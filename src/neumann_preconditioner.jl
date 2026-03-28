using LinearAlgebra
using .Threads
using Polyester
using Base.Threads: Atomic, atomic_add!
using IncompleteLU

function setup_2lp(A_p::AbstractMatrix{T}, int_indices, g_indices, schur_ops::SchurOperator{T}, ::Type{W}=Float32, ::Type{V}=Int32; τ = 0.01, ε=1e-9) where {T, W, V}
    lck = ReentrantLock()
    n_parts = length(int_indices)
    n_gamma = length(g_indices)
    
    # store one LU factorization per subdomain "Si⁻¹"
    # Each LU is for the matrix [A_ii  A_ig ; A_gi  A_gg_i]
    dummy_matrix = sparse(V[1, 2], V[1, 2], W[1.0, 1.0], 2, 2)
    F_type = typeof(ilu(dummy_matrix, τ = τ))
    local_LUs = Vector{F_type}(undef, n_parts)

    local_to_global_maps = Vector{Vector{Int}}(undef, n_parts)
    E = zeros(T, n_parts, n_parts)

    # Scaling Dᵢ + list of "interface nodes per subdomain"
    scaling, part_to_g_nodes, node_to_parts = compute_topology(A_p, int_indices, g_indices)

    # For a standard Laplacian, nodes on the boundary have a row sum > 0
    global_row_sums = sum(A_p, dims=2) |> vec
    
    @threads for i in 1:n_parts
        # local interface nodes
        i_idx = int_indices[i]
        g_idx_i = part_to_g_nodes[i] # absolute indices in A_p
        local_to_global_maps[i] = findall(x -> x ∈ g_idx_i, g_indices) # relative index (between 1 and the size of int_indices[i])
        
        subdomain_indices = vcat(i_idx, g_idx_i)
        Ai = A_p[subdomain_indices, subdomain_indices] # Diagonal is not correct
        
        # Diagonal needs to be the sum of all contributions in Ai
        # not very general, designed four our 2D Laplacian FD scheme
        rows = rowvals(Ai)
        vals = nonzeros(Ai)
        for col in 1:size(Ai, 2)
            local_connectivity_sum = zero(T)
            diag_idx = -1
            
            for k in nzrange(Ai, col)
                row = rows[k]
                if row == col
                    diag_idx = k
                else
                    # sum of absolute values of off-diagonals
                    local_connectivity_sum += abs(vals[k])
                end
            end
            
            if diag_idx != -1
                # Diagonal = Local Neighbors + Global Boundary Condition
                global_node_idx = subdomain_indices[col]
                vals[diag_idx] = local_connectivity_sum + global_row_sums[global_node_idx]
            end
        end

        # Regularization for Floating Subdomains
        is_floating = sum(abs.(global_row_sums[subdomain_indices])) < 1e-10
        # no connection to the global boundary
        # sum of rows is 0: matrix is singular
        if is_floating Ai[1, 1] += one(T) end

        # factorization of Ai
        Ai_prec = SparseMatrixCSC{W, V}(Ai) # factorization of local Ai in lower precision
        local_LUs[i] = ilu(Ai_prec, τ= τ)

        # local coarse basis Z_i (Size: n_local_interface x n_neighbors) 
        Zi = build_local_coarse_basis(local_to_global_maps[i], scaling, n_parts, node_to_parts)

        A_ii_fact = schur_ops.LUs[i]

        n_i = length(i_idx)
        
        # Extract the correctly Neumann-weighted blocks from Ai
        # Views are perfectly safe and allocate no memory here
        A_ig = @view Ai[1:n_i, n_i+1:end]
        A_gi = @view Ai[n_i+1:end, 1:n_i]
        A_gg_i = @view Ai[n_i+1:end, n_i+1:end]
        Ei = compute_local_coarse_contribution(Zi, A_ii_fact, A_ig, A_gi, A_gg_i) # a n_part x n_part matrix is allocated
        
        # Locked assembly into the global E
        lock(lck) do
            E .+= Ei
        end
    end

    E_fact = lu(E)
    
    # allocates buffer    
    workspaces = Vector{NeumannWorkspace{W}}(undef, n_parts) # workspace vectors, one workspace per domain
    for i in 1:n_parts
        local_size = length(int_indices[i]) + length(local_to_global_maps[i])
        workspaces[i] = NeumannWorkspace(zeros(W, local_size), zeros(W, local_size))
    end
    
    buffer_r = zeros(T, n_gamma)
    r_c = zeros(T, n_parts)
    e_c = zeros(T, n_parts)
    
    return NeumannPreconditioner{T, W, F_type, typeof(E_fact)}(local_LUs, scaling, local_to_global_maps, workspaces, E_fact, buffer_r, r_c, e_c, int_indices, g_indices)
end

"""
    solve_preconditioner!(w, r, prec::NeumannPreconditioner)

Applies the Neumann-Neumann preconditioner: w = Σ (Ri' * Di * Si⁻¹ * Di * Ri) * r
"""
function solve_preconditioner!(w::AbstractVector{T}, r::AbstractVector{T}, prec::NeumannPreconditioner{T, W, F, EF}) where {T, W, F, EF}
    fill!(w, zero(T))
    
    # computes Si⁻¹ * Di * Ri * r
    @batch for p in 1:length(prec.local_LUs)
        ws = prec.workspaces[p]
        l_map = prec.local_to_global_maps[p]
        n_i = length(prec.int_idxs[p])

        fill!(ws.rhs, zero(W))
        
        # rhs = [ 0 ; D_i * r_i ]
        for (i, rel_idx) in enumerate(l_map)
            ws.rhs[n_i + i] = W(r[rel_idx] * prec.scaling[rel_idx]) # cast to lower precision
        end
        
        # sol = Ai \ rhs
        ldiv!(ws.sol, prec.local_LUs[p], ws.rhs)
    end

    # sequential add of all buffer
    for p in 1:length(prec.local_LUs)
        ws = prec.workspaces[p]
        l_map = prec.local_to_global_maps[p]
        n_i = length(prec.int_idxs[p])
        
        for (i, rel_idx) in enumerate(l_map)
            # w = Σ Rᵢ'Dᵢsolᵢ
            w[rel_idx] += T(ws.sol[n_i + i] * prec.scaling[rel_idx])
        end
    end
    return w
end

# call solve_preconditioner!
import LinearAlgebra: ldiv!

# we simply add the Neumann part to the coarse part
function ldiv!(P::NeumannPreconditioner, x::AbstractVector)
    copyto!(P.buffer_r, x) # Save the original residual into our buffer

    # Neumann-Neumann Part: M_NN⁻¹ * buffer_r -> x
    solve_preconditioner!(x, P.buffer_r, P) # x holds the local contributions

    # Coarse Part: M_C⁻¹ * buffer_r -> x (additive)
    
    # Restriction: r_c = Zᵀ * buffer_r
    fill!(P.r_c, 0.0)
    for (i, l_map) in enumerate(P.local_to_global_maps)
        for k in l_map
            P.r_c[i] += P.buffer_r[k] * P.scaling[k]
        end
    end

    # Coarse Solve: e_c = E \ r_c
    ldiv!(P.e_c, P.E_fact, P.r_c)

    # Prolongation: x += Z * e_c
    for i in 1:length(P.local_to_global_maps)
        l_map = P.local_to_global_maps[i]
        for k in l_map
            x[k] += P.e_c[i] * P.scaling[k]
        end
    end

    return x
end

function ldiv!(y::AbstractVector, P::NeumannPreconditioner, x::AbstractVector)

    # Neumann-Neumann Part: M_NN⁻¹ * buffer_r -> x
    solve_preconditioner!(y, x, P) # x holds the local contributions

    # Coarse Part: M_C⁻¹ * buffer_r -> x (additive)
    
    # Restriction: r_c = Zᵀ * buffer_r
    fill!(P.r_c, 0.0)
    for (i, l_map) in enumerate(P.local_to_global_maps)
        for k in l_map
            P.r_c[i] += x[k] * P.scaling[k]
        end
    end

    # Coarse Solve: e_c = E \ r_c
    ldiv!(P.e_c, P.E_fact, P.r_c)

    # Prolongation: x += Z * e_c
    for i in 1:length(P.local_to_global_maps)
        l_map = P.local_to_global_maps[i]
        for k in l_map
            y[k] += P.e_c[i] * P.scaling[k]
        end
    end

    return y
end