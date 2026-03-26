import Base: show
using LinearAlgebra
using SparseArrays

function create_laplacian_2d(n)
    # Create 1D Laplacian: tridiagonal [-1, 2, -1]
    L1D = spdiagm(
        -1 => fill(-1.0, n-1), 
         0 => fill(2.0, n), 
         1 => fill(-1.0, n-1)
    )
    
    In = sparse(1.0I, n, n)
    
    # 2D Laplacian: L2D = (L1D ⊗ In) + (In ⊗ L1D)
    return kron(L1D, In) + kron(In, L1D)
end

function build_local_coarse_basis(l_map, scaling, n_parts, node_to_parts)
    n_li = length(l_map)
    Zi = zeros(eltype(scaling), n_li, n_parts)
    
    for k in 1:n_li
        global_g_idx = l_map[k]  # The relative index in the global interface
        weight = scaling[global_g_idx]
        
        # This node contributes to the coarse modes of ALL its owner subdomains
        for domain_id in node_to_parts[global_g_idx]
            Zi[k, domain_id] = weight
        end
    end
    return Zi
end

function compute_local_coarse_contribution(Zi, A_ii_fact, A_ig, A_gi, A_gg_i) # allocates 3 vectors
    # Zi: (n_local_g x n_parts)
    # W = A_ig * Zi
    W = A_ig * Zi
    
    # Y = A_ii \ W
    Y = A_ii_fact \ W
    
    # SiZi = A_gg_i * Zi - A_gi * Y
    SiZi = A_gg_i * Zi - A_gi * Y
    
    # Ei = Zi' * SiZi
    return Zi' * SiZi
end

function show(io::IO, ::MIME"text/plain", S::SchurOperator{T}) where T
    n_domains = length(S.LUs)
    gamma_size = length(S.gamma_indices)
    
    println(io, "SchurOperator{$T}")
    println(io, "  Domains:      $n_domains")
    println(io, "  Interface:    $gamma_size unknowns")
    print(io,   "  Precision:    $T")
end

function show(io::IO, ::MIME"text/plain", N::NeumannPreconditioner{T, W}) where {T, W}
    n_domains = length(N.local_LUs)
    
    println(io, "NeumannPreconditioner{$T}")
    println(io, "  Domains:      $n_domains")
    print(io,   "  Precision:    $T")
    print(io,   "  Local Precision:  $W")
end