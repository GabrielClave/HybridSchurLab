using LinearAlgebra
using .Threads
using LinearMaps
using Polyester

## Schur operator

function create_schur_operator(LUs, A_I_g, A_g_I, A_g_g, int_idxs, g_idxs)
    T = eltype(A_g_g)
    n_interior = sum(length.(int_idxs))
    n_gamma = length(g_idxs)
    n_parts = length(int_idxs)

    t1 = zeros(T, n_interior)
    t2 = zeros(T, n_interior)
    x_buffer = zeros(T, n_gamma)

    local_in = [zeros(T, length(int_idxs[p])) for p in 1:n_parts]
    local_out = [zeros(T, length(int_idxs[p])) for p in 1:n_parts]
    
    return SchurOperator(
        LUs, A_I_g, A_g_I, A_g_g, int_idxs, g_idxs, t1, t2, x_buffer, local_in, local_out
    )
end

"""
    setup_schur_operator(A_p, int_indices, g_indices)

setup the matrix-free Schur operator
"""
function setup_schur_operator(A_p, int_indices, g_indices)
    n_parts = length(int_indices)
    
    n_interior = sum(length.(int_indices))
    int_range = 1:n_interior
    
    # hard copy, using views of sparse matrix was terrible
    A_Ig = A_p[int_range, g_indices]
    A_gI = A_p[g_indices, int_range]
    A_gg = A_p[g_indices, g_indices]

    T = eltype(A_p)
    F_type = typeof(lu(sparse(one(T)*I, 1, 1)))
    local_LUs = Vector{F_type}(undef, n_parts)

    # factorize the interior domains
    @batch for p in 1:n_parts
        local_LUs[p] = lu(A_p[int_indices[p], int_indices[p]])
    end

    return create_schur_operator(local_LUs, A_Ig, A_gI, A_gg, int_indices, g_indices)
end

"""
    solve_interior(op::SchurOperator, b_I)

returns x_I = A_II⁻¹ * b_I.
"""
function solve_interior(op::SchurOperator, b_I)
    x_I = similar(b_I)
    @batch for p in 1:length(op.LUs)
        idx = op.interior_indices[p]
        x_view = view(x_I, idx)
        b_view = view(b_I, idx)
        ldiv!(x_view, op.LUs[p], b_view)
    end
    return x_I
end

"""
    schur_product!(y, x, op::SchurOperator)

the application of the Schur complement matrix S = A_gg - A_gI * A_II⁻¹ * A_Ig.
"""
function schur_product!(y::AbstractVector{T}, x::AbstractVector{T}, op::SchurOperator{T}) where T

    # Force the Krylov view into a contiguous standard Vector
    copyto!(op.x_buffer, x)
    
    # use the buffer for multiplication
    mul!(y, op.A_gamma_gamma, op.x_buffer) # y = A_γγ * x
    mul!(op.tmp1, op.A_I_gamma, op.x_buffer) # tmp1 = A_I_γ * x

    LUs = op.LUs
    int_idxs = op.interior_indices
    local_in = op.local_in
    local_out = op.local_out
    tmp1 = op.tmp1
    tmp2 = op.tmp2
    
    # tmp2 = A_II⁻¹ * tmp1
    @batch for p in 1:length(op.LUs)
        idx = int_idxs[p]        
        local_in[p] .= @view tmp1[idx]    
        ldiv!(local_out[p], LUs[p], local_in[p])        
        tmp2[idx] .= local_out[p]
    end
    
    # y = y - A_γI * tmp2
    mul!(y, op.A_gamma_I, op.tmp2, -1.0, 1.0) # y = 1.0 * y + (-1.0) * (A_γI * tmp2)
    
    return y
end

"""
    as_linear_map(op::SchurOperator)

Wraps the SchurOperator into a LinearMap.
"""
function as_linear_map(op::SchurOperator{T}) where T
    n_gamma = length(op.gamma_indices) # S has size n_gamma x n_gamma
    return LinearMap{T}(
        (y, x) -> schur_product!(y, x, op),
        n_gamma, n_gamma,
        ismutating = true
    )
end