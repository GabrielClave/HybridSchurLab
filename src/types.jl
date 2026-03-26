"""
    SchurOperator{T, M, F}

A structure representing a Domain Decomposition solver based on the Schur Complement.

# Fields
- `LUs`: A vector of local LU factorizations for each subdomain.
- `A_I_gamma`: The coupling matrix from Interior nodes to Interface nodes.
- `A_gamma_I`: The coupling matrix from Interface nodes to Interior nodes.
- `A_gamma_gamma`: The matrix representing direct interactions on the interface.
- `interior_indices`: A list of ranges, one per subdomain, in the reordered system.
- `gamma_indices`: The range of the interface nodes in the reordered system.
- `tmp1`: A pre-allocated buffer for the interior vector (A_I_gamma * x).
- `tmp2`: A pre-allocated buffer for the interior result (A_II \\ tmp1).
"""
struct SchurOperator{T, M, F}
    # factorizations of interior domain
    LUs::Vector{F} 

    # Matrix blocks
    A_I_gamma::M
    A_gamma_I::M
    A_gamma_gamma::M

    # Indexing metadata
    interior_indices::Vector{UnitRange{Int}} #index of each domain
    gamma_indices::UnitRange{Int} #index of interface

    # Pre-allocated work buffers
    tmp1::Vector{T}
    tmp2::Vector{T}
    x_buffer::Vector{T}
    local_in::Vector{Vector{T}}
    local_out::Vector{Vector{T}}
end

struct NeumannWorkspace{T}
    rhs::Vector{T}
    sol::Vector{T}
end

"""
    NeumannPreconditioner{T, W, F, EF}

A structure representing a Neumann Preconditioner.
"""
struct NeumannPreconditioner{T, W, F, EF}
    # --- Neumann-Neumann Part ---
    local_LUs::Vector{F}
    scaling::Vector{T}
    local_to_global_maps::Vector{Vector{Int}}
    workspaces::Vector{NeumannWorkspace{W}}
    
    # --- Coarse Part ---
    E_fact::EF  # Factorized coarse matrix
    # Solve Buffers (GMRES)
    buffer_r::Vector{T} 
    r_c::Vector{T}
    e_c::Vector{T}
    
    # --- indexing ---
    int_idxs::Vector{UnitRange{Int}}
    g_idxs::UnitRange{Int}
end
