using Metis
using SparseArrays
using Polyester
using Base.Threads: Atomic, atomic_add!

"""
    get_partition_indices(A::SparseMatrixCSC, n_parts::Int)

Given a sparse matrix A, partitions the underlying graph into `n_parts` 
and identifies interior vs interface nodes.

Returns:
- `perm`: The permutation vector to transform A into Bordered Block Diagonal form.
- `interior_indices`: Vector of UnitRanges for each subdomain in the reordered system.
- `gamma_indices`: UnitRange for the interface in the reordered system.
"""
function get_partition_indices(A::SparseMatrixCSC, n_parts::Int)
    N = size(A, 1)
    
    # Compute the partition
    partition = Metis.partition(A, n_parts)
    
    # interface nodes
    is_interface = zeros(Bool, N)
    rows = rowvals(A)
    
    for i in 1:N
        part_i = partition[i]
        for j in nzrange(A, i)
            neighbor = rows[j]
            if partition[neighbor] != part_i
                is_interface[i] = true
                break
            end
        end
    end
    
    perm = Int[] # permutation index
    interior_indices = Vector{UnitRange{Int}}(undef, n_parts)
    current_ptr = 1
    
    for p in 1:n_parts
        # interior nodes
        p_interior = findall(i -> (partition[i] == p && !is_interface[i]), 1:N)
        append!(perm, p_interior)
        
        # create a UnitRange
        len = length(p_interior)
        interior_indices[p] = current_ptr:(current_ptr + len - 1)
        current_ptr += len
    end
    
    g_nodes = findall(is_interface)
    append!(perm, g_nodes)
    gamma_indices = current_ptr:(current_ptr + length(g_nodes) - 1)
    
    return perm, interior_indices, gamma_indices
end

function compute_topology(A_p, int_indices, g_indices)
    T = eltype(A_p)
    n_parts = length(int_indices)
    n_g = length(g_indices)
    
    node_to_parts = [Set{Int}() for _ in 1:n_g]
    global_to_rel_g = Dict(idx => i for (i, idx) in enumerate(g_indices))
    
    # Step 1: Direct connections (Interface touching Interior)
    for p in 1:n_parts
        interior_set = Set(int_indices[p]) 
        for g_node in g_indices
            for row_idx in nzrange(A_p, g_node)
                if rowvals(A_p)[row_idx] ∈ interior_set
                    rel_idx = global_to_rel_g[g_node]
                    push!(node_to_parts[rel_idx], p)
                    break 
                end
            end
        end
    end
    
    # Step 2: The 1-Hop Bridge (Interface touching Interface)
    # This bridges your 2-node thick interface so subdomains correctly overlap
    node_to_parts_bridged = deepcopy(node_to_parts)
    for (rel_idx, g_node) in enumerate(g_indices)
        for row_idx in nzrange(A_p, g_node)
            neighbor = rowvals(A_p)[row_idx]
            if neighbor ∈ g_indices
                rel_neighbor = global_to_rel_g[neighbor]
                # Share domain assignments across the gap!
                union!(node_to_parts_bridged[rel_idx], node_to_parts[rel_neighbor])
            end
        end
    end
    
    # Convert Sets to standard Arrays
    n2p_arrays = [collect(s) for s in node_to_parts_bridged]
    
    part_to_g_nodes = [Int[] for _ in 1:n_parts]
    for (rel_idx, parts) in enumerate(n2p_arrays)
        for p in parts
            push!(part_to_g_nodes[p], g_indices[rel_idx])
        end
    end
    
    # Partition of unity scaling
    scaling = [length(list) > 0 ? one(T)/length(list) : 1.0 for list in n2p_arrays]
    
    return scaling, part_to_g_nodes, n2p_arrays
end

# Metis.partition(A, n_parts) is super expensive for the toy example we use
# this uses the "create_laplacian_2d" logic instead
function get_partition_indices_laplacian_FD(n, n_parts)
    N = n^2
    # 1. Determine grid dimensions (px * py = n_parts)
    px = 1
    for i in 1:floor(Int, sqrt(n_parts))
        if n_parts % i == 0; px = i; end
    end
    py = div(n_parts, px)
    
    row_edges = [round(Int, i * n / py) for i in 0:py]
    col_edges = [round(Int, i * n / px) for i in 0:px]

    partition = zeros(Int32, N)
    is_interface = zeros(Bool, N)

    # 2. Parallel Grid Sweep: Assign Partitions and Detect Interfaces
    @batch for rp in 1:py
        for cp in 1:px
            p_id = (rp - 1) * px + cp
            for r in (row_edges[rp]+1):row_edges[rp+1]
                for c in (col_edges[cp]+1):col_edges[cp+1]
                    idx = (r-1)*n + c
                    partition[idx] = p_id
                    
                    # Boundary check logic
                    if r == row_edges[rp]+1 || r == row_edges[rp+1] || 
                       c == col_edges[cp]+1 || c == col_edges[cp+1]
                       # Check if it's a true interior boundary (not a global one)
                       if (r > 1 && r < n) || (c > 1 && c < n)
                           is_interface[idx] = true
                       end
                    end
                end
            end
        end
    end

    # 3. Parallel Assembly (Bucket Sort logic)
    counts = [Atomic{Int}(0) for _ in 1:n_parts]
    @batch for i in 1:N
        if !is_interface[i]
            atomic_add!(counts[partition[i]], 1)
        end
    end

    # Compute offsets for interior blocks
    interior_indices = Vector{UnitRange{Int}}(undef, n_parts)
    offsets = zeros(Int, n_parts)
    current_ptr = 1
    for p in 1:n_parts
        len = counts[p][]
        interior_indices[p] = current_ptr:(current_ptr + len - 1)
        offsets[p] = current_ptr
        current_ptr += len
    end

    # Fill Permutation Vector (Parallel)
    perm = zeros(Int, N)
    # Track local fill positions per partition
    local_ptrs = [Atomic{Int}(0) for _ in 1:n_parts]
    
    # Fill Interior
    @batch for i in 1:N
        if !is_interface[i]
            p = partition[i]
            pos = offsets[p] + atomic_add!(local_ptrs[p], 1)
            perm[pos] = i
        end
    end

    # Fill Interface (Sequential is usually fine here as it's small)
    g_nodes = findall(is_interface)
    perm[current_ptr:end] .= g_nodes
    gamma_indices = current_ptr:N

    return perm, interior_indices, gamma_indices
end