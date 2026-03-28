import Pkg
Pkg.activate(dirname(@__DIR__), io=devnull)
using HybridSchurLab, LinearAlgebra, BenchmarkTools, Krylov, DelimitedFiles

# Parameters
n_parts = 64
# N_list = [500,750]
N_list = [50, 100, 250, 400, 500, 600, 750, 1000, 1250, 1500, 2000]
output_file = "benchmark_results.csv"

println("Starting benchmark")

for n_grid in N_list
    println(">>> Testing Grid Size N = $n_grid")
    
    A_raw = create_laplacian_2d(n_grid)
    perm, int_idxs, g_idxs = get_partition_indices_laplacian_FD(n_grid, n_parts)
    A_p = A_raw[perm, perm]

    op = setup_schur_operator(A_p, int_idxs, g_idxs)
    
    b_p = ones(size(A_p, 1))
    b_I = b_p[vcat(int_idxs...)]
    b_g = b_p[g_idxs]
    # g = b_g - A_ΓI * (A_II \ b_I)
    g = b_g - op.A_gamma_I * solve_interior(op, b_I)

    # Test both precisions for the preconditioner
    for types in [[Float64, Int64], [Float32, Int32]]
        prec = setup_2lp(A_p, int_idxs, g_idxs, op, types[1], types[2])

        lu_mem_bytes = Base.summarysize(prec.local_LUs)
        lu_mem_mib = lu_mem_bytes / (1024^2)
        
        # Benchmark the preconditioner application: solve Mx = y
        y_bench = similar(g)
        t_apply = @belapsed ldiv!($y_bench, $prec, $g)
        
        W_type_int = ifelse(types[1] == Float64, 64, 32)        
        println("  [$types] | LU Mem: $(round(lu_mem_mib, digits=2)) MiB | Apply: $(round(t_apply, digits=6))s")
        
        # Save data
        open(output_file, "a") do io
            writedlm(io, [[n_grid, W_type_int, round(lu_mem_mib, digits=3), t_apply*1000]], ',')
        end
    end
end

println("\nBenchmark complete. Results saved to $output_file")