using Test
using HybridSchurLab
using LinearAlgebra
using SparseArrays

@testset "Tests" begin
    # Setup small problem
    n_grid = 10
    n_parts = 4
    
    A_raw = create_laplacian_2d(n_grid)
    perm, int_idxs, g_idxs = get_partition_indices_laplacian_FD(n_grid, n_parts)
    A_p = A_raw[perm, perm]

    n_interior = sum(length.(int_idxs))
    n_gamma = length(g_idxs)

    @testset "Schur Operator" begin
        # Setup the operator
        op = setup_schur_operator(A_p, int_idxs, g_idxs)
        
        @test typeof(op) <: SchurOperator
        @test length(op.LUs) == n_parts
        @test size(op.A_gamma_gamma) == (n_gamma, n_gamma)
        
        S = as_linear_map(op)
        
        # Compare against dense Schur Complement
        # S = A_gg - A_gI * A_II⁻¹ * A_Ig

        int_range = 1:n_interior
        A_II = A_p[int_range, int_range]
        A_Ig = A_p[int_range, g_idxs]
        A_gI = A_p[g_idxs, int_range]
        A_gg = A_p[g_idxs, g_idxs]
        
        # Explicitly forming the dense Schur complement
        S_dense = Matrix(A_gg) - Matrix(A_gI) * (Matrix(A_II) \ Matrix(A_Ig))
        
        x = rand(n_gamma)
        y_op = similar(x)
        
        # matrix-free Schur product
        mul!(y_op, S, x) 
        
        # dense matrix multiplication
        y_dense = S_dense * x 
        
        @test y_op ≈ y_dense rtol=1e-10
    end

    @testset "Neumann Preconditioner" begin
        op = setup_schur_operator(A_p, int_idxs, g_idxs)
        
        # Test Float64 Preconditioner
        prec64 = setup_2lp(A_p, int_idxs, g_idxs, op, Float64, Int64)
        
        @test typeof(prec64) <: NeumannPreconditioner
        @test length(prec64.local_LUs) == n_parts
        @test eltype(prec64.workspaces[1].sol) == Float64
        
        r = rand(n_gamma)
        w64 = similar(r)
        
        # Apply preconditioner
        ldiv!(w64, prec64, r)
        
        @test length(w64) == n_gamma
        @test any(w64 .!= 0.0)
        
        # Test Mixed Precision Float32 Preconditioner
        prec32 = setup_2lp(A_p, int_idxs, g_idxs, op, Float32, Int32)
        
        @test typeof(prec32) <: NeumannPreconditioner
        @test eltype(prec32.workspaces[1].sol) == Float32 # Ensure lower precision buffer was allocated
        
        w32 = similar(r)
        ldiv!(w32, prec32, r)
        
        @test norm(w64 - w32) / norm(w64) < 1e-4
    end
end