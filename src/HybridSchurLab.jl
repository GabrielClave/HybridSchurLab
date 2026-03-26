module HybridSchurLab

using SparseArrays
using LinearAlgebra
using Metis
using LinearMaps
using .Threads
using IncompleteLU

include("types.jl")
include("partition.jl")
include("schur_operator.jl")
include("neumann_preconditioner.jl")
include("utils.jl")

export SchurOperator, NeumannPreconditioner, NeumannWorkspace
export create_schur_operator, setup_schur_operator
export as_linear_map, schur_product!, solve_interior
export setup_2lp, solve_preconditioner!
export create_laplacian_2d, get_partition_indices_laplacian_FD, get_partition_indices

end