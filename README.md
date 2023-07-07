# SVDSubset

[![Build Status](https://github.com/stevenlele/SVDSubset.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stevenlele/SVDSubset.jl/actions/workflows/CI.yml?query=branch%3Amain)

This work is led by the [THU-numbda](https://github.com/THU-numbda) group, and based on the [svds-C](https://github.com/XuFengthucs/svds-c) project.

## Usage

```julia
function svds(A::AbstractMatrix, k::Integer=6;
    tol::Real=1e-10, dim::Integer=max(3k, 15), maxiter::Integer=10)
```

Computes the `k` largest singular values of `A`.

- `tol`: convergence tolerance (relative)
- `dim`: subspace dimension (maximum size of Krylov subspace)
- `maxiter`: maximum number of algorithm iterations

Returns `(U, S, V)` where `S` is the vector of singular values and
`U`, `V` are matrices of the left and right singular vectors.

## See also

- `svds` in MATLAB: https://www.mathworks.com/help/matlab/ref/svds.html
- `svdl` in IterativeSolvers.jl: https://iterativesolvers.julialinearalgebra.org/stable/svd/svdl/
  - `SVDSubset.jl` is much faster and memory-efficient than `svdl`. When computing SVD subset of
    [SNAP dataset](https://snap.stanford.edu/data/soc-Slashdot0902.html) on a Apple M2 chip,
    it takes 2.11s / 254MB (memory allocations) for `SVDSubset.jl` or 4.45s / 21.3GB for `svdl` when k=50;
    it takes 6.36s / 506MB for `SVDSubset.jl` or 17.8s / 82.3GB for `svdl` when k=100.
