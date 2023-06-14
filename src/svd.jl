using LinearAlgebra
using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: libblastrampoline, @blasfunc
using LinearAlgebra.LAPACK: chklapackerror

# Calculate SVD of square matrix A
function svd!(A::Matrix{Float64}, U::Matrix{Float64}, S::Vector{Float64}, Vt::Matrix{Float64}, work::Vector{Float64})
    m = size(A, 1)
    lwork = BlasInt(5m)
    info = Ref{BlasInt}()
    ccall((@blasfunc(dgesvd_), libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
            Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
            Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
        'S', 'S', m, m, A, m, S, U, m, Vt, m, work, lwork, info, 1, 1)
    chklapackerror(info[])
end
