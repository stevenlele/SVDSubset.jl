module SVDSubset

export svds
include("svd.jl")

using LinearAlgebra

function svds(A::AbstractMatrix, k::Integer=6;
    tol::Real=1e-10, dim::Integer=max(3k, 15), maxiter::Integer=10)

    Arows, Acols = size(A)
    r = min(Arows, Acols)

    k = min(r, k)
    k > 0 || throw(ArgumentError("A is empty or k is not positive"))

    tol >= 0 || throw(ArgumentError("Tolerence should not be negative"))

    m = min(r, dim)
    k < m || throw(ArgumentError("Subspace dimension should be greater than k"))

    maxiter > 0 || throw(ArgumentError("Max iterations should be positive"))

    T = float(eltype(A))
    Tr = real(T)

    Um = Matrix{T}(undef, Arows, m)
    Vm = Matrix{T}(undef, Acols, m)
    Bm = zeros(Tr, m, m)

    u = Vector{T}(undef, Arows)
    v = normalize!(randn(T, Acols))

    # Temp buffer of u, v
    ut = Vector{T}(undef, Arows)
    vt = Vector{T}(undef, Acols)

    # SVD of Bm
    U = Matrix{Tr}(undef, m, m)
    S = Vector{Tr}(undef, m)
    Vt = Matrix{Tr}(undef, m, m)
    work = Vector{Tr}(undef, 5m)

    Uk = Matrix{T}(undef, Arows, k)
    Vk = Matrix{T}(undef, Acols, k)

    ρ = @view work[1:k]

    β = lanczos_bidiag!(A, 0, m, Um, Bm, Vm, u, v, ut, vt)

    for iter = 1:maxiter
        svd!(Bm, U, S, Vt, work)

        mul!(Uk, Um, @view U[:, 1:k])
        mul!(Vk, Vm, @view Vt'[:, 1:k])

        ρ .= β .* @view U[end, 1:k]

        if all(abs.(ρ) .<= tol .* @view S[1:k])
            return Uk, S[1:k], Vk
        end

        if iter == maxiter
            @warn "Max iterations reached!"
            return Uk, S[1:k], Vk
        end

        copyto!(Um, Uk)
        copyto!(Vm, Vk)

        fill!(Bm, zero(Tr))
        for i = 1:k
            Bm[i, i] = S[i]
        end
        Bm[1:k, k+1] = ρ
        lanczos_bidiag!(A, k, m, Um, Bm, Vm, u, v, ut, vt)
    end
end

function lanczos_bidiag!(A::AbstractMatrix, k::Integer, m::Integer,
    U::Matrix{T}, B::Matrix{Tr}, V::Matrix{T}, u::Vector{T}, v::Vector{T},
    ut::Vector{T}, vt::Vector{T}) where {T,Tr}

    V[:, k+1] = v

    mul!(u, A, v)
    if k > 0
        reorth!(u, @view(U[:, 1:k]), ut, @view vt[1:k])
    end
    u ./= (B[k+1, k+1] = α = norm(u))
    U[:, k+1] = u

    for i = 1:m-k-1
        mul!(vt, A', u)        # A' * u
        v .= vt .- α .* v  # v = A' * u - α * v
        reorth!(v, @view(V[:, 1:k+i]), vt, @view ut[1:k+i])
        v ./= (B[k+i, k+i+1] = β = norm(v))
        V[:, k+i+1] = v

        mul!(ut, A, v)         # A * v
        u .= ut .- β .* u  # u = A * v - β * u
        reorth!(u, @view(U[:, 1:k+i]), ut, @view vt[1:k+i])
        u ./= (B[k+i+1, k+i+1] = α = norm(u))
        U[:, k+i+1] = u
    end

    mul!(vt, A', u)        # A' * u
    v .= vt .- α .* v  # v = A' * u - α * v
    reorth!(v, V, vt, @view ut[1:m])
    v ./= (β = norm(v))

    return β
end

# Reorthogonalization: v .-= V * V' * v
@inline function reorth!(v, V, tcol, trow)
    mul!(trow, V', v)
    mul!(tcol, V, trow)
    v .-= tcol
end

end
