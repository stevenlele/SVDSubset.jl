module SVDS

export svds

using LinearAlgebra

function svds(A::AbstractMatrix, k::Integer=6;
    tol::Real=1e-10, dim::Integer=max(3k, 15), maxiter::Integer=10)

    r = minimum(size(A))

    k = min(r, k)
    k <= 0 && throw(ArgumentError("A is empty or k is not positive"))

    tol < 0 && throw(ArgumentError("Tolerence should not be negative"))

    m = min(r, dim)
    k > m && throw(ArgumentError("Subspace dimension should be greater than k"))

    maxiter <= 0 && throw(ArgumentError("Max iterations should be positive"))


    T = float(eltype(A))
    v = normalize!(randn(T, size(A, 2)))
    U, B, V, v, β = lanczos_bidiag(A, 0, m, Nothing[], Nothing[], v)
    B = Matrix(B)

    for iter = 1:maxiter
        F = svd!(B)

        U *= @view F.U[:, 1:k]
        V *= @view F.V[:, 1:k]
        s = @view F.S[1:k]

        ρ = β * @view F.U[end, 1:k]

        if all(abs.(ρ) .<= tol * s)
            return U, Vector(s), V
        end

        if iter == maxiter
            @warn "Max iterations reached!"
            return U, Vector(s), V
        end

        U, B, V, v, β = lanczos_bidiag(A, k, m, U, V, v)
        B = [
            Diagonal(s) ρ zeros(k, m - k - 1)
            zeros(m - k, k) B
        ]
    end
end

function lanczos_bidiag(A::AbstractMatrix, k::Integer, m::Integer,
    Uk::AbstractArray, Vk::AbstractArray, v::Vector{T}) where {T}

    Arows, Acols = size(A)

    α = Vector{T}(undef, m - k)
    β = Vector{T}(undef, m - k - 1)

    U = Matrix{T}(undef, Arows, m)
    V = Matrix{T}(undef, Acols, m)

    if k > 0
        copyto!(U, Uk)
        copyto!(V, Vk)
    end

    V[:, k+1] = v

    u = A * v
    if k > 0
        u -= Uk * Uk' * u # Reorthogonalization
    end
    u /= (α[1] = norm(u))
    U[:, k+1] = u

    for i = 1:m-k-1
        Vi = @view V[:, 1:k+i]
        Ui = @view U[:, 1:k+i]

        v = A' * u - α[i] * v
        v -= Vi * Vi' * v # Reorthogonalization
        v /= (β[i] = norm(v))
        V[:, k+i+1] = v

        u = A * v - β[i] * u
        u -= Ui * Ui' * u # Reorthogonalization
        u /= (α[i+1] = norm(u))
        U[:, k+i+1] = u
    end

    v = A' * u - α[m-k] * v
    v -= V * V' * v # Reorthogonalization
    v /= (βm = norm(v))

    B = Bidiagonal(α, β, :U)

    return U, B, V, v, βm
end

end
