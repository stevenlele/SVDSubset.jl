using SVDSubset
using Test, LinearAlgebra, SparseArrays

function testSVDMatch(r1, r2, rank=size(r1.S, 1))
    approxEq(a, b) = sum(abs, a .* b) ≈ 1

    @test r1.S[1:rank] ≈ r2.S[1:rank]
    @testset "singular vectors" begin
        for j = 1:rank
            @test approxEq(r1.U[:, j], r2.U[:, j])
            @test approxEq(r1.V[:, j], r2.V[:, j])
        end
    end
end

# Taken from Arpack.jl
@testset "sparse" begin
    @testset "real" begin
        A = sparse([1, 1, 2, 3, 4], [2, 1, 1, 3, 1], [2.0, -1.0, 6.1, 7.0, 1.5])
        U, S, V = svds(A, 2)
        r2 = svd(Array(A))

        testSVDMatch(SVD(U, S, V'), r2)
    end

    @testset "complex" begin
        A = sparse([1, 1, 2, 3, 4], [2, 1, 1, 3, 1], exp.(im*[2.0:2:10;]), 5, 4)
        U, S, V = svds(A, 2)
        r2 = svd(Array(A))

        testSVDMatch(SVD(U, S, V'), r2)
    end
end
