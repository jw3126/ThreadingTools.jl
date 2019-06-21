module TestTMapreduce

using Test
import ThreadingTools; const TT=ThreadingTools
using BenchmarkTools: @benchmark

struct FreeSemiGroup{T}
    word::Vector{T}
end

const FSG{T} = FreeSemiGroup{T}

function Base.:*(x::FSG, y::FSG)
    FSG(vcat(x.word, y.word))
end
Base.:(==)(x::FSG, y::FSG) = x.word == y.word
pure(x) = FSG([x])

@testset "sum, prod, minimum, maximum" begin
    
    for (f, arr) in [
            (identity, [1]),
            (sin, randn(10)),
            (identity, randn(10)),
            (x -> 2x, rand(Int, 10000)),
        ]
        @test Base.prod(arr) ≈ @inferred TT.prod(arr)
        @test Base.sum(arr) ≈ @inferred TT.sum(arr)
        @test Base.minimum(arr) === @inferred TT.minimum(arr)
        @test Base.maximum(arr) === @inferred TT.maximum(arr)
        
        @test Base.prod(f, arr) ≈ @inferred TT.prod(f, arr)
        @test Base.sum(f, arr) ≈ @inferred TT.sum(f, arr)
        @test Base.minimum(f, arr) === @inferred TT.minimum(f, arr)
        @test Base.maximum(f, arr) === @inferred TT.maximum(f, arr)
    end
    
    # performance
    data = randn(10^5)
    for red in [TT.sum, TT.prod, TT.minimum, TT.maximum]
        b = @benchmark ($red)($data) samples=1 evals=1
        @test b.allocs < 100
        b = @benchmark ($red)(sin, $data) samples=1 evals=1
        @test b.allocs < 100
    end
end

@testset "TT.reduce, TT.mapreduce" begin
    for setup in [
        (f=identity, op=+, src=1:10,            init=0),
        (f=identity, op=+, src=1:10,            init=21),
        (f=x->2x,    op=*, src=collect(1:10),   init=21),
        (f=x->2x,    op=*, src=rand(Int, 10^5), init=rand(Int)),
        (f=pure,     op=*, src=randn(4),        init=pure(42.0)),
        (f=pure,     op=*, src=randn(10^4),     init=pure(42.0)),
       ]
        res_base = @inferred Base.mapreduce(setup.f, setup.op, setup.src, init=setup.init)
        res_tt   = @inferred   TT.mapreduce(setup.f, setup.op, setup.src, init=setup.init)
        @test res_base == res_tt

        res_base = @inferred Base.mapreduce(setup.f, setup.op, setup.src)
        res_tt   = @inferred   TT.mapreduce(setup.f, setup.op, setup.src)
        @test res_base == res_tt

        res_base = @inferred Base.reduce(setup.op, map(setup.f, setup.src), init=setup.init)
        res_tt   = @inferred   TT.reduce(setup.op, map(setup.f, setup.src), init=setup.init)
        @test res_base == res_tt

        res_base = @inferred Base.reduce(setup.op, map(setup.f, setup.src))
        res_tt   = @inferred   TT.reduce(setup.op, map(setup.f, setup.src))
        @test res_base == res_tt
    end
    @test TT.mapreduce(pure, *, 1:1, init=pure(-1)) == pure(-1) * pure(1)
    @test TT.mapreduce(pure, *, 1:1, init=pure(-1)) == mapreduce(pure, *, 1:1, init=pure(-1))

    @test_broken TT.mapreduce(identity, +, Int[], init=0) == 0
    @test_broken TT.reduce(identity, +, Int[]) == 0

    # performance

    data = randn(10^5)
    b = @benchmark TT.mapreduce(sin, +, $data) samples=1 evals=1
    @test b.allocs < 200

    b = @benchmark TT.reduce(+, $data) samples=1 evals=1
    @test b.allocs < 200
end

end#module
