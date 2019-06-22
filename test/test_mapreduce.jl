module TestTMapreduce

using Test
import ThreadingTools; const TT=ThreadingTools
using BenchmarkTools: @benchmark

struct FreeMonoid{T}
    word::Vector{T}
end

function Base.:*(x::FreeMonoid, y::FreeMonoid)
    FreeMonoid(vcat(x.word, y.word))
end
Base.:(==)(x::FreeMonoid, y::FreeMonoid) = x.word == y.word
pure(x) = FreeMonoid([x])
Base.one(::Type{FreeMonoid{T}}) where {T} = FreeMonoid{T}(T[])

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

    # empty
    @test_throws ArgumentError TT.maximum(Int[])
    @test_throws ArgumentError TT.minimum(Int[])
    @test TT.prod(Int[]) === 1
    @test TT.sum(Int[]) === 0

    # performance
    data = randn(10^5)
    for red in [TT.sum, TT.prod, TT.minimum, TT.maximum]
        b = @benchmark ($red)($data) samples=1 evals=1
        @test b.allocs < 400
        b = @benchmark ($red)(sin, $data) samples=1 evals=1
        @test b.allocs < 400
    end
end

@testset "TT.reduce, TT.mapreduce" begin
    @test TT.mapreduce(pure, *, 1:1, init=pure(-1)) == pure(-1) * pure(1)
    @test TT.mapreduce(pure, *, 1:1, init=pure(-1)) == mapreduce(pure, *, 1:1, init=pure(-1))

    @test TT.mapreduce(identity, +, Int[], init=4) == 4
    @test TT.mapreduce(identity, +, Int[]) == 0
    @test TT.reduce(+, Int[]) == 0

    setups = [
                  (f=identity,       op=+, srcs=(1:10,),                    init=0),
                  (f=identity,       op=*, srcs=(String[],),                init="Hello"),
                  (f=identity,       op=+, srcs=(1:10,),                    init=21),
                  (f=x->2x,          op=*, srcs=(collect(1:10),),           init=21),
                  (f=x->2x,          op=*, srcs=(rand(Int, 10^5),),         init=rand(Int)),
                  (f=pure,           op=*, srcs=(randn(4),),                init=pure(42.0)),
                  (f=pure,           op=*, srcs=(randn(10^4),),             init=pure(42.0)),
       ]

    # multi arg mapreduce
    if VERSION >= v"1.2-"
        for n in 1:4
            s= (f=pure∘tuple,  op=*, srcs=[Float64[1] for _ in 1:n], init=pure(tuple(1.0:n...)))
            push!(setups, s)
        end
    else
        @warn "Skipping multi arg mapreduce tests, on julia $VERSION"
    end

    for setup in setups

        res_base = @inferred Base.reduce(setup.op, map(setup.f, setup.srcs...))
        res_tt   = @inferred   TT.reduce(setup.op, map(setup.f, setup.srcs...))
        @test res_base == res_tt

        res_base = @inferred Base.reduce(setup.op, map(setup.f, setup.srcs...), init=setup.init)
        res_tt   = @inferred   TT.reduce(setup.op, map(setup.f, setup.srcs...), init=setup.init)
        @test res_base == res_tt

        res_base = @inferred Base.mapreduce(setup.f, setup.op, setup.srcs...)
        res_tt   = @inferred   TT.mapreduce(setup.f, setup.op, setup.srcs...)
        @test res_base == res_tt

        res_base = @inferred Base.mapreduce(setup.f, setup.op, setup.srcs..., init=setup.init)
        res_tt   = @inferred   TT.mapreduce(setup.f, setup.op, setup.srcs..., init=setup.init)
        @test res_base == res_tt
    end

    # performance
    data = randn(10^5)
    b = @benchmark TT.mapreduce(sin, +, $data) samples=1 evals=1
    @test b.allocs < 400

    b = @benchmark TT.reduce(+, $data) samples=1 evals=1
    @test b.allocs < 400
end

end#module
