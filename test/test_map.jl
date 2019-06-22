module TestTMap

using Test
import ThreadingTools; const TT=ThreadingTools
using .TT: Batches
using BenchmarkTools
using OffsetArrays

@testset "Batches" begin
    for _ in 1:10
        start = rand(-1000:1000)
        stop = start + rand(1:1000)
        inds = start:stop
        batch_size = rand(1:1000)
        b = Batches(inds, batch_size)
        @test vcat([b[i] for i in eachindex(b)]...) == inds
    end
    
    inds = Base.OneTo(rand(1:1000))
    batch_size = rand(1:1000)
    b = Batches(inds, batch_size)
    @test vcat([b[i] for i in eachindex(b)]...) == inds
end

@testset "test TT.map, TT.map!" begin

    src = OffsetArray(rand(3), -2:0)
    dst = similar(src)
    @test TT.map!(sqrt, dst, src) == Base.map(sqrt, src)
    @test Base.map(sqrt, src) == TT.map(sqrt, src)

    src = (1:2, [1.0, 2.0], [false, true])
    @test Base.map(*, src...) ==  TT.map(*, src...)

    for (f, src) in [
              (tuple, tuple([randn(2) for _ in 1:3]...)),
              (sin  , (randn(3),)                      ),
              (+    , (randn(3), randn(3),)            ),
              (+    , (randn(6), randn(2,3),)          ),
              (x->x , (randn(100,100),)                ),
              (atan , (randn(10^4), randn(10^4))       ),
              (*    , (1:2, randn(2), [true, false])   ),
              (tuple, tuple([randn(1) for _ in 1:3]...)),
              (sqrt , (OffsetArray(rand(3), -2:0),)   ),
             ]
        src1 = first(src)
        y = f(map(first, src)...)
        T = typeof(y)
        dst_map = similar(src1, T)
        dst_TTmap = similar(src1, T)

        res_map!  = @inferred map!(f, dst_map, src...)
        res_TTmap! = @inferred TT.map!(f, dst_TTmap, src...)
        res_map   = @inferred map(f, src...)
        res_TTmap  = @inferred TT.map(f, src...)
        @test typeof(res_map) == typeof(res_TTmap)
        @test res_map == res_TTmap
        @test res_map! == res_TTmap!
    end

    # TT.map empty
    @test TT.map(+, Int[], Float64[]) == Float64[]

    @test_throws DimensionMismatch TT.map(+, randn(5), randn(4))
    @test_throws DimensionMismatch TT.map!(+, randn(4), randn(5), randn(4))
    @test_throws DimensionMismatch TT.map!(+, randn(5), randn(5), randn(4))

    # allocations
    for n in 1:4
        srcs = [randn(10^4) for _ in 1:n]
        # circumvent bug in julia < 1.2:
        # https://discourse.julialang.org/t/debugging-strange-allocations/25488/5?u=jw3126
        TT.map(tuple, srcs...)
        b = @benchmark TT.map($tuple, $(srcs...)) evals=1 samples=1
        @test b.allocs < 400
    end
end

end#module
