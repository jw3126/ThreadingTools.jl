# ThreadingTools

[![Build Status](https://travis-ci.com/jw3126/ThreadingTools.jl.svg?branch=master)](https://travis-ci.com/jw3126/ThreadingTools.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jw3126/ThreadingTools.jl?svg=true)](https://ci.appveyor.com/project/jw3126/ThreadingTools-jl)
[![Codecov](https://codecov.io/gh/jw3126/ThreadingTools.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jw3126/ThreadingTools.jl)
[![Coveralls](https://coveralls.io/repos/github/jw3126/ThreadingTools.jl/badge.svg?branch=master)](https://coveralls.io/github/jw3126/ThreadingTools.jl?branch=master)

# Usage

ThreadingTools defines threaded versions of the following functions: `map, map!, mapreduce, reduce, sum, prod, minimum, maximum`
```julia
julia> import ThreadingTools; const TT=ThreadingTools;

julia> using BenchmarkTools

julia> Threads.nthreads()
4

julia> data = randn(10^6);

julia> @btime sum(sin, data)
  13.114 ms (1 allocation: 16 bytes)
279.2390057547361

julia> @btime TT.sum(sin,data)
  3.722 ms (60 allocations: 4.09 KiB)
279.23900575473743

julia> @btime mapreduce(sin,*,data)
  15.607 ms (1 allocation: 16 bytes)
0.0

julia> @btime TT.mapreduce(sin,*,data)
  3.718 ms (60 allocations: 4.08 KiB)
0.0
```

# Credits
ThreadingTools was inspired and reuses some code of [`KissThreading`](https://github.com/mohamed82008/KissThreading.jl)
