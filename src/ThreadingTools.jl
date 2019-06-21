module ThreadingTools
using ArgCheck
using OffsetArrays

############################## docs ##############################
function _make_docstring(signature, fname)
    """
        $signature

    Threaded analog of [`Base.$fname`](@ref). See [`Base.$fname`](@ref) for a description of arguments.
    """
end

const SYMBOLS_MAPREDUCE_LIKE = [:sum, :prod, :minimum, :maximum]
for fun in SYMBOLS_MAPREDUCE_LIKE
    signature = "$fun([f,] src::AbstractArray)"
    docstring = _make_docstring(signature, fun)
    @eval begin
        """
        $($docstring)
        """
        function $fun end
    end
end

"""
$(_make_docstring("map!(f, dst::AbstractArray, srcs::AbstractArray...)", :map!))
"""
function map! end

"""
$(_make_docstring("map(f, srcs::AbstractArray...)", :map))
"""
function map end

"""
$(_make_docstring("mapreduce(f, op, src::AbstractArray [;init])", :mapreduce))
"""
function mapreduce end

"""
$(_make_docstring("reduce(op, src::AbstractArray [;init])", :reduce))
"""
function reduce end

############################## Helper functions ##############################
struct Batches{V}
    firstindex::Int
    lastindex::Int
    batch_size::Int
    values::V
    length::Int
end

function Batches(values, batch_size::Integer)
    @argcheck batch_size > 0
    r = eachindex(values)
    @assert r isa AbstractUnitRange
    len = ceil(Int, length(r) / batch_size)
    Batches(first(r), last(r), batch_size, values, len)
end

Base.length(o::Batches) = o.length
Base.eachindex(o::Batches) = Base.OneTo(length(o))
function Base.getindex(o::Batches, i)
    @boundscheck (@argcheck i in eachindex(o))
    start = o.firstindex + (i-1) * o.batch_size
    stop  = min(start + (o.batch_size) -1, o.lastindex)
    o.values[start:stop]
end

function default_batch_size(len)
    len <= 1 && return 1
    nthreads=Threads.nthreads()
    items_per_thread = len/nthreads
    items_per_batch = items_per_thread/4
    clamp(1, round(Int, items_per_batch), len)
end

function Base.iterate(o::Batches, state=1)
    if state in eachindex(o)
        o[state], state+1
    else
        nothing
    end
end

mutable struct _RollingCutOut{A,I<:AbstractUnitRange,T} <: AbstractVector{T}
    array::A
    eachindex::I
end

function _RollingCutOut(array::AbstractArray, indices)
    T = eltype(array)
    A = typeof(array)
    I = typeof(indices)
    _RollingCutOut{A, I, T}(array, indices)
end

Base.size(r::_RollingCutOut) = (length(r.eachindex),)

function Base.eachindex(r::_RollingCutOut, rs::_RollingCutOut...)
    for r2 in rs
        @assert r.eachindex == r2.eachindex
    end
    Base.IdentityUnitRange(r.eachindex)
end
Base.axes(r::_RollingCutOut) = (eachindex(r),)

@inline function Base.getindex(o::_RollingCutOut, i)
    @boundscheck checkbounds(o, i)
    @inbounds o.array[i]
end

@inline function Base.setindex!(o::_RollingCutOut, val, i)
    @boundscheck checkbounds(o, i)
    @inbounds o.array[i] = val
end

############################## map, map! ##############################
struct MapWorkspace{F,B,AD,AS}
    f::F
    batches::B
    arena_dst_view::AD
    arena_src_views::AS
end

@noinline function run!(o::MapWorkspace)
    let o=o
        Threads.@threads for i in 1:length(o.batches)
            tid = Threads.threadid()
            dst_view  = o.arena_dst_view[tid]
            src_views = o.arena_src_views[tid]
            inds = o.batches[i]
            dst_view.eachindex = inds
            for view in src_views
                view.eachindex = inds
            end
            Base.map!(o.f, dst_view, src_views...)
        end
    end
end

@noinline function prepare(::typeof(map!), f, dst, srcs)
    isempty(first(srcs)) && return dst
    batch_size = default_batch_size(length(dst))
    # we use IndexLinear since _RollingCutOut implementation
    # does not support other indexing well
    all_inds  = eachindex(IndexLinear(), srcs...)
    batches   = Batches(all_inds, batch_size)
    sample_inds = batches[1]
    nt = Threads.nthreads()
    arena_dst_view  = [_RollingCutOut(dst, sample_inds) for _ in 1:nt]
    arena_src_views = [[_RollingCutOut(src, sample_inds) for src in srcs] for _ in 1:nt]
    return MapWorkspace(f, batches, arena_dst_view, arena_src_views)
end

@noinline function map!(f, dst, srcs::AbstractArray...)
    w = prepare(map!, f, dst, srcs)
    run!(w)
    dst
end

function map(f, srcs::AbstractArray...)
    g = Base.Generator(f,srcs...)
    T = Base.@default_eltype(g)
    dst = similar(first(srcs), T)
    map!(f, dst, srcs...)
end

############################## mapreduce(like) ##############################
struct Reduction{O}
    op::O
end
(red::Reduction)(f, src) = Base.mapreduce(f, red.op, src)

struct MapReduceWorkspace{R,F,B,V<:_RollingCutOut,OA<:OffsetArray}
    reduction::R
    f::F
    batches::Batches{B}
    arena_src_view::Vector{V}
    batch_reductions::OA
end

struct NoInit end

function create_reduction(::typeof(mapreduce), op)
    Reduction(op)
end

function prepare(::typeof(mapreduce), f, op, src; init)
    red = create_reduction(mapreduce, op)
    w = prepare_mapreduce_like(red, f, src, init)
    return w
end

function mapreduce(f, op, src::AbstractArray; init=NoInit())
    @argcheck !isempty(src)
    w = prepare(mapreduce, f, op, src, init=init)
    run!(w)
end

function reduce(op, src::AbstractArray; init=NoInit())
    mapreduce(identity, op, src, init=init)
end

for red in SYMBOLS_MAPREDUCE_LIKE
    @eval function $red end
    
    @eval function prepare(::typeof($red), f, src)
        base_red = Base.$red
        prepare_mapreduce_like(base_red, f, src)
    end
    
    @eval function $red(f, src::AbstractArray)
        @argcheck !isempty(src)
        w = prepare($red, f, src)
        run!(w)
    end
    @eval $red(src) = $red(identity, src)
end

function prepare_mapreduce_like(red, f, src::AbstractArray, init=NoInit())
    batch_size = default_batch_size(length(src))
    all_inds  = eachindex(IndexLinear(), src)
    batches   = Batches(all_inds, batch_size)
    sample_inds = batches[1]
    nt = Threads.nthreads()
    arena_src_view = [_RollingCutOut(src, sample_inds) for _ in 1:nt]
    T = get_return_type(red, f, src)

    if (init isa NoInit)
        batch_reductions = OffsetVector{T}(undef, 1:length(batches))
    else
        batch_reductions = OffsetVector{T}(undef, 0:length(batches))
        batch_reductions[0] = init
    end
    MapReduceWorkspace(red, f, batches, arena_src_view, batch_reductions)
end

@inline function get_return_type(red, f, src)
    Core.Compiler.return_type(red, Tuple{typeof(f), typeof(src)})
end

@noinline function run!(o::MapReduceWorkspace)
    Threads.@threads for i in 1:length(o.batches)
        tid = Threads.threadid()
        src_view = o.arena_src_view[tid]
        inds = o.batches[i]
        src_view.eachindex = inds
        o.batch_reductions[i] = o.reduction(o.f, src_view)
    end
    o.reduction(identity, o.batch_reductions)
end

end # module
