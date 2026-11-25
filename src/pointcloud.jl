# A terrain model from a lidar point cloud: the points of LAS or LAZ files, local or on the web, or of a COPC file read
# in part over the web, gathered into one LAZ file by PDAL; the ground points kept as the survey classified them, or
# the ground found again with a morphological filter; the lowest ground point of each cell of a grid, the cells without
# one filled from the cells around them, up to a distance, and the ground points of each cell counted, so that what is
# filled shows. The grids run from the south-west corner, x first; the extension for Rasters.jl turns them into rasters.

"""The classes of the LAS specification that are noise, left out of every grid: low point (7) and high noise (18)."""
const NOISE_CLASSES = (7, 18)

"""The classes of the LAS specification that are ground: ground (2) and, in LAS 1.0 to 1.3, model key points (8), the
ground points a survey keeps for a thinned terrain model."""
const GROUND_CLASSES = (2, 8)

"""
	point_cloud_summary(source)

What the header of the point cloud `source` says, read by PDAL without reading its points: a local LAS, LAZ or COPC
file, or the address of a COPC file on the web, of which only the header is read. Its `points`, its `bounds`
(`xmin`, `ymin`, `zmin`, `xmax`, `ymax`, `zmax`) and its coordinate system, as `wkt` (empty when the file has none) and
`epsg` (`nothing` when PDAL names no EPSG code).
"""
function point_cloud_summary(source::AbstractString)
	reader = is_copc(source) ? `--driver readers.copc` : ``
	info = JSON.parse(read(`$(PDAL_jll.pdal()) info --summary $reader $source`, String))["summary"]
	b = info["bounds"]
	srs = get(info, "srs", Dict())
	wkt = srs isa AbstractDict ? string(get(srs, "wkt", "")) : ""
	return (points = Int(info["num_points"]), bounds = (xmin = b["minx"], ymin = b["miny"], zmin = b["minz"], xmax = b["maxx"], ymax = b["maxy"], zmax = b["maxz"]),
		wkt = wkt, epsg = epsg_code(wkt))
end

is_copc(source) = endswith(lowercase(source), ".copc.laz")
is_web(source) = startswith(source, "https://") || startswith(source, "http://")

function epsg_code(wkt)  # the last AUTHORITY["EPSG", n] of a WKT is that of the whole system
	codes = [m[1] for m in eachmatch(r"AUTHORITY\[\"EPSG\",\s*\"?(\d+)\"?\]", wkt)]
	ids = [m[1] for m in eachmatch(r"ID\[\"EPSG\",\s*(\d+)\]", wkt)]
	found = isempty(ids) ? codes : ids
	return isempty(found) ? nothing : parse(Int, last(found))
end

"""
	fetch_points(sources, path; bounds = nothing, resolution = nothing, folder = dirname(path))

Gathers the points of `sources` into one LAZ file at `path`, with PDAL, and returns `path`; when `path` exists it is
kept and nothing is read again. `sources` is a file or address, or several: LAS and LAZ files, local or on the web
(downloaded once into `folder`), and COPC files (`.copc.laz`), of which over the web only the parts within `bounds`
are read. `bounds = (xmin, ymin, xmax, ymax)`, in the coordinates of the points, keeps only the points within it.
`resolution`, in metres, reads a COPC file only to the level of its tree where the points are that far apart, fewer
points and quicker, which only COPC files can do.
"""
function fetch_points(sources, path::AbstractString; bounds = nothing, resolution = nothing, folder = dirname(path))
	isfile(path) && return path
	sources = sources isa AbstractString ? [sources] : collect(sources)
	isempty(sources) && throw(ArgumentError("no point cloud to read: no source given, or no tile of the index crosses the bounds"))
	box = bounds === nothing ? nothing : "([$(bounds[1]),$(bounds[3])],[$(bounds[2]),$(bounds[4])])"
	stages = Any[]
	for s in sources
		if is_copc(s)
			stage = Dict{String, Any}("type" => "readers.copc", "filename" => s)
			box === nothing || (stage["bounds"] = box)
			resolution === nothing || (stage["resolution"] = resolution)
		else
			resolution === nothing || @warn "resolution applies only to COPC files: $(s) is read whole"
			local_file = is_web(s) ? downloaded(s, folder) : s
			stage = Dict{String, Any}("type" => "readers.las", "filename" => local_file)
		end
		push!(stages, stage)
	end
	length(stages) > 1 && push!(stages, Dict("type" => "filters.merge"))
	box === nothing || push!(stages, Dict("type" => "filters.crop", "bounds" => box))
	mkpath(dirname(abspath(path)))
	partial = path * ".part"
	push!(stages, Dict("type" => "writers.las", "filename" => partial, "compression" => true, "forward" => "all", "minor_version" => 4))
	spec = tempname() * ".json"
	write(spec, JSON.json(Dict("pipeline" => stages)))
	report = IOBuffer()
	ok = success(pipeline(`$(PDAL_jll.pdal()) pipeline $spec`; stdout = devnull, stderr = report))
	rm(spec; force = true)
	if !ok
		rm(partial; force = true)
		error("PDAL could not read the point cloud: " * strip(String(take!(report))))
	end
	mv(partial, path; force = true)
	return path
end

function downloaded(url, folder)
	file = joinpath(mkpath(folder), basename(split(url, '?')[1]))
	isfile(file) || mv(Downloads.download(url, file * ".part"), file; force = true)
	return file
end

"""
	read_points(path)

The points of the LAS or LAZ file `path`, read with LazIO, each with its `geometry` (x, y and z), `classification`,
`return_number`, `number_of_returns` and `intensity`.
"""
read_points(path::AbstractString) = collect(LazIO.open(path))

"""
	point_classes(points)

The number of `points` of each class, as `class => number` pairs, the classes in order.
"""
function point_classes(points)
	n = Dict{Int, Int}()
	for p in points
		c = Int(p.classification)
		n[c] = get(n, c, 0) + 1
	end
	return sort(collect(n))
end

"""
	ground_grid(points, cellsize; classes = GROUND_CLASSES, bounds = nothing)

The lowest point of `classes` (the ground as the survey classified it) in each cell of `cellsize` metres: a grid with
its `heights` (`NaN` where a cell has no such point), the number of those points in each cell, `counts`, and its
`west` and `south` edges and `cellsize`, x first and from the south. `bounds = (xmin, ymin, xmax, ymax)` sets its edges,
else those of the points.
"""
ground_grid(points, cellsize::Real; classes = GROUND_CLASSES, bounds = nothing) =
	lowest_points(p -> p.classification in classes, points, cellsize, bounds)

"""
	lowest_grid(points, cellsize; bounds = nothing)

The lowest point of each cell of `cellsize` metres among all the points but noise ([`NOISE_CLASSES`](@ref)), whatever
their class, and the number of those points in each cell: the surface from which [`find_ground`](@ref) finds the
ground. A grid as [`ground_grid`](@ref) gives.
"""
lowest_grid(points, cellsize::Real; bounds = nothing) =
	lowest_points(p -> !(p.classification in NOISE_CLASSES), points, cellsize, bounds)

function lowest_points(keep, points, cellsize, bounds)
	cellsize > 0 || throw(ArgumentError("the cell size must be above 0"))
	chosen = filter(keep, points)
	isempty(chosen) && throw(ArgumentError("no points of those classes"))
	if bounds === nothing
		xs = (p.geometry[1] for p in chosen); ys = (p.geometry[2] for p in chosen)
		west, east = floor(minimum(xs) / cellsize) * cellsize, ceil(maximum(xs) / cellsize) * cellsize
		south, north = floor(minimum(ys) / cellsize) * cellsize, ceil(maximum(ys) / cellsize) * cellsize
	else
		west, south, east, north = Float64.(bounds)
	end
	nx, ny = max(1, round(Int, (east - west) / cellsize)), max(1, round(Int, (north - south) / cellsize))
	heights = fill(NaN, nx, ny)
	counts = zeros(Int, nx, ny)
	for p in chosen
		x, y, z = p.geometry
		i, j = floor(Int, (x - west) / cellsize) + 1, floor(Int, (y - south) / cellsize) + 1
		(1 <= i <= nx && 1 <= j <= ny) || continue
		counts[i, j] += 1
		(isnan(heights[i, j]) || z < heights[i, j]) && (heights[i, j] = z)
	end
	return (heights = heights, counts = counts, west = west, south = south, cellsize = Float64(cellsize))
end

"""
	find_ground(points, cellsize; method = :zhang, window = 20, slope = 0.15, threshold = 2.5, initial = 0.2, bounds = nothing)

The ground of `points` found again, whatever their classes, on cells of `cellsize` metres: the lowest point of each
cell but noise (see [`lowest_grid`](@ref)), kept where a morphological filter of Geomorphometry.jl takes it for
ground, the other cells `NaN`. `method = :zhang` is the progressive morphological filter of Zhang, Chen, Whitman,
Shyu, Yan and Zhang (2003), which opens the surface with windows growing to `window` metres and keeps the cells within
a height that grows with `slope` (metres a metre) from `initial` to `threshold` metres; `method = :pingel`, the simple
morphological filter of Pingel, Clarke and McBride (2013), with `window` and `slope` alone, slower. The `counts` of a
cell of ground are its points at most `initial` metres above its height, those of other cells 0. A grid as
[`ground_grid`](@ref) gives.
"""
function find_ground(points, cellsize::Real; method::Symbol = :zhang, window::Real = 20, slope::Real = 0.15,
		threshold::Real = 2.5, initial::Real = 0.2, bounds = nothing)
	lowest = lowest_grid(points, cellsize; bounds)
	h = lowest.heights
	known = .!isnan.(h)
	surface = copy(h)
	surface[.!known] .= maximum(h[known])  # an empty cell is set high, so that the filters never take it for ground
	if method === :zhang
		_, flags = Geomorphometry.progressive_morphological_filter(surface; ωₘ = Float64(window), slope = Float64(slope),
			dhₘ = Float64(threshold), dh₀ = Float64(initial), cellsize = lowest.cellsize)
		ground = known .& (flags .== 0)
	elseif method === :pingel
		kept = Geomorphometry.simple_morphological_filter(surface; ω = Float64(window), slope = Float64(slope), cellsize = lowest.cellsize)
		ground = known .& .!ismissing.(kept)
	else
		throw(ArgumentError("method is :zhang or :pingel"))
	end
	heights = ifelse.(ground, h, NaN)
	counts = zeros(Int, size(h))
	nx, ny = size(h)
	for p in points
		p.classification in NOISE_CLASSES && continue
		x, y, z = p.geometry
		i, j = floor(Int, (x - lowest.west) / lowest.cellsize) + 1, floor(Int, (y - lowest.south) / lowest.cellsize) + 1
		(1 <= i <= nx && 1 <= j <= ny && ground[i, j]) || continue
		z <= heights[i, j] + initial && (counts[i, j] += 1)
	end
	return merge(lowest, (heights = heights, counts = counts))
end

"""
	fill_gaps(grid; distance = 10, neighbours = 8)

The grid `grid` with its cells without a height filled from the `neighbours` nearest cells with one, weighted by the
inverse square of their distance, when the nearest is at most `distance` metres away; farther cells stay `NaN`, as no
height is made up there. Also `filled`, the cells that were filled, true where filled.
"""
function fill_gaps(grid; distance::Real = 10, neighbours::Integer = 8)
	h = grid.heights
	known = findall(!isnan, h)
	isempty(known) && throw(ArgumentError("the grid has no heights"))
	centre(I) = [grid.west + (I[1] - 0.5) * grid.cellsize, grid.south + (I[2] - 0.5) * grid.cellsize]
	tree = NearestNeighbors.KDTree(reduce(hcat, centre.(known)))
	z = h[known]
	k = min(neighbours, length(known))
	out = copy(h)
	filled = falses(size(h))
	for I in CartesianIndices(h)
		isnan(h[I]) || continue
		idx, d = NearestNeighbors.knn(tree, centre(I), k)
		minimum(d) <= distance || continue
		w = 1 ./ d .^ 2
		out[I] = sum(w .* z[idx]) / sum(w)
		filled[I] = true
	end
	return merge(grid, (heights = out, filled = filled))
end

"""
	ground_density(grid)

The ground points of each cell of `grid` per square metre: where few are, the terrain model is less sure, and where
none are, its height was filled from the cells around (see [`fill_gaps`](@ref)) or is missing.
"""
ground_density(grid) = grid.counts ./ grid.cellsize^2

"""
	terrain_raster(grid, crs)

The heights of `grid` as a raster of Rasters.jl in the coordinate system `crs` (as `EPSG(32615)` or a WKT), `missing`
where `NaN`; given by the extension for Rasters.jl.
"""
function terrain_raster end

"""
	index_tiles(index, bounds; field = "URL")

The tiles of the tile index `index`, a vector file of one polygon a tile (as the shapefile OpenTopography gives with a
survey), whose polygons cross `bounds = (xmin, ymin, xmax, ymax)`, in the coordinates of the index: the value of their
`field`, the address or file of each tile, for [`fetch_points`](@ref). Given by the extension for ArchGDAL.jl and
Rasters.jl.
"""
function index_tiles end
