# The multitemporal Sentinel-2 composite, read with ArchGDAL.jl from the Cloud-Optimised GeoTIFFs of Earth Search (only
# the cells of the area, each scene warped onto one grid of the UTM zone), and the cells of known sites from a vector file

module ArchaeologicalDetectionSentinelExt

using ArchaeologicalDetection, Rasters, ArchGDAL
import ArchaeologicalDetection: sentinel_composite, site_cells
using ArchaeologicalDetection: SENTINEL_BANDS, CLOUDY_CLASSES

const L = Rasters.Lookups

# Reading a Cloud-Optimised GeoTIFF over HTTPS: no listing of its folder, a few tries when the server is busy
const HTTP_OPTIONS = ("GDAL_DISABLE_READDIR_ON_OPEN" => "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS" => ".tif", "GDAL_HTTP_MAX_RETRY" => "4", "GDAL_HTTP_RETRY_DELAY" => "2", "GDAL_HTTP_MULTIRANGE" => "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES" => "YES")

# The EPSG code of the UTM zone of the centre of the box
function utm(box)
	lon, lat = (box[1] + box[3]) / 2, (box[2] + box[4]) / 2
	return (lat >= 0 ? 32600 : 32700) + clamp(floor(Int, (lon + 180) / 6) + 1, 1, 60)
end

# The box in the UTM zone, its edges taken at eleven points each, out to whole cells: west, south, east, north
function extent(box, epsg, cellsize)
	points = [[x, y] for x in range(box[1], box[3]; length = 11) for y in range(box[2], box[4]; length = 11) if x in (box[1], box[3]) || y in (box[2], box[4])]
	p = ArchGDAL.reproject(points, EPSG(4326), EPSG(epsg); order = :trad)
	xs, ys = first.(p), last.(p)
	return (floor(minimum(xs) / cellsize) * cellsize, floor(minimum(ys) / cellsize) * cellsize, ceil(maximum(xs) / cellsize) * cellsize, ceil(maximum(ys) / cellsize) * cellsize)
end

# The integers of the file at href on the grid of e and cellsize in epsg, cells of x by cells of y from the north-west
# corner, the 20 m bands taking the value of the nearest cell; 0 where the file has none
function warped(href, epsg, e, cellsize)
	ArchGDAL.read("/vsicurl/" * href) do source
		options = ["-t_srs", "EPSG:$(epsg)", "-te", string.(e)..., "-tr", string(cellsize), string(cellsize), "-r", "near", "-dstnodata", "0", "-of", "MEM"]
		ArchGDAL.gdalwarp([source], options) do w
			ArchGDAL.read(w, 1)
		end
	end
end

# Reads one scene onto the grid: the reflectance of each band and the cells that are clear, or nothing when no cell is
function scene_on_grid(scene, epsg, e, cellsize)
	clear = .!in.(warped(scene.assets["scl"].href, epsg, e, cellsize), Ref(CLOUDY_CLASSES))
	any(clear) || return nothing
	values = map(SENTINEL_BANDS) do band
		a = scene.assets[band]
		v = warped(a.href, epsg, e, cellsize)
		clear .&= v .> 0
		Float32.(v .* a.scale .+ a.offset)
	end
	return (values = values, clear = clear)
end

function sentinel_composite(scenes, box; path = nothing, cellsize::Real = 10)
	if path !== nothing && isfile(path)
		return replace_missing(read(Raster(path)), missing)
	end
	isempty(scenes) && error("No scenes: widen the years or raise the cloud limit of sentinel_scenes")
	epsg = utm(box)
	e = extent(box, epsg, cellsize)
	nx, ny, nb = round(Int, (e[3] - e[1]) / cellsize), round(Int, (e[4] - e[2]) / cellsize), length(SENTINEL_BANDS)
	total, counted = zeros(Float64, nx, ny, nb), zeros(Int32, nx, ny)
	lock, done, failed = ReentrantLock(), Threads.Atomic{Int}(0), String[]
	ArchGDAL.environment(; globalconfig = collect(HTTP_OPTIONS)) do
		Threads.@threads :dynamic for scene in scenes
			s = try
				scene_on_grid(scene, epsg, e, cellsize)
			catch err
				Base.@lock lock push!(failed, scene.id)
				nothing
			end
			if s !== nothing
				Base.@lock lock begin
					for b in 1:nb
						total[:, :, b] .+= ifelse.(s.clear, s.values[b], 0)
					end
					counted .+= s.clear
				end
			end
			k = Threads.atomic_add!(done, 1) + 1
			(k % 10 == 0 || k == length(scenes)) && println("Read ", k, " of ", length(scenes), " scenes")
		end
	end
	isempty(failed) || @warn "$(length(failed)) scenes could not be read and are left out: $(join(first(failed, 5), ", "))$(length(failed) > 5 ? ", ..." : "")"
	data = Array{Union{Missing, Float32}}(missing, nx, ny, nb)
	for b in 1:nb, j in 1:ny, i in 1:nx
		counted[i, j] > 0 && (data[i, j, b] = Float32(total[i, j, b] / counted[i, j]))
	end
	# With Y descending, Rasters puts each cell at its lower edge (as it reads a GeoTIFF of GDAL): the first is a cell below the top
	x = X(Rasters.Projected(range(e[1]; step = cellsize, length = nx); order = L.ForwardOrdered(), span = L.Regular(Float64(cellsize)), sampling = L.Intervals(L.Start()), crs = EPSG(epsg)))
	y = Y(Rasters.Projected(range(e[4] - cellsize; step = -cellsize, length = ny); order = L.ReverseOrdered(), span = L.Regular(-Float64(cellsize)), sampling = L.Intervals(L.Start()), crs = EPSG(epsg)))
	composite = Raster(data, (x, y, Band(1:nb)); missingval = missing, crs = EPSG(epsg), name = :reflectance)
	if path !== nothing
		mkpath(dirname(abspath(path)))
		write(path, replace_missing(composite, NaN32); missingval = NaN32)
	end
	println(count(>(0), counted), " of ", nx * ny, " cells clear in at least one scene, ", round(Int, sum(counted) / max(count(>(0), counted), 1)), " scenes a cell on average")
	return composite
end

function site_cells(composite::Rasters.AbstractRaster, sites::AbstractString; radius::Real = 0)
	grid = composite[Band(1)]
	target = crs(grid)
	isnothing(target) && error("The composite has no coordinate system")
	cellsize = abs(step(lookup(grid, X)))
	geometries = ArchGDAL.read(sites) do ds
		layer = ArchGDAL.getlayer(ds, 0)
		ref = ArchGDAL.getspatialref(layer)
		ref.ptr == C_NULL && error("$(sites) has no coordinate system")
		source = WellKnownText(ArchGDAL.toWKT(ref))
		map(layer) do feature
			g = ArchGDAL.reproject(ArchGDAL.clone(ArchGDAL.getgeom(feature)), source, target; order = :trad)
			area = ArchGDAL.getgeomtype(g) in (ArchGDAL.wkbPolygon, ArchGDAL.wkbMultiPolygon, ArchGDAL.wkbPolygon25D, ArchGDAL.wkbMultiPolygon25D)
			radius > 0 || !area ? ArchGDAL.buffer(g, max(radius, cellsize / 2)) : g  # a point or line takes at least half a cell
		end
	end
	isempty(geometries) && error("$(sites) has no features")
	return boolmask(geometries; to = grid)
end

end
