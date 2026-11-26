# Detection in a multitemporal Sentinel-2 composite with a random forest trained on known sites, as Orengo and 6
# coauthors (2020) found mounds in Cholistan: the mean reflectance of many scenes, each without its cloudy cells, at
# every cell of 10 m, a random forest of 128 trees trained on the cells of known sites and cells away from them, and
# the share of the trees that vote for a site at every cell. The scenes are those of Level-2A of Earth Search, in
# Cloud-Optimised GeoTIFFs, or of Level-1C of Google's public copy of the archive, in JPEG 2000, over HTTPS and read
# without an account; the composite itself is made by the extension for Rasters.jl and ArchGDAL.jl.

const EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search"

"""The bands of the composite, as Orengo and 6 coauthors (2020) took them: B2, B3, B4, B5, B6, B7, B8, B8A, B11 and B12,
by their names in Earth Search."""
const SENTINEL_BANDS = ("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22")

"""
    sentinel_scenes(box; years, cloud = 30, most = 1000, level = :l2a)

The Sentinel-2 scenes over `box = (west, south, east, north)`, in degrees, acquired in `years` (as `2023:2025`), with
at most `cloud` per cent of clouds, read without an account; at most `most` of them, the clearest first.

With `level = :l2a`, the scenes of Level-2A, surface reflectance, from Earth Search's catalogue, from November 2016,
with at most `cloud` per cent of clouds over the whole scene. With `level = :l1c`, the scenes of Level-1C, reflectance
at the top of the atmosphere, from June 2015, from Google's public copy of the Sentinel-2 archive, with at most
`cloud` per cent of cloudy cells in the tile, as Earth Engine's `CLOUDY_PIXEL_PERCENTAGE` gives it; an acquisition
processed more than once is taken as it was processed first. Their files are JPEG 2000, of which only the blocks over
the area are read.

Each scene has its `id`, `acquired`, `cloud`, `epsg` (its coordinate system) and `assets`: for each band of
[`SENTINEL_BANDS`](@ref), the address of its file with its `scale` and `offset`, by which the stored integers become
reflectance; and the cloud mask, the scene classification `scl` for Level-2A, the mask `clouds` of opaque clouds and
cirrus for Level-1C.
"""
function sentinel_scenes(box; years, cloud::Real = 30, most::Integer = 1000, level::Symbol = :l2a)
	length(box) == 4 && box[1] < box[3] && box[2] < box[4] || throw(ArgumentError("box is (west, south, east, north), in degrees"))
	level === :l1c && return l1c_scenes(box, years, cloud, most)
	level === :l2a || throw(ArgumentError("level is :l2a or :l1c"))
	days = "$(first(years))-01-01T00:00:00Z/$(last(years))-12-31T23:59:59Z"
	url = EARTH_SEARCH * "?collections=sentinel-2-l2a&limit=100&bbox=" * join(box, ",") * "&datetime=" * escapeuri(days)
	scenes = NamedTuple[]
	while url !== nothing
		page = JSON.parse(String(take!(paged(url))))
		for f in page["features"]
			p, a = f["properties"], f["assets"]
			c = get(p, "eo:cloud_cover", nothing)
			(c === nothing || c > cloud) && continue
			all(k -> haskey(a, k) && startswith(a[k]["href"], "https://"), (SENTINEL_BANDS..., "scl")) || continue  # a few scenes have their files only where an account is needed
			applied = get(p, "earthsearch:boa_offset_applied", false) === true  # the stored integers already have the offset of processing baseline 04.00 taken off
			assets = Dict(k => (href = a[k]["href"], scale = band_value(a[k], "scale", 1.0), offset = applied ? 0.0 : band_value(a[k], "offset", 0.0)) for k in (SENTINEL_BANDS..., "scl"))
			push!(scenes, (id = f["id"], acquired = p["datetime"], cloud = Float64(c), epsg = Int(p["proj:epsg"]), assets = assets))
		end
		next = [l["href"] for l in page["links"] if l["rel"] == "next"]
		url = isempty(next) ? nothing : first(next)
	end
	unique!(s -> s.id, scenes)
	sort!(scenes; by = s -> s.cloud)
	return scenes[1:min(most, end)]
end

# Level-1C from Google's public copy of the Sentinel-2 archive, read over HTTPS without an account
const SENTINEL_ARCHIVE = "https://storage.googleapis.com/gcp-public-data-sentinel-2"
const ARCHIVE_LISTING = "https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o"
const L1C_FILES = Dict("blue" => "B02", "green" => "B03", "red" => "B04", "rededge1" => "B05", "rededge2" => "B06",
	"rededge3" => "B07", "nir" => "B08", "nir08" => "B8A", "swir16" => "B11", "swir22" => "B12")
const L1C_PRODUCT = r"(?<sensed>\d{8}T\d{6})_N(?<baseline>\d{4})_R(?<orbit>\d{3})_T(?<tile>\w{5})_(?<processed>\d{8}T\d{6})\.SAFE/?$"

# The names under prefix in the archive: the objects, or with folders = true the folders one level down
function archive_names(prefix; folders::Bool = false)
	names, token = String[], nothing
	while true
		url = ARCHIVE_LISTING * "?prefix=" * escapeuri(prefix) * (folders ? "&delimiter=%2F&fields=prefixes,nextPageToken" : "&fields=items(name),nextPageToken") *
			(token === nothing ? "" : "&pageToken=" * escapeuri(token))
		page = JSON.parse(String(take!(retried(() -> Downloads.download(url, IOBuffer(); timeout = 120)))))
		append!(names, folders ? get(page, "prefixes", String[]) : [i["name"] for i in get(page, "items", [])])
		token = get(page, "nextPageToken", nothing)
		token === nothing && return names
	end
end

# The tiles of Sentinel-2's grid over the box, as 29TNG, from the Level-2A scenes of Earth Search of a year: the grid is
# the same at both levels
function sentinel_tiles(box)
	url = EARTH_SEARCH * "?collections=sentinel-2-l2a&limit=100&bbox=" * join(box, ",") * "&datetime=" * escapeuri("2020-01-01T00:00:00Z/2020-12-31T23:59:59Z")
	tiles = Set{String}()
	while url !== nothing
		page = JSON.parse(String(take!(paged(url))))
		for f in page["features"]
			p = f["properties"]
			push!(tiles, string(p["mgrs:utm_zone"], p["mgrs:latitude_band"], p["mgrs:grid_square"]))
		end
		next = [l["href"] for l in page["links"] if l["rel"] == "next"]
		url = isempty(next) ? nothing : first(next)
	end
	isempty(tiles) && error("No tile of Sentinel-2 covers the box $(box)")
	return sort!(collect(tiles))
end

function l1c_scenes(box, years, cloud, most)
	products = String[]
	for tile in sentinel_tiles(box)
		append!(products, archive_names("tiles/$(parse(Int, tile[1:end-3]))/$(tile[end-2])/$(tile[end-1:end])/"; folders = true))
	end
	found = Dict{Tuple{String, String, String}, Tuple{String, String}}()  # each acquisition, by day, orbit and tile, as processed first
	for p in products
		m = match(L1C_PRODUCT, p)
		(m === nothing || parse(Int, m[:sensed][1:4]) ∉ years) && continue
		key = (m[:sensed][1:8], m[:orbit], m[:tile])
		(!haskey(found, key) || m[:processed] < found[key][2]) && (found[key] = (p, m[:processed]))
	end
	scenes = asyncmap(p -> l1c_scene(p), [first(v) for v in values(found)]; ntasks = 16)
	scenes = [s for s in scenes if s !== nothing && s.cloud <= cloud]
	sort!(scenes; by = s -> s.cloud)
	return scenes[1:min(most, end)]
end

# A scene of Level-1C from the files of its product: nothing when a band or the cloud mask is not there
function l1c_scene(product)
	m = match(L1C_PRODUCT, product)
	names = archive_names(product)
	file(suffix) = (k = findfirst(n -> endswith(n, suffix), names); k === nothing ? nothing : SENTINEL_ARCHIVE * "/" * names[k])
	metadata = file("/MTD_TL.xml")
	metadata === nothing && return nothing
	xml = String(take!(retried(() -> Downloads.download(metadata, IOBuffer(); timeout = 120))))
	c = match(r"<CLOUDY_PIXEL_PERCENTAGE>(?<c>[^<]+)<", xml)
	e = match(r"EPSG:(?<epsg>\d+)", xml)
	(c === nothing || e === nothing) && return nothing
	offset = parse(Int, m[:baseline]) >= 400 ? -0.1 : 0.0  # from processing baseline 04.00 the stored integers are 1,000 above
	assets = Dict{String, Any}()
	for (band, name) in L1C_FILES
		href = file("_$(name).jp2")
		href === nothing && return nothing
		assets[band] = (href = href, scale = 1.0e-4, offset = offset)
	end
	gml, classes = file("/MSK_CLOUDS_B00.gml"), file("/MSK_CLASSI_B00.jp2")  # the mask before processing baseline 04.00, and from it
	gml === nothing && classes === nothing && return nothing
	assets["clouds"] = gml === nothing ? (href = classes, kind = :raster) : (href = gml, kind = :gml)
	s = m[:sensed]
	acquired = "$(s[1:4])-$(s[5:6])-$(s[7:8])T$(s[10:11]):$(s[12:13]):$(s[14:15])Z"
	return (id = splitext(basename(rstrip(product, '/')))[1], acquired = acquired, cloud = parse(Float64, c[:c]), epsg = parse(Int, e[:epsg]), assets = assets)
end

# A page of Earth Search's results: a page that fails each time it is asked, as some pages of 100 scenes do with 502,
# is asked again with half as many scenes, down to 25
function paged(url)
	try
		return retried(() -> Downloads.download(url, IOBuffer(); timeout = 120))
	catch e
		n = match(r"limit=(?<n>\d+)", url)
		(e isa Downloads.RequestError && n !== nothing && parse(Int, n[:n]) > 25) || rethrow()
		return paged(replace(url, r"limit=\d+" => "limit=$(parse(Int, n[:n]) ÷ 2)"))
	end
end

# Earth Search answers now and then with an error of its own, such as 502; asked again a little later, it answers
function retried(f; tries = 4)
	for k in 1:tries
		try
			return f()
		catch e
			(e isa Downloads.RequestError && k < tries) || rethrow()
			sleep(2^k)
		end
	end
end

band_value(asset, key, default) = Float64(get(first(get(asset, "raster:bands", [Dict()])), key, default))

escapeuri(s) = join(c in ('-', '.', '_', '~') || isletter(c) || isdigit(c) ? string(c) : join("%" .* uppercase.(string.(codeunits(string(c)); base = 16, pad = 2))) for c in s)

"""
    lonlat_box(raster)

The box of `raster`, a `Raster` of Rasters.jl in any coordinate system, in degrees: `(west, south, east, north)`, as
[`sentinel_scenes`](@ref) and [`sentinel_composite`](@ref) take it. Needs Rasters.jl and ArchGDAL.jl.
"""
function lonlat_box end

"""The classes of Sentinel-2's scene classification a cell of a scene is left out of the composite for: no data (0),
saturated or defective (1), cloud shadow (3), cloud of medium and of high probability (8, 9) and thin cirrus (10)."""
const CLOUDY_CLASSES = (0, 1, 3, 8, 9, 10)

"""
    sentinel_composite(scenes, box; path = nothing, cellsize = 10)

The multitemporal composite of `scenes` (see [`sentinel_scenes`](@ref)) over `box = (west, south, east, north)`, in
degrees: at every cell of `cellsize` metres, in the UTM zone of the centre of the box, the mean reflectance of each band
of [`SENTINEL_BANDS`](@ref) over the scenes where the cell is clear, and `missing` where it is clear in none. A cell is
clear where the scene classification says so, for Level-2A (see [`CLOUDY_CLASSES`](@ref)), and outside the opaque
clouds and cirrus of the cloud mask, for Level-1C, as Earth Engine's band `QA60` gives them. The 20 m bands take the value of the nearest cell.
Only the cells of the box are read from each scene, a few seconds a scene, several scenes at once when Julia has
several threads. With `path`, the composite is written there as a GeoTIFF and read from there later, whatever the
scenes. Needs Rasters.jl and ArchGDAL.jl.
"""
function sentinel_composite end

"""
    site_cells(composite, sites; radius = 0)

The cells of the composite (see [`sentinel_composite`](@ref)) that are known sites: inside a polygon of `sites`, a
vector file (as a GeoPackage, GeoJSON or Shapefile) in any coordinate system, or within `radius` metres of a point or
line of it. Needs Rasters.jl and ArchGDAL.jl.
"""
function site_cells end

"""
    forest_model(features, sites, cellsize; trees = 128, negatives = 10, distance = 500, seed = 1)

A random forest of `trees` trees (128 in Orengo and 6 coauthors, 2020), trained to tell the cells of known sites from
the rest: `features` is an array of cells by bands (as the composite of [`sentinel_composite`](@ref)),
`sites` a matrix of the same cells, true at the known sites (see [`site_cells`](@ref)), and `cellsize` the width of a
cell in metres. Every cell of a site is a positive; `negatives` times as many cells, drawn at random at least `distance`
metres from every site, are the negatives. Each tree takes half the cells, drawn with replacement, and the square root
of the number of bands at each split, as Earth Engine's random forest does by default. Cells with a band missing are left out.
"""
function forest_model(features::AbstractArray{<:Union{Missing, Real}, 3}, sites::AbstractMatrix{Bool}, cellsize::Real; trees::Integer = 128, negatives::Real = 10, distance::Real = 500, seed::Integer = 1)
	size(sites) == size(features)[1:2] || throw(DimensionMismatch("sites has $(size(sites)) cells, features $(size(features)[1:2])"))
	clear = [all(b -> valid(features[i, j, b]), axes(features, 3)) for i in axes(sites, 1), j in axes(sites, 2)]
	positive = findall(sites .& clear)
	isempty(positive) && error("No cell of a known site has all its bands")
	near = nearby(sites, ceil(Int, distance / cellsize))
	pool = findall(.!near .& clear)
	n = min(length(pool), round(Int, negatives * length(positive)))
	n == 0 && error("No cell is $(distance) m from every known site: take a larger area")
	rng = Random.Xoshiro(seed)
	negative = pool[Random.randperm(rng, length(pool))[1:n]]
	cells = [positive; negative]
	x = Float64[features[c, b] for c in cells, b in axes(features, 3)]
	y = [fill(1, length(positive)); fill(0, n)]
	forest = DecisionTree.build_forest(y, x, -1, trees, 0.5; rng = Random.Xoshiro(seed))
	return (forest = forest, positives = length(positive), negatives = n, bands = size(features, 3))
end

valid(v) = !ismissing(v) && isfinite(v)

# The cells within r cells of a true cell of m, by a square window: r cells along each axis
function nearby(m::AbstractMatrix{Bool}, r::Integer)
	rows = [any(m[max(1, i - r):min(end, i + r), j]) for i in axes(m, 1), j in axes(m, 2)]
	return [any(rows[i, max(1, j - r):min(end, j + r)]) for i in axes(m, 1), j in axes(m, 2)]
end

"""
    forest_probability(model, features)

The probability of a site at every cell of `features` (cells by bands, as the composite of
[`sentinel_composite`](@ref)), from the random forest `model` (see [`forest_model`](@ref)): the share of its trees
that vote for a site, as Earth Engine's probability mode gives it. `NaN` where a band is missing.
"""
function forest_probability(model, features::AbstractArray{<:Union{Missing, Real}, 3})
	size(features, 3) == model.bands || throw(DimensionMismatch("the model was trained on $(model.bands) bands, features has $(size(features, 3))"))
	p = fill(NaN, size(features)[1:2])
	cells = [c for c in CartesianIndices(p) if all(b -> valid(features[c, b]), axes(features, 3))]
	isempty(cells) && return p
	x = Float64[features[c, b] for c in cells, b in axes(features, 3)]
	p[cells] = DecisionTree.apply_forest_proba(model.forest, x, [0, 1])[:, 2]
	return p
end

"""
    forest_features(probability, cellsize; threshold = 0.55, min_area = 0, min_roundness = 0)

The features of the probability map of a random forest (see [`forest_probability`](@ref)): the cells of probability
at least `threshold` (0.55 in Orengo and 6 coauthors, 2020) joined into regions through the sides of the cells, each with
its `id`, the position `i`, `j` of its centre, its `area`, its `roundness` and its highest `probability`, as
[`adaf_features`](@ref) measures them, those of area up to `min_area` or roundness up to `min_roundness` left out.
Returns `candidates` and `labels`, their cells.
"""
forest_features(probability::AbstractMatrix{<:Real}, cellsize; threshold::Real = 0.55, kw...) = adaf_features(probability, cellsize; threshold, margin = 0, kw...)
