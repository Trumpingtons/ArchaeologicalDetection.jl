# ADAF's input, probability and features for a terrain model read with Rasters.jl: the results on its grid (missing
# where undefined), and the features at x and y

module ArchaeologicalDetectionRastersExt

using ArchaeologicalDetection, Rasters
import ArchaeologicalDetection: adaf_input, adaf_probability, adaf_features, forest_model, forest_probability, forest_features, enki_pooled, enki_examples, enki_probability, enki_features, terrain_raster, tumuli_input, tumuli_boxes, tumuli_features
using ArchaeologicalDetection: DarknetModel, EnkiModel, ENKI_CELL, enki_head, share_threshold, NNlib, Random

const L = Rasters.Lookups

steps(r) = (Float64(step(lookup(r, X))), Float64(step(lookup(r, Y))))
heights(r) = Float64.(coalesce.(parent(r), NaN))
on_grid(r, v) = rebuild(r; data = [isfinite(x) ? x : missing for x in v], missingval = missing)
centres(r, D) = collect(L.shiftlocus(L.Center(), lookup(r, D)))

function located(r, candidates)  # the positions i, j of the candidates turned into x, y, the centres of their cells
	x, y = centres(r, X), centres(r, Y)
	at(c, t) = c[1] + (t - 1) * (c[end] - c[1]) / max(length(c) - 1, 1)
	return [merge((id = c.id, x = at(x, c.i), y = at(y, c.j)), Base.structdiff(c, NamedTuple{(:id, :i, :j)})) for c in candidates]
end

function with_grid(r, found)  # candidates at x, y, and the matrices of the result as rasters on the grid of r
	extra = Dict(k => rebuild(r; data = v, missingval = nothing) for (k, v) in pairs(found) if k != :candidates)
	return merge((candidates = located(r, found.candidates),), NamedTuple(extra))
end

# ADAF reads a terrain model from its north-west corner: the matrices are turned north first, and back
north_first(r) = first(lookup(r, Y)) > last(lookup(r, Y))
turned(r, v) = north_first(r) ? v : reverse(v; dims = 2)

adaf_input(dem::Rasters.AbstractRaster) = on_grid(dem, turned(dem, adaf_input(turned(dem, heights(dem)), steps(dem))))

"""
    adaf_probability(model, dem::Raster; tile = 1024)

The probability of an archaeological feature at each cell of the terrain model `dem`, from ADAF's model `model`, on
the grid of `dem`: its local relief image, tiled from the north-west corner as ADAF tiles it. Missing where `dem` is.
"""
function adaf_probability(model, dem::Rasters.AbstractRaster; kw...)
	p = turned(dem, adaf_probability(model, turned(dem, Float32.(parent(adaf_input(dem)) .|> x -> coalesce(x, 0.0f0))); kw...))
	return on_grid(dem, ifelse.(isfinite.(heights(dem)), Float64.(p), NaN))
end

"""
    adaf_features(probability::Raster; kw...)

The features of the probability map `probability` (see `adaf_features(probability, cellsize)`), with their tiles
counted from the north-west corner, at `x` and `y`, and their cells as a raster `labels`.
"""
function adaf_features(probability::Rasters.AbstractRaster; kw...)
	ny = size(probability, Y)
	found = adaf_features(turned(probability, heights(probability)), steps(probability); kw...)
	if !north_first(probability)
		found = (candidates = [merge(c, (j = ny + 1 - c.j,)) for c in found.candidates], labels = reverse(found.labels; dims = 2))
	end
	return with_grid(probability, found)
end

"""
    tumuli_input(dem::Raster; soil = nothing, grey = TUMULI_GREY)

The image the tumulus detector reads from the terrain model `dem`, on its grid (see `tumuli_input(z, cellsize)`), 0
where `soil`, a raster on the grid of `dem`, is false, 0 or missing.
"""
function tumuli_input(dem::Rasters.AbstractRaster; soil = nothing, kw...)
	s = soil === nothing ? nothing : turned(dem, Float64.(coalesce.(parent(soil), 0)))
	return rebuild(dem; data = turned(dem, tumuli_input(turned(dem, heights(dem)), steps(dem); soil = s, kw...)), missingval = nothing)
end

"""
    tumuli_boxes(model, image::Raster; threshold = 0.25, overlap = 0.45)

The boxes of the tumuli that the detector finds in `image` (see `tumuli_input(dem)`), tiled from its north-west corner
as the authors tiled their terrain model, with the positions `i`, `j` of their centres on the grid of `image`.
"""
function tumuli_boxes(model::DarknetModel, image::Rasters.AbstractRaster; kw...)
	found = tumuli_boxes(model, turned(image, Float32.(coalesce.(parent(image), 0.0f0))); kw...)
	ny = size(image, Y)
	return north_first(image) ? found : [merge(b, (j = ny + 1 - b.j,)) for b in found]
end

"""
    tumuli_features(boxes, image::Raster; min_similarity = 0.25)

The tumuli of `boxes` (see `tumuli_boxes(model, image)`) of similarity at least `min_similarity`, as the authors listed
them: `candidates`, each with its `id`, `x` and `y`, the centre of its box, its `size` in metres, the larger side of
the box, and its `similarity`; `labels`, the cells of each box, labelled with its `id`, on the grid of `image`; and
`similarity`, the highest similarity of a box over each cell, 0 outside every box.
"""
function tumuli_features(boxes, image::Rasters.AbstractRaster; min_similarity::Real = 0.25)
	kept = filter(b -> b.similarity >= min_similarity, boxes)
	cells = ArchaeologicalDetection.tumuli_cells(kept, size(image, X), size(image, Y))
	grid = rebuild(image; data = cells.similarity, missingval = nothing, name = :similarity)
	return (candidates = located(image, tumuli_features(kept, abs(steps(image)[1]))),
		labels = rebuild(image; data = cells.labels, missingval = nothing, name = :labels), similarity = grid)
end

# The random forest on a composite read with Rasters.jl: its cells by bands, X and Y first
cells_by_bands(r) = parent(permutedims(r, (X, Y, Band)))

"""
    forest_model(composite::Raster, sites::Raster; kw...)

A random forest trained on the composite `composite` (see `sentinel_composite`) to tell the cells of `sites` (see
`site_cells`), on its grid, from the rest: see `forest_model(features, sites, cellsize)`.
"""
function forest_model(composite::Rasters.AbstractRaster, sites::Rasters.AbstractRaster; kw...)
	grid = composite[Band(1)]
	size(sites) == size(grid) || throw(DimensionMismatch("sites is not on the grid of the composite"))
	return forest_model(cells_by_bands(composite), Bool.(parent(permutedims(sites, (X, Y)))), abs(steps(grid)[1]); kw...)
end

"""
    forest_probability(model, composite::Raster)

The probability of a site at every cell of `composite`, from the random forest `model`, on its grid: missing where a
band is.
"""
forest_probability(model, composite::Rasters.AbstractRaster) = on_grid(permutedims(composite[Band(1)], (X, Y)), forest_probability(model, cells_by_bands(composite)))

"""
    forest_features(probability::Raster; threshold = 0.55, kw...)

The features of the probability map `probability` of a random forest, at `x` and `y`, and their cells as a raster
`labels`: see `forest_features(probability, cellsize)`.
"""
forest_features(probability::Rasters.AbstractRaster; threshold::Real = 0.55, kw...) = adaf_features(probability; threshold, margin = 0, kw...)

# Enki on an RGB image read with Rasters.jl: its pixels from the north-west corner, red, green and blue, as Enki reads
# them, kept in the image's own type (bytes for an image of aerial_image) and copied only when the axes must be turned
function north_first_rgb(image)
	size(image, Band) >= 3 || error("The image has $(size(image, Band)) bands: red, green and blue are needed")
	r = size(image, Band) == 3 ? image : image[Band(1:3)]
	a = Rasters.dimnum(r, (X, Y, Band)) == (1, 2, 3) ? parent(r) : parent(permutedims(r, (X, Y, Band)))
	return north_first(image) ? a : reverse(a; dims = 2)
end

pixel(v) = Float32(coalesce(v, 0))

# Whether the cell of side n from i, j (north-west corner) lies in the image and has an image over half of it at least
function covered(a, i, j, n)
	(i >= 1 && j >= 1 && i + n - 1 <= size(a, 1) && j + n - 1 <= size(a, 2)) || return false
	black = count(pixel(a[x, y, 1]) == 0 && pixel(a[x, y, 2]) == 0 && pixel(a[x, y, 3]) == 0 for y in j:j + n - 1, x in i:i + n - 1)
	return black <= n^2 / 2
end

# The pixels of the cell of side n from i, j, as Enki's 350 by 350
function window(a, i, j, n)
	w = pixel.(view(a, i:i + n - 1, j:j + n - 1, :))
	return n == ENKI_CELL ? w : NNlib.upsample_bilinear(reshape(w, n, n, 3, 1); size = (ENKI_CELL, ENKI_CELL))[:, :, :, 1]
end

# The pooled channels of the cells of side n from the corners (i, j), in batches: only a batch of images at a time
function pooled_batches(model, a, corners, n; batch, label = "images")
	f = Matrix{Float32}(undef, 1280, length(corners))
	t = time()
	for (k, chunk) in enumerate(Iterators.partition(eachindex(corners), batch))
		f[:, chunk] = enki_pooled(model, stack(window(a, corners[c]..., n) for c in chunk))
		done = last(chunk)
		(k % 10 == 0 || done == length(corners)) && println("Read ", done, " of ", length(corners), " ", label, " (", round(Int, time() - t), " s)")
	end
	return f
end

# The cell size of the image and its pixels to a cell of Enki's
function pixels_per_cell(image)
	cellsize = abs(steps(image)[1])
	n = round(Int, ENKI_CELL / cellsize)
	abs(n * cellsize - ENKI_CELL) <= cellsize / 2 || error("Cells of $(cellsize) m do not make a cell of $(ENKI_CELL) m")
	return cellsize, n
end

"""
    enki_pooled(model, image::Raster; batch = 16)

The pooled channels of Enki's network (see `enki_pooled(model, images)`) for every cell of 350 m of the RGB image
`image` (see `aerial_image`), cells counted from its north-west corner: a raster of cells of 350 m by `Band` 1 to 1,280,
missing where half a cell has no image. The images are read `batch` at a time. These are the most of Enki's time: the
head, trained or not, is then applied to them (see `enki_probability`).
"""
function enki_pooled(model::EnkiModel, image::Rasters.AbstractRaster; batch::Integer = 16)
	a = north_first_rgb(image)
	cellsize, n = pixels_per_cell(image)
	nx, ny = size(a, 1) ÷ n, size(a, 2) ÷ n
	(nx > 0 && ny > 0) || error("The image is smaller than a cell of $(ENKI_CELL) m")
	kept = [(i, j) for j in 1:ny for i in 1:nx if covered(a, (i - 1) * n + 1, (j - 1) * n + 1, n)]
	f = pooled_batches(model, a, [((i - 1) * n + 1, (j - 1) * n + 1) for (i, j) in kept], n; batch, label = "cells")
	data = Array{Union{Missing, Float32}}(missing, nx, ny, 1280)
	for (k, (i, j)) in enumerate(kept)
		data[i, j, :] = f[:, k]
	end
	side = n * cellsize
	west, top = Rasters.bounds(image, X)[1], Rasters.bounds(image, Y)[2]
	x = X(Rasters.Projected(range(west; step = side, length = nx); order = L.ForwardOrdered(), span = L.Regular(Float64(side)), sampling = L.Intervals(L.Start()), crs = crs(image)))
	y = Y(Rasters.Projected(range(top - side; step = -side, length = ny); order = L.ReverseOrdered(), span = L.Regular(-Float64(side)), sampling = L.Intervals(L.Start()), crs = crs(image)))
	return Raster(data, (x, y, Band(1:1280)); missingval = missing, crs = crs(image), name = :pooled)
end

"""
    enki_probability(model, cells::Raster)

The share Enki's network `model` gives each cell of 350 m for "positive", a tell or mound-like feature, on the grid of
`cells`: the pooled channels of `enki_pooled` (`Band` 1 to 1,280), or an RGB image (three bands), read first with
`enki_pooled`. Missing where a cell has no image.
"""
function enki_probability(model::EnkiModel, cells::Rasters.AbstractRaster)
	size(cells, Band) == 1280 || return enki_probability(model, enki_pooled(model, cells))
	c = permutedims(cells, (X, Y, Band))
	a = parent(c)
	nx, ny = size(a, 1), size(a, 2)
	v = fill(NaN, nx, ny)
	kept = [I for I in CartesianIndices((nx, ny)) if !ismissing(a[I, 1])]
	if !isempty(kept)
		p = enki_head(model.hidden, model.output, Float32.(stack(a[I, :] for I in kept)))
		v[kept] = p
	end
	return on_grid(c[Band(1)], v)
end

function enki_examples(model::EnkiModel, image::Rasters.AbstractRaster, cells::Rasters.AbstractRaster, sites; shift::Real = 50, negatives::Real = 5, distance::Real = 700, seed::Integer = 1, batch::Integer = 16)
	isempty(sites) && error("No known sites inside the image")
	a = north_first_rgb(image)
	cellsize, n = pixels_per_cell(image)
	west, top = Rasters.bounds(image, X)[1], Rasters.bounds(image, Y)[2]
	half = ENKI_CELL / 2
	corners = Tuple{Int, Int}[]
	for (sx, sy) in sites, dy in (-shift, 0, shift), dx in (-shift, 0, shift)
		i = floor(Int, (sx + dx - half - west) / cellsize) + 1
		j = floor(Int, (top - (sy + dy + half)) / cellsize) + 1
		covered(a, i, j, n) && push!(corners, (i, j))
	end
	isempty(corners) && error("No image of 350 m around a known site lies inside the image")
	positives = pooled_batches(model, a, corners, n; batch, label = "images of sites")
	c = permutedims(cells, (X, Y, Band))
	p = parent(c)
	xs, ys = centres(c, X), centres(c, Y)
	away = [I for I in CartesianIndices((size(p, 1), size(p, 2))) if !ismissing(p[I, 1]) && all(hypot(xs[I[1]] - sx, ys[I[2]] - sy) >= distance for (sx, sy) in sites)]
	wanted = min(round(Int, negatives * length(corners)), length(away))
	wanted > 0 || error("No cell is $(distance) m from every known site")
	drawn = away[Random.randperm(Random.Xoshiro(seed), length(away))[1:wanted]]
	others = Float32.(stack(p[I, :] for I in drawn))
	println(length(corners), " images of sites, ", wanted, " cells away from them")
	return (pooled = hcat(positives, others), labels = [trues(length(corners)); falses(wanted)], positives = length(corners), negatives = wanted)
end

"""
    enki_features(probability::Raster; threshold = 0.99, share = nothing, kw...)

The features of Enki's map of shares `probability` (see `enki_probability`), at `x` and `y`, and their cells as a
raster `labels`: see `enki_features(probability, cellsize)`, also for `share`.
"""
enki_features(probability::Rasters.AbstractRaster; threshold::Real = 0.99, share::Union{Nothing, Real} = nothing, kw...) =
	adaf_features(probability; threshold = isnothing(share) ? threshold : share_threshold(probability, share), margin = 0, kw...)

# A grid of a point cloud (x first, from the south-west corner) as a raster north up, as GeoTIFFs usually are, its
# cells by their centres
function terrain_raster(grid, crs)
	nx, ny = size(grid.heights)
	c = grid.cellsize
	north = grid.south + ny * c
	x = X(L.Sampled(range(grid.west + c / 2; step = c, length = nx); sampling = L.Intervals(L.Center()), order = L.ForwardOrdered(), span = L.Regular(c)))
	y = Y(L.Sampled(range(north - c / 2; step = -c, length = ny); sampling = L.Intervals(L.Center()), order = L.ReverseOrdered(), span = L.Regular(-c)))
	data = [isnan(v) ? missing : Float32(v) for v in reverse(grid.heights; dims = 2)]
	return Raster(data, (x, y); crs = crs isa AbstractString ? WellKnownText(crs) : crs, missingval = missing)
end

end
