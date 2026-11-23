# ADAF's input, probability and features for a terrain model read with Rasters.jl: the results on its grid (missing
# where undefined), and the features at x and y

module ArchaeologicalDetectionRastersExt

using ArchaeologicalDetection, Rasters
import ArchaeologicalDetection: adaf_input, adaf_probability, adaf_features, forest_model, forest_probability, forest_features

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

end
