# Enki (Rosati 2025), a classifier of RGB satellite imagery for tells and mound-like archaeological features of the
# Ancient Near East: EfficientNet-B0 of Keras (Tan and Le 2019) with a head of its own, that gives an image of a cell of
# 350 by 350 m, at 1 m, its share for "positive". The network is rebuilt here with NNlib.jl, each batch normalisation
# folded into the convolution before it, from Enki's published Keras file: its config.json names the layers and their
# order, its model.weights.h5 (read with JLD2.jl, without Python) holds the weights. The head can be trained again on the
# user's known sites, the network beneath it unchanged, as the first stage of Enki's own training does.

const ENKI_FILE = "https://raw.githubusercontent.com/archeorosati/enki/e47ea9a52986a9926c51ae68bb02a37a7a63353e/model_enki/enki0_0.keras"

"""The side of Enki's cells, in metres and in pixels: an image of 350 by 350 m at 1 m."""
const ENKI_CELL = 350

"""
    enki_weights(folder)

The file of Enki's trained model, `enki0_0.keras` (about 52 MB), downloaded once from its GitHub repository to
`folder/enki` and returned from there later.
"""
enki_weights(folder::AbstractString) = cached(ENKI_FILE, joinpath(folder, "enki", "enki0_0.keras"))

# A convolution, or a depthwise convolution, with the batch normalisation after it folded in: weights (kw, kh, in, out),
# or (kw, kh, 1, channels), as NNlib takes them; padding (left, right, top, bottom)
struct EnkiConv
	weight::Array{Float32, 4}
	bias::Vector{Float32}
	stride::Int
	pad::NTuple{4, Int}
	depthwise::Bool
end
function (c::EnkiConv)(x)
	y = c.depthwise ? NNlib.depthwiseconv(x, c.weight; stride = c.stride, pad = c.pad, flipped = true) :
		NNlib.conv(x, c.weight; stride = c.stride, pad = c.pad, flipped = true)
	return y .+ reshape(c.bias, 1, 1, :, 1)
end

# A mobile inverted bottleneck: expansion, depthwise convolution, squeeze and excitation, projection, and the input
# added back when the block keeps its size
struct MBConv
	expand::Union{EnkiConv, Nothing}
	depthwise::EnkiConv
	reduce::Tuple{Matrix{Float32}, Vector{Float32}}
	excite::Tuple{Matrix{Float32}, Vector{Float32}}
	project::EnkiConv
	skip::Bool
end
function (b::MBConv)(x)
	y = b.expand === nothing ? x : NNlib.swish.(b.expand(x))
	y = NNlib.swish.(b.depthwise(y))
	s = dropdims(mean(y; dims = (1, 2)); dims = (1, 2))  # channels by images
	s = NNlib.swish.(b.reduce[1] * s .+ b.reduce[2])
	s = NNlib.sigmoid.(b.excite[1] * s .+ b.excite[2])
	y = b.project(y .* reshape(s, 1, 1, size(s)...))
	return b.skip ? y .+ x : y
end

"""
Enki's network: the preprocessing of Keras's EfficientNet-B0 (`mean`, `deviation`), its stem, sixteen blocks and top
convolution, and the head, two dense layers (`hidden`, `output`) on the average of the top's 1,280 channels. See
[`enki_model`](@ref).
"""
struct EnkiModel
	mean::Vector{Float32}
	deviation::Vector{Float32}
	stem::EnkiConv
	blocks::Vector{MBConv}
	top::EnkiConv
	hidden::Tuple{Matrix{Float32}, Vector{Float32}}
	output::Tuple{Matrix{Float32}, Vector{Float32}}
end

# The weights of a Keras 3 file, by the names of config.json: model.weights.h5 names a layer by its class, numbered
# in the order of the layers of that class (conv2d, conv2d_1...)
function keras_layers(config, weights)
	inner = only(l for l in config["config"]["layers"] if l["class_name"] == "Functional")
	seen = Dict{String, Int}()
	layers = Dict{String, Any}()
	for l in inner["config"]["layers"]
		class = lowercase(replace(replace(l["class_name"], r"(.)([A-Z][a-z]+)" => s"\1_\2"), r"([a-z])([A-Z])" => s"\1_\2"))  # Keras's snake case
		k = get(seen, class, 0)
		seen[class] = k + 1
		key = "layers/functional/layers/" * (k == 0 ? class : "$(class)_$(k)") * "/vars"
		vars = haskey(weights, key) ? [Float32.(weights["$(key)/$(i)"]) for i in 0:length(keys(weights[key])) - 1] : []
		layers[l["config"]["name"]] = (class = l["class_name"], config = l["config"], vars = vars)
	end
	return layers
end

# A convolution of Keras's (read reversed: out, in, kw, kh; depthwise 1, channels, kw, kh) and its batch normalisation
function enki_conv(layers, conv, bn; pad = nothing)
	c = layers[conv]
	W = c.vars[1]
	depthwise = c.class == "DepthwiseConv2D"
	K = depthwise ? permutedims(W, (3, 4, 1, 2)) : permutedims(W, (3, 4, 2, 1))
	gamma, beta, mu, var = layers[bn].vars
	s = gamma ./ sqrt.(var .+ Float32(layers[bn].config["epsilon"]))
	K = K .* reshape(s, 1, 1, 1, :)  # the output channels are last in both kinds
	stride = c.config["strides"][1]
	k = c.config["kernel_size"][1]
	p = pad === nothing ? (k ÷ 2, k ÷ 2, k ÷ 2, k ÷ 2) : pad
	return EnkiConv(K, beta .- mu .* s, stride, p, depthwise)
end

# The padding of a ZeroPadding2D, ((top, bottom), (left, right)), as (left, right, top, bottom)
padding(layer) = (layer.config["padding"][2][1], layer.config["padding"][2][2], layer.config["padding"][1][1], layer.config["padding"][1][2])

dense(layer) = (layer.vars[1], layer.vars[2])  # Keras's kernel (in, out) read reversed: (out, in)
squeeze_conv(layer) = (reshape(layer.vars[1], size(layer.vars[1], 1), size(layer.vars[1], 2)), layer.vars[2])

"""
    enki_model(path)

Enki's trained network read from `path`, its Keras file (see [`enki_weights`](@ref)), without Python: the layers and
their order from its `config.json`, the weights from its `model.weights.h5`.
"""
function enki_model(path::AbstractString)
	zip = ZipArchives.ZipReader(read(path))
	config = JSON.parse(String(ZipArchives.zip_readentry(zip, "config.json")))
	h5 = tempname() * ".h5"
	write(h5, ZipArchives.zip_readentry(zip, "model.weights.h5"))
	model = try
		JLD2.jldopen(h5, "r") do weights
			layers = keras_layers(config, weights)
			mu, var = layers["normalization"].vars[1], layers["normalization"].vars[2]
			scale = Float32.(layers["rescaling_1"].config["scale"])
			deviation = max.(sqrt.(var), 1.0f-7) ./ scale  # Keras's Normalization, then the Rescaling after it
			stem = enki_conv(layers, "stem_conv", "stem_bn"; pad = padding(layers["stem_conv_pad"]))
			names = sort(unique(m.match for m in eachmatch(r"block\d+[a-z]", join(keys(layers), " "))))
			blocks = map(names) do p
				expand = haskey(layers, "$(p)_expand_conv") ? enki_conv(layers, "$(p)_expand_conv", "$(p)_expand_bn") : nothing
				pad = haskey(layers, "$(p)_dwconv_pad") ? padding(layers["$(p)_dwconv_pad"]) : nothing
				MBConv(expand, enki_conv(layers, "$(p)_dwconv", "$(p)_bn"; pad), squeeze_conv(layers["$(p)_se_reduce"]),
					squeeze_conv(layers["$(p)_se_expand"]), enki_conv(layers, "$(p)_project_conv", "$(p)_project_bn"), haskey(layers, "$(p)_add"))
			end
			top = enki_conv(layers, "top_conv", "top_bn")
			head(name) = (Array{Float32}(weights["layers/$(name)/vars/0"]), Array{Float32}(weights["layers/$(name)/vars/1"]))
			EnkiModel(mu, deviation, stem, blocks, top, head("dense"), head("dense_1"))
		end
	finally
		rm(h5; force = true)
	end
	return model
end

"""
    enki_pooled(model, images)

The 1,280 channels of the top of Enki's network averaged over each image: `images` is an array of x by y by the
three bands red, green, blue by images, in values from 0 to 255, x from west to east and y from north to south.
Returns 1,280 by images.
"""
function enki_pooled(m::EnkiModel, images::AbstractArray{<:Real, 4})
	x = (Float32.(images) ./ 255.0f0 .- reshape(m.mean, 1, 1, 3, 1)) ./ reshape(m.deviation, 1, 1, 3, 1)
	x = NNlib.swish.(m.stem(x))
	for b in m.blocks
		x = b(x)
	end
	x = NNlib.swish.(m.top(x))
	return dropdims(mean(x; dims = (1, 2)); dims = (1, 2))
end

# The head on pooled channels: the share of each image for "positive", the second of Enki's two classes
enki_head(hidden, output, f) = NNlib.softmax(output[1] * NNlib.relu.(hidden[1] * f .+ hidden[2]) .+ output[2]; dims = 1)[2, :]

"""
    enki_probability(model, images)

The share Enki's network gives each image for "positive", a tell or mound-like feature: `images` as for
[`enki_pooled`](@ref), each 350 by 350 pixels of 1 m as Enki reads them.
"""
enki_probability(m::EnkiModel, images::AbstractArray{<:Real, 4}) = enki_head(m.hidden, m.output, enki_pooled(m, images))

"""
    enki_finetune(model, pooled, labels; epochs = 15, rate = 1e-4, batch = 128, dropout = 0.3, validation = 0.2, seed = 1)

Enki's network with its head trained again on the user's images, the network beneath it unchanged, as the first stage
of Enki's own training trains its head: `pooled` are the images' channels (see [`enki_pooled`](@ref)), `labels` true
for the images of sites. The head starts from Enki's trained weights and is trained with Adam at the learning rate
`rate` on the cross-entropy of its two classes, in batches of `batch` images, with a dropout of `dropout` on the
channels. A share `validation` of the images, drawn at random, is kept out of training; after each epoch the head is
scored on them, and the head of the lowest loss on them is kept. Returns the model, with `history`, the loss and the
accuracy of each epoch on the images trained on and on those kept out.
"""
function enki_finetune(m::EnkiModel, pooled::AbstractMatrix{<:Real}, labels::AbstractVector{Bool}; epochs::Integer = 15, rate::Real = 1e-4, batch::Integer = 128, dropout::Real = 0.3, validation::Real = 0.2, seed::Integer = 1)
	n = length(labels)
	size(pooled, 2) == n || throw(DimensionMismatch("$(size(pooled, 2)) images, $(n) labels"))
	any(labels) && !all(labels) || error("Fine-tuning needs images of sites and images of other ground")
	rng = Random.Xoshiro(seed)
	order = Random.randperm(rng, n)
	held = order[1:round(Int, validation * n)]
	train = order[round(Int, validation * n) + 1:end]
	F, y = Float32.(pooled), [l ? 2 : 1 for l in labels]
	params = Any[copy(m.hidden[1]), copy(m.hidden[2]), copy(m.output[1]), copy(m.output[2])]
	moments = [(zero(p), zero(p)) for p in params]
	step = 0
	scored(ps, idx) = begin
		p = NNlib.softmax(ps[3] * NNlib.relu.(ps[1] * F[:, idx] .+ ps[2]) .+ ps[4]; dims = 1)
		(loss = -mean(log(max(p[y[i], k], 1.0f-7)) for (k, i) in enumerate(idx)), accuracy = mean((p[2, k] > 0.5) == (y[i] == 2) for (k, i) in enumerate(idx)))
	end
	best, kept, history = Inf, deepcopy(params), NamedTuple[]
	for epoch in 1:epochs
		shuffled = train[Random.randperm(rng, length(train))]
		for first in 1:batch:length(shuffled)
			idx = shuffled[first:min(first + batch - 1, end)]
			keep = (rand(rng, Float32, size(F, 1), length(idx)) .>= dropout) ./ Float32(1 - dropout)
			f = F[:, idx] .* keep
			a = params[1] * f .+ params[2]
			h = NNlib.relu.(a)
			p = NNlib.softmax(params[3] * h .+ params[4]; dims = 1)
			d = copy(p)
			for (k, i) in enumerate(idx)
				d[y[i], k] -= 1
			end
			d ./= length(idx)
			dh = (transpose(params[3]) * d) .* (a .> 0)
			grads = (dh * transpose(f), vec(sum(dh; dims = 2)), d * transpose(h), vec(sum(d; dims = 2)))
			step += 1
			for (k, g) in enumerate(grads)  # Adam, with Keras's defaults
				mo, v = moments[k]
				mo .= 0.9f0 .* mo .+ 0.1f0 .* g
				v .= 0.999f0 .* v .+ 0.001f0 .* g .^ 2
				params[k] .-= Float32(rate) .* (mo ./ (1 - 0.9f0^step)) ./ (sqrt.(v ./ (1 - 0.999f0^step)) .+ 1.0f-7)
			end
		end
		t = scored(params, train)
		v = isempty(held) ? (loss = NaN, accuracy = NaN) : scored(params, held)
		push!(history, (epoch = epoch, loss = t.loss, accuracy = t.accuracy, validation_loss = v.loss, validation_accuracy = v.accuracy))
		score = isempty(held) ? t.loss : v.loss
		if score < best
			best, kept = score, deepcopy(params)
		end
	end
	tuned = EnkiModel(m.mean, m.deviation, m.stem, m.blocks, m.top, (kept[1], kept[2]), (kept[3], kept[4]))
	return (model = tuned, history = history)
end

"""
    share_threshold(values, share)

The value at or above which the `share` of the finite `values` lie, the highest first: the threshold that keeps that
share of the cells of a map.
"""
function share_threshold(values, share::Real)
	0 < share <= 1 || throw(ArgumentError("share must be above 0 and at most 1, not $(share)"))
	v = sort!([Float64(x) for x in values if !ismissing(x) && isfinite(x)]; rev = true)
	isempty(v) && error("The map has no cell with a share")
	return v[max(1, ceil(Int, share * length(v)))]
end

"""
    enki_features(probability, cellsize; threshold = 0.99, share = nothing, min_area = 0, min_roundness = 0)

The features of Enki's map of shares (see `enki_probability` for a raster): the cells of a share at least `threshold`
(Enki's README filters its candidates at 99 per cent) joined into regions through the sides of the cells, measured as
[`adaf_features`](@ref) measures them. With `share`, from 0 to 1, the threshold is instead the one that keeps that
share of the cells with imagery, those of highest share first: the rule for a model fine-tuned with `enki_finetune`,
whose shares are on a scale of their own, so that Enki's 99 per cent does not apply to them.
"""
enki_features(probability::AbstractMatrix{<:Real}, cellsize; threshold::Real = 0.99, share::Union{Nothing, Real} = nothing, kw...) =
	adaf_features(probability, cellsize; threshold = isnothing(share) ? threshold : share_threshold(probability, share), margin = 0, kw...)

"""
    aerial_image(box; source = :esri, path = nothing, cellsize = 1)

An RGB image of the area `box`, west, south, east and north in degrees, at `cellsize` metres in the UTM zone of the
centre of the box, written as a GeoTIFF to `path` and read from there when the file is there. `source` is `:esri`
(Esri World Imagery, with an ArcGIS key in `ENV["ARCGIS_API_KEY"]`), `:google` (Google's satellite map tiles, with a
key in `ENV["GOOGLE_MAPS_API_KEY"]`), a connection string of GDAL's WMS driver (`"WMS:https://..."`), or the path of
a raster file of the user's, whose first three bands are red, green and blue. Tiles are taken at the zoom level just
finer than `cellsize` and warped bilinearly onto the grid; cells without imagery are 0. Needs Rasters.jl and
ArchGDAL.jl.
"""
function aerial_image end

"""
    site_points(grid, sites)

The points of the known sites of the vector file `sites` in the coordinate system of the raster `grid`, as x and y:
each feature's centroid, those inside `grid` only. Needs Rasters.jl and ArchGDAL.jl.
"""
function site_points end

"""
    enki_examples(model, image, cells, sites; shift = 50, negatives = 5, distance = 700, seed = 1)

The images to train Enki's head on, by their pooled channels (see [`enki_finetune`](@ref)): for each known site of
`sites` (x and y, see `site_points`), nine images of 350 m centred on it and on the points `shift` metres from it in
x, y or both, taken from `image`; and `negatives` times as many cells of `cells` (see `enki_pooled` for a raster),
drawn at random among those whose centre is at least `distance` metres from every site. Returns `pooled`, `labels`
(true for the images of sites), and the numbers of `positives` and `negatives`. Needs Rasters.jl.
"""
function enki_examples end
