# YOLOv3 (Redmon and Farhadi 2018) as Darknet runs it on the CPU: the network that a Darknet .cfg file describes, with
# the weights of a Darknet .weights file, its batch normalisation folded into the convolutions. An image is a matrix
# from its north-west corner, x first, as Darknet holds it; the boxes are those that `darknet detector test` keeps.

"""A YOLOv3 network read from Darknet's files (see [`darknet_model`](@ref))."""
struct DarknetModel
	width::Int
	height::Int
	channels::Int
	layers::Vector{Any}
	kept::Set{Int}  # the layers whose outputs later layers read again
end

Base.show(io::IO, m::DarknetModel) = print(io, "DarknetModel($(length(m.layers)) layers, $(m.width)×$(m.height)×$(m.channels))")

# The sections of a Darknet .cfg file, each its name and its options
function darknet_sections(path::AbstractString)
	sections = Tuple{String, Dict{String, String}}[]
	for line in eachline(path)
		line = strip(split(line, ('#', ';'))[1])
		isempty(line) && continue
		if startswith(line, '[')
			push!(sections, (strip(line, ['[', ']', ' ']), Dict{String, String}()))
		elseif occursin('=', line) && !isempty(sections)
			k, v = strip.(split(line, '='; limit = 2))
			sections[end][2][k] = v
		end
	end
	return sections
end

option(o, key, default) = haskey(o, key) ? parse(typeof(default), o[key]) : default
numbers(T, s) = [parse(T, strip(x)) for x in split(s, ',') if !isempty(strip(x))]

"""
	darknet_model(cfg, weights)

The YOLOv3 network that the Darknet configuration file `cfg` describes, with the weights of the Darknet file
`weights`: convolutional layers with batch normalisation and leaky or linear activations, shortcuts, routes, nearest
upsampling and YOLO layers, run with NNlib. Throws when the file has another number of weights than the network.
"""
function darknet_model(cfg::AbstractString, weights::AbstractString)
	sections = darknet_sections(cfg)
	isempty(sections) || first(sections)[1] in ("net", "network") || error("$(cfg) does not start with [net]")
	net = sections[1][2]
	width, height, channels = option(net, "width", 416), option(net, "height", 416), option(net, "channels", 3)
	layers = Any[]
	outputs = Int[]  # the channels of each layer's output
	open(weights) do io
		major, minor = read(io, Int32), read(io, Int32)
		read(io, Int32)  # revision
		major * 10 + minor >= 2 && major < 1000 && minor < 1000 ? read(io, Int64) : read(io, Int32)  # images seen in training
		c = channels
		for (index, (kind, o)) in enumerate(sections[2:end])
			at(k) = k < 0 ? index + k : k + 1  # Darknet counts layers from 0, and back from the present one
			if kind == "convolutional"
				n, k, stride = option(o, "filters", 1), option(o, "size", 1), option(o, "stride", 1)
				pad = option(o, "pad", 0) == 1 ? k ÷ 2 : option(o, "padding", 0)
				option(o, "groups", 1) == 1 || error("grouped convolutions are not read")
				activation = get(o, "activation", "logistic")
				activation in ("leaky", "linear") || error("the activation $(activation) is not read")
				bias = read!(io, Vector{Float32}(undef, n))
				normalised = option(o, "batch_normalize", 0) == 1
				scales, means, variances = normalised ? ntuple(_ -> read!(io, Vector{Float32}(undef, n)), 3) : (nothing, nothing, nothing)
				weight = read!(io, Array{Float32}(undef, k, k, c, n))
				if normalised
					f = scales ./ sqrt.(variances .+ 1.0f-5)
					weight .*= reshape(f, 1, 1, 1, n)
					bias = bias .- means .* f
				end
				push!(layers, (kind = :conv, weight = weight, bias = bias, stride = stride, pad = pad, leaky = activation == "leaky"))
				c = n
			elseif kind == "shortcut"
				get(o, "activation", "linear") == "linear" || error("a shortcut with an activation is not read")
				push!(layers, (kind = :shortcut, from = at(option(o, "from", -1))))
			elseif kind == "route"
				from = at.(numbers(Int, o["layers"]))
				haskey(o, "groups") && error("routes of groups are not read")
				push!(layers, (kind = :route, from = from))
				c = sum(outputs[from])
			elseif kind == "upsample"
				push!(layers, (kind = :upsample, stride = option(o, "stride", 2)))
			elseif kind == "yolo"
				option(o, "scale_x_y", 1.0) == 1 || error("YOLO layers with scale_x_y are not read")
				option(o, "new_coords", 0) == 0 || error("YOLO layers with new_coords are not read")
				anchors = numbers(Float32, o["anchors"])
				push!(layers, (kind = :yolo, mask = numbers(Int, o["mask"]) .+ 1, anchors = [(anchors[2k - 1], anchors[2k]) for k in 1:length(anchors) ÷ 2],
					classes = option(o, "classes", 20)))
			else
				error("the layer [$(kind)] is not read")
			end
			push!(outputs, c)
		end
		eof(io) || error("$(weights) has more weights than the network of $(cfg)")
	end
	kept = Set{Int}(f for l in layers if l.kind in (:shortcut, :route) for f in l.from)
	return DarknetModel(width, height, channels, layers, kept)
end

leaky(x::Float32) = ifelse(x > 0, x, 0.1f0 * x)
logistic(x) = 1 / (1 + exp(-x))

# The outputs of the YOLO layers for the image x (width × height × channels × 1)
function darknet_forward(model::DarknetModel, x::Array{Float32, 4})
	saved = Dict{Int, Array{Float32, 4}}()
	heads = Tuple{Array{Float32, 4}, Any}[]
	for (i, l) in enumerate(model.layers)
		if l.kind === :conv
			y = NNlib.conv(x, l.weight; stride = l.stride, pad = l.pad, flipped = true)
			y .+= reshape(l.bias, 1, 1, :, 1)
			l.leaky && (y .= leaky.(y))
			x = y
		elseif l.kind === :shortcut
			x = x .+ saved[l.from]
		elseif l.kind === :route
			x = length(l.from) == 1 ? saved[only(l.from)] : cat((saved[f] for f in l.from)...; dims = 3)
		elseif l.kind === :upsample
			x = NNlib.upsample_nearest(x, (l.stride, l.stride))
		else
			push!(heads, (x, l))
		end
		i in model.kept && (saved[i] = x)
	end
	return heads
end

"""
	darknet_boxes(model, image; threshold = 0.25, overlap = 0.45)

The boxes that `darknet detector test` finds in `image`, a matrix of the size of the network's input (from its
north-west corner, x first, values from 0 to 1): those whose objectness and probability of a class both exceed
`threshold`, the boxes of a class that overlap a more probable box by more than `overlap` (intersection over union)
left out. Each box has its centre `x`, `y`, its width `w` and height `h`, in cells of the image, its `class` and its
`probability`.
"""
function darknet_boxes(model::DarknetModel, image::AbstractMatrix{<:Real}; threshold::Real = 0.25, overlap::Real = 0.45)
	model.channels == 1 || error("the network reads $(model.channels) channels: give it an image of as many")
	size(image) == (model.width, model.height) || throw(DimensionMismatch("the network reads images of $(model.width)×$(model.height) cells"))
	x = reshape(Float32.(image), model.width, model.height, 1, 1)
	found = NamedTuple{(:x, :y, :w, :h, :class, :probability), Tuple{Float32, Float32, Float32, Float32, Int, Float32}}[]
	for (out, l) in darknet_forward(model, x)
		nx, ny = size(out, 1), size(out, 2)
		per = 5 + l.classes
		for (a, m) in enumerate(l.mask), j in 1:ny, i in 1:nx
			base = (a - 1) * per
			objectness = logistic(out[i, j, base + 5, 1])
			objectness > threshold || continue
			bx = (i - 1 + logistic(out[i, j, base + 1, 1])) / nx * model.width
			by = (j - 1 + logistic(out[i, j, base + 2, 1])) / ny * model.height
			bw = exp(out[i, j, base + 3, 1]) * l.anchors[m][1]
			bh = exp(out[i, j, base + 4, 1]) * l.anchors[m][2]
			for k in 1:l.classes
				p = objectness * logistic(out[i, j, base + 5 + k, 1])
				p > threshold && push!(found, (x = bx, y = by, w = bw, h = bh, class = k, probability = p))
			end
		end
	end
	kept = similar(found, 0)
	for k in unique(b.class for b in found)
		ofclass = filter(b -> b.class == k, found)
		corners = [(b.x - b.w / 2, b.y - b.h / 2, b.x + b.w / 2, b.y + b.h / 2) for b in ofclass]
		append!(kept, ofclass[nms(corners, [b.probability for b in ofclass], Float32(overlap))])
	end
	return kept
end
