# ADAF's Faster R-CNN object detection models in Julia: torchvision's fasterrcnn_resnet50_fpn_v2 with two classes (a
# feature and the background), as AiTLAS runs it: the ResNet-50 and its feature pyramid, the region proposal network,
# RoIAlign and the box head, with each batch normalisation folded into the convolution before it, the weights read
# from ADAF's PyTorch checkpoints with Pickle.jl and the layers computed with NNlib.jl. Images are (x, y) matrices, as
# PyTorch's are (y, x), and boxes are x0, y0, x1, y1 from the corner of the first cell, as torchvision's.

const IMAGE_MEAN = Float32[0.485, 0.456, 0.406]
const IMAGE_STD = Float32[0.229, 0.224, 0.225]
const ANCHOR_SIZES = (32, 64, 128, 256, 512)
const ASPECT_RATIOS = Float32[0.5, 1.0, 2.0]
const BOX_CLIP = Float32(log(1000 / 16))

struct ADAFDetector
	stem::Folded
	layers::Vector{Vector{Block}}
	inner::Vector{Folded}
	outer::Vector{Folded}
	rpn::Vector{Folded}
	objectness::Folded
	deltas::Folded
	head::Vector{Folded}
	fc::Matrix{Float32}
	fc_bias::Vector{Float32}
	score::Matrix{Float32}
	score_bias::Vector{Float32}
	box::Matrix{Float32}
	box_bias::Vector{Float32}
end

# A bottleneck of the ResNet-50: its stride on the 3 x 3 convolution, as torchvision's
function bottleneck(sd, p, stride)
	convs = [folded(sd, "$(p).conv1", "$(p).bn1"), folded(sd, "$(p).conv2", "$(p).bn2"; stride), folded(sd, "$(p).conv3", "$(p).bn3")]
	return Block(convs, has(sd, "$(p).downsample.0") ? folded(sd, "$(p).downsample.0", "$(p).downsample.1"; stride) : nothing)
end

function detector(sd)
	matrix(key) = Matrix{Float32}(sd[key])
	vector(key) = Vector{Float32}(sd[key])
	layers = [[bottleneck(sd, "backbone.body.layer$(l).$(k)", l > 1 && k == 0 ? 2 : 1) for k in upto(k -> has(sd, "backbone.body.layer$(l).$(k).conv1"))] for l in 1:4]
	return ADAFDetector(folded(sd, "backbone.body.conv1", "backbone.body.bn1"; stride = 2), layers,
		[folded(sd, "backbone.fpn.inner_blocks.$(k).0", "backbone.fpn.inner_blocks.$(k).1") for k in 0:3],
		[folded(sd, "backbone.fpn.layer_blocks.$(k).0", "backbone.fpn.layer_blocks.$(k).1") for k in 0:3],
		[folded(sd, "rpn.head.conv.$(k).0", nothing) for k in upto(k -> has(sd, "rpn.head.conv.$(k).0"))],
		folded(sd, "rpn.head.cls_logits", nothing), folded(sd, "rpn.head.bbox_pred", nothing),
		[folded(sd, "roi_heads.box_head.$(k).0", "roi_heads.box_head.$(k).1") for k in 0:3],
		matrix("roi_heads.box_head.5.weight"), vector("roi_heads.box_head.5.bias"),
		matrix("roi_heads.box_predictor.cls_score.weight"), vector("roi_heads.box_predictor.cls_score.bias"),
		matrix("roi_heads.box_predictor.bbox_pred.weight"), vector("roi_heads.box_predictor.bbox_pred.bias"))
end

# The feature maps of the pyramid, finest first: four from the ResNet-50 and a fifth of every other cell of the last
function pyramid(model::ADAFDetector, x)
	x = NNlib.maxpool(relu.(model.stem(x)), (3, 3); pad = 1, stride = 2)
	features = Array{Float32, 4}[]
	for layer in model.layers
		x = foldl((y, b) -> b(y), layer; init = x)
		push!(features, x)
	end
	last = model.inner[4](features[4])
	results = [model.outer[4](last)]
	for k in 3:-1:1
		lateral = model.inner[k](features[k])
		last = lateral .+ last[nearest(size(last, 1), size(lateral, 1)), nearest(size(last, 2), size(lateral, 2)), :, :]
		pushfirst!(results, model.outer[k](last))
	end
	return push!(results, results[4][1:2:end, 1:2:end, :, :])
end

# torchvision's boxes from deltas: centres, widths and heights in single precision, the scale of the deltas clipped
function decode(deltas, anchors, weights)
	w = anchors[3] - anchors[1]
	h = anchors[4] - anchors[2]
	cx = anchors[1] + 0.5f0 * w
	cy = anchors[2] + 0.5f0 * h
	dx, dy = deltas[1] / weights[1], deltas[2] / weights[2]
	dw, dh = min(deltas[3] / weights[3], BOX_CLIP), min(deltas[4] / weights[4], BOX_CLIP)
	px, py = dx * w + cx, dy * h + cy
	pw, ph = exp(dw) * w, exp(dh) * h
	return (px - 0.5f0 * pw, py - 0.5f0 * ph, px + 0.5f0 * pw, py + 0.5f0 * ph)
end

clip(b, n) = (clamp(b[1], 0.0f0, Float32(n)), clamp(b[2], 0.0f0, Float32(n)), clamp(b[3], 0.0f0, Float32(n)), clamp(b[4], 0.0f0, Float32(n)))
area(b) = (b[3] - b[1]) * (b[4] - b[2])

# torchvision's non-maximum suppression: the boxes by decreasing score, each dropped if it overlaps a box kept before it
# by more than `threshold` of their union. Returns the indices kept, by decreasing score
function nms(boxes, scores, threshold)
	order = sortperm(scores; rev = true)
	removed = falses(length(boxes))
	keep = Int[]
	for (k, i) in enumerate(order)
		removed[i] && continue
		push!(keep, i)
		a = boxes[i]
		for j in @view order[k + 1:end]
			removed[j] && continue
			b = boxes[j]
			iw = max(0.0f0, min(a[3], b[3]) - max(a[1], b[1]))
			ih = max(0.0f0, min(a[4], b[4]) - max(a[2], b[2]))
			inter = iw * ih
			inter / (area(a) + area(b) - inter) > threshold && (removed[j] = true)
		end
	end
	return keep
end

# torchvision's batched_nms: the suppression within each group, on its own for more than 1,000 boxes and otherwise
# with the boxes of each group moved apart, as torchvision does on the CPU
function batched_nms(boxes, scores, groups, threshold)
	isempty(boxes) && return Int[]
	if 4 * length(boxes) > 4000
		keep = reduce(vcat, [(members = findall(==(g), groups); members[nms(boxes[members], scores[members], threshold)]) for g in unique(groups)])
		return keep[sortperm(scores[keep]; rev = true)]
	end
	shift = maximum(maximum, boxes) + 1.0f0
	return nms([b .+ Float32(g) * shift for (b, g) in zip(boxes, groups)], scores, threshold)
end

# The anchors of a feature map of n1 x n2 cells, `stride` cells of the image apart, in the order of the network's outputs
function anchors(size, n1, n2, stride)
	base = [round.(Float32[-w, -h, w, h] ./ 2) for (w, h) in (((1 / sqrt(r)) * Float32(size), sqrt(r) * Float32(size)) for r in ASPECT_RATIOS)]
	return vec([Tuple(base[a] .+ Float32[(i - 1) * stride[1], (j - 1) * stride[2], (i - 1) * stride[1], (j - 1) * stride[2]]) for a in eachindex(base), i in 1:n1, j in 1:n2])
end

# The region proposal network: the 1,000 best anchors of each level moved by their deltas, clipped to the image of m
# cells, then the best 1,000 left after a suppression within each level
function proposals(model::ADAFDetector, features, m, padded)
	boxes, logits, levels = NTuple{4, Float32}[], Float32[], Int[]
	for (level, f) in enumerate(features)
		t = foldl((y, c) -> relu.(c(y)), model.rpn; init = f)
		o = vec(permutedims(model.objectness(t)[:, :, :, 1], (3, 1, 2)))
		d = model.deltas(t)[:, :, :, 1]
		d = reshape(permutedims(reshape(d, size(d, 1), size(d, 2), 4, :), (3, 4, 1, 2)), 4, :)
		a = anchors(ANCHOR_SIZES[level], size(f, 1), size(f, 2), (padded ÷ size(f, 1), padded ÷ size(f, 2)))
		top = partialsortperm(o, 1:min(1000, length(o)); rev = true)
		append!(boxes, [clip(decode(d[:, k], a[k], (1.0f0, 1.0f0, 1.0f0, 1.0f0)), m) for k in top])
		append!(logits, o[top])
		append!(levels, fill(level, length(top)))
	end
	scores = 1 ./ (1 .+ exp.(-logits))
	kept = findall(b -> b[3] - b[1] >= 1.0f-3 && b[4] - b[2] >= 1.0f-3, boxes)
	keep = kept[batched_nms(boxes[kept], scores[kept], levels[kept], 0.7f0)]
	return boxes[keep[1:min(1000, length(keep))]]
end

# torchvision's RoIAlign, not aligned, two samples a side in each of the 7 x 7 bins: the box `b` of the image on the
# feature map `f`, `scale` of its cells to a cell of the image
function roi_align(f, b, scale)
	nx, ny = size(f, 1), size(f, 2)
	x0, y0 = b[1] * scale, b[2] * scale
	w, h = max(b[3] * scale - x0, 1.0f0), max(b[4] * scale - y0, 1.0f0)
	bw, bh = w / 7, h / 7
	out = zeros(Float32, 7, 7, size(f, 3))
	samples(start, bin, p, n) = map(0:1) do s
		v = start + p * bin + (s + 0.5f0) * bin / 2
		(v < -1 || v > n) && return nothing
		v = max(v, 0.0f0)
		low = floor(Int, v)
		low >= n - 1 ? (n, n, 0.0f0) : (low + 1, low + 2, v - low)
	end
	for ph in 0:6, pw in 0:6
		for sy in samples(y0, bh, ph, ny), sx in samples(x0, bw, pw, nx)
			(sx === nothing || sy === nothing) && continue
			(x1, x2, lx), (y1, y2, ly) = sx, sy
			hx, hy = 1 - lx, 1 - ly
			for c in axes(f, 3)
				out[pw + 1, ph + 1, c] += hy * hx * f[x1, y1, c, 1] + hy * lx * f[x2, y1, c, 1] + ly * hx * f[x1, y2, c, 1] + ly * lx * f[x2, y2, c, 1]
			end
		end
	end
	return out ./ 4
end

# The box head on the proposals, by groups of 100: each proposal pooled from the level of the pyramid its size calls
# for, then the four convolutions, the fully connected layer and the predictor. Returns the boxes of the feature and their
# scores, the softmax of the two classes
function detections(model::ADAFDetector, features, boxes, m)
	scales = [2.0f0^round(log2(size(f, 1) / m)) for f in features[1:4]]
	levels = [clamp(floor(Int, 4 + log2(sqrt(area(b)) / 224.0f0) + 1.0f-6), 2, 5) - 1 for b in boxes]
	found, scores = NTuple{4, Float32}[], Float32[]
	for group in Iterators.partition(eachindex(boxes), 100)
		x = cat([roi_align(features[levels[k]], boxes[k], scales[levels[k]]) for k in group]...; dims = 4)
		x = foldl((y, c) -> relu.(c(y)), model.head; init = x)
		x = relu.(model.fc * reshape(x, :, size(x, 4)) .+ model.fc_bias)
		logits = model.score * x .+ model.score_bias
		deltas = model.box * x .+ model.box_bias
		for (n, k) in enumerate(group)
			e = exp.(logits[:, n] .- maximum(logits[:, n]))
			push!(scores, e[2] / sum(e))
			push!(found, clip(decode(deltas[5:8, n], boxes[k], (10.0f0, 10.0f0, 5.0f0, 5.0f0)), m))
		end
	end
	return found, scores
end

"""
    adaf_boxes(model, image; tile = 1024)

The boxes ADAF's Faster R-CNN model `model` finds on `image` (see [`adaf_input`](@ref)), run as ADAF runs it: on tiles
of `tile` cells from the first cell of the matrix, the image beyond the last tile filled with 0, each tile as AiTLAS
reads it (the image repeated into three bands, normalised and resized to 800 cells a side by torchvision), the
detections with a score above 0.05 after torchvision's suppressions, then AiTLAS's suppression of boxes that overlap
by more than 0.2. Each box has `x0`, `y0`, `x1` and `y1`, its sides in cells from the corner of the first cell of
`image`, and its `score`, from 0 to 1.
"""
function adaf_boxes(model::ADAFDetector, image::StridedMatrix{<:Real}; tile::Integer = 1024)
	nx, ny = size(image)
	found = NamedTuple{(:x0, :y0, :x1, :y1, :score), NTuple{5, Float32}}[]
	for j in 1:tile:ny, i in 1:tile:nx
		piece = zeros(Float32, tile, tile)
		ri, rj = i:min(i + tile - 1, nx), j:min(j + tile - 1, ny)
		piece[1:length(ri), 1:length(rj)] .= image[ri, rj]
		for b in boxes_of_tile(model, piece)
			push!(found, (x0 = b[1] + (i - 1), y0 = b[2] + (j - 1), x1 = b[3] + (i - 1), y1 = b[4] + (j - 1), score = b[5]))
		end
	end
	return found
end

function boxes_of_tile(model::ADAFDetector, piece::Matrix{Float32})
	n = size(piece, 1)
	m = floor(Int, n * min(800 / n, 1333 / n))
	padded = 32 * cld(m, 32)
	x = zeros(Float32, padded, padded, 3, 1)
	for c in 1:3
		x[1:m, 1:m, c, 1] .= bilinear((piece .- IMAGE_MEAN[c]) ./ IMAGE_STD[c], m)
	end
	features = pyramid(model, x)
	boxes, scores = detections(model, features, proposals(model, features, m, padded), m)
	keep = findall(b -> b[3] - b[1] >= 1.0f-2 && b[4] - b[2] >= 1.0f-2, boxes)
	keep = filter(k -> scores[k] > 0.05f0, keep)
	keep = keep[batched_nms(boxes[keep], scores[keep], ones(Int, length(keep)), 0.5f0)]
	keep = keep[1:min(100, length(keep))]
	ratio = Float32(n) / Float32(m)
	boxes, scores = [b .* ratio for b in boxes[keep]], scores[keep]
	return [(boxes[k]..., scores[k]) for k in nms(boxes, scores, 0.2f0)]
end

"""
    adaf_probability(model::ADAFDetector, image; tile = 1024)

For a Faster R-CNN model, the score of the best box covering each cell of `image` (see [`adaf_boxes`](@ref)), 0 where
no box covers it: the sides of each box rounded to whole cells, as ADAF rounds them, so that the cells above a threshold
are those of the boxes whose score is above it, which [`adaf_features`](@ref) joins into features.
"""
function adaf_probability(model::ADAFDetector, image::StridedMatrix{<:Real}; tile::Integer = 1024)
	nx, ny = size(image)
	out = zeros(Float32, nx, ny)
	for b in adaf_boxes(model, image; tile)
		ri = max(round(Int, b.x0) + 1, 1):min(round(Int, b.x1), nx)
		rj = max(round(Int, b.y0) + 1, 1):min(round(Int, b.y1), ny)
		out[ri, rj] .= max.(out[ri, rj], b.score)
	end
	return out
end
