# The tumulus detector of Berganzo-Besga, Orengo, Lumbreras, Carrero-Pazos, Fonte and Vilas-Estévez (2021), applied
# as they applied it to Galicia: a YOLOv3 network of Darknet (Redmon and Farhadi 2018), trained on tumuli of
# north-western Iberia, reads tiles of 1024 cells of the Multi-Scale Relief Model of a 1 m lidar terrain model (Orengo
# and Petrie 2018), set to 0 where a random forest of a multitemporal Sentinel-2 composite finds no soil fit for
# tumuli, and gives boxes with their similarity. The weights are those the authors published on MEGA; the network
# itself is in yolov3.jl.

const TUMULI_MEGA = (handle = "NddlzYbQ", key = "Ea_mT8uKiZmHZYrHr0SMJJ5YEXma1Gwqdy5b3cw1GbE", bytes = 246_303_084,
	sha256 = "0b1a5fdf595b9998c321df010396dc2ef49cf1cb04f3b0d61f6aced227095314")
const TUMULI_CFG = "https://raw.githubusercontent.com/iberganzo/darknet/78a530919a9ba5698f89f7798ae017e8e60aca20/cfg/yolov3-custom-detection.cfg"

"""
	tumuli_weights(folder)

The file of the trained tumulus detector of Berganzo-Besga and 5 coauthors (2021), `yolov3-custom_last.weights`
(Darknet's format, about 246 MB), downloaded once from the authors' MEGA link to `folder/tumuli`, decrypted, checked
and returned from there later; the network's configuration, `yolov3-custom-detection.cfg`, from the authors' Darknet
repository, is kept beside it.
"""
function tumuli_weights(folder::AbstractString)
	path = joinpath(folder, "tumuli", "yolov3-custom_last.weights")
	cached(TUMULI_CFG, joinpath(folder, "tumuli", "yolov3-custom-detection.cfg"))
	isfile(path) && return path
	mkpath(dirname(path))
	reply = IOBuffer()
	request = JSON.json([Dict("a" => "g", "g" => 1, "p" => TUMULI_MEGA.handle, "ssl" => 2)])
	Downloads.request("https://g.api.mega.co.nz/cs?id=1"; method = "POST", input = IOBuffer(request), output = reply,
		headers = ["Content-Type" => "application/json"])
	answer = JSON.parse(String(take!(reply)))
	link = answer isa AbstractVector && !isempty(answer) && first(answer) isa AbstractDict ? get(first(answer), "g", nothing) : nothing
	link isa AbstractString || error("MEGA did not give the address of the weights (its answer: $(answer)); it may be busy, or the file gone")
	encrypted = path * ".encrypted"
	try
		Downloads.download(link, encrypted)
		filesize(encrypted) == TUMULI_MEGA.bytes || error("the weights from MEGA have $(filesize(encrypted)) bytes, not $(TUMULI_MEGA.bytes)")
		a = base64url(TUMULI_MEGA.key)
		aes128ctr(encrypted, path * ".part", a[1:16] .⊻ a[17:32], [a[17:24]; zeros(UInt8, 8)])
	finally
		rm(encrypted; force = true)
	end
	if bytes2hex(open(SHA.sha256, path * ".part")) != TUMULI_MEGA.sha256
		rm(path * ".part"; force = true)
		error("the weights from MEGA are not those the authors published: their SHA-256 differs")
	end
	mv(path * ".part", path; force = true)
	return path
end

"""
	tumuli_model(path)

The tumulus detector of Berganzo-Besga and 5 coauthors (2021), a YOLOv3 network of Darknet reading tiles of 1024
cells and one channel, from its weights at `path` (see [`tumuli_weights`](@ref)) and the configuration beside them.
"""
tumuli_model(path::AbstractString) = darknet_model(joinpath(dirname(path), "yolov3-custom-detection.cfg"), path)

"""
	msrm(z, cellsize; fmin = 1, fmax = 19, x = 2)

The Multi-Scale Relief Model of the terrain model `z` (Orengo and Petrie 2018), as Berganzo-Besga and 5 coauthors
(2021) computed it in Earth Engine: the mean of the differences between successive mean filters over squares of
radius `k^x` cells, `k` from `i` to `n`, which the smallest and largest features sought, `fmin` and `fmax` metres, set
with the cell size, rounded to the millimetre. The mean of those differences is the first surface less the last over
their number. Positive on mounds, negative in hollows; `NaN` within the largest radius of the edge.
"""
function msrm(z::AbstractMatrix{<:Real}, cellsize; fmin::Real = 1, fmax::Real = 19, x::Real = 2)
	rr = round(cellwidth(cellsize); digits = 3)
	fmin = max(fmin, rr)
	i = floor(Int, ((fmin - rr) / (2rr))^(1 / x))
	n = ceil(Int, ((fmax - rr) / (2rr))^(1 / x))
	n > i || throw(ArgumentError("fmax is too close to fmin for this cell size"))
	heights = Float64.(z)
	lowpass(k) = k == 0 ? heights : heights .- local_relief(heights, cellsize; radius = round(Int, k^x) * rr)
	return round.((lowpass(i) .- lowpass(n)) ./ (n - i); digits = 3)
end

"""The grey of a cell of the detector's input is `TUMULI_GREY` times the square root of its Multi-Scale Relief Model, in
metres, 0 where it is 0 or below: the stretch of the authors' images, which were converted to JPEG and PNG in Photoshop
without the stretch being stated, estimated from the grey levels of their validation images."""
const TUMULI_GREY = 207

"""
	tumuli_input(z, cellsize; soil = nothing, grey = TUMULI_GREY)

The image the tumulus detector reads, from the terrain model `z` (from its north-west corner, x first): its
Multi-Scale Relief Model (see [`msrm`](@ref)), set to 0 where `soil`, a matrix of the cells of `z`, is false or 0, as
the authors multiplied it by the soil classes of a random forest, and stretched to 8-bit grey, `grey` times its square
root (see [`TUMULI_GREY`](@ref)), from 0 to 1. 0 where it is undefined.
"""
function tumuli_input(z::AbstractMatrix{<:Real}, cellsize; soil = nothing, grey::Real = TUMULI_GREY)
	m = msrm(z, cellsize)
	if soil !== nothing
		size(soil) == size(m) || throw(DimensionMismatch("soil has $(size(soil)) cells, the terrain model $(size(m))"))
		m = ifelse.(isfinite.(soil) .& (soil .> 0), m, 0.0)
	end
	return Float32.(round.(clamp.(grey .* sqrt.(ifelse.(isfinite.(m) .& (m .> 0), m, 0.0)), 0, 255)) ./ 255)
end

"""
	tumuli_boxes(model, image; threshold = 0.25, overlap = 0.45)

The tumuli the detector `model` (see [`tumuli_model`](@ref)) finds in `image` (see [`tumuli_input`](@ref)), read in
tiles of 1024 cells from its north-west corner, as the authors cut their terrain model, the image beyond the last tile
filled with 0: the boxes of similarity above `threshold` (25% in Berganzo-Besga and 5 coauthors, 2021), those that
overlap a more similar box of the same tile by more than `overlap` left out, as Darknet leaves them out. Each box has
the position `i`, `j` of its centre and its width `w` and height `h`, in cells of `image`, and its `similarity`.
"""
function tumuli_boxes(model::DarknetModel, image::AbstractMatrix{<:Real}; threshold::Real = 0.25, overlap::Real = 0.45)
	tile = model.width
	nx, ny = size(image)
	found = NamedTuple{(:i, :j, :w, :h, :similarity), NTuple{5, Float64}}[]
	for j0 in 1:tile:ny, i0 in 1:tile:nx
		piece = zeros(Float32, tile, model.height)
		ri, rj = i0:min(i0 + tile - 1, nx), j0:min(j0 + model.height - 1, ny)
		piece[1:length(ri), 1:length(rj)] .= image[ri, rj]
		for b in darknet_boxes(model, piece; threshold, overlap)
			# Darknet's x runs from 0 at the west edge of the tile; cell k spans k - 1 to k
			push!(found, (i = i0 - 0.5 + b.x, j = j0 - 0.5 + b.y, w = Float64(b.w), h = Float64(b.h), similarity = Float64(b.probability)))
		end
	end
	return found
end

const TUMULI_SOIL_FILE = joinpath(@__DIR__, "..", "data", "tumuli_soil.csv")

"""
	tumuli_soil_cells()

The training cells of the soil mask of Berganzo-Besga and 5 coauthors (2021): the cells of 10 m inside their 19
polygons of soil fit for tumuli (class 1) and 13 of other land cover, such as rock outcrops, towns and water (class 0),
in Galicia, with the mean reflectance of each band of [`SENTINEL_BANDS`](@ref) in the Sentinel-2 Level-2A scenes of
Earth Search from the first, in November 2016, to 5 June 2020, the end of the authors' period, under 20% cloud, each
cell where it is clear. Returns `features` (cells by bands), `classes`, and each cell's `polygon`, `lon` and `lat`.
"""
function tumuli_soil_cells()
	lines = readlines(TUMULI_SOIL_FILE)
	header = split(first(lines), ',')
	Tuple(header[5:end]) == SENTINEL_BANDS || error("the columns of $(TUMULI_SOIL_FILE) are not the bands of SENTINEL_BANDS")
	values = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
	return (features = [v[4 + b] for v in values, b in 1:length(SENTINEL_BANDS)], classes = [Int(v[1]) for v in values],
		polygon = [Int(v[2]) for v in values], lon = [v[3] for v in values], lat = [v[4] for v in values])
end

"""
	tumuli_soil_model(; trees = 128, seed = 1)

The random forest of the soil mask of Berganzo-Besga and 5 coauthors (2021), trained on their cells (see
[`tumuli_soil_cells`](@ref)): `trees` trees, as in Earth Engine's random forest of 128 trees, each taking half the cells,
drawn with replacement, and the square root of the number of bands at each split, as Earth Engine does by default.
"""
function tumuli_soil_model(; trees::Integer = 128, seed::Integer = 1)
	c = tumuli_soil_cells()
	forest = DecisionTree.build_forest(c.classes, c.features, -1, trees, 0.5; rng = Random.Xoshiro(seed))
	return (forest = forest, bands = size(c.features, 2), positives = count(==(1), c.classes), negatives = count(==(0), c.classes))
end

"""
	tumuli_soil(model, features)

The soil fit for tumuli at every cell of `features` (cells by bands, as the composite of [`sentinel_composite`](@ref)):
true where most trees of `model` (see [`tumuli_soil_model`](@ref)) vote for it, as Earth Engine classifies; false where
a band is missing. The authors multiplied the Multi-Scale Relief Model by it (see [`tumuli_input`](@ref)).
"""
tumuli_soil(model, features::AbstractArray{<:Union{Missing, Real}, 3}) = (p = forest_probability(model, features); isfinite.(p) .& (p .> 0.5))

"""
	tumuli_features(boxes, cellsize)

The tumuli of the boxes of [`tumuli_boxes`](@ref), as the authors listed them: each with its `id`, the position `i`,
`j` of its centre, its `size`, the larger side of its box in the units of `cellsize`, and its `similarity`.
"""
tumuli_features(boxes, cellsize) = [(id = k, i = b.i, j = b.j, size = max(b.w, b.h) * cellwidth(cellsize), similarity = b.similarity) for (k, b) in enumerate(boxes)]

# The cells of an nx by ny grid within each box, labelled with its number, and the highest similarity of a box over
# each cell, 0 outside every box: the more similar box takes a cell where two overlap
function tumuli_cells(boxes, nx::Integer, ny::Integer)
	labels, similarity = zeros(Int, nx, ny), zeros(Float64, nx, ny)
	for k in sortperm([b.similarity for b in boxes])
		b = boxes[k]
		ri = max(1, ceil(Int, b.i - b.w / 2)):min(nx, floor(Int, b.i + b.w / 2))
		rj = max(1, ceil(Int, b.j - b.h / 2)):min(ny, floor(Int, b.j + b.h / 2))
		labels[ri, rj] .= k
		similarity[ri, rj] .= b.similarity
	end
	return (labels = labels, similarity = similarity)
end

base64url(s) = Base64.base64decode(rpad(replace(s, '-' => '+', '_' => '/'), 4 * cld(length(s), 4), '='))

# MEGA encrypts a file with AES-128 in counter mode, the key and the first half of the counter from the link
function aes128ctr(input, output, key::Vector{UInt8}, iv::Vector{UInt8})
	context = ccall((:EVP_CIPHER_CTX_new, OpenSSL_jll.libcrypto), Ptr{Cvoid}, ())
	try
		cipher = ccall((:EVP_aes_128_ctr, OpenSSL_jll.libcrypto), Ptr{Cvoid}, ())
		ccall((:EVP_DecryptInit_ex, OpenSSL_jll.libcrypto), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{UInt8}, Ptr{UInt8}),
			context, cipher, C_NULL, key, iv) == 1 || error("OpenSSL could not start AES-128-CTR")
		buffer = Vector{UInt8}(undef, 1 << 22)
		plain = similar(buffer)
		n = Ref{Cint}(0)
		open(input) do source
			open(output, "w") do target
				while !eof(source)
					k = readbytes!(source, buffer)
					ccall((:EVP_DecryptUpdate, OpenSSL_jll.libcrypto), Cint, (Ptr{Cvoid}, Ptr{UInt8}, Ref{Cint}, Ptr{UInt8}, Cint),
						context, plain, n, buffer, k) == 1 || error("OpenSSL could not decrypt the weights")
					write(target, view(plain, 1:n[]))
				end
			end
		end
	finally
		ccall((:EVP_CIPHER_CTX_free, OpenSSL_jll.libcrypto), Cvoid, (Ptr{Cvoid},), context)
	end
	return output
end
