# Point clouds read with PDAL and LazIO.jl: the header of a point cloud, the points of several sources gathered into one
# LAZ file, and the points of a LAS or LAZ file. An extension, so that installing the package does not hold GDAL back
# to the GDAL_jll on which PDAL_jll is built.

module ArchaeologicalDetectionPointCloudExt

using ArchaeologicalDetection, PDAL_jll, LazIO
import ArchaeologicalDetection: point_cloud_summary, fetch_points, read_points
using ArchaeologicalDetection: JSON, SRS_VLR_ORDER, is_copc, is_web, epsg_code, downloaded

function point_cloud_summary(source::AbstractString)
	reader = is_copc(source) ? `--driver readers.copc` : `--readers.las.srs_vlr_order=$SRS_VLR_ORDER`
	output, report = IOBuffer(), IOBuffer()
	success(pipeline(`$(PDAL_jll.pdal()) info --summary $reader $source`; stdout = output, stderr = report)) ||
		error("PDAL could not read the point cloud: " * strip(String(take!(report))))
	info = JSON.parse(String(take!(output)))["summary"]
	b = info["bounds"]
	srs = get(info, "srs", Dict())
	wkt = srs isa AbstractDict ? string(get(srs, "wkt", "")) : ""
	return (points = Int(info["num_points"]), bounds = (xmin = b["minx"], ymin = b["miny"], zmin = b["minz"], xmax = b["maxx"], ymax = b["maxy"], zmax = b["maxz"]),
		wkt = wkt, epsg = epsg_code(wkt))
end

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
			stage = Dict{String, Any}("type" => "readers.las", "filename" => local_file, "srs_vlr_order" => SRS_VLR_ORDER)
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

read_points(path::AbstractString) = collect(LazIO.open(path))

end
