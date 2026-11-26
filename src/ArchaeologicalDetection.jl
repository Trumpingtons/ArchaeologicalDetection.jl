"""
    ArchaeologicalDetection

Trained models that detect archaeological features, run in Julia: the HRNet segmentation and Faster R-CNN object
detection models of ADAF, the Automatic Detection of Archaeological Features tool, with their input image, the
probability of a feature at each cell of a lidar terrain model (or the score of the boxes found) and the features
drawn from it; and a random forest trained on known sites in a multitemporal Sentinel-2 composite, after Orengo and
6 coauthors (2020); and Enki, an EfficientNet-B0 classifier of 1 m RGB imagery for tells and mounds (Rosati 2025),
whose head can be trained again on known sites; and a terrain model made from a lidar point cloud, its ground kept as
the survey classified it or found again with a morphological filter. Rasters read with Rasters.jl keep their grid.
"""
module ArchaeologicalDetection

using NNlib, Pickle
using Statistics: mean
import DecisionTree, Downloads, JLD2, JSON, Random, ZipArchives
import Geomorphometry, LazIO, NearestNeighbors, PDAL_jll
import Base64, OpenSSL_jll, SHA
using SmartRegions: local_relief, label_regions, cached, cellwidth

export adaf_weights, adaf_input, adaf_model, adaf_probability, adaf_boxes, adaf_features
export SENTINEL_BANDS, sentinel_scenes, sentinel_composite, lonlat_box, site_cells, forest_model, forest_probability, forest_features
export ENKI_CELL, aerial_image, site_points, enki_weights, enki_model, enki_pooled, enki_examples, enki_probability, enki_finetune, enki_save, enki_features
export DarknetModel, darknet_model, darknet_boxes, TUMULI_GREY, tumuli_weights, tumuli_model, msrm, tumuli_input, tumuli_boxes, tumuli_features, tumuli_soil_cells, tumuli_soil_model, tumuli_soil
export NOISE_CLASSES, GROUND_CLASSES, point_cloud_summary, index_tiles, fetch_points, read_points, point_classes, ground_grid, lowest_grid, find_ground, fill_gaps, ground_density, terrain_raster

include("adaf.jl")
include("hrnet.jl")
include("fasterrcnn.jl")
include("sentinel.jl")
include("enki.jl")
include("pointcloud.jl")
include("yolov3.jl")
include("tumuli.jl")

end
