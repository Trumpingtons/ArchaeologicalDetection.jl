"""
    ArchaeologicalDetection

Trained models that detect archaeological features, run in Julia: the HRNet segmentation and Faster R-CNN object
detection models of ADAF, the Automatic Detection of Archaeological Features tool, with their input image, the
probability of a feature at each cell of a lidar terrain model (or the score of the boxes found) and the features
drawn from it; and a random forest trained on known sites in a multitemporal Sentinel-2 composite, after Orengo and
6 coauthors (2020). Rasters read with Rasters.jl keep their grid.
"""
module ArchaeologicalDetection

using NNlib, Pickle
using Statistics: mean
import DecisionTree, Downloads, JSON, Random
using SmartRegions: local_relief, label_regions, cached, cellwidth

export adaf_weights, adaf_input, adaf_model, adaf_probability, adaf_boxes, adaf_features
export SENTINEL_BANDS, sentinel_scenes, sentinel_composite, site_cells, forest_model, forest_probability, forest_features

include("adaf.jl")
include("hrnet.jl")
include("fasterrcnn.jl")
include("sentinel.jl")

end
