# ArchaeologicalDetection.jl

Trained models that detect archaeological features, run in Julia: neural networks on lidar terrain models, and a random forest on Sentinel-2.

The package holds a port of the HRNet segmentation and Faster R-CNN object detection models of [ADAF](https://github.com/EarthObservation/adaf), the Automatic Detection of Archaeological Features tool of Čož, Corns, Curran, Kocev and Kokalj (2026), [doi:10.1016/j.jasrep.2026.105733](https://doi.org/10.1016/j.jasrep.2026.105733). Each has four models, trained on airborne laser scanning data of Ireland: all archaeology, barrows, enclosures and ringforts. The [weights](https://doi.org/10.5281/zenodo.15848663) are downloaded from Zenodo at first use, about 860 MB an HRNet model and 520 MB a Faster R-CNN model, and read with Pickle.jl without running Python.

```julia
using ArchaeologicalDetection, Rasters, ArchGDAL

dem = replace_missing(read(Raster("terrain.tif")), NaN32)  # a bare-earth terrain model in metres
model = adaf_model(adaf_weights("models"; feature = :all))  # :all, :barrow, :enclosure or :ringfort
probability = adaf_probability(model, dem)
found = adaf_features(probability; threshold = 0.5)
found.candidates
```

With `network = :fasterrcnn`, `adaf_weights` gives an object detection model. Its `adaf_probability` is the score of the best box covering each cell, and its features are drawn with `adaf_features(scores; threshold = 0.5, margin = 0)`. `adaf_boxes` gives the boxes themselves, on a matrix.

| Function | Returns |
|---|---|
| `adaf_weights` | The file of a trained model, downloaded once |
| `adaf_input` | The image the models read: the simple local relief model, from 0 to 1 |
| `adaf_model` | The network, HRNet or Faster R-CNN, with the weights of that file |
| `adaf_boxes` | Faster R-CNN: the boxes and their scores |
| `adaf_probability` | The probability of a feature at each cell, or the score of the best box covering it |
| `adaf_features` | The features, with their area, roundness and highest probability, and their cells |

## Random Forest on Sentinel-2

After Orengo, Conesa, Garcia-Molsosa, Lobo, Green, Madella and Petrie (2020), [doi:10.1073/pnas.2005583117](https://doi.org/10.1073/pnas.2005583117), who found mounds in Cholistan this way. A multitemporal composite of Sentinel-2 is the mean reflectance of many scenes, each without its cloudy cells, in ten bands at 10 m. A random forest of 128 trees is trained on the cells of known sites and on cells away from them, and gives at every cell the share of its trees that vote for a site. The scenes are those of Level-2A of [Earth Search](https://element84.com/earth-search/), read without an account from Cloud-Optimised GeoTIFFs, only the cells of the area.

```julia
using ArchaeologicalDetection, Rasters, ArchGDAL

box = (-8.16, 41.95, -8.13, 41.97)  # west, south, east, north, in degrees
scenes = sentinel_scenes(box; years = 2023:2025, cloud = 30)
composite = sentinel_composite(scenes, box; path = "composite.tif")  # written once, read from there later
sites = site_cells(composite, "known-sites.gpkg"; radius = 50)  # polygons, or points and lines with a radius
model = forest_model(composite, sites)
probability = forest_probability(model, composite)
found = forest_features(probability; threshold = 0.55)
```

| Function | Returns |
|---|---|
| `sentinel_scenes` | The Level-2A scenes over an area, in some years, below a share of clouds |
| `sentinel_composite` | The mean reflectance of the clear cells of the scenes, in ten bands at 10 m, in the UTM zone of the area |
| `site_cells` | The cells of the composite that are known sites, from a vector file |
| `forest_model` | The random forest, trained on the cells of the sites and cells away from them |
| `forest_probability` | The share of its trees that vote for a site, at each cell |
| `forest_features` | The features, with their area, roundness and highest probability, and their cells |

## Vignette

[vignette/ArchaeologicalDetection.tex](vignette/ArchaeologicalDetection.tex) says how the networks are built and how they were checked against PyTorch.
