# ArchaeologicalDetection.jl

Trained models that detect archaeological features, run in Julia: neural networks on lidar terrain models, a random forest on Sentinel-2, and Enki, a classifier of tells on RGB imagery of 1 m.

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

## Enki on Imagery of 1 m

[Enki](https://github.com/archeorosati/enki), by Paolo Rosati (2025), classifies RGB images of cells of 350 by 350 m, at 1 m, for tells and mound-like features of the Ancient Near East: EfficientNet-B0 of Keras with two dense layers on top, trained on more than 21,500 images of Google and Bing imagery ([dataset](https://doi.org/10.5281/zenodo.14950564)). Its Keras file, about 52 MB, is downloaded at first use from a fixed commit of Enki's repository and read with JLD2.jl and ZipArchives.jl, without Python. The network runs with NNlib.jl. Its two dense layers can be trained again on known sites, the network beneath unchanged, as the first stage of Enki's own training trains them.

```julia
using ArchaeologicalDetection, Rasters, ArchGDAL

box = (41.02, 36.66, 41.16, 36.77)  # west, south, east, north, in degrees
enki = enki_model(enki_weights("models"))
image = aerial_image(box; source = :esri, path = "image.tif")  # ENV["ARCGIS_API_KEY"]; or :google, "WMS:...", a file
cells = enki_pooled(enki, image)  # the network's 1,280 channels for every cell of 350 m
probability = enki_probability(enki, cells)  # Enki as published
points = site_points(image, "known-sites.gpkg")
examples = enki_examples(enki, image, cells, points)
tuned = enki_finetune(enki, examples.pooled, examples.labels)  # or train the dense layers on the known sites
enki_save("models/tuned-head.jld2", tuned.model)  # the two dense layers; enki_model(weights; head = ...) reads them back
probability = enki_probability(tuned.model, cells)
found = enki_features(probability; share = 0.1)  # the tenth of the cells of highest share; threshold = 0.99 for Enki as published
```

| Function | Returns |
|---|---|
| `enki_weights` | Enki's Keras file, downloaded once |
| `enki_model` | Enki's network, read from its Keras file, with the head of a file of `enki_save` if given |
| `aerial_image` | An RGB image of an area at 1 m in its UTM zone, from Esri World Imagery, Google's satellite tiles, a WMS or a raster file |
| `site_points` | The centres of the features of a vector file, in the coordinates of a raster |
| `enki_pooled` | The network's 1,280 averaged channels, for images or for every cell of 350 m of an image |
| `enki_examples` | Images of the known sites and cells away from them, by their channels, to train on |
| `enki_finetune` | Enki with its two dense layers trained again, and the loss and accuracy of each epoch |
| `enki_save` | The two dense layers of a model written to a JLD2 file, to read back with `enki_model` |
| `enki_probability` | The share for a tell, of images or at each cell of 350 m |
| `enki_features` | The features, with their area, roundness and highest share, and their cells: the cells at or above a threshold, or a share of the cells, those of highest share first |

## Vignette

[vignette/ArchaeologicalDetection.tex](vignette/ArchaeologicalDetection.tex) says how the networks are built and how they were checked against PyTorch and Keras.
