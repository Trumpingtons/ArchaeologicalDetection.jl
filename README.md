# ArchaeologicalDetection.jl

Trained models that detect archaeological features, run in Julia: neural networks on lidar terrain models, among them a detector of tumuli with its Sentinel-2 soil mask, a random forest on Sentinel-2, and Enki, a classifier of tells on RGB imagery of 1 m; and the terrain model itself, made from a lidar point cloud.

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

After Orengo and 6 coauthors (2020), [doi:10.1073/pnas.2005583117](https://doi.org/10.1073/pnas.2005583117), who found mounds in Cholistan this way. A multitemporal composite of Sentinel-2 is the mean reflectance of many scenes, each without its cloudy cells, in ten bands at 10 m. A random forest of 128 trees is trained on the cells of known sites and on cells away from them, and gives at every cell the share of its trees that vote for a site. The scenes are those of Level-2A of [Earth Search](https://element84.com/earth-search/), read without an account from Cloud-Optimised GeoTIFFs, only the cells of the area.

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

## Tumuli of North-Western Iberia

The tumulus detector of Berganzo-Besga, Orengo, Lumbreras, Carrero-Pazos, Fonte and Vilas-Estévez (2021), [doi:10.3390/rs13204181](https://doi.org/10.3390/rs13204181), which found 10,527 tumuli in Galicia. It is a YOLOv3 network of Darknet, trained on the burial mounds of Galicia and northern Portugal in the Multi-Scale Relief Model of Orengo and Petrie (2018), [doi:10.1002/esp.4317](https://doi.org/10.1002/esp.4317), computed from the Xunta de Galicia's 1 m terrain model (`MDT_1m_h50`).

Its weights, about 246 MB, are downloaded at first use from the authors' link and checked by their SHA-256. The network's configuration comes from a fixed commit of the authors' repository. Darknet's files are read in Julia, and the network runs with NNlib.jl.

The authors set to 0 the relief of the land where tumuli cannot be, such as towns, roads, water and rock. They found that land with a random forest of 128 trees on a multitemporal Sentinel-2 composite. Here the forest is trained on the cells of their 32 polygons, shipped in `data/tumuli_soil.csv`, in the Level-2A scenes of Earth Search from November 2016 to 5 June 2020.

```julia
using ArchaeologicalDetection, Rasters, ArchGDAL

dem = replace_missing(read(Raster("MDT_1m_h50_301.img")), NaN32)  # a sheet of the Xunta's terrain model, or a part of it
model = tumuli_model(tumuli_weights("models"))
scenes = filter(s -> s.acquired < "2020-06-06", sentinel_scenes(lonlat_box(dem); years = 2016:2020, cloud = 20))
composite = sentinel_composite(scenes, lonlat_box(dem); path = "sentinel.tif")  # written once, read from there later
soil = tumuli_soil(tumuli_soil_model(), composite; to = dem)  # 1 on land where tumuli can be, on the grid of dem
image = tumuli_input(dem; soil)  # or soil = nothing, for no soil mask
boxes = tumuli_boxes(model, image)  # tiles of 1024 cells from the north-west corner, as the authors cut Galicia
found = tumuli_features(boxes, image; min_similarity = 0.25)
found.candidates  # id, x, y, size and similarity of each tumulus
```

On the authors' 30 validation images, the Julia network finds the same 72 tumuli as Darknet built from the authors' repository. 71 have the same centre and size to the cell and the same similarity to the percent Darknet prints; the last has a similarity of 95.5%, which Darknet prints as 95% and Julia rounds to 96%. On the Xunta's terrain model of the authors' tile 41722, without the soil mask, it finds 59 of the 60 tumuli they list there, a median of 0.75 m from their positions.

| Function | Returns |
|---|---|
| `tumuli_weights` | The file of the trained detector, downloaded once, with the network's configuration beside it |
| `tumuli_model` | The detector, a `DarknetModel` |
| `darknet_model` | A YOLOv3 network from a Darknet configuration and weights file |
| `darknet_boxes` | The boxes Darknet finds in an image of the network's size, with their class and probability |
| `msrm` | The Multi-Scale Relief Model of a terrain model, as the authors computed it in Earth Engine |
| `tumuli_input` | The image the detector reads: the Multi-Scale Relief Model, 0 where it is 0 or below and off the soil mask, in the authors' grey levels |
| `tumuli_boxes` | The boxes of the tumuli found, tile by tile, with their centres, sides and similarity |
| `tumuli_features` | The tumuli of at least a similarity, with their centres, size and similarity, and their cells |
| `tumuli_soil_cells` | The authors' training cells of the soil mask, with their bands |
| `tumuli_soil_model` | The random forest of the soil mask, trained on those cells |
| `tumuli_soil` | The land where tumuli can be found, at each cell of a composite, or on the grid of a terrain model |
| `lonlat_box` | The box of a raster in degrees, for `sentinel_scenes` |

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

## Terrain Model from a Point Cloud

A bare-earth terrain model made from the points of a lidar survey. [PDAL](https://pdal.io), through PDAL_jll, reads LAS and LAZ files, local or on the web, and COPC files over the web, only the part within an area, and gathers them into one LAZ file, which LazIO.jl reads into Julia. The ground is kept as the survey classified it, classes 2 and 8, or found again with the progressive morphological filter of Zhang, Chen, Whitman, Shyu, Yan and Zhang (2003), [doi:10.1109/TGRS.2003.810682](https://doi.org/10.1109/TGRS.2003.810682), or the simple morphological filter of Pingel, Clarke and McBride (2013), [doi:10.1016/j.isprsjprs.2012.12.002](https://doi.org/10.1016/j.isprsjprs.2012.12.002), both of Geomorphometry.jl. The cells without ground are filled by inverse distance weighting from the nearest cells with ground, up to a distance, and the ground points of each cell are counted, so that what was filled shows.

```julia
using ArchaeologicalDetection, Rasters, ArchGDAL

area = (696600, 1968700, 697700, 1970300)  # xmin, ymin, xmax, ymax, in the coordinates of the points
tiles = index_tiles("/vsizip/MX19_Inomata_TileIndex.zip/MX19_Inomata_TileIndex.shp", area)  # a tile index; or files, a folder, a COPC address
cloud = fetch_points(tiles, "aguada-fenix.laz"; bounds = area, folder = "tiles")  # written once, read from there later
info = point_cloud_summary(cloud)
points = read_points(cloud)
ground = ground_grid(points, 1.0; bounds = area)  # or find_ground(points, 1.0; method = :zhang, bounds = area)
terrain = fill_gaps(ground; distance = 10)
dem = terrain_raster(terrain, EPSG(info.epsg))  # north up
write("aguada-fenix-dtm.tif", dem; driver = "COG")
```

| Function | Returns |
|---|---|
| `point_cloud_summary` | The number of points, the bounds and the coordinate system of a point cloud, from its header |
| `index_tiles` | The tiles of a tile index whose polygons cross an area |
| `fetch_points` | The points of LAS, LAZ and COPC files within an area, gathered into one LAZ file, written once |
| `read_points` | The points of a LAS or LAZ file |
| `point_classes` | The number of points of each class |
| `ground_grid` | The lowest point of the ground classes in each cell, and the number of them |
| `lowest_grid` | The lowest point of each cell but noise, whatever its class |
| `find_ground` | The ground found again by Zhang's or Pingel's filter, and the ground points of each cell |
| `fill_gaps` | The cells without ground filled from the nearest cells with ground, up to a distance, and which were filled |
| `ground_density` | The ground points a square metre of each cell |
| `terrain_raster` | A grid of heights as a `Raster`, north up |

## Vignette

[vignette/ArchaeologicalDetection.tex](vignette/ArchaeologicalDetection.tex) says how the networks are built and how they were checked against PyTorch and Keras.
