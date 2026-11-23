# ArchaeologicalDetection.jl

Trained neural networks that detect archaeological features, run in Julia.

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

## Vignette

[vignette/ArchaeologicalDetection.tex](vignette/ArchaeologicalDetection.tex) says how the networks are built and how they were checked against PyTorch.
