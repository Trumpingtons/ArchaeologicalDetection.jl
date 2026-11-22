# ArchaeologicalDetection.jl

Trained neural networks that detect archaeological features, run in Julia.

The package holds a port of the HRNet segmentation models of [ADAF](https://github.com/EarthObservation/adaf), the Automatic Detection of Archaeological Features tool of Čož, Corns, Curran, Kocev and Kokalj (2026), [doi:10.1016/j.jasrep.2026.105733](https://doi.org/10.1016/j.jasrep.2026.105733). It has four models, trained on airborne laser scanning data of Ireland: all archaeology, barrows, enclosures and ringforts. The [weights](https://doi.org/10.5281/zenodo.15848663) are downloaded from Zenodo at first use, about 860 MB a model, and read with Pickle.jl without running Python.

```julia
using ArchaeologicalDetection, Rasters, ArchGDAL

dem = replace_missing(read(Raster("terrain.tif")), NaN32)  # a bare-earth terrain model in metres
model = adaf_model(adaf_weights("models"; feature = :all))  # :all, :barrow, :enclosure or :ringfort
probability = adaf_probability(model, dem)
found = adaf_features(probability; threshold = 0.5)
found.candidates
```

| Function | Returns |
|---|---|
| `adaf_weights` | The file of a trained model, downloaded once |
| `adaf_input` | The image the models read: the simple local relief model, from 0 to 1 |
| `adaf_model` | The network, with the weights of that file |
| `adaf_probability` | The probability of a feature at each cell |
| `adaf_features` | The features, with their area, roundness and highest probability, and their cells |

## Vignette

[vignette/ArchaeologicalDetection.tex](vignette/ArchaeologicalDetection.tex) says how the network is built and how it was checked against PyTorch.
