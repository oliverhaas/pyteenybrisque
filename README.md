# pyteenybrisque

[![PyPI version](https://img.shields.io/pypi/v/pyteenybrisque.svg?style=flat)](https://pypi.org/project/pyteenybrisque/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyteenybrisque.svg)](https://pypi.org/project/pyteenybrisque/)
[![CI](https://github.com/oliverhaas/pyteenybrisque/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverhaas/pyteenybrisque/actions/workflows/ci.yml)

Tiny BRISQUE no-reference image quality scorer. One call, two runtime
dependencies (`numpy` and `Pillow`), ~250 KB of vendored model weights.

```python
import pyteenybrisque

score = pyteenybrisque.score(image="photo.jpg")
print(score)  # lower is better; ~0-100 scale
```

`score()` accepts a path, a `PIL.Image.Image`, or a numpy array (`HxW`
grayscale or `HxWx{3,4}` RGB / RGBA, uint8 or float in `[0, 1]`).
To score with weights trained on your own labelled images, see
[Training on your own labels](#training-on-your-own-labels).

## Installation

```console
pip install pyteenybrisque
```

## Training on your own labels

The bundled weights were trained on LIVE IQA and its five distortion types.
For images from another domain, label a few thousand of them with a quality
rating (for example 1-5, or a mean opinion score) and train new weights:

```console
pip install pyteenybrisque[train]
```

```python
from pathlib import Path

from pyteenybrisque import train

if __name__ == "__main__":  # extract_features starts worker processes
    paths = sorted(Path("photos").glob("*.jpg"))
    labels = [...]  # one rating per path
    features = train.extract_features(images=paths, workers=8)
    weights = train.fit(features=features, labels=labels)
    print(weights.info)  # chosen c and gamma, cross-validated RMSE and Spearman correlation
    weights.save("my_weights.npz")
```

Scoring with the saved weights needs only the base install:

```python
import pyteenybrisque

weights = pyteenybrisque.Weights.load("my_weights.npz")
pyteenybrisque.score(image="photo.jpg", weights=weights)
```

With your weights, `score()` predicts on the scale of your labels, so higher
means better if your ratings say so. `extract_features` gives a NaN row and a
warning for an image it cannot score; drop those rows and their labels
before `fit`. Keep the features with `np.save` to refit without extracting
them again.

## What it computes

BRISQUE (Mittal, Moorthy, Bovik 2012) is a no-reference image quality metric.
It extracts 36 natural-scene-statistics features from the luma channel at two
scales and runs them through an RBF SVR trained on LIVE IQA. Lower scores mean
higher perceived quality.

The implementation matches [`pyiqa`](https://github.com/chaofengc/IQA-PyTorch)'s
BRISQUE within ~0.1 BRISQUE points on natural images.

## How it compares

Each metric in the table below was scored on the [Kodak True Color test
set](https://r0k.us/graphics/kodak/) (8 lossless 768×512 PNGs) under six
degradation sweeps. Per source and metric, scores are min-max normalised
across the sweep so 0 = best in run, 1 = worst; the line is the median
across sources, the shaded band is the inter-quartile range.

<table>
<tr>
<td><img src="https://raw.githubusercontent.com/oliverhaas/pyteenybrisque/main/benchmarks/jpeg_quality.png" alt="JPEG quality sweep" width="100%"/></td>
<td><img src="https://raw.githubusercontent.com/oliverhaas/pyteenybrisque/main/benchmarks/webp_quality.png" alt="WebP quality sweep" width="100%"/></td>
<td><img src="https://raw.githubusercontent.com/oliverhaas/pyteenybrisque/main/benchmarks/gaussian_blur.png" alt="Gaussian blur sweep" width="100%"/></td>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/oliverhaas/pyteenybrisque/main/benchmarks/gaussian_noise.png" alt="Gaussian noise sweep" width="100%"/></td>
<td><img src="https://raw.githubusercontent.com/oliverhaas/pyteenybrisque/main/benchmarks/blocky_upscale.png" alt="Blocky upscale sweep" width="100%"/></td>
<td><img src="https://raw.githubusercontent.com/oliverhaas/pyteenybrisque/main/benchmarks/blurry_upscale.png" alt="Blurry upscale sweep" width="100%"/></td>
</tr>
</table>

BRISQUE is competitive with the deep-learning metrics on every degradation.
The benchmark script lives at `tools/benchmark_metrics.py` and is
reproducible end-to-end.

## Why "teeny"

`pyiqa` is the right tool if you want every IQA metric in one place. It pulls
in PyTorch and ~2 GB of dependencies. This package does one metric, on top of
just `numpy` and `Pillow`, in ~250 KB. Use it when BRISQUE is all you need.

## License

MIT
