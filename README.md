# pyteenybrisque

[![PyPI version](https://img.shields.io/pypi/v/pyteenybrisque.svg?style=flat)](https://pypi.org/project/pyteenybrisque/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyteenybrisque.svg)](https://pypi.org/project/pyteenybrisque/)
[![CI](https://github.com/oliverhaas/pyteenybrisque/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverhaas/pyteenybrisque/actions/workflows/ci.yml)

Tiny BRISQUE no-reference image quality scorer. One function, two runtime
dependencies (`numpy` and `Pillow`), ~250 KB of vendored model weights.

```python
import pyteenybrisque

score = pyteenybrisque.score(image="photo.jpg")
print(score)  # lower is better; ~0-100 scale
```

`score()` accepts a path, a `PIL.Image.Image`, or a numpy array (`HxW`
grayscale or `HxWx{3,4}` RGB / RGBA, uint8 or float in `[0, 1]`).

## Installation

```console
pip install pyteenybrisque
```

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

The benchmark script lives at `tools/benchmark_metrics.py` and is
reproducible end-to-end.

## Why "teeny"

`pyiqa` is the right tool if you want every IQA metric in one place. It pulls
in PyTorch and ~2 GB of dependencies. This package does one metric, on top of
just `numpy` and `Pillow`, in ~250 KB. Use it when BRISQUE is all you need.

## License

MIT
