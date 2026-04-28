# pyteenybrisque

[![PyPI version](https://img.shields.io/pypi/v/pyteenybrisque.svg?style=flat)](https://pypi.org/project/pyteenybrisque/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyteenybrisque.svg)](https://pypi.org/project/pyteenybrisque/)
[![CI](https://github.com/oliverhaas/pyteenybrisque/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverhaas/pyteenybrisque/actions/workflows/ci.yml)

Tiny numpy-only BRISQUE no-reference image quality scorer. One function, two
runtime dependencies (`numpy` and `Pillow`), ~250 KB of vendored model weights.

```python
import pyteenybrisque

score = pyteenybrisque.score("photo.jpg")
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
BRISQUE within ~0.1 BRISQUE points on natural images. Bit-for-bit agreement
isn't possible from numpy alone -- pyiqa's float32 PyTorch conv2d adds its own
~0.01-1.0 point noise via the catastrophic cancellation in the local-variance
step.

## Why "teeny"

`pyiqa` is the right tool if you want every IQA metric in one place. It pulls
in PyTorch and ~2 GB of dependencies. This package does one metric, in pure
numpy, in ~250 KB. Use it when BRISQUE is all you need.

## Documentation

[oliverhaas.github.io/pyteenybrisque](https://oliverhaas.github.io/pyteenybrisque/)

## License

MIT
