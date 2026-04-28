# pyteenybrisque

Tiny numpy-only BRISQUE no-reference image quality scorer.

```python
import pyteenybrisque

score = pyteenybrisque.score("photo.jpg")
```

A single public function returns a BRISQUE no-reference quality score.
Lower is better; scores typically fall in `[0, 100]`.

## Inputs

`score()` accepts:

- a path (`str` or `os.PathLike`)
- a `PIL.Image.Image`
- a numpy array, either `HxW` grayscale or `HxWx{3,4}` RGB / RGBA,
  with dtype `uint8` or float in `[0, 1]`

Color images are converted to BT.601 luma before scoring.

## How it works

The implementation reproduces the original MATLAB BRISQUE pipeline:

1. Convert to luma (Y channel of YIQ, BT.601 weights).
2. Compute the MSCN (mean-subtracted contrast-normalised) map using a
   separable 7-tap Gaussian.
3. Fit a generalised Gaussian to the MSCN values and an asymmetric
   generalised Gaussian to four shifted-product MSCN maps.
   That gives 18 features per scale.
4. Repeat at half-resolution (matlab-style bicubic with antialiasing).
5. Run the resulting 36-feature vector through an RBF SVR trained on
   LIVE IQA.

The vendored model weights, the SVR support vectors, and the alpha lookup
table together take ~250 KB. The only runtime dependencies are numpy and
Pillow.

## Agreement with pyiqa

The score agrees with [`pyiqa`](https://github.com/chaofengc/IQA-PyTorch)'s
BRISQUE within ~0.1 BRISQUE points on natural images. Bit-for-bit agreement
isn't possible from numpy alone -- pyiqa's float32 PyTorch conv2d adds
~0.01-1.0 point noise via catastrophic cancellation in the local-variance
step. This package uses float64 for the Gaussian, so it's typically closer
to the original MATLAB reference than pyiqa is.

## Limitations

- Single image, single score, sync. No batch API, no GPU, no async.
- Color BRISQUE is not implemented (uses the standard luma-only variant).
- ~130 ms per 1 MP image on a modern CPU.
