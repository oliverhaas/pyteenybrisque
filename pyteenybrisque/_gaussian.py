"""Separable 7-tap Gaussian filter with zero ('same') padding.

Matches `pyiqa.archs.func_util.normalize_img_with_gauss` byte-for-byte:
that path uses `fspecial(7, 7/6)` (a 2D isotropic Gaussian which factors
exactly into two 1D Gaussians of the same sigma) followed by `imfilter`
with `padding='same'` (which collapses to zero padding).
"""

import numpy as np
import numpy.typing as npt

_KERNEL_SIZE = 7
_SIGMA = 7.0 / 6.0


def _build_kernel() -> npt.NDArray[np.float64]:
    # pyiqa's `fspecial` builds a 2D Gaussian in float64 then `.float()`s the kernel
    # to float32. We separate the same isotropic Gaussian and round to float32 so
    # that downstream float64 arithmetic carries the exact same kernel coefficients.
    half = (_KERNEL_SIZE - 1) // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * _SIGMA * _SIGMA))
    k /= k.sum()
    return k.astype(np.float32).astype(np.float64)


_KERNEL = _build_kernel()
_PAD = _KERNEL_SIZE // 2


def _conv_axis(img: npt.NDArray[np.float64], axis: int) -> npt.NDArray[np.float64]:
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (_PAD, _PAD)
    padded = np.pad(img, pad_width)
    windows = np.lib.stride_tricks.sliding_window_view(padded, _KERNEL_SIZE, axis=axis)
    return windows @ _KERNEL


def gauss_filter(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """2D separable Gaussian, zero-padded to keep input shape.

    Computes in float64 to keep the catastrophic cancellation in
    `mu_sq - mu**2` (used downstream to recover local variance) bounded.
    """
    return _conv_axis(_conv_axis(img.astype(np.float64), axis=0), axis=1)
