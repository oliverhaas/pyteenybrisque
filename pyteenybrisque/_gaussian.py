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
_KERNEL.flags.writeable = False
_PAD = _KERNEL_SIZE // 2


def gauss_moments(
    img: npt.NDArray[np.floating | np.integer],
    start: int,
    stop: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rows [start, stop) of the zero-padded Gaussian filter of `img` and of `img**2`.

    Computes in float64 to keep the catastrophic cancellation in
    `mu_sq - mu**2` (used downstream to recover local variance) bounded.
    """
    h, w = img.shape
    padded_rows = np.zeros((stop - start + 2 * _PAD, w), dtype=np.float64)
    src_start, src_stop = max(start - _PAD, 0), min(stop + _PAD, h)
    dst_start = src_start - (start - _PAD)
    padded_rows[dst_start : dst_start + src_stop - src_start] = img[src_start:src_stop]
    mu = _filter(padded_rows)
    np.square(padded_rows, out=padded_rows)
    return mu, _filter(padded_rows)


def _filter(padded_rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    padded_cols = np.zeros((padded_rows.shape[0] - 2 * _PAD, padded_rows.shape[1] + 2 * _PAD), dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(padded_rows, _KERNEL_SIZE, axis=0)
    np.matmul(windows, _KERNEL, out=padded_cols[:, _PAD:-_PAD])
    return np.lib.stride_tricks.sliding_window_view(padded_cols, _KERNEL_SIZE, axis=1) @ _KERNEL
