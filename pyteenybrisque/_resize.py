"""MATLAB-style bicubic 0.5x downsample with antialiasing.

Numpy port of pyiqa's `pyiqa.matlab_utils.resize.imresize` for the single
configuration this package needs: cubic kernel, antialiasing on, scale 0.5,
2D input. The reference implementation uses reflect padding where the
boundary pixel is duplicated -- equivalent to numpy's `mode='symmetric'`.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

_CUBIC_A = -0.5
_KERNEL_SIZE_ORIG = 4
_KERNEL_MARGIN = 2  # MATLAB imresize adds two-tap margin to each side


def _cubic(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Keys cubic (a = -0.5): support is [-2, 2], piecewise on [0, 1] and [1, 2].
    a = _CUBIC_A
    ax = np.abs(x)
    ax2 = ax * ax
    ax3 = ax * ax2
    inner, outer = 1.0, 2.0
    cont_01 = ((a + 2) * ax3 - (a + 3) * ax2 + 1) * (ax <= inner)
    cont_12 = (a * ax3 - 5 * a * ax2 + 8 * a * ax - 4 * a) * ((ax > inner) & (ax <= outer))
    return cont_01 + cont_12


def _resize_axis(x: npt.NDArray[np.float32], axis: int, scale: float) -> npt.NDArray[np.float32]:
    h_in = x.shape[axis]
    size = math.ceil(h_in * scale)

    if scale < 1:
        antialiasing_factor = scale
        kernel_size = math.ceil(_KERNEL_SIZE_ORIG / scale) + _KERNEL_MARGIN
    else:
        antialiasing_factor = 1.0
        kernel_size = _KERNEL_SIZE_ORIG + _KERNEL_MARGIN

    pos = (np.arange(size, dtype=np.float64) + 0.5) / scale - 0.5
    base = np.floor(pos).astype(np.int64) - (kernel_size // 2) + 1
    dist = pos - base.astype(np.float64)

    buffer_pos = (dist[None, :] - np.arange(kernel_size, dtype=np.float64)[:, None]) * antialiasing_factor
    weight = _cubic(buffer_pos)
    weight = weight / weight.sum(axis=0, keepdims=True)
    weight = weight.astype(x.dtype)

    r_min = int(base.min())
    r_max = int(base.max() + kernel_size - 1)
    pad_pre = max(0, -r_min)
    pad_post = max(0, r_max - h_in + 1)
    base = base + pad_pre

    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (pad_pre, pad_post)
    x_pad = np.pad(x, pad_width, mode="symmetric")

    idx = base[None, :] + np.arange(kernel_size, dtype=np.int64)[:, None]

    if axis == 0:
        gathered = x_pad[idx, :]
        return np.einsum("ks,ksw->sw", weight, gathered)
    gathered = x_pad[:, idx]
    return np.einsum("ks,hks->hs", weight, gathered)


def downsample_half(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    out = _resize_axis(img, axis=0, scale=0.5)
    return _resize_axis(out, axis=1, scale=0.5)
