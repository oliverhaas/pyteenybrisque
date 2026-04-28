"""MATLAB-style bicubic 2x downsample with antialiasing.

Numpy port of pyiqa's `pyiqa.matlab_utils.resize.imresize` specialised to the
one configuration this package needs: cubic kernel, antialiasing on, scale
0.5, 2D input. The reference uses reflect padding where the boundary pixel
is duplicated -- equivalent to numpy's `mode='symmetric'`.
"""

import math

import numpy as np
import numpy.typing as npt

_CUBIC_A = -0.5
_SCALE = 0.5
# At scale 0.5 the antialiased cubic kernel grows to ceil(4 / 0.5) + 2 = 10 taps.
_KERNEL_SIZE = math.ceil(4 / _SCALE) + 2


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


def _resize_axis(x: npt.NDArray[np.floating], axis: int) -> npt.NDArray[np.floating]:
    h_in = x.shape[axis]
    size = math.ceil(h_in * _SCALE)

    pos = (np.arange(size, dtype=np.float64) + 0.5) / _SCALE - 0.5
    base = np.floor(pos).astype(np.int64) - (_KERNEL_SIZE // 2) + 1
    dist = pos - base.astype(np.float64)

    buffer_pos = (dist[None, :] - np.arange(_KERNEL_SIZE, dtype=np.float64)[:, None]) * _SCALE
    weight = _cubic(buffer_pos)
    weight = weight / weight.sum(axis=0, keepdims=True)
    weight = weight.astype(x.dtype)

    pad_pre = max(0, -int(base.min()))
    pad_post = max(0, int(base.max() + _KERNEL_SIZE - 1) - h_in + 1)
    base = base + pad_pre

    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (pad_pre, pad_post)
    x_pad = np.pad(x, pad_width, mode="symmetric")

    idx = base[None, :] + np.arange(_KERNEL_SIZE, dtype=np.int64)[:, None]

    if axis == 0:
        return np.einsum("ks,ksw->sw", weight, x_pad[idx, :])
    return np.einsum("ks,hks->hs", weight, x_pad[:, idx])


def downsample_half(img: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return _resize_axis(_resize_axis(img, axis=0), axis=1)
