"""pyiqa's MATLAB-style `imresize` (cubic, antialiased) fixed to a 2x downsample of a 2D image.

Borders reflect with the edge pixel repeated, as numpy's `mode='symmetric'` does.
Output pixel s reads 10 inputs from 2s - 4 with the same weights for every s.
"""

import math

import numpy as np
import numpy.typing as npt

_CUBIC_A = -0.5
_SCALE = 0.5
_STRIDE = round(1 / _SCALE)
# At scale 0.5 the antialiased cubic kernel grows to ceil(4 / 0.5) + 2 = 10 taps.
_KERNEL_SIZE = math.ceil(4 / _SCALE) + 2
# Input position of output pixel 0, and the first input pixel of its window.
_POS_0 = 0.5 / _SCALE - 0.5
_START_0 = math.floor(_POS_0) - _KERNEL_SIZE // 2 + 1
_PAD_PRE = -_START_0
_STRIP_PIXELS = 1 << 17


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


def _build_weight() -> npt.NDArray[np.float64]:
    dist = _POS_0 - _START_0 - np.arange(_KERNEL_SIZE, dtype=np.float64)
    weight = _cubic(dist * _SCALE)
    return weight / weight.sum()


# pyiqa resizes in float32.
_WEIGHT = _build_weight().astype(np.float32)
_WEIGHT.flags.writeable = False


def _padded_index(n: int) -> npt.NDArray[np.intp]:
    size = math.ceil(n * _SCALE)
    idx = np.mod(np.arange(-_PAD_PRE, _STRIDE * (size - 1) + _KERNEL_SIZE - _PAD_PRE), 2 * n)
    return np.where(idx < n, idx, 2 * n - 1 - idx)


def _windows(x: npt.NDArray[np.floating], axis: int) -> npt.NDArray[np.floating]:
    every_stride = [slice(None), slice(None)]
    every_stride[axis] = slice(None, None, _STRIDE)
    return np.lib.stride_tricks.sliding_window_view(x, _KERNEL_SIZE, axis=axis)[tuple(every_stride)]


def downsample_half(img: npt.NDArray[np.floating | np.integer]) -> npt.NDArray[np.float32]:
    h, w = img.shape
    rows, cols = _padded_index(h), _padded_index(w)
    out = np.empty((math.ceil(h * _SCALE), math.ceil(w * _SCALE)), dtype=np.float32)
    step = max(1, _STRIP_PIXELS // w)
    for top in range(0, out.shape[0], step):
        bottom = min(top + step, out.shape[0])
        slab = img[rows[_STRIDE * top : _STRIDE * (bottom - 1) + _KERNEL_SIZE]].astype(np.float32, copy=False)
        strip = _windows(slab, axis=0) @ _WEIGHT
        out[top:bottom] = _windows(strip[:, cols], axis=1) @ _WEIGHT
    return out
