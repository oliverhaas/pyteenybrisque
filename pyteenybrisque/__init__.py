"""Minimal BRISQUE no-reference image quality scorer (numpy + Pillow)."""

from os import PathLike

import numpy as np
import numpy.typing as npt
from PIL import Image

from ._features import features_per_scale
from ._resize import downsample_half
from ._svr import BUNDLED_WEIGHTS, Weights, predict

__all__ = ["Weights", "features", "score"]

# BT.601 luma weights -- matches pyiqa's `to_y_channel` (YIQ Y channel).
_LUMA_RGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)
_LUMA_RGB.flags.writeable = False
_GRAY_NDIM = 2
_RGB_NDIM = 3
_VALID_CHANNELS = frozenset({3, 4})
_STRIP_PIXELS = 1 << 17


def _to_luma(image: object) -> npt.NDArray[np.uint8]:
    if isinstance(image, (str, PathLike)):
        with Image.open(image) as pil:
            return _to_luma(pil)

    if isinstance(image, Image.Image):
        w, h = image.size
        step = max(1, _STRIP_PIXELS // w)
        strips = (np.asarray(image.crop((0, top, w, min(top + step, h))).convert("RGB")) for top in range(0, h, step))
    else:
        arr = np.asarray(image)
        if arr.ndim == _GRAY_NDIM:
            arr = np.broadcast_to(arr[..., None], (*arr.shape, 3))
        elif arr.ndim != _RGB_NDIM or arr.shape[2] not in _VALID_CHANNELS:
            raise ValueError(f"unsupported image shape {arr.shape}")
        h, w = arr.shape[:2]
        step = max(1, _STRIP_PIXELS // w)
        strips = (arr[top : top + step, :, :3] for top in range(0, h, step))

    luma = np.empty((h, w), dtype=np.uint8)
    for top, strip in zip(range(0, h, step), strips, strict=True):
        rgb01 = strip.astype(np.float32) / 255.0 if np.issubdtype(strip.dtype, np.integer) else strip.astype(np.float32)
        luma[top : top + step] = np.clip(np.round((rgb01 @ _LUMA_RGB) * 255.0), 0, 255)
    return luma


def score(*, image: object, weights: Weights | None = None) -> float:
    """BRISQUE quality score, ~0-100 with lower meaning higher quality, or on the label scale of custom `weights`.

    Accepts a path (str / `os.PathLike`), a `PIL.Image.Image`, or a numpy
    array (HxW grayscale or HxWx{3,4} RGB / RGBA, uint8 or float in [0, 1]).
    """
    if weights is None:
        weights = BUNDLED_WEIGHTS
    elif not isinstance(weights, Weights):
        raise TypeError(
            f"weights must be a pyteenybrisque.Weights, got {type(weights).__name__}; read a file with Weights.load(path)",
        )
    return predict(features(image=image), weights)


def features(*, image: object) -> npt.NDArray[np.float32]:
    """The 36 BRISQUE features of an image: 18 at full size, then 18 at half size."""
    luma = _to_luma(image)
    return np.concatenate([features_per_scale(luma), features_per_scale(downsample_half(luma))])
