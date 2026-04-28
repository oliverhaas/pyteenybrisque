"""Minimal numpy-only BRISQUE no-reference image quality scorer."""

from os import PathLike

import numpy as np
import numpy.typing as npt
from PIL import Image

from ._features import features_per_scale
from ._resize import downsample_half
from ._svr import predict

__all__ = ["score"]

# BT.601 luma weights -- matches pyiqa's `to_y_channel` (YIQ Y channel).
_LUMA_RGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)
_GRAY_NDIM = 2
_RGB_NDIM = 3


def _to_luma(image: object) -> npt.NDArray[np.float32]:
    if isinstance(image, (str, PathLike)):
        arr = np.asarray(Image.open(image).convert("RGB"))  # ty: ignore[invalid-argument-type]
    elif isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)

    if arr.ndim == _GRAY_NDIM:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim != _RGB_NDIM or arr.shape[2] not in (3, 4):
        raise ValueError(f"unsupported image shape {arr.shape}")
    arr = arr[..., :3]

    rgb01 = arr.astype(np.float32) / 255.0 if np.issubdtype(arr.dtype, np.integer) else arr.astype(np.float32)
    luma01 = rgb01 @ _LUMA_RGB
    return np.round(luma01 * 255.0).astype(np.float32)


def score(image: object) -> float:
    """BRISQUE no-reference quality score (lower is better, ~0-100).

    Accepts a path (str / `os.PathLike`), a `PIL.Image.Image`, or a numpy
    array (HxW grayscale or HxWx{3,4} RGB / RGBA, uint8 or float in [0, 1]).
    """
    luma = _to_luma(image)
    feats = np.concatenate([features_per_scale(luma), features_per_scale(downsample_half(luma))])
    return predict(feats)
