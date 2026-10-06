"""Train BRISQUE regression weights on labelled images; needs `pip install pyteenybrisque[train]`."""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from . import features as features_of

if TYPE_CHECKING:
    from collections.abc import Iterable

_N_FEATURES = 36
_MAX_LISTED_FAILURES = 10


def extract_features(*, images: Iterable[object], workers: int | None = None) -> npt.NDArray[np.float32]:
    """BRISQUE features of each image as one row of an (n, 36) array; an image that fails gets a NaN row."""
    images = list(images)
    workers = workers or os.process_cpu_count() or 1
    if workers == 1:
        results = [_features_or_error(image) for image in images]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_features_or_error, images))

    rows = np.full((len(images), _N_FEATURES), np.nan, dtype=np.float32)
    failures = []
    for i, result in enumerate(results):
        if isinstance(result, str):
            failures.append(f"{i}: {result}")
        else:
            rows[i] = result
    if failures:
        listed = "; ".join(failures[:_MAX_LISTED_FAILURES])
        more = "; ..." if len(failures) > _MAX_LISTED_FAILURES else ""
        warnings.warn(
            f"{len(failures)} of {len(images)} images failed and got NaN features: {listed}{more}",
            RuntimeWarning,
            stacklevel=2,
        )
    return rows


def _features_or_error(image: object) -> npt.NDArray[np.float32] | str:
    try:
        return features_of(image=image)
    except Exception as e:  # noqa: BLE001 -- one bad image must not discard the rest of the batch
        return f"{type(e).__name__}: {e}"
