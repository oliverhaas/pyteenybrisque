"""Training custom BRISQUE weights with `pyteenybrisque.train`."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pyteenybrisque
from pyteenybrisque import train

_DATA = Path(__file__).parent / "data"
_CLEAN = [_DATA / f"{name}.jpg" for name in ("img_10", "img_23", "img_2", "img_47", "img_88", "src134")]


def _small(path):
    return Image.open(path).convert("RGB").resize((192, 128))


def test_extract_features_rows_match_across_workers():
    images = [np.asarray(_small(p)) for p in _CLEAN[:4]]
    serial = train.extract_features(images=images, workers=1)
    parallel = train.extract_features(images=iter(images), workers=2)
    np.testing.assert_array_equal(serial[2], pyteenybrisque.features(image=images[2]))
    np.testing.assert_array_equal(parallel, serial)


def test_extract_features_gives_nan_row_for_failed_image():
    images = [np.zeros((64, 64), np.uint8), np.asarray(_small(_CLEAN[0]))]
    with pytest.warns(RuntimeWarning, match="1 of 2 images failed"):
        rows = train.extract_features(images=images, workers=1)
    assert np.isnan(rows[0]).all()
    assert np.isfinite(rows[1]).all()
