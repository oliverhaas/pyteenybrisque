"""Training custom BRISQUE weights with `pyteenybrisque.train`."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter
from scipy.stats import spearmanr

import pyteenybrisque
from pyteenybrisque import train

_DATA = Path(__file__).parent / "data"
_CLEAN = [_DATA / f"{name}.jpg" for name in ("img_10", "img_23", "img_2", "img_47", "img_88", "src134")]
_RADII = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_C, _GAMMA = 8.0, 0.03125
_SOLVER_TOL = 0.01
_MIN_SROCC = 0.8


def _small(path):
    return Image.open(path).convert("RGB").resize((192, 128))


@pytest.fixture(scope="module")
def blur_set():
    """Features and blur radii of all but the last clean image, blurred at each radius."""
    sources = [_small(p) for p in _CLEAN[:-1]]
    images = [src.filter(ImageFilter.GaussianBlur(r)) for src in sources for r in _RADII]
    radii = np.array(_RADII * len(sources))
    return train.extract_features(images=images, workers=1), radii


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


def test_trained_weights_rank_held_out_blur(blur_set):
    x, radii = blur_set
    weights = train.fit(features=x, labels=list(radii), c=_C, gamma=_GAMMA, workers=1)
    held_out = _small(_CLEAN[-1])
    scores = [pyteenybrisque.score(image=held_out.filter(ImageFilter.GaussianBlur(r)), weights=weights) for r in _RADII]
    assert spearmanr(scores, _RADII).statistic > _MIN_SROCC


def test_label_scale_carries_through_to_scores(blur_set):
    x, radii = blur_set
    shifted_labels = radii * 10 + 100
    base = train.fit(features=x, labels=radii, c=_C, gamma=_GAMMA, workers=1)
    shifted = train.fit(features=x, labels=shifted_labels, c=_C, gamma=_GAMMA, workers=1)
    image = _small(_CLEAN[-1])
    expected = 10 * pyteenybrisque.score(image=image, weights=base) + 100
    actual = pyteenybrisque.score(image=image, weights=shifted)
    assert actual == pytest.approx(expected, abs=_SOLVER_TOL * shifted_labels.std())


def test_fit_tunes_missing_settings_and_reports_cross_validation(blur_set):
    x, radii = blur_set
    ratings = (radii * 2).astype(int)
    weights = train.fit(features=x, labels=ratings, gamma=_GAMMA, tune_samples=20, folds=2, workers=1)
    assert weights.info["gamma"] == _GAMMA
    assert weights.info["c"] > 0
    assert weights.info["n_train"] == len(ratings)
    assert np.isfinite(weights.info["cv_rmse"])
    assert np.isfinite(weights.info["cv_srocc"])


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda x, y: (x[:, :35], y), "features must have shape"),
        (lambda x, y: (x, y[:-1]), "labels must have shape"),
        (lambda x, y: (np.vstack([x, np.full((1, 36), np.nan)]), np.append(y, 1.0)), "not finite"),
        (lambda x, y: (x, np.where(np.arange(len(y)) == 0, np.nan, y)), "labels must be finite"),
        (lambda x, y: (x[:5], y[:5]), "cross-validation"),
        (lambda x, y: (x, np.ones_like(y)), "equal"),
    ],
    ids=["feature width", "label count", "nan feature row", "nan label", "too few rows", "equal labels"],
)
def test_fit_rejects_unusable_training_data(blur_set, change, match):
    x, y = change(*blur_set)
    with pytest.raises(ValueError, match=match):
        train.fit(features=x, labels=y, c=_C, gamma=_GAMMA, workers=1)


def test_import_names_extra_when_sklearn_missing(monkeypatch):
    for name in ["sklearn", *(m for m in sys.modules if m.startswith("sklearn."))]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "pyteenybrisque.train")
    with pytest.raises(ImportError, match=r"pyteenybrisque\[train\]"):
        importlib.import_module("pyteenybrisque.train")
