"""Scoring with custom `pyteenybrisque.Weights`, and saving and loading them."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import pyteenybrisque
from pyteenybrisque import Weights

_IMAGE = Path(__file__).parent / "data" / "src134.jpg"
_OFFSET = 7.0


def test_score_returns_prediction_of_custom_weights():
    weights = Weights(
        sv=np.zeros((1, 36)),
        sv_coef=np.zeros(1),
        gamma=0.05,
        rho=-_OFFSET,
        feat_min=np.zeros(36),
        feat_range=np.ones(36),
    )
    assert pyteenybrisque.score(image=_IMAGE, weights=weights) == _OFFSET


@pytest.fixture
def weights():
    rng = np.random.default_rng(0)
    return Weights(
        sv=rng.uniform(-1, 1, (5, 36)),
        sv_coef=rng.normal(size=5),
        gamma=0.05,
        rho=-3.0,
        feat_min=np.zeros(36),
        feat_range=np.full(36, 4.0),
        info={"cv_rmse": 1.5},
    )


def test_saved_weights_load_back_identically(tmp_path, weights):
    path = tmp_path / "my_weights"
    weights.save(path)
    loaded = Weights.load(path)
    assert pyteenybrisque.score(image=_IMAGE, weights=loaded) == pyteenybrisque.score(image=_IMAGE, weights=weights)
    assert dict(loaded.info) == {"cv_rmse": 1.5}


def test_custom_weights_score_in_worker_process(weights):
    with ProcessPoolExecutor(max_workers=1) as pool:
        remote = pool.submit(pyteenybrisque.score, image=_IMAGE, weights=weights).result()
    assert remote == pyteenybrisque.score(image=_IMAGE, weights=weights)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [("sv", np.zeros((5, 35)), "'sv' must have shape"), ("rho", None, "lacks rho")],
    ids=["wrong sv width", "missing rho"],
)
def test_load_rejects_malformed_file(tmp_path, key, value, match):
    arrays = {
        "sv": np.zeros((5, 36)),
        "sv_coef": np.zeros(5),
        "gamma": 0.05,
        "rho": 1.0,
        "feat_min": np.zeros(36),
        "feat_range": np.ones(36),
    }
    if value is None:
        del arrays[key]
    else:
        arrays[key] = value
    path = tmp_path / "bad.npz"
    np.savez(path, allow_pickle=False, **arrays)
    with pytest.raises(ValueError, match=match):
        Weights.load(path)


def test_score_rejects_path_as_weights():
    with pytest.raises(TypeError, match=r"Weights\.load"):
        pyteenybrisque.score(image=_IMAGE, weights="my_weights.npz")  # ty: ignore[invalid-argument-type]
