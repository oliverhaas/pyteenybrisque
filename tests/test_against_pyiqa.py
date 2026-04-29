"""End-to-end check that `pyteenybrisque.score` agrees with `pyiqa`'s BRISQUE
within a tolerance sized to the cross-implementation noise floor."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import pyteenybrisque

_DATA = Path(__file__).parent / "data"
_IMAGES = sorted(p for p in _DATA.iterdir() if p.suffix.lower() in {".jpg", ".png", ".webp"})

# Lower is better. Typical scores 0-100. A 1-point disagreement is
# perceptually meaningless -- it's swamped by pyiqa's float32 noise.
_TOL = 1.0


@pytest.fixture(scope="module")
def pyiqa_metric():
    pyiqa = pytest.importorskip("pyiqa")
    return pyiqa.create_metric("brisque")


def _pyiqa_score(metric, path: Path) -> float:
    arr = np.asarray(Image.open(path).convert("RGB")).copy()
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return float(metric(tensor).item())


@pytest.mark.parametrize("path", _IMAGES, ids=lambda p: p.name)
def test_score_matches_pyiqa(pyiqa_metric, path):
    expected = _pyiqa_score(pyiqa_metric, path)
    actual = pyteenybrisque.score(image=path)
    assert abs(actual - expected) < _TOL, f"{path.name}: pyiqa={expected:.4f}, mine={actual:.4f}"


def test_accepts_pil_image():
    path = _IMAGES[0]
    direct = pyteenybrisque.score(image=path)
    via_pil = pyteenybrisque.score(image=Image.open(path))
    assert direct == pytest.approx(via_pil, abs=1e-9)


def test_accepts_numpy_array():
    path = _IMAGES[0]
    arr = np.asarray(Image.open(path).convert("RGB"))
    direct = pyteenybrisque.score(image=path)
    via_arr = pyteenybrisque.score(image=arr)
    assert direct == pytest.approx(via_arr, abs=1e-9)


def test_accepts_str_and_path_equivalently():
    path = _IMAGES[0]
    via_path = pyteenybrisque.score(image=path)
    via_str = pyteenybrisque.score(image=str(path))
    assert via_path == via_str


def test_grayscale_array():
    arr = np.asarray(Image.open(_IMAGES[0]).convert("L"))
    s = pyteenybrisque.score(image=arr)
    assert isinstance(s, float)
    assert np.isfinite(s)


def test_deterministic():
    path = _IMAGES[0]
    a = pyteenybrisque.score(image=path)
    b = pyteenybrisque.score(image=path)
    assert a == b


def test_rejects_unsupported_shape():
    bad = np.zeros((4, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="unsupported image shape"):
        pyteenybrisque.score(image=bad)
