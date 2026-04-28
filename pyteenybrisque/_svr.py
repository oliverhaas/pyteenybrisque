"""RBF SVR scoring against the BRISQUE-trained support vectors."""

from importlib import resources

import numpy as np
import numpy.typing as npt

with resources.files(__package__).joinpath("_weights/svm.npz").open("rb") as _fh:
    _W = np.load(_fh)
    _SV: npt.NDArray[np.float32] = _W["sv"]
    _SV_COEF: npt.NDArray[np.float32] = _W["sv_coef"]
    _GAMMA = float(_W["gamma"])
    _RHO = float(_W["rho"])
    _FEAT_MIN: npt.NDArray[np.float32] = _W["feat_min"]
    _FEAT_RANGE: npt.NDArray[np.float32] = _W["feat_range"]


def predict(features: npt.NDArray[np.float32]) -> float:
    scaled = (-1.0 + 2.0 * (features.astype(np.float64) - _FEAT_MIN) / _FEAT_RANGE).astype(np.float64)
    diff = scaled - _SV.astype(np.float64)
    dist = np.einsum("ij,ij->i", diff, diff)
    kernel = np.exp(-_GAMMA * dist)
    return float(kernel @ _SV_COEF.astype(np.float64) - _RHO)
