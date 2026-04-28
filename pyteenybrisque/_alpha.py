"""Discrete-grid lookup of GGD/AGGD shape parameter alpha.

pyiqa's `estimate_ggd_param` / `estimate_aggd_param` perform an argmin search
over alpha in [0.2, 10] step 0.001. We bake the same grid into a vendored
npz and reproduce the search exactly.
"""

from importlib import resources

import numpy as np
import numpy.typing as npt

with resources.files(__package__).joinpath("_weights/alpha.npz").open("rb") as _fh:
    _T = np.load(_fh)
    _GAMMA: npt.NDArray[np.float32] = _T["gamma"]
    _R_GGD: npt.NDArray[np.float32] = _T["r_ggd"]
    _R_AGGD: npt.NDArray[np.float32] = _T["r_aggd"]
    for _arr in (_GAMMA, _R_GGD, _R_AGGD):
        _arr.flags.writeable = False


def find_alpha_ggd(rho: float) -> float:
    return float(_GAMMA[np.abs(_R_GGD - rho).argmin()])


def find_alpha_aggd(rhatnorm: float) -> float:
    return float(_GAMMA[np.abs(_R_AGGD - rhatnorm).argmin()])
