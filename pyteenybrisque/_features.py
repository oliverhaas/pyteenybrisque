"""BRISQUE NSS features: MSCN map -> GGD + 4 AGGD fits = 18 features per scale."""

import math

import numpy as np
import numpy.typing as npt

from ._alpha import find_alpha_aggd, find_alpha_ggd
from ._gaussian import gauss_filter

_EPS = float(np.finfo(np.float32).eps)
_SHIFTS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (-1, 1))


def _ggd(mscn: npt.NDArray[np.float32]) -> tuple[float, float]:
    sigma_sq = float((mscn * mscn).mean())
    e = float(np.abs(mscn).mean())
    rho = sigma_sq / (e * e)
    alpha = find_alpha_ggd(rho)
    return alpha, sigma_sq


def _aggd(block: npt.NDArray[np.float32]) -> tuple[float, float, float, float]:
    mask_left = block < 0
    mask_right = block > 0
    count_left = int(mask_left.sum())
    count_right = int(mask_right.sum())

    sq = block * block
    left_std = float(np.sqrt(sq[mask_left].sum() / count_left))
    right_std = float(np.sqrt(sq[mask_right].sum() / count_right))

    gammahat = left_std / right_std
    abs_mean = float(np.abs(block).mean())
    sq_mean = float(sq.mean())
    rhat = (abs_mean * abs_mean) / sq_mean
    rhatnorm = rhat * (gammahat**3 + 1.0) * (gammahat + 1.0) / (gammahat * gammahat + 1.0) ** 2

    alpha = find_alpha_aggd(rhatnorm)
    log_eta = math.lgamma(2.0 / alpha) - (math.lgamma(1.0 / alpha) + math.lgamma(3.0 / alpha)) / 2.0
    eta = (right_std - left_std) * math.exp(log_eta)
    return alpha, eta, left_std, right_std


def features_per_scale(luma: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """18 BRISQUE NSS features for a single scale of a luma image in [0, 255]."""
    mu = gauss_filter(luma)
    mu_sq = gauss_filter(luma * luma)
    sigma = np.sqrt(np.abs(mu_sq - mu * mu) + _EPS)
    mscn = (luma - mu) / (sigma + 1.0)

    out = np.empty(18, dtype=np.float32)
    alpha, sigma_sq = _ggd(mscn)
    out[0] = alpha
    out[1] = sigma_sq

    for i, shift in enumerate(_SHIFTS):
        shifted = np.roll(mscn, shift, axis=(0, 1))
        alpha, eta, left_std, right_std = _aggd(mscn * shifted)
        base = 2 + 4 * i
        out[base] = alpha
        out[base + 1] = eta
        out[base + 2] = left_std * left_std
        out[base + 3] = right_std * right_std
    return out
