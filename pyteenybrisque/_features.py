"""BRISQUE NSS features: MSCN map -> GGD + 4 AGGD fits = 18 features per scale.

The MSCN map is built and reduced one row strip at a time, so memory use is
bounded by the strip size instead of the image size.
"""

import math

import numpy as np
import numpy.typing as npt

from ._alpha import find_alpha_aggd, find_alpha_ggd
from ._gaussian import gauss_moments

_EPS = float(np.finfo(np.float32).eps)
# Neighbours as `np.roll(mscn, shift, axis=(0, 1))` would place them, wrapping at the border.
_SHIFTS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (-1, 1))
_STRIP_PIXELS = 1 << 17


def _mscn(luma: npt.NDArray[np.floating | np.integer], start: int, stop: int) -> npt.NDArray[np.float64]:
    mu, mu_sq = gauss_moments(luma, start, stop)
    sigma = np.sqrt(np.abs(mu_sq - mu * mu) + _EPS)
    return (luma[start:stop] - mu) / (sigma + 1.0)


def _signed_sums(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Count of x < 0, count of x > 0, sum of x**2 over each of those, sum of |x|."""
    neg = np.minimum(x, 0.0)
    pos = np.maximum(x, 0.0)
    # einsum keeps these sums out of OpenBLAS, which spreads dot products this long over all cores.
    sq_neg, sq_pos = np.einsum("ij,ij->", neg, neg), np.einsum("ij,ij->", pos, pos)
    return np.array([np.count_nonzero(neg), np.count_nonzero(pos), sq_neg, sq_pos, pos.sum() - neg.sum()])


def _ggd(sums: npt.NDArray[np.float64], n: int) -> tuple[float, float]:
    _, _, sq_left, sq_right, abs_sum = sums
    sigma_sq = float(sq_left + sq_right) / n
    e = float(abs_sum) / n
    rho = sigma_sq / (e * e)
    alpha = find_alpha_ggd(rho)
    return alpha, sigma_sq


def _aggd(sums: npt.NDArray[np.float64], n: int) -> tuple[float, float, float, float]:
    count_left, count_right, sq_left, sq_right, abs_sum = sums
    left_std = math.sqrt(sq_left / count_left)
    right_std = math.sqrt(sq_right / count_right)

    gammahat = left_std / right_std
    abs_mean = float(abs_sum) / n
    sq_mean = float(sq_left + sq_right) / n
    rhat = (abs_mean * abs_mean) / sq_mean
    rhatnorm = rhat * (gammahat**3 + 1.0) * (gammahat + 1.0) / (gammahat * gammahat + 1.0) ** 2

    alpha = find_alpha_aggd(rhatnorm)
    log_eta = math.lgamma(2.0 / alpha) - (math.lgamma(1.0 / alpha) + math.lgamma(3.0 / alpha)) / 2.0
    eta = (right_std - left_std) * math.exp(log_eta)
    return alpha, eta, left_std, right_std


def features_per_scale(luma: npt.NDArray[np.floating | np.integer]) -> npt.NDArray[np.float32]:
    """18 BRISQUE NSS features for a single scale of a luma image in [0, 255]."""
    h, w = luma.shape
    step = max(1, _STRIP_PIXELS // w)
    first_row, last_row = _mscn(luma, 0, 1), _mscn(luma, h - 1, h)
    # Row 0 sums the MSCN map, row 1 + i its products with neighbour i; columns as `_signed_sums`.
    totals = np.zeros((1 + len(_SHIFTS), 5))
    for start in range(0, h, step):
        stop = min(start + step, h)
        # MSCN rows start - 1 through stop, wrapping around at the top and bottom.
        rows = [_mscn(luma, max(start - 1, 0), min(stop + 1, h))]
        if start == 0:
            rows.insert(0, last_row)
        if stop == h:
            rows.append(first_row)
        mscn = np.concatenate(rows)
        center = mscn[1:-1]
        totals[0] += _signed_sums(center)
        for i, (dy, dx) in enumerate(_SHIFTS, start=1):
            neighbour = np.roll(mscn[1 - dy : len(mscn) - 1 - dy], dx, axis=1)
            totals[i] += _signed_sums(center * neighbour)

    out = np.empty(18, dtype=np.float32)
    out[0], out[1] = _ggd(totals[0], h * w)
    for i in range(len(_SHIFTS)):
        alpha, eta, left_std, right_std = _aggd(totals[1 + i], h * w)
        base = 2 + 4 * i
        out[base] = alpha
        out[base + 1] = eta
        out[base + 2] = left_std * left_std
        out[base + 3] = right_std * right_std
    return out
