"""Extract BRISQUE SVM weights and feature scaling from pyiqa.

Run once to (re)produce `pyteenybrisque/_weights/svm.npz`.
Requires the dev environment (`uv sync --group dev`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyiqa

# Feature ranges copied verbatim from pyiqa.archs.brisque_arch.scale_features.
# Source: official BRISQUE on MATLAB
# (https://live.ece.utexas.edu/research/Quality/index_algorithms.htm).
_FEATURE_RANGES = np.array(
    [
        [0.338, 10],
        [0.017204, 0.806612],
        [0.236, 1.642],
        [-0.123884, 0.20293],
        [0.000155, 0.712298],
        [0.001122, 0.470257],
        [0.244, 1.641],
        [-0.123586, 0.179083],
        [0.000152, 0.710456],
        [0.000975, 0.470984],
        [0.249, 1.555],
        [-0.135687, 0.100858],
        [0.000174, 0.684173],
        [0.000913, 0.534174],
        [0.258, 1.561],
        [-0.143408, 0.100486],
        [0.000179, 0.685696],
        [0.000888, 0.536508],
        [0.471, 3.264],
        [0.012809, 0.703171],
        [0.218, 1.046],
        [-0.094876, 0.187459],
        [1.5e-005, 0.442057],
        [0.001272, 0.40803],
        [0.222, 1.042],
        [-0.115772, 0.162604],
        [1.6e-005, 0.444362],
        [0.001374, 0.40243],
        [0.227, 0.996],
        [-0.117188, 0.09832299999999999],
        [3e-005, 0.531903],
        [0.001122, 0.369589],
        [0.228, 0.99],
        [-0.12243, 0.098658],
        [2.8e-005, 0.530092],
        [0.001118, 0.370399],
    ],
    dtype=np.float64,
)


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "pyteenybrisque" / "_weights" / "svm.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    metric = pyiqa.create_metric("brisque")
    sv = metric.net.sv.detach().cpu().numpy().astype(np.float32)
    sv_coef = metric.net.sv_coef.detach().cpu().numpy().astype(np.float32).ravel()
    gamma = float(metric.net.gamma)
    rho = float(metric.net.rho)

    feat_min = _FEATURE_RANGES[:, 0].astype(np.float32)
    feat_range = (_FEATURE_RANGES[:, 1] - _FEATURE_RANGES[:, 0]).astype(np.float32)

    np.savez(
        out,
        sv=sv,
        sv_coef=sv_coef,
        gamma=np.float32(gamma),
        rho=np.float32(rho),
        feat_min=feat_min,
        feat_range=feat_range,
    )
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({size_kb:.1f} KB), sv shape={sv.shape}, sv_coef shape={sv_coef.shape}")


if __name__ == "__main__":
    main()
