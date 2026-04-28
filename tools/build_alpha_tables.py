"""Precompute the GGD/AGGD gamma lookup table.

pyiqa's `estimate_ggd_param` / `estimate_aggd_param` perform a discrete-grid
search over alpha in [0.2, 10] step 0.001, picking the alpha whose moment ratio
is closest to the empirical one. We bake the same grid into a vendored npz so
the runtime can do an identical argmin without recomputing lgammas.

Run once to (re)produce `pyteenybrisque/_weights/alpha.npz`.
"""

from pathlib import Path

import numpy as np
from scipy.special import gammaln


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "pyteenybrisque" / "_weights" / "alpha.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    gamma = np.arange(0.2, 10 + 0.001, 0.001, dtype=np.float64)

    # GGD: rho = Gamma(1/g) Gamma(3/g) / Gamma(2/g)^2
    r_ggd = np.exp(gammaln(1.0 / gamma) + gammaln(3.0 / gamma) - 2.0 * gammaln(2.0 / gamma))
    # AGGD: r_gam = Gamma(2/g)^2 / (Gamma(1/g) Gamma(3/g))   (= 1 / r_ggd)
    r_aggd = np.exp(2.0 * gammaln(2.0 / gamma) - (gammaln(1.0 / gamma) + gammaln(3.0 / gamma)))

    np.savez(
        out,
        gamma=gamma.astype(np.float32),
        r_ggd=r_ggd.astype(np.float32),
        r_aggd=r_aggd.astype(np.float32),
    )
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({size_kb:.1f} KB), grid size={gamma.size}")


if __name__ == "__main__":
    main()
