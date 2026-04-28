# Plan: minimal numpy-only BRISQUE

## Scope

One public function:

```python
from pyteenybrisque import score
score(image) -> float
```

Match `pyiqa`'s BRISQUE within ~1e-3 on the 5 test images we already have.

## Package layout

```
pyteenybrisque/
  __init__.py             # score() + load_image helper
  _gaussian.py            # separable 1D Gaussian (numpy)
  _features.py            # MSCN, GGD/AGGD moment fits, 36-feature vector
  _alpha.py               # Chebyshev poly: ρ → α
  _svr.py                 # RBF SVR eval
  _weights/
    svm.npz               # vendored: SVs, dual coefs, b, gamma, feature-scaling min/range
    alpha.npy             # vendored: ~12 Chebyshev coefficients
tests/
  test_against_pyiqa.py
tools/
  build_alpha_chebyshev.py   # offline: math.gamma to fit α(ρ)
  extract_pyiqa_weights.py   # offline: pyiqa state_dict → svm.npz
```

## Steps (in order)

1. **`_gaussian.py`** -- separable Gaussian (σ = 7/6, 7×7) via `np.convolve` per axis. Validate against `scipy.ndimage` on a fixture, then drop scipy.
2. **`_alpha.py`** -- build script generates Chebyshev coeffs for α(ρ) on ρ ∈ [1, 16]; runtime is one `np.polynomial.chebyshev.chebval`.
3. **`_features.py`** -- MSCN map → GGD fit (2 features) → 4 pairwise products → AGGD fit each (4 × 4 features) → 18 features per scale, two scales → 36 total.
4. **`_svr.py`** -- vectorized `(α · exp(−γ‖x − SV‖²)).sum() + b`. Single matmul-style op.
5. **`extract_pyiqa_weights.py`** -- pull SVs, duals, b, γ, feature scaling out of pyiqa's BRISQUE module → `svm.npz` (~112 KB).
6. **End-to-end `score()`** -- PIL load → grayscale → two scales (full + 2× downsampled) → features → scaling → SVR.

## Validation

- Run all 5 test images, diff against the pyiqa scores already captured by the benchmark.
- Acceptance: max abs diff < 1e-3.

## Perf budget

- Target < 50 ms per ~1 MP image, CPU.
- If `np.convolve` per row/col is the bottleneck → Cython kernel for the separable Gaussian only. Everything else is already vector ops over ≤ 1 MP arrays.

## Decision points

- After step 1: if pure-numpy Gaussian is < 20 ms → keep numpy, ship.
- After step 6: if total > 50 ms → Cython only the Gaussian. Don't pull scipy.

## Out of scope

Color BRISQUE, batch API, GPU, async. Single image, single score, sync.
