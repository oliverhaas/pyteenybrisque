"""Train BRISQUE regression weights on labelled images; needs `pip install pyteenybrisque[train]`."""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

try:
    from scipy.stats import spearmanr
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
    from sklearn.svm import SVR
except ImportError as e:
    raise ImportError("pyteenybrisque.train needs scikit-learn: pip install pyteenybrisque[train]") from e

from . import Weights
from . import features as features_of

if TYPE_CHECKING:
    from collections.abc import Iterable

_N_FEATURES = 36
_MAX_LISTED_FAILURES = 10
# Epsilon, C and gamma act on labels standardised to mean 0 and standard deviation 1.
_EPSILON = 0.1
_C_GRID = tuple(2.0**p for p in range(-1, 8, 2))
_GAMMA_GRID = tuple(2.0**p for p in range(-7, 0, 2))


def extract_features(*, images: Iterable[object], workers: int | None = None) -> npt.NDArray[np.float32]:
    """BRISQUE features of each image as one row of an (n, 36) array; an image that fails gets a NaN row."""
    images = list(images)
    workers = workers or os.process_cpu_count() or 1
    if workers == 1:
        results = [_features_or_error(image) for image in images]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_features_or_error, images))

    rows = np.full((len(images), _N_FEATURES), np.nan, dtype=np.float32)
    failures = []
    for i, result in enumerate(results):
        if isinstance(result, str):
            failures.append(f"{i}: {result}")
        else:
            rows[i] = result
    if failures:
        listed = "; ".join(failures[:_MAX_LISTED_FAILURES])
        more = "; ..." if len(failures) > _MAX_LISTED_FAILURES else ""
        warnings.warn(
            f"{len(failures)} of {len(images)} images failed and got NaN features: {listed}{more}",
            RuntimeWarning,
            stacklevel=2,
        )
    return rows


def fit(  # noqa: PLR0913 -- keyword-only settings with defaults
    *,
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    c: float | None = None,
    gamma: float | None = None,
    tune_samples: int = 5000,
    folds: int = 5,
    seed: int = 0,
    workers: int | None = None,
) -> Weights:
    """RBF SVR weights that predict `labels` from rows of `extract_features`.

    `c` and `gamma` left as None are chosen by a cross-validated grid search on at most `tune_samples` rows.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.shape[1:] != (_N_FEATURES,):
        raise ValueError(f"features must have shape (n, {_N_FEATURES}), got {x.shape}")
    if y.shape != (len(x),):
        raise ValueError(f"labels must have shape ({len(x)},) to match the features, got {y.shape}")
    bad_rows = int((~np.isfinite(x).all(axis=1)).sum())
    if bad_rows:
        raise ValueError(
            f"{bad_rows} feature rows are not finite, such as NaN rows of images that failed in extract_features; "
            "drop them first with keep = np.isfinite(features).all(axis=1)",
        )
    if not np.isfinite(y).all():
        raise ValueError("labels must be finite")
    if min(len(y), tune_samples) < 2 * folds:
        raise ValueError(
            f"{folds}-fold cross-validation needs at least {2 * folds} rows, got {min(len(y), tune_samples)}",
        )
    if y.std() == 0:
        raise ValueError("all labels are equal, so there is nothing to learn")

    feat_min = x.min(axis=0).astype(np.float32)
    feat_range = (x.max(axis=0) - x.min(axis=0)).astype(np.float32)
    feat_range[feat_range == 0] = 1.0
    # Same arithmetic as `_svr.predict`, so training and scoring scale features identically.
    scaled = -1.0 + 2.0 * (x - feat_min) / feat_range
    mean, sd = y.mean(), y.std()
    target = (y - mean) / sd

    subset = np.random.default_rng(seed).permutation(len(y))[:tune_samples]
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    n_jobs = workers or os.process_cpu_count() or 1
    if c is None or gamma is None:
        grid = {"C": list(_C_GRID) if c is None else [c], "gamma": list(_GAMMA_GRID) if gamma is None else [gamma]}
        search = GridSearchCV(
            SVR(epsilon=_EPSILON),
            grid,
            scoring="neg_root_mean_squared_error",
            cv=kfold,
            n_jobs=n_jobs,
            refit=False,
        )
        search.fit(scaled[subset], target[subset])
        c, gamma = float(search.best_params_["C"]), float(search.best_params_["gamma"])

    predicted = cross_val_predict(
        SVR(C=c, gamma=gamma, epsilon=_EPSILON),
        scaled[subset],
        target[subset],
        cv=kfold,
        n_jobs=n_jobs,
    )
    model = SVR(C=c, gamma=gamma, epsilon=_EPSILON).fit(scaled, target)
    # sklearn predicts kernel @ dual_coef_ + intercept_ on the standardised labels; undo the standardisation.
    return Weights(
        sv=model.support_vectors_,
        sv_coef=sd * model.dual_coef_[0],
        gamma=gamma,
        rho=-(mean + sd * model.intercept_[0]),
        feat_min=feat_min,
        feat_range=feat_range,
        info={
            "c": c,
            "gamma": gamma,
            "epsilon": _EPSILON,
            "n_train": len(y),
            "n_support": len(model.support_),
            "cv_rmse": sd * float(np.sqrt(np.mean((predicted - target[subset]) ** 2))),
            "cv_srocc": float(spearmanr(predicted, target[subset]).statistic),
        },
    )


def _features_or_error(image: object) -> npt.NDArray[np.float32] | str:
    try:
        return features_of(image=image)
    except Exception as e:  # noqa: BLE001 -- one bad image must not discard the rest of the batch
        return f"{type(e).__name__}: {e}"
