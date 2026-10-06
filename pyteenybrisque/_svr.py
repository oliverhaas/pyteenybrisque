"""RBF SVR prediction from the 36 BRISQUE features, with the bundled LIVE weights or user-trained ones."""

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

_N_FEATURES = 36
_INFO_PREFIX = "info_"
_KEYS = frozenset({"sv", "sv_coef", "gamma", "rho", "feat_min", "feat_range"})


@dataclass(frozen=True, eq=False)
class Weights:
    """RBF SVR parameters: features scaled to [-1, 1] by `feat_min` and `feat_range` score `kernel @ sv_coef - rho`."""

    sv: npt.NDArray[np.float32]
    sv_coef: npt.NDArray[np.float32]
    gamma: float
    rho: float
    feat_min: npt.NDArray[np.float32]
    feat_range: npt.NDArray[np.float32]
    info: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_sv = len(self.sv)
        shapes = {
            "sv": (n_sv, _N_FEATURES),
            "sv_coef": (n_sv,),
            "feat_min": (_N_FEATURES,),
            "feat_range": (_N_FEATURES,),
        }
        for name, shape in shapes.items():
            arr = np.array(getattr(self, name), dtype=np.float32)
            if arr.shape != shape:
                raise ValueError(f"weights {name!r} must have shape {shape}, got {arr.shape}")
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "gamma", float(self.gamma))
        object.__setattr__(self, "rho", float(self.rho))
        object.__setattr__(self, "info", MappingProxyType({k: float(v) for k, v in self.info.items()}))

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Self:
        """Read weights written by `save`."""
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(_KEYS - set(data.files))
            if missing:
                raise ValueError(f"weights file {path} lacks {', '.join(missing)}")
            info = {k.removeprefix(_INFO_PREFIX): float(data[k]) for k in data.files if k.startswith(_INFO_PREFIX)}
            return cls(
                sv=data["sv"],
                sv_coef=data["sv_coef"],
                gamma=float(data["gamma"]),
                rho=float(data["rho"]),
                feat_min=data["feat_min"],
                feat_range=data["feat_range"],
                info=info,
            )

    def save(self, path: str | PathLike[str]) -> None:
        """Write the weights as an `.npz` file to exactly `path`."""
        info = {f"{_INFO_PREFIX}{k}": np.float64(v) for k, v in self.info.items()}
        with Path(path).open("wb") as fh:
            np.savez(
                fh,
                sv=self.sv,
                sv_coef=self.sv_coef,
                gamma=np.float64(self.gamma),
                rho=np.float64(self.rho),
                feat_min=self.feat_min,
                feat_range=self.feat_range,
                allow_pickle=False,
                **info,
            )


with resources.as_file(resources.files(__package__).joinpath("_weights/svm.npz")) as _path:
    BUNDLED_WEIGHTS = Weights.load(_path)


def predict(features: npt.NDArray[np.float32], weights: Weights) -> float:
    scaled = (-1.0 + 2.0 * (features.astype(np.float64) - weights.feat_min) / weights.feat_range).astype(np.float64)
    diff = scaled - weights.sv.astype(np.float64)
    dist = np.einsum("ij,ij->i", diff, diff)
    kernel = np.exp(-weights.gamma * dist)
    return float(kernel @ weights.sv_coef.astype(np.float64) - weights.rho)
