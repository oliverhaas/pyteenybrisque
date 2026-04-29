"""Sweep the Kodak test set through several degradation curves and plot how
four no-reference IQA metrics react.

Sources: 8 Kodak images (lossless 768x512 PNGs, the canonical IQA test set).
Downloaded once into `benchmarks/sources/` and verified by sha256.

Metrics:
- BRISQUE (this package)
- MUSIQ (pyiqa, transformer trained on KonIQ-10k)
- MANIQA (pyiqa, multi-dimensional attention)
- CLIP-IQA (pyiqa, vision-language)

For every (source, degradation, step) combination we compute all four
metrics. Per source-and-metric we min-max normalise across the sweep so
0 = best score the metric gave that source in that sweep, 1 = worst.
This factors out content-specific scale differences. We then aggregate
across the 8 sources by reporting the median and the inter-quartile
range (IQR) per step.

Plots: one PNG per degradation in `benchmarks/`, with the median line
and a shaded IQR band per metric.
"""

import hashlib
import io
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pyiqa
import torch
from PIL import Image, ImageFilter

import pyteenybrisque

_ROOT = Path(__file__).resolve().parent.parent
_SOURCES_DIR = _ROOT / "benchmarks" / "sources"
_OUT = _ROOT / "benchmarks"
_RNG = np.random.default_rng(0)

# (filename, sha256). Kodak True Color images, 768x512 PNG, lossless.
# Mirror: https://r0k.us/graphics/kodak/
_KODAK = [
    ("kodim01.png", "a56e27cbf5f843c048b6af1d6e090760e9c92fadba88b7dee0205918a37523bd"),
    ("kodim02.png", "4f4b74a79237e311d72cad958237b5f7088d8bce1c82305ebefe1a70e3022dfd"),
    ("kodim03.png", "e25ca1ff2f0c0cb5fdfd5f9b0a0bb21ac4c3de3c84a67f35b09a85d3306249db"),
    ("kodim04.png", "e3b946107c5d3441c022f678d0c3caf1e224d81b1604ba840a4f88e562de61aa"),
    ("kodim05.png", "10349e963c5c813d327852f82c1795fa4148d69fedffc4c589bee458e3ac3d53"),
    ("kodim06.png", "363510303b715d4cbc384e1ce227e466b613a09e1b71ae985882bf8e7fbd9b18"),
    ("kodim07.png", "b77d3f006f42414bb242222e0482e750c0fb9e5ee8d4bed2f6f11c5605fe54a4"),
    ("kodim08.png", "ba23983c76b4832ee0e8af0592664756841a16779acd69f792e268fb6d13d6e7"),
]
_KODAK_URL_TEMPLATE = "https://r0k.us/graphics/kodak/kodak/{name}"


def _download_sources() -> list[Path]:
    _SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, expected_sha in _KODAK:
        path = _SOURCES_DIR / name
        if not path.exists():
            url = _KODAK_URL_TEMPLATE.format(name=name)
            print(f"  downloading {url}")
            urlretrieve(url, path)  # noqa: S310
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise SystemExit(
                f"sha256 mismatch for {name}: got {actual_sha}, expected {expected_sha}",
            )
        paths.append(path)
    return paths


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _webp(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blur(img: Image.Image, sigma: float) -> Image.Image:
    return img if sigma == 0 else img.filter(ImageFilter.GaussianBlur(sigma))


def _noise(img: Image.Image, sigma: float) -> Image.Image:
    if sigma == 0:
        return img.copy()
    arr = np.asarray(img).astype(np.float32)
    arr = np.clip(arr + _RNG.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _resample(img: Image.Image, factor: int, mode: Image.Resampling) -> Image.Image:
    if factor == 1:
        return img.copy()
    small = img.resize((img.width // factor, img.height // factor), mode)
    return small.resize(img.size, mode)


def _blocky(img: Image.Image, factor: int) -> Image.Image:
    return _resample(img, factor, Image.Resampling.NEAREST)


def _blurry(img: Image.Image, factor: int) -> Image.Image:
    return _resample(img, factor, Image.Resampling.LANCZOS)


_DEGRADATIONS: dict[str, tuple] = {
    "jpeg_quality": (_jpeg, "JPEG quality", [95, 80, 65, 50, 35, 25, 15, 10, 5]),
    "webp_quality": (_webp, "WebP quality", [95, 80, 65, 50, 35, 25, 15, 10, 5]),
    "gaussian_blur": (_blur, "Gaussian blur sigma (px)", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
    "gaussian_noise": (_noise, "Gaussian noise sigma (8-bit)", [0, 2, 5, 10, 15, 20, 30, 50]),
    "blocky_upscale": (_blocky, "Downscale-then-upscale factor (NEAREST)", [1, 2, 3, 4, 5, 6, 8]),
    "blurry_upscale": (_blurry, "Downscale-then-upscale factor (LANCZOS)", [1, 2, 3, 4, 5, 6, 8]),
}


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0


def _normalise_per_source(scores: np.ndarray, *, lower_is_better: bool) -> np.ndarray:
    # scores shape: (n_sources, n_steps). Normalise each row to [0, 1] with
    # 0 = best score that source got, 1 = worst.
    s = scores if lower_is_better else -scores
    lo = s.min(axis=1, keepdims=True)
    hi = s.max(axis=1, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return (s - lo) / rng


def main() -> None:
    _OUT.mkdir(exist_ok=True)
    print(f"sources -> {_SOURCES_DIR}")
    sources = _download_sources()
    print(f"  {len(sources)} sources ready")

    print("loading metrics...")
    pyiqa_metrics = {name: pyiqa.create_metric(name) for name in ("musiq", "maniqa", "clipiqa")}
    metrics: dict[str, tuple] = {
        "BRISQUE (ours)": (lambda img: pyteenybrisque.score(image=img), True),
        "MUSIQ": (lambda img: float(pyiqa_metrics["musiq"](_to_tensor(img)).item()), False),
        "MANIQA": (lambda img: float(pyiqa_metrics["maniqa"](_to_tensor(img)).item()), False),
        "CLIP-IQA": (lambda img: float(pyiqa_metrics["clipiqa"](_to_tensor(img)).item()), False),
    }

    # Pre-load all sources as RGB
    pil_sources = []
    for p in sources:
        with Image.open(p) as fh:
            pil_sources.append(fh.convert("RGB"))

    for deg_name, (fn, xlabel, values) in _DEGRADATIONS.items():
        print(f"\n[{deg_name}] {len(values)} steps x {len(pil_sources)} sources x {len(metrics)} metrics")
        raw: dict[str, np.ndarray] = {m: np.zeros((len(pil_sources), len(values))) for m in metrics}
        for j, v in enumerate(values):
            for i, src in enumerate(pil_sources):
                variant = fn(src, v)
                for m_name, (m_fn, _) in metrics.items():
                    raw[m_name][i, j] = m_fn(variant)
            print(f"  v={v}: done {len(pil_sources)} sources")

        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(values))
        for m_name, (_, lower_is_better) in metrics.items():
            normalised = _normalise_per_source(raw[m_name], lower_is_better=lower_is_better)
            median = np.median(normalised, axis=0)
            q25 = np.percentile(normalised, 25, axis=0)
            q75 = np.percentile(normalised, 75, axis=0)
            line = ax.plot(x, median, marker="o", linewidth=2, label=m_name)[0]
            ax.fill_between(x, q25, q75, color=line.get_color(), alpha=0.18)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(f"{xlabel}  (pristine -> worst)")
        ax.set_ylabel("normalised perceived badness  (median across sources, IQR shaded)")
        ax.set_title(f"{deg_name} - {len(pil_sources)} Kodak sources")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(visible=True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        out_path = _OUT / f"{deg_name}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
