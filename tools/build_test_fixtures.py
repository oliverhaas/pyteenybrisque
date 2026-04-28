"""Build BRISQUE test fixtures: high-res sources + quality-degraded variants.

Run once. Downloads stable Picsum images and produces, for each source:
- a heavy JPEG-compressed variant
- a heavy WebP-compressed variant
- a 4x downscale + nearest upscale (blocky)
- a 4x downscale + Lanczos upscale (soft)

Output goes to `tests/data/`. Idempotent on overwrite.
"""

from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image

_DATA = Path(__file__).resolve().parent.parent / "tests" / "data"

# (picsum seed, local name). 2000x1333 is ~2.7 MP.
_SOURCES = [
    (134, "src134"),
    (177, "src177"),
    (231, "src231"),
]
_W, _H = 2000, 1333
_DOWN = 4


def _save_jpeg(img: Image.Image, path: Path, quality: int) -> None:
    img.save(path, "JPEG", quality=quality, optimize=True)


def _save_webp(img: Image.Image, path: Path, quality: int) -> None:
    img.save(path, "WEBP", quality=quality, method=6)


def _save_png(img: Image.Image, path: Path) -> None:
    img.save(path, "PNG", optimize=True)


def _resample(img: Image.Image, mode: Image.Resampling) -> Image.Image:
    small = img.resize((img.width // _DOWN, img.height // _DOWN), mode)
    return small.resize(img.size, mode)


def main() -> None:
    _DATA.mkdir(parents=True, exist_ok=True)
    for seed, name in _SOURCES:
        src = _DATA / f"{name}.jpg"
        url = f"https://picsum.photos/seed/{seed}/{_W}/{_H}.jpg"
        print(f"downloading {url} -> {src.name}")
        urlretrieve(url, src)  # noqa: S310

        img = Image.open(src).convert("RGB")
        _save_jpeg(img, _DATA / f"{name}_jpeg10.jpg", quality=10)
        _save_webp(img, _DATA / f"{name}_webp10.webp", quality=10)
        # Blocky resample compresses well as PNG (4x4 blocks of identical pixels).
        _save_png(_resample(img, Image.Resampling.NEAREST), _DATA / f"{name}_blocky.png")
        # Blurry resample doesn't compress with PNG (every pixel unique). The
        # upscale artefact already dominates so JPEG q=95 is visually transparent.
        _save_jpeg(_resample(img, Image.Resampling.LANCZOS), _DATA / f"{name}_blurry.jpg", quality=95)
        print(f"  variants written for {name}")


if __name__ == "__main__":
    main()
