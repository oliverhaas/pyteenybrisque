"""Build BRISQUE test fixtures: high-res sources + quality-degraded variants.

For each source seed: downloads from Picsum, verifies the downloaded bytes
match the manifest below, and produces variants:
- a heavy JPEG-compressed variant (q=10)
- a heavy WebP-compressed variant (q=10)
- a 4x downscale + nearest upscale (blocky)
- a 4x downscale + Lanczos upscale (soft)

The hash check exists because Picsum's CDN may serve different bytes for the
same seed over time (re-encoding, mirror drift). If the check fails, either
update `_SOURCE_SHA256` to accept the new bytes (and re-baseline the pyiqa
scores in tests) or pin a different seed.

Output goes to `tests/data/`. Tests are pinned to the committed bytes; this
script is for one-time regeneration.
"""

import hashlib
from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image

_DATA = Path(__file__).resolve().parent.parent / "tests" / "data"
_W, _H = 2000, 1333
_DOWN = 4

# (picsum seed, local name, sha256 of the downloaded JPEG bytes).
_SOURCES = [
    (134, "src134", "e71404e1c956b5abf9df40b213f927050bc6f3145d6bce24e5ce7312cf3fb3f3"),
    (177, "src177", "d0999d9e0e4da370d5745dc1e1cf00515a99224c71bbf429e3745cdd90a67689"),
    (231, "src231", "6a68c623988d0d5953dcc0e9909f8a01deaad81921a0094b7a6c9f02b6602d0c"),
]


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
    for seed, name, expected_sha in _SOURCES:
        src = _DATA / f"{name}.jpg"
        url = f"https://picsum.photos/seed/{seed}/{_W}/{_H}.jpg"
        print(f"downloading {url} -> {src.name}")
        urlretrieve(url, src)  # noqa: S310

        actual_sha = hashlib.sha256(src.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise SystemExit(
                f"sha256 mismatch for seed {seed}: "
                f"got {actual_sha}, expected {expected_sha}. "
                f"Picsum may have re-encoded the image. Update _SOURCE_SHA256 "
                f"and re-baseline the pyiqa scores if the new bytes are acceptable.",
            )

        with Image.open(src) as pil:
            img = pil.convert("RGB")
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
