# Changelog

## 0.1.0a1 (Unreleased)

First public release. Implements the original (non-MATLAB) BRISQUE pipeline
on top of `numpy` and `Pillow`, with the SVR weights and feature scaling
extracted from [`pyiqa`](https://github.com/chaofengc/IQA-PyTorch) and
vendored in the package (~250 KB total).

- `pyteenybrisque.score(image=...)` — single public entry point. Accepts a
  path (str or `os.PathLike`), a `PIL.Image.Image`, or a numpy array
  (`HxW` grayscale or `HxWx{3,4}` RGB / RGBA, uint8 or float in `[0, 1]`).
- Two runtime dependencies: `numpy` and `Pillow`.
- Agreement with `pyiqa`'s BRISQUE: within ~0.1 BRISQUE points on natural
  images.
- Performance: ~130 ms per 1 MP image, single-threaded CPU.
