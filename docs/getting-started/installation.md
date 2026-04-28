# Installation

```console
pip install pyteenybrisque
```

Requires Python 3.14 or later. Pulls in `numpy` and `Pillow`.

## Quickstart

```python
import pyteenybrisque

# from a path
score = pyteenybrisque.score("photo.jpg")

# from a PIL image
from PIL import Image
score = pyteenybrisque.score(Image.open("photo.jpg"))

# from a numpy array (uint8 RGB)
import numpy as np
arr = np.asarray(Image.open("photo.jpg").convert("RGB"))
score = pyteenybrisque.score(arr)
```

Lower scores mean higher perceived quality. Typical range is `[0, 100]`.
