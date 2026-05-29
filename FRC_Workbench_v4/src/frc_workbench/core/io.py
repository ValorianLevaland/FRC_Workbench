from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Optional TIFF I/O backends
_tifffile = None
with contextlib.suppress(Exception):
    import tifffile as _tifffile  # type: ignore

_PIL_Image = None
with contextlib.suppress(Exception):
    from PIL import Image as _PIL_Image  # type: ignore


def ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def read_tif(path: Path) -> np.ndarray:
    """Read a 2D TIFF into float32."""
    path = Path(path)
    if _tifffile is not None:
        arr = _tifffile.imread(str(path))
        arr = np.asarray(arr)
        if arr.ndim > 2:
            arr = arr[0]
        return arr.astype(np.float32, copy=False)
    if _PIL_Image is not None:
        im = _PIL_Image.open(str(path))
        arr = np.array(im)
        if arr.ndim > 2:
            arr = arr[..., 0]
        return arr.astype(np.float32, copy=False)
    raise RuntimeError("No TIFF reader available. Install 'tifffile' or 'Pillow'.")


def save_tif_float32(path: Path, arr: np.ndarray) -> None:
    """Save a 2D float32 TIFF."""
    path = Path(path)
    arr = np.asarray(arr, dtype=np.float32, order="C")
    ensure_dir(path.parent)

    if _tifffile is not None:
        _tifffile.imwrite(str(path), arr, dtype=np.float32)
        return
    if _PIL_Image is not None:
        m = float(np.nanmax(arr))
        if not np.isfinite(m) or m <= 0:
            m = 1.0
        scaled = np.clip(arr / m, 0.0, 1.0) * 65535.0
        _PIL_Image.fromarray(scaled.astype(np.uint16)).save(str(path), format="TIFF")
        return
    raise RuntimeError("No TIFF writer available. Install 'tifffile' or 'Pillow'.")


def sha256_small_file(path: Path, max_bytes: int = 512 * 1024) -> str:
    """SHA256 over the first `max_bytes` of the file (fast traceability hint)."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        h.update(f.read(int(max_bytes)))
    return h.hexdigest()
