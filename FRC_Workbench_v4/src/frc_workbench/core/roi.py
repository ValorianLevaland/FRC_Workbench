from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def polygon_bbox_rc(polygon_rc: np.ndarray, shape_rc: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
    """Bounding box of a polygon in (row, col), optionally clipped to image shape."""
    poly = np.asarray(polygon_rc, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("ROI polygon must be (N,2) with N>=3")
    rmin = int(np.floor(np.nanmin(poly[:, 0])))
    rmax = int(np.ceil(np.nanmax(poly[:, 0])))
    cmin = int(np.floor(np.nanmin(poly[:, 1])))
    cmax = int(np.ceil(np.nanmax(poly[:, 1])))
    # include last pixel
    y0, y1 = rmin, rmax + 1
    x0, x1 = cmin, cmax + 1
    if shape_rc is not None:
        H, W = map(int, shape_rc)
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))
        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
    return y0, y1, x0, x1


def mask_from_polygon(shape_rc: Tuple[int, int], polygon_rc: np.ndarray) -> np.ndarray:
    """Convert a polygon (row, col) to a boolean mask (rows, cols).

    Implementation notes
    --------------------
    - Uses a bbox-restricted evaluation to avoid allocating full (H*W) coordinate grids
      for very large images.
    - Prefers Matplotlib Path if available; otherwise falls back to a ray-casting method.
    """
    poly = np.asarray(polygon_rc, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("ROI polygon must be (N,2) with N>=3")
    H, W = map(int, shape_rc)
    out = np.zeros((H, W), dtype=bool)

    y0, y1, x0, x1 = polygon_bbox_rc(poly, shape_rc=(H, W))
    if y1 <= y0 or x1 <= x0:
        return out

    # Add a small safety margin inside the image bounds
    pad = 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(H, y1 + pad)
    x1 = min(W, x1 + pad)

    hh = y1 - y0
    ww = x1 - x0

    try:
        from matplotlib.path import Path as MplPath  # type: ignore

        yy, xx = np.mgrid[y0:y1, x0:x1]
        pts_xy = np.vstack([xx.ravel(), yy.ravel()]).T  # (x,y) = (col,row)
        poly_xy = poly[:, [1, 0]]
        inside = MplPath(poly_xy).contains_points(pts_xy).reshape((hh, ww))
        out[y0:y1, x0:x1] = inside
        return out
    except Exception:
        pass

    # Ray-casting fallback (vectorized; restricted to bbox)
    y = np.arange(y0, y1)[:, None]
    x = np.arange(x0, x1)[None, :]
    poly_xy = poly[:, [1, 0]]
    xpoly, ypoly = poly_xy[:, 0], poly_xy[:, 1]
    n = len(xpoly)
    mask_local = np.zeros((hh, ww), dtype=bool)
    for i in range(n):
        j = (i - 1) % n
        xi, yi, xj, yj = xpoly[i], ypoly[i], xpoly[j], ypoly[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        mask_local ^= cond
    out[y0:y1, x0:x1] = mask_local
    return out
