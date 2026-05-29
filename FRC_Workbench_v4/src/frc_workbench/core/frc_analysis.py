from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .frc_backend import compute_frc_curve, find_cutoff_frequency_robust, gaussian_blur


def _roi_bbox(mask: np.ndarray):
    """Return (y0, y1, x0, x1) bbox for a boolean mask, or None if empty."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.size == 0:
        return None
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)


def _apodize_mask(mask: np.ndarray, sigma_px: float = 2.0) -> np.ndarray:
    """Softens a binary ROI mask (0/1) to reduce edge artifacts."""
    mask = np.asarray(mask, dtype=np.float32)
    if mask.max() <= 0:
        return (mask > 0).astype(np.float32)
    if sigma_px <= 0:
        return (mask > 0).astype(np.float32)
    w = gaussian_blur(mask, float(sigma_px))
    m = float(np.nanmax(w))
    if not np.isfinite(m) or m <= 0:
        return (mask > 0).astype(np.float32)
    w = w / (m + 1e-12)
    return np.clip(w, 0.0, 1.0).astype(np.float32, copy=False)


@dataclass
class FRCPlotData:
    freqs_cyc_per_nm: np.ndarray
    frc: np.ndarray
    threshold: float
    cutoff_cyc_per_pix: Optional[float]
    resolution_nm: Optional[float]
    notes: str = ""


def compute_frc_plot_data(
    img_even: np.ndarray,
    img_odd: np.ndarray,
    *,
    pixel_size_nm: float,
    threshold: float = 1.0 / 7.0,
    smooth_bins: int = 5,
    nyquist_guard: float = 0.49,
    roi_mask: Optional[np.ndarray] = None,
    apod_sigma_px: float = 2.0,
) -> FRCPlotData:
    """Compute an ROI-aware FRC curve and resolution estimate for plotting.

    Parameters
    ----------
    img_even, img_odd:
        Two statistically independent reconstructions (float images).
    pixel_size_nm:
        Reconstruction pixel size in nanometers.
    threshold:
        FRC threshold crossing (default 1/7).
    smooth_bins:
        Moving-average smoothing length (NaN-aware) before finding crossing.
    nyquist_guard:
        If crossing is too close to Nyquist (0.5 cyc/px), return no cutoff.
    roi_mask:
        Optional boolean mask selecting pixels to include.
    apod_sigma_px:
        Gaussian sigma (px) used to soften ROI edges (apodization).

    Returns
    -------
    FRCPlotData
    """
    img_even = np.asarray(img_even, dtype=np.float32)
    img_odd = np.asarray(img_odd, dtype=np.float32)

    if img_even.shape != img_odd.shape:
        raise ValueError("Odd/even images must have the same shape.")
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be > 0")

    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask, dtype=bool)
        if roi_mask.shape != img_even.shape:
            raise ValueError("ROI mask shape mismatch.")
        bbox = _roi_bbox(roi_mask)
        if bbox is None:
            raise ValueError("ROI mask is empty.")
        y0, y1, x0, x1 = bbox
        e = img_even[y0:y1, x0:x1].copy()
        o = img_odd[y0:y1, x0:x1].copy()
        m = roi_mask[y0:y1, x0:x1]
        w = _apodize_mask(m, sigma_px=float(apod_sigma_px))
        if np.nanmax(w) <= 0:
            raise ValueError("ROI apodization produced an empty window.")
        e = (e - float(np.mean(e[m]))) * w
        o = (o - float(np.mean(o[m]))) * w
        notes = f"ROI bbox: y[{y0}:{y1}] x[{x0}:{x1}], apod_sigma_px={apod_sigma_px:g}"
    else:
        e = img_even - float(np.mean(img_even))
        o = img_odd - float(np.mean(img_odd))
        notes = "No ROI (full image)"

    freqs_cyc_per_pix, frc = compute_frc_curve(e, o)
    cutoff = find_cutoff_frequency_robust(
        freqs_cyc_per_pix,
        frc,
        threshold=float(threshold),
        smooth_bins=int(max(1, smooth_bins)),
        nyquist_guard=float(nyquist_guard),
    )
    resolution_nm = float(pixel_size_nm / cutoff) if (cutoff is not None and cutoff > 0) else None
    freqs_cyc_per_nm = freqs_cyc_per_pix / float(pixel_size_nm)

    return FRCPlotData(
        freqs_cyc_per_nm=np.asarray(freqs_cyc_per_nm, dtype=float),
        frc=np.asarray(frc, dtype=float),
        threshold=float(threshold),
        cutoff_cyc_per_pix=float(cutoff) if cutoff is not None else None,
        resolution_nm=resolution_nm,
        notes=notes,
    )
