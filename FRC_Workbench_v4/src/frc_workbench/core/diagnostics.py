"""Diagnostic metadata helpers for FRC parameter traceability.

These helpers intentionally do not change numerical defaults.  They only make
parameter interpretation explicit in logs, exports, and tests.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


BACKEND_ROI_APOD_SIGMA_PX = 2.0
BACKEND_ROI_APOD_SIGMA_SOURCE = "backend_hardcoded_default"
GUI_ROI_APOD_SIGMA_SOURCE = "gui_control"


def gaussian_sigma_px(gaussian_sigma_nm: float, pixel_size_nm: float) -> float:
    """Return the rendering Gaussian sigma in reconstructed-image pixels."""
    return float(gaussian_sigma_nm) / max(1e-9, float(pixel_size_nm))


def frc_parameter_metadata(
    *,
    pixel_size_nm: float,
    gaussian_sigma_nm: Optional[float] = None,
    roi_apod_sigma_px: Optional[float] = None,
    threshold: Optional[float] = None,
    split_mode: Optional[str] = None,
    roi_source: Optional[str] = None,
    random_seed: Optional[int] = None,
    roi_apod_sigma_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compact metadata dict describing FRC-relevant parameters."""
    meta: Dict[str, Any] = {"pixel_size_nm": float(pixel_size_nm)}
    if gaussian_sigma_nm is not None:
        meta["gaussian_sigma_nm"] = float(gaussian_sigma_nm)
        meta["gaussian_sigma_px"] = gaussian_sigma_px(float(gaussian_sigma_nm), float(pixel_size_nm))
    if roi_apod_sigma_px is not None:
        meta["roi_apod_sigma_px"] = float(roi_apod_sigma_px)
    if threshold is not None:
        meta["threshold"] = float(threshold)
    if split_mode is not None:
        meta["split_mode"] = str(split_mode)
    if roi_source is not None:
        meta["roi_source"] = str(roi_source)
    if random_seed is not None:
        meta["random_seed"] = int(random_seed)
    if roi_apod_sigma_source is not None:
        meta["roi_apod_sigma_source"] = str(roi_apod_sigma_source)
    return meta
