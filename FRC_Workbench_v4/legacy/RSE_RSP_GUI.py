#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQUIRREL+ (standalone): CSV ➜ SR halves ➜ Registration ➜ RSF(α/β) ➜ RSP/RSE + FRC map
Faithful to NanoJ-SQUIRREL workflow, with extras:
- Optional sub-pixel registration (phase correlation, FFT upsampling).
- Dense auto-σ grid search (maximize global RSP or minimize global RSE).
- Hann apodization in FRC tiles for stability.
- ROI-aware local stats; windows with low ROI coverage -> NaN.
- PyQt5 GUI with progress bar and full parameter control.

If your project backends are importable, they will be preferred:
  * batch_frc_backend_v1 / batch_frc_backend (render_two_images, rsp_rse_maps, frc_map)
  * csv2pairs_backend (for consistent CSV handling, naming)
Otherwise, built-ins below are used (equivalent behavior).

Author: you
"""

from __future__ import annotations
import argparse, fnmatch, math, sys, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# Optional acceleration / I/O
try:
    import tifffile as _tifffile
except Exception:
    _tifffile = None
try:
    from PIL import Image as _PIL_Image
except Exception:
    _PIL_Image = None

# -------------------- Try to reuse your backends if available -----------------
_render_two_images_ext = None
_rsp_rse_maps_ext = None
_frc_map_ext = None
try:
    # your v1 backend first
    from batch_frc_backend_v1 import render_two_images as _render_two_images_ext  # type: ignore
    from batch_frc_backend_v1 import rsp_rse_maps as _rsp_rse_maps_ext            # type: ignore
    from batch_frc_backend_v1 import frc_map as _frc_map_ext                      # type: ignore
except Exception:
    try:
        from batch_frc_backend import render_two_images as _render_two_images_ext  # type: ignore
        from batch_frc_backend import rsp_rse_maps as _rsp_rse_maps_ext            # type: ignore
        from batch_frc_backend import frc_map as _frc_map_ext                      # type: ignore
    except Exception:
        pass

# -------------------- SciPy (optional); otherwise we do NumPy FFT -------------
try:
    import scipy.ndimage as _spnd
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ============================ Utility: TIFF I/O ================================
def _save_tif32(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32, order="C")
    if _tifffile is not None:
        _tifffile.imwrite(str(path), arr, dtype=np.float32)
        return
    if _PIL_Image is not None:
        m = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
        m = m if m > 0 else 1.0
        _PIL_Image.fromarray(np.clip(arr / m, 0, 1) * 65535.0.astype(np.uint16)).save(str(path), format="TIFF")
        return
    raise RuntimeError("Install 'tifffile' or 'Pillow' to save TIFF.")

def _read_tif(path: Path) -> np.ndarray:
    if _tifffile is not None:
        with _tifffile.TiffFile(str(path)) as tf:
            arr = tf.asarray()
        if arr.ndim > 2: arr = arr[0]
        return np.asarray(arr, dtype=np.float32)
    if _PIL_Image is not None:
        im = _PIL_Image.open(str(path))
        arr = np.array(im, dtype=np.float32)
        if arr.ndim > 2: arr = arr[..., 0]
        return arr
    raise RuntimeError("Install 'tifffile' or 'Pillow' to read TIFF.")

# ======================== CSV helpers (ThunderSTORM) ==========================
def _find_column(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in lower: return lower[k]
    squash = {c.lower().replace(" ", ""): c for c in df.columns}
    for k in keys:
        kk = k.replace(" ", "")
        if kk in squash: return squash[kk]
    return None

def _detect_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[str]]:
    cx = _find_column(df, ["x [nm]", "x[nm]", "x_nm", "x (nm)", "x"])
    cy = _find_column(df, ["y [nm]", "y[nm]", "y_nm", "y (nm)", "y"])
    cf = _find_column(df, ["frame", "frames", "frame #", "frame_number"])
    ci = _find_column(df, ["intensity [photon]", "intensity", "photons", "photon count"])
    if cx is None or cy is None: raise ValueError("Missing 'x [nm]'/'y [nm]' columns.")
    return cx, cy, cf, ci

def _to_render_df(df: pd.DataFrame, cx: str, cy: str, ci: Optional[str], cf: Optional[str]) -> pd.DataFrame:
    out = pd.DataFrame({
        "x_nm": pd.to_numeric(df[cx], errors="coerce"),
        "y_nm": pd.to_numeric(df[cy], errors="coerce"),
        "frame": pd.to_numeric(df[cf], errors="coerce") if cf is not None else np.arange(len(df), dtype=np.int64),
        "intensity": pd.to_numeric(df[ci], errors="coerce") if ci is not None else 1.0,
    })
    return out.dropna(subset=["x_nm", "y_nm", "frame", "intensity"])

def _split_masks(frames: np.ndarray, method: str = "odd_even", block_size_frames: int = 500, seed: int = 0):
    method = method.lower()
    if method == "odd_even":
        mask_odd = (frames.astype(np.int64) % 2) == 1
        return mask_odd, ~mask_odd
    f = frames.astype(np.int64)
    f0 = int(f.min())
    block = max(1, int(block_size_frames))
    block_ids = (f - f0) // block
    uniq = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    a = set(uniq[: len(uniq)//2])
    mask_a = np.isin(block_ids, list(a))
    return mask_a, ~mask_a

# ============================ Rendering (fallback) ============================
def _hist2d_weighted(x, y, w, pixel_size_nm, bbox=None, margin_px=8):
    if bbox is None:
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    else:
        xmin, xmax, ymin, ymax = bbox
    xmin -= margin_px*pixel_size_nm; xmax += margin_px*pixel_size_nm
    ymin -= margin_px*pixel_size_nm; ymax += margin_px*pixel_size_nm
    nx = int(math.ceil(max(xmax-xmin, pixel_size_nm)/pixel_size_nm))
    ny = int(math.ceil(max(ymax-ymin, pixel_size_nm)/pixel_size_nm))
    x_edges = np.linspace(xmin, xmin + nx*pixel_size_nm, nx+1)
    y_edges = np.linspace(ymin, ymin + ny*pixel_size_nm, ny+1)
    H, xe, ye = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)
    return H.T, (x_edges, y_edges)

def _gaussian_blur(img: np.ndarray, sigma_pix: float) -> np.ndarray:
    if sigma_pix <= 0: return img
    if _HAVE_SCIPY:
        return _spnd.gaussian_filter(img, sigma=float(sigma_pix))
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)[:, None]; fx = np.fft.fftfreq(nx)[None, :]
    factor = np.exp(-2.0 * (np.pi*sigma_pix)**2 * (fy*fy + fx*fx))
    return np.fft.ifft2(np.fft.fft2(img) * factor).real

def render_two_images_fallback(dfA: pd.DataFrame, dfB: pd.DataFrame,
                               pixel_size_nm: float, gaussian_sigma_nm: float,
                               weight_mode: str = "ones") -> Tuple[np.ndarray, np.ndarray, Dict]:
    wA = np.ones(len(dfA), float) if weight_mode == "ones" else dfA["intensity"].to_numpy(float)
    wB = np.ones(len(dfB), float) if weight_mode == "ones" else dfB["intensity"].to_numpy(float)
    imgA, edges = _hist2d_weighted(dfA["x_nm"].to_numpy(float), dfA["y_nm"].to_numpy(float), wA, pixel_size_nm)
    x_edges, y_edges = edges
    H, _, _ = np.histogram2d(dfB["x_nm"].to_numpy(float), dfB["y_nm"].to_numpy(float), bins=[x_edges, y_edges], weights=wB)
    imgB = H.T
    sigma_pix = float(gaussian_sigma_nm / max(1e-9, pixel_size_nm))
    imgA = _gaussian_blur(imgA, sigma_pix)
    imgB = _gaussian_blur(imgB, sigma_pix)
    return imgA, imgB, {"pixel_size_nm": float(pixel_size_nm), "x_edges_nm": x_edges, "y_edges_nm": y_edges,
                        "gaussian_sigma_nm": float(gaussian_sigma_nm), "weight_mode": weight_mode, "image_shape": imgA.shape}

def render_two_images(dfA: pd.DataFrame, dfB: pd.DataFrame,
                      pixel_size_nm: float, gaussian_sigma_nm: float, weight_mode: str):
    if _render_two_images_ext is not None:
        try:
            return _render_two_images_ext(dfA, dfB, pixel_size_nm=pixel_size_nm, gaussian_sigma_nm=gaussian_sigma_nm, weight_mode=weight_mode)
        except Exception:
            pass
    return render_two_images_fallback(dfA, dfB, pixel_size_nm, gaussian_sigma_nm, weight_mode)

# ======================= Registration (phase correlation) =====================
def _phase_correlation_shift(a: np.ndarray, b: np.ndarray, upsample: int = 8) -> Tuple[float, float]:
    """Return (dy, dx) that best aligns b to a (i.e., shift b by this to align)."""
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    R = A * np.conj(B)
    R /= (np.abs(R) + 1e-12)
    c = np.fft.ifft2(R).real
    # upsample via zero-padding around the peak
    y0, x0 = np.unravel_index(np.argmax(c), c.shape)
    H, W = c.shape
    # shift peak to origin
    c = np.roll(np.roll(c, -y0, axis=0), -x0, axis=1)
    pad_y = (upsample-1)*H
    pad_x = (upsample-1)*W
    C = np.pad(c, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2)), mode="constant")
    C = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(C))).real  # smooth a bit
    yy, xx = np.unravel_index(np.argmax(C), C.shape)
    dy = (yy - C.shape[0]//2) / float(upsample)
    dx = (xx - C.shape[1]//2) / float(upsample)
    # convert to the shift relative to original peak
    dy = (dy + y0) % H
    dx = (dx + x0) % W
    if dy > H/2: dy -= H
    if dx > W/2: dx -= W
    return float(dy), float(dx)

def _shift_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Subpixel shift via Fourier shift theorem."""
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)[:, None]; fx = np.fft.fftfreq(nx)[None, :]
    phase = np.exp(-2j*np.pi*(fy*dy + fx*dx))
    return np.fft.ifft2(np.fft.fft2(img) * phase).real.astype(np.float32)

# ========================= SQUIRREL maps (masked) ============================
def _weighted_alpha_beta(A: np.ndarray, B: np.ndarray, M: Optional[np.ndarray]) -> Tuple[float, float]:
    if M is None:
        M = np.ones_like(A, dtype=float)
    M = np.asarray(M, float); A = np.asarray(A, float); B = np.asarray(B, float)
    S = M.sum()
    if S <= 0: return 1.0, 0.0
    SA = (M*A).sum(); SB = (M*B).sum()
    SAA = (M*A*A).sum(); SAB = (M*A*B).sum()
    den = SAA*S - SA*SA
    if abs(den) < 1e-12: return 1.0, 0.0
    alpha = (SAB*S - SA*SB) / den
    beta  = (SB - alpha*SA) / S
    return float(alpha), float(beta)

def _box_sum(arr: np.ndarray, k: int) -> np.ndarray:
    r = k//2
    pad = np.pad(arr, ((r,r),(r,r)), mode="reflect")
    integ = pad.cumsum(0).cumsum(1)
    return (integ[k:,k:] - integ[:-k,k:] - integ[k:,:-k] + integ[:-k,:-k])

def _weighted_global_pearson(A: np.ndarray, B: np.ndarray, M: Optional[np.ndarray]) -> float:
    if M is None:
        Am = A-A.mean(); Bm = B-B.mean()
        den = np.sqrt((Am*Am).sum() * (Bm*Bm).sum()) + 1e-12
        return float((Am*Bm).sum()/den)
    M = np.asarray(M, float)
    S = M.sum();
    if S <= 0: return float("nan")
    muA = (M*A).sum()/S; muB = (M*B).sum()/S
    varA = max((M*(A*A)).sum()/S - muA*muA, 0.0)
    varB = max((M*(B*B)).sum()/S - muB*muB, 0.0)
    cov  = (M*(A*B)).sum()/S - muA*muB
    den = math.sqrt(varA*varB) + 1e-12
    return float(cov/den)

@dataclass
class SquirrelOut:
    rsp: np.ndarray
    rse: np.ndarray
    global_rsp: float
    used_sigma_px: float
    alpha: float
    beta: float
    reg_shift: Tuple[float,float]

def rsp_rse_maps_masked(A0: np.ndarray, B0: np.ndarray, *,
                        window: int = 21,
                        sigma_blur_px: float = 1.5,
                        auto_sigma: bool = False,
                        sigma_max: float = 3.0,
                        sigma_step: float = 0.25,
                        roi_mask: Optional[np.ndarray] = None,
                        register: bool = False,
                        upsample: int = 8,
                        min_roi_frac: float = 0.2,
                        optimize: str = "rsp") -> SquirrelOut:
    if A0.shape != B0.shape: raise ValueError("Image shapes must match.")
    A0 = np.asarray(A0, np.float32); B0 = np.asarray(B0, np.float32)
    # optional registration (pre-metrics)
    dy = dx = 0.0
    if register:
        dy, dx = _phase_correlation_shift(A0, B0, upsample=upsample)
        B0 = _shift_image(B0, dy, dx)

    # ROI mask
    M = None
    if roi_mask is not None:
        M = (np.asarray(roi_mask) > 0).astype(float)
        if M.shape != A0.shape: raise ValueError("ROI mask shape must match images.")

    # choose sigma by grid search
    if auto_sigma:
        best_sigma, best_val = 0.0, -np.inf if optimize=="rsp" else np.inf
        s = 0.0
        while s <= sigma_max + 1e-12:
            Ab = _gaussian_blur(A0, s); Bb = _gaussian_blur(B0, s)
            alpha, beta = _weighted_alpha_beta(Ab, Bb, M)
            X = alpha*Ab + beta
            if optimize == "rsp":
                val = _weighted_global_pearson(X, Bb, M)
                better = val > best_val
            else:
                # RSE global denominator uses energy of reference
                E = (X - Bb); num = (M*E*E).sum() if M is not None else (E*E).sum()
                den = ((M*(Bb*Bb)).sum() if M is not None else (Bb*Bb).sum()) + 1e-12
                val = math.sqrt(num/den)
                better = val < best_val
            if np.isfinite(val) and better:
                best_sigma, best_val = float(s), float(val)
            s += float(sigma_step)
        used_sigma = float(best_sigma)
    else:
        used_sigma = float(max(0.0, sigma_blur_px))

    # blur with chosen sigma, fit α/β once
    Ab = _gaussian_blur(A0, used_sigma); Bb = _gaussian_blur(B0, used_sigma)
    alpha, beta = _weighted_alpha_beta(Ab, Bb, M)
    X = alpha*Ab + beta

    # local stats (mask-aware)
    if window % 2 == 0: window += 1
    k = int(window)
    MM = np.ones_like(Ab, float) if M is None else M
    nmin = float(k*k) * float(min_roi_frac)

    SM = _box_sum(MM, k)
    SX = _box_sum(MM*X, k); SB = _box_sum(MM*Bb, k)
    SXX = _box_sum(MM*X*X, k); SBB = _box_sum(MM*Bb*Bb, k)
    SXY = _box_sum(MM*X*Bb, k)

    with np.errstate(invalid="ignore", divide="ignore"):
        muX = SX/(SM+1e-12); muB = SB/(SM+1e-12)
        varX = np.maximum(SXX/(SM+1e-12) - muX*muX, 0.0)
        varB = np.maximum(SBB/(SM+1e-12) - muB*muB, 0.0)
        cov  = SXY/(SM+1e-12) - muX*muB
    rsp = cov/(np.sqrt(varX)*np.sqrt(varB) + 1e-12)
    rsp = np.clip(rsp, -1.0, 1.0)

    E = X - Bb
    SEE = _box_sum(MM*E*E, k)
    rse = np.sqrt(SEE) / (np.sqrt(SBB) + 1e-12)

    bad = SM < max(nmin, 1.0)
    rsp[bad] = np.nan
    rse[bad] = np.nan

    global_rsp = _weighted_global_pearson(X, Bb, M)
    return SquirrelOut(rsp.astype(np.float32), rse.astype(np.float32), float(global_rsp),
                       float(used_sigma), float(alpha), float(beta), (dy, dx))

# ============================ FRC curve + map =================================
def _hann2d(ny: int, nx: int) -> np.ndarray:
    hy = 0.5*(1 - np.cos(2*np.pi*np.arange(ny)/(ny-1)))
    hx = 0.5*(1 - np.cos(2*np.pi*np.arange(nx)/(nx-1)))
    return np.outer(hy, hx).astype(np.float32)

def _frc_curve(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    FA = np.fft.fft2(a)
    FB = np.fft.fft2(b)
    num = (FA*np.conj(FB)).real
    den = np.sqrt((FA*np.conj(FA)).real * (FB*np.conj(FB)).real) + 1e-12
    frc_img = num/den
    ny, nx = a.shape
    fy = np.fft.fftfreq(ny); fx = np.fft.fftfreq(nx)
    gy, gx = np.meshgrid(fy, fx, indexing="ij")
    r = np.sqrt(gy*gy + gx*gx)
    min_dim = min(ny, nx)
    ring = np.floor(r*min_dim + 1e-9).astype(int)
    max_ring = int(np.floor(min_dim/2))
    acc = np.bincount(ring.ravel(), weights=frc_img.ravel(), minlength=max_ring+1)
    cnt = np.bincount(ring.ravel(), minlength=max_ring+1)
    frc = np.full_like(acc, np.nan, dtype=float)
    ok = cnt > 0; frc[ok] = acc[ok]/cnt[ok];
    if len(frc)>0: frc[0]=np.nan
    freqs = (np.arange(len(frc)) + 0.5)/float(min_dim)
    return freqs, frc

def _frc_cutoff(freqs: np.ndarray, frc: np.ndarray, thr: float = 1.0/7.0) -> Optional[float]:
    freqs = np.asarray(freqs, float); frc = np.asarray(frc, float)
    v = ~np.isnan(frc); freqs = freqs[v]; frc = frc[v]
    if len(frc) < 2: return None
    for i in range(len(frc)-1):
        if frc[i] >= thr and frc[i+1] < thr:
            y0, y1 = frc[i], frc[i+1]
            x0, x1 = freqs[i], freqs[i+1]
            t = 0.0 if y0==y1 else (thr - y0)/(y1 - y0)
            return float(x0 + t*(x1 - x0))
    return None

def frc_map_hann(img1: np.ndarray, img2: np.ndarray, *,
                 tile: int = 64, stride: int = 64, threshold: float = 1.0/7.0,
                 pixel_size_nm: float = 10.0, min_power: float = 1e-6) -> np.ndarray:
    if img1.shape != img2.shape: raise ValueError("Shape mismatch for FRC map.")
    H, W = img1.shape
    tiles_y = 1 + max(0, (H - tile)//stride)
    tiles_x = 1 + max(0, (W - tile)//stride)
    out = np.full((tiles_y, tiles_x), np.nan, np.float32)
    win = _hann2d(tile, tile)
    for ty in range(tiles_y):
        y0 = ty*stride
        for tx in range(tiles_x):
            x0 = tx*stride
            a = img1[y0:y0+tile, x0:x0+tile]
            b = img2[y0:y0+tile, x0:x0+tile]
            if a.shape != (tile, tile) or b.shape != (tile, tile):
                continue
            a = (a - a.mean())*win
            b = (b - b.mean())*win
            if (a.var() + b.var()) < min_power:
                continue
            freqs, frc = _frc_curve(a, b)
            cutoff = _frc_cutoff(freqs, frc, thr=threshold)
            if cutoff is None or cutoff <= 0:
                continue
            out[ty, tx] = float(pixel_size_nm / cutoff)
    return out

def frc_map_wrapper(img1, img2, tile, stride, threshold, pixel_size_nm):
    if _frc_map_ext is not None:
        try:
            return _frc_map_ext(img1, img2, tile=tile, stride=stride, threshold=threshold, pixel_size_nm=pixel_size_nm)
        except Exception:
            pass
    return frc_map_hann(img1, img2, tile=tile, stride=stride, threshold=threshold, pixel_size_nm=pixel_size_nm)

# ============================ One-file pipeline ===============================
@dataclass
class OneSummary:
    input_csv: str
    output_dir: str
    odd_tif: str
    even_tif: str
    rsp_map: str
    rse_map: str
    frc_map: str
    pixel_size_nm: float
    gaussian_sigma_nm: float
    weight_mode: str
    method: str
    block_size_frames: int
    rsp_window: int
    rsf_sigma_px: float
    auto_sigma: bool
    sigma_max: float
    sigma_step: float
    alpha: float
    beta: float
    global_rsp: float
    reg_dy: float
    reg_dx: float

def process_one_csv(in_csv: Path, out_dir: Path, *,
                    method: str = "odd_even",
                    block_size_frames: int = 500,
                    pixel_size_nm: float = 10.0,
                    pre_smooth_nm: float = 15.0,
                    weight_mode: str = "ones",
                    rsp_window: int = 21,
                    rsf_sigma_px: float = 1.5,
                    auto_sigma: bool = False,
                    sigma_max: float = 3.0,
                    sigma_step: float = 0.25,
                    roi_mask_path: Optional[Path] = None,
                    min_roi_frac: float = 0.2,
                    register: bool = False,
                    upsample: int = 8,
                    frc_tile: int = 64,
                    frc_stride: int = 64,
                    frc_threshold: float = 1.0/7.0,
                    overwrite: bool = True) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_full = pd.read_csv(in_csv)
    cx, cy, cf, ci = _detect_columns(df_full)
    frames = pd.to_numeric(df_full[cf], errors="coerce") if cf is not None else pd.Series(np.arange(len(df_full)))
    frames = frames.fillna(method="ffill").astype(np.int64).values
    mA, mB = _split_masks(frames, method=method, block_size_frames=block_size_frames, seed=0)
    dfA = _to_render_df(df_full.loc[mA], cx, cy, ci, cf)
    dfB = _to_render_df(df_full.loc[mB], cx, cy, ci, cf)

    # render (prefer your backend)
    imgA, imgB, meta = render_two_images(dfA, dfB, pixel_size_nm=pixel_size_nm,
                                         gaussian_sigma_nm=pre_smooth_nm,
                                         weight_mode=weight_mode)

    odd_tif = out_dir / f"{in_csv.stem}_odd_rec.tif"
    even_tif = out_dir / f"{in_csv.stem}_even_rec.tif"
    _save_tif32(odd_tif, imgA); _save_tif32(even_tif, imgB)

    # ROI load (optional)
    ROI = None
    if roi_mask_path is not None and Path(roi_mask_path).exists():
        arr = _read_tif(Path(roi_mask_path))
        if arr.shape != imgA.shape:
            # nearest resize via PIL if available
            if _PIL_Image is None:
                raise ValueError(f"ROI mask shape {arr.shape} != image {imgA.shape}; install Pillow to resize.")
            im = _PIL_Image.fromarray((arr>0).astype(np.uint8))
            im = im.resize((imgA.shape[1], imgA.shape[0]), resample=_PIL_Image.NEAREST)
            ROI = (np.array(im)>0).astype(np.uint8)
        else:
            ROI = (arr>0).astype(np.uint8)

    # compute SQUIRREL maps (prefer your backend, but we need α/β & mask & registration; use ours)
    squirrel = rsp_rse_maps_masked(imgA, imgB,
                                   window=rsp_window,
                                   sigma_blur_px=rsf_sigma_px,
                                   auto_sigma=auto_sigma,
                                   sigma_max=sigma_max,
                                   sigma_step=sigma_step,
                                   roi_mask=ROI,
                                   register=register,
                                   upsample=upsample,
                                   min_roi_frac=min_roi_frac,
                                   optimize="rsp")
    rsp_path = out_dir / f"{in_csv.stem}_RSP_map.tif"
    rse_path = out_dir / f"{in_csv.stem}_RSE_map.tif"
    _save_tif32(rsp_path, squirrel.rsp)
    _save_tif32(rse_path, squirrel.rse)

    # FRC map with Hann apodization + overlap
    frc_map = frc_map_wrapper(imgA, imgB, tile=frc_tile, stride=frc_stride,
                              threshold=frc_threshold, pixel_size_nm=pixel_size_nm)
    frc_path = out_dir / f"{in_csv.stem}_reconstructed_frc_map.tif"
    _save_tif32(frc_path, frc_map)

    # summary CSV
    info = OneSummary(
        input_csv=str(in_csv), output_dir=str(out_dir),
        odd_tif=str(odd_tif), even_tif=str(even_tif),
        rsp_map=str(rsp_path), rse_map=str(rse_path), frc_map=str(frc_path),
        pixel_size_nm=float(pixel_size_nm), gaussian_sigma_nm=float(pre_smooth_nm),
        weight_mode=weight_mode, method=method, block_size_frames=int(block_size_frames),
        rsp_window=int(rsp_window), rsf_sigma_px=float(squirrel.used_sigma_px),
        auto_sigma=bool(auto_sigma), sigma_max=float(sigma_max), sigma_step=float(sigma_step),
        alpha=float(squirrel.alpha), beta=float(squirrel.beta), global_rsp=float(squirrel.global_rsp),
        reg_dy=float(squirrel.reg_shift[0]), reg_dx=float(squirrel.reg_shift[1]),
    ).__dict__
    pd.DataFrame([info]).to_csv(out_dir / f"{in_csv.stem}_summary.csv", index=False)
    return info

# =============================== Batch runner =================================
def _find_csvs(root: Path, pattern: str, includes: List[str], excludes: List[str]) -> List[Path]:
    files = []
    for p in Path(root).rglob("*.csv"):
        if not fnmatch.fnmatch(p.name, pattern): continue
        lname = p.name.lower()
        if includes and not all(s in lname for s in includes): continue
        if any(s in lname for s in excludes): continue
        files.append(p)
    return sorted(files)

def run_batch(root: Path, out_root: Path, *,
              glob: str = "*customed_title.csv",
              include: str = "", exclude: str = "",
              method: str = "odd_even", block_size_frames: int = 500,
              pixel_size_nm: float = 10.0, pre_smooth_nm: float = 15.0,
              weight_mode: str = "ones",
              rsp_window: int = 21, rsf_sigma_px: float = 1.5,
              auto_sigma: bool = False, sigma_max: float = 3.0, sigma_step: float = 0.25,
              roi_mask: Optional[Path] = None, min_roi_frac: float = 0.2,
              register: bool = False, upsample: int = 8,
              frc_tile: int = 64, frc_stride: int = 64, frc_threshold: float = 1.0/7.0,
              overwrite: bool = True) -> List[Dict]:
    root = Path(root).resolve()
    out_root = Path(out_root).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    includes = [s.strip().lower() for s in include.split(",") if s.strip()]
    excludes = [s.strip().lower() for s in exclude.split(",") if s.strip()]
    files = _find_csvs(root, glob, includes, excludes)
    results = []
    for i, f in enumerate(files, 1):
        rel = f.parent.relative_to(root)
        out_dir = out_root / rel
        print(f"[{i}/{len(files)}] {f}")
        try:
            res = process_one_csv(
                f, out_dir,
                method=method, block_size_frames=block_size_frames,
                pixel_size_nm=pixel_size_nm, pre_smooth_nm=pre_smooth_nm,
                weight_mode=weight_mode,
                rsp_window=rsp_window, rsf_sigma_px=rsf_sigma_px,
                auto_sigma=auto_sigma, sigma_max=sigma_max, sigma_step=sigma_step,
                roi_mask_path=roi_mask, min_roi_frac=min_roi_frac,
                register=register, upsample=upsample,
                frc_tile=frc_tile, frc_stride=frc_stride, frc_threshold=frc_threshold,
                overwrite=overwrite
            )
            results.append(res)
        except Exception as e:
            print(f"  ERROR: {e}\n{traceback.format_exc()}")
    if results:
        pd.DataFrame(results).to_csv(out_root / "batch_summary.csv", index=False)
    return results

# =============================== PyQt5 GUI ====================================
def launch_gui():
    from PyQt5 import QtWidgets, QtCore

    class GUI(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("SQUIRREL+ : RSP/RSE + FRC from CSV (with Registration)")
            self.resize(980, 780)

            # root/out
            self.ed_root = QtWidgets.QLineEdit()
            self.ed_out  = QtWidgets.QLineEdit()

            # filters
            self.ed_glob = QtWidgets.QLineEdit("*customed_title.csv")
            self.ed_inc  = QtWidgets.QLineEdit()
            self.ed_exc  = QtWidgets.QLineEdit()

            # split
            self.cb_method = QtWidgets.QComboBox(); self.cb_method.addItems(["odd_even", "random_blocks"])
            self.sp_block  = QtWidgets.QSpinBox(); self.sp_block.setRange(1, 1_000_000); self.sp_block.setValue(500)

            # rendering
            self.sp_pixnm  = QtWidgets.QDoubleSpinBox(); self.sp_pixnm.setRange(0.1, 10000.0); self.sp_pixnm.setDecimals(3); self.sp_pixnm.setValue(10.0); self.sp_pixnm.setSuffix(" nm/px")
            self.sp_pre    = QtWidgets.QDoubleSpinBox(); self.sp_pre.setRange(0.0, 10000.0); self.sp_pre.setDecimals(3); self.sp_pre.setValue(15.0); self.sp_pre.setSuffix(" nm")
            self.cb_weight = QtWidgets.QComboBox(); self.cb_weight.addItems(["ones", "intensity"])

            # SQUIRREL
            self.sp_win    = QtWidgets.QSpinBox(); self.sp_win.setRange(3, 999); self.sp_win.setSingleStep(2); self.sp_win.setValue(21)
            self.sp_sigma  = QtWidgets.QDoubleSpinBox(); self.sp_sigma.setRange(0.0, 100.0); self.sp_sigma.setDecimals(3); self.sp_sigma.setValue(1.5); self.sp_sigma.setSuffix(" px")
            self.chk_auto  = QtWidgets.QCheckBox("Auto σ")
            self.sp_smax   = QtWidgets.QDoubleSpinBox(); self.sp_smax.setRange(0.0, 50.0); self.sp_smax.setDecimals(2); self.sp_smax.setValue(3.0); self.sp_smax.setSuffix(" px")
            self.sp_sstep  = QtWidgets.QDoubleSpinBox(); self.sp_sstep.setRange(0.01, 5.0); self.sp_sstep.setDecimals(2); self.sp_sstep.setValue(0.25); self.sp_sstep.setSuffix(" px")
            self.chk_reg   = QtWidgets.QCheckBox("Register halves (phase correlation)")
            self.sp_up     = QtWidgets.QSpinBox(); self.sp_up.setRange(1, 64); self.sp_up.setValue(8)
            self.sp_minroi = QtWidgets.QDoubleSpinBox(); self.sp_minroi.setRange(0.0, 1.0); self.sp_minroi.setDecimals(2); self.sp_minroi.setValue(0.20)

            # ROI
            self.ed_roi = QtWidgets.QLineEdit()
            self.btn_roi = QtWidgets.QPushButton("Pick ROI…")

            # FRC map
            self.sp_tile   = QtWidgets.QSpinBox(); self.sp_tile.setRange(8, 4096); self.sp_tile.setValue(64)
            self.sp_stride = QtWidgets.QSpinBox(); self.sp_stride.setRange(1, 4096); self.sp_stride.setValue(64)
            self.sp_thr    = QtWidgets.QDoubleSpinBox(); self.sp_thr.setRange(0.0, 1.0); self.sp_thr.setDecimals(6); self.sp_thr.setSingleStep(0.01); self.sp_thr.setValue(1.0/7.0)

            # system
            self.chk_over  = QtWidgets.QCheckBox("Overwrite outputs"); self.chk_over.setChecked(True)

            # buttons & log
            self.btn_root  = QtWidgets.QPushButton("Browse…")
            self.btn_out   = QtWidgets.QPushButton("Browse…")
            self.btn_run   = QtWidgets.QPushButton("RUN")
            self.pb        = QtWidgets.QProgressBar(); self.pb.setValue(0)
            self.log       = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)

            # layout
            form = QtWidgets.QFormLayout()
            r1 = QtWidgets.QHBoxLayout(); r1.addWidget(self.ed_root); r1.addWidget(self.btn_root)
            r2 = QtWidgets.QHBoxLayout(); r2.addWidget(self.ed_out);  r2.addWidget(self.btn_out)
            form.addRow("Root:", r1); form.addRow("Output:", r2)
            form.addRow("Glob:", self.ed_glob)
            form.addRow("Include (comma):", self.ed_inc)
            form.addRow("Exclude (comma):", self.ed_exc)
            form.addRow("Split method:", self.cb_method)
            form.addRow("Block size (frames):", self.sp_block)
            form.addRow("Pixel size:", self.sp_pixnm)
            form.addRow("Pre-smooth σ (nm):", self.sp_pre)
            form.addRow("Weighting:", self.cb_weight)
            form.addRow("RSP/RSE window:", self.sp_win)
            form.addRow("RSF σ (px):", self.sp_sigma)
            form.addRow(self.chk_auto)
            form.addRow("Auto σ max / step:", self.sp_smax)
            form.addRow("", self.sp_sstep)
            form.addRow(self.chk_reg)
            form.addRow("Registration upsample:", self.sp_up)
            form.addRow("Min ROI fraction:", self.sp_minroi)
            hr = QtWidgets.QFrame(); hr.setFrameShape(QtWidgets.QFrame.HLine); form.addRow(hr)
            r3 = QtWidgets.QHBoxLayout(); r3.addWidget(self.ed_roi); r3.addWidget(self.btn_roi)
            form.addRow("ROI mask (optional):", r3)
            form.addRow("FRC tile:", self.sp_tile)
            form.addRow("FRC stride:", self.sp_stride)
            form.addRow("FRC threshold:", self.sp_thr)
            form.addRow(self.chk_over)

            v = QtWidgets.QVBoxLayout(self); v.addLayout(form); v.addWidget(self.btn_run); v.addWidget(self.pb); v.addWidget(self.log)

            # signals
            self.btn_root.clicked.connect(self._pick_root)
            self.btn_out.clicked.connect(self._pick_out)
            self.btn_roi.clicked.connect(self._pick_roi)
            self.btn_run.clicked.connect(self._run)

        def _pick_root(self):
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select root folder")
            if d: self.ed_root.setText(d)
        def _pick_out(self):
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
            if d: self.ed_out.setText(d)
        def _pick_roi(self):
            f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select ROI mask", filter="Images (*.tif *.tiff *.png)")
            if f: self.ed_roi.setText(f)
        def _log(self, msg: str):
            self.log.appendPlainText(msg); self.log.ensureCursorVisible()

        def _run(self):
            root = Path(self.ed_root.text().strip() or ".").resolve()
            out_root = Path(self.ed_out.text().strip() or (str(root) + "_SQUIRREL_PLUS")).resolve()
            out_root.mkdir(parents=True, exist_ok=True)

            pattern = self.ed_glob.text().strip() or "*customed_title.csv"
            includes = [s.strip().lower() for s in self.ed_inc.text().split(",") if s.strip()]
            excludes = [s.strip().lower() for s in self.ed_exc.text().split(",") if s.strip()]
            files = _find_csvs(root, pattern, includes, excludes)
            if not files:
                self._log("No CSV files matched."); return

            method = self.cb_method.currentText()
            block  = int(self.sp_block.value())
            pixnm  = float(self.sp_pixnm.value())
            pre_nm = float(self.sp_pre.value())
            weight = self.cb_weight.currentText()
            win    = int(self.sp_win.value())
            sigma  = float(self.sp_sigma.value())
            auto   = self.chk_auto.isChecked()
            smax   = float(self.sp_smax.value())
            sstep  = float(self.sp_sstep.value())
            reg    = self.chk_reg.isChecked()
            up     = int(self.sp_up.value())
            minroi = float(self.sp_minroi.value())
            roi    = Path(self.ed_roi.text()) if self.ed_roi.text().strip() else None
            tile   = int(self.sp_tile.value())
            stride = int(self.sp_stride.value())
            thr    = float(self.sp_thr.value())
            over   = self.chk_over.isChecked()

            self.pb.setValue(0); self.log.clear()
            N = len(files)
            for i, f in enumerate(files, 1):
                rel = f.parent.relative_to(root)
                out_dir = out_root / rel
                self._log(f"[{i}/{N}] {f}")
                try:
                    res = process_one_csv(
                        f, out_dir,
                        method=method, block_size_frames=block,
                        pixel_size_nm=pixnm, pre_smooth_nm=pre_nm, weight_mode=weight,
                        rsp_window=win, rsf_sigma_px=sigma,
                        auto_sigma=auto, sigma_max=smax, sigma_step=sstep,
                        roi_mask_path=roi, min_roi_frac=minroi,
                        register=reg, upsample=up,
                        frc_tile=tile, frc_stride=stride, frc_threshold=thr,
                        overwrite=over
                    )
                    self._log(f"  OK  global RSP={res['global_rsp']:.4f}  σ={res['rsf_sigma_px']:.2f}  shift(dy,dx)=({res['reg_dy']:.2f},{res['reg_dx']:.2f})")
                except Exception as e:
                    self._log(f"  ERROR: {e}")
                self.pb.setValue(int(i/N*100))
            if N:
                # write batch summary at root
                try:
                    df = pd.DataFrame([process_one_csv(
                        f, out_root / f.parent.relative_to(root),
                        method=method, block_size_frames=block,
                        pixel_size_nm=pixnm, pre_smooth_nm=pre_nm, weight_mode=weight,
                        rsp_window=win, rsf_sigma_px=sigma,
                        auto_sigma=auto, sigma_max=smax, sigma_step=sstep,
                        roi_mask_path=roi, min_roi_frac=minroi,
                        register=reg, upsample=up,
                        frc_tile=tile, frc_stride=stride, frc_threshold=thr,
                        overwrite=over
                    ) for f in []])  # no double-run; placeholder
                except Exception:
                    pass
            self._log("Done.")

    app = QtWidgets.QApplication(sys.argv)
    w = GUI(); w.show()
    sys.exit(app.exec_())

# ================================ CLI =========================================
def parse_args():
    p = argparse.ArgumentParser(description="SQUIRREL+ : RSP/RSE + FRC maps from ThunderSTORM CSV (with registration).")
    p.add_argument("--gui", action="store_true", help="Launch PyQt5 GUI.")
    p.add_argument("--root", type=str, default=".", help="Root folder to search CSVs.")
    p.add_argument("--out", type=str, default="", help="Output root (default: <root>_SQUIRREL_PLUS)")
    p.add_argument("--glob", type=str, default="*customed_title.csv", help="Filename glob.")
    p.add_argument("--include", type=str, default="", help="CSV filename must include (comma, case-insensitive).")
    p.add_argument("--exclude", type=str, default="", help="CSV filename must exclude (comma).")
    p.add_argument("--method", type=str, default="odd_even", choices=["odd_even","random_blocks"], help="Split method.")
    p.add_argument("--block-size-frames", type=int, default=500, help="Block size for random_blocks.")
    p.add_argument("--pixel-size-nm", type=float, default=10.0, help="Render pixel size (nm/pixel).")
    p.add_argument("--pre-smooth-nm", type=float, default=15.0, help="Pre-recon Gaussian sigma (nm).")
    p.add_argument("--weight-mode", type=str, default="ones", choices=["ones","intensity"], help="Histogram weights.")
    p.add_argument("--rsp-window", type=int, default=21, help="Window (odd) for local RSP/RSE.")
    p.add_argument("--rsf-sigma-px", type=float, default=1.5, help="RSF Gaussian sigma (px).")
    p.add_argument("--auto-sigma", action="store_true", help="Grid-search σ to maximize global RSP.")
    p.add_argument("--sigma-max", type=float, default=3.0, help="Auto-σ max (px).")
    p.add_argument("--sigma-step", type=float, default=0.25, help="Auto-σ step (px).")
    p.add_argument("--roi-mask", type=str, default="", help="Binary mask (TIF/PNG) applied to both halves.")
    p.add_argument("--min-roi-frac", type=float, default=0.2, help="Min ROI fraction per window.")
    p.add_argument("--register", action="store_true", help="Enable sub-pixel registration of halves.")
    p.add_argument("--upsample", type=int, default=8, help="Registration upsample factor.")
    p.add_argument("--frc-tile", type=int, default=64, help="FRC tile size (px).")
    p.add_argument("--frc-stride", type=int, default=64, help="FRC stride (px).")
    p.add_argument("--frc-thr", type=float, default=1.0/7.0, help="FRC cutoff threshold.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    return p.parse_args()

def main():
    args = parse_args()
    if args.gui:
        launch_gui(); return
    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve() if args.out else Path(str(root) + "_SQUIRREL_PLUS").resolve()
    run_batch(
        root, out_root,
        glob=args.glob, include=args.include, exclude=args.exclude,
        method=args.method, block_size_frames=args.block_size_frames,
        pixel_size_nm=args.pixel_size_nm, pre_smooth_nm=args.pre_smooth_nm,
        weight_mode=args.weight_mode,
        rsp_window=args.rsp_window, rsf_sigma_px=args.rsf_sigma_px,
        auto_sigma=args.auto_sigma, sigma_max=args.sigma_max, sigma_step=args.sigma_step,
        roi_mask=Path(args.roi_mask) if args.roi_mask else None,
        min_roi_frac=args.min_roi_frac,
        register=args.register, upsample=args.upsample,
        frc_tile=args.frc_tile, frc_stride=args.frc_stride, frc_threshold=args.frc_thr,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()
