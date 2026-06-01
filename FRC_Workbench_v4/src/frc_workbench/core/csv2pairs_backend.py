# csv2pairs_backend.py (package module)
"""
CSV ➜ Odd/Even (or Random Blocks) ➜ Recon TIFFs ➜ RSP/RSE + FRC maps

This backend turns a ThunderSTORM-style localization CSV into:
  • *_odd.csv / *_even.csv (optional; full-column halves)
  • *_odd<recon_suffix>.tif and *_even<recon_suffix>.tif (float32)
  • <basename>_RSP_map.tif, <basename>_RSE_map.tif, <basename>_reconstructed_frc_map.tif
  • <basename>_batch_summary.csv (per-file summary)

Design notes
- Reuses functions from batch_frc_backend: rendering (hist2d + Gaussian), SQUIRREL-like RSP/RSE, FRC tile map.
- Parallel-friendliness: this file contains only pure functions with no global state.
- Robust column finding for ThunderSTORM headers (“x [nm]”, “y [nm]”, “frame”, “intensity [photon]”).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fnmatch

import numpy as np
import pandas as pd

# Optional TIFF I/O
try:
    import tifffile as _tifffile  # type: ignore
except Exception:
    _tifffile = None
try:
    from PIL import Image as _PIL_Image  # type: ignore
except Exception:
    _PIL_Image = None

# Reuse math from previous backend
from .frc_backend import (
   render_two_images, rsp_rse_maps, frc_map
)
from .diagnostics import gaussian_sigma_px
#from backend_frc_batch import render_two_images, rsp_rse_maps, frc_map

# ---------------------- Helpers ----------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_tif_float32(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32, order="C")
    if _tifffile is not None:
        _tifffile.imwrite(str(path), arr, dtype=np.float32)
        return
    if _PIL_Image is not None:
        # Fallback: scale to 16-bit
        m = float(np.nanmax(arr))
        if not np.isfinite(m) or m <= 0:
            m = 1.0
        scaled = np.clip(arr / m, 0, 1) * 65535.0
        _PIL_Image.fromarray(scaled.astype(np.uint16)).save(str(path), format="TIFF")
        return
    raise RuntimeError("No TIFF backend available. Install 'tifffile' or 'Pillow'.")

def _find_column(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in lower:
            return lower[k]
    squash = {c.lower().replace(" ", ""): c for c in df.columns}
    for k in keys:
        kk = k.replace(" ", "")
        if kk in squash:
            return squash[kk]
    return None

def _detect_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[str]]:
    cx = _find_column(df, ["x [nm]", "x[nm]", "x_nm", "x (nm)", "x"])
    cy = _find_column(df, ["y [nm]", "y[nm]", "y_nm", "y (nm)", "y"])
    cf = _find_column(df, ["frame", "frames", "frame #", "frame_number"])
    ci = _find_column(df, ["intensity [photon]", "intensity", "photons", "photon count"])
    if cx is None or cy is None:
        raise ValueError("Could not find 'x [nm]'/'y [nm]' columns in CSV.")
    return cx, cy, cf, ci

def _split_masks(frames: np.ndarray,
                 method: str = "odd_even",
                 block_size_frames: int = 500,
                 seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return boolean masks (mask_odd, mask_even_like) for splitting rows.
    For 'odd_even': true odd/evens by parity.
    For 'random_blocks': we still name halves 'odd'/'even' but they are A/B random blocks.
    """
    method = method.lower()
    n = frames.size
    if method == "odd_even":
        # odd = 1,3,5,... ; even = 0,2,4,...
        mask_odd = (frames % 2) == 1
        mask_even = ~mask_odd
        return mask_odd, mask_even
    # random_blocks
    f = frames.astype(np.int64, copy=False)
    f0 = int(f.min())
    block = max(1, int(block_size_frames))
    block_ids = (f - f0) // block
    uniq = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    half = len(uniq) // 2
    a_blocks = set(uniq[:half])
    mask_a = np.isin(block_ids, list(a_blocks))
    mask_b = ~mask_a
    return mask_a, mask_b

def _trim_basename(stem: str, trim_tokens: List[str]) -> str:
    s = stem
    low = s.lower()
    for tok in trim_tokens:
        t = tok.strip()
        if not t:
            continue
        i = low.find(t.lower())
        if i >= 0:
            s = s[:i] + s[i+len(t):]
            low = s.lower()
    # Clean potential double underscores after trimming
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def _to_render_df(df: pd.DataFrame, cx: str, cy: str, ci: Optional[str], cf: Optional[str]) -> pd.DataFrame:
    """Minimal numeric DataFrame for rendering with columns x_nm, y_nm, frame, intensity."""
    out = pd.DataFrame({
        "x_nm": pd.to_numeric(df[cx], errors="coerce"),
        "y_nm": pd.to_numeric(df[cy], errors="coerce"),
        "frame": pd.to_numeric(df[cf], errors="coerce") if cf is not None else np.arange(len(df), dtype=np.int64),
        "intensity": pd.to_numeric(df[ci], errors="coerce") if ci is not None else 1.0
    })
    out = out.dropna(subset=["x_nm", "y_nm", "frame", "intensity"])
    return out

# ---------------------- Core API ----------------------

@dataclass
class CSV2PairsSummary:
    input_csv: str
    output_dir: str
    odd_csv: Optional[str]
    even_csv: Optional[str]
    odd_tif: str
    even_tif: str
    rsp_map: str
    rse_map: str
    frc_map: str
    pixel_size_nm: float
    gaussian_sigma_nm: float
    gaussian_sigma_px: float
    weight_mode: str
    method: str
    block_size_frames: int
    rng_seed: int
    rsp_window: int
    rsp_sigma_px: float
    rsp_auto_sigma: bool
    frc_threshold: float
    frc_tile: int
    frc_stride: int
    global_rsp: float
    frc_map_median_nm: float
    frc_map_p10_nm: float
    frc_map_p90_nm: float

def process_csv_to_pairs(
    in_csv: Path,
    out_dir: Path,
    *,
    method: str = "odd_even",
    block_size_frames: int = 500,
    pixel_size_nm: float = 10.0,
    gaussian_sigma_nm: float = 15.0,
    weight_mode: str = "ones",
    recon_suffix: str = "_rec",
    save_half_csv: bool = True,
    trim_from_map_basename: Optional[List[str]] = None,
    squirrel_window: int = 21,
    squirrel_sigma_px: float = 1.5,
    squirrel_auto_sigma: bool = False,
    frc_threshold: float = 1.0/7.0,
    frc_tile: int = 64,
    frc_stride: int = 64,
    seed: int = 0,
    overwrite: bool = False,
) -> Dict:
    """
    Full pipeline for one CSV.
    Returns a dict (JSON-serializable) and writes a per-file summary CSV.
    """
    in_csv = Path(in_csv)
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Load once (preserve all columns for saving halves)
    df_full = pd.read_csv(in_csv)
    cx, cy, cf, ci = _detect_columns(df_full)
    frames = pd.to_numeric(df_full[cf], errors="coerce").fillna(method="ffill") if cf is not None else pd.Series(np.arange(len(df_full)), index=df_full.index)
    frames = frames.astype(np.int64).values

    mask_odd, mask_even = _split_masks(frames, method=method, block_size_frames=block_size_frames, seed=seed)
    df_odd_full = df_full.loc[mask_odd]
    df_even_full = df_full.loc[mask_even]

    # Optional: save half CSVs
    odd_csv_path = None
    even_csv_path = None
    if save_half_csv:
        odd_csv_path = out_dir / f"{in_csv.stem}_odd.csv"
        even_csv_path = out_dir / f"{in_csv.stem}_even.csv"
        if overwrite or True:
            df_odd_full.to_csv(odd_csv_path, index=False)
            df_even_full.to_csv(even_csv_path, index=False)

    # Minimal numeric DataFrames for rendering
    df_odd = _to_render_df(df_odd_full, cx, cy, ci, cf)
    df_even = _to_render_df(df_even_full, cx, cy, ci, cf)

    # Render two images (same binning for both halves, with smoothing)
    img_odd, img_even, meta = render_two_images(
        df_odd, df_even,
        pixel_size_nm=pixel_size_nm,
        gaussian_sigma_nm=gaussian_sigma_nm,
        weight_mode=weight_mode
    )

    # Save recon TIFFs
    odd_tif = out_dir / f"{in_csv.stem}_odd{recon_suffix}.tif"
    even_tif = out_dir / f"{in_csv.stem}_even{recon_suffix}.tif"
    _save_tif_float32(odd_tif, img_odd)
    _save_tif_float32(even_tif, img_even)

    # Choose the map base name (optionally trim tokens like "_thund")
    trim_tokens = list(trim_from_map_basename or [])
    map_base = _trim_basename(in_csv.stem, trim_tokens) if trim_tokens else in_csv.stem

    # Compute SQUIRREL-style maps and FRC tile map
    rsp_map, rse_map, global_rsp, used_sigma = rsp_rse_maps(
        img_odd, img_even,
        window=squirrel_window,
        sigma_blur_px=squirrel_sigma_px,
        auto_sigma=squirrel_auto_sigma
    )
    frc_res_map = frc_map(
        img_odd, img_even,
        tile=frc_tile, stride=frc_stride,
        threshold=frc_threshold,
        pixel_size_nm=pixel_size_nm
    )

    # Save maps (single underscore style to match your example)
    rsp_path = out_dir / f"{map_base}_RSP_map.tif"
    rse_path = out_dir / f"{map_base}_RSE_map.tif"
    frc_map_path = out_dir / f"{map_base}_reconstructed_frc_map.tif"
    _save_tif_float32(rsp_path, rsp_map)
    _save_tif_float32(rse_path, rse_map)
    _save_tif_float32(frc_map_path, frc_res_map)

    # Per-file summary
    med_res = float(np.nanmedian(frc_res_map)) if np.isfinite(np.nanmedian(frc_res_map)) else float('nan')
    p10 = float(np.nanpercentile(frc_res_map, 10)) if np.isfinite(np.nanpercentile(frc_res_map, 10)) else float('nan')
    p90 = float(np.nanpercentile(frc_res_map, 90)) if np.isfinite(np.nanpercentile(frc_res_map, 90)) else float('nan')

    one_summary_csv = out_dir / f"{map_base}_batch_summary.csv"
    pd.DataFrame([{
        "input_csv": str(in_csv),
        "odd_csv": str(odd_csv_path) if odd_csv_path else "",
        "even_csv": str(even_csv_path) if even_csv_path else "",
        "odd_tif": str(odd_tif),
        "even_tif": str(even_tif),
        "rsp_map": str(rsp_path),
        "rse_map": str(rse_path),
        "frc_map": str(frc_map_path),
        "pixel_size_nm": float(pixel_size_nm),
        "gaussian_sigma_nm": float(gaussian_sigma_nm),
        "gaussian_sigma_px": gaussian_sigma_px(gaussian_sigma_nm, pixel_size_nm),
        "weight_mode": weight_mode,
        "method": method,
        "block_size_frames": int(block_size_frames),
        "rng_seed": int(seed),
        "rsp_window": int(squirrel_window),
        "rsp_sigma_px": float(used_sigma if squirrel_auto_sigma else squirrel_sigma_px),
        "rsp_auto_sigma": bool(squirrel_auto_sigma),
        "frc_threshold": float(frc_threshold),
        "frc_tile": int(frc_tile),
        "frc_stride": int(frc_stride),
        "global_rsp": float(global_rsp),
        "frc_map_median_nm": med_res,
        "frc_map_p10_nm": p10,
        "frc_map_p90_nm": p90,
    }]).to_csv(one_summary_csv, index=False)

    return CSV2PairsSummary(
        input_csv=str(in_csv),
        output_dir=str(out_dir),
        odd_csv=str(odd_csv_path) if odd_csv_path else None,
        even_csv=str(even_csv_path) if even_csv_path else None,
        odd_tif=str(odd_tif),
        even_tif=str(even_tif),
        rsp_map=str(rsp_path),
        rse_map=str(rse_path),
        frc_map=str(frc_map_path),
        pixel_size_nm=float(pixel_size_nm),
        gaussian_sigma_nm=float(gaussian_sigma_nm),
        gaussian_sigma_px=gaussian_sigma_px(gaussian_sigma_nm, pixel_size_nm),
        weight_mode=weight_mode,
        method=method,
        block_size_frames=int(block_size_frames),
        rng_seed=int(seed),
        rsp_window=int(squirrel_window),
        rsp_sigma_px=float(used_sigma if squirrel_auto_sigma else squirrel_sigma_px),
        rsp_auto_sigma=bool(squirrel_auto_sigma),
        frc_threshold=float(frc_threshold),
        frc_tile=int(frc_tile),
        frc_stride=int(frc_stride),
        global_rsp=float(global_rsp),
        frc_map_median_nm=med_res,
        frc_map_p10_nm=p10,
        frc_map_p90_nm=p90,
    ).__dict__


def collect_csvs(root: Path,
                 pattern: str = "*customed_title.csv",
                 includes: Optional[List[str]] = None,
                 excludes: Optional[List[str]] = None) -> List[Path]:
    root = Path(root).resolve()
    includes = [s.lower() for s in (includes or []) if s]
    excludes = [s.lower() for s in (excludes or []) if s]
    files: List[Path] = []
    for p in root.rglob("*.csv"):
        name = p.name
        if not fnmatch.fnmatch(name, pattern):
            continue
        lname = name.lower()
        if includes and not all(s in lname for s in includes):
            continue
        if any(s in lname for s in excludes):
            continue
        files.append(p)
    return sorted(files)
