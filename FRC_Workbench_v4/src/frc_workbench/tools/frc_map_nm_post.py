#!/usr/bin/env python3
"""
FRC Map → Nanometers (nm) Post-Processor

Covers two scenarios:
A) Float32/float64 maps already in nm (direct use).
B) Display-intensity maps (uint8/uint16) requiring calibration to nm.

Features:
- Optional upsampling of tile-grid maps to image size (given tile/stride).
- Optional overlay on a reconstruction image with a chosen colormap.
- Robust handling of NaNs/Infs, zeros, and clipping.
- CSV summaries (global + ROI if mask provided).
- Optional uint16 export with sidecar JSON describing the nm↔uint16 mapping.

Usage examples are at the bottom of this file (search for "CLI EXAMPLES").
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
from tifffile import imread, imwrite

# Optional imports (used only if available / requested)
try:
    from skimage.transform import resize as _resize
except Exception:
    _resize = None

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from matplotlib import cm
except Exception:
    plt = None
    cm = None


# ----------------------------- Utilities -----------------------------

def _parse_shape(s: str) -> Tuple[int, int]:
    """
    Parse WxH or HxW strings into (H, W). Accepts '1024x2048' or '2048,1024'.
    """
    m = re.match(r"^\s*(\d+)\s*[x,]\s*(\d+)\s*$", s)
    if not m:
        raise ValueError(f"Invalid shape string '{s}'. Use 'H x W' or 'W x H'.")
    a, b = int(m.group(1)), int(m.group(2))
    # Heuristic: height is the smaller or the first? We prefer HxW input.
    # To avoid confusion, assume user gives HxW, but accept WxH if they say so.
    # We'll not guess — document to pass HxW.
    return (a, b)


def _safe_float_array(a: np.ndarray) -> np.ndarray:
    """
    Convert to float32 and sanitize NaNs/Infs.
    """
    out = a.astype(np.float32, copy=False)
    out = np.nan_to_num(out, nan=np.nan, posinf=np.nan, neginf=np.nan)
    return out


def _nan_sanitize(a: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """
    Replace +/-inf with NaN; keep NaNs as NaN, then optionally replace NaN with fill.
    """
    a = a.copy()
    a[~np.isfinite(a)] = np.nan
    if not math.isnan(fill):
        a = np.nan_to_num(a, nan=fill)
    return a


def _percentiles(a: np.ndarray, q: List[float]) -> List[float]:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float('nan') for _ in q]
    return list(np.percentile(a, q).astype(float))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------- Tile-grid upsampling -------------------------

def upsample_tile_map_to_image(tile_map: np.ndarray,
                               image_shape: Tuple[int, int],
                               tile: int,
                               stride: int) -> np.ndarray:
    """
    Expand a (Ty,Tx) tile grid to an image-sized (H,W) map by tessellating
    each tile's value over its footprint. Overlaps are averaged; the last
    row/col extend to the image edge so no strips are left uncovered.

    Parameters
    ----------
    tile_map : (Ty, Tx) array (float or int)
        Per-tile values (e.g., resolution in nm or display intensity).
    image_shape : (H, W)
        Target full image shape.
    tile : int
        Tile size in pixels used during FRC tiling.
    stride : int
        Stride in pixels used during FRC tiling.

    Returns
    -------
    (H, W) float32 array
    """
    H, W = map(int, image_shape)
    Ty, Tx = map(int, tile_map.shape)

    up = np.zeros((H, W), dtype=np.float32)
    wt = np.zeros((H, W), dtype=np.float32)

    for ty in range(Ty):
        y0 = ty * int(stride)
        last_row = (ty == Ty - 1)
        for tx in range(Tx):
            x0 = tx * int(stride)
            last_col = (tx == Tx - 1)
            y1 = H if last_row else min(y0 + int(tile), H)
            x1 = W if last_col else min(x0 + int(tile), W)
            v = float(tile_map[ty, tx])
            if np.isfinite(v) and (y1 > y0) and (x1 > x0):
                up[y0:y1, x0:x1] += v
                wt[y0:y1, x0:x1] += 1.0

    out = np.full((H, W), np.nan, dtype=np.float32)
    np.divide(up, wt, out=out, where=(wt > 0))
    return out


# -------------------------- Calibration logic --------------------------

@dataclass
class Calibration:
    a: float  # slope for nm = a * I + b
    b: float  # intercept

    def to_dict(self) -> Dict[str, float]:
        return {"a": float(self.a), "b": float(self.b)}

    @staticmethod
    def from_two_points(i0: float, nm0: float, i1: float, nm1: float) -> "Calibration":
        if i0 == i1:
            raise ValueError("i0 and i1 must be different for two-point calibration.")
        a = (nm1 - nm0) / (i1 - i0)
        b = nm0 - a * i0
        return Calibration(a=a, b=b)

    @staticmethod
    def from_range(int_min: float, int_max: float, nm_min: float, nm_max: float) -> "Calibration":
        if int_min == int_max:
            raise ValueError("int_min and int_max must differ for range calibration.")
        a = (nm_max - nm_min) / (int_max - int_min)
        b = nm_min - a * int_min
        return Calibration(a=a, b=b)


def apply_calibration(intensity: np.ndarray,
                      calib: Calibration,
                      clamp_nm: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Convert intensity image to nm using nm = a*I + b, return float32.
    Optionally clamp to [nm_min, nm_max].
    """
    I = intensity.astype(np.float32, copy=False)
    nm = calib.a * I + calib.b
    nm = _nan_sanitize(nm)
    if clamp_nm is not None:
        nm = np.clip(nm, clamp_nm[0], clamp_nm[1])
    return nm.astype(np.float32, copy=False)


# --------------------------- Statistics & I/O ---------------------------

def summarize_nm_map(nm: np.ndarray,
                     mask: Optional[np.ndarray],
                     thresholds_nm: Optional[List[float]]) -> Dict[str, object]:
    """
    Compute basic stats and area fractions under thresholds (if provided).
    """
    if mask is not None:
        if mask.shape != nm.shape:
            raise ValueError(f"ROI mask shape {mask.shape} does not match map {nm.shape}")
        valid = np.isfinite(nm) & (mask.astype(bool))
    else:
        valid = np.isfinite(nm)

    vals = nm[valid]
    n_pix = int(valid.sum())
    summary = {
        "count": n_pix,
        "min": float(np.nanmin(vals)) if n_pix else float('nan'),
        "max": float(np.nanmax(vals)) if n_pix else float('nan'),
        "mean": float(np.nanmean(vals)) if n_pix else float('nan'),
        "std": float(np.nanstd(vals)) if n_pix else float('nan'),
        "p10_p25_p50_p75_p90": _percentiles(vals, [10, 25, 50, 75, 90]),
    }

    if thresholds_nm:
        total = float(n_pix) if n_pix else float('nan')
        thr_fractions = {}
        for t in thresholds_nm:
            thr_fractions[f"area_frac_nm_<={t}"] = (float((vals <= t).sum()) / total) if n_pix else float('nan')
        summary.update(thr_fractions)

    return summary


def save_float32_tif(path: str, arr: np.ndarray, description: Optional[str] = None) -> None:
    imwrite(path, arr.astype(np.float32), metadata=None, description=description)


def save_uint16_with_sidecar(path: str, nm: np.ndarray,
                             nm_min: float, nm_max: float) -> None:
    """
    Save a uint16 version (for lightweight viewing) and drop a JSON
    sidecar with the linear mapping details so you can invert it.

    Mapping used: u16 = round( (nm - nm_min) / (nm_max - nm_min) * 65535 )
    """
    nm = nm.copy()
    nm = np.nan_to_num(nm, nan=np.nanmax(nm))  # put NaNs at max
    nm = np.clip(nm, nm_min, nm_max)
    denom = (nm_max - nm_min) if (nm_max > nm_min) else 1.0
    u16 = np.round((nm - nm_min) / denom * 65535.0).astype(np.uint16)
    imwrite(path, u16, metadata=None)
    meta = {
        "kind": "uint16_nm",
        "nm_min": float(nm_min),
        "nm_max": float(nm_max),
        "forward": "u16 = round((nm - nm_min) / (nm_max - nm_min) * 65535)",
        "inverse": "nm = nm_min + u16/65535 * (nm_max - nm_min)"
    }
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_histogram_png(path: str, nm: np.ndarray) -> None:
    if plt is None:
        print("[WARN] matplotlib not available; skipping histogram.")
        return
    data = nm[np.isfinite(nm)]
    if data.size == 0:
        print("[WARN] no finite data for histogram.")
        return
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=128)
    plt.xlabel("Resolution (nm)")
    plt.ylabel("Pixel count")
    plt.title("FRC Resolution Histogram")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_overlay_png(path: str,
                     recon: np.ndarray,
                     nm: np.ndarray,
                     opacity: float = 0.6,
                     colormap: str = "turbo",
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None) -> None:
    """
    Create a PNG overlay: grayscale recon + colorized nm map.
    Lower nm (better resolution) will appear 'hotter' by default if you
    choose an appropriate vmin/vmax (e.g., 10..50 nm).
    """
    if plt is None or cm is None:
        print("[WARN] matplotlib not available; skipping overlay.")
        return
    if recon.shape != nm.shape:
        raise ValueError(f"Overlay requires shape match: recon {recon.shape} vs nm {nm.shape}")
    base = recon.astype(np.float32)
    base = base - np.nanmin(base)
    denom = np.nanmax(base)
    base = base / (denom if denom > 0 else 1.0)
    base_rgb = np.stack([base, base, base], axis=-1)  # HxWx3

    data = nm.copy()
    if vmin is None or vmax is None:
        valid = data[np.isfinite(data)]
        if valid.size:
            # use robust percentiles to avoid outliers dominating
            vmin = np.percentile(valid, 5) if vmin is None else vmin
            vmax = np.percentile(valid, 95) if vmax is None else vmax
        else:
            vmin, vmax = 0.0, 1.0

    # Normalize nm to [0,1] for colormap (smaller nm = better)
    # We map smaller nm -> larger color intensity by inverting scale if desired.
    # Here we map linearly: 0 -> vmin, 1 -> vmax (so "good" = small = near 0.0)
    norm = np.clip((data - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    color_rgba = cmap(norm)  # HxWx4
    color_rgb = color_rgba[..., :3]

    overlay = (1.0 - opacity) * base_rgb + opacity * color_rgb
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(overlay, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200)
    plt.close()


# ------------------------------- Main ----------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Convert/standardize FRC maps to nanometers, optionally upsample tile maps, "
                    "create overlays, and export statistics."
    )
    p.add_argument("--frc-map", required=True, help="Path to FRC map image (TIFF/PNG).")
    p.add_argument("--out-dir", default=None, help="Output directory (default: alongside input).")
    p.add_argument("--basename", default=None, help="Base name for outputs (default: derived from input).")

    # If your input is a tile grid (Ty x Tx) and you want to expand it:
    p.add_argument("--is-tile-grid", action="store_true", help="Input is a tile grid (Ty x Tx).")
    p.add_argument("--tile", type=int, default=None, help="Tile size (px) used during FRC.")
    p.add_argument("--stride", type=int, default=None, help="Stride (px) used during FRC.")
    p.add_argument("--recon", default=None, help="Path to reconstruction image (for shape & overlay).")
    p.add_argument("--target-shape", default=None, help="Explicit target shape 'HxW' if no recon provided.")

    # Calibration choices if input is integer (display intensity)
    calib = p.add_argument_group("Calibration (only needed if input is integer)")
    calib.add_argument("--i0", type=float, help="Intensity at point 0.")
    calib.add_argument("--nm0", type=float, help="Nanometers at point 0.")
    calib.add_argument("--i1", type=float, help="Intensity at point 1.")
    calib.add_argument("--nm1", type=float, help="Nanometers at point 1.")
    calib.add_argument("--nm-min", type=float, help="nm at intensity min.")
    calib.add_argument("--nm-max", type=float, help="nm at intensity max.")
    calib.add_argument("--int-min", type=float, help="Override image intensity min.")
    calib.add_argument("--int-max", type=float, help="Override image intensity max.")
    calib.add_argument("--a", type=float, help="Direct slope: nm = a*I + b")
    calib.add_argument("--b", type=float, help="Direct intercept: nm = a*I + b")
    calib.add_argument("--clamp-min", type=float, default=None, help="Clamp nm minimum after conversion.")
    calib.add_argument("--clamp-max", type=float, default=None, help="Clamp nm maximum after conversion.")

    # ROI & thresholds
    p.add_argument("--roi-mask", default=None, help="Binary mask image (same size) to report ROI stats.")
    p.add_argument("--thresholds", type=float, nargs="*", default=[25, 35, 50],
                   help="nm thresholds for area fractions (default: 25 35 50).")

    # Overlays / plots
    p.add_argument("--make-overlay", action="store_true", help="Create a color overlay on --recon.")
    p.add_argument("--overlay-opacity", type=float, default=0.6)
    p.add_argument("--colormap", default="turbo", help="Matplotlib colormap (default: turbo).")
    p.add_argument("--hist", action="store_true", help="Save resolution histogram PNG.")

    # Optional uint16 export (for fast browsing in external tools)
    p.add_argument("--save-uint16", action="store_true", help="Also save a uint16 nm map + JSON sidecar.")

    args = p.parse_args()

    # I/O prep
    in_path = args.frc_map
    arr = imread(in_path)
    H_in, W_in = int(arr.shape[-2]), int(arr.shape[-1])
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(in_path)) or "."
    _ensure_dir(out_dir)
    base = args.basename or os.path.splitext(os.path.basename(in_path))[0]

    # Possibly upsample tile grid -> image size
    if args.is_tile_grid:
        if args.tile is None or args.stride is None:
            raise SystemExit("--is-tile-grid requires --tile and --stride.")
        if args.recon is None and args.target_shape is None:
            raise SystemExit("--is-tile-grid also requires either --recon or --target-shape (HxW).")

        if args.recon:
            recon = imread(args.recon)
            target_shape = recon.shape[-2], recon.shape[-1]
        else:
            target_shape = _parse_shape(args.target_shape)  # (H, W)

        tile_up = upsample_tile_map_to_image(arr, target_shape, args.tile, args.stride)
        arr = tile_up  # replace input with upsampled map
        H_in, W_in = arr.shape

    # Load optional recon (for overlay or to verify shape)
    recon_img = None
    if args.recon:
        recon_img = imread(args.recon)
        if recon_img.shape[-2:] != (H_in, W_in):
            raise SystemExit(f"--recon shape {recon_img.shape[-2:]} does not match map {arr.shape}.")

    # Load optional ROI mask
    roi = None
    if args.roi_mask:
        roi = imread(args.roi_mask)
        roi = (roi > 0).astype(np.uint8)
        if roi.shape[-2:] != (H_in, W_in):
            raise SystemExit(f"--roi-mask shape {roi.shape[-2:]} does not match map {arr.shape}.")

    # Determine if map is already in nm (float) or needs calibration (int)
    is_float = np.issubdtype(arr.dtype, np.floating)

    if is_float:
        nm_map = _safe_float_array(arr)
        nm_map = _nan_sanitize(nm_map)  # keep NaNs for empty regions
        conversion_info = {"mode": "float_nm", "note": "input values assumed to be nm"}
    else:
        # Integer intensity: build calibration
        I = arr.astype(np.float32, copy=False)
        i_min = float(np.min(I)) if args.int_min is None else float(args.int_min)
        i_max = float(np.max(I)) if args.int_max is None else float(args.int_max)

        calib_used: Optional[Calibration] = None

        # 1) Direct affine
        if args.a is not None and args.b is not None:
            calib_used = Calibration(a=float(args.a), b=float(args.b))

        # 2) Two-point
        elif (args.i0 is not None and args.nm0 is not None and
              args.i1 is not None and args.nm1 is not None):
            calib_used = Calibration.from_two_points(args.i0, args.nm0, args.i1, args.nm1)

        # 3) Range-based
        elif (args.nm_min is not None) and (args.nm_max is not None):
            calib_used = Calibration.from_range(
                args.int_min if args.int_min is not None else i_min,
                args.int_max if args.int_max is not None else i_max,
                args.nm_min, args.nm_max
            )

        if calib_used is None:
            raise SystemExit(
                "Input is integer (display intensity) but no calibration provided.\n"
                "Provide either: (--a --b) OR (--i0 --nm0 --i1 --nm1) OR (--nm-min --nm-max [--int-min --int-max])."
            )

        clamp_range = None
        if args.clamp_min is not None and args.clamp_max is not None:
            clamp_range = (float(args.clamp_min), float(args.clamp_max))

        nm_map = apply_calibration(I, calib_used, clamp_nm=clamp_range)
        conversion_info = {
            "mode": "intensity_calibrated",
            "calibration": calib_used.to_dict(),
            "intensity_min_used": i_min,
            "intensity_max_used": i_max,
            "clamp": clamp_range
        }

    # Save float32 nm map
    nm_path = os.path.join(out_dir, f"{base}__frc_nm_map.tif")
    save_float32_tif(nm_path, nm_map, description=json.dumps(conversion_info))
    print(f"[OK] Saved nm map: {nm_path}")

    # Optional uint16 map for quick viewing
    if args.save_uint16:
        # Use robust min/max for viewing; or clamp to percentiles
        valid = nm_map[np.isfinite(nm_map)]
        if valid.size:
            vmin = float(np.percentile(valid, 2))
            vmax = float(np.percentile(valid, 98))
            if vmin >= vmax:
                vmin = float(np.min(valid))
                vmax = float(np.max(valid))
        else:
            vmin, vmax = 0.0, 1.0
        nm_u16_path = os.path.join(out_dir, f"{base}__frc_nm_map_uint16.tif")
        save_uint16_with_sidecar(nm_u16_path, nm_map, vmin, vmax)
        print(f"[OK] Saved uint16 view + JSON: {nm_u16_path}")

    # Overlay (if requested and recon provided)
    if args.make_overlay:
        if recon_img is None:
            print("[WARN] --make-overlay requested but no --recon given; skipping.")
        else:
            overlay_path = os.path.join(out_dir, f"{base}__frc_overlay.png")
            make_overlay_png(overlay_path, recon_img, nm_map,
                             opacity=args.overlay_opacity,
                             colormap=args.colormap)
            print(f"[OK] Saved overlay: {overlay_path}")

    # Histogram (optional)
    if args.hist:
        hist_path = os.path.join(out_dir, f"{base}__frc_nm_hist.png")
        save_histogram_png(hist_path, nm_map)
        print(f"[OK] Saved histogram: {hist_path}")

    # CSV summary (global + ROI)
    summary: Dict[str, object] = {"input": in_path, "shape": [H_in, W_in]}
    summary.update(conversion_info)

    # Global
    summary["global"] = summarize_nm_map(nm_map, mask=None, thresholds_nm=args.thresholds)

    # ROI (if mask provided)
    if roi is not None:
        summary["roi"] = summarize_nm_map(nm_map, mask=roi, thresholds_nm=args.thresholds)

    # Save CSV-like JSON (more structured, easy to read)
    json_path = os.path.join(out_dir, f"{base}__frc_nm_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved summary JSON: {json_path}")

    # Also write a minimal CSV (common columns)
    csv_path = os.path.join(out_dir, f"{base}__frc_nm_summary.csv")
    import csv as _csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["scope", "count", "min", "max", "mean", "std", "p10", "p25", "p50", "p75", "p90"] +
                   [f"frac_<=_{int(t)}nm" for t in args.thresholds])
        # Global row
        g = summary["global"]
        p10, p25, p50, p75, p90 = g["p10_p25_p50_p75_p90"]
        row = ["global", g["count"], g["min"], g["max"], g["mean"], g["std"],
               p10, p25, p50, p75, p90] + [g.get(f"area_frac_nm_<={t}", "") for t in args.thresholds]
        w.writerow(row)
        # ROI row
        if "roi" in summary:
            r = summary["roi"]
            p10, p25, p50, p75, p90 = r["p10_p25_p50_p75_p90"]
            row = ["roi", r["count"], r["min"], r["max"], r["mean"], r["std"],
                   p10, p25, p50, p75, p90] + [r.get(f"area_frac_nm_<={t}", "") for t in args.thresholds]
            w.writerow(row)
    print(f"[OK] Saved summary CSV: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()


# --------------------------- CLI EXAMPLES ---------------------------
# 1) Your current float nm map (image-sized) -> overlay + stats:
#    python frc_map_nm_post.py --frc-map path/to/__frc_map_upsampled_to_recon.tif \
#           --recon path/to/reconstruction.tif --make-overlay --hist
#
# 2) Integer display map (uint16), with two-point calibration:
#    Example: intensity 475 => 200 nm, intensity 1200 => 35 nm
#    python frc_map_nm_post.py --frc-map path/to/FRC_display_uint16.tif \
#           --i0 475 --nm0 200 --i1 1200 --nm1 35 --save-uint16 --hist
#
# 3) Integer display map, simple range calibration:
#    Map min should represent 20 nm, max represents 60 nm:
#    python frc_map_nm_post.py --frc-map path/to/FRC_display_uint16.tif \
#           --nm-min 20 --nm-max 60
#
# 4) Tile grid input (Ty x Tx) that you want to expand to image size:
#    python frc_map_nm_post.py --frc-map path/to/__reconstructed_frc_map.tif \
#           --is-tile-grid --tile 32 --stride 16 \
#           --recon path/to/reconstruction.tif --make-overlay
#
# 5) Same tile grid but no reconstruction on disk; pass explicit target size:
#    python frc_map_nm_post.py --frc-map path/to/__reconstructed_frc_map.tif \
#           --is-tile-grid --tile 64 --stride 64 --target-shape 2560x4096
