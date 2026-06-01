# frc_backend.py
"""
FRC Workbench backend for SMLM/ThunderSTORM-style CSV files + odd/even TIF pair maps.

Fixes & upgrades:
- Robust, user-drawn ROI with Napari that blocks correctly until Save is clicked.
- Viewer uses robust contrast (2–99.5%) and tries "fire" LUT (fallback "inferno").
- FRC inside ROI now crops to ROI bbox, de-means each half, and apodizes edges → avoids Nyquist peg.
- Nyquist guard + smoothing: if cutoff ~0.5 cyc/px, re-estimate or fall back to tile-FRC median.
- ROI is saved (.roi.json) and additionally exported as a mask TIFF, so you can re-run with different
  parameters without redrawing.
- CSV summary now records which ROI file was used, and the ROI path is reusable between runs.
"""

from __future__ import annotations
import math, re, time, json, hashlib, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Sequence

from .diagnostics import BACKEND_ROI_APOD_SIGMA_PX, BACKEND_ROI_APOD_SIGMA_SOURCE, gaussian_sigma_px
import numpy as np
import pandas as pd

# ---------- Optional SciPy for gaussian_filter; we provide a numpy-only fallback ----------
try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------- Optional TIFF I/O ----------
_TIFF_BACKENDS = {}
try:
    import tifffile as _tifffile  # type: ignore
    _TIFF_BACKENDS["tifffile"] = _tifffile
except Exception:
    _tifffile = None
try:
    from PIL import Image as _PIL_Image  # type: ignore
    _TIFF_BACKENDS["pil"] = _PIL_Image
except Exception:
    _PIL_Image = None


# =========================================================================================
# ROI TOOLKIT
# =========================================================================================
try:
    import napari
    from qtpy import QtWidgets, QtCore
    from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QLabel, QMessageBox
    _HAS_NAPARI = True
except Exception:
    _HAS_NAPARI = False

try:
    from matplotlib.path import Path as MplPath
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


@dataclass
class ROIRecord:
    version: int
    created_at: str
    nm_per_px: float
    image_shape: Tuple[int, int]     # (rows, cols)
    polygon_px: List[List[float]]    # [[row, col], ...]
    source: Dict[str, Any]           # {"basename": "...", "recon_tag": "...", "sha256": "..."}

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_path(fpath: Path) -> "ROIRecord":
        with open(fpath, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        for key in ("version", "created_at", "nm_per_px", "image_shape", "polygon_px", "source"):
            if key not in obj:
                raise ValueError(f"ROI file missing key: {key}")
        obj["image_shape"] = tuple(obj["image_shape"])
        return ROIRecord(**obj)

    def mask(self, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        shape = image_shape or self.image_shape
        poly = np.asarray(self.polygon_px, dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            raise ValueError("ROI polygon must be Nx2 with N>=3")

        if _HAS_MPL:
            rr, cc = shape
            yy, xx = np.mgrid[:rr, :cc]
            pts = np.vstack([xx.ravel(), yy.ravel()]).T
            path = MplPath(poly[:, [1, 0]])  # (x, y) = (col, row)
            inside = path.contains_points(pts).reshape(shape)
            return inside

        # Ray-casting fallback (no external deps)
        rr, cc = shape
        mask = np.zeros((rr, cc), dtype=bool)
        y = np.arange(rr)[:, None]
        x = np.arange(cc)[None, :]
        poly_xy = poly[:, [1, 0]]
        xpoly, ypoly = poly_xy[:, 0], poly_xy[:, 1]
        n = len(xpoly)
        for i in range(n):
            j = (i - 1) % n
            xi, yi, xj, yj = xpoly[i], ypoly[i], xpoly[j], ypoly[j]
            cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            mask ^= cond
        return mask


def _sha256_small_file(path: Path, max_bytes: int = 512 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class ROIManager:
    """Handles ROI save/load/search and drawing in reconstructed image pixels."""
    def __init__(self,
                 roi_dir: Optional[Path] = None,
                 nm_per_px: Optional[float] = None,
                 recon_tag: str = "recon",
                 min_roi_pixels: int = 400):
        self.roi_dir = Path(roi_dir) if roi_dir else None
        self.nm_per_px = nm_per_px
        self.recon_tag = recon_tag
        self.min_roi_pixels = int(min_roi_pixels)

    # ---------- save/load/search ----------
    def _candidate_patterns(self, base: Path) -> List[str]:
        base_noext = base.stem
        nm_tag = f"{int(round(self.nm_per_px))}nm" if self.nm_per_px else "*nm"
        return [
            f"{base_noext}__{self.recon_tag}__{nm_tag}__*.roi.json",
            f"{base_noext}__*__{nm_tag}__*.roi.json",
            f"{base_noext}__*.roi.json",
        ]

    def find_existing(self, source_file: Path) -> List[Path]:
        hits: List[Path] = []
        search_dirs = []
        if self.roi_dir: search_dirs.append(self.roi_dir)
        search_dirs.append(source_file.parent)
        patterns = self._candidate_patterns(source_file)
        for d in search_dirs:
            for pat in patterns:
                hits.extend(sorted(d.glob(pat)))
        uniq: List[Path] = []
        seen = set()
        for p in sorted(set(hits), key=lambda p: p.stat().st_mtime, reverse=True):
            if p not in seen:
                uniq.append(p); seen.add(p)
        return uniq

    def save(self,
             source_file: Path,
             image_shape: Tuple[int, int],
             polygon_px: Sequence[Sequence[float]],
             sha256: Optional[str] = None) -> Path:
        if self.nm_per_px is None:
            raise ValueError("ROIManager.nm_per_px must be set for saving")
        base = source_file.stem
        nm_tag = f"{int(round(self.nm_per_px))}nm"
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"{base}__{self.recon_tag}__{nm_tag}__{ts}.roi.json"
        out_dir = self.roi_dir if self.roi_dir else source_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname
        record = ROIRecord(
            version=1,
            created_at=_now_iso(),
            nm_per_px=float(self.nm_per_px),
            image_shape=(int(image_shape[0]), int(image_shape[1])),
            polygon_px=[[float(r), float(c)] for r, c in polygon_px],
            source=dict(basename=source_file.name, recon_tag=self.recon_tag, sha256=sha256 or ""),
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(record.to_json())
        return out_path

    def load_and_adapt(self, roi_path: Path, target_shape: Tuple[int, int]) -> ROIRecord:
        rec = ROIRecord.from_path(roi_path)
        # adapt nm/px
        if self.nm_per_px is not None and rec.nm_per_px != self.nm_per_px:
            sf = self.nm_per_px / rec.nm_per_px
            poly = np.asarray(rec.polygon_px, float) * sf
            rec = ROIRecord(**{**asdict(rec), "nm_per_px": float(self.nm_per_px), "polygon_px": poly.tolist()})
        # adapt shape if uniform scale
        if rec.image_shape != tuple(target_shape):
            r0, c0 = rec.image_shape; r1, c1 = target_shape
            sry, srx = r1 / max(r0, 1), c1 / max(c0, 1)
            if abs(sry - srx) < 1e-6:
                poly = np.asarray(rec.polygon_px, float)
                poly[:, 0] *= sry; poly[:, 1] *= srx
                rec = ROIRecord(**{**asdict(rec), "image_shape": (int(r1), int(c1)), "polygon_px": poly.tolist()})
            else:
                raise ValueError(f"ROI image_shape mismatch: saved {rec.image_shape}, now {target_shape}")
        return rec

    # ---------- drawing ----------
    def draw(self, image: np.ndarray, title: str = "Draw ROI",
             existing: Optional[ROIRecord] = None,
             ui_mode: str = "gui") -> Optional[np.ndarray]:
        if ui_mode == "gui" and _HAS_NAPARI:
            return self._draw_napari(image, title, existing)
        return self._draw_matplotlib(image, title, existing)

    def _draw_napari(self, image: np.ndarray, title: str, existing: Optional[ROIRecord]) -> Optional[np.ndarray]:
        viewer = napari.Viewer(title=title)

        # Robust contrast & LUT with safe fallback
        try:
            vmin = float(np.nanpercentile(image, 2.0))
            vmax = float(np.nanpercentile(image, 99.5))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                raise ValueError
        except Exception:
            vmin = float(np.nanmin(image)) if np.isfinite(np.nanmin(image)) else 0.0
            vmax = float(np.nanmax(image)) if np.isfinite(np.nanmax(image)) else 1.0
            if vmax <= vmin: vmax = vmin + 1.0

        try:
            viewer.add_image(image, name="SR Image", colormap="fire",
                             blending="additive", contrast_limits=(vmin, vmax))
        except Exception:
            viewer.add_image(image, name="SR Image", colormap="inferno",
                             blending="additive", contrast_limits=(vmin, vmax))

        shapes = viewer.add_shapes(name="ROI", shape_type='polygon', edge_width=2)

        if existing is not None and existing.polygon_px:
            try:
                shapes.add(np.asarray(existing.polygon_px, float), shape_type='polygon', edge_color='yellow')
            except Exception:
                pass

        # Dock widget
        dock = QWidget()
        v = QVBoxLayout(dock)
        v.addWidget(QLabel("1) Draw or edit a polygon in the 'ROI' layer."))
        v.addWidget(QLabel("2) Click 'Save ROI & Continue' when done."))
        btn_save = QPushButton("Save ROI & Continue")
        v.addWidget(btn_save)
        viewer.window.add_dock_widget(dock, area="right")

        result_container = {"poly": None}

        def on_save():
            if len(shapes.data) == 0:
                try:
                    QMessageBox.warning(dock, "No ROI", "Please draw at least one polygon in the 'ROI' layer.")
                except Exception:
                    pass
                return
            poly = np.asarray(shapes.data[-1], float)
            if poly.shape[0] < 3:
                try:
                    QMessageBox.warning(dock, "Invalid ROI", "Polygon must have at least 3 points.")
                except Exception:
                    pass
                return
            result_container["poly"] = poly
            try:
                viewer.close()
            except Exception:
                pass

        btn_save.clicked.connect(on_save)

        # Block correctly even if a Qt app is running
        app = QtWidgets.QApplication.instance()
        if app is None:
            napari.run()
        else:
            viewer.show()
            loop = QtCore.QEventLoop()
            try:
                qtwin = viewer.window._qt_window
                qtwin.destroyed.connect(loop.quit)
            except Exception:
                t = QtCore.QTimer()
                t.setSingleShot(True)
                t.timeout.connect(loop.quit)
                t.start(1)
            loop.exec_()

        return result_container["poly"]

    def _draw_matplotlib(self, image: np.ndarray, title: str, existing: Optional[ROIRecord]) -> Optional[np.ndarray]:
        if not _HAS_MPL:
            raise RuntimeError("Matplotlib not available for ROI drawing. Use GUI/Napari.")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import PolygonSelector

        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        coords: List[Tuple[float, float]] = []

        def onselect(verts):
            coords.clear()
            coords.extend(verts)

        selector = PolygonSelector(ax, onselect)
        if existing is not None and existing.polygon_px:
            poly = np.asarray(existing.polygon_px, float)
            ax.plot(poly[:, 1], poly[:, 0], 'r-')

        print("Draw polygon; press 'enter' to accept; 'esc' to cancel.")
        accepted = {"ok": False}

        def on_key(event):
            if event.key == "enter":
                accepted["ok"] = True
                plt.close(fig)
            elif event.key == "escape":
                accepted["ok"] = False
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        try:
            plt.show(block=True)
        except TypeError:
            plt.show()

        if accepted["ok"] and len(coords) >= 3:
            return np.asarray([[y, x] for (x, y) in coords], float)
        return None

    # ---------- validation & control flow ----------
    def is_roi_sufficient(self, image: np.ndarray, roi_mask: np.ndarray) -> bool:
        if roi_mask is None or roi_mask.dtype != bool:
            return False
        nz = int(roi_mask.sum())
        if nz < self.min_roi_pixels:
            return False
        img_roi = np.asarray(image)[roi_mask]
        if img_roi.size == 0:
            return False
        return bool(np.nanmax(img_roi) > np.nanmin(img_roi))

    def apply_roi_or_prompt_redraw(self,
                                   source_file: Path,
                                   recon_even: np.ndarray,
                                   recon_odd: np.ndarray,
                                   compute_frc_fn,
                                   ui_mode: str = "gui",
                                   reuse_if_found: bool = True,
                                   ask_redraw_on_fail: bool = True,
                                   sha256_hint: Optional[str] = None) -> Tuple[Optional[float], Optional[Path], str]:
        """
        Returns: (resolution_nm or None, roi_path or None, status in {"APPLIED","SKIPPED"}).
        """
        assert recon_even.shape == recon_odd.shape, "recon halves must have same shape"
        image_shape = recon_even.shape[:2]

        chosen_rec: Optional[ROIRecord] = None
        chosen_path: Optional[Path] = None
        if reuse_if_found:
            cands = self.find_existing(source_file)
            if cands:
                if ui_mode == "gui" and len(cands) > 1:
                    try:
                        from qtpy.QtWidgets import QInputDialog
                        items = [str(p) for p in cands]
                        item, ok = QInputDialog.getItem(None, "Reuse ROI", "Choose an ROI to reuse:",
                                                        items, 0, False)
                        if ok:
                            chosen_path = Path(item)
                    except Exception:
                        chosen_path = cands[0]
                else:
                    chosen_path = cands[0]
                if chosen_path is not None:
                    try:
                        chosen_rec = self.load_and_adapt(chosen_path, image_shape)
                    except Exception:
                        chosen_rec = None
                        chosen_path = None

        # draw if none
        if chosen_rec is None:
            poly = self.draw(image=recon_even, title=f"Draw ROI — {source_file.name}",
                             existing=None, ui_mode=ui_mode)
            if poly is None:
                return None, None, "SKIPPED"
            roi_path = self.save(source_file, image_shape, poly, sha256=sha256_hint)
            chosen_path = roi_path
            chosen_rec = ROIRecord.from_path(roi_path)

        # compute with up to one redraw
        attempts = 0
        while True:
            mask = chosen_rec.mask(image_shape)
            if not self.is_roi_sufficient(recon_even, mask):
                if ask_redraw_on_fail and self._confirm_redraw_or_skip(ui_mode, "ROI looks too small or empty.\nRedraw ROI?"):
                    poly2 = self.draw(image=recon_even, title=f"Redraw ROI — {source_file.name}",
                                      existing=chosen_rec, ui_mode=ui_mode)
                    if poly2 is None:
                        return None, chosen_path, "SKIPPED"
                    chosen_path = self.save(source_file, image_shape, poly2, sha256=sha256_hint)
                    chosen_rec = ROIRecord.from_path(chosen_path)
                    attempts += 1
                    if attempts > 1:
                        return None, chosen_path, "SKIPPED"
                    continue
                else:
                    return None, chosen_path, "SKIPPED"

            even_roi = np.where(mask, recon_even, 0)
            odd_roi  = np.where(mask, recon_odd , 0)

            res_nm = None
            try:
                res_nm = compute_frc_fn(even_roi, odd_roi)
            except Exception:
                res_nm = None

            if res_nm is not None and np.isfinite(_safe_float(res_nm, np.nan)):
                return res_nm, chosen_path, "APPLIED"

            # failed; offer redraw
            if ask_redraw_on_fail and self._confirm_redraw_or_skip(ui_mode, "No valid FRC from ROI.\nRedraw ROI?"):
                poly3 = self.draw(image=recon_even, title=f"Redraw ROI — {source_file.name}",
                                  existing=chosen_rec, ui_mode=ui_mode)
                if poly3 is None:
                    return None, chosen_path, "SKIPPED"
                chosen_path = self.save(source_file, image_shape, poly3, sha256=sha256_hint)
                chosen_rec = ROIRecord.from_path(chosen_path)
                attempts += 1
                if attempts > 1:
                    return None, chosen_path, "SKIPPED"
                continue
            else:
                return None, chosen_path, "SKIPPED"

    @staticmethod
    def _confirm_redraw_or_skip(ui_mode: str, message: str) -> bool:
        """
        Returns True = Redraw, False = Skip. Never silently auto-skip.
        """
        if ui_mode == "gui":
            try:
                from qtpy.QtWidgets import QMessageBox
                box = QMessageBox()
                box.setIcon(QMessageBox.Question)
                box.setWindowTitle("ROI check")
                box.setText(message)
                redraw = box.addButton("Redraw ROI", QMessageBox.AcceptRole)
                skip   = box.addButton("Skip file", QMessageBox.RejectRole)
                box.exec_()
                return box.clickedButton() == redraw
            except Exception:
                # In GUI mode but couldn't show the box: default to "Redraw"
                return True
        # CLI fallback
        try:
            ans = input(f"{message} [R]edraw / [S]kip ? ").strip().lower()
            return ans.startswith("r")
        except Exception:
            # No stdin: default to Redraw (never silently skip)
            return True


# =========================================================================================
# Core math
# =========================================================================================

@dataclass
class FRCResult:
    frequencies_cyc_per_pix: np.ndarray    # shape (nbins,)
    frc: np.ndarray                        # shape (nbins,)
    cutoff_freq_cyc_per_pix: Optional[float]
    resolution_nm: Optional[float]
    threshold: float
    metadata: Dict


# -------------------------- CSV utilities --------------------------

def _find_column(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in lower_map:
            return lower_map[k]
    normalized = {c.lower().replace(" ", ""): c for c in df.columns}
    for k in keys:
        kk = k.replace(" ", "")
        if kk in normalized:
            return normalized[kk]
    return None

def load_localizations_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cx = _find_column(df, ["x [nm]", "x[nm]", "x_nm", "x (nm)", "x"])
    cy = _find_column(df, ["y [nm]", "y[nm]", "y_nm", "y (nm)", "y"])
    cf = _find_column(df, ["frame", "frames", "frame #", "frame_number"])
    ci = _find_column(df, ["intensity [photon]", "intensity", "photons", "photon count"])

    if cx is None or cy is None:
        raise ValueError("Could not find 'x [nm]'/'y [nm]' columns in CSV. Available columns: "
                         + ", ".join(df.columns))

    df = df.rename(columns={cx: "x_nm", cy: "y_nm"})
    if cf is not None:
        df = df.rename(columns={cf: "frame"})
    else:
        df["frame"] = np.arange(len(df), dtype=np.int64)
    if ci is not None:
        df = df.rename(columns={ci: "intensity"})
    else:
        df["intensity"] = 1.0

    for col in ["x_nm", "y_nm", "frame", "intensity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["x_nm", "y_nm", "frame", "intensity"])
    return df


# -------------------------- Splitting strategies --------------------------

def split_localizations(df: pd.DataFrame,
                        method: str = "odd_even",
                        block_size_frames: int = 500,
                        seed: Optional[int] = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    method = method.lower()
    if method not in {"odd_even", "random_blocks"}:
        raise ValueError(f"Unknown split method: {method}")

    if method == "odd_even":
        mask = (df["frame"].astype(np.int64) % 2) == 0
        return df[mask].copy(), df[~mask].copy()

    frames = df["frame"].to_numpy(dtype=np.int64)
    minf = int(frames.min())
    block_ids = (frames - minf) // int(max(1, block_size_frames))
    unique_blocks = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)
    half = len(unique_blocks) // 2
    blocks_a = set(unique_blocks[:half])
    mask_a = np.isin(block_ids, list(blocks_a))
    return df[mask_a].copy(), df[~mask_a].copy()


# -------------------------- Rendering utilities --------------------------

def _hist2d_weighted(x_nm: np.ndarray,
                     y_nm: np.ndarray,
                     weights: np.ndarray,
                     pixel_size_nm: float,
                     bbox: Optional[Tuple[float, float, float, float]] = None,
                     margin_px: int = 8) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if bbox is None:
        xmin = float(np.nanmin(x_nm)); xmax = float(np.nanmax(x_nm))
        ymin = float(np.nanmin(y_nm)); ymax = float(np.nanmax(y_nm))
    else:
        xmin, xmax, ymin, ymax = bbox

    xmin -= margin_px * pixel_size_nm; xmax += margin_px * pixel_size_nm
    ymin -= margin_px * pixel_size_nm; ymax += margin_px * pixel_size_nm

    width_nm = max(xmax - xmin, pixel_size_nm)
    height_nm = max(ymax - ymin, pixel_size_nm)

    nx = int(math.ceil(width_nm / pixel_size_nm))
    ny = int(math.ceil(height_nm / pixel_size_nm))

    x_edges = np.linspace(xmin, xmin + nx * pixel_size_nm, nx + 1)
    y_edges = np.linspace(ymin, ymin + ny * pixel_size_nm, ny + 1)

    H, _, _ = np.histogram2d(x_nm, y_nm, bins=[x_edges, y_edges], weights=weights)
    img = H.T
    return img, (x_edges, y_edges)

def _gaussian_smooth_numpy_fft(img: np.ndarray, sigma_pix: float) -> np.ndarray:
    if sigma_pix <= 0:
        return img
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    f2 = fy[:, None]**2 + fx[None, :]**2
    factor = np.exp(-2.0 * (np.pi * sigma_pix) ** 2 * f2)
    F = np.fft.fft2(img); F *= factor
    return np.fft.ifft2(F).real

def gaussian_blur(img: np.ndarray, sigma_pix: float) -> np.ndarray:
    if sigma_pix <= 0:
        return img
    if _HAVE_SCIPY:
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(img, sigma=sigma_pix)
        except Exception:
            pass
    return _gaussian_smooth_numpy_fft(img, sigma_pix)

def render_two_images(dfA: pd.DataFrame,
                      dfB: pd.DataFrame,
                      pixel_size_nm: float = 10.0,
                      gaussian_sigma_nm: float = 15.0,
                      weight_mode: str = "ones",
                      bbox: Optional[Tuple[float, float, float, float]] = None,
                      margin_px: int = 8) -> Tuple[np.ndarray, np.ndarray, Dict]:
    weightsA = np.ones(len(dfA), dtype=float) if weight_mode == "ones" else dfA["intensity"].to_numpy(float)
    weightsB = np.ones(len(dfB), dtype=float) if weight_mode == "ones" else dfB["intensity"].to_numpy(float)

    imgA, edges = _hist2d_weighted(dfA["x_nm"].to_numpy(float),
                                   dfA["y_nm"].to_numpy(float),
                                   weightsA,
                                   pixel_size_nm,
                                   bbox=bbox,
                                   margin_px=margin_px)
    x_edges, y_edges = edges
    H, _, _ = np.histogram2d(dfB["x_nm"].to_numpy(float),
                             dfB["y_nm"].to_numpy(float),
                             bins=[x_edges, y_edges],
                             weights=weightsB)
    imgB = H.T

    sigma_pix = gaussian_sigma_px(gaussian_sigma_nm, pixel_size_nm)
    imgA = gaussian_blur(imgA, sigma_pix)
    imgB = gaussian_blur(imgB, sigma_pix)

    meta = {
        "pixel_size_nm": float(pixel_size_nm),
        "x_edges_nm": x_edges,
        "y_edges_nm": y_edges,
        "gaussian_sigma_nm": float(gaussian_sigma_nm),
        "gaussian_sigma_px": gaussian_sigma_px(gaussian_sigma_nm, pixel_size_nm),
        "weight_mode": weight_mode,
        "image_shape": imgA.shape,
    }
    return imgA, imgB, meta


# -------------------------- FRC computation --------------------------

def _radial_bins(ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    grid_fy, grid_fx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(grid_fy**2 + grid_fx**2)
    min_dim = min(ny, nx)
    ring_index = np.floor(radius * min_dim + 1e-9).astype(int)
    return radius, ring_index

def compute_frc_curve(imgA: np.ndarray, imgB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if imgA.shape != imgB.shape:
        raise ValueError(f"Images must have same shape. Got {imgA.shape} vs {imgB.shape}")
    ny, nx = imgA.shape
    FA = np.fft.fft2(imgA)
    FB = np.fft.fft2(imgB)

    num = (FA * np.conj(FB)).real
    denA = (FA * np.conj(FA)).real
    denB = (FB * np.conj(FB)).real

    _, ring_index = _radial_bins(ny, nx)
    max_ring = int(np.floor(min(ny, nx) / 2))

    num_acc  = np.bincount(ring_index.ravel(), weights=num.ravel(),  minlength=max_ring + 1)
    denA_acc = np.bincount(ring_index.ravel(), weights=denA.ravel(), minlength=max_ring + 1)
    denB_acc = np.bincount(ring_index.ravel(), weights=denB.ravel(), minlength=max_ring + 1)
    counts   = np.bincount(ring_index.ravel(),                minlength=max_ring + 1)

    frc = np.zeros_like(num_acc, dtype=float)
    valid = (counts > 0) & (denA_acc > 0) & (denB_acc > 0)
    frc[valid] = num_acc[valid] / np.sqrt(denA_acc[valid] * denB_acc[valid])
    frc[~valid] = np.nan
    if len(frc) > 0:
        frc[0] = np.nan

    min_dim = min(ny, nx)
    ring_indices = np.arange(len(frc))
    freqs = (ring_indices + 0.5) / float(min_dim)
    freqs = np.clip(freqs, 0.0, 0.5)
    return freqs, frc

def _smooth_valid(y: np.ndarray, win: int = 3) -> np.ndarray:
    """NaN-aware moving average with window 'win' (odd)."""
    y = y.astype(float)
    win = max(1, int(win))
    if win == 1:
        return y
    k = np.ones(win, dtype=float)
    yy = y.copy()
    isn = np.isnan(yy)
    yy[isn] = 0.0
    num = np.convolve(yy, k, mode="same")
    den = np.convolve((~isn).astype(float), k, mode="same")
    out = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)
    return out

def find_cutoff_frequency(freqs: np.ndarray,
                          frc: np.ndarray,
                          threshold: float = 1.0/7.0) -> Optional[float]:
    """Basic crossing finder (kept for compatibility)."""
    freqs = np.asarray(freqs, dtype=float).ravel()
    frc = np.asarray(frc, dtype=float).ravel()
    if freqs.size != frc.size:
        raise ValueError("freqs and frc must have same length")
    valid = ~np.isnan(frc)
    freqs = freqs[valid]; frc = frc[valid]
    if freqs.size < 2:
        return None
    thr = float(threshold)
    for i in range(len(frc) - 1):
        if (frc[i] >= thr) and (frc[i + 1] < thr):
            x0, y0 = freqs[i], frc[i]
            x1, y1 = freqs[i + 1], frc[i + 1]
            if y0 == y1:
                return x1
            t = (thr - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return None

def find_cutoff_frequency_robust(freqs: np.ndarray,
                                 frc: np.ndarray,
                                 threshold: float = 1.0/7.0,
                                 smooth_bins: int = 5,
                                 nyquist_guard: float = 0.49) -> Optional[float]:
    """Smoothed crossing with Nyquist guard; returns None if at/near Nyquist."""
    freqs = np.asarray(freqs, float).ravel()
    frc = np.asarray(frc, float).ravel()
    if freqs.size != frc.size or freqs.size < 3:
        return None
    y = _smooth_valid(frc, win=smooth_bins)
    valid = ~np.isnan(y)
    f = freqs[valid]; y = y[valid]
    if f.size < 3:
        return None
    thr = float(threshold)
    # if never drops below threshold → no crossing
    if np.nanmin(y) >= thr:
        return None
    for i in range(len(y) - 1):
        if (y[i] >= thr) and (y[i + 1] < thr):
            x0, y0 = f[i], y[i]
            x1, y1 = f[i + 1], y[i + 1]
            if y0 == y1:
                x_cross = x1
            else:
                t = (thr - y0) / (y1 - y0)
                x_cross = x0 + t * (x1 - x0)
            # Nyquist guard
            if x_cross >= nyquist_guard:
                return None
            return float(x_cross)
    return None


# -------------------------- ROI‑robust FRC helper --------------------------

def _apodize_from_mask(mask: np.ndarray, sigma_pix: float = 2.0) -> np.ndarray:
    """Softens a binary ROI mask (0/1) to reduce edge artifacts."""
    m = np.asarray(mask, dtype=np.float32)
    if m.max() <= 0:
        return m
    w = gaussian_blur(m, sigma_pix)
    w = w / (w.max() + 1e-8)
    return np.clip(w, 0.0, 1.0)

def _compute_frc_nm_robust(even_roi: np.ndarray,
                           odd_roi: np.ndarray,
                           pixel_size_nm: float,
                           threshold: float) -> Optional[float]:
    """Compute FRC resolution (nm) robustly inside ROI‑masked images."""
    e = np.asarray(even_roi, dtype=np.float32)
    o = np.asarray(odd_roi , dtype=np.float32)

    # Detect ROI support from nonzeros (outside-ROI was zeroed)
    m = np.logical_or(e != 0, o != 0)
    if m.sum() < 9:
        return None
    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    e = e[y0:y1, x0:x1]
    o = o[y0:y1, x0:x1]
    m = m[y0:y1, x0:x1]

    # Apodize & demean to avoid DC and edge ringing
    w = _apodize_from_mask(m.astype(np.float32), sigma_pix=BACKEND_ROI_APOD_SIGMA_PX)
    if w.max() <= 0:
        return None
    e = (e - np.mean(e[m])) * w
    o = (o - np.mean(o[m])) * w

    # Power check
    if (np.var(e[m]) + np.var(o[m])) <= 1e-12:
        return None

    freqs, frc = compute_frc_curve(e, o)
    cutoff = find_cutoff_frequency_robust(freqs, frc, threshold=threshold, smooth_bins=5, nyquist_guard=0.49)
    if cutoff is not None and cutoff > 0:
        return float(pixel_size_nm / cutoff)

    # Fallback: tile-FRC median inside the cropped ROI
    tile = max(16, min(64, min(e.shape) // 2))
    res_map = frc_map(e, o, tile=tile, stride=tile, threshold=threshold, pixel_size_nm=pixel_size_nm)
    med = np.nanmedian(res_map)
    if np.isfinite(med):
        return float(med)
    return None


# -------------------------- I/O helpers --------------------------

def save_curve_csv(path: Path, freqs_cyc_per_nm: np.ndarray, frc: np.ndarray) -> None:
    df = pd.DataFrame({"frequency_cyc_per_nm": freqs_cyc_per_nm, "FRC": frc})
    df.to_csv(path, index=False)

def save_value_csv(path: Path, resolution_nm: Optional[float]) -> None:
    df = pd.DataFrame({"frc_resolution_nm": [resolution_nm if resolution_nm is not None else np.nan]})
    df.to_csv(path, index=False)

def save_curve_png(path: Path,
                   freqs_cyc_per_nm: np.ndarray,
                   frc: np.ndarray,
                   threshold: float,
                   resolution_nm: Optional[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 4.0), dpi=140)
    ax = plt.gca()
    ax.plot(freqs_cyc_per_nm, frc, label="FRC")
    ax.axhline(threshold, linestyle="--", label=f"Threshold = {threshold:g}")
    if resolution_nm is not None and np.isfinite(resolution_nm):
        f_c = 1.0 / resolution_nm
        ax.axvline(f_c, linestyle=":", label=f"Cutoff @ {resolution_nm:.1f} nm")
    ax.set_xlabel("Spatial frequency (cycles / nm)")
    ax.set_ylabel("FRC")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_tif_float32(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if _tifffile is not None:
        _tifffile.imwrite(str(path), arr, dtype=np.float32)
        return
    if _PIL_Image is not None:
        m = np.nanmax(arr)
        if not np.isfinite(m) or m <= 0:
            m = 1.0
        scaled = np.clip(arr / m, 0, 1) * 65535.0
        _PIL_Image.fromarray(scaled.astype(np.uint16)).save(str(path), format="TIFF")
        return
    raise RuntimeError("No TIFF backend available to save. Install 'tifffile'.")


# -------------------------- High-level: CSV (WITH ROI) --------------------------

def compute_frc_for_localizations(df: pd.DataFrame,
                                  method: str = "odd_even",
                                  block_size_frames: int = 500,
                                  pixel_size_nm: float = 10.0,
                                  gaussian_sigma_nm: float = 15.0,
                                  weight_mode: str = "ones",
                                  threshold: float = 1.0/7.0,
                                  seed: Optional[int] = 0) -> FRCResult:
    dfA, dfB = split_localizations(df, method=method, block_size_frames=block_size_frames, seed=seed)
    imgA, imgB, meta = render_two_images(dfA, dfB,
                                         pixel_size_nm=pixel_size_nm,
                                         gaussian_sigma_nm=gaussian_sigma_nm,
                                         weight_mode=weight_mode)
    freqs, frc = compute_frc_curve(imgA, imgB)
    cutoff = find_cutoff_frequency_robust(freqs, frc, threshold=threshold, smooth_bins=5, nyquist_guard=0.49)
    resolution_nm = float(pixel_size_nm / cutoff) if (cutoff is not None and cutoff > 0) else None
    return FRCResult(frequencies_cyc_per_pix=freqs,
                     frc=frc,
                     cutoff_freq_cyc_per_pix=cutoff,
                     resolution_nm=resolution_nm,
                     threshold=float(threshold),
                     metadata=meta)

def process_single_csv(in_csv: Path,
                       out_dir: Path,
                       method: str = "odd_even",
                       block_size_frames: int = 500,
                       pixel_size_nm: float = 10.0,
                       gaussian_sigma_nm: float = 15.0,
                       weight_mode: str = "ones",
                       threshold: float = 1.0/7.0,
                       seed: Optional[int] = 0,
                       overwrite: bool = False,
                       # NEW (optional) ROI knobs:
                       roi_dir: Optional[Path] = None,
                       reuse_roi: bool = True,
                       min_roi_pixels: int = 400,
                       ui_mode: str = "gui",
                       export_roi_mask_tif: bool = True) -> Dict:
    """
    CSV → split into halves → render → prompt/apply ROI → FRC curve/value.
    ROI sidecar JSON is saved; optional ROI mask TIFF is exported for reuse.
    """
    _ensure_dir(out_dir)

    # Load & split
    df = load_localizations_csv(in_csv)
    dfA, dfB = split_localizations(df, method=method,
                                   block_size_frames=block_size_frames,
                                   seed=seed)

    # Render both halves
    imgA, imgB, meta = render_two_images(dfA, dfB,
                                         pixel_size_nm=pixel_size_nm,
                                         gaussian_sigma_nm=gaussian_sigma_nm,
                                         weight_mode=weight_mode)

    # ROI manager
    mgr = ROIManager(roi_dir=Path(roi_dir) if roi_dir else None,
                     nm_per_px=pixel_size_nm,
                     recon_tag="csv",
                     min_roi_pixels=int(min_roi_pixels))

    # Optional file hash for traceability
    try:
        sha_hint = _sha256_small_file(Path(in_csv))
    except Exception:
        sha_hint = ""

    # Compute resolution (nm) robustly from ROI
    def _compute_frc(even_roi: np.ndarray, odd_roi: np.ndarray) -> Optional[float]:
        return _compute_frc_nm_robust(even_roi, odd_roi, pixel_size_nm=pixel_size_nm, threshold=threshold)

    res_nm, roi_path, status = mgr.apply_roi_or_prompt_redraw(
        source_file=Path(in_csv),
        recon_even=imgA,
        recon_odd=imgB,
        compute_frc_fn=_compute_frc,
        ui_mode=ui_mode,
        reuse_if_found=bool(reuse_roi),
        ask_redraw_on_fail=True,
        sha256_hint=sha_hint,
    )

    # Export an ROI mask TIFF for reuse, if we have a ROI file
    roi_mask_tif = ""
    if roi_path is not None:
        try:
            roi_rec = ROIRecord.from_path(roi_path)
            hard_mask = roi_rec.mask(imgA.shape).astype(np.float32)
            # put the mask next to the ROI json
            mask_name_root = roi_path.name.replace(".roi.json", "")
            mask_path = roi_path.parent / f"{mask_name_root}_roi_mask.tif"
            if export_roi_mask_tif:
                _save_tif_float32(mask_path, hard_mask)
            roi_mask_tif = str(mask_path)
        except Exception:
            pass

    # If no ROI success, return minimal summary (still record ROI json path if any)
    if res_nm is None:
        return {
            "type": "csv",
            "input_csv": str(in_csv),
            "output_dir": str(out_dir),
            "method": method,
            "block_size_frames": int(block_size_frames),
            "pixel_size_nm": float(pixel_size_nm),
            "threshold": float(threshold),
            "seed": int(seed if seed is not None else 0),
            "frc_resolution_nm": None,
            "gaussian_sigma_nm": float(gaussian_sigma_nm),
            "gaussian_sigma_px": gaussian_sigma_px(gaussian_sigma_nm, pixel_size_nm),
            "roi_apod_sigma_px_effective": BACKEND_ROI_APOD_SIGMA_PX,
            "roi_apod_sigma_source": BACKEND_ROI_APOD_SIGMA_SOURCE,
            "roi_file": str(roi_path) if roi_path else "",
            "roi_mask_tif": roi_mask_tif,
            "roi_status": "SKIPPED",
        }

    # Build ROI-masked images for saving the curve (use the same soft windowing)
    imgA_curve, imgB_curve = imgA.copy(), imgB.copy()
    cutoff = None
    if roi_path is not None and status == "APPLIED":
        try:
            roi_rec = ROIRecord.from_path(roi_path)
            hard_mask = roi_rec.mask(imgA.shape)
            w = _apodize_from_mask(hard_mask.astype(np.float32), sigma_pix=BACKEND_ROI_APOD_SIGMA_PX)
            imgA_curve = (imgA - np.mean(imgA[hard_mask])) * w
            imgB_curve = (imgB - np.mean(imgB[hard_mask])) * w
        except Exception:
            pass

    # Save curve & value
    freqs, frc = compute_frc_curve(imgA_curve, imgB_curve)
    cutoff = find_cutoff_frequency_robust(freqs, frc, threshold=threshold, smooth_bins=5, nyquist_guard=0.49)

    pix_nm = meta["pixel_size_nm"]
    freqs_cyc_per_nm = freqs / pix_nm

    stem = Path(in_csv).stem
    curve_csv = out_dir / f"{stem}__FRC_curve.csv"
    value_csv = out_dir / f"{stem}__FRC_value.csv"
    plot_png = out_dir / f"{stem}__FRC_curve.png"

    if overwrite or True:
        save_curve_csv(curve_csv, freqs_cyc_per_nm, frc)
        save_value_csv(value_csv, res_nm)
        save_curve_png(plot_png, freqs_cyc_per_nm, frc, threshold, res_nm)

    summary = {
        "type": "csv",
        "input_csv": str(in_csv),
        "output_dir": str(out_dir),
        "method": method,
        "block_size_frames": int(block_size_frames),
        "pixel_size_nm": float(pixel_size_nm),
        "gaussian_sigma_nm": float(gaussian_sigma_nm),
        "gaussian_sigma_px": gaussian_sigma_px(gaussian_sigma_nm, pixel_size_nm),
        "weight_mode": weight_mode,
        "threshold": float(threshold),
        "seed": int(seed if seed is not None else 0),
        "frc_resolution_nm": res_nm,
        "cutoff_freq_cyc_per_pix": cutoff,
        "image_shape": "x".join(map(str, meta.get("image_shape", (None, None)))),
        "n_localizations": int(len(df)),
        "roi_file": str(roi_path) if roi_path else "",
        "roi_mask_tif": roi_mask_tif,
        "roi_status": status,
        "roi_apod_sigma_px_effective": BACKEND_ROI_APOD_SIGMA_PX,
        "roi_apod_sigma_source": BACKEND_ROI_APOD_SIGMA_SOURCE,
    }
    return summary


# -------------------------- TIFF utilities (pairs) + maps --------------------------

_BASE_SUFFIX_RE = re.compile(r"(?i)(.*?)(?:_odd|_even)$")

def _read_tif(path: Path) -> np.ndarray:
    if _tifffile is not None:
        with _tifffile.TiffFile(str(path)) as tf:
            arr = tf.asarray()
        if arr.ndim > 2:
            arr = arr[0]
        return np.asarray(arr, dtype=np.float32)
    if _PIL_Image is not None:
        im = _PIL_Image.open(str(path))
        arr = np.array(im, dtype=np.float32)
        if arr.ndim > 2:
            arr = arr[..., 0]
        return arr
    raise RuntimeError("No TIFF backend available. Please install 'tifffile' or 'pillow'.")

def _pair_key_from_stem(stem: str) -> Optional[str]:
    m = _BASE_SUFFIX_RE.fullmatch(stem)
    return m.group(1) if m else None

def find_odd_even_pairs(folder: Path,
                        odd_suffix: str = "_odd",
                        even_suffix: str = "_even",
                        ext: str = ".tif") -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []
    by_dir: Dict[Tuple[Path, str], Dict[bool, Path]] = {}
    for p in folder.rglob(f"*{ext}"):
        stem = p.stem
        key = _pair_key_from_stem(stem)
        if key is None:
            continue
        by_dir.setdefault((p.parent, key), {})[stem.lower().endswith(even_suffix.lower())] = p
    for (d, key), dct in by_dir.items():
        odd_path = dct.get(False, None)
        even_path = dct.get(True,  None)
        if odd_path is None or even_path is None:
            maybe_odd  = d / f"{key}{odd_suffix}{ext}"
            maybe_even = d / f"{key}{even_suffix}{ext}"
            if maybe_odd.exists() and maybe_even.exists():
                pairs.append((maybe_odd, maybe_even, key))
            continue
        pairs.append((odd_path, even_path, key))
    return sorted(pairs)


# -------------------------- Local map utilities (box filter via integral image) --------------------------

def _box_filter_sum(img: np.ndarray, k: int) -> np.ndarray:
    """Sum over kxk window for each pixel using integral images (reflect padding)."""
    assert k >= 1 and k % 2 == 1, "window size k must be odd >=1"
    r = k // 2
    pad_img = np.pad(img, ((r, r), (r, r)), mode="reflect")
    integ = pad_img.cumsum(axis=0).cumsum(axis=1)
    s = (integ[k:, k:] - integ[:-k, k:] - integ[k:, :-k] + integ[:-k, :-k])
    return s

def rsp_rse_maps(img1: np.ndarray,
                 img2: np.ndarray,
                 window: int = 21,
                 sigma_blur_px: float = 1.5,
                 auto_sigma: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute SQUIRREL-like RSP and RSE maps between two SR images.
    Returns (RSP_map, RSE_map, global_RSP, used_sigma).
    """
    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)
    if img1.shape != img2.shape:
        raise ValueError(f"Images must match. Got {img1.shape} vs {img2.shape}")

    used_sigma = float(max(0.0, sigma_blur_px))
    if auto_sigma:
        sigmas = np.linspace(0.0, 3.0, 13)
        best = (None, -np.inf)
        for s in sigmas:
            a = gaussian_blur(img1, s); b = gaussian_blur(img2, s)
            am = a - a.mean(); bm = b - b.mean()
            denom = (np.sqrt((am**2).sum()) * np.sqrt((bm**2).sum()) + 1e-12)
            rsp = float((am*bm).sum() / denom)
            if rsp > best[1]:
                best = (s, rsp)
        used_sigma = float(best[0]) if best[0] is not None else used_sigma

    a = gaussian_blur(img1, used_sigma)
    b = gaussian_blur(img2, used_sigma)

    k = int(window);  k += (k % 2 == 0)
    Sa = _box_filter_sum(a, k);  Sb = _box_filter_sum(b, k)
    Saa = _box_filter_sum(a*a, k); Sbb = _box_filter_sum(b*b, k)
    Sab = _box_filter_sum(a*b, k); n = float(k*k)

    mu_a = Sa / n; mu_b = Sb / n
    var_a = np.maximum(Saa / n - mu_a*mu_a, 0.0)
    var_b = np.maximum(Sbb / n - mu_b*mu_b, 0.0)
    std_a = np.sqrt(var_a); std_b = np.sqrt(var_b)

    cov_ab = (Sab / n) - (mu_a * mu_b)
    rsp = cov_ab / (std_a * std_b + 1e-12)
    rsp = np.clip(rsp, -1.0, 1.0)

    se = _box_filter_sum((a - b)**2, k)
    ref_energy = _box_filter_sum(a*a, k)
    #rse = np.sqrt(se) / (np.sqrt(ref_energy) + 1e-12)
    se = np.nan_to_num(se, nan=0.0, posinf=0.0, neginf=0.0)
    ref_energy = np.nan_to_num(ref_energy, nan=0.0, posinf=0.0, neginf=0.0)

    num = np.sqrt(np.maximum(se, 0.0))
    den = np.sqrt(np.maximum(ref_energy, 0.0)) + 1e-12

    rse = np.divide(num, den, out=np.zeros_like(num), where=den>0)

    aa = a - a.mean(); bb = b - b.mean()
    denom = (np.sqrt((aa**2).sum()) * np.sqrt((bb**2).sum()) + 1e-12)
    global_rsp = float((aa*bb).sum() / denom)

    return rsp.astype(np.float32), rse.astype(np.float32), global_rsp, used_sigma


# -------------------------- FRC map from odd/even images (tile grid) --------------------------

def frc_map(img1: np.ndarray,
            img2: np.ndarray,
            tile: int = 64,
            stride: int = 64,
            threshold: float = 1.0/7.0,
            pixel_size_nm: float = 10.0,
            min_power: float = 1e-6,
            apodize_hann: bool = True) -> np.ndarray:
    """
    Compute a tile-based FRC resolution map (nm). Each tile produces one value.
    If a tile has too little power, the value is set to NaN.
    """

    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)
    if img1.shape != img2.shape:
        raise ValueError("Images must have same shape for FRC map.")
    H, W = img1.shape

    tile = int(tile);
    stride = int(stride) if stride > 0 else int(tile)
    if tile <= 8:
        raise ValueError("Tile size too small.")

    tiles_y = 1 + max(0, (H - tile) // stride)
    tiles_x = 1 + max(0, (W - tile) // stride)
    out = np.full((tiles_y, tiles_x), np.nan, dtype=np.float32)

    W = None
    if apodize_hann:
        win1d = np.hanning(tile).astype(np.float32)
        W = np.outer(win1d, win1d).astype(np.float32)

    for ty in range(tiles_y):
        y0 = ty * stride
        for tx in range(tiles_x):
            x0 = tx * stride
            a = img1[y0:y0+tile, x0:x0+tile]
            b = img2[y0:y0+tile, x0:x0+tile]
            if a.shape != (tile, tile) or b.shape != (tile, tile):
                continue

            # Jump over nearly void tiles
            if (a.var() + b.var()) < min_power:
                continue

            a = a - a.mean()
            b = b - b.mean()

            # Hann apodisation if activated
            if W is not None:
                a = a * W
                b = b * W

            if (a.var() + b.var()) < min_power:
                continue

            freqs, frc = compute_frc_curve(a, b)
            cutoff = find_cutoff_frequency(freqs, frc, threshold=threshold)
            if cutoff is None or cutoff <=0:
                continue
            out[ty, tx] = float(pixel_size_nm / cutoff)

    return out


def upsample_tile_map_to_image(tile_map: np.ndarray,
                               image_shape: Tuple[int, int],
                               tile: int,
                               stride: int) -> np.ndarray:
    """
    Expand a (Ty,Tx) tile grid to an image-sized (H,W) map by
    tessellating each tile's value over its footprint.
    Overlaps are averaged; the last row/col are extended to the image edge
    so there are no uncovered strips when H or W is not a multiple of stride.
    """
    H, W = map(int, image_shape)
    Ty, Tx = tile_map.shape
    up = np.zeros((H, W), dtype=np.float32)
    wt = np.zeros((H, W), dtype=np.float32)

    for ty in range(Ty):
        y0 = ty * int(stride)
        last_row = (ty == Ty - 1)
        for tx in range(Tx):
            x0 = tx * int(stride)
            last_col = (tx == Tx - 1)
            # Extend last tiles to the image edge
            y1 = H if last_row else min(y0 + int(tile), H)
            x1 = W if last_col else min(x0 + int(tile), W)
            v = float(tile_map[ty, tx])
            if np.isfinite(v) and y0 < H and x0 < W and y1 > y0 and x1 > x0:
                up[y0:y1, x0:x1] += v
                wt[y0:y1, x0:x1] += 1.0
    out = np.full((H, W), np.nan, dtype=np.float32)
    np.divide(up, wt, out=out, where=(wt > 0))
    return out



# -------------------------- Process a TIF pair (WITH ROI) --------------------------

def process_tif_pair(odd_path: Path,
                     even_path: Path,
                     out_dir: Path,
                     roi_mask: np.ndarray = None,
                     roi_apod_sigma_px: float = 2.0,
                     pixel_size_nm: float = 10.0,
                     squirrel_window: int = 21,
                     squirrel_sigma_px: float = 1.5,
                     squirrel_auto_sigma: bool = False,
                     frc_threshold: float = 1.0/7.0,
                     frc_tile: int = 64,
                     frc_stride: int = 64,
                     overwrite: bool = False,
                     roi_dir: Optional[Path] = None,
                     reuse_roi: bool = True,
                     min_roi_pixels: int = 400,
                     ui_mode: str = "gui",
                     apodize_hann: bool = True,
                     debug_qc: bool = False,
                     debug_n_tiles: int = 6) -> Dict:
    """
    Read odd/even TIFs, prompt/apply ROI, compute RSP/RSE maps and FRC map, and save.
    Returns summary dict.
    """

    dbg_dir = out_dir / "_debug"
    if debug_qc:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    _ensure_dir(out_dir)
    img_odd = _read_tif(odd_path)
    img_even = _read_tif(even_path)
    if img_odd.shape != img_even.shape:
        raise ValueError(f"Shape mismatch: {odd_path.name} {img_odd.shape} vs {even_path.name} {img_even.shape}")

    mgr = ROIManager(roi_dir=Path(roi_dir) if roi_dir else None,
                     nm_per_px=pixel_size_nm,
                     recon_tag="tifpair",
                     min_roi_pixels=int(min_roi_pixels))

    if roi_mask is not None:
        from scipy.ndimage import gaussian_filter
        m = roi_mask.astype(np.float32)
        if roi_apod_sigma_px > 0: m = gaussian_filter(m, roi_apod_sigma_px)
        m /= (m.max() + 1e-12)
        img_odd = img_odd * m; img_even = img_even * m

    def _compute_frc(even_roi: np.ndarray, odd_roi: np.ndarray) -> Optional[float]:
        return _compute_frc_nm_robust(even_roi, odd_roi, pixel_size_nm=pixel_size_nm, threshold=frc_threshold)

    try:
        sha_hint = _sha256_small_file(Path(odd_path))
    except Exception:
        sha_hint = ""

    res_nm, roi_path, status = mgr.apply_roi_or_prompt_redraw(
        source_file=odd_path,
        recon_even=img_even,
        recon_odd=img_odd,
        compute_frc_fn=_compute_frc,
        ui_mode=ui_mode,
        reuse_if_found=bool(reuse_roi),
        ask_redraw_on_fail=True,
        sha256_hint=sha_hint,
    )

    # Apply ROI if we have one
    if roi_path is not None and status == "APPLIED":
        try:
            roi_rec = ROIRecord.from_path(roi_path)
            hard_mask = roi_rec.mask(img_odd.shape)
            img_odd  = np.where(hard_mask, img_odd, 0)
            img_even = np.where(hard_mask, img_even, 0)
        except Exception:
            pass

    # Maps (within ROI if applied)
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

    frc_res_full = upsample_tile_map_to_image(frc_res_map, img_odd.shape, frc_tile, frc_stride)

    base = _pair_key_from_stem(odd_path.stem) or odd_path.stem.replace("_odd", "").replace("_even", "")
    rsp_path = out_dir / f"{base}__RSP_map.tif"
    rse_path = out_dir / f"{base}__RSE_map.tif"
    frc_map_path = out_dir / f"{base}__reconstructed_frc_map.tif"
    frc_full_path = out_dir / f"{base}__frc_map_upsampled_to_recon.tif"

    if overwrite or True:
        _save_tif_float32(rsp_path, rsp_map)
        _save_tif_float32(rse_path, rse_map)
        _save_tif_float32(frc_map_path, frc_res_map)
        _save_tif_float32(frc_full_path, frc_res_full)

    med_res = float(np.nanmedian(frc_res_map)) if np.isfinite(np.nanmedian(frc_res_map)) else float('nan')
    p10 = float(np.nanpercentile(frc_res_map, 10)) if np.isfinite(np.nanpercentile(frc_res_map, 10)) else float('nan')
    p90 = float(np.nanpercentile(frc_res_map, 90)) if np.isfinite(np.nanpercentile(frc_res_map, 90)) else float('nan')

    summary = {
        "type": "pair",
        "odd_tif": str(odd_path),
        "even_tif": str(even_path),
        "output_dir": str(out_dir),
        "pixel_size_nm": float(pixel_size_nm),
        "rsp_window": int(squirrel_window),
        "rsp_sigma_px": float(used_sigma),
        "global_rsp": float(global_rsp),
        "frc_threshold": float(frc_threshold),
        "frc_tile": int(frc_tile),
        "frc_stride": int(frc_stride),
        "frc_resolution_nm": res_nm if res_nm is not None else med_res,
        "requested_roi_apod_sigma_px": float(roi_apod_sigma_px),
        "roi_apod_sigma_px_effective": BACKEND_ROI_APOD_SIGMA_PX,
        "roi_apod_sigma_source": BACKEND_ROI_APOD_SIGMA_SOURCE,
        "frc_map_median_nm": med_res,
        "frc_map_p10_nm": p10,
        "frc_map_p90_nm": p90,
        "shape": f"{img_odd.shape[1]}x{img_odd.shape[0]}",
        "roi_file": str(roi_path) if roi_path else "",
        "roi_status": status,
        "rsp_map": str(rsp_map),
        "rse_map": str(rse_map),
        "frc_map": str(frc_map_path),
    }
    return summary
