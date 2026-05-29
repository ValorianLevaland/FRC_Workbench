from __future__ import annotations

import contextlib
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from qtpy import QtCore, QtWidgets

import napari

from frc_workbench.core.frc_backend import (
    ROIRecord,
    gaussian_blur,
    load_localizations_csv,
    split_localizations,
    render_two_images,
    rsp_rse_maps,
    frc_map,
    upsample_tile_map_to_image,
)
from frc_workbench.core.frc_analysis import compute_frc_plot_data, FRCPlotData
from frc_workbench.core.reference import compute_squirrel_vs_reference
from frc_workbench.core.roi import mask_from_polygon
from frc_workbench.core.io import ensure_dir, read_tif, save_tif_float32, sha256_small_file

from .mpl_canvas import FRCMatplotlibCanvas
from .workers import FunctionWorker


# =========================================================================================
# Small utilities
# =========================================================================================

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_dump(path: Path, obj: Any) -> None:
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _collect_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "tifffile", "napari", "qtpy"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[name] = "<not installed>"
    return versions


def _percentile_limits(img: np.ndarray, lo: float = 2.0, hi: float = 99.5) -> Tuple[float, float]:
    """Robust contrast limits for Napari layers."""
    a = np.asarray(img)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    vmin = float(np.percentile(a, lo))
    vmax = float(np.percentile(a, hi))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(a))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(a))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _remove_layer_if_exists(viewer: napari.Viewer, name: str) -> None:
    if name in viewer.layers:
        layer = viewer.layers[name]
        with contextlib.suppress(Exception):
            viewer.layers.remove(layer)


# =========================================================================================
# ROI helpers (Napari Shapes)
# =========================================================================================

def _ensure_roi_layer(viewer: napari.Viewer, name: str = "ROI"):
    if name in viewer.layers:
        layer = viewer.layers[name]
        if layer.__class__.__name__.lower() == "shapes":
            return layer
        layer.name = f"{name}_old"
    return viewer.add_shapes(name=name, shape_type="polygon", edge_width=2)


def _get_latest_polygon_from_roi_layer(viewer: napari.Viewer, name: str = "ROI") -> Optional[np.ndarray]:
    if name not in viewer.layers:
        return None
    layer = viewer.layers[name]
    if layer.__class__.__name__.lower() != "shapes":
        return None
    data = getattr(layer, "data", None)
    if not data:
        return None
    poly = np.asarray(data[-1], dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        return None
    return poly


# =========================================================================================
# Session model
# =========================================================================================

@dataclass
class FRCSession:
    # Input
    input_type: Optional[str] = None  # "csv" or "pair"
    csv_path: Optional[Path] = None
    odd_tif_path: Optional[Path] = None
    even_tif_path: Optional[Path] = None

    # Data
    df: Any = None
    df_odd: Any = None
    df_even: Any = None

    # Rendered images
    recon_odd: Optional[np.ndarray] = None
    recon_even: Optional[np.ndarray] = None
    recon_sum: Optional[np.ndarray] = None

    raw_odd: Optional[np.ndarray] = None
    raw_even: Optional[np.ndarray] = None
    raw_sum: Optional[np.ndarray] = None

    render_meta: Dict[str, Any] = field(default_factory=dict)

    # ROI
    roi_record: Optional[ROIRecord] = None
    roi_mask: Optional[np.ndarray] = None

    # Maps
    rsp_map: Optional[np.ndarray] = None
    rse_map: Optional[np.ndarray] = None
    frc_tile_map: Optional[np.ndarray] = None
    frc_full_map: Optional[np.ndarray] = None
    global_rsp: Optional[float] = None

    # FRC curve
    frc_plot: Optional[FRCPlotData] = None

    # Reference
    ref_image: Optional[np.ndarray] = None
    ref_path: Optional[Path] = None
    ref_shift_yx: Optional[Tuple[float, float]] = None
    rsf_sigma_px: Optional[float] = None
    rsf_alpha_beta: Optional[Tuple[float, float]] = None
    squirrel_rsp_map: Optional[np.ndarray] = None
    squirrel_rse_map: Optional[np.ndarray] = None
    squirrel_global_rsp: Optional[float] = None
    squirrel_global_rse: Optional[float] = None
    sr_degraded: Optional[np.ndarray] = None

    # Output
    output_dir: Optional[Path] = None

    def clear(self):
        self.__dict__.update(FRCSession().__dict__)


# =========================================================================================
# Main Workbench widget (Napari dock)
# =========================================================================================

class FRCWorkbenchWidget(QtWidgets.QWidget):
    """Load → render → ROI → FRC curve → maps → export, all from Napari."""

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.pool = QtCore.QThreadPool.globalInstance()
        self.session = FRCSession()
        self._busy_count = 0

        # ----------------------- UI widgets -----------------------

        self.lbl_dataset = QtWidgets.QLabel("No dataset loaded.")
        self.lbl_dataset.setWordWrap(True)

        # Display controls (keep UI uncluttered)
        self.chk_show_batch = QtWidgets.QCheckBox("Show batch tools (Batch FRC, CSV→Pairs)")
        self.chk_show_batch.setChecked(False)
        self.chk_show_batch.toggled.connect(self._toggle_batch_docks)

        self.chk_advanced = QtWidgets.QCheckBox("Show advanced panels")
        self.chk_advanced.setChecked(False)
        self.chk_advanced.toggled.connect(self._apply_compact_mode)

        # Output directory
        self.ed_out = QtWidgets.QLineEdit("")
        self.btn_out = QtWidgets.QPushButton("Set…")
        self.btn_out.clicked.connect(self._pick_output_dir)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.ed_out)
        out_row.addWidget(self.btn_out)

        # Load buttons
        self.btn_load_csv = QtWidgets.QPushButton("Load CSV…")
        self.btn_load_pair = QtWidgets.QPushButton("Load odd/even TIFF pair…")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_load_csv.clicked.connect(self._on_load_csv)
        self.btn_load_pair.clicked.connect(self._on_load_pair)
        self.btn_clear.clicked.connect(self._clear_session)

        # Render params
        self.sp_px = QtWidgets.QDoubleSpinBox(); self.sp_px.setRange(0.1, 10000.0); self.sp_px.setDecimals(3); self.sp_px.setValue(10.0); self.sp_px.setSuffix(" nm/px")
        self.sp_sigma_nm = QtWidgets.QDoubleSpinBox(); self.sp_sigma_nm.setRange(0.0, 10000.0); self.sp_sigma_nm.setDecimals(3); self.sp_sigma_nm.setValue(15.0); self.sp_sigma_nm.setSuffix(" nm")
        self.cb_weight = QtWidgets.QComboBox(); self.cb_weight.addItems(["ones", "intensity"])
        self.chk_raw = QtWidgets.QCheckBox("Also render raw histograms (σ=0)")
        self.sp_max_points = QtWidgets.QSpinBox(); self.sp_max_points.setRange(0, 5_000_000); self.sp_max_points.setValue(200_000)

        # Split params
        self.cb_method = QtWidgets.QComboBox(); self.cb_method.addItems(["odd_even", "random_blocks"])
        self.sp_block = QtWidgets.QSpinBox(); self.sp_block.setRange(1, 1_000_000); self.sp_block.setValue(500)
        self.sp_seed = QtWidgets.QSpinBox(); self.sp_seed.setRange(0, 10_000_000); self.sp_seed.setValue(0)

        self.btn_render = QtWidgets.QPushButton("Render SR halves (from CSV)")
        self.btn_render.clicked.connect(self._on_render)

        # ROI controls
        self.btn_roi_layer = QtWidgets.QPushButton("Create/Focus ROI layer")
        self.btn_roi_clear = QtWidgets.QPushButton("Clear ROI shapes")
        self.btn_roi_load = QtWidgets.QPushButton("Load ROI…")
        self.btn_roi_save = QtWidgets.QPushButton("Save ROI…")
        self.chk_show_roi_mask = QtWidgets.QCheckBox("Show ROI mask layer")
        self.sp_apod = QtWidgets.QDoubleSpinBox(); self.sp_apod.setRange(0.0, 50.0); self.sp_apod.setDecimals(2); self.sp_apod.setValue(2.0); self.sp_apod.setSuffix(" px")

        self.btn_roi_layer.clicked.connect(self._on_roi_layer)
        self.btn_roi_clear.clicked.connect(self._on_roi_clear)
        self.btn_roi_load.clicked.connect(self._on_roi_load)
        self.btn_roi_save.clicked.connect(self._on_roi_save)
        self.chk_show_roi_mask.toggled.connect(self._update_roi_mask_layer)

        # FRC curve controls
        self.sp_thr = QtWidgets.QDoubleSpinBox(); self.sp_thr.setRange(0.0, 1.0); self.sp_thr.setDecimals(6); self.sp_thr.setValue(1.0/7.0)
        self.sp_smooth = QtWidgets.QSpinBox(); self.sp_smooth.setRange(1, 99); self.sp_smooth.setValue(5)
        self.sp_nyq = QtWidgets.QDoubleSpinBox(); self.sp_nyq.setRange(0.1, 0.5); self.sp_nyq.setDecimals(3); self.sp_nyq.setValue(0.49)
        self.chk_require_roi = QtWidgets.QCheckBox("Require ROI for FRC")
        self.chk_require_roi.setChecked(True)
        self.btn_frc = QtWidgets.QPushButton("Compute FRC curve")
        self.btn_frc.clicked.connect(self._on_compute_frc)

        self.lbl_res = QtWidgets.QLabel("Resolution: —")
        self.lbl_res.setWordWrap(True)

        self.canvas = FRCMatplotlibCanvas()

        self.btn_export_curve = QtWidgets.QPushButton("Export curve (CSV+PNG)")
        self.btn_export_curve.clicked.connect(self._on_export_curve)

        # Maps controls
        self.sp_win = QtWidgets.QSpinBox(); self.sp_win.setRange(3, 999); self.sp_win.setSingleStep(2); self.sp_win.setValue(21)
        self.sp_sigma_px = QtWidgets.QDoubleSpinBox(); self.sp_sigma_px.setRange(0.0, 100.0); self.sp_sigma_px.setDecimals(3); self.sp_sigma_px.setValue(1.5); self.sp_sigma_px.setSuffix(" px")
        self.chk_auto_sigma = QtWidgets.QCheckBox("Auto σ (maximize global RSP)")
        self.sp_tile = QtWidgets.QSpinBox(); self.sp_tile.setRange(16, 4096); self.sp_tile.setValue(64)
        self.sp_stride = QtWidgets.QSpinBox(); self.sp_stride.setRange(1, 4096); self.sp_stride.setValue(64)
        self.sp_map_thr = QtWidgets.QDoubleSpinBox(); self.sp_map_thr.setRange(0.0, 1.0); self.sp_map_thr.setDecimals(6); self.sp_map_thr.setValue(1.0/7.0)
        self.chk_upsample = QtWidgets.QCheckBox("Upsample tile FRC map to recon size")
        self.chk_upsample.setChecked(True)
        self.chk_apply_roi_to_maps = QtWidgets.QCheckBox("Apply ROI (zero outside) before maps")
        self.chk_apply_roi_to_maps.setChecked(True)
        self.btn_maps = QtWidgets.QPushButton("Compute maps (RSP/RSE + FRC map)")
        self.btn_maps.clicked.connect(self._on_compute_maps)

        # Reference (SQUIRREL-like)
        self.grp_ref = QtWidgets.QGroupBox("Reference comparison (SQUIRREL-like)")
        self.grp_ref.setCheckable(True)
        self.grp_ref.setChecked(False)

        self.btn_ref_load = QtWidgets.QPushButton("Load reference image…")
        self.btn_ref_load.clicked.connect(self._on_load_reference)

        self.chk_ref_register = QtWidgets.QCheckBox("Register SR to reference (phase correlation)")
        self.chk_ref_register.setChecked(True)
        self.sp_ref_ups = QtWidgets.QSpinBox(); self.sp_ref_ups.setRange(1, 64); self.sp_ref_ups.setValue(8)

        self.sp_ref_smin = QtWidgets.QDoubleSpinBox(); self.sp_ref_smin.setRange(0.0, 100.0); self.sp_ref_smin.setDecimals(3); self.sp_ref_smin.setValue(0.0); self.sp_ref_smin.setSuffix(" px")
        self.sp_ref_smax = QtWidgets.QDoubleSpinBox(); self.sp_ref_smax.setRange(0.0, 100.0); self.sp_ref_smax.setDecimals(3); self.sp_ref_smax.setValue(3.0); self.sp_ref_smax.setSuffix(" px")
        self.sp_ref_steps = QtWidgets.QSpinBox(); self.sp_ref_steps.setRange(2, 201); self.sp_ref_steps.setValue(13)
        self.cb_ref_opt = QtWidgets.QComboBox(); self.cb_ref_opt.addItems(["max_rsp", "min_rse"])
        self.sp_ref_win = QtWidgets.QSpinBox(); self.sp_ref_win.setRange(3, 999); self.sp_ref_win.setSingleStep(2); self.sp_ref_win.setValue(21)
        self.btn_ref_run = QtWidgets.QPushButton("Compute ref maps")
        self.btn_ref_run.clicked.connect(self._on_compute_reference_maps)

        ref_form = QtWidgets.QFormLayout(self.grp_ref)
        ref_form.addRow(self.btn_ref_load)
        ref_form.addRow(self.chk_ref_register)
        ref_form.addRow("Registration upsample:", self.sp_ref_ups)
        ref_form.addRow("σ min:", self.sp_ref_smin)
        ref_form.addRow("σ max:", self.sp_ref_smax)
        ref_form.addRow("σ steps:", self.sp_ref_steps)
        ref_form.addRow("Optimize:", self.cb_ref_opt)
        ref_form.addRow("Local window:", self.sp_ref_win)
        ref_form.addRow(self.btn_ref_run)

        # Export all
        self.btn_export_all = QtWidgets.QPushButton("Export ALL outputs (recon/maps/ROI/manifest)")
        self.btn_export_all.clicked.connect(self._on_export_all)

        # Log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)

        # ----------------------- layout -----------------------
        # Build the full panel as a scrollable area (prevents oversized dock widgets on small displays)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        content = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(content)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        lay.addWidget(self.lbl_dataset)

        row_display = QtWidgets.QHBoxLayout()
        row_display.addWidget(self.chk_advanced)
        row_display.addStretch(1)
        row_display.addWidget(self.chk_show_batch)
        lay.addLayout(row_display)

        g_file = QtWidgets.QGroupBox("Dataset")
        f = QtWidgets.QFormLayout(g_file)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btn_load_csv)
        btn_row.addWidget(self.btn_load_pair)
        btn_row.addWidget(self.btn_clear)
        f.addRow(btn_row)
        f.addRow("Output directory:", out_row)
        lay.addWidget(g_file)

        # ---------------- CSV rendering ----------------
        g_r = QtWidgets.QGroupBox("CSV rendering (odd/even reconstructions)")
        fr = QtWidgets.QFormLayout(g_r)

        # Basic controls (always visible)
        fr.addRow("Pixel size:", self.sp_px)
        fr.addRow("Gaussian σ:", self.sp_sigma_nm)
        fr.addRow("Weighting:", self.cb_weight)

        # Advanced controls (hidden when compact mode is enabled)
        self._w_render_adv = QtWidgets.QWidget()
        fr_adv = QtWidgets.QFormLayout(self._w_render_adv)
        fr_adv.setContentsMargins(0, 0, 0, 0)
        fr_adv.addRow(self.chk_raw)
        fr_adv.addRow("Max points to display (0=off):", self.sp_max_points)
        fr_adv.addRow("Split method:", self.cb_method)
        fr_adv.addRow("Block size (frames):", self.sp_block)
        fr_adv.addRow("Random seed:", self.sp_seed)

        fr.addRow(self._w_render_adv)
        fr.addRow(self.btn_render)
        lay.addWidget(g_r)

        # ---------------- ROI ----------------
        g_roi = QtWidgets.QGroupBox("ROI (Napari Shapes polygon)")
        froi = QtWidgets.QFormLayout(g_roi)
        row_roi_btn = QtWidgets.QHBoxLayout()
        row_roi_btn.addWidget(self.btn_roi_layer)
        row_roi_btn.addWidget(self.btn_roi_clear)
        row_roi_btn.addWidget(self.btn_roi_load)
        row_roi_btn.addWidget(self.btn_roi_save)
        froi.addRow(row_roi_btn)
        froi.addRow("Apodization σ:", self.sp_apod)
        froi.addRow(self.chk_show_roi_mask)
        lay.addWidget(g_roi)

        # ---------------- Global FRC ----------------
        g_frc = QtWidgets.QGroupBox("Global FRC curve (Matplotlib)")
        ffrc = QtWidgets.QFormLayout(g_frc)
        ffrc.addRow("Threshold:", self.sp_thr)
        ffrc.addRow("Smoothing bins:", self.sp_smooth)
        ffrc.addRow("Nyquist guard:", self.sp_nyq)
        ffrc.addRow(self.chk_require_roi)
        ffrc.addRow(self.btn_frc)
        ffrc.addRow(self.lbl_res)
        ffrc.addRow(self.canvas)
        ffrc.addRow(self.btn_export_curve)
        lay.addWidget(g_frc)

        # ---------------- Local maps (advanced) ----------------
        g_maps = QtWidgets.QGroupBox("Local maps (odd vs even)")
        self._g_maps = g_maps
        fm = QtWidgets.QFormLayout(g_maps)
        fm.addRow("RSP/RSE window:", self.sp_win)
        fm.addRow("σ (px):", self.sp_sigma_px)
        fm.addRow(self.chk_auto_sigma)
        fm.addRow("FRC tile:", self.sp_tile)
        fm.addRow("FRC stride:", self.sp_stride)
        fm.addRow("FRC threshold:", self.sp_map_thr)
        fm.addRow(self.chk_upsample)
        fm.addRow(self.chk_apply_roi_to_maps)
        fm.addRow(self.btn_maps)
        lay.addWidget(g_maps)

        lay.addWidget(self.grp_ref)
        self._g_ref = self.grp_ref
        lay.addWidget(self.btn_export_all)

        g_log = QtWidgets.QGroupBox("Log")
        self._g_log = g_log
        gl = QtWidgets.QVBoxLayout(g_log)
        gl.addWidget(self.log)
        lay.addWidget(g_log)

        lay.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        # Start in compact mode.

        self._apply_compact_mode(self.chk_advanced.isChecked())

        self._log("Workbench ready.")

    # ----------------------- UI visibility helpers -----------------------

    def _iter_qdockwidgets(self):
        """Yield QDockWidget instances from the Napari main window (best-effort)."""
        try:
            qtwin = self.viewer.window._qt_window
        except Exception:
            return
        for dw in qtwin.findChildren(QtWidgets.QDockWidget):
            yield dw

    def _toggle_batch_docks(self, show: bool):
        """Show/hide optional batch docks to reduce clutter."""
        targets = {"Batch FRC", "CSV → Pairs"}
        for dw in self._iter_qdockwidgets():
            title = dw.windowTitle()
            if title in targets:
                with contextlib.suppress(Exception):
                    dw.setVisible(bool(show))

    def _apply_compact_mode(self, advanced: bool):
        """If advanced=False, hide heavier/less-used panels."""
        # Panels
        for w in (getattr(self, "_g_maps", None), getattr(self, "_g_ref", None), getattr(self, "_g_log", None)):
            if w is not None:
                w.setVisible(bool(advanced))
        # Fine-grained controls inside the render group
        w_adv = getattr(self, "_w_render_adv", None)
        if w_adv is not None:
            w_adv.setVisible(bool(advanced))

    # ----------------------- logging & dialogs -----------------------

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")
        self.log.ensureCursorVisible()

    def _error_box(self, title: str, details: str):
        self._log(f"ERROR: {title}")
        m = QtWidgets.QMessageBox(self)
        m.setIcon(QtWidgets.QMessageBox.Critical)
        m.setWindowTitle(title)
        m.setText(title)
        m.setDetailedText(details)
        m.exec_()

    def _set_busy(self, starting: bool):
        """Centralized busy-state handler."""
        if starting:
            self._busy_count += 1
        else:
            self._busy_count = max(0, self._busy_count - 1)

        busy = self._busy_count > 0
        for w in (
            self.btn_load_csv, self.btn_load_pair, self.btn_clear,
            self.btn_render, self.btn_roi_load, self.btn_roi_save,
            self.btn_frc, self.btn_maps, self.btn_export_curve, self.btn_export_all,
            self.btn_ref_load, self.btn_ref_run,
        ):
            w.setEnabled(not busy)

        self.setCursor(QtCore.Qt.WaitCursor if busy else QtCore.Qt.ArrowCursor)

    # ----------------------- dataset handling -----------------------

    def _pick_output_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output directory")
        if d:
            self.ed_out.setText(d)
            self.session.output_dir = Path(d)
            self._log(f"Output dir set: {d}")

    def _base_stem(self) -> str:
        if self.session.csv_path:
            return self.session.csv_path.stem
        if self.session.odd_tif_path:
            return self.session.odd_tif_path.stem.replace("_odd", "").replace("_even", "")
        return "dataset"

    def _update_dataset_label(self):
        if self.session.input_type == "csv" and self.session.csv_path:
            self.lbl_dataset.setText(f"CSV: {self.session.csv_path}\nPoints: {len(self.session.df) if self.session.df is not None else '?'}")
        elif self.session.input_type == "pair" and self.session.odd_tif_path and self.session.even_tif_path:
            self.lbl_dataset.setText(f"TIFF pair:\n  odd: {self.session.odd_tif_path.name}\n  even: {self.session.even_tif_path.name}")
        else:
            self.lbl_dataset.setText("No dataset loaded.")

    def _clear_session(self):
        self.session.clear()
        self.canvas.update_plot(None)
        self.lbl_res.setText("Resolution: —")
        for name in ("recon_odd", "recon_even", "recon_sum",
                     "raw_odd", "raw_even", "raw_sum",
                     "RSP_map", "RSE_map", "FRC_tile_map", "FRC_map_full",
                     "ROI_mask",
                     "Reference", "SR_degraded", "SQUIRREL_RSP", "SQUIRREL_RSE",
                     "localizations"):
            _remove_layer_if_exists(self.viewer, name)
        _ensure_roi_layer(self.viewer, "ROI")
        self._update_dataset_label()
        self._log("Session cleared.")

    def _on_load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open localization CSV", filter="CSV (*.csv)")
        if not path:
            return
        p = Path(path)
        self._log(f"Loading CSV: {p}")
        try:
            df = load_localizations_csv(p)
        except Exception:
            self._error_box("Failed to load CSV", traceback.format_exc())
            return

        self.session.clear()
        self.session.input_type = "csv"
        self.session.csv_path = p
        self.session.df = df
        self._update_dataset_label()

        # Suggest output directory beside input
        if not self.ed_out.text():
            self.ed_out.setText(str(p.parent))
            self.session.output_dir = p.parent

        self._log(f"Loaded CSV with {len(df)} rows.")
        _ensure_roi_layer(self.viewer, "ROI")

    def _on_load_pair(self):
        odd_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open odd reconstruction TIFF", filter="TIFF (*.tif *.tiff)")
        if not odd_path:
            return
        even_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open even reconstruction TIFF", filter="TIFF (*.tif *.tiff)")
        if not even_path:
            return

        p_odd = Path(odd_path); p_even = Path(even_path)
        self._log(f"Loading TIF pair: {p_odd.name} / {p_even.name}")
        try:
            img_odd = read_tif(p_odd)
            img_even = read_tif(p_even)
        except Exception:
            self._error_box("Failed to read TIFF", traceback.format_exc())
            return
        if img_odd.shape != img_even.shape:
            self._error_box("Shape mismatch", f"{img_odd.shape} vs {img_even.shape}")
            return

        self.session.clear()
        self.session.input_type = "pair"
        self.session.odd_tif_path = p_odd
        self.session.even_tif_path = p_even
        self.session.recon_odd = img_odd
        self.session.recon_even = img_even
        self.session.recon_sum = (img_odd + img_even).astype(np.float32, copy=False)
        self.session.render_meta = {"pixel_size_nm": float(self.sp_px.value())}  # user-controlled

        self._upsert_image("recon_odd", self.session.recon_odd, colormap="inferno", blending="additive", opacity=1.0)
        self._upsert_image("recon_even", self.session.recon_even, colormap="inferno", blending="additive", opacity=0.85)
        self._upsert_image("recon_sum", self.session.recon_sum, colormap="inferno", blending="additive", opacity=0.85)

        # Suggest output directory beside input
        if not self.ed_out.text():
            self.ed_out.setText(str(p_odd.parent))
            self.session.output_dir = p_odd.parent

        self._update_dataset_label()
        _ensure_roi_layer(self.viewer, "ROI")
        self._log("TIFF pair loaded.")

    # ----------------------- napari layer utilities -----------------------

    def _upsert_image(self, name: str, img: np.ndarray, *, colormap: str = "inferno", blending: str = "additive", opacity: float = 1.0):
        if img is None:
            return
        img = np.asarray(img, dtype=np.float32)
        vmin, vmax = _percentile_limits(img)
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if layer.__class__.__name__.lower() == "image":
                layer.data = img
                layer.contrast_limits = (vmin, vmax)
                layer.opacity = opacity
                layer.blending = blending
                return
            else:
                layer.name = f"{name}_old"
        try:
            self.viewer.add_image(
                img,
                name=name,
                colormap=colormap,
                blending=blending,
                opacity=opacity,
                contrast_limits=(vmin, vmax),
            )
        except Exception:
            # Fallback if a colormap name is unavailable in the user's napari build.
            self.viewer.add_image(
                img,
                name=name,
                colormap="gray",
                blending=blending,
                opacity=opacity,
                contrast_limits=(vmin, vmax),
            )

    def _upsert_points(self, name: str, pts_rc: np.ndarray, *, size: float = 1.5, opacity: float = 0.6):
        pts_rc = np.asarray(pts_rc, dtype=float)
        if pts_rc.ndim != 2 or pts_rc.shape[1] != 2:
            return
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if layer.__class__.__name__.lower() == "points":
                layer.data = pts_rc
                layer.size = size
                layer.opacity = opacity
                return
            else:
                layer.name = f"{name}_old"
        self.viewer.add_points(pts_rc, name=name, size=size, opacity=opacity)

    # ----------------------- rendering (threaded) -----------------------

    def _on_render(self):
        if self.session.input_type != "csv" or self.session.df is None:
            self._error_box("No CSV loaded", "Load a CSV file first.")
            return

        px_nm = float(self.sp_px.value())
        sigma_nm = float(self.sp_sigma_nm.value())
        weight = str(self.cb_weight.currentText())
        show_raw = bool(self.chk_raw.isChecked())
        max_pts = int(self.sp_max_points.value())
        method = str(self.cb_method.currentText())
        block = int(self.sp_block.value())
        seed = int(self.sp_seed.value())

        def task():
            dfA, dfB = split_localizations(self.session.df, method=method,
                                           block_size_frames=block, seed=seed)
            imgA, imgB, meta = render_two_images(
                dfA, dfB,
                pixel_size_nm=px_nm,
                gaussian_sigma_nm=sigma_nm,
                weight_mode=weight
            )
            out = {
                "df_odd": dfA, "df_even": dfB,
                "recon_odd": imgA, "recon_even": imgB,
                "recon_sum": (imgA + imgB).astype(np.float32, copy=False),
                "meta": meta,
            }
            if show_raw:
                rawA, rawB, _ = render_two_images(
                    dfA, dfB,
                    pixel_size_nm=px_nm,
                    gaussian_sigma_nm=0.0,
                    weight_mode=weight
                )
                out["raw_odd"] = rawA
                out["raw_even"] = rawB
                out["raw_sum"] = (rawA + rawB).astype(np.float32, copy=False)

            # Optional point mapping to pixel coords (done in background)
            points = None
            if max_pts > 0 and meta and "x_edges_nm" in meta and "y_edges_nm" in meta:
                try:
                    x_edges = np.asarray(meta["x_edges_nm"])
                    y_edges = np.asarray(meta["y_edges_nm"])
                    x0 = float(x_edges[0]); y0 = float(y_edges[0])
                    ix = np.floor((self.session.df["x_nm"].to_numpy(float) - x0) / px_nm).astype(np.int32)
                    iy = np.floor((self.session.df["y_nm"].to_numpy(float) - y0) / px_nm).astype(np.int32)
                    pts = np.stack([iy, ix], axis=1)
                    H, W = imgA.shape
                    m = (pts[:, 0] >= 0) & (pts[:, 0] < H) & (pts[:, 1] >= 0) & (pts[:, 1] < W)
                    pts = pts[m]
                    if pts.shape[0] > max_pts:
                        rng = np.random.default_rng(0)
                        sel = rng.choice(pts.shape[0], size=max_pts, replace=False)
                        pts = pts[sel]
                    points = pts
                except Exception:
                    points = None
            out["points"] = points
            return out

        self._log("Rendering SR halves…")
        self._set_busy(True)
        worker = FunctionWorker(task)
        worker.signals.result.connect(self._on_render_done)
        worker.signals.error.connect(lambda tb: self._error_box("Render failed", tb))
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self.pool.start(worker)

    def _on_render_done(self, out: Dict[str, Any]):
        self.session.df_odd = out.get("df_odd")
        self.session.df_even = out.get("df_even")
        self.session.recon_odd = out.get("recon_odd")
        self.session.recon_even = out.get("recon_even")
        self.session.recon_sum = out.get("recon_sum")
        self.session.render_meta = out.get("meta", {})
        self.session.raw_odd = out.get("raw_odd")
        self.session.raw_even = out.get("raw_even")
        self.session.raw_sum = out.get("raw_sum")

        self._upsert_image("recon_odd", self.session.recon_odd, colormap="inferno", blending="additive", opacity=1.0)
        self._upsert_image("recon_even", self.session.recon_even, colormap="inferno", blending="additive", opacity=0.85)
        self._upsert_image("recon_sum", self.session.recon_sum, colormap="inferno", blending="additive", opacity=0.85)

        if self.session.raw_odd is not None:
            self._upsert_image("raw_odd", self.session.raw_odd, colormap="gray", blending="additive", opacity=0.6)
            self._upsert_image("raw_even", self.session.raw_even, colormap="gray", blending="additive", opacity=0.6)
            self._upsert_image("raw_sum", self.session.raw_sum, colormap="gray", blending="additive", opacity=0.6)

        # Points (optional)
        pts = out.get("points")
        if pts is not None:
            self._upsert_points("localizations", pts, size=1.5, opacity=0.6)

        self._log(f"Rendered images shape: {self.session.recon_even.shape if self.session.recon_even is not None else '?'}")
        self._update_roi_mask_layer()

    # ----------------------- ROI handling -----------------------

    def _on_roi_layer(self):
        layer = _ensure_roi_layer(self.viewer, "ROI")
        self.viewer.layers.selection.active = layer
        self._log("ROI layer ready. Draw a polygon (Shapes) then compute FRC/maps.")

    def _on_roi_clear(self):
        if "ROI" in self.viewer.layers:
            layer = self.viewer.layers["ROI"]
            if layer.__class__.__name__.lower() == "shapes":
                layer.data = []
        self.session.roi_record = None
        self.session.roi_mask = None
        self._update_roi_mask_layer()
        self._log("ROI cleared.")

    def _on_roi_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load ROI JSON", filter="ROI json (*.roi.json);;JSON (*.json)")
        if not path:
            return
        try:
            rec = ROIRecord.from_path(Path(path))
        except Exception:
            self._error_box("Failed to load ROI", traceback.format_exc())
            return

        # If we have a reconstruction loaded, adapt to its shape if needed
        if self.session.recon_even is not None:
            try:
                # manual adaptation: if nm/px differs, scale polygon
                if abs(float(rec.nm_per_px) - float(self.sp_px.value())) > 1e-9:
                    sf = float(self.sp_px.value()) / float(rec.nm_per_px)
                    poly = np.asarray(rec.polygon_px, float) * sf
                    rec = ROIRecord(**{**rec.__dict__, "nm_per_px": float(self.sp_px.value()), "polygon_px": poly.tolist()})
                if tuple(rec.image_shape) != tuple(self.session.recon_even.shape):
                    r0, c0 = rec.image_shape
                    r1, c1 = self.session.recon_even.shape
                    sry, srx = r1 / max(r0, 1), c1 / max(c0, 1)
                    if abs(sry - srx) < 1e-6:
                        poly = np.asarray(rec.polygon_px, float)
                        poly[:, 0] *= sry
                        poly[:, 1] *= srx
                        rec = ROIRecord(**{**rec.__dict__, "image_shape": (int(r1), int(c1)), "polygon_px": poly.tolist()})
                    else:
                        raise ValueError(f"ROI shape mismatch: saved {rec.image_shape}, now {self.session.recon_even.shape}")
            except Exception:
                self._error_box("ROI adaptation failed", traceback.format_exc())
                return

        # Put polygon into Napari layer
        layer = _ensure_roi_layer(self.viewer, "ROI")
        try:
            layer.data = []
            layer.add(np.asarray(rec.polygon_px, float), shape_type="polygon")
        except Exception:
            self._error_box("Failed to set ROI layer", traceback.format_exc())
            return

        self.session.roi_record = rec
        self._log(f"Loaded ROI: {path}")
        self._update_roi_mask_layer()

    def _on_roi_save(self):
        if self.session.recon_even is None:
            self._error_box("No reconstruction loaded", "Load/render recon images first, then save ROI.")
            return
        poly = _get_latest_polygon_from_roi_layer(self.viewer, "ROI")
        if poly is None:
            self._error_box("No ROI polygon", "Draw a polygon in the ROI Shapes layer first.")
            return

        base = self._base_stem()
        nm_tag = f"{int(round(float(self.sp_px.value())))}nm"
        suggested = f"{base}__app__{nm_tag}__{time.strftime('%Y%m%d-%H%M%S')}.roi.json"
        out_dir = self.session.output_dir or (self.session.csv_path.parent if self.session.csv_path else Path.cwd())

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ROI JSON",
                                                        str(Path(out_dir) / suggested),
                                                        filter="ROI json (*.roi.json);;JSON (*.json)")
        if not path:
            return

        src = {"basename": base, "recon_tag": "app", "sha256": ""}
        try:
            if self.session.csv_path:
                src["sha256"] = sha256_small_file(Path(self.session.csv_path))
            elif self.session.odd_tif_path:
                src["sha256"] = sha256_small_file(Path(self.session.odd_tif_path))
        except Exception:
            pass

        rec = ROIRecord(
            version=1,
            created_at=_now_iso(),
            nm_per_px=float(self.sp_px.value()),
            image_shape=tuple(map(int, self.session.recon_even.shape)),
            polygon_px=[[float(r), float(c)] for r, c in np.asarray(poly, float)],
            source=src,
        )
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(rec.to_json())
            mask = mask_from_polygon(self.session.recon_even.shape, np.asarray(poly, float)).astype(np.float32)
            mask_path = Path(path).with_name(Path(path).stem.replace(".roi", "") + "_roi_mask.tif")
            save_tif_float32(mask_path, mask)
        except Exception:
            self._error_box("Failed to save ROI", traceback.format_exc())
            return

        self.session.roi_record = rec
        self._log(f"Saved ROI: {path}")
        self._log(f"Saved ROI mask: {mask_path}")
        self._update_roi_mask_layer()

    def _current_roi_mask(self) -> Optional[np.ndarray]:
        if self.session.recon_even is None:
            return None
        poly = _get_latest_polygon_from_roi_layer(self.viewer, "ROI")
        if poly is None:
            return None
        try:
            return mask_from_polygon(self.session.recon_even.shape, poly)
        except Exception:
            return None

    def _update_roi_mask_layer(self):
        show = bool(self.chk_show_roi_mask.isChecked())
        if not show:
            _remove_layer_if_exists(self.viewer, "ROI_mask")
            return
        m = self._current_roi_mask()
        if m is None:
            _remove_layer_if_exists(self.viewer, "ROI_mask")
            return
        img = m.astype(np.float32)
        if "ROI_mask" in self.viewer.layers:
            layer = self.viewer.layers["ROI_mask"]
            if layer.__class__.__name__.lower() == "image":
                layer.data = img
                layer.opacity = 0.25
                return
            else:
                layer.name = "ROI_mask_old"
        self.viewer.add_image(img, name="ROI_mask", colormap="gray", opacity=0.25, blending="additive", contrast_limits=(0.0, 1.0))

    # ----------------------- FRC curve (threaded) -----------------------

    def _on_compute_frc(self):
        if self.session.recon_even is None or self.session.recon_odd is None:
            self._error_box("No reconstructions", "Load/render recon images first.")
            return

        require_roi = bool(self.chk_require_roi.isChecked())
        roi_mask = self._current_roi_mask()
        if roi_mask is None and require_roi:
            self._error_box("ROI required", "Draw a ROI polygon first (ROI layer), or uncheck 'Require ROI'.")
            return

        px_nm = float(self.sp_px.value())
        thr = float(self.sp_thr.value())
        smooth = int(self.sp_smooth.value())
        nyq = float(self.sp_nyq.value())
        apod = float(self.sp_apod.value())

        def task():
            return compute_frc_plot_data(
                self.session.recon_even,
                self.session.recon_odd,
                pixel_size_nm=px_nm,
                threshold=thr,
                smooth_bins=smooth,
                nyquist_guard=nyq,
                roi_mask=roi_mask,
                apod_sigma_px=apod,
            )

        self._log("Computing FRC curve…")
        self._set_busy(True)
        worker = FunctionWorker(task)
        worker.signals.result.connect(self._on_frc_done)
        worker.signals.error.connect(lambda tb: self._error_box("FRC failed", tb))
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self.pool.start(worker)

    def _on_frc_done(self, plot: FRCPlotData):
        self.session.frc_plot = plot
        self.canvas.update_plot(plot)
        if plot.resolution_nm is not None and np.isfinite(plot.resolution_nm):
            self.lbl_res.setText(f"Resolution: {plot.resolution_nm:.1f} nm\n{plot.notes}")
            self._log(f"FRC resolution: {plot.resolution_nm:.2f} nm")
        else:
            self.lbl_res.setText("Resolution: (no cutoff)\n" + plot.notes)
            self._log("FRC cutoff not found (or too close to Nyquist).")

    def _on_export_curve(self):
        plot = self.session.frc_plot
        if plot is None:
            self._error_box("No curve", "Compute the FRC curve first.")
            return
        out_dir = self.session.output_dir or Path(self.ed_out.text() or Path.cwd())
        ensure_dir(out_dir)
        base = self._base_stem()
        csv_path = out_dir / f"{base}_FRC_curve.csv"
        png_path = out_dir / f"{base}_FRC_curve.png"

        try:
            arr = np.column_stack([plot.freqs_cyc_per_nm, plot.frc])
            header = "freq_cyc_per_nm,frc"
            np.savetxt(str(csv_path), arr, delimiter=",", header=header, comments="")
            self.canvas.save_png(png_path)
        except Exception:
            self._error_box("Export failed", traceback.format_exc())
            return
        self._log(f"Saved curve CSV: {csv_path}")
        self._log(f"Saved curve PNG: {png_path}")

    # ----------------------- Maps (threaded) -----------------------

    def _on_compute_maps(self):
        if self.session.recon_even is None or self.session.recon_odd is None:
            self._error_box("No reconstructions", "Load/render recon images first.")
            return

        win = int(self.sp_win.value())
        sig = float(self.sp_sigma_px.value())
        auto_sig = bool(self.chk_auto_sigma.isChecked())
        tile = int(self.sp_tile.value())
        stride = int(self.sp_stride.value())
        thr = float(self.sp_map_thr.value())
        upsample = bool(self.chk_upsample.isChecked())
        apply_roi = bool(self.chk_apply_roi_to_maps.isChecked())
        px_nm = float(self.sp_px.value())
        apod = float(self.sp_apod.value())

        roi_mask = self._current_roi_mask() if apply_roi else None

        def task():
            odd = self.session.recon_odd.copy()
            even = self.session.recon_even.copy()
            if roi_mask is not None:
                # apodize edges for map stability; outside ROI -> 0
                w = roi_mask.astype(np.float32)
                if apod > 0:
                    w = gaussian_blur(w, float(apod))
                    w /= (w.max() + 1e-12)
                odd *= w
                even *= w

            rsp, rse, g_rsp, used_sigma = rsp_rse_maps(even, odd, window=win, sigma_blur_px=sig, auto_sigma=auto_sig)
            tmap = frc_map(even, odd, tile=tile, stride=stride, threshold=thr, pixel_size_nm=px_nm)
            full = upsample_tile_map_to_image(tmap, even.shape, tile=tile, stride=stride) if upsample else None
            return dict(rsp=rsp, rse=rse, g_rsp=g_rsp, used_sigma=used_sigma, tmap=tmap, full=full)

        self._log("Computing maps…")
        self._set_busy(True)
        worker = FunctionWorker(task)
        worker.signals.result.connect(self._on_maps_done)
        worker.signals.error.connect(lambda tb: self._error_box("Maps failed", tb))
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self.pool.start(worker)

    def _on_maps_done(self, out: Dict[str, Any]):
        self.session.rsp_map = out["rsp"]
        self.session.rse_map = out["rse"]
        self.session.global_rsp = float(out["g_rsp"])
        self.session.frc_tile_map = out["tmap"]
        self.session.frc_full_map = out.get("full")

        self._upsert_image("RSP_map", self.session.rsp_map, colormap="viridis", blending="additive", opacity=0.85)
        self._upsert_image("RSE_map", self.session.rse_map, colormap="magma", blending="additive", opacity=0.85)
        self._upsert_image("FRC_tile_map", self.session.frc_tile_map, colormap="turbo", blending="additive", opacity=0.85)
        if self.session.frc_full_map is not None:
            self._upsert_image("FRC_map_full", self.session.frc_full_map, colormap="turbo", blending="additive", opacity=0.85)

        self._log(f"Global RSP: {self.session.global_rsp:.4f} (σ used: {float(out['used_sigma']):.3f} px)")
        if self.session.frc_tile_map is not None:
            med = float(np.nanmedian(self.session.frc_tile_map))
            self._log(f"Tile-FRC median: {med:.2f} nm")

    # ----------------------- Reference workflow -----------------------

    def _on_load_reference(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open reference image (TIFF)", filter="TIFF (*.tif *.tiff)")
        if not path:
            return
        try:
            ref = read_tif(Path(path))
        except Exception:
            self._error_box("Failed to read reference", traceback.format_exc())
            return
        self.session.ref_path = Path(path)
        self.session.ref_image = ref
        self._upsert_image("Reference", ref, colormap="gray", blending="additive", opacity=0.7)
        self._log(f"Loaded reference: {path} (shape {ref.shape})")

    def _on_compute_reference_maps(self):
        if self.session.recon_sum is None:
            self._error_box("No SR image", "Load/render SR reconstructions first.")
            return
        if self.session.ref_image is None:
            self._error_box("No reference", "Load a reference image first.")
            return
        if self.session.ref_image.shape != self.session.recon_sum.shape:
            self._error_box("Shape mismatch", f"SR {self.session.recon_sum.shape} vs ref {self.session.ref_image.shape}.\nThis simplified mode requires identical shapes.")
            return

        register = bool(self.chk_ref_register.isChecked())
        ups = int(self.sp_ref_ups.value())
        smin = float(self.sp_ref_smin.value())
        smax = float(self.sp_ref_smax.value())
        steps = int(self.sp_ref_steps.value())
        opt = str(self.cb_ref_opt.currentText())
        win = int(self.sp_ref_win.value())

        def task():
            return compute_squirrel_vs_reference(
                self.session.recon_sum,
                self.session.ref_image,
                register=register,
                upsample_factor=ups,
                sigma_min_px=smin,
                sigma_max_px=smax,
                sigma_steps=steps,
                optimize=opt,
                window=win,
            )

        self._log("Computing reference comparison…")
        self._set_busy(True)
        worker = FunctionWorker(task)
        worker.signals.result.connect(self._on_reference_done)
        worker.signals.error.connect(lambda tb: self._error_box("Reference comparison failed", tb))
        worker.signals.finished.connect(lambda: self._set_busy(False))
        self.pool.start(worker)

    def _on_reference_done(self, out: Dict[str, Any]):
        self.session.ref_shift_yx = tuple(out["shift_yx"])
        self.session.rsf_sigma_px = float(out["sigma_px"])
        self.session.rsf_alpha_beta = (float(out["alpha"]), float(out["beta"]))
        self.session.squirrel_global_rsp = float(out["global_rsp"])
        self.session.squirrel_global_rse = float(out["global_rse"])
        self.session.sr_degraded = out["degraded_sr"]
        self.session.squirrel_rsp_map = out["rsp_map"]
        self.session.squirrel_rse_map = out["rse_map"]

        self._upsert_image("SR_degraded", self.session.sr_degraded, colormap="inferno", blending="additive", opacity=0.85)
        self._upsert_image("SQUIRREL_RSP", self.session.squirrel_rsp_map, colormap="viridis", blending="additive", opacity=0.85)
        self._upsert_image("SQUIRREL_RSE", self.session.squirrel_rse_map, colormap="magma", blending="additive", opacity=0.85)

        self._log(f"Ref shift (dy,dx): {self.session.ref_shift_yx}")
        self._log(f"RSF sigma: {self.session.rsf_sigma_px:.3f} px, alpha/beta: {self.session.rsf_alpha_beta}")
        self._log(f"Global RSP: {self.session.squirrel_global_rsp:.4f}, Global RSE: {self.session.squirrel_global_rse:.4f}")

    # ----------------------- Export ALL -----------------------

    def _on_export_all(self):
        if self.session.recon_even is None or self.session.recon_odd is None:
            self._error_box("Nothing to export", "Load/render recon images first.")
            return
        out_dir = self.session.output_dir or Path(self.ed_out.text() or Path.cwd())
        ensure_dir(out_dir)
        base = self._base_stem()
        out = Path(out_dir) / f"{base}__FRC_Workbench"
        ensure_dir(out)

        try:
            # Reconstructions
            save_tif_float32(out / "recon_odd.tif", self.session.recon_odd)
            save_tif_float32(out / "recon_even.tif", self.session.recon_even)
            save_tif_float32(out / "recon_sum.tif", self.session.recon_sum)

            # Maps (if computed)
            if self.session.rsp_map is not None:
                save_tif_float32(out / "RSP_map.tif", self.session.rsp_map)
            if self.session.rse_map is not None:
                save_tif_float32(out / "RSE_map.tif", self.session.rse_map)
            if self.session.frc_tile_map is not None:
                save_tif_float32(out / "FRC_tile_map_nm.tif", self.session.frc_tile_map)
            if self.session.frc_full_map is not None:
                save_tif_float32(out / "FRC_map_full_nm.tif", self.session.frc_full_map)

            # ROI
            poly = _get_latest_polygon_from_roi_layer(self.viewer, "ROI")
            roi_path = None
            mask_path = None
            if poly is not None:
                rec = ROIRecord(
                    version=1,
                    created_at=_now_iso(),
                    nm_per_px=float(self.sp_px.value()),
                    image_shape=tuple(map(int, self.session.recon_even.shape)),
                    polygon_px=[[float(r), float(c)] for r, c in np.asarray(poly, float)],
                    source={"basename": base, "recon_tag": "app", "sha256": ""},
                )
                roi_path = out / "ROI.roi.json"
                with open(roi_path, "w", encoding="utf-8") as f:
                    f.write(rec.to_json())
                mask = mask_from_polygon(self.session.recon_even.shape, np.asarray(poly, float)).astype(np.float32)
                mask_path = out / "ROI_mask.tif"
                save_tif_float32(mask_path, mask)

            # FRC curve (if computed)
            if self.session.frc_plot is not None:
                plot = self.session.frc_plot
                arr = np.column_stack([plot.freqs_cyc_per_nm, plot.frc])
                np.savetxt(str(out / "FRC_curve.csv"), arr, delimiter=",", header="freq_cyc_per_nm,frc", comments="")
                self.canvas.save_png(out / "FRC_curve.png")

            # Reference outputs
            if self.session.ref_image is not None:
                save_tif_float32(out / "Reference.tif", self.session.ref_image)
            if self.session.sr_degraded is not None:
                save_tif_float32(out / "SR_degraded.tif", self.session.sr_degraded)
            if self.session.squirrel_rsp_map is not None:
                save_tif_float32(out / "SQUIRREL_RSP.tif", self.session.squirrel_rsp_map)
            if self.session.squirrel_rse_map is not None:
                save_tif_float32(out / "SQUIRREL_RSE.tif", self.session.squirrel_rse_map)

            # Manifest
            manifest = {
                "created_at": _now_iso(),
                "base": base,
                "input_type": self.session.input_type,
                "csv_path": str(self.session.csv_path) if self.session.csv_path else "",
                "odd_tif": str(self.session.odd_tif_path) if self.session.odd_tif_path else "",
                "even_tif": str(self.session.even_tif_path) if self.session.even_tif_path else "",
                "reference_path": str(self.session.ref_path) if self.session.ref_path else "",
                "pixel_size_nm": float(self.sp_px.value()),
                "render_sigma_nm": float(self.sp_sigma_nm.value()),
                "render_weight_mode": str(self.cb_weight.currentText()),
                "split_method": str(self.cb_method.currentText()),
                "block_size_frames": int(self.sp_block.value()),
                "seed": int(self.sp_seed.value()),
                "frc_threshold": float(self.sp_thr.value()),
                "frc_smooth_bins": int(self.sp_smooth.value()),
                "nyquist_guard": float(self.sp_nyq.value()),
                "roi_apod_sigma_px": float(self.sp_apod.value()),
                "maps_window": int(self.sp_win.value()),
                "maps_sigma_px": float(self.sp_sigma_px.value()),
                "maps_auto_sigma": bool(self.chk_auto_sigma.isChecked()),
                "frc_tile": int(self.sp_tile.value()),
                "frc_stride": int(self.sp_stride.value()),
                "maps_threshold": float(self.sp_map_thr.value()),
                "versions": _collect_versions(),
                "outputs": {
                    "folder": str(out),
                    "roi_json": str(roi_path) if roi_path else "",
                    "roi_mask": str(mask_path) if mask_path else "",
                },
            }
            _json_dump(out / "manifest.json", manifest)

        except Exception:
            self._error_box("Export failed", traceback.format_exc())
            return

        self._log(f"Exported to: {out}")

