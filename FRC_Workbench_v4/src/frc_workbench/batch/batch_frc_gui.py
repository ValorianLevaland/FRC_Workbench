# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 19:55:00 2025

@author: oavar
"""

# batch_frc_gui.py
import sys
import fnmatch
from pathlib import Path
import subprocess, sys  # add near other imports at top if you prefer

import pandas as pd
from PyQt5 import QtWidgets, QtCore

# Main backends
from frc_workbench.core.frc_backend import (
    process_single_csv, find_odd_even_pairs, process_tif_pair
)

# Conversion backend (CSV -> odd/even CSV+TIFFs + maps)
from frc_workbench.core import csv2pairs_backend


class BatchFRCGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Batch FRC Estimator (CSV + odd/even TIF maps)")
        self.setMinimumWidth(880)

        # Root/out
        self.rootEdit = QtWidgets.QLineEdit()
        self.outEdit = QtWidgets.QLineEdit()

        # Modes
        self.csvCheck = QtWidgets.QCheckBox("Process CSV files")
        self.csvCheck.setChecked(True)
        self.pairCheck = QtWidgets.QCheckBox("Process odd/even TIF pairs")
        self.pairCheck.setChecked(True)

        # --- CSV params ---
        self.globEdit = QtWidgets.QLineEdit("*customed_title.csv")
        self.includeEdit = QtWidgets.QLineEdit()
        self.excludeEdit = QtWidgets.QLineEdit()
        self.methodCombo = QtWidgets.QComboBox(); self.methodCombo.addItems(["odd_even", "random_blocks"])
        self.blockSpin = QtWidgets.QSpinBox(); self.blockSpin.setRange(1, 1_000_000); self.blockSpin.setValue(500)
        self.pixSpin = QtWidgets.QDoubleSpinBox(); self.pixSpin.setRange(0.1, 10_000.0); self.pixSpin.setValue(10.0); self.pixSpin.setSuffix(" nm/px")
        self.sigmaSpin = QtWidgets.QDoubleSpinBox(); self.sigmaSpin.setRange(0.0, 10_000.0); self.sigmaSpin.setValue(15.0); self.sigmaSpin.setSuffix(" nm")
        self.weightCombo = QtWidgets.QComboBox(); self.weightCombo.addItems(["ones", "intensity"])
        self.thresholdSpin = QtWidgets.QDoubleSpinBox(); self.thresholdSpin.setRange(0.0, 1.0); self.thresholdSpin.setDecimals(6); self.thresholdSpin.setSingleStep(0.01); self.thresholdSpin.setValue(1.0/7.0)
        self.seedSpin = QtWidgets.QSpinBox(); self.seedSpin.setRange(0, 10_000_000); self.seedSpin.setValue(0)

        # **NEW** CSV workflow selector
        self.csvWorkflow = QtWidgets.QComboBox()
        self.csvWorkflow.addItems([
            "Direct analysis (no TIFFs)",
            "Convert to odd/even TIFFs then analyze"
        ])

        # --- Pair params (used in pair mode and in Convert→Analyze) ---
        self.winSpin = QtWidgets.QSpinBox(); self.winSpin.setRange(3, 999); self.winSpin.setSingleStep(2); self.winSpin.setValue(21)
        self.sigmaPxSpin = QtWidgets.QDoubleSpinBox(); self.sigmaPxSpin.setRange(0.0, 100.0); self.sigmaPxSpin.setDecimals(3); self.sigmaPxSpin.setValue(1.5); self.sigmaPxSpin.setSuffix(" px")
        self.autoSigmaCheck = QtWidgets.QCheckBox("Auto sigma (maximize global RSP)")
        self.tileSpin = QtWidgets.QSpinBox(); self.tileSpin.setRange(16, 4096); self.tileSpin.setValue(64)
        self.strideSpin = QtWidgets.QSpinBox(); self.strideSpin.setRange(1, 4096); self.strideSpin.setValue(64)
        self.pairThrSpin = QtWidgets.QDoubleSpinBox(); self.pairThrSpin.setRange(0.0, 1.0); self.pairThrSpin.setDecimals(6); self.pairThrSpin.setSingleStep(0.01); self.pairThrSpin.setValue(1.0/7.0)

        # ROI options (simple)
        self.roiOpts = ROIOptionsWidget()

        # System
        self.workersSpin = QtWidgets.QSpinBox(); self.workersSpin.setRange(0, 128); self.workersSpin.setValue(0)
        self.overwriteCheck = QtWidgets.QCheckBox("Overwrite existing outputs")

        # Buttons
        browseRootBtn = QtWidgets.QPushButton("Browse…"); browseRootBtn.clicked.connect(self._pick_root)
        browseOutBtn = QtWidgets.QPushButton("Browse…"); browseOutBtn.clicked.connect(self._pick_out)
        self.runBtn = QtWidgets.QPushButton("RUN"); self.runBtn.clicked.connect(self._run_batch)

        # Log pane
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)

        # Layout (form)
        form = QtWidgets.QFormLayout()

        # Root / Out
        rootRow = QtWidgets.QHBoxLayout(); rootRow.addWidget(self.rootEdit); rootRow.addWidget(browseRootBtn)
        outRow = QtWidgets.QHBoxLayout(); outRow.addWidget(self.outEdit); outRow.addWidget(browseOutBtn)
        form.addRow("Root folder:", rootRow)
        form.addRow("Output folder:", outRow)

        # Modes
        form.addRow(self.csvCheck); form.addRow(self.pairCheck)

        # CSV group
        csvGroup = QtWidgets.QGroupBox("CSV → Overall FRC")
        csvForm = QtWidgets.QFormLayout(csvGroup)
        csvForm.addRow("CSV workflow:", self.csvWorkflow)
        csvForm.addRow("Glob pattern:", self.globEdit)
        csvForm.addRow("Include keywords (comma):", self.includeEdit)
        csvForm.addRow("Exclude keywords (comma):", self.excludeEdit)
        csvForm.addRow("Method:", self.methodCombo)
        csvForm.addRow("Block size (frames):", self.blockSpin)
        csvForm.addRow("Pixel size:", self.pixSpin)
        csvForm.addRow("Gaussian sigma:", self.sigmaSpin)
        csvForm.addRow("Weighting:", self.weightCombo)
        csvForm.addRow("FRC threshold:", self.thresholdSpin)
        csvForm.addRow("Random seed:", self.seedSpin)

        # Pair group
        pairGroup = QtWidgets.QGroupBox("Odd/Even TIF → RSP/RSE/FRC maps")
        pairForm = QtWidgets.QFormLayout(pairGroup)
        pairForm.addRow("RSP/RSE window (odd):", self.winSpin)
        pairForm.addRow("SQUIRREL sigma:", self.sigmaPxSpin)
        pairForm.addRow(self.autoSigmaCheck)
        pairForm.addRow("FRC tile:", self.tileSpin)
        pairForm.addRow("FRC stride:", self.strideSpin)
        pairForm.addRow("FRC threshold:", self.pairThrSpin)
        self.exportUpsampled = QtWidgets.QCheckBox("Export upsampled FRC map (image-size)")
        self.exportUpsampled.setChecked(True)
        pairForm.addRow(self.exportUpsampled)

        # --- add under the Pair group form (pairForm) ---
        self.postProcCheck = QtWidgets.QCheckBox("Post-process FRC maps (nm overlay / stats)")
        self.postProcCheck.setChecked(True)

        self.overlayCheck = QtWidgets.QCheckBox("Make overlay PNG");
        self.overlayCheck.setChecked(True)
        self.histCheck = QtWidgets.QCheckBox("Save histogram");
        self.histCheck.setChecked(True)
        self.u16Check = QtWidgets.QCheckBox("Also save uint16 view (for fast browsing)");
        self.u16Check.setChecked(False)

        pairForm.addRow(self.postProcCheck)
        pairForm.addRow(self.overlayCheck)
        pairForm.addRow(self.histCheck)
        pairForm.addRow(self.u16Check)

        # ROI group
        roiGroup = QtWidgets.QGroupBox("ROI options")
        roiLay = QtWidgets.QVBoxLayout(roiGroup)
        roiLay.addWidget(self.roiOpts)

        # System group
        sysGroup = QtWidgets.QGroupBox("System")
        sysForm = QtWidgets.QFormLayout(sysGroup)
        sysForm.addRow("Workers (0=auto):", self.workersSpin)
        sysForm.addRow(self.overwriteCheck)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(csvGroup)
        v.addWidget(pairGroup)
        v.addWidget(roiGroup)
        v.addWidget(sysGroup)
        v.addWidget(self.runBtn)
        v.addWidget(self.log)

        # Dynamic enabling based on workflow + toggles
        def _update_groups():
            do_csv = self.csvCheck.isChecked()
            do_pairs = self.pairCheck.isChecked()
            w_idx = self.csvWorkflow.currentIndex()  # 0=Direct, 1=Convert->Analyze

            # CSV group: active when CSV mode is on
            csvGroup.setEnabled(do_csv)

            # Pair group: active when TIF-pair mode is on, or when CSV mode is on *and* workflow is Convert->Analyze.
            pairGroup.setEnabled(do_pairs or (do_csv and w_idx == 1))

            # ROI options: relevant when either analysis path may trigger ROI
            roiGroup.setEnabled(do_pairs or (do_csv and (w_idx in (0, 1))))

        self.csvCheck.toggled.connect(_update_groups)
        self.pairCheck.toggled.connect(_update_groups)
        self.csvWorkflow.currentIndexChanged.connect(_update_groups)
        _update_groups()

    # --- helpers ---
    def _pick_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select root folder")
        if d: self.rootEdit.setText(d)

    def _pick_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
        if d: self.outEdit.setText(d)

    def _log(self, msg: str):
        self.log.appendPlainText(msg); self.log.ensureCursorVisible()

    def _mirror_out_dir(self, out_root: Path, root: Path, fpath: Path) -> Path:
        rel = fpath.parent.relative_to(root)
        out_dir = out_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # --- main run ---
    def _run_batch(self):
        root = Path(self.rootEdit.text().strip() or ".").resolve()
        out_root = Path(self.outEdit.text().strip() or (str(root) + "_FRC")).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        # CSV params
        do_csv = self.csvCheck.isChecked()
        workflow_idx = self.csvWorkflow.currentIndex()  # 0=Direct, 1=Convert->Analyze
        pattern = self.globEdit.text().strip() or "*customed_title.csv"
        includes = [s.strip().lower() for s in self.includeEdit.text().split(",") if s.strip()]
        excludes = [s.strip().lower() for s in self.excludeEdit.text().split(",") if s.strip()]
        method = self.methodCombo.currentText()
        block_size = int(self.blockSpin.value())
        pixel_size_nm = float(self.pixSpin.value())
        sigma_nm = float(self.sigmaSpin.value())
        weight_mode = self.weightCombo.currentText()
        threshold = float(self.thresholdSpin.value())
        seed = int(self.seedSpin.value())

        # Pair params
        do_pairs = self.pairCheck.isChecked()
        win = int(self.winSpin.value())
        sigma_px = float(self.sigmaPxSpin.value())
        auto_sigma = self.autoSigmaCheck.isChecked()
        tile = int(self.tileSpin.value())
        stride = int(self.strideSpin.value())
        pair_thr = float(self.pairThrSpin.value())

        # ROI params
        roi_dir_txt = self.roiOpts.ed_dir.text().strip()
        roi_dir = Path(roi_dir_txt) if roi_dir_txt else None
        reuse_roi = self.roiOpts.chk_reuse.isChecked()
        min_roi_pixels = int(self.roiOpts.spin_minpix.value())
        # (Manual ROI toggle is advisory; backend will still prompt if it needs a ROI.)

        # System
        workers = int(self.workersSpin.value())
        overwrite = self.overwriteCheck.isChecked()

        # Collect files
        csv_files = []
        if do_csv:
            for p in root.rglob("*.csv"):
                if not fnmatch.fnmatch(p.name, pattern):
                    continue
                lname = p.name.lower()
                if includes and not all(s in lname for s in includes):
                    continue
                if any(s in lname for s in excludes):
                    continue
                csv_files.append(p)
            if not csv_files:
                self._log("No CSV files matched your criteria.")

        pairs = []
        if do_pairs:
            pairs = find_odd_even_pairs(root)
            if not pairs:
                self._log("No odd/even TIF pairs found. (Expect *_odd.tif / *_even.tif in same folder)")

        summaries = []

        # -------- CSV branch --------
        if do_csv and csv_files:
            if workflow_idx == 0:
                # Direct CSV analysis (no TIFFs)
                self._log(f"CSV: {len(csv_files)} files (direct analysis)")
                for i, f in enumerate(csv_files, 1):
                    out_dir = self._mirror_out_dir(out_root, root, f)
                    self._log(f"[CSV {i}/{len(csv_files)}] {f}")
                    try:
                        # Try with ROI kwargs (new v1); if not supported, fallback
                        try:
                            summary = process_single_csv(
                                f, out_dir,
                                method=method,
                                block_size_frames=block_size,
                                pixel_size_nm=pixel_size_nm,
                                gaussian_sigma_nm=sigma_nm,
                                weight_mode=weight_mode,
                                threshold=threshold,
                                seed=seed,
                                overwrite=overwrite,
                                roi_dir=roi_dir,
                                reuse_roi=reuse_roi,
                                min_roi_pixels=min_roi_pixels,
                                ui_mode="gui",
                                export_roi_mask_tif=True,
                            )
                        except TypeError:
                            summary = process_single_csv(
                                f, out_dir,
                                method=method,
                                block_size_frames=block_size,
                                pixel_size_nm=pixel_size_nm,
                                gaussian_sigma_nm=sigma_nm,
                                weight_mode=weight_mode,
                                threshold=threshold,
                                seed=seed,
                                overwrite=overwrite,
                            )
                        summaries.append(summary)
                    except Exception as e:
                        self._log(f"ERROR: {e}")

            else:
                # Convert to odd/even TIFFs then analyze those pairs
                self._log(f"CSV: {len(csv_files)} files (convert to pairs → analyze)")
                for i, f in enumerate(csv_files, 1):
                    out_dir = self._mirror_out_dir(out_root, root, f)
                    self._log(f"[CSV→PAIRS {i}/{len(csv_files)}] {f}")
                    try:
                        conv_summary = csv2pairs_backend.process_csv_to_pairs(
                            in_csv=f, out_dir=out_dir,
                            method=method,
                            block_size_frames=block_size,
                            pixel_size_nm=pixel_size_nm,
                            gaussian_sigma_nm=sigma_nm,
                            weight_mode=weight_mode,
                            recon_suffix="_rec",
                            save_half_csv=True,
                            trim_from_map_basename=None,
                            squirrel_window=win,
                            squirrel_sigma_px=sigma_px,
                            squirrel_auto_sigma=auto_sigma,
                            frc_threshold=pair_thr,
                            frc_tile=tile,
                            frc_stride=stride,
                            seed=seed,
                            overwrite=overwrite,
                        )
                        summaries.append(conv_summary)
                        odd_tif = Path(conv_summary["odd_tif"])
                        even_tif = Path(conv_summary["even_tif"])
                        self._log(f"  -> Generated pairs: {odd_tif.name} / {even_tif.name}")

                        # Now analyze those pairs (ROI-enabled)
                        self._log(f"[ANALYZE] {odd_tif.name} & {even_tif.name}")
                        try:
                            pair_summary = process_tif_pair(
                                odd_tif, even_tif, out_dir,
                                pixel_size_nm=pixel_size_nm,
                                squirrel_window=win,
                                squirrel_sigma_px=sigma_px,
                                squirrel_auto_sigma=auto_sigma,
                                frc_threshold=pair_thr,
                                frc_tile=tile,
                                frc_stride=stride,
                                overwrite=overwrite,
                                roi_dir=roi_dir,
                                reuse_roi=reuse_roi,
                                min_roi_pixels=min_roi_pixels,
                                ui_mode="gui",
                            )
                        except TypeError:
                            # Fallback if your current v1 doesn't yet accept ROI kwargs
                            pair_summary = process_tif_pair(
                                odd_tif, even_tif, out_dir,
                                pixel_size_nm=pixel_size_nm,
                                squirrel_window=win,
                                squirrel_sigma_px=sigma_px,
                                squirrel_auto_sigma=auto_sigma,
                                frc_threshold=pair_thr,
                                frc_tile=tile,
                                frc_stride=stride,
                                overwrite=overwrite,
                            )
                        summaries.append(pair_summary)
                        self._run_postproc(pair_summary)
                    except Exception as e:
                        self._log(f"ERROR: {e}")

        # -------- Existing TIF pairs branch --------
        if do_pairs and pairs:
            self._log(f"TIF pairs: {len(pairs)}")
            for i, (odd_path, even_path, base) in enumerate(pairs, 1):
                out_dir = self._mirror_out_dir(out_root, root, odd_path)
                self._log(f"[PAIR {i}/{len(pairs)}] {odd_path.name}  &  {even_path.name}")
                try:
                    try:
                        summary = process_tif_pair(
                            odd_path, even_path, out_dir,
                            pixel_size_nm=pixel_size_nm,
                            squirrel_window=win,
                            squirrel_sigma_px=sigma_px,
                            squirrel_auto_sigma=auto_sigma,
                            frc_threshold=pair_thr,
                            frc_tile=tile,
                            frc_stride=stride,
                            overwrite=overwrite,
                            roi_dir=roi_dir,
                            reuse_roi=reuse_roi,
                            min_roi_pixels=min_roi_pixels,
                            ui_mode="gui",
                        )
                    except TypeError:
                        summary = process_tif_pair(
                            odd_path, even_path, out_dir,
                            pixel_size_nm=pixel_size_nm,
                            squirrel_window=win,
                            squirrel_sigma_px=sigma_px,
                            squirrel_auto_sigma=auto_sigma,
                            frc_threshold=pair_thr,
                            frc_tile=tile,
                            frc_stride=stride,
                            overwrite=overwrite,
                        )

                    summaries.append(summary)
                    self._run_postproc(summary)
                except Exception as e:
                    self._log(f"ERROR: {e}")

        # -------- Save batch summary --------
        if summaries:
            df_sum = pd.DataFrame(summaries)
            batch_csv = out_root / "batch_summary.csv"
            df_sum.to_csv(batch_csv, index=False)
            self._log(f"Saved batch summary: {batch_csv} ({len(df_sum)} rows)")

        self._log("Done.")

    def _run_postproc(self, summary_dict):
        if not self.postProcCheck.isChecked():
            return
        from pathlib import Path  # Path is already imported at top; this keeps it local if needed

        frc_map = summary_dict.get("frc_map")  # tile grid or (if you added it) image-sized
        odd_tif = summary_dict.get("odd_tif")
        if not frc_map or not odd_tif:
            return  # nothing to do

        cmd = [
            sys.executable,
            "-m", "frc_workbench.tools.frc_map_nm_post",
            "--frc-map", frc_map,
            "--is-tile-grid",  # because our saved map is the tile grid
            "--tile", str(self.tileSpin.value()),
            "--stride", str(self.strideSpin.value()),
            "--recon", odd_tif,  # align to recon; odd is fine
        ]
        if self.overlayCheck.isChecked():
            cmd.append("--make-overlay")
        if self.histCheck.isChecked():
            cmd.append("--hist")
        if self.u16Check.isChecked():
            cmd.append("--save-uint16")

        # thresholds for summary CSV (optional)
        # cmd.extend(["--thresholds", "25", "35", "50"])

        try:
            self._log(f"[POST] nm-post: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except Exception as e:
            self._log(f"[POST] ERROR: {e}")


# Minimal ROI options widget (PyQt5)
from qtpy.QtWidgets import (  # type: ignore
    QWidget, QFormLayout, QCheckBox, QSpinBox, QLineEdit, QPushButton, QFileDialog, QHBoxLayout
)

class ROIOptionsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chk_manual = QCheckBox("Manual ROI per image")
        self.chk_manual.setChecked(True)
        self.chk_reuse = QCheckBox("Reuse saved ROI if found")
        self.chk_reuse.setChecked(True)

        self.spin_minpix = QSpinBox()
        self.spin_minpix.setRange(50, 10_000)
        self.spin_minpix.setValue(400)

        self.ed_dir = QLineEdit("")
        btn_browse = QPushButton("Browse…")
        def _browse():
            d = QFileDialog.getExistingDirectory(self, "Choose ROI directory")
            if d: self.ed_dir.setText(d)
        btn_browse.clicked.connect(_browse)
        h = QHBoxLayout(); h.addWidget(self.ed_dir); h.addWidget(btn_browse)

        lay = QFormLayout(self)
        lay.addRow(self.chk_manual)
        lay.addRow(self.chk_reuse)
        lay.addRow("Min ROI pixels:", self.spin_minpix)
        lay.addRow("ROI directory:", h)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = BatchFRCGui()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
