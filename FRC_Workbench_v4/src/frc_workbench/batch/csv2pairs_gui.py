# csv2pairs_gui.py
"""
GUI: ThunderSTORM CSV ➜ odd/even (or random blocks) ➜ recon TIFFs ➜ RSP/RSE + FRC maps
- Optional: save half CSVs
- Optional: parallel processing (unchecked by default)
"""

from __future__ import annotations
from pathlib import Path
import fnmatch
import traceback

import pandas as pd

from PyQt5 import QtWidgets, QtCore

from frc_workbench.core.csv2pairs_backend import process_csv_to_pairs, collect_csvs


class Worker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)   # percent, message
    finished = QtCore.pyqtSignal(object)     # summaries list

    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks

    def run(self):
        summaries = []
        n = len(self.tasks)
        for i, (args, kwargs) in enumerate(self.tasks, 1):
            try:
                out = process_csv_to_pairs(*args, **kwargs)
                msg = f"[{i}/{n}] OK  {out.get('input_csv','')}"
                self.progress.emit(int(i / n * 100), msg)
                summaries.append(out)
            except Exception as e:
                msg = f"[{i}/{n}] ERROR {args[0]} → {e}"
                self.progress.emit(int(i / n * 100), msg)
        self.finished.emit(summaries)


class CSV2PairsGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV ➜ Odd/Even ➜ Recon TIFFs ➜ RSP/RSE + FRC maps")
        self.resize(900, 720)

        # Paths
        self.rootEdit = QtWidgets.QLineEdit()
        self.outEdit  = QtWidgets.QLineEdit()

        # Filters
        self.globEdit = QtWidgets.QLineEdit("*customed_title.csv")
        self.includeEdit = QtWidgets.QLineEdit()
        self.excludeEdit = QtWidgets.QLineEdit()

        # Split
        self.methodCombo = QtWidgets.QComboBox(); self.methodCombo.addItems(["odd_even", "random_blocks"])
        self.blockSpin   = QtWidgets.QSpinBox(); self.blockSpin.setRange(1, 1_000_000); self.blockSpin.setValue(500)

        # Recon
        self.pixSpin   = QtWidgets.QDoubleSpinBox(); self.pixSpin.setRange(0.1, 10000.0); self.pixSpin.setDecimals(3); self.pixSpin.setValue(10.0); self.pixSpin.setSuffix(" nm/px")
        self.sigmaSpin = QtWidgets.QDoubleSpinBox(); self.sigmaSpin.setRange(0.0, 10000.0); self.sigmaSpin.setDecimals(3); self.sigmaSpin.setValue(15.0); self.sigmaSpin.setSuffix(" nm")
        self.weightCombo = QtWidgets.QComboBox(); self.weightCombo.addItems(["ones", "intensity"])
        self.reconSuffixEdit = QtWidgets.QLineEdit("_rec")

        # SQUIRREL & FRC map
        self.winSpin = QtWidgets.QSpinBox(); self.winSpin.setRange(3, 999); self.winSpin.setSingleStep(2); self.winSpin.setValue(21)
        self.sigmaPxSpin = QtWidgets.QDoubleSpinBox(); self.sigmaPxSpin.setRange(0.0, 100.0); self.sigmaPxSpin.setDecimals(3); self.sigmaPxSpin.setValue(1.5); self.sigmaPxSpin.setSuffix(" px")
        self.autoSigmaCheck = QtWidgets.QCheckBox("Auto sigma (maximize global RSP)")
        self.tileSpin = QtWidgets.QSpinBox(); self.tileSpin.setRange(8, 4096); self.tileSpin.setValue(64)
        self.strideSpin = QtWidgets.QSpinBox(); self.strideSpin.setRange(1, 4096); self.strideSpin.setValue(64)
        self.thrSpin = QtWidgets.QDoubleSpinBox(); self.thrSpin.setRange(0.0, 1.0); self.thrSpin.setDecimals(6); self.thrSpin.setSingleStep(0.01); self.thrSpin.setValue(1.0/7.0)

        # Save halves + trim
        self.saveHalvesCheck = QtWidgets.QCheckBox("Save odd/even CSV halves")
        self.trimEdit = QtWidgets.QLineEdit("_thund,_thunderstorm")  # tokens to trim from map basename

        # System
        self.parallelCheck = QtWidgets.QCheckBox("Enable parallel processing (experimental)")
        self.workersSpin   = QtWidgets.QSpinBox(); self.workersSpin.setRange(0, 128); self.workersSpin.setValue(0)
        self.seedSpin      = QtWidgets.QSpinBox(); self.seedSpin.setRange(0, 10_000_000); self.seedSpin.setValue(0)
        self.overwriteCheck = QtWidgets.QCheckBox("Overwrite existing outputs")

        # Buttons
        self.btnBrowseRoot = QtWidgets.QPushButton("Browse…"); self.btnBrowseRoot.clicked.connect(self._pick_root)
        self.btnBrowseOut  = QtWidgets.QPushButton("Browse…"); self.btnBrowseOut.clicked.connect(self._pick_out)
        self.btnRun        = QtWidgets.QPushButton("RUN"); self.btnRun.clicked.connect(self._run)
        self.btnCancel     = QtWidgets.QPushButton("Cancel"); self.btnCancel.setEnabled(False)

        # Log
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        self.progress = QtWidgets.QProgressBar(); self.progress.setValue(0)

        # Layouts
        form = QtWidgets.QFormLayout()
        r1 = QtWidgets.QHBoxLayout(); r1.addWidget(self.rootEdit); r1.addWidget(self.btnBrowseRoot)
        r2 = QtWidgets.QHBoxLayout(); r2.addWidget(self.outEdit);  r2.addWidget(self.btnBrowseOut)

        form.addRow("Root folder:", r1)
        form.addRow("Output folder:", r2)
        form.addRow("Glob pattern:", self.globEdit)
        form.addRow("Include keywords (comma):", self.includeEdit)
        form.addRow("Exclude keywords (comma):", self.excludeEdit)

        form.addRow("Split method:", self.methodCombo)
        form.addRow("Block size (frames):", self.blockSpin)

        form.addRow("Pixel size:", self.pixSpin)
        form.addRow("Gaussian sigma:", self.sigmaSpin)
        form.addRow("Weighting:", self.weightCombo)
        form.addRow("Recon suffix:", self.reconSuffixEdit)

        form.addRow("RSP/RSE window:", self.winSpin)
        form.addRow("SQUIRREL σ (px):", self.sigmaPxSpin)
        form.addRow(self.autoSigmaCheck)
        form.addRow("FRC tile:", self.tileSpin)
        form.addRow("FRC stride:", self.strideSpin)
        form.addRow("FRC threshold:", self.thrSpin)

        form.addRow(self.saveHalvesCheck)
        form.addRow("Trim from map basename (comma):", self.trimEdit)

        form.addRow(self.parallelCheck)
        form.addRow("Workers (0=auto):", self.workersSpin)
        form.addRow("Random seed:", self.seedSpin)
        form.addRow(self.overwriteCheck)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(self.btnRun)
        v.addWidget(self.btnCancel)
        v.addWidget(self.progress)
        v.addWidget(self.log)

        self.worker = None

    # ---------- UI helpers ----------

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

    # ---------- Run ----------

    def _run(self):
        self.log.clear()
        root = Path(self.rootEdit.text().strip() or ".").resolve()
        out_root = Path(self.outEdit.text().strip() or (str(root) + "_CSV2PAIRS")).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        pattern = self.globEdit.text().strip() or "*customed_title.csv"
        includes = [s.strip().lower() for s in self.includeEdit.text().split(",") if s.strip()]
        excludes = [s.strip().lower() for s in self.excludeEdit.text().split(",") if s.strip()]

        files = []
        for p in root.rglob("*.csv"):
            if not fnmatch.fnmatch(p.name, pattern):
                continue
            lname = p.name.lower()
            if includes and not all(s in lname for s in includes):
                continue
            if any(s in lname for s in excludes):
                continue
            files.append(p)
        files = sorted(files)
        if not files:
            self._log("No CSV files matched your criteria."); return

        # Gather params
        method = self.methodCombo.currentText()
        block_size = int(self.blockSpin.value())
        px_nm = float(self.pixSpin.value())
        sigma_nm = float(self.sigmaSpin.value())
        weight = self.weightCombo.currentText()
        recon_sfx = self.reconSuffixEdit.text().strip() or "_rec"

        win = int(self.winSpin.value())
        sigma_px = float(self.sigmaPxSpin.value())
        auto_sigma = self.autoSigmaCheck.isChecked()
        tile = int(self.tileSpin.value())
        stride = int(self.strideSpin.value())
        thr = float(self.thrSpin.value())

        save_halves = self.saveHalvesCheck.isChecked()
        trim_tokens = [s.strip() for s in self.trimEdit.text().split(",") if s.strip()]

        workers = int(self.workersSpin.value())
        seed = int(self.seedSpin.value())
        overwrite = self.overwriteCheck.isChecked()

        # Assemble tasks for Worker
        tasks = []
        for f in files:
            out_dir = self._mirror_out_dir(out_root, root, f)
            args = (f, out_dir)
            kwargs = dict(
                method=method,
                block_size_frames=block_size,
                pixel_size_nm=px_nm,
                gaussian_sigma_nm=sigma_nm,
                weight_mode=weight,
                recon_suffix=recon_sfx,
                save_half_csv=save_halves,
                trim_from_map_basename=trim_tokens,
                squirrel_window=win,
                squirrel_sigma_px=sigma_px,
                squirrel_auto_sigma=auto_sigma,
                frc_threshold=thr,
                frc_tile=tile,
                frc_stride=stride,
                seed=seed,
                overwrite=overwrite,
            )
            tasks.append((args, kwargs))

        # Serial in a QThread (keeps UI responsive without introducing multi-proc complexity in GUI)
        self.btnRun.setEnabled(False); self.btnCancel.setEnabled(True)
        self.worker = Worker(tasks)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct); self._log(msg)

    def _on_finished(self, summaries):
        self.btnRun.setEnabled(True); self.btnCancel.setEnabled(False)
        self.progress.setValue(100)

        # Save master batch summary at the output root (deduce from any item)
        if not summaries:
            self._log("No outputs were produced."); return

        out_root = Path(self.outEdit.text().strip() or (str(Path(self.rootEdit.text().strip() or '.').resolve()) + "_CSV2PAIRS")).resolve()
        batch_csv = out_root / "batch_summary.csv"
        try:
            df = pd.DataFrame(summaries)
            df.to_csv(batch_csv, index=False)
            self._log(f"Saved batch summary: {batch_csv} ({len(df)} rows)")
        except Exception as e:
            self._log(f"Failed to write batch summary: {e}")


def main():
    import sys
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    w = CSV2PairsGUI(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
