from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")  # embed in Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from qtpy import QtWidgets

from frc_workbench.core.frc_analysis import FRCPlotData


class FRCMatplotlibCanvas(QtWidgets.QWidget):
    """Matplotlib dock widget dedicated to displaying the FRC curve."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5.0, 3.2), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(111)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

        self._last: Optional[FRCPlotData] = None
        self._render_empty()

    def _render_empty(self):
        self.ax.clear()
        self.ax.set_title("FRC curve")
        self.ax.set_xlabel("Spatial frequency (cycles / nm)")
        self.ax.set_ylabel("FRC")
        self.ax.set_ylim(0.0, 1.0)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update_plot(self, plot: Optional[FRCPlotData]):
        self._last = plot
        if plot is None:
            self._render_empty()
            return

        f = plot.freqs_cyc_per_nm
        y = plot.frc
        thr = float(plot.threshold)

        self.ax.clear()
        self.ax.plot(f, y, label="FRC")
        self.ax.axhline(thr, linestyle="--", label=f"Threshold = {thr:g}")

        if plot.resolution_nm is not None and np.isfinite(plot.resolution_nm):
            f_c = 1.0 / float(plot.resolution_nm)
            self.ax.axvline(f_c, linestyle=":", label=f"Cutoff @ {plot.resolution_nm:.1f} nm")

        self.ax.set_title("FRC curve")
        self.ax.set_xlabel("Spatial frequency (cycles / nm)")
        self.ax.set_ylabel("FRC")
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_xlim(left=0.0)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def save_png(self, path: Path) -> None:
        self.fig.savefig(str(Path(path)), bbox_inches="tight")

    def last_plot(self) -> Optional[FRCPlotData]:
        return self._last
