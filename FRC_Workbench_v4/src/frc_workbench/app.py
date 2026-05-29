from __future__ import annotations

import contextlib
import os

# NOTE: We intentionally import napari lazily inside `main()`.
# This allows launchers (run_workbench.py, console scripts, etc.) to set
# environment variables such as NAPARI_CONFIG / QT_API *before* napari loads.

from frc_workbench.gui.workbench_widget import FRCWorkbenchWidget


def make_main_widget(viewer):
    """Napari plugin entry: return the main dock widget."""
    return FRCWorkbenchWidget(viewer)


def _add_optional_batch_docks(viewer) -> None:
    """Add batch-related docks if their modules are importable."""
    try:
        from frc_workbench.batch.batch_frc_gui import BatchFRCGui
        dock = viewer.window.add_dock_widget(BatchFRCGui(), area="right", name="Batch FRC")
        # Keep the interface uncluttered by default; the user can show these from the main dock.
        with contextlib.suppress(Exception):
            dock.setVisible(False)
    except Exception:
        # Keep the app usable even if optional imports fail.
        pass

    try:
        from frc_workbench.batch.csv2pairs_gui import CSV2PairsGUI
        dock = viewer.window.add_dock_widget(CSV2PairsGUI(), area="right", name="CSV → Pairs")
        with contextlib.suppress(Exception):
            dock.setVisible(False)
    except Exception:
        pass


def main() -> None:
    """Launch the full GUI application (Napari viewer + docks)."""
    # Napari will pick the appropriate Qt backend; for safety we keep Qt API explicit.
    os.environ.setdefault("QT_API", "pyqt5")

    import napari

    viewer = napari.Viewer(title="FRC Workbench")
    viewer.window.add_dock_widget(FRCWorkbenchWidget(viewer), area="right", name="FRC Workbench")

    _add_optional_batch_docks(viewer)

    napari.run()
