#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-click launcher for the GUI from a source checkout.

This keeps things simple: it adds ./src to sys.path so you can run without an editable install.
Dependencies (napari, pyqt5, etc.) must still be installed in your environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# IMPORTANT (Windows/Qt): Napari restores window geometry from its global config.
# If that config was saved on a different monitor (e.g. 4K portrait), Qt may emit
# very noisy QWindowsWindow::setGeometry warnings and clamp the window.
# We isolate Napari config to this project folder to guarantee sane geometry.
NAPARI_CONFIG_DIR = HERE / "_napari_config"
NAPARI_CONFIG_DIR.mkdir(exist_ok=True)

# NapariSettings expects a *file* path ending in .yaml or .json (not a directory).
NAPARI_SETTINGS_FILE = NAPARI_CONFIG_DIR / "settings.yaml"
os.environ.setdefault("NAPARI_CONFIG", str(NAPARI_SETTINGS_FILE))
os.environ.setdefault("QT_API", "pyqt5")

from frc_workbench.app import main

if __name__ == "__main__":
    main()
