#!/usr/bin/env python3
"""
FRC_Batch_main.py
==================
A robust entry-point / dispatcher for the FRAC_Batch (FRC analysis) project.

Goals
-----
- Provide a single command to run the GUI or CLI backends that ship in this repo:
    * batch_frc_gui.py
    * batch_frc_cli.py
    * csv2pairs_gui.py / csv2pairs_backend.py
- Be resilient to different environments (Spyder, PyCharm, terminal) and Qt setups.
- Centralize logging setup with safe fallbacks.

Usage examples
--------------
- Auto-select GUI (if possible) or else fallback to CLI:
    python FRC_Batch_main.py

- Force GUI:
    python FRC_Batch_main.py --mode gui

- Force CLI, passing *all* remaining args directly to the CLI module:
    python FRC_Batch_main.py --mode cli -- --input /path/to/data --workers 8

- Run the CSV→pairs helper (GUI if Qt available, else backend only):
    python FRC_Batch_main.py --mode csv2pairs -- --input my.csv

- Just print versions and exit:
    python FRC_Batch_main.py --check --versions

Notes
-----
- Anything after a lone "--" is forwarded to the chosen submodule as-is.
- If the submodule calls sys.exit(), we propagate its exit code instead of crashing.
- If you run inside Spyder, we try hard to avoid double-creating a QApplication.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import runpy
import sys
import time
import traceback
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Prefer PyQt5; allow fallback to PySide6 via QtPy abstraction if downstream uses QtPy.
os.environ.setdefault("QT_API", "pyqt5")

# --------------- Logging helpers ---------------

def _default_log_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(f"frc_batch_{ts}.log")

def _setup_logging(level: str = "INFO", logfile: str | None = None) -> logging.Logger:
    """Configure a reasonable logging pipeline.
    Tries project-specific logger first (batch_frc_log), else falls back to stdlib logging.
    """
    logger = logging.getLogger("FRC_BATCH")
    if logger.handlers:
        # Already configured (e.g., called twice or running under Spyder)
        return logger

    # Attempt to use user-provided logging module if it exists.
    with contextlib.suppress(Exception):
        import batch_frc_log  # type: ignore

        if hasattr(batch_frc_log, "setup_logging"):
            return batch_frc_log.setup_logging(level=level, logfile=logfile)

    # Fallback: stdlib logging with both console and rotating file handler.
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if logfile is None:
        logfile = _default_log_path()
    fh = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.debug("Logging initialized; logfile=%s", logfile)
    return logger

# --------------- Environment checks ---------------

def _collect_versions() -> dict[str, str]:
    info: dict[str, str] = {}
    def put(name: str, module_name: str | None = None):
        modname = module_name or name
        try:
            mod = __import__(modname)
            ver = getattr(mod, "__version__", "unknown")
            info[name] = str(ver)
        except Exception:
            info[name] = "<not installed>"
    put("python", "sys")
    info["python"] = sys.version.split()[0]
    for name in ["numpy", "scipy", "pandas", "matplotlib", "tifffile",
                 "imageio", "PIL", "psutil", "numba", "pyfftw", "qtpy"]:
        put(name)
    # Try to report Qt binding details if present
    try:
        import qtpy  # type: ignore
        info["qtpy_api"] = getattr(qtpy, "API_NAME", "unknown")
        info["qtpy_qt_version"] = getattr(qtpy, "QT_VERSION", "unknown")
    except Exception:
        pass
    for qt_api in ("PyQt5", "PySide6"):
        with contextlib.suppress(Exception):
            m = __import__(qt_api)
            info[qt_api] = getattr(m, "__version__", "unknown")
    return info

def _has_display() -> bool:
    # Conservative: Windows usually OK; on POSIX check DISPLAY/WAYLAND.
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

# --------------- Submodule runners ---------------

class SubmoduleResult(Exception):
    """Internal control-flow to propagate submodule exit codes cleanly."""
    def __init__(self, code: int = 0):
        self.code = code
        super().__init__(f"Submodule exited with code {code}")

def _run_module_as_main(modname: str, forwarded_argv: list[str]) -> int:
    """Run a Python module as if invoked via "python -m {modname} ...".
    We temporarily replace sys.argv so the target module can parse its own arguments.
    Any SystemExit is captured and returned as an exit code.
    """
    old_argv = sys.argv[:]
    try:
        sys.argv = [modname] + forwarded_argv
        runpy.run_module(modname, run_name="__main__")
        return 0
    except SystemExit as e:
        # Submodule chose to exit normally with a code.
        code = int(getattr(e, "code", 0) or 0)
        return code
    finally:
        sys.argv = old_argv

def _try_run_first(logger: logging.Logger, names: list[str], forwarded_argv: list[str]) -> int:
    errors = []
    for name in names:
        try:
            __import__(name)
        except Exception as e:
            errors.append(f"{name}: import failed ({e})")
            continue
        logger.info("Dispatching to submodule: %s", name)
        return _run_module_as_main(name, forwarded_argv)
    if errors:
        for e in errors:
            logger.error(e)
    return 2  # No module found

def run_gui(logger: logging.Logger, forwarded_argv: list[str]) -> int:
    # Try GUI frontends in order of preference
    candidates = ["batch_frc_gui", "csv2pairs_gui"]
    return _try_run_first(logger, candidates, forwarded_argv)

def run_cli(logger: logging.Logger, forwarded_argv: list[str]) -> int:
    candidates = ["batch_frc_cli"]
    # If there's a csv2pairs backend-only mode
    if forwarded_argv and forwarded_argv[0].lower() == "csv2pairs":
        candidates = ["csv2pairs_backend"] + candidates
        forwarded_argv = forwarded_argv[1:]
    return _try_run_first(logger, candidates, forwarded_argv)

# --------------- Main argument parser ---------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="FRC_Batch_main",
        description="Dispatcher for FRAC_Batch (FRC analysis) GUI/CLI and helpers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "gui", "cli", "csv2pairs"],
        default="auto",
        help="Which frontend to launch. 'csv2pairs' prefers the GUI helper if available.",
    )
    parser.add_argument(
        "--loglevel", default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Console/file logging verbosity.",
    )
    parser.add_argument(
        "--logfile", default=None,
        help="Optional path to a log file. If omitted, a timestamped log is created in CWD.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Perform environment checks (versions, display availability) and exit.",
    )
    parser.add_argument(
        "--versions", action="store_true",
        help="Print library versions as a diagnostic step and exit (implies --check)."
    )
    parser.add_argument(
        "--no-gui-fallback", action="store_true",
        help="If GUI launch fails under --mode auto, do NOT fallback to CLI (return nonzero)."
    )
    parser.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to the selected submodule verbatim.",
    )
    ns = parser.parse_args(argv)
    # Strip leading "--" if present in remainder
    if ns.rest and ns.rest[0] == "--":
        ns.rest = ns.rest[1:]
    return ns

# --------------- Program entry ---------------

def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    logger = _setup_logging(level=ns.loglevel, logfile=ns.logfile)

    if ns.check or ns.versions:
        info = _collect_versions()
        logger.info("Environment diagnostics:")
        width = max(len(k) for k in info) + 2
        for k in sorted(info):
            logger.info(f"  {k:<{width}}{info[k]}")
        logger.info("Display available: %s", _has_display())
        return 0

    logger.info("FRC_Batch_main starting with mode=%s", ns.mode)

    try:
        if ns.mode == "gui" or (ns.mode == "auto" and _has_display()):
            code = run_gui(logger, ns.rest or [])
            if code == 0 or ns.mode == "gui" or ns.no_gui_fallback:
                return code
            logger.warning("GUI dispatch failed; falling back to CLI.")
            return run_cli(logger, ns.rest or [])
        elif ns.mode == "csv2pairs":
            # Prefer GUI helper if present
            code = _try_run_first(logger, ["csv2pairs_gui", "csv2pairs_backend"], ns.rest or [])
            return code
        else:
            # CLI
            return run_cli(logger, ns.rest or [])
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Exiting.")
        return 130
    except Exception as e:
        logger.error("Unhandled exception: %s", e)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
