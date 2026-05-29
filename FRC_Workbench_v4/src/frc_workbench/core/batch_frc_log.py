# batch_frc_log.py (package module)
"""
Lightweight structured logging for the FRAC_Batch pipeline.

Features
--------
- Text log (.log) + structured JSON Lines (.jsonl) per run.
- Run-level and file-level events: params, decisions, metrics, warnings, errors, artifacts.
- Context managers for file scope and timers, so you don't forget to close a file's log.
- Exception capture (with traceback) without crashing your app.
- Safe serialization of Paths, numpy scalars, etc.

Usage (quick)
-------------
from pathlib import Path
from batch_frc_log import BatchLogger

# 1) Create a logger once per "batch run"
blog = BatchLogger(log_dir=Path(out_root) / "_logs")         # creates a run id automatically
blog.log_run_start({"do_csv": True, "do_pairs": False, "user": "..."})

# 2) Log per-file with a context manager
with blog.file_ctx(in_csv, mode="csv") as L:
    L.params(method=method, block_size_frames=block, pixel_size_nm=pix_nm, gaussian_sigma_nm=sigma_nm,
             weight_mode=weight, threshold=thr, seed=int(seed), ui_mode="gui",
             roi_dir=str(roi_dir) if roi_dir else None, reuse_roi=bool(reuse_roi), min_roi_pixels=int(min_roi_pixels))
    # ... do work ...
    L.decision("roi_reused", path=str(roi_path))             # examples
    L.metric("frc_resolution_nm", res_nm, unit="nm")
    L.add_output(curve_csv=curve_csv, value_csv=value_csv, plot_png=plot_png,
                 roi_json=roi_path, roi_mask_tif=roi_mask_tif)
    L.set_status("OK")  # or "SKIPPED"/"ERROR"

blog.log_run_end({"n_files": N, "n_ok": ok, "n_skipped": skipped, "n_errors": err})

Integration points
------------------
- GUI: create one BatchLogger at start, call .log_run_start(cfg), use L = blog.file_ctx(...) around each file.
- Backend (optional, better): pass the logger down and log *decisions* inside ROI flow.
  For example, when you reuse vs draw a ROI, or when Nyquist guard triggers.

"""

from __future__ import annotations
import json, logging, os, sys, time, traceback, platform, socket, uuid, threading
from contextlib import ContextDecorator
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import numpy as _np  # optional; used for safe encoding
except Exception:
    _np = None


# --------------------------- helpers ---------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _coerce_scalar(x: Any) -> Any:
    """Convert numpy scalars and Paths to plain JSON-safe Python types."""
    if isinstance(x, Path):
        return str(x)
    if _np is not None:
        if isinstance(x, _np.generic):
            return x.item()
    return x

def _to_jsonable(obj: Any) -> Any:
    """Safely convert complex objects to JSON. Large arrays are summarized (not dumped)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if _np is not None:
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            try:
                fin = _np.isfinite(obj)
                _min = float(_np.nanmin(obj[fin])) if fin.any() else float("nan")
                _max = float(_np.nanmax(obj[fin])) if fin.any() else float("nan")
            except Exception:
                _min = _max = None
            return {"__ndarray__": True,
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                    "min": _min, "max": _max}
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)

def _short_kv(d: Dict[str, Any], maxlen: int = 120) -> str:
    try:
        s = ", ".join(f"{k}={v}" for k, v in d.items())
        if len(s) > maxlen: s = s[:maxlen-3] + "..."
        return s
    except Exception:
        return ""


# --------------------------- BatchLogger ---------------------------

class BatchLogger:
    """
    High-level run/file logger writing both human text and JSONL.

    Attributes
    ----------
    run_id : str
        Unique id (timestamp + random) for this run; used in log file names.
    text_path : Path
        Path to the human-readable .log file for this run.
    jsonl_path : Path
        Path to the structured .jsonl file for this run.
    """
    def __init__(self,
                 log_dir: Path,
                 run_id: Optional[str] = None,
                 console: bool = True,
                 level: int = logging.INFO):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_id = run_id or f"run-{ts}-{uuid.uuid4().hex[:6]}"

        self.text_path  = self.log_dir / f"{self.run_id}.log"
        self.jsonl_path = self.log_dir / f"{self.run_id}.jsonl"

        self._lock = threading.Lock()
        self._ctx: Dict[str, Any] = {}   # file-level context (file, mode)
        self._logger = logging.getLogger(f"FRCBatch[{self.run_id}]")
        self._logger.propagate = False
        self._logger.setLevel(level)

        # Avoid duplicate handlers if multiple instances with same run_id
        if not self._logger.handlers:
            fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S")
            fh = logging.FileHandler(self.text_path, encoding="utf-8")
            fh.setFormatter(fmt); fh.setLevel(level)
            self._logger.addHandler(fh)
            if console:
                sh = logging.StreamHandler(stream=sys.stdout)
                sh.setFormatter(fmt); sh.setLevel(level)
                self._logger.addHandler(sh)

        # Write run header into text log
        self._logger.info(f"Logging to: {self.text_path}")
        self._logger.info(f"JSONL:      {self.jsonl_path}")

    # --------------- core emitters ---------------

    def _emit_jsonl(self, event: str, level: str, data: Dict[str, Any]) -> None:
        payload = {
            "ts": _now_iso(),
            "event": event,
            "level": level,
            "run_id": self.run_id,
        }
        payload.update(self._ctx)  # add file/mode if set
        payload["data"] = _to_jsonable(data)
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _emit_text(self, level: int, msg: str) -> None:
        self._logger.log(level, msg)

    def log_event(self, event: str, level: int = logging.INFO, **data) -> None:
        """Generic event emitter."""
        level_name = logging.getLevelName(level)
        self._emit_text(level, f"{event}: {_short_kv(data)}")
        self._emit_jsonl(event, level_name, data)

    # --------------- run-level helpers ---------------

    def log_run_start(self, cfg: Dict[str, Any]) -> None:
        env = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }
        self.log_event("run_start", logging.INFO, cfg=cfg, env=env)

    def log_run_end(self, stats: Dict[str, Any]) -> None:
        self.log_event("run_end", logging.INFO, **stats)

    # --------------- file-scope context manager ---------------

    class _FileContext(ContextDecorator):
        def __init__(self, blog: "BatchLogger", file_path: Path, mode: str):
            self.blog = blog
            self.file_path = str(file_path)
            self.mode = mode
            self.outputs: Dict[str, Any] = {}
            self.status: str = "OK"
        def __enter__(self):
            self.blog._ctx = {"file": self.file_path, "mode": self.mode}
            self.blog.log_event("file_start", logging.INFO, file=self.file_path, mode=self.mode)
            return self
        def __exit__(self, exc_type, exc, tb):
            if exc is not None:
                self.status = "ERROR"
                self.blog.log_exception("file_error", error=str(exc))
                # Let exception propagate so caller can handle if desired
                self.blog.log_event("file_end", logging.INFO, file=self.file_path, mode=self.mode,
                                    status=self.status, outputs=self.outputs)
                return False
            self.blog.log_event("file_end", logging.INFO, file=self.file_path, mode=self.mode,
                                status=self.status, outputs=self.outputs)
            # clear context
            self.blog._ctx = {}
            return False
        # Convenience API
        def params(self, **kwargs): self.blog.log_event("params", logging.INFO, **kwargs)
        def decision(self, message: str, **kwargs): self.blog.log_event("decision", logging.INFO, message=message, **kwargs)
        def metric(self, name: str, value: Any, unit: Optional[str] = None, **kwargs):
            self.blog.log_event("metric", logging.INFO, name=name, value=_coerce_scalar(value), unit=unit, **kwargs)
        def warning(self, message: str, **kwargs):
            self.blog.log_event("warning", logging.WARNING, message=message, **kwargs); self.status = "WARN"
        def error(self, message: str, **kwargs):
            self.blog.log_event("error", logging.ERROR, message=message, **kwargs); self.status = "ERROR"
        def add_output(self, **paths_dict):
            for k, v in paths_dict.items():
                self.outputs[k] = _coerce_scalar(v)
        def set_status(self, status: str):
            self.status = status

    def file_ctx(self, file_path: Path, mode: str) -> "_FileContext":
        """Start logging a single file (csv or pair). Use as: with blog.file_ctx(p, "csv") as L: ..."""
        return BatchLogger._FileContext(self, file_path, mode)

    # --------------- timers ---------------

    class _Timer(ContextDecorator):
        def __init__(self, blog: "BatchLogger", name: str, **fields):
            self.blog = blog
            self.name = name
            self.fields = fields
            self._t0 = 0.0
        def __enter__(self):
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = (time.perf_counter() - self._t0) * 1000.0
            self.blog.log_event("timer", logging.INFO, name=self.name, ms=round(dt, 3), **self.fields)
            return False

    def timer(self, name: str, **fields) -> "_Timer":
        """Measure and log elapsed time (ms) for a code block."""
        return BatchLogger._Timer(self, name, **fields)

    # --------------- shortcuts ---------------

    def params(self, **kwargs):   self.log_event("params", logging.INFO, **kwargs)
    def decision(self, message: str, **kwargs): self.log_event("decision", logging.INFO, message=message, **kwargs)
    def metric(self, name: str, value: Any, unit: Optional[str] = None, **kwargs):
        self.log_event("metric", logging.INFO, name=name, value=_coerce_scalar(value), unit=unit, **kwargs)
    def warning(self, message: str, **kwargs): self.log_event("warning", logging.WARNING, message=message, **kwargs)
    def error(self, message: str, **kwargs):   self.log_event("error", logging.ERROR, message=message, **kwargs)

    # --------------- exceptions ---------------

    def log_exception(self, event: str = "exception", **fields) -> None:
        tb = traceback.format_exc()
        data = dict(traceback=tb, **fields)
        self._emit_text(logging.ERROR, f"{event}: {fields.get('error','')}")  # short console line
        self._emit_jsonl(event, "ERROR", data)
