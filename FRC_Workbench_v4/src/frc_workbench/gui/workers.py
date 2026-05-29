from __future__ import annotations

import traceback
from typing import Any, Callable

from qtpy import QtCore


class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    error = QtCore.Signal(str)
    result = QtCore.Signal(object)


class FunctionWorker(QtCore.QRunnable):
    """Run a function in a Qt threadpool and emit result/error signals."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self):
        try:
            out = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(out)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()
