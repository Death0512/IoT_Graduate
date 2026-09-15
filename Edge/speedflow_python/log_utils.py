"""
Edge logging resilience and crash diagnostic utilities.
Provides unhandled exception hooks, thread crash logging, faulthandler,
and auto-flushing rotating file handlers for freeze/crash survival.
"""
import contextlib
import faulthandler
import logging
import logging.handlers
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Union

# Keep a module-level reference to the faulthandler file descriptor so it is not GC'd
_faulthandler_file = None

# Interval (seconds) between repeated hang-diagnostic traceback dumps. 0 disables.
# Loaded from env via settings if available, else 300s default.
try:
    from speedflow_python.settings import HANG_DIAGNOSTIC_INTERVAL_S as _HANG_DIAGNOSTIC_INTERVAL_S
except (ImportError, AttributeError):
    _HANG_DIAGNOSTIC_INTERVAL_S = 300.0


@contextlib.contextmanager
def timed_lock(
    lock: threading.RLock,
    name: str,
    warn_threshold_s: float = 0.05,
    logger: Optional[logging.Logger] = None,
) -> Iterator[None]:
    """Context manager for acquiring locks while diagnosing contention and hold durations."""
    t0 = time.monotonic()
    lock.acquire()
    t_acquired = time.monotonic()
    wait_s = t_acquired - t0
    log = logger or logging.getLogger(__name__)
    if wait_s >= warn_threshold_s:
        log.warning("[DiagLock] High wait for lock '%s': wait=%.4fs (threshold=%.4fs)", name, wait_s, warn_threshold_s)
    try:
        yield
    finally:
        t_released = time.monotonic()
        hold_s = t_released - t_acquired
        lock.release()
        if hold_s >= warn_threshold_s:
            log.warning("[DiagLock] High hold duration for lock '%s': hold=%.4fs (threshold=%.4fs)", name, hold_s, warn_threshold_s)


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes and fsyncs immediately on WARNING+ records."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        if record.levelno >= logging.WARNING:
            self.flush()
            try:
                if self.stream and hasattr(self.stream, "fileno"):
                    os.fsync(self.stream.fileno())
            except (OSError, ValueError):
                pass


class FlushRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that flushes and fsyncs immediately on WARNING+ records."""

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        maxBytes: int = 50 * 1024 * 1024,
        backupCount: int = 3,
        encoding: Optional[str] = "utf-8",
        delay: bool = False,
    ) -> None:
        # ponytail: 50MB x 3 default cap protects eMMC from unbounded debug log growth
        super().__init__(
            str(filename),
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
        )

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        if record.levelno >= logging.WARNING:
            self.flush()
            try:
                if self.stream and hasattr(self.stream, "fileno"):
                    os.fsync(self.stream.fileno())
            except (OSError, ValueError):
                pass


def _uncaught_exception_handler(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical(
        "Uncaught main exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )
    for h in logging.root.handlers:
        try:
            h.flush()
            stream = getattr(h, "stream", None)
            if stream and hasattr(stream, "fileno"):
                os.fsync(stream.fileno())
        except (OSError, ValueError):
            pass


def _uncaught_thread_exception_handler(args) -> None:
    if issubclass(args.exc_type, KeyboardInterrupt):
        return
    thread_name = getattr(args.thread, "name", "unknown")
    thread_ident = getattr(args.thread, "ident", "unknown")
    logging.critical(
        f"Uncaught exception in thread {thread_name} (id={thread_ident})",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
    for h in logging.root.handlers:
        try:
            h.flush()
            stream = getattr(h, "stream", None)
            if stream and hasattr(stream, "fileno"):
                os.fsync(stream.fileno())
        except (OSError, ValueError):
            pass


def install_crash_hooks(log_dir: Optional[Union[Path, str]] = None) -> None:
    """Install sys.excepthook, threading.excepthook, and faulthandler.

    If log_dir is None, defaults to EDGE_LOG_DIR from settings (env-backed),
    falling back to None (stderr).
    # ponytail: stdlib faulthandler + excepthooks, zero third-party deps
    """
    global _faulthandler_file
    sys.excepthook = _uncaught_exception_handler
    if hasattr(threading, "excepthook"):
        threading.excepthook = _uncaught_thread_exception_handler

    if log_dir is None:
        try:
            from speedflow_python.settings import EDGE_LOG_DIR as _eld
            if _eld:
                log_dir = _eld
        except (ImportError, AttributeError):
            pass

    try:
        if log_dir is not None:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            fault_file = log_path / "faulthandler.log"
            # Open unbuffered or line-buffered append file for crash signals
            _faulthandler_file = open(fault_file, "a", buffering=1, encoding="utf-8")
            faulthandler.enable(file=_faulthandler_file, all_threads=True)
        else:
            faulthandler.enable(all_threads=True)
    except Exception:
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass

    install_hang_diagnostic()


def install_hang_diagnostic(interval_s: Optional[float] = None) -> None:
    """Start a repeated in-process hang diagnostic via faulthandler.

    Dumps a traceback of all threads every ``interval_s`` seconds so a wedged
    decision loop / lock is visible in the crash log instead of a silent hang.
    Uses the existing crash-hook file handle (``_faulthandler_file``) when
    available, else stderr — never crashes when logging is unavailable.

    Cancellation/cleanup path: ``cancel_hang_diagnostic()`` stops the timer.
    The ``repeat=True`` timer is intended to run for the process lifetime; the
    cancel function is the graceful-shutdown cleanup path.
    # ponytail: stdlib faulthandler only, no new deps.
    """
    if interval_s is None:
        interval_s = _HANG_DIAGNOSTIC_INTERVAL_S
    if interval_s <= 0:
        return
    try:
        faulthandler.dump_traceback_later(
            interval_s,
            repeat=True,
            file=_faulthandler_file if _faulthandler_file is not None else sys.stderr,
        )
    except Exception:
        # Never let a diagnostic failure take down the process.
        pass


def cancel_hang_diagnostic() -> None:
    """Cancel the repeated hang-diagnostic timer (cleanup path)."""
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass


def notify_watchdog() -> None:
    """Send ``WATCHDOG=1`` to systemd via ``$NOTIFY_SOCKET`` (no-op when absent).

    stdlib-only sd_notify — the ``systemd`` module is not installed. Safe when
    running outside systemd (NOTIFY_SOCKET unset): returns immediately, so
    startup never fails. Any send error is swallowed (watchdog is best-effort).
    # ponytail: stdlib socket datagram; abstract sockets start with '@'.
    """
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    if sock_path.startswith("@"):
        sock_path = "\0" + sock_path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(sock_path)
            s.sendall(b"WATCHDOG=1")
    except Exception:
        pass

