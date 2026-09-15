"""
speedflow_python/offload_receiver.py

Receives offloaded crops from peer nodes and runs lightweight
TensorRT inference directly (no DeepStream pipeline required on the
receiver side).

Subscribes to:
  offload/plates/*/{my_node_id}   — L2 plate-crop (source offload_level==1): run LPR, return plate text
  offload/results/*/{my_node_id}  — results returned when this node is sender

The vehicle-crop tier (Level 2, offload/vehicles/*) was a dead runtime tier
with no orchestrator trigger and was removed — see ADR-0002.

Results are published back to the sender:
  offload/results/{my_node_id}/{sender_node_id}
    payload: {stid, camera_id, frame_no, plate_text, confidence,
              inference_ok, ts, vote, source}

    vote=True and source="peer" are added so the sender's SpeedProbe routes
    the decoded crop through the SAME multi-frame voting path used for local
    LPR (instead of locking the track on the first single reading).

inference_ok is True for every published payload: only successful inference
runs publish.  An inference failure — engine unavailable (not loaded) or an
inference exception (LPR or LPD) — suppresses the result entirely and
increments the offload_inference_errors counter, with a concise reason carried
to the rate-limited WARNING log.  A valid empty observation (LPD found no
plate bbox, or LPR decoded no text) is *not* an error — it still publishes
with inference_ok=True and an empty plate_text.

The inference engines are loaded lazily on the first request (avoids startup
cost when offload is not active).  Both engines are loaded in a dedicated
worker thread so the Zenoh callback thread is never blocked.

TensorRT Python API (tensorrt / pycuda) is required.  If unavailable the
receiver falls back to a no-op stub that logs a warning — the sender will
simply not receive results for that frame.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import msgpack
import numpy as np

from .zenoh_session import make_session

logger = logging.getLogger(__name__)

# LPR inference helpers (_TRTEngine, _preprocess_lpr, _decode_lpr_output,
# label cache + lock, _try_import_trt, _safe_infer_reason) live in
# lpr_worker.py — single source of truth, reused by both the offload receiver
# and the local LPR worker.  Import them here to avoid duplicated code.
from .lpr_worker import (  # noqa: E402  (kept after stdlib imports above)
    _TRTEngine,
    _decode_lpr_output,
    _lpr_labels_cache,
    _lpr_labels_lock,
    _preprocess_lpr,
    _safe_infer_reason,
    _try_import_trt,
)

# Rate limit for inference-exception warning logs: at most one warning per
# this many seconds.  A crashed TRT engine can fail on every crop, so without
# a limit the log would drown in repetitive stack traces.
_inference_error_log_interval = 5.0


class _InferenceFailure:
    """Internal outcome distinguishing inference *failure* from a valid empty
    observation.  Returned by _run_lpr when the engine is unavailable
    (not loaded) or raises.  Carries a concise, safe reason for the
    rate-limited WARNING log.  A None (LPD) or ("", 0.0) (LPR) return remains a
    *valid* empty result that still publishes.

    fatal=True marks a permanent, unrecoverable condition (e.g. TRT not
    installed) so callers can distinguish it from a transient engine failure
    and suppress per-crop repeats after the one-time ERROR has been logged.
    """

    __slots__ = ("reason", "fatal")

    def __init__(self, reason: str, *, fatal: bool = False) -> None:
        self.reason = reason
        self.fatal  = fatal


# ---------------------------------------------------------------------------
# OffloadReceiver
# ---------------------------------------------------------------------------

class OffloadReceiver:
    """
    Receives plate crops from peer nodes, runs TRT inference (LPR),
    and publishes results back on offload/results/{my_node}/{sender_node}.

    Engines are loaded lazily in the worker thread.

    Args:
        node_id:        this node's ID
        session:        shared Zenoh session (from PeerOrchestrator)
        lpr_engine_path: path to lpr.engine (L2 plate-crop, source offload_level==1)
        lpd_engine_path: retained for backward compatibility (unused — the
                         Level 2 vehicle-crop tier was removed per ADR-0002)
        labels_path:    path to labels_lpr.txt
    """

    def __init__(
        self,
        node_id: str,
        session,
        lpr_engine_path: str,
        labels_path: str,
        lpd_engine_path: str = "",
        session_idle_s: float = 10.0,
        lpr_worker: Optional[Any] = None,
    ) -> None:
        self._node_id         = node_id
        self._session         = session
        self._lpr_engine_path = lpr_engine_path
        self._lpd_engine_path = lpd_engine_path
        self._labels_path     = labels_path
        self._session_idle_s  = float(session_idle_s)
        # Phase 3: optional LocalLprWorker already holding the LPR TRT engine.
        # When provided we SHARE its engine (no second GPU load) and delegate
        # _run_lpr to it (reusing its CUDA lock) instead of loading our own.
        self._lpr_worker = lpr_worker

        self._work_q: queue.Queue[Optional[dict]] = queue.Queue(maxsize=32)
        self._thread: Optional[threading.Thread]  = None
        self._running = False

        # Lazy-loaded engines (None until first use)
        self._lpr_engine: Optional[_TRTEngine] = None
        self._lpd_engine: Optional[_TRTEngine] = None
        self._engines_loaded = False
        # Three-valued: None = not yet checked; True = TRT available;
        # False = tensorrt/pycuda absent (permanent — logged once at ERROR).
        # ponytail: None sentinel avoids importing TRT at construction time,
        # keeping the receiver cheap to instantiate on host/CI.
        self._trt_available: Optional[bool] = None

        # Result publisher cache
        self._result_pubs: Dict[str, Any] = {}

        # F2 Session Handshake state: (src_node, camera_id) -> {"level": int, "last_seen": float}
        self._sessions: Dict[Tuple[str, str], dict] = {}
        self._session_dropped_count = 0
        self._sessions_lock = threading.Lock()
        self._last_session_clean_ts = time.time()

        # Thread-safe lifetime E2E counters (int reads are atomic in CPython)
        self._received       = 0
        self._queue_dropped  = 0
        self._processed      = 0
        self._errors         = 0
        self._results_sent   = 0

        # Cumulative inference-execution errors (distinct from worker-loop
        # errors above).  Incremented only when an inference run itself raises
        # inside _run_lpr — NOT for valid empty results such as
        # "no plate decoded".  Rate-limited warning logging.
        self._inference_errors = 0
        self._last_inference_error_log = float("-inf")

        # Result subscription used when this node sends crops to a peer.
        self._result_handler: Optional[Any] = None
        self._result_sub_declared = False

    @property
    def queue_depth(self) -> int:
        """Current number of items queued in the receiver work queue."""
        return self._work_q.qsize()

    @property
    def queue_full(self) -> bool:
        """Whether the receiver work queue is considered saturated (>= 80% capacity)."""
        return self._work_q.qsize() >= int(self._work_q.maxsize * 0.8)

    @property
    def queue_depth_ratio(self) -> float:
        """Work queue occupancy ratio [0.0, 1.0]."""
        if self._work_q.maxsize <= 0:
            return 0.0
        return self._work_q.qsize() / self._work_q.maxsize

    @property
    def session_dropped_count(self) -> int:
        """Crops dropped because no active handshake session existed. Thread-safe (int)."""
        return self._session_dropped_count

    @property
    def offload_session_dropped_count(self) -> int:
        """Alias for session_dropped_count for metric consistency."""
        return self._session_dropped_count

    @property
    def offload_queue_depth(self) -> int:
        """Alias for queue_depth."""
        return self.queue_depth

    @property
    def offload_queue_full(self) -> bool:
        """Alias for queue_full."""
        return self.queue_full

    @property
    def offload_queue_depth_ratio(self) -> float:
        """Alias for queue_depth_ratio."""
        return self.queue_depth_ratio

    def snapshot_counters(self) -> dict:
        """Snapshot of receiver metrics and queue state."""
        return {
            "received": self._received,
            "processed": self._processed,
            "queue_dropped": self._queue_dropped,
            "errors": self._errors,
            "results_sent": self._results_sent,
            "inference_errors": self._inference_errors,
            "queue_depth": self.queue_depth,
            "queue_full": self.queue_full,
            "queue_depth_ratio": self.queue_depth_ratio,
            "session_dropped_count": self._session_dropped_count,
        }

    @property
    def offload_processed_count(self) -> int:
        """Monotonic count of successfully handled crop items. Thread-safe (int)."""
        return self._processed

    @property
    def offload_received_count(self) -> int:
        """Crops successfully decoded from the wire. Thread-safe (int)."""
        return self._received

    @property
    def offload_queue_dropped_count(self) -> int:
        """Crops dropped because the work queue was full. Thread-safe (int)."""
        return self._queue_dropped

    @property
    def offload_errors_count(self) -> int:
        """Worker-loop handle errors. Thread-safe (int)."""
        return self._errors

    @property
    def offload_results_sent_count(self) -> int:
        """Results published back to senders. Thread-safe (int)."""
        return self._results_sent

    @property
    def offload_inference_errors_count(self) -> int:
        """Cumulative inference-execution errors (distinct from worker-loop
        errors).  Incremented only when inference itself raises, not for
        valid empty results. Thread-safe (int)."""
        return self._inference_errors

    @property
    def trt_available(self) -> Optional[bool]:
        """Three-valued TRT dependency state.

        None  — not yet probed (engine load hasn't been triggered yet).
        True  — tensorrt/pycuda imported successfully on first use.
        False — tensorrt/pycuda absent; a one-time ERROR was logged at
                _load_engines_once time; all crops are suppressed.

        Downstream (health_agent, dashboard) can surface this to distinguish
        "receiver running but TRT missing" from "receiver idle".
        """
        return self._trt_available

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        # BUG-18 fix: Zenoh uses '*' for single-level wildcards, not '+'.
        # '+' is MQTT syntax and is silently ignored by Zenoh, causing these
        # subscribers to never match any published key expression.
        self._session.declare_subscriber(
            f"offload/plates/*/{self._node_id}",
            self._on_plate_sample,
        )
        self._session.declare_subscriber(
            f"offload/session/*/{self._node_id}",
            self._on_session_msg,
        )
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"OffloadReceiver-{self._node_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[OffloadReceiver] Started. Listening on offload/*/*/%s and offload/session/*/%s",
            self._node_id, self._node_id,
        )

    def stop(self) -> None:
        self._running = False
        self._work_q.put(None)
        if self._thread:
            self._thread.join(timeout=5)
        # Clean up CUDA TRT engines on stop to prevent device memory leak
        if self._lpr_engine is not None:
            try:
                self._lpr_engine.close()
            except Exception:
                pass
            self._lpr_engine = None
        logger.info(
            "[OffloadReceiver] Stopped. received=%d processed=%d queue_dropped=%d errors=%d results_sent=%d",
            self._received, self._processed, self._queue_dropped,
            self._errors, self._results_sent,
        )

    def set_result_handler(self, handler) -> None:
        """
        Register a callback to receive offload results when this node is the
        SENDER.  The handler receives the decoded result dict:
          {src, dst, camera_id, stid, frame_no, plate_text, confidence, ts}

        Declares a Zenoh subscriber on offload/results/*/{my_node_id} once,
        and forwards every decoded message to handler(payload).
        Called during probe setup before the pipeline starts PLAYING.
        """
        self._result_handler = handler
        if not self._result_sub_declared:
            self._result_sub_declared = True
            self._session.declare_subscriber(
                f"offload/results/*/{self._node_id}",
                self._on_result_sample,
            )
            logger.info(
                "[OffloadReceiver] Result subscriber declared: offload/results/*/%s",
                self._node_id,
            )

    def _on_result_sample(self, sample) -> None:
        """Callback for offload/results/*/{my_node_id} — dispatch to handler."""
        try:
            payload = msgpack.unpackb(sample.payload.to_bytes(), raw=False)
            if self._result_handler is not None:
                self._result_handler(payload)
        except Exception as exc:
            logger.warning("[OffloadReceiver] Result dispatch error: %s", exc)

    # ------------------------------------------------------------------
    # Zenoh subscriber callbacks — called on the Zenoh recv thread
    # Must NOT block.
    # ------------------------------------------------------------------

    def _on_session_msg(self, sample) -> None:
        """Callback for offload/session/*/{my_node_id} — handle start/stop handshake."""
        try:
            payload = msgpack.unpackb(sample.payload.to_bytes(), raw=False)
            action = payload.get("action", "")
            src_node = payload.get("src", "")
            camera_id = payload.get("camera_id", "")
            level = int(payload.get("level", 0))

            if not src_node or not camera_id:
                return

            key = (src_node, camera_id)
            with self._sessions_lock:
                if action == "start":
                    self._sessions[key] = {"level": level, "last_seen": time.time()}
                    logger.info("[OffloadReceiver] Session started: src=%s cam=%s level=%d",
                                src_node, camera_id, level)
                elif action == "stop":
                    if self._sessions.pop(key, None) is not None:
                        logger.info("[OffloadReceiver] Session closed: src=%s cam=%s",
                                    src_node, camera_id)
        except Exception as exc:
            logger.warning("[OffloadReceiver] Session message parse error: %s", exc)

    def _on_plate_sample(self, sample) -> None:
        self._enqueue_sample(sample, crop_type="plate")

    def _enqueue_sample(self, sample, crop_type: str) -> None:
        try:
            payload = msgpack.unpackb(sample.payload.to_bytes(), raw=False)
            payload["_crop_type"] = crop_type
            # Received is credited as soon as the wire payload decodes — even
            # if dropped due to session handshake or queue full.
            self._received += 1

            src = payload.get("src", "")
            camera_id = payload.get("camera_id", "")
            sess_key = (src, camera_id)

            with self._sessions_lock:
                session = self._sessions.get(sess_key)
                if session is None:
                    # Phase 3 L1 P2P: plate crops may arrive without a prior
                    # handshake.  Treat the first plate as session start so we
                    # never drop a valid observation.
                    if crop_type == "plate":
                        session = {"level": 3, "last_seen": time.time()}
                        self._sessions[sess_key] = session
                    else:
                        self._session_dropped_count += 1
                        return
                session["last_seen"] = time.time()

            try:
                self._work_q.put_nowait(payload)
            except queue.Full:
                self._queue_dropped += 1
                if self._queue_dropped == 1 or self._queue_dropped % 500 == 0:
                    logger.warning("[OffloadReceiver] Work queue full — dropping (count=%d)",
                                   self._queue_dropped)
        except Exception as exc:
            logger.warning("[OffloadReceiver] Decode error: %s", exc)

    # ------------------------------------------------------------------
    # Worker loop — runs in dedicated thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while self._running:
            # Periodically clean up stale sessions
            now = time.time()
            if now - self._last_session_clean_ts >= 2.0:
                self._last_session_clean_ts = now
                with self._sessions_lock:
                    stale_keys = [
                        k for k, v in self._sessions.items()
                        if now - v.get("last_seen", 0.0) > self._session_idle_s
                    ]
                    for k in stale_keys:
                        self._sessions.pop(k, None)
                        logger.info("[OffloadReceiver] Session timeout (idle > %.1fs): src=%s cam=%s",
                                    self._session_idle_s, k[0], k[1])

            try:
                item = self._work_q.get(timeout=2.0)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self._handle_item(item)
                self._processed += 1
            except Exception as exc:
                self._errors += 1
                logger.warning("[OffloadReceiver] Handle error: %s", exc)

    def _load_engines_once(self) -> None:
        if self._engines_loaded:
            return

        # --- TRT dependency check (once, actionable ERROR if absent) ----------
        trt, cuda = _try_import_trt()
        if trt is None:
            self._trt_available = False
            self._engines_loaded = True  # don't retry — dependency is permanent
            logger.error(
                "[OffloadReceiver] tensorrt / pycuda are NOT installed on this node. "
                "All offloaded crops will be silently suppressed until TRT is available. "
                "Install JetPack TensorRT bindings (pip install tensorrt pycuda) and "
                "restart the service to enable crop-offload inference."
            )
            return
        self._trt_available = True

        # --- Engine file loading ----------------------------------------------
        # Mark loaded BEFORE the attempts so a repeated first-crop race in the
        # worker thread doesn't trigger a second load.  Partial failure (one
        # engine missing / corrupt) is logged once at ERROR here; per-crop
        # _run_lpr will still produce _InferenceFailure with a clear
        # reason distinguishing "engine not loaded" from "inference exception".
        self._engines_loaded = True
        try:
            # Share the LocalLprWorker's engine if it already loaded one —
            # avoids loading the same TRT engine twice into GPU memory.
            if self._lpr_worker is not None and getattr(self._lpr_worker, "_engine", None) is not None:
                self._lpr_engine = self._lpr_worker._engine
                logger.info(
                    "[OffloadReceiver] LPR engine shared from LocalLprWorker: %s",
                    self._lpr_engine_path,
                )
            elif Path(self._lpr_engine_path).exists():
                self._lpr_engine = _TRTEngine(self._lpr_engine_path)
                logger.info("[OffloadReceiver] LPR engine ready: %s", self._lpr_engine_path)
            else:
                logger.error(
                    "[OffloadReceiver] LPR engine file not found: %s — "
                    "rebuild with trtexec and set lpr_engine_path in config.",
                    self._lpr_engine_path,
                )
        except Exception as exc:
            logger.error(
                "[OffloadReceiver] LPR engine load failed (%s) — "
                "LPR inference will be suppressed until engine is rebuilt.",
                exc,
            )

    def _handle_item(self, item: dict) -> None:
        self._load_engines_once()

        crop_type = item.get("_crop_type", "plate")
        src_node  = item.get("src", "")
        camera_id = item.get("camera_id", "")
        stid      = tuple(item.get("stid", [0, 0]))
        frame_no  = item.get("frame_no", 0)
        jpeg_data = item.get("jpeg", b"")

        # Decode JPEG crop
        arr = np.frombuffer(jpeg_data, dtype=np.uint8)
        crop_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if crop_bgr is None:
            return

        plate_text = ""
        confidence = 0.0
        inference_ok = True

        if crop_type == "plate":
            result = self._run_lpr(crop_bgr)
            if isinstance(result, _InferenceFailure):
                self._record_inference_error("LPR", result.reason, fatal=result.fatal)
                return
            plate_text, confidence = result

        # Publish result back to sender (only reached on successful inference)
        level = 1
        self._publish_result(
            dst_node  = src_node,
            camera_id = camera_id,
            stid      = stid,
            frame_no  = frame_no,
            plate_text= plate_text,
            confidence= confidence,
            inference_ok = inference_ok,
            level     = level,
        )

    def _record_inference_error(self, stage: str, reason: str, *,
                                fatal: bool = False) -> None:
        """
        Count an inference failure (engine unavailable or exception) and log it
        at WARNING rate-limited to one message per
        _inference_error_log_interval seconds.  Never debug-only, never silent.
        The reason is a concise one-liner already produced by
        _safe_infer_reason or a static "engine not loaded" string — never a
        traceback.

        fatal=True (TRT not installed): skips the per-crop rate-limited
        WARNING because the actionable one-time ERROR was already emitted in
        _load_engines_once.  Still increments the counter so the health
        snapshot surfaces the suppressed crop count.
        """
        self._inference_errors += 1
        if fatal:
            # One-time ERROR already logged at engine-load time; per-crop
            # WARNING would be misleading noise.  Counter is still incremented.
            return
        now = time.monotonic()
        if now - self._last_inference_error_log >= _inference_error_log_interval:
            self._last_inference_error_log = now
            logger.warning(
                "[OffloadReceiver] %s inference error #%d: %s; result suppressed",
                stage, self._inference_errors, reason,
            )

    def _run_lpr(self, plate_bgr: np.ndarray) -> Any:
        # Phase 3: if a LocalLprWorker was wired in, delegate to its shared
        # engine (and its CUDA lock) rather than running our own inference.
        if self._lpr_worker is not None:
            if getattr(self._lpr_worker, "_engine", None) is None:
                return _InferenceFailure("LPR engine not loaded (LocalLprWorker engine unavailable)")
            text, conf = self._lpr_worker.run_lpr(plate_bgr)
            return (text, conf)
        if self._trt_available is False:
            # Fatal permanent condition — TRT not installed.  Don't spam the
            # rate-limited warning on every crop; the one-time ERROR in
            # _load_engines_once is the actionable signal.
            return _InferenceFailure("tensorrt/pycuda not installed", fatal=True)
        if self._lpr_engine is None:
            return _InferenceFailure("LPR engine not loaded (file missing or failed to parse)")
        inp = _preprocess_lpr(plate_bgr)
        try:
            outputs = self._lpr_engine.infer(inp)
            return _decode_lpr_output(outputs, self._labels_path)
        except Exception as exc:
            return _InferenceFailure(_safe_infer_reason(exc))

    def _publish_result(
        self, dst_node: str, camera_id: str, stid: tuple,
        frame_no: int, plate_text: str, confidence: float,
        inference_ok: bool = True,
        level: Optional[int] = None,
    ) -> None:
        key = f"offload/results/{self._node_id}/{dst_node}"
        pub = self._result_pubs.get(key)
        if pub is None:
            pub = self._session.declare_publisher(key)
            self._result_pubs[key] = pub

        now = time.time()
        payload = {
            "schema_version": 1,
            "version":     1,
            "src":         self._node_id,
            "dst":         dst_node,
            "camera_id":   camera_id,
            "stid":        list(stid),
            "frame_no":    frame_no,
            "plate_text":  plate_text,
            "confidence":  float(confidence),
            "inference_ok": bool(inference_ok),
            "timestamp":   now,
            "ts":          now,
            # Phase 3 plate-crop fidelity: peer-decoded LPR results carry
            # vote=True so the sender's SpeedProbe routes them through the SAME
            # multi-frame voting path as local LocalLprWorker results instead of
            # locking immediately on a single (possibly empty) reading.  'source'
            # is the explicit peer-vs-local provenance marker consumed by the
            # probe's voting path.
            "vote":   True,
            "source": "peer",
        }
        if level is not None:
            payload["level"] = int(level)
        pub.put(msgpack.packb(payload, use_bin_type=True))
        self._results_sent += 1
