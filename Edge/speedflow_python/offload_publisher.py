"""
speedflow_python/offload_publisher.py

Non-blocking Zenoh publisher for plate-crop offload.

Publishes on one key expression:
  offload/plates/{src_node}/{dst_node} — L2 plate-crop (source offload_level==1; ~1–3 KB)

The channel runs a single background thread and queue.
The caller enqueues a payload dict; the thread serialises and publishes.

Key design constraints (same as ZenohPublisher):
  - put() is non-blocking and safe to call from the GLib main loop thread.
  - If the queue is full the oldest entry is dropped (freshness > completeness).
  - Uses the shared Zenoh session from PeerOrchestrator when available.

The vehicle-crop tier (Level 2, offload/vehicles/*) was a dead runtime tier
with no orchestrator trigger and was removed — see ADR-0002.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

import cv2
import msgpack
import numpy as np

from .zenoh_session import make_session

logger = logging.getLogger(__name__)

# Payload type constants (included in every message)
TYPE_PLATE   = "plate"    # L2 plate-crop (source offload_level==1)

OFFLOAD_PAYLOAD_SCHEMA_VERSION = 1

_QUEUE_MAXSIZE = 64  # ~2 s of bursts at 30 fps × 1 cam; older entries dropped


class OffloadPublisher:
    """
    Non-blocking crop publisher for L2 plate-crop offload (source offload_level==1).

    Usage:
        pub = OffloadPublisher(node_id="edge-01", session=shared_session)
        pub.start()

        # In SpeedProbe (GLib main loop) — NON-BLOCKING:
        pub.put_plate(target_node="edge-02", stid=(0, 42),
                      camera_id="cam_01", frame_no=1234,
                      crop_bgr=plate_bgr, confidence=0.91)

        pub.stop()
    """

    def __init__(self, node_id: str, session=None) -> None:
        self._node_id  = node_id
        self._ext_sess = session        # shared from PeerOrchestrator if provided
        self._session  = None
        self._pubs: Dict[str, Any] = {}  # key_expr → declared publisher

        self._queue: queue.Queue[Optional[dict]] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Thread-safe lifetime E2E counters (int reads are atomic in CPython)
        self._encoded      = 0
        self._enqueued     = 0
        self._sent         = 0
        self._dropped      = 0
        self._send_errors  = 0

    # ------------------------------------------------------------------
    # Counter accessors (thread-safe: atomic int reads in CPython)
    # ------------------------------------------------------------------

    @property
    def offload_encoded_count(self) -> int:
        return self._encoded

    @property
    def offload_enqueued_count(self) -> int:
        return self._enqueued

    @property
    def offload_sent_count(self) -> int:
        return self._sent

    @property
    def offload_dropped_count(self) -> int:
        return self._dropped

    @property
    def offload_send_errors_count(self) -> int:
        return self._send_errors

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._ext_sess is not None:
            self._session = self._ext_sess
            logger.info("[OffloadPub] Using shared Zenoh session.")
        else:
            self._session = make_session()
            logger.info("[OffloadPub] Opened own Zenoh session.")

        self._running = True
        self._thread  = threading.Thread(
            target=self._publish_loop,
            name=f"OffloadPublisher-{self._node_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)           # sentinel
        if self._thread:
            self._thread.join(timeout=5)
        # Undeclare publishers (no-op if session is shared — caller closes it)
        if self._ext_sess is None:
            for pub in self._pubs.values():
                try:
                    pub.undeclare()
                except Exception:
                    pass
            if self._session:
                self._session.close()
        logger.info("[OffloadPub] Stopped. encoded=%d enqueued=%d sent=%d dropped=%d send_errors=%d",
                    self._encoded, self._enqueued, self._sent, self._dropped, self._send_errors)

    # ------------------------------------------------------------------
    # Public API — non-blocking enqueue
    # ------------------------------------------------------------------

    def put_plate(
        self,
        target_node: str,
        stid: tuple,
        camera_id: str,
        frame_no: int,
        crop_bgr: np.ndarray,
        confidence: float = 0.0,
    ) -> None:
        """
        Enqueue a plate crop for L2 plate-crop offload (source offload_level==1).
        crop_bgr: (H, W, 3) BGR uint8 array — typically ~120×48.
        """
        jpeg = self._encode_jpeg(crop_bgr, quality=85)
        if jpeg is None:
            return
        self._encoded += 1
        now = time.time()
        self._enqueue({
            "schema_version": OFFLOAD_PAYLOAD_SCHEMA_VERSION,
            "version":     OFFLOAD_PAYLOAD_SCHEMA_VERSION,
            "level":       3,
            "type":        TYPE_PLATE,
            "src":         self._node_id,
            "dst":         target_node,
            "camera_id":   camera_id,
            "stid":        list(stid),       # msgpack requires list not tuple
            "frame_no":    frame_no,
            "jpeg":        jpeg,             # raw bytes (not base64)
            "confidence":  float(confidence),
            "timestamp":   now,
            "ts":          now,
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_jpeg(bgr: np.ndarray, quality: int) -> Optional[bytes]:
        if bgr is None or bgr.size == 0:
            return None
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return buf.tobytes()

    def _enqueue(self, item: dict) -> None:
        try:
            self._queue.put_nowait(item)
            self._enqueued += 1
        except queue.Full:
            try:
                self._queue.get_nowait()   # drop oldest
                self._queue.put_nowait(item)
                self._dropped += 1
                self._enqueued += 1
                if self._dropped % 50 == 1:
                    logger.warning(
                        "[OffloadPub] Queue full — dropping oldest (total=%d)", self._dropped
                    )
            except (queue.Empty, queue.Full):
                pass

    def _get_or_declare_pub(self, key: str):
        pub = self._pubs.get(key)
        if pub is None:
            pub = self._session.declare_publisher(key)
            self._pubs[key] = pub
        return pub

    def _publish_loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                crop_type   = item["type"]
                dst         = item["dst"]

                key = f"offload/plates/{self._node_id}/{dst}"

                pub = self._get_or_declare_pub(key)
                pub.put(msgpack.packb(item, use_bin_type=True))
                self._sent += 1
            except Exception as exc:
                self._send_errors += 1
                logger.warning("[OffloadPub] Send error: %s", exc)
