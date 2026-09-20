# speedflow_python/probes.py
# -*- coding: utf-8 -*-
"""
GStreamer pad-probe callbacks.

Performance changes vs the original:
  - ROIFilterProbe._check_obj_in_roi  → sf.point_in_polygon  (C, no cv2 call)
  - SpeedProbe: homography applied as ONE batched call per camera per frame
    instead of one numpy array allocation + cv2.perspectiveTransform per object
  - _compute_speed_kmh     → sf.compute_speed_kmh     (C)
  - _valid_measurement_full → sf.valid_measurement     (C)
  - np.median on deque     → sf.median_speed           (C, no full sort)
  - _center_distance       → sf.center_distance        (C)
  - _calculate_plate_quality → sf.plate_quality        (C)
"""

import base64
import json
import logging
import os
import queue
import tempfile
import time
import threading
import uuid
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
import cv2

from . import speedflow_c as sf
from .settings import (
    VEHICLE_CLASS_IDS, SPEED_LOG,
    JPEG_QUALITY, SNAP_DIR, MAX_SNAPSHOT_PER_ID,
    MIN_WORLD_DISPL_M, MAX_ABS_KMH,
    BBOX_AREA_JUMP, MIN_DET_CONF, MEDIAN_WINDOW, LICENSE_PLATE_CLASS_IDS,
    FPS_STATS_FILE, NODE_ID, TELEMETRY_INTERVAL,
)
from .draw import add_polygon_display
from .camera_config import CameraManager, CameraConfig

logger = logging.getLogger(__name__)


def muxer_live_source(source_type_by_camera: Dict[str, str]) -> int:
    """Mirror core_pipeline's streammux live-source decision for telemetry.

    0 → every camera is a file source (PTS-paced realtime playback, so
    output FPS cannot exceed the source file's own FPS).
    1 → at least one live source present (arrival-rate push; a "file"
    camera in such a pipeline runs at decoder throughput and its output
    FPS may exceed the source FPS).

    Kept as a pure function so the writer loop and host tests agree on the
    contract without racing the 1 s telemetry writer thread.
    """
    if source_type_by_camera and all(
        v == "file" for v in source_type_by_camera.values()
    ):
        return 0
    return 1


class CSVLogger:
    """Lightweight CSV appender — optional, non-critical path.

    Opens the header file lazily and holds a persistent append handle so we
    don't pay open/close cost on every row.  close() must be called on
    shutdown to flush and release the fd.
    """
    def __init__(self, path, header):
        self.path = path
        self.header = header
        self._f = None
        self._lock = threading.Lock()
        try:
            needs_header = not os.path.exists(path)
            # r+ would fail if missing; use a+ and seek to decide header
            self._f = open(path, "a", encoding="utf-8")
            if needs_header or os.path.getsize(path) == 0:
                self._f.write(",".join(header) + "\n")
                self._f.flush()
        except OSError as exc:
            logger.warning("[CSVLogger] cannot open %s: %s", path, exc)
            self._f = None

    def write(self, row):
        if self._f is None:
            return
        try:
            with self._lock:
                self._f.write(",".join(map(str, row)) + "\n")
                self._f.flush()
        except OSError as exc:
            logger.warning("[CSVLogger] write failed to %s: %s", self.path, exc)

    def close(self):
        if self._f is not None:
            try:
                self._f.flush()
                self._f.close()
            except OSError:
                pass
            self._f = None


# ---------------------------------------------------------------------------
# ROI filter probe
# ---------------------------------------------------------------------------

class ROIFilterProbe:
    """
    Filters NvDs objects that are outside their camera's ROI polygon.

    The point-in-polygon test is now handled by sf.point_in_polygon (C),
    removing both the cv2.pointPolygonTest overhead and the Python dispatch.
    """

    def __init__(self, camera_manager: CameraManager):
        self.camera_manager = camera_manager

    def analytics_src_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            source_id  = frame_meta.source_id

            cam_cfg = self.camera_manager.get_config(source_id)
            if not cam_cfg:
                l_frame = l_frame.next
                continue

            roi = cam_cfg.roi_polygon          # (N,2) int32 ndarray or None
            # Fix #13: if no ROI is configured, keep all objects (no filter).
            if roi is None or len(roi) == 0:
                l_frame = l_frame.next
                continue
            objects_to_remove = []
            l_obj = frame_meta.obj_meta_list

            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                cx       = obj_meta.rect_params.left + obj_meta.rect_params.width  * 0.5
                bottom_y = obj_meta.rect_params.top  + obj_meta.rect_params.height

                # C extension — no cv2 import on the hot path
                if not sf.point_in_polygon(roi, cx, bottom_y):
                    objects_to_remove.append(obj_meta)

                l_obj = l_obj.next

            for obj_meta in objects_to_remove:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)

            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK


# ---------------------------------------------------------------------------
# Speed + LPR probe
# ---------------------------------------------------------------------------

class SpeedProbe:
    """
    Multi-stream speed measurement and license-plate tracking probe.

    State is keyed by (source_id, track_id) — one entry per tracked vehicle.

    Hot-path compute is delegated to the C extension (speedflow_c.py):
      • Perspective transform: one batched call per camera per frame
        instead of N individual calls with N NumPy array allocations.
      • Speed, validation, median, plate quality, center distance: all C.
    """

    def __init__(self, camera_manager: CameraManager, cooldown_s: float = 2.5,
                 peer_orch=None):
        self.camera_manager = camera_manager
        self._node_id       = NODE_ID
        # PeerOrchestrator reference — used to query offload level per camera.
        # None means all cameras run local inference (default / backward-compat).
        self._peer_orch = peer_orch

        # Per-track state  (stid = (source_id, track_id))
        # maxlen=45 bounds history to ~1.5s at 30fps to prevent unbounded memory growth
        self.history_positions  = defaultdict(lambda: deque(maxlen=45))   # world-Y history
        self.last_speed_text    = defaultdict(str)
        self.last_update_frame  = defaultdict(lambda: -1000)
        self.last_alert_ts      = defaultdict(float)
        self.cooldown_s         = float(cooldown_s)
        self.snap_count         = defaultdict(int)
        self.speed_history      = defaultdict(lambda: deque(maxlen=MEDIAN_WINDOW))
        self.track_birth_frame  = {}
        self.last_area          = {}

        # Zenoh publisher (set externally via set_publisher)
        self.publisher     = None
        # OffloadPublisher for plate-crop (L2) offload sending (set via set_offload_publisher)
        self._offload_pub  = None
        self._offload_rcv  = None

        try:
            os.makedirs(str(SNAP_DIR), exist_ok=True)
        except OSError as exc:
            logger.warning("[SpeedProbe] cannot create SNAP_DIR %s: %s", SNAP_DIR, exc)

        # License plate accumulation window.
        # Adaptive per-camera: ~0.67 s of video, bounded [10, 40] frames.
        #   • 0.67 s  ≃ a typical "all plates should appear" interval
        #     (5 SGIE inference runs at 4-frame batch → 20 frames on a
        #     30-fps camera), scaled to each camera's actual frame rate.
        #   • Lower-bound 10 frames — prevents thrashing on very-low-fps
        #     cameras (e.g. 5 fps doorbell).  Less than 10 produces
        #     unreliable OCR readings.
        #   • Upper-bound 40 frames — avoids keeping stale history on
        #     60-fps sources (5 SGIE cycles ≈ 20 frames is quite enough;
        #     40 is a generous ceiling).
        self.plate_detection_start_frame = {}
        self.plate_candidates            = defaultdict(list)
        self.plate_locked                = {}
        # Local LPR worker (Phase 2): runs TRT LPR off the DeepStream pipeline.
        # Fed from osd_sink_pad_buffer_probe with plate crops; results arrive
        # asynchronously via _drain_offload_results-style injection.  None until
        # run_python.py wires a LocalLprWorker instance via set_lpr_worker.
        self._lpr_worker = None
        # In-flight guard: stids whose crop has been submitted to the worker
        # but not yet answered/rejected — prevents re-submit / crop flooding.
        self._lpr_inflight: set = set()
        self.plate_detection_attempts    = defaultdict(int)

        self.last_cleanup_time = time.time()

        # ── Step-3: warmup timing ──────────────────────────────────────────
        # Set once by run_python.py after pipeline.set_state(PLAYING).
        # health_agent reads it on the next heartbeat cycle.
        self._warmup_ms: Optional[float] = None

        # ── Step-7: Δτ measurement ─────────────────────────────────────────
        # Records wall-clock time of the FIRST valid speed measurement per
        # camera after it was added (used to compute Application Blind-spot).
        # Key: camera_id (str), Value: unix timestamp (float)
        self._first_valid_speed_ts: dict = {}

        # Per-camera source type ("live" | "file"), derived from the camera URI.
        # Written by run_python.py once at startup; read by the FPS writer to
        # populate _source_modes in the telemetry window metadata (see _telemetry).
        # It is also consumed by run_python's source-starved detection gate on
        # the health loop (camera_id → "file" cameras are never starved).
        self._source_type_by_camera: dict = {}

        # ── Service completion tracking (V2 score) ────────────────────────
        self._svc_lock = threading.Lock()
        self._svc_counts: Dict[str, int] = {
            "plates_finalized": 0,      # plates finalized/locked with text or valid observation
            "plates_processed": 0,      # total plate extraction/inference attempts
            "tracks_born": 0,           # new track IDs registered
            "tracks_expired": 0,        # tracks evicted during periodic cleanup
            "tracks_expired_locked": 0, # evicted tracks that were successfully locked
            "tracks_missed": 0,         # evicted tracks with >= min_frames that were never locked
        }
        # Track-level service tracking: stid -> {"birth_frame": int, "finalized": bool}
        self._track_service_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # ── Offload result injector ────────────────────────────────────────
        # offload_receiver.py feeds decoded plate results into this queue.
        # Drained at the top of every osd_sink_pad_buffer_probe call so results
        # appear in the OSD overlay on the next frame after they arrive.
        # Queue is thread-safe; maxsize prevents unbounded growth if results
        # arrive faster than frames.
        self._offload_result_q: queue.Queue = queue.Queue(maxsize=256)
        # Lifetime counter of results successfully enqueued from the receiver
        # (sender-probe side of the offload E2E loop). Thread-safe (int).
        self._results_received: int = 0
        # Lifetime counter of results rejected at inject time because the
        # receiver explicitly flagged an inference failure (inference_ok=False).
        # Thread-safe (int). Surfaced in _snapshot_offload_crops as
        # "results_rejected".
        self._results_rejected: int = 0

        # ── Producer-gate counters for governance L2 (plate-crop) offload ────
        # Canonical source enum: 0 = local, 1 = plate-crop offload, 2 = full-stream migration/RFO.
        # Lifetime, cumulative, lock-protected. Incremented on the
        # GStreamer/GLib thread inside osd_sink_pad_buffer_probe; read by the
        # FPS writer thread during the 1 s snapshot.  Each gate outcome is
        # counted exactly when it happens — no per-frame logging.  Reading
        # under the lock (dict copy) means a snapshot can never raise, even
        # if a counter name is missing.
        #
        # Gate meaning (L2 = plate-crop offload, source offload_level==1):
        #   l2_active_frames       — frames where offload_level==1 AND a
        #                            publisher + target peer were configured
        #   l2_plate_objects       — plates that reached the crop stage
        #   l2_surface_unavailable — frame surface fetch returned None
        #   l2_valid_crops         — crops that passed size checks and were
        #                            handed to put_plate
        #   l2_crop_errors         — exceptions during crop fetch/crop/put
        #
        # l2_valid_crops doubles as the publish-attempt counter: put_plate is
        # called exactly once per valid crop, so the sender's encoded/enqueued
        # counters remain the unambiguous follow-through.
        self._offload_gate_lock: threading.Lock = threading.Lock()
        self._offload_gate_counts: Dict[str, int] = defaultdict(int)

        # ── Crop error type telemetry ─────────────────────────────────────
        # Bounded breakdown of exception class names per offload level.
        # Protected by _offload_gate_lock (same lock as gate counts).
        # Max 16 distinct types per level; overflow still counted in total
        # l*_crop_errors but no new type key is created.
        self._l2_crop_error_types: Dict[str, int] = {}

        # ── Load-score breakdown (set by health-push loop, read by writer) ──
        self._load_score_breakdown: Optional[Dict] = None
        self._lb_lock = threading.Lock()
        # Per-level count for rate-limited warning (fires on 1, 101, 201, …)
        # pony tail: separate counter from gate_inc so gate_inc stays unchanged.
        self._l2_crop_error_total: int = 0

        # ── Unified telemetry counters (reset on every writer flush) ───────
        # Per-camera FPS frame counters — incremented in _tick_fps, drained
        # by the writer thread so fps = frames/window_duration within the
        # SAME actual writer window.
        self._fps_frame_count: Dict[str, int] = defaultdict(int)
        self._fps_frame_lock = threading.Lock()

        # ── Configured native FPS cache (per camera) ───────────────────────
        # Authoritative native rate bound against burst delivery. Updated from
        # cam_cfg.fps on every frame inside _tick_fps, read by the writer
        # to bound published FPS = min(raw callback FPS, configured camera FPS).
        self._configured_fps_lock = threading.Lock()
        self._configured_fps: Dict[str, float] = {}

        # ── Proactive feature cache (per camera, updated every frame) ──────
        # Written on the GLib/GStreamer thread; read by the FPS writer thread
        # under _feature_lock.  Values are per-frame instantaneous counts that
        # the writer loop time-averages before flushing.
        #
        # Per-camera accumulators:
        #   n_track_sum         — sum of active vehicle tracks seen this window
        #   n_plate_sum         — sum of plate detections seen this window
        #   n_stationary_sum    — sum of stationary (≈stopped) vehicle counts
        #   frame_count         — number of frames accumulated since last flush
        #
        # "Stationary" threshold: speed_history median < STATIONARY_KMH_THRESH.
        # Any vehicle whose smoothed speed is below the threshold (or whose
        # speed history is empty, i.e., not yet computed) is counted as stopped.
        self._feature_lock    = threading.Lock()
        self._feature_acc: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"n_track_sum": 0.0, "n_plate_sum": 0.0,
                     "n_stationary_sum": 0.0, "frame_count": 0.0}
        )

        # ── Telemetry session ──────────────────────────────────────────────
        self._session_id = uuid.uuid4().hex[:12]
        self._seq: int = 0
        self._window_started_mono: Optional[float] = None

        # Published FPS snapshot for get_fps_stats() API compatibility.
        # Populated by the writer thread after each flush.
        self._fps_stats_cache: Dict[str, float] = {}
        self._fps_stats_lock = threading.Lock()

        # Feature snapshot cache for get_feature_stats() API compatibility.
        self._feature_snapshot_cache: Dict[str, Dict[str, float]] = {}

        self._fps_writer_running = True
        self._fps_writer_thread  = threading.Thread(
            target=self._fps_writer_loop, name="FPSStatsWriter", daemon=True
        )
        self._fps_writer_thread.start()

    # ------------------------------------------------------------------
    # Publisher
    # ------------------------------------------------------------------

    def remove_camera(self, camera_id: str, source_id: Optional[int] = None) -> None:
        """
        Thread-safe removal cleanup for a camera that was removed from the pipeline.
        Evicts configured FPS, frame counters, proactive feature accumulators,
        and associated per-camera / per-track probe state.
        """
        if not camera_id:
            return

        with self._configured_fps_lock:
            self._configured_fps.pop(camera_id, None)

        with self._fps_frame_lock:
            self._fps_frame_count.pop(camera_id, None)

        with self._feature_lock:
            self._feature_acc.pop(camera_id, None)
            self._feature_snapshot_cache.pop(camera_id, None)

        with self._fps_stats_lock:
            self._fps_stats_cache.pop(camera_id, None)

        self._source_type_by_camera.pop(camera_id, None)
        self._first_valid_speed_ts.pop(camera_id, None)

        if source_id is not None:
            # Purge stid = (source_id, track_id) records
            with self._svc_lock:
                stids_to_purge = [
                    stid for stid in self._track_service_meta if stid[0] == source_id
                ]
                for stid in stids_to_purge:
                    self._track_service_meta.pop(stid, None)

            track_dicts = [
                self.history_positions,
                self.last_speed_text,
                self.last_update_frame,
                self.last_alert_ts,
                self.snap_count,
                self.speed_history,
                self.track_birth_frame,
                self.last_area,
                self.plate_detection_start_frame,
                self.plate_candidates,
                self.plate_locked,
                self.plate_detection_attempts,
            ]
            for d in track_dicts:
                keys = [k for k in d if isinstance(k, tuple) and len(k) >= 1 and k[0] == source_id]
                for k in keys:
                    d.pop(k, None)

    def set_publisher(self, publisher) -> None:
        self.publisher = publisher

    def record_warmup_ms(self, warmup_ms: float) -> None:
        """
        Called by run_python.py immediately after pipeline.set_state(PLAYING).
        The value is forwarded to the health_agent on the next heartbeat so it
        appears in the peers/status heartbeat as 'warmup_ms' (paper §D_setup).
        """
        self._warmup_ms = warmup_ms

    def set_offload_publisher(self, pub) -> None:
        """
        Set the OffloadPublisher instance used for plate-crop (L2) offload sending.
        Called by run_python_mode after the orchestrator and publisher are ready.
        """
        self._offload_pub = pub

    def set_offload_receiver(self, rcv) -> None:
        """
        Store the OffloadReceiver so the FPS writer can query its processed count
        and surface offload_crops_received_per_s in the telemetry snapshot.
        """
        self._offload_rcv = rcv

    def set_lpr_worker(self, worker) -> None:
        """
        Wire the LocalLprWorker (Phase 2).  When set, plate crops extracted in
        osd_sink_pad_buffer_probe are submitted here for local TRT LPR instead of
        relying on the removed in-pipeline sgie2 classifier.
        """
        self._lpr_worker = worker

    def set_source_types(self, source_type_by_camera: dict) -> None:
        """
        Wire per-camera source type ("live" | "file") derived from camera URIs.
        Consumed by the FPS writer (_telemetry._source_modes) and by run_python's
        source-starved gate.  Called once at startup before the pipeline runs.
        """
        if isinstance(source_type_by_camera, dict):
            self._source_type_by_camera = dict(source_type_by_camera)

    def set_load_score_breakdown(self, breakdown: Dict) -> None:
        """
        Thread-safe setter for the latest load-score breakdown.
        Called from the health push loop; writer reads it without blocking.
        """
        with self._lb_lock:
            self._load_score_breakdown = breakdown

    def inject_offload_result(self, result: dict) -> None:
        """
        Thread-safe entry point for OffloadReceiver to push decoded plate text
        back into this probe.  Called from the offload receiver worker thread.
        result: {stid, camera_id, frame_no, plate_text, confidence, ts,
                 inference_ok?, vote?, source?}

        Result contract (receiver → sender):
          • inference_ok present and False → inference failure; the result is
            rejected (not queued, not injected into the OSD overlay) and the
            rejection counter is incremented.
          • inference_ok missing (legacy payloads) or True → valid observation,
            accepted even when plate_text is empty / confidence is 0.  An empty
            plate_text is a legitimate "plate not readable" outcome and still
            participates in voting so it is not re-offloaded.
          • vote present and True → route through the SAME multi-frame voting
            path as local LocalLprWorker results (plate_candidates +
            _select_best_plate_from_candidates).  This is the correct contract
            for peer-decoded LPR crops: a single possibly-empty peer reading
            must NOT lock the track — it accumulates with later readings and
            only the majority/quality-best text locks.  Empty plate_text is
            filtered out by the voter until a real reading arrives.
          • vote absent/False (legacy payloads) → lock the plate immediately
            (backward-compatible L1 behavior).  source marks provenance:
            "peer" for offloaded crops, "worker" for local LPR; default
            "worker" when the key is absent.
        """
        # Strict identity check: only explicit boolean False is rejected.
        # Legacy payloads lacking the key get default True → accepted.
        if result.get("inference_ok", True) is False:
            self._results_rejected += 1
            return
        try:
            self._offload_result_q.put_nowait(result)
            self._results_received += 1
        except queue.Full:
            pass   # discard stale result — next frame will get a fresher one

    def _drain_offload_results(self) -> None:
        """
        Drain receiver results queued by inject_offload_result into the OSD
        plate-lock table.  Only valid observations (not flagged as inference
        failures) reach this queue — rejection happens at inject time.

        An empty plate_text is a valid observation (plate not readable) and
        still locks the track via plate_locked so it is not re-offloaded.
        Results with no stid are silently dropped (no track to associate).
        """
        while True:
            try:
                res    = self._offload_result_q.get_nowait()
                stid_r = tuple(res.get("stid", []))
                if stid_r:
                    plate_txt = res.get("plate_text", "")
                    vote = res.get("vote", False)
                    if vote:
                        # Local LPR (Phase 2/3): multi-frame voting.  Accumulate a
                        # dict-shaped candidate (matches _select_best_plate_from_candidates)
                        # and let it decide the locked plate — do NOT lock immediately.
                        conf = res.get("confidence", 0.0)
                        ts   = res.get("ts", time.time())
                        # source marker carried by the result payload: "peer"
                        # (offloaded crop decoded by a peer) or default "worker"
                        # (local LocalLprWorker result).  Lets the voting path
                        # distinguish provenance without a second voter.
                        src = res.get("source", "worker")
                        self.plate_candidates[stid_r].append({
                            "text": plate_txt, "quality": conf, "ts": ts, "source": src,
                        })
                        best = self._select_best_plate_from_candidates(
                            self.plate_candidates[stid_r])
                        if best is not None:
                            # Sufficient evidence: lock, clear candidates, finalize.
                            self.plate_locked[stid_r] = best
                            self.plate_candidates[stid_r] = []
                            if isinstance(best, str) and len(best.strip()) >= 4:
                                with self._svc_lock:
                                    meta = self._track_service_meta.get(stid_r)
                                    if meta is not None and not meta.get("finalized"):
                                        meta["finalized"] = True
                                        self._svc_counts["plates_finalized"] += 1
                            # Release the in-flight guard — lifecycle complete.
                            self._lpr_inflight.discard(stid_r)
                        else:
                            # Inconclusive vote (no non-empty best): reopen the
                            # in-flight gate so a future frame can re-submit the
                            # crop and retry, but bound retries via the existing
                            # plate_detection_attempts counter so a track that
                            # never yields a readable plate cannot re-submit
                            # unbounded.  (Mirror of the Phase-1 give-up bound.)
                            self.plate_detection_attempts[stid_r] += 1
                            if self.plate_detection_attempts[stid_r] < 3:
                                self._lpr_inflight.discard(stid_r)
                            else:
                                # Retry budget exhausted — terminate this track's
                                # LPR with no plate (matches Phase-1 give-up).
                                self.plate_locked[stid_r] = None
                                self._lpr_inflight.discard(stid_r)
                    else:
                        # Legacy / no-vote payload (backward-compatible L1):
                        # lock the plate immediately.
                        self.plate_locked[stid_r] = plate_txt
                        if isinstance(plate_txt, str) and len(plate_txt.strip()) >= 4:
                            with self._svc_lock:
                                meta = self._track_service_meta.get(stid_r)
                                if meta is not None and not meta.get("finalized"):
                                    meta["finalized"] = True
                                    self._svc_counts["plates_finalized"] += 1
                        # Always release the in-flight guard for this stid.
                        self._lpr_inflight.discard(stid_r)
            except queue.Empty:
                break

    def _gate_inc(self, name: str, n: int = 1) -> None:
        """
        Increment a producer-gate counter (lifetime cumulative, thread-safe).
        Called from the GStreamer thread; must never raise.
        """
        try:
            with self._offload_gate_lock:
                self._offload_gate_counts[name] += n
        except Exception:
            pass   # counters must never break the GStreamer probe

    def _gate_counts_copy(self) -> Dict[str, int]:
        """Snapshot of gate counters for the writer thread — never raises."""
        try:
            with self._offload_gate_lock:
                return dict(self._offload_gate_counts)
        except Exception:
            return {}

    # Bounded error-type telemetry for governance L2 (plate-crop) exception path.
    # Canonical source enum: 0 = local, 1 = plate-crop offload, 2 = full-stream migration/RFO.
    # Used to diagnose a swallowed-at-debug crop failure on matched Jetson
    # All access under _offload_gate_lock so the GStreamer callback and the
    # writer thread can never raise; cap is 16 distinct class names per level.
    _CROP_ERROR_TYPES_CAP: int = 16
    # Warning fires on the 1st error and every 100th error per level.
    _CROP_ERROR_WARN_EVERY: int = 100

    def _record_crop_error_type(self, level: str, exc: Exception) -> None:
        """
        Record an exception class name for crop-error diagnosis.

        Called from the governance L2 (plate-crop) exception handler on the GStreamer
        thread.
        Must never raise and must not spam: a warning is emitted on the first
        error and every 100th error per level (type + message, no traceback).
        """
        exc_name = type(exc).__name__
        should_warn = False
        warn_total = 0
        warn_name = exc_name
        try:
            with self._offload_gate_lock:
                if level == "l2":
                    types = self._l2_crop_error_types
                    self._l2_crop_error_total += 1
                    total = self._l2_crop_error_total
                else:
                    return
                if exc_name in types:
                    types[exc_name] += 1
                elif len(types) < self._CROP_ERROR_TYPES_CAP:
                    types[exc_name] = 1
                # else: cap reached — drop the new type name, total still
                # recorded via l*_crop_errors gate counter.
                if total == 1 or total % self._CROP_ERROR_WARN_EVERY == 0:
                    should_warn = True
                    warn_total = total
                    warn_name = exc_name
        except Exception:
            return  # counters must never break the GStreamer probe

        if should_warn:
            logger.warning(
                "[SpeedProbe] L%s crop offload error #%s: %s: %s",
                level[-1], warn_total, warn_name, exc,
            )

    def _crop_error_types_copy(self) -> Dict[str, Dict[str, int]]:
        """Snapshot of crop error-type dicts for the writer thread — never raises."""
        try:
            with self._offload_gate_lock:
                return {
                    "l2_crop_error_types": dict(self._l2_crop_error_types),
                }
        except Exception:
            return {"l2_crop_error_types": {}}

    def _snapshot_offload_crops(self, prev_count: int, prev_ts: float):
        """
        Build the _offload_crops telemetry dict from the live offload
        publisher/receiver counter objects.

        Counter reads are wrapped in getattr + int() so a missing or partially
        initialized object can never raise inside the GStreamer/writer thread.
        Returns (offload_crops, prev_count, prev_ts) so the caller can keep
        the received_per_s window state.

        Kept as a standalone method so host tests can exercise the flush
        logic without a live GStreamer pipeline.
        """
        now_ts = time.time()

        def _cnt(obj, name: str, default: int = 0) -> int:
            try:
                return int(getattr(obj, name, default))
            except Exception:
                return default

        offload_crops = {
            "processed_count": _cnt(self._offload_rcv, "offload_processed_count"),
            "received_per_s":  0.0,
            "ts":              now_ts,
        }

        # Receiver-side lifetime counters
        for name in (
            "offload_received_count",
            "offload_queue_dropped_count",
            "offload_errors_count",
            "offload_results_sent_count",
            "offload_inference_errors_count",
            "offload_session_dropped_count",
            "offload_queue_full",
            "offload_queue_depth",
            "offload_queue_depth_ratio",
        ):
            if name == "offload_queue_full":
                offload_crops[name] = bool(getattr(self._offload_rcv, name, False))
            elif name == "offload_queue_depth_ratio":
                try:
                    offload_crops[name] = float(getattr(self._offload_rcv, name, 0.0))
                except Exception:
                    offload_crops[name] = 0.0
            else:
                offload_crops[name] = _cnt(self._offload_rcv, name)

        # Sender-side lifetime counters (crops encoded/enqueued/sent)
        for name in (
            "offload_encoded_count",
            "offload_enqueued_count",
            "offload_sent_count",
            "offload_dropped_count",
            "offload_send_errors_count",
        ):
            offload_crops[name] = _cnt(self._offload_pub, name)

        # Sender-probe side: results successfully enqueued by OffloadReceiver
        # into the OSD result queue.
        offload_crops["results_received"] = self._results_received
        # Results rejected at inject time because the receiver flagged an
        # inference failure (inference_ok=False).
        offload_crops["results_rejected"] = self._results_rejected

        # Local LPR worker queue saturation (Phase 2/3).  0.0 when no worker
        # is wired — used by the orchestrator to decide L1 offload escalation.
        # Snapshot depth+ratio in one call: the peak resets on read, so fetching
        # the two separately would always starve the second sample.
        lpr_queue_depth, lpr_queue_ratio = (
            self._lpr_worker.queue_snapshot() if self._lpr_worker else (0, 0.0)
        )
        offload_crops["lpr_queue_ratio"] = lpr_queue_ratio
        offload_crops["lpr_queue_depth"] = lpr_queue_depth
        offload_crops["lpr_submit_full_dropped"] = (
            self._lpr_worker.submit_full_dropped() if self._lpr_worker else 0
        )

        # Phase 3: feed ratio + depth to the orchestrator so it can escalate
        # plate-crop offload to a peer when the local LPR pool saturates.
        if self._peer_orch is not None and hasattr(self._peer_orch, "set_lpr_queue_ratio"):
            try:
                self._peer_orch.set_lpr_queue_ratio(lpr_queue_ratio, depth=lpr_queue_depth)
            except Exception:
                pass

        # Producer-gate counters (governance L1 plate-crop offload) — cumulative lifetime
        # counters incremented on the GStreamer thread. Read via a lock-held
        # dict copy so the writer thread can never see a torn value.
        gate_counts = self._gate_counts_copy()
        for name in (
            # L2 (plate crops)
            "l2_active_frames", "l2_plate_objects", "l2_surface_unavailable",
            "l2_valid_crops", "l2_crop_errors",
        ):
            offload_crops[name] = gate_counts.get(name, 0)

        # Bounded crop error-type breakdowns — added to the snapshot so the
        # swallowed-at-debug governance L1 (plate-crop) failure is visible at runtime.
        for name, err_types in self._crop_error_types_copy().items():
            offload_crops[name] = err_types

        # Rate (received_per_s) — backward-compat computation on the monotonic
        # processed counter.
        count = offload_crops["processed_count"]
        if prev_ts > 0:
            dt = max(0.001, now_ts - prev_ts)
            rate = (count - prev_count) / dt
        else:
            rate = 0.0
        prev_count = count
        prev_ts = now_ts
        offload_crops["received_per_s"] = round(max(0.0, rate), 3)

        return offload_crops, prev_count, prev_ts

    # ------------------------------------------------------------------
    # FPS counter
    # ------------------------------------------------------------------

    def _tick_fps(self, camera_id: str, pts_ns: Optional[int] = None,
                 configured_fps: Optional[float] = None) -> None:
        """Increment per-camera frame counter for the current writer window.

        ``pts_ns`` is accepted for signature compatibility but no longer used
        to bound published FPS (frame_meta.buf_pts is batch-level and duplicates
        halve FPS on Jetson).

        ``configured_fps`` is the camera's authored native FPS
        (CameraConfig.fps). Stored per-camera so the writer bounds published
        FPS = min(raw callback FPS, configured camera FPS). ``None`` or
        non-positive values are ignored (fallback to raw callback rate).
        """
        with self._fps_frame_lock:
            self._fps_frame_count[camera_id] += 1
        if configured_fps is not None and configured_fps > 0:
            with self._configured_fps_lock:
                self._configured_fps[camera_id] = configured_fps

    def get_fps_stats(self) -> Dict[str, float]:
        """Return the last-published FPS snapshot — API-compatible."""
        with self._fps_stats_lock:
            return dict(self._fps_stats_cache)

    def _get_current_fps(self, camera_id: str) -> float:
        """Return current display-ready FPS for a camera (fallback 0.0)."""
        with self._fps_stats_lock:
            return float(self._fps_stats_cache.get(camera_id, 0.0))

    # ------------------------------------------------------------------
    # Proactive feature counter
    # ------------------------------------------------------------------

    # Vehicles with smoothed speed below this are counted as stationary
    # (stopped at red light).  3 km/h tolerates GPS/homography noise.
    _STATIONARY_KMH_THRESH: float = 3.0

    def _tick_features(self, camera_id: str, source_id: int,
                       vehicles_in_frame: dict) -> None:
        """
        Accumulate per-frame feature counts for camera_id.

        Called once per frame inside osd_sink_pad_buffer_probe, after Pass 1
        (vehicles_in_frame is populated) and Pass 2 (plate counts available via
        plates_in_frame length — passed through n_plate argument).

        This method only handles vehicle-side counts; plate count is injected
        separately via _tick_features_plates so the call site stays clean.
        """
        n_track = len(vehicles_in_frame)
        n_stationary = 0
        for tid in vehicles_in_frame:
            stid = (source_id, tid)
            sh = self.speed_history.get(stid)
            if sh is None or len(sh) == 0:
                # No speed history yet (new track) — assume stopped
                n_stationary += 1
            else:
                smoothed = sf.median_speed(list(sh))
                if smoothed < self._STATIONARY_KMH_THRESH:
                    n_stationary += 1

        with self._feature_lock:
            acc = self._feature_acc[camera_id]
            acc["n_track_sum"]      += n_track
            acc["n_stationary_sum"] += n_stationary
            acc["frame_count"]      += 1.0

    def _tick_features_plates(self, camera_id: str, n_plate: int) -> None:
        """Add plate count for this frame (called after Pass 2)."""
        with self._feature_lock:
            self._feature_acc[camera_id]["n_plate_sum"] += n_plate

    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Return the last-published per-camera feature snapshot."""
        with self._feature_lock:
            return dict(self._feature_snapshot_cache)

    def _fps_writer_loop(self) -> None:
        _prev_offload_count: int = 0
        _prev_offload_ts: float = 0.0
        while self._fps_writer_running:
            window_started = time.monotonic()
            # Record the window start so FPS = frames / window_duration uses
            # the same interval the features accumulated over.
            with self._fps_frame_lock:
                self._window_started_mono = window_started

            time.sleep(TELEMETRY_INTERVAL)

            window_ended = time.monotonic()
            window_dur   = max(0.001, window_ended - window_started)

            try:
                # ── Drain per-camera FPS frame counters (reset-per-window) ─
                with self._fps_frame_lock:
                    frame_counts = dict(self._fps_frame_count)
                    self._fps_frame_count.clear()

                # Compute fps = frames / window_duration for each camera
                # from raw OSD callback counts (the "arrival rate" the writer
                # always saw).
                fps: Dict[str, float] = {}
                for cam_id, n_frames in frame_counts.items():
                    fps[cam_id] = round(n_frames / window_dur, 1)

                raw_cb_fps: Dict[str, float] = dict(fps)
                burst_cams: Dict[str, float] = {}
                fps_bound_by: Dict[str, str] = {}

                # ── Configured camera FPS bound ────────────────────────────
                # Published FPS = min(raw callback FPS, configured camera FPS).
                # Live CSI/USB or file sources can deliver callbacks in bursts;
                # the camera's authored native rate (CameraConfig.fps, pushed
                # per frame by _tick_fps) is the authoritative upper bound.
                # If no configured FPS is available, fallback to raw callback FPS.
                with self._configured_fps_lock:
                    cfg_snapshot = dict(self._configured_fps)
                for cam_id, cfg_fps in cfg_snapshot.items():
                    cb_fps = raw_cb_fps.get(cam_id, 0.0)
                    if cfg_fps > 0 and cb_fps > cfg_fps:
                        fps[cam_id] = cfg_fps
                        burst_cams[cam_id] = round(cb_fps - cfg_fps, 1)
                        fps_bound_by[cam_id] = "configured"
                    elif cam_id not in fps:
                        fps[cam_id] = 0.0
                # Rebuild the API cache and the per-camera bare-float out dict
                # with the now-bounded values.
                with self._fps_stats_lock:
                    self._fps_stats_cache = fps

                # ── Drain proactive features ───────────────────────────────
                feats = self._flush_features()

                # Update API-compatible caches
                with self._feature_lock:
                    self._feature_snapshot_cache = feats

                # ── Advance sequence ───────────────────────────────────────
                self._seq += 1

                with self._svc_lock:
                    svc_snapshot = dict(self._svc_counts)

                # ── Build unified atomic payload ───────────────────────────
                out: Dict = {
                    "_service": svc_snapshot,
                    "_updated_at":    time.time(),
                    "_telemetry": {
                        "session_id":                        self._session_id,
                        "sequence":                          self._seq,
                        "pipeline_window_started_monotonic": window_started,
                        "pipeline_window_ended_monotonic":   window_ended,
                        "pipeline_window_duration_s":        round(window_dur, 3),
                        # Phase 1 source-mode map: camera_id → "live" | "file"
                        # so downstream consumers can distinguish real-time live
                        # feeds from file-playback cameras before using input
                        # FPS for any overload/QoS logic (never treat file
                        # throughput as source starvation).
                        "source_modes": dict(self._source_type_by_camera),
                        # Mirror of core_pipeline's streammux live-source
                        # decision (0 = all files → PTS-paced realtime playback,
                        # output FPS ≤ source FPS; 1 = any live source →
                        # arrival-rate push).  Diagnostic contract: when
                        # muxer_live_source=1, a "file" camera's output FPS is
                        # decoder throughput and may exceed its source FPS.
                        "muxer_live_source": (
                            muxer_live_source(self._source_type_by_camera)
                        ),
                        # FPS diagnostics under _telemetry:
                        #   raw_callback_fps_per_camera — OSD callback rate
                        #     this window (legacy arrival-rate semantics),
                        #   configured_fps_per_camera — configured camera FPS map
                        #   fps_burst_per_camera — camera_id: (callback FPS −
                        #     configured FPS) for windows where delivery exceeded
                        #     the configured rate,
                        #   fps_bound_by_per_camera — camera_id → "configured"
                        #     indicating upper bound clamped published FPS
                        #     (absent when no clamping occurred).
                        "raw_callback_fps_per_camera": raw_cb_fps,
                        "configured_fps_per_camera":   cfg_snapshot,
                        "fps_burst_per_camera":        burst_cams,
                        "fps_bound_by_per_camera":     fps_bound_by,
                    },
                    "_features":       feats,
                }
                # FPS entries (bare floats, backward-compat for direct key lookup)
                for cam_id, f in fps.items():
                    out[cam_id] = f

                # ── Input FPS ─────────────────────────────────────────────
                # _input_fps reflects the configured/native frame rate or
                # bounded output FPS.
                input_fps: Dict[str, float] = {
                    cam_id: fps[cam_id]
                    for cam_id in fps
                }
                out["_input_fps"] = input_fps

                # ── Offload receiver processed count ───────────────────────
                if self._offload_rcv is not None or self._offload_pub is not None:
                    offload_crops, _prev_offload_count, _prev_offload_ts = \
                        self._snapshot_offload_crops(_prev_offload_count, _prev_offload_ts)
                    out["_offload_crops"] = offload_crops

                # ── Load-score breakdown (set by health-push loop) ─────────
                # The writer may observe a value one tick old — acceptable,
                # never block the writer on the health loop.
                with self._lb_lock:
                    if self._load_score_breakdown is not None:
                        out["load_score_breakdown"] = dict(self._load_score_breakdown)

                # ── Atomic write: unique temp file + os.replace ───────────
                # Filesystem hardening: os.replace raises ENOENT when the
                # destination's parent directory does not exist (e.g. a
                # nested /dev/shm/<subdir>/ path whose subdir was never
                # created, or a tmpfs that was cleared/recreated under us).
                # Recreate the parent first, use unique tempfile in same dir,
                # fsync, close, atomically replace, and clean up temp on failure.
                tmp_path = None
                try:
                    _parent = os.path.dirname(os.path.abspath(FPS_STATS_FILE))
                    if _parent and not os.path.isdir(_parent):
                        os.makedirs(_parent, exist_ok=True)
                    fd, tmp_path = tempfile.mkstemp(
                        prefix=".speedflow_fps_",
                        suffix=".tmp",
                        dir=_parent or None,
                    )
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            json.dump(out, f)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp_path, FPS_STATS_FILE)
                        tmp_path = None
                    except Exception:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
                        raise
                except OSError as exc:
                    logger.warning(
                        "[FPSWriter] atomic write failed — retrying: %s",
                        exc,
                    )
            except Exception as exc:
                logger.warning(
                    "[FPSWriter] writer failed — retrying next cycle: %s",
                    exc,
                    exc_info=True,
                )

    def _flush_features(self) -> Dict[str, Dict[str, float]]:
        """
        Drain accumulators, compute per-camera averages.
        Returns a snapshot dict {camera_id: {n_track, n_plate, stationary_fraction}}.
        Accumulators are reset after draining.
        """
        snapshot: Dict[str, Dict[str, float]] = {}
        with self._feature_lock:
            for cam_id, acc in self._feature_acc.items():
                fc = acc["frame_count"]
                if fc > 0:
                    n_track  = acc["n_track_sum"]      / fc
                    n_plate  = acc["n_plate_sum"]       / fc
                    n_stat   = acc["n_stationary_sum"]  / fc
                    stat_frac = n_stat / max(1.0, n_track)
                else:
                    n_track = n_plate = stat_frac = 0.0
                snapshot[cam_id] = {
                    # per-frame averages over the telemetry window
                    "n_track":                   round(n_track,  2),
                    "n_plate":                   round(n_plate,  2),
                    "stationary_fraction":       round(stat_frac, 3),
                    # unambiguous aliases — same values, clearer names for
                    # dashboard / downstream consumers (ponytail: aliases only,
                    # remove n_track/n_plate when all callers migrate)
                    "avg_vehicles_per_frame":    round(n_track,  2),
                    "avg_plates_per_frame":      round(n_plate,  2),
                }
                # Reset accumulator for next window
                acc["n_track_sum"] = acc["n_plate_sum"] = \
                    acc["n_stationary_sum"] = acc["frame_count"] = 0.0
        return snapshot

    def stop_fps_writer(self) -> None:
        self._fps_writer_running = False
        if self._fps_writer_thread.is_alive():
            self._fps_writer_thread.join(timeout=2.5)

    def _get_frame_bgr_cached(self, gst_buffer, frame_meta, cache: dict):
        """Return CPU BGR frame for this batch/frame, copying GPU surface once."""
        key = (frame_meta.batch_id, frame_meta.frame_num)
        if key not in cache:
            cache[key] = self._frame_bgr_from_gst_buffer(gst_buffer, frame_meta)
        return cache[key]

    # ------------------------------------------------------------------
    # Plate helpers
    # ------------------------------------------------------------------

    def _select_best_plate_from_candidates(self, candidates):
        if not candidates:
            return None
        valid = [c for c in candidates if c.get("text")]
        if not valid:
            return None

        text_groups: Dict[str, list] = defaultdict(list)
        for c in valid:
            text_groups[c["text"]].append(c)

        best_text = max(text_groups, key=lambda t: len(text_groups[t]))
        best_entry = max(text_groups[best_text], key=lambda x: x.get("quality", 0))
        return best_text

    @staticmethod
    def _extract_lpr_text(obj_meta) -> Optional[str]:
        try:
            class_meta_list = obj_meta.classifier_meta_list
            while class_meta_list is not None:
                class_meta = pyds.NvDsClassifierMeta.cast(class_meta_list.data)
                if class_meta and class_meta.unique_component_id == 3:
                    label_info_list = class_meta.label_info_list
                    if label_info_list is not None:
                        label_info = pyds.NvDsLabelInfo.cast(label_info_list.data)
                        if label_info:
                            text = getattr(label_info, "result_label", None)
                            if text:
                                return text
                            text = getattr(label_info, "result_class_label", None)
                            if text:
                                return text
                class_meta_list = class_meta_list.next
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Vehicle association helper (plate → vehicle)
    # ------------------------------------------------------------------

    def _associate_plate_to_vehicle(
        self,
        plate_bbox: dict,
        vehicles_in_frame: dict,
    ) -> Optional[int]:
        """
        Returns the track_id of the closest vehicle whose bounding box
        contains the plate horizontally, or None if no match within 300 px.
        Uses sf.center_distance (C) instead of np.sqrt.
        """
        pl, pt, pw, ph = (plate_bbox["left"], plate_bbox["top"],
                          plate_bbox["width"], plate_bbox["height"])
        plate_cx = pl + pw * 0.5

        best_vid  = None
        best_dist = float("inf")

        for vid, vbox in vehicles_in_frame.items():
            vl, vt, vw, vh = (vbox["left"], vbox["top"],
                               vbox["width"], vbox["height"])
            dist = sf.center_distance(pl, pt, pw, ph, vl, vt, vw, vh)
            if dist < best_dist and dist < 300:
                h_tol   = vw * 0.5
                v_right = vl + vw
                if vl - h_tol <= plate_cx <= v_right + h_tol:
                    best_dist = dist
                    best_vid  = vid

        return best_vid

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_bgr_from_gst_buffer(gst_buffer, frame_meta):
        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        try:
            img = np.array(surface, copy=True, order="C")
        finally:
            pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    @staticmethod
    def _crop_bbox(image_bgr, obj_meta):
        h, w = image_bgr.shape[:2]
        x  = max(0, int(round(obj_meta.rect_params.left)))
        y  = max(0, int(round(obj_meta.rect_params.top)))
        x2 = min(w, x + max(1, int(round(obj_meta.rect_params.width))))
        y2 = min(h, y + max(1, int(round(obj_meta.rect_params.height))))
        if x >= x2 or y >= y2:
            return None
        return image_bgr[y:y2, x:x2]

    @staticmethod
    def _jpg_b64_and_bytes(image_bgr, quality=85):
        ok, buf = cv2.imencode(".jpg", image_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None, None
        b = buf.tobytes()
        return base64.b64encode(b).decode("ascii"), b

    def _maybe_publish_and_save(self, stid, cam_id, frame_iso_ts,
                                 speed_kmh, crop_bgr):
        now = time.time()
        image_b64 = None
        if crop_bgr is not None:
            image_b64, _ = self._jpg_b64_and_bytes(crop_bgr, JPEG_QUALITY)

        if self.publisher and (now - self.last_alert_ts[stid] >= self.cooldown_s):
            self.last_alert_ts[stid] = now
            license_plate = self.plate_locked.get(stid, None)
            payload = {
                "type":          "overspeed",
                "node_id":       self._node_id,
                "camera_id":     cam_id,
                "ts":            frame_iso_ts,
                "track_id":      int(stid[1]),
                "speed_kmh":     float(speed_kmh),
                "license_plate": license_plate,
                "image_b64":     image_b64,
                "dedup_key":     f"{int(stid[1])}_{frame_iso_ts}",
            }
            try:
                if hasattr(self.publisher, "put"):
                    self.publisher.put(payload)
                else:
                    self.publisher(payload)
            except Exception as exc:
                logger.warning("[SpeedProbe] overspeed publish failed: %s", exc)

        if self.snap_count[stid] < MAX_SNAPSHOT_PER_ID and image_b64 is not None:
            self.snap_count[stid] += 1

    # ------------------------------------------------------------------
    # Stale-track cleanup  (incremental every 1.0 s)
    # ------------------------------------------------------------------

    def _periodic_cleanup(self, current_time: float, current_frame: int) -> None:
        if current_time - self.last_cleanup_time < 1.0:
            return
        self.last_cleanup_time = current_time

        all_stids: set = set()
        all_stids.update(self.history_positions.keys())
        all_stids.update(self.track_birth_frame.keys())
        all_stids.update(self.last_area.keys())
        all_stids.update(self.last_speed_text.keys())
        all_stids.update(self.plate_locked.keys())

        # BUG-5 fix: use last_update_frame age to detect stale tracks, not
        # last_alert_ts (which is only set on overspeed events and remains 0
        # for slow vehicles).  A track is stale if it hasn't produced a valid
        # speed reading for >60 s worth of frames AND its history deque is
        # empty (vehicle has left the scene).  Using history being non-empty
        # alone was wrong — the deque retains up to 1.5 s of old positions
        # even after the vehicle leaves the ROI.
        # We use a 60-second wall-clock staleness guard via last_alert_ts as
        # a secondary gate only when last_update_frame was never set (i.e.,
        # the track never produced a valid speed).
        max_fps_estimate = 30  # conservative upper bound
        stale_frame_age  = int(60 * max_fps_estimate)  # 60 s × 30 fps = 1800 frames
        stale_cutoff_ts  = current_time - 60.0

        stale_keys = []
        for stid in all_stids:
            last_frame = self.last_update_frame.get(stid, -1000)
            age_frames = current_frame - last_frame
            if age_frames < stale_frame_age:
                # Recently updated — keep
                continue
            # Old enough; also verify history is empty (vehicle gone)
            hist = self.history_positions.get(stid)
            if hist and len(hist) > 0:
                # Still has positional history: vehicle may still be in frame.
                # Only evict if the alert timestamp is also stale (>60 s).
                if self.last_alert_ts.get(stid, 0.0) > stale_cutoff_ts:
                    continue
            stale_keys.append(stid)

        for stid in stale_keys:
            # Service tracking: record eviction metrics
            # ponytail: only count as missed if track was born reasonably recently (within ~30s / 900 frames).
            # Evictions of ancient tracks from long-past overload bursts must not spike d_miss.
            with self._svc_lock:
                self._svc_counts["tracks_expired"] += 1
                meta = self._track_service_meta.pop(stid, None)
                if meta is not None:
                    if meta.get("finalized"):
                        self._svc_counts["tracks_expired_locked"] += 1
                    else:
                        birth = meta.get("birth_frame", current_frame)
                        age = current_frame - birth
                        lpd_engaged = (
                            stid in self.plate_detection_start_frame
                            or self.plate_detection_attempts.get(stid, 0) > 0
                        )
                        # Guard: must be >= 5 frames (real vehicle) AND not ancient
                        # (> 900 frames = 30s). Only count as missed if LPD actually
                        # engaged — vehicles that never presented a readable plate
                        # (trucks, occlusions, bad angles) are NOT pipeline collapse.
                        if 5 <= age <= 900 and lpd_engaged:
                            self._svc_counts["tracks_missed"] += 1
                elif stid in self.plate_locked and self.plate_locked[stid] is not None:
                    self._svc_counts["tracks_expired_locked"] += 1

            self.history_positions.pop(stid, None)
            self.last_speed_text.pop(stid, None)
            self.last_update_frame.pop(stid, None)
            self.last_alert_ts.pop(stid, None)
            self.snap_count.pop(stid, None)
            self.speed_history.pop(stid, None)
            self.track_birth_frame.pop(stid, None)
            self.last_area.pop(stid, None)
            self.plate_detection_start_frame.pop(stid, None)
            self.plate_candidates.pop(stid, None)
            self.plate_locked.pop(stid, None)
            self.plate_detection_attempts.pop(stid, None)
            self._lpr_inflight.discard(stid)

    # ------------------------------------------------------------------
    # Main probe callback
    # ------------------------------------------------------------------

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame    = batch_meta.frame_meta_list

        _NTP_UNIX_OFFSET_NS = 2_208_988_800 * 1_000_000_000
        _MIN_VALID_UNIX_NS  = 946_684_800   * 1_000_000_000

        # ── Drain offload results from peer receiver ───────────────────────
        # Must happen before the frame loop so results land in OSD this frame.
        # Rejection of inference-failure results happened at inject time.
        self._drain_offload_results()

        # Update adaptive PGIE interval based on total tracks in this batch
        total_tracks_in_batch = 0

        while l_frame:
            frame_meta   = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_number = frame_meta.frame_num
            source_id    = frame_meta.source_id
            frame_bgr_cache = {}

            cam_cfg = self.camera_manager.get_config(source_id)
            if not cam_cfg:
                l_frame = l_frame.next
                continue

            # ── Offload level for this camera ──────────────────────────────
            # Canonical source enum: 0 = local, 1 = plate-crop offload, 2 = full-stream migration/RFO.
            offload_level  = 0
            offload_target = ""
            if self._peer_orch is not None:
                offload_level  = self._peer_orch.get_offload_level(cam_cfg.camera_id)
                offload_target = self._peer_orch.get_offload_target(cam_cfg.camera_id)

            # ── Timestamp ─────────────────────────────────────────────────
            ts_ns = getattr(frame_meta, "ntp_timestamp", 0)
            if ts_ns and ts_ns > _NTP_UNIX_OFFSET_NS:
                unix_ns = ts_ns - _NTP_UNIX_OFFSET_NS
            else:
                unix_ns = int(time.time() * 1e9)
            if unix_ns < _MIN_VALID_UNIX_NS:
                unix_ns = int(time.time() * 1e9)
            ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(unix_ns / 1e9))

            # FPS counter tick.  buf_pts is the frame presentation timestamp
            # (ns) assigned by the source/encoder; it feeds the PTS source-rate
            # measurement that bounds published FPS against burst delivery.
            # cam_cfg.fps is the authored native rate — cached per-camera so
            # the writer can fall back to it when buf_pts is unavailable
            # (live CSI/USB sources where buf_pts is never populated).
            self._tick_fps(
                cam_cfg.camera_id,
                pts_ns=getattr(frame_meta, "buf_pts", 0),
                configured_fps=cam_cfg.fps,
            )

            # ── Pass 1: collect all vehicles and plates into flat lists ────
            # We accumulate vehicle bottom-center points for a SINGLE batched
            # perspective transform call — one C call instead of N Python calls.
            vehicles_in_frame: Dict[int, dict] = {}
            plates_in_frame = []

            # Working buffers for the batch transform (pre-allocated per frame)
            track_ids_ordered = []       # vehicle track IDs, same order as pts
            raw_pts = []                 # [cx, by] pairs — filled below

            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                if obj_meta.class_id in VEHICLE_CLASS_IDS:
                    tid = obj_meta.object_id
                    r   = obj_meta.rect_params
                    cx       = r.left + r.width  * 0.5
                    bottom_y = r.top  + r.height

                    vehicles_in_frame[tid] = {
                        "left": r.left, "top": r.top,
                        "width": r.width, "height": r.height,
                        "obj_meta": obj_meta,
                    }
                    total_tracks_in_batch += 1
                    track_ids_ordered.append(tid)
                    raw_pts.extend([cx, bottom_y])

                elif obj_meta.class_id in LICENSE_PLATE_CLASS_IDS:
                    obj_meta.text_params.display_text = "license_plate"
                    r = obj_meta.rect_params
                    plates_in_frame.append({
                        "bbox": {
                            "left": r.left, "top": r.top,
                            "width": r.width, "height": r.height,
                        },
                        "obj_meta": obj_meta,
                        "conf": getattr(obj_meta, "confidence", 0.0),
                    })

                l_obj = l_obj.next

            # ── Batch perspective transform (ONE C call for all vehicles) ──
            n_veh = len(track_ids_ordered)
            if n_veh > 0:
                in_arr   = np.array(raw_pts, dtype=np.float32).reshape(n_veh, 2)
                world_xy = sf.perspective_batch(cam_cfg.homo_matrix, in_arr)
                # world_xy[i] = (world_x, world_y) for track_ids_ordered[i]
                y_world_by_tid = {
                    track_ids_ordered[i]: float(world_xy[i, 1])
                    for i in range(n_veh)
                }
            else:
                y_world_by_tid = {}

            # ── Proactive feature tick (vehicle counts) ────────────────────
            # Called after Pass 1 so vehicles_in_frame is complete.
            # speed_history is still valid from previous frames — stationary
            # classification uses cached median, free of the batch-transform result.
            self._tick_features(cam_cfg.camera_id, source_id, vehicles_in_frame)

            # ── Pass 2: License plate accumulation or plate-crop offload ──
            # Plate-crop offload (source offload_level==1): plate crops are sent
            # to the peer; local accumulation skipped.
            if offload_level == 1 and self._offload_pub is not None and offload_target:
                self._gate_inc("l2_active_frames")
                # Backpressure gate: check if offload target peer is saturated
                target_saturated = False
                if self._peer_orch is not None and hasattr(self._peer_orch, "is_offload_target_saturated"):
                    target_saturated = self._peer_orch.is_offload_target_saturated(offload_target)

                if target_saturated:
                    self._gate_inc("l2_dropped_backpressure", len(plates_in_frame))
                else:
                    for plate_info in plates_in_frame:
                        vehicle_id = self._associate_plate_to_vehicle(
                            plate_info["bbox"], vehicles_in_frame
                        )
                        if vehicle_id is None:
                            continue
                        stid = (source_id, vehicle_id)
                        if stid in self.plate_locked:
                            continue
                        # Guard against duplicate sends: once a crop for this
                        # stid has been handed to put_plate, skip until the
                        # track expires and the in-flight marker is cleared.
                        if stid in self._lpr_inflight:
                            continue
                        # Reached the crop stage — a plate the origin would send.
                        self._gate_inc("l2_plate_objects")
                        try:
                            frame_bgr_off = self._get_frame_bgr_cached(
                                gst_buffer, frame_meta, frame_bgr_cache)
                            if frame_bgr_off is not None:
                                bb = plate_info["bbox"]
                                px = max(0, int(bb["left"]))
                                py = max(0, int(bb["top"]))
                                pw = max(1, int(bb["width"]))
                                ph = max(1, int(bb["height"]))
                                plate_crop = frame_bgr_off[py:py+ph, px:px+pw].copy()
                                if plate_crop.size > 0:
                                    self._gate_inc("l2_valid_crops")
                                    self._offload_pub.put_plate(
                                        target_node=offload_target,
                                        stid=stid,
                                        camera_id=cam_cfg.camera_id,
                                        frame_no=frame_number,
                                        crop_bgr=plate_crop,
                                        confidence=plate_info.get("conf", 0.0),
                                    )
                                    # Mark in-flight only after put_plate returns
                                    # (no exception) so a failed enqueue can retry.
                                    self._lpr_inflight.add(stid)
                            else:
                                self._gate_inc("l2_surface_unavailable")
                        except Exception as exc:
                            self._gate_inc("l2_crop_errors")
                            self._record_crop_error_type("l2", exc)
                            logger.debug(
                                "[SpeedProbe] L2 plate crop offload error for camera=%s frame=%s: %s",
                                cam_cfg.camera_id, frame_number, exc, exc_info=True,
                            )

            else:
                # Normal local plate accumulation OR local LPR via LocalLprWorker (Phase 2)
                frame_bgr_local = None
                if self._lpr_worker is not None:
                    try:
                        frame_bgr_local = self._get_frame_bgr_cached(
                            gst_buffer, frame_meta, frame_bgr_cache)
                    except Exception as exc:
                        logger.debug(
                            "[SpeedProbe] local LPR frame fetch error cam=%s frame=%s: %s",
                            cam_cfg.camera_id, frame_number, exc, exc_info=True,
                        )
                        frame_bgr_local = None

                for plate_info in plates_in_frame:
                    vehicle_id = self._associate_plate_to_vehicle(
                        plate_info["bbox"], vehicles_in_frame
                    )
                    if vehicle_id is None:
                        continue

                    stid = (source_id, vehicle_id)
                    if stid in self.plate_locked:
                        continue

                    # Phase 2: feed crop to LocalLprWorker; multi-frame voting
                    # (drained in _drain_offload_results) decides the locked plate.
                    if self._lpr_worker is not None:
                        if stid in self._lpr_inflight:
                            continue
                        if frame_bgr_local is not None:
                            bb = plate_info["bbox"]
                            px = max(0, int(bb["left"]))
                            py = max(0, int(bb["top"]))
                            pw = max(1, int(bb["width"]))
                            ph = max(1, int(bb["height"]))
                            plate_crop = frame_bgr_local[py:py+ph, px:px+pw].copy()
                            if plate_crop.size > 0:
                                if self._lpr_worker.submit(
                                    stid, cam_cfg.camera_id, frame_number,
                                    plate_crop, plate_info.get("conf", 0.0), vote=True):
                                    self._lpr_inflight.add(stid)
                        continue  # worker owns this track's LPR now

                    if stid not in self.plate_detection_start_frame:
                        self.plate_detection_start_frame[stid] = frame_number

                    frames_in_window = (frame_number
                                        - self.plate_detection_start_frame[stid])

                    # Per-camera adaptive window: 0.67 s × fps, clamped [10, 40]
                    plate_detection_frames = max(10, min(40, int(round(cam_cfg.fps * 0.67))))
                    if frames_in_window < plate_detection_frames:
                        plate_text = self._extract_lpr_text(plate_info["obj_meta"])
                        if plate_text:
                            bb = plate_info["bbox"]
                            quality = sf.plate_quality(
                                bb["width"], bb["height"], plate_info["conf"]
                            )
                            self.plate_candidates[stid].append({
                                "text":    plate_text,
                                "conf":    plate_info["conf"],
                                "bbox":    bb,
                                "quality": quality,
                                "frame":   frame_number,
                            })
                    else:
                        best = self._select_best_plate_from_candidates(
                            self.plate_candidates[stid]
                        )
                        if best:
                            self.plate_locked[stid] = best
                            if isinstance(best, str) and len(best.strip()) >= 4:
                                with self._svc_lock:
                                    meta = self._track_service_meta.get(stid)
                                    if meta is not None and not meta.get("finalized"):
                                        meta["finalized"] = True
                                        self._svc_counts["plates_finalized"] += 1
                        else:
                            self.plate_detection_attempts[stid] += 1
                            if self.plate_detection_attempts[stid] < 3:
                                self.plate_detection_start_frame[stid] = frame_number
                                self.plate_candidates[stid] = []
                            else:
                                self.plate_locked[stid] = None

            # ── Proactive feature tick (plate count) ───────────────────────
            # plates_in_frame is populated in Pass 1 regardless of offload level;
            # tick here so n_plate reflects the true detector output.
            self._tick_features_plates(cam_cfg.camera_id, len(plates_in_frame))

            # ── Pass 3: Speed & OSD display ───────────────────────────────
            fps_int = int(cam_cfg.fps)

            for tid, veh_info in vehicles_in_frame.items():
                stid     = (source_id, tid)
                obj_meta = veh_info["obj_meta"]

                y_world = y_world_by_tid.get(tid, None)
                if y_world is None:
                    # Edge case: track appeared but was not in the batch
                    continue

                hist = self.history_positions[stid]
                hist.append(y_world)

                # Trim history to ~1.5 s — O(1) with deque
                max_hist_len = int(cam_cfg.fps * 1.5)
                while len(hist) > max_hist_len:
                    hist.popleft()

                if stid not in self.track_birth_frame:
                    self.track_birth_frame[stid] = frame_number
                    with self._svc_lock:
                        self._svc_counts["tracks_born"] += 1
                        self._track_service_meta[stid] = {
                            "birth_frame": frame_number,
                            "finalized": False,
                        }

                area_now  = max(1.0, obj_meta.rect_params.width) * \
                             max(1.0, obj_meta.rect_params.height)
                area_prev = self.last_area.get(stid, None)
                det_conf  = getattr(obj_meta, "confidence", -1.0)
                if det_conf is None:
                    det_conf = -1.0

                display_text = self.last_speed_text[stid] or f"#{tid}"

                if (len(hist) >= fps_int
                        and (frame_number - self.last_update_frame[stid]) >= fps_int):

                    hist_list = list(hist)
                    speed_kmh = sf.compute_speed_kmh(hist_list, cam_cfg.fps)

                    if speed_kmh is not None:
                        bbox_ratio = (area_now / area_prev
                                      if area_prev and area_prev > 0 else 0.0)
                        age_frames = frame_number - self.track_birth_frame.get(
                            stid, frame_number)

                        if sf.valid_measurement(
                            hist_list, speed_kmh,
                            age_frames, cam_cfg.min_track_age_frames,
                            MIN_WORLD_DISPL_M, MAX_ABS_KMH,
                            bbox_ratio, BBOX_AREA_JUMP,
                            det_conf, MIN_DET_CONF,
                        ):
                            sh = self.speed_history[stid]
                            sh.append(speed_kmh)

                            # C median — no numpy allocation
                            if len(sh) >= 3:
                                speed_smooth = sf.median_speed(list(sh))
                            else:
                                speed_smooth = speed_kmh

                            display_text                     = f"{int(speed_smooth)} km/h"
                            self.last_speed_text[stid]       = display_text
                            self.last_update_frame[stid]     = frame_number

                            # ── Step-7: Δτ — record first valid speed ──
                            if cam_cfg.camera_id not in self._first_valid_speed_ts:
                                self._first_valid_speed_ts[cam_cfg.camera_id] = time.time()

                            if speed_smooth >= cam_cfg.speed_limit_kmh:
                                crop = None
                                try:
                                    frame_bgr = self._get_frame_bgr_cached(
                                        gst_buffer, frame_meta, frame_bgr_cache)
                                    if frame_bgr is not None and frame_bgr.size > 0:
                                        crop = self._crop_bbox(frame_bgr, obj_meta)
                                except Exception as exc:
                                    logger.debug(
                                        "[SpeedProbe] Overspeed snapshot capture error for camera=%s frame=%s track=%s: %s",
                                        cam_cfg.camera_id, frame_number, tid, exc, exc_info=True,
                                    )
                                self._maybe_publish_and_save(
                                    stid, cam_cfg.camera_id,
                                    ts_iso, speed_smooth, crop,
                                )
                        else:
                            display_text               = ""
                            self.last_speed_text[stid] = display_text

                # ── OSD text assembly ──────────────────────────────────────
                speed_text = self.last_speed_text.get(stid, "")
                plate_text = ""
                if stid in self.plate_locked and self.plate_locked[stid]:
                    plate_text = self.plate_locked[stid]

                if speed_text and plate_text:
                    final_display = f"{speed_text}\n{plate_text}"
                elif speed_text:
                    final_display = speed_text
                elif plate_text:
                    final_display = plate_text
                else:
                    final_display = ""

                obj_meta.text_params.display_text          = final_display
                obj_meta.text_params.font_params.font_size  = 6
                obj_meta.text_params.font_params.font_name  = "Serif"
                self.last_area[stid] = area_now

            # ── ROI / homography overlay ───────────────────────────────────
            if cam_cfg.roi_polygon is not None and len(cam_cfg.roi_polygon) > 0:
                add_polygon_display(batch_meta, frame_meta,
                                    cam_cfg.roi_polygon,
                                    color=(1.0, 0.0, 0.0, 1.0))

            if cam_cfg.source_points is not None and len(cam_cfg.source_points) > 0:
                add_polygon_display(batch_meta, frame_meta,
                                    cam_cfg.source_points,
                                    color=(0.0, 1.0, 0.0, 1.0))

            # ── Per-camera telemetry overlay ──────────────────────────────
            cam_id = cam_cfg.camera_id
            fps_val = self._get_current_fps(cam_id)
            feats = self.get_feature_stats().get(cam_id, {})
            n_track = feats.get("n_track", 0.0)
            n_plate = feats.get("n_plate", 0.0)

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            if display_meta is not None:
                display_meta.num_labels = 1
                txt = display_meta.text_params[0]
                txt.display_text = f"FPS: {fps_val:.1f} | N_TRACK: {n_track:.1f} | N_PLATE: {n_plate:.1f}"
                txt.x_offset = 10
                txt.y_offset = 10
                txt.font_params.font_name = "Courier"
                txt.font_params.font_size = 7
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                txt.set_bg_clr = 1
                txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            # Fix #8: always use wall-clock time for the cleanup timer, not the
            # buffer NTP timestamp.  A replayed video file carries old NTP
            # timestamps that diverge wildly from wall clock, making the
            # cleanup interval fire every frame (or never).
            self._periodic_cleanup(time.time(), frame_number)
            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK
