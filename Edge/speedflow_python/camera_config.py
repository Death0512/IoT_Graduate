#!/usr/bin/env python3
# speedflow_python/camera_config.py
"""
CameraManager — Manage multi-camera configuration for Multi-Stream system.

Provides:
  - Read/parse cameras.yml file
  - Pre-compute Homography matrix for each camera
  - Fast lookup API by source_id
  - Low-latency watcher (inotify via watchdog) + 100ms debounce
  - REST API (FastAPI) for programmatic add/remove
  - Thread-safe delta queue → GLib.idle_add() to ensure GStreamer ops
    always run on GLib Main Loop thread.

Requirements:
    pip install watchdog fastapi uvicorn
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Complete configuration information for a camera."""
    camera_id: str
    source_id: int
    uri: str
    enabled: bool
    name: str
    fps: float
    speed_limit_kmh: float

    # Homography
    source_points: np.ndarray          # shape (4, 2) float32
    target_points: np.ndarray          # shape (4, 2) float32
    homo_matrix: np.ndarray            # shape (3, 3) float64 — pre-computed

    # ROI polygon (pixel coords) for ROI filter probe
    roi_polygon: np.ndarray            # shape (N, 2) int32

    # Output
    record: bool
    record_path: str

    # Dynamic / foreign stream flag (set True when added via P2P migration or failover rescue)
    is_dynamic: bool = False

    # --- Derived speed validation params (from fps) ---
    @property
    def min_track_age_frames(self) -> int:
        return int(self.fps * 0.5)

@dataclass
class StreamDelta:
    """Changes detected between two config file reads."""
    to_add: List[CameraConfig] = field(default_factory=list)
    to_remove: List[Any] = field(default_factory=list)   # list of source_id (int) or tuple (source_id, callback)

# ---------------------------------------------------------------------------
# Config Parser
# ---------------------------------------------------------------------------

def _parse_cameras_yml(yml_path: Path) -> Dict[str, CameraConfig]:
    """Read cameras.yml, return dict camera_id -> CameraConfig."""
    with open(yml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cameras = raw.get("cameras", {}) or {}
    result: Dict[str, CameraConfig] = {}

    for cam_id, cfg in cameras.items():
        if cfg is None:
            continue

        src_pts_raw = cfg["homography"]["source_points"]
        tw = cfg["homography"]["target_width"]
        th = cfg["homography"]["target_height"]

        source_pts = np.array(src_pts_raw, dtype=np.float32)  # (4,2)
        target_pts = np.array(
            [[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32
        )
        # Pre-compute homography matrix when reading config
        homo_matrix, _ = cv2.findHomography(source_pts, target_pts)
        if homo_matrix is None:
            # Fallback: getPerspectiveTransform requires exactly 4 points
            homo_matrix = cv2.getPerspectiveTransform(source_pts, target_pts)

        roi_raw = cfg.get("roi_polygon", [])
        # roi_polygon: [x1,y1, x2,y2, x3,y3, x4,y4] → reshape to (N,2)
        roi_arr = np.array(roi_raw, dtype=np.int32).reshape(-1, 2)

        out_cfg = cfg.get("output", {})

        result[cam_id] = CameraConfig(
            camera_id=cam_id,
            source_id=int(cfg["source_id"]),
            uri=cfg["uri"],
            enabled=bool(cfg.get("enabled", True)),
            name=cfg.get("name", cam_id),
            fps=float(cfg.get("fps", 25.0)),
            speed_limit_kmh=float(cfg.get("speed_limit_kmh", 80.0)),
            source_points=source_pts,
            target_points=target_pts,
            homo_matrix=homo_matrix,
            roi_polygon=roi_arr,
            record=bool(out_cfg.get("record", False)),
            record_path=str(out_cfg.get("record_path", f"output/{cam_id}.mp4")),
        )

    return result

# ---------------------------------------------------------------------------
# Tiler layout helper
# ---------------------------------------------------------------------------

def compute_tiler_layout(num_streams: int) -> tuple[int, int]:
    """
    Compute optimal rows × cols for nvmultistreamtiler.
    Prefer square (or near-square) layout.
    """
    if num_streams <= 0:
        return 1, 1
    cols = math.ceil(math.sqrt(num_streams))
    rows = math.ceil(num_streams / cols)
    return rows, cols

# ---------------------------------------------------------------------------
# CameraManager
# ---------------------------------------------------------------------------

class CameraManager:
    """
    Manage the complete lifecycle of camera configuration.

    Usage:
        manager = CameraManager("configs/cameras.yml")
        manager.start(on_add_callback, on_remove_callback, glib_idle_add_fn)
        ...
        cfg = manager.get_config(source_id=0)
        ...
        manager.stop()
    """

    def __init__(self, yml_path: str | Path) -> None:
        self.yml_path = Path(yml_path).resolve()
        if not self.yml_path.exists():
            raise FileNotFoundError(f"Camera config not found: {self.yml_path}")

        # Current state: camera_id -> CameraConfig
        self._configs: Dict[str, CameraConfig] = {}
        # Fast lookup by source_id (immutable view, rebuild on reload)
        self._by_source_id: Dict[int, CameraConfig] = {}
        # Set by the first decoded buffer, not by config registration.
        self._stream_ready: Dict[int, threading.Event] = {}
        # Wall-clock of the last ADD enqueue per source_id (initial load counts
        # as process start). Lets live-held distinguish warming branches (added
        # within the window, pre first-buffer) from stale ones whose negotiation
        # died long ago but whose config was never removed.
        self._stream_added_at: Dict[int, float] = {}
        self._lock = threading.RLock()

        # Delta queue: [StreamDelta, ...] — thread-safe
        self._delta_q: queue.Queue[StreamDelta] = queue.Queue()

        # Callbacks (set when start() is called)
        self._on_add: Optional[Callable[[CameraConfig], None]] = None
        self._on_remove: Optional[Callable[..., None]] = None
        self._glib_idle_add: Optional[Callable] = None
        # Optional read-only diagnostic hook (camera_id, source_id) -> str,
        # set by run_python to snapshot a wedged bin at recreate-retry time.
        self.bin_diagnose_fn: Optional[Callable[[str, int], str]] = None

        # Control flags
        self._running = False
        self._watcher_thread: Optional[threading.Thread] = None
        self._processor_thread: Optional[threading.Thread] = None
        self._observer = None   # watchdog Observer
        self._max_streams: int = 4

        # Initial load
        self._load_initial()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        on_add: Callable[[CameraConfig], None],
        on_remove: Callable[..., None],
        glib_idle_add: Callable,
    ) -> None:
        """
        Start watcher and processor.

        BUG-11 fix: this method is now safe to call multiple times (e.g. on
        pipeline restart inside run_rtsp_push_mode).  On a re-call we update
        the callbacks and restart only the threads/observer that have stopped;
        the CameraManager's in-memory config state is preserved across calls.

        Args:
            on_add:         Called when a new stream needs to be added to GStreamer pipeline.
            on_remove:      Called when a stream (source_id) needs to be removed from pipeline.
            glib_idle_add:  GLib.idle_add function to ensure GStreamer ops
                            run on GLib Main Loop thread.
        """
        self._on_add = on_add
        self._on_remove = on_remove
        self._glib_idle_add = glib_idle_add
        self._running = True

        # Restart processor thread if it has exited
        if (self._processor_thread is None
                or not self._processor_thread.is_alive()):
            self._processor_thread = threading.Thread(
                target=self._processor_loop,
                name="CameraManager-Processor",
                daemon=True,
            )
            self._processor_thread.start()

        # Restart watchdog observer if it has stopped
        if self._observer is None or not self._observer.is_alive():
            self._start_watchdog()

        logger.info("[CameraManager] Started. Watching: %s", self.yml_path)

    def stop(self) -> None:
        """Stop watcher and processor."""
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join()
        # Unblock processor
        self._delta_q.put(None)  # type: ignore[arg-type]
        if self._processor_thread:
            self._processor_thread.join(timeout=3)
        logger.info("[CameraManager] Stopped.")

    def get_config(self, source_id: int) -> Optional[CameraConfig]:
        """Look up CameraConfig by source_id. Thread-safe."""
        with self._lock:
            return self._by_source_id.get(source_id)

    def get_config_by_camera_id(self, camera_id: str) -> Optional[CameraConfig]:
        """Look up CameraConfig by camera_id. Thread-safe."""
        with self._lock:
            return self._configs.get(camera_id)

    def handle_add_command(self, cmd: dict) -> bool:
        """
        Build a CameraConfig from an ADD command dict and enqueue it for
        dynamic addition to the running pipeline.

        Used by PeerOrchestrator's direct-dispatch path (when this node wins a
        migration) so the ADD is applied even if no ZenohCommandSubscriber is
        running.  Mirrors the config-building logic in
        ZenohCommandSubscriber._handle_add (without the Zenoh status/ack
        publishing — those belong to the subscriber path).

        cmd keys: camera_id, source_id, uri, homography{source_points,
        target_width, target_height}, roi_polygon, name, fps, speed_limit_kmh,
        output{record, record_path}.

        Returns True if the ADD was queued, False if it was a no-op
        (e.g. source_id already active).
        """
        cam_id    = cmd["camera_id"]
        source_id = int(cmd["source_id"])
        uri       = cmd["uri"]

        # Reject if camera_id or source_id is already in use (including disabled configs pending removal)
        # to prevent source_id / lookup collisions between cameras.
        with self._lock:
            existing_cam = self._configs.get(cam_id)
            if existing_cam and existing_cam.enabled:
                ready_ev = self._stream_ready.get(source_id)
                if ready_ev is not None and ready_ev.is_set():
                    logger.warning(
                        "[CameraManager] ADD ignored: camera_id='%s' already active and playing. "
                        "Firing ready event so reclaim ACK is not missed.",
                        cam_id,
                    )
                    # Fire the ready event even though we skip the ADD so the
                    # reclaim orchestrator's zenoh_subscriber._send_ack path gets
                    # its ACK signal.  Without this, a reclaim ADD for a camera
                    # that happened to already be PLAYING (e.g. after orphan
                    # recovery) leaves the reclaim loop waiting 15s per attempt
                    # then retrying forever (confirmed 2026-09-22, bug R7).
                    ready_ev.set()
                    return False
                else:
                    logger.info(
                        "[CameraManager] ADD for camera_id='%s' (source_id=%d) replacing unready/retrying branch.",
                        cam_id, source_id,
                    )

            existing_by_sid = self._by_source_id.get(source_id)
            if existing_by_sid and existing_by_sid.enabled and existing_by_sid.camera_id != cam_id:
                logger.warning(
                    "[CameraManager] ADD ignored: source_id=%d ('%s') already active on different camera_id='%s'.",
                    source_id, existing_by_sid.camera_id, cam_id,
                )
                return False

            # Check if any other config in _configs is using this source_id (even if not yet in _by_source_id)
            for c in self._configs.values():
                if c.source_id == source_id and c.camera_id != cam_id and c.enabled:
                    logger.warning(
                        "[CameraManager] ADD rejected: source_id=%d already assigned to enabled camera '%s'.",
                        source_id, c.camera_id,
                    )
                    return False

        # Gate against max_streams: count currently enabled cameras (excluding cam_id if replacing)
        with self._lock:
            enabled_count = sum(1 for c in self._configs.values() if c.enabled and c.camera_id != cam_id)
        if enabled_count >= self._max_streams:
            logger.error(
                "[CameraManager] ADD rejected for source_id=%d ('%s'): "
                "max_streams=%d reached (%d active).",
                source_id, cam_id, self._max_streams, enabled_count,
            )
            return False

        homo_cfg = cmd["homography"]
        src_pts  = np.array(homo_cfg["source_points"], dtype=np.float32)
        tw       = int(homo_cfg["target_width"])
        th       = int(homo_cfg["target_height"])
        tgt_pts  = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
        homo_mat, _ = cv2.findHomography(src_pts, tgt_pts)
        if homo_mat is None:
            homo_mat = cv2.getPerspectiveTransform(src_pts, tgt_pts)

        roi_raw = cmd.get("roi_polygon", [])
        roi_arr = (np.array(roi_raw, dtype=np.int32).reshape(-1, 2)
                   if roi_raw else np.zeros((0, 2), dtype=np.int32))

        out_cfg = cmd.get("output", {})

        cam_cfg = CameraConfig(
            camera_id=cam_id,
            source_id=source_id,
            uri=uri,
            enabled=True,
            name=cmd.get("name", cam_id),
            fps=float(cmd.get("fps", 25.0)),
            speed_limit_kmh=float(cmd.get("speed_limit_kmh", 80.0)),
            source_points=src_pts,
            target_points=tgt_pts,
            homo_matrix=homo_mat,
            roi_polygon=roi_arr,
            record=bool(out_cfg.get("record", False)),
            record_path=str(out_cfg.get("record_path", f"output/{cam_id}.mp4")),
            is_dynamic=True,
        )

        with self._lock:
            self._configs[cam_id] = cam_cfg
            self._rebuild_lookup()
            self._stream_ready[source_id] = threading.Event()
            self._stream_added_at[source_id] = time.time()

        self._delta_q.put(StreamDelta(to_add=[cam_cfg]))
        logger.info(
            "[CameraManager] ADD queued via handle_add_command: "
            "camera_id='%s', source_id=%d", cam_id, source_id,
        )
        return True

    def get_enabled_configs(self) -> List[CameraConfig]:
        """Return list of all enabled cameras."""
        with self._lock:
            return [c for c in self._configs.values() if c.enabled]

    def get_held_camera_ids(self) -> List[str]:
        """Return camera_ids of all enabled cameras (pipeline branches that exist).

        Unlike FPS-based active cameras, this includes warming-up and stalled
        streams that have a live pipeline branch (enabled config).
        """
        with self._lock:
            return [c.camera_id for c in self._configs.values() if c.enabled]

    def get_max_streams(self) -> int:
        """Read max_streams from yml file (cached on init)."""
        return self._max_streams

    def get_tiler_layout(self) -> tuple[int, int]:
        """rows, cols for nvmultistreamtiler based on enabled camera count."""
        n = len(self.get_enabled_configs())
        return compute_tiler_layout(n)

    # ------------------------------------------------------------------
    # Internal — Load & Diff
    # ------------------------------------------------------------------

    def _load_initial(self) -> None:
        """Initial load, no delta creation."""
        try:
            raw = yaml.safe_load(self.yml_path.read_text(encoding="utf-8")) or {}
            self._max_streams = int(raw.get("max_streams", 4))
            logger.info("[CameraManager] Loaded max_streams=%d from %s", self._max_streams, self.yml_path)
            new_configs = _parse_cameras_yml(self.yml_path)
            with self._lock:
                self._configs = new_configs
                self._rebuild_lookup()
            enabled = self.get_enabled_configs()
            logger.info(
                "[CameraManager] Loaded %d cameras (%d enabled): %s",
                len(new_configs),
                len(enabled),
                [c.camera_id for c in enabled],
            )
        except Exception as exc:
            logger.error("[CameraManager] Failed to load config: %s", exc)
            raise

    def _reload_and_diff(self) -> Optional[StreamDelta]:
        """
        Re-read YAML file, compare with current state.
        Return StreamDelta if changed, None otherwise.
        """
        try:
            new_configs = _parse_cameras_yml(self.yml_path)
        except Exception as exc:
            logger.warning("[CameraManager] Reload failed (skipped): %s", exc)
            return None

        with self._lock:
            old_enabled = {
                c.source_id: c
                for c in self._configs.values()
                if c.enabled
            }
            new_enabled = {
                c.source_id: c
                for c in new_configs.values()
                if c.enabled
            }

            to_add_ids = set(new_enabled) - set(old_enabled)
            to_remove_ids = set(old_enabled) - set(new_enabled)

            # Detect URI or config changes of running cameras
            # → remove then re-add to restart stream
            for sid in set(old_enabled) & set(new_enabled):
                old_c = old_enabled[sid]
                new_c = new_enabled[sid]
                if old_c.uri != new_c.uri or old_c.fps != new_c.fps:
                    logger.info(
                        "[CameraManager] source_id=%d config changed → restart", sid
                    )
                    to_remove_ids.add(sid)
                    to_add_ids.add(sid)

            # Commit new state
            self._configs = new_configs
            self._rebuild_lookup()

        if not to_add_ids and not to_remove_ids:
            return None

        # Gate reload against max_streams: only add streams that fit within
        # the configured limit.  This prevents a config-file change from
        # silently overflowing the muxer batch-size past what the Jetson can
        # handle.  Streams that can't fit are left disabled and a clear
        # warning is logged.
        # Use the pre-reload running set.  self._configs already contains
        # new_configs here, so counting it would include pending additions
        # and subtract removals a second time.
        current_active = len(old_enabled) - len(to_remove_ids)
        allowed_add = max(0, self._max_streams - current_active)
        add_list = [new_enabled[sid] for sid in sorted(to_add_ids) if sid in new_enabled]
        if len(add_list) > allowed_add:
            overflow = add_list[allowed_add:]
            add_list = add_list[:allowed_add]
            for config in overflow:
                config.enabled = False
            with self._lock:
                self._rebuild_lookup()
            logger.warning(
                "[CameraManager] max_streams=%d limits ADD to %d stream(s); "
                "dropped %d: %s",
                self._max_streams, allowed_add,
                len(overflow), [c.camera_id for c in overflow],
            )

        delta = StreamDelta(
            to_add=add_list,
            to_remove=list(to_remove_ids),
        )
        logger.info(
            "[CameraManager] Delta detected — add: %s, remove: %s",
            [c.camera_id for c in delta.to_add],
            delta.to_remove,
        )
        return delta

    def _rebuild_lookup(self) -> None:
        """Rebuild _by_source_id from _configs. Call while holding lock."""
        new_by_sid = {}
        for c in self._configs.values():
            if not c.enabled:
                continue
            if c.source_id in new_by_sid:
                existing = new_by_sid[c.source_id]
                logger.critical(
                    "[CameraManager] CRITICAL source_id collision: source_id=%d claimed by both '%s' and '%s'. "
                    "Retaining earlier camera to prevent cross-camera draw mixup.",
                    c.source_id, existing.camera_id, c.camera_id,
                )
                continue
            new_by_sid[c.source_id] = c
        self._by_source_id = new_by_sid

    def reload(self) -> Optional[StreamDelta]:
        """Public method to re-read YAML file and return StreamDelta if changed."""
        return self._reload_and_diff()

    # ------------------------------------------------------------------
    # Internal — Watchdog (inotify)
    # ------------------------------------------------------------------

    def _start_watchdog(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            manager = self   # capture for closure

            class _Handler(FileSystemEventHandler):
                _debounce_timer: Optional[threading.Timer] = None
                _debounce_lock = threading.Lock()

                def on_modified(self, event):
                    if Path(event.src_path).resolve() != manager.yml_path:
                        return
                    with self._debounce_lock:
                        if self._debounce_timer:
                            self._debounce_timer.cancel()
                        # Debounce 100ms — avoid reading file while still writing
                        self._debounce_timer = threading.Timer(
                            0.1, manager._trigger_reload
                        )
                        self._debounce_timer.start()

            self._observer = Observer()
            self._observer.schedule(
                _Handler(), str(self.yml_path.parent), recursive=False
            )
            self._observer.start()
            logger.info("[CameraManager] inotify watcher active (debounce=100ms)")

        except ImportError:
            logger.warning(
                "[CameraManager] 'watchdog' not installed. "
                "Falling back to 1s polling. Run: pip install watchdog"
            )
            # Fallback: polling thread
            self._watcher_thread = threading.Thread(
                target=self._polling_loop,
                name="CameraManager-Poller",
                daemon=True,
            )
            self._watcher_thread.start()

    def _trigger_reload(self) -> None:
        """Called when file changes — compute delta and push to queue."""
        delta = self._reload_and_diff()
        if delta:
            self._delta_q.put(delta)

    def _polling_loop(self) -> None:
        """Fallback when watchdog unavailable: poll file every 1s."""
        last_mtime = self.yml_path.stat().st_mtime
        while self._running:
            time.sleep(1.0)
            try:
                mtime = self.yml_path.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    time.sleep(0.1)   # debounce
                    self._trigger_reload()
            except Exception as exc:
                logger.debug("[CameraManager] Polling error: %s", exc)

    # ------------------------------------------------------------------
    # Internal — Processor (consume delta_q)
    # ------------------------------------------------------------------

    def _processor_loop(self) -> None:
        """
        Consume StreamDelta from queue and schedule GStreamer ops
        via GLib.idle_add to ensure thread safety.
        """
        while self._running:
            try:
                delta = self._delta_q.get(timeout=5.0)
            except queue.Empty:
                continue

            if delta is None:   # stop signal
                break

            # Remove first, add second (avoid source_id conflict)
            remove_events = []
            for item in delta.to_remove:
                if isinstance(item, tuple):
                    sid, on_complete_cb = item
                else:
                    sid, on_complete_cb = item, None

                if self._glib_idle_add and self._on_remove:
                    done = threading.Event()
                    on_rem = self._on_remove

                    def _remove_with_ack(source_id=sid, event=done, remove_fn=on_rem):
                        try:
                            if remove_fn is not None:
                                try:
                                    remove_fn(source_id, done_event=event)
                                except TypeError:
                                    remove_fn(source_id)
                                    event.set()
                            else:
                                event.set()
                        except Exception as exc:
                            logger.error(
                                "[CameraManager] Error in on_remove for source_id=%d: %s",
                                source_id,
                                exc,
                            )
                            event.set()
                        return False

                    self._glib_idle_add(_remove_with_ack)
                    remove_events.append((sid, done, on_complete_cb))
                    logger.info(
                        "[CameraManager] Scheduled REMOVE source_id=%d on GLib loop", sid
                    )
                else:
                    if on_complete_cb is not None:
                        try:
                            on_complete_cb()
                        except Exception as cb_exc:
                            logger.error(
                                "[CameraManager] Error in on_complete callback for source_id=%d: %s",
                                sid,
                                cb_exc,
                            )

            all_removed_cleanly = True
            if remove_events:
                for sid, done, cb in remove_events:
                    # Wait for GLib MainLoop to finish teardown. Worst case is
                    # EOS drain (1s) + 3 sequential states x up to 6s TSG-unbind
                    # wait each (~19s), plus up to 3x8s bounded sink-abandon
                    # on a wedged rtspclientsink (~24s).
                    if not done.wait(timeout=45.0):
                        all_removed_cleanly = False
                        logger.warning(
                            "[CameraManager] Timeout waiting for REMOVE source_id=%d on GLib loop", sid
                        )
                    else:
                        if cb is not None:
                            try:
                                cb()
                            except Exception as cb_exc:
                                logger.error(
                                    "[CameraManager] Error in on_complete callback for source_id=%d: %s",
                                    sid,
                                    cb_exc,
                                )

            if delta.to_remove and delta.to_add:
                if not all_removed_cleanly:
                    logger.error(
                        "[CameraManager] Aborting ADD in delta: one or more REMOVEs "
                        "did not complete within timeout."
                    )
                    continue

            for cam_cfg in delta.to_add:
                cfg = cam_cfg  # capture for lambda
                if self._glib_idle_add and self._on_add:
                    with self._lock:
                        self._glib_idle_add(self._on_add, cfg)
                    logger.info(
                        "[CameraManager] Scheduled ADD camera=%s source_id=%d on GLib loop",
                        cfg.camera_id, cfg.source_id,
                    )

    def stream_ready_event(self, source_id: int) -> Optional[threading.Event]:
        """Return the readiness event for a pending dynamic stream."""
        with self._lock:
            return self._stream_ready.get(source_id)

    def cleanup_stream_ready(self, source_id: int) -> None:
        """Remove readiness event on stream removal to prevent leaks and stale reuse."""
        with self._lock:
            self._stream_ready.pop(source_id, None)
            self._stream_added_at.pop(source_id, None)

    def get_live_held_camera_ids(self, warmup_s: float = 60.0) -> List[str]:
        """Cameras with a live or warming pipeline branch.

        Enabled configs whose branch never produced AND whose negotiation is
        older than warmup_s are excluded: a held claim with no local evidence
        (e.g. a migrated-out camera whose config lingered, or a starved ADD)
        must not advertise holder-self to peers or the Server — it splits the
        ownership truth while another node verifiably streams the camera.
        Initial/static cameras carry no readiness event and are always
        included (their liveness is covered by FPS/watchdog separately).
        """
        now = time.time()
        with self._lock:
            out = []
            for c in self._configs.values():
                if not c.enabled:
                    continue
                ev = self._stream_ready.get(c.source_id)
                if ev is None:
                    out.append(c.camera_id)
                elif ev.is_set():
                    out.append(c.camera_id)
                elif now - self._stream_added_at.get(c.source_id, 0.0) < warmup_s:
                    out.append(c.camera_id)
            return out

    def retry_add(self, source_id: int) -> bool:
        """Re-queue one ADD for an existing enabled config (recreate-retry).

        A starved uridecodebin (RTSP handshake OK, decodebin never exposes a
        pad) never recovers in place — only a fresh source bin renegotiates
        pads. The re-queued ADD runs on the GLib loop via the normal delta
        path; dynamic_add_stream tears down the wedged branch first (stale
        cleanup). Installs a FRESH ready Event so the new branch's first
        buffer — not a late buffer from the dying branch — signals readiness.

        Thread-safe; returns False when no enabled config owns source_id
        (e.g. NVDEC-limit refusal already disabled it) or the branch turned
        PLAYING concurrently (caller must re-check is_set() first).
        """
        with self._lock:
            cfg = self._by_source_id.get(source_id)
            if cfg is None or not cfg.enabled:
                return False
            cur = self._stream_ready.get(source_id)
            if cur is not None and cur.is_set():
                # Turned PLAYING concurrently — caller re-checks and acks;
                # never tear down a good branch.
                return False
            self._stream_ready[source_id] = threading.Event()
            self._stream_added_at[source_id] = time.time()
        diag = ""
        if self.bin_diagnose_fn is not None:
            try:
                diag = f" [{self.bin_diagnose_fn(cfg.camera_id, source_id)}]"
            except Exception as exc_dg:
                diag = f" [diag-unavailable: {exc_dg}]"
        self._delta_q.put(StreamDelta(to_add=[cfg]))
        logger.info(
            "[CameraManager] Recreate-retry queued for camera='%s' (source_id=%d).%s",
            cfg.camera_id, source_id, diag,
        )
        return True

    # ------------------------------------------------------------------
    # REST API (optional — Phase 3)
    # ------------------------------------------------------------------

    def start_rest_api(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """
        Start REST API server (FastAPI + uvicorn) on separate thread.
        Endpoints:
          POST   /cameras/add    body: CameraConfig JSON
          DELETE /cameras/{camera_id}
          GET    /cameras        list all cameras
        """
        try:
            import uvicorn
            from fastapi import FastAPI

            app = FastAPI(title="CameraManager API")
            manager = self

            @app.get("/cameras")
            def list_cameras():
                with manager._lock:
                    return {
                        cam_id: {
                            "source_id": c.source_id,
                            "uri": c.uri,
                            "enabled": c.enabled,
                            "name": c.name,
                        }
                        for cam_id, c in manager._configs.items()
                    }

            @app.delete("/cameras/{camera_id}")
            def remove_camera(camera_id: str):
                with manager._lock:
                    cfg = manager._configs.get(camera_id)
                    if not cfg or not cfg.enabled:
                        return {"status": "not_running", "camera_id": camera_id}
                    # Disable and push delta with callback for proper ordering
                    cfg.enabled = False
                    manager._rebuild_lookup()
                    source_id = cfg.source_id
                    delta = StreamDelta(to_remove=[(source_id, lambda: manager.cleanup_stream_ready(source_id))])
                manager._delta_q.put(delta)
                return {"status": "removing", "source_id": source_id}

            def _run():
                uvicorn.run(app, host=host, port=port, log_level="warning")

            api_thread = threading.Thread(
                target=_run, name="CameraManager-API", daemon=True
            )
            api_thread.start()
            logger.info("[CameraManager] REST API listening on %s:%d", host, port)

        except ImportError:
            logger.warning(
                "[CameraManager] FastAPI/uvicorn not installed. "
                "REST API disabled. Run: pip install fastapi uvicorn"
            )
