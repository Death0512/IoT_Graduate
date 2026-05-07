#!/usr/bin/env python3
# speedflow_python/camera_config.py
"""
CameraManager — Quản lý cấu hình đa camera cho hệ thống Multi-Stream.

Cung cấp:
  - Đọc/parse file cameras.yml
  - Pre-compute ma trận Homography cho từng camera
  - API tra cứu nhanh theo source_id
  - Watcher độ trễ thấp (inotify qua watchdog) + debounce 100ms
  - REST API (FastAPI) cho thêm/bớt lập trình
  - Thread-safe delta queue → GLib.idle_add() để đảm bảo GStreamer ops
    luôn chạy trên GLib Main Loop thread.

Yêu cầu:
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
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Thông tin cấu hình đầy đủ của một camera."""
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

    # ROI polygon (pixel coords) cho ROI filter probe
    roi_polygon: np.ndarray            # shape (N, 2) int32

    # Output
    record: bool
    record_path: str

    # --- Derived speed validation params (từ fps) ---
    @property
    def min_track_age_frames(self) -> int:
        return int(self.fps * 0.5)


@dataclass
class StreamDelta:
    """Thay đổi được phát hiện giữa 2 lần đọc file config."""
    to_add: List[CameraConfig] = field(default_factory=list)
    to_remove: List[int] = field(default_factory=list)   # list of source_id


# ---------------------------------------------------------------------------
# Config Parser
# ---------------------------------------------------------------------------

def _parse_cameras_yml(yml_path: Path) -> Dict[str, CameraConfig]:
    """Đọc cameras.yml, trả về dict camera_id -> CameraConfig."""
    with open(yml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cameras = raw.get("cameras", {})
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
        # Pre-compute homography matrix ngay lúc đọc config
        homo_matrix, _ = cv2.findHomography(source_pts, target_pts)
        if homo_matrix is None:
            # Fallback: getPerspectiveTransform yêu cầu đúng 4 điểm
            homo_matrix = cv2.getPerspectiveTransform(source_pts, target_pts)

        roi_raw = cfg.get("roi_polygon", [])
        # roi_polygon: [x1,y1, x2,y2, x3,y3, x4,y4] → reshape (N,2)
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
    Tính rows × cols tối ưu cho nvmultistreamtiler.
    Ưu tiên layout vuông (hoặc gần vuông).
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
    Quản lý toàn bộ vòng đời cấu hình camera.

    Sử dụng:
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

        # Trạng thái hiện tại: camera_id -> CameraConfig
        self._configs: Dict[str, CameraConfig] = {}
        # Lookup nhanh theo source_id (immutable view, rebuild khi reload)
        self._by_source_id: Dict[int, CameraConfig] = {}
        self._lock = threading.RLock()

        # Delta queue: [StreamDelta, ...] — thread-safe
        self._delta_q: queue.Queue[StreamDelta] = queue.Queue()

        # Callbacks (set khi start())
        self._on_add: Optional[Callable[[CameraConfig], None]] = None
        self._on_remove: Optional[Callable[[int], None]] = None
        self._glib_idle_add: Optional[Callable] = None

        # Control flags
        self._running = False
        self._watcher_thread: Optional[threading.Thread] = None
        self._processor_thread: Optional[threading.Thread] = None
        self._observer = None   # watchdog Observer

        # Load lần đầu
        self._load_initial()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        on_add: Callable[[CameraConfig], None],
        on_remove: Callable[[int], None],
        glib_idle_add: Callable,
    ) -> None:
        """
        Khởi động watcher và processor.

        Args:
            on_add:         Gọi khi cần thêm luồng mới vào GStreamer pipeline.
            on_remove:      Gọi khi cần xóa luồng (source_id) khỏi pipeline.
            glib_idle_add:  Hàm GLib.idle_add để đảm bảo GStreamer ops
                            chạy trên GLib Main Loop thread.
        """
        self._on_add = on_add
        self._on_remove = on_remove
        self._glib_idle_add = glib_idle_add
        self._running = True

        # Khởi động processor thread (consume delta_q)
        self._processor_thread = threading.Thread(
            target=self._processor_loop,
            name="CameraManager-Processor",
            daemon=True,
        )
        self._processor_thread.start()

        # Khởi động watchdog observer (inotify)
        self._start_watchdog()
        logger.info("[CameraManager] Started. Watching: %s", self.yml_path)

    def stop(self) -> None:
        """Dừng watcher và processor."""
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
        """Tra cứu CameraConfig theo source_id. Thread-safe."""
        with self._lock:
            return self._by_source_id.get(source_id)

    def get_enabled_configs(self) -> List[CameraConfig]:
        """Trả về danh sách tất cả camera đang enabled."""
        with self._lock:
            return [c for c in self._configs.values() if c.enabled]

    def get_max_streams(self) -> int:
        """Đọc max_streams từ file yml (cached khi init)."""
        return self._max_streams

    def get_tiler_layout(self) -> tuple[int, int]:
        """rows, cols cho nvmultistreamtiler dựa trên số camera enabled."""
        n = len(self.get_enabled_configs())
        return compute_tiler_layout(n)

    # ------------------------------------------------------------------
    # Internal — Load & Diff
    # ------------------------------------------------------------------

    def _load_initial(self) -> None:
        """Load lần đầu, không tạo delta."""
        try:
            raw = yaml.safe_load(self.yml_path.read_text(encoding="utf-8"))
            self._max_streams = int(raw.get("max_streams", 4))
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
        Đọc lại file YAML, so sánh với trạng thái hiện tại.
        Trả về StreamDelta nếu có thay đổi, None nếu không.
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

            # Phát hiện thay đổi URI hoặc config của camera đang chạy
            # → remove rồi re-add để restart luồng
            for sid in set(old_enabled) & set(new_enabled):
                old_c = old_enabled[sid]
                new_c = new_enabled[sid]
                if old_c.uri != new_c.uri or old_c.fps != new_c.fps:
                    logger.info(
                        "[CameraManager] source_id=%d config changed → restart", sid
                    )
                    to_remove_ids.add(sid)
                    to_add_ids.add(sid)

            # Commit state mới
            self._configs = new_configs
            self._rebuild_lookup()

        if not to_add_ids and not to_remove_ids:
            return None

        delta = StreamDelta(
            to_add=[new_enabled[sid] for sid in to_add_ids if sid in new_enabled],
            to_remove=list(to_remove_ids),
        )
        logger.info(
            "[CameraManager] Delta detected — add: %s, remove: %s",
            [c.camera_id for c in delta.to_add],
            delta.to_remove,
        )
        return delta

    def _rebuild_lookup(self) -> None:
        """Rebuild _by_source_id từ _configs. Gọi khi đang giữ lock."""
        self._by_source_id = {
            c.source_id: c
            for c in self._configs.values()
            if c.enabled
        }

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
                        # Debounce 100ms — tránh đọc file khi đang ghi dở
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
        """Gọi khi file thay đổi — tính delta và đẩy vào queue."""
        delta = self._reload_and_diff()
        if delta:
            self._delta_q.put(delta)

    def _polling_loop(self) -> None:
        """Fallback khi watchdog không có: poll file mỗi 1s."""
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
        Consume StreamDelta từ queue và lên lịch GStreamer ops
        thông qua GLib.idle_add để đảm bảo thread safety.
        """
        while self._running:
            try:
                delta = self._delta_q.get(timeout=5.0)
            except queue.Empty:
                continue

            if delta is None:   # stop signal
                break

            # Xóa trước, thêm sau (tránh source_id conflict)
            for source_id in delta.to_remove:
                sid = source_id  # capture for lambda
                if self._glib_idle_add and self._on_remove:
                    self._glib_idle_add(self._on_remove, sid)
                    logger.info(
                        "[CameraManager] Scheduled REMOVE source_id=%d on GLib loop", sid
                    )

            # Chờ GLib một cycle trước khi add (phòng ngừa race condition)
            if delta.to_remove and delta.to_add:
                time.sleep(0.05)

            for cam_cfg in delta.to_add:
                cfg = cam_cfg  # capture for lambda
                if self._glib_idle_add and self._on_add:
                    self._glib_idle_add(self._on_add, cfg)
                    logger.info(
                        "[CameraManager] Scheduled ADD camera=%s source_id=%d on GLib loop",
                        cfg.camera_id, cfg.source_id,
                    )

    # ------------------------------------------------------------------
    # REST API (tuỳ chọn — Giai đoạn 3)
    # ------------------------------------------------------------------

    def start_rest_api(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """
        Khởi động REST API server (FastAPI + uvicorn) trên thread riêng.
        Endpoint:
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
                    # Disable và push delta
                    cfg.enabled = False
                    manager._rebuild_lookup()
                    delta = StreamDelta(to_remove=[cfg.source_id])
                manager._delta_q.put(delta)
                return {"status": "removing", "source_id": cfg.source_id}

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
