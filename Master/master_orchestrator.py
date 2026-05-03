#!/usr/bin/env python3
"""
Master/master_orchestrator.py

Master Orchestrator — Bộ não điều phối Multi-Edge Coordination.

Chức năng:
  - Subscribe MQTT để theo dõi trạng thái tất cả Worker Nodes (Jetson Edge)
  - Duy trì Global State Table in-memory chứa Load Score, FPS, cameras của từng node
  - Tự động phát hiện node quá tải và ra quyết định chuyển tải (Load Balancing)
  - Thực thi chiến lược Make-before-Break khi di chuyển camera giữa các node
  - Xử lý Timeout & Rollback khi node đích không phản hồi
  - Ghi log migration vào CSV để phục vụ báo cáo thực nghiệm

Chiến lược Make-before-Break:
  1. Publish lệnh ADD vào edge/control/{node_target} — khởi tạo camera trên node nhẹ
  2. Chờ tín hiệu "PLAYING" từ node target (timeout 15s) trước khi xóa ở node cũ
  3. Publish lệnh REMOVE vào edge/control/{node_overloaded}
  4. Giai đoạn giao thời ~1-2s: 2 node cùng xử lý camera → Deduplication ở consumer

Topics MQTT:
  Subscribe: edge/status/+       (nhận metrics từ tất cả Worker Nodes)
  Subscribe: edge/status/+       (nhận xác nhận ADD/REMOVE từ Worker)
  Publish:   edge/control/{id}   (gửi lệnh ADD/REMOVE đến Worker cụ thể)

Biến môi trường:
  MQTT_BROKER_HOST      (mặc định: localhost)
  MQTT_BROKER_PORT      (mặc định: 1883)
  MQTT_USER / MQTT_PASS (tuỳ chọn)
  OVERLOAD_THRESHOLD    (mặc định: 85.0 — Load Score %)
  TARGET_THRESHOLD      (mặc định: 45.0 — Load Score %)
  OVERLOAD_DURATION_S   (mặc định: 15 — giây liên tục quá tải để trigger)
  MIGRATION_TIMEOUT_S   (mặc định: 15 — giây chờ xác nhận PLAYING)
  COOLDOWN_S            (mặc định: 45 — giây sau mỗi lần offload)
  MIN_FPS_THRESHOLD     (mặc định: 18 — FPS tối thiểu)

Yêu cầu:
  pip install paho-mqtt
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("master_orchestrator")

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------

BROKER_HOST         = os.environ.get("MQTT_BROKER_HOST", "localhost")
BROKER_PORT         = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
MQTT_USER           = os.environ.get("MQTT_USER", None)
MQTT_PASS           = os.environ.get("MQTT_PASS", None)

OVERLOAD_THRESHOLD  = float(os.environ.get("OVERLOAD_THRESHOLD", "85.0"))
TARGET_THRESHOLD    = float(os.environ.get("TARGET_THRESHOLD", "45.0"))
OVERLOAD_DURATION_S = float(os.environ.get("OVERLOAD_DURATION_S", "15.0"))
MIGRATION_TIMEOUT_S = float(os.environ.get("MIGRATION_TIMEOUT_S", "15.0"))
COOLDOWN_S          = float(os.environ.get("COOLDOWN_S", "45.0"))
MIN_FPS_THRESHOLD   = float(os.environ.get("MIN_FPS_THRESHOLD", "18.0"))

LOG_DIR  = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "orchestrator.csv"

CAMERA_CONFIGS_DIR = Path(__file__).parent.parent / "Edge" / "configs"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NodeState:
    """Trạng thái hiện tại của một Worker Node."""
    node_id: str
    api_url: str = ""              # Không dùng nữa (MQTT C2), giữ để tham chiếu
    load_score: float = 0.0
    gpu_percent: float = 0.0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    gpu_temp_c: float = 0.0
    avg_fps: Optional[float] = None
    fps_per_camera: Dict[str, float] = field(default_factory=dict)
    active_cameras: List[str] = field(default_factory=list)
    max_streams: int = 4
    last_seen: float = field(default_factory=time.time)

    # Metadata dùng cho quyết định Orchestrator
    overload_since: Optional[float] = None   # thời điểm bắt đầu quá tải
    last_migration: float = 0.0              # thời điểm migration gần nhất
    penalty_until: float = 0.0              # đánh dấu node lỗi tạm thời


# ---------------------------------------------------------------------------
# Migration Log
# ---------------------------------------------------------------------------

class MigrationLogger:
    """Ghi log mỗi lần migration ra file CSV để phân tích sau."""

    HEADER = [
        "timestamp_iso", "from_node", "to_node", "camera_id",
        "trigger_reason", "trigger_load", "trigger_fps",
        "migration_time_ms", "result",
    ]

    def __init__(self, log_file: Path) -> None:
        self._path = log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not log_file.exists():
            with open(log_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)

    def log(
        self,
        from_node: str,
        to_node: str,
        camera_id: str,
        trigger_reason: str,
        trigger_load: float,
        trigger_fps: Optional[float],
        migration_time_ms: float,
        result: str,
    ) -> None:
        row = [
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            from_node, to_node, camera_id,
            trigger_reason,
            round(trigger_load, 1),
            round(trigger_fps, 1) if trigger_fps is not None else "",
            round(migration_time_ms, 0),
            result,
        ]
        try:
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as exc:
            logger.warning("MigrationLogger write error: %s", exc)


# ---------------------------------------------------------------------------
# Master Orchestrator
# ---------------------------------------------------------------------------

class MasterOrchestrator:
    """
    Lõi điều phối tập trung. Chạy vòng lặp MQTT để:
      1. Theo dõi trạng thái tất cả Worker Nodes
      2. Ra quyết định Load Balancing
      3. Thực thi di chuyển camera theo chiến lược Make-before-Break
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeState] = {}
        self._lock = threading.Lock()

        # Sự kiện chờ xác nhận Make-before-Break
        # key: camera_id, value: threading.Event
        self._playing_events: Dict[str, threading.Event] = {}

        self._migration_log = MigrationLogger(LOG_FILE)
        self._client = None
        self._running = False
        self._decision_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Khởi động Orchestrator (blocking)."""
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client(client_id="master_orchestrator")
        if MQTT_USER:
            self._client.username_pw_set(MQTT_USER, MQTT_PASS)

        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

        self._running = True

        # Khởi động Decision Engine trong thread riêng
        self._decision_thread = threading.Thread(
            target=self._decision_loop,
            name="DecisionEngine",
            daemon=True,
        )
        self._decision_thread.start()

        logger.info("Connecting to broker %s:%d ...", BROKER_HOST, BROKER_PORT)
        while self._running:
            try:
                self._client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
                self._client.loop_forever()
            except Exception as exc:
                logger.error("Broker connection error: %s. Retrying in 5s...", exc)
                time.sleep(5)

    def stop(self) -> None:
        self._running = False
        if self._client:
            self._client.disconnect()

    # ------------------------------------------------------------------
    # MQTT callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc != 0:
            logger.error("MQTT connect failed, rc=%d", rc)
            return
        logger.info("Connected to MQTT broker. Subscribing...")
        # Lắng nghe trạng thái của TẤT CẢ Worker Nodes
        client.subscribe("edge/status/+", qos=1)
        logger.info("Subscribed to: edge/status/+")

    def _on_disconnect(self, client, userdata, rc) -> None:
        if rc != 0:
            logger.warning("Unexpected disconnect (rc=%d). Reconnecting...", rc)

    def _on_message(self, client, userdata, msg) -> None:
        """Xử lý mọi message từ Worker Nodes."""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return

        topic = msg.topic
        event = payload.get("event", "")
        node_id = payload.get("node_id", "")

        # --- Cập nhật Global State Table từ health metrics ---
        if "load_score" in payload:
            self._update_node_state(payload)
            return

        # --- Xử lý xác nhận từ Worker (Make-before-Break flow) ---
        if event == "ADD_PROCESSING":
            cam_id = payload.get("camera_id", "")
            logger.info("[M-b-B] Node '%s' đang khởi tạo camera '%s'...", node_id, cam_id)

        elif event in ("STREAM_PLAYING", "ADD_COMPLETED"):
            cam_id = payload.get("camera_id", "")
            logger.info("[M-b-B] Node '%s' xác nhận camera '%s' PLAYING.", node_id, cam_id)
            # Giải phóng Event để migration thread tiếp tục bước REMOVE
            ev = self._playing_events.get(cam_id)
            if ev:
                ev.set()

        elif event == "ADD_FAILED":
            cam_id = payload.get("camera_id", "")
            reason = payload.get("reason", "unknown")
            logger.error("[M-b-B] Node '%s' thất bại khi ADD camera '%s': %s", node_id, cam_id, reason)
            # Giải phóng Event để migration thread biết thất bại
            ev = self._playing_events.get(cam_id)
            if ev:
                ev.set()   # migration thread sẽ kiểm tra node state

        elif event == "NODE_ONLINE":
            logger.info("Node '%s' vừa online. Cameras: %s", node_id, payload.get("active_cameras", []))

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

    def _update_node_state(self, payload: dict) -> None:
        """Cập nhật trạng thái node từ health metrics publish."""
        node_id = payload.get("node_id", "")
        if not node_id:
            return

        pipeline = payload.get("pipeline", {})
        fps_per_cam = pipeline.get("fps_per_camera", {})
        avg_fps = pipeline.get("avg_fps")
        active_cams = pipeline.get("active_cameras", [])

        with self._lock:
            if node_id not in self._nodes:
                self._nodes[node_id] = NodeState(node_id=node_id)
                logger.info("Discovered new node: '%s'", node_id)

            node = self._nodes[node_id]
            node.load_score     = float(payload.get("load_score", 0))
            node.gpu_percent    = float(payload.get("gpu_percent", 0))
            node.cpu_percent    = float(payload.get("cpu_percent", 0))
            node.ram_percent    = float(payload.get("ram_percent", 0))
            node.gpu_temp_c     = float(payload.get("gpu_temp_c", 0))
            node.avg_fps        = avg_fps
            node.fps_per_camera = fps_per_cam
            node.active_cameras = active_cams
            node.last_seen      = time.time()

            now = time.time()
            is_overloaded = (
                node.load_score > OVERLOAD_THRESHOLD or
                (node.avg_fps is not None and node.avg_fps < MIN_FPS_THRESHOLD)
            )
            if is_overloaded:
                if node.overload_since is None:
                    node.overload_since = now
            else:
                node.overload_since = None  # Reset nếu đã ổn định

    # ------------------------------------------------------------------
    # Decision Engine
    # ------------------------------------------------------------------

    def _decision_loop(self) -> None:
        """
        Vòng lặp kiểm tra định kỳ (mỗi 5s) và ra quyết định offload.
        Chạy trong thread riêng — không block MQTT loop.
        """
        logger.info("[Decision] Engine started. Checking every 5s.")
        while self._running:
            time.sleep(5.0)
            try:
                self._evaluate_and_act()
            except Exception as exc:
                logger.error("[Decision] Unexpected error: %s", exc)

    def _evaluate_and_act(self) -> None:
        """Kiểm tra trạng thái và trigger migration nếu cần."""
        now = time.time()

        with self._lock:
            nodes_snapshot = {nid: n for nid, n in self._nodes.items()}

        for node_id, node in nodes_snapshot.items():
            # Bỏ qua node mất kết nối (>15s không nhận tin)
            if now - node.last_seen > 15:
                if node.active_cameras:
                    logger.critical("[Decision] Node '%s' OFFLINE with %d cameras! Triggering fail-over...",
                                   node_id, len(node.active_cameras))
                    self._trigger_failover(node, nodes_snapshot)
                    with self._lock:
                        if node_id in self._nodes:
                            self._nodes[node_id].active_cameras = []
                else:
                    logger.warning("[Decision] Node '%s' offline (last seen %.0fs ago).",
                                   node_id, now - node.last_seen)
                continue

            # Cooldown: không xử lý nếu vừa migration xong
            if now - node.last_migration < COOLDOWN_S:
                continue

            # Kiểm tra tình trạng quá tải liên tục
            if node.overload_since is None:
                continue
            overload_duration = now - node.overload_since
            if overload_duration < OVERLOAD_DURATION_S:
                continue

            # Node đã quá tải đủ lâu → tìm camera nhẹ nhất để di chuyển
            trigger_reason = "fps_drop" if (node.avg_fps and node.avg_fps < MIN_FPS_THRESHOLD) else "load_score"
            logger.warning(
                "[Decision] Node '%s' OVERLOADED for %.0fs! "
                "Load=%.1f%%, FPS=%s. Reason: %s",
                node_id, overload_duration, node.load_score,
                node.avg_fps, trigger_reason,
            )

            cam_to_offload = self._pick_camera_to_offload(node)
            if not cam_to_offload:
                logger.warning("[Decision] Node '%s' has no camera to offload.", node_id)
                continue

            target_node = self._pick_target_node(nodes_snapshot, node_id)
            if not target_node:
                logger.warning("[Decision] No available target node for offloading from '%s'.", node_id)
                continue

            logger.info(
                "[Decision] Migrating camera '%s' from '%s' → '%s'",
                cam_to_offload, node_id, target_node.node_id,
            )

            # Thực thi migration trong thread riêng (không block Decision loop)
            threading.Thread(
                target=self._execute_migration,
                args=(node, target_node, cam_to_offload, trigger_reason),
                daemon=True,
            ).start()

            # Cập nhật timestamp migration ngay để tránh trigger lại trong cooldown
            with self._lock:
                if node_id in self._nodes:
                    self._nodes[node_id].last_migration = now
                if target_node.node_id in self._nodes:
                    self._nodes[target_node.node_id].last_migration = now

    def _pick_camera_to_offload(self, node: NodeState) -> Optional[str]:
        """
        Chọn camera 'nhẹ nhất' để di chuyển khỏi node quá tải.
        Ưu tiên camera có FPS cao nhất (tải cao) → di chuyển sẽ giảm tải nhiều nhất.
        Nếu không có FPS data → chọn camera cuối trong danh sách.
        """
        if not node.active_cameras:
            return None
        if node.fps_per_camera:
            # Chọn camera đang chiếm FPS nhiều nhất
            return max(
                (c for c in node.active_cameras if c in node.fps_per_camera),
                key=lambda c: node.fps_per_camera.get(c, 0),
                default=node.active_cameras[-1],
            )
        return node.active_cameras[-1]

    def _pick_target_node(self, nodes: Dict[str, NodeState], exclude_id: str) -> Optional[NodeState]:
        """
        Chọn node nhẹ nhất để nhận camera mới.
        Điều kiện: load_score < TARGET_THRESHOLD, còn chỗ, không đang bị penalty.
        """
        now = time.time()
        candidates = [
            n for nid, n in nodes.items()
            if nid != exclude_id
            and n.load_score < TARGET_THRESHOLD
            and len(n.active_cameras) < n.max_streams
            and now - n.last_seen <= 15    # node phải online
            and now >= n.penalty_until     # không bị phạt
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.load_score)

    def _trigger_failover(self, dead_node: NodeState, nodes_snapshot: Dict[str, NodeState]) -> None:
        """
        Gán lại các camera của node đã chết sang các node khác đang sống.
        Lưu ý: Bỏ qua bước REMOVE ở node cũ (vì nó đã chết).
        """
        for cam_id in dead_node.active_cameras:
            target_node = self._pick_target_node(nodes_snapshot, dead_node.node_id)
            if not target_node:
                logger.error("[Fail-over] NO AVAILABLE NODE to rescue camera '%s' from '%s'.", cam_id, dead_node.node_id)
                continue
            
            # Cập nhật tạm thời snapshot để hàm _pick_target_node chạy đúng cho camera tiếp theo
            target_node.active_cameras.append(cam_id)
            
            logger.info("[Fail-over] Rescuing camera '%s' from dead node '%s' → '%s'", cam_id, dead_node.node_id, target_node.node_id)
            
            # Thực thi fail-over trong thread riêng
            threading.Thread(
                target=self._execute_failover,
                args=(dead_node, target_node, cam_id),
                daemon=True,
            ).start()

    def _execute_failover(
        self,
        dead_node: NodeState,
        target_node: NodeState,
        camera_id: str,
    ) -> None:
        """Thực thi fail-over (chỉ ADD vào node mới, không REMOVE từ node cũ)."""
        start_ms = time.time() * 1000
        
        cam_config = self._get_camera_config(camera_id)
        if not cam_config:
            logger.error("[Fail-over] Cannot get config for camera '%s'. Aborting.", camera_id)
            return

        add_cmd = {**cam_config, "cmd": "ADD"}
        self._publish_command(target_node.node_id, add_cmd)
        logger.info("[Fail-over] ADD command sent to node '%s' for camera '%s'.", target_node.node_id, camera_id)

        self._migration_log.log(
            dead_node.node_id, target_node.node_id, camera_id,
            "node_offline", 0.0, 0.0,
            time.time() * 1000 - start_ms, "FAILOVER_ADD",
        )

    # ------------------------------------------------------------------
    # Migration Execution (Make-before-Break)
    # ------------------------------------------------------------------

    def _execute_migration(
        self,
        from_node: NodeState,
        to_node: NodeState,
        camera_id: str,
        trigger_reason: str,
    ) -> None:
        """
        Thực thi luân chuyển camera theo chiến lược Make-before-Break:
          1. ADD camera trên node target
          2. Chờ xác nhận PLAYING (timeout 15s)
          3. REMOVE camera trên node nguồn
          4. Ghi log
        """
        start_ms = time.time() * 1000
        trigger_load = from_node.load_score
        trigger_fps  = from_node.avg_fps

        logger.info(
            "[Migration] START: '%s' | %s → %s | Trigger: %s (Load=%.1f%%)",
            camera_id, from_node.node_id, to_node.node_id,
            trigger_reason, trigger_load,
        )

        # --- BƯỚC 1: Lấy cấu hình camera để gửi sang node mới ---
        cam_config = self._get_camera_config(camera_id)
        if not cam_config:
            logger.error("[Migration] Cannot get config for camera '%s'. Aborting.", camera_id)
            self._migration_log.log(
                from_node.node_id, to_node.node_id, camera_id,
                trigger_reason, trigger_load, trigger_fps,
                0, "ABORTED_NO_CONFIG",
            )
            return

        # --- BƯỚC 2: Gửi lệnh ADD vào node target ---
        playing_event = threading.Event()
        self._playing_events[camera_id] = playing_event

        add_cmd = {**cam_config, "cmd": "ADD"}
        self._publish_command(to_node.node_id, add_cmd)
        logger.info("[Migration] ADD command sent to node '%s'.", to_node.node_id)

        # --- BƯỚC 3: Chờ xác nhận PLAYING từ node target ---
        confirmed = playing_event.wait(timeout=MIGRATION_TIMEOUT_S)
        self._playing_events.pop(camera_id, None)

        if not confirmed:
            # Timeout: Rollback — đánh dấu node target bị penalty
            logger.error(
                "[Migration] TIMEOUT (%ds) waiting for PLAYING from node '%s'. "
                "Rolling back — marking node as faulty.",
                int(MIGRATION_TIMEOUT_S), to_node.node_id,
            )
            with self._lock:
                if to_node.node_id in self._nodes:
                    self._nodes[to_node.node_id].penalty_until = (
                        time.time() + COOLDOWN_S * 2   # Phạt 2× cooldown
                    )
            self._migration_log.log(
                from_node.node_id, to_node.node_id, camera_id,
                trigger_reason, trigger_load, trigger_fps,
                time.time() * 1000 - start_ms, "TIMEOUT_ROLLBACK",
            )
            return

        # --- BƯỚC 4: Gửi lệnh REMOVE vào node nguồn ---
        remove_cmd = {"cmd": "REMOVE", "camera_id": camera_id}
        self._publish_command(from_node.node_id, remove_cmd)
        logger.info(
            "[Migration] REMOVE command sent to node '%s'. Camera '%s' migrated successfully.",
            from_node.node_id, camera_id,
        )

        elapsed_ms = time.time() * 1000 - start_ms
        self._migration_log.log(
            from_node.node_id, to_node.node_id, camera_id,
            trigger_reason, trigger_load, trigger_fps,
            elapsed_ms, "SUCCESS",
        )
        logger.info(
            "[Migration] DONE in %.0fms: '%s' | %s → %s",
            elapsed_ms, camera_id, from_node.node_id, to_node.node_id,
        )

    def _publish_command(self, node_id: str, payload: dict) -> None:
        """Publish lệnh điều khiển xuống Worker Node qua MQTT."""
        topic = f"edge/control/{node_id}"
        try:
            self._client.publish(topic, json.dumps(payload), qos=1)
            logger.debug("[Publish] %s → %s", topic, payload.get("cmd"))
        except Exception as exc:
            logger.error("[Publish] Failed to publish to '%s': %s", topic, exc)

    # ------------------------------------------------------------------
    # Camera Config Lookup
    # ------------------------------------------------------------------

    def _get_camera_config(self, camera_id: str) -> Optional[dict]:
        """
        Lấy cấu hình camera để gửi kèm lệnh ADD sang node mới.
        Đọc từ file cameras.yml trong thư mục Edge/configs.
        """
        try:
            import yaml
            yml_path = CAMERA_CONFIGS_DIR / "cameras.yml"
            with open(yml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            cameras = raw.get("cameras", {})
            cfg = cameras.get(camera_id)
            if not cfg:
                return None
            # Đóng gói lại thành định dạng lệnh ADD
            return {
                "camera_id":       camera_id,
                "source_id":       int(cfg.get("source_id", 0)),
                "uri":             cfg.get("uri", ""),
                "name":            cfg.get("name", camera_id),
                "fps":             float(cfg.get("fps", 25.0)),
                "speed_limit_kmh": float(cfg.get("speed_limit_kmh", 80.0)),
                "homography":      cfg.get("homography", {}),
                "roi_polygon":     cfg.get("roi_polygon", []),
                "output":          cfg.get("output", {}),
            }
        except Exception as exc:
            logger.error("Failed to load camera config for '%s': %s", camera_id, exc)
            return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import paho.mqtt.client
    except ImportError:
        logger.error("paho-mqtt not installed. Run: pip install paho-mqtt")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  Multi-Edge Master Orchestrator")
    logger.info("  Broker: %s:%d", BROKER_HOST, BROKER_PORT)
    logger.info("  Overload threshold: %.0f%% (duration: %.0fs)",
                OVERLOAD_THRESHOLD, OVERLOAD_DURATION_S)
    logger.info("  Target threshold:   %.0f%%", TARGET_THRESHOLD)
    logger.info("  Cooldown:           %.0fs", COOLDOWN_S)
    logger.info("  Migration timeout:  %.0fs", MIGRATION_TIMEOUT_S)
    logger.info("  Log file:           %s", LOG_FILE)
    logger.info("=" * 60)

    orchestrator = MasterOrchestrator()
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Shutting down Orchestrator...")
        orchestrator.stop()
