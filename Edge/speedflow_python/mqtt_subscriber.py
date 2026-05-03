#!/usr/bin/env python3
"""
speedflow_python/mqtt_subscriber.py

MQTT Command & Control Subscriber cho Worker Node (Jetson Edge).

Thay thế kiến trúc REST API, giúp Master ra lệnh xuyên tường lửa/NAT.

Topic lắng nghe:
    edge/control/{node_id}

Định dạng lệnh ADD (JSON):
    {
        "cmd": "ADD",
        "camera_id": "cam_03",
        "source_id": 2,
        "uri": "rtsp://...",
        "name": "Intersection North",
        "fps": 25.0,
        "speed_limit_kmh": 60.0,
        "homography": {
            "source_points": [[x,y], ...],   // 4 điểm
            "target_width": 1000,
            "target_height": 600
        },
        "roi_polygon": [x1,y1, x2,y2, ...],  // tuỳ chọn
        "output": {
            "record": false,
            "record_path": "output/cam_03.mp4"
        }
    }

Định dạng lệnh REMOVE (JSON):
    {
        "cmd": "REMOVE",
        "camera_id": "cam_03"
    }

Bảo mật: Hỗ trợ TLS và Username/Password authentication.
         Đặt cert path và credentials qua biến môi trường hoặc tham số.

Yêu cầu:
    pip install paho-mqtt
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

import cv2
import numpy as np

from .camera_config import CameraConfig, CameraManager, StreamDelta

logger = logging.getLogger(__name__)


class MQTTCommandSubscriber:
    """
    Subscribe vào topic MQTT để nhận lệnh điều khiển từ Master Orchestrator.

    Lệnh được nhận trên topic: `edge/control/{node_id}`
    Phản hồi trạng thái được gửi về: `edge/status/{node_id}`

    Thread-safety:
        Khi nhận lệnh, không gọi pipeline trực tiếp. Chỉ push StreamDelta
        vào CameraManager._delta_q. CameraManager Processor Thread sẽ
        lên lịch thực thi qua GLib.idle_add → đảm bảo GStreamer thread-safety.
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        node_id: str,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        # TLS (tuỳ chọn): truyền đường dẫn file cert
        tls_ca_cert: Optional[str] = None,
        tls_certfile: Optional[str] = None,
        tls_keyfile: Optional[str] = None,
        keepalive: int = 60,
    ) -> None:
        self._camera_manager = camera_manager
        self._node_id = node_id
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._username = username
        self._password = password
        self._tls_ca_cert = tls_ca_cert
        self._tls_certfile = tls_certfile
        self._tls_keyfile = tls_keyfile
        self._keepalive = keepalive

        self._control_topic = f"edge/control/{node_id}"
        self._status_topic = f"edge/status/{node_id}"
        self._client = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Khởi động MQTT client trên daemon thread."""
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client(client_id=f"edge_{self._node_id}")

        # Xác thực
        if self._username:
            self._client.username_pw_set(self._username, self._password)

        # TLS (nếu có cấu hình)
        if self._tls_ca_cert:
            self._client.tls_set(
                ca_certs=self._tls_ca_cert,
                certfile=self._tls_certfile,
                keyfile=self._tls_keyfile,
            )
            self._client.tls_insecure_set(False)
            logger.info("[MQTT C2] TLS enabled.")

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"MQTTSubscriber-{self._node_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Dừng MQTT client."""
        self._running = False
        if self._client:
            self._client.disconnect()
        if self._thread:
            self._thread.join(timeout=5)

    def publish_status(self, payload: dict) -> None:
        """Publish trạng thái/event từ Edge lên Broker."""
        if self._client:
            try:
                self._client.publish(
                    self._status_topic,
                    json.dumps(payload),
                    qos=1,
                )
            except Exception as exc:
                logger.warning("[MQTT C2] Failed to publish status: %s", exc)

    # ------------------------------------------------------------------
    # Internal — MQTT loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Chạy vòng lặp kết nối MQTT với cơ chế tự kết nối lại."""
        import paho.mqtt.client as mqtt

        retry_delay = 5  # giây
        while self._running:
            try:
                self._client.connect(
                    self._broker_host,
                    self._broker_port,
                    keepalive=self._keepalive,
                )
                self._client.loop_forever()
            except Exception as exc:
                logger.error(
                    "[MQTT C2] Connection error: %s. Retrying in %ds...",
                    exc, retry_delay,
                )
                # Tự kết nối lại sau khi mất kết nối
                import time
                time.sleep(retry_delay)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            logger.info(
                "[MQTT C2] Connected to broker %s:%d. Subscribing to '%s'",
                self._broker_host, self._broker_port, self._control_topic,
            )
            client.subscribe(self._control_topic, qos=1)
            # Báo hiệu cho Master biết node đã online
            self.publish_status({
                "node_id": self._node_id,
                "event": "NODE_ONLINE",
                "active_cameras": [
                    c.camera_id
                    for c in self._camera_manager.get_enabled_configs()
                ],
            })
        else:
            logger.error("[MQTT C2] Connection failed, rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        if rc != 0:
            logger.warning("[MQTT C2] Unexpected disconnect (rc=%d). Will retry.", rc)

    def _on_message(self, client, userdata, msg) -> None:
        """Xử lý lệnh điều khiển từ Master."""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("[MQTT C2] Invalid JSON received: %s", exc)
            return

        cmd = payload.get("cmd", "").upper()
        logger.info("[MQTT C2] Received command: %s", cmd)

        if cmd == "ADD":
            self._handle_add(payload)
        elif cmd == "REMOVE":
            self._handle_remove(payload)
        elif cmd == "STATUS":
            # Master yêu cầu báo cáo trạng thái ngay lập tức
            self._handle_status_request()
        else:
            logger.warning("[MQTT C2] Unknown command: '%s'", cmd)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_add(self, payload: dict) -> None:
        """
        Xử lý lệnh ADD camera mới.
        Parse payload → CameraConfig → push StreamDelta vào CameraManager.
        """
        try:
            cam_id    = payload["camera_id"]
            source_id = int(payload["source_id"])
            uri       = payload["uri"]

            # Kiểm tra trùng lặp
            existing = self._camera_manager.get_config(source_id)
            if existing and existing.enabled:
                logger.warning(
                    "[MQTT C2] ADD ignored: source_id=%d ('%s') already active.",
                    source_id, existing.camera_id,
                )
                self.publish_status({
                    "node_id": self._node_id,
                    "event": "ADD_REJECTED",
                    "camera_id": cam_id,
                    "reason": "source_id_conflict",
                })
                return

            # Parse Homography
            homo_cfg   = payload["homography"]
            src_pts    = np.array(homo_cfg["source_points"], dtype=np.float32)
            tw         = int(homo_cfg["target_width"])
            th         = int(homo_cfg["target_height"])
            tgt_pts    = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
            homo_mat, _ = cv2.findHomography(src_pts, tgt_pts)
            if homo_mat is None:
                homo_mat = cv2.getPerspectiveTransform(src_pts, tgt_pts)

            # Parse ROI (tuỳ chọn)
            roi_raw = payload.get("roi_polygon", [])
            roi_arr = np.array(roi_raw, dtype=np.int32).reshape(-1, 2) if roi_raw else np.zeros((0, 2), dtype=np.int32)

            out_cfg = payload.get("output", {})

            cam_cfg = CameraConfig(
                camera_id=cam_id,
                source_id=source_id,
                uri=uri,
                enabled=True,
                name=payload.get("name", cam_id),
                fps=float(payload.get("fps", 25.0)),
                speed_limit_kmh=float(payload.get("speed_limit_kmh", 80.0)),
                source_points=src_pts,
                target_points=tgt_pts,
                homo_matrix=homo_mat,
                roi_polygon=roi_arr,
                record=bool(out_cfg.get("record", False)),
                record_path=str(out_cfg.get("record_path", f"output/{cam_id}.mp4")),
            )

            # Cập nhật trạng thái nội bộ CameraManager
            with self._camera_manager._lock:
                self._camera_manager._configs[cam_id] = cam_cfg
                self._camera_manager._rebuild_lookup()

            # Push delta vào queue để Processor Thread lên lịch GLib.idle_add
            delta = StreamDelta(to_add=[cam_cfg])
            self._camera_manager._delta_q.put(delta)

            logger.info("[MQTT C2] ADD queued: camera_id='%s', source_id=%d", cam_id, source_id)

            # Phản hồi Master — node đang xử lý lệnh ADD
            self.publish_status({
                "node_id": self._node_id,
                "event": "ADD_PROCESSING",
                "camera_id": cam_id,
                "source_id": source_id,
            })

        except KeyError as exc:
            logger.error("[MQTT C2] ADD command missing required field: %s", exc)
            self.publish_status({
                "node_id": self._node_id,
                "event": "ADD_FAILED",
                "reason": f"missing_field_{exc}",
            })
        except Exception as exc:
            logger.error("[MQTT C2] ADD command error: %s", exc)
            self.publish_status({
                "node_id": self._node_id,
                "event": "ADD_FAILED",
                "reason": str(exc),
            })

    def _handle_remove(self, payload: dict) -> None:
        """
        Xử lý lệnh REMOVE camera.
        Tìm source_id từ camera_id, push StreamDelta.
        """
        try:
            cam_id = payload["camera_id"]

            with self._camera_manager._lock:
                cfg = self._camera_manager._configs.get(cam_id)
                if not cfg or not cfg.enabled:
                    logger.warning(
                        "[MQTT C2] REMOVE ignored: camera_id='%s' not active.", cam_id
                    )
                    self.publish_status({
                        "node_id": self._node_id,
                        "event": "REMOVE_REJECTED",
                        "camera_id": cam_id,
                        "reason": "not_active",
                    })
                    return
                source_id = cfg.source_id
                cfg.enabled = False
                self._camera_manager._rebuild_lookup()

            # Push delta
            delta = StreamDelta(to_remove=[source_id])
            self._camera_manager._delta_q.put(delta)

            logger.info(
                "[MQTT C2] REMOVE queued: camera_id='%s', source_id=%d", cam_id, source_id
            )
            self.publish_status({
                "node_id": self._node_id,
                "event": "REMOVE_PROCESSING",
                "camera_id": cam_id,
                "source_id": source_id,
            })

        except KeyError as exc:
            logger.error("[MQTT C2] REMOVE command missing required field: %s", exc)
        except Exception as exc:
            logger.error("[MQTT C2] REMOVE command error: %s", exc)

    def _handle_status_request(self) -> None:
        """Phản hồi yêu cầu STATUS từ Master."""
        active = self._camera_manager.get_enabled_configs()
        self.publish_status({
            "node_id": self._node_id,
            "event": "STATUS_REPORT",
            "active_cameras": [c.camera_id for c in active],
            "active_count": len(active),
        })
