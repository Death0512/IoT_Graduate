#!/usr/bin/env python3
"""
speedflow_python/mqtt_publisher.py

MQTT Publisher cho Worker Node (Jetson Edge).

Chịu trách nhiệm xuất dữ liệu AI (vi phạm tốc độ, biển số) ra ngoài
theo kiến trúc non-blocking: SpeedProbe chỉ đặt data vào Queue,
một Thread riêng sẽ lấy ra và gửi MQTT.

Thiết kế phi chặn (Non-blocking):
    - SpeedProbe gọi `publisher.put(data)` (< 0.1ms, không chờ mạng).
    - Thread nội bộ `_publish_loop` tiêu thụ Queue và gửi MQTT thực sự.
    - Queue có kích thước tối đa (maxsize=1000) phòng tràn RAM khi mạng hỏng.
      Nếu Queue đầy → drop tin cũ (block=False) để pipeline không bao giờ bị khoá.

Topics:
    traffic/events/{node_id}/{camera_id}  — vi phạm tốc độ, biển số
    edge/status/{node_id}                 — trạng thái node (dùng health_agent)

Yêu cầu:
    pip install paho-mqtt
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Kích thước tối đa Queue.
# ~1000 sự kiện × ~2KB mỗi sự kiện ≈ 2MB RAM tối đa khi mạng sập hoàn toàn.
_QUEUE_MAXSIZE = int(os.environ.get("MQTT_QUEUE_MAXSIZE", "1000"))


class MQTTPublisher:
    """
    Non-blocking MQTT Publisher dành cho GStreamer probe.

    Cách dùng:
        publisher = MQTTPublisher(node_id="jetson_A", broker_host="192.168.1.10")
        publisher.start()

        # Trong SpeedProbe (30fps callback — không được block):
        publisher.put({"camera_id": "cam_01", "speed_kmh": 92.5, ...})

        publisher.stop()
    """

    def __init__(
        self,
        node_id: str,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tls_ca_cert: Optional[str] = None,
        tls_certfile: Optional[str] = None,
        tls_keyfile: Optional[str] = None,
        keepalive: int = 60,
    ) -> None:
        self._node_id = node_id
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._username = username
        self._password = password
        self._tls_ca_cert = tls_ca_cert
        self._tls_certfile = tls_certfile
        self._tls_keyfile = tls_keyfile
        self._keepalive = keepalive

        # Queue phi chặn — kích thước tối đa phòng OOM khi mạng gián đoạn
        self._queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(maxsize=_QUEUE_MAXSIZE)

        self._client = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False

        # Thống kê (hữu ích để debug)
        self._sent_count = 0
        self._drop_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Khởi động publisher thread."""
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client(client_id=f"pub_{self._node_id}")

        if self._username:
            self._client.username_pw_set(self._username, self._password)

        if self._tls_ca_cert:
            self._client.tls_set(
                ca_certs=self._tls_ca_cert,
                certfile=self._tls_certfile,
                keyfile=self._tls_keyfile,
            )
            self._client.tls_insecure_set(False)

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        self._running = True
        self._thread = threading.Thread(
            target=self._publish_loop,
            name=f"MQTTPublisher-{self._node_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Dừng publisher, flush Queue còn lại trước khi thoát."""
        self._running = False
        # Đặt sentinel để unblock Queue.get()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=10)
        if self._client:
            self._client.disconnect()
        logger.info(
            "[MQTTPublisher] Stopped. Sent=%d, Dropped=%d",
            self._sent_count, self._drop_count,
        )

    def put(self, data: Dict[str, Any]) -> None:
        """
        Đưa data vào Queue để gửi đi (NON-BLOCKING).

        Nếu Queue đầy (mạng sập lâu), data CŨ NHẤT sẽ bị drop
        để nhường chỗ cho data mới hơn và pipeline không bị treo.

        Luôn an toàn để gọi từ bất kỳ thread nào (kể cả GStreamer Probe thread).
        """
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            # Queue đầy → drop data cũ nhất, thêm data mới vào
            try:
                self._queue.get_nowait()   # loại bỏ phần tử cũ nhất
                self._queue.put_nowait(data)
                self._drop_count += 1
                if self._drop_count % 100 == 1:
                    logger.warning(
                        "[MQTTPublisher] Queue full (%d slots). "
                        "Dropping oldest events (total dropped: %d). "
                        "Check network connectivity.",
                        _QUEUE_MAXSIZE, self._drop_count,
                    )
            except queue.Empty:
                pass  # Race condition hiếm gặp — bỏ qua

    def publish_event(self, data: Dict[str, Any]) -> None:
        """Alias của put() — tương thích với interface cũ."""
        self.put(data)

    # ------------------------------------------------------------------
    # Internal — publish loop (chạy trên thread riêng)
    # ------------------------------------------------------------------

    def _publish_loop(self) -> None:
        """
        Vòng lặp chính: kết nối Broker và tiêu thụ Queue để gửi MQTT.
        Tự động kết nối lại khi mất kết nối.
        """
        retry_delay = 5
        while self._running:
            # Kết nối (non-blocking loop_start)
            try:
                self._client.connect(
                    self._broker_host,
                    self._broker_port,
                    keepalive=self._keepalive,
                )
                self._client.loop_start()
            except Exception as exc:
                logger.error(
                    "[MQTTPublisher] Cannot connect to %s:%d — %s. "
                    "Retrying in %ds...",
                    self._broker_host, self._broker_port, exc, retry_delay,
                )
                time.sleep(retry_delay)
                continue

            # Tiêu thụ Queue
            while self._running:
                try:
                    item = self._queue.get(timeout=2.0)
                except queue.Empty:
                    continue

                if item is None:  # stop sentinel
                    break

                if not self._connected:
                    # Broker chưa kết nối — đặt lại vào Queue và chờ
                    try:
                        self._queue.put_nowait(item)
                    except queue.Full:
                        self._drop_count += 1
                    time.sleep(0.5)
                    continue

                self._send(item)

            # Dừng vòng lặp MQTT nếu còn chạy
            try:
                self._client.loop_stop()
            except Exception:
                pass
            break

    def _send(self, data: Dict[str, Any]) -> None:
        """Serialize và publish một sự kiện lên MQTT Broker."""
        try:
            camera_id = data.get("camera_id", "unknown")
            topic = f"traffic/events/{self._node_id}/{camera_id}"
            payload = json.dumps(data, ensure_ascii=False)
            result = self._client.publish(topic, payload, qos=1)
            if result.rc == 0:
                self._sent_count += 1
            else:
                logger.debug("[MQTTPublisher] Publish rc=%d (topic=%s)", result.rc, topic)
        except Exception as exc:
            logger.warning("[MQTTPublisher] Send error: %s", exc)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            logger.info(
                "[MQTTPublisher] Connected to broker %s:%d",
                self._broker_host, self._broker_port,
            )
        else:
            logger.error("[MQTTPublisher] Connection failed, rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        if rc != 0:
            logger.warning(
                "[MQTTPublisher] Unexpected disconnect (rc=%d). "
                "Events will queue up until reconnected.",
                rc,
            )
