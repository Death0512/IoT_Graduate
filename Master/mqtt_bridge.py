#!/usr/bin/env python3
"""
Master/mqtt_bridge.py

MQTT → WebSocket Bridge dự phòng.

Dùng khi Mosquitto KHÔNG có WebSocket listener (cổng 9001).
Bridge này subscribe MQTT TCP (1883) và forward message lên
WebSocket cho Web UI thông qua asyncio WebSocket server.

Chạy: python3 mqtt_bridge.py
      Sau đó mở Web UI với: ?broker=ws://localhost:9001
      (Bridge lắng nghe trên ws://0.0.0.0:9001)

Yêu cầu:
    pip install paho-mqtt websockets
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading

import paho.mqtt.client as mqtt
import websockets

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("mqtt_bridge")

BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
WS_PORT     = int(os.environ.get("BRIDGE_WS_PORT", "9001"))
TOPICS      = ["traffic/events/#", "edge/status/#"]

# Thread-safe queue giữa MQTT thread và asyncio WebSocket loop
_msg_queue: queue.Queue = queue.Queue(maxsize=500)
# Tập hợp tất cả WebSocket clients đang kết nối
_clients: set = set()
_clients_lock = threading.Lock()


def _mqtt_thread() -> None:
    """Chạy MQTT client trong thread riêng, đẩy messages vào _msg_queue."""
    client = mqtt.Client(client_id="mqtt_ws_bridge")

    def on_connect(c, u, f, rc):
        if rc == 0:
            logger.info("MQTT connected. Subscribing to: %s", TOPICS)
            for topic in TOPICS:
                c.subscribe(topic, qos=0)
        else:
            logger.error("MQTT connect failed, rc=%d", rc)

    def on_message(c, u, msg):
        try:
            _msg_queue.put_nowait({
                "topic":   msg.topic,
                "payload": msg.payload.decode("utf-8"),
            })
        except queue.Full:
            pass  # Drop nếu queue đầy

    client.on_connect = on_connect
    client.on_message = on_message

    while True:
        try:
            client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
            client.loop_forever()
        except Exception as exc:
            logger.error("MQTT error: %s. Retrying in 5s...", exc)
            import time; time.sleep(5)


async def _broadcast_loop() -> None:
    """Lấy messages từ queue và broadcast cho tất cả WebSocket clients."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            msg = await loop.run_in_executor(None, _msg_queue.get, True, 1.0)
        except Exception:
            continue

        frame = json.dumps(msg)
        with _clients_lock:
            dead = set()
            for ws in list(_clients):
                try:
                    await ws.send(frame)
                except Exception:
                    dead.add(ws)
            _clients -= dead


async def _ws_handler(ws) -> None:
    """Đăng ký WebSocket client mới và giữ kết nối."""
    with _clients_lock:
        _clients.add(ws)
    logger.info("WebSocket client connected. Total: %d", len(_clients))
    try:
        async for _ in ws:
            pass  # Không cần nhận dữ liệu từ client
    finally:
        with _clients_lock:
            _clients.discard(ws)
        logger.info("WebSocket client disconnected. Total: %d", len(_clients))


async def _main() -> None:
    threading.Thread(target=_mqtt_thread, daemon=True).start()
    asyncio.create_task(_broadcast_loop())
    logger.info("WebSocket bridge listening on ws://0.0.0.0:%d", WS_PORT)
    async with websockets.serve(_ws_handler, "0.0.0.0", WS_PORT):
        await asyncio.Future()  # chạy mãi mãi


if __name__ == "__main__":
    asyncio.run(_main())
