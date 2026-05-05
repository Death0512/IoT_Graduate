#!/usr/bin/env python3
"""
webrtc_client.py — WebRTC Client cho Edge Node (Camera + AI Pipeline).

Vị trí: Chạy trên mỗi Edge Node (Jetson).
Vai trò: "Người gửi" (WebRTC Peer) — kết nối đến Master Signaling Server,
         đăng ký định danh, và gửi luồng video WebRTC khi có Browser yêu cầu.

Luồng hoạt động:
  1. Kết nối WebSocket đến Master Signaling Server (ws://<master_ip>:8080/ws?role=edge&edge_id=<id>)
  2. Gửi gói register để định danh: {"type":"register","role":"edge","edge_id":"<id>"}
  3. Chờ Browser gửi SDP Offer (qua Master forward)
  4. Tạo SDP Answer và gửi ngược lại
  5. Trao đổi ICE candidates
  6. Gửi video track qua WebRTC PeerConnection

Yêu cầu:
  pip install aiortc websockets
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import websockets

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("webrtc_client")

# ---------------------------------------------------------------------------
# Cấu hình — lấy từ biến môi trường hoặc edge_node.yml
# ---------------------------------------------------------------------------
MASTER_HOST = os.environ.get("SIGNALING_HOST", "192.168.1.100")
MASTER_PORT = int(os.environ.get("SIGNALING_PORT", "8080"))
EDGE_ID     = os.environ.get("EDGE_ID", "jetson_A")

# ---------------------------------------------------------------------------
# WebRTC imports — chỉ import nếu dùng thực tế
# ---------------------------------------------------------------------------
try:
    from aiortc import (
        RTCPeerConnection,
        RTCSessionDescription,
        VideoStreamTrack,
    )
    from aiortc.contrib.media import MediaRelay
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    logger.warning("aiortc chưa cài đặt. WebRTC streaming sẽ không hoạt động.")
    logger.warning("Cài đặt: pip install aiortc")


# ---------------------------------------------------------------------------
# WebRTC Signaling Client
# ---------------------------------------------------------------------------

class EdgeWebRTCClient:
    """
    WebRTC Client cho Edge Node.

    Kết nối đến Master Signaling Server, đăng ký edge_id,
    và xử lý các yêu cầu WebRTC từ Browser.
    """

    def __init__(
        self,
        master_host: str = MASTER_HOST,
        master_port: int = MASTER_PORT,
        edge_id: str = EDGE_ID,
    ) -> None:
        self._master_host = master_host
        self._master_port = master_port
        self._edge_id     = edge_id
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._pc: Optional[RTCPeerConnection] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Kết nối đến Master Signaling Server và duy trì kết nối."""
        uri = f"ws://{self._master_host}:{self._master_port}/ws?role=edge&edge_id={self._edge_id}"
        self._running = True

        while self._running:
            try:
                async with websockets.connect(
                    uri,
                    ping_interval=30,
                    ping_timeout=10,
                ) as websocket:
                    self._ws = websocket
                    logger.info("[Edge] Đã kết nối đến Master Signaling: %s", uri)

                    # Gửi gói định danh (register)
                    await self._register()

                    # Lắng nghe thông điệp từ Master
                    await self._message_loop(websocket)

            except websockets.ConnectionClosed as exc:
                logger.warning("[Edge] Mất kết nối Signaling: %s. Thử lại sau 5s...", exc)
            except OSError as exc:
                logger.error("[Edge] Lỗi mạng: %s. Thử lại sau 5s...", exc)
            except Exception as exc:
                logger.error("[Edge] Lỗi không mong đợi: %s. Thử lại sau 5s...", exc)

            if self._running:
                await asyncio.sleep(5)

    async def disconnect(self) -> None:
        """Ngắt kết nối WebSocket và đóng PeerConnection."""
        self._running = False
        await self._cleanup_peerconnection()
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("[Edge] Đã ngắt kết nối khỏi Master Signaling.")

    # ------------------------------------------------------------------
    # Internal — Signaling
    # ------------------------------------------------------------------

    async def _register(self) -> None:
        """Gửi gói tin định danh đến Master Signaling Server."""
        if not self._ws:
            return
        register_msg = json.dumps({
            "type": "register",
            "role": "edge",
            "edge_id": self._edge_id,
        })
        await self._ws.send(register_msg)
        logger.info("[Edge] Đã gửi register: '%s'", self._edge_id)

    async def _message_loop(self, websocket) -> None:
        """Vòng lặp xử lý thông điệp từ Master Signaling Server."""
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("[Edge] Nhận được message không phải JSON.")
                continue

            msg_type = msg.get("type", "")
            action   = msg.get("action", "")

            logger.info("[Edge] Nhận từ signaling: type=%s, action=%s", msg_type, action)

            # --- Browser muốn xem Edge này ---
            if action == "view" or msg_type == "offer":
                target_edge = msg.get("target_edge", "")

                # Chỉ xử lý nếu target_edge là chính mình
                if target_edge and target_edge != self._edge_id:
                    continue

                if msg_type == "offer":
                    await self._handle_offer(msg)
                else:
                    # Browser gửi yêu cầu view — Edge chủ động tạo offer
                    # (Browser thường gửi offer, nên trường hợp này ít xảy ra)
                    logger.info("[Edge] Browser yêu cầu xem, chờ Browser gửi SDP Offer...")

            # --- ICE candidate từ Browser ---
            elif msg_type == "ice":
                await self._handle_ice(msg)

            # --- Message khác ---
            else:
                logger.debug("[Edge] Message không xử lý: %s", msg_type)

    # ------------------------------------------------------------------
    # Internal — WebRTC (SDP + ICE)
    # ------------------------------------------------------------------

    async def _handle_offer(self, msg: dict) -> None:
        """Xử lý SDP Offer từ Browser (forward qua Master)."""
        if not AIORTC_AVAILABLE:
            logger.error("[Edge] aiortc chưa cài đặt. Không thể xử lý SDP Offer.")
            return

        sdp = msg.get("sdp", "")
        if not sdp:
            logger.warning("[Edge] Offer không có SDP.")
            return

        logger.info("[Edge] Nhận SDP Offer từ Browser.")

        # Tạo PeerConnection mới
        await self._cleanup_peerconnection()
        self._pc = RTCPeerConnection()

        @self._pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            """Gửi ICE candidate lên Master (Master sẽ forward đến Browser)."""
            if self._ws and candidate:
                await self._ws.send(json.dumps({
                    "type": "ice",
                    "candidate": {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    },
                }))
                logger.debug("[Edge] Đã gửi ICE candidate.")

        @self._pc.on("connectionstatechange")
        async def on_connection_state():
            state = self._pc.connectionState
            logger.info("[Edge] WebRTC connection state: %s", state)

        # Thêm video track (cần được implement bởi pipeline cụ thể)
        # TODO: Thay thế bằng track từ GStreamer / Jetson camera pipeline
        # Hiện tại dùng VideoStreamTrack rỗng để test kết nối
        try:
            # Placeholder: pipeline thực tế sẽ cung cấp VideoStreamTrack
            # from your_pipeline import get_video_track
            # video_track = get_video_track()
            # self._pc.addTrack(video_track)
            logger.info("[Edge] Chưa có video track thực tế. Cần tích hợp GStreamer/aiortc track.")
        except Exception as exc:
            logger.error("[Edge] Lỗi khi thêm video track: %s", exc)

        # Set remote description (Offer) và tạo Answer
        try:
            offer = RTCSessionDescription(sdp=sdp, type="offer")
            await self._pc.setRemoteDescription(offer)

            answer = await self._pc.createAnswer()
            await self._pc.setLocalDescription(answer)

            # Gửi Answer lên Master
            if self._ws:
                await self._ws.send(json.dumps({
                    "type": "answer",
                    "sdp": self._pc.localDescription.sdp,
                }))
                logger.info("[Edge] Đã gửi SDP Answer.")
        except Exception as exc:
            logger.error("[Edge] Lỗi khi xử lý SDP: %s", exc)
            await self._cleanup_peerconnection()

    async def _handle_ice(self, msg: dict) -> None:
        """Xử lý ICE candidate từ Browser."""
        if not self._pc:
            return
        candidate = msg.get("candidate", {})
        if not candidate:
            return
        try:
            from aiortc import RTCIceCandidate
            ice = RTCIceCandidate(
                component=1,  # RTP
                foundation=candidate.get("foundation", ""),
                ip=candidate.get("ip", ""),
                port=candidate.get("port", 0),
                priority=candidate.get("priority", 0),
                protocol=candidate.get("protocol", "udp"),
                type=candidate.get("type", "host"),
            )
            await self._pc.addIceCandidate(ice)
            logger.debug("[Edge] Đã thêm ICE candidate từ Browser.")
        except Exception as exc:
            logger.error("[Edge] Lỗi thêm ICE candidate: %s", exc)

    async def _cleanup_peerconnection(self) -> None:
        """Đóng PeerConnection cũ nếu có."""
        if self._pc:
            try:
                await self._pc.close()
            except Exception:
                pass
            self._pc = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config_from_yml(yml_path: str) -> dict:
    """Đọc cấu hình từ file edge_node.yml."""
    try:
        import yaml
        with open(yml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return {
            "master_host": cfg.get("signaling", {}).get("master_host", MASTER_HOST),
            "master_port": int(cfg.get("signaling", {}).get("master_port", MASTER_PORT)),
            "edge_id": cfg.get("node_id", EDGE_ID),
        }
    except Exception:
        return {
            "master_host": MASTER_HOST,
            "master_port": MASTER_PORT,
            "edge_id": EDGE_ID,
        }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Entry point cho Edge WebRTC Client."""
    logger.info("=" * 55)
    logger.info("  Edge WebRTC Client — Multi-Edge")
    logger.info("  Edge ID:   %s", EDGE_ID)
    logger.info("  Master:    %s:%d", MASTER_HOST, MASTER_PORT)
    logger.info("=" * 55)

    # Thử load từ edge_node.yml nếu có
    config_dir = Path(__file__).parent.parent / "configs" / "edge_node.yml"
    if config_dir.exists():
        cfg = load_config_from_yml(str(config_dir))
        master_host = cfg["master_host"]
        master_port = cfg["master_port"]
        edge_id = cfg["edge_id"]
        logger.info("  Loaded config from: %s", config_dir)
    else:
        master_host = MASTER_HOST
        master_port = MASTER_PORT
        edge_id = EDGE_ID
        logger.info("  Using env/fallback config.")

    client = EdgeWebRTCClient(
        master_host=master_host,
        master_port=master_port,
        edge_id=edge_id,
    )

    try:
        await client.connect()
    except KeyboardInterrupt:
        logger.info("Shutting down Edge WebRTC Client...")
        await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
