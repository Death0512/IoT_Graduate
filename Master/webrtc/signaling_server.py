#!/usr/bin/env python3
"""
signaling_server.py — WebRTC Signaling Server cho Multi-Edge Architecture.

Vị trí: Chạy trên Master Node.
Vai trò: "Mai mối" (Signaling) kết nối WebRTC giữa Browser (Người xem) và Edge Nodes (Camera).

Kiến trúc:
  - edges: dict[edge_id → websocket] — Lưu các kết nối WebSocket từ Edge Node
  - clients: list[websocket]         — Lưu các kết nối từ Browser
  - Khi Browser gửi {"action":"view","target_edge":"edge_01"}, Server forward SDP Offer
    đến đúng Edge và chuyển tiếp Answer ngược lại.

Cổng mặc định: 8080 (TCP) — lắng nghe trên tất cả interface (0.0.0.0)
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from aiohttp import web, WSMsgType

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("signaling_server")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
INDEX_HTML = SCRIPT_DIR / "index.html"

# ---------------------------------------------------------------------------
# Global State — Quản lý kết nối
# ---------------------------------------------------------------------------
# edges:  edge_id → {"ws": websocket, "registered_at": float}
edges: dict[str, dict] = {}

# clients: danh sách Browser WebSocket đang mở
clients: list[web.WebSocketResponse] = []


# ---------------------------------------------------------------------------
# WebSocket Handler — Điểm vào chính
# ---------------------------------------------------------------------------
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    """
    Xử lý kết nối WebSocket từ Edge hoặc Browser.

    Phân biệt bằng query param `role`:
      - ?role=edge&edge_id=jetson_A  → Edge Node
      - ?role=browser                → Browser (Người xem)
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    role = request.query.get("role", "browser")
    edge_id = request.query.get("edge_id", "")

    if role == "edge" and edge_id:
        await _handle_edge(ws, edge_id)
    else:
        await _handle_browser(ws)

    return ws


# ---------------------------------------------------------------------------
# Edge Handler
# ---------------------------------------------------------------------------
async def _handle_edge(ws: web.WebSocketResponse, edge_id: str) -> None:
    """Xử lý kết nối từ Edge Node (Camera + AI Pipeline)."""
    edges[edge_id] = {"ws": ws, "registered_at": asyncio.get_event_loop().time()}
    logger.info("[EDGE] '%s' đã kết nối. Tổng edges: %d", edge_id, len(edges))

    # Thông báo cho tất cả Browser đang kết nối cập nhật danh sách Edge
    await _broadcast_edge_list()

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = msg.data
                try:
                    msg_obj = json.loads(data)
                    msg_type = msg_obj.get("type", "unknown")
                    logger.info("[EDGE] %s → signaling: %s", edge_id, msg_type)
                except json.JSONDecodeError:
                    pass

                # Forward SDP Answer từ Edge → Browser (tất cả client đang xem)
                await _forward_to_viewer(edge_id, data)

            elif msg.type == WSMsgType.ERROR:
                logger.error("[EDGE] %s WS error: %s", edge_id, ws.exception())
    finally:
        edges.pop(edge_id, None)
        logger.info("[EDGE] '%s' đã ngắt kết nối. Còn lại: %d", edge_id, len(edges))
        await _broadcast_edge_list()


# ---------------------------------------------------------------------------
# Browser Handler
# ---------------------------------------------------------------------------
async def _handle_browser(ws: web.WebSocketResponse) -> None:
    """Xử lý kết nối từ Browser (Người xem)."""
    clients.append(ws)
    client_idx = len(clients)
    logger.info("[BROWSER] Client #%d đã kết nối. Tổng browsers: %d", client_idx, len(clients))

    # Gửi ngay danh sách Edge hiện có cho Browser
    await _send_edge_list(ws)

    # Theo dõi edge mà browser này đang xem
    viewing_edge: str | None = None

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    msg_obj = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                action = msg_obj.get("action", "")
                msg_type = msg_obj.get("type", "")

                # --- Browser muốn xem một Edge cụ thể ---
                if action == "view":
                    target_edge = msg_obj.get("target_edge", "")
                    if target_edge:
                        viewing_edge = target_edge
                        logger.info("[BROWSER] Client #%d muốn xem edge '%s'", client_idx, target_edge)
                        # Forward toàn bộ message (có thể chứa SDP Offer hoặc chỉ là yêu cầu view)
                        await _forward_to_edge(target_edge, msg.data)

                # --- Browser gửi ICE candidate hoặc SDP Answer ---
                elif viewing_edge and msg_type in ("offer", "ice", "answer"):
                    await _forward_to_edge(viewing_edge, msg.data)

                # --- Browser yêu cầu danh sách Edge ---
                elif action == "list_edges":
                    await _send_edge_list(ws)

            elif msg.type == WSMsgType.ERROR:
                logger.error("[BROWSER] Client #%d WS error: %s", client_idx, ws.exception())
    finally:
        if ws in clients:
            clients.remove(ws)
        logger.info("[BROWSER] Client #%d đã ngắt kết nối. Còn lại: %d", client_idx, len(clients))


# ---------------------------------------------------------------------------
# Helpers — Forwarding
# ---------------------------------------------------------------------------
async def _forward_to_edge(edge_id: str, data: str) -> None:
    """Chuyển tiếp thông điệp từ Browser đến Edge cụ thể."""
    edge_entry = edges.get(edge_id)
    if edge_entry is None:
        logger.warning("[FORWARD] Edge '%s' không tồn tại. Không thể forward.", edge_id)
        return
    edge_ws = edge_entry["ws"]
    try:
        await edge_ws.send_str(data)
    except Exception as exc:
        logger.error("[FORWARD] Lỗi gửi đến edge '%s': %s", edge_id, exc)


async def _forward_to_viewer(edge_id: str, data: str) -> None:
    """Chuyển tiếp thông điệp từ Edge đến TẤT CẢ Browser đang kết nối."""
    dead_clients = []
    for ws in clients:
        try:
            await ws.send_str(data)
        except Exception:
            dead_clients.append(ws)

    # Dọn dẹp client đã chết
    for ws in dead_clients:
        if ws in clients:
            clients.remove(ws)


# ---------------------------------------------------------------------------
# Helpers — Edge List Broadcast
# ---------------------------------------------------------------------------
async def _send_edge_list(ws: web.WebSocketResponse) -> None:
    """Gửi danh sách Edge đang online cho một Browser cụ thể."""
    edge_list = [
        {"edge_id": eid, "online": True}
        for eid in edges.keys()
    ]
    try:
        await ws.send_str(json.dumps({
            "type": "edge_list",
            "edges": edge_list,
        }))
    except Exception as exc:
        logger.error("[EDGE_LIST] Lỗi gửi đến browser: %s", exc)


async def _broadcast_edge_list() -> None:
    """Broadcast danh sách Edge mới nhất đến TẤT CẢ Browser."""
    edge_list = [
        {"edge_id": eid, "online": True}
        for eid in edges.keys()
    ]
    message = json.dumps({"type": "edge_list", "edges": edge_list})
    dead_clients = []
    for ws in clients:
        try:
            await ws.send_str(message)
        except Exception:
            dead_clients.append(ws)

    for ws in dead_clients:
        if ws in clients:
            clients.remove(ws)


# ---------------------------------------------------------------------------
# HTTP Handlers
# ---------------------------------------------------------------------------
async def index(request: web.Request) -> web.Response:
    """Phục vụ giao diện Web (index.html)."""
    if not INDEX_HTML.exists():
        return web.Response(
            text=f"ERROR: index.html not found at {INDEX_HTML}", status=404
        )
    return web.FileResponse(INDEX_HTML)


async def health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({
        "status": "ok",
        "edges": len(edges),
        "clients": len(clients),
    })


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------
def create_app() -> web.Application:
    """Tạo aiohttp Application với các route."""
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/health", health)
    return app


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    HOST = os.environ.get("SIGNALING_HOST", "0.0.0.0")
    PORT = int(os.environ.get("SIGNALING_PORT", "8080"))

    logger.info("=" * 55)
    logger.info("  WebRTC Signaling Server — Multi-Edge")
    logger.info("  Host: %s", HOST)
    logger.info("  Port: %s", PORT)
    logger.info("  Index: %s", INDEX_HTML)
    logger.info("=" * 55)

    app = create_app()
    web.run_app(app, host=HOST, port=PORT)
