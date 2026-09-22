from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web, WSMsgType

_SERVER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SERVER_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=_SERVER_DIR / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s \u2014 %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server_app")

from Server.edge_registry import EdgeRegistry
from Server.violation_store import ViolationStore
from Server.telemetry_store import TelemetryStore
from Server.camera_projection import CameraProjection

"""
Server/app.py — Main Server Entry Point (aiohttp + WebSocket + Zenoh Bridge)

This is the Server backbone. It runs three concurrent services:

1. aiohttp HTTP + WebSocket Server (port 8080):
   - REST API: /api/cameras, /api/nodes, /api/telemetry, /api/violations
   - WebSocket: /ws pushes live dashboard updates (camera topology, FPS, violations)

2. Zenoh Subscriber (peer mode, multicast scouting):
   - Subscribes to peers/status/{node} (Edge heartbeats at 1 Hz)
   - Subscribes to offload/plates/{src}/{dst} (plate-crop offload metadata)
   - Merges heartbeats → CameraProjection.update_from_heartbeat()
   - Merges offload metadata → TelemetryStore.save_async()

3. Background Tasks:
   - EdgeRegistry watchdog: Marks nodes offline after HEARTBEAT_TIMEOUT (30s)
   - TelemetryStore periodic flush (async CSV append)
   - ViolationStore periodic flush

Architecture Principle (constraint #4017):
Server is a PASSIVE Zenoh-to-dashboard/media relay, NOT a central orchestrator.
All ownership, failover, migration, and offload decisions are made AUTONOMOUSLY
by Edge nodes (PeerOrchestrator). Server only projects the resulting topology.

Communication Planes (constraint #4018):
- Video: RTSP via MediaMTX (host networking) → WebRTC/HLS for browsers
- Metadata: Zenoh (peer mode, multicast scouting) for heartbeat, ownership,
  control, failover, offload

Key Components (initialized in main()):
- EdgeRegistry: Node liveness, cluster registry (30s heartbeat timeout)
- CameraProjection: Authoritative camera topology (epoch/boot fencing)
- TelemetryStore: Production CSV recorder (14 cols, per-node, async)
- ViolationStore: Overspeed violation persistence (JSONL, rotated)
- MediaMTX: External process (Docker) — RTSP ingress + WebRTC/HLS egress

Deployment:
- Runs as systemd service (deploy/speedflow-server.service)
- Requires Edge/.env, Server/.env with matching ZENOH_ROUTER
- Docker MediaMTX on host network (veth module absent on Jetson image)
"""
class ServerState:
    def __init__(self) -> None:
        self.registry: EdgeRegistry = None
        self.cameras: CameraProjection = None
        self.browser_ws: List[web.WebSocketResponse] = []
        self.store: ViolationStore = None
        self.telemetry_store: TelemetryStore = None
        self.http_session: aiohttp.ClientSession = None
        self._watchdog_task: asyncio.Task = None
        self._zenoh_session = None
        self._zenoh_sub = None
        self._zenoh_events_sub = None
        self._loop: asyncio.AbstractEventLoop = None

    def broadcast(self, msg: Dict[str, Any]) -> None:
        """Queue a JSON push to all connected browsers (thread-safe)."""
        payload = json.dumps(msg, default=str)
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._send_all(payload))
            )
        else:
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._send_all(payload))
            except RuntimeError:
                logger.warning("[ServerState] broadcast() called outside event loop — message dropped")

    async def _send_all(self, payload: str) -> None:
        dead: List[web.WebSocketResponse] = []
        for ws in list(self.browser_ws):
            try:
                await ws.send_str(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.browser_ws:
                self.browser_ws.remove(ws)

async def index(request: web.Request) -> web.Response:
    html_path = _SERVER_DIR / "static" / "index.html"
    if not html_path.exists():
        return web.Response(text="Dashboard not found", status=404)

    html = html_path.read_text(encoding="utf-8")
    # Derive WS scheme from the request scheme so WSS works behind TLS.
    ws_scheme = "wss" if request.secure else "ws"
    config_json = json.dumps({
        "MEDIAMTX_API": "/api/streams",  # always use the server-side proxy
        "MEDIAMTX_WEBRTC_URL": os.getenv("MEDIAMTX_WEBRTC_URL", ""),
        "WS_URL": f"{ws_scheme}://{request.host}/ws/server",
    })
    html = html.replace("<!-- SERVER_CONFIG -->",
                        f'<script id="server-config" type="application/json">{config_json}</script>')
    return web.Response(text=html, content_type="text/html", charset="utf-8")

async def serve_static(request: web.Request) -> web.Response:
    # BUG-03/12: Validate filename to prevent path traversal (../app.py etc.)
    filename = request.match_info["filename"]
    if PurePath(filename).name != filename:
        return web.Response(text="Invalid filename", status=400)
    filepath = _SERVER_DIR / "static" / filename
    # Resolve symlinks and ensure the result is still inside static/
    static_root = (_SERVER_DIR / "static").resolve()
    try:
        resolved = filepath.resolve()
        resolved.relative_to(static_root)  # raises ValueError if outside
    except (ValueError, OSError):
        return web.Response(text="Not found", status=404)
    if not resolved.exists() or not resolved.is_file():
        return web.Response(text="Not found", status=404)
    return web.FileResponse(resolved)

async def handle_edges(request: web.Request) -> web.Response:
    state: ServerState = request.app["state"]
    return web.json_response(state.registry.get_all())

async def handle_clusters(request: web.Request) -> web.Response:
    state: ServerState = request.app["state"]
    return web.json_response(state.registry.get_clusters())

async def handle_cameras(request: web.Request) -> web.Response:
    state: ServerState = request.app["state"]
    return web.json_response(state.cameras.get_all())

async def handle_violations(request: web.Request) -> web.Response:
    state: ServerState = request.app["state"]
    node_id = request.query.get("node_id") or None
    date = request.query.get("date") or None
    try:
        limit = int(request.query.get("limit", "50"))
    except ValueError:
        limit = 50
    try:
        page = int(request.query.get("page", "0"))
    except ValueError:
        page = 0
    offset = page * limit
    # BUG-8 fix: use query_async() (runs in a thread via asyncio.to_thread)
    # instead of the synchronous query() which blocked the event loop and
    # stalled WebSocket heartbeats during large JSONL reads.
    results = await state.store.query_async(node_id=node_id, date=date, limit=limit, offset=offset)
    return web.json_response(results)

async def handle_snapshot(request: web.Request) -> web.Response:
    node_id = request.match_info["node_id"]
    filename = request.match_info["filename"]

    if PurePath(node_id).name != node_id:
        return web.Response(text="Invalid node_id", status=400)
    if PurePath(filename).name != filename:
        return web.Response(text="Invalid filename", status=400)

    data_dir = _SERVER_DIR / os.getenv("DATA_DIR", "violations")
    if not data_dir.exists():
        return web.Response(text="No data", status=404)

    # BUG-08: use asyncio.to_thread to avoid blocking the event loop while
    # iterating potentially hundreds of date directories synchronously.
    def _find_snapshot() -> Optional[Path]:
        for date_dir in sorted(data_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            snap_path = date_dir / node_id / filename
            if snap_path.exists() and snap_path.is_file():
                return snap_path
        return None

    snap_path = await asyncio.to_thread(_find_snapshot)
    if snap_path is None:
        return web.Response(text="Snapshot not found", status=404)
    return web.FileResponse(snap_path)

async def handle_health_check(request: web.Request) -> web.Response:
    state: ServerState = request.app["state"]
    return web.json_response({
        "status": "ok",
        "edges_registered": len(state.registry.get_all()),
        "edges_online": len(state.registry.get_online()),
        "browsers": len(state.browser_ws),
    })

async def handle_streams(request: web.Request) -> web.Response:
    mtx_api = os.getenv("MEDIAMTX_API", "http://localhost:9997")
    state: ServerState = request.app["state"]
    try:
        async with state.http_session.get(f"{mtx_api}/v3/paths/list") as resp:
            data = await resp.json()
            return web.json_response(data)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)

def _start_zenoh_subscriber(state: ServerState) -> Optional[Any]:
    """
    Start Zenoh peer-mode subscriber for edge status messages.
    Server listens on TCP for edges outside LAN, plus multicast scouting.
    Read-only — Server never sends commands back to edges.
    """
    try:
        import msgpack
        import zenoh

        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"peer"')
        cfg.insert_json5("scouting/multicast/enabled", "true")
        zenoh_listen = os.getenv("ZENOH_LISTEN", "tcp/0.0.0.0:7447")
        if zenoh_listen:
            cfg.insert_json5("listen/endpoints", f'["{zenoh_listen}"]')

        session = zenoh.open(cfg)
        logger.info("[Zenoh Server] Session opened (peer mode, listening on %s)", zenoh_listen)

        def _on_status(sample) -> None:
            try:
                payload = msgpack.unpackb(sample.payload.to_bytes(), raw=False)
            except Exception as exc:
                logger.warning("[Zenoh Server] Failed to unpack status: %s", exc)
                return

            node_id = payload.get("node_id", "")
            if not node_id:
                return

            handle_status(state, payload)

        sub = session.declare_subscriber("peers/status/**", _on_status)
        logger.info("[Zenoh Server] Subscribed to 'peers/status/**'")

        def _on_traffic_event(sample) -> None:
            try:
                payload = msgpack.unpackb(sample.payload.to_bytes(), raw=False)
            except Exception as exc:
                logger.warning("[Zenoh Server] Failed to unpack traffic event: %s", exc)
                return

            if not isinstance(payload, dict):
                return

            # Normalize Edge image_b64 to store snapshot_b64
            if "snapshot_b64" not in payload and "image_b64" in payload:
                payload["snapshot_b64"] = payload.pop("image_b64")

            # Fallback camera_id / node_id from key expr if missing
            key_expr = str(sample.key_expr)
            key_parts = key_expr.split("/")
            if "node_id" not in payload and len(key_parts) >= 3:
                payload["node_id"] = key_parts[2]
            if "camera_id" not in payload and len(key_parts) >= 4:
                payload["camera_id"] = key_parts[3]

            def _save_task(rec=payload):
                asyncio.create_task(state.store.save_async(rec))

            if state.store and state._loop and state._loop.is_running():
                state._loop.call_soon_threadsafe(_save_task)

        events_sub = session.declare_subscriber("traffic/events/**", _on_traffic_event)
        logger.info("[Zenoh Server] Subscribed to 'traffic/events/**'")

        state._zenoh_session = session
        state._zenoh_sub = sub
        state._zenoh_events_sub = events_sub
        return session
    except Exception as exc:
        logger.warning("[Zenoh Server] Failed to start Zenoh subscriber: %s", exc)
        return None

def handle_status(state: ServerState, payload: Dict[str, Any]) -> None:
    """Process one edge status message over the actual app path.

    NODE_ONLINE (register) is the only transition that re-arms a node that
    has been swept offline. A plain health frame from an already-offline node
    is dropped by the registry and must not resurrect its camera rows.
    """
    node_id = payload.get("node_id", "")
    if not node_id:
        return

    event = payload.get("event")
    if event == "NODE_ONLINE":
        ip = payload.get("advertise_ip", "")
        is_new = state.registry.register(node_id, ip)
        state.cameras.on_node_online(node_id)
        if is_new:
            state.broadcast({"type": "edge_registered", "node_id": node_id, "ip": ip})
        return

    # Normal health update. The registry rejects it (returns False) when the
    # node is already offline, so a stale/buffered frame cannot re-arm the
    # registry or the projection. Online nodes update normally.
    applied = state.registry.update_health(node_id, payload)
    if not applied:
        return
    state.cameras.apply_health(node_id, payload)
    health_msg = {**payload, "type": "health_update", "node_id": node_id}
    state.broadcast(health_msg)
    # handle_status runs on the Zenoh C thread, not the event loop — create
    # the task via the loop (same BUG pattern as the traffic handler above).
    if state.telemetry_store is not None and state._loop and state._loop.is_running():
        def _save_telemetry(node_id=node_id, payload=payload):
            asyncio.create_task(state.telemetry_store.save_async(node_id, payload))
        state._loop.call_soon_threadsafe(_save_telemetry)

async def handle_ws_server(request: web.Request) -> web.WebSocketResponse:
    state: ServerState = request.app["state"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    state.browser_ws.append(ws)
    client_idx = len(state.browser_ws)
    logger.info("[WS] Browser #%d connected", client_idx)

    try:
        await ws.send_str(json.dumps({
            "type": "init",
            "edges": state.registry.get_all(),
            "clusters": state.registry.get_clusters(),
        }))
    except Exception:
        pass

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get("action") == "list_edges":
                        await ws.send_str(json.dumps({
                            "type": "edge_list",
                            "edges": state.registry.get_all(),
                        }))
                    elif data.get("action") == "ping":
                        pass
                except json.JSONDecodeError:
                    pass
            elif msg.type == WSMsgType.ERROR:
                logger.warning("[WS] Browser #%d error: %s", client_idx, ws.exception())
    finally:
        if ws in state.browser_ws:
            state.browser_ws.remove(ws)
        logger.info("[WS] Browser #%d disconnected", client_idx)

    return ws

def create_app() -> web.Application:
    state = ServerState()
    data_dir = os.getenv("DATA_DIR", "violations")
    state.store = ViolationStore(_SERVER_DIR / data_dir)
    state.telemetry_store = TelemetryStore(_SERVER_DIR / "data" / "telemetry")

    def on_registry_change(event: str, node_id: str) -> None:
        if event == "offline":
            state.cameras.on_node_offline(node_id)
            state.broadcast({"type": "edge_offline", "node_id": node_id})

    state.registry = EdgeRegistry(on_change=on_registry_change)
    state.cameras = CameraProjection()

    app = web.Application()
    app["state"] = state

    async def on_startup(app: web.Application) -> None:
        state._loop = asyncio.get_running_loop()
        state.http_session = aiohttp.ClientSession()
        # BUG-05: store the task reference so GC cannot collect a pending task
        state._watchdog_task = state.registry.start_watchdog()
        logger.info("[Server] Watchdog started, HTTP session created")
        _start_zenoh_subscriber(state)

    async def on_shutdown(app: web.Application) -> None:
        if state._watchdog_task and not state._watchdog_task.done():
            state._watchdog_task.cancel()
        if state.http_session and not state.http_session.closed:
            await state.http_session.close()
        if state._zenoh_sub:
            try:
                state._zenoh_sub.undeclare()
            except Exception:
                pass
        if state._zenoh_events_sub:
            try:
                state._zenoh_events_sub.undeclare()
            except Exception:
                pass
        if state._zenoh_session:
            try:
                state._zenoh_session.close()
            except Exception:
                pass
        logger.info("[Server] All connections closed")

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/", index)
    app.router.add_get("/static/{filename}", serve_static)
    app.router.add_get("/api/edges", handle_edges)
    app.router.add_get("/api/clusters", handle_clusters)
    app.router.add_get("/api/cameras", handle_cameras)
    app.router.add_get("/api/violations", handle_violations)
    app.router.add_get("/api/streams", handle_streams)
    app.router.add_get("/api/snapshots/{node_id}/{filename}", handle_snapshot)
    app.router.add_get("/ws/server", handle_ws_server)
    app.router.add_get("/health", handle_health_check)

    return app

def main() -> None:
    parser = argparse.ArgumentParser(description="IoT Graduate \u2014 Central Monitoring Server")
    parser.add_argument("--host", default=os.getenv("SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", "9090")))
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO").upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Use DEBUG to see every health payload from each edge.",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    logger.info("=" * 55)
    logger.info("  IoT Graduate \u2014 Central Monitoring Server")
    logger.info("  Dashboard: http://%s:%d", args.host, args.port)
    logger.info("  Health:    http://%s:%d/health", args.host, args.port)
    logger.info("  Data dir:  %s", os.getenv("DATA_DIR", "violations"))
    logger.info("=" * 55)

    app = create_app()

    async def _start() -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.port, reuse_address=True)
        await site.start()
        logger.info("Server listening on %s:%d (SO_REUSEADDR)", args.host, args.port)

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)
        await stop_event.wait()
        await runner.cleanup()

    try:
        asyncio.run(_start())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
