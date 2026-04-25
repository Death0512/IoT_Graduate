# signaling_server.py
import asyncio, json, os
from aiohttp import web, WSMsgType
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
INDEX_HTML = SCRIPT_DIR / "index.html"

# Lưu kết nối WS theo room
ROOMS = {}
async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    room = request.query.get("room", "demo")
    role = request.query.get("role", "unknown")
    peers = ROOMS.setdefault(room, set())
    peers.add(ws)
    print(f"[SRV] {role} joined room={room}, total peers={len(peers)}")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                # Broadcast cho các peer khác cùng room
                data = msg.data
                try:
                    msg_obj = json.loads(data)
                    msg_type = msg_obj.get("type", "unknown")
                    print(f"[SRV] room={room} {role} → * : {msg_type}")
                except:
                    pass
                
                for p in list(peers):
                    if p is not ws:
                        await p.send_str(data)
            elif msg.type == WSMsgType.ERROR:
                print("ws error:", ws.exception())
    finally:
        peers.discard(ws)
        print(f"[SRV] {role} left room={room}, remaining={len(peers)}")
        if not peers:
            ROOMS.pop(room, None)
    return ws

async def index(request):
    if not INDEX_HTML.exists():
        return web.Response(text=f"ERROR: index.html not found at {INDEX_HTML}", status=404)
    return web.FileResponse(INDEX_HTML)

app = web.Application()
app.router.add_get('/', index)
app.router.add_get('/ws', ws_handler)

if __name__ == "__main__":
    print(f"[SRV] Serving index.html from: {INDEX_HTML}")
    web.run_app(app, host="0.0.0.0", port=8080)
