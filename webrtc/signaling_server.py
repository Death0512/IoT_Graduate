# signaling_server.py
import asyncio, json, os
from aiohttp import web, WSMsgType
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
INDEX_HTML = SCRIPT_DIR / "index.html"

# Lưu kết nối WS theo room: {room_name: {ws_conn, role, ...}}
ROOMS = {}

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    room = request.query.get("room", "demo")
    role = request.query.get("role", "unknown")
    
    # Track connections với role info
    if room not in ROOMS:
        ROOMS[room] = {"connections": set(), "publishers": 0, "subscribers": 0}
    
    ROOMS[room]["connections"].add(ws)
    if role == "pub":
        ROOMS[room]["publishers"] += 1
    elif role == "sub":
        ROOMS[room]["subscribers"] += 1
    
    total_peers = len(ROOMS[room]["connections"])
    print(f"[SRV] {role} joined room={room}, total peers={total_peers}")
    
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
                
                for p in list(ROOMS[room]["connections"]):
                    if p is not ws:
                        await p.send_str(data)
            elif msg.type == WSMsgType.ERROR:
                print("ws error:", ws.exception())
    finally:
        ROOMS[room]["connections"].discard(ws)
        if role == "pub":
            ROOMS[room]["publishers"] -= 1
        elif role == "sub":
            ROOMS[room]["subscribers"] -= 1
            
        remaining = len(ROOMS[room]["connections"])
        print(f"[SRV] {role} left room={room}, remaining={remaining}")
        
        if remaining == 0:
            ROOMS.pop(room, None)
    return ws

async def index(request):
    if not INDEX_HTML.exists():
        return web.Response(text=f"ERROR: index.html not found at {INDEX_HTML}", status=404)
    return web.FileResponse(INDEX_HTML)

async def api_rooms(request):
    """API endpoint to get list of active rooms with publishers"""
    active_rooms = []
    for room_name, room_data in ROOMS.items():
        if room_data["publishers"] > 0:  # Only include rooms with active publishers
            active_rooms.append({
                "name": room_name,
                "publishers": room_data["publishers"],
                "subscribers": room_data["subscribers"]
            })
    return web.json_response({"rooms": active_rooms})

app = web.Application()
app.router.add_get('/', index)
app.router.add_get('/ws', ws_handler)
app.router.add_get('/api/rooms', api_rooms)  # New API endpoint

if __name__ == "__main__":
    print(f"[SRV] Serving index.html from: {INDEX_HTML}")
    web.run_app(app, host="0.0.0.0", port=8080)
