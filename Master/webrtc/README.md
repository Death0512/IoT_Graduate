# WebRTC Streaming Component (Master side)

This folder contains the **WebRTC signaling server** that enables low‑latency video streaming from an Edge node to any web browser.

## How It Works

1. The Edge node runs a DeepStream pipeline with `--mode webrtc`.  
   It connects to the signaling server via WebSocket and acts as a **publisher** (offering SDP).
2. One or more **viewers** open a browser page that also connects to the same signaling server (as a **viewer**).
3. The server exchanges SDP offers/answers and ICE candidates between the publisher and viewers.
4. Once the connection is established, the video flows directly from the Edge node to the browsers (peer‑to‑peer, VP8 codec).

## Starting the Signaling Server

```bash
cd ~/IoT_Graduate/Master/webrtc
python3 signaling_server.py
```

By default it listens on `0.0.0.0:8080` (WebSocket).  
You can change the port with `--port 9090` if needed.

## Using WebRTC Mode on the Edge Node

On the Edge node, run:

```bash
python3 main.py --backend python \
  --source rtsp://<CAMERA_IP>:8554/cam_01 \
  --mode webrtc \
  --server <SIGNALING_SERVER_IP> \
  --port 8080 \
  --room any_unique_name \
  --cfg configs/config_cam.txt
```

- `--server` : IP address of the machine running the signaling server  
- `--port` : port of the signaling server (default 8080)  
- `--room` : room identifier – all viewers using the same room will see the stream  
- `--cfg` : optional, but recommended to set correct FPS / resolution for WebRTC encoding

## Viewing the Stream in a Browser

Open the following URL in Chrome, Edge, or Firefox:

```
http://<SIGNALING_SERVER_IP>:8080/?room=any_unique_name
```

The page will automatically request camera/microphone permissions (not needed) and start receiving the VP8 video.  
**Overspeed alerts** are also sent via the same WebSocket connection – they appear as JSON messages in the browser console and can be used to display violation popups.

## Example Workflow

1. **Server** (e.g. 192.168.1.100):
   ```bash
   cd ~/IoT_Graduate/Master/webrtc
   python3 signaling_server.py
   ```

2. **Edge node** (e.g. 192.168.1.101):
   ```bash
   export MQTT_BROKER_HOST=192.168.1.100   # Master IP
   export NODE_ID=edge_01
   python3 health_agent.py &   # background
   python3 main.py --backend python --source rtsp://192.168.1.200:8554/cam_01 \
     --mode webrtc --server 192.168.1.100 --port 8080 --room main_road
   ```

3. **User** opens browser: `http://192.168.1.100:8080/?room=main_road` → sees live video with speed overlay and licence plates.

## Notes

- The signaling server is very lightweight (plain WebSocket + JSON). It does **not** relay the video – only the handshake.  
- For production, you may want to run the server behind nginx with SSL (WebRTC requires HTTPS for most browsers).  
- The HTML client (`index.html`) is automatically served by the signaling server when you request the base URL. You can customise it for branded dashboards.