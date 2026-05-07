# 🚀 IoT_Graduate – Multi‑Edge Traffic Monitoring & Coordination System

## 1. System Overview
**IoT_Graduate** is a distributed, real‑time traffic monitoring system that uses **AI (NVIDIA DeepStream)** to measure vehicle speed, detect vehicles, and read license plates.  
Its key innovation is **multi‑edge load balancing** with **make‑before‑break** migration: when an Edge node becomes overloaded (low FPS, high GPU/CPU), the Master orchestrator automatically moves cameras to other nodes without interrupting video analysis.

All coordination between Master and Edge nodes uses **MQTT** for lightweight, reliable messaging. The processing pipeline supports two backends:
- **Python** – easy to customise and debug
- **C++** – maximum performance on constrained hardware (e.g. Jetson)

## 2. Component Overview

| Component | Folder | Role |
|-----------|--------|------|
| **Camera** | `Camera/` | Simulates RTSP IP cameras using Docker + MediaMTX. Pushes looped video streams. |
| **Master** | `Master/` | Central orchestrator: monitors Edge metrics via MQTT, decides when to migrate cameras, and runs an MQTT broker. |
| **Edge** | `Edge/` | AI processing node (NVIDIA Jetson). Runs DeepStream pipelines for speed measurement, LPR, and overspeed alerts. Reports health to Master. |

## 3. Prerequisites

- **Camera node**: Docker & Docker Compose
- **Master node**: Ubuntu 20.04/22.04, Python 3.8+, Mosquitto MQTT broker
- **Edge node**: NVIDIA Jetson (Orin/NX/Nano) with JetPack 6.x and DeepStream SDK 7.x

## 4. Detailed Startup Instructions

### A. Start the Camera Node (RTSP simulator)

```bash
cd ~/IoT_Graduate/Camera

# Place video files (e.g. cam_01.mp4, cam_02.mp4) into ./videos/
chmod +x generate-compose.sh start.sh

# Generate docker-compose.yml based on videos/ content
./generate-compose.sh 4    # 4 = number of camera streams

# Launch RTSP server
docker-compose up -d
```

After startup, you will have RTSP streams available at:
```
rtsp://<CAMERA_NODE_IP>:8554/cam_01
rtsp://<CAMERA_NODE_IP>:8554/cam_02
...
```

### B. Start the Master Node (Orchestrator + MQTT Broker)

```bash
cd ~/IoT_Graduate/Master

# Install required Python packages
pip3 install paho-mqtt pyyaml

# Install and start Mosquitto (MQTT broker)
sudo apt update && sudo apt install mosquitto mosquitto-clients -y
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# (Optional) Edit orchestrator.yml to adjust overload thresholds / FPS limits

# Run the orchestrator
export MQTT_BROKER_HOST="localhost"   # IP of your MQTT broker
export OVERLOAD_THRESHOLD="85.0"       # percentage, e.g. 85% GPU usage
python3 master_orchestrator.py
```

The orchestrator now listens on MQTT topics `edge/status/+` and publishes commands to `edge/command/<node_id>`.

### C. Start an Edge Node (Jetson with DeepStream)

#### 1. Prepare the environment
```bash
cd ~/IoT_Graduate/Edge
chmod +x setup_system.sh
./setup_system.sh
pip3 install -r requirements.txt
```

#### 2. Run the Health Agent (reports metrics to Master)
```bash
export MQTT_BROKER_HOST="<IP_OF_MASTER_NODE>"
export NODE_ID="worker_jetson_01"
python3 health_agent.py
```

#### 3. Run the DeepStream pipeline

**Display mode (HDMI output)**
```bash
python3 main.py --backend python \
  --source rtsp://<CAMERA_NODE_IP>:8554/cam_01 \
  --mode display \
  --homo configs/points_rtsp.yml
```

**File mode (save to MP4)**
```bash
python3 main.py --backend cpp \
  --source rtsp://<CAMERA_NODE_IP>:8554/cam_01 \
  --mode file --output result.mp4 \
  --homo configs/points_rtsp.yml
```

**WebRTC mode (stream to browser)**
First start the signaling server (on Master or another machine – see `Master/webrtc/README.md`):
```bash
cd ~/IoT_Graduate/Master/webrtc
python3 signaling_server.py
```

Then on the Edge node:
```bash
python3 main.py --backend python \
  --source rtsp://<CAMERA_NODE_IP>:8554/cam_01 \
  --mode webrtc \
  --server <SIGNALING_SERVER_IP> --port 8080 --room demo \
  --cfg configs/config_cam.txt
```

Open a browser at `http://<SIGNALING_SERVER_IP>:8080/?room=demo` to view the live stream.

## 5. Load Balancing in Action

Once multiple Edge nodes are running and sending metrics, the Master will:
- Detect an overloaded node (e.g. GPU >85% or FPS drop below threshold)
- Choose a less loaded node
- Send an `ADD` command to the target node and a `REMOVE` command to the overloaded node (make‑before‑break)
- The Edge node will dynamically add or remove the RTSP stream without restarting the pipeline

You can observe migration logs in the Master terminal.