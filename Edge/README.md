# 🚀 IoT_Graduate - Dual Mode Traffic Monitoring System

## Hệ thống Giám sát Giao thông với 2 Backend: Python và C++

---

## 📁 Cấu trúc thư mục

```
IoT_Graduate/
├── main.py                      # Entry point (chọn backend)
├── configs/                     # Config files chung
├── models/                      # TensorRT engines
│
├── speedflow_python/            # 🐍 PYTHON BACKEND
│   ├── core_pipeline.py         # Pipeline builder
│   ├── probes.py               # Speed/Plate logic (Python)
│   ├── homography.py           # Perspective transform
│   ├── plate_preprocessor.py   # Image enhancement
│   ├── settings.py             # Configuration
│   └── run_python.py           # Python mode runner
│
├── speedflow_cpp/               # 🔧 C++ BACKEND
│   ├── CMakeLists.txt          # CMake build config
│   ├── build.sh                # Build script
│   ├── include/
│   │   ├── gst_speedflow.h     # Plugin header
│   │   ├── speed_calculator.h
│   │   ├── homography.h
│   │   └── plate_associator.h
│   ├── src/
│   │   ├── gst_speedflow.cpp   # Main plugin
│   │   ├── speed_calculator.cpp
│   │   ├── homography.cpp
│   │   └── plate_associator.cpp
│   ├── pipeline_cpp.py         # C++ mode runner
│   └── build/                  # Compiled plugin (.so)
│
└── webrtc/                      # WebRTC streaming
```

---

## 🎯 Cách sử dụng

### 0. Nguồn RTSP (Camera / Máy tính khác)

Hệ thống hỗ trợ **bất kỳ nguồn RTSP nào** — camera IP, NVR, OBS, FFmpeg, VLC, v.v.
Chỉ cần thay `video.mp4` bằng URL RTSP tương ứng.

#### Bước 1 – Kiểm tra kết nối RTSP
```bash
# Test xem RTSP có hoạt động không (cần gst-launch)
gst-launch-1.0 uridecodebin uri="rtsp://192.168.1.100:8554/stream" ! autovideosink

# Hoặc dùng ffplay
ffplay rtsp://192.168.1.100:8554/stream
```

#### Bước 2 – Calibrate homography cho camera mới
```bash
# Lấy 1 frame để chọn 4 điểm ROI:
ffmpeg -i rtsp://192.168.1.100:8554/stream -frames:v 1 frame_calib.jpg

# Mở ảnh, chọn 4 góc vùng đo, sửa configs/points_rtsp.yml
# SOURCE: 4 điểm pixel trên ảnh camera
# TARGET: kích thước thực (mét) của vùng đó ngoài đường
```

Ví dụ `configs/points_rtsp.yml`:
```yaml
SOURCE:
- [960,  400]   # top-left
- [1280, 400]   # top-right
- [1800, 900]   # bottom-right
- [400,  900]   # bottom-left
TARGET_WIDTH: 12    # 3 làn × 4m
TARGET_HEIGHT: 30   # đoạn đo 30m
TARGET:
- [0, 0]
- [12, 0]
- [12, 30]
- [0, 30]
```

#### Bước 3 – Chạy với RTSP

**Display mode** (xem trực tiếp trên màn hình):
```bash
python3 main.py --backend python \
  --source rtsp://192.168.1.100:8554/stream \
  --mode display \
  --homo configs/points_rtsp.yml
```

**File mode** (ghi ra file MP4):
```bash
python3 main.py --backend python \
  --source rtsp://192.168.1.100:8554/stream \
  --mode file \
  --output output/result.mp4 \
  --homo configs/points_rtsp.yml
```

**WebRTC mode** (stream lên browser):
```bash
# Terminal 1: khởi động signaling server
python3 webrtc/signaling_server.py

# Terminal 2: chạy pipeline
python3 main.py --backend python \
  --source rtsp://192.168.1.100:8554/stream \
  --mode webrtc \
  --server 192.168.1.200 --port 8080 --room cam1 \
  --cfg configs/config_cam_rtsp.txt
```

**C++ backend** (hiệu năng cao hơn, có NVOF):
```bash
python3 main.py --backend cpp \
  --source rtsp://192.168.1.100:8554/stream \
  --mode display \
  --homo configs/points_rtsp.yml
```

> **Lưu ý FPS**: Đặt `VIDEO_FPS` trong `config_cam_rtsp.txt` đúng với FPS thực của camera
> (25 cho PAL, 30 cho NTSC, v.v.) để tốc độ được tính chính xác.

---

#### URL RTSP phổ biến theo thiết bị

| Thiết bị | URL mẫu |
|----------|---------|
| Camera Hikvision | `rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101` |
| Camera Dahua | `rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0` |
| OBS (stream từ PC) | `rtsp://192.168.1.200:8554/live` |
| FFmpeg test stream | `ffmpeg -re -i video.mp4 -f rtsp rtsp://localhost:8554/test` |
| VLC stream | `rtsp://192.168.1.200:8554/` |

---

#### Phát RTSP từ máy tính khác bằng FFmpeg

```bash
# Cài mediamtx (RTSP server nhẹ)
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_linux_amd64.tar.gz
tar xf mediamtx_linux_amd64.tar.gz && ./mediamtx &

# Push video qua RTSP
ffmpeg -re -stream_loop -1 -i video.mp4 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtsp rtsp://localhost:8554/stream
```

---

### 1. Python Backend (Flexible)

```bash
# Display mode
python3 main.py --backend python --source video.mp4 --mode display

# File mode
python3 main.py --backend python --source video.mp4 --mode file --output result.mp4

# WebRTC mode
python3 main.py --backend python --source video.mp4 --mode webrtc --server 192.168.0.158 --room demo --cfg configs/config_cam.txt
```

### 2. C++ Backend (High Performance)

**Bước 1: Build C++ plugin**
```bash
cd speedflow_cpp
./build.sh
```

**Bước 2: Chạy với C++ backend**
```bash
# Display mode
python3 main.py --backend cpp --source video.mp4 --mode display

# File mode
python3 main.py --backend cpp --source video.mp4 --mode file --output result.mp4
```

---

## 📊 So sánh Python vs C++

| Metric | Python Backend | C++ Backend |
|--------|----------------|-------------|
| **FPS** | ~25-28 | ~32-35 |
| **Latency** | ~120-150ms | ~80-100ms |
| **CPU Usage** | ~40-50% | ~20-30% |
| **Development** | Fast | Slower |
| **Flexibility** | High | Medium |
| **NVOF Support** | Limited | Full |

---

## 🔧 Build Requirements (C++ Backend)

### System Dependencies
```bash
sudo apt install -y cmake build-essential pkg-config libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libopencv-dev libyaml-cpp-dev
```

### DeepStream SDK
- DeepStream 7.1 installed at `/opt/nvidia/deepstream/deepstream`

### Build
```bash
cd speedflow_cpp
./build.sh
```

Output: `speedflow_cpp/build/libgstspeedflow.so`

---

## 📈 Pipeline Architecture

### Python Backend
```
Source → Streammux → PGIE → Tracker → SGIE1 → SGIE2 → Analytics
                                                          ↓
                                               [Python Probes]
                                               - ROIFilterProbe
                                               - PlatePreprocessorProbe  
                                               - SpeedProbe
                                                          ↓
                                                   OSD → Sink
```

### C++ Backend
```
Source → Streammux → PGIE → Tracker → SGIE1 → SGIE2 → Analytics
                                                          ↓
                                              [C++ SpeedFlow Plugin]
                                              (All logic in single plugin)
                                                          ↓
                                                   OSD → Sink
```

---

## 🎓 Mục đích

Hệ thống dual-mode này cho phép:
1. **So sánh thực nghiệm** giữa Python và C++ trong DeepStream
2. **Benchmark** FPS, latency, CPU/GPU usage
3. **Đánh giá trade-offs** giữa tốc độ development và performance

---

## 📝 Ghi chú

- Python backend: Full features, dễ debug, phù hợp development
- C++ backend: High performance, khó debug hơn, phù hợp production
- WebRTC chỉ hỗ trợ Python backend (hiện tại)
