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
sudo apt install -y \
    cmake \
    build-essential \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libopencv-dev \
    libyaml-cpp-dev
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
