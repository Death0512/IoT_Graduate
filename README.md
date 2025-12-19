# Traffic Monitoring & Speed Detection System

![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson-green?style=for-the-badge&logo=nvidia)
![DeepStream](https://img.shields.io/badge/DeepStream-7.1-blue?style=for-the-badge)
![JetPack](https://img.shields.io/badge/JetPack-6.2-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-yellow?style=for-the-badge)

Hệ thống giám sát giao thông thông minh sử dụng **NVIDIA DeepStream SDK 7.1**, hỗ trợ phát hiện phương tiện, đo tốc độ, và nhận diện biển số xe (LPR) theo thời gian thực.

---

## Tính năng chính

- **Đo tốc độ xe:** Theo dõi và tính toán tốc độ dựa trên phép biến đổi Homography.
- **Nhận diện biển số (LPR):** Tích hợp pipeline phát hiện và đọc biển số xe Việt Nam.
- **Đa nền tảng Output:**
  - **Display Mode:** Xem trực tiếp trên màn hình HDMI.
  - **WebRTC Mode:** Streaming video độ trễ thấp tới trình duyệt (Browser).
  - **File Mode:** Lưu kết quả video ra file MP4.
- **Quản lý vi phạm:** Tự động chụp ảnh phương tiện vi phạm tốc độ.
- **Tối ưu hóa:** Sử dụng TensorRT engine (FP16) cho hiệu năng cao trên Jetson Orin/Nano.

---

## Quick Start

### 1. Chạy WebRTC Streaming (Khuyên dùng)
Xem video trực tiếp từ trình duyệt trên mọi thiết bị trong mạng LAN.

**B1: Bật Signaling Server**
```bash
python3 webrtc/signaling_server.py
```

**B2: Chạy Pipeline xử lý (Terminal mới)**
```bash
# Với file video test
python3 main.py --source videodemo/sample.mp4 --mode webrtc \
  --server 127.0.0.1 --room cam01

# Với RTSP Camera
python3 main.py --source rtsp://admin:pass@192.168.1.x:554/stream --mode webrtc \
  --server 127.0.0.1 --room cam01
```
> Truy cập trình duyệt: `http://<IP_JETSON>:8080/?room=cam01`

### 2. Chạy Display Mode (Màn hình gắn trực tiếp)
```bash
python3 main.py --source videodemo/sample.mp4 --mode display
```

---

## Cài đặt & Môi trường

<details>
<summary><b>1. Yêu cầu hệ thống (Click to expand)</b></summary>

- **Phần cứng:** NVIDIA Jetson Orin Nano / AGX Orin
- **OS:** Ubuntu 22.04 (JetPack 6.x)
- **DeepStream:** 7.1.0
- **CUDA:** 12.6
- **TensorRT:** 10.3
</details>

<details>
<summary><b>2. Cài đặt Dependencies</b></summary>

```bash
# System packages
sudo apt update
sudo apt install -y python3-pip python3-dev python3-gi python3-gst-1.0 \
    libgstrtspserver-1.0-0 gstreamer1.0-rtsp libgirepository1.0-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev

# Python packages
pip3 install numpy opencv-python pyyaml websockets aiohttp ultralytics
```
</details>

<details>
<summary><b>3. Cài đặt DeepStream Python Bindings</b></summary>

```bash
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
cd deepstream_python_apps
git switch -c v1.2.0
git submodule update --init
cd bindings && mkdir build && cd build
cmake ..
make -j$(nproc)
python3 -m pip install ./..
```
</details>

---

## Cấu hình (Configuration)

Các file cấu hình nằm trong thư mục `configs/`:

| File Config | Mô tả |
|-------------|-------|
| `config_infer_primary_yolo11.txt` | Cấu hình model phát hiện xe (YOLO11) |
| `config_infer_secondary_lpd.txt` | Cấu hình model phát hiện biển số (LPD) |
| `config_infer_secondary_lpr.txt` | Cấu hình model đọc biển số (LPR OCR) |
| `config_nvdsanalytics.txt` | Cấu hình vùng ROI và line đếm xe |
| `points_1.yml` | Điểm tham chiếu Homography để đo tốc độ |

**Chỉnh sửa tham số hệ thống** trong `speedflow/settings.py`:
- `VIDEO_FPS`: FPS của nguồn video (quan trọng để tính tốc độ đúng).
- `SPEED_LIMIT_KMH`: Ngưỡng cảnh báo tốc độ.
- `DEBUG_MODE`: Bật/tắt log chi tiết.

---
