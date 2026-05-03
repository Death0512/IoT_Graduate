# 🚀 IoT_Graduate - Multi-Edge Traffic Monitoring & Coordination System

## 1. Mô tả tóm tắt hệ thống (System Overview)
**IoT_Graduate** là một hệ thống phân tán hỗ trợ phân tích và giám sát giao thông qua video theo thời gian thực. Hệ thống ứng dụng công nghệ **AI/Deep Learning (thông qua NVIDIA DeepStream SDK)** để đo tốc độ phương tiện, nhận diện xe và đọc biển số. 

Điểm nổi bật cốt lõi của hệ thống là khả năng **Multi-Edge Load Balancing** (cân bằng tải đa cụm) và độ sẵn sàng cao (High Availability). Hệ thống cho phép luân chuyển camera tự động giữa các Edge Node (Worker) một cách an toàn thông qua chiến lược **Make-before-Break** khi phát hiện một node bị quá tải (tụt FPS hoặc tải GPU/CPU quá mức). Mọi giao tiếp và điều phối được thực hiện thông qua giao thức MQTT gọn nhẹ. Ngoài ra, module xử lý pipeline AI hỗ trợ linh hoạt hai backend: **Python** (dễ dàng tuỳ biến, phát triển nhanh) và **C++** (tối đa hoá hiệu năng trên phần cứng yếu).

---

## 2. Ý nghĩa cấu trúc thư mục & file code (Folder Structure & Components)

Dưới đây là sơ đồ và giải thích chi tiết vai trò của các thành phần trong dự án:

```text
IoT_Graduate/
├── Camera/                          # Node giả lập Camera (Camera Node)
│   ├── docker-compose.yml           # File triển khai MediaMTX (RTSP Server)
│   ├── mediamtx.yml                 # Cấu hình chi tiết cho RTSP Server
│   ├── generate-compose.sh          # Script sinh tự động compose file hỗ trợ push nhiều camera
│   ├── start.sh                     # Script khởi động việc phát luồng video loop liên tục
│   └── videos/                      # Thư mục chứa các file video mẫu (.mp4)
│
├── Master/                          # Nút điều phối trung tâm (Master Node)
│   ├── master_orchestrator.py       # Bộ não trung tâm: theo dõi metrics, ra quyết định migration và failover
│   ├── mqtt_bridge.py               # Bridge hỗ trợ đồng bộ dữ liệu giữa các cluster MQTT
│   ├── mosquitto.conf               # Cấu hình MQTT Broker cục bộ
│   └── orchestrator.yml             # Cấu hình các tham số ngưỡng (threshold) cho Master
│
├── Edge/                            # Nút xử lý AI / Worker (Edge Node - thiết bị NVIDIA Jetson)
    ├── main.py                      # Entry point chính để khởi chạy pipeline DeepStream
    ├── health_agent.py              # Agent chạy ngầm đọc thông số (CPU, GPU, RAM, FPS) và publish lên Master
    ├── configs/                     # Chứa các file cấu hình tracker, model và toạ độ camera (.yml, .txt)
    ├── models/                      # Chứa model AI đã được chuyển đổi tối ưu sang TensorRT engine
    ├── speedflow_python/            # PYTHON BACKEND: Logic DeepStream pipeline viết bằng Python
    │   ├── core_pipeline.py         # Khởi tạo GStreamer pipeline, hỗ trợ dynamic add/remove stream
    │   ├── mqtt_subscriber.py       # Xử lý lệnh ADD/REMOVE camera từ Master gửi xuống
    │   └── probes.py                # Xử lý logic AI từng frame (Bounding box, đo tốc độ, gửi log)
    ├── speedflow_cpp/               # C++ BACKEND: Logic DeepStream viết bằng C/C++ (dạng plugin GStreamer)
    └── webrtc/                      # Hỗ trợ stream video phân tích thời gian thực qua trình duyệt

```

---

## 3. Hướng dẫn sử dụng chi tiết từng thư mục

Để chạy toàn bộ hệ thống phân tán này, bạn cần thiết lập theo thứ tự: **Camera Node -> Master Node -> Edge Node**.

### A. Khởi tạo nguồn luồng (Thư mục `Camera/`)
Thư mục này dùng để giả lập môi trường có nhiều camera IP RTSP. Bạn có thể chạy nó trên một máy tính trung tâm hoặc PC riêng. Đảm bảo máy tính đã cài đặt **Docker**.

1. Copy các file video `.mp4` vào thư mục `Camera/videos/` (ví dụ: `cam_01.mp4`, `cam_02.mp4`).
2. Khởi tạo RTSP Server và bắt đầu đẩy luồng:
   ```bash
   cd Camera
   chmod +x generate-compose.sh start.sh
   ./generate-compose.sh 4     # Phân tích thư mục videos/ và tạo file docker-compose tương ứng
   docker-compose up -d       # Khởi động MediaMTX RTSP Server
   ```
3. Luồng RTSP thu được sẽ có dạng: `rtsp://<IP_CAMERA_NODE>:8554/cam_01`.

### B. Khởi động bộ điều phối (Thư mục `Master/`)
Thành phần này quản lý mạng lưới các thiết bị Edge. Khuyên dùng một máy chủ Cloud hoặc PC trung tâm.

1. **Cài đặt thư viện Python cần thiết:**
   ```bash
   pip3 install paho-mqtt pyyaml
   ```
2. **Cài đặt & Chạy MQTT Broker (Mosquitto):**
   ```bash
   sudo apt install mosquitto mosquitto-clients -y
   sudo mosquitto -c mosquitto.conf -d
   ```
3. **Khởi chạy Master Orchestrator:**
   ```bash
   cd Master
   export MQTT_BROKER_HOST="localhost"     # Trỏ về địa chỉ IP của MQTT Broker
   export OVERLOAD_THRESHOLD="85.0"        # Ngưỡng quá tải tính bằng %
   python3 master_orchestrator.py
   ```
   *Lưu ý: Sau khi chạy, chương trình sẽ túc trực lắng nghe tại topic MQTT `edge/status/+` để đợi các Edge Node kết nối.*

### C. Khởi chạy luồng xử lý AI (Thư mục `Edge/`)
Thư mục này dành riêng cho các thiết bị như **NVIDIA Jetson (Nano, NX, AGX, Orin)** đã cài đặt sẵn thư viện **DeepStream SDK 7.x**.

1. **Chuẩn bị môi trường & cài dependencies:**
   ```bash
   cd Edge
   chmod +x setup_system.sh
   ./setup_system.sh
   pip3 install -r requirements.txt
   ```

2. **Bật Health Agent báo cáo tình trạng thiết bị:**
   Mở một terminal, chạy Agent để thiết bị bắt đầu bắn telemtry (health metrics) về Master:
   ```bash
   export MQTT_BROKER_HOST="<IP_CỦA_MASTER_NODE>"
   export NODE_ID="worker_jetson_01"   # Tên định danh tuỳ chọn
   python3 health_agent.py
   ```

3. **Chạy DeepStream Pipeline:**
   Ở một terminal khác, khởi chạy luồng phân tích. Khi bắt đầu chạy, `mqtt_subscriber` sẽ tự động kết nối với Master và nhận lệnh điều phối:
   
   **Chế độ Python Backend (Mặc định, dễ debug):**
   ```bash
   python3 main.py --backend python \
     --source rtsp://<IP_CAMERA_NODE>:8554/cam_01 \
     --mode display \
     --homo configs/points_rtsp.yml
   ```
   
   **Chế độ C++ Backend (Tối ưu hiệu năng CPU/GPU, thích hợp production):**
   ```bash
   # Build plugin C++ trước khi chạy lần đầu
   cd speedflow_cpp && ./build.sh && cd ..
   
   # Chạy pipeline ghi ra file
   python3 main.py --backend cpp \
     --source rtsp://<IP_CAMERA_NODE>:8554/cam_01 \
     --mode file \
     --output output/result.mp4 \
     --homo configs/points_rtsp.yml
   ```
   
   **Chế độ WebRTC (Truyền video thực tiếp lên trình duyệt):**
   ```bash
   # Terminal phụ: Bật Signaling server
   python3 webrtc/signaling_server.py
   
   # Chạy pipeline:
   python3 main.py --backend python \
     --source rtsp://<IP_CAMERA_NODE>:8554/cam_01 \
     --mode webrtc \
     --server 127.0.0.1 --port 8080 --room stream_room \
     --cfg configs/config_cam_rtsp.txt
   ```
   
   *Khi chạy trong mạng lưới phân tán, hãy theo dõi log trên cửa sổ Master Node, bạn sẽ thấy tiến trình Load Balancing tự động khi thực hiện đẩy tải (stress test) trên GPU của một thiết bị Edge.*
