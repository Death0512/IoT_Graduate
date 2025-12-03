Tài liệu này hướng dẫn thiết lập môi trường cho **NVIDIA Jetson** chạy **JetPack 6.x** với cấu hình:
**DeepStream SDK:** 7.1.0
**CUDA:** 12.6
**TensorRT:** 10.3
**Mô hình:** YOLO11 (Ultralytics)
**Python:** 3.10

---
## 1. Cài đặt các gói hệ thống cần thiết
sudo apt update
sudo apt install -y \
    python3-pip python3-dev python3-gi python3-gst-1.0 \
    libgstrtspserver-1.0-0 gstreamer1.0-rtsp libgirepository1.0-dev \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    python3-pyqt5 pyqt5-dev-tools qttools5-dev-tools \
    libprotobuf-dev protobuf-compiler
pip3 install numpy opencv-python pyyaml websockets aiohttp ultralytics onnx onnxslim onnxruntime-gpu

## 2. Cài đặt DeepStream Python Bindings (cho DS 7.1)
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
cd deepstream_python_apps
git submodule update --init
cd bindings
mkdir build && cd build
cmake ..
make -j$(nproc)
pip3 install ./pyds-*.whl
 ## Kiểm tra: Chạy python3 -c "import pyds; print('Pyds installed successfully')" không báo lỗi là được.

## 3. Chuẩn bị DeepStream-Yolo (Custom Parser)
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo
CUDA_VER=12.6 make -C nvdsinfer_custom_impl_Yolo
 ## Sau khi chạy xong, bạn sẽ có file libnvdsinfer_custom_impl_Yolo.so trong thư mục nvdsinfer_custom_impl_Yolo.

## 4. Export Model YOLO11 sang ONNX (Fix lỗi PyTorch 2.6+)
yolo export model=yolo11n.pt format=onnx dynamic=True simplify=True opset=12
 ## dynamic=True: Bắt buộc để DeepStream có thể đổi kích thước input.
 ## opset=12: Giúp tương thích tốt nhất với TensorRT.
 ## simplify=True: Tối ưu graph (yêu cầu pip install onnxslim hoặc onnx-simplifier).
 ## Sau đó, copy file yolo11n.onnx vừa tạo vào thư mục DeepStream-Yolo hoặc nơi chứa config.

## 5. Cập nhật Code & Config (Migration từ 6.3 -> 7.1)
 ## Sửa file Config Model (config_infer_primary_yolo11.txt)
[property]
# Trỏ đúng file ONNX mới tạo
onnx-file=yolo11n.onnx
# Trỏ đúng thư viện Custom Lib vừa biên dịch ở Bước 3
custom-lib-path=/đường/dẫn/tới/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
# Quan trọng: XÓA hoặc Comment dòng model-engine-file cũ để nó tự tạo lại
# model-engine-file=model_b1_gpu0_fp32.engine
# Chế độ mạng (0=FP32, 1=INT8, 2=FP16). Trên Jetson nên dùng FP16
network-mode=2

 ## Sửa code Python (settings.py & pipeline*.py)
# File: speedflow/settings.py
## Tìm: /opt/nvidia/deepstream/deepstream-6.3/...
## Sửa thành: /opt/nvidia/deepstream/deepstream/... (đặc biệt là dòng TRACKER_CFG).

# File: speedflow/pipeline.py (và các file pipeline khác)
## Cập nhật tracker.set_property('ll-lib-file', ...):
## Sửa thành: /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so

## 6. Chạy Hệ Thống
# Trước khi chạy (Sửa lỗi SSL WebRTC), nếu gặp lỗi kết nối WebRTC hoặc SSL, chạy lệnh này trước:
export GIO_USE_TLS_GNUTLS=1

Cách 1: Chạy GUI trên Jetson (Cần màn hình)
Bash

python3 speed_gui.py
Cách 2: Chạy WebRTC Streaming
B1: Bật Server Signaling (trên Jetson hoặc Server riêng)

Bash

python3 webrtc/signaling_server.py
B2: Chạy Pipeline xử lý

Bash

# Ví dụ chạy với file video
python3 run_webrtc.py file:///home/mta/video.mp4 \
    --server 127.0.0.1 \
    --room test_room \
    --cfg configs/config_cam.txt
B3: Xem kết quả Mở trình duyệt truy cập http://<IP_JETSON>:8080

7. Các lỗi thường gặp (Troubleshooting)
Lỗi: Failed to load libnvdsinfer_custom_impl_Yolo.so

Nguyên nhân: Chưa biên dịch lại thư viện với CUDA_VER=12.6.

Khắc phục: Làm lại Bước 3.

Lỗi: Input shape not supported hoặc engine build failed

Nguyên nhân: Dùng file engine cũ của bản DeepStream trước.

Khắc phục: Xóa file .engine trong thư mục chứa model và chạy lại để hệ thống tự build file mới.

Lỗi: AttributeError: 'float' object has no attribute 'node' khi export

Nguyên nhân: Dùng script export cũ với PyTorch mới.

Khắc phục: Dùng lệnh CLI yolo export như hướng dẫn ở Bước 4.