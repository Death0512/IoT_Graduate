# IoT Graduate — Distributed Edge Traffic Monitoring

Hệ thống giám sát giao thông phân tán trên Jetson: phát hiện xe, bám vết, ước lượng tốc độ qua homography, phát hiện biển số, ghi nhận vi phạm, và chia tải camera giữa các nút khi quá tải.

## Cách hệ thống hoạt động

```
Camera (MediaMTX RTSP) → Edge Jetson (1 process) → Server (MediaMTX relay + Dashboard)
                          DeepStream pipeline
                          + HealthAgent (tính S_load, heartbeat 1Hz)
                          + PeerOrchestrator (quyết định L0/L1/L2)
                          + Zenoh subs/pubs
```

**Camera** — Docker Compose chạy MediaMTX `:8554` + FFmpeg loop MP4 thành RTSP.

**Edge** — Mỗi Jetson chạy **một process** `run_edge.sh → run_python_mode()` chứa toàn bộ pipeline và control plane, chia sẻ một Zenoh session.

**Server** — aiohttp `:9090` nhận health + overspeed qua Zenoh, hiển thị dashboard, lưu violation, relay RTSP. Có thể chạy Zenoh router `:7447` cho cross-subnet.

## Pipeline DeepStream

```
N × uridecodebin → nvstreammux → PGIE (YOLO11) → NvDCF tracker → SGIE (LPD)
→ nvdsanalytics → SpeedProbe (probe) → rtsp_push / display / file
```

LPR (nhận dạng biển số) chạy **ngoài pipeline** trên `LocalLprWorker` (L0) hoặc `OffloadReceiver` của peer (L1). Probe ghi snapshot 1s ra `/dev/shm/speedflow_fps.json`.

## Load Score

```
S_load = 100 * rho / (1 + rho),  rho = 0.15*rho_s + 0.35*rho_d + 0.35*rho_r + 0.15*rho_v
rho_s: đồng thời luồng (knee 4.0, exp 2.0)
rho_d: tải xe (n_track+n_plate / 16.0, EMA 0.33)
rho_r: nghẽn CPU/RAM (u_safe 0.60)
rho_v: thâm hụt hoàn thành dịch vụ (target 0.95, floor 0.50)
```

FPS chỉ là cầu chì khẩn cấp (floor 80 khi FPS<12), không phải trục chính.

## Chia tải

- **L0** — xử lý cục bộ toàn pipeline (mặc định)
- **L1** — đẩy crop biển số qua `offload/plates/{src}/{dst}` cho peer làm LPR (giảm queue LPR)
- **L2** — di trú nguyên luồng RTSP qua RFO/lease (giảm decode/tracking), make-before-break

**Thang bắt buộc:** quá tải `S≥55` giữ `3.0s` + `stream_pressure≥0.30` → L1, giữ `12.0s` quan sát, nếu vẫn quá tải → L2. Hạ cấp khi `S<50` và queue rút (`ratio<0.08`) giữ `5.0s`. Thu hồi khi `S<35` giữ `5.0s`.

Chọn camera theo workload (`n_track+n_plate`), không theo FPS. Mỗi nút giữ ≥1 camera gốc, không chuyển tiếp camera cứu hộ.

## Truyền thông Zenoh (peer mode)

`peers/status/{node}` heartbeat, `peers/vote/*` RFO/bid/decision/ack, `offload/plates/*`, `offload/results/*`, `traffic/events/*`.

## Phần cứng triển khai

| Node | IP | Camera gốc |
|---|---|---|
| jetson_A | 192.168.212.20 | cam_01, cam_02 |
| jetson_B | 192.168.212.21 | cam_03, cam_04 |
| jetson_C | 192.168.212.22 | cam_05, cam_06 |
| Server | 116.118.9.125 | MediaMTX 8554, Web 9090, Zenoh 7447 |

Jetson AGX Orin 32GB, JetPack 6.x, LAN 1Gbps. `max_streams=4`, `nvdec_limit=14`.

## Hướng dẫn sử dụng

### Chạy Edge (từ `Edge/`, env `DoAn`)

```bash
./run_edge.sh                              # rtsp_push mặc định
./run_edge.sh --mode display               # hiện màn hình
./run_edge.sh --mode rtsp_push --rtsp-push-url rtsp://116.118.9.125:8554/jetson_A
nohup ./run_edge.sh >/dev/null 2>&1 &      # chạy nền
```

### Chạy Camera

```bash
cd Camera && docker compose up -d
docker compose down
```

Video đặt trong `Camera/videos/`. Dùng `network_mode: host` trên Jetson thiếu `veth.ko`.

### Chạy Server

```bash
cd Server && python3 app.py   # http://0.0.0.0:9090
```

Cần `Server/.env` (`SERVER_PORT`, `MEDIAMTX_API`).

### Triển khai fleet

```bash
# 1. Teardown: Jetson (kill PID logs/run_edge.pid) → Camera down → Server pkill
# 2. Sync
for ip in 192.168.212.20 192.168.212.21 192.168.212.22; do
  rsync -avz --exclude='.env' --exclude='logs/' Edge/ mta@$ip:/home/mta/Documents/IoT_Graduate/Edge/
done
rsync -avz --exclude='logs/' Server/ mta@116.118.9.125:/home/mta/Documents/IoT_Graduate/Server/

# 3. Provision (nếu đổi identity)
sudo bash Edge/setup_system.sh jetson_A  # hoặc B/C

# 4. Khởi động: Server → Camera → Jetson, kiểm 6/6 RTSP ffprobe, soak 300s/1800s
```

Đồng bộ **không** ghi đè `Edge/.env` (chứa `NODE_ID`, `ADVERTISE_IP`).

## File cấu hình chính

| File | Dùng cho |
|---|---|
| `Edge/.env` | NODE_ID, ADVERTISE_IP, RTSP_PUSH_URL, TARGET_FPS, HEALTH_INTERVAL |
| `Edge/configs/cameras.yml` | URI RTSP, homography, ROI (quyết định ownership) |
| `Edge/configs/edge_node.yml` | Ngưỡng P2P (overload, hold, pressure, heartbeat), load_score |
| `Server/.env` | SERVER_PORT, MEDIAMTX_API |
