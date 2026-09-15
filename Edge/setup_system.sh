#!/bin/bash
# =============================================================
# Edge/setup_system.sh — One-shot Edge node provisioning
#
# Provision a fresh Jetson Orin to run the SpeedFlow pipeline.
# Tested on: JetPack 6.x / DeepStream 7.1 / Ubuntu 22.04 aarch64
#
# Usage:
#   chmod +x setup_system.sh
#   sudo ./setup_system.sh          # interactive — prompts for node identity
#   sudo ./setup_system.sh jetson_B # non-interactive
#
# Node → IP → local camera streams → global Edge camera IDs:
#   jetson_A  192.168.212.20  Docker cam1,cam2  →  cam_01, cam_02
#   jetson_B  192.168.212.21  Docker cam3,cam4  →  cam_03, cam_04
#   jetson_C  192.168.212.22  Docker cam5,cam6  →  cam_05, cam_06
#
# Fleet deploy note: Fleet deployment must explicitly exclude Edge/.env from
# rsync/scp/copy logic so per-node NODE_ID/ADVERTISE_IP provisioned here is not overwritten.
#
# What it does:
#   1. Install system packages (GStreamer, build tools, Python, Docker)
#   2. Verify DeepStream SDK + install Python bindings
#   3. Install Python dependencies (requirements.txt)
#   4. Configure .env for this node
#   5. Generate Edge/configs/cameras.yml for this node's two cameras
#   6. Generate Camera/docker-compose.yml for this node's two camera services
#   7. Start Camera simulator (Docker)
#   8. Set up /dev/shm, /etc/hosts, runtime dirs
# =============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
info()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[ OK ]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m %s\n" "$*" >&2; }
die()   { printf "\033[1;31m[FAIL]\033[0m %s\n" "$*" >&2; exit 1; }

[[ "$(id -u)" -eq 0 ]] || die "Run with sudo: sudo ./setup_system.sh"

# ──────────────────────────────────────────────────────────────
# 0a. Persistent journald + forensic evidence provisioning
# ──────────────────────────────────────────────────────────────
# Preserve kernel-hang forensics (journald + pstore) OUTSIDE Edge/logs so they
# survive Edge/logs cleanup. This is system-level evidence, NOT Edge project
# log/CSV collection. The historical Edge/tools diag collectors were removed;
# persistent journald is provisioned here by setup_system.sh.
info "Provisioning persistent journald…"

# 1. Create the persistent journal storage directory (absent → journald uses
#    volatile /run/journal). Owned by systemd-journal with the setgid bit so
#    journald can write/seal as a non-root group.
mkdir -p /var/log/journal
chown root:systemd-journal /var/log/journal 2>/dev/null || true
chmod 2755 /var/log/journal 2>/dev/null || true

# 2. Ensure exactly ONE Storage=persistent in /etc/systemd/journald.conf:
#    strip any existing Storage= (active or commented) directives, then append.
#    Idempotent: repeat runs only re-append if the line is missing.
JOURNALD_CONF=/etc/systemd/journald.conf
if [ -f "$JOURNALD_CONF" ]; then
    sed -i -E '/^[[:space:]]*#?[[:space:]]*(Storage|STORAGE)[[:space:]]*=/d' "$JOURNALD_CONF"
    if ! grep -qE '^[[:space:]]*Storage[[:space:]]*=' "$JOURNALD_CONF"; then
        printf '\n# Enabled by Edge/setup_system.sh — persistent forensic logging\nStorage=persistent\n' >> "$JOURNALD_CONF"
    fi
    ok "journald.conf: Storage=persistent enforced"
else
    warn "/etc/systemd/journald.conf not found — skipping Storage= enforcement"
fi

# 3. Apply the storage rules for /var/log/journal when systemd-tmpfiles is present.
if command -v systemd-tmpfiles &>/dev/null; then
    systemd-tmpfiles --create --prefix /var/log/journal 2>/dev/null || true
fi

# 4. Restart systemd-journald to adopt persistent storage. Fail-open: a failed
#    restart must not abort provisioning.
if systemctl restart systemd-journald 2>/dev/null; then
    ok "systemd-journald restarted (persistent logging active)"
else
    warn "systemd-journald restart failed — continuing; persistent logging may need a manual restart"
fi

# 5. Report forensic/evidence surfaces WITHOUT arming the watchdog. stat/read
#    only — no open() of /dev/watchdog, which would start its countdown.
if [ -d /sys/fs/pstore ]; then
    info "pstore: /sys/fs/pstore available ($(find /sys/fs/pstore -maxdepth 1 -type f 2>/dev/null | wc -l) recorded record(s))"
else
    warn "pstore: /sys/fs/pstore not mounted — kernel crash records will not persist"
fi
if [ -e /dev/watchdog ]; then
    info "watchdog: /dev/watchdog present (not armed by this script)"
else
    warn "watchdog: /dev/watchdog absent"
fi

# ──────────────────────────────────────────────────────────────
# 0. Determine node identity
# ──────────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    NODE_ID="$1"
else
    echo ""
    echo "  Available node identities:"
    echo "    jetson_A  →  192.168.212.20  (cam1, cam2  →  cam_01, cam_02)"
    echo "    jetson_B  →  192.168.212.21  (cam3, cam4  →  cam_03, cam_04)"
    echo "    jetson_C  →  192.168.212.22  (cam5, cam6  →  cam_05, cam_06)"
    echo ""
    read -rp "  Enter NODE_ID for this device [jetson_A]: " NODE_ID
    NODE_ID="${NODE_ID:-jetson_A}"
fi

# CAM_LOCAL_NUMS : the two Docker service names (cam1 cam2 OR cam3 cam4)
# CAM_EDGE_IDS   : the two global Edge camera IDs (cam_01 cam_02 OR cam_03 cam_04)
# CAM_SOURCE_IDS : the two global pipeline slot IDs, stable across all Jetsons
case "$NODE_ID" in
    jetson_A)
        ADVERTISE_IP="192.168.212.20"
        CAM_LOCAL_NUMS=(1 2)
        CAM_EDGE_IDS=(cam_01 cam_02)
        CAM_SOURCE_IDS=(0 1)
        CAM_VIDEO_FILES=(/videos/sparse/video_in_N.mp4 /videos/sparse/video_out_S.mp4)
        ;;
    jetson_B)
        ADVERTISE_IP="192.168.212.21"
        CAM_LOCAL_NUMS=(3 4)
        CAM_EDGE_IDS=(cam_03 cam_04)
        CAM_SOURCE_IDS=(2 3)
        CAM_VIDEO_FILES=(/videos/video_in_N.mp4 /videos/video_out_S.mp4)
        ;;
    jetson_C)
        ADVERTISE_IP="192.168.212.22"
        CAM_LOCAL_NUMS=(5 6)
        CAM_EDGE_IDS=(cam_05 cam_06)
        CAM_SOURCE_IDS=(4 5)
        CAM_VIDEO_FILES=(/videos/moderate/video_in_N.mp4 /videos/moderate/video_out_S.mp4)
        ;;
    *)
        die "Unknown NODE_ID '$NODE_ID'. Expected jetson_A, jetson_B, or jetson_C."
        ;;
esac

SERVER_IP="116.118.9.125"

info "Provisioning node: $NODE_ID ($ADVERTISE_IP)"
info "Local camera streams : cam${CAM_LOCAL_NUMS[0]}, cam${CAM_LOCAL_NUMS[1]}"
info "Global Edge camera IDs: ${CAM_EDGE_IDS[0]}, ${CAM_EDGE_IDS[1]}"
info "Server: $SERVER_IP"
echo ""

# ──────────────────────────────────────────────────────────────
# 1. System packages
# ──────────────────────────────────────────────────────────────
if [ "${QUICK_PROVISION:-0}" = "1" ]; then
    info "QUICK_PROVISION=1: Skipping apt, docker and pip installations"
else
    info "Installing system packages…"
    apt-get update -qq

    apt-get install -y -qq \
        build-essential cmake pkg-config \
        \
        libglib2.0-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstrtspserver-1.0-dev \
        \
        gstreamer1.0-tools \
        gstreamer1.0-rtsp \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        \
        python3-gi \
        python3-gi-cairo \
        python3-dev \
        python3-pip \
        python3-venv \
        \
        v4l-utils curl jq sshpass

    ok "System packages installed"

    # ──────────────────────────────────────────────────────────────
    # 2. Docker (for Camera simulator)
    # ──────────────────────────────────────────────────────────────
    if ! command -v docker &>/dev/null; then
        info "Installing Docker…"
        curl -fsSL https://get.docker.com | sh
        usermod -aG docker "${SUDO_USER:-$USER}" 2>/dev/null || true
        systemctl enable --now docker
        ok "Docker installed"
    else
        ok "Docker already installed ($(docker --version | cut -d' ' -f3))"
    fi

    if ! docker compose version &>/dev/null; then
        info "Installing Docker Compose plugin…"
        apt-get install -y -qq docker-compose-plugin 2>/dev/null \
            || pip3 install -q docker-compose
        ok "Docker Compose installed"
    else
        ok "Docker Compose already installed"
    fi

    # ──────────────────────────────────────────────────────────────
    # 3. Verify DeepStream SDK
    # ──────────────────────────────────────────────────────────────
    DEEPSTREAM_DIR="/opt/nvidia/deepstream/deepstream"
    if [ -d "$DEEPSTREAM_DIR" ]; then
        DS_VER=$(cat "$DEEPSTREAM_DIR/version" 2>/dev/null | head -1 || echo "unknown")
        ok "DeepStream SDK found: $DS_VER"

        # Install DeepStream Python bindings if available
        DS_PYTHON="$DEEPSTREAM_DIR/sources/deepstream_python_apps"
        if [ -d "$DS_PYTHON" ]; then
            DS_WHL=$(find "$DS_PYTHON" -name "*.whl" -path "*python*" 2>/dev/null | head -1)
            if [ -n "$DS_WHL" ]; then
                pip3 install -q "$DS_WHL" && ok "DeepStream Python bindings installed"
            fi
        fi
    else
        warn "DeepStream SDK not found at $DEEPSTREAM_DIR"
        warn "Install JetPack + DeepStream first, then re-run this script."
    fi

    # ──────────────────────────────────────────────────────────────
    # 4. Python dependencies
    # ──────────────────────────────────────────────────────────────
    info "Installing Python packages…"
    pip3 install --upgrade pip setuptools wheel -q
    pip3 install -r "$SCRIPT_DIR/requirements.txt" -q
    ok "Python packages installed"
fi

# ──────────────────────────────────────────────────────────────
# 5. Configure .env
# ──────────────────────────────────────────────────────────────
info "Writing .env for $NODE_ID…"
cat > "$SCRIPT_DIR/.env" <<EOF
# =============================================================
# Edge/.env — Generated by setup_system.sh ($(date +%F))
# Node: $NODE_ID ($ADVERTISE_IP)
# =============================================================

# --- Node identity ---
# NODE_ID is the sole node identity consumed by the runtime
# (settings.py:54, health_agent.py, run_python.py). EDGE_ID was a legacy alias
# of NODE_ID and is NOT read by any runtime code, so it is intentionally NOT
# emitted here to avoid a misleading dead identity on provisioned devices.
NODE_ID=$NODE_ID

# --- Load balancing experiment mode ---
# LOAD_POLICY: actual | predict_no_base | predict_with_base
LOAD_POLICY=actual
# LOAD_MODEL: formula | dl
LOAD_MODEL=formula

# --- Network ---
ADVERTISE_IP=$ADVERTISE_IP

# --- Central Monitoring Server / Zenoh Router ---
ZENOH_ROUTER=tcp/$SERVER_IP:7447

# --- RTSP Push (→ MediaMTX on Server) ---
RTSP_PUSH_URL=rtsp://$SERVER_IP:8554/$NODE_ID
# Max 750 kbps per camera (750000 bps) to fit shared WAN uplink bandwidth
RTSP_PUSH_BITRATE=750000
RTSP_PUSH_MAX_RETRIES=3
RTSP_PUSH_RETRY_DELAY_S=1.0

# --- DeepStream Pipeline Session & Slot Limits ---
SPEEDFLOW_SLOT_CAPACITY=16
SPEEDFLOW_NVDEC_SESSION_LIMIT=14

# --- Zenoh (P2P peer mode) ---
ZENOH_QUEUE_MAXSIZE=1000
ZENOH_ROUTER_STALE_S=15.0

# --- Health Agent ---
HEALTH_INTERVAL=1.0
# Log the LoadScore line once every N health cycles.
HEALTH_LOG_EVERY=1
TARGET_FPS=27.0
FPS_STATS_FILE=/dev/shm/speedflow_fps.json

# --- Pipeline / Video ---
VIDEO_FPS=30.0
GPU_ID=0
# 1280x720: each tile is 640x360 at 4-cam tiling — saves ~44% GPU memory
# bandwidth and ~30% encoder bitrate vs 1920x1080.
MUX_WIDTH=1920
MUX_HEIGHT=1080

# --- AI / DeepStream paths (relative to Edge/) ---
INFER_CONFIG=configs/config_infer_primary_yolo11.txt
SGIE_CONFIG=configs/config_infer_secondary_lpd.txt
LPR_CONFIG=configs/config_infer_secondary_lpr.txt
ANALYTICS_CFG=configs/config_nvdsanalytics.txt
TRACKER_CFG=configs/config_tracker_NvDCF_perf.yml
TRACKER_LPD_CFG=configs/config_tracker_lpd.yml
CAMERAS_YML=configs/cameras.yml
TRACKER_LIB=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so

# --- Detection / Speed thresholds ---
SPEED_LIMIT_KMH=80.0
# 82 is visually indistinguishable from 100 for license plate images
# but produces files ~3-4x smaller — critical for WS bandwidth on violations.
JPEG_QUALITY=82
MAX_SNAPSHOT_PER_ID=1
MIN_WORLD_DISPL_M=0.5
MAX_ABS_KMH=160.0
BBOX_AREA_JUMP=2.5
MIN_DET_CONF=0.45
MEDIAN_WINDOW=5

# --- Output paths (relative to Edge/) ---
SPEED_LOG=logs/speed_log.csv
SNAP_DIR=logs/overspeed_snaps
EOF
ok ".env written for $NODE_ID"

# ──────────────────────────────────────────────────────────────
# 6. Generate cameras.yml for this node's two cameras
#
# Mapping:
#   jetson_A: source_id 0→cam_01 (rtsp://$ADVERTISE_IP:8554/cam1)
#             source_id 1→cam_02 (rtsp://$ADVERTISE_IP:8554/cam2)
#   jetson_B: source_id 2→cam_03 (rtsp://$ADVERTISE_IP:8554/cam3)
#             source_id 3→cam_04 (rtsp://$ADVERTISE_IP:8554/cam4)
#   jetson_C: source_id 4→cam_05 (rtsp://$ADVERTISE_IP:8554/cam5)
#             source_id 5→cam_06 (rtsp://$ADVERTISE_IP:8554/cam6)
# ──────────────────────────────────────────────────────────────
CAMERAS_YML="$SCRIPT_DIR/configs/cameras.yml"
mkdir -p "$(dirname "$CAMERAS_YML")"
info "Generating cameras.yml for $NODE_ID (${CAM_EDGE_IDS[0]}, ${CAM_EDGE_IDS[1]})…"

cat > "$CAMERAS_YML" <<EOF
# =============================================================
# Edge/configs/cameras.yml
# GENERATED by setup_system.sh on $(date +%F) — DO NOT EDIT BY HAND
# Node : $NODE_ID ($ADVERTISE_IP)
# Cameras: ${CAM_EDGE_IDS[0]} (local cam${CAM_LOCAL_NUMS[0]}) · ${CAM_EDGE_IDS[1]} (local cam${CAM_LOCAL_NUMS[1]})
#
# Node → IP → local Docker stream → global camera ID:
#   jetson_A  192.168.212.20  cam1 → cam_01 · cam2 → cam_02
#   jetson_B  192.168.212.21  cam3 → cam_03 · cam4 → cam_04
#
# source_points and roi_polygon are in MUX resolution (1920×1080).
# If MUX_WIDTH/MUX_HEIGHT changes, scale all coordinates by the new ratio.
# =============================================================
max_streams: 4

tiler_mode: auto
tiler_rows: 4
tiler_cols: 2

cameras:
  ${CAM_EDGE_IDS[0]}:
    camera_id: "${CAM_EDGE_IDS[0]}"
    source_id: ${CAM_SOURCE_IDS[0]}
    uri: "rtsp://$ADVERTISE_IP:8554/cam${CAM_LOCAL_NUMS[0]}"
    enabled: true
    name: "Camera ${CAM_LOCAL_NUMS[0]}"
    fps: 30.0
    speed_limit_kmh: 80.0
    homography:
      source_points:
        - [656, 26]
        - [1176, 32]
        - [1592, 1016]
        - [72, 1000]
      target_width: 15
      target_height: 60
    roi_polygon: [656, 26, 1176, 32, 1592, 1016, 72, 1000]
    output:
      record: true
      record_path: "output/${CAM_EDGE_IDS[0]}.mp4"

  ${CAM_EDGE_IDS[1]}:
    camera_id: "${CAM_EDGE_IDS[1]}"
    source_id: ${CAM_SOURCE_IDS[1]}
    uri: "rtsp://$ADVERTISE_IP:8554/cam${CAM_LOCAL_NUMS[1]}"
    enabled: true
    name: "Camera ${CAM_LOCAL_NUMS[1]}"
    fps: 30.0
    speed_limit_kmh: 80.0
    homography:
      source_points:
        - [656, 26]
        - [1176, 32]
        - [1592, 1016]
        - [72, 1000]
      target_width: 15
      target_height: 60
    roi_polygon: [656, 26, 1176, 32, 1592, 1016, 72, 1000]
    output:
      record: true
      record_path: "output/${CAM_EDGE_IDS[1]}.mp4"
EOF
ok "cameras.yml generated"

# ──────────────────────────────────────────────────────────────
# 7. Generate Camera/docker-compose.yml + Camera/.env for this node
#
# rtsp_server + exactly the two local camera services for this node.
# VIDEO_FILE and RTSP_URL are resolved from Camera/.env at compose runtime.
# Each node keeps its global camera numbering (cam1/cam2 or cam3/cam4).
#
# Mapping:
#   jetson_A: services cam1, cam2  (RTSP paths /cam1, /cam2)
#   jetson_B: services cam3, cam4  (RTSP paths /cam3, /cam4)
#   jetson_C: services cam5, cam6  (RTSP paths /cam5, /cam6)
# ──────────────────────────────────────────────────────────────
CAMERA_DIR="$PROJECT_DIR/Camera"
COMPOSE_FILE="$CAMERA_DIR/docker-compose.yml"
ENV_FILE="$CAMERA_DIR/.env"
mkdir -p "$CAMERA_DIR"
info "Generating Camera/docker-compose.yml for $NODE_ID (cam${CAM_LOCAL_NUMS[0]}, cam${CAM_LOCAL_NUMS[1]})…"

cat > "$COMPOSE_FILE" <<EOF
# =============================================================
# Camera/docker-compose.yml
# GENERATED by Edge/setup_system.sh on $(date +%F) — DO NOT EDIT BY HAND
# Node : $NODE_ID ($ADVERTISE_IP)
# Services: rtsp_server · cam${CAM_LOCAL_NUMS[0]} · cam${CAM_LOCAL_NUMS[1]}
#
# Node → local camera services:
#   jetson_A  →  cam1, cam2
#   jetson_B  →  cam3, cam4
#   jetson_C  →  cam5, cam6
#
# To swap a video source, set CAM${CAM_LOCAL_NUMS[0]}_VIDEO_FILE or
# CAM${CAM_LOCAL_NUMS[1]}_VIDEO_FILE in Camera/.env, then restart the services.
# =============================================================
services:
  rtsp_server:
    image: bluenviron/mediamtx:latest
    container_name: rtsp_server
    ports:
      - "8554:8554"
    volumes:
      - ./mediamtx.yml:/mediamtx.yml
    restart: unless-stopped

  cam${CAM_LOCAL_NUMS[0]}:
    build: .
    container_name: cam${CAM_LOCAL_NUMS[0]}
    depends_on:
      - rtsp_server
    environment:
      - VIDEO_FILE=\${CAM${CAM_LOCAL_NUMS[0]}_VIDEO_FILE:-/videos/sample.mp4}
      - RTSP_URL=\${CAM${CAM_LOCAL_NUMS[0]}_RTSP_URL:-rtsp://rtsp_server:8554/cam${CAM_LOCAL_NUMS[0]}}
    volumes:
      - ./videos:/videos
    restart: unless-stopped

  cam${CAM_LOCAL_NUMS[1]}:
    build: .
    container_name: cam${CAM_LOCAL_NUMS[1]}
    depends_on:
      - rtsp_server
    environment:
      - VIDEO_FILE=\${CAM${CAM_LOCAL_NUMS[1]}_VIDEO_FILE:-/videos/sample.mp4}
      - RTSP_URL=\${CAM${CAM_LOCAL_NUMS[1]}_RTSP_URL:-rtsp://rtsp_server:8554/cam${CAM_LOCAL_NUMS[1]}}
    volumes:
      - ./videos:/videos
    restart: unless-stopped
EOF
ok "Camera/docker-compose.yml generated"

# Camera/.env — defines CAM${CAM_LOCAL_NUMS[0]}_* and CAM${CAM_LOCAL_NUMS[1]}_*
# for the two local cameras of this node. Whole-file overwrite so no stale
# CAM1/CAM2 entries leak to jetson_B / jetson_C. RTSP_PORT and HLS_PORT
# belong to the rtsp_server service and stay fixed.
info "Generating Camera/.env for $NODE_ID (cam${CAM_LOCAL_NUMS[0]}, cam${CAM_LOCAL_NUMS[1]})…"
cat > "$ENV_FILE" <<EOF
# =============================================================
# Camera/.env — Generated by Edge/setup_system.sh ($(date +%F))
# Node: $NODE_ID ($ADVERTISE_IP)
# Per-node local cameras: cam${CAM_LOCAL_NUMS[0]}, cam${CAM_LOCAL_NUMS[1]}
# =============================================================

# --- RTSP server ports (rtsp_server service, shared by all nodes) ---
RTSP_PORT=8554
HLS_PORT=8888

# --- Local camera ${CAM_LOCAL_NUMS[0]} ---
CAM${CAM_LOCAL_NUMS[0]}_VIDEO_FILE=${CAM_VIDEO_FILES[0]}
CAM${CAM_LOCAL_NUMS[0]}_RTSP_URL=rtsp://rtsp_server:8554/cam${CAM_LOCAL_NUMS[0]}

# --- Local camera ${CAM_LOCAL_NUMS[1]} ---
CAM${CAM_LOCAL_NUMS[1]}_VIDEO_FILE=${CAM_VIDEO_FILES[1]}
CAM${CAM_LOCAL_NUMS[1]}_RTSP_URL=rtsp://rtsp_server:8554/cam${CAM_LOCAL_NUMS[1]}
EOF
ok "Camera/.env generated"

# ──────────────────────────────────────────────────────────────
# 8. Runtime directories
# ──────────────────────────────────────────────────────────────
info "Setting up runtime environment…"
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/logs/overspeed_snaps"
if [ -n "${SUDO_USER:-}" ]; then
    chown -R "$SUDO_USER:$SUDO_USER" "$SCRIPT_DIR/logs"
fi
# /dev/shm/speedflow_fps.json is NOT pre-created here — the runtime
# Python pipeline creates it atomically via os.replace. Pre-creating it
# as root left a root-owned file that the unprivileged runtime user
# could not replace on some tmpfs configurations.

# ──────────────────────────────────────────────────────────────
# 9. /etc/hosts peer hints
# ──────────────────────────────────────────────────────────────
PEER_HINT="# IoT_Graduate peer nodes"
if ! grep -q "$PEER_HINT" /etc/hosts 2>/dev/null; then
    info "Adding peer hints to /etc/hosts…"
    cat >> /etc/hosts <<'HOSTS'

# IoT_Graduate peer nodes
192.168.212.20 jetson_A
192.168.212.21 jetson_B
192.168.212.22 jetson_C
HOSTS
    ok "/etc/hosts updated"
fi

# ──────────────────────────────────────────────────────────────
# 10. Start Camera simulator
# ──────────────────────────────────────────────────────────────
if [ -d "$CAMERA_DIR" ] && [ -f "$COMPOSE_FILE" ]; then
    info "Starting Camera simulator (Docker)…"
    cd "$CAMERA_DIR"
    docker compose up -d --build 2>/dev/null || docker-compose up -d --build 2>/dev/null || warn "Camera Docker start failed"
    ok "Camera simulator running"
    cd "$SCRIPT_DIR"
else
    warn "Camera/ directory not found — skip camera simulator"
fi

# ──────────────────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────────────────
printf "\n"
ok "═══════════════════════════════════════════════"
ok "  $NODE_ID ($ADVERTISE_IP) provisioned!"
ok "═══════════════════════════════════════════════"
echo ""
info "Camera streams : rtsp://$ADVERTISE_IP:8554/cam${CAM_LOCAL_NUMS[0]}  rtsp://$ADVERTISE_IP:8554/cam${CAM_LOCAL_NUMS[1]}"
info "Edge camera IDs: ${CAM_EDGE_IDS[0]}  ${CAM_EDGE_IDS[1]}"
echo ""
info "Run the pipeline:"
info "  cd $SCRIPT_DIR && python3 main.py --mode rtsp_push"
echo ""
