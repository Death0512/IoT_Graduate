import os
from pathlib import Path

# Lấy đường dẫn thư mục gốc dựa trên vị trí file settings.py
# speedflow/settings.py -> parents[1] = thư mục gốc dự án
ROOT = Path(__file__).resolve().parents[1]

# --- Video / Model ---
VIDEO_FPS = 60.0
GPU_ID = 0

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # Car, Motorbike, Bus, Truck (COCO IDs)
LICENSE_PLATE_CLASS_IDS = {0}

# --- Paths ---
PATH_LOGS = ROOT / "logs"
PATH_LOGS.mkdir(parents=True, exist_ok=True)

# Sửa lại các đường dẫn trỏ đúng vào cấu trúc thư mục của bạn
INFER_CONFIG = ROOT / "configs/config_infer_primary_yolo11.txt"
# LPD
SGIE_CONFIG = ROOT / "configs/config_infer_secondary_lpd.txt"
# LPR 
LPR_CONFIG = ROOT / "configs/config_infer_secondary_lpr.txt"

# Lưu ý: configs nằm ở ROOT, không phải trong DeepStream-Yolo
ANALYTICS_CFG = ROOT / "configs/config_nvdsanalytics.txt"
HOMO_YML      = ROOT / "configs/points_1.yml"

# Tracker config and library
TRACKER_CFG   = ROOT / "configs/config_tracker_NvDCF_perf.yml"
TRACKER_LPD_CFG = ROOT / "configs/config_tracker_lpd.yml"  # Tracker for license plates
TRACKER_LIB   = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
SPEED_LOG     = str(PATH_LOGS / "speed_log.csv")

# --- Overspeed config ---
SPEED_LIMIT_KMH = 80.0
JPEG_QUALITY    = 100
SNAP_DIR        = PATH_LOGS / "overspeed_snaps"
SNAP_DIR.mkdir(parents=True, exist_ok=True)
MAX_SNAPSHOT_PER_ID = 1

MIN_TRACK_AGE_FRAMES = int(VIDEO_FPS * 0.5)
MIN_WORLD_DISPL_M    = 0.5
MAX_ABS_KMH          = 160.0
BBOX_AREA_JUMP       = 2.5
MIN_DET_CONF         = 0.45
MEDIAN_WINDOW        = 5
