"""
speedflow_python/settings.py

Single source of truth for all runtime configuration.
Values are loaded from Edge/.env (via python-dotenv).

All other modules import constants from here — never call
os.environ directly with hardcoded fallback strings.

Usage:
    from .settings import CAMERAS_YML, ...
"""

import math
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

import yaml
from dotenv import load_dotenv

# Edge/ root — two levels up from this file (speedflow_python/settings.py)
ROOT = Path(__file__).resolve().parents[1]

def validate_p2p_offline_threshold(p2p_cfg: dict, path: Any) -> None:
    """Fail closed on the peer-offline threshold inputs.

    The effective PeerOrchestrator offline threshold is
    ``p2p.heartbeat_timeout_s + p2p.failover_grace_s``. A malformed or missing
    value must abort startup rather than silently fall back to a default that
    no longer matches the Server's HEARTBEAT_TIMEOUT. Raises ValueError.
    """
    for key in ("heartbeat_timeout_s", "failover_grace_s"):
        val = p2p_cfg.get(key)
        try:
            f = float(val)
        except (TypeError, ValueError):
            raise ValueError(
                f"Edge configuration file {path} p2p.{key} must be a number "
                f"(got {val!r})"
            )
        if not math.isfinite(f) or f <= 0:  # NaN, infinity, or non-positive
            raise ValueError(
                f"Edge configuration file {path} p2p.{key} must be a finite "
                f"positive number (got {val!r})"
            )

def load_edge_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Load and validate edge_node.yml.

    Fails closed with a clear exception if the file is missing, unreadable,
    malformed, empty, or missing required sections (e.g. 'p2p').
    """
    path = Path(config_path) if config_path is not None else (ROOT / "configs" / "edge_node.yml")
    if not path.exists():
        raise FileNotFoundError(f"Required edge configuration file does not exist: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"Failed to parse edge configuration file {path}: {exc}") from exc

    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"Edge configuration file {path} is empty or not a valid dictionary")
    if "p2p" not in cfg or not isinstance(cfg["p2p"], dict):
        raise ValueError(f"Edge configuration file {path} missing required 'p2p' section")

    # Fail closed on the peer-offline threshold inputs.
    validate_p2p_offline_threshold(cfg["p2p"], path)
    return cfg

# Load .env from Edge/.env (silent if missing — allows overrides via real env)
_env_path = ROOT / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

def _require(key: str) -> str:
    """Read env var; raise clearly if missing (no silent defaults)."""
    val = os.environ.get(key)
    if val is None:
        raise RuntimeError(
            f"Required env var '{key}' is not set. "
            f"Check {_env_path}"
        )
    return val

def _get(key: str, cast: Any = str) -> Any:
    """Read env var with cast; raise if missing."""
    return cast(_require(key))

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").strip().upper()

# -----------------------------------------------------------
# Central Monitoring Server / Zenoh Router
# -----------------------------------------------------------
ZENOH_ROUTER = os.environ.get("ZENOH_ROUTER", "").strip()   # empty → multicast only

# -----------------------------------------------------------
# Node identity
# -----------------------------------------------------------
NODE_ID = _get("NODE_ID")

# -----------------------------------------------------------
# Load balancing experiment mode
# -----------------------------------------------------------
LOAD_POLICY = os.environ.get("LOAD_POLICY", "actual").strip().lower()
LOAD_MODEL  = os.environ.get("LOAD_MODEL", "formula").strip().lower()
EDGE_LOAD_SCORE_MODE = os.environ.get("EDGE_LOAD_SCORE_MODE", "").strip().lower()

_VALID_LOAD_POLICIES = {"actual", "predict_no_base", "predict_with_base"}
_VALID_LOAD_MODELS   = {"formula", "dl"}
if LOAD_POLICY not in _VALID_LOAD_POLICIES:
    raise RuntimeError(
        f"LOAD_POLICY={LOAD_POLICY!r} invalid. "
        f"Must be one of: {', '.join(sorted(_VALID_LOAD_POLICIES))}"
    )
if LOAD_MODEL not in _VALID_LOAD_MODELS:
    raise RuntimeError(
        f"LOAD_MODEL={LOAD_MODEL!r} invalid. "
        f"Must be one of: {', '.join(sorted(_VALID_LOAD_MODELS))}"
    )

# -----------------------------------------------------------
# Zenoh (P2P peer mode — no broker needed)
# -----------------------------------------------------------
ZENOH_QUEUE_MAXSIZE = _get("ZENOH_QUEUE_MAXSIZE", int)

# -----------------------------------------------------------
# Health Agent
# -----------------------------------------------------------
HEALTH_INTERVAL  = _get("HEALTH_INTERVAL", float)
# Zenoh transport-liveness gate: if the session reports no live router transport
# for this many seconds while ZENOH_ROUTER is configured, infer silent transport
# death and force reconnect. Must be >> HEALTH_INTERVAL. 0 disables the check.
ZENOH_ROUTER_STALE_S = float(os.environ.get("ZENOH_ROUTER_STALE_S", "15.0"))
# Log the LoadScore line only once every N health cycles.
# e.g. HEALTH_LOG_EVERY=15 + HEALTH_INTERVAL=2.0 → log every 30 s.
# Set to 1 to log every cycle (original behaviour).
HEALTH_LOG_EVERY = int(os.environ.get("HEALTH_LOG_EVERY", "1"))
# Periodic NODE_ONLINE re-announcement interval (seconds). The Server only
# re-arms a node that was swept offline via an explicit NODE_ONLINE event, so
# a live node re-announces itself periodically to recover from a transient
# partition. Must stay well under the Server's HEARTBEAT_TIMEOUT (30s).
NODE_ONLINE_REANNOUNCE_INTERVAL = float(os.environ.get("NODE_ONLINE_REANNOUNCE_INTERVAL", "5.0"))
TARGET_FPS       = _get("TARGET_FPS", float)
FPS_STATS_FILE   = _get("FPS_STATS_FILE")

def _safe_float(val: Any, default: float, min_val: float = 0.1, max_val: float = 3600.0) -> float:
    try:
        f = float(val)
        if math.isfinite(f) and min_val <= f <= max_val:
            return f
    except (ValueError, TypeError):
        pass
    return default

JTOP_STALE_S = _safe_float(os.environ.get("JTOP_STALE_S"), default=10.0, min_val=1.0, max_val=120.0)
JTOP_WARN_INTERVAL_S = _safe_float(os.environ.get("JTOP_WARN_INTERVAL_S"), default=300.0, min_val=5.0, max_val=3600.0)

# Time to wait after a worker dies before reopening a jtop session (cooldown).
JTOP_COOLDOWN_S = _safe_float(os.environ.get("JTOP_COOLDOWN_S"), default=30.0, min_val=1.0, max_val=300.0)
# If a worker is alive but not producing fresh values for this many seconds,
# it is considered hung and will be abandoned (a new worker is spawned).
JTOP_HANG_ABANDON_S = _safe_float(os.environ.get("JTOP_HANG_ABANDON_S"), default=60.0, min_val=10.0, max_val=600.0)
# Maximum number of abandoned (hung) workers before we stop reopening sessions.
# Prevents unbounded worker-thread leak when jtop is permanently broken.
JTOP_MAX_ABANDONED_WORKERS = int(os.environ.get("JTOP_MAX_ABANDONED_WORKERS", "5"))

# Rescue ADD unconfirmed (no PLAYING ack) this long → re-arm for retry.
# Mirrors p2p.rescue_ack_timeout_s in edge_node.yml; env override wins.
RESCUE_ACK_TIMEOUT_S = _safe_float(os.environ.get("RESCUE_ACK_TIMEOUT_S"), default=30.0, min_val=1.0, max_val=600.0)

# -----------------------------------------------------------
# Hang diagnostic (faulthandler) interval
# -----------------------------------------------------------
HANG_DIAGNOSTIC_INTERVAL_S = _safe_float(os.environ.get("HANG_DIAGNOSTIC_INTERVAL_S"), default=300.0, min_val=5.0, max_val=3600.0)

# -----------------------------------------------------------
# Log directory override (tmpfs / persistent)
# -----------------------------------------------------------
# Default: Edge/logs (relative, on eMMC).  Set EDGE_LOG_DIR to a tmpfs path
# (e.g. /mnt/ramdisk/edge_logs) to move logs off eMMC without changing run scripts.
EDGE_LOG_DIR = os.environ.get("EDGE_LOG_DIR", "").strip()

# -----------------------------------------------------------
# RTSP Push (Centralized Streaming to Server)
# -----------------------------------------------------------
RTSP_PUSH_URL           = os.environ.get("RTSP_PUSH_URL", "").strip()
# ponytail: 750kbps default fits shared WAN uplink (3 nodes * 2 cams * 750kbps = 4.5Mbps measured capacity)
RTSP_PUSH_BITRATE       = int(os.environ.get("RTSP_PUSH_BITRATE", "750000"))
RTSP_PUSH_MAX_RETRIES   = int(os.environ.get("RTSP_PUSH_MAX_RETRIES", "3"))
RTSP_PUSH_RETRY_DELAY_S = float(os.environ.get("RTSP_PUSH_RETRY_DELAY_S", "1.0"))

# -----------------------------------------------------------
# DeepStream Pipeline Session & Slot Limits
# -----------------------------------------------------------
SPEEDFLOW_SLOT_CAPACITY       = int(os.environ.get("SPEEDFLOW_SLOT_CAPACITY", "16"))
SPEEDFLOW_NVDEC_SESSION_LIMIT = int(os.environ.get("SPEEDFLOW_NVDEC_SESSION_LIMIT", "14"))

# -----------------------------------------------------------
# Network identity
# -----------------------------------------------------------
ADVERTISE_IP = os.environ.get("ADVERTISE_IP", "").strip()

# -----------------------------------------------------------
# Pipeline / Video
# -----------------------------------------------------------
VIDEO_FPS   = _get("VIDEO_FPS", float)
GPU_ID      = _get("GPU_ID", int)
MUX_WIDTH   = _get("MUX_WIDTH", int)
MUX_HEIGHT  = _get("MUX_HEIGHT", int)

VEHICLE_CLASS_IDS       = {2, 3, 5, 7}   # COCO: car, motorbike, bus, truck
LICENSE_PLATE_CLASS_IDS = {0}

# -----------------------------------------------------------
# Paths — relative to ROOT (Edge/)
# Stored as Path objects; absolute only where system requires it
# -----------------------------------------------------------
CAMERAS_YML     = ROOT / _get("CAMERAS_YML")
INFER_CONFIG    = ROOT / _get("INFER_CONFIG")
SGIE_CONFIG     = ROOT / _get("SGIE_CONFIG")
LPR_CONFIG      = ROOT / _get("LPR_CONFIG")
ANALYTICS_CFG   = ROOT / _get("ANALYTICS_CFG")
TRACKER_CFG     = ROOT / _get("TRACKER_CFG")
TRACKER_LPD_CFG = ROOT / _get("TRACKER_LPD_CFG")

# Local LPR worker (Phase 2/3): plate-crop TRT engine + label file.
LPR_ENGINE = ROOT / "models" / "lpr.engine"
LPR_LABELS = ROOT / "configs" / "labels_lpr.txt"

# Absolute — DeepStream system library
TRACKER_LIB     = _get("TRACKER_LIB")

PATH_LOGS       = Path(EDGE_LOG_DIR) if EDGE_LOG_DIR else ROOT / "logs"
PATH_LOGS.mkdir(parents=True, exist_ok=True)

SPEED_LOG = str(ROOT / _get("SPEED_LOG"))
SNAP_DIR  = ROOT / _get("SNAP_DIR")
SNAP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# -----------------------------------------------------------
# Telemetry writer cadence (1-second windows)
# -----------------------------------------------------------
# FPS/feature snapshot is flushed to FPS_STATS_FILE every
# TELEMETRY_INTERVAL seconds.  Both input and output FPS counters
# are drained in the same atomic payload window.
# Default: 1.0 s.  Override per-run via TELEMETRY_INTERVAL env var.
# Both cadences MUST be exactly 1.0 — the profile collector, proactive
# model, SpeedProbe poll, and health-load loop all assume a 1 s cadence.
# Changing either value breaks the fabric and produces non-stationary data.
TELEMETRY_INTERVAL: float = float(os.environ.get("TELEMETRY_INTERVAL", "1.0"))

if abs(HEALTH_INTERVAL - 1.0) > 1e-9:
    raise RuntimeError(
        f"HEALTH_INTERVAL must be 1.0 second (got {HEALTH_INTERVAL!r}). "
        f"Set HEALTH_INTERVAL=1.0 in {_env_path}"
    )
if abs(TELEMETRY_INTERVAL - 1.0) > 1e-9:
    raise RuntimeError(
        f"TELEMETRY_INTERVAL must be 1.0 second (got {TELEMETRY_INTERVAL!r}). "
        f"Set TELEMETRY_INTERVAL=1.0 in {_env_path}, or remove it to accept"
        f" the default"
    )

# -----------------------------------------------------------
# Detection / Speed thresholds
# -----------------------------------------------------------
SPEED_LIMIT_KMH      = _get("SPEED_LIMIT_KMH", float)
JPEG_QUALITY         = _get("JPEG_QUALITY", int)
MAX_SNAPSHOT_PER_ID  = _get("MAX_SNAPSHOT_PER_ID", int)
MIN_WORLD_DISPL_M    = _get("MIN_WORLD_DISPL_M", float)
MAX_ABS_KMH          = _get("MAX_ABS_KMH", float)
BBOX_AREA_JUMP       = _get("BBOX_AREA_JUMP", float)
MIN_DET_CONF         = _get("MIN_DET_CONF", float)
MEDIAN_WINDOW        = _get("MEDIAN_WINDOW", int)

MIN_TRACK_AGE_FRAMES = int(VIDEO_FPS * 0.5)
