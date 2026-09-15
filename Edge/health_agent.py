#!/usr/bin/env python3
"""
Edge/health_agent.py

Health Agent — Collect hardware metrics and publish via Zenoh (peer mode).

Reads all configuration from Edge/.env via speedflow_python.settings.
No default values in this file — all values must be set in .env.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
import threading
import collections
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import os
import msgpack

try:
    import psutil
except ImportError:
    psutil = None

from speedflow_python.log_utils import timed_lock
from speedflow_python.sys_telemetry import read_hw_metrics as _read_hw_sysfs
from speedflow_python.zenoh_session import make_session

# Load settings from .env (must run from Edge/ or have Edge/ in path)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from speedflow_python.settings import (
    NODE_ID,
    HEALTH_INTERVAL,
    HEALTH_LOG_EVERY,
    NODE_ONLINE_REANNOUNCE_INTERVAL,
    TARGET_FPS,
    FPS_STATS_FILE,
    ADVERTISE_IP,
    LOAD_POLICY,
    LOAD_MODEL,
    TELEMETRY_INTERVAL,
    LOG_LEVEL,
    EDGE_LOAD_SCORE_MODE,
    ZENOH_ROUTER,
    ZENOH_ROUTER_STALE_S,
)

try:
    from speedflow_python.settings import (
        JTOP_STALE_S,
        JTOP_WARN_INTERVAL_S,
        JTOP_COOLDOWN_S,
        JTOP_HANG_ABANDON_S,
        JTOP_MAX_ABANDONED_WORKERS,
    )
except (ImportError, AttributeError):
    JTOP_STALE_S = float(os.environ.get("JTOP_STALE_S", "10.0"))
    JTOP_WARN_INTERVAL_S = float(os.environ.get("JTOP_WARN_INTERVAL_S", "300.0"))
    JTOP_COOLDOWN_S = float(os.environ.get("JTOP_COOLDOWN_S", "30.0"))
    JTOP_HANG_ABANDON_S = float(os.environ.get("JTOP_HANG_ABANDON_S", "60.0"))
    JTOP_MAX_ABANDONED_WORKERS = int(os.environ.get("JTOP_MAX_ABANDONED_WORKERS", "5"))
# NOTE: JTOP_* settings above are retained for test/back-compat only — telemetry
# now reads /proc + /sys directly via sys_telemetry (no jtop daemon, no IPC).

def _setup_logging() -> logging.Logger:
    raw_level = LOG_LEVEL
    level = getattr(logging, raw_level, logging.INFO)
    root_level = logging.INFO if raw_level == "DEBUG" else level

    try:
        from speedflow_python.log_utils import install_crash_hooks, FlushFileHandler
        install_crash_hooks()
    except Exception:
        FlushFileHandler = None

    if not logging.root.handlers:
        logging.basicConfig(
            level=root_level,
            format="[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.root.setLevel(root_level)

    if FlushFileHandler is not None:
        try:
            # Use EDGE_LOG_DIR if set, else default to Edge/logs
            from speedflow_python.settings import EDGE_LOG_DIR as _eld
            if _eld:
                log_dir = Path(_eld)
            else:
                log_dir = Path(__file__).resolve().parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = FlushFileHandler(str(log_dir / "health_agent.log"), mode="a", encoding="utf-8")
            fh.setLevel(root_level)
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
            logging.root.addHandler(fh)
        except Exception:
            pass

    if raw_level == "DEBUG":
        for name in (
            "peer_orchestrator",
            "health_agent",
            "speedflow_python.probes",
            "speedflow_python.offload_receiver",
        ):
            logging.getLogger(name).setLevel(logging.DEBUG)

    return logging.getLogger("health_agent")

logger = _setup_logging()


# ---------------------------------------------------------------------------
# Payload freshness/integrity tracking (module-level state)
# ---------------------------------------------------------------------------
# Committed pipeline session_id and last seen sequence number.
# Reset on session change (pipeline restart) or when the payload goes stale
# beyond the operational window.
_state_session_id: str = ""
_state_last_seq: int = -1

# ponytail: bounded age derived from 1s operational cadence.
# 3 * max(TELEMETRY_INTERVAL, HEALTH_INTERVAL) gives ~3 s of headroom
# for a 1 s cadence.  If the atomic payload writer stalls for 3+ s,
# report the pipeline as unavailable rather than replaying stale data.
_STALE_MAX_AGE_S = 3.0 * max(TELEMETRY_INTERVAL, HEALTH_INTERVAL)


# ---------------------------------------------------------------------------
# Unified Payload Reader
# ---------------------------------------------------------------------------

def _read_payload() -> Optional[dict]:
    """
    Read and parse the unified JSON payload written atomically by SpeedProbe.
    Returns the full parsed dict or None on any error (missing file, partial
    JSON, parse error).
    """
    try:
        with open(FPS_STATS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _validate_payload(payload: Optional[dict]) -> bool:
    """
    Freshness + integrity check on a payload dict.
    Rejects:
      - None / empty dict
      - Missing or empty _telemetry.session_id
      - Missing or non-integer _telemetry.sequence
      - Stale _updated_at (older than _STALE_MAX_AGE_S relative to now)

    Only advances _state_last_seq when session matches the committed session.
    Session changes are accepted (pipeline restart) — the first valid payload
    of a new session resets both session_id and last_seq.
    """
    global _state_session_id, _state_last_seq

    if not payload:
        return False

    telemetry = payload.get("_telemetry")
    if not isinstance(telemetry, dict):
        return False

    sess_id = telemetry.get("session_id")
    seq = telemetry.get("sequence")
    if not isinstance(sess_id, str) or not sess_id:
        return False
    if not isinstance(seq, int) or seq < 0:
        return False

    # Staleness: reject if _updated_at is too old.
    # Note: _updated_at is generated by the local SpeedProbe on the same Jetson host
    # (single-node producer-consumer snapshot file), so local wall-clock comparison is safe.
    updated_at = payload.get("_updated_at")
    if not isinstance(updated_at, (int, float)):
        return False
    if time.time() - updated_at > _STALE_MAX_AGE_S:
        logger.debug(
            "[HealthAgent] Stale payload: _updated_at=%.1f age=%.1fs > %.1fs",
            updated_at, time.time() - updated_at, _STALE_MAX_AGE_S,
        )
        return False

    # Sequence advancement tracking
    if sess_id != _state_session_id:
        # New session (pipeline restart) — reset
        _state_session_id = sess_id
        _state_last_seq = seq
        return True

    # Same session: must strictly advance
    if seq <= _state_last_seq:
        logger.debug(
            "[HealthAgent] Non-advancing seq: got %d, last %d (session %s)",
            seq, _state_last_seq, sess_id,
        )
        return False

    _state_last_seq = seq
    return True


def _payload_parts(
    payload: Optional[dict],
) -> tuple:
    """
    Safely extract telemetry parts from a single payload dict.
    Returns (fps_stats, feature_stats, offload_crops, service_stats) with safe defaults.

    Caller must have validated payload freshness via _validate_payload()
    before calling this function.  An invalid payload passed here will
    produce an empty fps_stats dict (score 100 = unavailable).
    """
    if payload is None:
        return {}, {}, {"received_per_s": 0.0}, {}
    fps_stats = {k: v for k, v in payload.items()
                 if not k.startswith("_") and isinstance(v, (int, float))}
    feature_stats = payload.get("_features", {})
    offload_crops = payload.get("_offload_crops", {"received_per_s": 0.0})
    service_stats = payload.get("_service", {})
    if not isinstance(service_stats, dict):
        service_stats = {}
    return fps_stats, feature_stats, offload_crops, service_stats


def _detect_source_starved(
    fps_stats: dict,
    input_fps: dict,
    edge_cfg: dict,
    source_type_map: Optional[dict] = None,
) -> set:
    """
    Detect cameras whose source (upstream feed) is starved.

    A camera is source-starved ONLY when BOTH conditions hold:
      1. Input rate is absent/zero or materially below the expected source rate.
      2. Output rate is also absent/low (not a pure output transient).

    Source-type gate (Phase 1 validity contract):
      ``source_type_map`` maps camera_id → ``"live"`` | ``"file"`` (derived
      from the camera URI in cameras.yml).  File-playback cameras are NEVER
      classified as source-starved: even with PTS-derived input FPS (which
      reflects the native source rate, not decoder throughput), file playback
      is not a live upstream feed — a low PTS-measured rate means the file
      is playing slowly or the muxer is PTS-paced, not that the source is
      starved.  This is a DEVICE GATE — realtime source-starvation enforcement
      for file playback is not implemented (see core_pipeline.streammux
      ``live-source``); do not apply live-feed starvation math to files.
      When source_type_map is None/missing the gate is inert — every camera
      is evaluated exactly as before (backward compatible).

    When _input_fps is unavailable (empty/missing/malformed), returns an empty
    set — preserving current FPS-score behaviour exactly.

    Malformed fps_stats values (None, string, bool, NaN, inf, negative) are
    treated as unavailable (0.0) — the camera is still evaluated with its
    paired input_fps value, so a valid input alone cannot induce starvation.

    Malformed config scalars fall back to defaults (expected_source_rate=25.0,
    starved_threshold_ratio=0.2).  An unusable threshold (≤0, NaN, inf) causes
    an early empty-set return — nothing is starvable.

    Configuration (edge_node.yml, ``source_starved`` section):
      expected_source_rate   float   default 25.0  (fps, matches cameras.yml)
      starved_threshold_ratio float  default 0.2   (below 20 % = starved → 5.0 fps)
    """
    if not isinstance(input_fps, dict) or not input_fps:
        return set()

    # ponytail: guard against non-dict callers before any .get/.items
    if not isinstance(fps_stats, dict):
        fps_stats = {}
    if not isinstance(edge_cfg, dict):
        edge_cfg = {}
    if not isinstance(source_type_map, dict):
        source_type_map = {}

    sc_cfg = edge_cfg.get("source_starved", {})
    if not isinstance(sc_cfg, dict):
        sc_cfg = {}

    # ── Safe config scalars: malformed → default; negative/non-finite → default ──
    def _safe_cfg_float(v, default):
        """Convert *v* to a non-negative finite float; any malformed input → *default*."""
        if v is None:
            return default
        if isinstance(v, bool):
            return default
        if isinstance(v, (int, float)):
            if math.isfinite(v) and v >= 0.0:
                return float(v)
            return default
        if isinstance(v, str):
            try:
                fv = float(v)
                if math.isfinite(fv) and fv >= 0.0:
                    return fv
            except (ValueError, TypeError):
                pass
        return default

    expected = _safe_cfg_float(sc_cfg.get("expected_source_rate"), 25.0)
    ratio    = _safe_cfg_float(sc_cfg.get("starved_threshold_ratio"), 0.2)
    threshold = expected * ratio

    # ponytail: unusable threshold → nothing is starvable (defensible empty)
    if not (math.isfinite(threshold) and threshold > 0.0):
        return set()

    # ── Safe FPS values: malformed → 0.0 (unavailable) ──
    def _safe_fps(v):
        """Convert *v* to a non-negative finite float; any malformed input → 0.0."""
        if v is None:
            return 0.0
        if isinstance(v, bool):
            return 0.0
        if isinstance(v, (int, float)):
            if math.isfinite(v) and v >= 0.0:
                return float(v)
            return 0.0
        if isinstance(v, str):
            try:
                fv = float(v)
                if math.isfinite(fv) and fv >= 0.0:
                    return fv
            except (ValueError, TypeError):
                pass
        return 0.0

    starved: set = set()
    for cam_id in set(fps_stats) | set(input_fps):
        # File-playback cameras are excluded from starvation classification:
        # their input FPS is decoder throughput, not an upstream feed rate.
        # See the source_type_map docstring above (device gate).
        if source_type_map.get(cam_id) == "file":
            continue
        in_fps  = _safe_fps(input_fps.get(cam_id))
        out_fps = _safe_fps(fps_stats.get(cam_id))
        if in_fps < threshold and out_fps < threshold:
            starved.add(cam_id)

    return starved


def _derive_camera_workload(
    feature_stats: dict,
    fps_stats: dict,
    starved_cams: set = None,
) -> dict:
    """
    Derive per-camera workload {camera_id: n_track + n_plate} from the same
    telemetry window's _features snapshot.

    Only active (fps > 0), non-source-starved cameras are included.
    A camera is skipped — never crashes the payload builder — when its
    n_track/n_plate are missing, non-numeric, bool, non-finite, or negative.
    """
    result: Dict[str, float] = {}
    starved = starved_cams or set()

    for cam_id, feats in feature_stats.items():
        if not isinstance(feats, dict) or cam_id in starved:
            continue
        fps = fps_stats.get(cam_id, 0.0)
        if not isinstance(fps, (int, float)) or fps <= 0.0:
            continue  # inactive camera
        n_track = feats.get("n_track")
        n_plate = feats.get("n_plate")
        if not isinstance(n_track, (int, float)) or isinstance(n_track, bool):
            continue
        if not isinstance(n_plate, (int, float)) or isinstance(n_plate, bool):
            continue
        if not math.isfinite(n_track) or n_track < 0:
            continue
        if not math.isfinite(n_plate) or n_plate < 0:
            continue
        result[cam_id] = n_track + n_plate

    return result


def _derive_camera_liveness(source_modes: dict, fps_stats: dict) -> tuple:
    """Split liveness from throughput for a pipeline snapshot.

    Returns (attached_cameras, streaming_cameras, active_cameras):
      - attached_cameras: pipeline-attached cameras (source_modes keys, falling
        back to fps_stats keys). Liveness for ownership/failover — independent
        of instantaneous FPS.
      - streaming_cameras: cameras with FPS>0. Throughput, for load only.
      - active_cameras: alias of attached_cameras (liveness).
    """
    attached = sorted(set(source_modes.keys()) | set(fps_stats.keys()))
    streaming = [k for k, v in fps_stats.items() if v > 0.0]
    return attached, streaming, attached


def _read_pipeline_snapshot() -> tuple:
    """
    Read the pipeline JSON once, validate freshness/integrity, return all parts.

    Returns (valid: bool, fps_stats, feature_stats, offload_crops,
             service_stats, input_fps, source_modes, telemetry).

    ``source_modes`` is ``_telemetry.source_modes`` from the probe payload
    (camera_id → "live" | "file").  Missing or malformed → {} so callers
    that don't yet pass it to _detect_source_starved remain backward
    compatible.

    ``telemetry`` is the full ``_telemetry`` dict from the probe payload
    (carries session_id, sequence, configured_fps_per_camera used as dedup
    keys / static camera metadata in the heartbeat).  {} on invalid.

    When valid=False:
      fps_stats is {} and the caller must not use telemetry-derived
      values (fps_stats, features, offload rate) for load scoring or
      payload publishing.  Callers report the pipeline as unavailable.
    """
    payload = _read_payload()
    if not _validate_payload(payload):
        return False, {}, {}, {}, {}, {}, {}, {}
    input_fps = payload.get("_input_fps", {})
    if not isinstance(input_fps, dict):
        input_fps = {}
    source_modes = {}
    telemetry = {}
    raw_telemetry = payload.get("_telemetry")
    if isinstance(raw_telemetry, dict):
        telemetry = raw_telemetry
        m = raw_telemetry.get("source_modes")
        if isinstance(m, dict):
            source_modes = {str(k): str(v) for k, v in m.items()}
    parts = _payload_parts(payload)
    return True, parts[0], parts[1], parts[2], parts[3], input_fps, source_modes, telemetry


# ---------------------------------------------------------------------------
# Metric Collector
# ---------------------------------------------------------------------------

# _JTOP_WARN_INTERVAL_S: low-stakes log rate limit (default 300s) sourced from settings.py/env;
# kept out of fast-path config to avoid churn while remaining environment-overridable.
_JTOP_WARN_INTERVAL_S = JTOP_WARN_INTERVAL_S
_JTOP_LAST_WARN_TS: float = 0.0


def _collect_jetson_metrics() -> Dict:
    """Read hardware metrics via direct /proc + /sys reads (no daemon, no IPC).

    The sysfs reader never fails hard — it always returns a valid dict with
    metrics defaulting to 0.0 on any read error, so the health loop never crashes.
    """
    return _read_hw_sysfs()


# Path to edge_node.yml and mtime for reload-on-use
_EDGE_NODE_YML = Path(__file__).resolve().parent / "configs" / "edge_node.yml"
_EDGE_CFG: dict = {}
_EDGE_CFG_MTIME: float = 0.0


def _load_edge_node_cfg() -> dict:
    """
    Read edge_node.yml. Returns the full parsed dict, or {} on error.
    Used by _maybe_reload_edge_cfg for mtime-based hot-reload.
    """
    try:
        import yaml
        with open(_EDGE_NODE_YML, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.debug("[HealthAgent] edge_node.yml load error: %s", exc)
        return {}


def _maybe_reload_edge_cfg() -> None:
    """
    Reload _EDGE_CFG if edge_node.yml mtime has changed since last read.
    Idempotent — called before every config-consuming operation.
    """
    global _EDGE_CFG, _EDGE_CFG_MTIME
    try:
        mtime = _EDGE_NODE_YML.stat().st_mtime
    except OSError:
        mtime = 0.0
    if mtime != _EDGE_CFG_MTIME:
        _EDGE_CFG = _load_edge_node_cfg()
        _EDGE_CFG_MTIME = mtime
        logger.info("[HealthAgent] edge_node.yml reloaded (mtime changed)")


def get_edge_cfg() -> dict:
    """Return the latest edge_node.yml config, reloading when the file changed."""
    _maybe_reload_edge_cfg()
    return _EDGE_CFG


# Seed at import time once, then mtime-based reload kicks in on each use.
_EDGE_CFG = _load_edge_node_cfg()
try:
    _EDGE_CFG_MTIME = _EDGE_NODE_YML.stat().st_mtime
except OSError:
    _EDGE_CFG_MTIME = 0.0
_FPS_HISTORY: Deque[Tuple[float, float]] = collections.deque(maxlen=20)


def _update_service_ema_state(
    service_stats: dict,
    prev_state: dict,
    now_mono: float,
    s_alpha: float = 0.30,
    s_stale: float = 30.0,
) -> dict:
    """
    Pure state updater for completion-primary service EMA.
    Uses deltas over interval, tracks pending tracks (born - expired),
    and enforces stale idle recovery only when there is zero active load.
    """
    fin = int(service_stats.get("plates_finalized", 0) or 0)
    miss = int(service_stats.get("tracks_missed", 0) or 0)
    born = int(service_stats.get("tracks_born", 0) or 0)
    exp = int(service_stats.get("tracks_expired", 0) or 0)

    pending_tracks = max(0, born - exp)
    prev_fin = prev_state.get("prev_fin")
    prev_miss = prev_state.get("prev_miss")
    current_ema = prev_state.get("service_ema")
    last_busy_ts = prev_state.get("last_busy_ts", now_mono)
    last_update_ts = prev_state.get("last_update_ts", now_mono)

    try:
        alpha_val = float(s_alpha)
        if not math.isfinite(alpha_val) or alpha_val <= 0.0 or alpha_val > 1.0:
            alpha_clamped = 0.30
        else:
            alpha_clamped = alpha_val
    except (TypeError, ValueError):
        alpha_clamped = 0.30

    try:
        stale_val = float(s_stale)
        if not math.isfinite(stale_val) or stale_val <= 0.0:
            stale_clamped = 30.0
        else:
            stale_clamped = max(0.1, stale_val)
    except (TypeError, ValueError):
        stale_clamped = 30.0

    if prev_fin is None or prev_miss is None:
        # First sample: initialize baseline counters.
        # If there are already historical lifetime metrics in /dev/shm snapshot,
        # seed EMA with the historical ratio instead of falsely claiming 1.0 (perfect).
        if current_ema is not None:
            initial_ema = min(1.0, max(0.0, float(current_ema)))
        elif (fin + miss) > 0:
            initial_ema = min(1.0, max(0.0, float(fin) / float(fin + miss)))
        else:
            initial_ema = 1.0

        return {
            "service_ema": initial_ema,
            "prev_fin": fin,
            "prev_miss": miss,
            "last_busy_ts": now_mono,
            "last_update_ts": now_mono,
            "delta_fin": 0,
            "delta_miss": 0,
            "pending_tracks": pending_tracks,
            "idle_s": 0.0,
            "cold_start": True,
        }

    # Handle counter reset / process restart
    d_fin = fin - prev_fin if fin >= prev_fin else fin
    d_miss = miss - prev_miss if miss >= prev_miss else miss
    d_fin = max(0, d_fin)
    d_miss = max(0, d_miss)

    d_denom = d_fin + d_miss
    new_ema = 1.0 if current_ema is None else min(1.0, max(0.0, float(current_ema)))

    if d_denom > 0:
        inst_c = d_fin / float(d_denom)
        new_ema = alpha_clamped * inst_c + (1.0 - alpha_clamped) * new_ema
        new_ema = min(1.0, max(0.0, new_ema))
        last_busy_ts = now_mono
        last_update_ts = now_mono
        idle_s = 0.0
    elif pending_tracks > 0:
        # Tracks in flight: system is actively processing, keep EMA and mark as busy
        last_busy_ts = now_mono
        last_update_ts = now_mono
        idle_s = 0.0
    else:
        # Zero completions and zero pending tracks in flight.
        idle_s = max(0.0, now_mono - last_busy_ts)
        elapsed_since_update = max(0.0, now_mono - last_update_ts)
        if idle_s >= stale_clamped and elapsed_since_update >= 1.0:
            # Smoothly recover towards 1.0 only after the full stale idle duration has elapsed
            recovery_delta = min(1.0 - new_ema, 0.10 * elapsed_since_update)
            new_ema = min(1.0, max(0.0, new_ema + max(0.0, recovery_delta)))
            last_update_ts = now_mono

    return {
        "service_ema": new_ema,
        "prev_fin": fin,
        "prev_miss": miss,
        "last_busy_ts": last_busy_ts,
        "last_update_ts": last_update_ts,
        "delta_fin": d_fin,
        "delta_miss": d_miss,
        "pending_tracks": pending_tracks,
        "idle_s": round(idle_s, 2),
        "cold_start": False,
    }


# ── Config-safe float and emergency helpers ────────────────────────────────
def _finite_positive(v):
    """Return float(v) for finite v > 0.0 (not bool); None otherwise."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0:
        return float(v)
    if isinstance(v, str):
        try:
            fv = float(v)
            if math.isfinite(fv) and fv > 0.0:
                return fv
        except (ValueError, TypeError):
            pass
    return None


def _finite_nonneg(v):
    """Return float(v) for finite numeric v >= 0.0 (not bool); None otherwise."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0.0:
        return float(v)
    if isinstance(v, str):
        try:
            fv = float(v)
            if math.isfinite(fv) and fv >= 0.0:
                return fv
        except (ValueError, TypeError):
            pass
    return None


def _unit_interval(v, default):
    """Return float(v) strictly within (0, 1) for finite numeric v (not bool);
    any malformed input (None, bool, non-finite, <=0, >=1) → *default*.

    Used for EMA smoothing weights so a bad config value can never invert the
    EMA (alpha > 1 would weight the new sample more than 100%) or zero it out.
    """
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)) and math.isfinite(v) and 0.0 < v < 1.0:
        return float(v)
    if isinstance(v, str):
        try:
            fv = float(v)
            if math.isfinite(fv) and 0.0 < fv < 1.0:
                return fv
        except (ValueError, TypeError):
            pass
    return default


def _resolve_emergency_thresholds(
    emergency: Any = None,
    wp_cfg: Optional[dict] = None,
) -> Tuple[float, float, float]:
    """Resolve emergency fuse thresholds (gpu_pct, gpu_fps, fps) with safe fallbacks.

    Returns a 3-tuple: (em_gpu_pct, em_gpu_fps, em_fps).
    Precedence:
      - emergency as a 3-tuple/list: (gpu_pct, gpu_fps, fps)
      - emergency dict: {gpu_pct, gpu_fps, fps}
      - fallback to _EDGE_CFG.load_score.emergency
      - fallback for fps: wp_cfg.fps_emergency or service.fps_emergency
      - final safe fallbacks: (99.0, 15.0, 12.0)

    Preserves GPU as witness-only sustained fuse, never rho input.
    """
    if isinstance(emergency, (tuple, list)) and len(emergency) >= 3:
        gpu_pct = _finite_positive(emergency[0])
        gpu_fps = _finite_positive(emergency[1])
        fps = _finite_positive(emergency[2])
        return (
            gpu_pct if gpu_pct is not None else 99.0,
            gpu_fps if gpu_fps is not None else 15.0,
            fps if fps is not None else 12.0,
        )

    em_cfg = emergency if isinstance(emergency, dict) else (
        _EDGE_CFG.get("load_score", {}).get("emergency", {})
        if isinstance(_EDGE_CFG.get("load_score"), dict) else {}
    )
    gpu_pct = _finite_positive(em_cfg.get("gpu_pct")) or 99.0
    gpu_fps = _finite_positive(em_cfg.get("gpu_fps")) or 15.0
    fps = _finite_positive(em_cfg.get("fps"))
    if fps is None and isinstance(wp_cfg, dict):
        fps = _finite_positive(wp_cfg.get("fps_emergency"))
    if fps is None:
        ls = _EDGE_CFG.get("load_score", {}) if isinstance(_EDGE_CFG.get("load_score"), dict) else {}
        svc = ls.get("service", {}) if isinstance(ls.get("service"), dict) else {}
        fps = _finite_positive(svc.get("fps_emergency"))
    if fps is None:
        fps = 12.0
    return (gpu_pct, gpu_fps, fps)


def _resolve_jtop_stale_s(cfg: Optional[dict] = None) -> float:
    """Resolve jtop stale threshold (seconds) with safe bounds [1.0, 120.0].
    Precedence: config (load_score.jtop_stale_s) -> settings.JTOP_STALE_S -> env -> 10.0.
    """
    raw = None
    if isinstance(cfg, dict):
        ls = cfg.get("load_score")
        if isinstance(ls, dict):
            raw = ls.get("jtop_stale_s")
    if raw is None:
        try:
            from speedflow_python.settings import JTOP_STALE_S as _s_stale
            raw = _s_stale
        except (ImportError, AttributeError):
            pass
    if raw is None:
        raw = os.environ.get("JTOP_STALE_S")
    val = _finite_positive(raw)
    if val is None:
        return 10.0
    return max(1.0, min(120.0, val))


def _gpu_fps_dwell_update(
    gpu_pct,
    eff_fps,
    prev_count: int,
    gpu_threshold: float = 95.0,
    fps_threshold: float = 24.0,
    dwell: int = 3,
) -> tuple:
    """
    GPU-as-witness dwell fuse state machine (pure).

    A sample qualifies when GPU% >= gpu_threshold AND effective FPS < fps_threshold.
    Each qualifying sample increments the consecutive count; a non-qualifying sample
    resets it to 0.  The fuse is ARMED once the count reaches ``dwell`` consecutive
    qualifying samples.

    Invalid/non-finite gpu_pct or eff_fps FAIL OPEN: the sample is treated as
    non-qualifying (count reset to 0, never armed) so bad telemetry cannot trip
    the fuse.

    Returns (armed: bool, new_count: int).
    """
    gt = _finite_positive(gpu_threshold)
    if gt is None:
        gt = 95.0
    ft = _finite_positive(fps_threshold)
    if ft is None:
        ft = 24.0
    d = int(dwell) if isinstance(dwell, (int, float)) and dwell > 0 else 3

    gpu = _finite_nonneg(gpu_pct)
    fps = _finite_nonneg(eff_fps)
    if gpu is None or fps is None:
        return False, 0
    if gpu >= gt and fps < ft:
        new_count = prev_count + 1
        return new_count >= d, new_count
    return False, 0


def _calc_workload_pressure(
    wp_cfg: dict,
    eff_wl: float,
    eff_fps: float,
    hw_fuse_score_floor: float,
    metrics: dict = None,
    service_ema: Optional[float] = None,
    n_active: int = 0,
    gpu_fps_dwell_armed: bool = False,
    emergency: Optional[dict] = None,
    **kwargs,
) -> float:
    """
    ponytail: asymptotic workload pressure [0..100).

    load_score = 100.0 * rho / (1.0 + rho), strictly < 100.0.
    100.0 represents infinite delay / hardware collapse, unreachable alive.
    rho = rho_s + rho_d + rho_r + rho_v (orthogonal load dimensions):
      - rho_s : stream concurrency (EMC LPDDR5 bus convex knee)
      - rho_d : vehicle workload demand vs capacity w_high
      - rho_r : CPU/RAM hardware contention (asymptotic)
      - rho_v : service completion deficit (odds of missed plates)
    Per ADR-0001: FPS is a safety witness only, never the primary driver.
    GPU% is excluded from rho (DVFS noise) and acts only as an emergency fuse.
    """
    k_s = _finite_positive(wp_cfg.get("stream_knee")) or 2.5
    p_s = _finite_positive(wp_cfg.get("stream_exp")) or 2.2
    w_sat = _finite_positive(wp_cfg.get("w_high")) or 10.0
    eps = 1e-3

    # ── Stream concurrency axis: non-linear convex knee on EMC bus ──
    rho_s = (max(0.0, float(n_active) - 1.0) / k_s) ** p_s

    # ── Demand axis: vehicle workload ratio ──
    rho_d = max(0.0, eff_wl) / max(0.001, w_sat)

    # ── Resource axis: CPU/RAM saturation (GPU excluded from rho due to DVFS) ──
    u_safe = _finite_positive(wp_cfg.get("u_safe")) or 0.60
    u = 0.0
    if isinstance(metrics, dict):
        vals = [
            v for v in (
                metrics.get("cpu_percent"),
                metrics.get("ram_percent"),
            )
            if isinstance(v, (int, float)) and math.isfinite(v)
        ]
        if vals:
            u = min(1.0, max(0.0, max(vals) / 100.0))
    rho_r = 0.0 if u <= u_safe else (u - u_safe) / max(0.05, 1.05 - u)

    # ── Service axis: bounded completion deficit vs achievable floor ──
    if service_ema is not None and math.isfinite(service_ema):
        svc = max(0.0, min(1.0, float(service_ema)))
        s_target = _finite_positive(wp_cfg.get("svc_target")) or 0.95
        s_floor = _finite_positive(wp_cfg.get("svc_floor")) or 0.50
        rho_v_max = _finite_positive(wp_cfg.get("rho_v_max")) or 1.0
        deficit = (s_target - svc) / max(1e-3, s_target - s_floor)
        rho_v = rho_v_max * max(0.0, min(1.0, deficit))
    else:
        rho_v = 0.0

    rho = rho_s + rho_d + rho_r + rho_v
    # Asymptotic kernel: strictly in [0.0, 100.0) for all finite rho >= 0
    raw = 100.0 * rho / (1.0 + rho)

    # ── Emergency fuses (floors, never primary path) ──
    if emergency is None and "emergency_cfg" in kwargs:
        emergency = kwargs["emergency_cfg"]
    em_gpu_pct, em_gpu_fps, em_fps = _resolve_emergency_thresholds(emergency, wp_cfg=wp_cfg)

    gpu = metrics.get("gpu_percent") if isinstance(metrics, dict) else None
    if isinstance(gpu, (int, float)) and math.isfinite(gpu) and gpu >= em_gpu_pct and eff_fps < em_gpu_fps:
        raw = max(raw, min(99.9, hw_fuse_score_floor))

    if eff_fps < em_fps:
        raw = max(raw, min(99.9, hw_fuse_score_floor))

    # GPU-as-witness dwell fuse: armed only after N consecutive qualifying
    # samples (GPU>=95% AND FPS<24).  Same floor as the other emergency fuses.
    if gpu_fps_dwell_armed:
        raw = max(raw, min(99.9, hw_fuse_score_floor))

    return min(99.9, max(0.0, raw))


def _compute_load_score(
    metrics: dict,
    fps_stats: dict,
    source_starved_cameras: set = None,
    feature_stats: Optional[dict] = None,
    workload_ema: Optional[float] = None,
    fps_ema: Optional[float] = None,
    service_ema: Optional[float] = None,
    gpu_fps_dwell_armed: bool = False,
    emergency: Optional[Any] = None,
) -> tuple:
    """
    Completion-primary load score with legacy fallback.
    Supports mode="service" (completion-primary) and mode="legacy" / "workload_primary".
    """
    _starved = source_starved_cameras or set()

    _maybe_reload_edge_cfg()
    ls_cfg = _EDGE_CFG.get("load_score", {})
    if not isinstance(ls_cfg, dict):
        ls_cfg = {}
    mode = str(EDGE_LOAD_SCORE_MODE or ls_cfg.get("mode", "service")).strip().lower()

    hw_fuse_threshold   = float(ls_cfg.get("hw_fuse_threshold",   90.0))
    hw_fuse_score_floor = float(ls_cfg.get("hw_fuse_score_floor", 80.0))
    fps_clamp_margin    = _finite_nonneg(ls_cfg.get("fps_clamp_margin"))
    if fps_clamp_margin is None:
        fps_clamp_margin = 2.0

    wp_cfg = ls_cfg.get("workload_policy", {})
    emergency_thresholds = _resolve_emergency_thresholds(
        emergency if emergency is not None else ls_cfg.get("emergency"),
        wp_cfg=wp_cfg if isinstance(wp_cfg, dict) else None,
    )

    # ── FPS component calculation ───────────────────────────────
    active_fps_vals = [
        v for k, v in fps_stats.items()
        if v > 0.0 and k not in _starved
    ] if isinstance(fps_stats, dict) else []
    if active_fps_vals:
        avg_fps = sum(active_fps_vals) / len(active_fps_vals)
    elif not (isinstance(ls_cfg.get("workload_policy"), dict)
              and ls_cfg["workload_policy"].get("enabled")):
        # No FPS and no workload policy → score undefined; report unavailable.
        return 0.0, "no_fps"
    else:
        # ADR-0001: workload/resource/service axes must still drive the score
        # when FPS is absent (FPS is a safety witness, not a requirement). Proceed
        # with zero FPS so resource saturation can still signal overload.
        avg_fps = 0.0

    fps_clamped = max(0.0, min(float(TARGET_FPS), avg_fps))
    curr_wl = sum(_derive_camera_workload(feature_stats or {}, fps_stats, _starved).values()) if isinstance(feature_stats, dict) else 0.0
    eff_wl = workload_ema if (workload_ema is not None and math.isfinite(workload_ema)) else curr_wl
    eff_fps = fps_ema if (fps_ema is not None and math.isfinite(fps_ema)) else avg_fps

    # ── Unified Service Score Mode (workload + completion + fps floors) ──
    if mode == "service":
        if isinstance(wp_cfg, dict) and wp_cfg.get("enabled", True):
            score = _calc_workload_pressure(
                wp_cfg, eff_wl, eff_fps, hw_fuse_score_floor,
                metrics=metrics, service_ema=service_ema,
                n_active=len(active_fps_vals),
                gpu_fps_dwell_armed=gpu_fps_dwell_armed,
                emergency=emergency_thresholds,
            )
        else:
            score = 0.0
        return round(min(99.9, max(0.0, score)), 1), "service_primary"

    # ── Workload-primary demand/resource/service policy ──────────
    if isinstance(wp_cfg, dict) and wp_cfg.get("enabled") is True:
        # Full demand/resource/service model (service axis via service_ema).
        score = _calc_workload_pressure(
            wp_cfg, eff_wl, eff_fps, hw_fuse_score_floor,
            metrics=metrics, service_ema=service_ema,
            n_active=len(active_fps_vals),
            gpu_fps_dwell_armed=gpu_fps_dwell_armed,
            emergency=emergency_thresholds,
        )

        # No de-escalation veto: healthy FPS must not mask high demand/resource.

        # Hardware emergency fuse
        hw_saturated = (
            isinstance(metrics, dict) and (
                float(metrics.get("cpu_percent", 0.0)) >= hw_fuse_threshold or
                float(metrics.get("ram_percent", 0.0)) >= hw_fuse_threshold
            )
        )
        if hw_saturated and fps_clamped < float(TARGET_FPS) - fps_clamp_margin:
            score = max(score, min(99.9, hw_fuse_score_floor))

        return round(min(99.9, max(0.0, score)), 1), "workload_primary"

    if fps_clamped >= float(TARGET_FPS):
        fps_score = 0.0
    elif fps_clamped >= 22.0:
        fps_score = 57.0 * (float(TARGET_FPS) - fps_clamped) / (float(TARGET_FPS) - 22.0)
    elif fps_clamped >= 19.0:
        fps_score = 57.0 + (65.0 - 57.0) * (22.0 - fps_clamped) / (22.0 - 19.0)
    elif fps_clamped >= 17.0:
        fps_score = 65.0 + (75.0 - 65.0) * (19.0 - fps_clamped) / (19.0 - 17.0)
    else:
        fps_score = 75.0 + (100.0 - 75.0) * (17.0 - fps_clamped) / (17.0 - 0.0)

    # ── Workload bonus (n_track + n_plate / capacity) ───────────
    workload_bonus = 0.0
    wl_cfg = ls_cfg.get("workload", {})
    if (isinstance(wl_cfg, dict) and wl_cfg.get("enabled") is True
            and isinstance(feature_stats, dict)):
        wl_cap = _finite_positive(wl_cfg.get("capacity"))
        wl_max = _finite_positive(wl_cfg.get("max_bonus")) or 15.0
        if wl_cap is not None:
            wl_total = sum(
                _derive_camera_workload(feature_stats, fps_stats, _starved).values()
            )
            workload_bonus = min(wl_max, max(0.0, wl_max * (wl_total / wl_cap)))

    # ── Thermal bonus (gpu_temp_c ramp [onset, critical]) ───────
    thermal_bonus = 0.0
    th_cfg = ls_cfg.get("thermal", {})
    if isinstance(th_cfg, dict) and th_cfg.get("enabled") is True and isinstance(metrics, dict):
        temp_val = _finite_nonneg(metrics.get("gpu_temp_c"))
        onset    = _finite_nonneg(th_cfg.get("onset_c"))
        critical = _finite_nonneg(th_cfg.get("critical_c"))
        th_max   = _finite_positive(th_cfg.get("max_bonus")) or 5.0
        if temp_val is not None and onset is not None and critical is not None and onset < critical:
            if temp_val >= critical:
                thermal_bonus = th_max
            elif temp_val > onset:
                thermal_bonus = th_max * (temp_val - onset) / (critical - onset)

    # ── FPS trend bonus (decline rate) ──────────────────────────
    trend_bonus = 0.0
    trend_cfg = ls_cfg.get("trend", {})
    now = time.monotonic()
    if isinstance(trend_cfg, dict) and trend_cfg.get("enabled") is True:
        max_decline = _finite_positive(trend_cfg.get("max_decline_fps_per_s")) or 2.0
        tr_max      = _finite_positive(trend_cfg.get("max_bonus")) or 5.0
        if _FPS_HISTORY:
            # find oldest sample within window
            t_past, fps_past = _FPS_HISTORY[0]
            for t_h, f_h in _FPS_HISTORY:
                if now - t_h >= 0.5:
                    t_past, fps_past = t_h, f_h
                    break
            dt = now - t_past
            slope = 0.0
            if dt >= 0.5:
                slope = (avg_fps - fps_past) / dt
            decline = max(0.0, -slope)
            trend_bonus = min(tr_max, max(0.0, tr_max * (decline / max_decline)))
    _FPS_HISTORY.append((now, avg_fps))

    raw_composite = fps_score + workload_bonus + thermal_bonus + trend_bonus
    composite = min(100.0, max(0.0, raw_composite))

    # ── Hardware emergency floor ────────────────────────────────
    hw_saturated = (
        isinstance(metrics, dict) and (
            float(metrics.get("cpu_percent", 0.0)) >= hw_fuse_threshold or
            float(metrics.get("ram_percent", 0.0)) >= hw_fuse_threshold
        )
    )
    fps_emergency = fps_clamped < float(TARGET_FPS) - fps_clamp_margin

    if hw_saturated and fps_emergency:
        score = max(composite, hw_fuse_score_floor)
    else:
        score = composite

    return round(score, 1), "fps_dominant"


def _compute_load_score_breakdown(
    metrics: dict,
    fps_stats: dict,
    source_starved_cameras: set = None,
    feature_stats: Optional[dict] = None,
    workload_ema: Optional[float] = None,
    fps_ema: Optional[float] = None,
    service_ema: Optional[float] = None,
    service_delta_fin: int = 0,
    service_delta_miss: int = 0,
    service_pending_tracks: int = 0,
    service_idle_s: float = 0.0,
    service_cold_start: bool = False,
    gpu_fps_dwell_armed: bool = False,
    emergency: Optional[Any] = None,
) -> dict:
    """
    Pure helper yielding auditable breakdown of the load score computation.
    """
    _starved = source_starved_cameras or set()

    if not isinstance(fps_stats, dict):
        return {
            "fps_score": 0.0,
            "workload_bonus": 0.0,
            "thermal_bonus": 0.0,
            "recv_bonus": 0.0,
            "trend_bonus": 0.0,
            "composite_score": 0.0,
            "load_score": 0.0,
        }

    _maybe_reload_edge_cfg()
    ls_cfg = _EDGE_CFG.get("load_score", {})
    if not isinstance(ls_cfg, dict):
        ls_cfg = {}
    mode = str(EDGE_LOAD_SCORE_MODE or ls_cfg.get("mode", "service")).strip().lower()

    hw_fuse_threshold   = float(ls_cfg.get("hw_fuse_threshold",   90.0))
    hw_fuse_score_floor = float(ls_cfg.get("hw_fuse_score_floor", 80.0))
    fps_clamp_margin    = _finite_nonneg(ls_cfg.get("fps_clamp_margin"))
    if fps_clamp_margin is None:
        fps_clamp_margin = 2.0

    wp_cfg = ls_cfg.get("workload_policy", {})
    emergency_thresholds = _resolve_emergency_thresholds(
        emergency if emergency is not None else ls_cfg.get("emergency"),
        wp_cfg=wp_cfg if isinstance(wp_cfg, dict) else None,
    )

    active_fps_vals = [
        v for k, v in fps_stats.items()
        if v > 0.0 and k not in _starved
    ]
    if active_fps_vals:
        avg_fps = sum(active_fps_vals) / len(active_fps_vals)
    elif not (isinstance(ls_cfg.get("workload_policy"), dict)
              and ls_cfg["workload_policy"].get("enabled")):
        return {
            "fps_score": 0.0,
            "workload_bonus": 0.0,
            "thermal_bonus": 0.0,
            "recv_bonus": 0.0,
            "trend_bonus": 0.0,
            "composite_score": 0.0,
            "load_score": 0.0,
        }
    else:
        # ADR-0001: score remains computable without FPS via workload/resource axes.
        avg_fps = 0.0

    fps_clamped = max(0.0, min(float(TARGET_FPS), avg_fps))
    curr_wl = sum(_derive_camera_workload(feature_stats or {}, fps_stats, _starved).values()) if isinstance(feature_stats, dict) else 0.0
    eff_wl = workload_ema if (workload_ema is not None and math.isfinite(workload_ema)) else curr_wl
    eff_fps = fps_ema if (fps_ema is not None and math.isfinite(fps_ema)) else avg_fps

    if mode == "service":
        svc_cfg = ls_cfg.get("service", {})
        if not isinstance(svc_cfg, dict):
            svc_cfg = {}
        c_target   = _finite_positive(svc_cfg.get("target")) or 0.95
        c_floor    = _finite_positive(svc_cfg.get("floor")) or 0.50
        fps_emerg  = emergency_thresholds[2]

        c = service_ema if (service_ema is not None and math.isfinite(service_ema)) else 1.0

        if c >= c_target:
            service_score = 0.0
        elif c <= c_floor:
            service_score = 100.0
        else:
            denom = max(0.001, c_target - c_floor)
            service_score = (c_target - c) / denom * 100.0

        if isinstance(wp_cfg, dict) and wp_cfg.get("enabled", True):
            workload_pressure = _calc_workload_pressure(
                wp_cfg, eff_wl, eff_fps, hw_fuse_score_floor,
                metrics=metrics, n_active=len(active_fps_vals),
                gpu_fps_dwell_armed=gpu_fps_dwell_armed,
                emergency=emergency_thresholds,
            )
        else:
            workload_pressure = 0.0

        fps_floor = 80.0 if (fps_clamped < fps_emerg) else 0.0
        hw_saturated = (
            isinstance(metrics, dict) and (
                float(metrics.get("cpu_percent", 0.0)) >= hw_fuse_threshold or
                float(metrics.get("ram_percent", 0.0)) >= hw_fuse_threshold
            )
        )
        hw_floor = hw_fuse_score_floor if (hw_saturated and fps_clamped < float(TARGET_FPS) - fps_clamp_margin) else 0.0

        # Primary load_score = full asymptotic kernel (demand+resource+service).
        if isinstance(wp_cfg, dict) and wp_cfg.get("enabled", True):
            load_score = _calc_workload_pressure(
                wp_cfg, eff_wl, eff_fps, hw_fuse_score_floor,
                metrics=metrics, service_ema=service_ema,
                n_active=len(active_fps_vals),
                gpu_fps_dwell_armed=gpu_fps_dwell_armed,
                emergency=emergency_thresholds,
            )
        else:
            load_score = 0.0
        load_score = round(min(99.9, max(0.0, load_score)), 1)

        if load_score >= 72.0:
            qos_state = "overloaded"
        elif load_score >= 55.0:
            qos_state = "degraded"
        elif load_score >= 30.0:
            qos_state = "moderate"
        else:
            qos_state = "healthy"

        return {
            "mode": "service",
            "service_c_ema": round(c, 4),
            "service_score": round(service_score, 1),
            "workload_pressure": round(workload_pressure, 1),
            "fps_score": round(fps_floor, 1),
            "hw_floor": round(hw_floor, 1),
            "composite_score": load_score,
            "load_score": load_score,
            "qos_state": qos_state,
            "workload_ema": round(eff_wl, 2),
            "fps_ema": round(eff_fps, 1),
            "raw_workload": round(curr_wl, 2),
            "raw_fps": round(avg_fps, 1),
            "service_delta_fin": int(service_delta_fin),
            "service_delta_miss": int(service_delta_miss),
            "service_pending_tracks": int(service_pending_tracks),
            "service_idle_s": float(service_idle_s),
            "service_cold_start": bool(service_cold_start),
        }

    # Determine qos_state based on demand/resource/service load
    if isinstance(wp_cfg, dict) and wp_cfg.get("enabled") is True:
        # Demand/resource/service pressure; full model (service axis via service_ema).
        # No FPS-confirmation gating — FPS is emergency fuse only (ADR-0001).
        base_score = _calc_workload_pressure(
            wp_cfg, eff_wl, eff_fps, hw_fuse_score_floor,
            metrics=metrics, service_ema=service_ema,
            n_active=len(active_fps_vals),
            gpu_fps_dwell_armed=gpu_fps_dwell_armed,
            emergency=emergency_thresholds,
        )
        load_score = base_score

        hw_saturated = (
            isinstance(metrics, dict) and (
                float(metrics.get("cpu_percent", 0.0)) >= hw_fuse_threshold or
                float(metrics.get("ram_percent", 0.0)) >= hw_fuse_threshold
            )
        )
        if hw_saturated and fps_clamped < float(TARGET_FPS) - fps_clamp_margin:
            load_score = max(load_score, min(99.9, hw_fuse_score_floor))

        load_score = min(99.9, max(0.0, load_score))

        if load_score >= 72.0:
            qos_state = "overloaded"
        elif load_score >= 55.0:
            qos_state = "degraded"
        elif load_score >= 30.0:
            qos_state = "moderate"
        else:
            qos_state = "healthy"

        fps_loss = max(0.0, (float(TARGET_FPS) - eff_fps) / float(TARGET_FPS))
        wl_ratio = max(0.0, eff_wl / max(1.0, _finite_positive(wp_cfg.get("w_high")) or 20.0))

        return {
            "fps_score": round(100.0 * fps_loss, 1),
            "workload_bonus": round(100.0 * wl_ratio, 1),
            "thermal_bonus": 0.0,
            "recv_bonus": 0.0,
            "trend_bonus": 0.0,
            "composite_score": round(base_score, 1),
            "load_score": round(load_score, 1),
            "workload_ema": round(eff_wl, 2),
            "fps_ema": round(eff_fps, 2),
            "raw_workload": round(curr_wl, 2),
            "raw_fps": round(avg_fps, 2),
            "qos_state": qos_state,
        }

    if fps_clamped >= float(TARGET_FPS):
        fps_score = 0.0
    elif fps_clamped >= 22.0:
        fps_score = 57.0 * (float(TARGET_FPS) - fps_clamped) / (float(TARGET_FPS) - 22.0)
    elif fps_clamped >= 19.0:
        fps_score = 57.0 + (65.0 - 57.0) * (22.0 - fps_clamped) / (22.0 - 19.0)
    elif fps_clamped >= 17.0:
        fps_score = 65.0 + (75.0 - 65.0) * (19.0 - fps_clamped) / (19.0 - 17.0)
    else:
        fps_score = 75.0 + (100.0 - 75.0) * (17.0 - fps_clamped) / (17.0 - 0.0)

    workload_bonus = 0.0
    wl_cfg = ls_cfg.get("workload", {})
    if (isinstance(wl_cfg, dict) and wl_cfg.get("enabled") is True
            and isinstance(feature_stats, dict)):
        wl_cap = _finite_positive(wl_cfg.get("capacity"))
        wl_max = _finite_positive(wl_cfg.get("max_bonus")) or 15.0
        if wl_cap is not None:
            wl_total = sum(
                _derive_camera_workload(feature_stats, fps_stats, _starved).values()
            )
            workload_bonus = min(wl_max, max(0.0, wl_max * (wl_total / wl_cap)))

    thermal_bonus = 0.0
    th_cfg = ls_cfg.get("thermal", {})
    if isinstance(th_cfg, dict) and th_cfg.get("enabled") is True and isinstance(metrics, dict):
        temp_val = _finite_nonneg(metrics.get("gpu_temp_c"))
        onset    = _finite_nonneg(th_cfg.get("onset_c"))
        critical = _finite_nonneg(th_cfg.get("critical_c"))
        th_max   = _finite_positive(th_cfg.get("max_bonus")) or 5.0
        if temp_val is not None and onset is not None and critical is not None and onset < critical:
            if temp_val >= critical:
                thermal_bonus = th_max
            elif temp_val > onset:
                thermal_bonus = th_max * (temp_val - onset) / (critical - onset)

    trend_bonus = 0.0
    trend_cfg = ls_cfg.get("trend", {})
    now = time.monotonic()
    if isinstance(trend_cfg, dict) and trend_cfg.get("enabled") is True:
        max_decline = _finite_positive(trend_cfg.get("max_decline_fps_per_s"))
        tr_max      = _finite_positive(trend_cfg.get("max_bonus")) or 5.0
        if max_decline is not None and _FPS_HISTORY:
            t_past, fps_past = _FPS_HISTORY[0]
            for t_h, f_h in _FPS_HISTORY:
                if now - t_h >= 0.5:
                    t_past, fps_past = t_h, f_h
                    break
            dt = now - t_past
            if dt >= 0.5:
                slope = (avg_fps - fps_past) / dt
                decline = max(0.0, -slope)
                trend_bonus = min(tr_max, max(0.0, tr_max * (decline / max_decline)))

    raw_composite = fps_score + workload_bonus + thermal_bonus + trend_bonus
    composite = min(100.0, max(0.0, raw_composite))

    hw_saturated = (
        isinstance(metrics, dict) and (
            float(metrics.get("cpu_percent", 0.0)) >= hw_fuse_threshold or
            float(metrics.get("ram_percent", 0.0)) >= hw_fuse_threshold
        )
    )
    fps_emergency = fps_clamped < float(TARGET_FPS) - fps_clamp_margin

    if hw_saturated and fps_emergency:
        load_score = max(composite, hw_fuse_score_floor)
    else:
        load_score = composite

    return {
        "fps_score": round(fps_score, 1),
        "workload_bonus": round(workload_bonus, 1),
        "thermal_bonus": round(thermal_bonus, 1),
        "recv_bonus": 0.0,
        "trend_bonus": round(trend_bonus, 1),
        "composite_score": round(composite, 1),
        "load_score": round(load_score, 1),
    }


def _update_load_score_ema(
    prev,
    load_score,
    ls_alpha,
    gpu_pct,
    raw_fps,
    emergency: Optional[Any] = None,
    **kwargs,
):
    """Two-stage load_score EMA: cold-start initializes directly; an FPS/GPU emergency
    bypasses the EMA and snaps to the instantaneous score so peers react immediately.

    Emergency thresholds are resolved via _resolve_emergency_thresholds (gpu_pct, gpu_fps, fps)
    with safe fallbacks (99.0, 15.0, 12.0).
    Malformed/non-numeric raw_fps and gpu_pct are treated as unavailable (no
    emergency, no crash) so a bad telemetry value cannot raise TypeError.
    """
    # Config-safe alpha: strictly within (0,1); malformed → 0.20 default.
    alpha = _unit_interval(ls_alpha, 0.20)

    # Safe numeric coercion: malformed → None (unavailable), never raises.
    gpu = _finite_nonneg(gpu_pct)
    fps = _finite_nonneg(raw_fps)

    if emergency is None and "emergency_cfg" in kwargs:
        emergency = kwargs["emergency_cfg"]
    em_gpu_pct, em_gpu_fps, em_fps = _resolve_emergency_thresholds(emergency)

    gpu_emerg = gpu is not None and gpu >= em_gpu_pct and fps is not None and fps < em_gpu_fps
    fps_emerg = fps is not None and fps < em_fps
    is_emergency = gpu_emerg or fps_emerg

    if prev is None or is_emergency:
        ema = load_score
    else:
        ema = alpha * load_score + (1.0 - alpha) * prev
    return round(min(99.9, max(0.0, ema)), 1)


# ---------------------------------------------------------------------------
# Health Agent Main Loop
# ---------------------------------------------------------------------------

# Staleness ceiling for the jtop reader's latest-slot value: a value older than
# this is treated as unavailable so the health loop never consumes minutes-old hardware
# metrics after the reader's worker has died on a hung manager.
_JTOP_STALE_S = _resolve_jtop_stale_s(_EDGE_CFG)


def _resolve_zenoh_router_stale_s() -> float:
    """Effective Zenoh router stale threshold (s).

    Returns the configured finite positive value; 0 disables the check
    (per settings comment); any non-finite (NaN/inf) or negative value
    falls back to the effective default 15.0 so a bad config never
    silently disables transport-death recovery.
    """
    try:
        v = float(ZENOH_ROUTER_STALE_S)
    except (TypeError, ValueError):
        return 15.0
    if v == 0.0:
        return 0.0
    if math.isfinite(v) and v > 0.0:
        return v
    return 15.0


def _check_zenoh_router_liveness(
    session,
    router_configured: bool,
    stale_s: float,
    absent_since_mono: Optional[float],
    now_mono: float,
) -> Tuple[Optional[float], bool]:
    """Pure staleness gate for Zenoh silent-transport death.

    Half-open TCP transport lets put() succeed locally while the router
    receives nothing. session.info.links() reflects real transport link
    state (synchronous, local, no round-trip) — non-empty on healthy
    peer-mode connect sessions, empties within Zenoh lease expiry on
    silent death. Returns (new_absent_since, should_reconnect). Skipped
    when no router is configured (fail-open: never reconnect).
    """
    if not router_configured or stale_s <= 0 or session is None:
        return absent_since_mono, False
    try:
        has_link = bool(session.info.links())
    except Exception:
        has_link = True  # fail-open: never tear down on a query error
    if has_link:
        return None, False
    if absent_since_mono is None:
        return now_mono, False
    if now_mono - absent_since_mono >= stale_s:
        return None, True
    return absent_since_mono, False


class HealthAgent:
    """
    Collect metrics and publish periodically via Zenoh (peer mode).
    Runs a daemon thread with a persistent jtop session to avoid
    socket/fd exhaustion from opening a new jtop() context each cycle.

    When run as a standalone process, opens its own MonitorClient
    WebSocket to push health payloads to the Central Monitor Server.
    """

    def __init__(self, external_session=None, ownership_provider: Optional[Callable[[], Dict[str, dict]]] = None, held_provider: Optional[Callable[[], List[str]]] = None, boot_id_provider: Optional[Callable[[], int]] = None) -> None:
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._pub = None
        self._external_session = external_session
        self._ownership_provider = ownership_provider
        self._held_provider = held_provider
        # P5 — provider returning THIS node's current monotonic boot_id, stamped
        # into the heartbeat so peers can fence pre-reboot ADD/REMOVE commands.
        self._boot_id_provider = boot_id_provider
        # Cached latest resolved boot_id, kept on the attribute so periodic
        # NODE_ONLINE re-announces can include it (R3).  Updated each heartbeat.
        self._boot_id = 0
        self._ready_event = threading.Event()
        self._jtop_stale_s = _resolve_jtop_stale_s(get_edge_cfg())
        # One-shot warmup_ms set by run_python.py after pipeline PLAYING
        self._warmup_ms: Optional[float] = None
        # Proactive load model — instantiated lazily in _run so that
        # edge_node.yml is read after the process fully starts.
        self._proactive_model = None
        self._cam_configs_cache: Dict[str, dict] = {}
        self._max_streams = 8   # from cameras.yml; fallback count of concurrent streams
        self._last_cfg_reload = 0.0
        # EMA state for workload-primary + FPS-confirmation policy
        self._workload_ema: Optional[float] = None
        self._fps_ema: Optional[float] = None
        # EMA of the reported load_score (smoothed heartbeat signal)
        self._load_score_ema: Optional[float] = None
        # Service completion score (V2) state
        self._service_state: dict = {
            "service_ema": None,
            "prev_fin": None,
            "prev_miss": None,
            "last_busy_ts": 0.0,
            "last_update_ts": 0.0,
            "delta_fin": 0,
            "delta_miss": 0,
            "pending_tracks": 0,
            "idle_s": 0.0,
        }
        # Heartbeat telemetry / watchdog metrics
        self._heartbeat_sent_count = 0
        self._heartbeat_error_count = 0
        self._heartbeat_consecutive_errors = 0
        self._last_heartbeat_sent_time: Optional[float] = None
        self._last_heartbeat_error_time: Optional[float] = None
        self._router_absent_since: Optional[float] = None
        # Periodic NODE_ONLINE re-announcement: monotonic timestamp of the last
        # NODE_ONLINE event published, so a live node re-arms itself on the
        # Server after a transient partition swept it offline.
        self._last_node_online_ts: float = 0.0
        # Bandwidth telemetry state (bytes, timestamp)
        self._last_net_bytes: Optional[Tuple[int, int]] = None
        self._last_net_time: Optional[float] = None
        self._net_lock = threading.Lock()
        # GPU-as-witness dwell fuse state: consecutive qualifying sample count.
        self._gpu_fps_dwell_count: int = 0

    def _reload_cam_configs(self) -> Dict[str, dict]:
        """Read cameras.yml for peer failover metadata in health payloads."""
        try:
            import yaml

            cam_yml = Path(__file__).resolve().parent / "configs" / "cameras.yml"
            with open(cam_yml, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            # max_streams sourced from cameras.yml; malformed values fall back to 8
            try:
                self._max_streams = int(raw.get("max_streams", 8) or 8)
            except (TypeError, ValueError):
                self._max_streams = 8

            result: Dict[str, dict] = {}
            for cam_id, cfg in raw.get("cameras", {}).items():
                if cfg and cfg.get("enabled", True):
                    result[cam_id] = {
                        "camera_id":       cam_id,
                        "source_id":       int(cfg.get("source_id", 0)),
                        "uri":             cfg.get("uri", ""),
                        "name":            cfg.get("name", cam_id),
                        "fps":             float(cfg.get("fps", 25.0)),
                        "speed_limit_kmh": float(cfg.get("speed_limit_kmh", 80.0)),
                        "homography":      cfg.get("homography", {}),
                        "roi_polygon":     cfg.get("roi_polygon", []),
                        "output":          cfg.get("output", {}),
                    }
            self._cam_configs_cache = result
            return result
        except Exception as exc:
            logger.warning("[HealthAgent] Failed to reload camera configs: %s", exc)
            return self._cam_configs_cache

    def run(self) -> None:
        """Run the health agent loop directly (blocking or thread target)."""
        self._running = True
        self._run()

    def start(self) -> None:
        """Start agent in daemon thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="HealthAgent",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[HealthAgent] Started. Node=%s, Interval=%.1fs",
            NODE_ID, HEALTH_INTERVAL,
        )

    def stop(self) -> None:
        self._running = False
        self._close_zenoh()

    def _close_zenoh(self) -> None:
        """Safely close and invalidate HealthAgent-owned Zenoh publisher and session."""
        if self._pub is not None:
            try:
                self._pub.undeclare()
            except Exception as exc:
                logger.debug("[HealthAgent] _close_zenoh undeclare error: %s", exc)
            self._pub = None
        if self._session is not None and self._external_session is None:
            try:
                self._session.close()
            except Exception as exc:
                logger.debug("[HealthAgent] _close_zenoh session.close() error: %s", exc)
            self._session = None
        elif self._session is not None:
            self._session = None

    def _maybe_reannounce_node_online(self, held_cameras) -> None:
        """Periodically re-announce NODE_ONLINE so the Server re-arms this node.

        The Server only re-arms a node swept offline via an explicit
        NODE_ONLINE event (anti-resurrection: plain health frames from an
        offline node are dropped). Re-announcing periodically lets a
        genuinely-alive node recover from a transient partition that caused
        the Server to sweep it offline. Reuses the NODE_ONLINE payload shape
        published by ZenohCommandSubscriber.start().
        """
        if self._pub is None:
            return
        if (time.monotonic() - self._last_node_online_ts) < NODE_ONLINE_REANNOUNCE_INTERVAL:
            return
        try:
            now = time.time()
            self._pub.put(msgpack.packb({
                "schema_version": 1,
                "version": 1,
                "node_id": NODE_ID,
                "event": "NODE_ONLINE",
                # R3: keep boot_id in re-announces so the Server can detect a
                # reboot even when the initial NODE_ONLINE was missed.
                "boot_id": self._boot_id,
                "active_cameras": list(held_cameras),
                "timestamp": now,
                "ts": now,
            }, use_bin_type=True))
            self._last_node_online_ts = time.monotonic()
            logger.debug("[HealthAgent] NODE_ONLINE re-announced (interval=%.1fs)", NODE_ONLINE_REANNOUNCE_INTERVAL)
        except Exception as exc:
            logger.warning("[HealthAgent] NODE_ONLINE re-announce failed: %s", exc)

    def _connect_zenoh(self, external_session=None):
        """Open Zenoh session and declare status publisher, or reuse external_session."""
        import zenoh
        target_session = external_session if external_session is not None else self._external_session
        try:
            if target_session is not None:
                self._external_session = target_session
                session = target_session
                logger.info("[HealthAgent] Using shared Zenoh session.")
            else:
                self._external_session = None
                session = make_session()
                logger.info("[HealthAgent] Zenoh session opened (peer mode).")
            pub = session.declare_publisher(f"peers/status/{NODE_ID}")
            return session, pub
        except Exception as exc:
            logger.error("[HealthAgent] Cannot open Zenoh session: %s", exc)
            return None, None

    def _collect_metrics(self) -> Dict:
        """Read hardware metrics via direct /proc + /sys reads (no daemon, no IPC)."""
        return _read_hw_sysfs()

    def _get_net_bytes(self) -> Tuple[int, int]:
        """Read cumulative (rx_bytes, tx_bytes) across non-loopback interfaces."""
        if psutil is not None:
            try:
                counters = psutil.net_io_counters()
                return int(counters.bytes_recv), int(counters.bytes_sent)
            except Exception:
                pass

        # Fallback to /proc/net/dev (eno1 / eth* / wlan* / all non-lo)
        rx_total, tx_total = 0, 0
        try:
            with open("/proc/net/dev", "r") as f:
                lines = f.readlines()
            for line in lines[2:]:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    iface = parts[0].strip()
                    if iface != "lo":
                        data = parts[1].split()
                        rx_total += int(data[0])
                        tx_total += int(data[8])
        except Exception:
            pass
        return rx_total, tx_total

    def _sample_network_bps(self) -> Tuple[float, float]:
        """Compute delta-per-second (network_bps_rx, network_bps_tx)."""
        with timed_lock(self._net_lock, "_net_lock.sample_network_bps", logger=logger):
            now = time.monotonic()
            curr_rx, curr_tx = self._get_net_bytes()
            if self._last_net_bytes is None or self._last_net_time is None:
                self._last_net_bytes = (curr_rx, curr_tx)
                self._last_net_time = now
                return 0.0, 0.0

            dt = now - self._last_net_time
            if dt <= 0.0:
                return 0.0, 0.0

            prev_rx, prev_tx = self._last_net_bytes
            rx_delta = max(0, curr_rx - prev_rx)
            tx_delta = max(0, curr_tx - prev_tx)

            # bits per second: bytes * 8 / dt
            bps_rx = (rx_delta * 8.0) / dt
            bps_tx = (tx_delta * 8.0) / dt

            self._last_net_bytes = (curr_rx, curr_tx)
            self._last_net_time = now
            return round(bps_rx, 1), round(bps_tx, 1)

    def _run(self) -> None:
        """Main loop — collect and publish periodically."""
        import traceback
        try:
            self._session, self._pub = self._connect_zenoh()
            if not self._session:
                logger.error("[HealthAgent] Zenoh unavailable. Running in log-only mode.")

            self._ready_event.set()
            logger.info("[HealthAgent] Collector loop ready, mono_ts=%.6f", time.monotonic())

            # Instantiate proactive model using the proactive: section of edge_node.yml.
            get_edge_cfg()
            from speedflow_python.load_model import ProactiveModel
            self._proactive_model = ProactiveModel(
                _EDGE_CFG.get("proactive", {}),
                policy=LOAD_POLICY,
                model_type=LOAD_MODEL,
            )
            logger.info("[HealthAgent] LOAD_POLICY=%s LOAD_MODEL=%s", LOAD_POLICY, LOAD_MODEL)
            if self._proactive_model.enabled:
                logger.info("[HealthAgent] Proactive load model ENABLED "
                            "(risk_threshold=%.2f)", self._proactive_model.risk_threshold)
            else:
                logger.info("[HealthAgent] Proactive load model disabled "
                            "(set proactive.enabled: true in edge_node.yml to activate)")

            _zenoh_retry_interval = 30.0  # seconds between Zenoh reconnect attempts
            _last_zenoh_attempt = time.time()
            _log_cycle = 0  # counts health cycles; log LoadScore every HEALTH_LOG_EVERY
            _cfg_reload_interval = 30.0
            self._last_cfg_reload = 0.0

            # ponytail: monotonic deadline sleep so work duration doesn't extend the period.
            _next_deadline = time.monotonic()

            while self._running:
                try:
                    if time.monotonic() - self._last_cfg_reload >= _cfg_reload_interval:
                        _maybe_reload_edge_cfg()
                        if self._proactive_model is not None:
                            self._proactive_model.reload_cfg(
                                get_edge_cfg().get("proactive", {})
                            )
                        self._reload_cam_configs()
                        self._last_cfg_reload = time.monotonic()

                    # Periodically retry Zenoh if the session is not established
                    if self._session is None:
                        if time.time() - _last_zenoh_attempt >= _zenoh_retry_interval:
                            logger.info("[HealthAgent] Retrying Zenoh connection...")
                            self._session, self._pub = self._connect_zenoh()
                            _last_zenoh_attempt = time.time()
                            if self._session:
                                logger.info("[HealthAgent] Zenoh reconnected successfully.")

                    metrics = self._collect_metrics()

                    snapshot_valid, fps_stats, feature_stats, offload_crops, service_stats, input_fps, source_modes, snap_telemetry = \
                        _read_pipeline_snapshot()

                    # ── Pipeline unavailable guard ─────────────────────────
                    # When snapshot is invalid (stale, missing telemetry,
                    # non-advancing seq), do NOT convert garbage into a
                    # healthy TARGET_FPS score.  Report unavailable
                    # (load_score 100 = worst) + empty pipeline section.
                    if snapshot_valid:
                        starved_cams = _detect_source_starved(
                            fps_stats, input_fps, get_edge_cfg(),
                            source_type_map=source_modes,
                        )
                        active_fps_vals_for_ema = [
                            v for k, v in fps_stats.items()
                            if v > 0.0 and k not in starved_cams
                        ]
                        raw_fps = sum(active_fps_vals_for_ema) / len(active_fps_vals_for_ema) if active_fps_vals_for_ema else 0.0
                        raw_workload = sum(
                            _derive_camera_workload(feature_stats or {}, fps_stats, starved_cams).values()
                        ) if isinstance(feature_stats, dict) else 0.0

                        # Update EMA state exactly once per HealthAgent cycle
                        ls_cfg = get_edge_cfg().get("load_score", {})
                        wp_cfg = ls_cfg.get("workload_policy", {}) if isinstance(ls_cfg, dict) else {}
                        alpha_ema = _unit_interval(wp_cfg.get("alpha_ema"), 0.33) if isinstance(wp_cfg, dict) else 0.33

                        if active_fps_vals_for_ema:
                            if self._workload_ema is None:
                                self._workload_ema = raw_workload
                            else:
                                self._workload_ema = alpha_ema * raw_workload + (1.0 - alpha_ema) * self._workload_ema

                            if self._fps_ema is None:
                                self._fps_ema = raw_fps
                            else:
                                self._fps_ema = alpha_ema * raw_fps + (1.0 - alpha_ema) * self._fps_ema
                        else:
                            self._workload_ema = None
                            self._fps_ema = None

                        # Service completion ratio calculation & EMA (delta-based)
                        svc_cfg = ls_cfg.get("service", {}) if isinstance(ls_cfg, dict) else {}
                        s_alpha = _finite_positive(svc_cfg.get("ema_alpha")) or 0.30
                        s_stale = _finite_positive(svc_cfg.get("ema_stale_s")) or 30.0

                        now_mono = time.monotonic()
                        self._service_state = _update_service_ema_state(
                            service_stats=service_stats or {},
                            prev_state=self._service_state,
                            now_mono=now_mono,
                            s_alpha=s_alpha,
                            s_stale=s_stale,
                        )

                        # GPU-as-witness dwell fuse: sustained GPU>=95% AND FPS<24
                        # for N consecutive samples floors the score.  State is
                        # maintained here (once per cycle); the armed flag is passed
                        # into the load-score fuses.  Invalid metrics fail open.
                        dwell_cfg = ls_cfg.get("gpu_fps_dwell", {}) if isinstance(ls_cfg, dict) else {}
                        if not isinstance(dwell_cfg, dict):
                            dwell_cfg = {}
                        dwell_enabled = bool(dwell_cfg.get("enabled", True))
                        gpu_pct_dwell = metrics.get("gpu_percent") if isinstance(metrics, dict) else None
                        eff_fps_dwell = self._fps_ema if self._fps_ema is not None else raw_fps
                        if dwell_enabled:
                            gpu_fps_dwell_armed, self._gpu_fps_dwell_count = _gpu_fps_dwell_update(
                                gpu_pct_dwell,
                                eff_fps_dwell,
                                self._gpu_fps_dwell_count,
                                gpu_threshold=float(dwell_cfg.get("gpu_threshold", 95.0)),
                                fps_threshold=float(dwell_cfg.get("fps_threshold", 24.0)),
                                dwell=int(dwell_cfg.get("dwell_samples", 3)),
                            )
                        else:
                            self._gpu_fps_dwell_count = 0
                            gpu_fps_dwell_armed = False

                        emergency_thresholds = _resolve_emergency_thresholds(
                            ls_cfg.get("emergency") if isinstance(ls_cfg, dict) else None,
                            wp_cfg=wp_cfg,
                        )

                        load_score, omega_preset = _compute_load_score(
                            metrics, fps_stats, source_starved_cameras=starved_cams,
                            feature_stats=feature_stats,
                            workload_ema=self._workload_ema,
                            fps_ema=self._fps_ema,
                            service_ema=self._service_state.get("service_ema"),
                            gpu_fps_dwell_armed=gpu_fps_dwell_armed,
                            emergency=emergency_thresholds,
                        )
                        # Compute breakdown for auditable payload
                        load_score_breakdown = _compute_load_score_breakdown(
                            metrics, fps_stats, source_starved_cameras=starved_cams,
                            feature_stats=feature_stats,
                            workload_ema=self._workload_ema,
                            fps_ema=self._fps_ema,
                            service_ema=self._service_state.get("service_ema"),
                            service_delta_fin=self._service_state.get("delta_fin", 0),
                            service_delta_miss=self._service_state.get("delta_miss", 0),
                            service_pending_tracks=self._service_state.get("pending_tracks", 0),
                            service_idle_s=self._service_state.get("idle_s", 0.0),
                            service_cold_start=self._service_state.get("cold_start", False),
                            gpu_fps_dwell_armed=gpu_fps_dwell_armed,
                            emergency=emergency_thresholds,
                        )
                        offload_crops_received_per_s = float(offload_crops.get("received_per_s", 0.0))
                        offload_queue_full = bool(offload_crops.get("offload_queue_full", False))
                        offload_queue_depth = int(offload_crops.get("offload_queue_depth", 0) or 0)
                        offload_queue_depth_ratio = float(offload_crops.get("offload_queue_depth_ratio", 0.0) or 0.0)

                        # BUG-G fix: EMA the reported load_score (configurable via
                        # load_score.load_score_alpha). On an FPS/GPU emergency the EMA is
                        # overridden to the instantaneous score so peers react immediately.
                        ls_alpha = _unit_interval(ls_cfg.get("load_score_alpha") if isinstance(ls_cfg, dict) else None, 0.20)
                        gpu_pct = metrics.get("gpu_percent") if isinstance(metrics, dict) else None
                        self._load_score_ema = _update_load_score_ema(
                            self._load_score_ema, load_score, ls_alpha, gpu_pct, raw_fps, emergency=emergency_thresholds,
                        )

                        # BUG-I fix: exclude 0-fps cameras from avg_fps,
                        # matching the exclusion applied in _compute_load_score()
                        # so the reported avg_fps is consistent with the load_score value.
                        active_fps_vals = [v for v in fps_stats.values() if v > 0.0]
                        avg_fps = round(sum(active_fps_vals) / len(active_fps_vals), 1) if active_fps_vals else None
                        # Liveness (ownership/failover) = pipeline-attached cameras,
                        # independent of instantaneous FPS. Throughput (FPS>0) is
                        # tracked separately as streaming_cameras for load only.
                        attached_cameras, streaming_cameras, active_cameras = \
                            _derive_camera_liveness(source_modes, fps_stats)
                    else:
                        self._workload_ema = None
                        self._fps_ema = None
                        # Reset the load_score EMA too: an invalid pipeline
                        # snapshot means no fresh load_score was computed, so a
                        # stale EMA must not be published as if it were current.
                        self._load_score_ema = None
                        starved_cams = set()
                        load_score, omega_preset = 0.0, "no_fps"
                        load_score_breakdown = {
                            "fps_score": 0.0,
                            "workload_bonus": 0.0,
                            "thermal_bonus": 0.0,
                            "recv_bonus": 0.0,
                            "trend_bonus": 0.0,
                            "composite_score": 0.0,
                            "load_score": 0.0,
                            "workload_ema": 0.0,
                            "fps_ema": 0.0,
                            "raw_workload": 0.0,
                            "raw_fps": 0.0,
                            "qos_state": None,
                        }
                        offload_crops_received_per_s = 0.0
                        offload_queue_full = False
                        offload_queue_depth = 0
                        offload_queue_depth_ratio = 0.0
                        active_fps_vals = []
                        avg_fps = None
                        active_cameras = []
                        streaming_cameras = []
                        # Zero out telemetry so downstream code (proactive model,
                        # logging) sees empty inputs, not stale data.
                        fps_stats = {}
                        feature_stats = {}
                        offload_crops = {"received_per_s": 0.0}

                    # Consume one-shot warmup_ms written by run_python.py after
                    # pipeline.set_state(PLAYING).  After the first heartbeat
                    # it appears in, reset so it only fires once per cold-start.
                    warmup_ms = self._warmup_ms
                    self._warmup_ms = None

                    # Active, non-source-starved cameras only.  Empty in the
                    # invalid-snapshot branch (feature_stats/fps_stats are {}) —
                    # nothing to derive from, matching the unavailable report.
                    camera_workload = _derive_camera_workload(
                        feature_stats, fps_stats, starved_cams
                    )

                    # Pipeline status: 'running' if snapshot valid and active cameras > 0, else 'waiting'
                    is_waiting = (
                        (not snapshot_valid)
                        or (not active_cameras)
                    )
                    pipeline_status = "waiting" if is_waiting else "running"
                    pipeline_idle = bool(snapshot_valid and not active_cameras)

                    # Retrieve live camera ownership & epochs if provider is configured
                    camera_owners = {}
                    camera_holders = {}
                    camera_epochs = {}
                    if self._ownership_provider is not None:
                        try:
                            records = self._ownership_provider()
                            if isinstance(records, dict):
                                for cam_k, rec in records.items():
                                    if isinstance(rec, dict):
                                        if "owner" in rec:
                                            camera_owners[cam_k] = rec["owner"]
                                        if "holder" in rec:
                                            camera_holders[cam_k] = rec["holder"]
                                        if "epoch" in rec:
                                            camera_epochs[cam_k] = rec["epoch"]
                        except Exception as exc:
                            logger.debug("[HealthAgent] Failed to retrieve live ownership records: %s", exc)

                    # Compute owned and foreign active cameras
                    owned_cam_ids = set(self._cam_configs_cache.keys())
                    owned_active = [c for c in active_cameras if camera_owners.get(c, c if c in owned_cam_ids else "") == NODE_ID or (c in owned_cam_ids and c not in camera_owners)]
                    foreign_active = [c for c in active_cameras if c not in owned_active]

                    # Get held cameras from CameraManager (enabled configs with live pipeline branches)
                    # This includes warming-up and stalled streams that streaming_cameras (FPS>0) would miss.
                    held_cameras = []
                    if self._held_provider is not None:
                        try:
                            held_cameras = self._held_provider() or []
                        except Exception as exc:
                            logger.debug("[HealthAgent] Failed to retrieve held cameras: %s", exc)

                    now_ts = time.time()
                    bps_rx, bps_tx = self._sample_network_bps()
                    # P5 — stamp this node's monotonic boot_id into the heartbeat
                    # so peers can fence pre-reboot ADD/REMOVE commands.
                    boot_id = 0
                    if self._boot_id_provider is not None:
                        try:
                            boot_id = int(self._boot_id_provider() or 0)
                        except Exception:
                            boot_id = 0
                    # R3: cache so periodic NODE_ONLINE re-announces carry it.
                    self._boot_id = boot_id

                    payload = {
                        "type":          "health",
                        "node_id":       NODE_ID,
                        "advertise_ip":  ADVERTISE_IP,
                        "boot_id":       boot_id,
                        "timestamp":     now_ts,
                        "ts":            now_ts,
                        # Dedup keys for TelemetryStore (Server): session_id +
                        # sequence identify the pipeline telemetry window that
                        # this heartbeat mirrors, so the Server skips rows it
                        # already persisted (avoids duplicate rows across the
                        # raw payload AND the derived aggregates). Empty when
                        # the pipeline snapshot is invalid.
                        "session_id":    snap_telemetry.get("session_id") if snapshot_valid else None,
                        "sequence":      snap_telemetry.get("sequence") if snapshot_valid else None,
                        # Static camera metadata: configured nominal FPS per
                        # camera (from cameras.yml via _cam_configs_cache).
                        "configured_fps_per_camera": {
                            cid: cfg.get("fps")
                            for cid, cfg in self._cam_configs_cache.items()
                            if isinstance(cfg, dict)
                        },
                        "network_bps_rx": bps_rx,
                        "network_bps_tx": bps_tx,
                        "load_score":    self._load_score_ema if self._load_score_ema is not None else load_score,
                        "omega_preset":  omega_preset,
                        "load_score_breakdown": load_score_breakdown,
                        "workload_ema":  load_score_breakdown.get("workload_ema"),
                        "fps_ema":       load_score_breakdown.get("fps_ema"),
                        "qos_state":     load_score_breakdown.get("qos_state"),
                        "gpu_percent":   metrics["gpu_percent"],
                        "cpu_percent":   metrics["cpu_percent"],
                        "ram_percent":   metrics["ram_percent"],
                        "gpu_temp_c":    metrics["gpu_temp_c"],
                        "power_mw":      metrics["power_mw"],
                        "source":        metrics.get("source", "jtop"),
                        "camera_owners": camera_owners,
                        "camera_holders": camera_holders,
                        "camera_epochs": camera_epochs,
                        "held_cameras": held_cameras,
                        "pipeline": {
                            # pipeline_available distinguishes "pipeline not yet
                            # started / stale snapshot" (False, load_score=100)
                            # from real overload or idle pipeline (True).
                            "pipeline_available": bool(snapshot_valid),
                            "pipeline_idle":      pipeline_idle,
                            "status":             pipeline_status,
                            # output_fps_per_camera = frames the probe actually
                            # processed this window (pipeline throughput).
                            # fps_per_camera is kept for backward compatibility.
                            "fps_per_camera":        fps_stats,
                            "output_fps_per_camera": fps_stats,
                            # input_fps_per_camera currently mirrors the bounded
                            # output/OSD sink tick rate (the same value as
                            # output_fps_per_camera).  It is NOT a PTS-derived
                            # upstream source measurement — no upstream PTS
                            # measurement exists in this build.  Used for
                            # source-starved detection.
                            "input_fps_per_camera":  input_fps if snapshot_valid else {},
                            "avg_fps":            avg_fps,
                            "active_cameras":     active_cameras,
                            "streaming_cameras":  streaming_cameras,
                            "held_cameras":       held_cameras,
                            "source_starved_cameras": sorted(starved_cams),
                            "camera_workload":    camera_workload,
                            "camera_features":    feature_stats if snapshot_valid else {},
                            "camera_configs":     self._cam_configs_cache,
                            "camera_owners":      camera_owners,
                            "camera_epochs":      camera_epochs,
                            "max_streams":        int(self._max_streams or 8),
                            "offload_crops_received_per_s": float(offload_crops_received_per_s or 0.0),
                            "offload_queue_full": bool(offload_queue_full),
                            "offload_queue_depth": int(offload_queue_depth),
                            "offload_queue_depth_ratio": float(offload_queue_depth_ratio),
                        },
                    }

                    if warmup_ms is not None:
                        payload["warmup_ms"] = warmup_ms

                    # ── Proactive model ────────────────────────────────────────
                    # Skip when snapshot is invalid — no features to compute on.
                    if snapshot_valid and self._proactive_model is not None:
                        _active_ids = {k for k, v in fps_stats.items() if v > 0.0}
                        proactive_result = self._proactive_model.compute(
                            metrics,
                            {k: v for k, v in feature_stats.items() if k in _active_ids},
                            offload_crops_received_per_s=offload_crops_received_per_s,
                            fps_stats={k: v for k, v in fps_stats.items() if k in _active_ids},
                        )
                        payload.update(proactive_result)

                    if self._pub:
                        t_pub_start = time.monotonic()
                        seq = self._heartbeat_sent_count + self._heartbeat_error_count + 1
                        logger.debug("[HealthAgent] Heartbeat publish attempt: seq=%d, mono_ts=%.6f", seq, t_pub_start)
                        try:
                            self._pub.put(msgpack.packb(payload, use_bin_type=True))
                            t_pub_end = time.monotonic()
                            self._heartbeat_sent_count += 1
                            self._heartbeat_consecutive_errors = 0
                            self._last_heartbeat_sent_time = time.time()
                            logger.debug("[HealthAgent] Heartbeat publish success: seq=%d, dur_ms=%.2f, mono_ts=%.6f", seq, (t_pub_end - t_pub_start) * 1000.0, t_pub_end)
                        except Exception as pub_exc:
                            t_pub_err = time.monotonic()
                            self._heartbeat_error_count += 1
                            self._heartbeat_consecutive_errors += 1
                            self._last_heartbeat_error_time = time.time()
                            logger.warning(
                                "[HealthAgent] Heartbeat publish failure: seq=%d, consecutive=%d, total=%d, err=%s, dur_ms=%.2f, mono_ts=%.6f",
                                seq, self._heartbeat_consecutive_errors, self._heartbeat_error_count, pub_exc, (t_pub_err - t_pub_start) * 1000.0, t_pub_err,
                            )
                            self._close_zenoh()
                            _last_zenoh_attempt = 0  # force immediate reconnect on next loop iteration

                    # Zenoh silent-transport-death gate: half-open TCP transport lets put()
                    # succeed locally while the router receives nothing.
                    # session.info.links() reflects real transport link state (synchronous,
                    # local, no round-trip). Skipped when no router is configured (fail-open).
                    # Uses the effective (validated non-finite-safe) stale threshold so a
                    # malformed ZENOH_ROUTER_STALE_S never silently disables recovery.
                    _router_stale_s = _resolve_zenoh_router_stale_s()
                    if ZENOH_ROUTER and _router_stale_s > 0.0 and self._session is not None:
                        _now_stale = time.monotonic()
                        self._router_absent_since, _should_reconnect = _check_zenoh_router_liveness(
                            self._session,
                            router_configured=True,
                            stale_s=_router_stale_s,
                            absent_since_mono=self._router_absent_since,
                            now_mono=_now_stale,
                        )
                        if _should_reconnect:
                            logger.warning(
                                "[HealthAgent] Zenoh transport link absent for %.1fs"
                                " — inferring silent transport death; reconnecting.",
                                ZENOH_ROUTER_STALE_S,
                            )
                            self._close_zenoh()
                            _last_zenoh_attempt = 0
                            self._router_absent_since = None

                    # Periodic NODE_ONLINE re-announcement. The Server only
                    # re-arms a node swept offline via an explicit NODE_ONLINE
                    # event (anti-resurrection: plain health frames from an
                    # offline node are dropped). Re-announcing periodically lets
                    # a genuinely-alive node recover from a transient partition
                    # that caused the Server to sweep it offline.
                    self._maybe_reannounce_node_online(held_cameras)

                    _log_cycle += 1
                    if _log_cycle % HEALTH_LOG_EVERY == 1:
                        _risk_str = (
                            f" | U={payload.get('risk_index', 0.0):.3f}"
                            f" L={payload.get('l_proactive', 0.0):.3f}"
                            f" H={payload.get('h_reactive', 0.0):.3f}"
                            if payload.get("proactive_enabled") else ""
                        )
                        _bd = load_score_breakdown if isinstance(load_score_breakdown, dict) else {}
                        logger.info(
                            "LoadScore=%.1f [%s] | qos=%s svc_score=%.1f (c=%.3f, +%d/-%d, pend=%d) "
                            "wl_press=%.1f fps_floor=%.1f hw_floor=%.1f | cams=%d (owned=%d, foreign=%d) q_full=%s | "
                            "GPU=%.1f%% CPU=%.1f%% RAM=%.1f%% Temp=%.1f°C Power=%.0fmW | FPS=%s%s",
                            load_score, omega_preset,
                            _bd.get("qos_state", "unknown"),
                            _bd.get("service_score", 0.0),
                            _bd.get("service_c_ema", 1.0) if _bd.get("service_c_ema") is not None else 1.0,
                            _bd.get("service_delta_fin", 0),
                            _bd.get("service_delta_miss", 0),
                            _bd.get("service_pending_tracks", 0),
                            _bd.get("workload_pressure", 0.0),
                            _bd.get("fps_score", 0.0),
                            _bd.get("hw_floor", 0.0),
                            len(active_cameras),
                            len(owned_active),
                            len(foreign_active),
                            offload_queue_full,
                            metrics["gpu_percent"],
                            metrics["cpu_percent"],
                            metrics["ram_percent"],
                            metrics["gpu_temp_c"],
                            metrics["power_mw"],
                            fps_stats,
                            _risk_str,
                        )

                except Exception as exc:
                    logger.error("[HealthAgent] Error in collect loop: %s", exc)

                # ponytail: deadline sleep — work duration does not extend the period.
                _next_deadline += HEALTH_INTERVAL
                _remaining = _next_deadline - time.monotonic()
                if _remaining > 0:
                    time.sleep(_remaining)
                else:
                    # Overran the interval — reset to next cycle to avoid burst.
                    _next_deadline = time.monotonic()

        except Exception:
            logger.critical("[HealthAgent] Fatal unhandled exception in _run:\n%s", traceback.format_exc())
            raise
        finally:
            logger.info("[HealthAgent] Cleaning up resources before exit...")
            self._close_zenoh()
            logger.info("[HealthAgent] Stopped.")


# ---------------------------------------------------------------------------
# Entry point (run standalone)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import zenoh
    except ImportError:
        logger.error("zenoh not installed. Run: pip install zenoh")
        sys.exit(1)

    # PID file lock — prevent two health_agent instances for the same node.
    # Two instances connecting with the same node_id cause the server to close
    # the older connection (code 1000) every time the newer one sends a message,
    # creating an endless close/reconnect loop.
    import os as _os
    _PID_FILE = Path(__file__).resolve().parent / "health_agent.pid"
    _my_pid   = _os.getpid()
    if _PID_FILE.exists():
        try:
            _old_pid = int(_PID_FILE.read_text().strip())
            if _old_pid != _my_pid:
                try:
                    _os.kill(_old_pid, 0)   # signal 0 = existence check only
                    logger.error(
                        "health_agent.py is already running (PID %d). "
                        "Kill it first: kill %d", _old_pid, _old_pid
                    )
                    sys.exit(1)
                except OSError:
                    pass   # old process is dead — stale PID file, safe to continue
        except ValueError:
            pass
    _PID_FILE.write_text(str(_my_pid))

    import atexit as _atexit
    _atexit.register(lambda: _PID_FILE.unlink(missing_ok=True))

    agent = HealthAgent()
    agent.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping HealthAgent...")
        agent.stop()
