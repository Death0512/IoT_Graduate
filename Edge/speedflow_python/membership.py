"""
Edge/speedflow_python/peer_orchestrator.py

Peer Orchestrator — P2P brain, replaces MasterOrchestrator.

Each Edge Node runs an independent PeerOrchestrator instance.
Instances communicate via Zenoh key expressions:
  - peers/status/<node_id>  ← heartbeat from all peers
  - peers/vote/request      ← RFO (Request for Offload) from overloaded peer
  - peers/vote/proposal     ← bid from capable peer
  - peers/vote/decision     ← election result
  - peers/vote/ack/{cam}    ← confirmation that stream is PLAYING

Migration uses Make-before-Break strategy:
  1. Requester opens vote window → collects proposals (3s)
  2. Select winner = proposal with lowest F(x) (ε-constraint)
  3. Publish decision → winner auto-ADD camera to pipeline
  4. Winner publishes peers/vote/ack/{cam} when stream PLAYING
  5. Requester receives ack → REMOVE camera from its pipeline
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import os
import random
import socket
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .log_utils import timed_lock

import msgpack

from .zenoh_session import make_session

# Settings loaded from Edge/.env
from .settings import ROOT as _ROOT, LOG_LEVEL

def _setup_logging() -> logging.Logger:
    raw_level = LOG_LEVEL
    level = getattr(logging, raw_level, logging.INFO)
    root_level = logging.INFO if raw_level == "DEBUG" else level

    if not logging.root.handlers:
        logging.basicConfig(
            level=root_level,
            format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.root.setLevel(root_level)

    if raw_level == "DEBUG":
        for name in (
            "peer_orchestrator",
            "health_agent",
            "speedflow_python.probes",
            "speedflow_python.offload_receiver",
        ):
            logging.getLogger(name).setLevel(logging.DEBUG)

    return logging.getLogger("peer_orchestrator")

logger = _setup_logging()

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PeerState:
    """Current state of a Peer Node (replaces NodeState)."""
    node_id: str
    load_score: float = 0.0
    gpu_percent: float = 0.0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    gpu_temp_c: Optional[float] = 0.0
    avg_fps: Optional[float] = None
    fps_per_camera: Dict[str, float] = field(default_factory=dict)
    active_cameras: List[str] = field(default_factory=list)
    streaming_cameras: List[str] = field(default_factory=list)
    held_cameras: List[str] = field(default_factory=list)
    camera_configs: Dict[str, dict] = field(default_factory=dict)
    camera_owners: Dict[str, str] = field(default_factory=dict)
    camera_holders: Dict[str, str] = field(default_factory=dict)
    camera_epochs: Dict[str, int] = field(default_factory=dict)
    max_streams: int = 8
    last_seen: float = field(default_factory=time.time)
    overload_since: Optional[float] = None
    _overload_peak: float = 0.0
    penalty_until: float = 0.0
    # P5 — monotonic boot counter published by each peer; used by receiver
    # fencing to reject pre-reboot ADD/REMOVE commands after a peer restart.
    boot_id: int = 0
    # Proactive model output -- populated when proactive.enabled is True.
    # Defaults to 0.0 (no risk) so legacy comparisons (load_score only) are unaffected.
    risk_index: float = 0.0
    # Per-camera workload (n_track + n_plate) from health payload.
    # L2 full-stream migration picks MIN workload; L1 plate-crop offload
    # (source offload_level==1) picks MAX workload (selected in
    # _pick_camera_for_lpr_offload). The L3 vehicle-crop tier was removed by the
    # P3 redesign.
    # Empty dict is the safe default when payload is missing/malformed.
    camera_workload: Dict[str, float] = field(default_factory=dict)
    # Camera IDs reported as source-starved by the health agent.
    # These are excluded from offload candidate selection (fail safe).
    source_starved_cameras: List[str] = field(default_factory=list)
    # Rate of offload crops received per second (for L1 plate-crop peer evaluation)
    offload_crops_received_per_s: float = 0.0
    offload_queue_full: bool = False
    consecutive_queue_not_full_count: int = 0
    offload_queue_depth: int = 0
    load_score_breakdown: Dict[str, float] = field(default_factory=dict)
    status: Optional[str] = None
    pipeline_idle: bool = False
    workload_ema: Optional[float] = None
    fps_ema: Optional[float] = None
    qos_state: Optional[str] = None


def _parse_camera_workload(raw) -> Dict[str, float]:
    """Parse camera_workload from a health payload; malformed → {}.

    Accepts only finite, non-negative numeric values keyed by str camera_id.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        if (isinstance(k, str)
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                and v >= 0):
            out[k] = float(v)
    return out


def _parse_starved_cameras(raw) -> List[str]:
    """Parse source_starved_cameras from a health payload; malformed → []."""
    if not isinstance(raw, (list, tuple)):
        return []
    return [c for c in raw if isinstance(c, str)]


def _pick_fps_dict(pipeline: dict) -> dict:
    """Return the canonical per-camera FPS dict from a pipeline section.

    Phase 1 added ``output_fps_per_camera`` as the unambiguous name for
    pipeline throughput FPS.  ``fps_per_camera`` is kept for backward
    compatibility.  Prefer the new name; fall back to the legacy key so
    peers on older firmware still work.
    """
    if not isinstance(pipeline, dict):
        return {}
    out = pipeline.get("output_fps_per_camera")
    if isinstance(out, dict):
        return out
    fallback = pipeline.get("fps_per_camera", {})
    return fallback if isinstance(fallback, dict) else {}


def _has_valid_positive_fps(fps_per_camera) -> bool:
    """Return True if fps_per_camera has at least one finite, positive (>0) value.

    ponytail: used for warmup / destination-readiness gating only — NOT for the
    overload decision itself.  Under ADR-0001 FPS is a safety witness, not a
    primary trigger: overload is driven by workload/resource/qos, and the
    decision fails closed only when *resource* telemetry is invalid.  Dashboard
    keeps the raw load_score; only the decision path is gated, preserving what
    the operator sees.
    """
    if not fps_per_camera or not isinstance(fps_per_camera, dict):
        return False
    for v in fps_per_camera.values():
        if (isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                and v > 0.0):
            return True
    return False


def _has_valid_or_unreported_fps(fps_per_camera) -> bool:
    """Return True if fps_per_camera has valid positive fps OR is not reported (dict empty/None).

    Used for rebalance backward-compatibility where older/synthetic peers might not populate fps_per_camera.
    """
    if fps_per_camera is None or (isinstance(fps_per_camera, dict) and len(fps_per_camera) == 0):
        return True
    return _has_valid_positive_fps(fps_per_camera)


def is_waiting_state(fps_per_camera, active_cameras: Optional[List[str]] = None, status: Optional[str] = None) -> bool:
    """Predicate for WAITING/RECOVERY state.

    Returns True if:
      - status string is explicitly 'waiting', 'recovering', or 'recovery', OR
      - active_cameras is empty or None (len == 0), OR
      - fps_per_camera is empty ({}, None) or has no valid positive FPS.

    In this state, the node is waiting/recovering and must NEVER be classified
    as overloaded or offline, and must not trigger failover/rescue or be picked as migration destination.
    """
    if status and str(status).strip().lower() in ("waiting", "recovering", "recovery"):
        return True
    if active_cameras is not None and len(active_cameras) == 0:
        return True
    if not _has_valid_positive_fps(fps_per_camera):
        return True
    return False


def _saturate(v) -> float:
    """Clamp a 0–100 hardware percentage into a [0,1] saturation ratio.

    Non-finite / non-numeric / bool values degrade to 0.0 (no pressure).
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, float(v) / 100.0))


def _resource_telemetry_valid(state) -> bool:
    """True when GPU/CPU/RAM saturation evidence is present and finite.

    ADR-0001: the migration decision fails closed ONLY when this is False
    (resource telemetry invalid). FPS absence is NOT a fail-closed condition.
    """
    for v in (state.gpu_percent, state.cpu_percent, state.ram_percent):
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return False
    return True


def _waiting_by_status_or_idle(status, active_cameras) -> bool:
    """Overload-decision waiting check: status waiting/recovering OR no active cameras.

    Deliberately omits the FPS clause used by is_waiting_state — under
    ADR-0001 FPS is a safety witness, not a gating signal, so its absence
    must NOT suppress overload detection.  (Startup/recovery suppression is
    handled by _check_self_overload's warmup + status/active guards.)
    """
    if status and str(status).strip().lower() in ("waiting", "recovering", "recovery"):
        return True
    if active_cameras is not None and len(active_cameras) == 0:
        return True
    return False


def _dwell_s(cfg: dict, key: str, default: float) -> float:
    """Read a dwell duration from config; malformed/missing → default (fail-safe)."""
    raw = cfg.get(key, default)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return default
    if not math.isfinite(float(raw)) or float(raw) < 0.0:
        return default
    return float(raw)


def _thermal_admission_ok(gpu_temp_c, therm_cfg) -> bool:
    """
    Shared thermal-admission gate for BOTH roles.

    Used by the sender side (_pick_best_peer, checking a peer before
    offloading to it) and the receiver side (_evaluate_and_bid, checking
    this node's own temperature before bidding on an RFO), so the rules
    cannot drift apart. Config is the `p2p.thermal` section of
    Edge/configs/edge_node.yml.

    Returns True when the (peer or self) node may accept offload work.
    An absent thermal section leaves behaviour unchanged (always accept).
    """
    if not therm_cfg:
        return True
    if not therm_cfg.get("admission_enabled", True):
        return True
    max_t = float(therm_cfg.get("max_gpu_temp_c", 85.0))
    if gpu_temp_c is None or not isinstance(gpu_temp_c, (int, float)):
        # Unknown / missing measurement. Default (conservative) policy
        # rejects so we never send work to a node whose thermal state is
        # unknown; permissive accepts on the assumption it is otherwise
        # healthy. Both behaviours are explicit config.
        policy = therm_cfg.get("invalid_policy", "conservative")
        if policy == "permissive":
            return True  # accept despite missing data
        # reject_invalid=True → reject (return False)
        # reject_invalid=False → accept (return True)
        return not bool(therm_cfg.get("reject_invalid", True))
    if gpu_temp_c <= 0:
        # Zero or negative is almost certainly a sensor failure
        policy = therm_cfg.get("invalid_policy", "conservative")
        return policy != "conservative"
    if gpu_temp_c > max_t:
        return False
    return True


# ---------------------------------------------------------------------------
# Migration Log
# ---------------------------------------------------------------------------

class MigrationLogger:
    """Log each migration to a CSV file — copied from master_orchestrator.py."""

    HEADER = [
        "timestamp_iso", "from_node", "to_node", "camera_id",
        "trigger_reason", "trigger_load", "trigger_fps",
        "migration_time_ms", "result", "blind_spot_ms",
    ]

    def __init__(self, log_file: Path) -> None:
        self._path = log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not log_file.exists():
            with open(log_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)

    def log(
        self,
        from_node: str,
        to_node: str,
        camera_id: str,
        trigger_reason: str,
        trigger_load: float,
        trigger_fps: Optional[float],
        migration_time_ms: float,
        result: str,
        blind_spot_ms: Optional[float] = None,
    ) -> None:
        row = [
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            from_node, to_node, camera_id,
            trigger_reason,
            round(trigger_load, 1),
            round(trigger_fps, 1) if trigger_fps is not None else "",
            round(migration_time_ms, 0),
            result,
            round(blind_spot_ms, 0) if blind_spot_ms is not None else "",
        ]
        try:
            # File size rotation guard: rotate if CSV exceeds 10MB to avoid filling Jetson eMMC
            if self._path.exists() and self._path.stat().st_size > 10 * 1024 * 1024:
                backup = self._path.with_suffix(".csv.old")
                try:
                    self._path.replace(backup)
                except Exception:
                    pass
                with open(self._path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(self.HEADER)

            with open(self._path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as exc:
            logger.warning("MigrationLogger write error: %s", exc)


# ---------------------------------------------------------------------------
# Peer Orchestrator
# ---------------------------------------------------------------------------



class MembershipMixin:
    def _on_peer_status(self, payload: dict) -> None:
        """Update PeerState from heartbeat."""
        node_id = payload.get("node_id", "")
        if not node_id:
            return

        # Update state of this node itself
        if node_id == self._node_id:
            with timed_lock(self._self_lock, "_self_lock.update_self_state", logger=logger):
                old_last_seen = self._self_state.last_seen
                new_last_seen = time.time()
                mono_now = time.monotonic()
                if old_last_seen > 0.0:
                    logger.debug(
                        "[PeerOrch] Self heartbeat received: node='%s', last_seen_gap_s=%.3f, mono_ts=%.6f",
                        node_id, new_last_seen - old_last_seen, mono_now,
                    )
                else:
                    logger.debug(
                        "[PeerOrch] First self heartbeat received: node='%s', mono_ts=%.6f",
                        node_id, mono_now,
                    )
                self._self_state.load_score  = payload.get("load_score",  0.0)
                self._self_state.load_score_breakdown = payload.get("load_score_breakdown", {})
                self._self_state.workload_ema = payload.get("workload_ema")
                self._self_state.fps_ema = payload.get("fps_ema")
                self._self_state.qos_state = payload.get("qos_state")
                self._self_state.gpu_percent = payload.get("gpu_percent", 0.0)
                self._self_state.cpu_percent = payload.get("cpu_percent", 0.0)
                self._self_state.ram_percent = payload.get("ram_percent", 0.0)
                self._self_state.gpu_temp_c  = payload.get("gpu_temp_c",  0.0)
                self._self_state.risk_index  = payload.get("risk_index",  0.0)
                pipeline = payload.get("pipeline", {}) or {}
                self._self_state.status = pipeline.get("status")
                self._self_state.pipeline_idle = bool(pipeline.get("pipeline_idle", False))
                self._self_state.avg_fps = pipeline.get("avg_fps")
                # Prefer output_fps_per_camera (Phase 1 unambiguous key); fall back
                # to fps_per_camera for backward compatibility with older firmware.
                self._self_state.fps_per_camera = _pick_fps_dict(pipeline)
                if isinstance(pipeline.get("active_cameras"), (list, tuple, set)):
                    self._self_state.active_cameras = [str(c) for c in pipeline["active_cameras"] if isinstance(c, (str, int))]
                if isinstance(pipeline.get("streaming_cameras"), (list, tuple, set)):
                    self._self_state.streaming_cameras = [str(c) for c in pipeline["streaming_cameras"] if isinstance(c, (str, int))]
                else:
                    self._self_state.streaming_cameras = list(self._self_state.active_cameras)
                if isinstance(pipeline.get("held_cameras"), (list, tuple, set)):
                    self._self_state.held_cameras = [str(c) for c in pipeline["held_cameras"] if isinstance(c, (str, int))]
                else:
                    self._self_state.held_cameras = list(self._self_state.active_cameras)
                raw_configs = pipeline.get("camera_configs")
                if isinstance(raw_configs, dict) and raw_configs:
                    self._self_state.camera_configs = raw_configs
                elif isinstance(raw_configs, dict) and len(raw_configs) == 0:
                    if (
                        pipeline.get("status") is not None
                        and str(pipeline.get("status")).strip().lower() not in ("waiting", "recovering", "recovery", "starting")
                        and "active_cameras" in pipeline
                        and len(self._self_state.active_cameras) == 0
                    ):
                        self._self_state.camera_configs = {}
                # Backwards-compatible: missing or malformed mapping → empty.
                self._self_state.camera_workload = _parse_camera_workload(
                    pipeline.get("camera_workload", {})
                )
                self._self_state.source_starved_cameras = _parse_starved_cameras(
                    pipeline.get("source_starved_cameras", [])
                )
                self._self_state.offload_crops_received_per_s = float(
                    pipeline.get("offload_crops_received_per_s", 0.0) or 0.0
                )
                self._self_state.last_seen = time.time()
                try:
                    self._self_state.max_streams = int(pipeline.get("max_streams", 8) or 8)
                except (TypeError, ValueError):
                    self._self_state.max_streams = 8

                # ponytail: startup + per-camera warmup timestamp tracking —
                # same logic as update_self_state so the Zenoh-self path and
                # the direct update_self_state path behave identically.
                fps_valid = _has_valid_positive_fps(self._self_state.fps_per_camera)
                if fps_valid and self._self_first_valid_fps_at is None:
                    self._self_first_valid_fps_at = time.time()
                if fps_valid:
                    now_ts = time.time()
                    for cam_id, fps_val in self._self_state.fps_per_camera.items():
                        if not isinstance(cam_id, str):
                            continue
                        if (isinstance(fps_val, (int, float))
                                and not isinstance(fps_val, bool)
                                and math.isfinite(fps_val)
                                and fps_val > 0.0):
                            self._camera_first_valid_fps_at.setdefault(cam_id, now_ts)

                # Overload onset: overloaded (workload/resource/qos) AND resource
                # telemetry valid AND not waiting/recovery.  ADR-0001: FPS is a
                # safety witness, NOT a gate — its absence does not suppress
                # overload (warmup + status/active guards handle startup/recovery).
                # Dashboard keeps the raw load_score; only the decision path is gated.
                in_waiting = _waiting_by_status_or_idle(
                    pipeline.get("status"),
                    self._self_state.active_cameras,
                )
                # ADR-0001: overload is driven by workload/resource/qos, NOT by
                # FPS.  Fail closed only when resource telemetry is invalid; FPS
                # absence is not a blocker (warmup + status/active guards handle
                # startup/recovery).  Source starvation is a symptom, not a veto.
                overloaded = (
                    self._is_overloaded(
                        self._self_state.load_score,
                        self._self_state.risk_index,
                        self._self_state.qos_state,
                    )
                    and _resource_telemetry_valid(self._self_state)
                    and not in_waiting
                )
                thr = float(self._cfg.get("overload_threshold", 55.0))
                if overloaded:
                    self._below_thr_since = None
                    if self._self_state.overload_since is None:
                        self._self_state.overload_since = time.time()
                        dur = float(self._cfg.get("overload_duration_s", 5.0))
                        logger.info(
                            "[PeerOrch] OVERLOAD ONSET load=%.1f thr=%.1f dur=%.1fs",
                            self._self_state.load_score, thr, dur,
                        )
                    self._self_state._overload_peak = max(
                        self._self_state._overload_peak,
                        float(self._self_state.load_score),
                    )
                    # Only reset reclaim eligibility outside the post-reclaim
                    # transition settle window — same guard as update_self_state.
                    if time.time() <= self._transition_settle_until:
                        pass  # preserve _reclaim_eligible_since
                    else:
                        self._reclaim_eligible_since = None
                else:
                    # ── Garbage-tick guard + hysteresis deadband ──────────────
                    # Prevent single-sample load=0.0 glitches and threshold
                    # dithering inside the deadband (clear_thr .. overload_thr)
                    # from resetting a genuine overload hold too early.
                    cur_load = float(self._self_state.load_score)
                    clear_thr = float(self._cfg.get("overload_clear_threshold", 50.0))
                    preserve = False
                    if self._self_state.overload_since is not None:
                        if cur_load <= 0.0:
                            # garbage tick: load_score 0.0 during overload
                            streak = int(getattr(self, "_overload_zero_streak", 0) or 0) + 1
                            self._overload_zero_streak = streak
                            if streak <= 5:
                                logger.debug(
                                    "[PeerOrch] OVERLOAD PRESERVED (garbage tick %d/5) load=%.1f",
                                    streak, cur_load,
                                )
                                preserve = True
                            else:
                                # 6+ consecutive zero ticks: probably not garbage, reset streak
                                self._overload_zero_streak = 0
                        elif cur_load >= clear_thr:
                            # still inside deadband (50-55) — hold the overload
                            logger.debug(
                                "[PeerOrch] OVERLOAD HELD load=%.1f (deadband %.1f-%.1f)",
                                cur_load, clear_thr, thr,
                            )
                            preserve = True
                        else:
                            # trustworthy tick below clear_thr → genuine recovery
                            self._overload_zero_streak = 0
                    if preserve:
                        return

                    now_ts = time.time()
                    if self._self_state.overload_since is not None:
                        peak = float(self._self_state._overload_peak)
                        held = now_ts - float(self._self_state.overload_since)
                        rsn = []
                        if float(self._self_state.load_score) < thr:
                            rsn.append("load-dip")
                        if not _resource_telemetry_valid(self._self_state):
                            rsn.append("telemetry-invalid")
                        if in_waiting:
                            status = str(self._self_state.status or "")
                            active = len(self._self_state.active_cameras or [])
                            rsn.append(f"waiting:{status}:{active}")
                        logger.info(
                            "[PeerOrch] OVERLOAD CLEAR peak=%.1f held=%.1fs reason=%s load=%.1f thr=%.1f",
                            peak, held, "+".join(rsn) if rsn else "none", float(self._self_state.load_score), thr,
                        )
                        self._self_state._overload_peak = 0.0
                    self._self_state.overload_since = None
                    self._ladder_l1_since = None
                    self._ladder_l1_camera = None
                    self._overload_zero_streak = 0
                    now_ts = time.time()
                    if self._below_thr_since is None:
                        self._below_thr_since = now_ts
                    dwell = float(self._cfg.get("offload_release_dwell_s", 15.0))
                    if now_ts - self._below_thr_since >= dwell:
                        for cam_id in list(self._self_state.active_cameras):
                            cl = self.get_offload_level(cam_id)
                            if cl in (1, 2):
                                start_ts = self._offload_started_at.pop(cam_id, 0.0)
                                dur = f" (duration: {now_ts - start_ts:.1f}s)" if start_ts > 0 else ""
                                self.set_offload_level(cam_id, 0)
                                logger.info("[PeerOrch] L%d offload ENDED for '%s': load_normalized%s", cl, cam_id, dur)
            return

        # Update state of other peers
        # BUG-04: update ALL mutable fields inside the lock to prevent torn
        # reads from _decision_loop running on a separate thread.
        with timed_lock(self._lock, "_lock.update_peer_state", logger=logger):
            is_new = node_id not in self._peers
            if is_new:
                self._peers[node_id] = PeerState(node_id=node_id)
                logger.info("[PeerOrch] Discovered peer '%s' via Zenoh", node_id)
            peer = self._peers[node_id]

            old_last_seen = peer.last_seen
            new_last_seen = time.time()
            mono_now = time.monotonic()
            if old_last_seen > 0.0:
                logger.debug(
                    "[PeerOrch] Peer heartbeat received: node='%s', last_seen_gap_s=%.3f, mono_ts=%.6f",
                    node_id, new_last_seen - old_last_seen, mono_now,
                )
            else:
                logger.debug(
                    "[PeerOrch] First peer heartbeat received: node='%s', mono_ts=%.6f",
                    node_id, mono_now,
                )

            peer.load_score  = payload.get("load_score",  0.0)
            peer.load_score_breakdown = payload.get("load_score_breakdown", {})
            peer.workload_ema = payload.get("workload_ema")
            peer.fps_ema = payload.get("fps_ema")
            peer.qos_state = payload.get("qos_state")
            peer.gpu_percent = payload.get("gpu_percent", 0.0)
            peer.cpu_percent = payload.get("cpu_percent", 0.0)
            peer.ram_percent = payload.get("ram_percent", 0.0)
            peer.gpu_temp_c  = payload.get("gpu_temp_c",  0.0)
            peer.risk_index  = payload.get("risk_index",  0.0)

            pipeline = payload.get("pipeline", {}) or {}
            peer.status         = pipeline.get("status")
            peer.pipeline_idle  = bool(pipeline.get("pipeline_idle", False))
            peer.avg_fps        = pipeline.get("avg_fps")
            # Prefer output_fps_per_camera (Phase 1 unambiguous key); fall back
            # to fps_per_camera for backward compatibility with older firmware.
            peer.fps_per_camera = _pick_fps_dict(pipeline)
            if not peer.fps_per_camera and peer.avg_fps and isinstance(peer.avg_fps, (int, float)) and peer.avg_fps > 0:
                # If per-camera fps dict is omitted in older/synthetic payloads, populate from avg_fps for active cameras
                if isinstance(pipeline.get("active_cameras"), (list, tuple, set)):
                    peer.fps_per_camera = {str(c): float(peer.avg_fps) for c in pipeline["active_cameras"] if isinstance(c, (str, int))}
            if isinstance(pipeline.get("active_cameras"), (list, tuple, set)):
                peer.active_cameras = [str(c) for c in pipeline["active_cameras"] if isinstance(c, (str, int))]
            if isinstance(pipeline.get("streaming_cameras"), (list, tuple, set)):
                peer.streaming_cameras = [str(c) for c in pipeline["streaming_cameras"] if isinstance(c, (str, int))]
            else:
                peer.streaming_cameras = list(peer.active_cameras)
            if isinstance(pipeline.get("held_cameras"), (list, tuple, set)):
                peer.held_cameras = [str(c) for c in pipeline["held_cameras"] if isinstance(c, (str, int))]
            else:
                peer.held_cameras = list(peer.active_cameras)
            # Preserve valid prior camera_configs when payload field is missing/malformed
            raw_configs = pipeline.get("camera_configs")
            if isinstance(raw_configs, dict) and raw_configs:
                peer.camera_configs = raw_configs
            elif isinstance(raw_configs, dict) and len(raw_configs) == 0:
                # Explicit valid empty snapshot only if pipeline is running and status indicates explicit state
                if (
                    pipeline.get("status") is not None
                    and str(pipeline.get("status")).strip().lower() not in ("waiting", "recovering", "recovery", "starting")
                    and "active_cameras" in pipeline
                    and len(peer.active_cameras) == 0
                ):
                    peer.camera_configs = {}
                # Otherwise (e.g. transient empty dictionary during recovery or startup), preserve prior valid configs
            # Backwards-compatible: missing or malformed mapping → empty.
            peer.camera_workload = _parse_camera_workload(
                pipeline.get("camera_workload", {})
            )
            peer.source_starved_cameras = _parse_starved_cameras(
                pipeline.get("source_starved_cameras", [])
            )
            peer.offload_crops_received_per_s = float(
                pipeline.get("offload_crops_received_per_s", 0.0) or 0.0
            )
            raw_queue_full = bool(pipeline.get("offload_queue_full", False))
            if raw_queue_full:
                peer.offload_queue_full = True
                peer.consecutive_queue_not_full_count = 0
            else:
                peer.consecutive_queue_not_full_count += 1
                release_count = int(self._cfg.get("offload_queue_full_release_count", 3))
                if peer.consecutive_queue_not_full_count >= release_count:
                    peer.offload_queue_full = False
            peer.offload_queue_depth = int(pipeline.get("offload_queue_depth", 0) or 0)
            peer.camera_owners = payload.get("camera_owners", {}) or {}
            peer.camera_holders = payload.get("camera_holders", {}) or {}
            peer.camera_epochs = payload.get("camera_epochs", {}) or {}
            # P5 — learn the peer's current boot_id for receiver fencing.
            try:
                peer.boot_id = int(payload.get("boot_id", 0) or 0)
            except (TypeError, ValueError):
                peer.boot_id = 0
            peer.last_seen = time.time()
            # max_streams from peer health payload; malformed values fall back to 8
            try:
                peer.max_streams = int(pipeline.get("max_streams", 8) or 8)
            except (TypeError, ValueError):
                peer.max_streams = 8

            # Track overload onset using same proactive-aware helper.
            # ADR-0001: overload is driven by workload/resource/qos, not FPS.
            # Fail closed only when resource telemetry is invalid; a peer with no
            # FPS is not automatically overloaded (warmup + status/active guards
            # suppress startup/recovery).  Source starvation is a symptom, not a
            # gating signal.
            peer_fps_valid = _has_valid_positive_fps(peer.fps_per_camera)
            peer_in_waiting = _waiting_by_status_or_idle(
                pipeline.get("status"),
                peer.active_cameras,
            )
            overloaded = (
                self._is_overloaded(
                    peer.load_score,
                    peer.risk_index,
                    peer.qos_state,
                )
                and _resource_telemetry_valid(peer)
                and not peer_in_waiting
            )
            if overloaded:
                if peer.overload_since is None:
                    peer.overload_since = time.time()
                    thr = float(self._cfg.get("overload_threshold", 55.0))
                    dur = float(self._cfg.get("overload_duration_s", 5.0))
                    logger.info(
                        "[PeerOrch] PEER OVERLOAD ONSET node='%s' load=%.1f thr=%.1f dur=%.1fs",
                        peer.node_id, peer.load_score, thr, dur,
                    )
                peer._overload_peak = max(
                    peer._overload_peak, float(peer.load_score),
                )
            else:
                now_ts = time.time()
                if peer.overload_since is not None:
                    peak = float(peer._overload_peak)
                    held = now_ts - float(peer.overload_since)
                    logger.info(
                        "[PeerOrch] PEER OVERLOAD CLEAR node='%s' peak=%.1f held=%.1fs",
                        peer.node_id, peak, held,
                    )
                    peer._overload_peak = 0.0
                peer.overload_since = None

            # Return / yield rescued camera only when owner is alive AND ready:
            # heartbeat fresh, peer not in waiting, valid positive FPS, not overloaded,
            # and cam_id is statically owned by the resuming node (matches Path B).
            # NOTE: cam_id in peer.held_cameras removed — inverted causality: A cannot
            # yield until C holds, but C cannot hold until A yields (deadlock). The
            # static-ownership guard replaces it safely (Oracle-confirmed 2026-09-15).
            peer_ready_to_resume = (
                peer_fps_valid
                and not peer_in_waiting
                and not overloaded
            )
            owner_static_cameras = self._get_node_owned_cameras(node_id)
            cameras_to_yield = [
                cam_id for cam_id, orig_owner in self._rescued_cameras.items()
                if orig_owner == node_id
                and peer_ready_to_resume
                and (owner_static_cameras is None or cam_id in owner_static_cameras)
            ]
            with self._lock:
                for cam_id in cameras_to_yield:
                    self._rescued_cameras.pop(cam_id, None)
                    self._rescued_at.pop(cam_id, None)
                    remove_cmd = self._build_remove_cmd(cam_id, context="immediate_yield")
                    if self._pubs.get("control") is not None:
                        self._pubs["control"].put(msgpack.packb(remove_cmd, use_bin_type=True))
                    logger.info(
                        "[PeerOrch] Original owner '%s' resumed '%s' (fps_valid=%s, load=%.1f). Immediate yield: sent REMOVE.",
                        node_id, cam_id, peer_fps_valid, peer.load_score,
                    )

            # Safe duplicate reconciliation defense-in-depth:
            # If self and an alive peer both report the same HELD camera, check monotonic epoch / identity:
            # 1. Higher epoch wins. Lower epoch node yields/removes.
            # 2. Equal epoch / absent identity: static owner is deterministic tie-breaker.
            # 3. Duplicate observation must be stable across at least 2 consecutive heartbeats.
            # Skip if camera is in-flight across rescue/reclaim/migration/warmup.
            to_remove_self_reconcile: List[str] = []
            with self._self_lock:
                self_held = set(self._self_state.held_cameras)

            # Update duplicate observation trackers
            peer_held = set(peer.held_cameras)
            now_ts = time.time()
            # _camera_added_at is set on every dynamic ADD and never popped:
            # a permanent membership test here silences reconciliation forever
            # (seen live: migrated cam_06 held 20+ min with zero reconcile).
            # Time-bound the in-flight guard past the ack window instead.
            inflight_grace_s = max(float(self._cfg.get("camera_warmup_s", 10.0)), 60.0)
            for cam_id in list(self_held & peer_held):
                key = (node_id, cam_id)
                self._duplicate_camera_seen[key] = self._duplicate_camera_seen.get(key, 0) + 1

                # Clean up if not active or not in-flight
                # Check in-flight and exclusion gates
                added_at = self._camera_added_at.get(cam_id, 0.0)
                recently_added = bool(added_at) and (now_ts - added_at) < inflight_grace_s
                if (
                    cam_id in self._rescued_cameras
                    or cam_id in self._reclaim_in_progress
                    or cam_id in self._pending_acks
                    or cam_id in self._pending_winner
                    or recently_added
                ):
                    continue

                if self._duplicate_camera_seen.get(key, 0) < 2:
                    continue

                # Monotonic epoch check
                self_epoch = self._camera_epochs.get(cam_id, 1)
                peer_epoch = peer.camera_epochs.get(cam_id, 1)

                should_self_yield = False
                if peer_epoch > self_epoch:
                    peer_fps = peer.fps_per_camera.get(cam_id, 0.0) if peer.fps_per_camera else 0.0
                    peer_streaming = (cam_id in peer.streaming_cameras) if peer.streaming_cameras else False
                    if peer_fps > 0.0 or peer_streaming:
                        logger.info(
                            "[PeerOrch][Reconcile] Peer '%s' has higher epoch (%d > %d) and is streaming '%s' (fps=%.1f). Self yielding.",
                            node_id, peer_epoch, self_epoch, cam_id, peer_fps,
                        )
                        should_self_yield = True
                    else:
                        logger.warning(
                            "[PeerOrch][Reconcile] Peer '%s' has higher epoch (%d > %d) for '%s' but stream is not verified playing (fps=%.1f, streaming=%s). Make-before-break: self holding.",
                            node_id, peer_epoch, self_epoch, cam_id, peer_fps, peer_streaming,
                        )
                        should_self_yield = False
                elif self_epoch > peer_epoch:
                    logger.info(
                        "[PeerOrch][Reconcile] Self has higher epoch (%d > %d) for duplicate camera '%s'. Peer '%s' expected to yield.",
                        self_epoch, peer_epoch, cam_id, node_id,
                    )
                    should_self_yield = False
                else:
                    # Equal epoch: use reported original owner vs current holder
                    # Check peer.camera_owners first, fall back to static mapping
                    peer_owner_claim = peer.camera_owners.get(cam_id)
                    peer_is_owner = (peer_owner_claim == node_id)

                    static_owned = self._get_owned_camera_ids()
                    self_is_owner = (cam_id in static_owned) or (self._rescued_cameras.get(cam_id) == self._node_id)
                    if peer_owner_claim is None:
                        peer_static = self._get_node_owned_cameras(node_id)
                        peer_is_owner = peer_static is not None and cam_id in peer_static

                    # Fail closed on ambiguity
                    if peer_is_owner and self_is_owner:
                        logger.warning(
                            "[PeerOrch][Reconcile] Ambiguous ownership for duplicate camera '%s': both '%s' and self configured/reporting as owner at epoch %d. Failing closed.",
                            cam_id, node_id, self_epoch,
                        )
                        continue
                    if not peer_is_owner and not self_is_owner:
                        logger.warning(
                            "[PeerOrch][Reconcile] Unresolved ownership for duplicate camera '%s' between '%s' and self at epoch %d. Failing closed.",
                            cam_id, node_id, self_epoch,
                        )
                        continue

                    if peer_is_owner and not self_is_owner:
                        should_self_yield = True

                if should_self_yield:
                    now_ts = time.time()
                    if now_ts - self._cam_cooldown.get(cam_id, 0.0) < 5.0:
                        continue
                    to_remove_self_reconcile.append(cam_id)
                    self._cam_cooldown[cam_id] = now_ts

            # Clean tracking for non-duplicate cameras on this peer
            for (p_node, p_cam) in list(self._duplicate_camera_seen.keys()):
                if p_node == node_id and p_cam not in (self_held & peer_held):
                    self._duplicate_camera_seen.pop((p_node, p_cam), None)

            # Issue REMOVE outside _lock
            for cam_id in to_remove_self_reconcile:
                remove_cmd = self._build_remove_cmd(cam_id, context="duplicate_reconciliation")
                if self._pubs.get("control") is not None:
                    self._pubs["control"].put(msgpack.packb(remove_cmd, use_bin_type=True))
                logger.info(
                    "[PeerOrch][Reconcile] Duplicate camera '%s' active on both self and static owner '%s'. Non-owner self yielding: REMOVE sent.",
                    cam_id, node_id,
                )

            # Phase 6 / Reviewer Finding 2.4: compute migration blind-spot metric (Δτ)
            # Check if any camera migrated out to this peer is now reporting valid FPS on the peer.
            for cam_id, mig_ts in list(self._migration_complete_ts.items()):
                if self._migrated_out.get(cam_id) == node_id:
                    cam_fps = peer.fps_per_camera.get(cam_id, 0.0) if peer.fps_per_camera else 0.0
                    if isinstance(cam_fps, (int, float)) and cam_fps > 0.0:
                        blind_spot_ms = max(0.0, (time.time() - mig_ts) * 1000.0)
                        logger.info(
                            "[PeerOrch] Migration blind-spot resolved for '%s' on peer '%s': %.0fms",
                            cam_id, node_id, blind_spot_ms,
                        )
                        # ponytail: log resolved blind spot row to CSV
                        self._migration_log.log(
                            self._node_id, node_id, cam_id,
                            "blind_spot_resolved", peer.load_score, cam_fps,
                            0.0, "RESOLVED",
                            blind_spot_ms=blind_spot_ms,
                        )
                        self._migration_complete_ts.pop(cam_id, None)

    def _is_overloaded(
        self,
        load_score: float,
        risk_index: float,
        qos_state: Optional[str] = None,
    ) -> bool:
        """
        Determine if this node (or a peer) should be considered overloaded.

        Load-driven (reactive) decision: qos_state is telemetry/display only
        and MUST NOT veto the score — a recognized 'healthy'/'moderate' qos
        with load_score above threshold is still overloaded (the qos bands
        mirror the same score, so a veto only injects snapshot-skew flapping).
        Proactive hard-fuse (risk_index >= hard_fuse) continues to override
        as a safety fuse.

        When proactive is enabled with a positive risk_index, the risk
        predictor decides; otherwise the legacy reactive load_score >=
        overload_threshold decides (shadow mode included — same rule).
        """
        proactive_cfg = self._cfg.get("proactive", {})
        hard_fuse = float(proactive_cfg.get("hard_fuse_threshold", 0.95))

        # Hard fuse — safety fuse against hardware saturation
        if not proactive_cfg.get("shadow_mode", False) and risk_index >= hard_fuse:
            return True

        # qos_state intentionally NOT consulted: it mirrors the same score
        # (health_agent bands: 72/55/30) so any veto only injects
        # snapshot-skew flapping around the threshold. Display only.

        # Shadow mode: telemetry only — strictly passive/reactive decisions
        if proactive_cfg.get("shadow_mode", False):
            return load_score >= self._cfg.get("overload_threshold", 55.0)

        if proactive_cfg.get("enabled", False) and risk_index > 0.0:
            threshold = float(proactive_cfg.get("risk_threshold", 0.85))
            return risk_index >= threshold

        # Legacy path
        return load_score >= self._cfg.get("overload_threshold", 55.0)

    def _effective_load(self, load_score: float, risk_index: float) -> float:
        """
        Return the score that best represents the current load for logging
        and for populating RFO payloads.  When proactive is enabled, returns
        risk_index × 100 (scaled to the same 0–100 range as load_score).
        """
        proactive_cfg = self._cfg.get("proactive", {})
        if proactive_cfg.get("enabled", False) and risk_index > 0.0:
            return round(risk_index * 100.0, 1)
        return load_score

    def _decision_loop(self) -> None:
        """Main loop — check overload + OFFLINE peers + rebalance."""
        logger.info("[PeerOrch] Decision loop started (interval=1s).")
        while self._running:
            time.sleep(1.0)
            try:
                self._check_offline_peers()
                self._check_rebalance()
                self._check_reclaim()
                self._check_stalled_branches()
                self._check_pending_migration_timeouts()
                self._check_self_overload()
            except Exception as exc:
                logger.error("[PeerOrch] Decision loop error: %s", exc)

    def _check_stalled_branches(self) -> None:
        """Dead-branch watchdog: restart owned-held cameras with zero FPS.

        A linked branch can wedge permanently with no error (EOS from a
        source flap, a decoder stall after a timestamp discontinuity): pads
        linked, zero buffers, forever. The static owner is the only node
        that may restart it — foreign branches are left to the owner's
        reclaim path. Bounded: stall_timeout_s (default 90s) of continuous
        zero FPS triggers one REMOVE+ADD restart; more than
        reclaim_max_retries consecutive failures park the camera 300s.
        """
        cfg = self._cfg
        now = time.time()
        stall_timeout_s = float(cfg.get("branch_stall_timeout_s", 90.0))
        max_restarts = int(cfg.get("reclaim_max_retries", 3))
        heartbeat_timeout = float(cfg.get("heartbeat_timeout_s", 5.0))

        with self._self_lock:
            if self._self_state.last_seen == 0.0 or (now - self._self_state.last_seen > heartbeat_timeout):
                return
            status = self._self_state.status
            fps_map = dict(self._self_state.fps_per_camera or {})
            held = set(self._self_state.held_cameras)
        if str(status or "").strip().lower() in ("waiting", "recovering", "recovery", "starting"):
            return
        try:
            owned_ids = self._get_owned_camera_ids()
        except Exception:
            return
        if not owned_ids:
            return
        with self._lock:
            in_flight = (set(self._pending_acks) | set(self._pending_winner)
                         | set(self._reclaim_in_progress) | set(self._rescued_cameras))
            retry_wait = dict(self._reclaim_retry_at)
            migrated = set(self._migrated_out)
        for cam_id in sorted(owned_ids & held):
            if cam_id in in_flight or cam_id in migrated:
                self._branch_zero_since.pop(cam_id, None)
                continue
            if now < self._branch_restart_parked_until.get(cam_id, 0.0):
                continue
            if now < retry_wait.get(cam_id, 0.0):
                continue
            # Warmup grace doubles as never-started grace: a freshly ADDed
            # branch gets stall_timeout_s before it counts as stalled.
            added_at = self._camera_added_at.get(cam_id, 0.0)
            if added_at and (now - added_at) < stall_timeout_s:
                continue
            if cam_id not in fps_map:
                if not fps_map:
                    # Writer itself silent — unknown, not proof of a stall.
                    self._branch_zero_since.pop(cam_id, None)
                    continue
                # Held branch with a live writer reporting siblings but zero
                # callbacks here (e.g. C/cam_06: key absent for hours while
                # cam_05 ticks 30fps): same stall signal as explicit 0.0.
                # Fall through to the since-clock below with fps=0.0.
                fps = 0.0
            else:
                fps = fps_map[cam_id]
            if isinstance(fps, (int, float)) and not isinstance(fps, bool) and fps > 0.0:
                self._branch_zero_since.pop(cam_id, None)
                self._branch_restart_fails.pop(cam_id, None)
                continue
            since = self._branch_zero_since.get(cam_id)
            if since is None:
                self._branch_zero_since[cam_id] = now
                continue
            if now - since < stall_timeout_s:
                continue
            fails = self._branch_restart_fails.get(cam_id, 0) + 1
            self._branch_restart_fails[cam_id] = fails
            self._branch_zero_since.pop(cam_id, None)
            if fails > max_restarts:
                self._branch_restart_parked_until[cam_id] = now + 300.0
                self._branch_restart_fails[cam_id] = 0
                logger.error(
                    "[PeerOrch][Watchdog] Branch '%s' still stalled after %d restarts — parking 300s.",
                    cam_id, max_restarts,
                )
                continue
            self._restart_owned_branch(cam_id)
            break  # one restart per tick

    def _restart_owned_branch(self, cam_id: str) -> None:
        """REMOVE an owned stalled branch now, re-ADD it after 10s.

        The delay lets the REMOVE teardown (EOS drain + sequential state
        walk) release the mux slot before the ADD recreates the branch.
        """
        with self._lock:
            cur_epoch = self._camera_epochs.get(cam_id, 1) + 1
            self._camera_epochs[cam_id] = cur_epoch
        remove_cmd = self._build_remove_cmd(cam_id, context="stalled_branch_restart")
        if self._pubs.get("control") is not None:
            try:
                self._pubs["control"].put(msgpack.packb(remove_cmd, use_bin_type=True))
            except Exception as exc:
                logger.warning("[PeerOrch][Watchdog] Failed to publish REMOVE for '%s': %s", cam_id, exc)
                return
        cam_config = self._get_camera_config(cam_id)
        if cam_config is None:
            logger.warning("[PeerOrch][Watchdog] Cannot restart '%s': no camera config", cam_id)
            return
        add_cmd = {
            **cam_config,
            "cmd": "ADD",
            "epoch": cur_epoch,
            "migration_id": f"restart_{cam_id}_{int(time.time() * 1000)}",
        }
        if getattr(self, "_boot_id", 0):
            add_cmd["boot_id"] = self._boot_id

        def _delayed_add() -> None:
            # Poll until mux slot is free to avoid racing teardown (ghost pad still linked).
            # Worst teardown is EOS 1s + 3*5s TSG + 3*8s sink ≈ 16-18s, so poll up to 20s.
            deadline = time.time() + 20.0
            while time.time() < deadline:
                try:
                    # If local pipeline still reports cam active, slot not free yet
                    with self._self_lock:
                        still_held = cam_id in (self._self_state.held_cameras or [])
                        still_active = cam_id in (self._self_state.active_cameras or [])
                    if not still_held and not still_active:
                        break
                except Exception:
                    break
                time.sleep(0.5)
            pubs = self._pubs.get("control")
            if pubs is not None:
                try:
                    pubs.put(msgpack.packb(add_cmd, use_bin_type=True))
                except Exception as exc:
                    logger.warning("[PeerOrch][Watchdog] Failed to publish ADD for '%s': %s", cam_id, exc)
                    return
            with self._lock:
                self._camera_added_at[cam_id] = time.time()
            self._camera_first_valid_fps_at.pop(cam_id, None)

        t = threading.Timer(18.0, _delayed_add)
        t.daemon = True
        t.start()
        logger.warning(
            "[PeerOrch][Watchdog] Restarting stalled owned branch '%s' (zero FPS): REMOVE sent, ADD in 10s (epoch=%d).",
            cam_id, cur_epoch,
        )

    def _check_offline_peers(self) -> None:
        """
        Detect offline peers (heartbeat timeout).
        If peer has active cameras → trigger leaderless failover.

        Grace/convergence guard: a peer is only declared OFFLINE after
        ``heartbeat_timeout_s + failover_grace_s`` of silence.  The extra
        grace window prevents transient Zenoh / network blips from
        triggering an expensive leaderless failover for a peer that is
        still alive but briefly unreachable.  Default grace equals
        ``heartbeat_timeout_s`` so the total wait is 2× the heartbeat
        timeout — configurable via ``p2p.failover_grace_s`` in
        ``Edge/configs/edge_node.yml``.

        Failover convergence grace:
        When a failover has just been triggered (within ``failover_convergence_grace_s``,
        default 15s), suppress declaring other peers offline to prevent cascading
        mutual false-offline cascades during convergence. Detection resumes once
        the grace window expires.
        """
        now = time.time()
        timeout = float(self._cfg.get("heartbeat_timeout_s", 5.0))
        grace_s = float(self._cfg.get("failover_grace_s", timeout))
        offline_threshold = timeout + grace_s
        convergence_grace_s = float(self._cfg.get("failover_convergence_grace_s", 15.0))

        with timed_lock(self._lock, "_lock.check_offline_peers.ready_check", logger=logger):
            if not self._pipeline_ready:
                logger.debug("[PeerOrch] Local pipeline not ready yet. Suppressing peer-offline detection and failover.")
                return

        with timed_lock(self._self_lock, "_self_lock.check_offline_peers.self_heartbeat", logger=logger):
            if self._self_state.last_seen == 0.0 or (now - self._self_state.last_seen > offline_threshold):
                logger.debug(
                    "[PeerOrch] Self heartbeat stale or missing (age=%.1fs > %.1fs). "
                    "Suppressing peer-offline detection.",
                    now - self._self_state.last_seen,
                    offline_threshold,
                )
                return

        with timed_lock(self._lock, "_lock.check_offline_peers.scan", logger=logger):
            # Per-dead-node / per-peer suppression:
            # Map of node_id -> timestamp of recent failover/offline event
            all_offline_events = {**self._failover_triggered, **self._peer_offline_at}
            to_check = list(self._peers.items())

        for node_id, peer in to_check:
            if node_id == self._node_id:
                continue
            silent_s = now - peer.last_seen
            # Harden peer offline handling: ONLY heartbeat silence (silent_s > offline_threshold)
            # can declare a peer offline.
            # If peer heartbeat is arriving (silent_s <= offline_threshold), even with
            # fps={}, 0 active cameras, or waiting/recovery state, the peer is alive
            # and MUST NOT be declared offline or trigger failover rescue.
            if silent_s > offline_threshold:
                with timed_lock(self._lock, "_lock.check_offline_peers.check_dead", logger=logger):
                    already_triggered = node_id in self._failover_triggered
                    last_event_time = all_offline_events.get(node_id, 0.0)
                # Per-dead-node suppression: if this specific node already triggered recently,
                # suppress re-declaring it offline during its own convergence window.
                # Other dead peers are NOT blocked.
                if not already_triggered and (now - last_event_time) < convergence_grace_s and last_event_time > 0.0:
                    continue

                with timed_lock(self._lock, "_lock.check_offline_peers.mark_offline", logger=logger):
                    self._peer_offline_at[node_id] = now
                self._clear_offload_target(node_id)
                # Reclaim cameras migrated out to this dead peer locally
                self._reclaim_migrated_from_dead_peer(node_id)

                # Compute orphan cameras from authoritative static config first, then dynamic states
                candidate_orphans: List[str] = []
                seen_candidates = set()

                # 1. Authoritative static node-owned cameras
                static_owned = self._get_node_owned_cameras(node_id)
                if static_owned:
                    for c in sorted(static_owned):
                        if c and c not in seen_candidates:
                            candidate_orphans.append(c)
                            seen_candidates.add(c)

                # 2. held_cameras (includes warming-up and stalled streams with live pipeline branches)
                for c in peer.held_cameras:
                    if c and c not in seen_candidates:
                        candidate_orphans.append(c)
                        seen_candidates.add(c)
                # 3. camera_configs keys
                if isinstance(peer.camera_configs, dict):
                    for c in peer.camera_configs.keys():
                        if c and c not in seen_candidates:
                            candidate_orphans.append(c)
                            seen_candidates.add(c)
                # 4. _migrated_out entries whose holder is the dead peer
                with timed_lock(self._lock, "_lock.check_offline_peers.find_migrated_out", logger=logger):
                    for c, holder in self._migrated_out.items():
                        if holder == node_id and c and c not in seen_candidates:
                            candidate_orphans.append(c)
                            seen_candidates.add(c)

                if candidate_orphans:
                    # Time-based re-arm: a failed attempt (e.g. RTSP sources
                    # unreachable while the dead host is down) must retry once
                    # sources come back, instead of latching until the peer's
                    # heartbeat revives.
                    retry_interval_s = float(self._cfg.get("failover_retry_interval_s", 30.0))
                    with timed_lock(self._lock, "_lock.check_offline_peers.trigger_failover", logger=logger):
                        last_attempt = self._failover_triggered.get(node_id, 0.0)
                        should_trigger = (now - last_attempt) >= retry_interval_s
                        if should_trigger:
                            self._failover_triggered[node_id] = now
                    if should_trigger:
                        orphans = list(candidate_orphans)
                        if node_id not in self._notified_offline:
                            logger.critical(
                                "[PeerOrch] Peer '%s' OFFLINE: reason=heartbeat_timeout, last_seen_s=%.2f, silent_s=%.2f, threshold_s=%.2f, mono_ts=%.6f, cameras=%d. Triggering failover...",
                                node_id, peer.last_seen, silent_s, offline_threshold, time.monotonic(), len(orphans),
                            )
                            self._notified_offline.add(node_id)
                        else:
                            logger.info(
                                "[PeerOrch] Peer '%s' still OFFLINE — re-attempting failover for %d cameras (retry interval %.1fs, mono_ts=%.6f).",
                                node_id, len(orphans), retry_interval_s, time.monotonic(),
                            )
                        self._executor.submit(self._leaderless_failover, node_id, orphans)
                else:
                    if node_id not in self._notified_offline:
                        logger.warning(
                            "[PeerOrch] Peer '%s' OFFLINE: reason=heartbeat_timeout_no_cameras, last_seen_s=%.2f, silent_s=%.2f, threshold_s=%.2f, mono_ts=%.6f",
                            node_id, peer.last_seen, silent_s, offline_threshold, time.monotonic(),
                        )
                        self._notified_offline.add(node_id)
            else:
                # Peer is alive — clear the notified/failover flags so we
                # react again if it goes offline a second time.
                self._notified_offline.discard(node_id)
                with timed_lock(self._lock, "_lock.check_offline_peers.alive_reset", logger=logger):
                    self._failover_triggered.pop(node_id, None)
                    self._peer_offline_at.pop(node_id, None)

    def _clear_offload_target(self, node_id: str) -> None:
        """Clear any offload entries that target an offline peer.

        After the P3 redesign only Level 1 (full-stream migration) exists; its
        lease state is tracked via _migrated_out / HRW claims, not _offload_targets,
        so this is now effectively a no-op safety net for any residual entry.
        """
        with self._offload_lock:
            affected = [
                camera_id for camera_id, target in self._offload_targets.items()
                if target == node_id and self._offload_table.get(camera_id, 0) != 0
            ]
            for camera_id in affected:
                old_level = self._offload_table.get(camera_id, 0)
                self._offload_table[camera_id] = 0
                self._offload_targets[camera_id] = ""
                self._offload_level_changed_at[camera_id] = time.time()
                start_ts = self._offload_started_at.pop(camera_id, 0.0)
                dur = f" (duration: {time.time() - start_ts:.1f}s)" if start_ts > 0 else ""
                logger.info("[PeerOrch] L%d offload ENDED for '%s': peer_offline%s", old_level, camera_id, dur)
                logger.warning(
                    "[PeerOrch] Offload target '%s' offline — clearing L%d for '%s'",
                    node_id, old_level, camera_id,
                )

    def _reclaim_migrated_from_dead_peer(self, dead_node_id: str) -> None:
        """Reclaim cameras owned by this node that were migrated out to a peer that just died."""
        with self._lock:
            migrated_to_dead = [
                cam_id for cam_id, holder in self._migrated_out.items()
                if holder == dead_node_id
            ]

        if not migrated_to_dead:
            return

        now = time.time()
        for camera_id in migrated_to_dead:
            with self._lock:
                if camera_id in self._reclaim_in_progress:
                    continue
                retry_at = self._reclaim_retry_at.get(camera_id, 0.0)
                if now < retry_at:
                    continue

            # Check if camera is already running locally
            with self._self_lock:
                if camera_id in self._self_state.active_cameras:
                    with self._lock:
                        self._migrated_out.pop(camera_id, None)
                        self._reclaim_in_progress.discard(camera_id)
                        self._reclaim_retry_at.pop(camera_id, None)
                    continue

            cam_config = self._get_camera_config(camera_id)
            if cam_config is None:
                logger.warning(
                    "[PeerOrch] Dead-peer reclaim: cannot get config for '%s', skipping",
                    camera_id,
                )
                continue

            now_ts = time.time()
            with self._lock:
                cur_epoch = self._camera_epochs.get(camera_id, 1) + 1
                self._camera_epochs[camera_id] = cur_epoch
                mig_id = f"mig_{camera_id}_{int(now_ts * 1000)}"
                self._camera_migration_ids[camera_id] = mig_id
                self._pending_migration_ids[camera_id] = mig_id
                self._pending_epochs[camera_id] = cur_epoch

            add_cmd = {
                **cam_config,
                "cmd": "ADD",
                "epoch": cur_epoch,
                "migration_id": mig_id,
            }
            # P5: stamp THIS node's boot_id so a pre-reboot reclaim ADD is fenced.
            if getattr(self, "_boot_id", 0):
                add_cmd["boot_id"] = self._boot_id
            event = threading.Event()
            with self._lock:
                # Register before publishing ADD: the receiver can ACK
                # immediately, before the waiter thread gets scheduled.
                self._pending_acks[camera_id] = event
            if self._pubs.get("control") is not None:
                self._pubs["control"].put(msgpack.packb(add_cmd, use_bin_type=True))
            self._camera_added_at[camera_id] = now
            self._camera_first_valid_fps_at.pop(camera_id, None)
            with self._lock:
                self._reclaim_in_progress.add(camera_id)

            logger.info(
                "[PeerOrch] Dead-peer reclaim: ADD '%s' back to self (holder '%s' dead), waiting for ack...",
                camera_id, dead_node_id,
            )
            self._reclaim_completed_at[camera_id] = now
            with self._lock:
                self._reclaim_retry_count[camera_id] = 0
            self._transition_settle_until = now + self._cfg.get("transition_settle_s", 5.0)

            self._executor.submit(self._wait_and_remove_reclaim, camera_id, dead_node_id)

    def _check_rebalance(self) -> None:
        """
        Return rescued cameras when their original owner comes back online.

        If peer X died and we rescued cam_01, cam_02, and then X restarts
        and reports cam_01, cam_02 in its active_cameras, we remove our
        rescued copies to avoid duplicate streams.

        Guards (per spec):
        - owner absent (peer is None) → skip; never REMOVE/return to a
          peer that is no longer in the routing table.
        - owner stale (now - last_seen > heartbeat_timeout_s, default 5 s)
          → skip; the heartbeat timeout is sourced from config so operators
          can tune it.
        - owner not running the camera (not in peer.active_cameras) → skip;
          rebalance is only valid when the owner has actually resumed
          processing it.
        - last-active-camera guard → never return a rescued camera that
          would leave this node processing zero streams.
        - hard owned-camera invariant (NEW) → never leave this node with
          zero locally-owned cameras (cameras.yml via _get_owned_camera_ids),
          not merely zero active cameras.  Implemented as two layered checks:
            1. aggregate guard: if no locally-owned camera is currently
               active, block ALL returns (the node is in a degraded state;
               foreign returns make it worse).
            2. per-camera guard: if a specific camera in _rescued_cameras is
               locally-owned (defensive — _rescued_cameras should only ever
               hold foreign cameras), skip its return.
          Failover rescue is unaffected because _leaderless_failover does
          not route through _check_rebalance.
        """
        with self._lock:
            if not self._rescued_cameras:
                return

        now = time.time()
        timeout = self._cfg.get("heartbeat_timeout_s", 5.0)
        to_return: List[str] = []

        # ponytail: snapshot self-active under _self_lock and rescued/peers
        # under _lock so we can compute "after return" atomically without
        # races.  _camera_added_at is not read here.
        with self._self_lock:
            self_active_snapshot = set(self._self_state.active_cameras)

        # Resolve locally-owned camera IDs (live CameraManager first, then
        # cameras.yml).  ponytail: wrap in try/except — the helper is
        # documented as fail-safe (returns empty on error) but we belt-and-
        # brace it because a leak here would silently violate the hard
        # invariant that protects L1 migration availability.
        try:
            owned_ids = self._get_owned_camera_ids()
        except Exception as exc:
            logger.warning(
                "[PeerOrch][Rebalance] _get_owned_camera_ids() raised: %s; "
                "treating ownership as unresolved and blocking all returns.",
                exc,
            )
            owned_ids = None
        if owned_ids is None:
            owned_ids = set()
        owned_active_snapshot = owned_ids & self_active_snapshot

        # ponytail: no rescue-hold expiry DROP. Rescued cameras stay on the
        # rescuer until the original owner revives and resumes them (normal
        # return path below). The old force-REMOVE left the camera owned by
        # nobody cluster-wide when the owner stayed dead.

        # Aggregate owned-camera guard: zero locally-owned active cameras
        # means the node cannot satisfy the L1 ownership invariant for
        # migration decisions (see _pick_camera_to_offload).  Returning
        # any rescued camera would not change owned_active (rescued cameras
        # are foreign) but it would also fail to fix the degradation, so
        # we hold all rescues here until at least one owned camera resumes.
        # This is the hard invariant requested in the spec.
        # ponytail: rate-limit the diagnostic — this branch fires every 1s
        # tick while the node is in the degraded state; one log per
        # BLOCKED_LOG_COOLDOWN (15 s) is enough to alert without spam.
        if not owned_active_snapshot:
            with self._lock:
                has_rescues = bool(self._rescued_cameras)
            if has_rescues and self._maybe_log_block("rebalance_no_owned", now):
                logger.warning(
                    "[PeerOrch][Rebalance] BLOCKED: no locally-owned active "
                    "cameras (owned_active=0, rescued=%d). Holding rescued "
                    "cameras until an owned stream resumes.",
                    len(self._rescued_cameras),
                )
            return

        with self._lock:
            rescued_snapshot = list(self._rescued_cameras.items())
            peers_snapshot = dict(self._peers)
        # Decrement as we commit each return so two concurrent eligible
        # returns don't both pass the "more than one active remains" check.
        remaining_after = set(self_active_snapshot)
        for camera_id, original_owner in rescued_snapshot:
            # Per-camera guard: an owned camera must NEVER be returned by
            # this rebalance path.  _rescued_cameras is only ever populated
            # by _leaderless_failover from a peer's active_cameras, but a
            # transition window or a misconfigured yml could in principle
            # place an owned camera here.  Skip defensively.
            if camera_id in owned_ids:
                if self._maybe_log_block(f"rebalance_owned_{camera_id}", now):
                    logger.warning(
                        "[PeerOrch][Rebalance] Skipping return of '%s' to '%s': "
                        "camera is locally-owned; this rebalance path only "
                        "handles foreign rescued cameras.",
                        camera_id, original_owner,
                    )
                continue
            # Return rescued camera only if original owner is online AND ready:
            # fresh owner heartbeat, owner not waiting/recovery, owner not overloaded,
            # and static ownership matches original owner. REMOVE locally without requiring
            # owner active_cameras.
            peer = peers_snapshot.get(original_owner)
            if peer is None:
                # Owner absent from routing table.
                continue
            if now - peer.last_seen > timeout:
                # Owner heartbeat older than configured timeout — stale.
                continue
            peer_in_waiting = is_waiting_state(peer.fps_per_camera, peer.active_cameras, peer.status)
            peer_overloaded = self._is_overloaded(
                peer.load_score,
                peer.risk_index,
                peer.qos_state,
            )
            if peer_in_waiting or peer_overloaded:
                continue
            # Accept return only if qos_state is healthy/moderate when supplied; legacy blank remains compatible.
            if peer.qos_state is not None and str(peer.qos_state).strip().lower() not in ("healthy", "moderate", ""):
                continue
            # Validate static ownership of the original owner
            owner_static_cameras = self._get_node_owned_cameras(original_owner)
            if owner_static_cameras is not None and camera_id not in owner_static_cameras:
                if self._maybe_log_block(f"rebalance_not_owned_{camera_id}", now):
                    logger.warning(
                        "[PeerOrch][Rebalance] Skipping return of '%s' to '%s': camera is not statically owned by peer.",
                        camera_id, original_owner,
                    )
                continue
            # Owner is alive, not waiting/recovering, not overloaded, and statically owns the camera! Yield.
            remaining_after.discard(camera_id)
            to_return.append(camera_id)

        for camera_id in to_return:
            with self._lock:
                original_owner = self._rescued_cameras.pop(camera_id, None)
                self._rescued_at.pop(camera_id, None)
            if original_owner is None:
                continue
            remove_cmd = self._build_remove_cmd(camera_id, context="rebalance_return")
            self._pubs["control"].put(msgpack.packb(remove_cmd, use_bin_type=True))
            logger.info(
                "[Rebalance] Returning '%s' to original owner '%s'. REMOVE sent.",
                camera_id, original_owner,
            )
            # BUG-1 fix: read _self_state under its lock
            with self._self_lock:
                _self_load = self._self_state.load_score
                _self_fps  = self._self_state.avg_fps
            self._migration_log.log(
                self._node_id, original_owner, camera_id,
                "rebalance_return", _self_load, _self_fps,
                0.0, "RETURNED",
            )

    def _check_reclaim(self) -> None:
        """
        Reclaim cameras that were migrated away due to overload, once this
        node's load drops sufficiently below the overload threshold.

        Reclaim condition:
          - This node's load_score has been below (overload_threshold - reclaim_margin)
            for at least reclaim_stable_s seconds.
          - This node stays below the capacity ceiling after taking the camera back
            (active_cameras + 1 < effective_capacity).
          - The camera is still being held by the peer it was migrated to
            (confirmed via peer heartbeat).
          - Cooldown has expired since the migration.

        Reclaim is done one camera at a time to avoid oscillation.
        """
        if not self._migrated_out:
            return

        cfg = self._cfg
        now = time.time()

        reclaim_threshold = cfg.get("overload_threshold", 55.0) - cfg.get("reclaim_margin", 12.0)
        reclaim_stable_s  = cfg.get("reclaim_stable_s", 5.0)
        cooldown_s        = cfg.get("cooldown_s", 6.0)
        heartbeat_timeout = cfg.get("heartbeat_timeout_s", 5.0)

        with self._self_lock:
            if self._self_state.last_seen == 0.0 or (now - self._self_state.last_seen > heartbeat_timeout):
                if self._maybe_log_block("stale_self_reclaim", now):
                    logger.warning(
                        "[PeerOrch] Self heartbeat stale or missing (age=%.1fs > %.1fs). "
                        "Skipping reclaim check.",
                        now - self._self_state.last_seen, heartbeat_timeout,
                    )
                return
            load         = self._self_state.load_score
            overload_since = self._self_state.overload_since
            self_active_count = len(self._self_state.active_cameras)
            self_max_streams = self._self_state.max_streams

        # Capacity guard: block reclaim when adding a camera would reach/exceed local stream capacity
        cm = self._camera_manager
        local_max = cm.get_max_streams() if cm is not None else self_max_streams
        configured_max = int(cfg.get("eps_streams_max", local_max))
        effective_capacity = min(local_max, configured_max)
        if self_active_count + 1 >= effective_capacity:
            logger.debug(
                "[PeerOrch] Reclaim blocked by local capacity: active=%d + 1 >= capacity=%d",
                self_active_count, effective_capacity,
            )
            return

        # Only reclaim if load has been stable and low
        if load >= reclaim_threshold:
            self._reclaim_eligible_since = None
            return
        # overload_since being None means load has dropped — good.
        # If it's still set, the node is still above overload_threshold.
        if overload_since is not None:
            return
        # Wait reclaim_stable_s after load dropped before reclaiming.
        # _reclaim_eligible_since is initialised to None in __init__ and set
        # here on first entry; the hasattr guard was dead code.
        if self._reclaim_eligible_since is None:
            self._reclaim_eligible_since = now
            return
        if now - self._reclaim_eligible_since < reclaim_stable_s:
            return

        # Orphan adoption (restart hole): _migrated_out is in-memory only, so
        # any Edge restart orphans migrated cameras cluster-wide — no node
        # tracks them and the owner never reclaims (seen live 2026-09-18:
        # cam_06 held by nobody after Edge restarts). Reconstruct tracking
        # for statically-owned cameras not held locally. One camera per tick.
        # Orphan confirmation requires a fresh heartbeat from EVERY static
        # peer — otherwise the holder may simply be undiscovered.
        try:
            owned_ids = self._get_owned_camera_ids()
        except Exception:
            owned_ids = set()
        if owned_ids:
            node_cam_map = self._cfg.get("node_camera_map") or {}
            static_peers = [n for n in node_cam_map if n != self._node_id]
            with self._self_lock:
                self_held = set(self._self_state.held_cameras)
                self_active = set(self._self_state.active_cameras)
            with self._lock:
                tracked = set(self._migrated_out) | set(self._reclaim_in_progress)
                peers_snap = dict(self._peers)
            for cam_id in sorted(owned_ids):
                if cam_id in self_held or cam_id in self_active or cam_id in tracked:
                    continue
                if not static_peers:
                    break
                if any(
                    n not in peers_snap or (now - peers_snap[n].last_seen > heartbeat_timeout)
                    for n in static_peers
                ):
                    break  # view not settled — retry next tick
                holder = None
                for pid in sorted(peers_snap):
                    p = peers_snap[pid]
                    if pid != self._node_id and cam_id in p.held_cameras:
                        holder = pid
                        break
                with self._lock:
                    # Sentinel "orphan": confirmed holderless. The reclaim loop
                    # below ADDs locally; the REMOVE step self-skips.
                    self._migrated_out[cam_id] = holder or "orphan"
                logger.warning(
                    "[PeerOrch][Reclaim] Adopting untracked owned camera '%s' (holder=%s).",
                    cam_id, holder or "orphan",
                )
                break

        # Find one camera to reclaim (oldest migration first)
        with self._lock:
            candidates = list(self._migrated_out.items())

        for camera_id, holder_node in candidates:
            # Check in-progress and retry timing
            with self._lock:
                if camera_id in self._reclaim_in_progress:
                    continue
                retry_at = self._reclaim_retry_at.get(camera_id, 0.0)
            if now < retry_at:
                continue

            # Check cooldown
            last_mig = self._cam_cooldown.get(camera_id, 0.0)
            if now - last_mig < cooldown_s:
                continue

            # 1. Check if holder is alive and still reports the camera.
            # Sentinel "orphan" (adoption gate) skips holder checks and goes
            # straight to local ADD; the REMOVE step self-skips for it.
            if holder_node == "orphan":
                pass
            else:
                with self._lock:
                    holder_peer = self._peers.get(holder_node)
                holder_alive = holder_peer is not None and (now - holder_peer.last_seen <= heartbeat_timeout)

            if holder_node == "orphan":
                # Confirmed holderless — proceed directly to local ADD below.
                pass
            elif holder_alive and holder_peer is not None and camera_id in holder_peer.held_cameras:
                # Still running fine on holder; check if load/cooldown allows normal reclaim
                pass
            elif holder_alive and holder_peer is not None and camera_id not in holder_peer.held_cameras:
                # Holder dropped the camera while still alive!
                # If camera is already active on self, reclaim has succeeded; clear state safely.
                with self._self_lock:
                    is_active_local = camera_id in self._self_state.active_cameras
                if is_active_local:
                    logger.info(
                        "[PeerOrch][Reclaim] Camera '%s' is already active locally and holder '%s' no longer has it. Clearing reclaim mapping.",
                        camera_id, holder_node,
                    )
                    with self._lock:
                        self._migrated_out.pop(camera_id, None)
                        self._reclaim_in_progress.discard(camera_id)
                        self._reclaim_retry_at.pop(camera_id, None)
                        self._reclaim_retry_count.pop(camera_id, None)
                        self._reclaim_attempts.pop(camera_id, None)
                        self._reclaim_pending_remove.pop(camera_id, None)
                    continue

                # Inspect latest fresh peer heartbeats for another alive peer reporting camera_id
                other_holder = None
                with self._lock:
                    for pid, p in self._peers.items():
                        if pid != holder_node and (now - p.last_seen <= heartbeat_timeout) and (camera_id in p.held_cameras):
                            other_holder = pid
                            break
                    if other_holder is not None:
                        # Found another alive peer reporting this camera; update mapping atomically & skip local ADD
                        self._migrated_out[camera_id] = other_holder
                        self._reclaim_in_progress.discard(camera_id)
                        self._reclaim_retry_at.pop(camera_id, None)
                        self._reclaim_retry_count.pop(camera_id, None)
                        self._reclaim_attempts.pop(camera_id, None)
                        self._reclaim_pending_remove.pop(camera_id, None)

                if other_holder is not None:
                    logger.info(
                        "[PeerOrch][Reclaim] Camera '%s' found active on alive peer '%s' (reassigned from '%s'). Skipping local ADD.",
                        camera_id, other_holder, holder_node,
                    )
                    continue

                logger.warning(
                    "[PeerOrch][Reclaim] Camera '%s' missing from active_cameras of alive holder '%s'! Initiating recovery...",
                    camera_id, holder_node,
                )
            else:
                # Holder offline or unknown — let dead-peer recovery / offline check handle it, but do not drop tracking
                continue

            # Make-before-Break: send ADD to self FIRST (do not REMOVE holder until stream PLAYING).
            # _wait_and_remove_reclaim() will send REMOVE to holder only after local ADD is confirmed.

            # Send ADD to self (reclaim)
            cam_config = self._get_camera_config(camera_id)
            if cam_config is None:
                logger.warning("[PeerOrch] Reclaim: cannot get config for '%s', skipping", camera_id)
                continue

            # Make-before-Break: Step 1 — ADD to self first, wait for stream PLAYING ack
            # Step 2 — Only then REMOVE from holder
            now_ts = time.time()
            with self._lock:
                cur_epoch = self._camera_epochs.get(camera_id, 1) + 1
                mig_id = f"mig_{camera_id}_{int(now_ts * 1000)}"
                self._camera_migration_ids[camera_id] = mig_id
                self._pending_migration_ids[camera_id] = mig_id
                self._pending_epochs[camera_id] = cur_epoch

            add_cmd = {
                **cam_config,
                "cmd": "ADD",
                "epoch": cur_epoch,
                "migration_id": mig_id,
            }
            # P5: stamp THIS node's boot_id so a pre-reboot reclaim ADD is fenced.
            if getattr(self, "_boot_id", 0):
                add_cmd["boot_id"] = self._boot_id
            event = threading.Event()
            with self._lock:
                # Register before publishing ADD: the receiver can ACK
                # immediately, before the waiter thread gets scheduled.
                self._pending_acks[camera_id] = event
                self._reclaim_in_progress.add(camera_id)
            if self._pubs.get("control") is not None:
                self._pubs["control"].put(msgpack.packb(add_cmd, use_bin_type=True))
            # ponytail: record the ADD so the per-camera warmup gate in
            # _pick_camera_to_offload suppresses offload actions on this
            # camera until its FPS has been valid for camera_warmup_s seconds.
            self._camera_added_at[camera_id] = now
            # Clear any stale first-valid-fps snapshot so the warmup restarts
            # from zero for this new ADD event.
            self._camera_first_valid_fps_at.pop(camera_id, None)

            logger.info(
                "[PeerOrch] Reclaim: load=%.1f < threshold=%.1f (risk=%.2f, active=%d) — "
                "ADD '%s' back to self (held by '%s'), waiting for ack...",
                load, reclaim_threshold, self._self_state.risk_index,
                len(self._self_state.active_cameras), camera_id, holder_node,
            )

            # Record reclaim start: this camera is ineligible for offload until
            # its FPS stabilises after returning home.
            self._reclaim_completed_at[camera_id] = now
            with self._lock:
                self._reclaim_retry_count[camera_id] = 0
                # ponytail: do NOT reset _reclaim_attempts here — it accumulates across
                # all retry cycles and is the give-up gate; resetting it here caused
                # infinite reclaim loops (always appeared as attempt 1).
                self._reclaim_pending_remove.pop(camera_id, None)

            # Suppress overload decisions for the settle window once reclaim
            # ADD is initiated: incoming stream warm-up FPS can look like a
            # fresh overload and re-escalate the node moments after reclaim.
            self._transition_settle_until = now + cfg.get("transition_settle_s", 5.0)

            # Spin up a thread that waits for the local ADD ack then removes holder
            self._executor.submit(self._wait_and_remove_reclaim, camera_id, holder_node)

            self._cam_cooldown[camera_id] = now
            # Reset eligible timer to avoid immediately reclaiming next camera
            self._reclaim_eligible_since = now

            self._migration_log.log(
                holder_node, self._node_id, camera_id,
                "reclaim", load, None,
                0.0, "RECLAIM_INITIATED",
            )
            # Reclaim one at a time
            return

    def _check_pending_migration_timeouts(self) -> None:
        """
        Periodically garbage-collect stale pending migration state that timed out
        without an ACK or where a background thread might have exited unexpectedly.
        Cleans up _pending_winner, _pending_started_at, _pending_acks, and inflight reservations.
        """
        now = time.time()
        timeout_s = float(self._cfg.get("migration_timeout_s", 15.0))
        rescue_timeout_s = float(self._cfg.get("rescue_ack_timeout_s", 30.0))
        # Garbage collect any pending migration running longer than 2x timeout or min 30s
        stale_threshold = max(30.0, timeout_s * 2.0)
        stale_cleanups = []
        rescue_cleanups = []
        with self._lock:
            for cam_id, started_at in list(self._pending_started_at.items()):
                if (now - started_at) >= stale_threshold:
                    winner_id = self._pending_winner.pop(cam_id, None)
                    self._pending_started_at.pop(cam_id, None)
                    self._pending_acks.pop(cam_id, None)
                    self._pending_epochs.pop(cam_id, None)
                    self._pending_migration_ids.pop(cam_id, None)
                    if winner_id:
                        stale_cleanups.append((cam_id, winner_id))
            # Rescue-ACK timeout: camera waiting for rescue acknowledgement.
            for cam_id, resc_at in list(self._pending_rescue_at.items()):
                if (now - resc_at) >= rescue_timeout_s:
                    self._pending_rescue.pop(cam_id, None)
                    self._pending_rescue_at.pop(cam_id, None)
                    winner_id = self._pending_winner.pop(cam_id, None)
                    self._pending_acks.pop(cam_id, None)
                    self._pending_epochs.pop(cam_id, None)
                    self._pending_migration_ids.pop(cam_id, None)
                    if winner_id:
                        rescue_cleanups.append((cam_id, winner_id))

        for cam_id, winner_id in stale_cleanups:
            with self._lock:
                self._peer_inflight[winner_id] = max(
                    0, self._peer_inflight.get(winner_id, 0) - 1
                )
            logger.warning(
                "[PeerOrch] Cleaned up stale pending migration for camera '%s' (winner '%s') after timeout.",
                cam_id, winner_id,
            )

        for cam_id, winner_id in rescue_cleanups:
            with self._lock:
                self._peer_inflight[winner_id] = max(
                    0, self._peer_inflight.get(winner_id, 0) - 1
                )
            logger.warning(
                "[PeerOrch] Cleaned up stale rescue-ack for camera '%s' (winner '%s') after timeout.",
                cam_id, winner_id,
            )

    def _compute_stream_pressure(self, state, cfg: dict) -> float:
        """
        Offload pressure signal in [0.0, 1.0] from workload/resource
        observables (ADR-0001).

        Crop offload (L2 vehicle-crop tier) was historically justified by ``load_score`` alone,
        which does NOT prove that decode/tracking pressure is relieved — the
        premise behind its removal.  This signal is computed ONLY from observed
        demand/resource observables (existing PeerState fields):

          * Resource pressure:
                max(saturate(gpu), saturate(cpu), saturate(ram))
            GPU is the primary saturation signal (evidence: GPU saturates while
            FPS stays high, so resource, not FPS, is the genuine saturation
            signal).
          * Demand/workload pressure:
                sum(camera_workload) / workload_saturation_point
            per-camera vehicle workload (n_track + n_plate) normalized to the
            provisional saturation point.
          * Source-starvation ratio:
                |starved ∩ active| / max(1, |active|)
            starved cameras are reported by the health agent when a source
            cannot keep the decoder fed — a direct decode-pressure symptom
            (ADR-0001: starvation may remain a direct symptom).

        FPS is NOT a primary driver: a high-FPS but resource-saturated node
        still yields pressure, and a low-FPS-but-idle node yields none.

        FAILS CLOSED: returns ``0.0`` (no trustworthy pressure) ONLY when
        resource telemetry is invalid — never merely because FPS is absent.
        """
        # Fail closed only when resource telemetry is invalid.
        if not _resource_telemetry_valid(state):
            return 0.0

        res_p = max(
            _saturate(state.gpu_percent),
            _saturate(state.cpu_percent),
            _saturate(state.ram_percent),
        )

        wl = state.camera_workload or {}
        sat = self._workload_saturation_point(cfg)
        total_wl = sum(
            v for v in wl.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
        )
        demand_p = max(0.0, min(1.0, total_wl / max(0.001, sat)))

        active = state.active_cameras or []
        starved = [c for c in (state.source_starved_cameras or []) if c in active]
        starve_p = len(starved) / max(1, len(active))

        return max(res_p, demand_p, starve_p)

    def _workload_saturation_point(self, cfg: dict) -> float:
        """Demand-pressure saturation point (sum n_track+n_plate).

        Default aligned with shipped edge_node.yml workload_saturation_point (10.0).
        """
        return float(cfg.get("workload_saturation_point", 10.0)) or 10.0

    def _stream_pressure_threshold(self, cfg: dict) -> float:
        """Minimum workload/resource pressure to authorize L1 migration.

        Default aligned with shipped edge_node.yml stream_pressure_threshold (0.30).
        """
        return float(cfg.get("stream_pressure_threshold", 0.30))

    def _check_self_overload(self) -> None:
        """
        Two orthogonal relief mechanisms are triggered from the overloaded
        state, and they must not be conflated:

        * L1 — Stream Lease Transfer: the full stream (decode + tracking +
          inference + the resources that come with it) is migrated to a peer
          via the existing RFO / lease path. This is the only mechanism that
          relieves decode/tracking/resource pressure, since the source stream
          is fully handed off.
        * LPR — LPR Queue Relief: only the plate-crop work (LPR inference on
          crops) is offloaded to a peer; the decode/tracking stream stays
          local. This relieves the LPR crop queue, not the decode/tracking load.

        Crop offload (L2 vehicle-crop tier) was removed by the P3 redesign — it cannot
        relieve decode/tracking pressure and caused receiver drops/oscillation.
        """
        cfg   = self._cfg
        now   = time.time()
        # BUG-1 fix: snapshot _self_state under lock so we work with a
        # consistent view for the entire overload-check decision.
        with self._self_lock:
            state = PeerState(
                node_id=self._self_state.node_id,
                load_score=self._self_state.load_score,
                gpu_percent=self._self_state.gpu_percent,
                cpu_percent=self._self_state.cpu_percent,
                ram_percent=self._self_state.ram_percent,
                gpu_temp_c=self._self_state.gpu_temp_c,
                avg_fps=self._self_state.avg_fps,
                fps_per_camera=dict(self._self_state.fps_per_camera),
                active_cameras=list(self._self_state.active_cameras),
                streaming_cameras=list(self._self_state.streaming_cameras),
                held_cameras=list(self._self_state.held_cameras),
                overload_since=self._self_state.overload_since,
                penalty_until=self._self_state.penalty_until,
                risk_index=self._self_state.risk_index,
                camera_workload=dict(self._self_state.camera_workload),
                source_starved_cameras=list(self._self_state.source_starved_cameras),
                last_seen=self._self_state.last_seen,
                status=self._self_state.status,
            )
            self_last_seen = self._self_state.last_seen

        # Stale self-heartbeat guard: if self state has never been received or
        # is stale (last_seen > offline_threshold), skip overload checks.
        # Use same threshold as _check_offline_peers to avoid inconsistent suppression.
        timeout = self._cfg.get("heartbeat_timeout_s", 5.0)
        grace_s = self._cfg.get("failover_grace_s", timeout)
        offline_threshold = timeout + grace_s
        if self_last_seen == 0.0 or (now - self_last_seen > offline_threshold):
            if self._maybe_log_block("stale_self_heartbeat", now):
                logger.warning(
                    "[PeerOrch] Self heartbeat stale or missing (age=%.1fs > %.1fs). "
                    "Skipping overload check.",
                    now - self_last_seen, timeout,
                )
            return

        # LPR Queue Relief: plate-crop (LPR) offload only. Runs every tick,
        # independently of node overload — a node can decode/track fine yet
        # saturate the local LPR worker pool (sgie2 removed in Phase 1). This
        # relieves the LPR crop queue, NOT the decode/tracking load; the stream
        # lease transfer in _trigger_level1_if_due handles decode/tracking relief.
        self._evaluate_lpr_offload(now, cfg)

        if state.overload_since is None:
            if self._maybe_log_block("not_overloaded", now):
                logger.debug("[PeerOrch] Not overloaded (overload_since=None)")
            return
        if now - state.overload_since < cfg.get("overload_duration_s", 3.0):
            if self._maybe_log_block("overload_too_recent", now):
                logger.debug("[PeerOrch] Overload too recent (%.1fs < %.1fs)",
                            now - state.overload_since, cfg.get("overload_duration_s", 3.0))
            return

        # ponytail: startup warmup gate.  Even with overload_since set,
        # suppress all overload escalation until self has had valid positive
        # FPS for overload_warmup_s seconds (default 10).  Without this, a
        # load_score that spiked before the pipeline stabilised could fire
        # RFO moments after start.  Field is sticky — once valid FPS appears
        # the timer counts from that moment; the gate never goes backwards.
        warmup_s = cfg.get("overload_warmup_s", 10.0)
        first_valid_fps_at = self._self_first_valid_fps_at
        if first_valid_fps_at is None:
            if self._maybe_log_block("warmup_no_fps", now):
                logger.warning(
                    "[PeerOrch] Overload check BLOCKED: startup warmup — "
                    "no valid positive FPS observed yet. No L1/L2 actions."
                )
            return
        if now - first_valid_fps_at < warmup_s:
            if self._maybe_log_block("warmup_active", now):
                logger.warning(
                    "[PeerOrch] Overload check BLOCKED: startup warmup "
                    "(%.1fs since first valid FPS, need %.1fs). No L1/L2 actions.",
                    now - first_valid_fps_at, warmup_s,
                )
            return

        # ── Decision suppression: pending-ack & post-migration settle ──
        # While a make-before-break migration is in flight (pending ack), no
        # L1/L2 action is allowed — stale FPS samples could escalate the
        # node prematurely.  After a migration completes or reclaim ADD is
        # initiated, a configurable settle window (`transition_settle_s`,
        # default 5.0 s) holds all overload decisions so draining samples
        # cannot trigger a second migration before the pipeline stabilises.
        with self._lock:
            has_pending = bool(self._pending_acks)
        if has_pending:
            if self._maybe_log_block("pending_ack", now):
                logger.warning(
                    "[PeerOrch] Overload check BLOCKED: %d pending ack(s) "
                    "still in flight — waiting for migration completion "
                    "before issuing any L1/L2 action.",
                    len(self._pending_acks),
                )
            return

        settle_remaining = self._transition_settle_until - now
        if settle_remaining > 0:
            if self._maybe_log_block("settle", now):
                logger.warning(
                    "[PeerOrch] Overload check BLOCKED: post-migration "
                    "settle window active (%.1f s remaining). No L1/L2 "
                    "actions until settle expires.",
                    settle_remaining,
                )
            return
        # ── End decision suppression ──

        # ── P3 redesign: single stream lease-transfer primitive ──────────
        # Crop offload (L2 vehicle-crop tier) is removed: it cannot relieve decode/tracking
        # pressure and caused receiver drops/oscillation. The ONLY decode/
        # tracking relief is full-stream migration (L2) via the existing RFO /
        # lease machinery. This is distinct from LPR Queue Relief (plate-crop
        # offload, L1), which leaves the stream local and only drains the LPR queue.
        #
        # The single L1 transfer decision is gated by an offload-aware stream
        # pressure signal reflecting actual workload/resource pressure
        # (GPU/CPU/RAM saturation + per-camera demand + source starvation),
        # fired through _trigger_level1_if_due. FPS is NOT a primary driver
        # (ADR-0001).  We require BOTH the existing load-based overload gate
        # (overload_since already set by here) AND stream_pressure, and we FAIL
        # CLOSED only when resource telemetry is invalid (never merely because
        # FPS is absent).
        pressure = self._compute_stream_pressure(state, cfg)
        pressure_thr = self._stream_pressure_threshold(cfg)
        if pressure < pressure_thr:
            if self._maybe_log_block("stream_pressure_low", now):
                logger.warning(
                    "[PeerOrch] Overload check BLOCKED: stream-pressure %.2f < %.2f "
                    "(workload/resource pressure insufficient; resource telemetry valid "
                    "but below threshold — no migration). res(gpu/cpu/ram)=%s/%s/%s "
                    "starved=%s",
                    pressure, pressure_thr,
                    state.gpu_percent, state.cpu_percent, state.ram_percent,
                    state.source_starved_cameras,
                )
            return

        if self._maybe_log_block("overload_decision_check", now):
            logger.debug(
                "[PeerOrch] Overload decision: stream-pressure=%.2f (>=%.2f), "
                "load=%.1f, avg_fps=%s, active=%d — eligible for L1 stream migration",
                pressure, pressure_thr, state.load_score, state.avg_fps,
                len(state.active_cameras),
            )

        # ── Mandatory ladder L0 -> L1 -> L2 ──────────────────────────────
        # Step 1: L1 (plate-crop) must be attempted before L2. Fast-escalate to
        # L2 ONLY when there is nothing to hold (no-candidate). On no-peer /
        # busy the node waits — escalating a saturated or busy mesh into the
        # heavier full-stream migration is never correct (user mandate: full
        # climb is mandatory in normal conditions).
        hold_s = float(cfg.get("ladder_l1_hold_s", 8.0))

        if self._ladder_l1_since is None:
            l1_res = self._activate_ladder_l1(now, cfg)
            if isinstance(l1_res, str) and l1_res.startswith("ok:"):
                l1_cam = l1_res[3:]
                self._ladder_l1_since = now
                self._ladder_l1_camera = l1_cam
                logger.info("[PeerOrch] LADDER L0->L1: escalated '%s' to plate-crop offload; holding %.1fs before L2", l1_cam, hold_s)
                return
            if l1_res == "no-candidate":
                best_peer = self._pick_best_peer(for_offload_level=1)
                if best_peer is None:
                    logger.info(
                        "[PeerOrch] LADDER L1 pending (no-peer), pressure=%.2f (thr=%.2f), deferring L2 this tick",
                        pressure, pressure_thr,
                    )
                    return
                logger.warning("[PeerOrch] LADDER L0->L1 nothing to hold; fast-escalating to L2")
                # Do NOT set _ladder_l1_since here — L1 never activated.
                # Leaving it None allows a later tick to retry L1 after conditions change.
            else:
                # "no-peer" or "busy": wait this tick, do not escalate.
                logger.info("[PeerOrch] LADDER L1 pending (%s), pressure=%.2f (thr=%.2f), deferring L2 this tick", l1_res, pressure, pressure_thr)
                return

        # Step 2: In L1 hold window. Only proceed to L2 if hold window expired (or fast-escalated)
        if self._ladder_l1_camera is not None and (now - self._ladder_l1_since) < hold_s:
            if self._maybe_log_block("ladder_l1_hold", now):
                logger.info("[PeerOrch] LADDER L1 hold: %.1f/%.1fs elapsed, still overloaded — deferring L2", now - self._ladder_l1_since, hold_s)
            return

        # Hold window expired or fast-escalated: proceed to L2.
        # Clear L1 ladder camera before proceeding to L2 to avoid orphaned L1 crops.
        if self._ladder_l1_camera:
            if self.get_offload_level(self._ladder_l1_camera) == 1:
                logger.info(
                    "[PeerOrch] LADDER L1->L2: cleared plate-crop offload on '%s' before L2 migration",
                    self._ladder_l1_camera,
                )
                self.set_offload_level(self._ladder_l1_camera, 0, "")
            self._ladder_l1_camera = None

        self._trigger_level1_if_due(state, now, cfg)

    def _is_peer_ready_for_yield(self, peer: Optional[PeerState], camera_id: str) -> bool:
        """Predicate to check if the original owner is alive, ready, and running the camera.

        Conditions:
          - peer is known and heartbeat is fresh
          - peer status is not waiting/recovering
          - peer has valid positive FPS
          - peer is not overloaded
          - camera_id is in peer.held_cameras
        """
        if peer is None:
            return False
        now = time.time()
        timeout = float(self._cfg.get("heartbeat_timeout_s", 5.0))
        if (now - peer.last_seen) > timeout:
            return False
        peer_fps_valid = _has_valid_positive_fps(peer.fps_per_camera)
        peer_in_waiting = is_waiting_state(peer.fps_per_camera, peer.streaming_cameras, peer.status)
        peer_overloaded = self._is_overloaded(
            peer.load_score,
            peer.risk_index,
            peer.qos_state,
        )
        if not peer_fps_valid or peer_in_waiting or peer_overloaded:
            return False
        if camera_id not in peer.held_cameras:
            return False
        # Also ensure peer has capacity (held_cameras <= max_streams)
        if len(peer.held_cameras) > peer.max_streams:
            return False
        return True

    def _on_failover_claim(self, payload: dict) -> None:
        """Handle incoming rescue claim broadcast on peers/failover/claim.

        Used for contention resolution when multiple nodes detect the same dead peer,
        or for startup preemption claims when the true owner boots up.
        """
        dead_node_id = payload.get("dead_node_id", "")
        camera_id = payload.get("camera_id", "")
        claimer_node_id = payload.get("claimer_node_id") or payload.get("claimer", "")
        priority_weight = payload.get("priority_weight", 0)
        claim_type = payload.get("type", "")
        ts = payload.get("ts", time.time())

        # Check for startup preemption claim: original owner booted and claims its cameras
        if claim_type == "startup_claim" or payload.get("action") == "startup_preempt":
            claimed_cameras = payload.get("cameras", [])
            if not claimed_cameras and camera_id:
                claimed_cameras = [camera_id]
            logger.info(
                "[Failover] Received startup preemption announcement from '%s' for cameras: %s",
                claimer_node_id, claimed_cameras,
            )
            with self._lock:
                peer = self._peers.get(claimer_node_id)
                for cam in claimed_cameras:
                    # Validate static ownership of the claimer
                    claimer_owned = self._get_node_owned_cameras(claimer_node_id)
                    if claimer_owned is not None and cam not in claimer_owned:
                        logger.warning(
                            "[Failover] Rejected startup claim from '%s' for '%s': not statically owned by claimer.",
                            claimer_node_id, cam,
                        )
                        continue

                    # Check if holding dynamically (either in _rescued_cameras, or active on self but not statically owned)
                    self_owned = self._get_owned_camera_ids()
                    if cam in self_owned:
                        # Own home camera — do not yield
                        continue

                    is_rescued = (self._rescued_cameras.get(cam) == claimer_node_id or cam in self._rescued_cameras)
                    with self._self_lock:
                        is_active_local = cam in self._self_state.active_cameras

                    if not is_rescued and not is_active_local:
                        continue

                    if not self._is_peer_ready_for_yield(peer, cam):
                        logger.info(
                            "[Failover] Deferred yield of camera '%s' on startup claim from '%s': owner not ready/active.",
                            cam, claimer_node_id,
                        )
                        continue

                    if not self._l1_remove_ownership_guard(cam):
                        logger.info(
                            "[Failover] Aborted yield of '%s' on startup claim from '%s': last owned camera guard.",
                            cam, claimer_node_id,
                        )
                        continue

                    orig = self._rescued_cameras.pop(cam, None)
                    self._rescued_at.pop(cam, None)
                    remove_cmd = self._build_remove_cmd(cam, context="startup_announcement_yield")
                    if self._pubs.get("control") is not None:
                        self._pubs["control"].put(msgpack.packb(remove_cmd, use_bin_type=True))
                    logger.info(
                        "[Failover] Yielded camera '%s' (orig owner '%s') due to startup announcement from '%s'.",
                        cam, orig or claimer_node_id, claimer_node_id,
                    )
            return

        if not dead_node_id or not camera_id or not claimer_node_id:
            return

        # If the claimer is the original owner claiming its camera back, yield only if ready
        with self._lock:
            if self._rescued_cameras.get(camera_id) == claimer_node_id:
                # Validate static ownership
                claimer_owned = self._get_node_owned_cameras(claimer_node_id)
                if claimer_owned is not None and camera_id not in claimer_owned:
                    logger.info(
                        "[Failover] Rejected claim from '%s' for '%s': not statically owned by claimer.",
                        claimer_node_id, camera_id,
                    )
                    return
                peer = self._peers.get(claimer_node_id)
                if not self._is_peer_ready_for_yield(peer, camera_id):
                    logger.info(
                        "[Failover] Deferred yield of rescued camera '%s' to original owner '%s' via failover_claim: owner not ready/active.",
                        camera_id, claimer_node_id,
                    )
                    return
                self._rescued_cameras.pop(camera_id, None)
                self._rescued_at.pop(camera_id, None)
                remove_cmd = self._build_remove_cmd(camera_id, context="failover_claim_yield")
                if self._pubs.get("control") is not None:
                    self._pubs["control"].put(msgpack.packb(remove_cmd, use_bin_type=True))
                logger.info(
                    "[Failover] Yielded rescued camera '%s' to original owner '%s' via failover_claim.",
                    camera_id, claimer_node_id,
                )
                return

        local_now = time.time()
        with self._claims_lock:
            # Claims are only needed during the claim window / lease period. Bound
            # this cache using configured rescue_claim_lease_s (default 15s) or 4x window.
            # Local receipt time is used for lease freshness to prevent cross-node clock skew.
            claim_lease_s = float(self._cfg.get("rescue_claim_lease_s", max(15.0, float(self._cfg.get("rescue_claim_window_s", 0.5)) * 4.0)))
            cutoff = local_now - claim_lease_s
            self._failover_claims = {
                key: claim for key, claim in self._failover_claims.items()
                if claim[1] >= cutoff
            }
            key = (dead_node_id, camera_id)
            existing = self._failover_claims.get(key)
            if existing is None or priority_weight > existing[2]:
                self._failover_claims[key] = (claimer_node_id, local_now, priority_weight, float(ts))
                logger.info(
                    "[Failover] Recorded rescue claim for '%s' (dead='%s') by '%s' (weight=%d, remote_ts=%.3f)",
                    camera_id, dead_node_id, claimer_node_id, priority_weight, float(ts),
                )

    def _publish_startup_announcement(self) -> None:
        """One-shot startup announcement for statically owned cameras."""
        try:
            owned = list(self._get_owned_camera_ids())
            if not owned:
                return
            now_ts = time.time()
            claim_payload = {
                "type": "startup_claim",
                "action": "startup_preempt",
                "claimer_node_id": self._node_id,
                "cameras": owned,
                "timestamp": now_ts,
                "ts": now_ts,
            }
            if "failover_claim" in self._pubs and self._pubs["failover_claim"] is not None:
                self._pubs["failover_claim"].put(msgpack.packb(claim_payload, use_bin_type=True))
                logger.info(
                    "[PeerOrch] Published startup announcement for owned cameras: %s",
                    owned,
                )
        except Exception as exc:
            logger.warning("[PeerOrch] Failed to publish startup announcement: %s", exc)

    @staticmethod
    def _consistent_hash(camera_id: str, peer_ids: List[str]) -> str:
        """
        Rendezvous / Highest Random Weight (HRW) hashing:
        Computes sha256(camera_id:peer_id) for each candidate peer; highest hash wins.

        Properties over modulo hashing:
          1. Minimal disruption on membership change: when a node leaves/fails,
             only its assigned cameras are reassigned; remaining cameras stay bound.
          2. Uniform distribution across alive peers.
          3. Deterministic across all nodes that see the same peer set.
        """
        if not peer_ids:
            return ""
        # ponytail: HRW rendezvous hash ensures true consistent hashing property
        best_peer = ""
        best_weight = -1
        for pid in sorted(peer_ids):
            combined = f"{camera_id}:{pid}".encode("utf-8")
            weight = int(hashlib.sha256(combined).hexdigest(), 16)
            if weight > best_weight:
                best_weight = weight
                best_peer = pid
        return best_peer

    def _leaderless_failover(self, dead_node_id: str, orphaned_cameras: List[str]) -> None:
        """
        Rescue orphaned cameras using consistent hash.

        Each surviving peer runs independently → same hash result.
        Winner executes ADD after jitter (0-2s) to avoid race.
        After jitter, check peers/status/+ to see if camera already rescued.

        Dead-owner filtering
        ────────────────────
        The dead peer's ``active_cameras`` can contain stale entries:
          * cameras that were being migrated TO the dead peer (in flight when it died)
          * cameras that the dead peer had previously rescued from another peer
          * cameras that were offloaded (L1/L2) to the dead peer

        Rescuing any of those creates duplicate streams — the original owner
        (or another alive peer) is already running them.  The dead peer's
        ``camera_configs`` (populated from its last heartbeat's
        ``pipeline.camera_configs``) is the authoritative "owned by the dead
        node" set: cameras it was configured to manage.  We intersect the
        orphan list with those keys before rescue.  If the dead peer never
        populated ``camera_configs`` (older firmware / missing field), we fall
        back to the full orphan list — never silently drop a mandatory rescue.

Duplicate prevention (local)
        ────────────────────────────
        The consistent hash ensures only one node wins each camera across the
        cluster.  Within this node, we additionally guard against double-rescue:
          * camera already in ``_rescued_cameras`` (rescued in a prior run)
          * camera already in ``_self_state.held_cameras`` (we're running it)
        Both are checked under their respective locks.
        """
        cfg = self._cfg
        now = time.time()
        timeout = cfg.get("heartbeat_timeout_s", 5.0)

        with self._lock:
            if not self._pipeline_ready:
                logger.debug("[Failover] Local pipeline not ready yet. Suppressing leaderless failover execution.")
                return

        cm = self._camera_manager
        default_max = cm.get_max_streams() if cm is not None else 4
        eps_max = int(self._cfg.get("eps_streams_max", default_max))
        # Configurable rescue ceiling, default eps_streams_max - 1 (e.g. 3 when eps_streams_max=4)
        rescue_ceiling = int(self._cfg.get("failover_rescue_max", eps_max - 1))

        # Read dead peer's camera configs AND build alive_peers in a single lock
        # acquisition to prevent torn reads between the two operations.
        with self._lock:
            dead_peer = self._peers.get(dead_node_id)
            peer_cam_configs = dict(dead_peer.camera_configs) if dead_peer else {}

        # Build alive candidate list -- includes self so this node can rescue too
        # BUG-1 fix: read held_cameras from _self_state under _self_lock.
        with self._self_lock:
            self_streams = len(self._self_state.held_cameras)
        with self._lock:
            self_eligible = (
                self._node_id != dead_node_id
                and self_streams < rescue_ceiling
            )
            alive_peers = sorted([
                nid for nid, peer in self._peers.items()
                if nid != dead_node_id
                and now - peer.last_seen <= timeout
                and len(peer.held_cameras) < peer.max_streams
            ] + ([self._node_id] if self_eligible else []))

        # Dead-owner filtering: camera MUST be originally owned by dead_node_id.
        # Check static mapping / dead peer's camera_configs.
        # Under NO circumstance can a node rescue a camera that belongs to itself or an alive peer.
        dead_owned_set = self._get_node_owned_cameras(dead_node_id)
        if dead_owned_set is not None:
            before_n = len(orphaned_cameras)
            orphaned_cameras = [c for c in orphaned_cameras if c in dead_owned_set]
            filtered_out = before_n - len(orphaned_cameras)
            if filtered_out:
                logger.info(
                    "[Failover] Filtered %d non-owned/stale entries from '%s' orphan list "
                    "(only rescue cameras originally owned by '%s').",
                    filtered_out, dead_node_id, dead_node_id,
                )
        elif peer_cam_configs:
            owned_set = set(peer_cam_configs.keys())
            before_n = len(orphaned_cameras)
            orphaned_cameras = [c for c in orphaned_cameras if c in owned_set]
            filtered_out = before_n - len(orphaned_cameras)
            if filtered_out:
                logger.info(
                    "[Failover] Filtered %d stale/non-owned entries from '%s' orphan list "
                    "(active_cameras included offload/migrated-in cameras).",
                    filtered_out, dead_node_id,
                )

        # Strict exclusion: NEVER rescue cameras owned by self or by an alive peer
        self_owned = self._get_owned_camera_ids()
        alive_peer_owned = set()
        with self._lock:
            for nid, p in self._peers.items():
                if nid != dead_node_id and nid != self._node_id and (now - p.last_seen <= timeout):
                    p_owned = self._get_node_owned_cameras(nid)
                    if p_owned:
                        alive_peer_owned.update(p_owned)

        excluded_cams = self_owned | alive_peer_owned
        if excluded_cams:
            before_ex = len(orphaned_cameras)
            orphaned_cameras = [c for c in orphaned_cameras if c not in excluded_cams]
            if len(orphaned_cameras) < before_ex:
                logger.info(
                    "[Failover] Excluded %d cameras belonging to self or alive peers from '%s' rescue.",
                    before_ex - len(orphaned_cameras), dead_node_id,
                )

        if not orphaned_cameras:
            logger.info(
                "[Failover] No valid owned orphans from '%s' to rescue. Skipping.",
                dead_node_id,
            )
            return

        # BUG-11: Track how many cameras this node has accepted during this
        # failover loop so we don't exceed eps_streams_max across iterations.
        self_accepted = 0

        # Single jitter sleep BEFORE the loop to avoid race conditions with
        # other peers, without accumulating delay per camera.
        jitter = random.uniform(0, cfg.get("failover_jitter_max_s", 2.0))
        time.sleep(jitter)

        for camera_id in orphaned_cameras:
            if not alive_peers:
                logger.error("[Failover] No alive peers to rescue '%s'", camera_id)
                continue

            winner = self._consistent_hash(camera_id, alive_peers)

            if winner == self._node_id:
                # Fail-closed admission gate (COLLECTOR_FAILOVER_RECOVERY Phase 2):
                # never rescue while the local node is still in a WAITING/RECOVERY
                # state — no valid positive FPS, empty active cameras, or an
                # explicit waiting/recovering status. Reuses is_waiting_state so
                # the gate matches the existing overload/rebalance semantics. A
                # warming-up or recovering node must not absorb orphan streams
                # until its own pipeline produces valid FPS (valid low-FPS still
                # passes). Placed before the duplicate/capacity/load checks.
                with self._self_lock:
                    _sfps = self._self_state.fps_per_camera
                    _sactive = self._self_state.active_cameras
                    _sstatus = self._self_state.status
                if is_waiting_state(_sfps, _sactive, _sstatus):
                    logger.info(
                        "[Failover] Skip rescue of '%s': local node WAITING/RECOVERY "
                        "(fps=%s, active=%d, status=%s). Failing closed.",
                        camera_id, _sfps, len(_sactive or []), _sstatus,
                    )
                    continue

                # Local-side duplicate guard: skip if WE already rescued this
                # camera in a prior failover run, or if we're already running it or have uncommitted migration.
                with self._lock:
                    if (camera_id in self._rescued_cameras
                            or camera_id in self._pending_acks
                            or camera_id in self._pending_winner
                            or camera_id in self._reclaim_in_progress):
                        logger.info(
                            "[Failover] Camera '%s' in uncommitted state or already in _rescued_cameras (owner='%s'). "
                            "Skipping duplicate rescue.",
                            camera_id, self._rescued_cameras.get(camera_id, "unknown"),
                        )
                        continue
                with self._self_lock:
                    if camera_id in self._self_state.held_cameras:
                        logger.info(
                            "[Failover] Camera '%s' already held on self. Skipping duplicate rescue.",
                            camera_id,
                        )
                        continue

                # Re-check capacity including cameras accepted earlier in this loop against rescue ceiling
                # BUG-1 fix: read held_cameras and load_score under _self_lock
                with self._self_lock:
                    current_streams = len(self._self_state.held_cameras)
                    current_load = self._self_state.load_score
                if current_streams + self_accepted >= rescue_ceiling:
                    logger.warning(
                        "[Failover] Cannot rescue '%s': at rescue ceiling (%d >= %d). Skipping.",
                        camera_id, current_streams + self_accepted, rescue_ceiling,
                    )
                    continue

                # Capacity load-score gate: do not rescue if node is already near/above overload
                overload_thresh = self._cfg.get("overload_threshold", 55.0)
                if current_load >= overload_thresh:
                    logger.warning(
                        "[Failover] Cannot rescue '%s': self load_score (%.1f) >= overload threshold (%.1f). Skipping.",
                        camera_id, current_load, overload_thresh,
                    )
                    continue

                # Double-check: camera already rescued by another peer?
                # Exclude both self AND the dead peer — the dead peer's stale
                # held_cameras still lists its own cameras and would otherwise
                # cause every rescue to be skipped in a 2-node cluster.
                with self._lock:
                    already_handled = any(
                        camera_id in peer.held_cameras
                        for nid, peer in self._peers.items()
                        if nid != self._node_id and nid != dead_node_id
                    )
                if already_handled:
                    logger.info(
                        "[Failover] Camera '%s' already rescued. Skipping.",
                        camera_id,
                    )
                    continue

                # Use camera config from the dead peer's heartbeat (original URIs),
                # fall back to local cameras.yml only if peer didn't share configs.
                cam_config = peer_cam_configs.get(camera_id) or self._get_camera_config(camera_id)
                if cam_config is None:
                    logger.error("[Failover] No config for '%s'. Skipping.", camera_id)
                    continue

                # Fix E: Broadcast rescue claim to resolve membership-divergence split-brain
                # Check if a fresh higher-priority claim lease already exists before publishing/claiming
                claim_lease_s = float(cfg.get("rescue_claim_lease_s", 15.0))
                with self._claims_lock:
                    existing_claim = self._failover_claims.get((dead_node_id, camera_id))
                    if (existing_claim is not None
                            and existing_claim[0] != self._node_id
                            and (time.time() - existing_claim[1]) < claim_lease_s):
                        logger.info(
                            "[Failover] Active rescue claim lease exists for '%s' by '%s' (age=%.1fs); skipping local claim.",
                            camera_id, existing_claim[0], time.time() - existing_claim[1],
                        )
                        continue

                # ponytail: mask to 63-bit int so msgpack integer serialization never overflows
                my_weight = int(hashlib.sha256(f"{camera_id}:{self._node_id}".encode()).hexdigest()[:15], 16)
                claim_now = time.time()
                claim_payload = {
                    "dead_node_id": dead_node_id,
                    "camera_id": camera_id,
                    "claimer_node_id": self._node_id,
                    "priority_weight": my_weight,
                    "timestamp": claim_now,
                    "ts": claim_now,
                }
                if "failover_claim" in self._pubs:
                    try:
                        self._pubs["failover_claim"].put(msgpack.packb(claim_payload, use_bin_type=True))
                    except Exception as e:
                        logger.warning("[Failover] Could not publish rescue claim: %s", e)

                # Record own claim locally
                with self._claims_lock:
                    self._failover_claims[(dead_node_id, camera_id)] = (self._node_id, claim_now, my_weight, claim_now)

                # Wait configured claim window to collect peer rescue claims
                claim_window_s = float(cfg.get("rescue_claim_window_s", 0.5))
                if claim_window_s > 0:
                    time.sleep(claim_window_s)

                # Check if a peer claimed with higher weight during the window
                with self._claims_lock:
                    best_claim = self._failover_claims.get((dead_node_id, camera_id))
                    if best_claim is not None and best_claim[0] != self._node_id and best_claim[2] > my_weight:
                        logger.info(
                            "[Failover] Yielding rescue of '%s' to '%s' (higher weight %d > %d)",
                            camera_id, best_claim[0], best_claim[2], my_weight,
                        )
                        continue

                # Stale holder guard before rescue ADD: verify dead holder is still stale
                # under current freshness rules (heartbeat timeout + grace) to avoid acting on transient missed heartbeat
                now_check = time.time()
                timeout = float(self._cfg.get("heartbeat_timeout_s", 5.0))
                grace_s = float(self._cfg.get("failover_grace_s", timeout))
                offline_threshold = timeout + grace_s
                with self._lock:
                    dead_peer_state = self._peers.get(dead_node_id)
                    if dead_peer_state is not None and (now_check - dead_peer_state.last_seen) <= offline_threshold:
                        logger.info(
                            "[Failover] Aborted rescue ADD of '%s': dead holder '%s' is no longer stale (last_seen %.1fs ago <= %.1fs threshold).",
                            camera_id, dead_node_id, now_check - dead_peer_state.last_seen, offline_threshold,
                        )
                        continue

                # P2: per-camera source-liveness gating. Verify the camera
                # SOURCE can actually produce/accept a stream BEFORE committing
                # the rescue ADD. We distinguish SOURCE_UNREACHABLE (definitively
                # down) from rescue-pending (still probing / backoff in effect)
                # and never ADD on either — this is what prevents rescue
                # ADD/REMOVE loops on a flaky or dead source.
                cam_uri = cam_config.get("uri", "")
                src_status = self._probe_source_liveness(camera_id, cam_uri)
                if src_status != "reachable":
                    logger.info(
                        "[Failover] Camera '%s' source not rescue-ready (status=%s, uri=%s). "
                        "Skipping ADD; will retry with backoff.",
                        camera_id, src_status, cam_uri,
                    )
                    continue

                # Alive-peer ownership guard: another surviving node may have
                # rescued this camera in an earlier round whose claim lease has
                # since expired. Never ADD a camera an alive peer currently
                # reports owning.
                alive_owner = None
                with self._lock:
                    for other_id, other_state in self._peers.items():
                        if other_id == dead_node_id:
                            continue
                        if ((time.time() - other_state.last_seen) <= timeout
                                and camera_id in other_state.held_cameras):
                            alive_owner = other_id
                            break
                if alive_owner is not None:
                    logger.info(
                        "[Failover] Camera '%s' already handled by alive peer '%s'. Skipping.",
                        camera_id, alive_owner,
                    )
                    continue

                # Re-check local active ownership immediately before publishing failover ADD
                with self._self_lock:
                    if camera_id in self._self_state.held_cameras:
                        logger.info(
                            "[Failover] Camera '%s' became held locally before publishing. Skipping.",
                            camera_id,
                        )
                        continue

                add_cmd = {**cam_config, "cmd": "ADD"}
                # P5: stamp THIS node's boot_id so a pre-reboot rescue ADD is fenced.
                if getattr(self, "_boot_id", 0):
                    add_cmd["boot_id"] = self._boot_id
                self._pubs["control"].put(msgpack.packb(add_cmd, use_bin_type=True))
                self_accepted += 1
                with self._lock:
                    self._rescued_cameras[camera_id] = dead_node_id
                    self._rescued_at[camera_id] = time.time()
                # ponytail: rescue ADD — record so _pick_camera_to_offload
                # applies the per-camera warmup gate to this camera too.
                self._camera_added_at[camera_id] = time.time()
                self._camera_first_valid_fps_at.pop(camera_id, None)
                logger.info("[Failover] Rescue ADD sent: '%s' → me (source reachable)", camera_id)

                self._migration_log.log(
                    dead_node_id, self._node_id, camera_id,
                    "node_offline", 0.0, 0.0,
                    0.0, "FAILOVER_ADD",
                )

        # A completed failover must not retain one claim per rescued camera.
        # Keep only claims that may still be observed by a peer in flight.
        with self._claims_lock:
            for camera_id in orphaned_cameras:
                self._failover_claims.pop((dead_node_id, camera_id), None)

    def _get_owned_camera_ids(self) -> set:
        """
        Return the set of camera IDs this node is configured to own.

        Ownership is resolved in order:
          1. Static node_camera_map in config: if configured for self._node_id,
             this static assignment strictly defines the node's owned cameras.
          2. Live CameraManager: enabled static cameras (is_dynamic=False).
          3. Fallback cameras.yml on disk.

        Ownership = cameras enabled in this node's local cameras.yml or node_camera_map.
        Rescued and migrated-in cameras are NOT owned by this node.
        """
        # Highest priority: static node_camera_map configured for this node
        node_cam_map = self._cfg.get("node_camera_map")
        if isinstance(node_cam_map, dict) and self._node_id in node_cam_map:
            cams = node_cam_map.get(self._node_id)
            if isinstance(cams, (list, tuple, set)):
                return set(cams)

        # Fast path: live CameraManager (hot-reloaded via inotify)
        cm = self._camera_manager
        if cm is not None:
            try:
                with cm._lock:
                    return {
                        c.camera_id for c in cm._configs.values()
                        if getattr(c, "enabled", False) and not getattr(c, "is_dynamic", False)
                    }
            except Exception as exc:
                logger.debug(
                    "[PeerOrch] CameraManager ownership lookup failed: %s", exc
                )

        # Fallback: cameras.yml with mtime-based cache invalidation.
        # Mirrors the same pattern used in _get_camera_config so reloads
        # are visible without restarting the orchestrator.
        try:
            import yaml
            yml_path = self._camera_configs_dir / "cameras.yml"
            try:
                current_mtime = yml_path.stat().st_mtime
            except OSError:
                current_mtime = 0.0

            if (self._cameras_cache is None
                    or getattr(self, "_cameras_cache_mtime", None) != current_mtime):
                with open(yml_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                self._cameras_cache = (raw or {}).get("cameras", {}) or {}
                self._cameras_cache_mtime = current_mtime

            return {
                cam_id for cam_id, cfg in (self._cameras_cache or {}).items()
                if isinstance(cfg, dict) and bool(cfg.get("enabled", True))
            }
        except Exception as exc:
            logger.debug(
                "[PeerOrch] cameras.yml ownership lookup failed: %s", exc
            )
            return set()

    def _get_node_owned_cameras(self, target_node_id: str) -> Optional[set]:
        """
        Return the set of cameras configured to be owned by target_node_id.
        Checks static node_camera_map in config first, then target peer's camera_configs.
        Returns None if ownership cannot be determined.
        """
        node_cam_map = self._cfg.get("node_camera_map")
        if isinstance(node_cam_map, dict) and target_node_id in node_cam_map:
            cams = node_cam_map.get(target_node_id)
            if isinstance(cams, (list, tuple, set)):
                return set(cams)

        with self._lock:
            peer = self._peers.get(target_node_id)
            if peer and peer.camera_configs:
                return set(peer.camera_configs.keys())
        return None
