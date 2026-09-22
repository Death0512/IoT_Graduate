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
from .settings import ROOT as _ROOT, LOG_LEVEL
from .membership import (
    PeerState,
    MigrationLogger,
    logger,
    _setup_logging,
    _parse_camera_workload,
    _parse_starved_cameras,
    _pick_fps_dict,
    _has_valid_positive_fps,
    _has_valid_or_unreported_fps,
    is_waiting_state,
    _dwell_s,
    _thermal_admission_ok,
)

"""
Edge/speedflow_python/offload.py — Offload Ladder L0/L1/L2 Mixin

OffloadMixin — Implements the three-tier offload ladder for PeerOrchestrator.
Mixed into peer_orchestrator.PeerOrchestrator.

Offload Levels (verify_offload_contract.py canonical enum):
  L0 = 0: Local processing — full pipeline runs on this node
  L1 = 1: Plate-crop offload — this node keeps stream, ships plate crops to peer for LPR
  L2 = 2: Full-stream migration (RFO) — stream moves to peer via make-before-break

Ladder Transitions (sustained gates, edge_node.yml):
  L0 → L1: load_score ≥ 60 sustained overload_duration_s=5s; clears < 55 (5pt deadband)
  L1 → L2: load_score ≥ 60 + stream_pressure ≥ 0.30 + thermal/capacity ε-checks
  L2 → L1/L0: Reclaim when load < 40 (60-20 margin) stable reclaim_stable_s=12s

Key Methods:
- get_offload_level(): Lock-free fast path (called per-frame from SpeedProbe)
- _pick_camera_for_lpr_offload(): Selects camera for plate-crop based on
  n_track + n_plate (workload evidence, NOT output FPS)
- _pick_camera_for_rfo(): Selects camera for full-stream RFO using HRW hash
  (sha256(camera:peer) highest wins) — deterministic, minimal disruption.
- set_lpr_queue_ratio(): Plate queue relief mechanism (lpr_enter_thr=0.20,
  lpr_exit_thr=0.08, sustain 1.5s)
- _update_offload_table(): Applied after migration COMMIT; sets new levels.

HSPW (Highest-Sustainable-Parallel-Workload) model:
  Predicts load_after = base_load + Σ(peer_workload) for bid evaluation.
  Refuses bid if predicted load_after ≥ hw_fuse_threshold (90).

Dependencies:
- membership.py — PeerState, HRW hash, ladder thresholds, thermal checks
- zenoh_publisher.py — OffloadPublisher for plate-crop wire payloads
- offload_receiver.py — OffloadReceiver for crop ingestion + worker pool
"""


class OffloadMixin:
    def get_offload_level(self, camera_id: str) -> int:
        """
        Return the current offload level for camera_id (0, 1, or 2).

        0 = local processing; 1 = plate-crop offload (L1; this node still owns
        the stream but ships plate crops to a peer for LPR). 2 = full-stream
        migration in flight (RFO / lease path).
        Set via set_offload_level; plate-crop selection lives in
        _pick_camera_for_lpr_offload. Called from
        SpeedProbe on every frame — must be lock-free fast. Uses a separate
        RLock from the main _lock to avoid priority inversion with the Zenoh
        callback thread.
        """
        with self._offload_lock:
            return self._offload_table.get(camera_id, 0)

    def get_offload_target(self, camera_id: str) -> str:
        """Return the node_id of the offload peer for camera_id, or '' if none."""
        with self._offload_lock:
            return self._offload_targets.get(camera_id, "")

    def is_offload_target_saturated(self, target_node: str) -> bool:
        """Check if target node has reported a saturated offload queue in its recent heartbeat."""
        if not target_node:
            return False
        with self._lock:
            peer = self._peers.get(target_node)
            if peer is not None:
                return bool(peer.offload_queue_full)
            return False

    def set_offload_level(self, camera_id: str, level: int, target_node: str = "") -> None:
        """
        Record the current offload state for camera_id. Called only from the
        decision loop.

        Levels:
          0 = local processing.
1 = L1 plate-crop offload: this node still owns the stream but
            ships plate crops to `target_node` for LPR. Triggered by local
            LPR worker queue saturation, reclaimed when it drains.
        2 = full-stream migration in flight (RFO / lease path).
        """
        with self._offload_lock:
            old = self._offload_table.get(camera_id, 0)
            old_target = self._offload_targets.get(camera_id, "")
            self._offload_table[camera_id] = level
            self._offload_targets[camera_id] = target_node
            now = time.time()
            self._offload_level_changed_at[camera_id] = now
            if level > 0 and old == 0:
                self._offload_started_at[camera_id] = now

        if old != level or (level > 0 and old_target != target_node):
            logger.info(
                "[PeerOrch] Offload level %d→%d for '%s' (target='%s' was='%s')",
                old, level, camera_id, target_node, old_target,
            )

    def set_lpr_queue_ratio(self, ratio: float, depth: int = 0) -> None:
        """Feed the node-local LPR worker queue saturation (0.0..1.0).

        Pushed from SpeedProbe's telemetry snapshot (offload_crops.lpr_queue_ratio)
        so the orchestrator can escalate plate-crop work to a peer when the local
        LPR worker pool saturates.  Defensive: clamps to [0.0, 1.0].
        """
        try:
            r = float(ratio)
        except Exception:
            r = 0.0
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        self._lpr_queue_ratio = r
        self._lpr_queue_depth = max(0, int(depth or 0))

    def _trigger_level1_if_due(self, state, now: float, cfg: dict) -> None:
        """L2 Stream Lease Transfer trigger (the decode/tracking relief path).

        Triggers full-stream migration via the RFO / lease machinery when the
        node is overloaded AND stream pressure is sufficient. This is the ONLY
        mechanism that relieves decode/tracking/resource pressure (the stream
        is fully handed to a peer). It is orthogonal to LPR Queue Relief, which
        only drains the plate-crop queue and leaves the stream local.
        NOTE: `state` is already a consistent snapshot captured by _check_self_overload.
        """
        # If ANY camera already has a vote in progress, wait for its outcome
        # before triggering another RFO. This prevents migrating more cameras
        # than needed — one migration at a time until load drops.
        with self._lock:
            if self._vote_in_progress:
                logger.debug("[PeerOrch] Vote already in progress for %s, skipping re-trigger",
                             self._vote_in_progress)
                return

        cam_to_offload = self._pick_camera_to_offload(state, level=2)
        if not cam_to_offload:
            logger.debug("[PeerOrch] No camera to offload (all inactive or locked)")
            return

        # Failover rescue guard: never L2-migrate a camera currently held as rescued.
        if cam_to_offload in self._rescued_cameras:
            logger.debug("[PeerOrch] '%s' is a rescued camera; cannot be L2-migrated", cam_to_offload)
            return

        # Ladder: if this camera is currently at L2 (plate-crop, offload_level==1),
        # clear it before L1->L2 full-stream migration to avoid double-offloading.
        if self.get_offload_level(cam_to_offload) == 1:
            self.set_offload_level(cam_to_offload, 0, "")
            logger.info("[PeerOrch] LADDER L1->L2: cleared plate-crop offload on '%s' before full-stream migration", cam_to_offload)

        last_mig = self._cam_cooldown.get(cam_to_offload, 0.0)
        time_since_mig = now - last_mig
        if time_since_mig < cfg.get("cooldown_s", 6.0):
            logger.debug("[PeerOrch] Cooldown not met for '%s' (%.1fs / %.1fs)",
                        cam_to_offload, time_since_mig, cfg.get("cooldown_s", 6.0))
            return
        trigger_reason = (
            "load_score+stream_pressure"
            + (";fps_witness_low" if (state.avg_fps and state.avg_fps < cfg.get("eps_fps_strict", 18.0)) else "")
        )
        logger.info(
            "[PeerOrch] OVERLOADED (%.1f%%, FPS=%s). Triggering RFO for '%s' (reason: %s)",
            state.load_score, state.avg_fps, cam_to_offload, trigger_reason,
        )
        # Fast-escalate marker: an L1 RFO is now in flight on behalf of an
        # unavailable L2 ladder — timestamp it so the ladder state reflects an
        # active migration (never set when no candidate/RFO fires, leaving the
        # wedge clear for a retry).
        self._ladder_l1_since = now
        self._trigger_rfo(cam_to_offload, relaxation_tier=0)

    def _pick_best_peer(self, for_offload_level: int = 1) -> Optional[str]:
        """
        Return the node_id of the alive peer with the lowest load score,
        subject to not being in cooldown. Used for plate-crop offload
        peer selection (source offload_level==1). Stream capacity and
        thermal admission are enforced. Full-stream migration (level 2)
        uses the lease/RFO lane and does not call this selector.
        Returns None if no suitable peer is found.

        ``for_offload_level`` is retained for signature compatibility; only
        the plate-crop selection behaviour exists.
        """
        now = time.time()
        timeout = self._cfg.get("heartbeat_timeout_s", 5.0)

        # Thermal admission gate: reject peers with unsafe or ambiguous
        # temperatures before evaluating load score. Configurable via
        # Edge/configs/edge_node.yml p2p.thermal section.
        # Reuses the shared _thermal_admission_ok so sender and receiver
        # rules cannot drift.
        # Allow policy override for baseline comparison experiments (Finding 2.5)
        policy_name = self._cfg.get("policy", "p2p_pareto")
        if policy_name == "no_offload":
            return None

        therm_cfg = self._cfg.get("thermal")
        best_id : Optional[str] = None
        best_load = float("inf")
        best_workload_ema = float("inf")

        with self._lock:
            # Baseline: Round-Robin offload policy
            if policy_name == "round_robin":
                eligible = [
                    nid for nid, peer in sorted(self._peers.items())
                    if nid != self._node_id
                    and (now - peer.last_seen <= timeout)
                    and not getattr(peer, "offload_queue_full", False)
                ]
                if not eligible:
                    return None
                rr_counter = self._rr_counter
                selected = eligible[rr_counter % len(eligible)]
                self._rr_counter = rr_counter + 1
                return selected

            zombie_timeout_count = int(self._cfg.get("zombie_timeout_count", 3))

            for nid, peer in self._peers.items():
                if nid == self._node_id:
                    continue
                if now - peer.last_seen > timeout:
                    continue
                # Plate-crop admission: admit below the overload threshold
                # directly. Crops add negligible pipeline load (only LPR
                # inference, ~1-3%), so no full-stream headroom delta applies
                # here; receiver-side LPR queue-full backpressure is the
                # operative guard.
                threshold = self._cfg.get("overload_threshold", 60.0)
                if peer.load_score >= threshold:
                    continue
                if self._peer_consecutive_timeouts.get(nid, 0) >= zombie_timeout_count:
                    continue
                # Skip peers in startup/recovery or without valid positive FPS (e.g. load_score=0 placeholder)
                if is_waiting_state(peer.fps_per_camera, peer.streaming_cameras, getattr(peer, "status", None)):
                    continue
                # Exclude peers whose offload queue is full to avoid routing to saturated receivers
                if getattr(peer, "offload_queue_full", False):
                    continue
                # For pure least-load greedy baseline, skip stream capacity and thermal admission gates
                if policy_name not in ("least_load_greedy", "centralized_greedy"):
                    if (
                        len(peer.held_cameras) + self._peer_inflight.get(nid, 0)
                        >= peer.max_streams
                    ):
                        continue
                    if not _thermal_admission_ok(peer.gpu_temp_c, therm_cfg):
                        continue
                    if now < peer.penalty_until:
                        continue

                # This selector serves L1 plate-crop offload only. Full-stream
                # migration (Level 2) uses the lease/RFO lane and never reaches
                # here. Peer selection is by pipeline load_score.
                candidate_score = peer.load_score

                peer_wl = peer.workload_ema if (peer.workload_ema is not None and math.isfinite(peer.workload_ema)) else float("inf")
                if candidate_score < best_load:
                    best_load = candidate_score
                    best_workload_ema = peer_wl
                    best_id   = nid
                elif abs(candidate_score - best_load) < 1e-6:
                    # Tiebreaker: workload_ema only
                    if peer_wl < best_workload_ema:
                        best_workload_ema = peer_wl
                        best_id = nid

        return best_id

    def _pick_camera_for_lpr_offload(self, cfg: dict) -> Optional[str]:
        """
        Phase 3: choose a local (Level 0) camera whose plate-crop work should be
        offloaded to a peer at L2 (source offload_level==1) when the local LPR worker queue saturates.

        Prefers the heaviest local camera by workload (most LPR pressure), never
        the last held camera (ownership invariant).  Returns None if no eligible
        camera exists — fail safe, never rank on missing evidence.
        """
        with self._self_lock:
            state = self._self_state
            held = list(state.held_cameras)
            workload = dict(state.camera_workload or {})
        with self._offload_lock:
            table = dict(self._offload_table)

        eligible = [
            c for c in held
            if table.get(c, 0) == 0
            and c in workload
            and isinstance(workload[c], (int, float))
            and not isinstance(workload[c], bool)
            and math.isfinite(workload[c])
        ]
        if len(held) <= 1:
            return None  # never offload the last camera
        if not eligible:
            return None
        # Heaviest local camera first — it generates the most plate crops.
        eligible.sort(key=lambda c: workload[c], reverse=True)
        return eligible[0]

    def _evaluate_lpr_offload(self, now: float, cfg: dict) -> None:
        """
        LPR Queue Relief — plate-crop offload driven by local LPR queue saturation.

        This is the LPR Queue Relief mechanism (distinct from L1 Stream Lease
        Transfer): only the plate-crop work (LPR inference on crops) is moved to
        a peer; the decode/tracking stream stays local, so it does NOT relieve
        decode/tracking pressure. The legacy "Level 3" tag (source
        offload_level==1) refers to this L1 plate-crop offload tier.

        Independent of node overload: a node can decode/track fine yet saturate
        the LPR worker pool (sgie2 was removed in Phase 1, so plate inference now
        runs on the LocalLprWorker pool).  When the worker queue saturates we
        move one camera's plate-crop work to a peer (L1, source offload_level==1); when it drains we
        reclaim that camera back to fully-local (Level 0).

        Thresholds are configurable:
          lpr_offload_up_threshold   (default 0.20) — saturate → escalate
          lpr_offload_down_threshold (default 0.08) — drained  → reclaim
          lpr_offload_sustain_s      (default 1.5)  — ratio must stay > up_thr this long
          lpr_offload_reclaim_cooldown_s (default 5.0) — wait this long after a reclaim
          lpr_offload_cooldown_s     (default 6.0)
        """
        ratio = getattr(self, "_lpr_queue_ratio", 0.0)
        depth = getattr(self, "_lpr_queue_depth", 0)
        up_thr = float(cfg.get("lpr_offload_up_threshold", 0.20))
        down_thr = float(cfg.get("lpr_offload_down_threshold", 0.08))
        up_depth = int(cfg.get("lpr_offload_up_depth", 5))
        down_depth = int(cfg.get("lpr_offload_down_depth", 1))
        sustain_s = float(cfg.get("lpr_offload_sustain_s", 1.5))
        reclaim_cooldown_s = float(cfg.get("lpr_offload_reclaim_cooldown_s", 5.0))
        cooldown_s = float(cfg.get("lpr_offload_cooldown_s", 6.0))
        # ponytail: simple per-camera sustain/reclaim timers; good enough to kill
        # flapping. If hysteresis needs to be adaptive later, track ratio EMA too.

        is_saturated = (ratio >= up_thr) or (depth >= up_depth)
        if is_saturated:
            if self._lpr_over_thr_since is None:
                self._lpr_over_thr_since = now
                logger.info(
                    "[PeerOrch] LPR SATURATED depth=%d ratio=%.2f (up_depth=%d up_thr=%.2f)",
                    depth, ratio, up_depth, up_thr,
                )
        else:
            if self._lpr_over_thr_since is not None:
                logger.debug(
                    "[PeerOrch] LPR desaturated — sustain timer reset (depth=%d ratio=%.2f)",
                    depth, ratio,
                )
            self._lpr_over_thr_since = None

        sustained = (
            is_saturated
            and self._lpr_over_thr_since is not None
            and (now - self._lpr_over_thr_since) >= sustain_s
        )
        if sustained and not self._lpr_sustained_logged:
            self._lpr_sustained_logged = True
            logger.info(
                "[PeerOrch] LPR SUSTAINED %.1fs — escalating L1 plate-crop",
                now - self._lpr_over_thr_since,
            )
        elif not sustained:
            self._lpr_sustained_logged = False

        # Ladder-hold lock: while the mandatory L0->L1 hold window is active for the
        # ladder-held camera, the local queue drains because work moved to the peer —
        # that must not trigger reclaim, or engage/reclaim flaps and the hold never
        # survives to the L2 decision. Overload-clear resets _ladder_l1_since, which
        # releases this lock, so no stuck L1 is possible; the 12s bound keeps it safe.
        hold_s = float(cfg.get("ladder_l1_hold_s", 8.0))
        with self._offload_lock:
            table = dict(self._offload_table)
            targets = dict(self._offload_targets)

        # Reclaim: any L1 (source offload_level==1) camera whose queue drained (or peer lost) → local.
        is_drained = (ratio < down_thr) and (depth <= down_depth)
        for cam_id, lvl in table.items():
            if lvl != 1:
                continue
            held_cam = getattr(self, "_ladder_l1_camera", None)
            held_since = getattr(self, "_ladder_l1_since", None)
            if held_cam is not None and cam_id == held_cam and held_since is not None and (now - held_since) < hold_s:
                continue
            peer = targets.get(cam_id, "")
            if is_drained or not peer or self.is_offload_target_saturated(peer):
                self.set_offload_level(cam_id, 0, "")
                self._lpr_reclaim_at[cam_id] = now + reclaim_cooldown_s
                logger.info(
                    "[PeerOrch] LPR offload RECLAIM '%s' (ratio=%.2f, depth=%d, peer='%s')",
                    cam_id, ratio, depth, peer,
                )
                continue
            if not sustained:
                continue

            if now < self._lpr_reclaim_at.get(cam_id, 0.0):
                logger.info(
                    "[PeerOrch] L1 escalation deferred: cam='%s' in post-reclaim cooldown",
                    cam_id,
                )
                continue  # still in post-reclaim cooldown
            if (now - self._cam_cooldown.get(cam_id, 0.0)) < cooldown_s:
                logger.info(
                    "[PeerOrch] L1 escalation deferred: cam='%s' in cam cooldown",
                    cam_id,
                )
                continue
            peer = self._pick_best_peer(for_offload_level=1)
            if peer is None:
                logger.info(
                    "[PeerOrch] L1 escalation stopped: no eligible peer (plate-crop admission)"
                )
                continue
            self.set_offload_level(cam_id, 1, peer)
            logger.info(
                "[PeerOrch] L1 ENGAGED cam='%s' peer='%s' depth=%d ratio=%.2f",
                cam_id, peer, depth, ratio,
            )

        # Fresh escalation: sustained saturation with no camera currently at L1 → promote one Level-0 camera.
        any_l1 = any(lvl == 1 for lvl in table.values())
        if sustained and not any_l1:
            with self._lock:
                _pending = bool(getattr(self, "_pending_acks", {}))
            if _pending:
                logger.info("[PeerOrch] L1 escalation deferred: migration in flight (pending acks)")
                return
            fresh = self._pick_camera_for_lpr_offload(cfg)
            if fresh is None:
                logger.info("[PeerOrch] L1 escalation stopped: no L0 camera candidate")
                return
            if now < self._lpr_reclaim_at.get(fresh, 0.0):
                logger.info("[PeerOrch] L1 escalation deferred: cam='%s' in post-reclaim cooldown", fresh)
                return
            if (now - self._cam_cooldown.get(fresh, 0.0)) < cooldown_s:
                logger.info("[PeerOrch] L1 escalation deferred: cam='%s' in cam cooldown", fresh)
                return
            fresh_peer = self._pick_best_peer(for_offload_level=1)
            if fresh_peer is None:
                logger.info("[PeerOrch] L1 escalation stopped: no eligible peer (plate-crop admission)")
                return
            self.set_offload_level(fresh, 1, fresh_peer)
            logger.info(
                "[PeerOrch] L1 ENGAGED cam='%s' peer='%s' depth=%d ratio=%.2f (fresh)",
                fresh, fresh_peer, depth, ratio,
            )

        # BUG-E: never pile plate-crop offload onto a camera that is mid
        # stream-migration (or while a migration is in flight on this node) — the
        # L2 full-stream path owns that camera's lifecycle.
        with self._lock:
            has_pending = bool(getattr(self, "_pending_acks", {}))
        if has_pending:
            logger.info(
                "[PeerOrch] L1 escalation deferred: migration in flight (pending acks)"
            )
            return

    def _activate_ladder_l1(self, now: float, cfg: dict) -> str:
        """Escalate a Level-0 camera to L1 plate-crop (offload_level==1)
        for the mandatory ladder, independent of LPR-queue trigger.

        Returns:
            "ok:<cam_id>" on success,
            "no-candidate" when no holdable plate-crop candidate exists,
            "no-peer"      when _pick_best_peer finds no eligible peer,
            "busy"         when pending acks block new offloads.
        """
        with self._lock:
            if bool(getattr(self, "_pending_acks", {})):
                return "busy"
        candidate = self._pick_camera_for_lpr_offload(cfg)
        if candidate is None:
            logger.info(
                "[PeerOrch] L1 escalation stopped: no L0 camera candidate"
            )
            return "no-candidate"
        if candidate in self._rescued_cameras:
            return "no-candidate"
        cooldown_s = float(cfg.get("lpr_offload_cooldown_s", 6.0))
        if (now - self._cam_cooldown.get(candidate, 0.0)) < cooldown_s:
            return "no-candidate"
        peer = self._pick_best_peer(for_offload_level=1)
        if peer is None:
            return "no-peer"
        self.set_offload_level(candidate, 1, peer)
        return f"ok:{candidate}"

    def _trigger_rfo(self, camera_id: str, relaxation_tier: int = 0) -> None:
        """
        Send Request for Offload (RFO) and open vote window.

        relaxation_tier:
          0 = strict (eps_fps_strict, eps_network_ms_strict)
          1 = tier1  (eps_network_ms_tier1)
          2 = tier2  (eps_fps_tier1/2)
        """
        cfg = self._cfg
        eps_fps_map = [
            cfg.get("eps_fps_strict", 18.0),
            cfg.get("eps_fps_tier1", 15.0),
            cfg.get("eps_fps_tier2", 12.0),
        ]
        eps_fps = eps_fps_map[min(relaxation_tier, 2)]
        eps_net = cfg.get("eps_network_ms_strict", 50.0) if relaxation_tier == 0 \
                  else cfg.get("eps_network_ms_tier1", 80.0)

        cam_uri = self._get_camera_uri(camera_id) or ""

        with self._self_lock:
            _rfo_load = self._self_state.load_score
            _rfo_fps  = self._self_state.avg_fps

        with self._lock:
            if camera_id not in self._rfo_snapshots:
                self._rfo_snapshots[camera_id] = (_rfo_load, _rfo_fps)

        now_ts = time.time()
        payload = {
            "requester":      self._node_id,
            "camera_id":      camera_id,
            "cam_uri":        cam_uri,
            "load_score":     _rfo_load,
            "avg_fps":        _rfo_fps,
            "eps_fps":        eps_fps,
            "eps_network_ms": eps_net,
            "tier":           relaxation_tier,
            "timestamp":      now_ts,
            "ts":             now_ts,
        }

        with self._lock:
            self._vote_windows[camera_id] = []
            # Mark this camera as having RFO in progress
            self._vote_in_progress.add(camera_id)

        self._pubs["vote_request"].put(msgpack.packb(payload, use_bin_type=True))
        logger.info("[PeerOrch] RFO sent for '%s' (tier=%d, eps_fps=%.1f, eps_net=%.0fms)",
                    camera_id, relaxation_tier, eps_fps, eps_net)

        # Timer to close vote window
        timer = threading.Timer(
            cfg.get("vote_window_s", 3.0),
            self._close_vote_window,
            args=(camera_id, relaxation_tier),
        )
        with self._lock:
            self._vote_timers[camera_id] = timer
        timer.start()

    def _close_vote_window(self, camera_id: str, relaxation_tier: int) -> None:
        """
        Close vote window, select winner.

        If no proposals → escalate relaxation tier.
        If max tier exhausted with no proposals → log CLUSTER_SATURATED.
        """
        with self._lock:
            proposals = self._vote_windows.pop(camera_id, [])
            self._vote_timers.pop(camera_id, None)
            # Keep _vote_in_progress set until we know we're not escalating,
            # to prevent _check_self_overload from re-triggering in the gap.
            # It will be cleared below if we're not escalating another tier.

        if not proposals:
            if relaxation_tier < 2:
                logger.info(
                    "[PeerOrch] Zero bids for '%s' (tier=%d). Relaxing ε...",
                    camera_id, relaxation_tier,
                )
                # _vote_in_progress stays set; _trigger_rfo will keep it set
                self._trigger_rfo(camera_id, relaxation_tier=relaxation_tier + 1)
            else:
                logger.error(
                    "[PeerOrch] CLUSTER_SATURATED: no peer can accept '%s'. "
                    "Continuing with current load.",
                    camera_id,
                )
                # All tiers exhausted — clear in_progress and set cooldown
                with self._lock:
                    self._vote_in_progress.discard(camera_id)
                    self._rfo_snapshots.pop(camera_id, None)
                self._cam_cooldown[camera_id] = time.time()
                logger.info(
                    "[PeerOrch] Cooldown set for '%s' (%.1fs) to prevent RFO spam",
                    camera_id, self._cfg.get("cooldown_s", 6.0),
                )
            return

        # Winner found — clear in_progress
        with self._lock:
            self._vote_in_progress.discard(camera_id)

        # Winner = lowest-F(x) proposal that passes the zombie/penalty gate.
        # The gate mirrors _pick_best_peer: skip bidders with consecutive
        # migration ACK timeouts >= zombie_timeout_count or inside their
        # post-timeout penalty window. Without this, a low-load peer whose
        # pipeline never completes ADD keeps winning every vote and the fleet
        # churns on timeout/rollback forever (seen live: 11x B->A cam_04).
        zombie_max = int(self._cfg.get("zombie_timeout_count", 3))
        now_f = time.time()
        eligible = []
        for p in proposals:
            bidder = p.get("bidder", "")
            if self._peer_consecutive_timeouts.get(bidder, 0) >= zombie_max:
                logger.warning(
                    "[PeerOrch] Election skipped zombie bidder '%s' for '%s' "
                    "(%d consecutive timeouts >= %d)",
                    bidder, camera_id,
                    self._peer_consecutive_timeouts.get(bidder, 0), zombie_max,
                )
                continue
            peer = self._peers.get(bidder)
            if peer is not None and now_f < getattr(peer, "penalty_until", 0.0):
                logger.warning(
                    "[PeerOrch] Election skipped penalized bidder '%s' for '%s' "
                    "(penalty %.0fs remaining)",
                    bidder, camera_id,
                    getattr(peer, "penalty_until", 0.0) - now_f,
                )
                continue
            if self._migrated_out.get(camera_id) == bidder:
                logger.warning(
                    "[PeerOrch] Election skipped previous winner '%s' for '%s' "
                    "(anti-affinity).",
                    bidder, camera_id,
                )
                continue
            eligible.append(p)
        if not eligible:
            logger.error(
                "[PeerOrch] All %d bidder(s) for '%s' gated (zombie/penalty). "
                "Cooling down instead of re-targeting a proven-dead peer.",
                len(proposals), camera_id,
            )
            with self._lock:
                self._vote_in_progress.discard(camera_id)
                self._rfo_snapshots.pop(camera_id, None)
            self._cam_cooldown[camera_id] = time.time()
            return
        winner = min(eligible, key=lambda p: p["score"])
        cam_config = self._get_camera_config(camera_id)
        if cam_config is None:
            logger.error("[PeerOrch] Cannot get config for camera '%s'. Aborting election.", camera_id)
            return

        now_ts = time.time()
        with self._lock:
            cur_epoch = self._camera_epochs.get(camera_id, 1) + 1
            self._camera_epochs[camera_id] = cur_epoch
            mig_id = f"mig_{camera_id}_{int(now_ts * 1000)}"
            self._camera_migration_ids[camera_id] = mig_id
            self._pending_migration_ids[camera_id] = mig_id
            self._pending_epochs[camera_id] = cur_epoch

        decision = {
            "winner":       winner["bidder"],
            "camera_id":    camera_id,
            "from_node":    self._node_id,
            "cam_config":   cam_config,
            "epoch":        cur_epoch,
            "migration_id": mig_id,
            "timestamp":    now_ts,
            "ts":           now_ts,
        }
        winner_id = winner["bidder"]

        # Phase 3 review fix 1+3: atomically (a) reserve a stream slot on the
        # winner and (b) register the pending-ACK event BEFORE the decision is
        # published.  A fast valid ACK from the winner can therefore never
        # arrive before its event exists, so it cannot be dropped into a false
        # timeout/penalty.  All three lifecycle actors — _close_vote_window
        # (reserve), _on_vote_ack (ACK release), _wait_and_remove (timeout
        # release) — mutate _peer_inflight/_pending_winner under _lock so a
        # single canonical owner always wins the pop.
        ack_event = threading.Event()
        with self._lock:
            self._peer_inflight[winner_id] = self._peer_inflight.get(winner_id, 0) + 1
            self._pending_winner[camera_id] = winner_id
            self._pending_started_at[camera_id] = time.time()
            self._pending_acks[camera_id] = ack_event

        self._pubs["vote_decision"].put(msgpack.packb(decision, use_bin_type=True))
        self._cam_cooldown[camera_id] = time.time()
        logger.info(
            "[PeerOrch] Election won by '%s' for '%s' (score=%.1f, fps_pred=%.1f, inflight=%d)",
            winner_id, camera_id, winner["score"], winner.get("fps_predicted", 0),
            self._peer_inflight[winner_id],
        )

    def _pick_camera_to_offload(self, state: PeerState, level: int) -> Optional[str]:
        """
        Select camera to offload based on intended offload level.

        Level 2 (full-stream migration / RFO): choose the
        **lightest** eligible camera (min workload) — migrate the easiest stream,
        keep heavy cameras local. This is the decode/tracking relief path.

        Note: this selector only handles full-stream (level 2) migration. Plate-crop
        (source offload_level==1) camera selection lives in
        _pick_camera_for_lpr_offload; any non-level-2 level
        returns None here (see the fail-safe at the end of the method).

        Workload comes from the health payload's camera_workload mapping
        (n_track + n_plate per camera) — NOT output FPS, which is a poor proxy
        for processing cost.  A candidate must have a finite, non-negative
        workload; cameras with missing evidence are skipped.  If no candidate
        has workload evidence, return None (fail safe) — never rank on FPS.

        Source-starved cameras (health agent reports them starved) and cameras
        within the reclaim post-return observation window are ineligible.

        Never offload the last camera — node must keep at least 1 camera
        to continue operation. If only 1 camera remains and still overloaded,
        that's a hardware limit that cannot be solved by migration.

        L1 ownership invariant
        ──────────────────────
Full-stream migration removes a stream from this node. We must keep
        at least one locally-owned camera (enabled in this node's cameras.yml)
        active at all times. Rescued/migrated-in cameras are NOT owned by
        this node -- the previous selector kept one arbitrary active camera,
        so a foreign camera could "stand in" while every owned camera got
        migrated away.

          * If no owned camera is held -> fail safe (return None).
          * Otherwise, among eligible candidates, pick the lightest whose
            migration still leaves >= 1 owned camera held.

        L1 plate-crop offload (source offload_level==1, selected elsewhere) does NOT remove the stream,
        so its ownership guard is handled in _evaluate_lpr_offload; this L1
        selector enforces the L1 ownership guard above.
        """
        now = time.time()
        if len(state.held_cameras) <= 1:
            if state.held_cameras:
                if self._maybe_log_block("pick_only_one_camera", now):
                    logger.debug(
                        "[PeerOrch] Only 1 camera left ('%s') — cannot offload last camera",
                        state.held_cameras[0],
                    )
            return None

        reclaim_stability = self._cfg.get("reclaim_stability_s", 6.0)
        reclaim_stable = self._cfg.get("reclaim_stable_s", 5.0)
        reclaim_window = max(reclaim_stability, reclaim_stable)
        starved = set(state.source_starved_cameras or [])

        # Bounce dampening: exclude cameras that have reached bounce_max migrations within bounce_window_s
        bounce_max = int(self._cfg.get("bounce_max", 3))
        bounce_window_s = float(self._cfg.get("bounce_window_s", 300.0))
        bounced_cameras = set()
        with self._lock:
            for cam_id, history in list(self._cam_migration_history.items()):
                # Filter out expired timestamps
                valid_history = [ts for ts in history if now - ts <= bounce_window_s]
                if len(valid_history) != len(history):
                    if valid_history:
                        self._cam_migration_history[cam_id] = valid_history
                    else:
                        self._cam_migration_history.pop(cam_id, None)
                if len(valid_history) >= bounce_max:
                    bounced_cameras.add(cam_id)

        # ponytail: per-camera warmup gate.  A freshly-ADDed camera whose FPS
        # hasn't stabilised is ineligible for offload — its workload is
        # untrustworthy and offloading it would just thrash.  Cameras NOT
        # recorded in _camera_added_at are pre-existing and skip the gate,
        # which keeps the existing L1/L2 selector tests passing.
        camera_warmup_s = self._cfg.get(
            "camera_warmup_s", self._cfg.get("overload_warmup_s", 10.0),
        )

        def _camera_warming_up(cam_id: str) -> bool:
            if cam_id not in self._camera_added_at:
                return False  # pre-existing — no warmup
            first_fps = self._camera_first_valid_fps_at.get(cam_id)
            if first_fps is None:
                return True   # added but no valid FPS observed yet
            return (now - first_fps) < camera_warmup_s

        # Workload evidence must come from the health payload. Require a
        # finite, non-negative workload per camera; missing/malformed values
        # are skipped (fail safe). Do NOT fall back to output FPS.
        cooldown_s = float(self._cfg.get("cooldown_s", 6.0))
        workload = state.camera_workload or {}
        eligible = {}
        for c in state.held_cameras:
            if c in starved:
                continue
            if now - self._reclaim_completed_at.get(c, 0.0) < reclaim_window:
                continue
            if (now - self._cam_cooldown.get(c, 0.0)) < cooldown_s:
                continue
            if _camera_warming_up(c):
                continue
            if c not in workload:
                continue
            w = workload[c]
            if not (isinstance(w, (int, float))
                    and not isinstance(w, bool)
                    and math.isfinite(w)
                    and w >= 0):
                continue
            eligible[c] = float(w)

        if not eligible:
            return None

        # Split eligible cameras into foreign (not in cameras.yml) and owned
        owned_cam_ids = self._get_owned_camera_ids()
        foreign_eligible = {c: w for c, w in eligible.items() if c not in owned_cam_ids}
        owned_eligible = {c: w for c, w in eligible.items() if c in owned_cam_ids}

        if level == 2:
            # Foreign and rescued camera filter for L1 candidates
            foreign_l1 = {c: w for c, w in foreign_eligible.items() if c not in bounced_cameras and c not in self._rescued_cameras}
            owned_l1 = {c: w for c, w in owned_eligible.items() if c not in bounced_cameras and c not in self._rescued_cameras}

            # L1 ownership guard: never migrate away the last owned camera.
            # Explicit L1 guard: never select an L1 candidate when <=1 locally-owned held camera.
            owned_held = owned_cam_ids & set(state.held_cameras)
            if len(owned_held) == 0:
                if self._maybe_log_block("l1_no_owned_active", now):
                    logger.info(
                        "[PeerOrch] L1 fail-safe: no locally-owned camera is held "
                        "(held=%d). Skipping migration to preserve ownership.",
                        len(state.held_cameras),
                    )
                return None

            # ponytail: foreign cameras MUST NOT be L1-migrated — they can only
            # return to their original owner via the owner's reclaim path.
            # Forwarding a foreign camera to a third node via RFO creates
            # chain migration (B→A→C) that breaks owner reclaim.
            if foreign_l1:
                if self._maybe_log_block("l1_skipping_foreign", now):
                    logger.info(
                        "[PeerOrch] L1: skipping %d foreign camera(s) — only owner can reclaim.",
                        len(foreign_l1),
                    )

            # If owned held <= 1, guard against offloading owned camera
            if len(owned_held) <= 1:
                if self._maybe_log_block("l1_guard_single_owned", now):
                    logger.info(
                        "[PeerOrch] L1 guard: <= 1 locally-owned camera held (owned_held=%d, held=%d) and no foreign candidates. Skipping L1 migration.",
                        len(owned_held), len(state.held_cameras),
                    )
                return None

            # Then owned MIN workload (guard <=1 owned)
            if owned_l1:
                for c in sorted(owned_l1, key=lambda cam: owned_l1[cam]):
                    if (owned_held - {c}):
                        return c

            if self._maybe_log_block("l1_no_safe_candidate", now):
                logger.info(
                    "[PeerOrch] L1 fail-safe: every eligible candidate would leave "
                    "zero owned cameras held (owned_held=%d).",
                    len(owned_held),
                )
            return None

        # P3 redesign: only Level 1 (full-stream migration) remains. Crop
        # offload (vehicle-crop tier, retired) was removed. Any non-L1 level is
        # treated as a fail-safe (no candidate) to avoid resurrecting crop
        # selection.
        return None
