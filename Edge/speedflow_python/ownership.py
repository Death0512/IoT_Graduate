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
Edge/speedflow_python/ownership.py — Camera Ownership & Migration Mixin

OwnershipMixin — Handles camera ownership, epoch fencing, and make-before-break
migration for the PeerOrchestrator. Mixed into peer_orchestrator.PeerOrchestrator.

Key Concepts:
- Static Ownership: Fixed per node (jetson_A=cam_01,02; jetson_B=cam_03,04; jetson_C=cam_05,06).
  Never changes. Used for fallback/reclaim destination.
- Dynamic Holding: Current pipeline runner for a camera. Changes via migration/rescue.
- Epoch Fencing: _camera_epochs advances ONLY on confirmed transfers (Migration DONE,
  reclaim-complete adopt from _proposed_epochs). Per-round fencing freshness from
  unique migration_id nonce. Failed rounds leave no advertised trace.
- Holder Sequence (_camera_holders): Tracks who currently holds each camera.
  Combined with epoch for Server CameraProjection anti-resurrection.

Make-Before-Break Protocol:
  1. Dest ADD → pipeline reaches PLAYING → publishes peers/vote/ack/{cam}
  2. Src waits for ack → REMOVE from local pipeline
  3. Src publishes COMMIT (ownership epoch advances) → Holder updates

Server CameraProjection prefers verified-streaming claimants (active + positive FPS);
hysteresis damps voluntary moves after 2 flips/600s (bounce dampening).

Dependencies:
- membership.py — PeerState, HRW hash, offload ladder, logging
- zenoh_session.py — Shared Zenoh session
- settings.py — All configuration
"""


class OwnershipMixin:
    def get_ownership_records(self) -> Dict[str, dict]:
        """
        Return thread-safe snapshot of locally active/owned camera ownership records.
        Distinguishes static/original owner from current holder.
        Maps camera_id -> {"owner": str, "holder": str, "epoch": int, "migration_id": Optional[str]}
        """
        records = {}
        static_owned = self._get_owned_camera_ids()
        with timed_lock(self._lock, "_lock.get_ownership_records", logger=logger):
            with timed_lock(self._self_lock, "_self_lock.get_ownership_records", logger=logger):
                active_cams = list(self._self_state.active_cameras)
            for cam_id in active_cams:
                epoch = self._camera_epochs.get(cam_id, 1)
                # P5 fix: floor the exported epoch against the persisted lease
                # high-water so a node restarted mid-migration (in-memory
                # _camera_epochs regressed to 0/default) never publishes a stale
                # epoch below what the lease already guarantees (cam_06=58 case).
                if self._lease is not None:
                    try:
                        lease_floor = self._lease.epoch_floor(cam_id)
                        if lease_floor > 0:
                            epoch = max(epoch, lease_floor)
                    except Exception as exc:
                        logger.debug("[PeerOrch] lease epoch floor lookup failed for '%s': %s", cam_id, exc)
                mig_id = self._camera_migration_ids.get(cam_id)
                # Static owner is this node if configured in cameras.yml;
                # otherwise check rescued_cameras or fallback to static mapping.
                orig_owner = self._rescued_cameras.get(cam_id)
                if orig_owner is None:
                    if cam_id in static_owned:
                        orig_owner = self._node_id
                    else:
                        # Check configured mapping for any peer
                        node_cam_map = self._cfg.get("node_camera_map")
                        if isinstance(node_cam_map, dict):
                            for nid, cams in node_cam_map.items():
                                if isinstance(cams, (list, tuple, set)) and cam_id in cams:
                                    orig_owner = nid
                                    break
                rec = {
                    "owner": orig_owner,
                    "holder": self._node_id,
                    "epoch": epoch,
                }
                if mig_id is not None:
                    rec["migration_id"] = mig_id
                records[cam_id] = rec
        return records

    def _on_vote_request(self, payload: dict) -> None:
        """
        Receive RFO from another peer.
        Check ε-constraints, if pass → send proposal.

        RTT measurement runs in a thread pool so we never block
        the Zenoh subscriber callback thread.
        """
        requester = payload.get("requester", "")
        if requester == self._node_id:
            return  # Ignore own RFO

        camera_id = payload.get("camera_id", "")
        logger.info("[PeerOrch] RFO received from '%s' for camera '%s'", requester, camera_id)

        # Offload the blocking work immediately so this callback returns fast
        self._safe_submit(self._evaluate_and_bid, payload)

    def _evaluate_and_bid(self, payload: dict) -> None:
        """
        Run ε-constraint checks and publish proposal if eligible.
        Runs in ThreadPoolExecutor — safe to block for RTT measurement.
        """
        requester     = payload.get("requester", "")
        camera_id     = payload.get("camera_id", "")
        eps_fps       = payload.get("eps_fps", 18.0)
        eps_net_ms    = payload.get("eps_network_ms", 50.0)

        # In-flight guard: do not accept/bid on an RFO if camera is already
        # undergoing uncommitted migration, rescue, or already active/held.
        now = time.time()
        heartbeat_timeout = float(self._cfg.get("heartbeat_timeout_s", 5.0))
        with self._lock:
            if (camera_id in self._pending_acks
                    or camera_id in self._pending_winner
                    or camera_id in self._rescued_cameras
                    or camera_id in self._migrated_out
                    or camera_id in self._reclaim_in_progress):
                logger.info("[PeerOrch] RFO rejected for '%s': uncommitted state/already handled", camera_id)
                return
            # Priority: finish reclaiming our own migrated-out cameras before
            # volunteering to host foreign streams. Accepting foreign RFO fills
            # local capacity and raises load_score, which blocks _check_reclaim's
            # capacity/load gates — a priority inversion that strands owned cams.
            # ponytail: blanket refuse-while-reclaiming; if a node must host
            # foreign cams during a long reclaim, gate per-owned-camera instead.
            if self._migrated_out:
                logger.info(
                    "[PeerOrch] RFO rejected for '%s': local reclaim pending for %s "
                    "(owned-camera reclaim takes priority over hosting foreign streams)",
                    camera_id, list(self._migrated_out.keys()),
                )
                return

            # Reject if an alive peer already reports this camera held
            for pid, p in self._peers.items():
                if pid != requester and (now - p.last_seen <= heartbeat_timeout) and (camera_id in p.held_cameras):
                    logger.info("[PeerOrch] RFO rejected for '%s': already held on alive peer '%s'", camera_id, pid)
                    return
        with self._self_lock:
            if camera_id in self._self_state.held_cameras:
                logger.info("[PeerOrch] RFO rejected for '%s': already held locally", camera_id)
                return

        # BUG-1 fix: read _self_state under its own lock
        with self._self_lock:
            if not _has_valid_positive_fps(self._self_state.fps_per_camera):
                logger.info(
                    "[PeerOrch] RFO rejected for '%s': no valid positive FPS locally (fps_per_camera=%s)",
                    camera_id, self._self_state.fps_per_camera,
                )
                return
            current_streams = len(self._self_state.active_cameras)
            self_load = self._self_state.load_score
            self_temp = self._self_state.gpu_temp_c

        # Phase 3 review fix 2 and Fix 5: explicit L1 capacity semantics.
        #
        # ``max_streams`` is the hard hardware/pipeline limit (e.g. GStreamer
        # streammux max-sources / decoder max slots) — enforced for direct
        # sender-side peer selection in ``_pick_best_peer`` and for failover
        # self-eligibility.          It is NEVER relaxed by plate-crop (L2, source offload_level==1) offload.
        #
        # ``eps_streams_max`` is the ε admission policy limit — it is the
        # ceiling used by the receiver's ε1 gate (_evaluate_and_bid) which
        # additionally accounts ``_self_inflight`` reservations (bids accepted
        # but not yet ADDed).  Both the capacity check and the reservation
        # increment are held under ``_self_inflight_lock`` as a single atomic
        # read–modify–write, with explicit rollback on every downstream reject
        # path so concurrent evaluators cannot overbook eps_streams_max.
        #
        # Canonical expression of L1 slot availability (used consistently in
        # _pick_best_peer, _evaluate_and_bid, and failover self-eligibility):
        #   sender gate (receiver hardware):           peer.active_cameras + peer_inflight < peer.max_streams
        #   receiver admission (receiver ε policy):    current_active + self_inflight < eps_streams_max
        #   failover self-eligibility (self hardware): current_active + self_accepted < eps_streams_max  (= eps_streams_max)

        # ε0 — Thermal admission gate for THIS node (receiver).
        # Same rule as _pick_best_peer's sender-side gate: do not bid when
        # this node is too hot or has an unknown reading under a conservative
        # policy. Accepting a stream onto a throttled node helps nobody.
        therm_cfg = self._cfg.get("thermal")
        if not _thermal_admission_ok(self_temp, therm_cfg):
            logger.info(
                "[PeerOrch] RFO rejected for '%s': ε0 (thermal-self) — "
                "gpu_temp_c=%s (max=%.1f)",
                camera_id, self_temp,
                therm_cfg.get("max_gpu_temp_c", 85.0) if therm_cfg else 85.0,
            )
            return

        # ε1 — Capacity constraint (Phase 3 review fix 2).
        #
        # The capacity check AND the _self_inflight reservation increment are
        # one atomic critical section: concurrent evaluators cannot both read
        # the same "free slot" and overbook past eps_streams_max.  The
        # reservation is made BEFORE any later ε-check (ε2..ε5) or the RTT
        # measurement, and rolled back via the finally block below on every
        # downstream rejection/exception — so a bid that never ships does not
        # hold capacity, and one that ships is counted exactly once.
        cm = self._camera_manager
        default_max = cm.get_max_streams() if cm is not None else 4
        eps_streams_max = int(self._cfg.get("eps_streams_max", default_max))
        with self._self_inflight_lock:
            accounted_streams = current_streams + self._self_inflight
            if accounted_streams >= eps_streams_max:
                logger.info(
                    "[PeerOrch] RFO rejected for '%s': ε1 (capacity) — "
                    "current=%d inflight=%d max=%d",
                    camera_id, current_streams,
                    accounted_streams - current_streams, eps_streams_max,
                )
                return
            self._self_inflight += 1  # reserve now, atomically with the check

        # Flag flipped to False once the bid is fully shipped (no rollback).
        reservation_committed = False
        try:
            # ε2 — FPS prediction (demoted from hard reject gate to soft bid scoring)
            # YAML parses bare integer keys as int; look up both int and str forms
            fps_model = self._cfg.get("fps_model", {})
            streams_after = current_streams + 1
            predicted_fps = fps_model.get(streams_after,
                            fps_model.get(str(streams_after), None))
            # ponytail: static fps_model is used for soft bid scoring rather than hard rejection
            logger.debug(
                "[PeerOrch] FPS model evaluation for '%s': streams_after=%d, predicted_fps=%s, eps_fps=%.1f (soft bid scoring only)",
                camera_id, streams_after, predicted_fps, eps_fps,
            )

            # ε3 — Network RTT to camera RTSP origin (blocking — safe here in thread pool)
            # Prefer URI from the RFO payload (sent by requester who owns the camera).
            # Fall back to local lookup for backward compatibility.
            cam_uri = payload.get("cam_uri") or self._get_camera_uri(camera_id)
            if not cam_uri:
                logger.info("[PeerOrch] RFO rejected for '%s': ε3 (network) — camera URI not found", camera_id)
                return
            rtt_ms = self._measure_rtt(cam_uri)
            if rtt_ms is None or rtt_ms > eps_net_ms:
                logger.info(
                    "[PeerOrch] RFO rejected for '%s': ε3 (network) — "
                    "RTT=%.1fms, threshold=%.1fms",
                    camera_id, rtt_ms if rtt_ms else -1.0, eps_net_ms,
                )
                return

            # ε4 — Per-camera cooldown
            last_mig = self._cam_cooldown.get(camera_id, 0.0)
            cooldown_s = self._cfg.get("cooldown_s", 6.0)
            time_since_last = time.time() - last_mig
            if time_since_last < cooldown_s:
                logger.info(
                    "[PeerOrch] RFO rejected for '%s': ε4 (cooldown) — "
                    "%.1fs since last migration, need %.1fs",
                    camera_id, time_since_last, cooldown_s,
                )
                return

            # ε5 — Penalty check (applied when this node previously caused a migration timeout)
            now = time.time()
            if now < self._self_penalty_until:
                logger.info(
                    "[PeerOrch] RFO rejected for '%s': ε5 (penalty) — "
                    "penalized until %.1f",
                    camera_id, self._self_penalty_until,
                )
                return

            # All constraints pass — compute multi-objective cost F(x)
            # Incorporates load pressure, predicted FPS degradation, network RTT, and thermal headroom.
            bid_weights = self._cfg.get("p2p", {}).get("bid_weights", {})
            w_load = float(bid_weights.get("w_load", 0.50))
            w_fps = float(bid_weights.get("w_fps", 0.25))
            w_rtt = float(bid_weights.get("w_rtt", 0.15))
            w_therm = float(bid_weights.get("w_therm", 0.10))

            target_fps = float(self._cfg.get("target_fps", 25.0))
            fps_degrade_ratio = max(0.0, min(1.0, (target_fps - (predicted_fps or target_fps)) / max(1.0, target_fps)))
            rtt_ratio = max(0.0, min(1.0, (rtt_ms or 0.0) / max(1.0, eps_net_ms)))

            # Thermal headroom cost above onset (70C -> 85C)
            onset_c = float(self._cfg.get("thermal", {}).get("onset_gpu_temp_c", 70.0))
            crit_c = float(self._cfg.get("thermal", {}).get("max_gpu_temp_c", 85.0))
            if self_temp is not None:
                therm_ratio = max(0.0, min(1.0, (self_temp - onset_c) / max(1.0, (crit_c - onset_c))))
            else:
                therm_ratio = 0.0
                logger.debug("[PeerOrch] Bid F(x): gpu_temp_c is None, w_therm contribution is 0.0")

            # ponytail: multi-objective incremental cost; lowest cost wins vote
            f_x = (
                w_load * (self_load / 100.0)
                + w_fps * fps_degrade_ratio
                + w_rtt * rtt_ratio
                + w_therm * therm_ratio
            ) * 100.0

            now_ts = time.time()
            proposal = {
                "bidder":        self._node_id,
                "camera_id":     camera_id,
                "score":         round(f_x, 2),
                "fps_predicted": predicted_fps,
                "rtt_ms":        round(rtt_ms, 1),
                "timestamp":     now_ts,
                "ts":            now_ts,
            }

            self._pubs["vote_proposal"].put(msgpack.packb(proposal, use_bin_type=True))
            logger.info(
                "[PeerOrch] RFO accepted for '%s' (ALL ε-constraints pass) — "
                "Bid: score=%.1f, fps_pred=%s, rtt=%.0fms",
                camera_id, f_x, f"{predicted_fps:.1f}" if predicted_fps is not None else "None", rtt_ms,
            )
            reservation_committed = True

            # Phase 3: hold the receiver-side reservation for the worst-case
            # resolution window (vote_window_s + migration_timeout_s), then
            # decay it.  Conservative timeout-only decay: the ADD command goes
            # to ZenohCommandSubscriber, not back through this orchestrator,
            # so we cannot hook the ADD arrival to release earlier.
            decay_s = (
                self._cfg.get("vote_window_s", 2.0)
                + self._cfg.get("migration_timeout_s", 15.0)
            )

            def _decay_self_inflight():
                with self._self_inflight_lock:
                    self._self_inflight = max(0, self._self_inflight - 1)
                logger.debug("[PeerOrch] Self inflight reservation decayed (camera='%s')", camera_id)

            threading.Timer(decay_s, _decay_self_inflight).start()
        finally:
            # Roll back the reservation if we never shipped a bid (any ε-check
            # rejection above, RTT failure, or an unexpected exception).
            if not reservation_committed:
                with self._self_inflight_lock:
                    self._self_inflight = max(0, self._self_inflight - 1)

    def _on_vote_proposal(self, payload: dict) -> None:
        """Collect proposals — only requester processes."""
        camera_id = payload.get("camera_id", "")
        if not camera_id:
            return
        with self._lock:
            if camera_id in self._vote_windows:
                self._vote_windows[camera_id].append(payload)

    def _on_vote_decision(self, payload: dict) -> None:
        """
        Receive election result.

        If I am winner → ADD camera.
        If I am requester → wait for ack then REMOVE.
        """
        winner    = payload.get("winner", "")
        camera_id = payload.get("camera_id", "")
        from_node = payload.get("from_node", "")

        if winner == self._node_id:
            # Duplicate / uncommitted guard on winner side
            now = time.time()
            heartbeat_timeout = float(self._cfg.get("heartbeat_timeout_s", 5.0))
            with self._lock:
                if (camera_id in self._pending_acks
                        or camera_id in self._pending_winner
                        or camera_id in self._rescued_cameras
                        or camera_id in self._migrated_out
                        or camera_id in self._reclaim_in_progress):
                    logger.info("[PeerOrch] Decision ignored: camera '%s' is in-flight/uncommitted or held.", camera_id)
                    return

                # Reject/abort if an alive peer already reports this camera active
                for pid, p in self._peers.items():
                    if pid != from_node and (now - p.last_seen <= heartbeat_timeout) and (camera_id in p.active_cameras):
                        logger.info("[PeerOrch] Decision ignored: camera '%s' is already active on alive peer '%s'.", camera_id, pid)
                        return
            with self._self_lock:
                if camera_id in self._self_state.active_cameras:
                    logger.info("[PeerOrch] Decision ignored: camera '%s' is already active locally.", camera_id)
                    return

            # --- I WON: ADD camera to pipeline ---
            cam_config = payload.get("cam_config", {})
            if not cam_config:
                logger.error("[PeerOrch] Decision missing cam_config for '%s'", camera_id)
                return

            epoch = payload.get("epoch")
            migration_id = payload.get("migration_id")
            if epoch is not None or migration_id is not None:
                with self._lock:
                    if epoch is not None:
                        # P5: never regress the epoch counter below the
                        # persisted high-water (or any epoch we have already
                        # minted/seen) — guarantees newly minted epochs are
                        # never lower than prior values.
                        e = int(epoch)
                        if e > self._camera_epochs.get(camera_id, 0):
                            self._camera_epochs[camera_id] = e
                    if migration_id is not None:
                        self._camera_migration_ids[camera_id] = str(migration_id)

            self._dispatch_add(camera_id, winner, cam_config, epoch, migration_id)

        elif from_node == self._node_id:
            # --- I AM REQUESTER: wait for ack then REMOVE ---
            self._executor.submit(self._wait_and_remove, camera_id, winner)

    def _dispatch_add(
        self,
        camera_id: str,
        winner: str,
        cam_config: dict,
        epoch: Optional[int],
        migration_id: Optional[str],
    ) -> None:
        add_cmd = {**cam_config, "cmd": "ADD"}
        if epoch is not None:
            add_cmd["epoch"] = epoch
        if migration_id is not None:
            add_cmd["migration_id"] = migration_id
        # P5: stamp recipient's boot_id so a pre-reboot ADD is fenced.
        self._attach_lease_fields(add_cmd, winner)
        # ZenohCommandSubscriber is the single owner of ADD/ACK: it
        # processes the ADD, waits for the stream to reach PLAYING,
        # and publishes peers/vote/ack/{cam}.  Publishing directly to
        # peers/control/{winner} routes through Zenoh — which loops
        # back to our own subscriber when we are the winner.
        winner_key = f"peers/control/{winner}"
        self._session.put(winner_key, msgpack.packb(add_cmd, use_bin_type=True))
        logger.info("[PeerOrch] ADD command published for '%s' to '%s'", camera_id, winner)
        # ponytail: camera now being added from vote winner — record the ADD
        # so _pick_camera_to_offload can apply the per-camera warmup gate.
        self._camera_added_at[camera_id] = time.time()
        self._camera_first_valid_fps_at.pop(camera_id, None)
        self._transition_settle_until = max(
            self._transition_settle_until,
            time.time() + self._cfg.get("transition_settle_s", 5.0),
        )

    def _redispatch_after_reject(self, camera_id: str, winner: str) -> None:
        with self._lock:
            attempts = self._reject_retries.get(camera_id, 0) + 1
            self._reject_retries[camera_id] = attempts
            epoch = self._camera_epochs.get(camera_id, 1)
        if attempts > 3:
            logger.error(
                "[PeerOrch] '%s' still rejected after %d fast-forwards; abandoning migration.",
                camera_id, attempts,
            )
            with self._lock:
                # Release ALL pending migration state so the camera is not
                # permanently blocked by the RFO / decision / ladder guards.
                self._reject_retries.pop(camera_id, None)
                self._pending_winner.pop(camera_id, None)
                ev = self._pending_acks.pop(camera_id, None)
                if ev is not None:
                    ev.set()
                self._pending_epochs.pop(camera_id, None)
                self._pending_migration_ids.pop(camera_id, None)
                self._pending_started_at.pop(camera_id, None)
                # The in-flight reservation increments here so a later
                # normal-completion path does not underflow the peer counter.
                self._peer_inflight[winner] = self._peer_inflight.get(winner, 0) + 1
            return
        cam_config = self._get_camera_config(camera_id)
        if cam_config is None:
            return
        mig_id = f"mig_{camera_id}_{int(time.time() * 1000)}"
        with self._lock:
            self._camera_migration_ids[camera_id] = mig_id
            self._pending_migration_ids[camera_id] = mig_id
            self._pending_epochs[camera_id] = epoch
            self._peer_inflight[winner] = self._peer_inflight.get(winner, 0) + 1
            self._pending_winner[camera_id] = winner
            ev = threading.Event()
            self._pending_acks[camera_id] = ev
        self._dispatch_add(camera_id, winner, cam_config, epoch, mig_id)
        self._executor.submit(self._wait_and_remove, camera_id, winner)

    def _l1_remove_ownership_guard(self, camera_id: str) -> bool:
        """
        Final atomic ownership guard, called immediately before an L1 REMOVE
        that would remove ``camera_id`` from this node's pipeline.

        The decision-time guard in ``_pick_camera_to_offload`` ran against a
        snapshot that is now stale: while we waited for the winner's ACK, a
        concurrent migration / reclaim / rebalance may have removed every
        other locally-owned camera, leaving this one as the last owned stream.
        Re-check ownership against the CURRENT active set so a stale decision
        can never REMOVE the final owned camera and leave only foreign streams.

        Logically atomic: reads ownership (``_get_owned_camera_ids``) and the
        active set (``_self_state.active_cameras`` under ``_self_lock``) with
        no intervening wait/await before the caller acts on the result.

        Returns True if the REMOVE may proceed; False if it must be aborted
        (this camera is the last locally-owned active camera, or ownership is
        unresolved).
        """
        try:
            owned = self._get_owned_camera_ids()
        except Exception as exc:
            logger.warning(
                "[PeerOrch] L1 ownership guard: ownership lookup failed: %s", exc
            )
            owned = set()

        # Removing a foreign (rescued/migrated-in) camera never reduces the
        # locally-owned-active count, so it may always proceed.
        if owned and camera_id not in owned:
            return True

        with self._self_lock:
            active = set(self._self_state.active_cameras)

        if not owned:
            # Foreign camera check when owned is empty: if camera_id is recorded
            # as rescued or migrated-in, it is definitely foreign.
            with self._lock:
                if camera_id in self._rescued_cameras:
                    return True
            # Fail closed: cannot prove another owned camera would remain.
            return False

        # Owned camera: proceed only if some OTHER owned camera stays active.
        owned_active = owned & active
        return bool(owned_active - {camera_id})

    def _wait_and_remove(self, camera_id: str, winner_node: str) -> None:
        """
        Make-before-Break: wait for winner to confirm PLAYING → REMOVE from self.

        If timeout → rollback (penalize winner node).

        Phase 3 review fix 1: the pending-ACK event was already registered in
        _close_vote_window BEFORE the decision was published, so a fast valid
        ACK can never arrive before its event exists (no dropped ACK → no false
        timeout/penalty).  We reuse that pre-registered event here instead of
        creating a fresh one.
        """
        with self._lock:
            event = self._pending_acks.get(camera_id)
        if event is None:
            # Defensive fallback (e.g. caller other than the requester path)
            # — create it now, even though it should already exist.
            event = threading.Event()
            with self._lock:
                self._pending_acks[camera_id] = event

        start_ms = time.time() * 1000
        timeout = self._cfg.get("migration_timeout_s", 15.0)
        # Carry RFO trigger snapshot metrics if available, else capture under _self_lock
        with self._lock:
            snap = self._rfo_snapshots.pop(camera_id, None)
        if snap is not None:
            trigger_load, trigger_fps = snap
        else:
            with self._self_lock:
                trigger_load = self._self_state.load_score
                trigger_fps  = self._self_state.avg_fps

        confirmed = event.wait(timeout=timeout)

        with self._lock:
            self._pending_acks.pop(camera_id, None)
            retry = self._add_rejected.pop(camera_id, False)

        if retry:
            logger.info("[PeerOrch] Re-dispatching ADD for '%s' after epoch fast-forward.", camera_id)
            self._redispatch_after_reject(camera_id, winner_node)
            return

        if not confirmed:
            # Timeout — rollback: penalise the winner so we don't pick it again soon
            logger.error(
                "[PeerOrch] TIMEOUT (%ds) waiting for ack from '%s' for '%s'. Rolling back.",
                int(timeout), winner_node, camera_id,
            )
            # Phase 3 review fix 3: release the in-flight reservation ONLY if
            # this camera still owns one, and ONLY once.  _on_vote_ack and the
            # timeout path both race to pop _pending_winner under _lock — the
            # single winner of the pop is the single owner of the decrement, so
            # an ACK/timeout race can never double-release or decrement a
            # reservation belonging to a different camera/winner.
            with self._lock:
                owned = self._pending_winner.pop(camera_id, None)
                self._pending_started_at.pop(camera_id, None)
                curr_timeouts = self._peer_consecutive_timeouts.get(winner_node, 0) + 1
                self._peer_consecutive_timeouts[winner_node] = curr_timeouts
            if owned == winner_node:
                self._peer_inflight[winner_node] = max(
                    0, self._peer_inflight.get(winner_node, 0) - 1
                )
                logger.debug(
                    "[PeerOrch] Timeout released reservation for '%s' (winner='%s', inflight=%d)",
                    camera_id, winner_node, self._peer_inflight[winner_node],
                )
            base_cooldown = self._cfg.get("cooldown_s", 6.0)
            multiplier = min(2 ** (curr_timeouts - 1), 8)
            penalty_duration = max(base_cooldown * 2, base_cooldown * multiplier)
            penalty_until = time.time() + penalty_duration
            if winner_node == self._node_id:
                # The winner is ourselves — set our own penalty field
                self._self_penalty_until = penalty_until
            else:
                with self._lock:
                    if winner_node in self._peers:
                        self._peers[winner_node].penalty_until = penalty_until
            self._migration_log.log(
                self._node_id, winner_node, camera_id,
                "timeout", trigger_load, trigger_fps,
                time.time() * 1000 - start_ms, "TIMEOUT_ROLLBACK",
            )
            return

        # Success — REMOVE from self.
        # Final atomic ownership guard: the decision-time check in
        # _pick_camera_to_offload ran against a snapshot that may now be
        # stale. A concurrent migration / reclaim / rebalance can have
        # removed every other owned camera while we were waiting for this
        # ACK, turning camera_id into the last owned stream. Block the
        # REMOVE rather than strip the node down to foreign-only streams.
        if not self._l1_remove_ownership_guard(camera_id):
            logger.error(
                "[PeerOrch] L1 REMOVE ABORTED for '%s' (winner='%s'): it is now "
                "the last locally-owned active camera. Keeping it local; rolling back winner.",
                camera_id, winner_node,
            )
            # Send targeted rollback REMOVE to winner so it does not keep playing the stream
            if winner_node and winner_node != self._node_id:
                try:
                    winner_sid: Optional[int] = None
                    with self._lock:
                        winner_p = self._peers.get(winner_node)
                        if winner_p and isinstance(winner_p.camera_configs, dict):
                            winner_c = winner_p.camera_configs.get(camera_id)
                            if isinstance(winner_c, dict) and "source_id" in winner_c:
                                try:
                                    winner_sid = int(winner_c["source_id"])
                                except (ValueError, TypeError):
                                    pass
                    rollback_cmd: dict = {"cmd": "REMOVE", "camera_id": camera_id}
                    if winner_sid is not None:
                        rollback_cmd["source_id"] = winner_sid
                    else:
                        rollback_cmd = self._build_remove_cmd(camera_id, context="l1_guard_abort_rollback")
                    # P5: stamp recipient's boot_id so a pre-reboot rollback is fenced.
                    self._attach_lease_fields(rollback_cmd, winner_node)

                    winner_control_key = f"peers/control/{winner_node}"
                    if self._session is not None:
                        self._session.put(
                            winner_control_key,
                            msgpack.packb(rollback_cmd, use_bin_type=True),
                        )
                    logger.info(
                        "[PeerOrch] Rollback REMOVE sent to winner '%s' for '%s' following L1 ownership guard abort.",
                        winner_node, camera_id,
                    )
                except Exception as exc:
                    logger.error(
                        "[PeerOrch] Failed to send rollback REMOVE to winner '%s' for '%s': %s",
                        winner_node, camera_id, exc,
                    )

            with self._lock:
                stale_winner = self._pending_winner.pop(camera_id, None)
                self._pending_started_at.pop(camera_id, None)
            if stale_winner == winner_node:
                self._peer_inflight[winner_node] = max(
                    0, self._peer_inflight.get(winner_node, 0) - 1
                )
                logger.debug(
                    "[PeerOrch] L1 REMOVE abort released reservation for '%s' "
                    "(winner='%s', inflight=%d)",
                    camera_id, winner_node, self._peer_inflight[winner_node],
                )
            self._migration_log.log(
                self._node_id, winner_node, camera_id,
                "overload", trigger_load, trigger_fps,
                time.time() * 1000 - start_ms, "OWNERSHIP_GUARD_BLOCK",
            )
            self._cam_cooldown[camera_id] = time.time()
            return

        remove_cmd = self._build_remove_cmd(camera_id, context="l1_migration")
        self._pubs["control"].put(msgpack.packb(remove_cmd, use_bin_type=True))
        logger.info(
            "[PeerOrch] REMOVE sent to self for '%s'. Migration complete.",
            camera_id,
        )

        # Commit reclaimable ownership only after ACK and REMOVE are sent.
        self._cam_cooldown[camera_id] = time.time()
        with self._lock:
            self._migrated_out[camera_id] = winner_node
            if camera_id not in self._cam_migration_history:
                self._cam_migration_history[camera_id] = []
            self._cam_migration_history[camera_id].append(time.time())
        self.set_offload_level(camera_id, 0)

        elapsed_ms = time.time() * 1000 - start_ms
        # Δτ: time from migration complete to first valid speed on the new node.
        # The SpeedProbe on winner_node will update _first_valid_speed_ts once
        # it produces its first valid measurement; that timestamp is compared
        # against the local time here to get the Application Blind-spot duration.
        # We record the migration-complete timestamp; blind_spot_ms is computed
        # once the winner's first heartbeat confirms active FPS on this camera.
        self._migration_log.log(
            self._node_id, winner_node, camera_id,
            "overload", trigger_load, trigger_fps,
            elapsed_ms, "SUCCESS",
            blind_spot_ms=None,   # filled in by _update_blind_spot() when known
        )
        # Store migration-complete timestamp so _update_blind_spot can reference it
        self._migration_complete_ts[camera_id] = time.time()

        # Suppress overload decisions for the settle window so stale/draining
        # FPS samples on the reduced pipeline cannot trigger a second offload.
        self._transition_settle_until = time.time() + self._cfg.get(
            "transition_settle_s", 5.0,
        )

        logger.info(
            "[PeerOrch] Migration DONE in %.0fms: '%s' → %s",
            elapsed_ms, camera_id, winner_node,
        )

    def _record_failed_migration(self, camera_id: str) -> None:
        """Record a failed migration attempt so the camera is not immediately
        re-selected for offload (bounce protection)."""
        with self._lock:
            if camera_id not in self._cam_migration_history:
                self._cam_migration_history[camera_id] = []
            self._cam_migration_history[camera_id].append(time.time())

    def _wait_and_remove_reclaim(self, camera_id: str, holder_node: str) -> None:
        """
        Make-before-Break for reclaim:
          - Wait for local ADD ack (stream PLAYING on self)
          - Then send REMOVE to holder node

        Reuses the same _pending_acks event mechanism as _wait_and_remove.
        If timeout, log error and schedule a bounded exponential retry.
        """
        # Reclaim retry: re-validate holder state from latest heartbeat
        # before each retry. If holder dropped camera, clear stale state safely.
        now = time.time()
        heartbeat_timeout = float(self._cfg.get("heartbeat_timeout_s", 5.0))
        with self._lock:
            holder_peer = self._peers.get(holder_node)
            if holder_peer is not None and (now - holder_peer.last_seen <= heartbeat_timeout):
                if camera_id not in holder_peer.held_cameras:
                    with self._self_lock:
                        is_active_local = camera_id in self._self_state.active_cameras
                    if is_active_local:
                        logger.info(
                            "[PeerOrch][Reclaim] Holder '%s' no longer has '%s' and stream active locally; clearing reclaim mapping.",
                            holder_node, camera_id,
                        )
                        self._migrated_out.pop(camera_id, None)
                        self._reclaim_in_progress.discard(camera_id)
                        self._reclaim_retry_at.pop(camera_id, None)
                        self._reclaim_retry_count.pop(camera_id, None)
                        return

        with self._lock:
            event = self._pending_acks.get(camera_id)
            if event is None:
                event = threading.Event()
                self._pending_acks[camera_id] = event

        timeout = self._cfg.get("migration_timeout_s", 15.0)
        confirmed = event.wait(timeout=timeout)

        with self._lock:
            self._pending_acks.pop(camera_id, None)

        if not confirmed:
            base_retry_s = float(self._cfg.get("reclaim_retry_s", 5.0))
            cooldown_s = float(self._cfg.get("cooldown_s", 6.0))
            if cooldown_s <= 0.0:
                cooldown_s = float("inf")
            max_backoff_s = float(self._cfg.get("reclaim_max_backoff_s", 30.0))

            with self._lock:
                max_retries = int(self._cfg.get("reclaim_max_retries", 3))
                attempts = self._reclaim_attempts.get(camera_id, 0) + 1
                self._reclaim_attempts[camera_id] = attempts
                current_retries = self._reclaim_retry_count.get(camera_id, 0) + 1
                self._reclaim_retry_count[camera_id] = current_retries
                self._reclaim_in_progress.discard(camera_id)
                self._pending_epochs.pop(camera_id, None)
                # reclaim_max_retries means give up fast retry (edge_node.yml):
                # park 300s instead of churning ADD/REMOVE on the live
                # pipeline every ~17s. Tracking retained, no orphan.
                if attempts >= max_retries:
                    backoff_s = 300.0
                else:
                    backoff_s = min(base_retry_s * (2 ** (current_retries - 1)), max_backoff_s)
                self._reclaim_retry_at[camera_id] = time.time() + backoff_s

            if attempts >= max_retries:
                logger.error(
                    "[PeerOrch] Reclaim: ADD for '%s' failed %d/%d fast attempts (holder '%s') — parking 300s, tracking retained.",
                    camera_id, attempts, max_retries, holder_node,
                )
                return

            logger.warning(
                "[PeerOrch] Reclaim: TIMEOUT (%ds) waiting for local ADD ack of '%s' from holder '%s' "
                "(active=%d, genuine-failures=%d) — retrying in %.1fs, tracking retained.",
                int(timeout), camera_id, holder_node,
                len(self._self_state.active_cameras), attempts, backoff_s,
            )
            return

        # Stream confirmed PLAYING on self
        with self._lock:
            if camera_id in self._pending_epochs:
                self._camera_epochs[camera_id] = self._pending_epochs.pop(camera_id)
        # Check if holder is known offline/dead. If so, skip sending REMOVE.
        now = time.time()
        timeout = self._cfg.get("heartbeat_timeout_s", 5.0)
        grace_s = self._cfg.get("failover_grace_s", timeout)
        offline_threshold = timeout + grace_s

        with self._lock:
            holder_peer = self._peers.get(holder_node)
            is_dead = (
                holder_node == "orphan"  # adoption gate: confirmed holderless
                or holder_node in self._failover_triggered
                or holder_node in self._peer_offline_at
                or (holder_peer is not None and (now - holder_peer.last_seen > offline_threshold))
            )

        if is_dead:
            with self._lock:
                self._migrated_out.pop(camera_id, None)
                self._reclaim_in_progress.discard(camera_id)
                self._reclaim_retry_at.pop(camera_id, None)
                self._reclaim_retry_count.pop(camera_id, None)
                self._reclaim_attempts.pop(camera_id, None)
                self._reclaim_pending_remove.pop(camera_id, None)
            logger.info(
                "[PeerOrch] Reclaim: stream PLAYING on self — skipped REMOVE to '%s' (holder dead/offline). Reclaim complete.",
                holder_node,
            )
            self._migration_log.log(
                holder_node, self._node_id, camera_id,
                "reclaim", getattr(self._self_state, "load_score", 0.0), None,
                0.0, "RECLAIMED",
            )
            return

        # Re-check ownership immediately before REMOVE. Heartbeat state may
        # have changed while the local ADD was waiting for PLAYING.
        with self._lock:
            holder_peer = self._peers.get(holder_node)
            holder_still_has_camera = bool(
                holder_peer is None or camera_id in holder_peer.active_cameras
            )
        if not holder_still_has_camera:
            with self._lock:
                self._migrated_out.pop(camera_id, None)
                self._reclaim_in_progress.discard(camera_id)
                self._reclaim_retry_at.pop(camera_id, None)
                self._reclaim_retry_count.pop(camera_id, None)
                self._reclaim_attempts.pop(camera_id, None)
                self._reclaim_pending_remove.pop(camera_id, None)
            logger.info(
                "[PeerOrch] Reclaim: holder '%s' no longer owns '%s'; skipped REMOVE.",
                holder_node, camera_id,
            )
            self._migration_log.log(
                holder_node, self._node_id, camera_id,
                "reclaim", getattr(self._self_state, "load_score", 0.0), None,
                0.0, "RECLAIMED",
            )
            return

        # Holder is alive — safe to remove from holder
        # Note: on holder node, source_id cannot be locally resolved, so we query holder peer camera_configs
        holder_sid: Optional[int] = None
        holder_epoch: Optional[int] = None
        holder_mig_id: Optional[str] = None
        with self._lock:
            holder_p = self._peers.get(holder_node)
            if holder_p and isinstance(holder_p.camera_configs, dict):
                holder_c = holder_p.camera_configs.get(camera_id)
                if isinstance(holder_c, dict) and "source_id" in holder_c:
                    try:
                        holder_sid = int(holder_c["source_id"])
                    except (ValueError, TypeError):
                        pass
            holder_epoch = self._camera_epochs.get(camera_id)
            holder_mig_id = self._camera_migration_ids.get(camera_id)

        remove_cmd: dict = {"cmd": "REMOVE", "camera_id": camera_id}
        if holder_sid is not None:
            remove_cmd["source_id"] = holder_sid
        else:
            logger.debug(
                "[PeerOrch] Reclaim REMOVE for '%s' to '%s' emitted without source_id (holder config not reporting source_id).",
                camera_id, holder_node,
            )
        if holder_epoch is not None:
            remove_cmd["epoch"] = holder_epoch
        if holder_mig_id is not None:
            remove_cmd["migration_id"] = holder_mig_id

        # Register dedicated remove ACK event before sending REMOVE
        remove_ack_event = threading.Event()
        with self._lock:
            self._reclaim_remove_acks[camera_id] = remove_ack_event
            self._reclaim_pending_remove[camera_id] = holder_node
            if holder_epoch is not None:
                self._reclaim_pending_remove_epoch[camera_id] = holder_epoch
            if holder_mig_id is not None:
                self._reclaim_pending_remove_mig_id[camera_id] = holder_mig_id

        # P5: stamp recipient's boot_id so a pre-reboot REMOVE is fenced.
        self._attach_lease_fields(remove_cmd, holder_node)

        holder_control_key = f"peers/control/{holder_node}"
        remove_sent = False
        try:
            if self._session is not None:
                self._session.put(
                    holder_control_key,
                    msgpack.packb(remove_cmd, use_bin_type=True),
                )
                remove_sent = True
        except Exception as exc:
            logger.error("[PeerOrch] Reclaim: failed to send REMOVE to '%s' for '%s': %s", holder_node, camera_id, exc)

        remove_confirmed = False
        if remove_sent:
            remove_timeout = float(self._cfg.get("remove_ack_timeout_s", 10.0))
            remove_confirmed = remove_ack_event.wait(timeout=remove_timeout)

        with self._lock:
            self._reclaim_remove_acks.pop(camera_id, None)
            self._reclaim_pending_remove.pop(camera_id, None)
            self._reclaim_pending_remove_epoch.pop(camera_id, None)
            self._reclaim_pending_remove_mig_id.pop(camera_id, None)

        if remove_confirmed:
            with self._lock:
                self._migrated_out.pop(camera_id, None)
                self._reclaim_in_progress.discard(camera_id)
                self._reclaim_retry_at.pop(camera_id, None)
                self._reclaim_retry_count.pop(camera_id, None)
                self._reclaim_attempts.pop(camera_id, None)
            logger.info(
                "[PeerOrch] Reclaim: stream PLAYING on self and REMOVE ACK confirmed from '%s' for '%s'. Reclaim complete.",
                holder_node, camera_id,
            )
            self._migration_log.log(
                holder_node, self._node_id, camera_id,
                "reclaim", getattr(self._self_state, "load_score", 0.0), None,
                0.0, "RECLAIMED",
            )
            # F2: After reclaim-complete, the rtspclientsink may have been rejected
            # by MediaMTX overridePublisher:no during the MBB overlap window
            # (holder's publisher was still active when reclaimer connected).
            # The rejection is silent — no publisher_failure bus message is emitted
            # during state-change — so no retry was ever scheduled.
            # Fix: schedule _rebuild_publisher_branch 3s after REMOVE-ACK (path is
            # guaranteed free by then) so any rejected rtspclientsink is recovered.
            # This is a no-op when the push branch is already PLAYING.
            _rebuild = getattr(self, "_rebuild_publisher_branch", None)
            _cam_mgr = getattr(self, "_camera_manager", None)
            if _rebuild is not None and _cam_mgr is not None:
                _src_id = None
                try:
                    _cfg_obj = _cam_mgr.get_config_by_camera_id(camera_id)
                    if _cfg_obj is not None:
                        _src_id = _cfg_obj.source_id
                except Exception:
                    pass
                if _src_id is not None:
                    def _deferred_rebuild(cam=camera_id, sid=_src_id):
                        try:
                            _rebuild(sid)
                            logger.info(
                                "[PeerOrch] Reclaim post-check: publisher recovery triggered for '%s' (source_id=%d)",
                                cam, sid,
                            )
                        except Exception as _exc:
                            logger.debug(
                                "[PeerOrch] Reclaim post-check: publisher recovery skipped for '%s': %s",
                                cam, _exc,
                            )
                    threading.Timer(3.0, _deferred_rebuild).start()
        else:
            base_retry_s = float(self._cfg.get("reclaim_retry_s", 5.0))
            cooldown_s = float(self._cfg.get("cooldown_s", 6.0))
            if cooldown_s <= 0.0:
                cooldown_s = float("inf")
            with self._lock:
                max_retries = int(self._cfg.get("reclaim_max_retries", 3))
                current_retries = self._reclaim_retry_count.get(camera_id, 0) + 1
                self._reclaim_retry_count[camera_id] = current_retries
                self._reclaim_in_progress.discard(camera_id)
                # Same give-up semantics as the ADD-timeout path above.
                if current_retries >= max_retries:
                    backoff_s = 300.0
                else:
                    backoff_s = min(base_retry_s * (2 ** (current_retries - 1)), cooldown_s)
                self._reclaim_retry_at[camera_id] = time.time() + backoff_s

            logger.error(
                "[PeerOrch] Reclaim: REMOVE unconfirmed (timeout or send failure) to '%s' for '%s' — scheduling retry in %.1fs (retry=%d)",
                holder_node, camera_id, backoff_s, current_retries,
            )

    def _on_vote_ack(self, payload: dict) -> None:
        """Receive ack that stream is PLAYING on winner node.

        Phase 3 review fix 4 — the ACK is authenticated against the expected
        winner/migration identity before it can release the reservation or
        trigger the REMOVE: the payload ``node_id`` (the winner, as published
        by ZenohCommandSubscriber) must equal the stored ``_pending_winner``
        for that camera.  A stale, wrong, or duplicate ACK therefore cannot
        release a reservation it does not own, and cannot set the pending-ack
        event (which is what drives the requester's REMOVE in _wait_and_remove).
        Fail-closed for ambiguous ACKs; safe legacy compatibility: a legacy ACK
        without ``node_id`` is only honoured when no migration is pending for
        that camera (i.e. there is nothing to authenticate against).
        """
        camera_id = payload.get("camera_id", "")
        if not camera_id:
            return
        ack_node = payload.get("node_id", "")
        event_type = payload.get("event")

        if event_type not in (None, "PLAYING", "REJECTED"):
            logger.warning(
                "[PeerOrch] Ignoring ACK for '%s': event='%s'.",
                camera_id, event_type,
            )
            return

        if event_type == "REJECTED":
            self._on_add_rejected(payload)
            return

        with self._lock:
            expected_winner = self._pending_winner.get(camera_id)
            event = self._pending_acks.get(camera_id)

            ack_epoch = payload.get("epoch")
            ack_mig_id = payload.get("migration_id")
            pending_epoch = self._pending_epochs.get(camera_id)
            pending_mig_id = self._pending_migration_ids.get(camera_id)

            if pending_epoch is not None and ack_epoch is not None:
                try:
                    if int(ack_epoch) != int(pending_epoch):
                        logger.warning(
                            "[PeerOrch] Ignoring stale ACK for '%s': ack epoch=%r != pending epoch=%r",
                            camera_id, ack_epoch, pending_epoch,
                        )
                        return
                except (ValueError, TypeError):
                    return
            if pending_mig_id is not None and ack_mig_id is not None:
                if str(ack_mig_id) != str(pending_mig_id):
                    logger.warning(
                        "[PeerOrch] Ignoring mismatched ACK for '%s': ack mig_id=%r != pending mig_id=%r",
                        camera_id, ack_mig_id, pending_mig_id,
                    )
                    return

            if expected_winner is not None and ack_node not in ("", expected_winner):
                # Wrong/forged/stale sender for an in-flight migration — fail closed.
                logger.debug(
                    "[PeerOrch] Ignoring ACK for '%s': sender='%s' != expected winner='%s'.",
                    camera_id, ack_node, expected_winner,
                )
                return
            if expected_winner is None and ack_node not in ("", self._node_id):
                # No pending migration but a foreign sender claims this camera —
                # ambiguous, fail closed (do not set event, do not release).
                logger.debug(
                    "[PeerOrch] Ignoring ACK for '%s': no pending migration but "
                    "sender='%s' != self.", camera_id, ack_node,
                )
                return

            # Authenticated — now atomically claim the reservation if we own it.
            winner_id = self._pending_winner.pop(camera_id, None)
            self._pending_started_at.pop(camera_id, None)
            self._pending_epochs.pop(camera_id, None)
            self._pending_migration_ids.pop(camera_id, None)
            if winner_id is not None:
                self._peer_inflight[winner_id] = max(
                    0, self._peer_inflight.get(winner_id, 0) - 1
                )
                self._peer_consecutive_timeouts.pop(winner_id, None)
            elif ack_node:
                self._peer_consecutive_timeouts.pop(ack_node, None)
            if event is not None:
                event.set()

        if winner_id is not None:
            logger.info(
                "[PeerOrch] Ack received for '%s' from '%s' — stream is PLAYING. "
                "Reservation released (inflight=%d).",
                camera_id, ack_node, self._peer_inflight.get(winner_id, 0),
            )
        elif event is not None:
            logger.info("[PeerOrch] Ack received for '%s' — stream is PLAYING.", camera_id)

    def _on_add_rejected(self, payload: dict) -> None:
        """P2: winner rejected ADD as stale_epoch. Fast-forward our epoch past its
        held_epoch and re-dispatch immediately — no penalty, no 15s timeout."""
        camera_id = payload.get("camera_id", "")
        ack_node  = payload.get("node_id", "")
        held      = payload.get("held_epoch")
        ack_mig   = payload.get("migration_id")
        if not camera_id or held is None:
            return
        with self._lock:
            expected = self._pending_winner.get(camera_id)
            pend_mig = self._pending_migration_ids.get(camera_id)
            if expected is None or ack_node not in ("", expected):
                return
            if pend_mig is not None and ack_mig is not None and str(ack_mig) != str(pend_mig):
                return
            new_epoch = max(self._camera_epochs.get(camera_id, 0), int(held) + 1)
            self._camera_epochs[camera_id] = new_epoch
            winner_id = self._pending_winner.pop(camera_id, None)
            self._pending_started_at.pop(camera_id, None)
            self._pending_epochs.pop(camera_id, None)
            self._pending_migration_ids.pop(camera_id, None)
            if winner_id is not None:
                self._peer_inflight[winner_id] = max(0, self._peer_inflight.get(winner_id, 0) - 1)
                self._peer_consecutive_timeouts.pop(winner_id, None)
            event = self._pending_acks.get(camera_id)
            self._add_rejected[camera_id] = True
        if self._lease is not None:
            try:
                self._lease.record_epoch(camera_id, new_epoch)
            except Exception as exc:
                logger.warning("[PeerOrch][P5] persist epoch after reject failed '%s': %s", camera_id, exc)
        logger.info(
            "[PeerOrch] ADD rejected by '%s' for '%s' (held=%s); fast-forward epoch->%d, retrying.",
            ack_node, camera_id, held, new_epoch,
        )
        if event is not None:
            event.set()

    def _on_remove_ack(self, payload: dict) -> None:
        """
        Handle incoming REMOVE ACK on peers/remove/ack/{cam_id}.
        Validates camera_id, source_id, epoch, and migration_id against in-flight reclaim/remove requests.
        """
        cam_id = payload.get("camera_id") or payload.get("cam_id")
        if not cam_id:
            return
        ack_node = payload.get("node_id", "")
        event_type = payload.get("event")
        if event_type is not None and event_type != "REMOVED":
            logger.warning(
                "[PeerOrch] Ignoring remove ACK for '%s': event='%s' != 'REMOVED'.",
                cam_id, event_type,
            )
            return

        with self._lock:
            pending_holder = self._reclaim_pending_remove.get(cam_id)
            event = self._reclaim_remove_acks.get(cam_id)
            if event is None:
                logger.debug("[PeerOrch] No pending remove ACK event for '%s', ignoring.", cam_id)
                return

            if pending_holder is not None and ack_node not in ("", pending_holder):
                logger.warning(
                    "[PeerOrch] Ignoring remove ACK for '%s': sender='%s' != pending holder='%s'.",
                    cam_id, ack_node, pending_holder,
                )
                return

            ack_epoch = payload.get("epoch")
            ack_mig_id = payload.get("migration_id")
            pending_epoch = self._reclaim_pending_remove_epoch.get(cam_id)
            pending_mig_id = self._reclaim_pending_remove_mig_id.get(cam_id)

            if pending_epoch is not None and ack_epoch is not None:
                try:
                    if int(ack_epoch) != int(pending_epoch):
                        logger.warning(
                            "[PeerOrch] Ignoring stale remove ACK for '%s': ack epoch=%r != pending epoch=%r",
                            cam_id, ack_epoch, pending_epoch,
                        )
                        return
                except (ValueError, TypeError):
                    return
            if pending_mig_id is not None and ack_mig_id is not None:
                if str(ack_mig_id) != str(pending_mig_id):
                    logger.warning(
                        "[PeerOrch] Ignoring mismatched remove ACK for '%s': ack mig_id=%r != pending mig_id=%r",
                        cam_id, ack_mig_id, pending_mig_id,
                    )
                    return

            event.set()
            logger.info("[PeerOrch] Remove ACK confirmed for '%s' from '%s'", cam_id, ack_node)
