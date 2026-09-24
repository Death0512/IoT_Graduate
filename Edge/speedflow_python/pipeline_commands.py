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

"""Edge/speedflow_python/pipeline_commands.py

Pipeline control / Zenoh command mixin for PeerOrchestrator (P4).
Methods relocated verbatim; shared helpers live in membership.py.
"""


class PipelineCommandsMixin:
    def set_pipeline_ready(self, ready: bool = True) -> None:
        """Set local pipeline readiness status (called when pipeline enters PLAYING)."""
        with self._lock:
            self._pipeline_ready = bool(ready)
        logger.info("[PeerOrch] Pipeline ready state set to %s", self._pipeline_ready)

    def is_pipeline_ready(self) -> bool:
        """Check whether the local pipeline is ready."""
        with self._lock:
            return self._pipeline_ready

    def start(self) -> None:
        """Open Zenoh session, declare pubs/subs, start decision thread."""
        import zenoh

        # P5 — gate startup on lease persistence. If the state cannot be loaded
        # and persisted we must not join the mesh (fail closed). This runs
        # before the Zenoh session is opened, so a persistence failure leaves
        # the node fully detached.
        self._init_lease_state()

        self._session = make_session()
        logger.info("[PeerOrch] Zenoh session opened (peer mode).")

        if hasattr(self, "_migration_log") and hasattr(self._migration_log, "set_session"):
            self._migration_log.set_session(self._session)

        # Declared publishers
        self._pubs["status"]        = self._session.declare_publisher(f"peers/status/{self._node_id}")
        self._pubs["vote_request"]  = self._session.declare_publisher("peers/vote/request")
        self._pubs["vote_proposal"] = self._session.declare_publisher("peers/vote/proposal")
        self._pubs["vote_decision"] = self._session.declare_publisher("peers/vote/decision")
        self._pubs["control"]       = self._session.declare_publisher(f"peers/control/{self._node_id}")
        self._pubs["failover_claim"] = self._session.declare_publisher("peers/failover/claim")

        # Subscribe to all P2P topics
        self._session.declare_subscriber("peers/status/**",      self._on_sample)
        self._session.declare_subscriber("peers/vote/request",   self._on_sample)
        self._session.declare_subscriber("peers/vote/proposal",  self._on_sample)
        self._session.declare_subscriber("peers/vote/decision",  self._on_sample)
        self._session.declare_subscriber("peers/vote/ack/**",    self._on_sample)
        self._session.declare_subscriber("peers/remove/ack/**",  self._on_sample)
        self._session.declare_subscriber("peers/failover/claim", self._on_sample)
        logger.info("[PeerOrch] Subscribed to: peers/status/**, peers/vote/*, peers/vote/ack/**, peers/remove/ack/**, peers/failover/claim")

        self._running = True
        self._ready_event.set()

        # Startup preemption announcement: notify cluster of owned cameras so peers holding them release immediately
        self._publish_startup_announcement()

        self._decision_thread = threading.Thread(
            target=self._decision_loop,
            name=f"PeerDecision-{self._node_id}",
            daemon=True,
        )
        logger.info("[Thread] Starting PeerDecision thread: name=%s, mono_ts=%.6f", self._decision_thread.name, time.monotonic())
        self._decision_thread.start()

        # Park — Zenoh peer mode needs no blocking loop
        self._stop_event.wait()

    def publish_status(self, payload: bytes) -> None:
        """Publish health status on peers/status/<node_id> (called by health push loop)."""
        pub = self._pubs.get("status")
        if pub:
            t_pub_start = time.monotonic()
            seq = self._status_sent_count + self._status_error_count + 1
            logger.debug("[PeerOrch] Status publish attempt: seq=%d, mono_ts=%.6f", seq, t_pub_start)
            try:
                pub.put(payload)
                t_pub_end = time.monotonic()
                self._status_sent_count += 1
                self._status_consecutive_errors = 0
                self._last_status_sent_time = time.time()
                logger.debug("[PeerOrch] Status publish success: seq=%d, dur_ms=%.2f, mono_ts=%.6f", seq, (t_pub_end - t_pub_start) * 1000.0, t_pub_end)
            except Exception as exc:
                t_pub_err = time.monotonic()
                self._status_error_count += 1
                self._status_consecutive_errors += 1
                self._last_status_error_time = time.time()
                logger.warning(
                    "[PeerOrch] Status publish failure: seq=%d, consecutive=%d, total=%d, err=%s, dur_ms=%.2f, mono_ts=%.6f",
                    seq, self._status_consecutive_errors, self._status_error_count, exc, (t_pub_err - t_pub_start) * 1000.0, t_pub_err,
                )

    def update_self_state(self, payload: dict) -> None:
        """Update this node's local state without publishing a Zenoh heartbeat.

        Thin compatibility wrapper routing directly to _on_peer_status(payload).
        """
        if not isinstance(payload, dict):
            return
        if "node_id" not in payload:
            payload = dict(payload)
            payload["node_id"] = self._node_id
        self._on_peer_status(payload)

    def stop(self) -> None:
        """Stop orchestrator."""
        self._running = False
        self._stop_event.set()
        if self._session:
            self._session.close()
        for timer in self._vote_timers.values():
            timer.cancel()
        self._vote_timers.clear()
        self._executor.shutdown(wait=False)

    def _on_sample(self, sample) -> None:
        """Route incoming Zenoh samples by key expression."""
        try:
            payload = msgpack.unpackb(sample.payload.to_bytes(), raw=False)
        except Exception:
            return

        key = str(sample.key_expr)

        if key.startswith("peers/status/"):
            self._on_peer_status(payload)
        elif key == "peers/vote/request":
            self._on_vote_request(payload)
        elif key == "peers/vote/proposal":
            self._on_vote_proposal(payload)
        elif key == "peers/vote/decision":
            self._on_vote_decision(payload)
        elif key.startswith("peers/vote/ack/"):
            self._on_vote_ack(payload)
        elif key.startswith("peers/remove/ack/"):
            self._on_remove_ack(payload)
        elif key == "peers/failover/claim":
            self._on_failover_claim(payload)
        else:
            logger.debug("[PeerOrch] Unknown key: %s", key)

    def _safe_submit(self, fn, *args, **kwargs):
        """Submit to executor safely; silently ignores if executor is shut down."""
        try:
            return self._executor.submit(fn, *args, **kwargs)
        except RuntimeError:
            logger.debug("[PeerOrch] Executor already shut down; dropped async task.")
            return None

    def _build_remove_cmd(self, camera_id: str, context: str = "") -> dict:
        """
        Build a REMOVE command dict for camera_id, attaching resolved source_id, epoch, and migration_id if available.
        """
        cmd: dict = {"cmd": "REMOVE", "camera_id": camera_id}
        sid = self._resolve_local_source_id(camera_id)
        if sid is not None:
            cmd["source_id"] = sid
        else:
            logger.info(
                "[PeerOrch] REMOVE for '%s' (%s) emitted without source_id: could not resolve active source_id.",
                camera_id, context or "unknown",
            )
        epoch = self._camera_epochs.get(camera_id)
        if epoch is not None:
            cmd["epoch"] = epoch
        mig_id = self._camera_migration_ids.get(camera_id) if hasattr(self, "_camera_migration_ids") else None
        if mig_id is not None:
            cmd["migration_id"] = mig_id
        # P5: stamp THIS node's boot_id on self-directed REMOVE so a pre-reboot
        # REMOVE cannot pass receiver fencing after a restart.
        if getattr(self, "_boot_id", 0):
            cmd["boot_id"] = self._boot_id
        return cmd
