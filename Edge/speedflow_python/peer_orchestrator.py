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

import threading
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

from .membership import MembershipMixin
from .ownership import OwnershipMixin
from .offload import OffloadMixin
from .rescue import RescueMixin
from .pipeline_commands import PipelineCommandsMixin
from .lease_state import _is_int


class PeerOrchestrator(
    MembershipMixin,
    OwnershipMixin,
    OffloadMixin,
    RescueMixin,
    PipelineCommandsMixin,
):
    """P2P version of MasterOrchestrator — runs on each Edge Node.

    Communicates via Zenoh (peer mode, key expressions).

    Mechanical decomposition (P4): methods are organized across mixin modules
    (membership / ownership / offload / rescue / pipeline_commands). Behavior,
    callbacks, locks, and P0-P3 semantics are preserved unchanged.
    """

    # Class-level constant referenced by self.BLOCKED_LOG_COOLDOWN (relocated
    # verbatim from the original class body).
    BLOCKED_LOG_COOLDOWN = 60.0

    def __init__(
        self,
        node_id: str,
        cfg: dict,
        camera_manager: object,
        camera_configs_dir: Optional[Path] = None,
        lease_state_path: Optional[Path] = None,
    ) -> None:
        self._node_id = node_id
        self._cfg = cfg
        self._camera_manager = camera_manager

        # P5 — Edge lease persistence / boot fencing.
        # The orchestrator refuses to join the mesh until this state persists.
        # If persistence fails, startup fails closed (see _init_lease_state).
        self._lease_state_path = (
            Path(lease_state_path)
            if lease_state_path is not None
            else (Path(__file__).resolve().parent.parent / "lease_state.json")
        )
        self._lease = None
        self._boot_id = 0

        # Startup validation: add_ack_timeout_s must be strictly less than migration_timeout_s
        migration_timeout = float(self._cfg.get("migration_timeout_s", 15.0))
        if "add_ack_timeout_s" in self._cfg and self._cfg["add_ack_timeout_s"] is not None:
            ack_timeout = float(self._cfg["add_ack_timeout_s"])
            if ack_timeout >= migration_timeout:
                raise ValueError(
                    f"add_ack_timeout_s ({ack_timeout:.1f}s) must be strictly less than "
                    f"migration_timeout_s ({migration_timeout:.1f}s) to ensure sender timeout safety"
                )

        # Peer state
        self._peers: Dict[str, PeerState] = {}
        self._lock = threading.RLock()

        # State of this node itself (updated from our own peers/status/+)
        self._self_state = PeerState(node_id=node_id)
        # BUG-1 fix: separate lock for _self_state so the Zenoh callback
        # thread and the decision loop never see a torn write.
        self._self_lock = threading.RLock()

        # Migration log — relative to Edge/logs/
        log_dir = _ROOT / "logs"
        self._migration_log = MigrationLogger(log_dir / "p2p_migrations.csv")

        # Cooldown per-camera: camera_id → timestamp of most recent migration
        self._cam_cooldown: Dict[str, float] = {}

        # Vote windows: camera_id → list[proposal]
        self._vote_windows: Dict[str, List[dict]] = {}
        self._vote_timers: Dict[str, threading.Timer] = {}

        # Status/heartbeat publish failure tracking
        self._status_sent_count = 0
        self._status_error_count = 0
        self._status_consecutive_errors = 0
        self._last_status_sent_time: Optional[float] = None
        self._last_status_error_time: Optional[float] = None
        # Cameras with RFO sent but vote window still open (prevent re-trigger)
        self._vote_in_progress: set = set()
        # RFO trigger snapshots: camera_id -> (trigger_load, trigger_fps)
        self._rfo_snapshots: Dict[str, Tuple[float, Optional[float]]] = {}

        # Pending ack events for Make-before-Break
        self._pending_acks: Dict[str, threading.Event] = {}
        self._add_rejected: Dict[str, bool] = {}
        self._reject_retries: Dict[str, int] = {}

        # Mandatory ladder L0->L1->L2 state
        self._ladder_l1_since: Optional[float] = None
        self._ladder_l1_camera: Optional[str] = None

        # Camera config lookup — relative to Edge/configs/
        if camera_configs_dir is None:
            camera_configs_dir = _ROOT / "configs"
        self._camera_configs_dir = camera_configs_dir

        # Zenoh session + publishers
        self._session = None
        self._pubs: dict = {}
        self._running = False
        self._decision_thread: Optional[threading.Thread] = None

        # Track peers already reported offline (no cameras) to avoid log spam
        self._notified_offline: set = set()

        # BUG-6 fix: track which dead peers have already had failover triggered
        # so we don't re-trigger on the next decision-loop tick.  We no longer
        # clear active_cameras on the dead peer's PeerState (that would race
        # with _check_rebalance reading it to detect camera returns).
        self._failover_triggered: Dict[str, float] = {}

        # Rescue claims received from other nodes: (dead_node_id, camera_id) → (node_id, local_received_ts, priority_weight, remote_ts)
        self._failover_claims: Dict[Tuple[str, str], Tuple[str, float, int, float]] = {}
        self._claims_lock = threading.Lock()

        # Cameras rescued via failover: camera_id → original_owner_node_id
        # Used to return cameras when the original owner comes back online.
        self._rescued_cameras: Dict[str, str] = {}
        # Timestamps when cameras were rescued: camera_id → timestamp
        self._rescued_at: Dict[str, float] = {}

        # P2: per-camera source-liveness gating for failover rescue.
        # Before committing a rescue ADD we verify the camera SOURCE can
        # actually produce/accept a stream. State keyed by camera_id:
        #   {"status": "reachable"|"unreachable"|"pending",
        #    "last_probe_ts": float, "next_probe_ts": float, "attempts": int}
        # We distinguish SOURCE_UNREACHABLE (probe definitively failed, bounded
        # attempts exhausted) from rescue-pending (still probing / backoff in
        # effect) and use slow bounded backoff so a flaky source never drives
        # rescue ADD/REMOVE loops. Driven entirely by the existing _measure_rtt
        # capability — no new deps, no central controller.
        self._source_liveness: Dict[str, dict] = {}
        self._source_liveness_lock = threading.Lock()

        # Duplicate camera observation counts for safe duplicate reconciliation:
        # (peer_node_id, camera_id) -> count of consecutive heartbeat observations
        self._duplicate_camera_seen: Dict[Tuple[str, str], int] = {}

        # Cameras migrated away due to overload: camera_id → winner_node_id
        # Used to reclaim cameras when this node's load drops below threshold.
        self._migrated_out: Dict[str, str] = {}

        # Per-camera migration timestamp history for bounce dampening: camera_id -> list[timestamp]
        self._cam_migration_history: Dict[str, List[float]] = {}

        # Consecutive migration ACK timeouts per target peer: node_id -> count
        self._peer_consecutive_timeouts: Dict[str, int] = {}

        # Phase 3 — bounded in-flight reservation accounting.
        #
        # Sender side: _peer_inflight[peer_node_id] counts how many L1 stream
        # migrations have been decided (decision published) toward that peer but
        # whose ADD ack has not yet arrived.  _pick_best_peer adds this to
        # len(peer.active_cameras) before applying the capacity gate so two
        # simultaneous RFOs cannot both pick the same already-full peer.
        # Decremented on ACK (stream PLAYING) or on migration timeout (rollback).
        # ponytail: per-peer int rather than per-camera set — O(1), no camera-id
        # coupling, naturally collapses when the reservation resolves.
        self._peer_inflight: Dict[str, int] = {}

        # Phase 3 — sender-side camera→winner mapping used to decrement
        # _peer_inflight on ACK or timeout.  Populated in _close_vote_window;
        # cleared in _on_vote_ack (ACK path) and _wait_and_remove (timeout path).
        self._pending_winner: Dict[str, str] = {}
        self._pending_started_at: Dict[str, float] = {}
        self._pending_rescue: Dict[str, Any] = {}
        self._pending_rescue_at: Dict[str, float] = {}

        # Receiver side: count of bids sent but whose ADD command has not yet
        # arrived.  ε1 in _evaluate_and_bid gates on
        # current_streams + _self_inflight >= eps_streams_max so a node with 3
        # streams that has already bid on one RFO won't bid on a second and
        # overflow capacity.  Decayed by a timer (vote_window_s +
        # migration_timeout_s) since the ADD command goes to ZenohCommandSubscriber
        # rather than back through this orchestrator — timeout-only decay is the
        # conservative safe path.
        self._self_inflight: int = 0
        self._self_inflight_lock = threading.Lock()  # separate from _lock to avoid nesting

        # Reclaim post-return observation window: camera_id → timestamp of reclaim
        # completion.  Prevents the reclaimed camera from being immediately
        # migrated away again during its transient FPS warm-up window on this node.
        self._reclaim_completed_at: Dict[str, float] = {}

        # Round-robin counter for baseline experiment policy
        self._rr_counter: int = 0

        # Reclaim retry tracking: camera_id → timestamp when reclaim can be retried
        self._reclaim_retry_at: Dict[str, float] = {}
        self._reclaim_retry_count: Dict[str, int] = {}
        self._reclaim_attempts: Dict[str, int] = {}
        self._reclaim_pending_remove: Dict[str, str] = {}
        # Dead-branch watchdog (2026-09-18): owned-held camera with zero FPS
        # for stall_timeout_s gets a local REMOVE+ADD restart. Tracks first
        # zero-FPS time, consecutive restart failures, and park deadline.
        self._branch_zero_since: Dict[str, float] = {}
        self._branch_restart_fails: Dict[str, int] = {}
        self._branch_restart_parked_until: Dict[str, float] = {}
        # Cameras currently undergoing reclaim Make-before-Break
        self._reclaim_in_progress: set = set()
        # Tracking camera owner epochs and active migration IDs
        self._camera_epochs: Dict[str, int] = {}
        self._camera_migration_ids: Dict[str, str] = {}
        self._pending_migration_ids: Dict[str, str] = {}
        self._pending_epochs: Dict[str, int] = {}
        self._reclaim_pending_remove_epoch: Dict[str, int] = {}
        self._reclaim_pending_remove_mig_id: Dict[str, str] = {}
        self._reclaim_remove_acks: Dict[str, threading.Event] = {}

        # ponytail: rate-limit blocked-decision diagnostics so they don't spew
        # every 1-second tick.  Keys are short reason strings; value is the Unix
        # timestamp of the last log.  Cooldown = 15 s (half a typical vote window);
        # increase to 30 s if log volume is still too high.
        self._blocked_logged_at: Dict[str, float] = {}

        # ponytail: single settle deadline that suppresses ALL L1/L2 overload
        # actions after a migration completes or reclaim ADD is initiated.
        # Stale/draining FPS samples can otherwise escalate the node during the
        # transition.  Configurable via p2p.transition_settle_s (default 5.0 s).
        self._transition_settle_until: float = 0.0

        # Timestamp when load first dropped below reclaim threshold (for stability check)
        self._reclaim_eligible_since: Optional[float] = None

        # Penalty timestamp for this node itself (set on migration timeout rollback)
        self._self_penalty_until: float = 0.0

        # ponytail: track when self FIRST observed valid positive FPS (startup
        # warmup gate).  Sticky — set on the first update that contains real
        # measurements, never reset, so the gate measures "time since valid FPS
        # first appeared" rather than "current sample validity".  Suppresses
        # overload escalation for overload_warmup_s (default 10 s) after start.
        self._self_first_valid_fps_at: Optional[float] = None

        # ponytail: per-camera "ADD timestamp".  Set whenever we issue an ADD
        # command (winner of vote, leaderless failover, reclaim).  Cameras NOT
        # in this dict are assumed pre-existing and skip the per-camera warmup
        # gate — this keeps the existing L1/L2 selector tests passing
        # without forcing every test to set a fake ADD timestamp.
        self._camera_added_at: Dict[str, float] = {}

        # Track when any peer was detected offline (node_id -> timestamp)
        # to ensure failover_convergence_grace activates for all peer departures.
        self._peer_offline_at: Dict[str, float] = {}

        # ponytail: per-camera "first valid positive FPS" timestamp.  Sticky
        # within the lifetime of the dict entry; cleared (overwritten) when we
        # ADD the camera again so the warmup restarts.
        self._camera_first_valid_fps_at: Dict[str, float] = {}

        # -----------------------------------------------------------------------
        # Offload level table — shared with SpeedProbe (read from probe thread).
        # Maps camera_id → offload level (0=L0 local, 1=L1 plate-crop, 2=L2 full-stream (lease-tracked, not table-written)).
        # Written only by the decision loop; read-only from the probe.
        # Protected by _offload_lock (separate from _lock to avoid deadlock with
        # the Zenoh callback thread which holds _lock).
        # -----------------------------------------------------------------------
        self._offload_table: Dict[str, int] = {}
        self._offload_lock = threading.RLock()
        # Phase 3: node-local LPR worker queue saturation (0.0..1.0), fed from
        # SpeedProbe telemetry; drives L1 plate-crop offload escalation (source offload_level==1).
        self._lpr_queue_ratio: float = 0.0
        self._lpr_queue_depth: int = 0
        self._lpr_over_thr_since: Optional[float] = None  # sustain-timer anchor
        self._lpr_sustained_logged: bool = False  # log LPR SUSTAINED once per episode
        self._lpr_reclaim_at: Dict[str, float] = {}  # per-camera reclaim cooldown

        # Per-camera timestamp of the last offload-level change (for cooldown)
        self._offload_level_changed_at: Dict[str, float] = {}

        # Target peer for each camera's offload (camera_id → node_id or "")
        self._offload_targets: Dict[str, str] = {}

        # Per-camera timestamp when offload (L1/L2) was started
        self._offload_started_at: Dict[str, float] = {}

        # Timestamp since self load dropped below overload threshold (for de-escalation dwell)
        self._below_thr_since: Optional[float] = None

        # Step-7: migration-complete timestamps for Δτ computation
        # camera_id → unix timestamp when REMOVE was confirmed sent
        self._migration_complete_ts: Dict[str, float] = {}

        # Flag indicating whether the local pipeline has finished initializing / reached PLAYING state
        self._pipeline_ready: bool = False

        # Stop event for cleanly blocking start()
        self._stop_event = threading.Event()
        # Signaled once Zenoh session and publishers are ready
        self._ready_event = threading.Event()

        # Thread pool for blocking I/O (RTT measurement) off the Zenoh callback thread
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="PeerOrch-IO")

        # BUG-15: Cache cameras.yml to avoid repeated disk reads on every vote.
        # Reset to None to force reload only if the file changes (not implemented
        # here — a future improvement could use inotify or mtime check).
        self._cameras_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # P5 — Edge lease persistence / boot fencing
    # ------------------------------------------------------------------

    def _init_lease_state(self) -> None:
        """Load + persist lease state; gate startup on success (fail closed).

        Loads the persisted per-camera epoch high-water into the in-memory
        epoch counters so newly minted epochs are never lower than a prior
        boot's values. Bumps the monotonic boot_id and persists immediately.
        Raises if the state cannot be persisted — caller must not join.
        """
        from .lease_state import LeaseState

        self._lease = LeaseState(self._lease_state_path)
        self._lease.load(fail_safe_high=True)

        # Load persisted epoch high-water into the live epoch counters. This is
        # what guarantees "newly minted epochs cannot be lower than prior
        # values" across reboots.
        with self._lock:
            for cam_id, epoch in self._lease.camera_epochs.items():
                cur = self._camera_epochs.get(cam_id)
                if cur is None or epoch > cur:
                    self._camera_epochs[cam_id] = epoch

        # Monotonic boot_id bump + persist. Fail closed if we cannot persist:
        # a node that cannot record its boot epoch must not join the mesh.
        try:
            self._lease.bump_and_persist()
            self._boot_id = self._lease.boot_id
        except Exception as exc:
            raise RuntimeError(
                f"[PeerOrch] Lease state persistence failed at "
                f"{self._lease_state_path}; refusing to join (fail-closed): {exc}"
            ) from exc

        logger.info(
            "[PeerOrch][P5] Lease state initialized: boot_id=%d, "
            "epochs=%s, first_boot=%s, corrupt=%s",
            self._boot_id,
            self._lease.camera_epochs,
            self._lease.first_boot,
            self._lease.corrupt,
        )

    def _recipient_boot_id(self, node_id: str) -> Optional[int]:
        """Boot_id the recipient expects on a command addressed to ``node_id``.

        Self-directed commands use our own boot_id. Cross-node commands use the
        peer's last-observed boot_id (learned from its heartbeat). Returns None
        when unknown so the command falls back to the legacy epoch-only fence
        rather than being wrongly stamped.
        """
        if node_id == self._node_id:
            return self._boot_id
        peer = self._peers.get(node_id)
        if peer is not None:
            b = getattr(peer, "boot_id", 0)
            if _is_int(b) and b:
                return int(b)
        return None

    def _attach_lease_fields(self, cmd: dict, recipient_node_id: str) -> dict:
        """Attach the recipient's boot_id to an outgoing ADD/REMOVE command.

        Preserves the existing integer epoch wire field (added by callers) and
        adds boot_id for receiver fencing. Returns the same dict.

        When the recipient's boot_id is unknown (e.g. peer not yet seen), any
        previously stamped boot_id is removed so a stale/self boot_id is never
        left on a cross-node command — leaving it would either fail the
        receiver's fence or, worse, let a pre-reboot command through. Legacy
        receivers that only understand epoch fall back to the epoch-only fence.
        """
        boot_id = self._recipient_boot_id(recipient_node_id)
        if boot_id is not None:
            cmd["boot_id"] = boot_id
        else:
            # Unknown recipient boot_id: drop any existing boot_id rather than
            # leaving a wrong (e.g. self) one on a cross-node command.
            cmd.pop("boot_id", None)
        return cmd
