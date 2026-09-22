"""
speedflow_python/zenoh_subscriber.py

Zenoh Command & Control Subscriber for Worker Node (Jetson Edge).

Subscribes to key expression:
    peers/control/{node_id}

Command format (msgpack):
    {"cmd": "ADD", "camera_id": "...", "source_id": N, "uri": "...", ...}
    {"cmd": "REMOVE", "camera_id": "..."}
    {"cmd": "STATUS"}

Requirements:
    pip install zenoh msgpack opencv-python numpy
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import msgpack

from .camera_config import CameraManager, StreamDelta
from .zenoh_session import make_session

logger = logging.getLogger(__name__)

class ZenohCommandSubscriber:
    """
    Subscribe to Zenoh key expression to receive control commands
    from the Peer Orchestrator.

    Key expression: peers/control/{node_id}
    Status replies: peers/status/{node_id}
    Vote acks:      peers/vote/ack/{camera_id}
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        node_id: str,
        session=None,
        ack_timeout_s: Optional[float] = None,
        boot_id: int = 0,
        lease=None,
    ) -> None:
        self._camera_manager = camera_manager
        self._node_id = node_id
        self._external_session = session
        self._ack_timeout_s = ack_timeout_s
        # P5: current boot_id (monotonic). Used as the receiver fence — a command
        # must carry THIS node's current boot_id or it is a stale pre-reboot
        # command and is rejected.
        self._boot_id = int(boot_id)
        # P5: lease state (optional). When provided, the persisted per-camera
        # epoch high-water seeds the held-epoch floor and accepted epochs are
        # recorded back so a future reboot keeps the floor.
        self._lease = lease

        self._control_key = f"peers/control/{node_id}"
        self._status_key = f"peers/status/{node_id}"
        self._session = None
        self._subscriber = None
        self._running = False

        # Per-camera held epoch for stale ADD/REMOVE rejection
        # Seeded from the persisted epoch high-water (P5) so a freshly rebooted
        # node rejects pre-reboot commands, then raised on every applied
        # ADD/REMOVE. Receiver rejects commands with epoch < held_epoch to
        # prevent partition-rejoin data-loss where a stale REMOVE tears down the
        # new owner's stream.
        if self._lease is not None:
            self._held_epochs: Dict[str, int] = dict(self._lease.camera_epochs)
        else:
            self._held_epochs: Dict[str, int] = {}

        from concurrent.futures import ThreadPoolExecutor
        self._ack_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ZenohC2-ACK")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open Zenoh session and subscribe to control key."""
        if self._external_session is not None:
            self._session = self._external_session
            logger.info("[Zenoh C2] Using shared Zenoh session.")
        else:
            import zenoh
            self._session = make_session()
            logger.info("[Zenoh C2] Session opened (peer mode).")

        self._subscriber = self._session.declare_subscriber(
            self._control_key,
            self._on_sample,
        )
        logger.info(
            "[Zenoh C2] Subscribed to '%s'. Node='%s'",
            self._control_key, self._node_id,
        )

        # Signal that this node is online
        now = time.time()
        self.publish_status({
            "schema_version": 1,
            "version": 1,
            "node_id": self._node_id,
            "event": "NODE_ONLINE",
            "active_cameras": [
                c.camera_id
                for c in self._camera_manager.get_enabled_configs()
            ],
            "timestamp": now,
            "ts": now,
        })

    def stop(self) -> None:
        """Unsubscribe and close Zenoh session."""
        if hasattr(self, '_ack_pool'):
            self._ack_pool.shutdown(wait=True)
        self._running = False
        if self._subscriber:
            self._subscriber.undeclare()
        # BUG-10 fix: only close the session if we opened it ourselves.
        # When an external (shared) session was supplied, the caller
        # (PeerOrchestrator / run_python.py) owns the lifecycle — closing it
        # here would crash all sibling modules that share the same session.
        if self._session and self._external_session is None:
            self._session.close()

    def publish_status(self, payload: dict) -> None:
        """Publish status/event from this node."""
        if self._session:
            try:
                p = dict(payload)
                now = time.time()
                if "timestamp" not in p and "ts" not in p:
                    p["timestamp"] = now
                    p["ts"] = now
                elif "timestamp" not in p and "ts" in p:
                    p["timestamp"] = p["ts"]
                elif "ts" not in p and "timestamp" in p:
                    p["ts"] = p["timestamp"]
                if "schema_version" not in p and "version" not in p:
                    p["schema_version"] = 1
                    p["version"] = 1
                elif "schema_version" not in p and "version" in p:
                    p["schema_version"] = p["version"]
                elif "version" not in p and "schema_version" in p:
                    p["version"] = p["schema_version"]
                data = msgpack.packb(p, use_bin_type=True)
                self._session.put(self._status_key, data)
            except Exception as exc:
                logger.warning("[Zenoh C2] Failed to publish status: %s", exc)

    # ------------------------------------------------------------------
    # Internal — Zenoh subscriber callback
    # ------------------------------------------------------------------

    def _on_sample(self, sample) -> None:
        """Handle incoming control command."""
        import zenoh
        try:
            payload = msgpack.unpackb(sample.payload.to_bytes(), raw=False)
        except Exception as exc:
            logger.error("[Zenoh C2] Invalid msgpack received: %s", exc)
            return

        cmd = payload.get("cmd", "").upper()
        logger.info("[Zenoh C2] Received command: %s", cmd)

        if cmd == "ADD":
            self._handle_add(payload)
        elif cmd == "REMOVE":
            self._handle_remove(payload)
        elif cmd == "STATUS":
            self._handle_status_request()
        else:
            logger.warning("[Zenoh C2] Unknown command: '%s'", cmd)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # P5 — boot / epoch fencing
    # ------------------------------------------------------------------

    def _lease_fence(self, cam_id: str, epoch, boot_id) -> Tuple[str, Optional[str]]:
        """Decide whether a command may pass receiver fencing.

        Returns (decision, reason) where decision is "accept" or "reject".
        Two independent fences:

          * boot_id fence: a command must carry THIS node's current boot_id.
            Pre-reboot commands carry the node's OLD boot_id and are rejected,
            so a stale ADD/REMOVE replayed after a restart cannot pass.
          * epoch floor fence: within a boot, the command epoch must be >= the
            persisted/observed high-water for the camera.

        A command missing boot_id falls back to the epoch-only fence (legacy
        wire compatibility). A command missing epoch skips the epoch fence.
        """
        # boot_id fence
        if boot_id is not None:
            try:
                b = int(boot_id)
            except (TypeError, ValueError):
                b = None
            if b is not None and b != self._boot_id:
                return "reject", "stale_boot_id"

        # epoch floor fence
        if epoch is not None:
            try:
                e = int(epoch)
            except (TypeError, ValueError):
                return "reject", "malformed_epoch"
            if e < self._held_epochs.get(cam_id, -1):
                return "reject", "stale_epoch"

        return "accept", None

    def _lease_accept(self, cam_id: str, epoch) -> None:
        """Raise the held-epoch floor and persist the high-water on accept."""
        if epoch is None:
            return
        try:
            e = int(epoch)
        except (TypeError, ValueError):
            return
        if e > self._held_epochs.get(cam_id, -1):
            self._held_epochs[cam_id] = e
            if self._lease is not None:
                try:
                    self._lease.record_epoch(cam_id, e)
                except Exception as exc:
                    logger.warning("[Zenoh C2][P5] Failed to persist epoch high-water for '%s': %s", cam_id, exc)

    def _handle_add(self, payload: dict) -> None:
        try:
            cam_id    = payload["camera_id"]
            source_id = int(payload["source_id"])
            epoch     = payload.get("epoch")
            boot_id   = payload.get("boot_id")

            # P5 — boot_id + epoch floor fencing. Reject stale pre-reboot or
            # out-of-order commands before touching the pipeline.
            decision, reason = self._lease_fence(cam_id, epoch, boot_id)
            if decision == "reject":
                held_epoch = self._held_epochs.get(cam_id, -1)
                logger.info(
                    "[Zenoh C2] ADD rejected for '%s': %s (epoch=%s, boot_id=%s, held_epoch=%d, self_boot_id=%d)",
                    cam_id, reason, epoch, boot_id, held_epoch, self._boot_id,
                )
                self.publish_status({
                    "node_id": self._node_id,
                    "event": "ADD_REJECTED",
                    "camera_id": cam_id,
                    "reason": reason,
                    "epoch": epoch,
                    "boot_id": boot_id,
                    "held_epoch": held_epoch,
                    "self_boot_id": self._boot_id,
                })
                # P2: wake the requester's _wait_and_remove immediately (no 15s timeout)
                # so it can fast-forward its epoch past held_epoch. Only stale_epoch is fast-forwardable.
                if reason == "stale_epoch" and self._session is not None:
                    try:
                        now = time.time()
                        self._session.put(f"peers/vote/ack/{cam_id}", msgpack.packb({
                            "schema_version": 1,
                            "version": 1,
                            "node_id": self._node_id,
                            "camera_id": cam_id,
                            "event": "REJECTED",
                            "reason": reason,
                            "held_epoch": held_epoch,
                            "epoch": epoch,
                            "migration_id": payload.get("migration_id"),
                            "timestamp": now,
                            "ts": now,
                        }, use_bin_type=True))
                        logger.info("[Zenoh C2] Published REJECTED ack for '%s' (held_epoch=%d) to peers/vote/ack/%s", cam_id, held_epoch, cam_id)
                    except Exception as exc:
                        logger.warning("[Zenoh C2] Failed to publish REJECTED ack for '%s': %s", cam_id, exc)
                return

            # Delegate config building + delta enqueue to CameraManager so the
            # same logic is shared with PeerOrchestrator's direct-dispatch path.
            queued = self._camera_manager.handle_add_command(payload)
            if not queued:
                self.publish_status({
                    "node_id": self._node_id,
                    "event": "ADD_REJECTED",
                    "camera_id": cam_id,
                    "reason": "camera_or_source_id_conflict",
                })
                return

            # P5: raise held-epoch floor + persist high-water on accept.
            self._lease_accept(cam_id, epoch)

            self.publish_status({
                "node_id": self._node_id,
                "event": "ADD_PROCESSING",
                "camera_id": cam_id,
                "source_id": source_id,
            })

            # P2P vote ack — wait until the stream actually reaches PLAYING
            # state (or timeout) before acknowledging Make-Before-Break.
            #
            # BUG-3 fix: the old implementation sent the ack unconditionally
            # after a hardcoded sleep(3), which could ack a failed ADD and
            # cause the requester to REMOVE its own stream before the new one
            # is actually up.
            #
            # Strategy: poll the CameraConfig.enabled flag AND verify that
            # the CameraManager has a live source_id mapping (set by on_add
            # which runs on the GLib main loop after dynamic_add_stream
            # succeeds).  We also apply a generous 15-second timeout so that
            # slow RTSP sources don't block forever.
            _session     = self._session
            _cam_id      = cam_id
            _source_id   = source_id
            _node_id     = self._node_id
            _cam_manager = self._camera_manager
            _epoch       = payload.get("epoch")
            _mig_id      = payload.get("migration_id")
            # Coordinate with migration_timeout_s: default 12.0s allows migration_timeout_s (15.0s) a 3.0s safety margin
            _ack_timeout = float(self._ack_timeout_s) if self._ack_timeout_s is not None else 12.0

            def _send_ack() -> None:
                deadline = time.monotonic() + _ack_timeout
                playing  = False

                # Config registration is synchronous and is not stream readiness.
                # Wait for the first decoded buffer from this source instead.
                # Recreate-retry: a wedged uridecodebin (handshake OK, decodebin
                # never exposes a pad — seen live on cross-node ADDs) never
                # recovers in place; only a fresh source bin renegotiates pads.
                # Spend the first part of the window waiting, then queue ONE
                # recreate-retry via CameraManager and wait out the remainder.
                # Worst case (retry also starves) falls through to the existing
                # withheld path — never worse than today. The 12s contract is
                # untouched.
                ready_event = _cam_manager.stream_ready_event(_source_id)
                first_wait = min(5.0, max(2.0, _ack_timeout - 7.0))
                if ready_event is not None:
                    playing = ready_event.wait(timeout=max(0.0, min(first_wait, deadline - time.monotonic())))
                if not playing:
                    _cur_ev = _cam_manager.stream_ready_event(_source_id)
                    if _cur_ev is not None and not _cur_ev.is_set():
                        try:
                            retried = _cam_manager.retry_add(_source_id)
                        except Exception as exc_rt:
                            logger.warning(
                                "[Zenoh C2] Recreate-retry failed to queue for '%s': %s",
                                _cam_id, exc_rt,
                            )
                            retried = False
                        if retried:
                            logger.warning(
                                "[Zenoh C2] First buffer missed %.0fs for '%s' — "
                                "recreate-retry queued, waiting out the window.",
                                first_wait, _cam_id,
                            )
                            _new_ev = _cam_manager.stream_ready_event(_source_id)
                            if _new_ev is not None:
                                playing = _new_ev.wait(timeout=max(0.0, deadline - time.monotonic()))
                            else:
                                playing = ready_event.wait(timeout=max(0.0, deadline - time.monotonic())) if ready_event is not None else False
                        else:
                            # Retry refused: either the config is gone (fall
                            # through to the withheld path) or the branch turned
                            # PLAYING concurrently (ack it below).
                            _recheck = _cam_manager.stream_ready_event(_source_id)
                            if _recheck is not None and _recheck.is_set():
                                playing = True
                            elif ready_event is not None:
                                playing = ready_event.wait(timeout=max(0.0, deadline - time.monotonic()))

                if not playing:
                    logger.warning(
                        "[Zenoh C2] ADD ack NOT sent for '%s': stream did not reach "
                        "PLAYING within %.0fs.", _cam_id, _ack_timeout,
                    )
                    # Self-eviction guard: only skip if the existing branch is
                    # genuinely PLAYING (ready_event set). A held branch that
                    # never reached PLAYING (slow re-ADD) must be cleaned up,
                    # otherwise the 15s migration timeout strands the camera.
                    _ready_ev = _cam_manager.stream_ready_event(_source_id)
                    if _ready_ev is not None and _ready_ev.is_set():
                        logger.info(
                            "[Zenoh C2] Skipping self-eviction for '%s': PLAYING branch already active (ready set).",
                            _cam_id,
                        )
                        # Duplicate ADD on an already-PLAYING branch — ack it
                        # so the requester can complete its MBB instead of timing out.
                        try:
                            now2 = time.time()
                            ack_dup = {
                                "schema_version": 1,
                                "version": 1,
                                "node_id": _node_id,
                                "camera_id": _cam_id,
                                "event": "PLAYING",
                                "timestamp": now2,
                                "ts": now2,
                            }
                            if _epoch is not None:
                                ack_dup["epoch"] = _epoch
                            if _mig_id is not None:
                                ack_dup["migration_id"] = _mig_id
                            if _session:
                                _session.put(f"peers/vote/ack/{_cam_id}", msgpack.packb(ack_dup, use_bin_type=True))
                            logger.info("[Zenoh C2] Duplicate ADD ack sent for PLAYING '%s' (epoch=%s).", _cam_id, _epoch)
                        except Exception as exc2:
                            logger.warning("[Zenoh C2] Failed to send duplicate PLAYING ack for '%s': %s", _cam_id, exc2)
                        return
                    # Root-cause fix (offload-induced camera-loss): do NOT disable the
                    # config or queue REMOVE just because the first buffer missed the
                    # ack window.  A slow first frame does not mean the camera is dead —
                    # evicting here permanently drops a statically-owned branch whose
                    # (watchdog-triggered) REMOVE+ADD never got to PLAYING in time.
                    # Preserve the config/branch and let the membership watchdog own
                    # bounded retry (REMOVE+ADD restart with backoff + park).  A genuine
                    # competing PLAYING branch or duplicate ADD is handled above.
                    logger.warning(
                        "[Zenoh C2] ADD ack withheld for '%s': first buffer missed the "
                        "%.0fs window. Branch remains ENABLED for bounded retry; "
                        "membership watchdog owns restart.", _cam_id, _ack_timeout,
                    )
                    # Reaffirm processing status: the branch is preserved, not failed —
                    # the requester's MBB may still complete via the watchdog retry.
                    self.publish_status({
                        "node_id": _node_id,
                        "event": "ADD_PROCESSING",
                        "camera_id": _cam_id,
                        "source_id": _source_id,
                    })
                    return

                try:
                    now = time.time()
                    ack_p = {
                        "schema_version": 1,
                        "version":   1,
                        "node_id":   _node_id,
                        "camera_id": _cam_id,
                        "event":     "PLAYING",
                        "timestamp": now,
                        "ts":        now,
                    }
                    if _epoch is not None:
                        ack_p["epoch"] = _epoch
                    if _mig_id is not None:
                        ack_p["migration_id"] = _mig_id
                    ack_payload = msgpack.packb(ack_p, use_bin_type=True)
                    if _session:
                        _session.put(f"peers/vote/ack/{_cam_id}", ack_payload)
                    logger.info("[Zenoh C2] ADD ack sent for '%s' (stream PLAYING, epoch=%s, migration_id=%s).", _cam_id, _epoch, _mig_id)
                except Exception as exc:
                    logger.warning("[Zenoh C2] Failed to send ack for '%s': %s", _cam_id, exc)

            self._ack_pool.submit(_send_ack)

        except KeyError as exc:
            logger.error("[Zenoh C2] ADD command missing required field: %s", exc)
            self.publish_status({
                "node_id": self._node_id,
                "event": "ADD_FAILED",
                "reason": f"missing_field_{exc}",
            })
        except Exception as exc:
            logger.error("[Zenoh C2] ADD command error: %s", exc)
            self.publish_status({
                "node_id": self._node_id,
                "event": "ADD_FAILED",
                "reason": str(exc),
            })

    def _handle_remove(self, payload: dict) -> None:
        try:
            cam_id = payload["camera_id"]
            epoch = payload.get("epoch")
            boot_id = payload.get("boot_id")
            migration_id = payload.get("migration_id")

            # P5 — boot_id + epoch floor fencing. Reject stale pre-reboot or
            # out-of-order commands before touching the pipeline.
            decision, reason = self._lease_fence(cam_id, epoch, boot_id)
            if decision == "reject":
                held_epoch = self._held_epochs.get(cam_id, -1)
                logger.info(
                    "[Zenoh C2] REMOVE rejected for '%s': %s (epoch=%s, boot_id=%s, held_epoch=%d, self_boot_id=%d)",
                    cam_id, reason, epoch, boot_id, held_epoch, self._boot_id,
                )
                self.publish_status({
                    "node_id": self._node_id,
                    "event": "REMOVE_REJECTED",
                    "camera_id": cam_id,
                    "reason": reason,
                    "epoch": epoch,
                    "boot_id": boot_id,
                    "held_epoch": held_epoch,
                    "self_boot_id": self._boot_id,
                    "migration_id": migration_id,
                })
                return

            with self._camera_manager._lock:
                cfg = self._camera_manager._configs.get(cam_id)
                if not cfg or not cfg.enabled:
                    logger.warning(
                        "[Zenoh C2] REMOVE ignored: camera_id='%s' not active.", cam_id
                    )
                    self.publish_status({
                        "node_id": self._node_id,
                        "event": "REMOVE_REJECTED",
                        "camera_id": cam_id,
                        "reason": "not_active",
                        "epoch": epoch,
                        "migration_id": migration_id,
                    })
                    return
                # If command specifies source_id, verify it matches current active source_id.
                # Robustly reject malformed non-integer source_id without throwing out of Zenoh callback.
                cmd_sid = payload.get("source_id")
                if cmd_sid is not None:
                    try:
                        parsed_sid = int(cmd_sid)
                    except (ValueError, TypeError):
                        logger.warning(
                            "[Zenoh C2] Malformed REMOVE source_id ignored: camera_id='%s' source_id=%r",
                            cam_id, cmd_sid,
                        )
                        self.publish_status({
                            "node_id": self._node_id,
                            "event": "REMOVE_REJECTED",
                            "camera_id": cam_id,
                            "reason": "malformed_source_id",
                            "epoch": epoch,
                            "migration_id": migration_id,
                        })
                        return

                    if parsed_sid != cfg.source_id:
                        logger.warning(
                            "[Zenoh C2] Stale REMOVE ignored: camera_id='%s' active source_id=%d != cmd source_id=%d",
                            cam_id, cfg.source_id, parsed_sid,
                        )
                        self.publish_status({
                            "node_id": self._node_id,
                            "event": "REMOVE_REJECTED",
                            "camera_id": cam_id,
                            "reason": "stale_source_id",
                            "epoch": epoch,
                            "migration_id": migration_id,
                        })
                        return
                source_id = cfg.source_id
                cfg.enabled = False
                self._camera_manager._rebuild_lookup()
                # P5: raise held-epoch floor + persist high-water on accept.
                self._lease_accept(cam_id, epoch)

            def _on_teardown_done():
                try:
                    now = time.time()
                    ack_p = {
                        "schema_version": 1,
                        "version": 1,
                        "node_id": self._node_id,
                        "camera_id": cam_id,
                        "source_id": source_id,
                        "event": "REMOVED",
                        "timestamp": now,
                        "ts": now,
                    }
                    if epoch is not None:
                        ack_p["epoch"] = epoch
                    if migration_id is not None:
                        ack_p["migration_id"] = migration_id

                    if self._session:
                        self._session.put(f"peers/remove/ack/{cam_id}", msgpack.packb(ack_p, use_bin_type=True))
                    self.publish_status({
                        "node_id": self._node_id,
                        "event": "REMOVED",
                        "camera_id": cam_id,
                        "source_id": source_id,
                        "epoch": epoch,
                        "migration_id": migration_id,
                    })
                    logger.info("[Zenoh C2] REMOVED ack sent for '%s' (source_id=%d, epoch=%s, migration_id=%s)", cam_id, source_id, epoch, migration_id)
                except Exception as ex:
                    logger.error("[Zenoh C2] Failed to send REMOVED ack for '%s': %s", cam_id, ex)

            delta = StreamDelta(to_remove=[(source_id, _on_teardown_done)])
            self._camera_manager._delta_q.put(delta)
            if hasattr(self._camera_manager, "cleanup_stream_ready"):
                self._camera_manager.cleanup_stream_ready(source_id)

            logger.info(
                "[Zenoh C2] REMOVE queued: camera_id='%s', source_id=%d", cam_id, source_id
            )
            self.publish_status({
                "node_id": self._node_id,
                "event": "REMOVE_PROCESSING",
                "camera_id": cam_id,
                "source_id": source_id,
                "epoch": epoch,
                "migration_id": migration_id,
            })

        except KeyError as exc:
            logger.error("[Zenoh C2] REMOVE command missing required field: %s", exc)
            self.publish_status({
                "node_id": self._node_id,
                "event": "REMOVE_FAILED",
                "reason": f"missing_field_{exc}",
            })
        except Exception as exc:
            logger.error("[Zenoh C2] REMOVE command error: %s", exc)
            self.publish_status({
                "node_id": self._node_id,
                "event": "REMOVE_FAILED",
                "reason": str(exc),
            })

    def _handle_status_request(self) -> None:
        active = self._camera_manager.get_enabled_configs()
        self.publish_status({
            "node_id": self._node_id,
            "event": "STATUS_REPORT",
            "active_cameras": [c.camera_id for c in active],
            "active_count": len(active),
        })
