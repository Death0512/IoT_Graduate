from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("camera_projection")

# ponytail: single-process in-memory projection. Server restart rebuilds it
# from edge heartbeats (no persistence needed). If multi-process ever needed,
# back this with the registry DB; the public API stays the same.


class CameraState:
    __slots__ = (
        "camera_id",
        "owner_node",
        "holder_node",
        "epoch",
        "active",
        "held",
        "node_id",       # authoritative reporting node (== holder_node)
        "online",        # whether that node is currently online
        "source_id",
        "last_seen",
    )

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.owner_node: Optional[str] = None
        self.holder_node: Optional[str] = None
        self.epoch: int = 0
        self.active: bool = False
        self.held: bool = False
        self.node_id: Optional[str] = None
        self.online: bool = True
        self.source_id: Optional[int] = None
        self.last_seen: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "owner_node": self.owner_node,
            "holder_node": self.holder_node,
            "epoch": self.epoch,
            "active": self.active,
            "held": self.held,
            "node_id": self.node_id,
            "online": self.online,
            "source_id": self.source_id,
            "last_seen": self.last_seen,
        }


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


class CameraProjection:
    """
    Authoritative, in-memory projection of every known camera keyed by
    camera_id. Only the camera's declared holder (or owner, falling back to
    the reporting node) is trusted to update its row. Updates from any other
    node, or with a lower epoch, are rejected (fail-closed) so a stale or
    non-authoritative edge can never corrupt the authoritative view.

    This does NOT invent central control: it only observes the owner/holder/
    epoch fields the edges already publish and collapses duplicate per-node
    active_cameras into exactly one row per camera.
    """

    def __init__(self) -> None:
        self._cams: Dict[str, CameraState] = {}
        self._node_online: Dict[str, bool] = {}
        self._node_boot_id: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ── authoritative update ─────────────────────────────────────
    def apply_health(self, node_id: str, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return

        # Fail-closed: once the registry has marked a node offline, any late
        # health report from it is stale and ignored. The node is re-enabled
        # only via on_node_online() (called when a fresh heartbeat arrives).
        if not self._node_online.get(node_id, True):
            logger.debug("[Projection] Ignoring health from offline node '%s'", node_id)
            return

        pipeline = payload.get("pipeline") or {}
        if not isinstance(pipeline, dict):
            pipeline = {}

        owners = payload.get("camera_owners") or pipeline.get("camera_owners") or {}
        holders = payload.get("camera_holders") or pipeline.get("camera_holders") or {}
        epochs = payload.get("camera_epochs") or pipeline.get("camera_epochs") or {}
        configs = pipeline.get("camera_configs") or {}

        active_cams = set(
            pipeline.get("active_cameras") or pipeline.get("streaming_cameras") or []
        )
        held_cams = set(pipeline.get("held_cameras") or [])

        # Every camera this node mentions in any ownership/activity field.
        cam_ids = (
            set(owners)
            | set(holders)
            | set(epochs)
            | set(configs)
            | active_cams
            | held_cams
        )

        boot_id = _as_int(payload.get("boot_id", 0))

        with self._lock:
            last_boot_id = self._node_boot_id.get(node_id, 0)
            is_new_boot = boot_id > 0 and boot_id > last_boot_id
            if is_new_boot:
                logger.info(
                    "[Projection] Node '%s' new boot_id detected (%d > %d) — resetting its camera epochs",
                    node_id, boot_id, last_boot_id,
                )
                self._node_boot_id[node_id] = boot_id
                # Reset epoch on cameras owned or held by this newly booted node
                for c in self._cams.values():
                    if c.node_id == node_id or c.holder_node == node_id or c.owner_node == node_id:
                        c.epoch = 0
            # Track whether this update comes from a freshly booted node so the
            # per-camera epoch check below can allow an epoch regression when the
            # owner node rebooted and legitimately restarted from epoch 0.
            node_is_fresh_boot = is_new_boot

        for cam in cam_ids:
            declared_holder = holders.get(cam)
            declared_owner = owners.get(cam)
            declared_epoch = _as_int(epochs.get(cam, 0))
            # Trusted source for this camera's authoritative state.
            authoritative_node = declared_holder or declared_owner or node_id

            # Fail-closed: only the holder/owner node may write this row.
            if node_id != authoritative_node:
                logger.debug(
                    "[Projection] Ignoring non-authoritative report for '%s' "
                    "from '%s' (authoritative node is '%s')",
                    cam, node_id, authoritative_node,
                )
                continue

            active = cam in active_cams
            held = cam in held_cams
            source_id = None
            cfg = configs.get(cam)
            if isinstance(cfg, dict) and cfg.get("source_id") is not None:
                source_id = _as_int(cfg.get("source_id"))

            with self._lock:
                cur = self._cams.get(cam)
                if cur is None:
                    cur = CameraState(cam)
                    self._cams[cam] = cur
                else:
                    # Stale epoch from the authoritative node (e.g. an old
                    # holder after a handoff): reject, do not regress.
                    # Exception: if the reporting node is the static owner and
                    # just rebooted (fresh boot_id), allow epoch regression so
                    # it can reclaim cameras held by a peer at higher epoch.
                    is_owner_reclaim = (
                        node_is_fresh_boot
                        and declared_owner == node_id
                        and cur.owner_node == node_id
                    )
                    if declared_epoch < cur.epoch and not is_owner_reclaim:
                        logger.warning(
                            "[Projection] Stale epoch for '%s' from '%s' "
                            "(%d < %d) — update rejected",
                            cam, node_id, declared_epoch, cur.epoch,
                        )
                        continue
                    if is_owner_reclaim and declared_epoch < cur.epoch:
                        logger.info(
                            "[Projection] Owner '%s' rebooted — allowing epoch regression for '%s' (%d < %d)",
                            node_id, cam, declared_epoch, cur.epoch,
                        )

                cur.owner_node = declared_owner
                cur.holder_node = declared_holder
                cur.epoch = declared_epoch
                cur.active = active
                cur.held = held
                cur.node_id = node_id
                cur.online = self._node_online.get(node_id, True)
                cur.source_id = source_id
                cur.last_seen = time.time()

    # ── offline / stale handling ─────────────────────────────────
    def on_node_offline(self, node_id: str) -> None:
        """Mark every camera held by this node as offline/stale.

        Any later health report arriving from the offline node is ignored by
        apply_health (node flagged offline), so a dead edge cannot keep a
        camera alive in the authoritative view.
        """
        with self._lock:
            self._node_online[node_id] = False
            for cam in self._cams.values():
                if cam.node_id == node_id or cam.holder_node == node_id:
                    cam.online = False
                    cam.active = False
                    cam.held = False

    def on_node_online(self, node_id: str) -> None:
        with self._lock:
            self._node_online[node_id] = True

    # ── reads ────────────────────────────────────────────────────
    def get_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [c.to_dict() for c in sorted(self._cams.values(), key=lambda c: c.camera_id)]

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cam = self._cams.get(camera_id)
            return cam.to_dict() if cam else None

    def count(self) -> int:
        with self._lock:
            return len(self._cams)
