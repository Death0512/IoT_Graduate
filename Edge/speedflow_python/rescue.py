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

"""Edge/speedflow_python/rescue.py

Source-liveness / rescue-probe mixin for PeerOrchestrator (P4).
Methods relocated verbatim; shared helpers live in membership.py.
"""

class RescueMixin:
    def _resolve_local_source_id(self, camera_id: str) -> Optional[int]:
        """
        Resolve the source_id for a locally active/configured camera.
        Returns int source_id if resolved, or None if unknown/not found.
        """
        if self._camera_manager is not None:
            try:
                with self._camera_manager._lock:
                    cfg_obj = self._camera_manager._configs.get(camera_id)
                    if cfg_obj is not None and getattr(cfg_obj, "enabled", True):
                        return int(cfg_obj.source_id)
            except Exception as exc:
                logger.debug("[PeerOrch] Could not resolve source_id from CameraManager for '%s': %s", camera_id, exc)

        cfg = self._get_camera_config(camera_id)
        if cfg and "source_id" in cfg:
            try:
                return int(cfg["source_id"])
            except (ValueError, TypeError):
                pass
        return None

    def _probe_source_liveness(self, camera_id: str, cam_uri: str) -> str:
        """
        P2: per-camera source-liveness gate for rescue ADD.

        Returns one of:
          "reachable"  -> source confirmed produce/accept a stream; safe to ADD.
          "unreachable"-> SOURCE_UNREACHABLE: probe definitively failed after
                          bounded attempts. Do NOT ADD. We still keep a slow
                          steady re-probe so a recovered source can be rescued.
          "pending"    -> rescue-pending: probe not yet conclusive / backoff in
                          effect. Do NOT ADD; a later failover round will retry.

        Bounded backoff: failures back off exponentially up to a ceiling; once
        ``rescue_source_probe_max_attempts`` is reached the status flips to
        "unreachable" but re-probing continues at the slow ceiling interval, so
        recovery is still detected. A fresh "reachable" result stays valid for
        ``rescue_source_liveness_ttl_s`` (no re-probe while fresh).
        """
        now = time.time()
        base = float(self._cfg.get("rescue_source_probe_backoff_base_s", 5.0))
        cap = float(self._cfg.get("rescue_source_probe_backoff_max_s", 60.0))
        max_attempts = int(self._cfg.get("rescue_source_probe_max_attempts", 12))
        ttl = float(self._cfg.get("rescue_source_liveness_ttl_s", 30.0))

        with self._source_liveness_lock:
            st = self._source_liveness.get(camera_id)
            if st is None:
                st = {
                    "status": "pending",
                    "last_probe_ts": 0.0,
                    "next_probe_ts": 0.0,
                    "attempts": 0,
                }
                self._source_liveness[camera_id] = st
            else:
                st = dict(st)  # snapshot; real mutation happens under lock below

        # Fresh reachable result → skip re-probe entirely.
        if st["status"] == "reachable" and (now - st["last_probe_ts"]) <= ttl:
            return "reachable"

        # Backoff not elapsed → return cached status (no probe, no ADD).
        if now < st["next_probe_ts"]:
            return st["status"]

        # Perform the probe (blocking, safe in the failover thread pool).
        ok = self._measure_rtt(cam_uri) is not None

        with self._source_liveness_lock:
            cur = self._source_liveness.setdefault(camera_id, {
                "status": "pending", "last_probe_ts": 0.0,
                "next_probe_ts": 0.0, "attempts": 0,
            })
            if ok:
                cur["status"] = "reachable"
                cur["last_probe_ts"] = now
                cur["attempts"] = 0
                cur["next_probe_ts"] = now + ttl
            else:
                cur["attempts"] += 1
                cur["last_probe_ts"] = now
                if cur["attempts"] >= max_attempts:
                    cur["status"] = "unreachable"
                    # Slow steady re-probe so a recovered source is still detected.
                    cur["next_probe_ts"] = now + cap
                else:
                    cur["status"] = "pending"
                    backoff = min(base * (2 ** (cur["attempts"] - 1)), cap)
                    cur["next_probe_ts"] = now + backoff
            return cur["status"]

    def _measure_rtt(self, rtsp_uri: str) -> Optional[float]:
        """
        Verify an RTSP stream is reachable AND the path exists.

        Sends an RTSP DESCRIBE request.  Returns RTT in ms on success
        (any 2xx response), or None if the host is down or the path
        does not exist (4xx).
        """
        try:
            parsed = urllib.parse.urlparse(rtsp_uri)
            host = parsed.hostname
            port = parsed.port or 554
            t0 = time.monotonic()
            with socket.create_connection((host, port), timeout=0.5) as sock:
                # Send a minimal RTSP DESCRIBE to check if the path exists
                req = (
                    f"DESCRIBE {rtsp_uri} RTSP/1.0\r\n"
                    f"CSeq: 1\r\n"
                    f"Accept: application/sdp\r\n"
                    f"\r\n"
                )
                sock.sendall(req.encode())
                resp = sock.recv(512).decode(errors="ignore")
                rtt = (time.monotonic() - t0) * 1000.0
                # Accept any 2xx response; reject 404/401/etc.
                if resp.startswith("RTSP/1.0 2"):
                    return rtt
                return None
        except Exception:
            return None

    def _get_camera_config(self, camera_id: str) -> Optional[dict]:
        """
        Read camera config from cameras.yml.

        BUG-2 fix: prefer the live CameraManager when available — it already
        maintains a hot-reloaded, up-to-date config dict so we never serve
        stale homography/ROI/URI data after a cameras.yml change.  Fall back
        to a YAML parse only when the manager is not set (standalone tests).

        BUG-15 (original BUG-15 from bug report): cache is now irrelevant
        because CameraManager owns the in-memory state.  The _cameras_cache
        field is kept for the YAML fallback path only.
        """
        # Fast path: use live CameraManager (always up-to-date)
        if self._camera_manager is not None:
            try:
                cfg_obj = None
                # CameraManager stores CameraConfig objects keyed by camera_id
                with self._camera_manager._lock:
                    cfg_obj = self._camera_manager._configs.get(camera_id)
                if cfg_obj is not None:
                    return {
                        "camera_id":       camera_id,
                        "source_id":       int(cfg_obj.source_id),
                        "uri":             cfg_obj.uri,
                        "name":            cfg_obj.name,
                        "fps":             float(cfg_obj.fps),
                        "speed_limit_kmh": float(cfg_obj.speed_limit_kmh),
                        "homography": {
                            "source_points": cfg_obj.source_points.tolist(),
                            "target_width":  int(cfg_obj.target_points[2, 0]),
                            "target_height": int(cfg_obj.target_points[2, 1]),
                        },
                        "roi_polygon": cfg_obj.roi_polygon.tolist(),
                        "output": {
                            "record":      cfg_obj.record,
                            "record_path": cfg_obj.record_path,
                        },
                    }
            except Exception as exc:
                logger.debug("CameraManager config lookup failed for '%s': %s", camera_id, exc)

        # Fallback: parse cameras.yml (used in tests / standalone mode)
        # BUG-2 fix: invalidate the cache based on file mtime so hot-reload is
        # respected even in the fallback path.
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
                self._cameras_cache = raw.get("cameras", {})
                self._cameras_cache_mtime = current_mtime

            cameras = self._cameras_cache
            cfg = cameras.get(camera_id)
            if not cfg:
                return None
            return {
                "camera_id":       camera_id,
                "source_id":       int(cfg.get("source_id", 0)),
                "uri":             cfg.get("uri", ""),
                "name":            cfg.get("name", camera_id),
                "fps":             float(cfg.get("fps", 25.0)),
                "speed_limit_kmh": float(cfg.get("speed_limit_kmh", 80.0)),
                "homography":      cfg.get("homography", {}),
                "roi_polygon":     cfg.get("roi_polygon", []),
                "output":          cfg.get("output", {}),
            }
        except Exception as exc:
            logger.error("Failed to load camera config for '%s': %s", camera_id, exc)
            return None

    def _maybe_log_block(self, reason: str, now: float) -> bool:
        """Return True the first time `reason` fires or after its cooldown expires."""
        last = self._blocked_logged_at.get(reason, 0.0)
        if now - last >= self.BLOCKED_LOG_COOLDOWN:
            self._blocked_logged_at[reason] = now
            return True
        return False

    def _get_camera_uri(self, camera_id: str) -> Optional[str]:
        """Get RTSP URI of camera from cameras.yml."""
        cfg = self._get_camera_config(camera_id)
        if cfg:
            return cfg.get("uri")
        return None
