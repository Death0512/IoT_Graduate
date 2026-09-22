from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("telemetry_store")

# Fields persisted from the health payload; intentionally NOT the full payload.
_KEPT_FIELDS = (
    "ts", "node_id", "session_id", "sequence", "load_score", "gpu_percent",
    "cpu_percent", "ram_percent", "gpu_temp_c", "power_mw", "n_track_total",
    "n_plate_total", "stationary_fraction_mean", "fps_avg",
    "offload_crops_received_per_s", "active_cameras_count", "pipeline_status",
)

# Camera per-stream FPS keys are bare floats (not prefixed with "_") — keep them.
_FPS_PREFIX_IGNORED = "_"


"""
Server/telemetry_store.py — Production Telemetry Recording

TelemetryStore — Persists 1 Hz edge node telemetry into CSV files.

This is the PRODUCTION telemetry recorder. All model evaluation, load analysis,
and research datasets are derived from these CSV files.

Data Flow:
  Edge HealthAgent (1 Hz) → Zenoh peers/status/{node} → Server app.py
    → TelemetryStore.save_async() → CSV append

File Format:
  One CSV per node: Server/data/telemetry/calibration_{node_id}.csv
  14 columns: ts, gpu_percent, cpu_percent, ram_percent, gpu_temp_c,
              session_id, sequence, fps_avg, n_active_cameras,
              n_track_total, n_plate_total, stationary_fraction_mean,
              offload_crops_received_per_s, load_score

Key Design Decisions:
- Async write via asyncio.to_thread: Never blocks the aiohttp event loop
- Deduplication: (session_id, sequence) pair must advance; duplicate heartbeats dropped
- Per-node asyncio.Lock: Serializes appends to same CSV (prevents interleaving)
- Derived fields computed server-side: n_track_total, n_plate_total,
  stationary_fraction_mean, fps_avg (from per-camera fps_per_camera)
- FPS keys starting with "_" ignored (internal bookkeeping, not real cameras)
- Directory created on demand: Server/data/telemetry/

No local CSV writing on Jetsons — all telemetry centralized on Server.
Jetsons do NOT write training CSV files locally (constraint #4013).
"""


class TelemetryStore:
    def __init__(self, base_dir: str | Path) -> None:
        self._root = Path(base_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        # Lazy lock per loop, same BUG-17 discipline as violation_store.
        self._locks: Dict[str, Optional[asyncio.Lock]] = {}
        # Last persisted (session_id, sequence) per node — dedup key so the
        # same pipeline telemetry window is never written twice.
        self._last_seq: Dict[str, tuple] = {}

    def _get_lock(self, node_id: str) -> asyncio.Lock:
        if node_id not in self._locks or self._locks[node_id] is None:
            self._locks[node_id] = asyncio.Lock()
        return self._locks[node_id]

    async def save_async(self, node_id: str, payload: Dict[str, Any]) -> None:
        # Dedup: skip when this row mirrors the last telemetry window for node.
        sess = payload.get("session_id")
        seq = payload.get("sequence")
        if isinstance(seq, int) and isinstance(sess, str) and sess:
            if self._last_seq.get(node_id) == (sess, seq):
                return
            self._last_seq[node_id] = (sess, seq)

        rec = {"ts": time.time(), "node_id": node_id}
        for f in _KEPT_FIELDS:
            if f in ("ts", "node_id"):
                continue
            if f in payload:
                rec[f] = payload[f]

        # Edge nests camera run metrics under payload["pipeline"] (older
        # versions may omit it entirely — degrade to empty dicts).
        pipeline_data = payload.get("pipeline") or {}

        # Camera FPS lives at pipeline.fps_per_camera; ignore top-level
        # scalars (boot_id, network_bps_rx, etc.) which are not FPS values.
        fps = {k: v for k, v in pipeline_data.get("fps_per_camera", {}).items()
               if isinstance(k, str) and not k.startswith(_FPS_PREFIX_IGNORED)}
        if fps:
            rec["fps_per_camera"] = fps

        # Derived aggregates from camera_features — mirrors
        # tools/profile_collect.py (n_track_total, n_plate_total,
        # stationary_fraction_mean) plus fps_avg (mean per-camera FPS).
        cf = pipeline_data.get("camera_features")
        active_feats = {k: v for k, v in cf.items() if isinstance(v, dict)} \
            if isinstance(cf, dict) else {}
        n_track_total = sum(v.get("n_track", 0.0) for v in active_feats.values())
        n_plate_total = sum(v.get("n_plate", 0.0) for v in active_feats.values())
        stat_vals = [v.get("stationary_fraction", 0.0) for v in active_feats.values()]
        rec["n_track_total"] = round(n_track_total, 2)
        rec["n_plate_total"] = round(n_plate_total, 2)
        rec["stationary_fraction_mean"] = round(
            sum(stat_vals) / len(stat_vals), 3) if stat_vals else 0.0
        rec["fps_avg"] = round(
            sum(fps.values()) / len(fps), 3) if fps else 0.0

        # Persist calibration_<NODE_ID>.csv directly on server so Jetsons don't write CSV locally.
        # Compatible with tools/profile_collect.py and training scripts schema.
        csv_path = self._root / f"calibration_{node_id}.csv"

        def _write() -> None:
            csv_exists = csv_path.exists()
            with csv_path.open("a") as cf:
                if not csv_exists:
                    header = [
                        "ts", "gpu_percent", "cpu_percent", "ram_percent", "gpu_temp_c",
                        "session_id", "sequence",
                        "fps_avg", "n_active_cameras",
                        "n_track_total", "n_plate_total", "stationary_fraction_mean",
                        "offload_crops_received_per_s", "load_score",
                    ]
                    cf.write(",".join(header) + "\n")
                row = [
                    str(rec.get("ts", "")),
                    str(rec.get("gpu_percent", "")),
                    str(rec.get("cpu_percent", "")),
                    str(rec.get("ram_percent", "")),
                    str(rec.get("gpu_temp_c", "")),
                    str(rec.get("session_id", "")),
                    str(rec.get("sequence", "")),
                    str(rec.get("fps_avg", "")),
                    str(rec.get("active_cameras_count", "")),
                    str(rec.get("n_track_total", "")),
                    str(rec.get("n_plate_total", "")),
                    str(rec.get("stationary_fraction_mean", "")),
                    str(rec.get("offload_crops_received_per_s", "")),
                    str(rec.get("load_score", "")),
                ]
                cf.write(",".join(row) + "\n")

        async with self._get_lock(node_id):
            try:
                await asyncio.to_thread(_write)
            except Exception as exc:
                logger.error("Failed to write telemetry for %s: %s", node_id, exc)