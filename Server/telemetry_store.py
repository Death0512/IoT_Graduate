from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("telemetry_store")

# Fields persisted from the health payload; intentionally NOT the full payload.
_KEPT_FIELDS = (
    "ts", "node_id", "session_id", "sequence", "load_score", "gpu_percent",
    "cpu_percent", "ram_percent", "gpu_temp_c", "power_mw", "n_track_total",
    "n_plate_total", "stationary_fraction_mean", "fps_avg",
    "offload_crops_received_per_s", "active_cameras_count", "pipeline_status",
    "lpr_queue_ratio", "lpr_queue_depth", "offload_queue_full",
    "offload_queue_depth", "offload_queue_depth_ratio",
    "camera_levels", "camera_targets", "stream_pressure",
)

# Camera per-stream FPS keys are bare floats (not prefixed with "_") — keep them.
_FPS_PREFIX_IGNORED = "_"

_LEGACY_CALIBRATION_HEADER = [
    "ts", "gpu_percent", "cpu_percent", "ram_percent", "gpu_temp_c",
    "session_id", "sequence",
    "fps_avg", "n_active_cameras",
    "n_track_total", "n_plate_total", "stationary_fraction_mean",
    "offload_crops_received_per_s", "load_score",
]

_EXTENDED_TELEMETRY_HEADER = [
    "ts", "gpu_percent", "cpu_percent", "ram_percent", "gpu_temp_c",
    "session_id", "sequence",
    "fps_avg", "n_active_cameras",
    "n_track_total", "n_plate_total", "stationary_fraction_mean",
    "offload_crops_received_per_s", "load_score",
    "lpr_queue_ratio", "lpr_queue_depth", "offload_queue_full",
    "offload_queue_depth", "offload_queue_depth_ratio",
    "camera_levels", "camera_targets", "stream_pressure",
]

_EVENT_HEADER = [
    "timestamp_iso", "from_node", "to_node", "camera_id",
    "trigger_reason", "trigger_load", "trigger_fps",
    "duration_ms", "result", "blind_spot_ms",
    "stream_pressure", "epoch",
]


def _format_cell(val: Any) -> Any:
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, separators=(",", ":"))
    return val


def _append_csv_safely(path: Path, header: List[str], row: List[Any]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)
        return

    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
    except Exception as exc:
        logger.warning("Could not read header of %s: %s", path.name, exc)
        return

    if file_header != header:
        logger.warning(
            "Skipping append to %s: existing header schema (%d cols) differs from expected (%d cols)",
            path.name,
            len(file_header) if file_header else 0,
            len(header),
        )
        return

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


"""
Server/telemetry_store.py — Production Telemetry Recording

TelemetryStore — Persists 1 Hz edge node telemetry and migration events into CSV files.

This is the PRODUCTION telemetry recorder. All model evaluation, load analysis,
and research datasets are derived from these CSV files.

Data Flow:
  Edge HealthAgent (1 Hz) → Zenoh peers/status/{node} → Server app.py
    → TelemetryStore.save_async() → CSV append
  Edge PeerOrchestrator → Zenoh peers/events/migration → Server app.py
    → TelemetryStore.save_event_async() → CSV append

File Format:
  Legacy calibration: Server/data/telemetry/calibration_{node_id}.csv (14 columns)
  Extended telemetry: Server/data/telemetry/telemetry_extended_{node_id}.csv (22 columns)
  Migration events:   Server/data/telemetry/migration_events.csv (12 columns)

Key Design Decisions:
- Async write via asyncio.to_thread: Never blocks the aiohttp event loop
- Deduplication: (session_id, sequence) pair must advance; duplicate heartbeats dropped
- Per-node asyncio.Lock: Serializes appends to same CSV (prevents interleaving)
- Event asyncio.Lock: Serializes event appends to migration_events.csv
- Derived fields computed server-side: n_track_total, n_plate_total,
  stationary_fraction_mean, fps_avg (from per-camera fps_per_camera)
- FPS keys starting with "_" ignored (internal bookkeeping, not real cameras)
- Safe CSV escaping via csv.writer; dict fields serialized as JSON
- Strict schema check: never appends mismatched row to existing CSV file
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
        self._event_lock: Optional[asyncio.Lock] = None
        # Last persisted (session_id, sequence) per node — dedup key so the
        # same pipeline telemetry window is never written twice.
        self._last_seq: Dict[str, tuple] = {}

    def _get_lock(self, node_id: str) -> asyncio.Lock:
        if node_id not in self._locks or self._locks[node_id] is None:
            self._locks[node_id] = asyncio.Lock()
        return self._locks[node_id]

    def _get_event_lock(self) -> asyncio.Lock:
        if self._event_lock is None:
            self._event_lock = asyncio.Lock()
        return self._event_lock

    async def save_async(self, node_id: str, payload: Dict[str, Any]) -> None:
        # Dedup: skip when this row mirrors the last telemetry window for node.
        # Filter out payloads lacking valid session_id/sequence (e.g. startup / stale snapshots)
        sess = payload.get("session_id")
        seq = payload.get("sequence")
        if not (isinstance(seq, int) and isinstance(sess, str) and sess):
            return
        if self._last_seq.get(node_id) == (sess, seq):
            return
        self._last_seq[node_id] = (sess, seq)

        rec = {"ts": time.time(), "node_id": node_id}
        # Edge nests camera run metrics under payload["pipeline"] (older
        # versions may omit it entirely — degrade to empty dicts).
        pipeline_data = payload.get("pipeline") or {}

        for f in _KEPT_FIELDS:
            if f in ("ts", "node_id"):
                continue
            if f in payload and payload[f] is not None:
                rec[f] = payload[f]
            elif f in pipeline_data and pipeline_data[f] is not None:
                rec[f] = pipeline_data[f]

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

        cal_path = self._root / f"calibration_{node_id}.csv"
        ext_path = self._root / f"telemetry_extended_{node_id}.csv"

        # Edge sends pipeline.active_cameras as a list; derive count from it.
        _ac_list = pipeline_data.get("active_cameras")
        if isinstance(_ac_list, (list, tuple, set)):
            active_cams = len(_ac_list)
        else:
            active_cams = rec.get("active_cameras_count", rec.get("n_active_cameras", ""))

        legacy_row = [
            _format_cell(rec.get("ts")),
            _format_cell(rec.get("gpu_percent")),
            _format_cell(rec.get("cpu_percent")),
            _format_cell(rec.get("ram_percent")),
            _format_cell(rec.get("gpu_temp_c")),
            _format_cell(rec.get("session_id")),
            _format_cell(rec.get("sequence")),
            _format_cell(rec.get("fps_avg")),
            _format_cell(active_cams),
            _format_cell(rec.get("n_track_total")),
            _format_cell(rec.get("n_plate_total")),
            _format_cell(rec.get("stationary_fraction_mean")),
            _format_cell(rec.get("offload_crops_received_per_s")),
            _format_cell(rec.get("load_score")),
        ]

        extended_row = [
            _format_cell(rec.get("ts")),
            _format_cell(rec.get("gpu_percent")),
            _format_cell(rec.get("cpu_percent")),
            _format_cell(rec.get("ram_percent")),
            _format_cell(rec.get("gpu_temp_c")),
            _format_cell(rec.get("session_id")),
            _format_cell(rec.get("sequence")),
            _format_cell(rec.get("fps_avg")),
            _format_cell(active_cams),
            _format_cell(rec.get("n_track_total")),
            _format_cell(rec.get("n_plate_total")),
            _format_cell(rec.get("stationary_fraction_mean")),
            _format_cell(rec.get("offload_crops_received_per_s")),
            _format_cell(rec.get("load_score")),
            _format_cell(rec.get("lpr_queue_ratio")),
            _format_cell(rec.get("lpr_queue_depth")),
            _format_cell(rec.get("offload_queue_full")),
            _format_cell(rec.get("offload_queue_depth")),
            _format_cell(rec.get("offload_queue_depth_ratio")),
            _format_cell(rec.get("camera_levels")),
            _format_cell(rec.get("camera_targets")),
            _format_cell(rec.get("stream_pressure")),
        ]

        def _write() -> None:
            _append_csv_safely(cal_path, _LEGACY_CALIBRATION_HEADER, legacy_row)
            _append_csv_safely(ext_path, _EXTENDED_TELEMETRY_HEADER, extended_row)

        async with self._get_lock(node_id):
            try:
                await asyncio.to_thread(_write)
            except Exception as exc:
                logger.error("Failed to write telemetry for %s: %s", node_id, exc)

    async def save_event_async(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return

        event_path = self._root / "migration_events.csv"
        row = [
            _format_cell(event.get("timestamp_iso", time.strftime("%Y-%m-%dT%H:%M:%S"))),
            _format_cell(event.get("from_node", "")),
            _format_cell(event.get("to_node", "")),
            _format_cell(event.get("camera_id", "")),
            _format_cell(event.get("trigger_reason", "")),
            _format_cell(event.get("trigger_load", event.get("load_score", ""))),
            _format_cell(event.get("trigger_fps", "")),
            _format_cell(event.get("duration_ms", event.get("migration_time_ms", ""))),
            _format_cell(event.get("result", "")),
            _format_cell(event.get("blind_spot_ms", "")),
            _format_cell(event.get("stream_pressure", "")),
            _format_cell(event.get("epoch", "")),
        ]

        def _write_event() -> None:
            _append_csv_safely(event_path, _EVENT_HEADER, row)

        async with self._get_event_lock():
            try:
                await asyncio.to_thread(_write_event)
            except Exception as exc:
                logger.error("Failed to write migration event: %s", exc)
