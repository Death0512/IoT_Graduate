from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

logger = logging.getLogger("violation_store")

"""
Server/violation_store.py — Overspeed Violation Persistence

ViolationStore — Persists overspeed violation events to JSONL files.

Data Flow:
  SpeedProbe (probes.py) detects overspeed → emits via Zenoh publisher
    → Server app.py receives → ViolationStore.save_async()

File Format:
  JSONL (one JSON object per line) at:
    Server/data/violations/YYYY-MM-DD/{node_id}/violations.jsonl
  Also saves plate crop JPEG snapshots to same directory.

Key Design Decisions:
- Async write via aiofiles + asyncio.to_thread: Never blocks event loop
- Lazy lock creation (BUG-17 fix): asyncio.Lock created on first use, not at
  __init__ time, to avoid "Future attached to different loop" on Python 3.8-3.9
- Date-based directory rotation: Automatic daily partition
- Per-node isolation: node_id subdirectory under date
- Snapshot JPEG saved alongside JSONL (base64 decoded from record)

Record Schema:
  node_id, camera_id, track_id, speed_kmh, speed_limit_kmh,
  timestamp, plate_text (optional), plate_conf (optional),
  plate_crop_base64 (optional), bbox, world_pos

No local persistence on Jetsons — all violations centralized on Server.
"""
class ViolationStore:
    def __init__(self, data_dir: str | Path) -> None:
        self._root = Path(data_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        # BUG-17 fix: do NOT create asyncio.Lock() at __init__ time.
        # On Python 3.8-3.9, a Lock created before asyncio.run() starts is
        # bound to the default (possibly wrong) event loop and raises
        # "Task got Future attached to a different loop" at runtime.
        # We create it lazily on the first coroutine call instead.
        self._async_write_lock: Optional[asyncio.Lock] = None

    def _get_write_lock(self) -> asyncio.Lock:
        """Return the write lock, creating it lazily inside the running loop."""
        if self._async_write_lock is None:
            self._async_write_lock = asyncio.Lock()
        return self._async_write_lock

    async def save_async(self, record: Dict[str, Any]) -> Optional[Path]:
        node_id = record.get("node_id", "unknown")
        camera_id = record.get("camera_id", "unknown")

        ts_raw = record.get("timestamp") or record.get("ts")
        if ts_raw is None:
            ts = time.time()
        elif isinstance(ts_raw, str):
            try:
                import datetime
                ts = datetime.datetime.fromisoformat(ts_raw).timestamp()
            except ValueError:
                ts = time.time()
        else:
            ts = float(ts_raw)

        if "timestamp" not in record:
            record["timestamp"] = ts

        date_str = time.strftime("%Y-%m-%d", time.localtime(ts))

        node_dir = self._root / date_str / node_id
        await asyncio.to_thread(node_dir.mkdir, parents=True, exist_ok=True)

        jsonl_path = node_dir / "violations.jsonl"
        snapshot_path: Optional[Path] = None

        record = dict(record)
        snapshot_b64 = record.pop("snapshot_b64", None)
        if snapshot_b64:
            ts_ms = int(ts * 1000)
            snapshot_path = node_dir / f"{camera_id}_{ts_ms}.jpg"
            try:
                img_bytes = base64.b64decode(snapshot_b64)
                await asyncio.to_thread(snapshot_path.write_bytes, img_bytes)
                record["snapshot_file"] = str(snapshot_path.relative_to(self._root))
            except Exception as exc:
                logger.warning("Failed to save snapshot for %s/%s: %s", node_id, camera_id, exc)

        async with self._get_write_lock():
            try:
                async with aiofiles.open(jsonl_path, "a") as f:
                    await f.write(json.dumps(record, default=str) + "\n")
            except Exception as exc:
                logger.error("Failed to write JSONL for %s/%s: %s", node_id, camera_id, exc)

        return snapshot_path

    async def query_async(
        self,
        node_id: Optional[str] = None,
        date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        def _query() -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []

            date_dirs: List[Path] = []
            if date:
                d = self._root / date
                if d.is_dir():
                    date_dirs.append(d)
            else:
                for d in sorted(self._root.iterdir(), reverse=True):
                    if d.is_dir():
                        date_dirs.append(d)

            # Fix #11: collect only offset+limit records across all files so the
            # in-memory footprint stays bounded regardless of how many violations
            # are stored.  Records are sorted descending by timestamp, so we read
            # the newest date directories first and stop as soon as we have enough.
            need = offset + limit

            for date_dir in date_dirs:
                node_dirs: List[Path] = []
                if node_id:
                    nd = date_dir / node_id
                    if nd.is_dir():
                        node_dirs.append(nd)
                else:
                    for nd in date_dir.iterdir():
                        if nd.is_dir():
                            node_dirs.append(nd)

                for nd in node_dirs:
                    jsonl = nd / "violations.jsonl"
                    if not jsonl.exists():
                        continue
                    try:
                        with open(jsonl) as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        rec = json.loads(line)
                                        results.append(rec)
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as exc:
                        logger.warning("Failed to read %s: %s", jsonl, exc)

                # Early-exit: once we have more than enough records for this
                # page we can stop reading further (older) date directories.
                if len(results) >= need:
                    break

            results.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
            return results[offset:offset + limit]

        return await asyncio.to_thread(_query)
