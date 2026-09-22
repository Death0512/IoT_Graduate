# speedflow_python/sys_telemetry.py
# -*- coding: utf-8 -*-
"""
Direct /proc + /sys hardware telemetry for Jetson Orin — no daemon, no IPC, no blocking.

Replaces jtop as the telemetry source for health_agent.py and profile_collect.py.
All reads are synchronous file reads (non-blocking); on any error the metric defaults
to 0.0 so callers always receive a valid dict, never None.
"""
from __future__ import annotations

import os
import time
from typing import Optional

_GPU_LOAD_PATH = "/sys/devices/platform/bus@0/17000000.gpu/load"
_PROC_STAT_PATH = "/proc/stat"
_PROC_MEMINFO_PATH = "/proc/meminfo"
_THERMAL_BASE = "/sys/class/thermal"

# Cache gpu-thermal zone path (found once at import time, re-checked lazily if missing)
_gpu_thermal_path: Optional[str] = None
_gpu_thermal_checked: float = 0.0

def _find_gpu_thermal_path() -> Optional[str]:
    """Find the thermal zone file for GPU temperature. Cached after first successful lookup."""
    global _gpu_thermal_path, _gpu_thermal_checked
    now = time.monotonic()
    if _gpu_thermal_path and now - _gpu_thermal_checked < 300.0:
        return _gpu_thermal_path
    _gpu_thermal_checked = now
    try:
        for entry in os.scandir(_THERMAL_BASE):
            if not entry.name.startswith("thermal_zone"):
                continue
            try:
                tp = open(os.path.join(entry.path, "type")).read().strip()
            except OSError:
                continue
            if tp == "gpu-thermal" or "gpu" in tp.lower():
                candidate = os.path.join(entry.path, "temp")
                if os.path.exists(candidate):
                    _gpu_thermal_path = candidate
                    return _gpu_thermal_path
    except OSError:
        pass
    return None

def read_gpu_percent() -> float:
    """GPU utilization in percent [0.0, 100.0]. Source: /sys/.../gpu/load (per-mille)."""
    try:
        with open(_GPU_LOAD_PATH) as f:
            return min(100.0, max(0.0, int(f.read().strip()) / 10.0))
    except Exception:
        return 0.0

def read_gpu_temp_c() -> float:
    """GPU temperature in Celsius. Source: gpu-thermal zone /temp (millidegrees)."""
    path = _find_gpu_thermal_path()
    if path is None:
        return 0.0
    try:
        with open(path) as f:
            v = int(f.read().strip())
        return round(v / 1000.0, 1)
    except Exception:
        return 0.0

def read_cpu_percent() -> float:
    """CPU utilization in percent [0.0, 100.0]. Source: /proc/stat first line."""
    try:
        with open(_PROC_STAT_PATH) as f:
            line = f.readline()
        parts = line.split()
        if parts[0] != "cpu" or len(parts) < 5:
            return 0.0
        fields = [int(x) for x in parts[1:]]
        # idle = fields[3], iowait = fields[4] if present
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        total = sum(fields)
        if total == 0:
            return 0.0
        return round(max(0.0, min(100.0, (1.0 - idle / total) * 100.0)), 1)
    except Exception:
        return 0.0

def read_ram_percent() -> float:
    """RAM utilization in percent [0.0, 100.0]. Source: /proc/meminfo."""
    try:
        mem_total = mem_available = None
        with open(_PROC_MEMINFO_PATH) as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1])
                if mem_total is not None and mem_available is not None:
                    break
        if not mem_total:
            return 0.0
        used = mem_total - (mem_available or 0)
        return round(max(0.0, min(100.0, used / mem_total * 100.0)), 1)
    except Exception:
        return 0.0

def read_hw_metrics() -> dict:
    """Return hardware metrics dict matching the legacy jtop dict shape.

    Always returns a valid dict (never raises). Values default to 0.0 on any
    read error so callers never need to handle None.
    """
    return {
        "gpu_percent": read_gpu_percent(),
        "cpu_percent": read_cpu_percent(),
        "ram_percent": read_ram_percent(),
        "gpu_temp_c":  read_gpu_temp_c(),
        "power_mw":    0.0,   # ponytail: no reliable sysfs equivalent; jtop-only metric
        "source":      "sysfs",
    }
