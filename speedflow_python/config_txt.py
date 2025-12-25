# config_txt.py
from pathlib import Path

REQUIRED_KEYS = {"ANALYTICS_CFG", "HOMO_YML", "VIDEO_FPS"}

def load_kv_txt(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config TXT not found: {p}")
    data = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid line in {p}: {line!r} (expected key=value)")
            k, v = [x.strip() for x in line.split("=", 1)]
            data[k] = v
    # validate
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"Missing required keys in TXT: {missing}")
    # cast
    try:
        data["VIDEO_FPS"] = float(data["VIDEO_FPS"])
    except Exception:
        raise ValueError("VIDEO_FPS must be a number (int/float)")
    return data
