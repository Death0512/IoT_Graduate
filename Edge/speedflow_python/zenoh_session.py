"""
speedflow_python/zenoh_session.py

Factory for a Zenoh peer-mode session on the LAN.
All modules call make_session() to get a configured session.

No broker required — Zenoh peer mode uses UDP multicast scouting
so nodes on the same LAN switch discover each other automatically.
"""

from __future__ import annotations

import zenoh

from .settings import ZENOH_ROUTER

def make_config() -> zenoh.Config:
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"peer"')
    cfg.insert_json5("scouting/multicast/enabled", "true")

    # Cross-network router / server endpoint (e.g. "tcp/116.118.9.125:7447")
    if ZENOH_ROUTER:
        cfg.insert_json5("connect/endpoints", f'["{ZENOH_ROUTER}"]')

    return cfg

def make_session() -> zenoh.Session:
    return zenoh.open(make_config())
