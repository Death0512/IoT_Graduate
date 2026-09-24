#!/usr/bin/env python3
"""
Python backend runner — Main entry point for the Edge DeepStream pipeline.

This module orchestrates the complete pipeline lifecycle:
- Builds the multi-stream DeepStream pipeline via core_pipeline.build_pipeline()
- Attaches pad probes (ROI filter, SpeedProbe) for analytics and telemetry
- Starts the HealthAgent (telemetry collection) and Zenoh publishers/subscribers
- Runs the GLib main loop with error/EOS handling
- Supports three sink modes: display (local GUI), file (recording), rtsp_push (MediaMTX)

Architecture:
    run_python.py (this file)
        ├── core_pipeline.py — GStreamer pipeline construction & dynamic ADD/REMOVE
        ├── probes.py — Pad probes: ROI filter, SpeedProbe (speed, plate, telemetry)
        ├── health_agent.py — Hardware telemetry + load scoring + Zenoh heartbeat
        ├── membership.py — Peer state, offload ladder (L0/L1/L2), reclaim, HRW hash
        ├── ownership.py — Camera ownership epochs, holder_seq, make-before-break
        ├── offload.py — L1 plate-crop offload, L2 full-stream RFO logic
        ├── offload_publisher.py — Zenoh plate-crop sender (bounded queue)
        ├── offload_receiver.py — Zenoh plate-crop receiver + worker pool
        ├── lpr_worker.py — Local/offload plate OCR (ONNX Runtime / TensorRT)
        ├── zenoh_publisher.py — Async Zenoh event publisher (overspeed, telemetry)
        ├── zenoh_subscriber.py — Zenoh control commands (ADD/REMOVE/REPAIR)
        └── camera_config.py — Camera configuration management

Three run modes (selected by run_python_mode()):
    1. display:   Local EGL sink — for Jetson with display attached
    2. file:      MP4 recording — for data collection / debugging
    3. rtsp_push: Push annotated streams to MediaMTX — production mode

Key design decisions:
- HEALTH_INTERVAL=1.0 and TELEMETRY_INTERVAL=1.0 are the ONLY supported cadences.
- EOS discrimination: RTSP sources auto-reconnect via rtspsrc reconnect-delay=5;
  only file sources trigger pipeline quit (fix-8).
- HealthAgent runs as in-process daemon thread (not separate process).
- Dual EMA: workload axis alpha_ema=0.33, score output load_score_alpha=0.20.
- Lazy frame fetch: NVMM→numpy copy deferred until plate reaches OCR submission.
"""
import sys
import os
import logging
import time
import threading
from pathlib import Path
from typing import Optional

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from .core_pipeline import (
    build_pipeline,
    dynamic_add_stream,
    dynamic_remove_stream,
    rebuild_rtsp_push_sink,
    PublisherRecovery,
    classify_pipeline_error,
    handle_publisher_failure,
)
from .camera_config import CameraManager
from .probes import SpeedProbe, ROIFilterProbe
from .lpr_worker import LocalLprWorker
from .offload_publisher import OffloadPublisher
from .offload_receiver import OffloadReceiver
from .common import is_file_uri
from . import settings as S
from .settings import (
    CAMERAS_YML,
    NODE_ID,
    RTSP_PUSH_BITRATE,
    RTSP_PUSH_MAX_RETRIES,
    RTSP_PUSH_RETRY_DELAY_S,
    LPR_ENGINE,
    LPR_LABELS,
)

import yaml

logger = logging.getLogger(__name__)

# Holder for the live SpeedProbe.  Set by the mode runners when a probe is
# created (rtsp_push restarts create a fresh probe per iteration).
ACTIVE_SPEED_PROBE: list = []

def _stop_active_speed_probes() -> None:
    """Stop/join FPS writer on all probes in ACTIVE_SPEED_PROBE and clear the list."""
    for p in ACTIVE_SPEED_PROBE:
        try:
            p.stop_fps_writer()
        except Exception:
            pass
    ACTIVE_SPEED_PROBE.clear()

# ---------------------------------------------------------------------------
# Shared probe setup
# ---------------------------------------------------------------------------

def _setup_probes(pipeline: Gst.Pipeline, nvdsosd: Gst.Element,
                  camera_manager: CameraManager,
                  sink_type: str = "display",
                  peer_orch=None,
                  offload_pub=None,
                  offload_rcv=None,
                  zenoh_pub=None,
                  lpr_worker=None) -> SpeedProbe:
    """
    Attach ROI filter, plate preprocessor, and speed probe to *pipeline*.
    Returns the SpeedProbe instance.

    peer_orch:   PeerOrchestrator — lets the probe query offload levels.
    offload_pub: OffloadPublisher — lets the probe send crops to peers.
    offload_rcv: OffloadReceiver — lets peer-returned crop results reach the probe.
    zenoh_pub:   ZenohPublisher — lets the probe publish overspeed events.
    lpr_worker:  LocalLprWorker — local plate-crop LPR off the DeepStream graph.
    """
    # 1. ROI filter
    analytics = pipeline.get_by_name("analytics")
    if analytics:
        roi_filter = ROIFilterProbe(camera_manager)
        analytics_srcpad = analytics.get_static_pad("src")
        if analytics_srcpad:
            analytics_srcpad.add_probe(
                Gst.PadProbeType.BUFFER,
                roi_filter.analytics_src_pad_buffer_probe,
                None,
            )
            print("[ROI Filter] Enabled (Multi-Stream Python Mode)")

    # 2. Speed + LPR probe (pass peer_orch so it can query offload levels)
    probe = SpeedProbe(camera_manager, peer_orch=peer_orch)
    # Keep PGIE interval fixed at 3. Adaptive switching is intentionally disabled.
    pgie_elem = pipeline.get_by_name("primary-infer")
    if pgie_elem is not None:
        pgie_elem.set_property("interval", 3)
    # Wire camera -> source_type ("live" | "file") so probes.py can
    # publish source_modes in _telemetry and downstream consumers can
    # distinguish decoder-throughput file playback from live source FPS.
    # ponytail: module-level is_file_uri reused
    _source_types = {}
    for _c in camera_manager.get_enabled_configs():
        _source_types[_c.camera_id] = "file" if is_file_uri(_c.uri or "") else "live"
    probe.set_source_types(_source_types)
    if offload_pub is not None:
        probe.set_offload_publisher(offload_pub)
    if offload_rcv is not None:
        offload_rcv.set_result_handler(probe.inject_offload_result)
        probe.set_offload_receiver(offload_rcv)
    if lpr_worker is not None:
        probe.set_lpr_worker(lpr_worker)
        lpr_worker.set_result_sink(probe.inject_offload_result)
    if zenoh_pub is not None:
        probe.set_publisher(zenoh_pub)

    tiler = pipeline.get_by_name("tiler")
    if tiler:
        pad = tiler.get_static_pad("sink")
        pad_name = "tiler sink"
    elif sink_type in ("rtsp_push", "file"):
        # Per-branch OSD: attach one probe per camera to its OSD sink pad.
        # Each branch's OSD element is named osd_{sink_type}_{sid}.
        osd_prefix = "osd_rtsp_push" if sink_type == "rtsp_push" else "osd_file"
        attached = 0
        for _c in camera_manager.get_enabled_configs():
            osd_el = pipeline.get_by_name(f"{osd_prefix}_{_c.source_id}")
            if osd_el is not None:
                osd_pad = osd_el.get_static_pad("sink")
                if osd_pad is not None:
                    osd_pad.add_probe(
                        Gst.PadProbeType.BUFFER,
                        probe.osd_sink_pad_buffer_probe,
                        None,
                    )
                    attached += 1
        if attached == 0:
            # Fallback: no per-branch OSD found (should not happen)
            raise RuntimeError(
                f"No per-branch OSD found for sink_type={sink_type}. "
                "Expected osd_{{rtsp_push,file}}_{{sid}} elements."
            )
        pad = None  # no single pad; probes already attached
    else:
        pad = nvdsosd.get_static_pad("sink") if nvdsosd else None
        pad_name = "nvdsosd sink"

    if pad is not None:
        pad.add_probe(Gst.PadProbeType.BUFFER, probe.osd_sink_pad_buffer_probe, None)

    return probe

# ---------------------------------------------------------------------------
# Dynamic Hooks Setup
# ---------------------------------------------------------------------------

def _attach_camera_manager(
    camera_manager: CameraManager,
    pipeline: Gst.Pipeline,
    streammux: Gst.Element,
    source_bins: dict,
    tiler: Gst.Element = None,
    rtsp_push_base_url: Optional[str] = None,
    rtsp_push_bitrate: Optional[int] = None,
    node_camera_map: Optional[dict] = None,
    recovery: Optional["PublisherRecovery"] = None,
):
    """
    Hooks up the CameraManager to safely add/remove streams dynamically.

    Thread-safety note:
        source_id_to_cam_id is ONLY written inside on_add/on_remove,
        which are always called via GLib.idle_add → run on GLib Main Loop
        thread → no concurrent mutation possible.
    """
    # Reverse mapping: source_id (int) → camera_id (str)
    # Initialized from enabled cameras when pipeline starts.
    # Only read/written from GLib Main Loop thread (via idle_add).
    source_id_to_cam_id: dict[int, str] = {
        cfg.source_id: cfg.camera_id
        for cfg in camera_manager.get_enabled_configs()
    }

    # Grab the live SpeedProbe (may be None during early init).
    active_probe = ACTIVE_SPEED_PROBE[0] if ACTIVE_SPEED_PROBE else None

    def on_add(cam_cfg):
        print(f"[Dynamic] Adding camera '{cam_cfg.camera_id}' (source_id={cam_cfg.source_id})")
        ready_ev = camera_manager.stream_ready_event(cam_cfg.source_id)
        try:
            dynamic_add_stream(
                pipeline, streammux, cam_cfg, tiler, source_bins,
                ready_event=ready_ev,
                rtsp_push_base_url=rtsp_push_base_url,
                rtsp_push_bitrate=rtsp_push_bitrate,
                node_camera_map=node_camera_map,
                probe=active_probe,
            )
            # Register mapping immediately after successful add.
            # This function runs in GLib Main Loop → safe, no lock needed.
            source_id_to_cam_id[cam_cfg.source_id] = cam_cfg.camera_id
            # Update source_modes so dynamic/migrated cameras appear in
            # _derive_camera_liveness — missing entry excludes them from
            # active_cameras and triggers phantom reclaim cycles.
            # ponytail: main-loop thread, no lock needed (same thread as set_source_types).
            for p in ACTIVE_SPEED_PROBE:
                p._source_type_by_camera[cam_cfg.camera_id] = (
                    "file" if is_file_uri(cam_cfg.uri or "") else "live"
                )
        except Exception as exc:
            print(f"[Dynamic] ERROR adding camera '{cam_cfg.camera_id}': {exc}", file=sys.stderr)
            with camera_manager._lock:
                if cam_cfg.camera_id in camera_manager._configs:
                    camera_manager._configs[cam_cfg.camera_id].enabled = False
                    camera_manager._rebuild_lookup()
            camera_manager.cleanup_stream_ready(cam_cfg.source_id)
            # Fail loud via logs, never re-raise from GLib idle callbacks.
            # Raising here kills the GLib idle callback with CRITICAL root.

    def on_remove(source_id, done_event=None):
        # Look up camera_id from the mapping dict.
        # Not using GStreamer pad scan because it's complex and not thread-safe.
        cam_id = source_id_to_cam_id.get(source_id)
        if cam_id is None:
            print(
                f"[Dynamic] WARN: No camera mapped to source_id={source_id}. "
                "Possibly already removed or never registered.",
                file=sys.stderr
            )
            if done_event is not None:
                done_event.set()
            return

        # P1: flag intentional teardown so any publisher (rtspclientsink) error
        # emitted while this branch is being torn down is NOT treated as an
        # unexpected failure requiring recovery.
        if recovery is not None:
            recovery.mark_intentional_teardown(cam_id)

        print(f"[Dynamic] Removing camera '{cam_id}' (source_id={source_id})")
        try:
            dynamic_remove_stream(
                pipeline, streammux, cam_id, source_id, tiler, source_bins, done_event=done_event
            )
        except Exception as exc:
            print(f"[Dynamic] ERROR removing camera '{cam_id}': {exc}", file=sys.stderr)
            raise
        finally:
            # Clean up key after initiating/scheduling removal.
            # Prevents memory leak during continuous operation over months
            # and avoids source_id conflicts if the same ID is reused later.
            removed = source_id_to_cam_id.pop(source_id, None)
            if removed:
                print(f"[Dynamic] Cleaned up mapping: source_id={source_id} → '{removed}'")
            camera_manager.cleanup_stream_ready(source_id)
            for p in ACTIVE_SPEED_PROBE:
                p.remove_camera(cam_id, source_id=source_id)
            # Clear the intentional-teardown flag after a short cooldown so
            # async publisher errors caused by the teardown are absorbed, but a
            # later real failure on this camera can still recover.
            if recovery is not None:
                _cid = cam_id
                GLib.timeout_add(2000, lambda: (recovery.clear_intentional_teardown(_cid), False)[1])

    camera_manager.start(on_add, on_remove, GLib.idle_add)

    def _diagnose_wedged_bin(camera_id, source_id):
        """Read-only snapshot of a starved source bin (never raises).

        Distinguishes 'no RTP arrived' (network/server) from 'RTP arrived
        but no decode pad' (decodebin/main-loop stall) at recreate-retry
        time, so the next starvation carries its cause in the log.
        """
        try:
            src = source_bins.get(camera_id)
            if src is None:
                return "no-bin"
            try:
                _, bstate, _ = src.get_state(0)
                bstate = bstate.value_nick if bstate is not None else "?"
            except Exception:
                bstate = "?"
            rtspsrc = [None]

            def _walk(el, depth=0):
                if rtspsrc[0] is not None or depth > 6:
                    return
                try:
                    fac = el.get_factory()
                    if fac is not None and fac.get_name() == "rtspsrc":
                        rtspsrc[0] = el
                        return
                except Exception:
                    return
                try:
                    it = el.iterate_elements()
                except Exception:
                    return
                while True:
                    try:
                        res, ch = it.next()
                    except Exception:
                        break
                    if res != Gst.IteratorResult.OK:
                        break
                    _walk(ch, depth + 1)

            try:
                _walk(src)
            except Exception:
                pass
            rs = rtspsrc[0]
            if rs is None:
                return f"bin={bstate} rtspsrc=not-found"
            try:
                _, rstate, _ = rs.get_state(0)
                rstate = rstate.value_nick if rstate is not None else "?"
            except Exception:
                rstate = "?"
            stats = ""
            try:
                st = rs.get_property("stats")
                if st is not None:
                    parts = []
                    for key in ("packets-received", "bytes-received", "packets-lost"):
                        try:
                            if st.has_field(key):
                                parts.append(f"{key}={st.get_value(key)}")
                        except Exception:
                            pass
                    stats = " " + " ".join(parts) if parts else " no-counters"
            except Exception:
                stats = " stats-unavailable"
            mux_linked = "?"
            try:
                mp = streammux.get_static_pad(f"sink_{source_id}")
                mux_linked = str(bool(mp is not None and mp.is_linked()))
            except Exception:
                pass
            return f"bin={bstate} rtspsrc={rstate}{stats} muxpad_linked={mux_linked}"
        except Exception as exc:
            return f"unavailable: {exc}"

    camera_manager.bin_diagnose_fn = _diagnose_wedged_bin

# ---------------------------------------------------------------------------
# GLib bus helpers
# ---------------------------------------------------------------------------

# Track NVMM buffer errors to detect persistent decoder starvation
_NVMM_ERROR_TIMESTAMPS: dict = {}  # src_name -> list of timestamps
_NVMM_ERROR_RATE_LIMIT = 10        # errors in 30s triggers critical warning

def _is_transient_nvmm_buffer_error(err, debug: Optional[str], src_name: str = "unknown") -> bool:
    """Narrow check for transient decoder buffer exhaustion.

    Returns True if the error is considered transient (below rate limit).
    Returns False if it is not an NVMM buffer error or if the rate limit is exceeded
    (so callers can trigger recovery/restart/source removal).
    """
    text = f"{err} {debug or ''}"
    if "OutputBufferUnavailable" not in text and "cbAllocPictureBuffer" not in text:
        return False

    now = time.monotonic()
    hist = _NVMM_ERROR_TIMESTAMPS.setdefault(src_name, [])
    hist.append(now)
    _NVMM_ERROR_TIMESTAMPS[src_name] = [t for t in hist if now - t < 30.0]
    if len(_NVMM_ERROR_TIMESTAMPS[src_name]) >= _NVMM_ERROR_RATE_LIMIT:
        logger.critical(
            "[Pipeline] Frequent NVDEC buffer starvation on %s (%d errors in 30s) — stream degraded",
            src_name, len(_NVMM_ERROR_TIMESTAMPS[src_name]),
        )
        _NVMM_ERROR_TIMESTAMPS[src_name].clear()
        return False
    return True

def _graceful_stop_pipeline(pipeline: Gst.Pipeline) -> None:
    """
    Tear down a GStreamer pipeline safely on Jetson (Tegra).
    Transitions sequentially PLAYING -> PAUSED -> READY -> NULL
    with get_state() checks to allow hardware NVDEC/NVENC blocks
    to release registers and kernel dmabuf pools gracefully.
    """
    if pipeline is None:
        return
    try:
        # 1. PAUSED — stops data flow, drains in-flight buffers
        pipeline.set_state(Gst.State.PAUSED)
        pipeline.get_state(1 * Gst.SECOND)
        # 2. READY — tears down element-specific hardware contexts (NVDEC/NVENC)
        pipeline.set_state(Gst.State.READY)
        pipeline.get_state(1 * Gst.SECOND)
        # 3. NULL — frees GStreamer structures and closes file descriptors
        pipeline.set_state(Gst.State.NULL)
        pipeline.get_state(1 * Gst.SECOND)
    except Exception:
        try:
            pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass

def _run_loop_until_eos_or_error(
    pipeline: Gst.Pipeline,
    camera_manager: CameraManager
) -> None:
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src_name = message.src.get_name() if message.src else "unknown"
            if _is_transient_nvmm_buffer_error(err, debug, src_name):
                print(f"WARNING (recoverable): Transient decoder buffer exhaustion from {src_name}: {err}", file=sys.stderr)
                if debug:
                    print(f"DEBUG INFO: {debug}", file=sys.stderr)
                return

            print(f"ERROR from {src_name}: {err}", file=sys.stderr)
            if debug:
                print(f"DEBUG INFO: {debug}", file=sys.stderr)
            loop.quit()
        elif t == Gst.MessageType.EOS:
            print("EOS received — processing complete")
            loop.quit()

    bus.connect("message", on_message)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        camera_manager.stop()
        _graceful_stop_pipeline(pipeline)
        print("Pipeline stopped")

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_display_mode(args, camera_manager: CameraManager, peer_orch=None, offload_pub=None, offload_rcv=None, zenoh_pub=None, lpr_worker=None) -> SpeedProbe:
    Gst.init(None)
    configs = camera_manager.get_enabled_configs()

    ret_build = build_pipeline(
        camera_configs=configs,
        sink_type="display",
        mux_width=args.width,
            mux_height=args.height,
        )
    pipeline, nvdsosd, streammux, source_bins = ret_build
    tiler = pipeline.get_by_name("tiler")

    # Stop any previous active probe's FPS writer before creating a new one
    _stop_active_speed_probes()

    probe = _setup_probes(pipeline, nvdsosd, camera_manager, sink_type="display",
                          peer_orch=peer_orch, offload_pub=offload_pub, offload_rcv=offload_rcv,
                          zenoh_pub=zenoh_pub, lpr_worker=lpr_worker)
    ACTIVE_SPEED_PROBE.append(probe)
    _attach_camera_manager(camera_manager, pipeline, streammux, source_bins, tiler)

    logger.info("[Pipeline] set_state(PLAYING) BEGIN: mode=display, mono_ts=%.6f", time.monotonic())
    t0_playing = time.monotonic()
    ret = pipeline.set_state(Gst.State.PLAYING)
    t1_playing = time.monotonic()
    logger.info("[Pipeline] set_state(PLAYING) END: mode=display, result=%s, duration_ms=%.2f, mono_ts=%.6f", ret.value_nick if hasattr(ret, "value_nick") else ret, (t1_playing - t0_playing) * 1000.0, t1_playing)
    if ret == Gst.StateChangeReturn.FAILURE:
        _stop_active_speed_probes()
        _graceful_stop_pipeline(pipeline)
        raise RuntimeError("Unable to set display pipeline to PLAYING state")
    warmup_ms = (t1_playing - t0_playing) * 1000.0
    probe.record_warmup_ms(warmup_ms)
    if peer_orch is not None:
        peer_orch.set_pipeline_ready(True)
    logger.info("[Display] Pipeline PLAYING after %.0f ms (warmup)", warmup_ms)

    try:
        _run_loop_until_eos_or_error(pipeline, camera_manager)
    finally:
        _stop_active_speed_probes()
    return probe

def run_file_mode(args, camera_manager: CameraManager, peer_orch=None, offload_pub=None, offload_rcv=None, zenoh_pub=None, lpr_worker=None) -> SpeedProbe:
    Gst.init(None)
    configs = camera_manager.get_enabled_configs()

    ret_build = build_pipeline(
        camera_configs=configs,
        sink_type="file",
        mux_width=args.width,
        mux_height=args.height,
        )
    pipeline, nvdsosd, streammux, source_bins = ret_build

    # Stop any previous active probe's FPS writer before creating a new one
    _stop_active_speed_probes()

    probe = _setup_probes(pipeline, nvdsosd, camera_manager, sink_type="file",
                          peer_orch=peer_orch, offload_pub=offload_pub, offload_rcv=offload_rcv,
                          zenoh_pub=zenoh_pub, lpr_worker=lpr_worker)
    ACTIVE_SPEED_PROBE.append(probe)
    _attach_camera_manager(camera_manager, pipeline, streammux, source_bins, None)

    print(f"[Python File Mode] Processing multi-streams to output files...")
    logger.info("[Pipeline] set_state(PLAYING) BEGIN: mode=file, mono_ts=%.6f", time.monotonic())
    t0_playing = time.monotonic()
    ret = pipeline.set_state(Gst.State.PLAYING)
    t1_playing = time.monotonic()
    logger.info("[Pipeline] set_state(PLAYING) END: mode=file, result=%s, duration_ms=%.2f, mono_ts=%.6f", ret.value_nick if hasattr(ret, "value_nick") else ret, (t1_playing - t0_playing) * 1000.0, t1_playing)
    if ret == Gst.StateChangeReturn.FAILURE:
        _stop_active_speed_probes()
        _graceful_stop_pipeline(pipeline)
        raise RuntimeError("Unable to set file pipeline to PLAYING state")
    warmup_ms = (t1_playing - t0_playing) * 1000.0
    probe.record_warmup_ms(warmup_ms)
    if peer_orch is not None:
        peer_orch.set_pipeline_ready(True)

    try:
        _run_loop_until_eos_or_error(pipeline, camera_manager)
    finally:
        _stop_active_speed_probes()
    return probe

def run_rtsp_push_mode(args, camera_manager: CameraManager, peer_orch=None, offload_pub=None, offload_rcv=None, zenoh_pub=None, lpr_worker=None) -> Optional["SpeedProbe"]:
    Gst.init(None)
    configs = camera_manager.get_enabled_configs()

    rtsp_url = args.rtsp_push_url or S.RTSP_PUSH_URL
    if not rtsp_url:
        raise ValueError("--rtsp-push-url or RTSP_PUSH_URL required")

    # Guard against two instances of this process publishing to the same path.
    # A second instance would cause MediaMTX to kick the first (overridePublisher)
    # leading to an endless reconnect loop and GLib-GIO-CRITICAL socket errors.
    _PID_FILE = S.ROOT / "run_python.pid"
    _my_pid = os.getpid()
    if _PID_FILE.exists():
        try:
            _old_pid = int(_PID_FILE.read_text().strip())
            if _old_pid != _my_pid:
                # Check if that process is actually alive
                try:
                    os.kill(_old_pid, 0)  # signal 0 = existence check
                except OSError:
                    pass  # old process is dead — stale PID file, safe to continue
                else:
                    # A second live instance would publish to the same RTSP push
                    # path and conflict (MediaMTX kicks the older publisher), so
                    # refuse to start rather than warn-and-continue into a fight.
                    logger.critical(
                        "Another pipeline process (PID %d) is already running. "
                        "Refusing to start to avoid RTSP push conflict. "
                        "Kill the old process first: kill %d",
                        _old_pid, _old_pid,
                    )
                    sys.exit(1)
        except ValueError:
            pass
    _PID_FILE.write_text(str(_my_pid))

    import atexit
    atexit.register(lambda: _PID_FILE.unlink(missing_ok=True))

    _RESTART_DELAYS = [5, 10, 20, 30]
    restart_idx = 0
    _last_probe = None
    _last_restart_cause = "initial_start"

    while True:
        # Stop any previous active probe's FPS writer before creating a new one / rebuilding RTSP
        _stop_active_speed_probes()

        print(f"[RTSP Push] Building pipeline (cause: {_last_restart_cause})...")
        rtsp_push_bitrate = RTSP_PUSH_BITRATE
        _edge_node_yml = S.ROOT / "configs" / "edge_node.yml"
        _edge_cfg_local = {}
        try:
            if _edge_node_yml.exists():
                with open(_edge_node_yml, "r") as _f:
                    _edge_cfg_local = yaml.safe_load(_f) or {}
        except Exception:
            pass
        _p2p_local = _edge_cfg_local.get("p2p", {}) if isinstance(_edge_cfg_local, dict) else {}
        _node_cam_map = _p2p_local.get("node_camera_map") if isinstance(_p2p_local, dict) else None
        ret_build = build_pipeline(
            camera_configs=camera_manager.get_enabled_configs(),
            sink_type="rtsp_push",
            mux_width=args.width,
            mux_height=args.height,
            rtsp_push_base_url=rtsp_url,
            rtsp_push_bitrate=rtsp_push_bitrate,
            node_camera_map=_node_cam_map,
        )
        pipeline, nvdsosd, streammux, source_bins = ret_build
        tiler = pipeline.get_by_name("tiler")

        _last_probe = _setup_probes(pipeline, nvdsosd, camera_manager, sink_type="rtsp_push",
                                    peer_orch=peer_orch, offload_pub=offload_pub, offload_rcv=offload_rcv,
                                    zenoh_pub=zenoh_pub, lpr_worker=lpr_worker)
        ACTIVE_SPEED_PROBE.append(_last_probe)

        # P1: per-camera publisher-failure recovery controller. Publisher
        # failures are isolated leaves — they never trigger loop.quit / full
        # DeepStream pipeline rebuild or an ADD/REMOVE storm.
        # Constructed before _attach_camera_manager so the initial attach
        # receives a live recovery controller (not an undefined name).
        recovery = PublisherRecovery(
            max_attempts=RTSP_PUSH_MAX_RETRIES,
            base_delay_s=RTSP_PUSH_RETRY_DELAY_S,
            max_delay_s=60.0,
            reset_after_s=300.0,
        )
        _attach_camera_manager(
            camera_manager, pipeline, streammux, source_bins, tiler,
            rtsp_push_base_url=rtsp_url, rtsp_push_bitrate=rtsp_push_bitrate,
            node_camera_map=_node_cam_map,
            recovery=recovery,
        )

        logger.info("[Pipeline] set_state(PLAYING) BEGIN: mode=rtsp_push, mono_ts=%.6f", time.monotonic())
        t0_playing = time.monotonic()
        ret = pipeline.set_state(Gst.State.PLAYING)
        t1_playing = time.monotonic()
        logger.info("[Pipeline] set_state(PLAYING) END: mode=rtsp_push, result=%s, duration_ms=%.2f, mono_ts=%.6f", ret.value_nick if hasattr(ret, "value_nick") else ret, (t1_playing - t0_playing) * 1000.0, t1_playing)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
            _stop_active_speed_probes()
            _graceful_stop_pipeline(pipeline)
            delay = _RESTART_DELAYS[min(restart_idx, len(_RESTART_DELAYS) - 1)]
            print(f"[RTSP Push] Retrying in {delay}s...")
            restart_idx += 1
            time.sleep(delay)
            continue
        elif ret == Gst.StateChangeReturn.ASYNC:
            state_ret, current_state, pending_state = pipeline.get_state(5 * Gst.SECOND)
            print(f"[RTSP Push] State change ASYNC resolved: return={state_ret.value_nick}, current={current_state.value_nick}, pending={pending_state.value_nick}")
            if state_ret == Gst.StateChangeReturn.FAILURE:
                print("ERROR: Unable to complete pipeline transition to PLAYING state (ASYNC timeout/failure)", file=sys.stderr)
                _stop_active_speed_probes()
                _graceful_stop_pipeline(pipeline)
                delay = _RESTART_DELAYS[min(restart_idx, len(_RESTART_DELAYS) - 1)]
                print(f"[RTSP Push] Retrying in {delay}s...")
                restart_idx += 1
                time.sleep(delay)
                continue
        elif ret == Gst.StateChangeReturn.NO_PREROLL:
            print("[RTSP Push] State change NO_PREROLL: live pipeline running (preroll not required)")
        elif ret == Gst.StateChangeReturn.SUCCESS:
            print("[RTSP Push] State change SUCCESS: pipeline is PLAYING")

        if peer_orch is not None:
            peer_orch.set_pipeline_ready(True)

        restart_idx = 0  # reset backoff on successful start
        print(f"[RTSP Push] Streaming to {rtsp_url}")

        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        _error_flag = [False]
        _error_reason = ["unknown"]
        _removing = set()  # guard against double-remove from multiple error msgs

        def _boot_data_watchdog():
            # ponytail: one-shot boot-stall trip only — catches "PLAYING but zero
            # OSD frames from the start"; won't retrigger on a mid-run stall (a
            # mid-run reader already exists via the publisher-failure branch).
            # Persistent get_fps_stats() (never drained) is the stall signal: a
            # camera that delivered any frame keeps a non-zero snapshot entry,
            # so a sum==0 means NO camera ever produced an OSD buffer.
            _total = sum((_last_probe.get_fps_stats() or {}).values())
            if _total <= 0.0:
                _error_flag[0] = True
                _error_reason[0] = ("boot data stall: PLAYING with pads linked "
                                    "but 0 OSD frame counters after 60s")
                logger.critical("[RTSP Push] CRITICAL %s — quitting GLib loop so "
                                "restart-with-backoff relaunches pipeline", _error_reason[0])
                loop.quit()
            return False

        GLib.timeout_add_seconds(60, _boot_data_watchdog)

        def on_message(bus, message):
            t = message.type
            if t == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                src_name = message.src.get_name() if message.src else "unknown"

                if _is_transient_nvmm_buffer_error(err, debug, src_name):
                    print(f"WARNING (recoverable): Transient decoder buffer exhaustion from {src_name}: {err}", file=sys.stderr)
                    if debug:
                        print(f"DEBUG INFO: {debug}", file=sys.stderr)
                    return

                # If the error comes from a camera branch element (uridecodebin,
                # queue, converter), remove just that stream instead of killing
                # the whole pipeline.
                cam_id = None
                elem = message.src
                while elem is not None:
                    n = elem.get_name()
                    for pfx in ("src-", "q_", "conv_"):
                        if n.startswith(pfx):
                            cand_id = n[len(pfx):]
                            if cand_id in source_bins or cand_id in [c.camera_id for c in camera_manager.get_enabled_configs()]:
                                cam_id = cand_id
                                break
                    if cam_id:
                        break
                    elem = elem.get_parent()

                if cam_id:
                    is_intentional = False
                    if recovery is not None and hasattr(recovery, "is_intentional_teardown"):
                        is_intentional = recovery.is_intentional_teardown(cam_id)
                    if is_intentional or cam_id in _removing:
                        # Teardown artifact (e.g. not-linked, flushing during intentional teardown)
                        print(f"INFO (teardown artifact): absorbed {src_name} ({cam_id}): {err}", file=sys.stderr)
                        return

                    if cam_id in source_bins and cam_id not in _removing:
                        _removing.add(cam_id)
                        print(f"ERROR (source) from {src_name} ({cam_id}): {err}", file=sys.stderr)
                        if debug:
                            print(f"DEBUG INFO: {debug}", file=sys.stderr)
                        print(f"[Pipeline] Removing failed source '{cam_id}'", file=sys.stderr)
                        sid = None
                        for cfg in camera_manager.get_enabled_configs():
                            if cfg.camera_id == cam_id:
                                sid = cfg.source_id
                                break
                        if sid is not None:
                            dynamic_remove_stream(pipeline, streammux, cam_id, sid, tiler, source_bins)
                        return  # don't quit the pipeline

                # Distinguish RTSP push sink (rtsp_push_sink / rtspclientsink / per-camera sinks) vs other pipeline errors
                is_rtsp_sink = False
                sink_el = None
                elem = message.src
                while elem is not None:
                    ename = elem.get_name()
                    fact = elem.get_factory()
                    fact_name = fact.get_name() if fact else ""
                    if ename == "rtsp_push_sink" or ename.startswith("sink_rtsp_push_") or fact_name == "rtspclientsink":
                        is_rtsp_sink = True
                        sink_el = elem
                        break
                    elem = elem.get_parent()

                err_category = classify_pipeline_error(
                    src_name, str(err), debug or "",
                    is_rtsp_sink=is_rtsp_sink,
                )

                if err_category == "publisher":
                    sink_el_name = sink_el.get_name() if sink_el else (message.src.get_name() if message.src else "unknown")
                    print(f"ERROR (publisher_failure) from {sink_el_name}: {err}", file=sys.stderr)
                    if debug:
                        print(f"DEBUG INFO: {debug}", file=sys.stderr)

                    # P1: a publisher failure is a per-camera leaf. Resolve the
                    # camera identity for per-camera push branches; legacy single
                    # sink uses a synthetic key.
                    if sink_el_name.startswith("sink_rtsp_push_"):
                        try:
                            _sid = int(sink_el_name.split("_")[-1])
                        except ValueError:
                            _sid = None
                        _cid = next(
                            (c.camera_id for c in camera_manager.get_enabled_configs() if c.source_id == _sid),
                            f"src_{_sid}" if _sid is not None else "unknown",
                        )
                    else:
                        _cid = "__rtsp_push_legacy__"
                        _sid = None

                    def _rebuild_publisher_branch():
                        if _sid is not None:
                            # _remove_rtsp_push_branch uses _set_state_bounded → done.wait()
                            # which blocks the calling thread. Called from GLib bus callback
                            # (GLib main thread), this would block all GStreamer callbacks
                            # including the EOS drain we just added. Run the rebuild on a
                            # worker thread; block the worker (not GLib) until done. (B1 fix)
                            result = [False]
                            done_ev = threading.Event()

                            def _worker():
                                try:
                                    from .core_pipeline import _remove_rtsp_push_branch, _add_rtsp_push_branch
                                    cam_cfg = next(
                                        (c for c in camera_manager.get_enabled_configs() if c.source_id == _sid),
                                        None,
                                    )
                                    demux = pipeline.get_by_name("demux")
                                    if cam_cfg and demux:
                                        _remove_rtsp_push_branch(pipeline, _sid)
                                        _add_rtsp_push_branch(
                                            pipeline, demux, cam_cfg, rtsp_url,
                                            bitrate=rtsp_push_bitrate, sync=False,
                                            node_camera_map=_node_cam_map,
                                        )
                                        result[0] = True
                                except Exception as rebuild_exc:
                                    print(f"[RTSP Push] Per-camera sink rebuild failed: {rebuild_exc}", file=sys.stderr)
                                finally:
                                    done_ev.set()

                            t = threading.Thread(target=_worker, daemon=True)
                            t.start()
                            done_ev.wait(timeout=30.0)
                            return result[0]
                        return rebuild_rtsp_push_sink(pipeline, rtsp_url, bitrate=S.RTSP_PUSH_BITRATE)

                    def _schedule_publisher_retry(delay):
                        def _fire():
                            recovery.clear_in_flight(_cid)
                            handle_publisher_failure(
                                recovery, _cid, _rebuild_publisher_branch, _schedule_publisher_retry
                            )
                            return False
                        GLib.timeout_add(int(delay * 1000), _fire)

                    result = handle_publisher_failure(
                        recovery, _cid, _rebuild_publisher_branch, _schedule_publisher_retry
                    )
                    if result == "recovered":
                        print(f"[RTSP Push] Publisher branch for '{_cid}' recovered; analytics pipeline PLAYING", file=sys.stderr)
                    elif result == "scheduled":
                        print(f"[RTSP Push] Publisher recovery scheduled for '{_cid}'; pipeline PLAYING", file=sys.stderr)
                    elif result == "intentional":
                        print(f"[RTSP Push] Publisher error on '{_cid}' during intentional teardown — ignored", file=sys.stderr)
                    else:
                        print(
                            f"[RTSP Push] Publisher recovery exhausted/circuit-open for '{_cid}'; "
                            f"branch left DOWN, analytics pipeline PLAYING (no full restart)",
                            file=sys.stderr,
                        )
                    return  # never quit the loop for publisher-only errors

                # Non-RTSP / non-publisher pipeline errors still trigger a full
                # restart (owned by the outer supervisor loop).
                _error_reason[0] = f"{err_category}:{src_name}:{err}"
                print(f"ERROR ({err_category}) from {src_name}: {err}", file=sys.stderr)
                if debug:
                    print(f"DEBUG INFO: {debug}", file=sys.stderr)

                _error_flag[0] = True
                loop.quit()
            elif t == Gst.MessageType.EOS:
                print("EOS received — processing complete")
                loop.quit()

        bus.connect("message", on_message)

        try:
            loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            # BUG-11 fix: stop the manager on intentional exit only (see below)
            camera_manager.stop()
            _stop_active_speed_probes()
            _graceful_stop_pipeline(pipeline)
            print("Pipeline stopped")
            return _last_probe
        finally:
            # BUG-11 fix: do NOT call camera_manager.stop() here on every
            # iteration — that kills the watchdog observer and processor thread
            # permanently.  On a restart those threads must stay alive so that
            # hot-reload and dynamic ADD/REMOVE keep working.
            # Stop the probe FPS writer and clear active probe for this iteration
            _stop_active_speed_probes()
            # We gracefully stop the pipeline state; _attach_camera_manager() at the
            # top of the next iteration re-registers on_add/on_remove callbacks
            # and calls camera_manager.start() again (which is idempotent for
            # the watchdog if it is still running).
            _graceful_stop_pipeline(pipeline)
            print("Pipeline stopped")

        if _error_flag[0]:
            delay = _RESTART_DELAYS[min(restart_idx, len(_RESTART_DELAYS) - 1)]
            _last_restart_cause = _error_reason[0]
            print(f"[RTSP Push] {_last_restart_cause} — reconnecting in {delay}s...", file=sys.stderr)
            restart_idx += 1
            time.sleep(delay)
        else:
            # Clean EOS or intentional stop — do not restart
            _last_restart_cause = "clean_stop"
            break

    # BUG-11 fix: stop the camera manager once, after the restart loop exits.
    camera_manager.stop()
    return _last_probe

# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def run_python_mode(args) -> None:
    """Entry point called by main.py for the Python backend."""
    # Logging handlers: level controlled by S.LOG_LEVEL (default INFO)
    log_level = getattr(logging, S.LOG_LEVEL, logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Clear existing root handlers to avoid duplicate logs
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")

    term_handler = logging.StreamHandler(sys.stderr)
    term_handler.setLevel(log_level)
    term_handler.setFormatter(formatter)
    root_logger.addHandler(term_handler)

    from .log_utils import FlushRotatingFileHandler, install_crash_hooks
    S.PATH_LOGS.mkdir(parents=True, exist_ok=True)
    install_crash_hooks(S.PATH_LOGS)

    debug_log_path = str(S.PATH_LOGS / "edge_debug.log")

    try:
        file_handler = FlushRotatingFileHandler(debug_log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as exc:
        print(f"[Logging] Failed to attach {debug_log_path} handler: {exc}", file=sys.stderr)

    logger.info("[Process] Startup: pid=%d, node_id='%s', mode='%s', args=%s, mono_ts=%.6f", os.getpid(), NODE_ID, args.mode, vars(args), time.monotonic())

    camera_manager = CameraManager(CAMERAS_YML)

    # --- P2P config (edge_node.yml) ---
    edge_cfg = {}
    edge_node_yml = S.ROOT / "configs" / "edge_node.yml"
    try:
        if edge_node_yml.exists():
            with open(edge_node_yml, "r") as f:
                edge_cfg = yaml.safe_load(f) or {}
            print("[P2P] edge_node.yml loaded.")
    except Exception as exc:
        print(f"[P2P] Failed to load edge_node.yml: {exc}", file=sys.stderr)

    # --- P2P Peer Orchestrator (single Zenoh session for ALL P2P traffic) ---
    p2p_cfg = edge_cfg.get("p2p", {})
    peer_orch = None
    try:
        from .peer_orchestrator import PeerOrchestrator
        # P5 — lease persistence path (Edge-local, stdlib-only JSON). Override
        # via LEASE_STATE_PATH; default is <configs_dir>/lease_state.json.
        lease_state_path = os.environ.get("LEASE_STATE_PATH") or (
            edge_cfg.get("_config_dir") if isinstance(edge_cfg, dict) else None
        )
        if not lease_state_path:
            lease_state_path = Path(__file__).resolve().parent.parent / "lease_state.json"
        peer_orch = PeerOrchestrator(
            node_id=NODE_ID,
            cfg=p2p_cfg,
            camera_manager=camera_manager,
            lease_state_path=Path(lease_state_path),
        )
        orch_thread = threading.Thread(target=peer_orch.start, daemon=True)
        logger.info("[Thread] Starting PeerOrchestrator thread: ident=%s, mono_ts=%.6f", orch_thread.name, time.monotonic())
        orch_thread.start()
        peer_orch._ready_event.wait(timeout=5)
        logger.info("[PeerOrch] Started. Node='%s', Overload threshold=%s%%, mono_ts=%.6f", NODE_ID, p2p_cfg.get("overload_threshold", 75.0), time.monotonic())
        print(f"[PeerOrch] Started. Node='{NODE_ID}', Overload threshold={p2p_cfg.get('overload_threshold', 75.0)}%")
    except Exception as exc:
        logger.error("[PeerOrch] Failed to start: %s, mono_ts=%.6f", exc, time.monotonic(), exc_info=True)
        print(f"[PeerOrch] Failed to start: {exc}", file=sys.stderr)
        peer_orch = None

    # --- Health Agent (shares PeerOrchestrator's Zenoh session) ---
    health_agent = None
    try:
        from health_agent import HealthAgent
        ownership_cb = peer_orch.get_ownership_records if peer_orch else None
        held_cb = camera_manager.get_live_held_camera_ids if camera_manager else None
        offload_cb = peer_orch.get_offload_status if peer_orch else None
        health_agent = HealthAgent(
            external_session=peer_orch._session if peer_orch else None,
            ownership_provider=ownership_cb,
            held_provider=held_cb,
            boot_id_provider=(lambda: peer_orch._boot_id) if peer_orch is not None else None,
            offload_provider=offload_cb,
        )
        ha_thread = threading.Thread(target=health_agent.run, daemon=True, name="HealthAgent")
        logger.info("[Thread] Starting HealthAgent thread: ident=%s, mono_ts=%.6f", ha_thread.name, time.monotonic())
        ha_thread.start()
        health_agent._ready_event.wait(timeout=5)
        logger.info("[HealthAgent] Started. Node='%s', Interval=%ss, mono_ts=%.6f", NODE_ID, S.HEALTH_INTERVAL, time.monotonic())
        print(f"[HealthAgent] Started. Node='{NODE_ID}', Interval={S.HEALTH_INTERVAL}s")
    except Exception as exc:
        logger.error("[HealthAgent] Failed to start: %s, mono_ts=%.6f", exc, time.monotonic(), exc_info=True)
        print(f"[HealthAgent] Failed to start: {exc}", file=sys.stderr)
        health_agent = None

    # --- Zenoh Command & Control (shares PeerOrchestrator's Zenoh session) ---
    zenoh_sub = None
    try:
        from .zenoh_subscriber import ZenohCommandSubscriber
        shared_session = peer_orch._session if peer_orch else None
        # Pass configured add_ack_timeout_s from edge_node.yml (falls back to migration_timeout_s - 3.0s margin, or 12.0s default)
        migration_timeout = float(p2p_cfg.get("migration_timeout_s", 15.0))
        if "add_ack_timeout_s" in p2p_cfg and p2p_cfg["add_ack_timeout_s"] is not None:
            ack_timeout = float(p2p_cfg["add_ack_timeout_s"])
        else:
            ack_timeout = max(1.0, migration_timeout - 3.0)
        if ack_timeout >= migration_timeout:
            raise ValueError(
                f"add_ack_timeout_s ({ack_timeout:.1f}s) must be strictly less than "
                f"migration_timeout_s ({migration_timeout:.1f}s) to ensure sender timeout safety"
            )
        if shared_session is None:
            # P5 fail-closed: the PeerOrchestrator did not join the mesh
            # (e.g. lease persistence failed), so the control plane must not
            # open its own rogue session and accept commands.
            print(
                "[Zenoh C2] Control plane disabled: PeerOrchestrator did not join "
                "(lease persistence failed or session not opened).",
                file=sys.stderr,
            )
        else:
            zenoh_sub = ZenohCommandSubscriber(
                camera_manager=camera_manager,
                node_id=NODE_ID,
                session=shared_session,
                ack_timeout_s=ack_timeout,
                # P5 — receiver fencing: pass the node's current boot_id and the
                # persisted epoch high-water so the subscriber rejects pre-reboot
                # commands after a restart.
                boot_id=peer_orch._boot_id if peer_orch is not None else 0,
                lease=peer_orch._lease if peer_orch is not None else None,
            )
            zenoh_sub.start()
            print(f"[Zenoh C2] Subscriber active. Node='{NODE_ID}', Key=peers/control/{NODE_ID}")
    except ImportError:
        print(
            "[Zenoh C2] zenoh/msgpack not installed — control disabled. "
            "Run: pip install zenoh msgpack",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"[Zenoh C2] Failed to start subscriber: {exc}", file=sys.stderr)

    # --- Zenoh event publisher (overspeed violations → traffic/events/...) ---
    # SpeedProbe enqueues overspeed payloads here; a daemon thread publishes
    # them via Zenoh. health_agent.py subscribes to traffic/events/{NODE_ID}/**
    # and forwards them to the Central Monitor Server. Without this, overspeed
    # events are computed but never leave the pipeline.
    zenoh_pub = None
    try:
        from .zenoh_publisher import ZenohPublisher
        shared_session = peer_orch._session if peer_orch else None
        zenoh_pub = ZenohPublisher(node_id=NODE_ID, session=shared_session)
        zenoh_pub.start()
        print(f"[ZenohPub] Started. Key=traffic/events/{NODE_ID}/<camera_id>")
    except ImportError:
        print(
            "[ZenohPub] zenoh/msgpack not installed — overspeed events disabled. "
            "Run: pip install zenoh msgpack",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"[ZenohPub] Failed to start: {exc}", file=sys.stderr)

    # --- Local LPR worker + L1 plate-crop + L2 full-stream offload (Phase 3) ---
    # LocalLprWorker runs TRT LPR on plate crops off the DeepStream graph
    # (sgie2 was removed in Phase 1).  The OffloadPublisher/OffloadReceiver
    # move plate crops to a peer (L1, source offload_level==1) and return decoded text; the
    # orchestrator escalates a camera to L1(plate-crop, offload_level==1) when this node's LPR queue saturates.
    # The worker runs even without a Zenoh session so local LPR always works.
    lpr_worker = LocalLprWorker(str(LPR_ENGINE), str(LPR_LABELS))
    lpr_worker.start()

    zenoh_session = peer_orch._session if peer_orch is not None else None
    offload_pub = None
    offload_rcv = None
    if zenoh_session is not None:
        try:
            offload_pub = OffloadPublisher(node_id=NODE_ID, session=zenoh_session)
            offload_pub.start()
            offload_rcv = OffloadReceiver(
                node_id=NODE_ID,
                session=zenoh_session,
                lpr_engine_path=str(LPR_ENGINE),
                lpd_engine_path="",
                labels_path=str(LPR_LABELS),
                lpr_worker=lpr_worker,
            )
            offload_rcv.start()
            print(f"[Offload] Started L1 plate-crop + L2 full-stream offload (source offload_level==1). Node='{NODE_ID}'")
        except Exception as exc:
            print(f"[Offload] Failed to start (plate-crop offload disabled): {exc}", file=sys.stderr)
            offload_pub = None
            offload_rcv = None

    # Note: Production health_agent.py is the sole metrics/load-score publisher
    # and publishes self-heartbeats to peers/status/<NODE_ID> over Zenoh.

    # Run pipeline — offload_pub + zenoh_pub references passed so SpeedProbe can use them
    # Supervisor loop around pipeline execution to handle recovery / stream parking
    # and keep Edge process alive without exiting when all streams are removed.
    probe = None
    while True:
        try:
            enabled_cams = camera_manager.get_enabled_configs()
            if not enabled_cams:
                recovery_wait = float(edge_cfg.get("p2p", {}).get("recovery_wait_s", 300.0))
                print(f"[Supervisor] Zero active cameras configured/enabled. Entering recovery state for {recovery_wait:.0f}s...")
                time.sleep(recovery_wait)
                # Re-check cameras config file
                camera_manager.reload()
                enabled_cams = camera_manager.get_enabled_configs()
                if not enabled_cams:
                    print("[Supervisor] Still 0 active cameras after recovery wait. Retrying...")
                    continue
                print(f"[Supervisor] Found {len(enabled_cams)} enabled cameras after recovery. Resuming pipeline.")

            if args.mode == "display":
                probe = run_display_mode(args, camera_manager, peer_orch=peer_orch, offload_pub=offload_pub, offload_rcv=offload_rcv, zenoh_pub=zenoh_pub, lpr_worker=lpr_worker)
            elif args.mode == "file":
                probe = run_file_mode(args, camera_manager, peer_orch=peer_orch, offload_pub=offload_pub, offload_rcv=offload_rcv, zenoh_pub=zenoh_pub, lpr_worker=lpr_worker)
            elif args.mode == "rtsp_push":
                probe = run_rtsp_push_mode(args, camera_manager, peer_orch=peer_orch, offload_pub=offload_pub, offload_rcv=offload_rcv, zenoh_pub=zenoh_pub, lpr_worker=lpr_worker)
            else:
                raise ValueError(f"Unknown mode: '{args.mode}'")
            # A GStreamer EOS/error returns here after the mode runner has
            # stopped the old pipeline.  Keep the Edge supervisor alive and
            # deliberately park before rebuilding; an empty-FPS interval is
            # WAITING/RECOVERY, never a reason to terminate the node.
            recovery_wait = float(edge_cfg.get("p2p", {}).get("recovery_wait_s", 300.0))
            print(f"[Supervisor] Pipeline stopped. Entering recovery state for {recovery_wait:.0f}s...")
            time.sleep(recovery_wait)
            camera_manager.reload()
            print("[Supervisor] Recovery wait complete. Rebuilding pipeline.")
            continue
        except KeyboardInterrupt:
            print("\n[Supervisor] Interrupted by user. Exiting.")
            break
        except Exception as exc:
            print(f"[Supervisor] Pipeline exception: {exc}. Retrying in 5s...", file=sys.stderr)
            time.sleep(5)
            continue
        break

    # Fix #1 / #7: stop the FPS writer thread cleanly now that the pipeline
    # has exited, regardless of which mode was used.
    if probe is not None:
        try:
            probe.stop_fps_writer()
        except Exception:
            pass

    # Clear the active probe reference so health loop doesn't push to stale probe
    ACTIVE_SPEED_PROBE.clear()

    # Stop the overspeed event publisher and flush its queue on exit.
    if zenoh_pub is not None:
        try:
            zenoh_pub.stop()
        except Exception as exc:
            print(f"[ZenohPub] Stop error: {exc}", file=sys.stderr)

    # Phase 3: tear down crop-offload + local LPR worker on exit.
    if offload_rcv is not None:
        try:
            offload_rcv.stop()
        except Exception as exc:
            print(f"[OffloadReceiver] Stop error: {exc}", file=sys.stderr)
    if offload_pub is not None:
        try:
            offload_pub.stop()
        except Exception as exc:
            print(f"[OffloadPublisher] Stop error: {exc}", file=sys.stderr)
    if lpr_worker is not None:
        try:
            lpr_worker.stop()
        except Exception as exc:
            print(f"[LocalLprWorker] Stop error: {exc}", file=sys.stderr)
