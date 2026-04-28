#!/usr/bin/env python3
"""
Python backend runner.
Uses speedflow_python pipeline + GStreamer pad probes for processing.
Hỗ trợ Multi-Stream Dynamic.
"""
import sys
import os
import asyncio
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from .core_pipeline import build_pipeline, dynamic_add_stream, dynamic_remove_stream
from .camera_config import CameraManager
from .settings import CAMERAS_YML
from .probes import SpeedProbe, ROIFilterProbe
from .plate_preprocessor import PlatePreprocessorProbe
from .config_txt import load_kv_txt
from .common import WebRTCSession
from . import settings as S


# ---------------------------------------------------------------------------
# Shared probe setup
# ---------------------------------------------------------------------------

def _setup_probes(pipeline: Gst.Pipeline, nvdsosd: Gst.Element, camera_manager: CameraManager) -> SpeedProbe:
    """
    Attach ROI filter, plate preprocessor, and speed probe to *pipeline*.
    Returns the SpeedProbe instance.
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

    # 2. Plate preprocessor
    tracker = pipeline.get_by_name("tracker")
    if tracker:
        plate_preprocessor = PlatePreprocessorProbe(
            enable_sharpening=True,
            enable_contrast=True,
            enable_denoise=True,
            adaptive_mode=True,
        )
        tracker_srcpad = tracker.get_static_pad("src")
        if tracker_srcpad:
            tracker_srcpad.add_probe(
                Gst.PadProbeType.BUFFER,
                plate_preprocessor.buffer_probe,
                None,
            )
            print("[Plate Preprocessor] Enabled")

    # 3. Speed + LPR probe
    probe = SpeedProbe(camera_manager)

    tiler = pipeline.get_by_name("tiler")
    if tiler:
        pad = tiler.get_static_pad("sink")
        pad_name = "tiler sink"
    else:
        pad = nvdsosd.get_static_pad("sink")
        pad_name = "nvdsosd sink"

    if not pad:
        print(f"ERROR: Unable to get {pad_name} pad", file=sys.stderr)
        sys.exit(1)
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
    tiler: Gst.Element = None
):
    """
    Hooks up the CameraManager to safely add/remove streams dynamically.
    """
    def on_add(cam_cfg):
        current_n = streammux.get_property("batch-size")
        print(f"[Dynamic] Adding camera '{cam_cfg.camera_id}' (source_id={cam_cfg.source_id})")
        dynamic_add_stream(pipeline, streammux, cam_cfg, tiler, source_bins, current_n)

    def on_remove(source_id):
        # Find camera_id by source_id from source_bins dict
        cam_id = None
        for cid, src_bin in source_bins.items():
            # Since source_id was mapped, we can rely on camera_manager
            # But the source_id might already be gone from camera_manager
            pass 
        
        # We need camera_id. Let's find it.
        for cid in list(source_bins.keys()):
            if f"src-{cid}" == source_bins[cid].get_name():
                # Since we don't store source_id directly in dict, we can query it
                # Actually dynamic_remove_stream requires camera_id.
                # In CameraManager, we only get source_id to remove.
                pass
        
        # Better: camera_manager delta to_remove only gives source_id. We need to match.
        cam_id_to_remove = None
        for cid, src in source_bins.items():
            if f"sink_{source_id}" in [pad.get_name() for pad in streammux.sinkpads if pad.get_peer() and pad.get_peer().get_parent() == src.get_by_name(f"conv_{cid}")]:
                pass # Too complex.

        # Let's simplify: In our setup, source bin name is f"src-{camera_id}".
        # Let's search all source_bins for the one linked to sink_{source_id}.
        for cid, src in source_bins.items():
            pad = streammux.get_static_pad(f"sink_{source_id}")
            if pad and pad.is_linked():
                peer = pad.get_peer()
                if peer and peer.get_parent().get_name() == f"conv_{cid}":
                    cam_id_to_remove = cid
                    break
            else:
                # If pad is unlinked, just check if we have a match
                # Wait, dynamic_remove_stream requires camera_id. Let's just pass camera_id to on_remove.
                pass
        
        # Hack to find camera_id from source_id
        for cid, src in source_bins.items():
            # In dynamic_add_stream, we used f"conv_{cid}"
            if pipeline.get_by_name(f"conv_{cid}") and pipeline.get_by_name(f"conv_{cid}").get_static_pad("src").is_linked():
                peer = pipeline.get_by_name(f"conv_{cid}").get_static_pad("src").get_peer()
                if peer and peer.get_name() == f"sink_{source_id}":
                    cam_id_to_remove = cid
                    break
        
        if cam_id_to_remove:
            current_n = streammux.get_property("batch-size")
            print(f"[Dynamic] Removing camera '{cam_id_to_remove}' (source_id={source_id})")
            dynamic_remove_stream(pipeline, streammux, cam_id_to_remove, source_id, tiler, source_bins, current_n)
        else:
            print(f"[Dynamic] Could not find camera for source_id={source_id} to remove.")

    camera_manager.start(on_add, on_remove, GLib.idle_add)


# ---------------------------------------------------------------------------
# GLib bus helpers
# ---------------------------------------------------------------------------

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
            print(f"ERROR from {message.src.get_name()}: {err}", file=sys.stderr)
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
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_display_mode(args, camera_manager: CameraManager) -> None:
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

    _setup_probes(pipeline, nvdsosd, camera_manager)
    _attach_camera_manager(camera_manager, pipeline, streammux, source_bins, tiler)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
        sys.exit(1)
        
    _run_loop_until_eos_or_error(pipeline, camera_manager)


def run_file_mode(args, camera_manager: CameraManager) -> None:
    Gst.init(None)
    configs = camera_manager.get_enabled_configs()

    ret_build = build_pipeline(
        camera_configs=configs,
        sink_type="file",
        mux_width=args.width,
        mux_height=args.height,
    )
    pipeline, nvdsosd, streammux, source_bins = ret_build

    _setup_probes(pipeline, nvdsosd, camera_manager)
    _attach_camera_manager(camera_manager, pipeline, streammux, source_bins, None)

    print(f"[Python File Mode] Processing multi-streams to output files...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
        sys.exit(1)

    _run_loop_until_eos_or_error(pipeline, camera_manager)


async def run_webrtc_mode_async(args, camera_manager: CameraManager) -> None:
    configs = list(camera_manager.configs.values())
    
    ret_build = build_pipeline(
        camera_configs=configs,
        sink_type="webrtc",
        mux_width=args.width,
        mux_height=args.height,
    )
    pipeline, nvdsosd, streammux, source_bins, webrtc_elem = ret_build
    tiler = pipeline.get_by_name("tiler")

    probe = _setup_probes(pipeline, nvdsosd, camera_manager)
    _attach_camera_manager(camera_manager, pipeline, streammux, source_bins, tiler)

    ws_uri  = f"ws://{args.server}:{args.port}/ws?room={args.room}&role=pub"
    session = WebRTCSession(webrtc_elem, ws_uri)
    probe.set_publisher(session.send_json_threadsafe)

    pipeline.set_state(Gst.State.PLAYING)
    print(f"[Python WebRTC Mode] Pipeline running")
    print(f"[Python WebRTC Mode] Room: {args.room}")
    print(f"[Python WebRTC Mode] View stream at: http://{args.server}:{args.port}/")

    await asyncio.sleep(1.5)
    await session.connect()

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            loop.quit()
        elif t == Gst.MessageType.EOS:
            loop.quit()

    bus.connect("message", on_message)

    try:
        running_loop = asyncio.get_running_loop()
        await running_loop.run_in_executor(None, loop.run)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        session.close()
        camera_manager.stop()
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


def run_webrtc_mode(args, camera_manager: CameraManager) -> None:
    Gst.init(None)
    asyncio.run(run_webrtc_mode_async(args, camera_manager))


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def run_python_mode(args) -> None:
    """Entry point called by main.py for the Python backend."""
    # Initialize CameraManager globally for python mode
    camera_manager = CameraManager(CAMERAS_YML)
    
    # Optional: Start REST API on port 8000
    camera_manager.start_rest_api(port=8000)

    if args.mode == "display":
        run_display_mode(args, camera_manager)
    elif args.mode == "file":
        run_file_mode(args, camera_manager)
    elif args.mode == "webrtc":
        run_webrtc_mode(args, camera_manager)
    else:
        raise ValueError(f"Unknown mode: '{args.mode}'")
