#!/usr/bin/env python3
"""
Python backend runner.
Uses speedflow_python pipeline + GStreamer pad probes for processing.
"""
import sys
import os
import asyncio

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from .core_pipeline import build_pipeline
from .homography import load_points, ViewTransformer
from .settings import HOMO_YML
from .probes import SpeedProbe, ROIFilterProbe
from .plate_preprocessor import PlatePreprocessorProbe
from .config_txt import load_kv_txt
from .common import WebRTCSession
from . import settings as S


# ---------------------------------------------------------------------------
# Shared probe setup  (eliminates copy-paste across all three modes)
# ---------------------------------------------------------------------------

def _setup_probes(pipeline: Gst.Pipeline, nvdsosd: Gst.Element, homo_path: str) -> SpeedProbe:
    """
    Attach ROI filter, plate preprocessor, and speed probe to *pipeline*.

    Returns the SpeedProbe instance so callers can set a publisher if needed.
    """
    # 1. ROI filter — remove objects outside the analytics ROI
    analytics = pipeline.get_by_name("analytics")
    if analytics:
        roi_filter = ROIFilterProbe()
        analytics_srcpad = analytics.get_static_pad("src")
        if analytics_srcpad:
            analytics_srcpad.add_probe(
                Gst.PadProbeType.BUFFER,
                roi_filter.analytics_src_pad_buffer_probe,
                None,
            )
            print("[ROI Filter] Enabled")

    # 2. Plate preprocessor — enhance vehicle crops before SGIE1
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
            print("[Plate Preprocessor] Enabled (vehicle-crop only)")

    # 3. Speed + LPR probe — attached to nvdsosd sink pad
    source_pts, target_pts = load_points(homo_path)
    vt = ViewTransformer(source_pts, target_pts)
    probe = SpeedProbe(vt, roi_source_points=source_pts)

    pad = nvdsosd.get_static_pad("sink")
    if not pad:
        print("ERROR: Unable to get sink pad of nvdsosd", file=sys.stderr)
        sys.exit(1)
    pad.add_probe(Gst.PadProbeType.BUFFER, probe.osd_sink_pad_buffer_probe, None)

    return probe


# ---------------------------------------------------------------------------
# GLib bus helpers
# ---------------------------------------------------------------------------

def _run_loop_until_eos_or_error(pipeline: Gst.Pipeline) -> None:
    """Block in a GLib.MainLoop until EOS or error, then stop the pipeline."""
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
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


# ---------------------------------------------------------------------------
# Display mode
# ---------------------------------------------------------------------------

def run_display_mode(args) -> None:
    Gst.init(None)
    pipeline, nvdsosd = build_pipeline(...)
    _setup_probes(pipeline, nvdsosd, args.homo)
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
        sys.exit(1)
    _run_loop_until_eos_or_error(pipeline)  # ← dùng hàm chung


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------

def run_file_mode(args) -> None:
    """Run pipeline in file (MP4 output) mode."""
    Gst.init(None)

    # Chỉ kiểm tra nếu không phải là RTSP/HTTP/etc.
    if not args.source.startswith(("rtsp://", "rtmp://", "http://", "file://")):
        if not os.path.exists(args.source):
            print(f"ERROR: Input file not found: {args.source}", file=sys.stderr)
            sys.exit(1)

    pipeline, nvdsosd = build_pipeline(
        source_uri=args.source,
        sink_type="file",
        output_path=args.output,
        mux_width=args.width,
        mux_height=args.height,
    )

    _setup_probes(pipeline, nvdsosd, args.homo)

    print(f"[Python File Mode] Processing: {args.source} → {args.output}")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR)
        if msg:
            err, debug = msg.parse_error()
            print(f"ERROR: Pipeline failed to start: {err}", file=sys.stderr)
            if debug:
                print(f"DEBUG INFO: {debug}", file=sys.stderr)
        else:
            print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
        sys.exit(1)

    _run_loop_until_eos_or_error(pipeline)


# ---------------------------------------------------------------------------
# WebRTC mode
# ---------------------------------------------------------------------------

async def run_webrtc_mode_async(args) -> None:
    """Run pipeline in WebRTC (browser stream) mode."""
    # Load per-camera config overrides
    kv = load_kv_txt(args.cfg)
    S.ANALYTICS_CFG = kv["ANALYTICS_CFG"]
    S.HOMO_YML      = kv["HOMO_YML"]
    S.VIDEO_FPS     = kv["VIDEO_FPS"]
    S.MUX_WIDTH     = int(kv.get("MUX_WIDTH",  args.width))
    S.MUX_HEIGHT    = int(kv.get("MUX_HEIGHT", args.height))

    print(f"[Config] ANALYTICS_CFG = {S.ANALYTICS_CFG}")
    print(f"[Config] HOMO_YML      = {S.HOMO_YML}")
    print(f"[Config] VIDEO_FPS     = {S.VIDEO_FPS}")
    print(f"[Config] Resolution    = {S.MUX_WIDTH}×{S.MUX_HEIGHT}")

    pipeline, nvdsosd, webrtc = build_pipeline(
        source_uri=args.source,
        sink_type="webrtc",
        mux_width=S.MUX_WIDTH,
        mux_height=S.MUX_HEIGHT,
        analytics_config=S.ANALYTICS_CFG,
    )

    probe = _setup_probes(pipeline, nvdsosd, str(S.HOMO_YML))

    # WebRTC signaling — uses shared WebRTCSession from common.py
    ws_uri  = f"ws://{args.server}:{args.port}/ws?room={args.room}&role=pub"
    session = WebRTCSession(webrtc, ws_uri)
    probe.set_publisher(session.send_json_threadsafe)

    pipeline.set_state(Gst.State.PLAYING)
    print(f"[Python WebRTC Mode] Pipeline running")
    print(f"[Python WebRTC Mode] Room: {args.room}")
    print(f"[Python WebRTC Mode] View stream at: http://{args.server}:{args.port}/")

    await asyncio.sleep(1.5)
    await session.connect()

    loop = GLib.MainLoop()
    try:
        # Use asyncio.get_running_loop() (replaces deprecated get_event_loop)
        running_loop = asyncio.get_running_loop()
        await running_loop.run_in_executor(None, loop.run)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        session.close()
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


def run_webrtc_mode(args) -> None:
    """Synchronous wrapper to run WebRTC mode with asyncio."""
    Gst.init(None)
    asyncio.run(run_webrtc_mode_async(args))


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def run_python_mode(args) -> None:
    """Entry point called by main.py for the Python backend."""
    if args.mode == "display":
        run_display_mode(args)
    elif args.mode == "file":
        run_file_mode(args)
    elif args.mode == "webrtc":
        run_webrtc_mode(args)
    else:
        raise ValueError(f"Unknown mode: '{args.mode}'")
