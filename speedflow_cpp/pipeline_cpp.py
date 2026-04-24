#!/usr/bin/env python3
"""
C++ Backend Pipeline Runner.
Uses the custom GStreamer C++ plugin (libgstspeedflow.so) for speed measurement.
All probe logic is handled inside the plugin, so no Python probes are needed.
"""
import sys
import os
import asyncio

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Import shared utilities from the Python module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speedflow_python.settings import (
    INFER_CONFIG, TRACKER_CFG, ANALYTICS_CFG,
    SGIE_CONFIG, TRACKER_LIB, LPR_CONFIG, HOMO_YML,
)
from speedflow_python.config_txt import load_kv_txt
from speedflow_python.common import make_element, gst_link, WebRTCSession

# Path to the compiled C++ plugin
PLUGIN_PATH = os.path.join(os.path.dirname(__file__), "build", "libgstspeedflow.so")


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline_cpp(
    source_uri: str,
    sink_type: str = "display",
    output_path: str = None,
    mux_width: int = 1920,
    mux_height: int = 1080,
    is_live: bool = None,
    analytics_config: str = None,
    homo_config: str = None,
    video_fps: int = 60,
    **kwargs,
):
    """
    Build a DeepStream pipeline using the C++ speedflow GStreamer plugin.

    The C++ plugin replaces all Python probes and additionally supports
    NVIDIA Optical Flow (NVOF) for improved speed estimation accuracy.

    Returns:
        (pipeline, nvdsosd)         for sink_type in {"display", "file"}
        (pipeline, nvdsosd, webrtc) for sink_type == "webrtc"
    """
    # ── Load C++ plugin ───────────────────────────────────────────────────────
    if not Gst.Registry.get().find_plugin("speedflow"):
        if os.path.exists(PLUGIN_PATH):
            Gst.Registry.get().scan_path(os.path.dirname(PLUGIN_PATH))
            print(f"[C++ Plugin] Loaded from {PLUGIN_PATH}")
        else:
            raise RuntimeError(
                f"C++ plugin not found at {PLUGIN_PATH}. "
                f"Run speedflow_cpp/build.sh first."
            )

    # ── Inputs ────────────────────────────────────────────────────────────────
    if uri := source_uri:
        if uri.startswith("file://") or uri.startswith("rtsp://"):
            pass
        elif os.path.exists(uri):
            uri = "file://" + os.path.abspath(uri)

    is_file = uri.startswith("file://") or (
        os.path.isabs(source_uri) and os.path.isfile(source_uri)
    )
    if is_live is None:
        is_live = 0 if is_file else 1

    if sink_type == "file" and not output_path:
        raise ValueError("output_path is required when sink_type='file'")

    if analytics_config is None:
        analytics_config = str(ANALYTICS_CFG)
    if homo_config is None:
        homo_config = str(HOMO_YML)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Gst.Pipeline.new(f"ds-cpp-pipeline-{sink_type}")

    # ── Source ────────────────────────────────────────────────────────────────
    source = make_element("source-bin", "uridecodebin")
    source.set_property("uri", uri)

    def on_source_setup(decodebin, src):
        if not is_file:
            for prop, val in [("latency", 100), ("drop-on-latency", True)]:
                try:
                    src.set_property(prop, val)
                except (TypeError, Exception):
                    pass

    source.connect("source-setup", on_source_setup)

    # ── Core processing ───────────────────────────────────────────────────────
    streammux = make_element("stream-muxer", "nvstreammux")
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", mux_width)
    streammux.set_property("height", mux_height)
    streammux.set_property("batched-push-timeout", 33000)
    streammux.set_property("live-source", is_live)

    # NVOF: GPU optical flow — C++ backend exclusive
    nvof = make_element("nvof", "nvof")
    nvof.set_property("gpu-id", 0)
    nvof.set_property("preset-level", 2)   # 0=slow 1=medium 2=fast
    nvof.set_property("grid-size", 0)      # 0 = 4×4 grid
    print("[C++ NVOF] Optical Flow enabled")

    pgie = make_element("primary-infer", "nvinfer")
    pgie.set_property("config-file-path", str(INFER_CONFIG))

    tracker = make_element("tracker", "nvtracker")
    tracker.set_property("ll-lib-file", str(TRACKER_LIB))
    tracker.set_property("ll-config-file", str(TRACKER_CFG))
    tracker.set_property("tracker-width", 224)
    tracker.set_property("tracker-height", 224)
    tracker.set_property("gpu_id", 0)

    sgie = make_element("secondary-infer", "nvinfer")
    sgie.set_property("config-file-path", str(SGIE_CONFIG))

    sgie2 = make_element("lpr-classifier", "nvinfer")
    sgie2.set_property("config-file-path", str(LPR_CONFIG))

    analytics = make_element("analytics", "nvdsanalytics")
    analytics.set_property("config-file", analytics_config)

    # ── C++ SpeedFlow plugin (replaces all Python probes) ─────────────────────
    speedflow = make_element("speedflow-plugin", "speedflow")
    speedflow.set_property("config-file", homo_config)
    speedflow.set_property("speed-limit", 80.0)
    speedflow.set_property("video-fps", video_fps)
    speedflow.set_property("enable-nvof", True)

    print(f"[C++ SpeedFlow] Homography config : {homo_config}")
    print(f"[C++ SpeedFlow] Speed limit       : 80.0 km/h")
    print(f"[C++ SpeedFlow] NVOF              : ENABLED")
    print(f"[C++ SpeedFlow] Video FPS         : {video_fps}")

    # ── OSD ───────────────────────────────────────────────────────────────────
    preosd_convert = make_element("preosd_convert", "nvvideoconvert")
    preosd_caps = make_element("preosd_caps", "capsfilter")
    preosd_caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )

    nvdsosd = make_element("onscreendisplay", "nvdsosd")
    nvdsosd.set_property("display-text", 1)
    nvdsosd.set_property("display-bbox", 1)
    nvdsosd.set_property("process-mode", 2)
    nvdsosd.set_property("gpu-id", 0)

    # ── Sink-specific elements ────────────────────────────────────────────────
    sink_elements: list = []

    if sink_type == "display":
        conv = make_element("conv", "nvvideoconvert")
        conv_caps = make_element("conv_caps", "capsfilter")
        conv_caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
        )
        eglT = make_element("eglT", "nvegltransform")
        sink = make_element("display", "nveglglessink")
        sink.set_property("sync", False)
        sink.set_property("qos", False)
        sink.set_property("async", False)
        sink.set_property("max-lateness", -1)
        sink_elements = [conv, conv_caps, eglT, sink]

    elif sink_type == "file":
        postosd_convert = make_element("postosd_convert", "nvvideoconvert")
        encoder = make_element("encoder", "nvv4l2h264enc")
        encoder.set_property("bitrate", 20_000_000)
        encoder.set_property("preset-level", 1)
        parser = make_element("parser", "h264parse")
        muxer  = make_element("muxer", "qtmux")
        sink   = make_element("filesink", "filesink")
        sink.set_property("location", os.path.abspath(output_path))
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sink_elements = [postosd_convert, encoder, parser, muxer, sink]

    elif sink_type == "webrtc":
        conv = make_element("conv", "nvvideoconvert")
        enc  = make_element("enc", "nvv4l2h264enc")
        enc.set_property("bitrate", 2_000_000)       # 2 Mbps — predictable bandwidth
        enc.set_property("iframeinterval", 15)        # fast loss recovery
        enc.set_property("insert-sps-pps", True)
        enc.set_property("control-rate", 1)           # CBR
        enc.set_property("preset-level", 1)
        enc.set_property("profile", 0)                # Baseline — best browser compat
        try:
            enc.set_property("maxperf-enable", True)
        except (TypeError, Exception):
            pass

        parse = make_element("parse", "h264parse")
        parse.set_property("config-interval", -1)     # SPS/PPS at every IDR

        pay = make_element("pay", "rtph264pay")
        pay.set_property("pt", 96)
        pay.set_property("config-interval", 1)
        pay.set_property("mtu", 1200)                 # less fragmentation

        rtp_caps = make_element("rtp_caps", "capsfilter")
        rtp_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                "application/x-rtp,media=video,encoding-name=H264,"
                "payload=96,clock-rate=90000"
            ),
        )
        webrtc = make_element("webrtc", "webrtcbin")
        webrtc.set_property("bundle-policy", 3)   # max-bundle
        webrtc.set_property("latency", 100)

        sink_elements = [conv, enc, parse, pay, rtp_caps, webrtc]

    else:
        raise ValueError(f"Unknown sink_type: '{sink_type}'")

    # ── Add elements ──────────────────────────────────────────────────────────
    core_elements = [
        source, streammux, nvof, pgie, tracker, sgie, sgie2,
        analytics, speedflow, preosd_convert, preosd_caps, nvdsosd,
    ]
    for el in core_elements + sink_elements:
        pipeline.add(el)

    # ── Dynamic pad linking (source → streammux) ──────────────────────────────
    def on_pad_added(decodebin, pad):
        caps = pad.get_current_caps()
        if not caps:
            return
        if caps.to_string().startswith("video/"):
            sinkpad = streammux.get_request_pad("sink_0")
            if sinkpad and not sinkpad.is_linked():
                queue   = make_element("source_queue", "queue")
                convert = make_element("source_convert", "nvvideoconvert")
                pipeline.add(queue)
                pipeline.add(convert)
                queue.sync_state_with_parent()
                convert.sync_state_with_parent()
                pad.link(queue.get_static_pad("sink"))
                gst_link(queue, convert)
                convert.get_static_pad("src").link(sinkpad)

    source.connect("pad-added", on_pad_added)

    # ── Link core chain ───────────────────────────────────────────────────────
    # Streammux → NVOF → PGIE → Tracker → SGIE → SGIE2 → Analytics → SpeedFlow → OSD
    gst_link(
        streammux, nvof, pgie, tracker, sgie, sgie2,
        analytics, speedflow, preosd_convert, preosd_caps, nvdsosd,
    )

    # ── Link sink chain ───────────────────────────────────────────────────────
    if sink_type == "display":
        conv, conv_caps, eglT, sink = sink_elements
        gst_link(nvdsosd, conv, conv_caps, eglT, sink)

    elif sink_type == "file":
        postosd_convert, encoder, parser, muxer, sink = sink_elements
        gst_link(nvdsosd, postosd_convert, encoder, parser, muxer, sink)

    elif sink_type == "webrtc":
        conv, enc, parse, pay, rtp_caps, webrtc = sink_elements
        gst_link(nvdsosd, conv, enc, parse, pay, rtp_caps)
        srcpad  = rtp_caps.get_static_pad("src")
        sinkpad = webrtc.get_request_pad("sink_%u")
        if not sinkpad:
            raise RuntimeError("Failed to get request pad from webrtcbin")
        link_res = srcpad.link(sinkpad)
        if link_res != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link RTP → webrtcbin: {link_res}")

    if sink_type == "webrtc":
        return pipeline, nvdsosd, sink_elements[-1]  # webrtcbin
    return pipeline, nvdsosd


# ---------------------------------------------------------------------------
# GLib bus helpers (shared logic)
# ---------------------------------------------------------------------------

def _run_loop_until_eos_or_error(pipeline: Gst.Pipeline) -> None:
    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"ERROR from {message.src.get_name()}: {err}", file=sys.stderr)
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

def run_display_mode_cpp(args) -> None:
    Gst.init(None)
    pipeline, nvdsosd = build_pipeline_cpp(
        source_uri=args.source,
        sink_type="display",
        mux_width=args.width,
        mux_height=args.height,
        homo_config=args.homo,
    )
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
        sys.exit(1)
    print(f"[C++ Display Mode] Running with source: {args.source}")
    print("Press Ctrl+C to stop…")
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------

def run_file_mode_cpp(args) -> None:
    Gst.init(None)
    if not os.path.exists(args.source):
        print(f"ERROR: Input file not found: {args.source}", file=sys.stderr)
        sys.exit(1)
    pipeline, nvdsosd = build_pipeline_cpp(
        source_uri=args.source,
        sink_type="file",
        output_path=args.output,
        mux_width=args.width,
        mux_height=args.height,
        homo_config=args.homo,
    )
    print(f"[C++ File Mode] Processing: {args.source} → {args.output}")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
        sys.exit(1)
    _run_loop_until_eos_or_error(pipeline)


# ---------------------------------------------------------------------------
# WebRTC mode
# ---------------------------------------------------------------------------

async def run_webrtc_mode_cpp_async(args) -> None:
    kv           = load_kv_txt(args.cfg)
    analytics_cfg = kv.get("ANALYTICS_CFG", str(ANALYTICS_CFG))
    homo_cfg     = kv.get("HOMO_YML",       str(HOMO_YML))
    video_fps    = int(kv.get("VIDEO_FPS",  30))
    mux_width    = int(kv.get("MUX_WIDTH",  args.width))
    mux_height   = int(kv.get("MUX_HEIGHT", args.height))

    print(f"[Config] ANALYTICS_CFG = {analytics_cfg}")
    print(f"[Config] HOMO_YML      = {homo_cfg}")
    print(f"[Config] VIDEO_FPS     = {video_fps}")
    print(f"[Config] Resolution    = {mux_width}×{mux_height}")

    pipeline, nvdsosd, webrtc = build_pipeline_cpp(
        source_uri=args.source,
        sink_type="webrtc",
        mux_width=mux_width,
        mux_height=mux_height,
        analytics_config=analytics_cfg,
        homo_config=homo_cfg,
        video_fps=video_fps,
    )

    ws_uri  = f"ws://{args.server}:{args.port}/ws?room={args.room}&role=pub"
    session = WebRTCSession(webrtc, ws_uri)   # shared from common.py

    pipeline.set_state(Gst.State.PLAYING)
    print(f"[C++ WebRTC Mode] Pipeline running")
    print(f"[C++ WebRTC Mode] Room: {args.room}")
    print(f"[C++ WebRTC Mode] View stream at: http://{args.server}:{args.port}/")

    await asyncio.sleep(1.5)
    await session.connect()

    loop = GLib.MainLoop()
    try:
        running_loop = asyncio.get_running_loop()   # replaces deprecated get_event_loop()
        await running_loop.run_in_executor(None, loop.run)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        session.close()
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


def run_webrtc_mode_cpp(args) -> None:
    Gst.init(None)
    asyncio.run(run_webrtc_mode_cpp_async(args))


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def run_cpp_mode(args) -> None:
    """Entry point called by main.py for the C++ backend."""
    if args.mode == "display":
        run_display_mode_cpp(args)
    elif args.mode == "file":
        run_file_mode_cpp(args)
    elif args.mode == "webrtc":
        run_webrtc_mode_cpp(args)
    else:
        raise ValueError(f"Unknown mode: '{args.mode}'")
