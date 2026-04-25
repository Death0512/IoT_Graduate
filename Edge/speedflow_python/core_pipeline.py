# speedflow/core_pipeline.py
import os

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from .common import make_element, gst_link
from .settings import (
    INFER_CONFIG, TRACKER_CFG, ANALYTICS_CFG,
    SGIE_CONFIG, TRACKER_LIB, LPR_CONFIG, TRACKER_LPD_CFG,
)

Gst.init(None)


def is_file_uri(uri: str) -> bool:
    """Check if URI points to a local file."""
    return uri.startswith("file://") or (os.path.isabs(uri) and os.path.isfile(uri))


def normalize_uri(uri: str) -> str:
    """Ensure URI has a proper scheme prefix."""
    if uri.startswith("file://") or uri.startswith("rtsp://"):
        return uri
    if os.path.exists(uri):
        return "file://" + os.path.abspath(uri)
    return uri


def build_pipeline(
    source_uri: str,
    sink_type: str = "display",
    output_path: str = None,
    mux_width: int = 1920,
    mux_height: int = 1080,
    is_live: bool = None,
    analytics_config: str = None,
    **kwargs,
):
    """
    Build a DeepStream GStreamer pipeline.

    Args:
        source_uri:       Input source (RTSP URL or file path).
        sink_type:        Output type — "display", "file", or "webrtc".
        output_path:      Output file path (required when sink_type="file").
        mux_width:        nvstreammux width.
        mux_height:       nvstreammux height.
        is_live:          Whether source is live (auto-detected when None).
        analytics_config: Path to nvdsanalytics config (defaults to settings).
        **kwargs:         Sink-specific parameters (ignored).

    Returns:
        (pipeline, nvdsosd)         for sink_type in {"display", "file"}
        (pipeline, nvdsosd, webrtc) for sink_type == "webrtc"
    """
    # ── Inputs ──────────────────────────────────────────────────────────────
    uri = normalize_uri(source_uri)
    is_file = is_file_uri(uri)

    if is_live is None:
        is_live = 0 if is_file else 1

    if sink_type == "file" and not output_path:
        raise ValueError("output_path is required when sink_type='file'")

    if analytics_config is None:
        analytics_config = str(ANALYTICS_CFG)

    # ── Pipeline ─────────────────────────────────────────────────────────────
    pipeline = Gst.Pipeline.new(f"ds-pipeline-{sink_type}")

    # ── Source ───────────────────────────────────────────────────────────────
    source = make_element("source-bin", "uridecodebin")
    source.set_property("uri", uri)

    def on_source_setup(decodebin, src):
        """Apply RTSP-specific latency settings when the source pad is set up."""
        if not is_file:
            for prop, val in [("latency", 100), ("drop-on-latency", True)]:
                try:
                    src.set_property(prop, val)
                except (TypeError, Exception):
                    pass

    source.connect("source-setup", on_source_setup)

    # ── Core processing ───────────────────────────────────────────────────────
    streammux = make_element("stream-muxer", "nvstreammux")
    streammux.set_property("batch-size", 12)
    streammux.set_property("width", mux_width)
    streammux.set_property("height", mux_height)
    streammux.set_property("batched-push-timeout", 33000)
    streammux.set_property("live-source", is_live)

    pgie = make_element("primary-infer", "nvinfer")
    pgie.set_property("config-file-path", str(INFER_CONFIG))


    # Tracker placed before SGIE to stabilise vehicle bboxes before plate detection
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

    # Convert to RGBA before OSD to prevent ghosting artefacts
    preosd_convert = make_element("preosd_convert", "nvvideoconvert")
    preosd_caps = make_element("preosd_caps", "capsfilter")
    preosd_caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )

    nvdsosd = make_element("onscreendisplay", "nvdsosd")
    nvdsosd.set_property("display-text", 1)
    nvdsosd.set_property("display-bbox", 1)
    nvdsosd.set_property("process-mode", 2)   # 2 = GPU (new API)
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
        muxer = make_element("muxer", "qtmux")
        sink = make_element("filesink", "filesink")
        sink.set_property("location", os.path.abspath(output_path))
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sink_elements = [postosd_convert, encoder, parser, muxer, sink]

    elif sink_type == "webrtc":
        conv = make_element("conv", "nvvideoconvert")
        enc = make_element("enc", "nvv4l2h264enc")
        enc.set_property("insert-sps-pps", True)
        enc.set_property("iframeinterval", 30)
        enc.set_property("bitrate", 4_000_000)
        try:
            enc.set_property("maxperf-enable", True)
        except (TypeError, Exception):
            pass
        parse = make_element("parse", "h264parse")
        pay = make_element("pay", "rtph264pay")
        pay.set_property("pt", 96)
        pay.set_property("config-interval", 1)
        rtp_caps = make_element("rtp_caps", "capsfilter")
        rtp_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                "application/x-rtp,media=video,encoding-name=H264,"
                "payload=96,clock-rate=90000"
            ),
        )
        webrtc = make_element("webrtc", "webrtcbin")
        sink_elements = [conv, enc, parse, pay, rtp_caps, webrtc]

    else:
        raise ValueError(
            f"Unknown sink_type: '{sink_type}'. Must be 'display', 'file', or 'webrtc'."
        )

    # ── Add elements to pipeline ──────────────────────────────────────────────
    core_elements = [
        source, streammux, pgie, tracker, sgie, sgie2,
        analytics, preosd_convert, preosd_caps, nvdsosd,
    ]
    for element in core_elements + sink_elements:
        pipeline.add(element)

    # ── Dynamic pad linking (source → streammux) ──────────────────────────────
    def on_pad_added(decodebin, pad):
        caps = pad.get_current_caps()
        if not caps:
            return
        if caps.to_string().startswith("video/"):
            sinkpad = streammux.get_request_pad("sink_0")
            if sinkpad and not sinkpad.is_linked():
                queue = make_element("source_queue", "queue")
                convert = make_element("source_convert", "nvvideoconvert")
                pipeline.add(queue)
                pipeline.add(convert)
                queue.sync_state_with_parent()
                convert.sync_state_with_parent()
                pad.link(queue.get_static_pad("sink"))
                gst_link(queue, convert)
                convert.get_static_pad("src").link(sinkpad)

    source.connect("pad-added", on_pad_added)

    # ── Link core processing chain ────────────────────────────────────────────
    # Streammux → PGIE → Tracker → SGIE (LPD) → SGIE2 (LPR) → Analytics → OSD
    gst_link(streammux, pgie, tracker, sgie, sgie2, analytics,
             preosd_convert, preosd_caps, nvdsosd)

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
        srcpad = rtp_caps.get_static_pad("src")
        sinkpad = webrtc.get_request_pad("sink_%u")
        if not sinkpad:
            raise RuntimeError("Failed to get request pad from webrtcbin")
        link_res = srcpad.link(sinkpad)
        if link_res != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link RTP → webrtcbin: {link_res}")

    if sink_type == "webrtc":
        return pipeline, nvdsosd, webrtc
    return pipeline, nvdsosd