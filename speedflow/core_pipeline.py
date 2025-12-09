# speedflow/core_pipeline.py
"""
Unified DeepStream pipeline builder supporting multiple sink types.
Consolidates pipeline.py, pipeline_file.py, and pipeline_webrtc.py into a single flexible implementation.
"""
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from .settings import INFER_CONFIG, TRACKER_CFG, ANALYTICS_CFG, SGIE_CONFIG, TRACKER_LIB

Gst.init(None)

def make_element(name: str, factory: str):
    """Create a GStreamer element with error handling."""
    element = Gst.ElementFactory.make(factory, name)
    if not element:
        raise RuntimeError(f"Failed to create element: {factory} (name: {name})")
    return element

def is_file_uri(uri: str) -> bool:
    """Check if URI points to a file."""
    return uri.startswith("file://") or (os.path.isabs(uri) and os.path.isfile(uri))

def normalize_uri(uri: str) -> str:
    """Normalize URI to proper format."""
    if uri.startswith("file://") or uri.startswith("rtsp://"):
        return uri
    if os.path.exists(uri):
        # Convert to absolute path, then to file:// URI
        abs_path = os.path.abspath(uri)
        return "file://" + abs_path
    return uri

def build_pipeline(source_uri: str, sink_type: str = "display", output_path: str = None,
                mux_width: int = 1280, mux_height: int = 720, is_live: bool = None, analytics_config: str = None, **kwargs):
    """   
    Args:
        source_uri: Input source (RTSP URL or file path)
        sink_type: Output type - "display", "file", or "webrtc"
        output_path: Output file path (required for sink_type="file")
        mux_width: Streammux width (default: 1280)
        mux_height: Streammux height (default: 720)
        is_live: Whether source is live (auto-detected if None)
        analytics_config: Path to analytics config file (default: from settings)
        **kwargs: Additional sink-specific parameters
    Returns:
        For display/file: (pipeline, nvdsosd)
        For webrtc: (pipeline, nvdsosd, webrtc)
    """
    # Normalize and validate inputs
    uri = normalize_uri(source_uri)
    is_file = is_file_uri(uri)
    
    if is_live is None:
        is_live = 0 if is_file else 1
    
    if sink_type == "file" and not output_path:
        raise ValueError("output_path is required when sink_type='file'")
    
    if analytics_config is None:
        analytics_config = str(ANALYTICS_CFG)
    
    # Create pipeline
    pipeline = Gst.Pipeline.new(f"ds-pipeline-{sink_type}")
    
    # ========== SOURCE ==========
    source = make_element("source-bin", "uridecodebin")
    source.set_property("uri", uri)
    
    def on_source_setup(decodebin, src):
        """Configure source element (especially for RTSP)."""
        if not is_file:
            # RTSP optimization settings
            for prop, val in [("latency", 100), ("drop-on-latency", True)]:
                try:
                    src.set_property(prop, val)
                except (TypeError, Exception):
                    pass
    
    source.connect("source-setup", on_source_setup)
    
    # ========== CORE PROCESSING ==========
    streammux = make_element("stream-muxer", "nvstreammux")
    streammux.set_property('batch-size', 1)
    streammux.set_property('width', mux_width)
    streammux.set_property('height', mux_height)
    streammux.set_property('batched-push-timeout', 33000)
    streammux.set_property('live-source', is_live)
    
    pgie = make_element("primary-infer", "nvinfer")
    pgie.set_property('config-file-path', str(INFER_CONFIG))
    sgie = make_element("secondary-infer", "nvinfer")
    sgie.set_property('config-file-path', str(SGIE_CONFIG))
    
    tracker = make_element("tracker", "nvtracker")
    tracker.set_property('ll-lib-file', str(TRACKER_LIB))
    tracker.set_property('ll-config-file', str(TRACKER_CFG))
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('gpu_id', 0)
    
    analytics = make_element("analytics", "nvdsanalytics")
    analytics.set_property('config-file', analytics_config)
    
    # For WebRTC and File modes, we need RGBA format before OSD for proper overlay rendering
    if sink_type in ["webrtc", "file"]:
        preosd_convert = make_element("preosd_convert", "nvvideoconvert")
        preosd_caps = make_element("preosd_caps", "capsfilter")
        preosd_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    
    nvdsosd = make_element("onscreendisplay", "nvdsosd")
    nvdsosd.set_property("display-text", 1)
    nvdsosd.set_property("display-bbox", 1)
    
    # ========== SINK-SPECIFIC ELEMENTS ==========
    sink_elements = []
    
    if sink_type == "display":
        # Display: nvvideoconvert → nvegltransform → nveglglessink
        conv = make_element("conv", "nvvideoconvert")
        eglT = make_element("eglT", "nvegltransform")
        sink = make_element("display", "nveglglessink")
        sink.set_property("sync", False)
        sink.set_property("qos", False)
        sink_elements = [conv, eglT, sink]
        
    elif sink_type == "file":
        # File: nvvideoconvert (RGBA→I420) → nvv4l2h264enc → h264parse → qtmux → filesink
        # Need conversion because OSD outputs RGBA but encoder needs I420/NV12
        postosd_convert = make_element("postosd_convert", "nvvideoconvert")
        encoder = make_element("encoder", "nvv4l2h264enc")
        parser = make_element("parser", "h264parse")
        muxer = make_element("muxer", "qtmux")
        sink = make_element("filesink", "filesink")
        sink.set_property("location", os.path.abspath(output_path))
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        sink_elements = [postosd_convert, encoder, parser, muxer, sink]
        
    elif sink_type == "webrtc":
        # WebRTC: nvvideoconvert → nvv4l2h264enc → h264parse → rtph264pay → webrtcbin
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
        rtp_caps.set_property("caps", Gst.Caps.from_string(
            "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"))
        
        webrtc = make_element("webrtc", "webrtcbin")
        # try:
        #     webrtc.set_property("stun-server", "stun://stun.l.google.com:19302")
        # except (TypeError, Exception):
        #     pass
        
        sink_elements = [conv, enc, parse, pay, rtp_caps, webrtc]
    else:
        raise ValueError(f"Unknown sink_type: {sink_type}. Must be 'display', 'file', or 'webrtc'")
    
    # ========== ADD ELEMENTS TO PIPELINE ==========
    core_elements = [source, streammux, pgie, sgie, tracker, analytics]
    
    if sink_type in ["webrtc", "file"]:
        core_elements.extend([preosd_convert, preosd_caps])
    
    core_elements.append(nvdsosd)
    core_elements.extend(sink_elements)
    
    for element in core_elements:
        pipeline.add(element)
    
    # ========== LINK ELEMENTS ==========
    def on_pad_added(decodebin, pad):
        """Link source to streammux when pad is available."""
        caps = pad.get_current_caps()
        if not caps:
            return
        if caps.to_string().startswith("video/"):
            sinkpad = streammux.get_request_pad("sink_0")
            if sinkpad and not sinkpad.is_linked():
                pad.link(sinkpad)
    
    source.connect("pad-added", on_pad_added)
    
    # Link core processing chain
    assert streammux.link(pgie), "Failed to link streammux → pgie"
    assert pgie.link(sgie), "Failed to link pgie → sgie"
    assert sgie.link(tracker), "Failed to link sgie → tracker"
    assert tracker.link(analytics), "Failed to link tracker → analytics"
    
    if sink_type in ["webrtc", "file"]:
        assert analytics.link(preosd_convert), "Failed to link analytics → preosd_convert"
        assert preosd_convert.link(preosd_caps), "Failed to link preosd_convert → preosd_caps"
        assert preosd_caps.link(nvdsosd), "Failed to link preosd_caps → nvdsosd"
    else:
        assert analytics.link(nvdsosd), "Failed to link analytics → nvdsosd"
    
    # Link sink-specific chain
    if sink_type == "display":
        conv, eglT, sink = sink_elements
        assert nvdsosd.link(conv), "Failed to link nvdsosd → conv"
        assert conv.link(eglT), "Failed to link conv → eglT"
        assert eglT.link(sink), "Failed to link eglT → sink"
        
    elif sink_type == "file":
        postosd_convert, encoder, parser, muxer, sink = sink_elements
        assert nvdsosd.link(postosd_convert), "Failed to link nvdsosd → postosd_convert"
        assert postosd_convert.link(encoder), "Failed to link postosd_convert → encoder"
        assert encoder.link(parser), "Failed to link encoder → parser"
        assert parser.link(muxer), "Failed to link parser → muxer"
        assert muxer.link(sink), "Failed to link muxer → sink"
        
    elif sink_type == "webrtc":
        conv, enc, parse, pay, rtp_caps, webrtc = sink_elements
        assert nvdsosd.link(conv), "Failed to link nvdsosd → conv"
        assert conv.link(enc), "Failed to link conv → enc"
        assert enc.link(parse), "Failed to link enc → parse"
        assert parse.link(pay), "Failed to link parse → pay"
        assert pay.link(rtp_caps), "Failed to link pay → rtp_caps"
        
        # Link to webrtcbin using request pad
        srcpad = rtp_caps.get_static_pad("src")
        sinkpad = webrtc.get_request_pad("sink_%u")
        if not sinkpad:
            raise RuntimeError("Failed to get request pad from webrtcbin")
        
        link_res = srcpad.link(sinkpad)
        if link_res != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link RTP to webrtcbin: {link_res}")
    
    # Return appropriate tuple based on sink type
    if sink_type == "webrtc":
        return pipeline, nvdsosd, webrtc
    else:
        return pipeline, nvdsosd
