# speedflow/core_pipeline.py
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from .settings import INFER_CONFIG, TRACKER_CFG, ANALYTICS_CFG, SGIE_CONFIG, TRACKER_LIB, LPR_CONFIG, TRACKER_LPD_CFG

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
                mux_width: int = 1920, mux_height: int = 1080, is_live: bool = None, analytics_config: str = None, **kwargs):
    """   
    Args:
        source_uri: Input source (RTSP URL or file path)
        sink_type: Output type - "display", "file", or "webrtc"
        output_path: Output file path (required for sink_type="file")
        mux_width: Streammux width
        mux_height: Streammux height
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
    
    # SGIE2: LPR (License Plate Recognition) - Character Classifier
    sgie2 = make_element("lpr-classifier", "nvinfer")
    sgie2.set_property('config-file-path', str(LPR_CONFIG))
    
    # OPTIMIZED: Single unified tracker after PGIE (tracks both vehicles and plates)
    # Moved here to stabilize vehicle bboxes BEFORE plate detection
    tracker = make_element("tracker", "nvtracker")
    tracker.set_property('ll-lib-file', str(TRACKER_LIB))
    tracker.set_property('ll-config-file', str(TRACKER_CFG))
    tracker.set_property('tracker-width', 224)
    tracker.set_property('tracker-height', 224)
    tracker.set_property('gpu_id', 0)
    
    analytics = make_element("analytics", "nvdsanalytics")
    analytics.set_property('config-file', analytics_config)
    
    # Pre-OSD conversion: Convert to RGBA for proper overlay rendering (fixes ghosting)
    preosd_convert = make_element("preosd_convert", "nvvideoconvert")
    preosd_caps = make_element("preosd_caps", "capsfilter")
    preosd_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    
    nvdsosd = make_element("onscreendisplay", "nvdsosd")
    nvdsosd.set_property("display-text", 1)
    nvdsosd.set_property("display-bbox", 1)
    # FIX: Set process-mode to GPU for better performance and no ghosting
    nvdsosd.set_property("process-mode", 2)  # 0=CPU, 1=GPU (legacy), 2=GPU (new)
    nvdsosd.set_property("gpu-id", 0)
    
    # ========== SINK-SPECIFIC ELEMENTS ==========
    sink_elements = []
    
    if sink_type == "display":
        # Display: nvvideoconvert → nvegltransform → nveglglessink
        conv = make_element("conv", "nvvideoconvert")
        # FIX: Force output format to avoid buffer issues
        conv_caps = make_element("conv_caps", "capsfilter")
        conv_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))
        
        eglT = make_element("eglT", "nvegltransform")
        sink = make_element("display", "nveglglessink")
        
        # OPTIMIZED: Anti-lag settings for smooth display
        sink.set_property("sync", False)           # Don't wait for clock sync
        sink.set_property("qos", False)            # Disable quality-of-service events
        sink.set_property("async", False)          # Don't wait for preroll
        sink.set_property("max-lateness", -1)      # Drop late frames immediately
        
        sink_elements = [conv, conv_caps, eglT, sink]
        
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
        # WebRTC: Low-Quality Optimized Pipeline for Smooth Streaming
        # nvvideoconvert (downscale to 720p, keep NVMM) → nvv4l2h264enc → h264parse → queue → rtph264pay → webrtcbin
        
        # Downscale to 720p using nvvideoconvert (keeps NVMM memory for encoder)
        conv = make_element("conv", "nvvideoconvert")
        conv_caps = make_element("conv_caps", "capsfilter")
        # Note: Only specify resolution, let framerate passthrough from source
        conv_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM),format=NV12,width=1280,height=720"))
        
        enc = make_element("enc", "nvv4l2h264enc")
        
        # --- Balanced Quality Settings (720p @ 2Mbps) ---
        enc.set_property("insert-sps-pps", True)       # Required for WebRTC
        enc.set_property("iframeinterval", 25)         # Keyframe every 1s (at 25fps)
        enc.set_property("bitrate", 700000)           # 2 Mbps: Good quality at 720p
        enc.set_property("profile", 0)                 # Baseline profile
        enc.set_property("preset-level", 1)            # UltraFast encoding
        
        try:
            enc.set_property("maxperf-enable", True)
        except (TypeError, Exception):
            pass
        
        parse = make_element("parse", "h264parse")
        
        # Queue to decouple encoder from network (prevents blocking)
        enc_queue = make_element("enc_queue", "queue")
        
        pay = make_element("pay", "rtph264pay")
        pay.set_property("pt", 96)
        pay.set_property("config-interval", -1)        # Send SPS/PPS with every IDR
        
        rtp_caps = make_element("rtp_caps", "capsfilter")
        rtp_caps.set_property("caps", Gst.Caps.from_string(
            "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"))
        
        webrtc = make_element("webrtc", "webrtcbin")
        
        sink_elements = [conv, conv_caps, enc, parse, enc_queue, pay, rtp_caps, webrtc]
    else:
        raise ValueError(f"Unknown sink_type: {sink_type}. Must be 'display', 'file', or 'webrtc'")
    
    # ========== ADD ELEMENTS TO PIPELINE ==========
    # OPTIMIZED: tracker moved right after pgie, tracker_lpd removed
    # FIX: preosd_convert added for all sink types to fix ghosting
    core_elements = [source, streammux, pgie, tracker, sgie, sgie2, analytics, 
                     preosd_convert, preosd_caps, nvdsosd]
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
                # Add queue and nvvideoconvert to fix format negotiation issues
                queue = make_element("source_queue", "queue")
                convert = make_element("source_convert", "nvvideoconvert")
                
                pipeline.add(queue)
                pipeline.add(convert)
                
                # Sync state since pipeline might be already running/paused
                queue.sync_state_with_parent()
                convert.sync_state_with_parent()
                
                # Link: decodebin_pad -> queue -> convert -> streammux_sinkpad
                pad.link(queue.get_static_pad("sink"))
                queue.link(convert)
                
                convert_src = convert.get_static_pad("src")
                convert_src.link(sinkpad)
    
    source.connect("pad-added", on_pad_added)
    
    # Link core processing chain
    # OPTIMIZED FLOW: PGIE detects vehicles → Tracker tracks and stabilizes → 
    # SGIE detects plates on stabilized bboxes → SGIE2 recognizes text → Analytics
    assert streammux.link(pgie), "Failed to link streammux → pgie"
    assert pgie.link(tracker), "Failed to link pgie → tracker"
    assert tracker.link(sgie), "Failed to link tracker → sgie (LPD)"
    assert sgie.link(sgie2), "Failed to link sgie → sgie2 (LPR)"
    assert sgie2.link(analytics), "Failed to link sgie2 → analytics"
    
    # FIX: Always use preosd_convert to fix ghosting issue
    assert analytics.link(preosd_convert), "Failed to link analytics → preosd_convert"
    assert preosd_convert.link(preosd_caps), "Failed to link preosd_convert → preosd_caps"
    assert preosd_caps.link(nvdsosd), "Failed to link preosd_caps → nvdsosd"
    
    # Link sink-specific chain
    if sink_type == "display":
        conv, conv_caps, eglT, sink = sink_elements
        assert nvdsosd.link(conv), "Failed to link nvdsosd → conv"
        assert conv.link(conv_caps), "Failed to link conv → conv_caps"
        assert conv_caps.link(eglT), "Failed to link conv_caps → eglT"
        assert eglT.link(sink), "Failed to link eglT → sink"
        
    elif sink_type == "file":
        postosd_convert, encoder, parser, muxer, sink = sink_elements
        assert nvdsosd.link(postosd_convert), "Failed to link nvdsosd → postosd_convert"
        assert postosd_convert.link(encoder), "Failed to link postosd_convert → encoder"
        assert encoder.link(parser), "Failed to link encoder → parser"
        assert parser.link(muxer), "Failed to link parser → muxer"
        assert muxer.link(sink), "Failed to link muxer → sink"
        
    elif sink_type == "webrtc":
        conv, conv_caps, enc, parse, enc_queue, pay, rtp_caps, webrtc = sink_elements
        assert nvdsosd.link(conv), "Failed to link nvdsosd → conv"
        assert conv.link(conv_caps), "Failed to link conv → conv_caps"
        assert conv_caps.link(enc), "Failed to link conv_caps → enc"
        assert enc.link(parse), "Failed to link enc → parse"
        assert parse.link(enc_queue), "Failed to link parse → enc_queue"
        assert enc_queue.link(pay), "Failed to link enc_queue → pay"
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
