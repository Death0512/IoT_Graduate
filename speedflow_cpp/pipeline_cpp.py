#!/usr/bin/env python3
"""
C++ Backend Pipeline Runner
Uses custom GStreamer plugin for speed measurement
"""
import sys
import os
import asyncio
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstWebRTC', '1.0')
gi.require_version('GstSdp', '1.0')

from gi.repository import Gst, GLib, GstWebRTC, GstSdp
import json

# Import shared configs from Python module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speedflow_python.settings import (INFER_CONFIG, TRACKER_CFG, ANALYTICS_CFG, SGIE_CONFIG, TRACKER_LIB, LPR_CONFIG, HOMO_YML)
from speedflow_python.config_txt import load_kv_txt

# WebRTC imports
try:
    import websockets
except ImportError:
    websockets = None


# Path to C++ plugin
PLUGIN_PATH = os.path.join(os.path.dirname(__file__), "build", "libgstspeedflow.so")


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
        abs_path = os.path.abspath(uri)
        return "file://" + abs_path
    return uri


def build_pipeline_cpp(source_uri: str, sink_type: str = "display", 
                       output_path: str = None, mux_width: int = 1920, 
                       mux_height: int = 1080, is_live: bool = None,
                       analytics_config: str = None, homo_config: str = None,
                       video_fps: int = 30, **kwargs):
    """
    Build DeepStream pipeline using C++ speedflow plugin.
    
    This pipeline replaces Python probes with a GStreamer C++ plugin
    for better performance.
    """
    # Load C++ plugin if not already registered
    if not Gst.Registry.get().find_plugin("speedflow"):
        if os.path.exists(PLUGIN_PATH):
            Gst.Registry.get().scan_path(os.path.dirname(PLUGIN_PATH))
            print(f"[C++ Plugin] Loaded from {PLUGIN_PATH}")
        else:
            raise RuntimeError(f"C++ plugin not found at {PLUGIN_PATH}. Please build it first.")
    
    # Normalize inputs
    uri = normalize_uri(source_uri)
    is_file = is_file_uri(uri)
    
    if is_live is None:
        is_live = 0 if is_file else 1
    
    if sink_type == "file" and not output_path:
        raise ValueError("output_path is required when sink_type='file'")
    
    if analytics_config is None:
        analytics_config = str(ANALYTICS_CFG)
    
    if homo_config is None:
        homo_config = str(HOMO_YML)
    
    # Create pipeline
    pipeline = Gst.Pipeline.new(f"ds-cpp-pipeline-{sink_type}")
    
    # ========== SOURCE ==========
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
    
    # ========== CORE PROCESSING ==========
    streammux = make_element("stream-muxer", "nvstreammux")
    streammux.set_property('batch-size', 1)
    streammux.set_property('width', mux_width)
    streammux.set_property('height', mux_height)
    streammux.set_property('batched-push-timeout', 33000)
    streammux.set_property('live-source', is_live)
    
    # ========== NVIDIA OPTICAL FLOW (C++ EXCLUSIVE) ==========
    # NVOF calculates motion vectors for improved tracking and speed estimation
    # Only available in C++ backend - Python cannot access NVOF metadata
    nvof = make_element("nvof", "nvof")
    nvof.set_property('gpu-id', 0)
    nvof.set_property('preset-level', 2)  # 0=slow, 1=medium, 2=fast
    nvof.set_property('grid-size', 4)     # 4x4 grid for detailed motion vectors
    print("[C++ NVOF] Optical Flow enabled - Motion vectors will improve speed accuracy")
    
    pgie = make_element("primary-infer", "nvinfer")
    pgie.set_property('config-file-path', str(INFER_CONFIG))
    
    sgie = make_element("secondary-infer", "nvinfer")
    sgie.set_property('config-file-path', str(SGIE_CONFIG))
    
    sgie2 = make_element("lpr-classifier", "nvinfer")
    sgie2.set_property('config-file-path', str(LPR_CONFIG))
    
    tracker = make_element("tracker", "nvtracker")
    tracker.set_property('ll-lib-file', str(TRACKER_LIB))
    tracker.set_property('ll-config-file', str(TRACKER_CFG))
    tracker.set_property('tracker-width', 224)
    tracker.set_property('tracker-height', 224)
    tracker.set_property('gpu_id', 0)
    
    analytics = make_element("analytics", "nvdsanalytics")
    analytics.set_property('config-file', analytics_config)
    
    # ========== C++ SPEEDFLOW PLUGIN ==========
    # This replaces Python probes (SpeedProbe, ROIFilterProbe, PlatePreprocessor)
    # In C++ mode, we enable NVOF to access motion vectors
    speedflow = make_element("speedflow-plugin", "speedflow")
    speedflow.set_property("config-file", homo_config)
    speedflow.set_property("speed-limit", 80.0)
    speedflow.set_property("video-fps", video_fps)
    speedflow.set_property("enable-nvof", True)  # NVOF ENABLED for C++ backend
    
    print(f"[C++ SpeedFlow] Homography config: {homo_config}")
    print(f"[C++ SpeedFlow] Speed limit: 80.0 km/h")
    print(f"[C++ SpeedFlow] NVOF: ENABLED (motion vector analysis)")
    print(f"[C++ SpeedFlow] Video FPS: {video_fps}")
    
    # ========== OSD ==========
    preosd_convert = make_element("preosd_convert", "nvvideoconvert")
    preosd_caps = make_element("preosd_caps", "capsfilter")
    preosd_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    
    nvdsosd = make_element("onscreendisplay", "nvdsosd")
    nvdsosd.set_property("display-text", 1)
    nvdsosd.set_property("display-bbox", 1)
    nvdsosd.set_property("process-mode", 2)
    nvdsosd.set_property("gpu-id", 0)
    
    # ========== SINK-SPECIFIC ELEMENTS ==========
    sink_elements = []
    
    if sink_type == "display":
        conv = make_element("conv", "nvvideoconvert")
        conv_caps = make_element("conv_caps", "capsfilter")
        conv_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))
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
        parser = make_element("parser", "h264parse")
        muxer = make_element("muxer", "qtmux")
        sink = make_element("filesink", "filesink")
        sink.set_property("location", os.path.abspath(output_path))
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sink_elements = [postosd_convert, encoder, parser, muxer, sink]
        
    elif sink_type == "webrtc":
        conv = make_element("conv", "nvvideoconvert")
        enc = make_element("enc", "nvv4l2h264enc")
        
        # OPTIMIZED SETTINGS FOR SMOOTH WEBRTC STREAMING
        # Bitrate: 2.5 Mbps (lower = less network congestion)
        enc.set_property("bitrate", 2_000_000)
        
        # Iframe interval: 15 frames (~0.6s at 25fps) for faster recovery from packet loss
        enc.set_property("iframeinterval", 15)
        
        # Insert SPS/PPS for WebRTC compatibility
        enc.set_property("insert-sps-pps", True)
        
        # Constant Bitrate (CBR) for predictable bandwidth
        enc.set_property("control-rate", 1)  # 0=variable, 1=constant
        
        # Encoding preset: UltraFast for low latency
        enc.set_property("preset-level", 1)  # 0=slow, 1=medium, 2=fast, 3=ultrafast
        
        # Performance optimizations
        try:
            enc.set_property("maxperf-enable", True)
        except:
            pass
        
        # Profile: Baseline for better compatibility
        enc.set_property("profile", 0)  # 0=Baseline, 2=Main, 4=High
        
        parse = make_element("parse", "h264parse")
        parse.set_property("config-interval", -1)  # Insert SPS/PPS at every IDR
        
        pay = make_element("pay", "rtph264pay")
        pay.set_property("pt", 96)
        pay.set_property("config-interval", 1)  # Send config every second
        
        # MTU optimization for network packets
        pay.set_property("mtu", 1200)  # Smaller MTU = less fragmentation
        
        rtp_caps = make_element("rtp_caps", "capsfilter")
        rtp_caps.set_property("caps", Gst.Caps.from_string(
            "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"))
        webrtc = make_element("webrtc", "webrtcbin")
        
        # WebRTC settings for stability
        webrtc.set_property("bundle-policy", 3)  # max-bundle
        webrtc.set_property("latency", 100)  # 100ms latency target
        
        sink_elements = [conv, enc, parse, pay, rtp_caps, webrtc]
    else:
        raise ValueError(f"Unknown sink_type: {sink_type}")
    
    # ========== ADD ELEMENTS TO PIPELINE ==========
    # Note: speedflow plugin replaces Python probes
    # NVOF added for C++ backend (can access motion vectors)
    core_elements = [source, streammux, nvof, pgie, tracker, sgie, sgie2, analytics, speedflow, preosd_convert, preosd_caps, nvdsosd]
    core_elements.extend(sink_elements)
    
    for element in core_elements:
        pipeline.add(element)
    
    # ========== LINK ELEMENTS ==========
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
                queue.link(convert)
                convert_src = convert.get_static_pad("src")
                convert_src.link(sinkpad)
    
    source.connect("pad-added", on_pad_added)
    
    # Link core processing chain (with NVOF and speedflow plugin)
    # Streammux → NVOF → PGIE → Tracker → SGIE → SGIE2 → Analytics → SpeedFlow → OSD
    assert streammux.link(nvof), "Failed to link streammux → nvof"
    assert nvof.link(pgie), "Failed to link nvof → pgie"
    assert pgie.link(tracker), "Failed to link pgie → tracker"
    assert tracker.link(sgie), "Failed to link tracker → sgie"
    assert sgie.link(sgie2), "Failed to link sgie → sgie2"
    assert sgie2.link(analytics), "Failed to link sgie2 → analytics"
    assert analytics.link(speedflow), "Failed to link analytics → speedflow"
    assert speedflow.link(preosd_convert), "Failed to link speedflow → preosd_convert"
    assert preosd_convert.link(preosd_caps), "Failed to link preosd_convert → preosd_caps"
    assert preosd_caps.link(nvdsosd), "Failed to link preosd_caps → nvdsosd"
    
    # Link sink-specific chain
    if sink_type == "display":
        conv, conv_caps, eglT, sink = sink_elements
        assert nvdsosd.link(conv)
        assert conv.link(conv_caps)
        assert conv_caps.link(eglT)
        assert eglT.link(sink)
        
    elif sink_type == "file":
        postosd_convert, encoder, parser, muxer, sink = sink_elements
        assert nvdsosd.link(postosd_convert)
        assert postosd_convert.link(encoder)
        assert encoder.link(parser)
        assert parser.link(muxer)
        assert muxer.link(sink)
        
    elif sink_type == "webrtc":
        conv, enc, parse, pay, rtp_caps, webrtc = sink_elements
        assert nvdsosd.link(conv)
        assert conv.link(enc)
        assert enc.link(parse)
        assert parse.link(pay)
        assert pay.link(rtp_caps)
        srcpad = rtp_caps.get_static_pad("src")
        sinkpad = webrtc.get_request_pad("sink_%u")
        if not sinkpad:
            raise RuntimeError("Failed to get request pad from webrtcbin")
        link_res = srcpad.link(sinkpad)
        if link_res != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link RTP to webrtcbin: {link_res}")
    
    # Return tuple based on sink type
    if sink_type == "webrtc":
        return pipeline, nvdsosd, sink_elements[-1]  # webrtcbin
    else:
        return pipeline, nvdsosd


def run_display_mode_cpp(args):
    """Run C++ pipeline in display mode."""
    Gst.init(None)
    
    pipeline, nvdsosd = build_pipeline_cpp(
        source_uri=args.source,
        sink_type="display",
        mux_width=args.width,
        mux_height=args.height,
        homo_config=args.homo
    )
    
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
        sys.exit(1)
    
    print(f"[C++ Display Mode] Pipeline running with source: {args.source}")
    print("Press Ctrl+C to stop...")
    
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


def run_file_mode_cpp(args):
    """Run C++ pipeline in file mode."""
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
        homo_config=args.homo
    )
    
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    
    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"ERROR from {message.src.get_name()}: {err}", file=sys.stderr)
            loop.quit()
        elif t == Gst.MessageType.EOS:
            print("EOS received - Processing complete")
            loop.quit()
    
    bus.connect("message", on_message)
    
    print(f"[C++ File Mode] Processing: {args.source} → {args.output}")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
        sys.exit(1)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


class WebRTCSessionCpp:
    """Handles WebRTC signaling for C++ backend."""
    
    def __init__(self, webrtc, ws_uri):
        self.webrtc = webrtc
        self.ws_uri = ws_uri
        self.ws = None
        self.loop = None
        self._closing = False
        self._ice_buffer = []

        self.webrtc.connect("on-negotiation-needed", self.on_negotiation_needed)
        self.webrtc.connect("on-ice-candidate", self.on_ice_candidate)

    async def _ws_connect(self):
        """Connect to WebSocket signaling server with retry."""
        while not self._closing:
            try:
                self.ws = await websockets.connect(self.ws_uri)
                print(f"[WebRTC] Connected to signaling server: {self.ws_uri}")
                return
            except Exception as e:
                print(f"[WebRTC] Connection failed, retrying... ({e})")
                await asyncio.sleep(1.2)

    async def connect(self):
        """Establish WebSocket connection and start receive loop."""
        self.loop = asyncio.get_running_loop()
        await self._ws_connect()
        asyncio.create_task(self._recv_loop())
        await self._flush_ice_buffer()
        await asyncio.sleep(0.2)
        self.on_negotiation_needed(self.webrtc)
    
    async def _flush_ice_buffer(self):
        """Send any buffered ICE candidates."""
        if self._ice_buffer:
            for ice_msg in self._ice_buffer:
                try:
                    await self.ws.send(json.dumps(ice_msg))
                except:
                    pass
            self._ice_buffer = []

    async def _recv_loop(self):
        """Receive and process signaling messages."""
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")
                
                if msg_type == "answer":
                    res, sdpmsg = GstSdp.SDPMessage.new_from_text(msg["sdp"])
                    if res != GstSdp.SDPResult.OK:
                        print("[WebRTC] SDP parse failed")
                        continue
                    answer = GstWebRTC.WebRTCSessionDescription.new(
                        GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg
                    )
                    self.webrtc.emit("set-remote-description", answer, None)
                    print("[WebRTC] Remote description set")
                    await self._flush_ice_buffer()
                    
                elif msg_type == "ice":
                    cand = msg["candidate"]["candidate"]
                    mline = int(msg["candidate"]["sdpMLineIndex"])
                    self.webrtc.emit("add-ice-candidate", mline, cand)
                    
        except Exception as e:
            print(f"[WebRTC] Receive loop ended: {e}")
            await self._handle_ws_drop()

    async def _handle_ws_drop(self):
        """Handle WebSocket disconnection with reconnection."""
        if self._closing:
            return
        try:
            if self.ws:
                await self.ws.close()
        except:
            pass
        self.ws = None
        print("[WebRTC] Reconnecting...")
        await asyncio.sleep(0.8)
        await self._ws_connect()
        await asyncio.sleep(0.2)
        self.on_negotiation_needed(self.webrtc)

    def on_negotiation_needed(self, element):
        """Handle negotiation needed signal."""
        print("[WebRTC] Negotiation needed")
        promise = Gst.Promise.new_with_change_func(self._on_offer_created, element, None)
        self.webrtc.emit("create-offer", None, promise)

    def _on_offer_created(self, promise, element, _):
        """Send offer to signaling server."""
        reply = promise.get_reply()
        offer = reply.get_value("offer")
        if not offer:
            print("[WebRTC] Failed to create offer - webrtcbin not ready yet")
            return
        if not self.ws:
            print("[WebRTC] WebSocket not connected yet, skipping offer send")
            return
        self.webrtc.emit("set-local-description", offer, None)
        text = offer.sdp.as_text()
        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps({"type": "offer", "sdp": text})), 
            self.loop
        )

    def on_ice_candidate(self, element, mline, candidate):
        """Send ICE candidate to signaling server."""
        if not candidate:
            return
            
        ice_msg = {
            "type": "ice",
            "candidate": {
                "candidate": candidate,
                "sdpMLineIndex": int(mline)
            }
        }
        
        if not self.ws:
            self._ice_buffer.append(ice_msg)
            return
            
        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps(ice_msg)),
            self.loop
        )


async def run_webrtc_mode_cpp_async(args):
    """Run C++ pipeline in WebRTC mode."""
    if websockets is None:
        print("ERROR: websockets module not installed. Run: pip install websockets", file=sys.stderr)
        sys.exit(1)
    
    # Load config from TXT file
    kv = load_kv_txt(args.cfg)
    analytics_config = kv.get("ANALYTICS_CFG", str(ANALYTICS_CFG))
    homo_config = kv.get("HOMO_YML", str(HOMO_YML))
    video_fps = int(kv.get("VIDEO_FPS", 30))
    mux_width = int(kv.get("MUX_WIDTH", args.width))
    mux_height = int(kv.get("MUX_HEIGHT", args.height))
    
    print(f"[Config] ANALYTICS_CFG = {analytics_config}")
    print(f"[Config] HOMO_YML = {homo_config}")
    print(f"[Config] VIDEO_FPS = {video_fps}")
    print(f"[Config] Resolution = {mux_width}x{mux_height}")
    
    pipeline, nvdsosd, webrtc = build_pipeline_cpp(
        source_uri=args.source,
        sink_type="webrtc",
        mux_width=mux_width,
        mux_height=mux_height,
        analytics_config=analytics_config,
        homo_config=homo_config,
        video_fps=video_fps
    )
    
    # Setup WebRTC session
    ws_uri = f"ws://{args.server}:{args.port}/ws?room={args.room}&role=pub"
    session = WebRTCSessionCpp(webrtc, ws_uri)
    
    pipeline.set_state(Gst.State.PLAYING)
    print(f"[C++ WebRTC Mode] Pipeline running")
    print(f"[C++ WebRTC Mode] Room: {args.room}")
    print(f"[C++ WebRTC Mode] View stream at: http://{args.server}:{args.port}/")
    
    await asyncio.sleep(1.5)
    await session.connect()
    
    loop = GLib.MainLoop()
    try:
        await asyncio.get_event_loop().run_in_executor(None, loop.run)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


def run_webrtc_mode_cpp(args):
    """Wrapper to run WebRTC mode with asyncio."""
    Gst.init(None)
    asyncio.run(run_webrtc_mode_cpp_async(args))


def run_cpp_mode(args):
    """Main dispatcher for C++ backend."""
    if args.mode == "display":
        run_display_mode_cpp(args)
    elif args.mode == "file":
        run_file_mode_cpp(args)
    elif args.mode == "webrtc":
        run_webrtc_mode_cpp(args)
