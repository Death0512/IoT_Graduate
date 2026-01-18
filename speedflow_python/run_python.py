#!/usr/bin/env python3
"""
Python backend runner - wraps existing Python pipeline logic.
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

from .core_pipeline import build_pipeline
from .homography import load_points, ViewTransformer
from .settings import HOMO_YML
from .probes import SpeedProbe, ROIFilterProbe
from .plate_preprocessor import PlatePreprocessorProbe
from .config_txt import load_kv_txt
from . import settings as S

# WebRTC imports
try:
    import websockets
except ImportError:
    websockets = None


class WebRTCSession:
    """Handles WebRTC signaling and session management."""
    
    def __init__(self, webrtc, ws_uri):
        self.webrtc = webrtc
        self.ws_uri = ws_uri
        self.ws = None
        self.loop = None
        self._closing = False

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
        if hasattr(self, '_ice_buffer') and self._ice_buffer:
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
            if not hasattr(self, '_ice_buffer'):
                self._ice_buffer = []
            self._ice_buffer.append(ice_msg)
            return
            
        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps(ice_msg)),
            self.loop
        )

    def send_json_threadsafe(self, data: dict):
        """Send JSON data to signaling server (thread-safe)."""
        if not self.ws:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps(data)), 
                self.loop
            )
        except Exception as e:
            print(f"[WebRTC] Failed to publish JSON: {e}")


def run_display_mode(args):
    """Run pipeline in display mode."""
    Gst.init(None)
    
    pipeline, nvdsosd = build_pipeline(
        source_uri=args.source,
        sink_type="display",
        mux_width=args.width,
        mux_height=args.height
    )
    
    # Setup ROI filter probe
    analytics = pipeline.get_by_name("analytics")
    if analytics:
        roi_filter = ROIFilterProbe()
        analytics_srcpad = analytics.get_static_pad("src")
        if analytics_srcpad:
            analytics_srcpad.add_probe(Gst.PadProbeType.BUFFER, roi_filter.analytics_src_pad_buffer_probe, None)
            print("[ROI Filter] Enabled")
    
    # Setup plate preprocessing probe
    tracker = pipeline.get_by_name("tracker")
    if tracker:
        plate_preprocessor = PlatePreprocessorProbe(
            enable_sharpening=True,
            enable_contrast=True, 
            enable_denoise=True,
            adaptive_mode=True  # ⚡ NEW: Auto-adjust based on motion blur
        )
        tracker_srcpad = tracker.get_static_pad("src")
        if tracker_srcpad:
            tracker_srcpad.add_probe(Gst.PadProbeType.BUFFER, plate_preprocessor.buffer_probe, None)
            print("[Plate Preprocessor] Enabled")
    
    # Setup homography and speed probe
    source_pts, target_pts = load_points(args.homo)
    vt = ViewTransformer(source_pts, target_pts)
    probe = SpeedProbe(vt, roi_source_points=source_pts)
    
    pad = nvdsosd.get_static_pad("sink")
    if not pad:
        print("ERROR: Unable to get sink pad of nvdsosd", file=sys.stderr)
        sys.exit(1)
    pad.add_probe(Gst.PadProbeType.BUFFER, probe.osd_sink_pad_buffer_probe, None)
    
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
        sys.exit(1)
    
    print(f"[Python Display Mode] Running with source: {args.source}")
    print("Press Ctrl+C to stop...")
    
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


def run_file_mode(args):
    """Run pipeline in file mode."""
    Gst.init(None)
    
    if not os.path.exists(args.source):
        print(f"ERROR: Input file not found: {args.source}", file=sys.stderr)
        sys.exit(1)
    
    pipeline, nvdsosd = build_pipeline(
        source_uri=args.source,
        sink_type="file",
        output_path=args.output,
        mux_width=args.width,
        mux_height=args.height
    )
    
    # Setup ROI filter probe
    analytics = pipeline.get_by_name("analytics")
    if analytics:
        roi_filter = ROIFilterProbe()
        analytics_srcpad = analytics.get_static_pad("src")
        if analytics_srcpad:
            analytics_srcpad.add_probe(Gst.PadProbeType.BUFFER, roi_filter.analytics_src_pad_buffer_probe, None)
            print("[ROI Filter] Enabled")
    
    # Setup plate preprocessing probe
    tracker = pipeline.get_by_name("tracker")
    if tracker:
        plate_preprocessor = PlatePreprocessorProbe(
            enable_sharpening=True,
            enable_contrast=True,
            enable_denoise=True
        )
        tracker_srcpad = tracker.get_static_pad("src")
        if tracker_srcpad:
            tracker_srcpad.add_probe(Gst.PadProbeType.BUFFER, plate_preprocessor.buffer_probe, None)
            print("[Plate Preprocessor] Enabled")
    
    # Setup homography and speed probe
    source_pts, target_pts = load_points(args.homo)
    vt = ViewTransformer(source_pts, target_pts)
    probe = SpeedProbe(vt, roi_source_points=source_pts)
    
    pad = nvdsosd.get_static_pad("sink")
    if not pad:
        print("ERROR: Unable to get sink pad of nvdsosd", file=sys.stderr)
        sys.exit(1)
    pad.add_probe(Gst.PadProbeType.BUFFER, probe.osd_sink_pad_buffer_probe, None)
    
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
            print("EOS received - Processing complete")
            loop.quit()
    
    bus.connect("message", on_message)
    
    print(f"[Python File Mode] Processing: {args.source} → {args.output}")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR)
        if msg:
            err, debug = msg.parse_error()
            print(f"ERROR: Pipeline failed to start: {err}", file=sys.stderr)
            if debug:
                print(f"DEBUG INFO: {debug}", file=sys.stderr)
        else:
            print("ERROR: Unable to set pipeline to PLAYING state", file=sys.stderr)
        sys.exit(1)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


async def run_webrtc_mode_async(args):
    """Run pipeline in WebRTC mode."""
    if websockets is None:
        print("ERROR: websockets module not installed. Run: pip install websockets", file=sys.stderr)
        sys.exit(1)
    
    kv = load_kv_txt(args.cfg)
    S.ANALYTICS_CFG = kv["ANALYTICS_CFG"]
    S.HOMO_YML = kv["HOMO_YML"]
    S.VIDEO_FPS = kv["VIDEO_FPS"]
    S.MUX_WIDTH = int(kv.get("MUX_WIDTH", args.width))
    S.MUX_HEIGHT = int(kv.get("MUX_HEIGHT", args.height))
    
    print(f"[Config] ANALYTICS_CFG = {S.ANALYTICS_CFG}")
    print(f"[Config] HOMO_YML = {S.HOMO_YML}")
    print(f"[Config] VIDEO_FPS = {S.VIDEO_FPS}")
    print(f"[Config] Resolution = {S.MUX_WIDTH}x{S.MUX_HEIGHT}")
    
    pipeline, nvdsosd, webrtc = build_pipeline(
        source_uri=args.source,
        sink_type="webrtc",
        mux_width=S.MUX_WIDTH,
        mux_height=S.MUX_HEIGHT,
        analytics_config=S.ANALYTICS_CFG
    )
    
    # Setup ROI filter probe
    analytics = pipeline.get_by_name("analytics")
    if analytics:
        roi_filter = ROIFilterProbe()
        analytics_srcpad = analytics.get_static_pad("src")
        if analytics_srcpad:
            analytics_srcpad.add_probe(Gst.PadProbeType.BUFFER, roi_filter.analytics_src_pad_buffer_probe, None)
            print("[ROI Filter] Enabled")
    
    # Setup plate preprocessing probe
    tracker = pipeline.get_by_name("tracker")
    if tracker:
        plate_preprocessor = PlatePreprocessorProbe(
            enable_sharpening=True,
            enable_contrast=True,
            enable_denoise=True
        )
        tracker_srcpad = tracker.get_static_pad("src")
        if tracker_srcpad:
            tracker_srcpad.add_probe(Gst.PadProbeType.BUFFER, plate_preprocessor.buffer_probe, None)
            print("[Plate Preprocessor] Enabled")
    
    # Setup homography and speed probe
    source_pts, target_pts = load_points(str(S.HOMO_YML))
    vt = ViewTransformer(source_pts, target_pts)
    probe = SpeedProbe(vt, roi_source_points=source_pts, cooldown_s=2.5)
    
    pad = nvdsosd.get_static_pad("sink")
    pad.add_probe(Gst.PadProbeType.BUFFER, probe.osd_sink_pad_buffer_probe, None)
    
    # Setup WebRTC session
    ws_uri = f"ws://{args.server}:{args.port}/ws?room={args.room}&role=pub"
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
        await asyncio.get_event_loop().run_in_executor(None, loop.run)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")


def run_webrtc_mode(args):
    """Wrapper to run WebRTC mode with asyncio."""
    Gst.init(None)
    asyncio.run(run_webrtc_mode_async(args))


def run_python_mode(args):
    """Main dispatcher for Python backend."""
    if args.mode == "display":
        run_display_mode(args)
    elif args.mode == "file":
        run_file_mode(args)
    elif args.mode == "webrtc":
        run_webrtc_mode(args)
