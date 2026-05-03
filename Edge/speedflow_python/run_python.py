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

    Thread-safety note:
        source_id_to_cam_id is ONLY written inside on_add/on_remove,
        which are always called via GLib.idle_add → run on GLib Main Loop
        thread → no concurrent mutation possible.
    """
    # Mapping ngược: source_id (int) → camera_id (str)
    # Khởi tạo từ các camera đã được enabled khi pipeline bắt đầu.
    # Chỉ được đọc/ghi từ GLib Main Loop thread (thông qua idle_add).
    source_id_to_cam_id: dict[int, str] = {
        cfg.source_id: cfg.camera_id
        for cfg in camera_manager.get_enabled_configs()
    }

    def on_add(cam_cfg):
        current_n = streammux.get_property("batch-size")
        print(f"[Dynamic] Adding camera '{cam_cfg.camera_id}' (source_id={cam_cfg.source_id})")
        dynamic_add_stream(pipeline, streammux, cam_cfg, tiler, source_bins, current_n)
        # Đăng ký ánh xạ ngay sau khi thêm thành công.
        # Hàm này chạy trong GLib Main Loop → an toàn, không cần lock.
        source_id_to_cam_id[cam_cfg.source_id] = cam_cfg.camera_id

    def on_remove(source_id):
        # Tra cứu camera_id từ dict ánh xạ —
        # không dùng GStreamer pad scan vì phức tạp và không thread-safe.
        cam_id = source_id_to_cam_id.get(source_id)
        if cam_id is None:
            print(
                f"[Dynamic] WARN: No camera mapped to source_id={source_id}. "
                "Possibly already removed or never registered.",
                file=sys.stderr
            )
            return

        current_n = streammux.get_property("batch-size")
        print(f"[Dynamic] Removing camera '{cam_id}' (source_id={source_id})")
        dynamic_remove_stream(pipeline, streammux, cam_id, source_id, tiler, source_bins, current_n)

        # Dọn dẹp key sau khi xóa thành công.
        # Phòng tránh memory leak khi hệ thống chạy liên tục nhiều tháng
        # và xung đột source_id nếu cùng ID được tái sử dụng sau này.
        removed = source_id_to_cam_id.pop(source_id, None)
        if removed:
            print(f"[Dynamic] Cleaned up mapping: source_id={source_id} → '{removed}'")

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

    ws_uri = f"ws://{args.server}:{args.port}/ws?room={args.room}&role=pub"
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
    camera_manager = CameraManager(CAMERAS_YML)

    # --- MQTT Command & Control ---
    # Thay thế REST API bằng MQTT subscriber để nhận lệnh ADD/REMOVE từ Master.
    # Kiến trúc này cho phép hoạt động qua tường lửa/NAT mà không cần mở cổng HTTP.
    # Cấu hình qua biến môi trường:
    #   NODE_ID           — định danh node này (mặc định: "jetson_default")
    #   MQTT_BROKER_HOST  — IP/hostname của MQTT Broker (mặc định: "localhost")
    #   MQTT_BROKER_PORT  — cổng Broker (mặc định: 1883)
    #   MQTT_USER         — username (tuỳ chọn, dùng khi Broker bật xác thực)
    #   MQTT_PASS         — password (tuỳ chọn)
    mqtt_sub = None
    try:
        from .mqtt_subscriber import MQTTCommandSubscriber
        node_id       = os.environ.get("NODE_ID", "jetson_default")
        broker_host   = os.environ.get("MQTT_BROKER_HOST", "localhost")
        broker_port   = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
        mqtt_user     = os.environ.get("MQTT_USER", None)
        mqtt_pass     = os.environ.get("MQTT_PASS", None)

        mqtt_sub = MQTTCommandSubscriber(
            camera_manager=camera_manager,
            node_id=node_id,
            broker_host=broker_host,
            broker_port=broker_port,
            username=mqtt_user,
            password=mqtt_pass,
        )
        mqtt_sub.start()
        print(
            f"[MQTT C2] Subscriber active. Node='{node_id}', "
            f"Broker={broker_host}:{broker_port}, "
            f"Topic=edge/control/{node_id}"
        )
    except ImportError:
        print(
            "[MQTT C2] paho-mqtt not installed — MQTT control disabled. "
            "Run: pip install paho-mqtt",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"[MQTT C2] Failed to start subscriber: {exc}", file=sys.stderr)

    if args.mode == "display":
        run_display_mode(args, camera_manager)
    elif args.mode == "file":
        run_file_mode(args, camera_manager)
    elif args.mode == "webrtc":
        run_webrtc_mode(args, camera_manager)
    else:
        raise ValueError(f"Unknown mode: '{args.mode}'")
