# speedflow/core_pipeline.py  (Multi-Stream Edition)
"""
Xây dựng DeepStream pipeline hỗ trợ đa luồng (Multi-Stream).

Kiến trúc:
  N × uridecodebin ──→ nvstreammux ──→ PGIE ──→ Tracker ──→ SGIE1 ──→ SGIE2
                                                                          │
                                                                    nvdsanalytics
                                                                          │
                               ┌──────────────────────────────────────────┘
                               │
                    sink_type == "display":   nvmultistreamtiler → OSD → EGL sink
                    sink_type == "file":      OSD → nvstreamdemux → N × encoder → filesink
                    sink_type == "webrtc":    nvmultistreamtiler → OSD → webrtcbin
"""
import logging
import os

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from .common import make_element, gst_link
from .settings import (
    INFER_CONFIG, TRACKER_CFG, ANALYTICS_CFG,
    SGIE_CONFIG, TRACKER_LIB, LPR_CONFIG,
)
from .camera_config import CameraConfig, compute_tiler_layout

logger = logging.getLogger(__name__)

Gst.init(None)


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

def normalize_uri(uri: str) -> str:
    """Đảm bảo URI có scheme hợp lệ."""
    if uri.startswith(("file://", "rtsp://", "rtmp://", "http://")):
        return uri
    if os.path.exists(uri):
        return "file://" + os.path.abspath(uri)
    return uri


def is_file_uri(uri: str) -> bool:
    return uri.startswith("file://") or (
        os.path.isabs(uri) and os.path.isfile(uri)
    )


# ---------------------------------------------------------------------------
# Source bin factory
# ---------------------------------------------------------------------------

def _make_source_bin(
    pipeline: Gst.Pipeline,
    streammux: Gst.Element,
    cam_cfg: CameraConfig,
) -> Gst.Element:
    """
    Tạo một source bin cho một camera và nối vào streammux.
    Trả về element source (uridecodebin) để có thể gỡ sau.

    Quy ước tên element: "src-{camera_id}"
    """
    uri = normalize_uri(cam_cfg.uri)
    is_file = is_file_uri(uri)
    source_id = cam_cfg.source_id
    elem_name = f"src-{cam_cfg.camera_id}"

    source = make_element(elem_name, "uridecodebin")
    source.set_property("uri", uri)

    def on_source_setup(decodebin, src):
        if not is_file:
            for prop, val in [("latency", 200), ("drop-on-latency", True)]:
                try:
                    src.set_property(prop, val)
                except (TypeError, Exception):
                    pass

    source.connect("source-setup", on_source_setup)
    pipeline.add(source)

    def on_pad_added(decodebin, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or not caps.to_string().startswith("video/"):
            return
        pad_name = f"sink_{source_id}"
        sinkpad = streammux.get_request_pad(pad_name)
        if sinkpad and not sinkpad.is_linked():
            q = make_element(f"q_{cam_cfg.camera_id}", "queue")
            q.set_property("max-size-buffers", 4)
            q.set_property("leaky", 2)          # leaky downstream
            conv = make_element(f"conv_{cam_cfg.camera_id}", "nvvideoconvert")
            pipeline.add(q)
            pipeline.add(conv)
            q.sync_state_with_parent()
            conv.sync_state_with_parent()
            pad.link(q.get_static_pad("sink"))
            gst_link(q, conv)
            conv.get_static_pad("src").link(sinkpad)
            logger.info(
                "[Pipeline] Camera '%s' (source_id=%d) linked → sink_%d",
                cam_cfg.camera_id, source_id, source_id,
            )

    source.connect("pad-added", on_pad_added)
    return source


# ---------------------------------------------------------------------------
# Main pipeline builder (Multi-Stream)
# ---------------------------------------------------------------------------

def build_pipeline(
    camera_configs: list[CameraConfig],
    sink_type: str = "display",
    max_streams: int = 4,
    mux_width: int = 1920,
    mux_height: int = 1080,
    analytics_config: str = None,
    **kwargs,
):
    """
    Xây dựng DeepStream pipeline đa luồng.
    """
    if not camera_configs:
        raise ValueError("camera_configs không được rỗng.")

    n_cameras = len(camera_configs)

    if analytics_config is None:
        analytics_config = str(ANALYTICS_CFG)

    pipeline = Gst.Pipeline.new(f"ds-multi-pipeline-{sink_type}")

    # ── Muxer ────────────────────────────────────────────────────────────────
    streammux = make_element("stream-muxer", "nvstreammux")
    streammux.set_property("batch-size", n_cameras)
    streammux.set_property("width", mux_width)
    streammux.set_property("height", mux_height)
    streammux.set_property("batched-push-timeout", 33_000)
    streammux.set_property("live-source", 1)   # đa phần là RTSP live
    streammux.set_property("attach-sys-ts", True)

    # ── Core AI processing ───────────────────────────────────────────────────
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

    # ── Xác định chiến lược hiển thị / ghi file ──────────────────────────────
    is_tiled = (sink_type in ["display", "webrtc"])

    # ── Tiler (chỉ tạo nếu cần ghép lưới) ────────────────────────────────────
    if is_tiled:
        tiler = make_element("tiler", "nvmultistreamtiler")
        # Sử dụng max_streams để cố định lưới (tránh VIC scaling error khi thay đổi động)
        rows, cols = compute_tiler_layout(max_streams)
        tiler.set_property("rows", int(rows))
        tiler.set_property("columns", int(cols))
        tiler.set_property("width", mux_width)
        tiler.set_property("height", mux_height)
        tiler.set_property("gpu-id", 0)

        logger.info("[Pipeline] Tiler layout: %d×%d for %d streams", rows, cols, n_cameras)
    else:
        tiler = None

    # ── Pre-OSD convert ──────────────────────────────────────────────────────
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

    # ── Sink-specific elements & Routing ─────────────────────────────────────
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

    elif sink_type == "webrtc":
        conv = make_element("conv", "nvvideoconvert")
        enc = make_element("enc", "nvv4l2h264enc")
        enc.set_property("insert-sps-pps", True)
        enc.set_property("iframeinterval", 30)
        enc.set_property("bitrate", 6_000_000)
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
        webrtc_elem = make_element("webrtc", "webrtcbin")
        webrtc_elem.set_property(
            "stun-server", "stun://stun.l.google.com:19302"
        )
        sink_elements = [conv, enc, parse, pay, rtp_caps, webrtc_elem]

    elif sink_type == "file":
        # ── Demuxer ──
        demux = make_element("demux", "nvstreamdemux")
        pipeline.add(demux)

        for cam_cfg in camera_configs:
            if not cam_cfg.record:
                continue
            
            sid = cam_cfg.source_id
            # Thêm queue để tách biệt luồng và ổn định timestamp (PTS)
            queue = make_element(f"queue_file_{sid}", "queue")
            postosd = make_element(f"postosd_{sid}", "nvvideoconvert")
            enc = make_element(f"enc_{sid}", "nvv4l2h264enc")
            enc.set_property("bitrate", 10_000_000)
            enc.set_property("preset-level", 1)
            enc.set_property("insert-sps-pps", True)
            
            parse = make_element(f"parse_{sid}", "h264parse")
            muxer = make_element(f"mux_{sid}", "qtmux")
            # faststart giúp file mp4 có thể xem được ngay cả khi bị crash giữa chừng
            muxer.set_property("faststart", True)
            
            fsink = make_element(f"fsink_{sid}", "filesink")
            fsink.set_property("sync", False) # Thường để False cho file recording từ live source
            
            os.makedirs(os.path.dirname(os.path.abspath(cam_cfg.record_path)), exist_ok=True)
            fsink.set_property("location", os.path.abspath(cam_cfg.record_path))
            
            for el in [queue, postosd, enc, parse, muxer, fsink]:
                pipeline.add(el)
            
            gst_link(queue, postosd, enc, parse, muxer, fsink)
            
            # Ghi nhớ srcpad để nối sau khi core linking hoàn tất
            setattr(demux, f"_delayed_link_{sid}", queue)

    else:
        raise ValueError(f"Unknown sink_type: '{sink_type}'")

    # ── Add core elements to pipeline ─────────────────────────────────────────
    core_elements = [
        streammux, pgie, tracker, sgie, sgie2,
        analytics, preosd_convert, preosd_caps, nvdsosd,
    ]
    if is_tiled:
        core_elements.insert(-3, tiler)  # Thêm tiler trước preosd_convert

    for el in core_elements + sink_elements:
        pipeline.add(el)

    # ── Link core chain ───────────────────────────────────────────────────────
    if is_tiled:
        gst_link(
            streammux, pgie, tracker, sgie, sgie2,
            analytics, tiler, preosd_convert, preosd_caps, nvdsosd,
        )
    else:
        gst_link(
            streammux, pgie, tracker, sgie, sgie2,
            analytics, preosd_convert, preosd_caps, nvdsosd,
        )

    # ── Link sink chain ───────────────────────────────────────────────────────
    if sink_type == "display":
        conv, conv_caps, eglT, sink = sink_elements
        gst_link(nvdsosd, conv, conv_caps, eglT, sink)

    elif sink_type == "webrtc":
        conv, enc, parse, pay, rtp_caps, webrtc_elem = sink_elements
        gst_link(nvdsosd, conv, enc, parse, pay, rtp_caps)
        srcpad = rtp_caps.get_static_pad("src")
        sinkpad = webrtc_elem.get_request_pad("sink_%u")
        res = srcpad.link(sinkpad)
        if res != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link RTP → webrtcbin: {res}")

    elif sink_type == "file":
        # Nối OSD vào Demux
        nvdsosd.get_static_pad("src").link(demux.get_static_pad("sink"))
        
        # Nối Demux src pads ra các nhánh file riêng biệt
        for cam_cfg in camera_configs:
            if not cam_cfg.record:
                continue
            sid = cam_cfg.source_id
            postosd = getattr(demux, f"_delayed_link_{sid}")
            srcpad = demux.get_request_pad(f"src_{sid}")
            sinkpad = postosd.get_static_pad("sink")
            srcpad.link(sinkpad)

    # ── Add source bins (N cameras) ───────────────────────────────────────────
    source_bins: dict[str, Gst.Element] = {}
    for cam_cfg in camera_configs:
        src = _make_source_bin(pipeline, streammux, cam_cfg)
        source_bins[cam_cfg.camera_id] = src

    logger.info(
        "[Pipeline] Built multi-stream pipeline: %d cameras, sink=%s",
        n_cameras, sink_type,
    )

    if sink_type == "webrtc":
        return pipeline, nvdsosd, streammux, source_bins, webrtc_elem
    return pipeline, nvdsosd, streammux, source_bins


# ---------------------------------------------------------------------------
# Dynamic stream add/remove helpers (Giai đoạn 3)
# ---------------------------------------------------------------------------

def dynamic_add_stream(
    pipeline: Gst.Pipeline,
    streammux: Gst.Element,
    cam_cfg: CameraConfig,
    tiler: Gst.Element,
    source_bins: dict,
    current_n: int,
) -> Gst.Element:
    # 1. Tăng batch-size của muxer
    streammux.set_property("batch-size", current_n + 1)
    
    # 2. Thêm và chạy source mới
    src = _make_source_bin(pipeline, streammux, cam_cfg)
    src.sync_state_with_parent()

    source_bins[cam_cfg.camera_id] = src
    return src




def dynamic_remove_stream(
    pipeline: Gst.Pipeline,
    streammux: Gst.Element,
    camera_id: str,
    source_id: int,
    tiler: Gst.Element,
    source_bins: dict,
    current_n: int,
) -> None:
    src = source_bins.get(camera_id)
    if not src:
        return

    from gi.repository import GLib

    conv = pipeline.get_by_name(f"conv_{camera_id}")
    conv_src_pad = conv.get_static_pad("src") if conv else None

    def _cleanup_bin(pad, probe_id):
        # 1. Đặt state về NULL để dừng luồng dữ liệu từ gốc đến ngọn
        if src:
            src.set_state(Gst.State.NULL)
        for prefix in [f"q_{camera_id}", f"conv_{camera_id}"]:
            el = pipeline.get_by_name(prefix)
            if el:
                el.set_state(Gst.State.NULL)

        # 2. Gỡ kết nối (unlink) khỏi bộ trộn (streammux)
        mux_sinkpad = streammux.get_static_pad(f"sink_{source_id}")
        if mux_sinkpad:
            if conv_src_pad:
                # Gỡ probe block nếu nó còn tồn tại
                if pad and probe_id:
                    try:
                        pad.remove_probe(probe_id)
                    except Exception:
                        pass
                conv_src_pad.unlink(mux_sinkpad)
            streammux.release_request_pad(mux_sinkpad)

        # 3. Xóa element khỏi pipeline
        if src:
            pipeline.remove(src)
        for prefix in [f"q_{camera_id}", f"conv_{camera_id}"]:
            el = pipeline.get_by_name(prefix)
            if el:
                pipeline.remove(el)

        if camera_id in source_bins:
            del source_bins[camera_id]

        new_n = max(1, current_n - 1)
        
        # Giảm batch-size
        streammux.set_property("batch-size", new_n)
        
        # Lưu ý: Không thay đổi rows/cols của tiler để tránh VIC error trên Jetson
            
        logger.info(f"[Pipeline] Cleaned up resources for camera {camera_id}")


        return False

    def _blocking_probe(pad, info, _user_data):
        # Không remove probe ở đây để giữ trạng thái block
        GLib.idle_add(_cleanup_bin, pad, info.id)
        # DROP buffer hiện tại để tránh nó lọt qua khi unlink
        return Gst.PadProbeReturn.DROP

    if conv_src_pad:
        conv_src_pad.add_probe(
            Gst.PadProbeType.BLOCK_DOWNSTREAM, _blocking_probe, None
        )
    else:
        # Nếu không có pad, dọn dẹp luôn
        GLib.idle_add(_cleanup_bin, None, None)