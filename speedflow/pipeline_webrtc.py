# ds_pipeline.py
import os, gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import speedflow.settings as S

Gst.init(None)

def is_file_uri(u: str) -> bool:
    return u.startswith("file://") or (os.path.isabs(u) and os.path.isfile(u))

def normalize_uri(u: str) -> str:
    return u if u.startswith("file://") or not is_file_uri(u) else "file://" + u

def build_webrtc_pipeline(rtsp_or_file_uri: str):
    uri = normalize_uri(rtsp_or_file_uri)
    is_file = is_file_uri(uri)

    pipeline = Gst.Pipeline.new("ds-webrtc")
    source = Gst.ElementFactory.make("uridecodebin", "source-bin")
    source.set_property("uri", uri)

    def on_source_setup(decodebin, src):
        if not is_file:
            for prop, val in [("latency", 100), ("drop-on-latency", True)]:
                try: src.set_property(prop, val)
                except Exception: pass
    source.connect("source-setup", on_source_setup)

    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property('batch-size', 1)
    streammux.set_property('width', int(getattr(S, "MUX_WIDTH", 1280)))
    streammux.set_property('height', int(getattr(S, "MUX_HEIGHT", 720)))
    streammux.set_property('batched-push-timeout', 40000)
    streammux.set_property('live-source', 0 if is_file else 1)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-infer")
    pgie.set_property('config-file-path', str(S.INFER_CONFIG))

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property('ll-lib-file', "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property('ll-config-file', str(S.TRACKER_CFG))
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('gpu_id', 0)

    analytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    analytics.set_property('config-file', str(S.ANALYTICS_CFG)) 

    preosd_convert = Gst.ElementFactory.make("nvvideoconvert", "preosd_convert")
    preosd_caps    = Gst.ElementFactory.make("capsfilter", "preosd_caps")
    preosd_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    nvdsosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvdsosd.set_property("display-text", 1)
    nvdsosd.set_property("display-bbox", 1)

    conv = Gst.ElementFactory.make("nvvideoconvert", "conv")
    enc = Gst.ElementFactory.make("nvv4l2h264enc", "enc")
    enc.set_property("insert-sps-pps", True)
    enc.set_property("iframeinterval", 30)
    enc.set_property("bitrate", 4_000_000)
    try: enc.set_property("maxperf-enable", True)
    except TypeError: pass

    parse = Gst.ElementFactory.make("h264parse", "parse")
    pay   = Gst.ElementFactory.make("rtph264pay", "pay")
    pay.set_property("pt", 96)
    pay.set_property("config-interval", 1)

    rtp_caps = Gst.ElementFactory.make("capsfilter", "rtp_caps")
    rtp_caps.set_property("caps", Gst.Caps.from_string(
        "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"))

    webrtc = Gst.ElementFactory.make("webrtcbin", "webrtc")
    try: webrtc.set_property("stun-server", "stun://stun.l.google.com:19302")
    except TypeError: pass

    for e in [source, streammux, pgie, tracker, analytics,
              preosd_convert, preosd_caps, nvdsosd, conv, enc, parse, pay, rtp_caps, webrtc]:
        if not e: raise RuntimeError("Failed to create a required Gst element")
        pipeline.add(e)

    def on_pad_added(decodebin, pad):
        caps = pad.get_current_caps()
        if caps and caps.to_string().startswith("video/"):
            sinkpad = streammux.get_request_pad("sink_0")
            if sinkpad and not sinkpad.is_linked():
                pad.link(sinkpad)
    source.connect("pad-added", on_pad_added)

    assert streammux.link(pgie)
    assert pgie.link(tracker)
    assert tracker.link(analytics)
    assert analytics.link(preosd_convert)
    assert preosd_convert.link(preosd_caps)
    assert preosd_caps.link(nvdsosd)
    assert nvdsosd.link(conv)
    assert conv.link(enc)
    assert enc.link(parse)
    assert parse.link(pay)
    assert pay.link(rtp_caps)

    # request pad vào webrtc
    srcpad  = rtp_caps.get_static_pad("src")
    sinkpad = webrtc.get_request_pad("sink_%u")
    print(f"[DEBUG] sinkpad: {sinkpad}")
    if sinkpad:
        link_res = srcpad.link(sinkpad)
        print(f"[DEBUG] link_res: {link_res}")
        if link_res != Gst.PadLinkReturn.OK:
             raise RuntimeError(f"Failed to link RTP to webrtcbin: {link_res}")
    else:
        raise RuntimeError("Failed to get request pad from webrtcbin")

    return pipeline, nvdsosd, webrtc
