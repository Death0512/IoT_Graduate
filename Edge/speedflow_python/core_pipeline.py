# speedflow/core_pipeline.py  (Multi-Stream Edition)
"""
Builds and manages a multi-stream DeepStream pipeline for Jetson Orin.

This module is the central pipeline factory and lifecycle manager. It constructs
a DeepStream GStreamer pipeline with the following architecture:

Core AI Pipeline (common to all sink types):
  N × uridecodebin ──→ nvstreammux ──→ PGIE ──→ Tracker ──→ SGIE ──→ nvdsanalytics
                                                                            │
                                ┌───────────────────────────────────────────┘
                                │
                     sink_type == "display":
                       analytics → nvmultistreamtiler → OSD (nvdsosd)
                       → nvvidconv → capsfilter → nvegltransform → nveglglessink
                       Used for local GUI display on Jetson.

                     sink_type == "file":
                       analytics → nvstreamdemux → per-stream:
                       OSD → nvvidconv → nvv4l2h264enc → h264parse → qtmux → filesink
                       Used for recording video to disk.

                     sink_type == "rtsp_push":
                       analytics → nvstreamdemux → per-stream:
                       OSD → nvvidconv → capsfilter → nvvidconv → capsfilter
                       → nvv4l2h264enc → h264parse → rtspclientsink
                       Pushes annotated streams to MediaMTX RTSP server for
                       cross-node consumption and dashboard WebRTC/HLS playback.

Key design principles:
- Permanent mux pads: nvstreammux pads are pre-created at build time (slot_capacity).
  Dynamic ADD/REMOVE reuses these pads; never destroys the mux. This avoids
  nvstreammux batch-size renegotiation which crashes on Orin.
- Source bin generations: Each dynamic ADD creates a new uridecodebin with
  generation suffix "src-{camera_id}-g{monotonic_ms}". This prevents orphan
  element name collisions from previous failed ADDs.
- NVDEC session gate: Hard limit (SPEEDFLOW_NVDEC_SESSION_LIMIT=14) enforced
  before every ADD. Exceeding this requires reboot — unrecoverable at runtime.
- RTSP reconnect: rtspsrc reconnect-delay=5 prevents camera TCP FIN from
  bubbling EOS to the pipeline (fix-8).
- EOS isolation: Per-branch conv pad probes drop spontaneous EOS before mux
  so one camera's stream end doesn't kill the shared multi-stream pipeline.
- Teardown ordering: Strict sequential PAUSED→READY→NULL with get_state waits.
  Direct PLAYING→NULL on nvv4l2decoder deadlocks Tegra v4l2 kernel driver.

This file provides:
- build_pipeline(): Main entry point, returns (pipeline, osd, streammux, source_bins)
- dynamic_add_stream() / dynamic_remove_stream(): Runtime camera lifecycle
- _make_source_bin(): Creates uridecodebin + queue + nvvideoconvert + mux link
- RTSP push / file recording branch management
- Teardown utilities with bounded state walks
"""
import logging
import os
import threading
import time
from typing import Optional

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

from .common import make_element, gst_link, is_file_uri
from .settings import (
    INFER_CONFIG, TRACKER_CFG, ANALYTICS_CFG,
    SGIE_CONFIG, TRACKER_LIB,
    SPEEDFLOW_SLOT_CAPACITY, SPEEDFLOW_NVDEC_SESSION_LIMIT,
    RTSP_PUSH_BITRATE,
)
from .camera_config import CameraConfig, compute_tiler_layout

logger = logging.getLogger(__name__)

# Initialize GStreamer
Gst.init(None)

# ---------------------------------------------------------------------------
# Global state for pipeline lifecycle management
# ---------------------------------------------------------------------------

# Track abandoned rtsp_push elements left parented in pipeline after bounded
# teardown (wedged rtspclientsink). Reaped across subsequent _remove calls to
# avoid leaking NVENC sessions over days of migration churn.
# Keyed by pipeline_id to support multiple pipeline instances.
# Max 64 entries to bound memory.
_ABANDONED_PUSH_ELEMENTS: set[str] = set()
_ABANDONED_PUSH_MAX = 64


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

def normalize_uri(uri: str) -> str:
    """Ensure the URI has a valid scheme for GStreamer.

    Args:
        uri: Raw URI string from camera config. Can be:
            - 'file:///path' (already valid)
            - 'rtsp://...' (already valid)
            - 'rtmp://...' / 'http://...' (already valid)
            - '/absolute/path' (local file, converted to file://)
            - 'relative/path' (returned as-is, will fail at runtime)

    Returns:
        URI with proper scheme prefix for GStreamer uridecodebin.
    """
    if uri.startswith(("file://", "rtsp://", "rtmp://", "http://")):
        return uri
    if os.path.exists(uri):
        return "file://" + os.path.abspath(uri)
    return uri


# is_file_uri imported from .common — single source of truth; no local redefinition here.

# ---------------------------------------------------------------------------
# Source bin factory
# ---------------------------------------------------------------------------

def _make_source_bin(
    pipeline: Gst.Pipeline,
    streammux: Gst.Element,
    cam_cfg: CameraConfig,
    ready_event: Optional[threading.Event] = None,
    reconnect=None,
) -> Gst.Element:
    """
    Create a source bin for one camera and connect it to streammux.
    This is the core dynamic camera ADD primitive.

    Builds the following sub-pipeline:
      uridecodebin (source) → queue → nvvideoconvert → nvstreammux sink_{source_id}

    The bin is named with generation suffix to prevent orphan collisions:
      "src-{camera_id}-g{generation}" where generation = process-monotonic ms.
    Static names caused the orphan-element collision class: a wedged leftover
    parented under the old name blocked every re-add forever. Generations can
    never collide (monotonic ms), and the reaper in dynamic_add_stream
    garbage-collects unmapped generations.

    Args:
        pipeline: The Gst.Pipeline to add elements to.
        streammux: The nvstreammux element with pre-created sink pads.
        cam_cfg: CameraConfig with camera_id, source_id, uri, record flag.
        ready_event: Optional threading.Event set when first buffer reaches OSD
                     (signals ADD acknowledgement to caller).
        reconnect: Optional callable(cfg) for watchdog retry on pad-add stall.

    Returns:
        The uridecodebin source element (so caller can track/remove it).
    """
    uri = normalize_uri(cam_cfg.uri)
    is_file = is_file_uri(uri)
    source_id = cam_cfg.source_id
    # Generation suffix prevents orphan-element name collision on re-ADD.
    # Static name "src-{camera_id}" causes silent stall: GStreamer element registry
    # retains the name from the torn-down generation while the new bin is being added,
    # so source-setup and pad-added never fire. Monotonic ms guarantees uniqueness.
    # Same generation is reused for q and conv so teardown can locate them via
    # src._q_name / src._conv_name instead of a fragile static get_by_name (R2/A3).
    _gen = int(time.monotonic() * 1000)
    elem_name = f"src-{cam_cfg.camera_id}-g{_gen}"
    _q_name   = f"q-{cam_cfg.camera_id}-g{_gen}"
    _conv_name = f"conv-{cam_cfg.camera_id}-g{_gen}"

    source = make_element(elem_name, "uridecodebin")
    source.set_property("uri", uri)

    def on_source_setup(decodebin, src):
        """Configure rtspsrc properties when uridecodebin creates the source.

        RTSP-specific properties for robust camera connectivity:
        - latency: 200ms buffer for jitter
        - drop-on-latency: Drop late buffers instead of queueing
        - protocols: 0x4 = GST_RTSP_LOWER_TRANS_TCP (TCP transport)
        - retry: 2 connection retries
        - timeout: 6s in μs — bounded under 12s ADD ack window
        - tcp-timeout: 6s CLOSE-WAIT stall bound; retry budget under ack window
        - ntp-sync: False — Disable NTP sync; prevents jitterbuffer RTCP-SR starve
        - do-rtcp: True — Enable RTCP for keepalive; prevents MediaMTX 60s timeout
        """
        if not is_file:
            for prop, val in [
                ("latency", 200),
                ("drop-on-latency", True),
                ("protocols", 0x4),  # rtspsrc TCP transport (GST_RTSP_LOWER_TRANS_TCP)
                ("retry", 2),
                ("timeout", 6_000_000),  # 6s in μs — bounded under 12s ADD ack window
                ("tcp-timeout", 6_000_000),  # bound CLOSE-WAIT stall to 6s; retry budget under ack window
                ("ntp-sync", False),
                ("do-rtcp", True),
            ]:
                try:
                    src.set_property(prop, val)
                except (TypeError, Exception) as _prop_exc:
                    logger.debug(
                        "[Pipeline] rtspsrc property '%s' not supported on %s: %s",
                        prop, src.get_name(), _prop_exc,
                    )

    source.connect("source-setup", on_source_setup)
    pipeline.add(source)

    def on_pad_added(decodebin, pad):
        """Callback when uridecodebin exposes a new pad (video stream).

        Links the decodebin video pad through queue → nvvideoconvert → streammux.
        Only handles video/ caps. Non-video pads are ignored.

        Key design:
        - Mux sink pads are pre-created at build time and NEVER released (#596).
        - A pad still linked here can only be our black filler from previous REMOVE.
        - Detach filler first, then link real branch into the same permanent pad.
        """
        try:
            caps = pad.get_current_caps() or pad.query_caps(None)
            caps_str = caps.to_string() if caps else "<no-caps>"
            logger.info(
                "[Pipeline] pad-added for camera '%s' (source_id=%d): pad='%s' caps='%s'",
                cam_cfg.camera_id, source_id, pad.get_name(), caps_str[:160],
            )
            if not caps or not caps.to_string().startswith("video/"):
                return
            pad_name = f"sink_{source_id}"
            # Mux sink pads are pre-created once at build time and are NEVER
            # requested/released post-init (#596 crash class). A pad still linked
            # here can only be owned by our own black filler left by a previous
            # REMOVE — detach it, then link the real branch into the same pad.
            sinkpad = streammux.get_static_pad(pad_name)
            if sinkpad is None:
                logger.error(
                    "[Pipeline] No permanent mux pad '%s' for camera '%s' "
                    "(source_id=%d beyond slot capacity); ADD aborted.",
                    pad_name, cam_cfg.camera_id, source_id,
                )
                return
            _detach_filler_from_pad(pipeline, streammux, source_id)
            if not sinkpad.is_linked():
                # Use generation-suffixed names (R2/A3 fix): static q_/conv_ names
                # cause orphan collision when teardown raises mid-way and elements
                # remain parented. Next ADD's get_by_name() finds the orphan and
                # tears down the wrong object → real branch leaks NVDEC session.
                # Names stored on source element so _teardown_source_branch can
                # look them up via pad graph without fragile static get_by_name.
                q = make_element(_q_name, "queue")
                q.set_property("max-size-buffers", 4)
                q.set_property("leaky", 2)          # Leaky downstream (drop old if full)
                conv = make_element(_conv_name, "nvvideoconvert")
                source._q_name = _q_name
                source._conv_name = _conv_name
                pipeline.add(q)
                pipeline.add(conv)
                q.sync_state_with_parent()
                conv.sync_state_with_parent()

                # ponytail: no BUFFER probe here. Input FPS is counted from the
                # same OSD sink-pad counter as output FPS (SpeedProbe._fps_frame_count),
                # so both share the same writer telemetry window — no independent
                # source probe to burst.
                # Gst.Pad.link returns PadLinkReturn (never raises): an unchecked
                # failure here logs a lying "linked" line and starves for 20s
                # with zero diagnosis (seen live on B/C 2026-09-18). Fail loud.
                _link_pads(pad, q.get_static_pad("sink"), cam_cfg.camera_id, source_id)
                gst_link(q, conv)
                conv_src_pad = conv.get_static_pad("src")
                def _drop_live_eos(pad, info):
                    event = info.get_event()
                    if event is not None and event.type == Gst.EventType.EOS:
                        logger.warning(
                            "[Pipeline] Dropped spontaneous EOS before mux for camera '%s' (source_id=%d)",
                            cam_cfg.camera_id, source_id,
                        )
                        return Gst.PadProbeReturn.DROP
                    return Gst.PadProbeReturn.OK
                conv_src_pad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, _drop_live_eos)
                if ready_event is not None:
                    def _first_buffer(pad, info, event=ready_event):
                        event.set()
                        return Gst.PadProbeReturn.REMOVE
                    conv_src_pad.add_probe(Gst.PadProbeType.BUFFER, _first_buffer)
                _link_pads(conv_src_pad, sinkpad, cam_cfg.camera_id, source_id)

                logger.info(
                    "[Pipeline] Camera '%s' (source_id=%d) linked → sink_%d",
                    cam_cfg.camera_id, source_id, source_id,
                )
            else:
                # Reclaim/failover ADD otherwise spins forever (ack timeout →
                # REMOVE → retry) with zero diagnosis. No force-unlink here —
                # unlinking a live pad while PLAYING is a #596 crash class.
                peer = sinkpad.get_peer()
                logger.error(
                    "[Pipeline] ADD blocked for camera '%s' (source_id=%d): mux pad '%s' still linked by '%s'.",
                    cam_cfg.camera_id, source_id, pad_name,
                    peer.get_name() if peer is not None else "unknown",
                )
        except Exception:
            logger.exception(
                "[Pipeline] on_pad_added failed for camera '%s' (source_id=%d)",
                cam_cfg.camera_id, source_id,
            )

    source.connect("pad-added", on_pad_added)
    return source


def _resolve_rtsp_push_location(
    rtsp_push_base_url: str,
    camera_id: str,
    node_camera_map: Optional[dict] = None,
) -> str:
    """
    ponytail: build RTSP push location keyed by camera owner identity.
    Preserves server dashboard WHEP mapping '{owner_node}/{camera_id}' across migrations.
    If camera_id belongs to jetson_B but is running on jetson_A, pushes to '.../jetson_B/cam_03'.
    """
    clean_base = rtsp_push_base_url.rstrip('/')
    if not node_camera_map or not isinstance(node_camera_map, dict):
        return f"{clean_base}/{camera_id}"

    # Find owner node in node_camera_map: {node_name: [cam_ids...]}
    owner = None
    for n_id, cams in node_camera_map.items():
        if isinstance(cams, list) and camera_id in cams:
            owner = n_id
            break

    if not owner:
        return f"{clean_base}/{camera_id}"

    # Extract host prefix if clean_base ends with a node identifier
    # e.g., 'rtsp://116.118.9.125:8554/jetson_A' -> 'rtsp://116.118.9.125:8554'
    parts = clean_base.rsplit('/', 1)
    if len(parts) == 2 and any(parts[1] == k for k in node_camera_map.keys()):
        host_prefix = parts[0]
        return f"{host_prefix}/{owner}/{camera_id}"

    return f"{clean_base}/{camera_id}"


def _add_rtsp_push_branch(
    pipeline: Gst.Pipeline,
    demux: Gst.Element,
    cam_cfg: CameraConfig,
    rtsp_push_base_url: str,
    bitrate: int = RTSP_PUSH_BITRATE,
    sync: bool = False,
    node_camera_map: Optional[dict] = None,
) -> list[Gst.Element]:
    """Create one nvstreamdemux -> queue -> osd -> encoder -> rtspclientsink branch for cam_cfg."""
    sid = cam_cfg.source_id
    suffix = f"_{sid}"

    # Clean up any stale RTSP push branch for this slot before adding fresh branch
    branch_names = [
        f"queue_rtsp_push{suffix}",
        f"queue_osd_rtsp_push{suffix}",
        f"osd_convert_rtsp_push{suffix}",
        f"osd_caps_rtsp_push{suffix}",
        f"osd_rtsp_push{suffix}",
        f"conv_rtsp_push{suffix}",
        f"caps_rtsp_push{suffix}",
        f"enc_rtsp_push{suffix}",
        f"parse_rtsp_push{suffix}",
        f"sink_rtsp_push{suffix}",
    ]
    if any(pipeline.get_by_name(name) is not None for name in branch_names):
        logger.info(
            "[Pipeline] Cleaning up stale RTSP push branch for '%s' (source_id=%d) before adding fresh branch",
            cam_cfg.camera_id, sid,
        )
        _remove_rtsp_push_branch(pipeline, sid)

    queue_osd = make_element(f"queue_osd_rtsp_push{suffix}", "queue")
    queue_osd.set_property("max-size-buffers", 30)
    osd_convert = make_element(f"osd_convert_rtsp_push{suffix}", "nvvideoconvert")
    osd_caps = make_element(f"osd_caps_rtsp_push{suffix}", "capsfilter")
    osd_caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )
    osd = make_element(f"osd_rtsp_push{suffix}", "nvdsosd")
    osd.set_property("display-text", 1)
    osd.set_property("display-bbox", 1)
    osd.set_property("process-mode", 2)
    osd.set_property("gpu-id", 0)

    conv = make_element(f"conv_rtsp_push{suffix}", "nvvideoconvert")
    caps = make_element(f"caps_rtsp_push{suffix}", "capsfilter")
    caps.set_property(
        "caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=NV12, width=1280, height=720"
        )
    )
    enc = make_element(f"enc_rtsp_push{suffix}", "nvv4l2h264enc")
    enc.set_property("bitrate", bitrate)  # default 750 kbps per camera
    enc.set_property("iframeinterval", 30)
    enc.set_property("insert-sps-pps", True)
    try:
        enc.set_property("maxperf-enable", True)
    except (TypeError, Exception):
        pass

    parse = make_element(f"parse_rtsp_push{suffix}", "h264parse")
    sink = make_element(f"sink_rtsp_push{suffix}", "rtspclientsink")
    location = _resolve_rtsp_push_location(rtsp_push_base_url, cam_cfg.camera_id, node_camera_map)
    sink.set_property("location", location)
    sink.set_property("protocols", "tcp")
    sink.set_property("latency", 0)

    elements = [queue_osd, osd_convert, osd_caps, osd, conv, caps, enc, parse, sink]
    for el in elements:
        try:
            ret = pipeline.add(el)
            if ret is False:
                raise RuntimeError(
                    f"Failed to add '{el.get_name()}' to pipeline (orphan element collision)"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to add '{el.get_name()}' to pipeline (orphan element collision): {e}"
            ) from e

    try:
        # Link all downstream chain from queue_osd (linked to demux below)
        gst_link(queue_osd, osd_convert, osd_caps, osd, conv, caps, enc, parse, sink)

        # Demux request pads are permanent; get pre-created static pad
        srcpad = demux.get_static_pad(f"src_{sid}")
        sinkpad = queue_osd.get_static_pad("sink")

        if srcpad is None:
            raise RuntimeError(
                f"Permanent demux src pad 'src_{sid}' not found for camera '{cam_cfg.camera_id}' (source_id={sid})"
            )
        if sinkpad is None:
            raise RuntimeError(
                f"Queue sink pad not found for camera '{cam_cfg.camera_id}' (source_id={sid})"
            )

        if hasattr(srcpad, "get_direction") and hasattr(Gst, "PadDirection"):
            if srcpad.get_direction() != Gst.PadDirection.SRC:
                raise RuntimeError(f"Demux pad for source_id={sid} is not a SRC pad")
        if hasattr(sinkpad, "get_direction") and hasattr(Gst, "PadDirection"):
            if sinkpad.get_direction() != Gst.PadDirection.SINK:
                raise RuntimeError(f"Queue sink pad for source_id={sid} is not a SINK pad")

        link_ret = srcpad.link(sinkpad)
        ok_val = getattr(getattr(Gst, "PadLinkReturn", None), "OK", 0)
        if link_ret is not None and link_ret != ok_val and link_ret is not True:
            link_nick = getattr(link_ret, "value_nick", str(link_ret))
            logger.error(
                "[Pipeline] Failed to link demux pad to RTSP push queue for camera '%s' (source_id=%d): return=%s",
                cam_cfg.camera_id, sid, link_nick,
            )
            raise RuntimeError(
                f"Failed to link demux pad to RTSP push queue for camera '{cam_cfg.camera_id}' (source_id={sid}): return={link_nick}"
            )

        if sync:
            # Use sync_state_with_parent() — NOT _set_state_bounded() here.
            # _set_state_bounded() offloads set_state to a worker thread but
            # calls done.wait() on the CALLER — which is the GLib main thread
            # (dynamic_add_stream runs via idle_add, camera_config.py:705).
            # Blocking the GLib main thread after src.sync_state_with_parent()
            # prevents rtspsrc's source-setup / pad-added signals (emitted from
            # rtspsrc's own streaming thread, marshalled through the GLib main
            # context by PyGObject) from being dispatched — rtspsrc stays stuck
            # PAUSED forever (root cause R6, confirmed 2026-09-22).
            #
            # sync_state_with_parent() is the correct async primitive: it kicks
            # each element toward the pipeline's current state (PLAYING) and
            # returns immediately. GStreamer's state machine handles the
            # PLAYING transition in the element's own streaming thread without
            # blocking the GLib main thread. rtspclientsink's TCP connect
            # happens in its own context; we observe completion via bus messages
            # rather than blocking the caller.
            for el in elements:
                el.sync_state_with_parent()

        return elements
    except Exception as exc:
        logger.error(
            "[Pipeline] Error setting up RTSP push branch for '%s' (source_id=%d): %s",
            cam_cfg.camera_id, sid, exc,
        )
        _remove_rtsp_push_branch(pipeline, sid)
        raise


def _set_state_bounded(el: Gst.Element, target_state: Gst.State, timeout_s: float = 8.0) -> bool:
    """set_state + get_state with a hard wall-clock bound.

    rtspclientsink.set_state can block forever inside C on a wedged TCP
    connection to the RTSP server, with no internal timeout. An unbounded
    call wedges the GLib main thread and kills every future ADD/REMOVE on
    the node (seen live on jetson_C 2026-09-17: faulthandler pinned the GLib
    thread at _remove_rtsp_push_branch set_state for 25+ min). Run the walk
    in a helper thread; on timeout abandon the element (left parented) so
    the teardown completes and the node stays operable. The abandoned sink
    may leak one session; the NVDEC gate still caps the ceiling.
    Returns True if the element reached the target state, False if abandoned.
    Raises RuntimeError on synchronous FAILURE (existing teardown semantics).
    """
    done = threading.Event()
    outcome: list = []

    def _walk() -> None:
        try:
            el.set_state(target_state)
            state_ret, current_state, _ = el.get_state(1 * Gst.SECOND)
            if state_ret == Gst.StateChangeReturn.ASYNC:
                state_ret, current_state, _ = el.get_state(5 * Gst.SECOND)
            outcome.append((state_ret, current_state))
        except Exception as exc:  # noqa: BLE001 — recorded, not raised across threads
            outcome.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=_walk, daemon=True)
    t.start()
    if not done.wait(timeout=timeout_s):
        target_nick = getattr(target_state, "value_nick", str(target_state))
        logger.critical(
            "[Pipeline] Abandoning '%s': set_state(%s) unresolved after %.0fs "
            "(wedged sink suspected) — teardown continues without it.",
            el.get_name(), target_nick, timeout_s,
        )
        return False
    result = outcome[0] if outcome else None
    if isinstance(result, Exception):
        raise result
    state_ret, current_state = result
    if state_ret == Gst.StateChangeReturn.FAILURE:
        target_nick = getattr(target_state, "value_nick", str(target_state))
        curr_nick = getattr(current_state, "value_nick", str(current_state))
        raise RuntimeError(
            f"RTSP push element {el.get_name()} failed to reach {target_nick}: state={curr_nick}"
        )
    if state_ret == Gst.StateChangeReturn.ASYNC:
        target_nick = getattr(target_state, "value_nick", str(target_state))
        curr_nick = getattr(current_state, "value_nick", str(current_state))
        raise RuntimeError(
            f"RTSP push element {el.get_name()} ASYNC teardown unresolved after 6s "
            f"(target={target_nick}, state={curr_nick}); refusing pipeline.remove() "
            f"to prevent TSG orphan / NVDEC session leak"
        )
    return True


def _remove_rtsp_push_branch(pipeline: Gst.Pipeline, source_id: int) -> None:
    # Reaper: try to clean up previously-abandoned push elements from this
    # pipeline. Never raises; per-element try/except, debug log on still-wedged.
    pid = id(pipeline)
    to_reap = [n for n in _ABANDONED_PUSH_ELEMENTS if n.startswith(f"p{pid}_")]
    reaped = 0
    for full_name in to_reap:
        ename = full_name.split("_", 1)[1]  # strip "p{pid}_" prefix
        try:
            el = pipeline.get_by_name(ename)
            if el is None:
                continue
            for pad in (el.get_static_pad("sink"),):
                if pad is not None and pad.is_linked():
                    peer = pad.get_peer() if hasattr(pad, "get_peer") else None
                    if peer is not None:
                        # GStreamer unlink requires srcpad.unlink(sinkpad)
                        peer.unlink(pad)
            if _set_state_bounded(el, Gst.State.NULL, timeout_s=4.0):
                pipeline.remove(el)
                _ABANDONED_PUSH_ELEMENTS.discard(full_name)
                reaped += 1
            else:
                logger.debug(
                    "[Pipeline] Reaper: '%s' still wedged, keeping in abandoned set.", ename
                )
        except Exception:
            logger.debug("[Pipeline] Reaper error on '%s', keeping in abandoned set.", ename)
    if reaped:
        logger.info("[Pipeline] Reaped %d previously-abandoned RTSP push element(s) for pipeline_id=%d.", reaped, pid)

    demux = pipeline.get_by_name("demux")
    suffix = f"_{source_id}"
    names = [
        f"queue_rtsp_push{suffix}",
        f"queue_osd_rtsp_push{suffix}",
        f"osd_convert_rtsp_push{suffix}",
        f"osd_caps_rtsp_push{suffix}",
        f"osd_rtsp_push{suffix}",
        f"conv_rtsp_push{suffix}",
        f"caps_rtsp_push{suffix}",
        f"enc_rtsp_push{suffix}",
        f"parse_rtsp_push{suffix}",
        f"sink_rtsp_push{suffix}",
    ]
    elements = [pipeline.get_by_name(n) for n in names if pipeline.get_by_name(n) is not None]
    if not elements:
        return

    # 1. EOS-drain the branch head BEFORE unlinking the demux srcpad.
    # Without this, nvv4l2h264enc holds in-flight V4L2 buffers that were pushed
    # by the live pipeline at 25fps and cannot reclaim them on PAUSED→NULL,
    # causing every queue/osd/conv/enc element to time out and get abandoned.
    # Sending EOS into queue_osd (the branch head, downstream of the demux tee
    # point) flushes the encoder's V4L2 STREAMOFF path cleanly. The EOS is
    # confined to this branch and cannot reach nvstreamdemux or cam_05.
    # (Same pattern as the proven _teardown_source_branch EOS-drain above.)
    queue_osd = pipeline.get_by_name(f"queue_osd_rtsp_push{suffix}")
    if queue_osd is not None:
        # Install EOS-catch probe on parse_rtsp_push (just before rtspclientsink) so
        # we wait until nvv4l2h264enc has fully flushed its V4L2 encode queue — not
        # just until queue_osd has forwarded the EOS, which would be too early.
        _drained = threading.Event()

        def _eos_probe(pad, info, _drained=_drained):
            if info.get_event() is not None and info.get_event().type == Gst.EventType.EOS:
                _drained.set()
                return Gst.PadProbeReturn.DROP
            return Gst.PadProbeReturn.OK

        parse_el = pipeline.get_by_name(f"parse_rtsp_push{suffix}")
        probe_pad = parse_el.get_static_pad("src") if parse_el else None
        probe_id = probe_pad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, _eos_probe) if probe_pad else None
        queue_osd.send_event(Gst.Event.new_eos())
        _drained.wait(timeout=2.0)
        if probe_pad and probe_id:
            probe_pad.remove_probe(probe_id)

    # 2. Unlink from nvstreamdemux (idempotent, demux request pad is permanent and never released)
    sinkpad = queue_osd.get_static_pad("sink") if queue_osd else None
    demux_srcpad = demux.get_static_pad(f"src_{source_id}") if demux else None

    if sinkpad and sinkpad.is_linked():
        peer = sinkpad.get_peer() if hasattr(sinkpad, "get_peer") else None
        if peer:
            peer.unlink(sinkpad)
    elif demux_srcpad and demux_srcpad.is_linked():
        peer = demux_srcpad.get_peer() if hasattr(demux_srcpad, "get_peer") else None
        if peer:
            demux_srcpad.unlink(peer)

    # 2. Sequential teardown PLAYING -> PAUSED -> READY -> NULL with bounded
    # waits. Each element walk is wall-clock bounded (_set_state_bounded): a
    # wedged rtspclientsink is abandoned instead of wedging the GLib thread.
    # Skipping ASYNC (only checking FAILURE) leaves nvv4l2decoder TSG unbind in-flight,
    # which orphans the TSG, accumulates NVDEC sessions, and causes AXI stall / RCU hang.
    abandoned: list[str] = []
    for target_state in (Gst.State.PAUSED, Gst.State.READY, Gst.State.NULL):
        for el in elements:
            if el.get_name() in abandoned:
                continue
            if not _set_state_bounded(el, target_state):
                abandoned.append(el.get_name())

    # 3. Remove elements from pipeline (abandoned wedged sinks stay parented).
    for el in elements:
        if el.get_name() in abandoned:
            continue
        pipeline.remove(el)
    if abandoned:
        logger.critical(
            "[Pipeline] RTSP push teardown for source_id=%d left %d abandoned element(s): %s.",
            source_id, len(abandoned), abandoned,
        )
        # Track newly-abandoned elements for later reaping. Prefix with pipeline id
        # to scope to this pipeline instance; cap set size to bound memory.
        for ename in abandoned:
            _ABANDONED_PUSH_ELEMENTS.add(f"p{pid}_{ename}")
        if len(_ABANDONED_PUSH_ELEMENTS) > _ABANDONED_PUSH_MAX:
            # Drop oldest (arbitrary order) to cap
            excess = len(_ABANDONED_PUSH_ELEMENTS) - _ABANDONED_PUSH_MAX
            for _ in range(excess):
                _ABANDONED_PUSH_ELEMENTS.pop()

    # 4. Unconditional final sweep pass: ensure all 10 elements in push-branch chain reach NULL + pipeline.remove
    # even when rtspclientsink or other elements wedged. Unlink remaining pads, force NULL,
    # and remove from pipeline so no orphan elements linger.
    swept = 0
    for name in names:
        el = pipeline.get_by_name(name)
        if el is None:
            continue
        # Unlink only sink pad to silence GST_PAD_IS_SRC noise
        sink_pad = el.get_static_pad("sink")
        if sink_pad is not None and sink_pad.is_linked():
            peer = sink_pad.get_peer() if hasattr(sink_pad, "get_peer") else None
            if peer is not None:
                try:
                    sink_pad.unlink(peer)
                except Exception:
                    pass
        if _set_state_bounded(el, Gst.State.NULL, timeout_s=4.0):
            try:
                pipeline.remove(el)
                swept += 1
            except Exception:
                pass
        else:
            _ABANDONED_PUSH_ELEMENTS.add(f"p{pid}_{name}")
            if len(_ABANDONED_PUSH_ELEMENTS) > _ABANDONED_PUSH_MAX:
                excess = len(_ABANDONED_PUSH_ELEMENTS) - _ABANDONED_PUSH_MAX
                for _ in range(excess):
                    _ABANDONED_PUSH_ELEMENTS.pop()
    if swept:
        logger.info(
            "[Pipeline] Swept %d leftover RTSP push element(s) for source_id=%d in final teardown pass",
            swept, source_id,
        )


def init_rtsp_push_branches(
    pipeline: Gst.Pipeline,
    demux: Gst.Element,
    present_cameras: list[CameraConfig],
    rtsp_push_base_url: str,
    bitrate: int = 750_000,
    node_camera_map: Optional[dict] = None,
) -> None:
    """Initialize RTSP push branches for all enabled initial cameras."""
    for cam_cfg in present_cameras:
        if getattr(cam_cfg, "enabled", True):
            _add_rtsp_push_branch(
                pipeline, demux, cam_cfg, rtsp_push_base_url, bitrate=bitrate, sync=True, node_camera_map=node_camera_map
            )


def _add_file_recording_branch(
    pipeline: Gst.Pipeline,
    demux: Gst.Element,
    cam_cfg: CameraConfig,
    sync: bool = False,
) -> None:
    """Create one nvstreamdemux → queue -> osd -> encoder → filesink branch for cam_cfg."""
    if not cam_cfg.record:
        return

    sid = cam_cfg.source_id
    suffix = f"_{sid}"
    if pipeline.get_by_name(f"queue_osd_file_{sid}"):
        _remove_file_recording_branch(pipeline, sid)

    queue_osd = make_element(f"queue_osd_file{suffix}", "queue")
    queue_osd.set_property("max-size-buffers", 30)
    osd_convert = make_element(f"osd_convert_file{suffix}", "nvvideoconvert")
    osd_caps = make_element(f"osd_caps_file{suffix}", "capsfilter")
    osd_caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )
    osd = make_element(f"osd_file{suffix}", "nvdsosd")
    osd.set_property("display-text", 1)
    osd.set_property("display-bbox", 1)
    osd.set_property("process-mode", 2)
    osd.set_property("gpu-id", 0)

    postosd = make_element(f"postosd_{sid}", "nvvideoconvert")
    enc = make_element(f"enc_{sid}", "nvv4l2h264enc")
    enc.set_property("bitrate", 10_000_000)
    enc.set_property("preset-level", 1)
    enc.set_property("insert-sps-pps", True)

    parse = make_element(f"parse_{sid}", "h264parse")
    muxer = make_element(f"mux_{sid}", "qtmux")
    # faststart allows MP4 file to be viewable even if it crashes midway
    muxer.set_property("faststart", True)

    fsink = make_element(f"fsink_{sid}", "filesink")
    fsink.set_property("sync", False)
    os.makedirs(os.path.dirname(os.path.abspath(cam_cfg.record_path)), exist_ok=True)
    fsink.set_property("location", os.path.abspath(cam_cfg.record_path))

    elements = [queue_osd, osd_convert, osd_caps, osd, postosd, enc, parse, muxer, fsink]
    for el in elements:
        pipeline.add(el)

    gst_link(queue_osd, osd_convert, osd_caps, osd, postosd, enc, parse, muxer, fsink)

    srcpad = demux.get_static_pad(f"src_{sid}")
    sinkpad = queue_osd.get_static_pad("sink")
    if srcpad and sinkpad and not sinkpad.is_linked():
        srcpad.link(sinkpad)

    if sync:
        for el in elements:
            el.sync_state_with_parent()


def _remove_file_recording_branch(pipeline: Gst.Pipeline, source_id: int) -> None:
    demux = pipeline.get_by_name("demux")
    suffix = f"_{source_id}"
    names = [
        f"queue_osd_file{suffix}",
        f"osd_convert_file{suffix}",
        f"osd_caps_file{suffix}",
        f"osd_file{suffix}",
        f"postosd_{source_id}",
        f"enc_{source_id}",
        f"parse_{source_id}",
        f"mux_{source_id}",
        f"fsink_{source_id}",
    ]
    elements = [pipeline.get_by_name(n) for n in names if pipeline.get_by_name(n) is not None]
    if not elements:
        return

    # 1. Unlink from nvstreamdemux (idempotent, demux request pad is permanent and never released)
    queue_osd = pipeline.get_by_name(f"queue_osd_file{suffix}")
    sinkpad = queue_osd.get_static_pad("sink") if queue_osd else None
    demux_srcpad = demux.get_static_pad(f"src_{source_id}") if demux else None

    if sinkpad and sinkpad.is_linked():
        peer = sinkpad.get_peer() if hasattr(sinkpad, "get_peer") else None
        if peer:
            peer.unlink(sinkpad)
    elif demux_srcpad and demux_srcpad.is_linked():
        peer = demux_srcpad.get_peer() if hasattr(demux_srcpad, "get_peer") else None
        if peer:
            demux_srcpad.unlink(peer)

    # 2. Sequential teardown PLAYING -> PAUSED -> READY -> NULL with get_state waits
    for target_state in (Gst.State.PAUSED, Gst.State.READY, Gst.State.NULL):
        for el in elements:
            el.set_state(target_state)
        for el in elements:
            try:
                el.get_state(1 * Gst.SECOND)
            except Exception:
                pass

    for el in elements:
        pipeline.remove(el)


# ---------------------------------------------------------------------------
# Main pipeline builder (Multi-Stream)
# ---------------------------------------------------------------------------

def build_pipeline(
    camera_configs: list[CameraConfig],
    sink_type: str = "display",
    mux_width: int = 1920,
    mux_height: int = 1080,
    analytics_config: Optional[str] = None,
    slot_capacity: Optional[int] = None,
    **kwargs,
):
    """
    Build a multi-stream DeepStream pipeline.
    """
    if not camera_configs:
        raise ValueError("camera_configs must not be empty.")

    n_cameras = len(camera_configs)

    if analytics_config is None:
        analytics_config = str(ANALYTICS_CFG)

    if slot_capacity is None:
        # Deployment-wide source_id universe, NOT this node's own camera
        # count: any camera may arrive here via migration/failover carrying
        # its original source_id (e.g. sids 4-5 landing on a 2-camera node).
        # Idle request pads are inert (no branch linked, batch-size excludes
        # them), so a generous bound costs nothing at runtime.
        slot_capacity = SPEEDFLOW_SLOT_CAPACITY
    slot_capacity = max(int(slot_capacity), n_cameras)

    pipeline = Gst.Pipeline.new(f"ds-multi-pipeline-{sink_type}")

    # ── Muxer ────────────────────────────────────────────────────────────────
    streammux = make_element("stream-muxer", "nvstreammux")
    # Set batch-size to slot_capacity (not n_cameras) so the NvBufSurface pool
    # is allocated for the maximum possible concurrent streams at build time.
    # Dynamic ADD mutates batch-size at runtime AFTER the pool is frozen →
    # decodebin never prerolls → on_pad_added never fires → nvdec 1→1 timeout.
    # Pool size is fixed at NULL→READY; resizing post-PLAYING is a no-op on Orin.
    streammux.set_property("batch-size", slot_capacity)
    streammux.set_property("width", mux_width)
    streammux.set_property("height", mux_height)
    streammux.set_property("batched-push-timeout", 33_000)
    # live-source=1 (arrival-rate push) is correct for live RTSP sources,
    # but for pure file playback it lets the muxer run at decode speed, so
    # output FPS can exceed the source file's native FPS.  live-source=0
    # paces the muxer to the sources' PTS → realtime playback (output FPS
    # ≤ source FPS).  Mixed live+file pipelines must stay 1 (a live source
    # would stall under PTS pacing); probes.py telemetry exposes the
    # decision as muxer_live_source so downstream can interpret FPS.
    live_source = (
        0 if all(is_file_uri(normalize_uri(c.uri)) for c in camera_configs)
        else 1
    )
    streammux.set_property("live-source", live_source)
    streammux.set_property("attach-sys-ts", True)

    # ── Permanent mux sink pads (crash-class fix) ─────────────────────────────
    # Every possible slot's request pad is created ONCE here, before the
    # pipeline ever runs, and is never released while the process lives.
    # Dynamic ADD/REMOVE only swaps the upstream branch linked into an
    # existing pad. Pad request/release on a PLAYING nvstreammux can corrupt
    # NvBufSurfacePool and wedge the Tegra kernel (#596), so post-init pad
    # churn is eliminated by construction. batch-size still tracks the number
    # of active branches exactly as before (unchanged GPU economics).
    for sid in range(slot_capacity):
        streammux.get_request_pad(f"sink_{sid}")
    logger.info(
        "[Pipeline] Pre-created %d permanent mux sink pads (slot_capacity=%d)",
        slot_capacity, slot_capacity,
    )

    # ── Core AI processing ───────────────────────────────────────────────────
    logger.info("[DeepStream] Configuring AI elements (PGIE, Tracker, SGIE, Analytics): mono_ts=%.6f", time.monotonic())
    pgie = make_element("primary-infer", "nvinfer")
    pgie.set_property("config-file-path", str(INFER_CONFIG))
    logger.info("[DeepStream] PGIE configured: config=%s", INFER_CONFIG)

    tracker = make_element("tracker", "nvtracker")
    tracker.set_property("ll-lib-file", str(TRACKER_LIB))
    tracker.set_property("ll-config-file", str(TRACKER_CFG))
    tracker.set_property("tracker-width", 224)
    tracker.set_property("tracker-height", 224)
    tracker.set_property("gpu_id", 0)
    logger.info("[DeepStream] Tracker configured: lib=%s, config=%s", TRACKER_LIB, TRACKER_CFG)

    sgie = make_element("secondary-infer", "nvinfer")
    sgie.set_property("config-file-path", str(SGIE_CONFIG))
    logger.info("[DeepStream] SGIE configured: config=%s", SGIE_CONFIG)

    analytics = make_element("analytics", "nvdsanalytics")
    analytics.set_property("config-file", analytics_config)
    logger.info("[DeepStream] Analytics configured: config=%s, mono_ts=%.6f", analytics_config, time.monotonic())

    # ── Determine display / file-write strategy ──────────────────────────────
    is_tiled = (sink_type == "display")

    # ── Tiler (only create when a tiled grid is needed) ───────────────────────
    if is_tiled:
        tiler = make_element("tiler", "nvmultistreamtiler")
        # Grid is computed from the INITIAL camera count so it looks square.
        # It must NOT change while the pipeline is running — resizing rows/cols
        # on a live tiler causes a VIC scaling crash on Jetson.  Dynamic add/remove
        # reuses the existing slots without touching the grid dimensions.
        rows, cols = compute_tiler_layout(n_cameras)
        tiler.set_property("rows", int(rows))
        tiler.set_property("columns", int(cols))
        tiler.set_property("width", mux_width)
        tiler.set_property("height", mux_height)
        tiler.set_property("gpu-id", 0)

        logger.info("[Pipeline] Tiler layout: %d×%d for %d streams", rows, cols, n_cameras)
    else:
        tiler = None

    # ── Pre-OSD convert (display mode only) ───────────────────────────────────
    # For rtsp_push/file the convert+caps+nvdsosd move into each per-camera
    # branch after nvstreamdemux, so OSD never sees the batched surface.
    if is_tiled:
        preosd_convert = make_element("preosd_convert", "nvvideoconvert")
        preosd_caps = make_element("preosd_caps", "capsfilter")
        preosd_caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        )
    else:
        preosd_convert = None
        preosd_caps = None

    # Shared OSD is display-mode only; rtsp_push/file use per-branch OSD.
    if is_tiled:
        nvdsosd = make_element("onscreendisplay", "nvdsosd")
        nvdsosd.set_property("display-text", 1)
        nvdsosd.set_property("display-bbox", 1)
        nvdsosd.set_property("process-mode", 2)
        nvdsosd.set_property("gpu-id", 0)
    else:
        nvdsosd = None

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

    elif sink_type == "rtsp_push":
        demux = make_element("demux", "nvstreamdemux")
        pipeline.add(demux)
        for sid in range(slot_capacity):
            demux.get_request_pad(f"src_{sid}")
        logger.info(
            "[Pipeline] Pre-created %d permanent demux src pads (slot_capacity=%d)",
            slot_capacity, slot_capacity,
        )
        rtsp_base_url = kwargs.get("rtsp_push_base_url") or kwargs.get("rtsp_push_url") or ""
        rtsp_bitrate = kwargs.get("rtsp_push_bitrate") or kwargs.get("bitrate", 750_000)
        node_cam_map = kwargs.get("node_camera_map")
        init_rtsp_push_branches(pipeline, demux, camera_configs, rtsp_base_url, bitrate=rtsp_bitrate, node_camera_map=node_cam_map)

    elif sink_type == "file":
        # ── Demuxer ──
        demux = make_element("demux", "nvstreamdemux")
        pipeline.add(demux)
        for sid in range(slot_capacity):
            demux.get_request_pad(f"src_{sid}")
        logger.info(
            "[Pipeline] Pre-created %d permanent demux src pads (slot_capacity=%d)",
            slot_capacity, slot_capacity,
        )

        for cam_cfg in camera_configs:
            _add_file_recording_branch(pipeline, demux, cam_cfg)

    else:
        raise ValueError(f"Unknown sink_type: '{sink_type}'")

    # ── Add core elements to pipeline ─────────────────────────────────────────
    # For display mode: keep full core chain including preosd_convert/caps/nvdsosd.
    # For rtsp_push/file: OSD moves into per-branch; link analytics→demux directly.
    if is_tiled:
        core_elements = [
            streammux, pgie, tracker, sgie,
            analytics, preosd_convert, preosd_caps, tiler, nvdsosd,
        ]
    elif sink_type in ("rtsp_push", "file"):
        core_elements = [
            streammux, pgie, tracker, sgie,
            analytics,
        ]
    else:
        core_elements = [
            streammux, pgie, tracker, sgie,
            analytics, preosd_convert, preosd_caps, nvdsosd,
        ]

    for el in core_elements + sink_elements:
        pipeline.add(el)

    # ── Link core chain ───────────────────────────────────────────────────────
    if is_tiled:
        gst_link(
            streammux, pgie, tracker, sgie,
            analytics, preosd_convert, preosd_caps, tiler, nvdsosd,
        )
    elif sink_type in ("rtsp_push", "file"):
        gst_link(streammux, pgie, tracker, sgie, analytics)

        # Link analytics → demux (per-branch OSD handles drawing downstream)
        demux_el = pipeline.get_by_name("demux")
        if demux_el is not None:
            analytics_srcpad = analytics.get_static_pad("src")
            demux_sinkpad = demux_el.get_static_pad("sink")
            if analytics_srcpad and demux_sinkpad:
                analytics_srcpad.link(demux_sinkpad)
    else:
        gst_link(
            streammux, pgie, tracker, sgie,
            analytics, preosd_convert, preosd_caps, nvdsosd,
        )

    # ── Link sink chain ───────────────────────────────────────────────────────
    if sink_type == "display":
        conv, conv_caps, eglT, sink = sink_elements
        gst_link(nvdsosd, conv, conv_caps, eglT, sink)

    # ── Add source bins (N cameras) ───────────────────────────────────────────
    source_bins: dict[str, Gst.Element] = {}
    reconnect_args = {
        "rtsp_push_base_url": kwargs.get("rtsp_push_base_url"),
        "rtsp_push_bitrate": kwargs.get("rtsp_push_bitrate"),
        "node_camera_map": kwargs.get("node_camera_map"),
        "probe": kwargs.get("probe"),
    }

    def _reconnect_initial_source(cfg):
        active_p = None
        try:
            from .run_python import ACTIVE_SPEED_PROBE
            if ACTIVE_SPEED_PROBE:
                active_p = ACTIVE_SPEED_PROBE[0]
        except Exception:
            pass
        cur_probe = active_p or reconnect_args.get("probe")
        return dynamic_add_stream(
            pipeline, streammux, cfg, pipeline.get_by_name("tiler"), source_bins,
            probe=cur_probe,
            rtsp_push_base_url=reconnect_args.get("rtsp_push_base_url"),
            rtsp_push_bitrate=reconnect_args.get("rtsp_push_bitrate"),
            node_camera_map=reconnect_args.get("node_camera_map"),
        )

    for cam_cfg in camera_configs:
        src = _make_source_bin(
            pipeline, streammux, cam_cfg,
            reconnect=_reconnect_initial_source,
        )
        source_bins[cam_cfg.camera_id] = src

    logger.info(
        "[Pipeline] Built multi-stream pipeline: %d cameras, sink=%s",
        n_cameras, sink_type,
    )

    return pipeline, nvdsosd, streammux, source_bins


def rebuild_rtsp_push_sink(
    pipeline: Gst.Pipeline,
    rtsp_push_url: str,
    bitrate: int = 4_000_000,
) -> bool:
    """Tear down and recreate only the RTSP push sink branch on failure.

    This replaces the old sink branch (conv, scale_caps, enc, parse, sink) with a fresh
    set of elements unlinked from nvdsosd, preserving all upstream AI pipeline state,
    nvstreammux, and NVDEC decoders without recreating the whole pipeline.

    # ponytail: minimal sink-only rebuild avoids tearing down NVDEC/AI sessions when MediaMTX drops.
    """
    nvdsosd = pipeline.get_by_name("onscreendisplay")

    # Per-branch OSD mode (rtsp_push): no shared OSD exists.  Each camera has
    # its own osd_rtsp_push_{sid} → encoder chain.  Recovery for per-branch
    # failures is handled by dynamic_remove + dynamic_add of the branch.
    if nvdsosd is None:
        logger.info(
            "[Pipeline] rebuild_rtsp_push_sink: no shared OSD found "
            "(per-branch mode); skipping shared rebuild."
        )
        return False

    old_names = ["conv_push", "scale_caps", "enc", "parse", "rtsp_push_sink"]
    # Fallback to checking older conv name if conv_push is not yet used
    old_elements = []
    for name in old_names:
        el = pipeline.get_by_name(name)
        if el is not None:
            old_elements.append(el)
    if not old_elements:
        conv_el = pipeline.get_by_name("conv")
        if conv_el is not None:
            old_elements.append(conv_el)

    # 1. Unlink nvdsosd from the sink branch
    nvdsosd_src = nvdsosd.get_static_pad("src")
    if nvdsosd_src and nvdsosd_src.is_linked():
        peer = nvdsosd_src.get_peer()
        if peer:
            nvdsosd_src.unlink(peer)

    # 2. Sequentially teardown old sink branch elements
    for target_state in (Gst.State.PAUSED, Gst.State.READY, Gst.State.NULL):
        for el in old_elements:
            el.set_state(target_state)
        for el in old_elements:
            el.get_state(1 * Gst.SECOND)

    # 3. Remove old sink branch elements from pipeline
    for el in old_elements:
        pipeline.remove(el)

    # 4. Create fresh sink branch elements
    try:
        conv = make_element("conv_push", "nvvideoconvert")
        scale_caps = make_element("scale_caps", "capsfilter")
        scale_caps.set_property(
            "caps", Gst.Caps.from_string(
                "video/x-raw(memory:NVMM), format=NV12, width=1280, height=720"
            )
        )
        enc = make_element("enc", "nvv4l2h264enc")
        enc.set_property("insert-sps-pps", True)
        enc.set_property("iframeinterval", 30)
        enc.set_property("bitrate", bitrate)
        try:
            enc.set_property("maxperf-enable", True)
        except (TypeError, Exception):
            pass
        parse = make_element("parse", "h264parse")
        sink = make_element("rtsp_push_sink", "rtspclientsink")
        sink.set_property("location", rtsp_push_url)
        sink.set_property("protocols", "tcp")
        sink.set_property("latency", 0)

        new_elements = [conv, scale_caps, enc, parse, sink]
        for el in new_elements:
            pipeline.add(el)

        # 5. Link nvdsosd -> conv -> scale_caps -> enc -> parse -> sink
        gst_link(nvdsosd, conv, scale_caps, enc, parse, sink)

        # 6. Synchronize state of new elements with parent pipeline
        for el in new_elements:
            el.sync_state_with_parent()

        logger.info("[Pipeline] RTSP push sink branch rebuilt and resynced to PLAYING")
        return True
    except Exception as exc:
        logger.error("[Pipeline] Failed to rebuild RTSP push sink branch: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Dynamic stream add/remove helpers (Phase 3)
# ---------------------------------------------------------------------------

# Orin NVDEC hardware decode-session ceiling is ~16-32 (#598); exceeding it
# yields unrecoverable OutputBufferUnavailable (reboot required). Gate ADDs at
# a conservative default well below the documented floor until device data
# justifies raising it.
NVDEC_SESSION_LIMIT = SPEEDFLOW_NVDEC_SESSION_LIMIT


def _iter_elements_deep(root: Gst.Element):
    """Yield every GstElement under root, recursing into bins."""
    if not isinstance(root, Gst.Bin):
        return
    it = root.iterate_recurse()
    while True:
        ret, el = it.next()
        if ret != Gst.IteratorResult.OK:
            return
        yield el


def _count_nvdec_decoders(pipeline: Gst.Pipeline) -> int:
    """Count live nvv4l2decoder elements ≈ active NVDEC hardware sessions."""
    n = 0
    for el in _iter_elements_deep(pipeline):
        factory = el.get_factory()
        if factory and factory.get_name() == "nvv4l2decoder":
            n += 1
    return n


def _proc_rss_mb() -> int:
    """Process RSS in MB (-1 if unreadable) for teardown-leak auditing."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _describe_source_for_teardown(src: Gst.Element) -> str:
    """One-line ground truth for a dying source bin (non-blocking).

    Reports the bin state plus the inner rtspsrc state (proves whether the
    RTSP handshake/RTP path was alive) and whether decodebin ever produced
    a decoder. Uses get_state(0) only — never waits. Every field is
    best-effort; unknowns render as '?'. Logged at every teardown entry so
    a timed-out ADD leaves evidence distinguishing "no RTP" from
    "RTP in, no decode pad".
    """
    try:
        _, bin_state, _ = src.get_state(0)
        bin_s = bin_state.value_nick if bin_state else "?"
    except Exception:
        bin_s = "?"
    rtspsrc_s, dec_s, n_inner = "?", "?", 0
    try:
        for el in _iter_elements_deep(src):
            n_inner += 1
            try:
                factory = el.get_factory()
                fname = factory.get_name() if factory else ""
            except Exception:
                fname = ""
            if fname == "rtspsrc":
                try:
                    _, st, _ = el.get_state(0)
                    rtspsrc_s = st.value_nick if st else "?"
                except Exception:
                    rtspsrc_s = "?"
            elif fname.startswith("nvv4l2"):
                try:
                    _, st, _ = el.get_state(0)
                    dec_s = st.value_nick if st else "?"
                except Exception:
                    dec_s = "?"
    except Exception:
        pass
    return f"bin={bin_s} rtspsrc={rtspsrc_s} decoder={dec_s} inner={n_inner}"


def _link_pads(src_pad, sink_pad, camera_id: str, source_id: int) -> None:
    """Link two pads, raising a descriptive error on failure.

    Gst.Pad.link() returns a PadLinkReturn and never raises — callers that
    ignore it log a lying success and then starve. Compare against OK with
    the same tolerance used in _add_rtsp_push_branch (None/True accepted).
    """
    if src_pad is None or sink_pad is None:
        raise RuntimeError(
            f"Cannot link pads for camera '{camera_id}' (source_id={source_id}): "
            f"src_pad={src_pad}, sink_pad={sink_pad}"
        )
    link_ret = src_pad.link(sink_pad)
    ok_val = getattr(getattr(Gst, "PadLinkReturn", None), "OK", 0)
    if link_ret is not None and link_ret != ok_val and link_ret is not True:
        raise RuntimeError(
            f"Failed to link pads for camera '{camera_id}' (source_id={source_id}): "
            f"return={getattr(link_ret, 'value_nick', link_ret)}"
        )


def _detach_filler_from_pad(pipeline: Gst.Pipeline, streammux: Gst.Element, source_id: int) -> None:
    """Stop and remove a black filler occupying a permanent mux sink pad.

    The mux pad itself is NEVER released (#596 crash class): only the
    upstream filler branch is torn down so the permanent pad becomes free
    for the real source branch.
    """
    fake_src = pipeline.get_by_name(f"fake_src_{source_id}")
    if not fake_src:
        return
    fake_conv = pipeline.get_by_name(f"fake_conv_{source_id}")
    fake_elements = [el for el in [fake_src, fake_conv] if el is not None]
    # Unlink from the mux side FIRST, before any state change: the permanent
    # mux pad is authoritative and never released (#596). Querying the filler's
    # src pad after NULL can report unlinked and skip the unlink, leaving the
    # mux pad half-linked → next ADD sees sinkpad.is_linked()==True and skips
    # the real branch (nvdec 1→1, no PLAYING). Oracle-confirmed 2026-09-16.
    sinkpad = streammux.get_static_pad(f"sink_{source_id}")
    if sinkpad is not None and sinkpad.is_linked():
        peer = sinkpad.get_peer()
        if peer is not None:
            peer.unlink(sinkpad)
    for el in fake_elements:
        el.set_state(Gst.State.NULL)
    for el in fake_elements:
        pipeline.remove(el)
    logger.info("[Pipeline] Detached black filler from permanent pad sink_%d", source_id)


def _add_fake_black_source(pipeline: Gst.Pipeline, streammux: Gst.Element, source_id: int) -> None:
    """Add videotestsrc pattern=2 (black) to keep tiler slot black."""
    try:
        # ponytail: never get_request_pad post-init (#596 crash class) — the
        # permanent pad was pre-created at build time via slot_capacity.
        sinkpad = streammux.get_static_pad(f"sink_{source_id}")
        if sinkpad is None:
            logger.warning(
                "[Pipeline] No permanent mux pad sink_%d; cannot attach black filler.",
                source_id,
            )
            return
        if sinkpad.is_linked():
            return
        fake_src = make_element(f"fake_src_{source_id}", "videotestsrc")
        fake_src.set_property("pattern", 2)  # 2 = black
        fake_conv = make_element(f"fake_conv_{source_id}", "nvvideoconvert")
        pipeline.add(fake_src)
        pipeline.add(fake_conv)
        gst_link(fake_src, fake_conv)
        conv_pad = fake_conv.get_static_pad("src")
        if conv_pad and sinkpad:
            conv_pad.link(sinkpad)
        fake_src.sync_state_with_parent()
        fake_conv.sync_state_with_parent()
        logger.info("[Pipeline] Added fake black source for freed slot source_id=%d", source_id)
    except Exception as exc:
        logger.warning("[Pipeline] Could not add fake black source for slot source_id=%d: %s", source_id, exc)


def attach_osd_probe(probe: Optional["object"], pipeline: Gst.Pipeline, source_id: int, is_rtsp: bool) -> None:
    """Attach the shared SpeedProbe to a newly created per-branch OSD sink pad.

    probe:      shared SpeedProbe instance (may be None).
    pipeline:   the Gst.Pipeline containing the OSD.
    source_id:  the demux src slot that the new camera occupies.
    is_rtsp:    True → lookup osd_rtsp_push_{sid}; False → osd_file_{sid}.
    """
    if probe is None:
        try:
            from .run_python import ACTIVE_SPEED_PROBE
            if ACTIVE_SPEED_PROBE:
                probe = ACTIVE_SPEED_PROBE[0]
        except Exception:
            pass
    if probe is None:
        logger.warning("[Pipeline] attach_osd_probe: no SpeedProbe available for source_id=%d", source_id)
        return
    pad_name = "osd_rtsp_push" if is_rtsp else "osd_file"
    osd_name = f"{pad_name}_{source_id}"
    osd_el = pipeline.get_by_name(osd_name)
    if osd_el is None:
        logger.warning("[Pipeline] attach_osd_probe: %s not found for source_id=%d", osd_name, source_id)
        return
    osd_pad = osd_el.get_static_pad("sink")
    if osd_pad is None:
        logger.warning("[Pipeline] attach_osd_probe: %s has no sink pad", osd_name)
        return
    # Idempotency guard: every GStreamer add_probe() stacks a new callback on
    # the same pad — no dedup in GStreamer itself. A keep-alive re-ADD or a
    # retry ADD that reuses the same OSD element would attach the probe again,
    # causing FPS double-count and plate/vehicle emit side-effects to run 2×
    # per buffer (Oracle A1/R5, confirmed 2026-09-22). Guard on the element
    # object so it clears automatically when the element is destroyed on real
    # teardown and a fresh branch gets a clean attach.
    if getattr(osd_el, "_speedprobe_attached", False):
        logger.debug("[Pipeline] attach_osd_probe: already attached to %s, skipping", osd_name)
        return
    osd_el._speedprobe_attached = True
    osd_pad.add_probe(Gst.PadProbeType.BUFFER, probe.osd_sink_pad_buffer_probe, None)
    logger.info("[Pipeline] attach_osd_probe: attached probe to %s for source_id=%d", osd_name, source_id)


def dynamic_add_stream(
    pipeline: Gst.Pipeline,
    streammux: Gst.Element,
    cam_cfg: CameraConfig,
    tiler: Optional[Gst.Element],
    source_bins: dict,
    ready_event: Optional[threading.Event] = None,
    rtsp_push_base_url: Optional[str] = None,
    rtsp_push_bitrate: Optional[int] = None,
    node_camera_map: Optional[dict] = None,
    probe: Optional["object"] = None,
) -> Gst.Element:
    # Cleanup stale source_bin from a previous failed ADD attempt.
    # If the prior _send_ack timed out without a REMOVE completing, the old
    # uridecodebin may still occupy this slot (its mux pad stays linked).
    # Route through the SAME sequential teardown as dynamic_remove_stream:
    # direct-to-NULL slams leave Tegra nvv4l2decoder registers undefined
    # (#597 kernel v4l2 deadlock), and pad release is forbidden post-init (#596).
    # NEVER tear down a PLAYING branch for bookkeeping: a live bin whose
    # readiness event decoupled (fresh event per retry + one-shot first-buffer
    # probe) still streams — replacing it destroyed C/cam_05 live (2026-09-20:
    # playing/playing/playing torn down as "unready", rebuild starved, node
    # went dark). Only replace when the URI actually changed (source moved).
    stale_src = source_bins.get(cam_cfg.camera_id)
    if stale_src is not None and stale_src.get_parent() is not None:
        try:
            _, _stale_state, _ = stale_src.get_state(0)
        except Exception:
            _stale_state = None
        try:
            _stale_uri = normalize_uri(str(stale_src.get_property("uri") or ""))
        except Exception:
            _stale_uri = ""
        if (_stale_state == Gst.State.PLAYING
                and _stale_uri
                and _stale_uri == normalize_uri(cam_cfg.uri or "")):
            logger.info(
                "[Pipeline] Keeping live PLAYING branch for '%s' (source_id=%d); "
                "re-ADD is bookkeeping-only, teardown skipped.",
                cam_cfg.camera_id, cam_cfg.source_id,
            )
            if ready_event is not None:
                try:
                    ready_event.set()
                except Exception:
                    pass
            _is_rtsp = pipeline.get_by_name(f"osd_rtsp_push_{cam_cfg.source_id}") is not None
            attach_osd_probe(probe, pipeline, cam_cfg.source_id, _is_rtsp)
            return stale_src
        logger.info(
            "[Pipeline] Removing stale source_bin for '%s' (source_id=%d) before retry ADD.",
            cam_cfg.camera_id, cam_cfg.source_id,
        )
        _teardown_source_branch(
            pipeline, streammux, cam_cfg.camera_id, cam_cfg.source_id,
            tiler, source_bins,
        )

    # NVDEC session gate (#598): exceeding the Orin decode-session ceiling is
    # unrecoverable (reboot required). Refuse BEFORE mutating anything; the
    # caller's exception path disables the config and a later REMOVE becomes
    # a clean no-op.
    nvdec_count = _count_nvdec_decoders(pipeline)
    if nvdec_count >= NVDEC_SESSION_LIMIT:
        raise RuntimeError(
            f"NVDEC session limit reached ({nvdec_count} >= "
            f"{NVDEC_SESSION_LIMIT}); refusing ADD '{cam_cfg.camera_id}' "
            f"(source_id={cam_cfg.source_id})"
        )

    # batch-size is fixed at slot_capacity (set at build time, pool frozen at
    # NULL→READY).  No runtime mutation here — attempting set_property after
    # PLAYING is a no-op on Orin and was the root cause of nvdec 1→1 timeouts.
    demux = pipeline.get_by_name("demux")
    recording_added = False
    rtsp_push_added = False
    src = None
    try:
        # STEP 1: Tear down any stale RTSP push branch OFF the GLib main thread.
        #
        # _remove_rtsp_push_branch uses _set_state_bounded → done.wait() which
        # blocks the caller. Since dynamic_add_stream runs on the GLib main thread
        # (camera_config.py:705 idle_add), blocking here would prevent rtspsrc's
        # source-setup / pad-added signals (marshalled through the GLib main context
        # by PyGObject) from being dispatched — rtspsrc stays stuck PAUSED forever.
        #
        # Fix (R8, confirmed 2026-09-22): run the stale-branch teardown on a
        # dedicated worker thread and join() BEFORE creating the source bin. This
        # frees the GLib main thread during the blocking teardown while ensuring
        # teardown is complete before the new source bin is added. The source bin
        # then gets a clear slot and rtspsrc's pad-added can dispatch normally.
        if demux is not None and rtsp_push_base_url:
            # Always run _remove_rtsp_push_branch on a worker thread regardless of
            # whether stale elements exist. _remove_rtsp_push_branch checks internally
            # and is a no-op when there is nothing to remove. Running unconditionally
            # closes the R8B gap: previously the stale-element check (line ~368 in
            # _add_rtsp_push_branch) could still find elements that arrived via
            # publisher-recovery and call _remove_rtsp_push_branch on the GLib main
            # thread, blocking rtspsrc source-setup / pad-added dispatch.
            # By clearing ALL stale elements here first, _add_rtsp_push_branch at
            # line ~368 finds nothing and skips its direct call. (R8B, 2026-09-22)
            _teardown_done = threading.Event()
            _teardown_exc: list = []
            def _teardown_worker():
                try:
                    _remove_rtsp_push_branch(pipeline, cam_cfg.source_id)
                except Exception as e:
                    _teardown_exc.append(e)
                finally:
                    _teardown_done.set()
            t = threading.Thread(target=_teardown_worker, daemon=True)
            t.start()
            _teardown_done.wait(timeout=20.0)  # bounded; GLib main thread stays free
            if _teardown_exc:
                logger.warning(
                    "[Pipeline] Stale RTSP push branch teardown for '%s' raised: %s",
                    cam_cfg.camera_id, _teardown_exc[0],
                )

        # STEP 2: Create source bin and kick rtspsrc to PLAYING.
        # GLib main thread is now free so source-setup / pad-added can dispatch.
        reconnect = lambda cfg: dynamic_add_stream(
            pipeline, streammux, cfg, tiler, source_bins,
            ready_event=ready_event,
            rtsp_push_base_url=rtsp_push_base_url,
            rtsp_push_bitrate=rtsp_push_bitrate,
            node_camera_map=node_camera_map,
            probe=probe,
        )
        src = _make_source_bin(
            pipeline, streammux, cam_cfg,
            ready_event=ready_event,
            reconnect=reconnect,
        )
        src.sync_state_with_parent()

        # STEP 3: Add RTSP push branch (may block briefly for NVENC init).
        # rtspsrc is already negotiating on its own thread; pad-added will
        # dispatch on the next GLib loop iteration after this returns.
        if demux is not None:
            if rtsp_push_base_url:
                bitrate = rtsp_push_bitrate if rtsp_push_bitrate is not None else 750_000
                _add_rtsp_push_branch(
                    pipeline, demux, cam_cfg, rtsp_push_base_url, bitrate=bitrate, sync=True, node_camera_map=node_camera_map
                )
                rtsp_push_added = True
            elif cam_cfg.record:
                _add_file_recording_branch(pipeline, demux, cam_cfg, sync=True)
                recording_added = True

        # Attach the shared probe to the newly created per-branch OSD so the
        # dynamically added camera draws its own overlay (it is not covered by
        # the static build-time probe loop in _setup_probes).
        is_rtsp_branch = bool(rtsp_push_added)
        attach_osd_probe(probe, pipeline, cam_cfg.source_id, is_rtsp_branch)
    except Exception:
        if src is not None:
            _teardown_source_branch(
                pipeline, streammux, cam_cfg.camera_id, cam_cfg.source_id,
                tiler, source_bins,
            )
        if recording_added:
            _remove_file_recording_branch(pipeline, cam_cfg.source_id)
        if rtsp_push_added:
            _remove_rtsp_push_branch(pipeline, cam_cfg.source_id)
        raise

    source_bins[cam_cfg.camera_id] = src
    logger.info(
        "[Pipeline] Added stream '%s' (source_id=%d)",
        cam_cfg.camera_id, cam_cfg.source_id,
    )
    return src




def _teardown_source_branch(
    pipeline: Gst.Pipeline,
    streammux: Gst.Element,
    camera_id: str,
    source_id: int,
    tiler: Optional[Gst.Element],
    source_bins: dict,
) -> None:
    """Single sequential teardown path for a live source branch.

    Used by dynamic_remove_stream AND the stale-bin cleanup in
    dynamic_add_stream so there is exactly one correct teardown
    implementation (#597): PLAYING→PAUSED→READY→NULL with get_state waits.
    The mux sink pad is NEVER released (#596 crash class); after the branch
    is gone the permanent pad is re-armed with a black filler (tiled sinks).
    """
    src = source_bins.get(camera_id)
    if not src:
        return

    pre_nvdec = _count_nvdec_decoders(pipeline)
    pre_rss = _proc_rss_mb()
    logger.info(
        "[Pipeline] Teardown entry '%s' (source_id=%d): %s",
        camera_id, source_id, _describe_source_for_teardown(src),
    )

    try:
        # Teardown downstream RTSP push / file recording branch first
        # so source removal cannot send EOS / broken state into live rtspclientsink
        _remove_rtsp_push_branch(pipeline, source_id)
        _remove_file_recording_branch(pipeline, source_id)

        # Use generation-aware names stored on the source element (R2/A3 fix).
        # Fallback to static names only for bins created before this fix.
        q_elem   = pipeline.get_by_name(getattr(src, "_q_name",   f"q_{camera_id}"))
        conv_elem = pipeline.get_by_name(getattr(src, "_conv_name", f"conv_{camera_id}"))
        conv_pad = conv_elem.get_static_pad("src") if conv_elem else None
        mux_sinkpad = conv_pad.get_peer() if conv_pad and conv_pad.is_linked() else None

        # 1) Install a buffer+EOS drop probe on conv:src BEFORE unlinking.
        #    Dropping all buffers at the pad prevents any buffer from reaching
        #    an unlinked pad (eliminating GST_FLOW_NOT_LINKED) while catching EOS.
        drained = threading.Event()
        drain_probe_id = None
        if conv_pad is not None:
            def _drain_probe(pad, info):
                if info.type & (Gst.PadProbeType.EVENT_DOWNSTREAM | Gst.PadProbeType.EVENT_UPSTREAM):
                    ev = info.get_event()
                    if ev and ev.type == Gst.EventType.EOS:
                        drained.set()
                        return Gst.PadProbeReturn.DROP
                # DROP all buffers so nothing flows downstream into unlinked pad or mux
                return Gst.PadProbeReturn.DROP

            drain_probe_id = conv_pad.add_probe(
                Gst.PadProbeType.BUFFER | Gst.PadProbeType.EVENT_DOWNSTREAM,
                _drain_probe,
            )

        # 2) Unlink conv->mux AFTER probe is active:
        #    conv->mux is unlinked so drain/EOS cannot reach the shared nvstreammux.
        if conv_pad and mux_sinkpad and conv_pad.is_linked():
            conv_pad.unlink(mux_sinkpad)

        # 3) Drain the decoder: inject EOS at the branch head and let nvv4l2decoder
        #    flush its picture-buffer pool downstream into the now-dead-ended conv.
        if conv_pad is not None:
            src.send_event(Gst.Event.new_eos())
            if not drained.wait(timeout=1.0):
                logger.warning("[Pipeline] '%s' EOS drain timed out (1.0s); forcing teardown.", camera_id)
            if drain_probe_id is not None:
                try:
                    conv_pad.remove_probe(drain_probe_id)
                except Exception:
                    pass

        # 4) State down DOWNSTREAM-FIRST (conv -> q -> src): the decoder reaches NULL
        #    last, after its surfaces are already drained and unreferenced.
        # ASYNC must be resolved before calling pipeline.remove(): an in-flight TSG unbind
        # (nvgpu Channel N unbind failed, EAGAIN) orphans the TSG if remove() races it,
        # accumulating NVDEC sessions toward the #598 ceiling and exhausting the AXI bus
        # → HDA timeout → CPU0 RCU stall → hard reset (confirmed pstore forensics, 2026-09-15).
        branch_elements = [el for el in [conv_elem, q_elem, src] if el is not None]
        for target_state in (Gst.State.PAUSED, Gst.State.READY, Gst.State.NULL):
            for el in branch_elements:
                el.set_state(target_state)
            for el in branch_elements:
                state_ret, current_state, _ = el.get_state(1 * Gst.SECOND)
                if state_ret == Gst.StateChangeReturn.ASYNC:
                    # TSG unbind in-flight; give nvv4l2decoder up to 5s to complete
                    state_ret, current_state, _ = el.get_state(5 * Gst.SECOND)
                if state_ret == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError(
                        f"Element {el.get_name()} failed to reach "
                        f"{target_state.value_nick}: state={current_state.value_nick}"
                    )
                if state_ret == Gst.StateChangeReturn.ASYNC:
                    raise RuntimeError(
                        f"Element {el.get_name()} ASYNC teardown unresolved after 6s "
                        f"(target={target_state.value_nick}, state={current_state.value_nick}); "
                        f"refusing pipeline.remove() to prevent TSG orphan / NVDEC session leak"
                    )

        # Verify hardware decoder(s) really reached NULL — a bin can report
        # NULL while an inner nvv4l2decoder is wedged mid-teardown, silently
        # leaking its NVDEC session toward the #598 accumulation ceiling.
        for dec_el in _iter_elements_deep(src):
            factory = dec_el.get_factory()
            fname = factory.get_name() if factory else ""
            if fname.startswith("nvv4l2"):
                _, dec_state, _ = dec_el.get_state(0)
                if dec_state != Gst.State.NULL:
                    logger.critical(
                        "[Pipeline] '%s' (%s) stuck at %s after bin NULL for "
                        "camera %s — NVDEC session leak risk!",
                        dec_el.get_name(), fname, dec_state.value_nick, camera_id,
                    )

        # Remove elements from pipeline
        for el in branch_elements:
            pipeline.remove(el)

        # Leak audit (on-device evidence): nvdec must drop by exactly the
        # number of removed branches (-1 here); rss creep across many cycles
        # flags Python/GObject ref leaks or unfreed NvBufSurface memory.
        logger.info(
            "[Pipeline] Teardown audit '%s': nvdec %d→%d, rss %dMB→%dMB",
            camera_id,
            pre_nvdec, _count_nvdec_decoders(pipeline),
            pre_rss, _proc_rss_mb(),
        )
        logger.info("[Pipeline] Cleaned up resources for camera %s", camera_id)
    except Exception as exc:
        logger.error("[Pipeline] Error during cleanup of camera %s: %s", camera_id, exc)
        raise
    finally:
        # Always run bookkeeping even when hardware teardown raised (A2 fix).
        # Leaving source_bins mapped to a dead/half-torn src causes the next ADD
        # stale-bin path to call _teardown_source_branch again on the same broken
        # src → infinite wedge. Clear mapping unconditionally so the next ADD can
        # attempt a fresh bin. Black filler re-arm is best-effort (tiled sinks only).
        if camera_id in source_bins:
            del source_bins[camera_id]
        if tiler is not None:
            try:
                _add_fake_black_source(pipeline, streammux, source_id)
            except Exception:
                pass


def dynamic_remove_stream(
    pipeline: Gst.Pipeline,
    streammux: Gst.Element,
    camera_id: str,
    source_id: int,
    tiler: Optional[Gst.Element],
    source_bins: dict,
    done_event: Optional[threading.Event] = None,
) -> None:
    src = source_bins.get(camera_id)
    if not src:
        if done_event is not None:
            done_event.set()
        return

    logger.info(
        "[Pipeline] Removing stream '%s' (source_id=%d) synchronously",
        camera_id,
        source_id,
    )
    # Synchronous teardown on the GLib main thread: avoids BLOCK_DOWNSTREAM pad
    # probe deadlocks when RTSP streams are stalled and buffers stop flowing.
    try:
        _teardown_source_branch(pipeline, streammux, camera_id, source_id, tiler, source_bins)
    finally:
        if done_event is not None:
            done_event.set()


# ---------------------------------------------------------------------------
# RTSP publisher failure isolation (P1)
# ---------------------------------------------------------------------------
#
# A publisher (rtspclientsink) failure must stay a *per-camera leaf*: it must
# never propagate into a full DeepStream pipeline restart or an ADD/REMOVE
# storm.  The logic below is intentionally free of any GStreamer dependency so
# it can be unit-tested in isolation (the recovery controller, the error
# classifier, and the decision router are pure Python).

def classify_pipeline_error(
    src_name: str,
    err_text: str = "",
    debug_text: str = "",
    *,
    is_rtsp_sink: bool = False,
) -> str:
    """Classify a GStreamer ERROR bus message.

    Returns one of:
      - "transient"   : benign, self-healing decoder starvation (NVDEC buffer
                        exhaustion) or teardown buffer artifact.
      - "publisher"   : the failure originated in an RTSP push publisher
                        (rtspclientsink / per-camera push branch) — isolate it.
      - "pipeline"    : any other genuine pipeline-level error.
    """
    text = f"{err_text} {debug_text or ''}"
    if "OutputBufferUnavailable" in text or "cbAllocPictureBuffer" in text:
        return "transient"
    if "not-linked" in text and any(src_name.startswith(pfx) for pfx in ("q_", "conv_", "src-")):
        return "transient"
    if is_rtsp_sink:
        return "publisher"
    return "pipeline"


class PublisherRecovery:
    """Per-camera single-flight bounded-exponential retry + circuit breaker.

    State is keyed by camera id.  The controller guarantees:
      * single-flight — at most one in-flight recovery per camera, so a burst
        of publisher errors cannot spawn an ADD/REMOVE storm (one rebuild at
        a time, scheduled retries coalesced).
      * bounded exponential backoff — delay grows base*2**n, capped at
        ``max_delay_s``.
      * circuit breaker — after ``max_attempts`` consecutive failures the
        branch is left DOWN (leaf) instead of thrashing; it auto-resets after
        ``reset_after_s`` so a later healthy window can retry.
      * intentional teardown — cameras being removed on purpose are excluded
        from recovery entirely.

    No GStreamer calls: safe to import and exercise under stdlib-only tests.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        base_delay_s: float = 2.0,
        max_delay_s: float = 60.0,
        reset_after_s: float = 300.0,
    ):
        self.max_attempts = int(max_attempts)
        self.base_delay_s = float(base_delay_s)
        self.max_delay_s = float(max_delay_s)
        self.reset_after_s = float(reset_after_s)
        self._state: dict = {}          # cam_id -> {attempts, last_failure, in_flight, circuit_open}
        self._intentional: set = set()

    # -- intentional teardown ------------------------------------------------
    def mark_intentional_teardown(self, cam_id) -> None:
        self._intentional.add(cam_id)

    def clear_intentional_teardown(self, cam_id) -> None:
        self._intentional.discard(cam_id)

    def is_intentional_teardown(self, cam_id) -> bool:
        return cam_id in self._intentional

    # -- state helpers ------------------------------------------------------
    def _st(self, cam_id) -> dict:
        s = self._state.get(cam_id)
        if s is None:
            s = {"attempts": 0, "last_failure": 0.0, "in_flight": False, "circuit_open": False}
            self._state[cam_id] = s
        return s

    def begin_attempt(self, cam_id) -> None:
        self._st(cam_id)["in_flight"] = True

    def clear_in_flight(self, cam_id) -> None:
        self._st(cam_id)["in_flight"] = False

    def _hold(self, cam_id) -> None:
        self._st(cam_id)["in_flight"] = True

    def record_success(self, cam_id) -> None:
        s = self._st(cam_id)
        s["attempts"] = 0
        s["in_flight"] = False
        s["circuit_open"] = False
        s["last_failure"] = 0.0

    def record_failure(self, cam_id, now: float = None) -> None:
        s = self._st(cam_id)
        s["in_flight"] = False
        s["attempts"] += 1
        s["last_failure"] = now if now is not None else time.monotonic()
        if s["attempts"] >= self.max_attempts:
            s["circuit_open"] = True

    def backoff_seconds(self, cam_id) -> float:
        s = self._st(cam_id)
        exp = self.base_delay_s * (2 ** max(0, s["attempts"] - 1))
        return min(self.max_delay_s, exp)

    def _maybe_reset(self, cam_id, now: float = None) -> None:
        s = self._state.get(cam_id)
        if not s or not s["last_failure"]:
            return
        now = now if now is not None else time.monotonic()
        if (now - s["last_failure"]) >= self.reset_after_s:
            s["attempts"] = 0
            s["circuit_open"] = False
            s["last_failure"] = 0.0

    def is_recoverable(self, cam_id) -> bool:
        if cam_id in self._intentional:
            return False
        s = self._state.get(cam_id)
        if s is None:
            return True
        if s["in_flight"]:
            return False
        self._maybe_reset(cam_id)
        s = self._state[cam_id]
        if s["circuit_open"]:
            return False
        if s["attempts"] >= self.max_attempts:
            return False
        return True


def handle_publisher_failure(
    recovery: "PublisherRecovery",
    cam_id,
    rebuild_fn,
    schedule_fn,
) -> str:
    """Route one publisher error for a single camera.

    ``rebuild_fn`` takes no arguments and returns True on success.
    ``schedule_fn(delay_seconds)`` schedules a later retry (single-flight).

    Returns one of:
      - "intentional" : camera is being intentionally removed — do nothing.
      - "recovered"   : rebuild succeeded; analytics pipeline stays PLAYING.
      - "scheduled"   : rebuild failed, a single bounded retry was scheduled.
      - "leaf"        : retry budget / circuit exhausted — branch left down,
                        analytics pipeline stays PLAYING (NO full restart).

    Callers MUST NOT call loop.quit() on any of these outcomes: publisher
    failure is a per-camera leaf by construction.
    """
    if recovery.is_intentional_teardown(cam_id):
        return "intentional"

    if not recovery.is_recoverable(cam_id):
        return "leaf"

    recovery.begin_attempt(cam_id)
    try:
        ok = bool(rebuild_fn())
    except Exception:
        ok = False

    if ok:
        recovery.record_success(cam_id)
        return "recovered"

    recovery.record_failure(cam_id)
    if recovery.is_recoverable(cam_id):
        delay = recovery.backoff_seconds(cam_id)
        recovery._hold(cam_id)
        schedule_fn(delay)
        return "scheduled"
    return "leaf"
