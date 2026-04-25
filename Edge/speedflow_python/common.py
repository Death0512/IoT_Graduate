# speedflow_python/common.py
"""
Shared utilities used by both Python and C++ backends.
Extracted to eliminate DRY violations across run_python.py and pipeline_cpp.py.
"""
import asyncio
import json
import sys

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstWebRTC', '1.0')
gi.require_version('GstSdp', '1.0')
from gi.repository import Gst, GstWebRTC, GstSdp

try:
    import websockets
except ImportError:
    websockets = None


# ---------------------------------------------------------------------------
# GStreamer helper
# ---------------------------------------------------------------------------

def make_element(name: str, factory: str) -> Gst.Element:
    """Create a GStreamer element, raising a clear error if the factory is missing."""
    element = Gst.ElementFactory.make(factory, name)
    if not element:
        raise RuntimeError(
            f"Failed to create GStreamer element '{factory}' (alias '{name}'). "
            f"Make sure the required GStreamer plugin is installed."
        )
    return element


def gst_link(*elements: Gst.Element) -> None:
    """
    Link a chain of GStreamer elements in order.
    Raises RuntimeError with a descriptive message on failure.
    Replaces bare `assert element.link(next)` calls which are disabled by -O.
    """
    for a, b in zip(elements, elements[1:]):
        if not a.link(b):
            raise RuntimeError(
                f"Failed to link GStreamer elements: "
                f"'{a.get_name()}' → '{b.get_name()}'"
            )


# ---------------------------------------------------------------------------
# WebRTC signaling session (shared between Python and C++ backends)
# ---------------------------------------------------------------------------

class WebRTCSession:
    """
    Manages WebRTC offer/answer signaling and ICE candidate exchange
    over a WebSocket connection to the signaling server.

    Thread-safety: GStreamer callbacks run in the GLib main loop thread.
    asyncio coroutines run in a separate thread via run_coroutine_threadsafe.
    """

    def __init__(self, webrtc: Gst.Element, ws_uri: str) -> None:
        if websockets is None:
            print(
                "ERROR: 'websockets' package not installed. "
                "Run: pip install websockets",
                file=sys.stderr,
            )
            sys.exit(1)

        self.webrtc = webrtc
        self.ws_uri = ws_uri
        self.ws = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._closing = False
        self._ice_buffer: list[dict] = []

        self.webrtc.connect("on-negotiation-needed", self.on_negotiation_needed)
        self.webrtc.connect("on-ice-candidate", self.on_ice_candidate)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish WebSocket connection and start the receive loop."""
        self.loop = asyncio.get_running_loop()
        await self._ws_connect()
        asyncio.create_task(self._recv_loop())
        await self._flush_ice_buffer()
        await asyncio.sleep(0.2)
        self.on_negotiation_needed(self.webrtc)

    def send_json_threadsafe(self, data: dict) -> None:
        """Send JSON data to the signaling server from any thread."""
        if not self.ws or self.loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps(data)),
                self.loop,
            )
        except Exception as exc:
            print(f"[WebRTC] Failed to publish JSON: {exc}")

    def close(self) -> None:
        """Signal the session to stop reconnecting."""
        self._closing = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ws_connect(self) -> None:
        """Connect to the signaling server, retrying until successful."""
        while not self._closing:
            try:
                self.ws = await websockets.connect(self.ws_uri)
                print(f"[WebRTC] Connected to signaling server: {self.ws_uri}")
                return
            except Exception as exc:
                print(f"[WebRTC] Connection failed, retrying… ({exc})")
                await asyncio.sleep(1.2)

    async def _recv_loop(self) -> None:
        """Receive signaling messages and feed them to webrtcbin."""
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

        except Exception as exc:
            print(f"[WebRTC] Receive loop ended: {exc}")
            await self._handle_ws_drop()

    async def _handle_ws_drop(self) -> None:
        """Reconnect after a WebSocket disconnect."""
        if self._closing:
            return
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass
        self.ws = None
        print("[WebRTC] Reconnecting…")
        await asyncio.sleep(0.8)
        await self._ws_connect()
        await asyncio.sleep(0.2)
        self.on_negotiation_needed(self.webrtc)

    async def _flush_ice_buffer(self) -> None:
        """Send any ICE candidates that were buffered before WS was ready."""
        if self._ice_buffer:
            for ice_msg in list(self._ice_buffer):
                try:
                    await self.ws.send(json.dumps(ice_msg))
                except Exception:
                    pass
            self._ice_buffer.clear()

    # ------------------------------------------------------------------
    # GStreamer signal callbacks (called from GLib main loop thread)
    # ------------------------------------------------------------------

    def on_negotiation_needed(self, element: Gst.Element) -> None:
        print("[WebRTC] Negotiation needed")
        promise = Gst.Promise.new_with_change_func(
            self._on_offer_created, element, None
        )
        self.webrtc.emit("create-offer", None, promise)

    def _on_offer_created(
        self, promise: Gst.Promise, element: Gst.Element, _
    ) -> None:
        reply = promise.get_reply()
        offer = reply.get_value("offer")
        if not offer:
            print("[WebRTC] Failed to create offer — webrtcbin not ready yet")
            return
        if not self.ws or self.loop is None:
            print("[WebRTC] WebSocket not connected yet, skipping offer send")
            return
        self.webrtc.emit("set-local-description", offer, None)
        text = offer.sdp.as_text()
        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps({"type": "offer", "sdp": text})),
            self.loop,
        )

    def on_ice_candidate(
        self, element: Gst.Element, mline: int, candidate: str
    ) -> None:
        if not candidate:
            return
        ice_msg = {
            "type": "ice",
            "candidate": {
                "candidate": candidate,
                "sdpMLineIndex": int(mline),
            },
        }
        if not self.ws or self.loop is None:
            self._ice_buffer.append(ice_msg)
            return
        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps(ice_msg)),
            self.loop,
        )
