import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

webrtc = Gst.ElementFactory.make("webrtcbin", "webrtc")
if not webrtc:
    print("Failed to create webrtcbin")
    exit(1)

print(f"Created webrtcbin: {webrtc}")

# Try request_pad_simple
try:
    pad = webrtc.request_pad_simple("sink_0")
    print(f"request_pad_simple('sink_0'): {pad}")
except AttributeError:
    print("request_pad_simple not available")

# Try request_pad with template
templ = webrtc.get_pad_template("sink_%u")
if templ:
    print(f"Found template: {templ.name_template}")
    pad = webrtc.request_pad(templ, "sink_0", None)
    print(f"request_pad(templ, 'sink_0'): {pad}")
else:
    print("Template sink_%u not found")
