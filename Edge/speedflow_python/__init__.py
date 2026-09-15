# speedflow_python/__init__.py
"""
Python-based speed measurement and license plate recognition module.
"""

from . import settings      # access via settings.VIDEO_FPS etc.

try:
    from . import speedflow_c   # C extension; available → speedflow_c.is_available()
except (RuntimeError, OSError, ImportError):
    pass

try:
    from .core_pipeline import build_pipeline
    from .probes import SpeedProbe, ROIFilterProbe
    from .common import make_element, gst_link
except ImportError:
    # Optional DeepStream / GStreamer / PyGObject dependencies in headless or test envs
    pass

__all__ = [
    'build_pipeline',
    'SpeedProbe',
    'ROIFilterProbe',
    'make_element',
    'gst_link',
    'settings',
    'speedflow_c',
]
