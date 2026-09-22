# speedflow_python/common.py
"""
Shared utilities for the Python backend pipeline.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os


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
# URI helpers — single source of truth for file vs RTSP discrimination
# ---------------------------------------------------------------------------

def is_file_uri(uri: str) -> bool:
    """
    Check if a URI is a file source (file:// or absolute path that exists).

    This is the SINGLE source of truth for file/RTSP discrimination.
    Both core_pipeline.py and run_python.py import from here.

    Args:
        uri: URI string to check

    Returns:
        True if file:// scheme or absolute path that exists on disk
    """
    if not uri:
        return False
    s = uri.strip().lower()
    if s.startswith("file://"):
        return True
    if s.startswith("/"):
        return os.path.exists(s)
    return False
