# speedflow_cpp/__init__.py
"""
C++ GStreamer plugin based speed measurement module.
Provides Python wrapper to launch C++ pipeline.
"""

from .pipeline_cpp import build_pipeline_cpp, run_cpp_mode

__all__ = ['build_pipeline_cpp', 'run_cpp_mode']
