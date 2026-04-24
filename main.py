#!/usr/bin/env python3
"""
Unified DeepStream Speed Measurement Entry Point
Supports dual backend: Python (flexibility) or C++ (performance)
"""
import sys
import os
import argparse

def main():
    """Main entry point with dual-backend support."""
    parser = argparse.ArgumentParser(
        description="DeepStream Traffic Monitor - Dual Mode (Python/C++)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Python backend (default)
  python3 main.py --backend python --source video.mp4 --mode display
  
  # C++ backend (high performance)
  python3 main.py --backend cpp --source video.mp4 --mode display
  
  # WebRTC mode with C++ backend
  python3 main.py --backend cpp --source video.mp4 --mode webrtc \\
      --server 192.168.0.158 --room demo --cfg configs/config_cam.txt
        """
    )
    
    # Backend selection
    parser.add_argument(
        "--backend",
        required=True,
        choices=["python", "cpp"],
        help="Processing backend: 'python' (flexible) or 'cpp' (high performance)"
    )
    
    # Required arguments
    parser.add_argument(
        "--source",
        required=True,
        help="Input source (RTSP URL or file path)"
    )
    
    parser.add_argument(
        "--mode",
        required=True,
        choices=["display", "file", "webrtc"],
        help="Output mode: display (screen), file (MP4), or webrtc (browser stream)"
    )
    
    # Common optional arguments
    parser.add_argument(
        "--homo",
        default="configs/points_1.yml",
        help="Homography points YAML file (default: configs/points_1.yml)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Streammux width (default: 1920)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Streammux height (default: 1080)"
    )
    
    # File mode specific
    parser.add_argument(
        "--output",
        help="Output file path (required for file mode)"
    )
    
    # WebRTC mode specific
    parser.add_argument(
        "--server",
        default="localhost",
        help="WebRTC signaling server IP (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="WebRTC signaling server port (default: 8080)"
    )
    
    parser.add_argument(
        "--room",
        default="demo",
        help="WebRTC room name (default: demo)"
    )
    
    parser.add_argument(
        "--cfg",
        help="Config TXT file for WebRTC mode (required for webrtc mode)"
    )
    
    args = parser.parse_args()
    
    # Validate mode-specific requirements
    if args.mode == "file" and not args.output:
        parser.error("--output is required when --mode is 'file'")
    
    if args.mode == "webrtc" and not args.cfg:
        parser.error("--cfg is required when --mode is 'webrtc'")
    
    # Select and run backend
    if args.backend == "python":
        print("=" * 60)
        print("🐍 [PYTHON BACKEND] Loading Python processing module...")
        print("=" * 60)
        from speedflow_python.run_python import run_python_mode
        run_python_mode(args)
        
    elif args.backend == "cpp":
        print("=" * 60)
        print("🔧 [C++ BACKEND] Loading C++ GStreamer plugin...")
        print("=" * 60)
        from speedflow_cpp.pipeline_cpp import run_cpp_mode
        run_cpp_mode(args)

    else:
        # Should never happen because argparse restricts choices,
        # but guard explicitly so future backends can't slip through silently.
        raise ValueError(f"Unsupported backend: '{args.backend}'")


if __name__ == "__main__":
    main()
