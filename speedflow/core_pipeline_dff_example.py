# Example: Modified core_pipeline.py with DFF Integration
# 
# This file shows how to integrate DFF optimization into build_pipeline()
# Copy relevant sections to your actual core_pipeline.py

from .settings import (INFER_CONFIG, TRACKER_CFG, ANALYTICS_CFG, SGIE_CONFIG, 
                       TRACKER_LIB, LPR_CONFIG, TRACKER_LPD_CFG,
                       ENABLE_DFF, DFF_INTERVAL, DFF_USE_OFA, 
                       DFF_ADAPTIVE_KEYFRAME, DFF_FLOW_THRESHOLD)
from .dff_optimizer import SimpleDFFOptimizer, DFFOptimizer

# ... (existing imports) ...

def build_pipeline(source_uri: str, sink_type: str = "display", output_path: str = None,
                mux_width: int = 1920, mux_height: int = 1080, is_live: bool = None, 
                analytics_config: str = None, **kwargs):
    """
    Build DeepStream pipeline with optional DFF optimization.
    
    NEW Args:
        enable_dff (bool): Override ENABLE_DFF setting (optional)
        dff_interval (int): Override DFF_INTERVAL setting (optional)
    """
    
    # Override DFF settings from kwargs if provided
    enable_dff = kwargs.get('enable_dff', ENABLE_DFF)
    dff_interval = kwargs.get('dff_interval', DFF_INTERVAL)
    
    # ... (existing pipeline setup code) ...
    
    # Create pipeline
    pipeline = Gst.Pipeline.new(f"ds-pipeline-{sink_type}")
    
    # SOURCE
    source = make_element("source-bin", "uridecodebin")
    source.set_property("uri", uri)
    # ... (existing source setup) ...
    
    # STREAMMUX
    streammux = make_element("stream-muxer", "nvstreammux")
    streammux.set_property('batch-size', 1)
    streammux.set_property('width', mux_width)
    streammux.set_property('height', mux_height)
    # ... (existing mux setup) ...
    
    
    # ========== DFF INTEGRATION START ==========
    
    # Option 1: Add OFA for optical flow (if DFF_USE_OFA=True)
    if enable_dff and DFF_USE_OFA:
        print("[DFF] Adding NVIDIA OFA (Optical Flow Accelerator)")
        ofa = make_element("ofa", "nvofa")
        try:
            ofa.set_property("preset-level", 1)  # 0=quality, 1=balanced, 2=performance
            ofa.set_property("grid-size", 1)     # 1=4x4, 2=8x8, 3=16x16
            print("[DFF] OFA configured: preset=balanced, grid=4x4")
        except Exception as e:
            print(f"[DFF] Warning: OFA configuration failed: {e}")
            print("[DFF] Falling back to simple interval-based DFF")
            DFF_USE_OFA = False
    
    # PGIE (Primary Inference)
    pgie = make_element("primary-infer", "nvinfer")
    pgie.set_property('config-file-path', str(INFER_CONFIG))
    
    # Option 2: Simple interval-based DFF (RECOMMENDED)
    if enable_dff and not DFF_USE_OFA:
        print(f"[DFF] Enabling Simple Interval-based optimization")
        simple_dff = SimpleDFFOptimizer(interval=dff_interval)
        simple_dff.configure_pgie(pgie)
        print(f"[DFF] → PGIE will run every {dff_interval} frames")
        print(f"[DFF] → Theoretical speedup: {dff_interval}x")
        print(f"[DFF] → Expected FPS: {30 * dff_interval} fps (from 30 fps baseline)")
    
    # Option 3: Full DFF with OFA warping (ADVANCED)
    elif enable_dff and DFF_USE_OFA:
        print(f"[DFF] Enabling Full DFF with OFA optical flow warping")
        dff_optimizer = DFFOptimizer(
            keyframe_interval=dff_interval,
            flow_threshold=DFF_FLOW_THRESHOLD,
            enable_adaptive=DFF_ADAPTIVE_KEYFRAME
        )
        
        # Add probes for DFF logic
        # Probe 1: BEFORE PGIE (decide keyframe/warp)
        pgie_sinkpad = pgie.get_static_pad("sink")
        pgie_sinkpad.add_probe(
            Gst.PadProbeType.BUFFER, 
            dff_optimizer.dff_probe, 
            None
        )
        print("[DFF] → Added pre-PGIE probe (keyframe decision)")
        
        # Probe 2: AFTER PGIE (cache detections)
        pgie_srcpad = pgie.get_static_pad("src")
        pgie_srcpad.add_probe(
            Gst.PadProbeType.BUFFER, 
            dff_optimizer.post_pgie_cache_probe, 
            None
        )
        print("[DFF] → Added post-PGIE probe (detection caching)")
        
        # Store optimizer for cleanup
        pipeline.dff_optimizer = dff_optimizer
    
    # ========== DFF INTEGRATION END ==========
    
    
    # SGIE, Tracker, Analytics (unchanged)
    sgie = make_element("secondary-infer", "nvinfer")
    sgie.set_property('config-file-path', str(SGIE_CONFIG))
    
    sgie2 = make_element("lpr-classifier", "nvinfer")
    sgie2.set_property('config-file-path', str(LPR_CONFIG))
    
    tracker = make_element("tracker", "nvtracker")
    tracker.set_property('ll-lib-file', str(TRACKER_LIB))
    tracker.set_property('ll-config-file', str(TRACKER_CFG))
    # ... (existing tracker setup) ...
    
    analytics = make_element("analytics", "nvdsanalytics")
    analytics.set_property('config-file', analytics_config or str(ANALYTICS_CFG))
    
    # ... (rest of pipeline: OSD, sinks, etc.) ...
    
    
    # ========== ADD ELEMENTS TO PIPELINE ==========
    if enable_dff and DFF_USE_OFA:
        # Include OFA in pipeline
        core_elements = [source, streammux, ofa, pgie, tracker, sgie, sgie2, analytics, ...]
    else:
        # Standard pipeline
        core_elements = [source, streammux, pgie, tracker, sgie, sgie2, analytics, ...]
    
    for element in core_elements:
        pipeline.add(element)
    
    
    # ========== LINK ELEMENTS ==========
    
    # ... (existing source → streammux linking) ...
    
    # Link core chain
    if enable_dff and DFF_USE_OFA:
        # With OFA: streammux → OFA → PGIE → tracker → ...
        assert streammux.link(ofa), "Failed to link streammux → ofa"
        assert ofa.link(pgie), "Failed to link ofa → pgie"
        assert pgie.link(tracker), "Failed to link pgie → tracker"
    else:
        # Standard: streammux → PGIE → tracker → ...
        assert streammux.link(pgie), "Failed to link streammux → pgie"
        assert pgie.link(tracker), "Failed to link pgie → tracker"
    
    # Rest of linking (unchanged)
    assert tracker.link(sgie), "Failed to link tracker → sgie"
    assert sgie.link(sgie2), "Failed to link sgie → sgie2"
    assert sgie2.link(analytics), "Failed to link sgie2 → analytics"
    
    # ... (rest of sink linking) ...
    
    
    # Print pipeline summary
    if enable_dff:
        print("\n" + "="*60)
        print("DFF OPTIMIZATION ENABLED")
        print("="*60)
        print(f"Mode:              {'OFA Warping' if DFF_USE_OFA else 'Simple Interval'}")
        print(f"Interval:          {dff_interval} frames")
        print(f"Adaptive keyframe: {DFF_ADAPTIVE_KEYFRAME}")
        if DFF_USE_OFA:
            print(f"Flow threshold:    {DFF_FLOW_THRESHOLD} pixels")
        print(f"Expected speedup:  ~{dff_interval}x")
        print("="*60 + "\n")
    
    # Return pipeline
    if sink_type == "webrtc":
        return pipeline, nvdsosd, webrtc
    else:
        return pipeline, nvdsosd


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Example 1: Enable DFF via settings.py
# In settings.py: ENABLE_DFF = True, DFF_INTERVAL = 10
# Then just call normally:
pipeline, nvdsosd = build_pipeline(source_uri="video.mp4", sink_type="display")

# Example 2: Override DFF settings
pipeline, nvdsosd = build_pipeline(
    source_uri="video.mp4",
    sink_type="display",
    enable_dff=True,      # Force enable
    dff_interval=5        # Use interval=5 for higher accuracy
)

# Example 3: Disable DFF for specific run
pipeline, nvdsosd = build_pipeline(
    source_uri="video.mp4",
    sink_type="display",
    enable_dff=False      # Force disable
)

# Example 4: Test DFF performance
# python3 main.py --source video.mp4 --mode display
# → See FPS in console output
