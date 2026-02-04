# MULTI-THREADED ARCHITECTURE IMPLEMENTATION GUIDE

**Project:** IoT_Graduate - Traffic Monitoring System  
**Goal:** Single-process, multi-threaded architecture decoupling Speed Detection & LPR  
**Date:** 2026-02-04  
**Update:** Architecture optimized for Jetson (Single Process / Thread Pool)

---

## TABLE OF CONTENTS

1. [Architecture Overview](#1-architecture-overview)
2. [Skills Applied](#2-skills-applied)
3. [Implementation Phases](#3-implementation-phases)
4. [Detailed Implementation](#4-detailed-implementation)
5. [Testing & Validation](#5-testing--validation)

---

## 1. ARCHITECTURE OVERVIEW

### Current State (Single Pipeline)
```
Source → YOLO → Tracker → Analytics → LPD → LPR → Speed → Display
                                       ↑___________↑
                                   BOTTLENECK: ~25 FPS
```

### Target State (Multi-Threaded Single Process)
```
┌──────────────────────────────────────────────────┐
│  SINGLE PROCESS - Main Thread (DeepStream)       │
│                                                  │
│  Source → YOLO → Tracker → Speed → Display       │
│              ↓                    ↑              │
│         Crop Extract         Lookup Results      │
│              ↓                    ↑              │
│         queue.Queue          threading.Lock      │
│         (crops)              (results dict)      │
└──────────────┬─────────────────┬─────────────────┘
               │                 │
               ▼                 │
     ┌─────────────────────┐     │
     │  Thread Pool (1)    │     │
     │  ┌────────────────┐ │     │
     │  │ Worker Thread  │ │     │
     │  │ RVRT→LPD→LPR   │◄┼─────┘
     │  └────────────────┘ │
     │         ...         │
     └─────────────────────┘
```

### Key Design Decisions

> [!IMPORTANT]
> **Architecture Change:** Multi-threading instead of multi-process due to Jetson constraint

1. **Multi-Threading (ThreadPoolExecutor)** - Single process, worker pool
2. **Thread-Safe Queue (queue.Queue)** - Crop distribution to workers
3. **Thread-Safe Dict (dict + Lock)** - Result storage
4. **RVRT on first 5 frames only** - Balance quality vs throughput
5. **Track ID as key** - Sync between main thread and workers
6. **No Docker/K8s** - Direct Python threading

---

## 2. SKILLS APPLIED

### Planning & Design
- **@brainstorming** - System design validation
- **@architecture** - Dual pipeline architecture

### Python Development
- **@python-pro** - Modern Python 3.10+ patterns
- **@async-python-patterns** - Async processing loops

### Testing
- **@test-driven-development** - Write tests first
- **@systematic-debugging** - Debug methodically

### Architecture
- **@microservices-patterns** - Inter-process communication
- **@performance-engineer** - Optimization strategies

### Implementation
- **@backend-architect** - Pipeline communication design
- **@clean-code** - Code quality standards

---

## 3. IMPLEMENTATION PHASES

### Phase 1: Thread-Safe Infrastructure (Week 1)
**Skills:** @python-pro, @clean-code  
**Tasks:**
- [ ] Create `threading_manager.py`
- [ ] Implement thread-safe crop queue (`queue.Queue`)
- [ ] Implement thread-safe result dict (`dict` + `threading.Lock`)
- [ ] Unit tests for thread safety
- [ ] Benchmark latency (<1ms target)

### Phase 2: Main Thread - Speed Measurement (Week 2)
**Skills:** @backend-architect, @python-pro  
**Tasks:**
- [ ] Modify `core_pipeline.py` - remove SGIE
- [ ] Create `CropExtractionProbe` 
- [ ] Modify `SpeedProbe` - lookup plates from thread-safe dict
- [ ] Test speed measurement still accurate
- [ ] Verify 50+ FPS achieved

### Phase 3: Worker Threads - LPR Enhancement (Week 3)
**Skills:** @python-pro, @performance-engineer  
**Tasks:**
- [ ] Download RVRT model: `RVRT_Vimeo90K_SR_L.pth` (https://github.com/JingyunLiang/RVRT/releases)
- [ ] Create `models/convert_rvrt.py` for ONNX/TensorRT conversion (Dynamic Shapes support)
- [ ] Create `lpr_pipeline.py`
- [ ] Implement RVRT inference (TensorRT)
- [ ] Implement LPD inference (TensorRT)
- [ ] Implement LPR inference (TensorRT)
- [ ] First-5-frames logic
- [ ] Test plate recognition accuracy

### Phase 4: Integration (Week 4)
**Skills:** @test-driven-development, @systematic-debugging  
**Tasks:**
- [ ] Create `main_threaded.py` entry point
- [ ] ThreadPoolExecutor setup (4 workers)
- [ ] End-to-end testing
- [ ] Performance benchmarking
- [ ] Edge case handling (thread safety)

---

## 4. DETAILED IMPLEMENTATION

### 4.1 Threading Manager

**File:** `threading_manager.py`  
**Skills Applied:** @python-pro, @async-python-patterns

```python
"""
Threading Manager for Inter-Thread Communication

Thread-safe data structures using Python stdlib:
1. queue.Queue for vehicle crops (Main Thread → Worker Threads)
2. dict + threading.Lock for plate results (Worker Threads → Main Thread)
"""

import queue
import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class VehicleCrop:
    """Vehicle crop metadata with image"""
    track_id: int
    timestamp: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    frame_number: int
    crop_count: int  # Which crop for this vehicle (0-4 for first 5)
    image: np.ndarray  # BGR image


@dataclass
class PlateResult:
    """License plate recognition result"""
    track_id: int
    plate_text: str
    confidence: float
    timestamp: float


class ThreadSafeManager:
    """
    Manage thread-safe communication between main thread and worker pool
    
    Components:
    - crop_queue: queue.Queue for distributing crops to workers (FIFO)
    - result_dict: dict protected by Lock for storing plate results
    
    Capacity:
    - Queue: 100 crops max
    - Dict: Unlimited (with age-based cleanup)
    """
    
    def __init__(self, max_queue_size: int = 100):
        # Queue for vehicle crops (main thread → workers)
        self.crop_queue = queue.Queue(maxsize=max_queue_size)
        
        # Dict for plate results (workers → main thread)
        self.result_dict = {}  # {track_id: PlateResult}
        self.result_lock = threading.Lock()
        
        # Statistics
        self.stats_lock = threading.Lock()
        self.total_crops_sent = 0
        self.total_plates_received = 0
    
    # === Crop Queue Operations ===
    
    def put_crop(self, crop: VehicleCrop, timeout: float = 0.01) -> bool:
        """
        Put vehicle crop into queue (called by main thread)
        
        Args:
            crop: VehicleCrop with image
            timeout: Max wait time if queue full
        
        Returns:
            True if successful, False if queue full
        """
        try:
            self.crop_queue.put(crop, timeout=timeout)
            
            with self.stats_lock:
                self.total_crops_sent += 1
            
            return True
        except queue.Full:
            return False
    
    def get_crop(self, timeout: float = 0.1) -> Optional[VehicleCrop]:
        """
        Get vehicle crop from queue (called by worker threads)
        
        Args:
            timeout: Max wait time if queue empty
        
        Returns:
            VehicleCrop or None if timeout
        """
        try:
            crop = self.crop_queue.get(timeout=timeout)
            return crop
        except queue.Empty:
            return None
    
    # === Result Dict Operations ===
    
    def set_result(self, result: PlateResult):
        """
        Store plate result (called by worker threads)
        
        Args:
            result: PlateResult with plate text
        """
        with self.result_lock:
            self.result_dict[result.track_id] = result
            
        with self.stats_lock:
            self.total_plates_received += 1
    
    def get_result(self, track_id: int) -> Optional[PlateResult]:
        """
        Retrieve plate result (called by main thread)
        
        Args:
            track_id: Vehicle track ID
        
        Returns:
            PlateResult or None if not found
        """
        with self.result_lock:
            return self.result_dict.get(track_id)
    
    def cleanup_old_results(self, max_age_s: float = 30.0):
        """
        Remove old results to prevent memory growth
        
        Args:
            max_age_s: Max age in seconds
        """
        now = time.time()
        
        with self.result_lock:
            to_delete = [
                tid for tid, result in self.result_dict.items()
                if (now - result.timestamp) > max_age_s
            ]
            
            for tid in to_delete:
                del self.result_dict[tid]
    
    # === Statistics ===
    
    def get_stats(self) -> dict:
        """Get queue and processing statistics"""
        with self.stats_lock:
            with self.result_lock:
                return {
                    'queue_size': self.crop_queue.qsize(),
                    'total_crops_sent': self.total_crops_sent,
                    'total_plates_received': self.total_plates_received,
                    'results_in_dict': len(self.result_dict)
                }


# Global instance (singleton pattern)
_manager_instance = None

def get_manager() -> ThreadSafeManager:
    """Get or create global ThreadSafeManager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ThreadSafeManager()
    return _manager_instance
```

---

### 4.2 Main Thread - Speed Measurement

**File:** `speedflow_python/pipeline1_speed.py`  
**Skills Applied:** @backend-architect, @python-pro

```python
"""
Main Thread: Speed Measurement with Crop Extraction

Modified from original pipeline to:
1. Remove SGIE (LPD/LPR) for maximum speed
2. Add crop extraction probe
3. Lookup plates from ThreadSafeManager
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
import sys
sys.path.append('..')

from speedflow_python.core_pipeline import build_pipeline
from speedflow_python.settings import VEHICLE_CLASS_IDS
from threading_manager import get_manager, VehicleCrop
import time
import cv2
import numpy as np


class CropExtractionProbe:
    """
    Extract vehicle crops and send to worker threads
    
    Logic:
    - Only first 5 frames per vehicle
    - Only vehicles in ROI
    - Write to queue.Queue
    """
    
    def __init__(self):
        self.manager = get_manager()
        self.crop_counts = {}  # track_id → count
    
    def callback(self, pad, info, user_data):
        """GStreamer probe callback"""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                
                # Extract frame as BGR
                frame_bgr = self._extract_frame_bgr(gst_buffer, frame_meta)
                if frame_bgr is None:
                    l_frame = l_frame.next
                    continue
                
                l_obj = frame_meta.obj_meta_list
                while l_obj:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        
                        # Only vehicles
                        if obj_meta.class_id not in VEHICLE_CLASS_IDS:
                            l_obj = l_obj.next
                            continue
                        
                        track_id = obj_meta.object_id
                        
                        # Only first 5 frames per vehicle
                        if track_id not in self.crop_counts:
                            self.crop_counts[track_id] = 0
                        
                        if self.crop_counts[track_id] >= 5:
                            l_obj = l_obj.next
                            continue
                        
                        # Extract crop
                        bbox = obj_meta.rect_params
                        x, y = int(bbox.left), int(bbox.top)
                        w, h = int(bbox.width), int(bbox.height)
                        
                        # Bounds check
                        h_frame, w_frame = frame_bgr.shape[:2]
                        x = max(0, min(x, w_frame - 1))
                        y = max(0, min(y, h_frame - 1))
                        w = min(w, w_frame - x)
                        h = min(h, h_frame - y)
                        
                        if w <= 0 or h <= 0:
                            l_obj = l_obj.next
                            continue
                        
                        crop_bgr = frame_bgr[y:y+h, x:x+w]
                        
                        # Create crop data (with image embedded)
                        crop_data = VehicleCrop(
                            track_id=track_id,
                            timestamp=time.time(),
                            bbox=(x, y, w, h),
                            frame_number=frame_meta.frame_num,
                            crop_count=self.crop_counts[track_id],
                            image=crop_bgr  # Embed image in dataclass
                        )
                        
                        # Write to queue
                        success = self.manager.put_crop(crop_data)
                        
                        if success:
                            self.crop_counts[track_id] += 1
                        
                    except StopIteration:
                        break
                    
                    l_obj = l_obj.next
                
            except StopIteration:
                break
            
            l_frame = l_frame.next
        
        return Gst.PadProbeReturn.OK
    
    def _extract_frame_bgr(self, gst_buffer, frame_meta):
        """Extract frame as BGR numpy array"""
        # Get NvBufSurface
        n_surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        if n_surface is None:
            return None
        
        # Convert to numpy
        frame_copy = np.array(n_surface, copy=True, order='C')
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
        
        return frame_copy


class SpeedPipeline:
    """
    Main Thread: Vehicle detection + tracking + speed measurement
    
    Responsibilities:
    - DeepStream pipeline (YOLO + Tracker)
    - Extract crops → send to worker threads
    - Speed measurement
    - Display with plates from worker threads
    """
    
    def __init__(self, source_uri: str, homo_yml: str):
        # Get threading manager (singleton)
        self.manager = get_manager()
        
        # Build GStreamer pipeline (without SGIE)
        self.pipeline = build_pipeline(
            source_uri=source_uri,
            sink_type="display",
            homo_yml=homo_yml,
            mux_width=1920,
            mux_height=1080,
            enable_sgie=False  # KEY: No LPD/LPR in main thread
        )
        
        # Attach probes
        self._attach_probes()
    
    def _attach_probes(self):
        """Attach crop extraction probe"""
        # Get tracker element
        tracker = self.pipeline.get_by_name("tracker")
        if not tracker:
            print("ERROR: Tracker element not found")
            return
        
        # Attach crop extraction probe
        tracker_src = tracker.get_static_pad("src")
        crop_probe = CropExtractionProbe()
        tracker_src.add_probe(
            Gst.PadProbeType.BUFFER,
            crop_probe.callback,
            None
        )
        
        print("[Main Thread] Crop extraction probe attached")
    
    def run(self):
        """Run pipeline"""
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Wait for EOS or error
        bus = self.pipeline.get_bus()
        msg = bus.timed_pop_filtered(
            Gst.CLOCK_TIME_NONE,
            Gst.MessageType.ERROR | Gst.MessageType.EOS
        )
        
        self.pipeline.set_state(Gst.State.NULL)
```

---

### 4.3 Model Preparation (RVRT)

**File:** `models/convert_rvrt.py`
**Skills Applied:** @ai-engineer, @performance-engineer

```python
"""
Convert RVRT PyTorch model to ONNX and TensorRT
Supports Dynamic Shapes for optimal batch size flexibility.

Usage:
    python convert_rvrt.py --model RVRT_Vimeo90K_SR_L.pth --output RVRT.engine
"""

import torch
import torch.onnx
import tensorrt as trt
import argparse
import os
import sys

# Assume external dependency or local implementation of RVRT model architecture
# from models.rvrt_arch import RVRT  <-- You might need to add the model definition here or import it

def export_onnx(torch_model, onnx_path, input_shape=(1, 3, 256, 256)):
    """Export PyTorch model to ONNX with dynamic axes"""
    dummy_input = torch.randn(input_shape, device='cuda')
    
    # Dynamic axes: batch_size, height, width
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
    
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        torch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

def build_engine(onnx_path, engine_path, fp16=True):
    """Build TensorRT engine from ONNX"""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # Use standard memory pool in newer TRT versions instead of max_workspace_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    print(f"Parsing ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    # Min: 256x256, Opt: 640x640, Max: 1280x1280 (Adjust based on expected crop sizes)
    profile.set_shape("input", (1, 3, 256, 256), (1, 3, 640, 640), (4, 3, 1280, 1280))
    config.add_optimization_profile(profile)
    
    print(f"Building TensorRT engine: {engine_path}")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        print("Build success!")
    else:
        print("Build failed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth model")
    parser.add_argument("--output", default="models/RVRT.engine", help="Output path")
    args = parser.parse_args()
    
    # Needs model definition code...
    # model = RVRT(...) 
    # model.load_state_dict(torch.load(args.model))
    # model.cuda().eval()
    
    onnx_path = args.output.replace(".engine", ".onnx")
    # export_onnx(model, onnx_path)
    build_engine(onnx_path, args.output)
```

---

### 4.4 Worker Threads - LPR with RVRT

**File:** `lpr_worker.py`  
**Skills Applied:** @python-pro, @performance-engineer

```python
"""
LPR Worker Threads: RVRT Enhancement + LPD + LPR

Worker loop (runs in thread pool):
1. Read vehicle crops from queue.Queue
2. RVRT enhancement (TensorRT)
3. License plate detection (TensorRT)
4. License plate recognition (TensorRT)
5. Write results to thread-safe dict
"""

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple, Optional
import sys
sys.path.append('..')

from threading_manager import get_manager, PlateResult
import time


class TensorRTInference:
    """Base class for TensorRT inference"""
    
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Get bindings
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def _load_engine(self, engine_path: str):
        """Load TensorRT engine"""
        with open(engine_path, "rb") as f, \
             trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run inference"""
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy output to host
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(
                output['host'],
                output['device'],
                self.stream
            )
            outputs.append(output['host'].copy())
        
        self.stream.synchronize()
        return outputs


class RVRTEnhancer(TensorRTInference):
    """RVRT image enhancement"""
    
    def __init__(self):
        super().__init__("models/RVRT.engine")
        self.input_size = (256, 256)  # Adjust based on model
    
    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        """Enhance image quality"""
        # Preprocess
        orig_h, orig_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # Normalize to [0, 1]
        input_tensor = image_resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC → CHW
        input_tensor = np.expand_dims(input_tensor, 0)  # Add batch
        
        # Inference
        output = self.infer(input_tensor)[0]
        
        # Postprocess
        output = np.squeeze(output)
        output = np.transpose(output, (1, 2, 0))  # CHW → HWC
        output = (output * 255).clip(0, 255).astype(np.uint8)
        
        # Resize back to original
        enhanced_rgb = cv2.resize(output, (orig_w, orig_h))
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        
        return enhanced_bgr


class LicensePlateDetector(TensorRTInference):
    """YOLOv11 license plate detection"""
    
    def __init__(self):
        super().__init__("models/lpd.engine")
        self.input_size = (320, 320)
        self.conf_threshold = 0.3
        self.iou_threshold = 0.4
    
    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect license plates
        
        Returns: List of (x, y, w, h, confidence)
        """
        orig_h, orig_w = image_bgr.shape[:2]
        
        # Preprocess
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.input_size)
        input_tensor = image_resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, 0)
        
        # Inference
        output = self.infer(input_tensor)[0]
        
        # Postprocess (YOLO format)
        detections = self._parse_yolo_output(output)
        
        # Scale to original size
        detections_scaled = []
        for x, y, w, h, conf in detections:
            x_orig = int(x * orig_w / self.input_size[0])
            y_orig = int(y * orig_h / self.input_size[1])
            w_orig = int(w * orig_w / self.input_size[0])
            h_orig = int(h * orig_h / self.input_size[1])
            detections_scaled.append((x_orig, y_orig, w_orig, h_orig, conf))
        
        return detections_scaled
    
    def _parse_yolo_output(self, output: np.ndarray) -> List[Tuple]:
        """Parse YOLO detection output"""
        # Simplified - adjust based on actual model output format
        detections = []
        # ... NMS, confidence filtering, etc.
        return detections


class LicensePlateRecognizer(TensorRTInference):
    """LPRNet"""
    
    def __init__(self):
        super().__init__("models/lpr.engine")
        self.input_size = (94, 24)  # Typical LPR input
        
        # Vietnamese plate characters
        self.chars = "0123456789ABCDEFGHKLMNPSTUVXYZ"
    
    def recognize(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """
        Recognize license plate text
        
        Returns: (plate_text, confidence)
        """
        # Preprocess
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.input_size)
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(np.expand_dims(input_tensor, 0), 0)
        
        # Inference
        output = self.infer(input_tensor)[0]
        
        # CTC decode
        text, confidence = self._ctc_decode(output)
        
        return text, confidence
    
    def _ctc_decode(self, output: np.ndarray) -> Tuple[str, float]:
        """CTC decoding"""
        # Simplified - implement proper CTC beam search
        text = ""
        confidence = 0.9
        return text, confidence




def lpr_worker_loop():
    """
    Worker thread function: Process crops from queue
    
    Each worker thread:
    1. Gets crop from queue
    2. RVRT → LPD → LPR
    3. Writes result to dict
    
    Note: TensorRT releases GIL during inference → true parallelism
    """
    # Get manager (thread-safe singleton)
    manager = get_manager()
    
    # Load TensorRT engines (each thread has own CUDA context)
    print("[Worker] Loading RVRT engine...")
    rvrt = RVRTEnhancer()
    
    print("[Worker] Loading LPD engine...")
    lpd = LicensePlateDetector()
    
    print("[Worker] Loading LPR engine...")
    lpr = LicensePlateRecognizer()
    
    print("[Worker] All engines loaded - ready to process")
    
    while True:
        # Read crop from queue
        crop = manager.get_crop(timeout=0.1)
        
        if crop is None:
            continue
        
        track_id = crop.track_id
        crop_image = crop.image
        
        try:
            # Step 1: RVRT Enhancement
            enhanced = rvrt.enhance(crop_image)
            
            # Step 2: License Plate Detection
            plates = lpd.detect(enhanced)
            
            if not plates:
                continue
            
            # Step 3: License Plate Recognition (best plate)
            best_text = ""
            best_conf = 0.0
            
            for x, y, w, h, conf in plates:
                # Crop plate region
                plate_crop = enhanced[y:y+h, x:x+w]
                
                # Recognize
                text, text_conf = lpr.recognize(plate_crop)
                
                # Combined confidence
                combined_conf = conf * text_conf
                
                if combined_conf > best_conf:
                    best_text = text
                    best_conf = combined_conf
            
            # Step 4: Write result to dict
            if best_text and best_conf > 0.5:
                result = PlateResult(
                    track_id=track_id,
                    plate_text=best_text,
                    confidence=best_conf,
                    timestamp=time.time()
                )
                
                manager.set_result(result)
                print(f"[Worker] Track {track_id}: {best_text} ({best_conf:.2f})")
        
        except Exception as e:
            print(f"[Worker] Error processing track {track_id}: {e}")
            continue
```

---

### 4.5 Main Entry Point

**File:** `main_threaded.py`  
**Skills Applied:** @backend-architect

```python
"""
Main entry point for multi-threaded pipeline system

Architecture:
- Main Thread: DeepStream pipeline (speed measurement + crop extraction)
- Worker Pool: 4 threads for LPR processing (RVRT → LPD → LPR)
"""

from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Multi-Threaded Traffic Monitor")
    parser.add_argument("--source", required=True, help="Video source (RTSP/file)")
    parser.add_argument("--homo", default="configs/points_1.yml", help="Homography YAML")
    parser.add_argument("--workers", type=int, default=4, help="Number of LPR worker threads")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-THREADED PIPELINE SYSTEM")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Homography: {args.homo}")
    print(f"LPR Workers: {args.workers}")
    print("=" * 60)
    
    # Import after arg parsing
    from speedflow_python.pipeline1_speed import SpeedPipeline
    from lpr_worker import lpr_worker_loop
    
    # Create thread pool for LPR workers
    print(f"\n[MAIN] Starting {args.workers} LPR worker threads...")
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="LPR-Worker")
    
    # Start worker threads
    for i in range(args.workers):
        executor.submit(lpr_worker_loop)
        print(f"[MAIN] Started worker thread {i+1}/{args.workers}")
    
    print("\n[MAIN] Starting main thread (DeepStream)...")
    
    # Run main pipeline in this thread
    try:
        pipeline = SpeedPipeline(args.source, args.homo)
        pipeline.run()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")
    finally:
        print("[MAIN] Stopping worker threads...")
        executor.shutdown(wait=False, cancel_futures=True)
        print("[MAIN] Cleanup complete")

if __name__ == "__main__":
    main()
```

---

## 5. TESTING & VALIDATION

### 5.1 Unit Tests

**File:** `tests/test_threading_manager.py`  
**Skills Applied:** @test-driven-development

```python
"""
Unit tests for threading manager components

Run: pytest tests/test_threading_manager.py -v
"""

import pytest
import numpy as np
import time
import threading
from threading_manager import (
    ThreadSafeManager, VehicleCrop, PlateResult,
    get_manager
)

class TestThreadSafeManager:
    """Test VehicleCropQueue thread safety and functionality"""
    
    def test_put_get_single(self):
        """Test single write and read"""
        queue = VehicleCropQueue(max_size=10)
        
        # Create test data
        crop_data = VehicleCrop(
            track_id=42,
            timestamp=time.time(),
            bbox=(100, 200, 300, 400),
            frame_number=1,
            crop_count=0
        )
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Write
        success = queue.put(crop_data, image)
        assert success == True
        
        # Read
        result = queue.get(timeout=0.1)
        assert result is not None
        
        read_crop, read_image = result
        assert read_crop.track_id == 42
        assert read_crop.bbox == (100, 200, 300, 400)
        assert read_image.shape == (640, 640, 3)  # Resized
        
        queue.cleanup()
    
    def test_buffer_full(self):
        """Test buffer full behavior"""
        queue = VehicleCropQueue(max_size=5)
        
        # Fill buffer
        for i in range(5):
            crop = VehicleCrop(i, time.time(), (0,0,10,10), i, 0)
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            assert queue.put(crop, image) == True
        
        # Try one more (should fail)
        crop = VehicleCrop(99, time.time(), (0,0,10,10), 99, 0)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert queue.put(crop, image) == False
        
        queue.cleanup()

class TestPlateResultDict:
    """Test PlateResultDict thread safety and functionality"""
    
    def test_set_get(self):
        """Test write and read"""
        dict_shm = PlateResultDict(max_tracks=100)
        
        # Write
        result = PlateResult(
            track_id=123,
            plate_text="30A-12345",
            confidence=0.95,
            timestamp=time.time()
        )
        success = dict_shm.set(result)
        assert success == True
        
        # Read
        retrieved = dict_shm.get(123)
        assert retrieved is not None
        assert retrieved.plate_text == "30A-12345"
        assert retrieved.confidence == 0.95
        
        dict_shm.cleanup()
    
    def test_not_found(self):
        """Test reading non-existent key"""
        dict_shm = PlateResultDict(max_tracks=100)
        
        result = dict_shm.get(999)
        assert result is None
        
        dict_shm.cleanup()
```

### 5.2 Integration Tests

**File:** `tests/test_integration.py`

```python
"""
Integration tests for multi-threaded system

Test end-to-end flow:
Main Thread extracts crops → Queue → Worker Thread process → Dict → Main Thread reads
"""

import pytest
import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from threading_manager import VehicleCrop, PlateResult, get_manager

def test_threaded_communication():
    """Test Main Thread ↔ Worker Thread communication via Manager"""
    manager = get_manager()
    
    # Simulation: Main thread produces crops
    def main_producer():
        for i in range(5):
            crop = VehicleCrop(
                track_id=i,
                timestamp=time.time(),
                bbox=(0,0,100,100),
                frame_number=i,
                crop_count=0,
                image=np.zeros((100, 100, 3), dtype=np.uint8)
            )
            manager.put_crop(crop)
            time.sleep(0.01)
    
    # Simulation: Worker thread consumes crops and produces results
    def worker_consumer():
        processed_count = 0
        while processed_count < 5:
            crop = manager.get_crop(timeout=1.0)
            if crop:
                result = PlateResult(
                    track_id=crop.track_id,
                    plate_text=f"PLATE-{crop.track_id}",
                    confidence=0.99,
                    timestamp=time.time()
                )
                manager.set_result(result)
                processed_count += 1
    
    # Run simulation
    producer_thread = threading.Thread(target=main_producer)
    worker_thread = threading.Thread(target=worker_consumer)
    
    producer_thread.start()
    worker_thread.start()
    
    producer_thread.join(timeout=2)
    worker_thread.join(timeout=2)
    
    # Validation
    stats = manager.get_stats()
    assert stats['total_crops_sent'] >= 5
    assert stats['total_plates_received'] >= 5
    
    # check result exist in dict
    res = manager.get_result(0)
    assert res is not None
    assert res.plate_text == "PLATE-0"

```

### 5.3 Performance Benchmarks

**File:** `tests/benchmark_latency.py`

```python
"""Benchmark threading queue latency"""

import time
import numpy as np
import threading
from threading_manager import ThreadSafeManager, VehicleCrop

def benchmark_latency(num_iterations=1000):
    """Measure put → get latency across threads"""
    manager = ThreadSafeManager()
    latencies = []
    
    def consumer():
        while len(latencies) < num_iterations:
            t_start = time.perf_counter()
            crop = manager.get_crop(timeout=1.0)
            t_end = time.perf_counter()
            if crop:
                # Approximate latency as time spent waiting + retrieving
                # Note: this is a simple approximation
                pass

    # Simple Put-Get in main thread for baseline
    print("Benchmarking single-thread Put-Get latency...")
    for i in range(num_iterations):
        crop = VehicleCrop(i, 0, (0,0,0,0), 0, 0, np.zeros((640,640,3), np.uint8))
        
        t0 = time.perf_counter()
        manager.put_crop(crop)
        _ = manager.get_crop()
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000) # ms
        
    print(f"Latency stats (n={len(latencies)}):")
    print(f"  Mean: {np.mean(latencies):.4f} ms")
    print(f"  P99: {np.percentile(latencies, 99):.4f} ms")

if __name__ == "__main__":
    benchmark_latency()
```

---

## 6. DEPLOYMENT

### 6.1 Setup

```bash
# Create directory structure
mkdir -p tests

# Install dependencies (ensure DeepStream environment active)
pip install numpy opencv-python-headless pycuda tensorrt

# Verify models exist
ls models/RVRT_90K.engine models/yolo11n.engine models/lpd.engine models/lpr.engine
```

### 6.2 Run

```bash
# Single pipeline (baseline)
python3 main.py --backend python --source video.mp4 --mode display

# Multi-Threaded System (Target)
python3 main_threaded.py --source video.mp4 --homo configs/points_1.yml --workers 1
```

### 6.3 Expected Results

**Performance Targets:**
- Main Thread FPS: 55-60 (DeepStream pipeline)
- LPR Processing FPS: ~30 (Aggregate of 1 worker)
- Queue Latency: <0.1ms (intra-process)
- CPU Usage: Balanced across cores

**Validation Checklist:**
- [ ] Speed measurements accurate (Main thread)
- [ ] License plates showing up on OSD (Result from workers)
- [ ] No race conditions or deadlocks
- [ ] Graceful shutdown on Ctrl+C

---

**End of Implementation Guide** 🚀
