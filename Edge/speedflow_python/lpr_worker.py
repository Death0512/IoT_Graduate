"""
speedflow_python/lpr_worker.py

Local LPR (license-plate recognition) worker for the edge pipeline.

Runs the LPR TensorRT engine on plate crops that were previously decoded by
sgie2 (the in-pipeline secondary classifier).  After Phase 1 removed sgie2
from the DeepStream graph, plate crops are now extracted in
`osd_sink_pad_buffer_probe` (probes.py) and fed here for inference on a
dedicated worker thread pool — decoupling LPR from the real-time pipeline and
allowing multi-frame voting.

The TensorRT engine loader (`_TRTEngine`), preprocessing (`_preprocess_lpr`)
and CTC decoder (`_decode_lpr_output`) live here so that
`offload_receiver.py` can reuse them without duplicating code (single source
of truth for the LPR inference contract).

TensorRT / pycuda are imported lazily and guarded by try/except so this module
still imports and `py_compile`s on dev/CI hosts without a GPU.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BUG-7 fix: module-level label cache protected by a lock.  Concurrent worker
# threads calling _decode_lpr_output must never see a partially-initialised
# label list.
_lpr_labels_cache: dict = {}          # labels_path → List[str]
_lpr_labels_lock = threading.Lock()


def _safe_infer_reason(exc: BaseException, limit: int = 120) -> str:
    """One-line exception summary for logs: type + message, truncated.
    Never includes a traceback — no stack spam when a crashed engine fails
    on every crop."""
    text = f"{type(exc).__name__}: {exc}".strip()
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


# ---------------------------------------------------------------------------
# TensorRT engine loader (lazy, cached)
# ---------------------------------------------------------------------------

def _try_import_trt():
    """Return (trt, cuda) or (None, None) if not installed."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        cuda.init()
        return trt, cuda
    except Exception:
        return None, None


class _TRTEngine:
    """
    Minimal TensorRT engine wrapper for single-batch inference.
    Loads a serialised .engine file produced by trtexec / DeepStream.
    """

    def __init__(self, engine_path: str) -> None:
        trt, cuda = _try_import_trt()
        if trt is None:
            raise RuntimeError("tensorrt / pycuda not installed")

        logger.info("[TRTEngine] Loading %s ...", engine_path)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self._engine = runtime.deserialize_cuda_engine(engine_data)
        self._context = self._engine.create_execution_context()
        self._trt = trt
        self._cuda = cuda

        # CUDA context affinity: bind this engine to the primary context of
        # device 0 and make it current for the whole setup + every infer/close.
        # Without an explicit context, TRT enqueueV3 raises
        # "Cuda Runtime (invalid resource handle)" because the stream/buffers
        # were created on a different (autoinit) context than the one active at
        # inference time on the worker thread.
        self._dev = cuda.Device(0)
        self._cuda_ctx = self._dev.retain_primary_context()
        self._cuda_ctx.push()
        try:
            # Allocate pinned host + device buffers.  JetPack 5 / TensorRT 8 uses
            # the legacy binding-index API; JetPack 6 / TensorRT 10 uses the
            # name-based tensor API.  Support both so Jetson deployments can move
            # between JetPack releases without breaking crop offload inference.
            self._bindings = []
            self._tensor_names: list = []
            self._use_trt10_api = hasattr(self._engine, "num_io_tensors")
            self._host_inputs: list = []
            self._host_outputs: list = []
            self._dev_inputs: list = []
            self._dev_outputs: list = []
            self._input_shapes: list = []
            self._output_shapes: list = []

            if self._use_trt10_api:
                self._allocate_trt10(trt, cuda)
            else:
                self._allocate_trt8(trt, cuda)

            if not self._host_inputs:
                raise RuntimeError(f"TensorRT engine has no input bindings/tensors: {engine_path}")

            self._stream = cuda.Stream()
            logger.info("[TRTEngine] Loaded %s — %d I/O tensor(s)", engine_path, len(self._bindings))
        finally:
            self._cuda_ctx.pop()

    def close(self) -> None:
        """Explicitly free CUDA device memory and destroy CUDA streams."""
        if hasattr(self, "_cuda_ctx") and self._cuda_ctx is not None:
            try:
                self._cuda_ctx.push()
            except Exception:
                pass
        try:
            if hasattr(self, "_dev_inputs") and self._dev_inputs:
                for dev in self._dev_inputs:
                    try:
                        dev.free()
                    except Exception:
                        pass
                self._dev_inputs.clear()

            if hasattr(self, "_dev_outputs") and self._dev_outputs:
                for dev in self._dev_outputs:
                    try:
                        dev.free()
                    except Exception:
                        pass
                self._dev_outputs.clear()

            if hasattr(self, "_stream") and self._stream is not None:
                try:
                    # pycuda stream cleanup if applicable
                    del self._stream
                    self._stream = None
                except Exception:
                    pass

            if hasattr(self, "_context") and self._context is not None:
                try:
                    del self._context
                    self._context = None
                except Exception:
                    pass

            if hasattr(self, "_engine") and self._engine is not None:
                try:
                    del self._engine
                    self._engine = None
                except Exception:
                    pass
        finally:
            if hasattr(self, "_cuda_ctx") and self._cuda_ctx is not None:
                try:
                    self._cuda_ctx.pop()
                except Exception:
                    pass
                try:
                    self._cuda_ctx.detach()
                except Exception:
                    pass
                self._cuda_ctx = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _allocate_trt10(self, trt, cuda) -> None:
        """TensorRT 10 name-based tensor API with dynamic profile 0 support."""
        n_io = int(self._engine.num_io_tensors)
        # First pass: collect tensor info and resolve dynamic INPUT shapes via profile 0
        tensor_infos: list[dict] = []
        for i in range(n_io):
            name = self._engine.get_tensor_name(i)
            tensor_infos.append({
                "name": name,
                "shape": tuple(self._engine.get_tensor_shape(name)),
                "dtype": trt.nptype(self._engine.get_tensor_dtype(name)),
                "mode": self._engine.get_tensor_mode(name),
            })

        # Resolve dynamic input shapes using profile 0 (MIN shape for batch-1 crops)
        for info in tensor_infos:
            if info["mode"] != trt.TensorIOMode.INPUT:
                continue
            shape = info["shape"]
            # Check if this input has dynamic dimensions
            if any(dim < 0 for dim in shape):
                name = info["name"]
                min_shape, opt_shape, max_shape = self._engine.get_tensor_profile_shape(name, 0)
                logger.info("[TRTEngine] Dynamic input '%s': min=%s opt=%s max=%s",
                            name, min_shape, opt_shape, max_shape)
                # Use MIN shape (index 0) for batch=1 crop preprocessing
                if any(dim <= 0 for dim in min_shape):
                    raise RuntimeError(
                        f"TensorRT profile 0 min shape for input '{name}' contains "
                        f"non-positive dimension(s): {min_shape}"
                    )
                info["shape"] = tuple(min_shape)
                # Require the context to accept the concrete input shape.
                if not self._context.set_input_shape(name, info["shape"]):
                    raise RuntimeError(
                        f"TensorRT set_input_shape failed for input '{name}' "
                        f"with shape {info['shape']} (profile 0 min)"
                    )
            # else: static input, shape already concrete

        # After all input shapes are set, validate that the context considers
        # all binding shapes specified (TRT 10+ provides all_binding_shapes_specified).
        # Explicit False -> RuntimeError. Missing attribute -> compatible (skip check).
        all_specified = getattr(self._context, "all_binding_shapes_specified", None)
        if all_specified is False:
            raise RuntimeError(
                "TensorRT context reports all_binding_shapes_specified=False. "
                "Not all required input shapes have been set."
            )

        # Second pass: now all input shapes are set, resolve output shapes from context
        for info in tensor_infos:
            name = info["name"]
            shape = info["shape"]
            dtype = info["dtype"]
            host_mem = None
            dev_mem = None
            if info["mode"] == trt.TensorIOMode.INPUT:
                # Input shape already resolved above
                host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                self._host_inputs.append(host_mem)
                self._dev_inputs.append(dev_mem)
                self._input_shapes.append(shape)
            else:
                # Output: get concrete shape from context after input shapes are set
                resolved_shape = tuple(self._context.get_tensor_shape(name))
                if any(dim <= 0 for dim in resolved_shape):
                    raise RuntimeError(
                        f"TensorRT context resolved non-positive dimension for output "
                        f"'{name}': {resolved_shape}. Ensure all dynamic inputs have "
                        f"valid shapes set before allocation."
                    )
                shape = resolved_shape
                host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                self._host_outputs.append(host_mem)
                self._dev_outputs.append(dev_mem)
                self._output_shapes.append(shape)

            self._bindings.append(int(dev_mem))
            self._tensor_names.append(name)
            self._context.set_tensor_address(name, int(dev_mem))

    def _allocate_trt8(self, trt, cuda) -> None:
        """TensorRT 8 binding-index API with dynamic profile 0 support."""
        binding_count = int(self._engine.num_bindings)

        # First pass: collect binding info
        binding_infos: list[dict] = []
        for i in range(binding_count):
            binding_infos.append({
                "index": i,
                "shape": tuple(self._engine.get_binding_shape(i)),
                "dtype": trt.nptype(self._engine.get_binding_dtype(i)),
                "is_input": self._engine.binding_is_input(i),
            })

        # Resolve dynamic input shapes using profile 0 (MIN shape for batch-1 crops)
        for info in binding_infos:
            if not info["is_input"]:
                continue
            shape = info["shape"]
            if any(dim < 0 for dim in shape):
                i = info["index"]
                min_shape, opt_shape, max_shape = self._engine.get_profile_shape(0, i)
                logger.info("[TRTEngine] Dynamic binding %d: min=%s opt=%s max=%s",
                            i, min_shape, opt_shape, max_shape)
                # Use MIN shape (index 0) for batch=1 crop preprocessing
                if any(dim <= 0 for dim in min_shape):
                    raise RuntimeError(
                        f"TensorRT profile 0 min shape for binding {i} contains "
                        f"non-positive dimension(s): {min_shape}"
                    )
                info["shape"] = tuple(min_shape)
                # Require the context to accept the concrete input shape.
                if not self._context.set_binding_shape(i, info["shape"]):
                    raise RuntimeError(
                        f"TensorRT set_binding_shape failed for binding {i} "
                        f"with shape {info['shape']} (profile 0 min)"
                    )
            # else: static input, shape already concrete

        # After all input shapes are set, validate that the context considers
        # all binding shapes specified (TRT 10+ provides all_binding_shapes_specified).
        # Explicit False -> RuntimeError. Missing attribute -> compatible (skip check).
        all_specified = getattr(self._context, "all_binding_shapes_specified", None)
        if all_specified is False:
            raise RuntimeError(
                "TensorRT context reports all_binding_shapes_specified=False. "
                "Not all required input shapes have been set."
            )

        # Second pass: allocate with resolved shapes
        for info in binding_infos:
            i = info["index"]
            shape = info["shape"]
            dtype = info["dtype"]
            host_mem = None
            dev_mem = None
            if info["is_input"]:
                # Input shape already resolved above
                host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                self._host_inputs.append(host_mem)
                self._dev_inputs.append(dev_mem)
                self._input_shapes.append(shape)
            else:
                # Output: get concrete shape from context
                resolved_shape = tuple(self._context.get_binding_shape(i))
                if any(dim <= 0 for dim in resolved_shape):
                    raise RuntimeError(
                        f"TensorRT context resolved non-positive dimension for output "
                        f"binding {i}: {resolved_shape}. Ensure all dynamic inputs have "
                        f"valid shapes set before allocation."
                    )
                shape = resolved_shape
                host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                self._host_outputs.append(host_mem)
                self._dev_outputs.append(dev_mem)
                self._output_shapes.append(shape)

            self._bindings.append(int(dev_mem))

    def infer(self, input_array: np.ndarray) -> list:
        """Run one inference pass. Returns list of output ndarrays."""
        expected_size = int(self._host_inputs[0].size)
        actual_size = int(input_array.size)
        if actual_size != expected_size:
            raise ValueError(
                f"Input array size {actual_size} does not match engine input 0 "
                f"size {expected_size} (shape {self._input_shapes[0] if self._input_shapes else '?'}; "
                f"preprocess crop to the engine's expected shape"
            )
        self._cuda_ctx.push()
        try:
            np.copyto(self._host_inputs[0], input_array.ravel())
            self._cuda.memcpy_htod_async(self._dev_inputs[0], self._host_inputs[0], self._stream)
            if self._use_trt10_api:
                self._context.execute_async_v3(stream_handle=self._stream.handle)
            else:
                self._context.execute_async_v2(self._bindings, self._stream.handle, None)
            for host, dev in zip(self._host_outputs, self._dev_outputs):
                self._cuda.memcpy_dtoh_async(host, dev, self._stream)
            self._stream.synchronize()
            return [
                np.array(host).reshape(shape)
                for host, shape in zip(self._host_outputs, self._output_shapes)
            ]
        finally:
            self._cuda_ctx.pop()


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_lpr(crop_bgr: np.ndarray, target_h: int = 48, target_w: int = 96) -> np.ndarray:
    """
    Resize + normalise a plate crop to the LPR model input format.
    Matches the preprocessing used during LPRNet training:
      - resize to (W, H)
      - convert to float32 in [0, 1]
      - channel-first layout: (1, 3, H, W)
    """
    resized = cv2.resize(crop_bgr, (target_w, target_h))
    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    return np.expand_dims(img, axis=0)          # (1, 3, H, W)


def _decode_lpr_output(outputs: list, labels_path: str) -> Tuple[str, float]:
    """
    CTC greedy decode: collapse repeats, skip blank.
    Mirrors the C++ NvDsInferParseCustomNVPlate logic in nvdsinfer_custom_impl_lpr.cpp.

    Returns (plate_text, avg_confidence).

    BUG-7 fix: labels are cached in a module-level dict protected by a lock
    so concurrent worker threads never see a partially-initialised label list.
    """
    # Thread-safe label loading
    with _lpr_labels_lock:
        if labels_path not in _lpr_labels_cache:
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    _lpr_labels_cache[labels_path] = [
                        l.rstrip() for l in f if l.strip()
                    ]
            except Exception:
                _lpr_labels_cache[labels_path] = []
        labels = _lpr_labels_cache[labels_path]

    blank = len(labels)

    # outputs[0]: argmax sequence  (seqLen,) int32
    # outputs[1]: max probs        (seqLen,) float32
    argmax_seq = outputs[0].ravel().astype(int)
    max_probs = outputs[1].ravel().astype(float) if len(outputs) > 1 else None

    plate = ""
    total_c = 0.0
    count = 0
    prev_idx = -1

    for t, idx in enumerate(argmax_seq):
        if idx == blank or idx < 0:
            prev_idx = idx
            continue
        if idx == prev_idx:
            continue
        prev_idx = idx

        conf = float(max_probs[t]) if max_probs is not None else 1.0
        if conf < 0.3:
            continue
        if idx < len(labels):
            plate += labels[idx]
        total_c += conf
        count += 1

    avg_conf = total_c / count if count > 0 else 0.0
    return plate, avg_conf


# ---------------------------------------------------------------------------
# LocalLprWorker — local crop LPR off the DeepStream pipeline
# ---------------------------------------------------------------------------

class LocalLprWorker:
    """
    Runs LPR inference on plate crops fed from the OSD probe, on a dedicated
    worker thread pool, so the real-time DeepStream graph (sgie2 removed in
    Phase 1) is never blocked by TRT execution.

    Crops are submitted non-blocking.  Decoded results are pushed to a result
    sink (typically the probe's voting handler) as dicts:

        {"stid", "camera_id", "frame_no", "plate_text",
         "confidence", "ts", "inference_ok", "vote"}

    The CUDA engine is wrapped by a single lock so concurrent worker threads
    never race on the TRT execution context.
    """

    def __init__(self, engine_path: str, labels_path: str,
                 num_workers: int = 1, maxsize: int = 64) -> None:
        self._engine_path = engine_path
        self._labels_path = labels_path
        self._num_workers = max(1, int(num_workers))
        self._maxsize = int(maxsize)

        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=self._maxsize)
        self._peak_depth: int = 0
        self._depth_lock = threading.Lock()
        self._submit_full_dropped: int = 0
        self._engine: Optional[_TRTEngine] = None
        self._engine_lock = threading.Lock()
        self._threads: list = []
        self._running = False
        self._result_sink: Optional[Callable[[dict], None]] = None
        # Three-valued: None=not probed, True=ok, False=unavailable
        self._trt_available: Optional[bool] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._engine = self._load_engine()
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"LocalLprWorker-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        logger.info(
            "[LocalLprWorker] Started: workers=%d maxsize=%d engine=%s",
            self._num_workers, self._maxsize,
            "ready" if self._engine is not None else "UNAVAILABLE",
        )

    def stop(self) -> None:
        self._running = False
        for _ in self._threads:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None

    def _load_engine(self) -> Optional[_TRTEngine]:
        trt, _ = _try_import_trt()
        if trt is None:
            self._trt_available = False
            logger.error(
                "[LocalLprWorker] tensorrt / pycuda NOT installed — LPR inference "
                "suppressed. Install JetPack TensorRT bindings and restart."
            )
            return None
        self._trt_available = True
        try:
            return _TRTEngine(self._engine_path)
        except Exception as exc:
            self._trt_available = False
            logger.error("[LocalLprWorker] LPR engine load failed: %s", _safe_infer_reason(exc))
            return None

    # -- submission --------------------------------------------------------
    def submit(self, stid, camera_id, frame_no, crop_bgr: np.ndarray,
               conf: float, vote: bool = True) -> bool:
        """Queue a crop non-blocking. Returns False if the queue is full."""
        try:
            self._queue.put_nowait((stid, camera_id, frame_no, crop_bgr, conf, vote))
            with self._depth_lock:
                cur = self._queue.qsize()
                if cur > self._peak_depth:
                    self._peak_depth = cur
            return True
        except queue.Full:
            self._submit_full_dropped += 1
            return False

    def set_result_sink(self, fn: Callable[[dict], None]) -> None:
        self._result_sink = fn

    def queue_depth(self) -> int:
        """Return peak queue depth since last read, then reset to current."""
        with self._depth_lock:
            cur = self._queue.qsize()
            peak = max(cur, self._peak_depth)
            self._peak_depth = cur
            return peak

    def queue_depth_ratio(self) -> float:
        if self._maxsize <= 0:
            return 0.0
        return self.queue_depth() / self._maxsize

    def queue_snapshot(self) -> "tuple[int, float]":
        """Single-sample (depth, ratio). Both queue_depth() and queue_depth_ratio()
        reset the peak, so reading them separately starves the second call; this
        takes one snapshot and returns both under a single lock acquisition."""
        with self._depth_lock:
            cur = self._queue.qsize()
            peak = max(cur, self._peak_depth)
            self._peak_depth = cur
            ratio = (peak / self._maxsize) if self._maxsize > 0 else 0.0
            return peak, ratio

    def submit_full_dropped(self) -> int:
        """Cumulative crops dropped because the submit queue was full."""
        return self._submit_full_dropped

    # -- worker ------------------------------------------------------------
    def run_lpr(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Synchronous LPR for the offload receiver (Phase 3).  Reuses the shared
        engine + CUDA lock so the receiver never loads a second TRT engine.
        """
        return self._infer(crop_bgr)

    def _infer(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        if self._engine is None:
            return "", 0.0
        with self._engine_lock:
            try:
                inp = _preprocess_lpr(crop_bgr)
                outputs = self._engine.infer(inp)
                return _decode_lpr_output(outputs, self._labels_path)
            except Exception as exc:
                logger.warning(
                    "[LocalLprWorker] infer error: %s", _safe_infer_reason(exc)
                )
                return "", 0.0

    def _worker_loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            stid, camera_id, frame_no, crop_bgr, conf, vote = item
            text, _dec_conf = self._infer(crop_bgr)
            result = {
                "stid": stid,
                "camera_id": camera_id,
                "frame_no": frame_no,
                "plate_text": text,
                "confidence": conf,
                "ts": time.time(),
                "inference_ok": True,
                "vote": vote,
            }
            if self._result_sink is not None:
                try:
                    self._result_sink(result)
                except Exception as exc:
                    logger.warning("[LocalLprWorker] result_sink error: %s", exc)


if __name__ == "__main__":
    # Lightweight self-check — runs without a GPU / TRT engine present.
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    captured: list = []

    def _sink(r: dict) -> None:
        captured.append(r)

    w = LocalLprWorker(
        engine_path="/nonexistent_lpr.engine",
        labels_path="/nonexistent_labels.txt",
        num_workers=1,
        maxsize=8,
    )
    w.set_result_sink(_sink)
    w.start()

    dummy = np.zeros((32, 96, 3), dtype=np.uint8)
    ok = w.submit("cam1:5", "cam1", 1, dummy, 0.9, vote=True)
    assert ok, "submit must succeed on an empty queue"
    assert w.queue_depth() == 1, "queue depth must be 1 after submit"

    for _ in range(100):
        if captured:
            break
        time.sleep(0.05)

    w.stop()
    assert captured, "result_sink must be called for the submitted crop"
    print("SELFCHECK_OK results=%d submit_ok=%s depth_ratio=%.2f"
          % (len(captured), ok, w.queue_depth_ratio()))
