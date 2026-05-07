#!/usr/bin/env python3
# speedflow/plate_preprocessor.py
"""
License Plate Preprocessing Probe
Improves plate detection and OCR accuracy by enhancing image quality.

PERFORMANCE FIX: Enhancement is now applied only to cropped vehicle bounding
boxes, NOT to the full 1920×1080 frame.  Bilateral filtering a full frame at
60 fps is prohibitively expensive on CPU (~100 ms/frame); cropping first
reduces the processed area by ~50-100×.
"""
import cv2
import numpy as np
import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


class PlatePreprocessorProbe:
    """
    Preprocessing probe attached BEFORE SGIE1 (License Plate Detector).

    Enhancement pipeline per vehicle crop:
      1. Denoising    – bilateral filter (edge-preserving)
      2. Sharpening   – convolution kernel adaptive to motion blur level
      3. Contrast     – CLAHE on the L channel of LAB color space

    The probe reads vehicle bounding boxes from NvDsObjectMeta and only
    processes those regions, keeping CPU load proportional to the number of
    detected vehicles rather than resolution.
    """

    # COCO class IDs that correspond to vehicles
    VEHICLE_CLASS_IDS = {2, 3, 5, 7}

    def __init__(
        self,
        enable_sharpening: bool = True,
        enable_contrast: bool = True,
        enable_denoise: bool = True,
        adaptive_mode: bool = True,
    ) -> None:
        self.enable_sharpening = enable_sharpening
        self.enable_contrast = enable_contrast
        self.enable_denoise = enable_denoise
        self.adaptive_mode = adaptive_mode
        self.processed_count = 0

        # Sharpening kernels for different motion blur levels
        self.sharpen_kernel_light = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0],
        ], dtype=np.float32)

        self.sharpen_kernel_medium = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1],
        ], dtype=np.float32)

        self.sharpen_kernel_strong = np.array([
            [-1, -2, -1],
            [-2, 13, -2],
            [-1, -2, -1],
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Public probe callback
    # ------------------------------------------------------------------

    def buffer_probe(self, pad, info, u_data):
        """
        GStreamer pad probe callback.
        Processes vehicle crops before the buffer reaches SGIE1.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            l_frame = batch_meta.frame_meta_list

            while l_frame:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

                # ── Decode surface once per frame ─────────────────────────
                n_frame = pyds.get_nvds_buf_surface(
                    hash(gst_buffer), frame_meta.batch_id
                )
                frame_copy = np.array(n_frame, copy=True, order='C')

                # Convert NVMM surface format (RGBA or GRAY) to BGR
                if frame_copy.ndim == 3 and frame_copy.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
                elif frame_copy.ndim == 2:
                    frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame_copy

                h_frame, w_frame = frame_bgr.shape[:2]
                modified = False  # track whether we need to write back

                # ── Process only vehicle bbox regions ─────────────────────
                l_obj = frame_meta.obj_meta_list
                while l_obj:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                    if obj_meta.class_id in self.VEHICLE_CLASS_IDS:
                        crop, x, y, x2, y2 = self._crop_vehicle(
                            frame_bgr, obj_meta, w_frame, h_frame
                        )
                        if crop is not None and crop.size > 0:
                            enhanced = self.preprocess_image(crop)
                            frame_bgr[y:y2, x:x2] = enhanced
                            modified = True

                    l_obj = l_obj.next

                # ── Write modified frame back to GPU surface ───────────────
                if modified:
                    if frame_copy.ndim == 3 and frame_copy.shape[2] == 4:
                        enhanced_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
                    else:
                        enhanced_rgba = frame_bgr
                    np.copyto(n_frame, enhanced_rgba)

                # VERY IMPORTANT: Unmap the surface buffer to sync back to GPU
                # and release the lock so other elements (like SGIE) can read it properly.
                pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)

                self.processed_count += 1
                l_frame = l_frame.next

        except Exception as e:
            # Never break the pipeline on preprocessing errors
            print(f"[PlatePreprocessorProbe] Error: {e}")


        return Gst.PadProbeReturn.OK

    # ------------------------------------------------------------------
    # Enhancement logic
    # ------------------------------------------------------------------

    def preprocess_image(self, image_bgr: np.ndarray, motion_level: str = 'medium') -> np.ndarray:
        """
        Apply adaptive enhancement to a *cropped* vehicle or plate region.

        Args:
            image_bgr:    OpenCV BGR image (vehicle crop).
            motion_level: 'low' | 'medium' | 'high'.  Auto-detected when
                          adaptive_mode is True.

        Returns:
            Enhanced BGR image (same shape as input).
        """
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        if self.adaptive_mode and motion_level == 'medium':
            motion_level = self._estimate_motion_blur(image_bgr)

        enhanced = image_bgr.copy()

        # ── Choose parameters based on blur level ─────────────────────────
        if motion_level == 'low':
            denoise_d, denoise_sigma = 3, 30
            sharpen_kernel = self.sharpen_kernel_light
            clahe_clip = 1.5
        elif motion_level == 'high':
            denoise_d, denoise_sigma = 7, 70
            sharpen_kernel = self.sharpen_kernel_strong
            clahe_clip = 2.5
        else:  # medium (default)
            denoise_d, denoise_sigma = 5, 50
            sharpen_kernel = self.sharpen_kernel_medium
            clahe_clip = 2.0

        # 1. Edge-preserving denoising
        if self.enable_denoise:
            enhanced = cv2.bilateralFilter(
                enhanced, d=denoise_d,
                sigmaColor=denoise_sigma,
                sigmaSpace=denoise_sigma,
            )

        # 2. Adaptive sharpening
        if self.enable_sharpening:
            enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)

        # 3. Adaptive contrast (CLAHE on L channel only)
        if self.enable_contrast:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)
            lab = cv2.merge([l_ch, a_ch, b_ch])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_vehicle(
        frame_bgr: np.ndarray,
        obj_meta,
        w_frame: int,
        h_frame: int,
    ):
        """
        Safely extract the vehicle bounding box from the full frame.

        Returns:
            (crop, x, y, x2, y2) if valid, otherwise (None, 0, 0, 0, 0).
        """
        x  = int(round(obj_meta.rect_params.left))
        y  = int(round(obj_meta.rect_params.top))
        bw = int(round(obj_meta.rect_params.width))
        bh = int(round(obj_meta.rect_params.height))

        x  = max(0, x);  y  = max(0, y)
        x2 = min(w_frame, x + max(1, bw))
        y2 = min(h_frame, y + max(1, bh))

        if x >= x2 or y >= y2:
            return None, 0, 0, 0, 0

        return frame_bgr[y:y2, x:x2], x, y, x2, y2

    @staticmethod
    def _estimate_motion_blur(image_bgr: np.ndarray) -> str:
        """
        Estimate motion blur level via Laplacian variance.
        Lower variance → more blur.

        Returns: 'low' | 'medium' | 'high'
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance > 500:
            return 'low'
        if variance > 200:
            return 'medium'
        return 'high'


class SimplePlateEnhancer:
    """
    Lightweight enhancement for cropped license plate images.
    Can be used as a preprocessing step on individual plate crops.
    """

    @staticmethod
    def enhance_plate_crop(image_bgr: np.ndarray) -> np.ndarray:
        """
        Quick contrast/clarity enhancement for a cropped plate image.

        Args:
            image_bgr: Cropped plate image (BGR).

        Returns:
            Enhanced BGR image.
        """
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def unsharp_mask(image: np.ndarray, sigma: float = 1.5, strength: float = 1.5) -> np.ndarray:
        """Apply unsharp masking to sharpen the image."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
