#!/usr/bin/env python3
# speedflow/plate_preprocessor.py
"""
License Plate Preprocessing Probe
Improves plate detection and OCR accuracy by enhancing image quality
"""
import cv2
import numpy as np
import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


class PlatePreprocessorProbe:
    """
    Preprocessing probe attached BEFORE SGIE1 (License Plate Detector)
    to enhance image quality for better plate detection and OCR.
    
    Techniques applied:
    1. Sharpening: Enhance edges for better detection
    2. Contrast Enhancement: Make text more visible
    3. Denoising: Reduce motion blur impact
    """
    
    def __init__(self, enable_sharpening=True, enable_contrast=True, enable_denoise=True):
        self.enable_sharpening = enable_sharpening
        self.enable_contrast = enable_contrast
        self.enable_denoise = enable_denoise
        self.processed_count = 0
        
        # Sharpening kernel (Laplacian-based)
        self.sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
    def preprocess_image(self, image_bgr):
        """
        Apply preprocessing to enhance license plate visibility.
        
        Args:
            image_bgr: OpenCV BGR image
            
        Returns:
            Enhanced BGR image
        """
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr
        
        enhanced = image_bgr.copy()
        
        # 1. Denoising (bilateral filter - preserves edges)
        if self.enable_denoise:
            enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
        
        # 2. Sharpening (enhance edges)
        if self.enable_sharpening:
            enhanced = cv2.filter2D(enhanced, -1, self.sharpen_kernel)
        
        # 3. Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        if self.enable_contrast:
            # Convert to LAB color space
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back to BGR
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def buffer_probe(self, pad, info, u_data):
        """
        GStreamer pad probe callback.
        Processes the buffer before it reaches SGIE1.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            l_frame = batch_meta.frame_meta_list
            
            while l_frame:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                
                # Get surface (GPU memory) and convert to CPU numpy array
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frame_copy = np.array(n_frame, copy=True, order='C')
                
                # Convert RGBA to BGR if needed
                if frame_copy.ndim == 3 and frame_copy.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
                elif frame_copy.ndim == 2:
                    frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame_copy
                
                # Apply preprocessing
                enhanced = self.preprocess_image(frame_bgr)
                
                # Convert back to RGBA
                if frame_copy.shape[2] == 4:
                    enhanced_rgba = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGBA)
                else:
                    enhanced_rgba = enhanced
                
                # Copy enhanced image back to surface
                # Note: This is a simplified approach. For production, use CUDA
                np.copyto(n_frame, enhanced_rgba)
                
                self.processed_count += 1
                
                l_frame = l_frame.next
                
        except Exception as e:
            # Silently continue on error to not break pipeline
            pass
        
        return Gst.PadProbeReturn.OK


class SimplePlateEnhancer:
    """
    Lightweight enhancement for license plate regions (applied to cropped plates in SGIE1).
    This can be used as a preprocessing function in nvinfer config.
    """
    
    @staticmethod
    def enhance_plate_crop(image_bgr):
        """
        Quick enhancement for cropped license plate images.
        Optimized for speed (used per detection).
        
        Args:
            image_bgr: Cropped plate image (BGR)
            
        Returns:
            Enhanced BGR image
        """
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr
        
        # 1. Convert to grayscale for processing
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. Adaptive thresholding to enhance contrast
        # This works well for text on license plates
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 4. Convert back to BGR
        enhanced_bgr = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        
        return enhanced_bgr
    
    @staticmethod
    def unsharp_mask(image, sigma=1.5, strength=1.5):
        """
        Unsharp masking for sharpening.
        
        Args:
            image: Input image
            sigma: Gaussian blur sigma
            strength: Sharpening strength
            
        Returns:
            Sharpened image
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened
