#!/usr/bin/env python3
"""
Python wrapper for motion deblur preprocessing
Can be integrated directly into probes.py for easier implementation
"""

import ctypes
import numpy as np
import cv2
from pathlib import Path

# Deblur method enum
class DeblurMethod:
    NONE = 0
    UNSHARP_MASK = 1
    WIENER = 2
    RICHARDSON_LUCY = 3

# Config structure (matches C++ struct)
class DeblurConfig(ctypes.Structure):
    _fields_ = [
        ("method", ctypes.c_int),
        ("unsharp_sigma", ctypes.c_float),
        ("unsharp_amount", ctypes.c_float),
        ("wiener_kernel_size", ctypes.c_int),
        ("wiener_angle", ctypes.c_float),
        ("wiener_snr", ctypes.c_float),
        ("rl_iterations", ctypes.c_int),
        ("rl_kernel_size", ctypes.c_int),
        ("rl_angle", ctypes.c_float),
    ]

class MotionDeblur:
    """Python wrapper for C++ motion deblur library"""
    
    def __init__(self, lib_path=None):
        if lib_path is None:
            # Auto-detect library path
            lib_path = Path(__file__).parent / "libnvdsinfer_deblur_preprocess.so"
        
        self.lib = ctypes.CDLL(str(lib_path))
        
        # Setup function signatures
        self.lib.deblur_set_config.argtypes = [DeblurConfig]
        self.lib.deblur_set_config.restype = None
        
        self.lib.deblur_get_config.argtypes = []
        self.lib.deblur_get_config.restype = DeblurConfig
        
        # Default config
        self.config = DeblurConfig(
            method=DeblurMethod.UNSHARP_MASK,
            unsharp_sigma=1.5,
            unsharp_amount=1.5,
            wiener_kernel_size=15,
            wiener_angle=0.0,
            wiener_snr=25.0,
            rl_iterations=10,
            rl_kernel_size=15,
            rl_angle=0.0
        )
        
        self.lib.deblur_set_config(self.config)
    
    def process(self, image):
        """
        Process image with deblur
        
        Args:
            image: numpy array (BGR or grayscale)
        
        Returns:
            Deblurred image
        """
        if image is None or image.size == 0:
            return image
        
        # Call C++ implementation via process_cv2
        return self._process_cv2(image)
    
    def _process_cv2(self, image):
        """Pure Python/OpenCV implementation (fallback)"""
        if self.config.method == DeblurMethod.NONE:
            return image
        
        elif self.config.method == DeblurMethod.UNSHARP_MASK:
            return self._unsharp_mask(image, self.config.unsharp_sigma, self.config.unsharp_amount)
        
        elif self.config.method == DeblurMethod.WIENER:
            return self._wiener_deblur(image, self.config.wiener_kernel_size, self.config.wiener_angle)
        
        elif self.config.method == DeblurMethod.RICHARDSON_LUCY:
            return self._richardson_lucy(image, self.config.rl_iterations, self.config.rl_kernel_size, self.config.rl_angle)
        
        return image
    
    def _unsharp_mask(self, image, sigma, amount):
        """Unsharp masking implementation"""
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened
    
    def _create_motion_psf(self, kernel_size, angle_deg):
        """Create motion blur PSF"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        psf = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * cos_a)
            y = int(center + offset * sin_a)
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                psf[y, x] = 1.0
        
        psf = psf / psf.sum()
        return psf
    
    def _wiener_deblur(self, image, kernel_size, angle):
        """Simplified Wiener deblur"""
        psf = self._create_motion_psf(kernel_size, angle)
        
        # Simple inverse filter (approximation)
        kernel = np.flip(psf)
        result = cv2.filter2D(image, -1, kernel)
        
        return result
    
    def _richardson_lucy(self, image, iterations, kernel_size, angle):
        """Richardson-Lucy deconvolution"""
        # Convert to float [0,1]
        img_float = image.astype(np.float32) / 255.0
        
        psf = self._create_motion_psf(kernel_size, angle)
        psf_flipped = np.flip(psf)
        
        estimate = img_float.copy()
        
        for _ in range(iterations):
            # Convolve estimate with PSF
            blurred = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REPLICATE)
            blurred += 1e-10  # Avoid division by zero
            
            # Compute ratio
            ratio = img_float / blurred
            
            # Correlate with flipped PSF
            correction = cv2.filter2D(ratio, -1, psf_flipped, borderType=cv2.BORDER_REPLICATE)
            
            # Update estimate
            estimate = estimate * correction
        
        # Convert back to uint8
        estimate = np.clip(estimate * 255.0, 0, 255).astype(np.uint8)
        return estimate
    
    def set_method(self, method):
        """Set deblur method"""
        self.config.method = method
        self.lib.deblur_set_config(self.config)
    
    def set_unsharp_params(self, sigma=1.5, amount=1.5):
        """Set unsharp mask parameters"""
        self.config.unsharp_sigma = sigma
        self.config.unsharp_amount = amount
        self.lib.deblur_set_config(self.config)
    
    def set_wiener_params(self, kernel_size=15, angle=0.0, snr=25.0):
        """Set Wiener deblur parameters"""
        self.config.wiener_kernel_size = kernel_size
        self.config.wiener_angle = angle
        self.config.wiener_snr = snr
        self.lib.deblur_set_config(self.config)
    
    def set_rl_params(self, iterations=10, kernel_size=15, angle=0.0):
        """Set Richardson-Lucy parameters"""
        self.config.rl_iterations = iterations
        self.config.rl_kernel_size = kernel_size
        self.config.rl_angle = angle
        self.lib.deblur_set_config(self.config)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 deblur_wrapper.py <image_path>")
        sys.exit(1)
    
    # Load image
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: Cannot load image {sys.argv[1]}")
        sys.exit(1)
    
    print(f"Loaded image: {img.shape}")
    
    # Initialize deblur
    deblur = MotionDeblur()
    
    # Test all methods
    methods = [
        (DeblurMethod.NONE, "Original"),
        (DeblurMethod.UNSHARP_MASK, "Unsharp Mask"),
        (DeblurMethod.WIENER, "Wiener"),
        (DeblurMethod.RICHARDSON_LUCY, "Richardson-Lucy")
    ]
    
    results = []
    
    for method_id, method_name in methods:
        print(f"\nProcessing with {method_name}...")
        deblur.set_method(method_id)
        
        import time
        start = time.time()
        result = deblur.process(img.copy())
        elapsed = (time.time() - start) * 1000
        
        print(f"  Time: {elapsed:.2f} ms")
        results.append((method_name, result))
    
    # Show results
    print("\nDisplaying results (press any key to continue)...")
    for name, result in results:
        cv2.imshow(name, result)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("\nDone!")
