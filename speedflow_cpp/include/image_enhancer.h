/**
 * @file image_enhancer.h
 * @brief Image preprocessing for motion blur reduction (C++ backend)
 */

#ifndef IMAGE_ENHANCER_H
#define IMAGE_ENHANCER_H

#include <opencv2/opencv.hpp>
#include <string>

/**
 * @brief Motion blur level estimation and adaptive enhancement
 */
class ImageEnhancer {
public:
    ImageEnhancer(bool enable_sharpening = true, 
                 bool enable_contrast = true,
                 bool enable_denoise = true,
                 bool adaptive_mode = true);
    
    /**
     * @brief Estimate motion blur level using Laplacian variance
     * @param image Input image (BGR)
     * @return Motion level: "low", "medium", or "high"
     */
    std::string estimate_motion_blur(const cv::Mat& image);
    
    /**
     * @brief Adaptive preprocessing based on motion level
     * @param image Input image (BGR)
     * @param motion_level Optional motion level override
     * @return Enhanced image
     */
    cv::Mat preprocess_image(const cv::Mat& image, 
                            const std::string& motion_level = "medium");
    
    /**
     * @brief Quick enhancement for cropped plates (optimized for speed)
     * @param image Cropped plate image
     * @return Enhanced plate image
     */
    static cv::Mat enhance_plate_crop(const cv::Mat& image);
    
    /**
     * @brief Unsharp masking for sharpening
     * @param image Input image
     * @param sigma Gaussian blur sigma
     * @param strength Sharpening strength
     * @return Sharpened image
     */
    static cv::Mat unsharp_mask(const cv::Mat& image, 
                               float sigma = 1.5f, 
                               float strength = 1.5f);

private:
    bool enable_sharpening_;
    bool enable_contrast_;
    bool enable_denoise_;
    bool adaptive_mode_;
    
    // Sharpening kernels for different motion levels
    cv::Mat sharpen_kernel_light_;
    cv::Mat sharpen_kernel_medium_;
    cv::Mat sharpen_kernel_strong_;
    
    /**
     * @brief Apply bilateral filter for denoising
     */
    cv::Mat apply_denoise(const cv::Mat& image, int d, double sigma);
    
    /**
     * @brief Apply sharpening with kernel
     */
    cv::Mat apply_sharpen(const cv::Mat& image, const cv::Mat& kernel);
    
    /**
     * @brief Apply CLAHE for contrast enhancement
     */
    cv::Mat apply_clahe(const cv::Mat& image, double clip_limit);
};

#endif /* IMAGE_ENHANCER_H */
