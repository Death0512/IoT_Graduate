/**
 * @file image_enhancer.cpp
 * @brief Image preprocessing implementation for C++ backend
 */

#include "image_enhancer.h"
#include <cmath>

ImageEnhancer::ImageEnhancer(bool enable_sharpening, bool enable_contrast,
                           bool enable_denoise, bool adaptive_mode)
    : enable_sharpening_(enable_sharpening)
    , enable_contrast_(enable_contrast)
    , enable_denoise_(enable_denoise)
    , adaptive_mode_(adaptive_mode) {
    
    // Light sharpen kernel
    sharpen_kernel_light_ = (cv::Mat_<float>(3, 3) << 
        0, -1, 0,
        -1, 5, -1,
        0, -1,  0
    );
    
    // Medium sharpen kernel
    sharpen_kernel_medium_ = (cv::Mat_<float>(3, 3) << 
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1
    );
    
    // Strong sharpen kernel
    sharpen_kernel_strong_ = (cv::Mat_<float>(3, 3) << 
        -1, -2, -1,
        -2, 13, -2,
        -1, -2, -1
    );
}

std::string ImageEnhancer::estimate_motion_blur(const cv::Mat& image) {
    if (image.empty()) {
        return "medium";
    }
    
    // Convert to grayscale
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Apply Laplacian
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    // Calculate variance
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double variance = stddev[0] * stddev[0];
    
    // Classify blur level
    if (variance > 500.0) {
        return "low";      // Sharp image
    } else if (variance > 200.0) {
        return "medium";   // Moderate blur
    } else {
        return "high";     // Significant blur
    }
}

cv::Mat ImageEnhancer::preprocess_image(const cv::Mat& image, 
                                       const std::string& motion_level) {
    if (image.empty()) {
        return image;
    }
    
    // Auto-detect motion level if adaptive
    std::string level = motion_level;
    if (adaptive_mode_ && level == "medium") {
        level = estimate_motion_blur(image);
    }
    
    cv::Mat enhanced = image.clone();
    
    // === ADAPTIVE PARAMETERS ===
    int denoise_d;
    double denoise_sigma;
    cv::Mat sharpen_kernel;
    double clahe_clip;
    
    if (level == "low") {
        denoise_d = 3;
        denoise_sigma = 30.0;
        sharpen_kernel = sharpen_kernel_light_;
        clahe_clip = 1.5;
    } else if (level == "medium") {
        denoise_d = 5;
        denoise_sigma = 50.0;
        sharpen_kernel = sharpen_kernel_medium_;
        clahe_clip = 2.0;
    } else {  // high motion
        denoise_d = 7;
        denoise_sigma = 70.0;
        sharpen_kernel = sharpen_kernel_strong_;
        clahe_clip = 2.5;
    }
    
    // 1. Denoising
    if (enable_denoise_) {
        enhanced = apply_denoise(enhanced, denoise_d, denoise_sigma);
    }
    
    // 2. Sharpening
    if (enable_sharpening_) {
        enhanced = apply_sharpen(enhanced, sharpen_kernel);
    }
    
    // 3. Contrast enhancement
    if (enable_contrast_) {
        enhanced = apply_clahe(enhanced, clahe_clip);
    }
    
    return enhanced;
}

cv::Mat ImageEnhancer::apply_denoise(const cv::Mat& image, int d, double sigma) {
    cv::Mat result;
    cv::bilateralFilter(image, result, d, sigma, sigma);
    return result;
}

cv::Mat ImageEnhancer::apply_sharpen(const cv::Mat& image, const cv::Mat& kernel) {
    cv::Mat result;
    cv::filter2D(image, result, -1, kernel);
    return result;
}

cv::Mat ImageEnhancer::apply_clahe(const cv::Mat& image, double clip_limit) {
    // Convert BGR to LAB
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    
    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    
    // Apply CLAHE to L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);
    
    // Merge and convert back
    cv::merge(channels, lab);
    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    
    return result;
}

// Static methods for quick plate enhancement
cv::Mat ImageEnhancer::enhance_plate_crop(const cv::Mat& image) {
    if (image.empty()) {
        return image;
    }
    
    // Convert to grayscale
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Adaptive thresholding
    cv::Mat thresh;
    cv::adaptiveThreshold(gray, thresh, 255, 
                         cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY, 11, 2);
    
    // Morphological closing
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat morph;
    cv::morphologyEx(thresh, morph, cv::MORPH_CLOSE, kernel);
    
    // Convert back to BGR
    cv::Mat result;
    cv::cvtColor(morph, result, cv::COLOR_GRAY2BGR);
    
    return result;
}

cv::Mat ImageEnhancer::unsharp_mask(const cv::Mat& image, float sigma, float strength) {
    if (image.empty()) {
        return image;
    }
    
    // Blur the image
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(0, 0), sigma);
    
    // Subtract blur from original
    cv::Mat sharpened;
    cv::addWeighted(image, 1.0 + strength, blurred, -strength, 0, sharpened);
    
    return sharpened;
}
