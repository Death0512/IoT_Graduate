/**
 * @file homography.cpp
 * @brief Homography transformation implementation
 */

#include "homography.h"
#include <yaml-cpp/yaml.h>
#include <iostream>

HomographyTransform::HomographyTransform() : is_valid_(false) {}

bool HomographyTransform::load_config(const std::string& yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        
        // Load SOURCE points
        if (!config["SOURCE"]) {
            std::cerr << "[Homography] Missing SOURCE in config file" << std::endl;
            return false;
        }
        
        source_points_.clear();
        for (const auto& pt : config["SOURCE"]) {
            float x = pt[0].as<float>();
            float y = pt[1].as<float>();
            source_points_.push_back(cv::Point2f(x, y));
        }
        
        if (source_points_.size() != 4) {
            std::cerr << "[Homography] SOURCE must have exactly 4 points" << std::endl;
            return false;
        }
        
        // Load TARGET points
        if (!config["TARGET"]) {
            std::cerr << "[Homography] Missing TARGET in config file" << std::endl;
            return false;
        }
        
        target_points_.clear();
        for (const auto& pt : config["TARGET"]) {
            float x = pt[0].as<float>();
            float y = pt[1].as<float>();
            target_points_.push_back(cv::Point2f(x, y));
        }
        
        if (target_points_.size() != 4) {
            std::cerr << "[Homography] TARGET must have exactly 4 points" << std::endl;
            return false;
        }
        
        // Load TARGET_WIDTH and TARGET_HEIGHT
        target_width_ = config["TARGET_WIDTH"] ? config["TARGET_WIDTH"].as<float>() : 50.0f;
        target_height_ = config["TARGET_HEIGHT"] ? config["TARGET_HEIGHT"].as<float>() : 100.0f;
        
        // Compute homography matrix
        std::vector<cv::Point2f> src_pts(source_points_.begin(), source_points_.end());
        std::vector<cv::Point2f> tgt_pts(target_points_.begin(), target_points_.end());
        
        homography_matrix_ = cv::getPerspectiveTransform(src_pts, tgt_pts);
        
        is_valid_ = true;
        
        std::cout << "[Homography] Loaded config from " << yaml_path << std::endl;
        std::cout << "[Homography] Target size: " << target_width_ << "m x " << target_height_ << "m" << std::endl;
        
        return true;
        
    } catch (const YAML::Exception& e) {
        std::cerr << "[Homography] YAML error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[Homography] Error: " << e.what() << std::endl;
        return false;
    }
}

cv::Point2f HomographyTransform::transform_point(float pixel_x, float pixel_y) {
    if (!is_valid_) {
        return cv::Point2f(0, 0);
    }
    
    std::vector<cv::Point2f> input_pts = {cv::Point2f(pixel_x, pixel_y)};
    std::vector<cv::Point2f> output_pts;
    
    cv::perspectiveTransform(input_pts, output_pts, homography_matrix_);
    
    return output_pts[0];
}

std::vector<cv::Point2f> HomographyTransform::transform_points(const std::vector<cv::Point2f>& pixel_points) {
    if (!is_valid_ || pixel_points.empty()) {
        return {};
    }
    
    std::vector<cv::Point2f> output_pts;
    cv::perspectiveTransform(pixel_points, output_pts, homography_matrix_);
    
    return output_pts;
}

std::vector<cv::Point2f> HomographyTransform::get_source_polygon() const {
    return source_points_;
}
