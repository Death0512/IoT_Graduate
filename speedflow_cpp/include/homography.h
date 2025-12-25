/**
 * @file homography.h
 * @brief Homography transformation for pixel to world coordinate conversion
 */

#ifndef HOMOGRAPHY_H
#define HOMOGRAPHY_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class HomographyTransform {
public:
    HomographyTransform();
    
    /**
     * @brief Load homography configuration from YAML file
     * @param yaml_path Path to YAML file containing SOURCE/TARGET points
     * @return true if loaded successfully
     */
    bool load_config(const std::string& yaml_path);
    
    /**
     * @brief Transform a point from pixel space to world space
     * @param pixel_x X coordinate in pixels
     * @param pixel_y Y coordinate in pixels
     * @return Point in world coordinates (meters)
     */
    cv::Point2f transform_point(float pixel_x, float pixel_y);
    
    /**
     * @brief Transform multiple points
     * @param pixel_points Vector of pixel coordinates
     * @return Vector of world coordinates
     */
    std::vector<cv::Point2f> transform_points(const std::vector<cv::Point2f>& pixel_points);
    
    /**
     * @brief Get the source polygon points (for ROI visualization)
     * @return Vector of source polygon points
     */
    std::vector<cv::Point2f> get_source_polygon() const;
    
    /**
     * @brief Check if homography is valid
     */
    bool is_valid() const { return is_valid_; }

private:
    cv::Mat homography_matrix_;
    std::vector<cv::Point2f> source_points_;
    std::vector<cv::Point2f> target_points_;
    float target_width_;
    float target_height_;
    bool is_valid_;
};

#endif /* HOMOGRAPHY_H */
