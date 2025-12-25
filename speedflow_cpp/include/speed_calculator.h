/**
 * @file speed_calculator.h
 * @brief Speed calculation and validation logic
 */

#ifndef SPEED_CALCULATOR_H
#define SPEED_CALCULATOR_H

#include <deque>
#include <unordered_map>
#include <opencv2/opencv.hpp>

class SpeedCalculator {
public:
    SpeedCalculator(int video_fps, int min_track_age_frames,
                   float min_displacement_m, float max_speed_kmh,
                   float bbox_area_jump, float min_conf, int median_window);
    
    /**
     * @brief Update position history for a track
     * @param track_id Vehicle track ID
     * @param y_world Y coordinate in world space (meters)
     * @param frame_num Current frame number
     */
    void update_history(uint64_t track_id, float y_world, int frame_num);
    
    /**
     * @brief Calculate speed for a track
     * @param track_id Vehicle track ID
     * @return Speed in km/h or -1 if not enough data
     */
    float calculate_speed(uint64_t track_id);
    
    /**
     * @brief Validate speed measurement
     * @param track_id Vehicle track ID
     * @param speed_kmh Calculated speed
     * @param frame_num Current frame number
     * @param bbox_area Current bbox area
     * @param confidence Detection confidence
     * @return true if measurement is valid
     */
    bool validate_measurement(uint64_t track_id, float speed_kmh,
                             int frame_num, float bbox_area, float confidence);
    
    /**
     * @brief Get smoothed speed using median filter
     * @param track_id Vehicle track ID
     * @param raw_speed Raw speed value
     * @return Smoothed speed
     */
    float get_smoothed_speed(uint64_t track_id, float raw_speed);
    
    /**
     * @brief Register track birth frame
     * @param track_id Vehicle track ID
     * @param frame_num Birth frame number
     */
    void register_track(uint64_t track_id, int frame_num);
    
    /**
     * @brief Update last bbox area for track
     */
    void update_bbox_area(uint64_t track_id, float area);
    
    /**
     * @brief Clean up old tracks that haven't been seen for a while
     * @param current_frame Current frame number
     * @param max_age Maximum frames since last seen
     */
    void cleanup_old_tracks(int current_frame, int max_age);

private:
    int video_fps_;
    int min_track_age_frames_;
    float min_displacement_m_;
    float max_speed_kmh_;
    float bbox_area_jump_;
    float min_conf_;
    int median_window_;
    
    // Track ID -> position history (y_world values)
    std::unordered_map<uint64_t, std::deque<float>> position_history_;
    
    // Track ID -> birth frame
    std::unordered_map<uint64_t, int> track_birth_frame_;
    
    // Track ID -> last bbox area
    std::unordered_map<uint64_t, float> last_bbox_area_;
    
    // Track ID -> speed history for median filter
    std::unordered_map<uint64_t, std::deque<float>> speed_history_;
    
    // Track ID -> last seen frame
    std::unordered_map<uint64_t, int> last_seen_frame_;
    
    /**
     * @brief Compute median of a deque
     */
    float compute_median(std::deque<float>& values);
};

#endif /* SPEED_CALCULATOR_H */
