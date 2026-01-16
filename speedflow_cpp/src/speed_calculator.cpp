/**
 * @file speed_calculator.cpp
 * @brief Speed calculation and validation implementation with TIME-BASED tracking
 */

#include "speed_calculator.h"
#include <algorithm>
#include <cmath>
#include <iostream>

SpeedCalculator::SpeedCalculator(int video_fps, int min_track_age_frames,
                                 float min_displacement_m, float max_speed_kmh,
                                 float bbox_area_jump, float min_conf, int median_window)
    : video_fps_(video_fps)
    , min_track_age_frames_(min_track_age_frames)
    , min_displacement_m_(min_displacement_m)
    , max_speed_kmh_(max_speed_kmh)
    , bbox_area_jump_(bbox_area_jump)
    , min_conf_(min_conf)
    , median_window_(median_window) {}

void SpeedCalculator::register_track(uint64_t track_id, int frame_num) {
    if (track_birth_frame_.find(track_id) == track_birth_frame_.end()) {
        track_birth_frame_[track_id] = frame_num;
    }
}

void SpeedCalculator::update_history(uint64_t track_id, const cv::Point2f& world_pos, 
                                     uint64_t timestamp_ns, int frame_num, 
                                     float nvof_magnitude) {
    auto& history = position_history_[track_id];
    
    // Create sample
    PositionSample sample;
    sample.world_pos = world_pos;
    sample.timestamp_ns = timestamp_ns;
    sample.frame_num = frame_num;
    sample.nvof_magnitude = nvof_magnitude;
    
    // Keep 1 second of history based on frame count (fallback if PTS unreliable)
    if (history.size() >= static_cast<size_t>(video_fps_ * 2)) {  // 2 seconds buffer
        history.pop_front();
    }
    
    history.push_back(sample);
    last_seen_frame_[track_id] = frame_num;
}

float SpeedCalculator::calculate_speed(uint64_t track_id) {
    auto it = position_history_.find(track_id);
    if (it == position_history_.end()) {
        return -1.0f;
    }
    
    auto& history = it->second;
    
    // Need at least 2 samples for speed calculation
    if (history.size() < 2) {
        return -1.0f;
    }
    
    // Strategy: Use oldest and newest sample within reasonable window
    // Prefer samples ~1 second apart for stable measurement
    const PositionSample& oldest = history.front();
    const PositionSample& newest = history.back();
    
    // Calculate 2D Euclidean distance in world space (meters)
    float distance_m = euclidean_distance(oldest.world_pos, newest.world_pos);
    
    // Calculate time delta in seconds
    float time_s;
    if (newest.timestamp_ns > oldest.timestamp_ns && newest.timestamp_ns > 0) {
        // Use PTS timestamp (nanoseconds -> seconds)
        time_s = static_cast<float>(newest.timestamp_ns - oldest.timestamp_ns) / 1e9f;
    } else {
        // Fallback to frame-based time (if PTS unavailable or invalid)
        int frame_delta = newest.frame_num - oldest.frame_num;
        time_s = static_cast<float>(frame_delta) / video_fps_;
    }
    
    if (time_s <= 0.01f) {  // Avoid division by very small numbers
        return -1.0f;
    }
    
    // Speed in km/h = (distance_m / time_s) * 3.6
    // m/s -> km/h conversion factor is 3.6
    float speed_mps = distance_m / time_s;
    float speed_kmh = speed_mps * 3.6f;
    
    // NVOF Fusion (optional enhancement)
    // If NVOF data available, we could use it for validation or correction
    // For now, just use geometric tracking (future work: Kalman filter fusion)
    
    return speed_kmh;
}

bool SpeedCalculator::validate_measurement(uint64_t track_id, float speed_kmh,
                                           int frame_num, float bbox_area, float confidence) {
    // 1. Track age check
    auto birth_it = track_birth_frame_.find(track_id);
    if (birth_it != track_birth_frame_.end()) {
        int age_frames = frame_num - birth_it->second;
        if (age_frames < min_track_age_frames_) {
            return false;
        }
    }
    
    // 2. Minimum displacement check (2D distance)
    auto hist_it = position_history_.find(track_id);
    if (hist_it != position_history_.end() && hist_it->second.size() >= 2) {
        const auto& oldest = hist_it->second.front();
        const auto& newest = hist_it->second.back();
        float displacement = euclidean_distance(oldest.world_pos, newest.world_pos);
        if (displacement < min_displacement_m_) {
            return false;
        }
    }
    
    // 3. Physical speed limits
    if (speed_kmh <= 0.0f || speed_kmh > max_speed_kmh_) {
        return false;
    }
    
    // 4. Bbox area stability check
    auto area_it = last_bbox_area_.find(track_id);
    if (area_it != last_bbox_area_.end() && area_it->second > 0) {
        float area_ratio = bbox_area / area_it->second;
        if (area_ratio > bbox_area_jump_) {
            return false;
        }
    }
    
    // 5. Detection confidence check
    if (confidence < min_conf_) {
        return false;
    }
    
    return true;
}

float SpeedCalculator::get_smoothed_speed(uint64_t track_id, float raw_speed) {
    auto& history = speed_history_[track_id];
    
    // Add to history
    if (history.size() >= static_cast<size_t>(median_window_)) {
        history.pop_front();
    }
    history.push_back(raw_speed);
    
    // Need at least 3 samples for meaningful median
    if (history.size() < 3) {
        return raw_speed;
    }
    
    return compute_median(history);
}

void SpeedCalculator::update_bbox_area(uint64_t track_id, float area) {
    last_bbox_area_[track_id] = area;
}

void SpeedCalculator::cleanup_old_tracks(int current_frame, int max_age) {
    std::vector<uint64_t> to_remove;
    
    for (const auto& [tid, last_frame] : last_seen_frame_) {
        if (current_frame - last_frame > max_age) {
            to_remove.push_back(tid);
        }
    }
    
    for (uint64_t tid : to_remove) {
        position_history_.erase(tid);
        track_birth_frame_.erase(tid);
        last_bbox_area_.erase(tid);
        speed_history_.erase(tid);
        last_seen_frame_.erase(tid);
    }
}

float SpeedCalculator::compute_median(std::deque<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    
    std::vector<float> sorted_values(values.begin(), values.end());
    std::sort(sorted_values.begin(), sorted_values.end());
    
    size_t n = sorted_values.size();
    if (n % 2 == 0) {
        return (sorted_values[n/2 - 1] + sorted_values[n/2]) / 2.0f;
    } else {
        return sorted_values[n/2];
    }
}

float SpeedCalculator::euclidean_distance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return std::sqrt(dx * dx + dy * dy);
}
