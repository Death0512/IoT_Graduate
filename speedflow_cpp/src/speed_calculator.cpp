/**
 * @file speed_calculator.cpp
 * @brief Speed calculation and validation implementation
 */

#include "speed_calculator.h"
#include <algorithm>
#include <cmath>

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

void SpeedCalculator::update_history(uint64_t track_id, float y_world, int frame_num) {
    auto& history = position_history_[track_id];
    
    // Keep only 1 second of history 
    if (history.size() >= static_cast<size_t>(video_fps_)) {
        history.pop_front();
    }
    history.push_back(y_world);
    
    last_seen_frame_[track_id] = frame_num;
}

float SpeedCalculator::calculate_speed(uint64_t track_id) {
    auto it = position_history_.find(track_id);
    if (it == position_history_.end()) {
        return -1.0f;
    }
    
    auto& history = it->second;
    
    // Need full 1-second window
    if (history.size() < static_cast<size_t>(video_fps_)) {
        return -1.0f;
    }
    
    // Distance = |y_end - y_start|
    float distance_m = std::abs(history.back() - history.front());
    
    // Time = (num_samples - 1) / fps
    float time_s = static_cast<float>(history.size() - 1) / video_fps_;
    
    if (time_s <= 0.0f) {
        return 0.0f;
    }
    
    // Speed in km/h = (distance_m / time_s) * 3.6
    float speed_kmh = (distance_m / time_s) * 3.6f;
    
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
    
    // 2. Minimum displacement check
    auto hist_it = position_history_.find(track_id);
    if (hist_it != position_history_.end() && hist_it->second.size() >= 2) {
        float displacement = std::abs(hist_it->second.back() - hist_it->second.front());
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
