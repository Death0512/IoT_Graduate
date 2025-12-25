/**
 * @file plate_associator.cpp
 * @brief License plate association and 12-frame voting implementation
 */

#include "plate_associator.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

PlateAssociator::PlateAssociator(int plate_detection_frames, int max_attempts)
    : plate_detection_frames_(plate_detection_frames)
    , max_attempts_(max_attempts) {}

void PlateAssociator::process_plates(
    const std::unordered_map<uint64_t, std::tuple<float, float, float, float>>& vehicles,
    const std::vector<PlateBBox>& plates,
    int frame_num)
{
    for (const auto& plate : plates) {
        // Find closest vehicle
        uint64_t vehicle_id = find_closest_vehicle(plate, vehicles);
        
        if (vehicle_id == 0) {
            continue; // No matching vehicle
        }
        
        // Get or create vehicle plate state
        auto& state = vehicle_plates_[vehicle_id];
        
        // Skip if already locked
        if (state.is_locked) {
            continue;
        }
        
        // Initialize detection window
        if (state.detection_start_frame == 0) {
            state.detection_start_frame = frame_num;
            state.attempts = 0;
            state.candidates.clear();
            state.is_locked = false;
        }
        
        // Calculate frames elapsed
        int frames_elapsed = frame_num - state.detection_start_frame;
        
        // Collect candidates within detection window
        if (frames_elapsed < plate_detection_frames_) {
            if (!plate.text.empty()) {
                PlateCandidate candidate;
                candidate.text = plate.text;
                candidate.confidence = plate.confidence;
                candidate.quality = calculate_quality(plate);
                candidate.frame_num = frame_num;
                state.candidates.push_back(candidate);
            }
        }
        // Window completed - select best plate
        else if (frames_elapsed == plate_detection_frames_) {
            std::string best_plate = select_best_plate(state.candidates);
            
            if (!best_plate.empty()) {
                state.locked_text = best_plate;
                state.is_locked = true;
            } else {
                // Retry if attempts remaining
                state.attempts++;
                if (state.attempts < max_attempts_) {
                    state.detection_start_frame = frame_num;
                    state.candidates.clear();
                } else {
                    // Mark as failed
                    state.is_locked = true;
                    state.locked_text = "";
                }
            }
        }
    }
}

std::string PlateAssociator::get_plate_text(uint64_t vehicle_id) const {
    auto it = vehicle_plates_.find(vehicle_id);
    if (it != vehicle_plates_.end() && it->second.is_locked) {
        return it->second.locked_text;
    }
    return "";
}

bool PlateAssociator::is_plate_locked(uint64_t vehicle_id) const {
    auto it = vehicle_plates_.find(vehicle_id);
    return (it != vehicle_plates_.end() && it->second.is_locked);
}

void PlateAssociator::cleanup_old_vehicles(int current_frame, int max_age) {
    std::vector<uint64_t> to_remove;
    
    for (const auto& [vid, state] : vehicle_plates_) {
        if (state.detection_start_frame > 0 && 
            current_frame - state.detection_start_frame > max_age) {
            to_remove.push_back(vid);
        }
    }
    
    for (uint64_t vid : to_remove) {
        vehicle_plates_.erase(vid);
    }
}

uint64_t PlateAssociator::find_closest_vehicle(
    const PlateBBox& plate,
    const std::unordered_map<uint64_t, std::tuple<float, float, float, float>>& vehicles)
{
    uint64_t best_id = 0;
    float min_distance = 300.0f; // Maximum distance threshold
    
    float plate_cx = plate.left + plate.width / 2.0f;
    float plate_cy = plate.top + plate.height / 2.0f;
    
    for (const auto& [vid, bbox] : vehicles) {
        auto& [vx, vy, vw, vh] = bbox;
        
        // Distance between centers
        float dist = center_distance(plate.left, plate.top, plate.width, plate.height,
                                     vx, vy, vw, vh);
        
        if (dist < min_distance) {
            // Additional check: plate should be within vehicle horizontal bounds ± 50%
            float v_left = vx;
            float v_right = vx + vw;
            float h_tolerance = vw * 0.5f;
            
            if (plate_cx >= v_left - h_tolerance && plate_cx <= v_right + h_tolerance) {
                min_distance = dist;
                best_id = vid;
            }
        }
    }
    
    return best_id;
}

float PlateAssociator::calculate_quality(const PlateBBox& plate) {
    // 1. Confidence score (70%)
    float conf_score = plate.confidence * 70.0f;
    
    // 2. Area score (20%) - larger = clearer
    float area = plate.width * plate.height;
    float area_score = std::min(20.0f, std::max(0.0f, (area - 4000.0f) / 12000.0f * 20.0f));
    
    // 3. Aspect ratio score (10%)
    float aspect = plate.width / std::max(1.0f, plate.height);
    float ideal_aspect = (aspect >= 1.8f) ? 2.5f : 1.1f; // 1-line vs 2-line plates
    float aspect_diff = std::abs(aspect - ideal_aspect);
    float aspect_score = std::max(0.0f, 10.0f - aspect_diff * 2.0f);
    
    return conf_score + area_score + aspect_score;
}

std::string PlateAssociator::select_best_plate(const std::vector<PlateCandidate>& candidates) {
    if (candidates.empty()) {
        return "";
    }
    
    // Group by text (voting)
    std::unordered_map<std::string, std::vector<const PlateCandidate*>> text_groups;
    
    for (const auto& candidate : candidates) {
        text_groups[candidate.text].push_back(&candidate);
    }
    
    // Find most frequent text
    std::string best_text = "";
    size_t max_count = 0;
    float best_quality = 0.0f;
    
    for (const auto& [text, group] : text_groups) {
        // Select by frequency first
        if (group.size() > max_count) {
            max_count = group.size();
            best_text = text;
            
            // Find best quality in this group
            best_quality = 0.0f;
            for (const auto* cand : group) {
                if (cand->quality > best_quality) {
                    best_quality = cand->quality;
                }
            }
        }
        // Same frequency - select by quality
        else if (group.size() == max_count) {
            float group_best_quality = 0.0f;
            for (const auto* cand : group) {
                if (cand->quality > group_best_quality) {
                    group_best_quality = cand->quality;
                }
            }
            if (group_best_quality > best_quality) {
                best_text = text;
                best_quality = group_best_quality;
            }
        }
    }
    
    return best_text;
}

float PlateAssociator::center_distance(float x1, float y1, float w1, float h1,
                                       float x2, float y2, float w2, float h2) {
    float cx1 = x1 + w1 / 2.0f;
    float cy1 = y1 + h1 / 2.0f;
    float cx2 = x2 + w2 / 2.0f;
    float cy2 = y2 + h2 / 2.0f;
    
    return std::sqrt((cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2));
}
