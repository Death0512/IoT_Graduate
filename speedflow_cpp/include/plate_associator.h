/**
 * @file plate_associator.h
 * @brief License plate association and 12-frame voting mechanism
 */

#ifndef PLATE_ASSOCIATOR_H
#define PLATE_ASSOCIATOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include <deque>

struct PlateBBox {
    float left, top, width, height;
    float confidence;
    std::string text;
};

// PlateCandidate defined here (used by both plate_associator and gst_speedflow)
struct PlateCandidate {
    std::string text;
    float confidence;
    float quality;
    int frame_num;
};

struct VehiclePlateState {
    int detection_start_frame;
    int attempts;
    std::vector<PlateCandidate> candidates;
    std::string locked_text;
    bool is_locked;
};

class PlateAssociator {
public:
    PlateAssociator(int plate_detection_frames = 12, int max_attempts = 3);
    
    /**
     * @brief Associate plates to vehicles based on spatial proximity
     * @param vehicles Map of vehicle_id -> bbox
     * @param plates List of detected plates in frame
     * @param frame_num Current frame number
     */
    void process_plates(
        const std::unordered_map<uint64_t, std::tuple<float, float, float, float>>& vehicles,
        const std::vector<PlateBBox>& plates,
        int frame_num
    );
    
    /**
     * @brief Get locked plate text for a vehicle
     * @param vehicle_id Vehicle track ID
     * @return Plate text or empty string if not locked
     */
    std::string get_plate_text(uint64_t vehicle_id) const;
    
    /**
     * @brief Check if plate is locked for vehicle
     */
    bool is_plate_locked(uint64_t vehicle_id) const;
    
    /**
     * @brief Clean up old vehicle states
     */
    void cleanup_old_vehicles(int current_frame, int max_age);

private:
    int plate_detection_frames_;
    int max_attempts_;
    
    std::unordered_map<uint64_t, VehiclePlateState> vehicle_plates_;
    
    /**
     * @brief Find closest vehicle to a plate
     * @return Vehicle ID or 0 if no match
     */
    uint64_t find_closest_vehicle(
        const PlateBBox& plate,
        const std::unordered_map<uint64_t, std::tuple<float, float, float, float>>& vehicles
    );
    
    /**
     * @brief Calculate plate quality score
     */
    float calculate_quality(const PlateBBox& plate);
    
    /**
     * @brief Select best plate from candidates using voting
     */
    std::string select_best_plate(const std::vector<PlateCandidate>& candidates);
    
    /**
     * @brief Euclidean distance between bbox centers
     */
    float center_distance(float x1, float y1, float w1, float h1,
                         float x2, float y2, float w2, float h2);
};

#endif /* PLATE_ASSOCIATOR_H */
