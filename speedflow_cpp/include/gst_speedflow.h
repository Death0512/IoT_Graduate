/**
 * @file gst_speedflow.h
 * @brief GStreamer SpeedFlow plugin for real-time speed measurement
 * 
 * This plugin performs:
 * - Vehicle speed calculation using homography
 * - License plate association with vehicles
 * - Overspeed detection and notification
 * - Optional NVOF (Optical Flow) integration
 */

#ifndef GST_SPEEDFLOW_H
#define GST_SPEEDFLOW_H

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <memory>
#include <unordered_map>
#include <deque>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Include PlateCandidate definition
#include "plate_associator.h"

// Forward declarations
class SpeedCalculator;
class HomographyTransform;
class PlateAssociator;

G_BEGIN_DECLS

/* Type macros */
#define GST_TYPE_SPEEDFLOW             (gst_speedflow_get_type())
#define GST_SPEEDFLOW(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_SPEEDFLOW, GstSpeedFlow))
#define GST_SPEEDFLOW_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_SPEEDFLOW, GstSpeedFlowClass))
#define GST_IS_SPEEDFLOW(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_SPEEDFLOW))
#define GST_IS_SPEEDFLOW_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_SPEEDFLOW))

typedef struct _GstSpeedFlow GstSpeedFlow;
typedef struct _GstSpeedFlowClass GstSpeedFlowClass;

/* Vehicle tracking data */
struct VehicleTrackData {
    guint64 track_id;
    gint class_id;
    gfloat left, top, width, height;
    gfloat confidence;
    gboolean in_roi;
    std::deque<gfloat> y_world_history;
    gint birth_frame;
    gfloat last_speed_kmh;
    std::string display_text;
    gfloat last_area;
};

/* Plate data for each vehicle (uses PlateCandidate from plate_associator.h) */
struct PlateData {
    std::string locked_text;        // Finalized plate after 12-frame window
    gint detection_start_frame;
    gint attempts;
    std::vector<PlateCandidate> candidates;
    gboolean is_locked;
};

/**
 * @brief Main GStreamer SpeedFlow element structure
 */
struct _GstSpeedFlow {
    GstBaseTransform parent;
    
    /* Properties */
    gchar *config_file;             // Homography config YAML
    gfloat speed_limit_kmh;         // Speed limit for overspeed detection
    gboolean enable_nvof;           // Enable NVIDIA Optical Flow
    gint video_fps;                 // Video FPS for speed calculation
    gchar *snap_dir;                // Directory for overspeed snapshots
    
    /* Instance-based helper classes (NO MORE GLOBALS!) */
    std::unique_ptr<SpeedCalculator> speed_calc;
    std::unique_ptr<HomographyTransform> homography;
    std::unique_ptr<PlateAssociator> plate_assoc;
    std::vector<std::pair<float, float>> roi_points;  // ROI polygon for display
    
    /* Frame dimensions (needed for NVOF grid mapping) */
    guint frame_width;
    guint frame_height;
    
    /* Homography matrix */
    cv::Mat homography_matrix;
    
    /* Vehicle tracking state */
    std::unordered_map<guint64, VehicleTrackData> vehicle_tracks;
    std::unordered_map<guint64, PlateData> vehicle_plates;
    std::unordered_map<guint64, std::deque<gfloat>> speed_history;
    
    /* Frame counter */
    guint64 frame_num;
    
    /* Processing state */
    gboolean is_initialized;
    
    /* Configuration */
    gint min_track_age_frames;
    gfloat min_world_displacement_m;
    gfloat max_speed_kmh;
    gfloat bbox_area_jump_threshold;
    gfloat min_detection_confidence;
    gint median_window_size;
    gint plate_detection_frames;    // 5-frame window for plate detection
};

struct _GstSpeedFlowClass {
    GstBaseTransformClass parent_class;
};

GType gst_speedflow_get_type(void);

G_END_DECLS

#endif /* GST_SPEEDFLOW_H */
