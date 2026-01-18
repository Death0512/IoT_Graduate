/**
 * @file gst_speedflow.cpp
 * @brief Main GStreamer SpeedFlow plugin implementation
 */

/* Required for GST_PLUGIN_DEFINE */
#ifndef PACKAGE
#define PACKAGE "speedflow"
#endif

#include "gst_speedflow.h"
#include "speed_calculator.h"
#include "homography.h"

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <nvdsmeta.h>
#include <gstnvdsmeta.h>
#include <nvds_analytics_meta.h>
#include <nvds_opticalflow_meta.h>  // NVOF metadata for motion vectors

#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <set>
#include <cmath>

/* Plugin signals and properties */
enum {
    PROP_0,
    PROP_CONFIG_FILE,
    PROP_SPEED_LIMIT,
    PROP_ENABLE_NVOF,
    PROP_VIDEO_FPS,
    PROP_SNAP_DIR
};

/* Default property values */
#define DEFAULT_CONFIG_FILE     "configs/points_1.yml"
#define DEFAULT_SPEED_LIMIT     80.0f
#define DEFAULT_ENABLE_NVOF     FALSE
#define DEFAULT_VIDEO_FPS       30
#define DEFAULT_SNAP_DIR        "logs/overspeed_snaps"

/* Vehicle class IDs (COCO) */
static const std::set<int> VEHICLE_CLASS_IDS = {2, 3, 5, 7}; // car, motorbike, bus, truck
static const std::set<int> PLATE_CLASS_IDS = {0}; // license plate

/* REMOVED GLOBAL VARIABLES - Now using instance members in speedflow struct */

GST_DEBUG_CATEGORY_STATIC(gst_speedflow_debug);
#define GST_CAT_DEFAULT gst_speedflow_debug

/* Define the element type */
G_DEFINE_TYPE(GstSpeedFlow, gst_speedflow, GST_TYPE_BASE_TRANSFORM);

/* Forward declarations */
static void gst_speedflow_set_property(GObject *object, guint prop_id,
                                       const GValue *value, GParamSpec *pspec);
static void gst_speedflow_get_property(GObject *object, guint prop_id,
                                       GValue *value, GParamSpec *pspec);
static void gst_speedflow_finalize(GObject *object);
static gboolean gst_speedflow_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps);
static gboolean gst_speedflow_start(GstBaseTransform *trans);
static gboolean gst_speedflow_stop(GstBaseTransform *trans);
static GstFlowReturn gst_speedflow_transform_ip(GstBaseTransform *trans, GstBuffer *buf);

/* Pad templates */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM)")
);

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:NVMM)")
);

/* Initialize the class */
static void gst_speedflow_class_init(GstSpeedFlowClass *klass) {
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);
    
    gobject_class->set_property = gst_speedflow_set_property;
    gobject_class->get_property = gst_speedflow_get_property;
    gobject_class->finalize = gst_speedflow_finalize;
    
    /* Install properties */
    g_object_class_install_property(gobject_class, PROP_CONFIG_FILE,
        g_param_spec_string("config-file", "Config File",
            "Path to homography configuration YAML file",
            DEFAULT_CONFIG_FILE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_SPEED_LIMIT,
        g_param_spec_float("speed-limit", "Speed Limit",
            "Speed limit in km/h for overspeed detection",
            0.0f, 300.0f, DEFAULT_SPEED_LIMIT,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_ENABLE_NVOF,
        g_param_spec_boolean("enable-nvof", "Enable NVOF",
            "Enable NVIDIA Optical Flow integration",
            DEFAULT_ENABLE_NVOF,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_VIDEO_FPS,
        g_param_spec_int("video-fps", "Video FPS",
            "Video frame rate for speed calculation",
            1, 120, DEFAULT_VIDEO_FPS,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    g_object_class_install_property(gobject_class, PROP_SNAP_DIR,
        g_param_spec_string("snap-dir", "Snapshot Directory",
            "Directory to save overspeed snapshots",
            DEFAULT_SNAP_DIR,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    
    /* Set element metadata */
    gst_element_class_set_static_metadata(element_class,
        "SpeedFlow",
        "Filter/Video/Analytics",
        "Real-time vehicle speed measurement and license plate recognition",
        "IoT Graduate Project");
    
    /* Add pad templates */
    gst_element_class_add_static_pad_template(element_class, &sink_factory);
    gst_element_class_add_static_pad_template(element_class, &src_factory);
    
    /* Set transform function */
    transform_class->set_caps = gst_speedflow_set_caps;
    transform_class->start = gst_speedflow_start;
    transform_class->stop = gst_speedflow_stop;
    transform_class->transform_ip = gst_speedflow_transform_ip;
    
    /* We modify the buffer in-place */
    GST_BASE_TRANSFORM_CLASS(klass)->passthrough_on_same_caps = FALSE;
    
    GST_DEBUG_CATEGORY_INIT(gst_speedflow_debug, "speedflow", 0, "SpeedFlow Plugin");
}

/* Initialize the element instance */
static void gst_speedflow_init(GstSpeedFlow *speedflow) {
    speedflow->config_file = g_strdup(DEFAULT_CONFIG_FILE);
    speedflow->speed_limit_kmh = DEFAULT_SPEED_LIMIT;
    speedflow->enable_nvof = DEFAULT_ENABLE_NVOF;
    speedflow->video_fps = DEFAULT_VIDEO_FPS;
    speedflow->snap_dir = g_strdup(DEFAULT_SNAP_DIR);
    
    speedflow->frame_num = 0;
    speedflow->is_initialized = FALSE;
    
    /* Frame dimensions (will be set by set_caps) */
    speedflow->frame_width = 0;
    speedflow->frame_height = 0;
    
    /* Default configuration values */
    speedflow->min_track_age_frames = 15; // 0.5s at 30fps
    speedflow->min_world_displacement_m = 0.5f;
    speedflow->max_speed_kmh = 160.0f;
    speedflow->bbox_area_jump_threshold = 2.5f;
    speedflow->min_detection_confidence = 0.45f;
    speedflow->median_window_size = 5;
    speedflow->plate_detection_frames = 5;
    
    /* Set in-place transform mode */
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(speedflow), TRUE);
}

/* Set property */
static void gst_speedflow_set_property(GObject *object, guint prop_id,
                                       const GValue *value, GParamSpec *pspec) {
    GstSpeedFlow *speedflow = GST_SPEEDFLOW(object);
    
    switch (prop_id) {
        case PROP_CONFIG_FILE:
            g_free(speedflow->config_file);
            speedflow->config_file = g_value_dup_string(value);
            break;
        case PROP_SPEED_LIMIT:
            speedflow->speed_limit_kmh = g_value_get_float(value);
            break;
        case PROP_ENABLE_NVOF:
            speedflow->enable_nvof = g_value_get_boolean(value);
            break;
        case PROP_VIDEO_FPS:
            speedflow->video_fps = g_value_get_int(value);
            break;
        case PROP_SNAP_DIR:
            g_free(speedflow->snap_dir);
            speedflow->snap_dir = g_value_dup_string(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/* Get property */
static void gst_speedflow_get_property(GObject *object, guint prop_id,
                                       GValue *value, GParamSpec *pspec) {
    GstSpeedFlow *speedflow = GST_SPEEDFLOW(object);
    
    switch (prop_id) {
        case PROP_CONFIG_FILE:
            g_value_set_string(value, speedflow->config_file);
            break;
        case PROP_SPEED_LIMIT:
            g_value_set_float(value, speedflow->speed_limit_kmh);
            break;
        case PROP_ENABLE_NVOF:
            g_value_set_boolean(value, speedflow->enable_nvof);
            break;
        case PROP_VIDEO_FPS:
            g_value_set_int(value, speedflow->video_fps);
            break;
        case PROP_SNAP_DIR:
            g_value_set_string(value, speedflow->snap_dir);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/* Finalize (destructor) */
static void gst_speedflow_finalize(GObject *object) {
    GstSpeedFlow *speedflow = GST_SPEEDFLOW(object);
    
    g_free(speedflow->config_file);
    g_free(speedflow->snap_dir);
    
    speedflow->vehicle_tracks.clear();
    speedflow->vehicle_plates.clear();
    speedflow->speed_history.clear();
    
    G_OBJECT_CLASS(gst_speedflow_parent_class)->finalize(object);
}

/* Set caps (called when pipeline negotiates formats) */
static gboolean gst_speedflow_set_caps(GstBaseTransform *trans, 
                                       GstCaps *incaps, GstCaps *outcaps) {
    GstSpeedFlow *speedflow = GST_SPEEDFLOW(trans);
    GstStructure *structure = gst_caps_get_structure(incaps, 0);
    
    gint width = 0, height = 0;
    if (gst_structure_get_int(structure, "width", &width) &&
        gst_structure_get_int(structure, "height", &height)) {
        speedflow->frame_width = (guint)width;
        speedflow->frame_height = (guint)height;
        GST_INFO_OBJECT(speedflow, "Frame dimensions set to: %ux%u", 
                        speedflow->frame_width, speedflow->frame_height);
    } else {
        GST_WARNING_OBJECT(speedflow, "Failed to extract frame dimensions from caps");
        // Fallback to default
        speedflow->frame_width = 1920;
        speedflow->frame_height = 1080;
    }
    
    return TRUE;
}

/* Start (called when pipeline starts) */
static gboolean gst_speedflow_start(GstBaseTransform *trans) {
    GstSpeedFlow *speedflow = GST_SPEEDFLOW(trans);
    
    GST_INFO_OBJECT(speedflow, "Starting SpeedFlow plugin");
    GST_INFO_OBJECT(speedflow, "Config file: %s", speedflow->config_file);
    GST_INFO_OBJECT(speedflow, "Speed limit: %.1f km/h", speedflow->speed_limit_kmh);
    GST_INFO_OBJECT(speedflow, "Video FPS: %d", speedflow->video_fps);
    GST_INFO_OBJECT(speedflow, "NVOF enabled: %s", speedflow->enable_nvof ? "YES" : "NO");
    
    /* Initialize homography (INSTANCE-BASED) */
    speedflow->homography = std::make_unique<HomographyTransform>();
    if (!speedflow->homography->load_config(speedflow->config_file)) {
        GST_ERROR_OBJECT(speedflow, "Failed to load homography config from %s", speedflow->config_file);
        return FALSE;
    }
    
    /* Initialize speed calculator (INSTANCE-BASED) */
    speedflow->speed_calc = std::make_unique<SpeedCalculator>(
        speedflow->video_fps,
        speedflow->min_track_age_frames,
        speedflow->min_world_displacement_m,
        speedflow->max_speed_kmh,
        speedflow->bbox_area_jump_threshold,
        speedflow->min_detection_confidence,
        speedflow->median_window_size
    );
    
    /* Initialize plate associator (INSTANCE-BASED) */
    speedflow->plate_assoc = std::make_unique<PlateAssociator>(
        speedflow->plate_detection_frames,
        3 // max attempts
    );
    
    /* Load ROI polygon points for display (same as homography source points) */
    speedflow->roi_points.clear();
    auto source_pts = speedflow->homography->get_source_polygon();
    for (const auto& pt : source_pts) {
        speedflow->roi_points.emplace_back(pt.x, pt.y);
    }
    GST_INFO_OBJECT(speedflow, "Loaded %zu ROI polygon points", speedflow->roi_points.size());
    
    /* Note: frame_width/height will be set by set_caps callback */
    
    speedflow->is_initialized = TRUE;
    speedflow->frame_num = 0;
    
    GST_INFO_OBJECT(speedflow, "SpeedFlow plugin initialized successfully");
    
    return TRUE;
}

/* Stop (called when pipeline stops) */
static gboolean gst_speedflow_stop(GstBaseTransform *trans) {
    GstSpeedFlow *speedflow = GST_SPEEDFLOW(trans);
    
    GST_INFO_OBJECT(speedflow, "Stopping SpeedFlow plugin");
    
    /* Reset instance-based helpers */
    speedflow->speed_calc.reset();
    speedflow->homography.reset();
    speedflow->plate_assoc.reset();
    speedflow->roi_points.clear();
    
    speedflow->vehicle_tracks.clear();
    speedflow->vehicle_plates.clear();
    speedflow->speed_history.clear();
    speedflow->is_initialized = FALSE;
    
    return TRUE;
}

/* Helper: Check if class ID is a vehicle */
static inline bool is_vehicle(int class_id) {
    return VEHICLE_CLASS_IDS.find(class_id) != VEHICLE_CLASS_IDS.end();
}

/* Helper: Check if class ID is a plate */
static inline bool is_plate(int class_id) {
    return PLATE_CLASS_IDS.find(class_id) != PLATE_CLASS_IDS.end();
}

/* Helper: Get ISO timestamp */
static std::string get_iso_timestamp() {
    auto now = std::time(nullptr);
    auto tm = std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

/* Helper: Check if object is in ROI (from nvdsanalytics metadata) */
static bool object_in_roi(NvDsObjectMeta *obj_meta) {
    NvDsMetaList *l_user = obj_meta->obj_user_meta_list;
    while (l_user != nullptr) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if (user_meta && user_meta->base_meta.meta_type == NVDS_USER_OBJ_META_NVDSANALYTICS) {
            NvDsAnalyticsObjInfo *analytics_info = (NvDsAnalyticsObjInfo *)user_meta->user_meta_data;
            if (analytics_info && !analytics_info->roiStatus.empty()) {
                return true;
            }
        }
        l_user = l_user->next;
    }
    return false;
}

/**
 * @brief Draw ROI polygon on frame using display metadata
 * @param batch_meta Batch metadata for acquiring display meta
 * @param frame_meta Frame metadata to add display to
 * @param roi_points Vector of ROI polygon points (x, y pairs)
 */
static void draw_roi_polygon(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                             const std::vector<std::pair<float, float>>& roi_points) {
    if (roi_points.size() < 3) return;  // Need at least 3 points for polygon
    
    NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (!display_meta) return;
    
    int n = roi_points.size();
    display_meta->num_lines = n;
    
    for (int i = 0; i < n && i < MAX_ELEMENTS_IN_DISPLAY_META; i++) {
        int x1 = (int)roi_points[i].first;
        int y1 = (int)roi_points[i].second;
        int x2 = (int)roi_points[(i + 1) % n].first;
        int y2 = (int)roi_points[(i + 1) % n].second;
        
        display_meta->line_params[i].x1 = x1;
        display_meta->line_params[i].y1 = y1;
        display_meta->line_params[i].x2 = x2;
        display_meta->line_params[i].y2 = y2;
        display_meta->line_params[i].line_width = 4;
        
        // Red color for ROI polygon
        display_meta->line_params[i].line_color.red = 1.0f;
        display_meta->line_params[i].line_color.green = 0.0f;
        display_meta->line_params[i].line_color.blue = 0.0f;
        display_meta->line_params[i].line_color.alpha = 1.0f;
    }
    
    nvds_add_display_meta_to_frame(frame_meta, display_meta);
}

/**
 * @brief Colorize bbox based on overspeed status
 * @param obj_meta Object metadata to modify
 * @param is_overspeed True if vehicle is overspeed (red), false for normal (green)
 */
static void colorize_bbox(NvDsObjectMeta *obj_meta, bool is_overspeed) {
    if (is_overspeed) {
        // RED for overspeed
        obj_meta->rect_params.border_width = 3;
        obj_meta->rect_params.border_color.red = 1.0f;
        obj_meta->rect_params.border_color.green = 0.0f;
        obj_meta->rect_params.border_color.blue = 0.0f;
        obj_meta->rect_params.border_color.alpha = 1.0f;
        
        // Red background for text
        obj_meta->text_params.text_bg_clr.red = 1.0f;
        obj_meta->text_params.text_bg_clr.green = 0.0f;
        obj_meta->text_params.text_bg_clr.blue = 0.0f;
        obj_meta->text_params.text_bg_clr.alpha = 0.6f;
        
        // White text
        obj_meta->text_params.font_params.font_color.red = 1.0f;
        obj_meta->text_params.font_params.font_color.green = 1.0f;
        obj_meta->text_params.font_params.font_color.blue = 1.0f;
        obj_meta->text_params.font_params.font_color.alpha = 1.0f;
    } else {
        // GREEN for normal
        obj_meta->rect_params.border_width = 2;
        obj_meta->rect_params.border_color.red = 0.0f;
        obj_meta->rect_params.border_color.green = 1.0f;
        obj_meta->rect_params.border_color.blue = 0.0f;
        obj_meta->rect_params.border_color.alpha = 1.0f;
        
        // Dark background for text
        obj_meta->text_params.text_bg_clr.red = 0.0f;
        obj_meta->text_params.text_bg_clr.green = 0.0f;
        obj_meta->text_params.text_bg_clr.blue = 0.0f;
        obj_meta->text_params.text_bg_clr.alpha = 0.4f;
        
        // White text
        obj_meta->text_params.font_params.font_color.red = 1.0f;
        obj_meta->text_params.font_params.font_color.green = 1.0f;
        obj_meta->text_params.font_params.font_color.blue = 1.0f;
        obj_meta->text_params.font_params.font_color.alpha = 1.0f;
    }
}

/* Helper: Extract LPR text from classifier metadata */
static std::string extract_lpr_text(NvDsObjectMeta *obj_meta) {
    NvDsMetaList *l_class = obj_meta->classifier_meta_list;
    while (l_class != nullptr) {
        NvDsClassifierMeta *class_meta = (NvDsClassifierMeta *)l_class->data;
        if (class_meta && class_meta->unique_component_id == 3) { // LPR GIE ID
            NvDsMetaList *l_label = class_meta->label_info_list;
            if (l_label != nullptr) {
                NvDsLabelInfo *label_info = (NvDsLabelInfo *)l_label->data;
                if (label_info && label_info->result_label) {
                    return std::string(label_info->result_label);
                }
            }
        }
        l_class = l_class->next;
    }
    return "";
}

/**
 * @brief Extract motion speed from NVOF metadata at object location
 * @param batch_meta Batch metadata
 * @param obj_meta Object metadata
 * @param frame_width Frame width
 * @param frame_height Frame height
 * @return Average motion magnitude (pixels/frame) or 0 if not available
 */
static float extract_nvof_motion_at_object(NvDsBatchMeta *batch_meta, 
                                           NvDsObjectMeta *obj_meta,
                                           guint frame_width, guint frame_height) {
    // Find NVOF metadata in user meta list
    NvDsMetaList *l_user = batch_meta->batch_user_meta_list;
    while (l_user != nullptr) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        
        if (user_meta && user_meta->base_meta.meta_type == NVDS_OPTICAL_FLOW_META) {
            NvDsOpticalFlowMeta *of_meta = (NvDsOpticalFlowMeta *)user_meta->user_meta_data;
            
            if (of_meta && of_meta->data) {
                // Get object bounding box center
                float obj_cx = obj_meta->rect_params.left + obj_meta->rect_params.width / 2.0f;
                float obj_cy = obj_meta->rect_params.top + obj_meta->rect_params.height / 2.0f;
                
                // Calculate grid indices for this position
                guint grid_cols = of_meta->cols;
                guint grid_rows = of_meta->rows;
                
                // Map pixel position to grid cell
                int grid_x = (int)(obj_cx * grid_cols / frame_width);
                int grid_y = (int)(obj_cy * grid_rows / frame_height);
                
                // Clamp to valid range
                grid_x = std::max(0, std::min((int)grid_cols - 1, grid_x));
                grid_y = std::max(0, std::min((int)grid_rows - 1, grid_y));
                
                // Get motion vector at this grid cell
                NvOFFlowVector *flow_vectors = (NvOFFlowVector *)of_meta->data;
                int idx = grid_y * grid_cols + grid_x;
                
                float flow_x = flow_vectors[idx].flowx / 32.0f;  // Fix-point to float
                float flow_y = flow_vectors[idx].flowy / 32.0f;
                
                // Calculate magnitude
                float magnitude = std::sqrt(flow_x * flow_x + flow_y * flow_y);
                
                return magnitude;
            }
        }
        l_user = l_user->next;
    }
    return 0.0f;
}

/**
 * @brief Log NVOF stats for debugging
 */
static void log_nvof_stats(NvDsBatchMeta *batch_meta, guint64 frame_num) {
    NvDsMetaList *l_user = batch_meta->batch_user_meta_list;
    while (l_user != nullptr) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        
        if (user_meta && user_meta->base_meta.meta_type == NVDS_OPTICAL_FLOW_META) {
            NvDsOpticalFlowMeta *of_meta = (NvDsOpticalFlowMeta *)user_meta->user_meta_data;
            
            if (of_meta && of_meta->data && frame_num % 100 == 0) {
                // Log stats every 100 frames
                g_print("[NVOF] Frame %lu: Grid %ux%u available\n", 
                        frame_num, of_meta->cols, of_meta->rows);
                
                // Sample center point motion for debugging
                int center_idx = (of_meta->rows / 2) * of_meta->cols + (of_meta->cols / 2);
                NvOFFlowVector *flow_vectors = (NvOFFlowVector *)of_meta->data;
                float flow_x = flow_vectors[center_idx].flowx / 32.0f;
                float flow_y = flow_vectors[center_idx].flowy / 32.0f;
                float magnitude = std::sqrt(flow_x * flow_x + flow_y * flow_y);
                g_print("[NVOF] Center motion: x=%.2f, y=%.2f, mag=%.2f px/frame\n",
                        flow_x, flow_y, magnitude);
            }
            return;
        }
        l_user = l_user->next;
    }
}


/* Main transform function - processes each buffer */
static GstFlowReturn gst_speedflow_transform_ip(GstBaseTransform *trans, GstBuffer *buf) {
    GstSpeedFlow *speedflow = GST_SPEEDFLOW(trans);
    
    if (!speedflow->is_initialized) {
        GST_WARNING_OBJECT(speedflow, "Plugin not initialized, passing through");
        return GST_FLOW_OK;
    }
    
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        return GST_FLOW_OK;
    }
    
    speedflow->frame_num++;
    
    /* Log NVOF stats if enabled (debug) */
    if (speedflow->enable_nvof) {
        log_nvof_stats(batch_meta, speedflow->frame_num);
    }
    
    /* Iterate through frames in batch */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; 
         l_frame != nullptr; 
         l_frame = l_frame->next) {
        
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        int frame_number = frame_meta->frame_num;
        
        /* PASS 1: Collect vehicles and plates */
        std::unordered_map<uint64_t, std::tuple<float, float, float, float>> vehicles;
        std::vector<PlateBBox> plates;
        std::unordered_map<uint64_t, NvDsObjectMeta*> vehicle_obj_metas;
        std::vector<NvDsObjectMeta*> objects_to_remove;
        
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj != nullptr;
             l_obj = l_obj->next) {
            
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            
            /* Check ROI status - skip processing but DON'T remove to avoid tracking disruption */
            if (!object_in_roi(obj_meta)) {
                continue;  // Skip but keep object in metadata for tracker continuity
            }
            
            if (is_vehicle(obj_meta->class_id)) {
                uint64_t tid = obj_meta->object_id;
                vehicles[tid] = std::make_tuple(
                    obj_meta->rect_params.left,
                    obj_meta->rect_params.top,
                    obj_meta->rect_params.width,
                    obj_meta->rect_params.height
                );
                vehicle_obj_metas[tid] = obj_meta;
                
                /* Register track if new */
                speedflow->speed_calc->register_track(tid, frame_number);
                
            } else if (is_plate(obj_meta->class_id)) {
                PlateBBox plate;
                plate.left = obj_meta->rect_params.left;
                plate.top = obj_meta->rect_params.top;
                plate.width = obj_meta->rect_params.width;
                plate.height = obj_meta->rect_params.height;
                plate.confidence = obj_meta->confidence;
                plate.text = extract_lpr_text(obj_meta);
                plates.push_back(plate);
                
                /* Update plate display text */
                if (obj_meta->text_params.display_text) {
                    g_free(obj_meta->text_params.display_text);
                }
                obj_meta->text_params.display_text = g_strdup("plate");
            }
        }
        
        /* PASS 2: Process plates with 5-frame window */
        speedflow->plate_assoc->process_plates(vehicles, plates, frame_number);
        
        /* PASS 3: Calculate speed for each vehicle */
        for (auto& [tid, bbox] : vehicles) {
            auto& [left, top, width, height] = bbox;
            NvDsObjectMeta *obj_meta = vehicle_obj_metas[tid];
            
            /* Get bottom center point */
            float cx = left + width / 2.0f;
            float bottom_y = top + height;
            
            /* Transform to world coordinates (2D) */
            cv::Point2f world_pt = speedflow->homography->transform_point(cx, bottom_y);
            
            /* Extract NVOF motion if enabled */
            float nvof_magnitude = 0.0f;
            if (speedflow->enable_nvof) {
                nvof_magnitude = extract_nvof_motion_at_object(
                    batch_meta, obj_meta, 
                    speedflow->frame_width, speedflow->frame_height
                );
            }
            
            /* Get PTS timestamp from buffer metadata (with validation) */
            guint64 pts_ns = frame_meta->buf_pts;
            if (pts_ns == GST_CLOCK_TIME_NONE || pts_ns == 0) {
                // Generate synthetic PTS if not available
                pts_ns = (guint64)(speedflow->frame_num * 1e9 / speedflow->video_fps);
            }
            
            /* Update history with 2D position + timestamp */
            speedflow->speed_calc->update_history(tid, world_pt, pts_ns, frame_number, nvof_magnitude);
            
            /* Calculate bbox area */
            float area = width * height;
            speedflow->speed_calc->update_bbox_area(tid, area);
            
            /* Calculate speed (now TIME-BASED with 2D vectors + NVOF fusion) */
            float raw_speed = speedflow->speed_calc->calculate_speed(tid);
            
            // OPTIONAL: Advanced NVOF Fusion Mode (experimental)
            // Uncomment to use full sensor fusion instead of just validation
            /*
            if (speedflow->enable_nvof && raw_speed > 0) {
                float nvof_speed = speedflow->speed_calc->calculate_nvof_speed(
                    tid, speedflow->frame_width, speedflow->frame_height
                );
                if (nvof_speed > 0) {
                    raw_speed = speedflow->speed_calc->fuse_speeds(raw_speed, nvof_speed, tid);
                    GST_DEBUG_OBJECT(speedflow, "NVOF Fusion: Track %lu, Geo=%.1f, NVOF=%.1f, Fused=%.1f",
                                    tid, raw_speed, nvof_speed, raw_speed);
                }
            }
            */
            
            std::string display_text = "";
            bool is_overspeed = false;  // Track overspeed status
            
            if (raw_speed > 0) {
                /* Validate measurement */
                if (speedflow->speed_calc->validate_measurement(tid, raw_speed, frame_number, 
                                                               area, obj_meta->confidence)) {
                    /* Apply median smoothing */
                    float smooth_speed = speedflow->speed_calc->get_smoothed_speed(tid, raw_speed);
                    
                    /* Format speed text */
                    std::ostringstream oss;
                    oss << static_cast<int>(smooth_speed) << " km/h";
                    display_text = oss.str();
                    
                    /* Check overspeed */
                    if (smooth_speed >= speedflow->speed_limit_kmh) {
                        is_overspeed = true;
                        GST_DEBUG_OBJECT(speedflow, "OVERSPEED: Track %lu at %.1f km/h",
                                        tid, smooth_speed);
                    }
                }
            }
            
            /* Get plate text if locked */
            std::string plate_text = speedflow->plate_assoc->get_plate_text(tid);
            
            /* Build final display text */
            std::string final_text = "";
            if (!display_text.empty() && !plate_text.empty()) {
                final_text = display_text + "\n" + plate_text;
            } else if (!display_text.empty()) {
                final_text = display_text;
            } else if (!plate_text.empty()) {
                final_text = plate_text;
            }
            
            /* Update display text */
            if (obj_meta->text_params.display_text) {
                g_free(obj_meta->text_params.display_text);
            }
            if (!final_text.empty()) {
                obj_meta->text_params.display_text = g_strdup(final_text.c_str());
            } else {
                obj_meta->text_params.display_text = nullptr;
            }
            
            /* Colorize bbox based on overspeed status */
            colorize_bbox(obj_meta, is_overspeed);
        }
        
        /* Draw ROI polygon on frame */
        draw_roi_polygon(batch_meta, frame_meta, speedflow->roi_points);
        
        /* Cleanup old tracks periodically */
        if (frame_number % 300 == 0) { // Every 10 seconds at 30fps
            speedflow->speed_calc->cleanup_old_tracks(frame_number, speedflow->video_fps * 5);
            speedflow->plate_assoc->cleanup_old_vehicles(frame_number, speedflow->video_fps * 5);
        }
    }
    
    return GST_FLOW_OK;
}

/* Plugin init function */
static gboolean plugin_init(GstPlugin *plugin) {
    return gst_element_register(plugin, "speedflow", GST_RANK_NONE, GST_TYPE_SPEEDFLOW);
}

/* Plugin definition */
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    speedflow,
    "SpeedFlow - Real-time vehicle speed measurement and LPR",
    plugin_init,
    "1.0",
    "LGPL",
    "IoT Graduate Project",
    "https://github.com/iot-graduate"
)
