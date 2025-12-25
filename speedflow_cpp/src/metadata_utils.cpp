/**
 * @file metadata_utils.cpp
 * @brief Utility functions for DeepStream metadata handling
 */

#include <nvdsmeta.h>
#include <gstnvdsmeta.h>
#include <string>
#include <cstring>

/**
 * @brief Attach display text to an object
 */
void attach_display_text(NvDsObjectMeta *obj_meta, const std::string& text) {
    if (obj_meta->text_params.display_text) {
        g_free(obj_meta->text_params.display_text);
    }
    
    if (!text.empty()) {
        obj_meta->text_params.display_text = g_strdup(text.c_str());
        
        // Configure text appearance
        obj_meta->text_params.font_params.font_size = 12;
        obj_meta->text_params.font_params.font_color.red = 1.0;
        obj_meta->text_params.font_params.font_color.green = 1.0;
        obj_meta->text_params.font_params.font_color.blue = 1.0;
        obj_meta->text_params.font_params.font_color.alpha = 1.0;
        
        obj_meta->text_params.set_bg_clr = 1;
        obj_meta->text_params.text_bg_clr.red = 0.0;
        obj_meta->text_params.text_bg_clr.green = 0.0;
        obj_meta->text_params.text_bg_clr.blue = 0.0;
        obj_meta->text_params.text_bg_clr.alpha = 0.5;
    } else {
        obj_meta->text_params.display_text = nullptr;
    }
}

/**
 * @brief Set bbox color (red for overspeed, green for normal)
 */
void set_bbox_color(NvDsObjectMeta *obj_meta, bool is_overspeed) {
    if (is_overspeed) {
        // Red for overspeed
        obj_meta->rect_params.border_color.red = 1.0;
        obj_meta->rect_params.border_color.green = 0.0;
        obj_meta->rect_params.border_color.blue = 0.0;
        obj_meta->rect_params.border_color.alpha = 1.0;
        obj_meta->rect_params.border_width = 3;
        
        // Red background for text
        obj_meta->text_params.set_bg_clr = 1;
        obj_meta->text_params.text_bg_clr.red = 1.0;
        obj_meta->text_params.text_bg_clr.green = 0.0;
        obj_meta->text_params.text_bg_clr.blue = 0.0;
        obj_meta->text_params.text_bg_clr.alpha = 0.6;
    } else {
        // Green for normal
        obj_meta->rect_params.border_color.red = 0.0;
        obj_meta->rect_params.border_color.green = 1.0;
        obj_meta->rect_params.border_color.blue = 0.0;
        obj_meta->rect_params.border_color.alpha = 1.0;
        obj_meta->rect_params.border_width = 2;
        
        // Dark background for text
        obj_meta->text_params.set_bg_clr = 1;
        obj_meta->text_params.text_bg_clr.red = 0.0;
        obj_meta->text_params.text_bg_clr.green = 0.0;
        obj_meta->text_params.text_bg_clr.blue = 0.0;
        obj_meta->text_params.text_bg_clr.alpha = 0.4;
    }
}
