# speedflow/draw.py
import pyds
import numpy as np

def add_polygon_display(batch_meta, frame_meta, points: np.ndarray):
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    n = len(points)
    display_meta.num_lines = n
    for i in range(n):
        x1,y1 = int(points[i][0]), int(points[i][1])
        x2,y2 = int(points[(i+1)%n][0]), int(points[(i+1)%n][1])
        display_meta.line_params[i].x1 = x1
        display_meta.line_params[i].y1 = y1
        display_meta.line_params[i].x2 = x2
        display_meta.line_params[i].y2 = y2
        display_meta.line_params[i].line_width = 4
        display_meta.line_params[i].line_color.set(1.0, 0.0, 0.0, 1.0)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

def _colorize_osd(self, obj_meta, red_alert: bool):
        # Viền bbox + nền text
    if red_alert:
        obj_meta.rect_params.border_width = max(2, int(obj_meta.rect_params.border_width) or 2)
        obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)  # đỏ
        obj_meta.text_params.text_bg_clr.set(1.0, 0.0, 0.0, 0.6)    # nền đỏ transparent
        obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0) # chữ trắng
    else:
        obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
        obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.4)
        obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
