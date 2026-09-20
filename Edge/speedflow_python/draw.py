# speedflow/draw.py
import pyds
import numpy as np

def add_polygon_display(batch_meta, frame_meta, points: np.ndarray, color=(1.0, 0.0, 0.0, 1.0)):
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    if display_meta is None:
        return
    display_meta.num_labels = 0
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
        display_meta.line_params[i].line_color.set(*color)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)



