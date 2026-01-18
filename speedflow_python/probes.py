
# probes.py
# -*- coding: utf-8 -*-
# speedflow/probes.py
import time, os, base64
from collections import defaultdict, deque
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
import cv2

from .settings import (VIDEO_FPS, VEHICLE_CLASS_IDS, SPEED_LOG,SPEED_LIMIT_KMH, JPEG_QUALITY, SNAP_DIR, MAX_SNAPSHOT_PER_ID,
    MIN_TRACK_AGE_FRAMES, MIN_WORLD_DISPL_M, MAX_ABS_KMH,BBOX_AREA_JUMP, MIN_DET_CONF, MEDIAN_WINDOW, LISENCE_PLATE_CLASS_IDS)
from .draw import add_polygon_display

class CSVLogger:
    """Nhẹ nhàng: ghi CSV nếu cần, không bắt buộc."""
    def __init__(self, path, header):
        self.path = path
        self.header = header
        try:
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(",".join(header) + "\n")
        except Exception:
            pass
    def write(self, row):
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(",".join(map(str, row)) + "\n")
        except Exception:
            pass

class ROIFilterProbe:
    """
    Probe to filter out objects that are outside the ROI.
    Removes objects from metadata if they don't have ROI status from nvdsanalytics.
    This prevents non-ROI vehicles from being displayed, tracked, or having license plates detected.
    """
    def __init__(self):
        self.filtered_count = 0
        self.total_count = 0
        self.frame_count = 0
        
    def analytics_src_pad_buffer_probe(self, pad, info, u_data):
        """
        Probe attached to analytics src pad to filter objects outside ROI.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            self.frame_count += 1
            
            # We need to iterate and remove objects that are NOT in ROI
            # Use a list to collect objects to remove (can't remove while iterating)
            objects_to_remove = []
            
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                self.total_count += 1
                
                # Check if object is in ROI by looking for analytics metadata
                in_roi = self._check_obj_in_roi(obj_meta)
                
                if not in_roi:
                    # Mark for removal
                    objects_to_remove.append(obj_meta)
                    self.filtered_count += 1
                
                l_obj = l_obj.next
            
            # Remove objects outside ROI
            for obj_meta in objects_to_remove:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
            
            l_frame = l_frame.next
        
        return Gst.PadProbeReturn.OK
    
    def _check_obj_in_roi(self, obj_meta) -> bool:
        """
        Check if object is inside ROI by examining nvdsanalytics metadata.
        Returns True if object has ROI status, False otherwise.
        """
        try:
            user_meta_list = obj_meta.obj_user_meta_list
            while user_meta_list is not None:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                if user_meta and hasattr(pyds, "nvds_get_user_meta_type"):
                    mt = pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META")
                    if user_meta.base_meta.meta_type == mt:
                        info = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
                        roi_status = getattr(info, "roiStatus", None)
                        if roi_status and len(roi_status) > 0:
                            # Object is in ROI
                            return True
                user_meta_list = user_meta_list.next
            # No ROI metadata found => object is NOT in ROI
            return False
        except Exception as e:
            # On error, assume object is NOT in ROI (safe default)
            return False


class SpeedProbe:
    """
    - Lấy điểm (cx, bottom_y) của bbox -> chuyển sang world bằng homography
    - Tốc độ = Δy_world / Δt trong cửa sổ ~1s -> km/h
    - Nếu > SPEED_LIMIT_KMH: crop bbox -> lưu JPG (1 ảnh/track_id) + (tuỳ chọn) publish base64
    """
    def __init__(self, view_transformer, roi_source_points, cooldown_s: float = 2.5):
        self.view_transformer = view_transformer
        self.roi_points = np.array(roi_source_points, dtype=np.float32)

        # Lịch sử vị trí y_world theo track_id (cửa sổ ~1s)
        self.history_positions = defaultdict(lambda: deque(maxlen=int(VIDEO_FPS)))
        self.last_speed_text   = defaultdict(lambda: "")
        self.last_update_frame = defaultdict(lambda: -int(VIDEO_FPS))

        # chống spam socket
        self.last_alert_ts     = defaultdict(lambda: 0.0)
        self.cooldown_s        = float(cooldown_s)

        # số ảnh đã chụp cho mỗi track_id
        self.snap_count        = defaultdict(int)

        # publisher để đẩy JSON sang web (tuỳ bạn set)
        self.publisher = None
        # cai thien hien thi toc do ao
        self.speed_history = defaultdict(lambda: deque(maxlen=MEDIAN_WINDOW))
        self.track_birth_frame = {}  # lưu frame first-seen cho từng track

        # đảm bảo thư mục tồn tại
        try:
            os.makedirs(str(SNAP_DIR), exist_ok=True)
        except Exception:
            pass

        # License plate tracking: vehicle_id -> {bbox, last_frame, confidence, text}
        self.vehicle_plates = {}  # tid -> {left, top, width, height, last_frame, conf, text}
        
        # === Plate Detection Window Mechanism (5-frame window) ===
        self.PLATE_DETECTION_FRAMES = 5  # Detect plate for first 5 frames
        self.plate_detection_start_frame = {}  # tid -> frame when detection started
        self.plate_candidates = defaultdict(list)  # tid -> [{text, conf, bbox, quality, frame}, ...]
        self.plate_locked = {}  # tid -> final locked plate text (after 5 frames)
        self.plate_detection_attempts = defaultdict(int)  # tid -> number of 5-frame windows tried
# cai thien hien thi toc do
    def _bbox_area(self, obj_meta):
        w = max(1.0, obj_meta.rect_params.width)
        h = max(1.0, obj_meta.rect_params.height)
        return float(w * h)

    def _valid_measurement(self, tid, frame_no, hist, speed_kmh, area_start, area_end, det_conf):
        # Tuổi track
        birth = self.track_birth_frame.get(tid, frame_no)
        age_frames = frame_no - birth
        if age_frames < MIN_TRACK_AGE_FRAMES:
            return False

        # Dịch chuyển tối thiểu
        if len(hist) >= 2:
            disp_m = abs(hist[-1] - hist[0])
            if disp_m < MIN_WORLD_DISPL_M:
                return False

        # Giới hạn vật lý
        if speed_kmh <= 0 or speed_kmh > MAX_ABS_KMH:
            return False

        # Ổn định kích thước bbox
        if area_start > 0 and area_end / area_start > BBOX_AREA_JUMP:
            return False

        # Độ tin cậy detection (nếu có)
        if det_conf is not None and det_conf < MIN_DET_CONF:
            return False

        return True

    def set_publisher(self, fn):
        """fn(payload: dict) -> None"""
        self.publisher = fn
    
    def _select_best_plate_from_candidates(self, candidates):
        """
        Select the best license plate from candidates using voting + quality.
        
        Strategy (realistic best choice):
        1. Group by plate text (voting)
        2. Select group with highest frequency (most common plate)
        3. Within that group, select entry with highest quality score
        
        Args:
            candidates: List of {text, conf, bbox, quality, frame}
        
        Returns:
            Best plate text (str) or None
        """
        if not candidates:
            return None
        
        # Filter out candidates without text
        valid_candidates = [c for c in candidates if c.get('text')]
        if not valid_candidates:
            return None
        
        # Group by plate text
        from collections import Counter
        text_groups = defaultdict(list)
        for candidate in valid_candidates:
            text_groups[candidate['text']].append(candidate)
        
        # Find the most common plate text (voting)
        text_frequencies = {text: len(entries) for text, entries in text_groups.items()}
        
        # Select plate text with highest frequency
        best_text = max(text_frequencies, key=text_frequencies.get)
        best_group = text_groups[best_text]
        
        # Within the best group, select entry with highest quality
        best_entry = max(best_group, key=lambda x: x.get('quality', 0))
        return best_text

    # -------------------- LPR text extraction --------------------
    @staticmethod
    def _extract_lpr_text(obj_meta):
        """
        Extract license plate text from classifier metadata (SGIE2 LPR output).
        Returns: string of recognized characters, or None if not available.
        """
        try:
            # Iterate through classifier metadata attached to this object
            class_meta_list = obj_meta.classifier_meta_list
            found_classifier = False
            while class_meta_list is not None:
                class_meta = pyds.NvDsClassifierMeta.cast(class_meta_list.data)
                
                # Check if this is from LPR classifier (gie-unique-id=3)
                if class_meta and class_meta.unique_component_id == 3:
                    # LPR typically outputs a single label with the full text
                    label_info_list = class_meta.label_info_list
                    if label_info_list is not None:
                        label_info = pyds.NvDsLabelInfo.cast(label_info_list.data)
                        if label_info and label_info.result_label:
                            return label_info.result_label
                
                class_meta_list = class_meta_list.next
            
            return None
        except Exception:
            return None

    # -------------------- license plate association --------------------
    @staticmethod
    def _bbox_iou(box1, box2):
        """Calculate IoU between two bounding boxes.
        box format: {left, top, width, height}
        """
        x1_min = box1['left']
        y1_min = box1['top']
        x1_max = x1_min + box1['width']
        y1_max = y1_min + box1['height']
        
        x2_min = box2['left']
        y2_min = box2['top']
        x2_max = x2_min + box2['width']
        y2_max = y2_min + box2['height']
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = box1['width'] * box1['height']
        box2_area = box2['width'] * box2['height']
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def _center_distance(box1, box2):
        """Calculate Euclidean distance between centers of two bounding boxes."""
        cx1 = box1['left'] + box1['width'] / 2.0
        cy1 = box1['top'] + box1['height'] / 2.0
        cx2 = box2['left'] + box2['width'] / 2.0
        cy2 = box2['top'] + box2['height'] / 2.0
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    def _associate_plate_to_vehicle(self, plate_bbox, vehicles_in_frame):
        """
        Associate license plate to nearest vehicle.
        Returns: vehicle_id of closest vehicle, or None
        """
        best_vehicle_id = None
        min_distance = float('inf')
        
        for vid, vbox in vehicles_in_frame.items():
            # Calculate distance between plate and vehicle
            dist = self._center_distance(plate_bbox, vbox)
            
            # Optional: also check if plate is spatially close (not too far away)
            # License plate should be within or very close to vehicle bbox
            if dist < min_distance and dist < 300:  # max 300 pixels distance
                # Additional check: plate should be roughly within vehicle's horizontal bounds
                # (to avoid matching plates from other lanes)
                plate_cx = plate_bbox['left'] + plate_bbox['width'] / 2.0
                v_left = vbox['left']
                v_right = vbox['left'] + vbox['width']
                
                # Allow some tolerance (±50%) for horizontal alignment
                h_tolerance = vbox['width'] * 0.5
                if v_left - h_tolerance <= plate_cx <= v_right + h_tolerance:
                    min_distance = dist
                    best_vehicle_id = vid
        
        return best_vehicle_id

    def _calculate_plate_quality(self, bbox, confidence):
        """
        Calculate quality score for license plate detection.
        Higher score = better quality plate.
        
        Factors:
        1. Confidence (weight: 70%) - Primary metric
        2. Bbox area (weight: 20%) - Larger = closer to camera = clearer
        3. Aspect ratio (weight: 10%) - Vietnamese plates support both 1-line and 2-line
        
        Returns: quality score (0-100)
        """
        # 1. Confidence score (0-70 points)
        conf_score = confidence * 70.0
        
        # 2. Area score (0-20 points) - normalized by typical plate size
        # Typical plate in 1920x1080: ~100x40 to 200x80 pixels
        area = bbox['width'] * bbox['height']
        # Normalize: 4000 (small) = 0, 16000 (large) = 20
        area_score = min(20.0, max(0.0, (area - 4000) / 12000 * 20))
        
        # 3. Aspect ratio score (0-10 points)
        # AUTO-DETECT plate type based on aspect ratio
        aspect = bbox['width'] / max(1.0, bbox['height'])
        
        # Vietnamese license plates:
        # - 1-line plates: aspect ≈ 2.5:1 to 3:1 (wider)
        # - 2-line plates: aspect ≈ 1:1 to 1.2:1 (more square)
        if aspect >= 1.8:
            # 1-line plate detected
            ideal_aspect = 2.5
            plate_type = "1-line"
        else:
            # 2-line plate detected
            ideal_aspect = 1.1
            plate_type = "2-line"
        
        aspect_diff = abs(aspect - ideal_aspect)
        # Score decreases as deviation from ideal increases
        aspect_score = max(0.0, 10.0 - aspect_diff * 2.0)
        
        total_score = conf_score + area_score + aspect_score
        
        # Optional: Store plate type for debugging (can be removed in production)
        # print(f"[DEBUG] Plate {plate_type}: aspect={aspect:.2f}, quality={total_score:.1f}")
        
        return total_score

    # -------------------- helpers --------------------
    def _compute_speed_kmh(self, hist):
        if len(hist) < int(VIDEO_FPS):
            return None
        distance_m = abs(hist[-1] - hist[0])
        time_s = (len(hist) - 1) / float(VIDEO_FPS)
        if time_s <= 0:
            return 0.0
        return (distance_m / time_s) * 3.6

    def _obj_in_analytics_roi(self, obj_meta) -> bool:
        """Check if object is inside ROI. Return True only if roiStatus indicates object is in ROI."""
        try:
            user_meta_list = obj_meta.obj_user_meta_list
            while user_meta_list is not None:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                if user_meta and hasattr(pyds, "nvds_get_user_meta_type"):
                    mt = pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META")
                    if user_meta.base_meta.meta_type == mt:
                        info = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
                        # Check roiStatus and return True if object is in any ROI
                        roi_status = getattr(info, "roiStatus", None)
                        if roi_status and len(roi_status) > 0:
                            # If any ROI flag is set, object is in ROI
                            return True
                user_meta_list = user_meta_list.next
            # No ROI metadata found => object is NOT in ROI => skip it
            return False
        except Exception as e:
            # On error, assume object is NOT in ROI (safe default)
            return False

    @staticmethod
    def _frame_bgr_from_gst_buffer(gst_buffer, frame_meta):
        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        img = np.array(surface, copy=True, order='C')
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


    @staticmethod
    def _crop_bbox(image_bgr, obj_meta):
        h, w = image_bgr.shape[:2]
        x  = int(round(obj_meta.rect_params.left))
        y  = int(round(obj_meta.rect_params.top))
        bw = int(round(obj_meta.rect_params.width))
        bh = int(round(obj_meta.rect_params.height))
        x = max(0, x); y = max(0, y)
        x2 = min(w, x + max(1, bw))
        y2 = min(h, y + max(1, bh))
        if x >= x2 or y >= y2:
            return None
        return image_bgr[y:y2, x:x2]

    @staticmethod
    def _jpg_b64_and_bytes(image_bgr, quality=85):
        ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None, None
        b = buf.tobytes()
        return base64.b64encode(b).decode("ascii"), b

    # speedflow/probes.py (thay _maybe_publish_and_save)
    def _maybe_publish_and_save(self, frame_iso_ts, track_id, speed_kmh, crop_bgr):
        now = time.time()

        image_b64 = None
        if crop_bgr is not None:
            image_b64, _ = self._jpg_b64_and_bytes(crop_bgr, JPEG_QUALITY)

        if self.publisher and (now - self.last_alert_ts[track_id] >= self.cooldown_s):
            self.last_alert_ts[track_id] = now
            
            # Get locked plate if available
            license_plate = self.plate_locked.get(track_id, None)
            
            payload = {
                "type": "overspeed",
                "ts": frame_iso_ts,
                "track_id": int(track_id),
                "speed_kmh": float(speed_kmh),
                "license_plate": license_plate,  # NEW: Add plate info
                "image_b64": image_b64,
            }
            try:
                self.publisher(payload)
            except Exception:
                pass

        if self.snap_count[track_id] < 1 and image_b64 is not None:
            self.snap_count[track_id] += 1


    # -------------------- main probe --------------------
    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        # ===== Ngưỡng mặc định (nếu bạn CHƯA thêm vào settings.py) =====
        # Gợi ý: đưa các hằng này sang settings.py để chỉnh từ 1 chỗ.
        try:
            MIN_TRACK_AGE_FRAMES
        except NameError:
            # cần tối thiểu ~0.5 giây tuổi track
            MIN_TRACK_AGE_FRAMES = int(VIDEO_FPS * 0.5)
        try:
            MIN_WORLD_DISPL_M
        except NameError:
            # dịch chuyển mặt đất tối thiểu trong cửa sổ
            MIN_WORLD_DISPL_M = 0.5
        try:
            MAX_ABS_KMH
        except NameError:
            # trần tốc độ hợp lý theo bối cảnh
            MAX_ABS_KMH = 160.0
        try:
            BBOX_AREA_JUMP
        except NameError:
            # nếu area_end / area_prev > BBOX_AREA_JUMP => coi là nhảy hình/zoom
            BBOX_AREA_JUMP = 2.5
        try:
            MIN_DET_CONF
        except NameError:
            # ngưỡng độ tin cậy detection (nếu có)
            MIN_DET_CONF = 0.45
        try:
            MEDIAN_WINDOW
        except NameError:
            # kích thước cửa sổ median smoothing cho tốc độ
            MEDIAN_WINDOW = 5

        # ===== Bộ nhớ tạm cần thiết (tự khởi tạo nếu chưa có trong __init__) =====

        if not hasattr(self, "speed_history"):
            self.speed_history = defaultdict(lambda: deque(maxlen=MEDIAN_WINDOW))
        if not hasattr(self, "track_birth_frame"):
            self.track_birth_frame = {}  # tid -> frame first-seen
        if not hasattr(self, "last_area"):
            self.last_area = {}          # tid -> bbox area ở frame trước

        def _bbox_area(obj_meta):
            w = max(1.0, obj_meta.rect_params.width)
            h = max(1.0, obj_meta.rect_params.height)
            return float(w * h)

        def _valid_measurement(tid, frame_no, hist, speed_kmh, area_prev, area_now, det_conf):
            # 1) tuổi track
            birth = self.track_birth_frame.get(tid, frame_no)
            age_frames = frame_no - birth
            if age_frames < MIN_TRACK_AGE_FRAMES:
                return False

            # 2) dịch chuyển mặt đất tối thiểu
            if len(hist) >= 2:
                disp_m = abs(hist[-1] - hist[0])
                if disp_m < MIN_WORLD_DISPL_M:
                    return False

            # 3) giới hạn vật lý
            if (speed_kmh is None) or (speed_kmh <= 0) or (speed_kmh > MAX_ABS_KMH):
                return False

            # 4) nhảy diện tích bbox (zoom/ID switch/dao động)
            if area_prev is not None and area_prev > 0:
                if (area_now / area_prev) > BBOX_AREA_JUMP:
                    return False

            # 5) độ tin cậy detection (nếu có)
            if det_conf is not None and det_conf < MIN_DET_CONF:
                return False

            return True

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_number = frame_meta.frame_num

            # timestamp (ISO)
            ts_ns = getattr(frame_meta, "ntp_timestamp", 0) or int(time.time() * 1e9)
            ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts_ns / 1e9))

            # ========== TWO-PASS APPROACH ==========
            # Pass 1: Collect all vehicles and license plates in current frame
            vehicles_in_frame = {}  # tid -> {bbox dict}
            plates_in_frame = []    # list of {bbox dict, obj_meta}
            
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                # chỉ xét các object nằm trong ROI analytics
                if not self._obj_in_analytics_roi(obj_meta):
                    l_obj = l_obj.next
                    continue

                # Collect vehicles
                if obj_meta.class_id in VEHICLE_CLASS_IDS:
                    tid = obj_meta.object_id
                    vehicles_in_frame[tid] = {
                        'left': obj_meta.rect_params.left,
                        'top': obj_meta.rect_params.top,
                        'width': obj_meta.rect_params.width,
                        'height': obj_meta.rect_params.height,
                        'obj_meta': obj_meta
                    }
                
                # Collect license plates
                elif obj_meta.class_id in LISENCE_PLATE_CLASS_IDS:
                    # Set display text to "license_plate" only (no ID)
                    obj_meta.text_params.display_text = "license_plate"
                    
                    plate_bbox = {
                        'left': obj_meta.rect_params.left,
                        'top': obj_meta.rect_params.top,
                        'width': obj_meta.rect_params.width,
                        'height': obj_meta.rect_params.height
                    }
                    plate_conf = getattr(obj_meta, "confidence", 0.0)
                    plates_in_frame.append({
                        'bbox': plate_bbox,
                        'obj_meta': obj_meta,
                        'conf': plate_conf
                    })
                
                l_obj = l_obj.next
            
            # Pass 2: Associate license plates to vehicles (5-Frame Detection Window)
            for plate_info in plates_in_frame:
                plate_bbox = plate_info['bbox']
                plate_conf = plate_info['conf']
                
                # Find closest vehicle
                vehicle_id = self._associate_plate_to_vehicle(plate_bbox, vehicles_in_frame)
                
                if vehicle_id is not None:
                    # === NEW: 5-Frame Detection Window Logic ===
                    
                    # Check if plate is already locked for this vehicle
                    if vehicle_id in self.plate_locked:
                        # Plate already locked - ignore new detections to save processing
                        continue
                    
                    # Initialize detection window for new vehicle
                    if vehicle_id not in self.plate_detection_start_frame:
                        self.plate_detection_start_frame[vehicle_id] = frame_number
                    
                    # Calculate frames elapsed since detection started
                    frames_in_window = frame_number - self.plate_detection_start_frame[vehicle_id]
                    
                    # Only collect candidates within the 5-frame window
                    if frames_in_window < self.PLATE_DETECTION_FRAMES:
                        # Extract LPR text
                        plate_text = self._extract_lpr_text(plate_info['obj_meta'])
                        
                        if plate_text:  # Only add if we got valid text
                            # Calculate quality score
                            quality = self._calculate_plate_quality(plate_bbox, plate_conf)
                            
                            # Add to candidates
                            self.plate_candidates[vehicle_id].append({
                                'text': plate_text,
                                'conf': plate_conf,
                                'bbox': plate_bbox,
                                'quality': quality,
                                'frame': frame_number
                            })
                    
                    # Window completed - select best plate and lock it
                    elif frames_in_window == self.PLATE_DETECTION_FRAMES:
                        candidates = self.plate_candidates[vehicle_id]
                        best_plate_text = self._select_best_plate_from_candidates(candidates)
                        
                        if best_plate_text:
                            # Lock the best plate
                            self.plate_locked[vehicle_id] = best_plate_text
                        else:
                            # No valid plate detected - retry another 5-frame window
                            self.plate_detection_attempts[vehicle_id] += 1
                            
                            if self.plate_detection_attempts[vehicle_id] < 3:  # Max 3 attempts (15 frames total)
                                # Reset for next window
                                self.plate_detection_start_frame[vehicle_id] = frame_number
                                self.plate_candidates[vehicle_id] = []
                            else:
                                self.plate_locked[vehicle_id] = None  # Mark as failed (no more attempts)
            
            
            # Pass 3: Process vehicles with speed calculation and display
            for tid, veh_info in vehicles_in_frame.items():
                obj_meta = veh_info['obj_meta']
                
                # tính world-coordinate từ chân bbox (cx, bottom_y)
                cx = obj_meta.rect_params.left + obj_meta.rect_params.width  / 2.0
                bottom_y = obj_meta.rect_params.top  + obj_meta.rect_params.height
                pts_world = self.view_transformer.transform_points(
                    np.array([[cx, bottom_y]], dtype=np.float32)
                )
                y_world = float(pts_world[0][1])

                hist = self.history_positions[tid]
                hist.append(y_world)

                # lưu thời điểm sinh track
                if tid not in self.track_birth_frame:
                    self.track_birth_frame[tid] = frame_number

                # area bbox hiện tại
                area_now = _bbox_area(obj_meta)
                area_prev = self.last_area.get(tid, None)

                # độ tin cậy detection (có thể None trên 1 số phiên bản)
                det_conf = getattr(obj_meta, "confidence", None)

                display_text = self.last_speed_text[tid] or f"#{tid}"

                # mỗi ~1s mới cập nhật một lần như code gốc
                if len(hist) >= int(VIDEO_FPS) and \
                (frame_number - self.last_update_frame[tid] >= int(VIDEO_FPS)):

                    speed_kmh = self._compute_speed_kmh(hist)

                    if _valid_measurement(tid, frame_number, hist, speed_kmh, area_prev, area_now, det_conf):
                        # median smoothing
                        sh = self.speed_history[tid]
                        sh.append(speed_kmh)
                        if len(sh) >= 3:
                            speed_smooth = float(np.median(sh))
                        else:
                            speed_smooth = speed_kmh

                        # Lưu tốc độ (không có # ID)
                        display_text = f"{int(speed_smooth)} km/h"
                        self.last_speed_text[tid]   = display_text
                        self.last_update_frame[tid] = frame_number

                        # --- OVERSPEED ---
                        if speed_smooth >= float(SPEED_LIMIT_KMH):
                            crop = None
                            # PERFORMANCE: Only extract frame when overspeed detected
                            try:
                                frame_bgr = self._frame_bgr_from_gst_buffer(gst_buffer, frame_meta)
                                if frame_bgr is not None and frame_bgr.size > 0:
                                    crop = self._crop_bbox(frame_bgr, obj_meta)
                            except Exception:
                                pass

                            self._maybe_publish_and_save(ts_iso, tid, speed_smooth, crop)
                    else:
                        # Phép đo không hợp lệ: không hiển thị gì
                        display_text = ""
                        self.last_speed_text[tid] = display_text


                # === Display Logic: Chỉ hiển thị khi đã có data ===
                final_display = ""
                
                # Get speed (if available)
                speed_text = self.last_speed_text.get(tid, "")
                
                # Get locked plate (if available) 
                plate_text = ""
                if tid in self.plate_locked:
                    locked_plate = self.plate_locked[tid]
                    if locked_plate:  # Successfully detected
                        plate_text = locked_plate
                
                # Build display: chỉ hiển thị khi có ít nhất 1 trong 2
                if speed_text or plate_text:
                    if speed_text and plate_text:
                        # Có cả tốc độ và biển số
                        final_display = f"{speed_text}\n{plate_text}"
                    elif speed_text:
                        # Chỉ có tốc độ
                        final_display = speed_text
                    elif plate_text:
                        # Chỉ có biển số
                        final_display = plate_text
                
                obj_meta.text_params.display_text = final_display

                # cập nhật area_prev cho lần sau
                self.last_area[tid] = area_now

            
            # Vẽ ROI box lên khung hình
            add_polygon_display(batch_meta, frame_meta, self.roi_points)
            
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK