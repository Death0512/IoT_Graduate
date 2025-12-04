
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

        # logger CSV (tuỳ)
        # self.logger = CSVLogger(SPEED_LOG, header=["frame","track_id","speed_km_h"])

        # đảm bảo thư mục tồn tại
        try:
            os.makedirs(str(SNAP_DIR), exist_ok=True)
        except Exception:
            pass
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
        """Nếu bạn có NvDsAnalytics và bật roiStatus thì lọc theo ROI; nếu không có thì luôn True."""
        try:
            user_meta_list = obj_meta.obj_user_meta_list
            while user_meta_list is not None:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                # Nếu SDK của bạn không có hằng số meta_type này, trả về True để không chặn
                if user_meta and hasattr(pyds, "nvds_get_user_meta_type"):
                    mt = pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META")
                    if user_meta.base_meta.meta_type == mt:
                        info = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
                        if getattr(info, "roiStatus", None):
                            return True
                user_meta_list = user_meta_list.next
            # không thấy meta => không lọc
            return True
        except Exception:
            return True

    @staticmethod
    def _frame_bgr_from_gst_buffer(gst_buffer, frame_meta):
        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        img = np.array(surface, copy=True, order='C')
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # DEBUG: in ra 1 lần để biết đã lấy được khung
        if not hasattr(SpeedProbe, "_dbg_frame_once"):
            print("[DBG] frame_bgr OK ->", img.shape)
            SpeedProbe._dbg_frame_once = True
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
            payload = {
                "type": "overspeed",
                "ts": frame_iso_ts,
                "track_id": int(track_id),
                "speed_kmh": float(speed_kmh),
                "image_b64": image_b64,
            }
            try:
                self.publisher(payload)
            except Exception as e:
                print("[WARN] publish overspeed failed:", e)

        if self.snap_count[track_id] < 1 and image_b64 is not None:
            self.snap_count[track_id] += 1


    # -------------------- main probe --------------------
    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        """
        - Chỉ hiển thị/tính tốc độ khi phép đo HỢP LỆ để loại bỏ tốc độ ảo.
        - Các ngưỡng có thể đặt trong settings.py; nếu chưa có, dùng default bên dưới.
        """

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

            # lấy frame BGR cho crop (nếu cần)
            try:
                frame_bgr = self._frame_bgr_from_gst_buffer(gst_buffer, frame_meta)
                if frame_bgr is None or frame_bgr.size == 0:
                    print("[DBG] frame_bgr is None/empty")
            except Exception as e:
                print("[ERR] get frame_bgr failed:", e)
                frame_bgr = None

            l_obj = frame_meta.obj_meta_list
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

                # chỉ xét các object nằm trong ROI analytics
                if not self._obj_in_analytics_roi(obj_meta):
                    l_obj = l_obj.next
                    continue

                if obj_meta.class_id in VEHICLE_CLASS_IDS:
                    # tính world-coordinate từ chân bbox (cx, bottom_y)
                    cx = obj_meta.rect_params.left + obj_meta.rect_params.width  / 2.0
                    bottom_y = obj_meta.rect_params.top  + obj_meta.rect_params.height
                    pts_world = self.view_transformer.transform_points(
                        np.array([[cx, bottom_y]], dtype=np.float32)
                    )
                    y_world = float(pts_world[0][1])

                    tid = obj_meta.object_id
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

                            display_text = f"#{tid} {int(speed_smooth)} km/h"
                            self.last_speed_text[tid]   = display_text
                            self.last_update_frame[tid] = frame_number

                            # --- OVERSPEED ---
                            if speed_smooth >= float(SPEED_LIMIT_KMH):
                                crop = None
                                if frame_bgr is not None:
                                    crop = self._crop_bbox(frame_bgr, obj_meta)
                                    if crop is not None and crop.size > 0 and not hasattr(self, "_dbg_crop_once"):
                                        print(f"[DBG] got first CROP shape={crop.shape} for track {tid}")
                                        self._dbg_crop_once = True

                                self._maybe_publish_and_save(ts_iso, tid, speed_smooth, crop)
                        else:
                            # phép đo không hợp lệ: chỉ hiển thị id
                            display_text = f"#{tid}"
                            self.last_speed_text[tid] = display_text

                    # Hiển thị OSD
                    obj_meta.text_params.display_text = display_text

                    # cập nhật area_prev cho lần sau
                    self.last_area[tid] = area_now

                l_obj = l_obj.next
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK