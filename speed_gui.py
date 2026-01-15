import sys
import os
import cv2
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QPoint, QProcess
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
    QLineEdit, QTabWidget, QSpinBox, QMessageBox, QGroupBox, QGridLayout, QListWidget,
    QListWidgetItem, QComboBox, QTextEdit
)

# ========= Constants =========
PROCESSING_WIDTH = 1920
PROCESSING_HEIGHT = 1080

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms"
os.environ["QT_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins"

# ========= Data classes =========

@dataclass
class Calibration:
    # Real-world measurements in meters
    width_meters: float = 3.5      # Default lane width
    length_meters: float = 20.0    # Default measurement distance
    
    # SOURCE points (pixel coordinates on video)
    points: List[Tuple[int, int]] = field(default_factory=list)
    
    # Expanded ROI points (auto-calculated, 15-20% larger than SOURCE)
    expanded_roi: List[Tuple[int, int]] = field(default_factory=list)
    
    def get_target_width_cm(self) -> int:
        """Convert width from meters to centimeters"""
        return int(self.width_meters * 100)
    
    def get_target_height_cm(self) -> int:
        """Convert length from meters to centimeters"""
        return int(self.length_meters * 100)
    
    def calculate_target_points(self) -> List[List[int]]:
        """Auto-calculate TARGET points based on real-world measurements in meters"""
        # Return in meters, not centimeters
        w = int(self.width_meters)
        h = int(self.length_meters)
        return [[0, 0], [w, 0], [w, h], [0, h]]
    
    def calculate_expanded_roi(self, expansion_factor: float = 1.2):
        """Calculate expanded ROI by scaling SOURCE polygon outward from centroid"""
        if len(self.points) != 4:
            return
        
        # Calculate centroid
        cx = sum(x for x, y in self.points) / 4
        cy = sum(y for x, y in self.points) / 4
        
        # Expand each point outward from centroid with boundary clamping
        expanded = []
        for x, y in self.points:
            dx = x - cx
            dy = y - cy
            new_x = int(cx + dx * expansion_factor)
            new_y = int(cy + dy * expansion_factor)
            
            # Clamp to image boundaries [0, PROCESSING_WIDTH] and [0, PROCESSING_HEIGHT]
            new_x = max(0, min(new_x, PROCESSING_WIDTH))
            new_y = max(0, min(new_y, PROCESSING_HEIGHT))
            
            expanded.append((new_x, new_y))
        
        self.expanded_roi = expanded
    
    def to_yaml_dict(self):
        """Generate YAML structure with auto-calculated TARGET in meters"""
        target_points = self.calculate_target_points()
        # Convert cm to meters for TARGET_WIDTH and TARGET_HEIGHT
        return {
            "SOURCE": [[int(x), int(y)] for (x, y) in self.points],
            "TARGET_WIDTH": int(self.width_meters),  # Store in meters, not cm
            "TARGET_HEIGHT": int(self.length_meters),  # Store in meters, not cm
            "TARGET": target_points,
        }

@dataclass
class SourceItem:
    uri: str
    calib: Calibration = field(default_factory=Calibration)
    captured_frame: Optional[any] = None  # numpy array (BGR)
    last_preview_frame: Optional[any] = None  # numpy array (BGR)

# ========= Helpers =========

class QMutexLocker:
    def __init__(self, mutex: QMutex):
        self.mutex = mutex
        self.mutex.lock()
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.mutex.unlock()

# ========= Video thread (single preview at a time) =========

class VideoThread(QThread):
    frame_ready = pyqtSignal(object)   # numpy frame (BGR)
    opened = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self._source = None
        self._running = False
        self._mutex = QMutex()

    def set_source(self, source: str):
        with QMutexLocker(self._mutex):
            self._source = source

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False

    def run(self):
        with QMutexLocker(self._mutex):
            src = self._source
        self._running = True

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.opened.emit(False, f"Cannot open source: {src}")
            return
        self.opened.emit(True, src)

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None: break
                # Resize to match DeepStream pipeline default
                frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
                self.frame_ready.emit(frame)
                self.msleep(15)
        finally:
            cap.release()
            self._running = False

# ========= VideoWidget with overlay =========

class VideoWidget(QLabel):
    clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(720, 405)
        self.setAlignment(Qt.AlignCenter)
        self._frame = None
        self._pix = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._points: List[Tuple[int, int]] = []
        self._expanded_roi: List[Tuple[int, int]] = []  # NEW: for expanded ROI display

    def set_frame(self, frame_bgr):
        self._frame = frame_bgr
        self._update_pixmap()

    def get_frame(self): return self._frame

    def set_points(self, pts: List[Tuple[int, int]]):
        self._points = pts[:]
        self.update()

    def clear_points(self):
        self._points = []
        self.update()
    
    def set_expanded_roi(self, pts: List[Tuple[int, int]]):
        """Set expanded ROI points for display"""
        self._expanded_roi = pts[:]
        self.update()
    
    def clear_expanded_roi(self):
        """Clear expanded ROI display"""
        self._expanded_roi = []
        self.update()

    def _update_pixmap(self):
        if self._frame is None:
            self._pix = None
            self.clear()
            return
        h, w = self._frame.shape[:2]
        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scale = scaled.width() / w
        x_off = (self.width() - scaled.width()) // 2
        y_off = (self.height() - scaled.height()) // 2
        self._offset = QPoint(x_off, y_off)
        self._pix = scaled
        self.setPixmap(self._pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event):
        if self._frame is None: return
        if event.button() == Qt.LeftButton:
            x = (event.x() - self._offset.x()) / self._scale
            y = (event.y() - self._offset.y()) / self._scale
            h, w = self._frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                self.clicked.emit(int(x), int(y))

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pix is None: return
        p = QPainter(self)
        p.translate(self._offset)
        p.scale(self._scale, self._scale)

        # Draw expanded ROI first (yellow, dashed) - background layer
        if self._expanded_roi and len(self._expanded_roi) == 4:
            pen = QPen(QColor(255, 255, 0), 3, Qt.DashLine)
            p.setPen(pen)
            for i in range(len(self._expanded_roi)):
                x1, y1 = self._expanded_roi[i]
                x2, y2 = self._expanded_roi[(i + 1) % len(self._expanded_roi)]
                p.drawLine(x1, y1, x2, y2)

        # Draw SOURCE polygon (green, solid) - foreground layer
        if self._points:
            p.setPen(QPen(QColor(0, 255, 0), 2))
            for i in range(len(self._points)):
                x1, y1 = self._points[i]
                x2, y2 = self._points[(i + 1) % len(self._points)]
                p.drawLine(x1, y1, x2, y2)
            # points + index
            for idx, (x, y) in enumerate(self._points):
                p.setPen(QPen(QColor(255, 0, 0), 6))
                p.drawPoint(x, y)
                p.setPen(QPen(QColor(255, 255, 0), 1))
                p.drawText(x + 5, y - 5, str(idx))

# ========= Main Window =========

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepStream Speed - Multi Source GUI")
        self.setMinimumSize(1200, 800)

        # Multi sources storage (uri -> SourceItem)
        self.sources: Dict[str, SourceItem] = {}

        self.tabs = QTabWidget(self)
        self.tab_sources = QWidget()
        self.tab_calib = QWidget()
        self.tab_run = QWidget()
        self.tabs.addTab(self.tab_sources, "Nguồn (nhiều File/RTSP)")
        self.tabs.addTab(self.tab_calib, "Hiệu chuẩn")
        self.tabs.addTab(self.tab_run, "Phát")

        root = QVBoxLayout(self)
        root.addWidget(self.tabs)

        self.vthread = VideoThread()
        self.vthread.frame_ready.connect(self.on_frame_ready)
        self.vthread.opened.connect(self.on_opened)

        self._build_tab_sources()
        self._build_tab_calib()
        self._build_tab_run()

    # ---------- Tab: Sources ----------
    def _build_tab_sources(self):
        lay = QVBoxLayout(self.tab_sources)

        # Input + buttons
        row = QHBoxLayout()
        self.le_uri = QLineEdit()
        self.le_uri.setPlaceholderText("Đường dẫn file hoặc rtsp://user:pass@host:port/...")
        self.btn_browse_file = QPushButton("Browse file")
        self.btn_add = QPushButton("Thêm nguồn")
        self.btn_remove = QPushButton("Xoá nguồn")
        row.addWidget(self.le_uri)
        row.addWidget(self.btn_browse_file)
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_remove)

        # List of sources
        self.list_sources = QListWidget()

        # Preview controls (selected)
        row2 = QHBoxLayout()
        self.btn_preview = QPushButton("Start")
        self.btn_stop_preview = QPushButton("Stop")
        row2.addWidget(self.btn_preview)
        row2.addWidget(self.btn_stop_preview)

        # Live preview widget
        self.preview_widget = VideoWidget()

        lay.addLayout(row)
        lay.addWidget(self.list_sources)
        lay.addLayout(row2)
        lay.addWidget(self.preview_widget)

        self.btn_browse_file.clicked.connect(self.on_browse_file)
        self.btn_add.clicked.connect(self.on_add_source)
        self.btn_remove.clicked.connect(self.on_remove_source)
        self.btn_preview.clicked.connect(self.on_start_preview_selected)
        self.btn_stop_preview.clicked.connect(self.on_stop_preview)

    def on_browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Videos (*.mp4 *.avi *.mkv *.MOV);;All files (*)")
        if path: self.le_uri.setText(path)

    def on_add_source(self):
        uri = self.le_uri.text().strip()
        if not uri:
            QMessageBox.warning(self, "Thiếu URL", "Nhập đường dẫn file hoặc RTSP URL.")
            return
        if not (uri.startswith("rtsp://") or os.path.exists(uri)):
            QMessageBox.warning(self, "Nguồn không hợp lệ", "Nhập file tồn tại hoặc RTSP URL bắt đầu bằng rtsp://")
            return
        if uri in self.sources:
            QMessageBox.information(self, "Đã tồn tại", "Nguồn này đã có trong danh sách.")
            return
        self.sources[uri] = SourceItem(uri=uri)
        self.list_sources.addItem(QListWidgetItem(uri))
        self.le_uri.clear()

        # cập nhật combo ở các tab khác
        self._refresh_source_selectors()

    def on_remove_source(self):
        item = self.list_sources.currentItem()
        if not item: return
        uri = item.text()
        self.sources.pop(uri, None)
        self.list_sources.takeItem(self.list_sources.row(item))
        self.preview_widget.set_frame(None)
        self._refresh_source_selectors()

    def on_start_preview_selected(self):
        item = self.list_sources.currentItem()
        if not item:
            QMessageBox.information(self, "Chưa chọn", "Hãy chọn một nguồn trong danh sách.")
            return
        uri = item.text()
        self.vthread.set_source(uri)
        self.vthread.start()

    def on_stop_preview(self):
        if self.vthread.isRunning():
            self.vthread.stop()
            self.vthread.wait(500)

    def on_opened(self, ok: bool, msg: str):
        if not ok:
            QMessageBox.critical(self, "Không mở được nguồn", msg)

    def on_frame_ready(self, frame):
        # show live
        self.preview_widget.set_frame(frame)
        # remember last frame for this source
        if self.vthread.isRunning():
            # try to deduce current source
            uri = self.le_uri.text().strip()
            # better: scan selected item text if any
            item = self.list_sources.currentItem()
            current_uri = item.text() if item else None
            key = current_uri or uri
            if key in self.sources:
                self.sources[key].last_preview_frame = frame

    # ---------- Helpers (combo sync) ----------
    def _refresh_source_selectors(self):
        uris = list(self.sources.keys())
        # Calibration combo
        self.cb_source_calib.blockSignals(True)
        self.cb_source_calib.clear()
        self.cb_source_calib.addItems(uris)
        self.cb_source_calib.blockSignals(False)

        # Run combo
        self.cb_source_run.blockSignals(True)
        self.cb_source_run.clear()
        self.cb_source_run.addItems(uris)
        self.cb_source_run.blockSignals(False)

    # ---------- Tab: Calibration ----------
    def _build_tab_calib(self):
        lay = QVBoxLayout(self.tab_calib)

        # choose source
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Nguồn:"))
        self.cb_source_calib = QComboBox()
        src_row.addWidget(self.cb_source_calib)

        self.video_widget = VideoWidget()
        self.video_widget.clicked.connect(self.on_video_clicked)

        # controls
        ctrl = QGroupBox("THIẾT LẬP CÁC THÔNG SỐ")
        grid = QGridLayout(ctrl)

        # Real-world measurements (meters) - import QDoubleSpinBox at top
        from PyQt5.QtWidgets import QDoubleSpinBox
        
        self.dsb_width_m = QDoubleSpinBox()
        self.dsb_width_m.setRange(0.5, 50.0)
        self.dsb_width_m.setValue(3.5)
        self.dsb_width_m.setSingleStep(0.5)
        self.dsb_width_m.setSuffix(" m")
        
        self.dsb_length_m = QDoubleSpinBox()
        self.dsb_length_m.setRange(1.0, 200.0)
        self.dsb_length_m.setValue(20.0)
        self.dsb_length_m.setSingleStep(1.0)
        self.dsb_length_m.setSuffix(" m")

        self.btn_input_measurements = QPushButton("📏 Nhập kích thước thực tế")
        self.btn_use_last = QPushButton("Khung hình hiện tại")
        self.btn_capture = QPushButton("Ảnh")
        self.btn_clear = QPushButton("Xoá điểm")
        self.btn_save_yaml = QPushButton("💾 Lưu cấu hình")

        grid.addWidget(QLabel("Chiều rộng làn đường:"), 0, 0)
        grid.addWidget(self.dsb_width_m, 0, 1)
        grid.addWidget(QLabel("Chiều dài vùng đo:"), 0, 2)
        grid.addWidget(self.dsb_length_m, 0, 3)
        grid.addWidget(self.btn_input_measurements, 1, 0, 1, 4)
        grid.addWidget(self.btn_use_last, 2, 0, 1, 2)
        grid.addWidget(self.btn_capture, 2, 2, 1, 2)
        grid.addWidget(self.btn_clear, 3, 0, 1, 2)
        grid.addWidget(self.btn_save_yaml, 3, 2, 1, 2)

        lay.addLayout(src_row)
        lay.addWidget(self.video_widget)
        lay.addWidget(ctrl)

        self.cb_source_calib.currentTextChanged.connect(self.on_change_calib_source)
        self.btn_input_measurements.clicked.connect(self.on_input_measurements)
        self.btn_use_last.clicked.connect(self.on_use_last_frame)
        self.btn_capture.clicked.connect(self.on_capture_freeze)
        self.btn_clear.clicked.connect(self.on_clear_points)
        self.btn_save_yaml.clicked.connect(self.on_save_yaml)

    def on_change_calib_source(self, uri: str):
        if not uri: 
            self.video_widget.set_frame(None)
            self.video_widget.clear_points()
            self.video_widget.clear_expanded_roi()
            return
        si = self.sources.get(uri)
        if not si:
            self.video_widget.set_frame(None)
            self.video_widget.clear_points()
            self.video_widget.clear_expanded_roi()
            return
        # load measurements (meters) and points
        self.dsb_width_m.setValue(si.calib.width_meters)
        self.dsb_length_m.setValue(si.calib.length_meters)
        self.video_widget.set_points(si.calib.points)
        self.video_widget.set_expanded_roi(si.calib.expanded_roi)
        # show captured frame or last preview
        frame = si.captured_frame or si.last_preview_frame
        self.video_widget.set_frame(frame)

    def on_use_last_frame(self):
        uri = self.cb_source_calib.currentText()
        if uri not in self.sources: return
        si = self.sources[uri]
        if si.last_preview_frame is None:
            QMessageBox.information(self, "Chưa có frame", "Hãy Start Preview ở tab Nguồn rồi thử lại.")
            return
        si.captured_frame = si.last_preview_frame.copy()
        self.video_widget.set_frame(si.captured_frame)

    def on_capture_freeze(self):
        uri = self.cb_source_calib.currentText()
        if uri not in self.sources: return
        # lấy frame đang hiển thị tại widget (có thể từ preview)
        frame = self.video_widget.get_frame()
        if frame is None:
            QMessageBox.information(self, "Chưa có frame", "Hãy Start Preview và bấm 'Dùng khung hình preview hiện tại'.")
            return
        self.sources[uri].captured_frame = frame.copy()
        self.video_widget.set_frame(self.sources[uri].captured_frame)
        # (không lưu ra đĩa)
    
    def on_input_measurements(self):
        """Show dialog to input real-world measurements"""
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox
        
        uri = self.cb_source_calib.currentText()
        if uri not in self.sources:
            QMessageBox.information(self, "Chưa chọn nguồn", "Hãy chọn một nguồn trong combo.")
            return
        
        si = self.sources[uri]
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Nhập kích thước thực tế")
        layout = QFormLayout(dialog)
        
        # Width input
        width_input = QDoubleSpinBox()
        width_input.setRange(0.5, 50.0)
        width_input.setValue(si.calib.width_meters)
        width_input.setSingleStep(0.5)
        width_input.setSuffix(" m")
        layout.addRow("Chiều rộng làn đường:", width_input)
        
        # Length input
        length_input = QDoubleSpinBox()
        length_input.setRange(1.0, 200.0)
        length_input.setValue(si.calib.length_meters)
        length_input.setSingleStep(1.0)
        length_input.setSuffix(" m")
        layout.addRow("Chiều dài vùng đo:", length_input)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            si.calib.width_meters = width_input.value()
            si.calib.length_meters = length_input.value()
            self.dsb_width_m.setValue(si.calib.width_meters)
            self.dsb_length_m.setValue(si.calib.length_meters)
            QMessageBox.information(
                self, "Đã cập nhật",
                f"Kích thước: {si.calib.width_meters}m x {si.calib.length_meters}m"
            )

    def on_video_clicked(self, x, y):
        uri = self.cb_source_calib.currentText()
        if uri not in self.sources: return
        si = self.sources[uri]
        # cần có frame cố định để chọn điểm
        if si.captured_frame is None and si.last_preview_frame is None:
            QMessageBox.information(self, "Chưa có frame", "Hãy lấy khung hình (Use preview / Capture).")
            return
        pts = si.calib.points
        if len(pts) >= 4:
            QMessageBox.information(self, "Đã đủ 4 điểm", "Bấm 'Xoá điểm' nếu muốn chọn lại.")
            return
        pts.append((x, y))
        self.video_widget.set_points(pts)
        
        # Auto-calculate expanded ROI when 4 points are selected
        if len(pts) == 4:
            si.calib.calculate_expanded_roi(expansion_factor=1.2)
            self.video_widget.set_expanded_roi(si.calib.expanded_roi)
            QMessageBox.information(
                self, "Hoàn tất",
                "Đã chọn đủ 4 điểm!\nVùng ROI mở rộng (vàng) đã được tính tự động."
            )

    def on_clear_points(self):
        uri = self.cb_source_calib.currentText()
        if uri not in self.sources: return
        self.sources[uri].calib.points = []
        self.sources[uri].calib.expanded_roi = []
        self.video_widget.clear_points()
        self.video_widget.clear_expanded_roi()

    def on_save_yaml(self):
        uri = self.cb_source_calib.currentText()
        if uri not in self.sources:
            QMessageBox.information(self, "Chưa chọn nguồn", "Hãy chọn một nguồn trong combo.")
            return
        si = self.sources[uri]
        if len(si.calib.points) != 4:
            QMessageBox.warning(self, "Thiếu điểm", "Hãy click đủ 4 điểm.")
            return
        if si.captured_frame is None and si.last_preview_frame is None:
            QMessageBox.warning(self, "Thiếu ảnh", "Hãy dùng khung hình preview hoặc đóng băng khung hình trước.")
            return

        # Update measurements from UI
        si.calib.width_meters = self.dsb_width_m.value()
        si.calib.length_meters = self.dsb_length_m.value()
        
        # Recalculate expanded ROI with current measurements
        si.calib.calculate_expanded_roi(expansion_factor=1.2)
        
        # Choose save location for YAML
        default_dir = os.path.join(os.getcwd(), "configs")
        os.makedirs(default_dir, exist_ok=True)
        default_path = os.path.join(default_dir, "points_source_target.yml")
        
        yaml_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu file YAML cấu hình",
            default_path,
            "YAML (*.yml *.yaml);;All files (*)"
        )
        if not yaml_path:
            return
        
        # Write YAML file
        with open(yaml_path, "w") as f:
            yaml.safe_dump(si.calib.to_yaml_dict(), f, sort_keys=False)
        
        # Write config_nvdsanalytics.txt
        analytics_path = os.path.join(default_dir, "config_nvdsanalytics.txt")
        self._write_analytics_config(analytics_path, si.calib.expanded_roi)
        
        QMessageBox.information(
            self, "Đã lưu",
            f"✅ YAML: {yaml_path}\n✅ Analytics: {analytics_path}"
        )
    
    def _write_analytics_config(self, path: str, roi_points: List[Tuple[int, int]]):
        """Write or update config_nvdsanalytics.txt with ROI coordinates"""
        if len(roi_points) != 4:
            return
        
        # Format ROI string: "x1;y1;x2;y2;x3;y3;x4;y4"
        roi_str = ";".join([f"{int(x)};{int(y)}" for x, y in roi_points])
        
        # Read existing config or create new
        config_lines = []
        if os.path.exists(path):
            with open(path, "r") as f:
                config_lines = f.readlines()
        
        # Update or add ROI line
        roi_updated = False
        new_lines = []
        for line in config_lines:
            if line.strip().startswith("roi-RF="):
                new_lines.append(f"roi-RF={roi_str}\n")
                roi_updated = True
            else:
                new_lines.append(line)
        
        # If ROI line doesn't exist, create minimal config
        if not roi_updated:
            new_lines = [
                "[property]\n",
                "enable=1\n",
                f"config-width={PROCESSING_WIDTH}\n",
                f"config-height={PROCESSING_HEIGHT}\n",
                "osd-mode=2\n",
                "display-font-size=12\n",
                "\n",
                "[roi-filtering-stream-0]\n",
                "enable=1\n",
                f"roi-RF={roi_str}\n",
                "inverse-roi=0\n",
                "class-id=-1\n",
            ]
        
        # Write config
        with open(path, "w") as f:
            f.writelines(new_lines)

    # ---------- Tab: Run ----------
    def _build_tab_run(self):
        lay = QVBoxLayout(self.tab_run)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Nguồn:"))
        self.cb_source_run = QComboBox()
        src_row.addWidget(self.cb_source_run)

        # Ô cấu hình + nút Browse
        cfg_row = QHBoxLayout()
        self.le_homo = QLineEdit(os.path.join(os.getcwd(), "configs", "points_1.yml"))
        self.btn_browse_homo = QPushButton("Browse...")
        cfg_row.addWidget(self.le_homo)
        cfg_row.addWidget(self.btn_browse_homo)

        btn_row = QHBoxLayout()
        self.btn_run_file_disp = QPushButton("Hiển thị file")
        self.btn_run_file_mp4 = QPushButton("Ghi MP4")
        self.btn_run_rtsp_disp = QPushButton("Hiển thị RTSP")
        self.btn_stop = QPushButton("Stop")

        btn_row.addWidget(self.btn_run_file_disp)
        btn_row.addWidget(self.btn_run_file_mp4)
        btn_row.addWidget(self.btn_run_rtsp_disp)
        btn_row.addWidget(self.btn_stop)

        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)

        lay.addLayout(src_row)
        lay.addWidget(QLabel("Cấu hình (YAML homography/points):"))
        lay.addLayout(cfg_row)
        lay.addLayout(btn_row)
        lay.addWidget(self.txt_log)

        self.proc: Optional[QProcess] = None

        self.btn_browse_homo.clicked.connect(self.on_browse_homo)
        self.btn_run_file_disp.clicked.connect(self.on_run_file_display)
        self.btn_run_file_mp4.clicked.connect(self.on_run_file_mp4)
        self.btn_run_rtsp_disp.clicked.connect(self.on_run_rtsp_display)
        self.btn_stop.clicked.connect(self.on_stop_proc)

    def on_browse_homo(self):
        default_dir = os.path.join(os.getcwd(), "configs")
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file YAML cấu hình", default_dir,
            "YAML (*.yml *.yaml);;All files (*)"
        )
        if path:
            self.le_homo.setText(path)

    def _ensure_proc(self):
        # Check if process is already running
        if self.proc and self.proc.state() == QProcess.Running:
            QMessageBox.warning(self, "Đang chạy", "Pipeline đang chạy. Hãy Stop trước khi chạy lệnh mới.")
            return False
        
        if self.proc is None:
            self.proc = QProcess(self)
            self.proc.setProcessChannelMode(QProcess.MergedChannels)
            self.proc.readyReadStandardOutput.connect(self._read_proc_out)
            self.proc.finished.connect(lambda: self.txt_log.append("=== Finished ==="))
        return True

    def _read_proc_out(self):
        if not self.proc: return
        out = bytes(self.proc.readAllStandardOutput()).decode(errors="ignore")
        if out: self.txt_log.append(out)

    def on_run_file_display(self):
        uri = self.cb_source_run.currentText()
        if not uri:
            QMessageBox.information(self, "Chưa chọn nguồn", "Chọn một nguồn trước.")
            return
        if not os.path.exists(uri):
            QMessageBox.warning(self, "Không phải file", "Nguồn không phải file nội bộ.")
            return
        
        # Check if main.py exists
        if not os.path.exists("main.py"):
            QMessageBox.critical(self, "Lỗi", "Không tìm thấy main.py trong thư mục hiện tại.")
            return
        
        # Use main.py with --mode display for file display
        homo = self.le_homo.text().strip()
        if not homo or not os.path.exists(homo):
            QMessageBox.warning(self, "Thiếu YAML", "Homography YAML không tồn tại.")
            return
        
        if not self._ensure_proc():
            return
        
        cmd = f"python3 main.py --source {uri} --mode display --homo {homo} --width {PROCESSING_WIDTH} --height {PROCESSING_HEIGHT}"
        self.txt_log.append(f"$ {cmd}")
        self.proc.start("bash", ["-c", cmd])

    def on_run_file_mp4(self):
        uri = self.cb_source_run.currentText()
        if not uri or not os.path.exists(uri):
            QMessageBox.warning(self, "Không phải file", "Chọn một file nội bộ để ghi MP4.")
            return
        
        # Check if main.py exists
        if not os.path.exists("main.py"):
            QMessageBox.critical(self, "Lỗi", "Không tìm thấy main.py trong thư mục hiện tại.")
            return
        
        homo = self.le_homo.text().strip()
        if not os.path.exists(homo):
            QMessageBox.warning(self, "Thiếu YAML", "Homography YAML không tồn tại.")
            return
        
        if not self._ensure_proc():
            return
        
        os.makedirs("output", exist_ok=True)
        out = os.path.join("output", "output.mp4")
        
        # Use main.py with --mode file to export MP4
        cmd = f"python3 main.py --source {uri} --mode file --output {out} --homo {homo} --width {PROCESSING_WIDTH} --height {PROCESSING_HEIGHT}"
        self.txt_log.append(f"$ {cmd}")
        self.proc.start("bash", ["-c", cmd])

    def on_run_rtsp_display(self):
        uri = self.cb_source_run.currentText()
        if not uri.startswith("rtsp://"):
            QMessageBox.warning(self, "Không phải RTSP", "Chọn một RTSP URL (rtsp://...).")
            return
        
        # Check if main.py exists
        if not os.path.exists("main.py"):
            QMessageBox.critical(self, "Lỗi", "Không tìm thấy main.py trong thư mục hiện tại.")
            return
        
        homo = self.le_homo.text().strip()
        if not homo or not os.path.exists(homo):
            QMessageBox.warning(self, "Thiếu YAML", "Homography YAML không tồn tại.")
            return
        
        if not self._ensure_proc():
            return
        
        cmd = f"python3 main.py --source {uri} --mode display --homo {homo} --width {PROCESSING_WIDTH} --height {PROCESSING_HEIGHT}"
        self.txt_log.append(f"$ {cmd}")
        self.proc.start("bash", ["-c", cmd])

    def on_stop_proc(self):
        if self.proc:
            self.proc.kill()
            self.proc = None
            self.txt_log.append("=== Stopped ===")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
