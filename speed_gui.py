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

# ========= Data classes =========

@dataclass
class Calibration:
    target_width: int = 25
    target_height: int = 170
    points: List[Tuple[int, int]] = field(default_factory=list)

    def to_yaml_dict(self):
        tw = int(self.target_width)
        th = int(self.target_height)
        return {
            "SOURCE": [[int(x), int(y)] for (x, y) in self.points],
            "TARGET_WIDTH": tw,
            "TARGET_HEIGHT": th,
            "TARGET": [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]],
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

        # polygon
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

        self.sb_tw = QSpinBox(); self.sb_tw.setRange(1, 10000); self.sb_tw.setValue(10)
        self.sb_th = QSpinBox(); self.sb_th.setRange(1, 10000); self.sb_th.setValue(100)

        self.btn_use_last = QPushButton("khung hình hiện tại")
        self.btn_capture = QPushButton("Ảnh")
        self.btn_clear = QPushButton("Xoá điểm")
        self.btn_save_yaml = QPushButton("Lưu")  # đổi tên để gợi ý Save As

        grid.addWidget(QLabel("TARGET_WIDTH:"), 0, 0); grid.addWidget(self.sb_tw, 0, 1)
        grid.addWidget(QLabel("TARGET_HEIGHT:"), 0, 2); grid.addWidget(self.sb_th, 0, 3)
        grid.addWidget(self.btn_use_last, 1, 0, 1, 2)
        grid.addWidget(self.btn_capture, 1, 2, 1, 2)
        grid.addWidget(self.btn_clear, 2, 0, 1, 2)
        grid.addWidget(self.btn_save_yaml, 2, 2, 1, 2)

        lay.addLayout(src_row)
        lay.addWidget(self.video_widget)
        lay.addWidget(ctrl)

        self.cb_source_calib.currentTextChanged.connect(self.on_change_calib_source)
        self.btn_use_last.clicked.connect(self.on_use_last_frame)
        self.btn_capture.clicked.connect(self.on_capture_freeze)
        self.btn_clear.clicked.connect(self.on_clear_points)
        self.btn_save_yaml.clicked.connect(self.on_save_yaml)

    def on_change_calib_source(self, uri: str):
        if not uri: 
            self.video_widget.set_frame(None)
            self.video_widget.clear_points()
            return
        si = self.sources.get(uri)
        if not si:
            self.video_widget.set_frame(None); self.video_widget.clear_points(); return
        # load target sizes and points
        self.sb_tw.setValue(si.calib.target_width)
        self.sb_th.setValue(si.calib.target_height)
        self.video_widget.set_points(si.calib.points)
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

    def on_clear_points(self):
        uri = self.cb_source_calib.currentText()
        if uri not in self.sources: return
        self.sources[uri].calib.points = []
        self.video_widget.clear_points()

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

        # cập nhật target
        si.calib.target_width = int(self.sb_tw.value())
        si.calib.target_height = int(self.sb_th.value())

        # Hộp thoại Save As để chọn nơi lưu/tên file
        default_dir = os.path.join(os.getcwd(), "configs")
        os.makedirs(default_dir, exist_ok=True)
        default_path = os.path.join(default_dir, "points_1.yml")
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu file YAML cấu hình",
            default_path,
            "YAML (*.yml *.yaml);;All files (*)"
        )
        if not path:
            return

        with open(path, "w") as f:
            yaml.safe_dump(si.calib.to_yaml_dict(), f, sort_keys=False)
        QMessageBox.information(self, "Đã lưu", f"YAML: {path}")

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
        if self.proc is None:
            self.proc = QProcess(self)
            self.proc.setProcessChannelMode(QProcess.MergedChannels)
            self.proc.readyReadStandardOutput.connect(self._read_proc_out)
            self.proc.finished.connect(lambda: self.txt_log.append("=== Finished ==="))

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
        # Use main.py with --mode display for file display
        homo = self.le_homo.text().strip()
        if not homo or not os.path.exists(homo):
            QMessageBox.warning(self, "Thiếu YAML", "Homography YAML không tồn tại.")
            return
        self._ensure_proc()
        self.txt_log.append(f"$ python3 main.py --source {uri} --mode display --homo {homo}")
        self.proc.start("python3", ["main.py", "--source", uri, "--mode", "display", "--homo", homo])

    def on_run_file_mp4(self):
        uri = self.cb_source_run.currentText()
        if not uri or not os.path.exists(uri):
            QMessageBox.warning(self, "Không phải file", "Chọn một file nội bộ để ghi MP4.")
            return
        homo = self.le_homo.text().strip()
        if not os.path.exists(homo):
            QMessageBox.warning(self, "Thiếu YAML", "Homography YAML không tồn tại.")
            return
        os.makedirs("outputs", exist_ok=True)
        out = os.path.join("outputs", "output.mp4")
        # Use main.py with --mode file to export MP4
        self._ensure_proc()
        self.txt_log.append(f"$ python3 main.py --source {uri} --mode file --output {out} --homo {homo}")
        self.proc.start("python3", ["main.py", "--source", uri, "--mode", "file", "--output", out, "--homo", homo])

    def on_run_rtsp_display(self):
        uri = self.cb_source_run.currentText()
        if not uri.startswith("rtsp://") and not uri.startswith("file://"):
            QMessageBox.warning(self, "Không phải RTSP", "Chọn một RTSP URL (rtsp://...).")
            return
        homo = self.le_homo.text().strip()
        if not homo or not os.path.exists(homo):
            QMessageBox.warning(self, "Thiếu YAML", "Homography YAML không tồn tại.")
            return
        self._ensure_proc()
        self.txt_log.append(f"$ python3 main.py --source {uri} --mode display --homo {homo}")
        self.proc.start("python3", ["main.py", "--source", uri, "--mode", "display", "--homo", homo])

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
