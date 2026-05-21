import sys
import os
import threading
import numpy as np
import cv2

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QRadioButton, QGroupBox, QButtonGroup, QFileDialog, 
                             QTextEdit, QMessageBox, QSplitter)
from PyQt6.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt6.QtGui import QFont, QImage, QPixmap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

VIDEO_POLYGON_MAP = {
    "easy_01": [[662, 1410], [1399, 1384], [1656, 1629], [805, 1671]],
    "easy_02": [[408, 873], [847, 846], [1030, 1071], [366, 1095]],
    "medium_01": [[751, 2257], [492, 2599], [1760, 2575], [1598, 2257]],
    "medium_02": [[1188, 568], [700, 1338], [3072, 1334], [2609, 663]],
    "medium_03": [[1372, 986], [1163, 1473], [2691, 1445], [2392, 982]],
    "medium_05": [[1827, 1424], [1630, 1793], [3133, 1768], [2904, 1383]],
}

class VideoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    log_msg = pyqtSignal(str)
    process_finished = pyqtSignal(int)
    
    def __init__(self, action_id, input_file, output_folder):
        super().__init__()
        self.action_id = action_id
        self.input_file = input_file
        self.output_folder = output_folder
        self.pipeline = None

    def run(self):
        try:
            filename = os.path.basename(self.input_file)
            name, _ = os.path.splitext(filename)
            
            if self.action_id in [1, 2]:
                from scripts.run_count_video import build_runtime_config
                from traffic_counter.detection.yolo_detector import YOLODetector
                from traffic_counter.tracking.tracker import SimpleIoUTracker
                from traffic_counter.counting.region_counter import RegionCounter
                from traffic_counter.counting.line_counter import LineCounter
                from traffic_counter.pipeline import TrafficPipeline

                self.log_msg.emit("Đang nạp cấu hình và khởi tạo mô hình AI...\n")
                config = build_runtime_config(None)
                
                # Sửa đổi key do YOLO update
                model_path = config["model"].get("path") or config["model"].get("yolo_path")
                
                detector = YOLODetector(
                    model_path=os.path.join(PROJECT_ROOT, model_path),
                    device=config["model"].get("device"),
                    allowed_classes=config.get("classes"),
                )
                
                tracker_cfg = config.get("tracker", {})
                tracker = SimpleIoUTracker(
                    iou_threshold=tracker_cfg.get("iou_threshold", 0.3),
                    max_lost=tracker_cfg.get("max_lost", 10),
                    min_hits=tracker_cfg.get("min_hits", 1),
                )
                
                count_cfg = config.get("counting", {})
                allowed_classes = set(config.get("classes", []))
                counting_type = count_cfg.get("counting_type", "line")
                
                if counting_type == "region":
                    polygon = count_cfg.get("polygon", [])
                    # Tự động tải polygon theo tên file
                    for key, mapped_poly in VIDEO_POLYGON_MAP.items():
                        if key in name:
                            polygon = mapped_poly
                            self.log_msg.emit(f"[*] Đã tự động tải vùng đếm riêng cho video: {key}\n")
                            break

                    counter = RegionCounter(
                        polygon=polygon,
                        allowed_classes=allowed_classes,
                        anchor_point=count_cfg.get("anchor_point", "bottom_center"),
                        count_once_per_track=count_cfg.get("count_once_per_track", True),
                        count_when=count_cfg.get("count_when", "enter"),
                    )
                else:
                    counter = LineCounter(
                        p1=tuple(count_cfg.get("line", [[100, 500], [1000, 500]])[0]),
                        p2=tuple(count_cfg.get("line", [[100, 500], [1000, 500]])[1]),
                        allowed_classes=allowed_classes,
                        count_once_per_track=count_cfg.get("count_once_per_track", True),
                    )

                self.pipeline = TrafficPipeline(
                    detector=detector,
                    tracker=tracker,
                    counter=counter,
                    predict_config=config.get("predict", {})
                )

                if self.action_id == 1:
                    out_file = os.path.join(self.output_folder, f"counted_{name}.mp4")
                    self.log_msg.emit(f"Bắt đầu đếm video: {self.input_file}\n")
                    self.pipeline.process_video(
                        self.input_file, 
                        out_file, 
                        display=False, 
                        frame_skip=config.get("video", {}).get("frame_skip", 0),
                        frame_callback=self.emit_frame
                    )
                elif self.action_id == 2:
                    out_file = os.path.join(self.output_folder, f"detected_{name}.jpg")
                    self.log_msg.emit(f"Bắt đầu nhận diện ảnh: {self.input_file}\n")
                    self.pipeline.process_image(self.input_file, out_file, frame_callback=self.emit_frame)
                    
            elif self.action_id == 3:
                self.log_msg.emit(f"Đang trích xuất frame từ video: {self.input_file}\n")
                from scripts.extract_frame import extract_frame
                out_file = os.path.join(self.output_folder, f"frame_{name}.jpg")
                extract_frame(self.input_file, out_file, frame_index=0)
                frame = cv2.imread(out_file)
                if frame is not None:
                    self.emit_frame(frame)
            
            self.process_finished.emit(0)
        except Exception as e:
            self.log_msg.emit(f"Lỗi: {e}\n")
            import traceback
            self.log_msg.emit(traceback.format_exc())
            self.process_finished.emit(1)

    def emit_frame(self, frame):
        self.frame_ready.emit(frame)
        
    def stop(self):
        if self.pipeline:
            self.pipeline.should_stop = True

class TrafficCounterUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Vehicle Counting System - Control Panel")
        self.resize(1100, 750)
        self.worker = None
        self._create_widgets()

    def _create_widgets(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Sử dụng QSplitter để chia đôi màn hình: Trái (Controls) - Phải (Video)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- PANEL ĐIỀU KHIỂN (BÊN TRÁI) ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 10, 0)

        font_bold = QFont()
        font_bold.setBold(True)

        # Input Frame
        input_group = QGroupBox("Cấu hình Đầu vào/Đầu ra")
        input_group.setFont(font_bold)
        input_layout = QVBoxLayout()
        
        lbl1 = QLabel("Input File (Video/Image):")
        lbl1.setFont(QFont())
        input_layout.addWidget(lbl1)
        
        row1 = QHBoxLayout()
        self.input_entry = QLineEdit()
        self.input_entry.setPlaceholderText("Chọn file video hoặc ảnh...")
        row1.addWidget(self.input_entry)
        btn_browse_in = QPushButton("Browse...")
        btn_browse_in.clicked.connect(self._browse_input)
        row1.addWidget(btn_browse_in)
        input_layout.addLayout(row1)

        lbl2 = QLabel("Output Folder:")
        lbl2.setFont(QFont())
        input_layout.addWidget(lbl2)
        
        row2 = QHBoxLayout()
        self.output_entry = QLineEdit()
        self.output_entry.setText(os.path.join(PROJECT_ROOT, "outputs"))
        row2.addWidget(self.output_entry)
        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.clicked.connect(self._browse_output)
        row2.addWidget(btn_browse_out)
        input_layout.addLayout(row2)

        input_group.setLayout(input_layout)
        control_layout.addWidget(input_group)

        # Action Frame
        action_group = QGroupBox("Chọn Chức năng")
        action_group.setFont(font_bold)
        action_layout = QVBoxLayout()
        
        self.action_btn_group = QButtonGroup(self)
        
        self.radio_count = QRadioButton("Count Video (Nhận diện & Đếm)")
        self.radio_count.setFont(QFont())
        self.radio_count.setChecked(True)
        self.action_btn_group.addButton(self.radio_count, 1)
        action_layout.addWidget(self.radio_count)

        self.radio_detect = QRadioButton("Detect Image (Chỉ nhận diện ảnh)")
        self.radio_detect.setFont(QFont())
        self.action_btn_group.addButton(self.radio_detect, 2)
        action_layout.addWidget(self.radio_detect)

        self.radio_extract = QRadioButton("Extract Frame (Trích xuất frame đầu)")
        self.radio_extract.setFont(QFont())
        self.action_btn_group.addButton(self.radio_extract, 3)
        action_layout.addWidget(self.radio_extract)

        action_group.setLayout(action_layout)
        control_layout.addWidget(action_group)

        # Run Button Frame
        run_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶ BẮT ĐẦU")
        self.run_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self._start_task)
        run_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("■ DỪNG")
        self.stop_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 10px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_task)
        run_layout.addWidget(self.stop_btn)

        control_layout.addLayout(run_layout)

        # Log Frame
        log_group = QGroupBox("Console Logs")
        log_group.setFont(font_bold)
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setFont(QFont())
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, Monaco, monospace; font-size: 12px;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        control_layout.addWidget(log_group)
        
        splitter.addWidget(control_panel)

        # --- MÀN HÌNH VIDEO (BÊN PHẢI) ---
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(10, 0, 0, 0)
        
        self.video_label = QLabel("Video Display Area")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 20px;")
        # Fix the minimum size so the layout doesn't collapse
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        
        splitter.addWidget(video_panel)
        
        # Thiết lập tỉ lệ kích thước 1:2 (Panel:Video)
        splitter.setSizes([350, 750])

    def _browse_input(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "Media Files (*.mp4 *.avi *.mov *.jpg *.png);;All Files (*.*)")
        if filepath:
            self.input_entry.setText(filepath)

    def _browse_output(self):
        dirpath = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if dirpath:
            self.output_entry.setText(dirpath)

    def _log(self, message):
        self.log_text.insertPlainText(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_logs(self):
        self.log_text.clear()

    def update_video_frame(self, frame_np):
        # Chuyển đổi BGR sang RGB cho PyQt
        rgb_image = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Convert thành QImage
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Convert sang QPixmap và scale cho vừa với video_label
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.video_label.width(), 
            self.video_label.height(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def _start_task(self):
        input_file = self.input_entry.text().strip()
        output_folder = self.output_entry.text().strip()

        if not input_file or not os.path.isfile(input_file):
            QMessageBox.critical(self, "Lỗi", "Vui lòng chọn một file đầu vào hợp lệ!")
            return

        os.makedirs(output_folder, exist_ok=True)
        
        self._clear_logs()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        action_id = self.action_btn_group.checkedId()
        
        self.worker = VideoWorker(action_id, input_file, output_folder)
        self.worker.log_msg.connect(self._log)
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.process_finished.connect(self._on_process_finish)
        
        self.worker.start()

    def _stop_task(self):
        if self.worker and self.worker.isRunning():
            self._log("\nĐang gửi tín hiệu dừng... Vui lòng đợi trong giây lát.\n")
            self.worker.stop()

    def _on_process_finish(self, exit_code):
        self._log(f"\nTiến trình hoàn tất.\n")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TrafficCounterUI()
    window.show()
    sys.exit(app.exec())
