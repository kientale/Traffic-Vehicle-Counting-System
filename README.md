# NCKH - Traffic Vehicle Counting System

Dự án này phục vụ đề tài nghiên cứu khoa học:
**"Nghiên cứu khả năng ứng dụng Ultralytics YOLO trong giám sát và đếm lưu lượng phương tiện giao thông theo thời gian thực từ camera cố định"**

---

## 🌟 Giới thiệu

Hệ thống cung cấp một giải pháp toàn diện để phát hiện, bám vết (tracking) và đếm số lượng phương tiện giao thông chạy qua các điểm cấu hình trước trong video hoặc luồng camera trực tiếp. 

Dự án được tích hợp sẵn một **giao diện người dùng đồ họa (GUI) hiện đại** xây dựng bằng thư viện `PyQt6`, hỗ trợ trình chiếu trực tiếp tiến trình nhận diện lên màn hình và cho phép tùy biến cấu hình dễ dàng mà không cần thao tác với dòng lệnh.

### ⚙️ Pipeline Xử lý Lõi:
```text
Video/Ảnh đầu vào
 ├── 1. Phát hiện phương tiện (Ultralytics YOLO)
 ├── 2. Bám vết - Tracking (Simple IoU Tracker)
 ├── 3. Đếm phương tiện (Line Counting / Region Counting)
 └── 4. Xuất kết quả (Video đính kèm Bounding box, Console Log, Data đếm)
```

---

## 🚀 Các Tính Năng Nổi Bật

- **AI Đỉnh cao**: Ứng dụng mô hình YOLO mới nhất (hỗ trợ cấu hình `yolo11n.pt` và các phiên bản khác).
- **Giao diện Trực quan (PyQt6)**: Màn hình trình chiếu video trực tiếp (Embedded Video) được tối ưu hóa bằng đa luồng (Multi-threading) không gây đơ/lag giao diện.
- **Tự động nhận diện Tọa độ (Auto-Polygon Map)**: Hệ thống có khả năng tự động nội suy tọa độ vùng đếm (Region of Interest) dựa trên tên file đầu vào (VD: `easy_01`, `medium_02`...).
- **Tùy chỉnh Dễ dàng**: Người dùng có thể chỉnh sửa mô hình, FPS, Threshold thông qua các tệp YAML (`configs/detect.yaml`, `configs/counting.yaml`).
- **Log Console Tích hợp**: Theo dõi trực tiếp tiến trình ngay trong giao diện ứng dụng.

---

## 🛠 Cài đặt

1. **Yêu cầu hệ thống**: Python 3.9+ 
2. **Cài đặt thư viện**:
   Sử dụng môi trường ảo (virtual environment) và chạy lệnh sau để cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
   *(Các thư viện chính bao gồm: `ultralytics`, `opencv-python`, `numpy`, `PyQt6`)*

---

## 💻 Hướng dẫn Sử dụng

Để khởi động bảng điều khiển (Control Panel) của hệ thống đếm xe, bạn chỉ cần gõ lệnh sau ở thư mục gốc:

```bash
python3 ui/app.py
```

### Các bước thao tác trên giao diện:
1. **Input Selection**: Nhấn `Browse...` để trỏ tới file video (hoặc ảnh) cần đếm.
2. **Output Folder**: Chọn thư mục lưu trữ kết quả đầu ra (mặc định là `outputs/`).
3. **Select Action**:
   - `Count Video`: Kích hoạt toàn bộ pipeline để đếm xe trong Video.
   - `Detect Image`: Chỉ sử dụng thuật toán nhận diện phương tiện trên 1 bức ảnh.
   - `Extract Frame`: Trích xuất nhanh khung hình đầu tiên của video.
4. **Bắt đầu**: Nhấn nút `▶ BẮT ĐẦU`. Khung đếm và số liệu sẽ hiển thị to, rõ ràng và mượt mà ngay phần màn hình màu đen bên phải của app.

---

## 📁 Cấu trúc Dự án

```text
Traffic-Vehicle-Counting-System/
├── configs/               # Chứa các file cấu hình YAML (detect, tracking, counting)
├── data/raws/             # Thư mục mẫu chứa video/ảnh gốc
├── models/yolo/           # Lưu trữ các tệp trọng số mô hình (.pt)
├── outputs/               # Nơi lưu trữ video/ảnh sau khi được hệ thống xử lý
├── scripts/               # Các script chạy độc lập (Dùng cho CLI)
├── src/
│   └── traffic_counter/   # Mã nguồn lõi (Detector, Tracker, Counter, Pipeline)
├── ui/
│   └── app.py             # File khởi chạy giao diện GUI PyQt6 chính
└── README.md              # Tài liệu hướng dẫn
```

---
*Dự án NCKH.*