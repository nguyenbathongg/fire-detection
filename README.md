# Hệ thống Phát hiện Đám Cháy từ Video

Hệ thống phát hiện đám cháy từ video sử dụng YOLOv11 và FastAPI. Ứng dụng này tự động phát hiện đám cháy từ video được tải lên hoặc video YouTube, với kết quả xử lý lưu trữ trên Cloudinary.

## Tính năng

- Đăng ký và xác thực người dùng
- Tải lên video từ máy tính hoặc từ YouTube
- Phát hiện đám cháy theo thời gian thực
- Xử lý video và đánh dấu vùng đám cháy
- Streaming kết quả xử lý theo thời gian thực 
- Lưu trữ video trên Cloudinary (không lưu cục bộ)
- API RESTful đầy đủ

## Yêu cầu hệ thống

- Python 3.8 hoặc cao hơn
- PostgreSQL
- Tài khoản Cloudinary
- Torch/PyTorch (hỗ trợ CUDA nếu có GPU)

## Cài đặt

1. Clone repository:
   ```
   git clone https://github.com/nguyenbathongg/fire-detection.git
   cd fire-detection
   ```

2. Tạo môi trường ảo và cài đặt dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # hoặc
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. Tạo file `.env` từ file mẫu `.env.example`:
   ```
   cp .env.example .env  # Linux/Mac
   # hoặc
   copy .env.example .env  # Windows
   ```

4. Cập nhật thông tin trong file `.env`:
   - Thông tin database PostgreSQL
   - Thông tin Cloudinary API
   - Secret key cho JWT

5. Tải model YOLOv11 và đặt vào thư mục `model/`:
   - Model YOLO đã train phát hiện đám cháy (bestyolov11-25k.pt)

6. Chạy migration để tạo cấu trúc database:
   ```
   python -m alembic upgrade head
   ```

## Cấu trúc thư mục

```
fire-detection/
├── app/                    # Mã nguồn chính
│   ├── api/                # API endpoints 
│   ├── core/               # Cấu hình
│   ├── db/                 # Kết nối database
│   ├── models/             # Mô hình dữ liệu
│   ├── schemas/            # Pydantic schemas
│   ├── services/           # Business logic
│   └── utils/              # Tiện ích
├── migrations/             # Alembic migrations
├── model/                  # Model YOLOv11
├── .env                    # Biến môi trường
├── .env.example           # Mẫu cấu hình biến môi trường
├── .gitignore             # Danh sách file bỏ qua khi đẩy lên Git
└── requirements.txt        # Packages
```

## Khởi chạy ứng dụng

```bash
# Khởi động ứng dụng
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mở trình duyệt và truy cập: http://localhost:8000

## Sử dụng API

### Streaming Video Processing

1. Tải lên video qua API POST `/api/v1/videos`
2. Lấy ID video từ phản hồi
3. Mở trang `http://localhost:8000
4. 
5. Xem quá trình xử lý và kết quả phát hiện đám cháy theo thời gian thực

### API Endpoints

- `POST /api/v1/videos`: Tải lên video mới
- `GET /api/v1/videos`: Lấy danh sách video của người dùng
- `GET /api/v1/videos/{video_id}`: Xem chi tiết video
- `DELETE /api/v1/videos/{video_id}`: Xóa video
- `WebSocket /api/v1/videos/ws/process/{video_id}`: Streaming xử lý video

## Lưu trữ media

- Tất cả video và ảnh được lưu trữ trực tiếp trên Cloudinary
- Không có thư mục cục bộ cho việc lưu trữ video, chỉ sử dụng thư mục tạm trong quá trình xử lý
- Hệ thống tự động xóa tất cả các file tạm sau khi xử lý xong 