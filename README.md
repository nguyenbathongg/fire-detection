# Hệ thống Phát hiện Đám Cháy từ Video

Hệ thống phát hiện đám cháy từ video sử dụng YOLOv11 (UltraLytics) và FastAPI cho back-end, React cho front-end. Ứng dụng này tự động phát hiện đám cháy từ video được tải lên hoặc video YouTube, với kết quả xử lý lưu trữ trên Cloudinary.

## Tính năng

- Đăng ký và xác thực người dùng
- Tải lên video từ máy tính hoặc từ YouTube
- Phát hiện đám cháy theo thời gian thực
- Xử lý video và đánh dấu vùng đám cháy
- Streaming kết quả xử lý theo thời gian thực 
- Lưu trữ video trên Cloudinary (không lưu cục bộ)
- Giao diện người dùng thân thiện với React, Material-UI và Ant Design
- API RESTful đầy đủ

## Yêu cầu hệ thống

- Python 3.8 hoặc cao hơn
- Node.js 14+ và npm
- PostgreSQL
- Tài khoản Cloudinary
- Torch/PyTorch (hỗ trợ CUDA nếu có GPU)

## Cài đặt Back-end

1. Clone repository:
   ```
   git clone https://github.com/nguyenbathongg/fire-detection.git
   cd fire-detection
   ```

2. Tạo môi trường ảo và cài đặt dependencies:
   ```
   python -m venv .venv
   # Kích hoạt môi trường ảo
   .venv\Scripts\activate  # Windows
   # hoặc
   source .venv/bin/activate  # Linux/Mac
   
   # Cài đặt các thư viện
   cd be
   pip install -r requirements.txt
   
   # Cài đặt yt-dlp (cần thiết cho tải video YouTube)
   python install_ytdlp.py
   ```

3. Tạo file `.env` từ file mẫu `.env.example`:
   ```
   copy .env.example .env  # Windows
   # hoặc
   cp .env.example .env  # Linux/Mac
   ```

4. Cập nhật thông tin trong file `.env`:
   - Thông tin database PostgreSQL
   - Thông tin Cloudinary API
   - Secret key cho JWT
   - Cấu hình cho SMTP thông báo

5. Tải model YOLOv11 và đặt vào thư mục `model/`: 
   - Model YOLO đã train phát hiện đám cháy (tên file `bestyolov11-27k.pt`)
   - Hoặc cập nhật tham số MODEL_PATH trong file .env

6. Chạy migration để tạo cấu trúc database:
   ```
   alembic upgrade head
   ```

7. Tạo tài khoản admin:
   ```
   # Chạy script tạo tài khoản admin
   python create_admin.py
   ```
   Nhập thông tin theo yêu cầu (email, mật khẩu) để tạo tài khoản admin.

8. Khởi động back-end:
   ```
   uvicorn app.main:app --reload
   ```
   Mặc định back-end sẽ chạy tại: http://localhost:8000

## Cài đặt Front-end

1. Di chuyển vào thư mục front-end:
   ```
   cd fe
   ```

2. Cài đặt các dependencies:
   ```
   npm install
   ```

3. Khởi động front-end:
   ```
   npm start
   ```
   Mặc định front-end sẽ chạy tại: http://localhost:3000

## Sử dụng

1. Mở trình duyệt và truy cập vào http://localhost:3000
2. Đăng ký tài khoản mới hoặc đăng nhập với tài khoản có sẵn
3. Sử dụng các tính năng từ menu điều hướng:
   - Tải video lên để phân tích
   - Xem kết quả phát hiện đám cháy
   - Quản lý tài khoản cá nhân

## Cấu trúc dự án

```
fire-detection/
├─ be/                     # Back-end
│   ├─ app/                # Mã nguồn FastAPI
│   │   ├─ api/            # API endpoints 
│   │   ├─ core/           # Cấu hình
│   │   ├─ db/             # Kết nối database
│   │   ├─ models/         # Mô hình dữ liệu
│   │   ├─ schemas/        # Pydantic schemas
│   │   ├─ services/       # Business logic
│   │   └─ utils/          # Tiện ích
│   ├─ migrations/         # Alembic migrations
│   ├─ model/              # Model YOLOv11
│   ├─ .env                # Biến môi trường
│   └─ requirements.txt    # Packages cho backend
├─ fe/                     # Front-end
│   ├─ public/             # Tài nguyên tĩnh
│   ├─ src/                # Mã nguồn React
│   └─ package.json        # Packages cho frontend
└─ README.md               # Tài liệu dự án
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/login-email`: Đăng nhập bằng email
- `POST /api/v1/auth/register`: Đăng ký tài khoản mới
- `POST /api/v1/auth/change-password`: Thay đổi mật khẩu
- `POST /api/v1/auth/verify-token`: Xác thực token

### Users
- `GET /api/v1/users/me`: Lấy thông tin người dùng hiện tại
- `PUT /api/v1/users/me`: Cập nhật thông tin người dùng hiện tại
- `GET /api/v1/users`: Lấy danh sách người dùng
- `POST /api/v1/users`: Tạo người dùng mới
- `GET /api/v1/users/{user_id}`: Lấy thông tin người dùng theo ID
- `DELETE /api/v1/users/{user_id}`: Xóa người dùng

### Videos
- `GET /api/v1/videos`: Lấy danh sách video người dùng tải lên
- `GET /api/v1/videos/all`: Lấy danh sách tất cả video 
- `GET /api/v1/videos/{video_id}`: Lấy thông tin chi tiết video
- `DELETE /api/v1/videos/{video_id}`: Xóa video

### Notifications
- `GET /api/v1/notifications`: Lấy danh sách thông báo
- `POST /api/v1/notifications`: Tạo thông báo mới
- `GET /api/v1/notifications/settings`: Lấy cài đặt thông báo
- `POST /api/v1/notifications/settings`: Cập nhật cài đặt thông báo

### User History
- `GET /api/v1/history/me`: Lấy lịch sử hoạt động của người dùng hiện tại
- `GET /api/v1/history`: Lấy lịch sử hoạt động của tất cả người dùng
- `GET /api/v1/history/{user_id}`: Lấy lịch sử hoạt động của người dùng cụ thể

### WebSocket
- `WS /ws/videos/{video_id}`: Stream kết quả xử lý video theo thời gian thực

## Lưu trữ media

- Tất cả video và ảnh được lưu trữ trực tiếp trên Cloudinary
- Không có thư mục cục bộ cho việc lưu trữ video, chỉ sử dụng thư mục tạm trong quá trình xử lý
- Hệ thống tự động xóa tất cả các file tạm sau khi xử lý xong 