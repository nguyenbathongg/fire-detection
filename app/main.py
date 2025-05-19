import logging
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.api import api_router
from app.core.config import settings
from app.utils.cloudinary_service import init_cloudinary

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hệ thống phát hiện đám cháy",
    description="API cho hệ thống phát hiện đám cháy từ video",
    version="1.0.0",
)

# Middleware CORS - Cấu hình cho phép frontend React và WebSocket truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # Thêm frontend React origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "*"],
    expose_headers=["Content-Length", "*"],
    max_age=600,  # 10 phút cache cho preflight requests
)

# Khởi tạo Cloudinary
init_cloudinary()

# Đăng ký router API
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    """
    Hiển thị trang chủ
    """
    return {"message": "Hệ thống phát hiện đám cháy API đang hoạt động"}

@app.get("/api/health-check")
def health_check():
    """
    Kiểm tra trạng thái API
    """
    return {"message": "Hệ thống phát hiện đám cháy API đang hoạt động"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Xử lý ngoại lệ toàn cục
    """
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Đã xảy ra lỗi hệ thống. Vui lòng thử lại sau."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 