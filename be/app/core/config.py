import os
from dotenv import load_dotenv
load_dotenv()
import tempfile
from typing import Any, Dict, Optional

from pydantic import PostgresDsn, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    DATABASE_URL: PostgresDsn
    
    MODEL_PATH: str
    
    # Cấu hình Cloudinary
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str
    CLOUDINARY_FOLDER: str = "fire_detection"
    DELETE_LOCAL_FILES_AFTER_UPLOAD: bool = True
    
    # Cấu hình Email (SMTP)
    EMAIL_SENDER: str = "firerdetection@example.com"
    EMAIL_HOST: str = "smtp.gmail.com"
    EMAIL_PORT: int = 587
    EMAIL_USE_TLS: bool = True
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    
    # Thư mục tạm của hệ thống
    TEMP_DIR: str = tempfile.gettempdir()
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings() 