import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base
from app.models.enums import VideoTypeEnum, StatusEnum


class Video(Base):
    __tablename__ = "videos"

    video_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    video_type = Column(String(10), nullable=False)
    youtube_url = Column(String(255), nullable=True)
    original_video_url = Column(String(255), nullable=False)
    processed_video_url = Column(String(255), nullable=True)
    status = Column(String(10), nullable=False, default="pending")
    fire_detected = Column(Boolean, nullable=False, default=False)
    file_name = Column(String(255), nullable=True)  # Tên file video gốc hoặc tiêu đề YouTube
    cloudinary_public_id = Column(String(255), nullable=True)  # ID công khai của video gốc trên Cloudinary
    cloudinary_processed_id = Column(String(255), nullable=True)  # ID công khai của video đã xử lý trên Cloudinary
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fire_detections = relationship("FireDetection", back_populates="video", cascade="all, delete-orphan")
    user = relationship("User") 