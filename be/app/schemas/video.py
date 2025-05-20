import uuid
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, HttpUrl

from app.models.enums import VideoTypeEnum, StatusEnum
from app.schemas.fire_detection import FireDetection


# Schemas cho Video
class VideoBase(BaseModel):
    video_type: Optional[VideoTypeEnum] = None
    youtube_url: Optional[str] = None
    file_name: Optional[str] = None


class VideoCreate(VideoBase):
    pass


class VideoUpdate(BaseModel):
    video_type: Optional[VideoTypeEnum] = None
    youtube_url: Optional[str] = None
    file_name: Optional[str] = None
    status: Optional[StatusEnum] = None
    fire_detected: Optional[bool] = None


class VideoInDBBase(VideoBase):
    video_id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    original_video_url: str
    processed_video_url: Optional[str] = None
    status: StatusEnum
    fire_detected: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Video(VideoInDBBase):
    pass


class VideoWithDetections(Video):
    fire_detections: List[FireDetection] = []


# Schema cho upload video
class VideoUpload(BaseModel):
    youtube_url: Optional[str] = None 