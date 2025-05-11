import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, confloat


# Schemas cho FireDetection
class FireDetectionBase(BaseModel):
    fire_start_time: Optional[float] = None
    fire_end_time: Optional[float] = None
    confidence: confloat(ge=0.0, le=1.0)


class FireDetectionCreate(FireDetectionBase):
    video_id: uuid.UUID


class FireDetectionUpdate(BaseModel):
    fire_start_time: Optional[float] = None
    fire_end_time: Optional[float] = None
    confidence: Optional[confloat(ge=0.0, le=1.0)] = None
    max_fire_frame: Optional[int] = None
    max_fire_frame_image_path: Optional[str] = None


class FireDetectionInDBBase(FireDetectionBase):
    detection_id: uuid.UUID
    video_id: uuid.UUID
    max_fire_frame: Optional[int] = None
    max_fire_frame_image_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class FireDetection(FireDetectionInDBBase):
    pass 