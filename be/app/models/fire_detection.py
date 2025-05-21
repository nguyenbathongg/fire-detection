import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Float, Integer, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base
from app.utils.datetime_utils import utcnow_vn


class FireDetection(Base):
    __tablename__ = "fire_detections"

    detection_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id", ondelete="CASCADE"))
    fire_start_time = Column(Float)
    fire_end_time = Column(Float)
    confidence = Column(Float, nullable=False)
    max_fire_frame = Column(Integer)
    max_fire_frame_image_path = Column(String(255))
    created_at = Column(DateTime, default=utcnow_vn)
    
    # Constraint
    __table_args__ = (
        CheckConstraint('fire_start_time < fire_end_time OR fire_end_time IS NULL', name='check_time_order'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence_range'),
    )
    
    # Relationships
    video = relationship("Video", back_populates="fire_detections") 