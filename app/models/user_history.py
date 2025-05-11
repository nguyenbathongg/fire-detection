import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class UserHistory(Base):
    __tablename__ = "user_history"

    history_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="SET NULL"))
    action_type = Column(String(50), nullable=False)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id", ondelete="SET NULL"))
    notification_id = Column(UUID(as_uuid=True), ForeignKey("notifications.notification_id", ondelete="SET NULL"))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    video = relationship("Video")
    notification = relationship("Notification") 