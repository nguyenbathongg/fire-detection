import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# Schemas cho Notification
class NotificationBase(BaseModel):
    title: str
    message: str
    enable_email_notification: bool = False
    enable_website_notification: bool = False


class NotificationCreate(NotificationBase):
    user_id: uuid.UUID
    video_id: Optional[uuid.UUID] = None


class NotificationUpdate(BaseModel):
    title: Optional[str] = None
    message: Optional[str] = None
    enable_email_notification: Optional[bool] = None
    enable_website_notification: Optional[bool] = None


class NotificationInDBBase(NotificationBase):
    notification_id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    video_id: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Notification(NotificationInDBBase):
    pass


# Settings cho notifications
class NotificationSettings(BaseModel):
    enable_email_notification: bool
    enable_website_notification: bool 