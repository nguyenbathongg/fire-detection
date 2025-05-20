import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# Schemas cho UserHistory
class UserHistoryBase(BaseModel):
    action_type: str
    description: Optional[str] = None


class UserHistoryCreate(UserHistoryBase):
    user_id: uuid.UUID
    video_id: Optional[uuid.UUID] = None
    notification_id: Optional[uuid.UUID] = None


class UserHistoryInDBBase(UserHistoryBase):
    history_id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    video_id: Optional[uuid.UUID] = None
    notification_id: Optional[uuid.UUID] = None
    created_at: datetime

    class Config:
        from_attributes = True


class UserHistory(UserHistoryInDBBase):
    pass 