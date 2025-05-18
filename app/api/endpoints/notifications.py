from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any, List
import uuid

from app.db.base import get_db
from app.api.deps import get_current_active_user, get_current_active_admin
from app.models import User
from app.schemas import Notification as NotificationSchema, NotificationCreate, NotificationSettings
from app.services import notification as notification_service 

router = APIRouter()


@router.get("", response_model=List[NotificationSchema])
def read_notifications(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    return notification_service.get_user_notifications(db, current_user.user_id, skip, limit)


@router.post("", response_model=NotificationSchema)
def create_notification(
    *,
    db: Session = Depends(get_db),
    notification_in: NotificationCreate,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    notification = notification_service.create_notification(db, notification_in, current_user)
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Người dùng không tồn tại",
        )
    return notification


@router.get("/settings", response_model=NotificationSettings)
def get_notification_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    return notification_service.get_notification_settings(db, current_user.user_id)


@router.post("/settings", response_model=NotificationSettings)
def update_notification_settings(
    *,
    db: Session = Depends(get_db),
    settings: NotificationSettings,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    return notification_service.update_notification_settings(db, current_user.user_id, settings)
