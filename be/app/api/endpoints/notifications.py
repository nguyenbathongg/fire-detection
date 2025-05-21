import uuid
from typing import Any, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.api.deps import get_current_active_user, get_current_active_admin
from app.models import User
from app.schemas import Notification as NotificationSchema, NotificationCreate, NotificationUpdate, NotificationSettings
from app.controllers.notification_controller import NotificationController

router = APIRouter()


@router.get("", response_model=List[NotificationSchema])
async def read_notifications(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Đọc danh sách thông báo của người dùng
    """
    return NotificationController.get_user_notifications(
        db=db, 
        user_id=current_user.user_id, 
        skip=skip, 
        limit=limit
    )


@router.post("", response_model=NotificationSchema)
async def create_notification(
    *,
    db: Session = Depends(get_db),
    notification_in: NotificationCreate,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Tạo thông báo mới (chỉ admin)
    """
    return NotificationController.create_notification(
        db=db, 
        notification_in=notification_in,
        admin_id=current_user.user_id
    )


@router.get("/settings", response_model=NotificationSettings)
async def get_notification_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Lấy cài đặt thông báo của người dùng
    """
    return NotificationController.get_notification_settings(
        db=db, 
        user_id=current_user.user_id
    )


@router.post("/settings", response_model=NotificationSettings)
async def update_notification_settings(
    *,
    db: Session = Depends(get_db),
    settings: NotificationSettings,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Cập nhật cài đặt thông báo
    """
    return NotificationController.update_notification_settings(
        db=db, 
        user_id=current_user.user_id, 
        settings=settings
    )