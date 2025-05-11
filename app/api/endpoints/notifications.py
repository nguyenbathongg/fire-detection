import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.base import get_db
from app.api.deps import get_current_active_user, get_current_active_admin
from app.models import User, Notification, UserHistory
from app.schemas import Notification as NotificationSchema, NotificationCreate, NotificationUpdate, NotificationSettings

router = APIRouter()


@router.get("", response_model=List[NotificationSchema])
def read_notifications(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Đọc danh sách thông báo của người dùng
    """
    notifications = db.query(Notification).filter(Notification.user_id == current_user.user_id) \
        .order_by(desc(Notification.created_at)) \
        .offset(skip).limit(limit).all()
    
    # Thêm lịch sử
    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=current_user.user_id,
        action_type="view_notifications",
        description="Xem danh sách thông báo"
    )
    db.add(history)
    db.commit()
    
    return notifications


@router.post("", response_model=NotificationSchema)
def create_notification(
    *,
    db: Session = Depends(get_db),
    notification_in: NotificationCreate,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Tạo thông báo mới (chỉ admin)
    """
    # Kiểm tra user_id tồn tại
    user = db.query(User).filter(User.user_id == notification_in.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Người dùng không tồn tại",
        )
    
    notification = Notification(
        notification_id=uuid.uuid4(),
        user_id=notification_in.user_id,
        video_id=notification_in.video_id,
        title=notification_in.title,
        message=notification_in.message,
        enable_email_notification=notification_in.enable_email_notification,
        enable_website_notification=notification_in.enable_website_notification,
    )
    db.add(notification)
    
    # Thêm lịch sử
    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=current_user.user_id,
        action_type="create_notification",
        notification_id=notification.notification_id,
        description=f"Tạo thông báo mới cho người dùng {user.username}"
    )
    db.add(history)
    db.commit()
    db.refresh(notification)
    
    return notification


@router.get("/settings", response_model=NotificationSettings)
def get_notification_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Lấy cài đặt thông báo của người dùng
    """
    # Lấy thông báo gần nhất để xem cài đặt
    notification = db.query(Notification).filter(Notification.user_id == current_user.user_id) \
        .order_by(desc(Notification.created_at)).first()
    
    if notification:
        settings = NotificationSettings(
            enable_email_notification=notification.enable_email_notification,
            enable_website_notification=notification.enable_website_notification
        )
    else:
        # Nếu chưa có thông báo nào, trả về mặc định
        settings = NotificationSettings(
            enable_email_notification=False,
            enable_website_notification=True
        )
    
    return settings


@router.post("/settings", response_model=NotificationSettings)
def update_notification_settings(
    *,
    db: Session = Depends(get_db),
    settings: NotificationSettings,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Cập nhật cài đặt thông báo
    """
    # Cập nhật tất cả thông báo của người dùng
    notifications = db.query(Notification).filter(Notification.user_id == current_user.user_id).all()
    
    if not notifications:
        # Nếu chưa có thì tạo bản ghi Notification mới để lưu setting
        notification = Notification(
            notification_id=uuid.uuid4(),
            user_id=current_user.user_id,
            title="Cài đặt thông báo",
            message="Bản ghi lưu cài đặt thông báo",
            enable_email_notification=settings.enable_email_notification,
            enable_website_notification=settings.enable_website_notification
        )
        db.add(notification)
    else:
        for notification in notifications:
            notification.enable_email_notification = settings.enable_email_notification
            notification.enable_website_notification = settings.enable_website_notification
    
    # Thêm lịch sử
    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=current_user.user_id,
        action_type="update_notification_settings",
        description=f"Cập nhật cài đặt thông báo. Email: {settings.enable_email_notification}, Website: {settings.enable_website_notification}"
    )
    db.add(history)
    db.commit()
    
    return settings