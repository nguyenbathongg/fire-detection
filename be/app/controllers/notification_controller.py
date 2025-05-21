import uuid
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import desc
from fastapi import HTTPException, status

from app.models import User, Notification, UserHistory
from app.schemas import NotificationCreate, NotificationSettings
from app.controllers.user_history_controller import UserHistoryController


class NotificationController:
    """
    Controller xử lý logic nghiệp vụ liên quan đến thông báo
    """

    @staticmethod
    def get_user_notifications(db: Session, user_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[Notification]:
        """
        Lấy danh sách thông báo của người dùng
        """
        notifications = (db.query(Notification)
                        .filter(Notification.user_id == user_id)
                        .order_by(desc(Notification.created_at))
                        .offset(skip)
                        .limit(limit)
                        .all())
        
        # Thêm lịch sử
        UserHistoryController.add_history(
            db=db,
            user_id=user_id,
            action_type="view_notifications",
            description="Xem danh sách thông báo"
        )
        
        return notifications

    @staticmethod
    def create_notification(db: Session, notification_in: NotificationCreate, admin_id: uuid.UUID) -> Notification:
        """
        Tạo thông báo mới
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
        UserHistoryController.add_history(
            db=db,
            user_id=admin_id,
            action_type="create_notification",
            notification_id=notification.notification_id,
            description=f"Tạo thông báo mới cho người dùng {user.username}"
        )
        
        db.commit()
        db.refresh(notification)
        
        return notification

    @staticmethod
    def get_notification_settings(db: Session, user_id: uuid.UUID) -> NotificationSettings:
        """
        Lấy cài đặt thông báo của người dùng
        """
        # Lấy thông báo gần nhất để xem cài đặt
        notification = (db.query(Notification)
                        .filter(Notification.user_id == user_id)
                        .order_by(desc(Notification.created_at))
                        .first())
        
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

    @staticmethod
    def update_notification_settings(db: Session, user_id: uuid.UUID, settings: NotificationSettings) -> NotificationSettings:
        """
        Cập nhật cài đặt thông báo
        """
        # Cập nhật tất cả thông báo của người dùng
        notifications = db.query(Notification).filter(Notification.user_id == user_id).all()
        
        if not notifications:
            # Nếu chưa có thì tạo bản ghi Notification mới để lưu setting
            notification = Notification(
                notification_id=uuid.uuid4(),
                user_id=user_id,
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
        UserHistoryController.add_history(
            db=db,
            user_id=user_id,
            action_type="update_notification_settings",
            description=f"Cập nhật cài đặt thông báo. Email: {settings.enable_email_notification}, Website: {settings.enable_website_notification}"
        )
        
        db.commit()
        
        return settings 