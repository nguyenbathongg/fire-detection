import uuid
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models import Notification, User, UserHistory
from app.schemas import NotificationCreate, NotificationSettings


def get_user_notifications(db: Session, user_id: uuid.UUID, skip: int = 0, limit: int = 100):
    return db.query(Notification)\
        .filter(Notification.user_id == user_id)\
        .order_by(desc(Notification.created_at))\
        .offset(skip).limit(limit).all()


def create_notification(db: Session, notification_in: NotificationCreate, admin_user: User):
    user = db.query(User).filter(User.user_id == notification_in.user_id).first()
    if not user:
        return None

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

    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=admin_user.user_id,
        action_type="create_notification",
        notification_id=notification.notification_id,
        description=f"Tạo thông báo mới cho người dùng {user.username}"
    )
    db.add(history)
    db.commit()
    db.refresh(notification)
    return notification


def get_notification_settings(db: Session, user_id: uuid.UUID):
    notification = db.query(Notification)\
        .filter(Notification.user_id == user_id)\
        .order_by(desc(Notification.created_at)).first()
    
    if notification:
        return NotificationSettings(
            enable_email_notification=notification.enable_email_notification,
            enable_website_notification=notification.enable_website_notification
        )
    return NotificationSettings(
        enable_email_notification=False,
        enable_website_notification=True
    )


def update_notification_settings(db: Session, user_id: uuid.UUID, settings: NotificationSettings):
    notifications = db.query(Notification).filter(Notification.user_id == user_id).all()

    if not notifications:
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

    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=user_id,
        action_type="update_notification_settings",
        description=f"Cập nhật cài đặt thông báo. Email: {settings.enable_email_notification}, Website: {settings.enable_website_notification}"
    )
    db.add(history)
    db.commit()

    return settings
