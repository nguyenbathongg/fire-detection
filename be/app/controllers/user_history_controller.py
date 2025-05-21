import uuid
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import desc
from fastapi import HTTPException, status

from app.models import User, UserHistory


class UserHistoryController:
    """
    Controller xử lý logic nghiệp vụ liên quan đến lịch sử người dùng
    """

    @staticmethod
    def get_user_history(db: Session, user_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[UserHistory]:
        """
        Lấy lịch sử hoạt động của người dùng
        """
        return (db.query(UserHistory)
                .filter(UserHistory.user_id == user_id)
                .order_by(desc(UserHistory.created_at))
                .offset(skip)
                .limit(limit)
                .all())

    @staticmethod
    def get_all_histories(db: Session, skip: int = 0, limit: int = 100) -> List[UserHistory]:
        """
        Lấy tất cả lịch sử hoạt động
        """
        return (db.query(UserHistory)
                .order_by(desc(UserHistory.created_at))
                .offset(skip)
                .limit(limit)
                .all())

    @staticmethod
    def add_history(db: Session, user_id: uuid.UUID, action_type: str, description: str, 
                    notification_id: Optional[uuid.UUID] = None) -> UserHistory:
        """
        Thêm lịch sử hoạt động
        """
        history = UserHistory(
            history_id=uuid.uuid4(),
            user_id=user_id,
            action_type=action_type,
            notification_id=notification_id,
            description=description
        )
        db.add(history)
        db.commit()
        db.refresh(history)
        return history

    @staticmethod
    def add_view_history(db: Session, user_id: uuid.UUID) -> UserHistory:
        """
        Thêm lịch sử xem lịch sử
        """
        return UserHistoryController.add_history(
            db=db,
            user_id=user_id,
            action_type="view_history",
            description="Xem lịch sử hoạt động cá nhân"
        )

    @staticmethod
    def add_view_user_history(db: Session, admin_id: uuid.UUID, target_user_id: uuid.UUID) -> UserHistory:
        """
        Thêm lịch sử xem lịch sử người dùng khác
        """
        # Kiểm tra người dùng cần xem lịch sử tồn tại
        user = db.query(User).filter(User.user_id == target_user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Không tìm thấy người dùng",
            )

        return UserHistoryController.add_history(
            db=db,
            user_id=admin_id,
            action_type="view_user_history",
            description=f"Xem lịch sử hoạt động của người dùng {user.username}"
        ) 