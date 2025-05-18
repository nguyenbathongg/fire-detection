from typing import Any, List
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.api.deps import get_current_active_user, get_current_active_admin
from app.models import User
from app.schemas import UserHistory as UserHistorySchema
from app.services import user_history as user_history_service  # ✅ Gọi service

router = APIRouter()


@router.get("/me", response_model=List[UserHistorySchema])
def read_user_history_me(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_active_user),
) -> Any:
    """
    Lấy lịch sử hoạt động của người dùng hiện tại
    """
    histories = user_history_service.get_user_history(db, current_user.user_id, skip, limit)
    user_history_service.add_view_history(
        db,
        user_id=current_user.user_id,
        action_type="view_history",
        description="Xem lịch sử hoạt động cá nhân"
    )
    return histories


@router.get("", response_model=List[UserHistorySchema])
def read_all_user_history(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_active_admin),
) -> Any:
    """
    Lấy tất cả lịch sử hoạt động (chỉ admin)
    """
    return user_history_service.get_user_history(db, current_user.user_id, skip, limit)


@router.get("/{user_id}", response_model=List[UserHistorySchema])
def read_user_history(
    *,
    db: Session = Depends(get_db),
    user_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_active_admin),
) -> Any:
    """
    Lấy lịch sử hoạt động của một người dùng cụ thể (chỉ admin)
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng",
        )

    histories = user_history_service.get_user_history(db, user_id, skip, limit)
    user_history_service.add_view_history(
        db,
        user_id=current_user.user_id,
        action_type="view_user_history",
        description=f"Xem lịch sử hoạt động của người dùng {user.username}"
    )
    return histories
