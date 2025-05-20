from typing import Any, List
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.base import get_db
from app.api.deps import get_current_active_user, get_current_active_admin
from app.models import User, UserHistory
from app.schemas import UserHistory as UserHistorySchema

router = APIRouter()


@router.get("/me", response_model=List[UserHistorySchema])
async def read_user_history_me(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Lấy lịch sử hoạt động của người dùng hiện tại
    """
    histories = db.query(UserHistory).filter(UserHistory.user_id == current_user.user_id) \
        .order_by(desc(UserHistory.created_at)) \
        .offset(skip).limit(limit).all()
    
    # Thêm lịch sử cho việc xem lịch sử
    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=current_user.user_id,
        action_type="view_history",
        description="Xem lịch sử hoạt động cá nhân"
    )
    db.add(history)
    db.commit()
    
    return histories


@router.get("", response_model=List[UserHistorySchema])
async def read_all_user_history(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Lấy tất cả lịch sử hoạt động (chỉ admin)
    """
    histories = db.query(UserHistory).order_by(desc(UserHistory.created_at)) \
        .offset(skip).limit(limit).all()
    return histories


@router.get("/{user_id}", response_model=List[UserHistorySchema])
async def read_user_history(
    *,
    db: Session = Depends(get_db),
    user_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Lấy lịch sử hoạt động của một người dùng cụ thể (chỉ admin)
    """
    # Kiểm tra người dùng tồn tại
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng",
        )
    
    histories = db.query(UserHistory).filter(UserHistory.user_id == user_id) \
        .order_by(desc(UserHistory.created_at)) \
        .offset(skip).limit(limit).all()
    
    # Thêm lịch sử
    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=current_user.user_id,
        action_type="view_user_history",
        description=f"Xem lịch sử hoạt động của người dùng {user.username}"
    )
    db.add(history)
    db.commit()
    
    return histories 