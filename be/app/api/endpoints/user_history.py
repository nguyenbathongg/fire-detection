from typing import Any, List
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.api.deps import get_current_active_user, get_current_active_admin
from app.models import User
from app.schemas import UserHistory as UserHistorySchema
from app.controllers.user_history_controller import UserHistoryController

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
    # Lấy lịch sử người dùng
    histories = UserHistoryController.get_user_history(
        db=db, 
        user_id=current_user.user_id, 
        skip=skip, 
        limit=limit
    )
    
    # Thêm lịch sử cho việc xem lịch sử
    UserHistoryController.add_view_history(db=db, user_id=current_user.user_id)
    
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
    return UserHistoryController.get_all_histories(db=db, skip=skip, limit=limit)


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
    # Thêm lịch sử xem lịch sử người dùng
    UserHistoryController.add_view_user_history(
        db=db, 
        admin_id=current_user.user_id, 
        target_user_id=user_id
    )
    
    # Lấy lịch sử người dùng
    return UserHistoryController.get_user_history(db=db, user_id=user_id, skip=skip, limit=limit) 