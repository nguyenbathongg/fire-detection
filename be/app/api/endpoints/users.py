import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.base import get_db
from app.api.deps import get_current_active_user, get_current_active_admin
from app.models import User
from app.schemas import User as UserSchema, UserUpdate, UserCreate
from app.controllers.user_controller import UserController

router = APIRouter()


@router.get("/me", response_model=UserSchema)
async def read_user_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Lấy thông tin người dùng hiện tại
    """
    return current_user


@router.put("/me", response_model=UserSchema)
async def update_user_me(
    *,
    db: Session = Depends(get_db),
    user_in: UserUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Cập nhật thông tin người dùng hiện tại
    """
    return UserController.update_user(db=db, current_user=current_user, user_in=user_in)


@router.get("", response_model=List[UserSchema])
async def read_users(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Lấy danh sách người dùng (chỉ admin)
    """
    return UserController.get_users(db=db, skip=skip, limit=limit)


@router.post("", response_model=UserSchema)
async def create_user(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Tạo người dùng mới (chỉ admin)
    """
    return UserController.create_user(db=db, user_in=user_in, created_by_id=current_user.user_id)


@router.get("/{user_id}", response_model=UserSchema)
async def read_user(
    *,
    db: Session = Depends(get_db),
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Lấy thông tin người dùng theo ID (chỉ admin)
    """
    user = UserController.get_user_by_id(db=db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng",
        )
    return user


@router.delete("/{user_id}")
async def delete_user(
    *,
    db: Session = Depends(get_db),
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Xóa người dùng (chỉ admin)
    """
    UserController.delete_user(db=db, user_id=user_id, current_user_id=current_user.user_id)
    return {"message": "Xóa người dùng thành công"} 