from typing import Any, List
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.api.deps import get_current_active_user, get_current_active_admin
from app.schemas import User as UserSchema, UserCreate, UserUpdate
from app.models import User
from app.services import user as user_service

router = APIRouter()


@router.get("/me", response_model=UserSchema)
def read_user_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    return current_user


@router.put("/me", response_model=UserSchema)
def update_user_me(
    *,
    db: Session = Depends(get_db),
    user_in: UserUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    return user_service.update_current_user(db, current_user, user_in)


@router.get("", response_model=List[UserSchema])
def read_users(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    return user_service.get_all_users(db, skip, limit)


@router.post("", response_model=UserSchema)
def create_user(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    return user_service.create_user(db, current_user, user_in)


@router.get("/{user_id}", response_model=UserSchema)
def read_user(
    *,
    db: Session = Depends(get_db),
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    user = user_service.get_user_by_id(db, user_id)
    if not user:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy người dùng")
    return user


@router.delete("/{user_id}")
def delete_user(
    *,
    db: Session = Depends(get_db),
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    return user_service.delete_user_by_admin(db, user_id, current_user)
