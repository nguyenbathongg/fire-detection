from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.api.deps import get_current_active_user
from app.schemas import Token, UserCreate, User as UserSchema, PasswordChange, LoginRequest
from app.models import User
from app.services import auth as auth_service  

router = APIRouter()


@router.post("/login", response_model=Token)
def login_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    Đăng nhập (dùng OAuth2 Form – nhập email qua trường username)
    """
    user = auth_service.authenticate_user(db, form_data.username, form_data.password)
    return auth_service.generate_access_token(user)


@router.post("/login-email", response_model=Token)
def login_with_email(
    login_data: LoginRequest,
    db: Session = Depends(get_db),
) -> Any:
    """
    Đăng nhập bằng email và mật khẩu (không dùng OAuth2 form)
    """
    user = auth_service.authenticate_user(db, login_data.email, login_data.password)
    return auth_service.generate_access_token(user)


@router.post("/register", response_model=UserSchema)
def register_user(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
) -> Any:
    return auth_service.register_user(db, user_in)


@router.post("/change-password")
def change_password(
    *,
    db: Session = Depends(get_db),
    password_change: PasswordChange,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    return auth_service.change_user_password(db, current_user, password_change)
