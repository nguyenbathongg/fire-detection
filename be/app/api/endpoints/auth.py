from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.api.deps import get_current_active_user
from app.models import User
from app.schemas import Token, UserCreate, User as UserSchema, PasswordChange, LoginRequest
from app.controllers.auth_controller import AuthController

router = APIRouter()


@router.post("/login", response_model=Token)
def login_access_token(
    db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    Đăng nhập để lấy token access (dùng username field để nhập email)
    """
    # Sử dụng form_data.username là email thực tế
    return AuthController.login(db=db, email=form_data.username, password=form_data.password)


@router.post("/login-email", response_model=Token)
def login_with_email(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
) -> Any:
    """
    Đăng nhập bằng email và mật khẩu
    """
    return AuthController.login(db=db, email=login_data.email, password=login_data.password)


@router.post("/register", response_model=UserSchema)
def register_user(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
) -> Any:
    """
    Đăng ký người dùng mới
    """
    return AuthController.register(db=db, user_in=user_in)


@router.post("/change-password")
def change_password(
    *,
    db: Session = Depends(get_db),
    password_change: PasswordChange,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Đổi mật khẩu
    """
    AuthController.change_password(
        db=db, 
        current_user=current_user, 
        current_password=password_change.current_password, 
        new_password=password_change.new_password
    )
    return {"message": "Đổi mật khẩu thành công"}


@router.post("/verify-token", response_model=UserSchema)
def verify_token(
    token: str,
    db: Session = Depends(get_db)
) -> Any:
    """
    Xác thực token và trả về thông tin người dùng nếu token hợp lệ
    """
    return AuthController.verify_token(db=db, token=token)