from datetime import timedelta
import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import verify_password, get_password_hash, create_access_token
from app.models import User
from app.schemas import UserCreate, PasswordChange
from app.core.config import settings


def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email hoặc mật khẩu không chính xác",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def generate_access_token(user: User):
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": create_access_token(
            str(user.user_id), user.role, expires_delta=access_token_expires
        ),
        "token_type": "bearer",
    }


def register_user(db: Session, user_in: UserCreate) -> User:
    if db.query(User).filter(User.email == user_in.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email đã tồn tại",
        )
    
    user = User(
        user_id=uuid.uuid4(),
        username=user_in.username,
        email=user_in.email,
        password_hash=get_password_hash(user_in.password),
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def change_user_password(db: Session, current_user: User, password_change: PasswordChange):
    if not verify_password(password_change.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mật khẩu hiện tại không chính xác",
        )
    
    current_user.password_hash = get_password_hash(password_change.new_password)
    db.add(current_user)
    db.commit()
    return {"message": "Đổi mật khẩu thành công"}
