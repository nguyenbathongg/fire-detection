from datetime import timedelta
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
import uuid

from app.models import User
from app.schemas import UserCreate, User as UserSchema
from app.core.security import create_access_token, get_password_hash, verify_password
from app.core.config import settings
from jose import jwt, JWTError
from app.schemas.user import TokenPayload


class AuthController:
    """
    Controller xử lý logic nghiệp vụ liên quan đến xác thực người dùng
    """

    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
        """
        Xác thực người dùng
        """
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.password_hash):
            return None
        return user

    @staticmethod
    def create_user_token(user: User) -> dict:
        """
        Tạo token xác thực cho người dùng
        """
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            str(user.user_id), user.role, expires_delta=access_token_expires
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
        }

    @staticmethod
    def login(db: Session, email: str, password: str) -> dict:
        """
        Xử lý đăng nhập
        """
        user = AuthController.authenticate_user(db, email, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Email hoặc mật khẩu không chính xác",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return AuthController.create_user_token(user)

    @staticmethod
    def register(db: Session, user_in: UserCreate) -> User:
        """
        Xử lý đăng ký
        """
        # Kiểm tra email đã tồn tại chưa
        if db.query(User).filter(User.email == user_in.email).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email đã tồn tại",
            )
        
        # Tạo người dùng mới
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

    @staticmethod
    def change_password(db: Session, current_user: User, current_password: str, new_password: str) -> bool:
        """
        Đổi mật khẩu
        """
        if not verify_password(current_password, current_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mật khẩu hiện tại không chính xác",
            )
        
        current_user.password_hash = get_password_hash(new_password)
        db.add(current_user)
        db.commit()
        return True

    @staticmethod
    def verify_token(db: Session, token: str) -> User:
        """
        Xác thực token
        """
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            token_data = TokenPayload(**payload)
            
            # Tìm người dùng trong database
            user = db.query(User).filter(User.user_id == token_data.sub).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Người dùng không tồn tại"
                )
            
            return user
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token không hợp lệ hoặc đã hết hạn",
                headers={"WWW-Authenticate": "Bearer"},
            ) 