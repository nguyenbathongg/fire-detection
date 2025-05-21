import uuid
from typing import List, Optional

from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.models import User, UserHistory
from app.schemas import UserCreate, UserUpdate
from app.core.security import get_password_hash


class UserController:
    """
    Controller xử lý logic nghiệp vụ liên quan đến người dùng
    """
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
        """Lấy người dùng theo ID"""
        return db.query(User).filter(User.user_id == user_id).first()
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Lấy người dùng theo email"""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Lấy người dùng theo username"""
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """Lấy danh sách người dùng"""
        return db.query(User).offset(skip).limit(limit).all()
    
    @staticmethod
    def create_user(db: Session, user_in: UserCreate, created_by_id: Optional[uuid.UUID] = None) -> User:
        """Tạo người dùng mới"""
        # Kiểm tra email đã tồn tại chưa
        if UserController.get_user_by_email(db, email=user_in.email):
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
            address=user_in.address,
            phone_number=user_in.phone_number,
            role="user",  # Mặc định là user
        )
        db.add(user)
        
        # Thêm lịch sử nếu có người tạo
        if created_by_id:
            history = UserHistory(
                history_id=uuid.uuid4(),
                user_id=created_by_id,
                action_type="create_user",
                description=f"Tạo người dùng mới: {user_in.username}"
            )
            db.add(history)
        
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def update_user(db: Session, current_user: User, user_in: UserUpdate) -> User:
        """Cập nhật thông tin người dùng"""
        # Kiểm tra username đã tồn tại chưa nếu đang thay đổi
        if user_in.username and user_in.username != current_user.username:
            if UserController.get_user_by_username(db, username=user_in.username):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tên đăng nhập đã tồn tại",
                )
        
        # Cập nhật thông tin
        if user_in.username:
            current_user.username = user_in.username
        # Không cập nhật email
        if user_in.password:
            current_user.password_hash = get_password_hash(user_in.password)
        if user_in.address is not None:
            current_user.address = user_in.address
        if user_in.phone_number is not None:
            current_user.phone_number = user_in.phone_number
        
        # Thêm lịch sử
        history = UserHistory(
            history_id=uuid.uuid4(),
            user_id=current_user.user_id,
            action_type="update_profile",
            description="Cập nhật thông tin cá nhân"
        )
        db.add(history)
        
        db.add(current_user)
        db.commit()
        db.refresh(current_user)
        return current_user
    
    @staticmethod
    def delete_user(db: Session, user_id: uuid.UUID, current_user_id: uuid.UUID) -> None:
        """Xóa người dùng"""
        user = UserController.get_user_by_id(db, user_id=user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Không tìm thấy người dùng",
            )
        
        # Không cho phép xóa chính mình
        if user.user_id == current_user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Không thể xóa tài khoản của chính mình",
            )
        
        # Thêm lịch sử
        history = UserHistory(
            history_id=uuid.uuid4(),
            user_id=current_user_id,
            action_type="delete_user",
            description=f"Xóa người dùng: {user.username}"
        )
        db.add(history)
        
        db.delete(user)
        db.commit() 