import uuid
from typing import Optional, List
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models import User, UserHistory
from app.schemas import UserCreate, UserUpdate
from app.core.security import get_password_hash


def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
    return db.query(User).filter(User.user_id == user_id).first()


def get_all_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    return db.query(User).offset(skip).limit(limit).all()


def create_user(
    db: Session,
    admin_user: User,
    user_in: UserCreate
) -> User:
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
        address=user_in.address,
        phone_number=user_in.phone_number,
        role="user",
    )
    db.add(user)

    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=admin_user.user_id,
        action_type="create_user",
        description=f"Tạo người dùng mới: {user_in.username}"
    )
    db.add(history)

    db.commit()
    db.refresh(user)
    return user


def update_current_user(
    db: Session,
    user: User,
    update_data: UserUpdate
) -> User:
    if update_data.username and update_data.username != user.username:
        if db.query(User).filter(User.username == update_data.username).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tên đăng nhập đã tồn tại",
            )

    if update_data.username:
        user.username = update_data.username
    if update_data.password:
        user.password_hash = get_password_hash(update_data.password)
    if update_data.address is not None:
        user.address = update_data.address
    if update_data.phone_number is not None:
        user.phone_number = update_data.phone_number

    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=user.user_id,
        action_type="update_profile",
        description="Cập nhật thông tin cá nhân"
    )
    db.add(history)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def delete_user_by_admin(
    db: Session,
    user_id: uuid.UUID,
    current_admin: User
):
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng",
        )
    if user.user_id == current_admin.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể xóa tài khoản của chính mình",
        )

    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=current_admin.user_id,
        action_type="delete_user",
        description=f"Xóa người dùng: {user.username}"
    )
    db.add(history)

    db.delete(user)
    db.commit()
    return {"message": "Xóa người dùng thành công"}
