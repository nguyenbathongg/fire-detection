import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.base import get_db
from app.api.deps import get_current_active_user, get_current_active_admin
from app.models import User, UserHistory
from app.schemas import User as UserSchema, UserUpdate, UserCreate
from app.core.security import get_password_hash

router = APIRouter()


@router.get("/me", response_model=UserSchema)
def read_user_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Lấy thông tin người dùng hiện tại
    """
    return current_user


@router.put("/me", response_model=UserSchema)
def update_user_me(
    *,
    db: Session = Depends(get_db),
    user_in: UserUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Cập nhật thông tin người dùng hiện tại
    """
    # Kiểm tra username đã tồn tại chưa nếu đang thay đổi
    if user_in.username and user_in.username != current_user.username:
        if db.query(User).filter(User.username == user_in.username).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tên đăng nhập đã tồn tại",
            )
    
    # Không cho phép cập nhật email
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


@router.get("", response_model=List[UserSchema])
def read_users(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Lấy danh sách người dùng (chỉ admin)
    """
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@router.post("", response_model=UserSchema)
def create_user(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Tạo người dùng mới (chỉ admin)
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
        address=user_in.address,
        phone_number=user_in.phone_number,
        role="user",  # Mặc định là user
    )
    db.add(user)
    
    # Thêm lịch sử
    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=current_user.user_id,
        action_type="create_user",
        description=f"Tạo người dùng mới: {user_in.username}"
    )
    db.add(history)
    
    db.commit()
    db.refresh(user)
    return user


@router.get("/{user_id}", response_model=UserSchema)
def read_user(
    *,
    db: Session = Depends(get_db),
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Lấy thông tin người dùng theo ID (chỉ admin)
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng",
        )
    return user


@router.delete("/{user_id}")
def delete_user(
    *,
    db: Session = Depends(get_db),
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Xóa người dùng (chỉ admin)
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng",
        )
    
    # Không cho phép xóa chính mình
    if user.user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể xóa tài khoản của chính mình",
        )
    
    # Thêm lịch sử
    history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=current_user.user_id,
        action_type="delete_user",
        description=f"Xóa người dùng: {user.username}"
    )
    db.add(history)
    
    db.delete(user)
    db.commit()
    return {"message": "Xóa người dùng thành công"} 