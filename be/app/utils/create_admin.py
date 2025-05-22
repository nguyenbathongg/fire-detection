import sys
import os
import getpass
from sqlalchemy.orm import Session

# Thêm thư mục gốc vào sys.path để import các module
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, root_dir)

from app.db.session import SessionLocal
from app.models import User
from app.core.security import get_password_hash
import uuid


def create_admin_user(db: Session, email: str, password: str, username: str) -> User:
    """
    Tạo tài khoản admin mới
    """
    # Kiểm tra email đã tồn tại chưa
    if db.query(User).filter(User.email == email).first():
        print(f"Lỗi: Email {email} đã tồn tại trong hệ thống.")
        return None
    
    # Tạo người dùng admin mới
    user = User(
        user_id=uuid.uuid4(),
        username=username,
        email=email,
        password_hash=get_password_hash(password),
        role="admin",  # Đặt quyền admin
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def main():
    print("===== Tạo tài khoản Admin =====")
    
    email = input("Nhập email admin: ")
    username = input("Nhập tên người dùng: ")
    
    # Sử dụng getpass để ẩn mật khẩu khi nhập
    password = getpass.getpass("Nhập mật khẩu: ")
    password_confirm = getpass.getpass("Nhập lại mật khẩu: ")
    
    # Kiểm tra mật khẩu nhập lại
    if password != password_confirm:
        print("Lỗi: Mật khẩu nhập lại không khớp.")
        return
    
    # Tạo session database
    db = SessionLocal()
    try:
        # Tạo tài khoản admin
        user = create_admin_user(db, email, password, username)
        if user:
            print(f"Đã tạo tài khoản admin thành công với email: {email}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
