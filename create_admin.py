import sys
import uuid
from datetime import datetime
import argparse
import os
import traceback

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Thêm đường dẫn hiện tại vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load biến môi trường
load_dotenv()

from app.core.security import get_password_hash


def create_admin_user(username, email, password):
    """
    Tạo tài khoản admin
    """
    # Lấy DATABASE_URL từ biến môi trường
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Không tìm thấy DATABASE_URL trong biến môi trường")
        return False
    
    print(f"Kết nối đến database: {database_url}")
    
    try:
        # Kết nối trực tiếp đến PostgreSQL
        conn = psycopg2.connect(
            database="fire-detection",
            user="postgres",
            password="1",
            host="localhost",
            port="5432"
        )
        
        # Tạo cursor
        cursor = conn.cursor()
        
        # Kiểm tra username đã tồn tại chưa
        cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            print(f"Tên đăng nhập '{username}' đã tồn tại")
            conn.close()
            return False
        
        # Kiểm tra email đã tồn tại chưa
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            print(f"Email '{email}' đã tồn tại")
            conn.close()
            return False
        
        # Tạo mật khẩu hash
        password_hash = get_password_hash(password)
        
        # Tạo user_id
        user_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        updated_at = created_at
        
        # Thực hiện INSERT
        query = """
        INSERT INTO users (user_id, username, email, password_hash, role, created_at, updated_at) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            user_id, 
            username, 
            email, 
            password_hash, 
            'admin',  # Sử dụng 'admin' trực tiếp
            created_at, 
            updated_at
        ))
        
        # Commit transaction
        conn.commit()
        print(f"Đã tạo tài khoản admin cho '{username}' thành công")
        
        # Đóng kết nối
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Lỗi khi tạo tài khoản admin:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo tài khoản admin cho hệ thống phát hiện đám cháy")
    parser.add_argument("--username", type=str, required=True, help="Tên đăng nhập")
    parser.add_argument("--email", type=str, required=True, help="Email")
    parser.add_argument("--password", type=str, required=True, help="Mật khẩu")
    
    args = parser.parse_args()
    
    # Tạo admin
    create_admin_user(args.username, args.email, args.password) 