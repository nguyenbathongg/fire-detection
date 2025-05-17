import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field

from app.models.enums import RoleEnum


# Schemas cho User
class UserBase(BaseModel):
    username: str
    email: EmailStr
    address: Optional[str] = None
    phone_number: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    address: Optional[str] = None
    phone_number: Optional[str] = None


class UserInDBBase(UserBase):
    user_id: uuid.UUID
    role: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class User(UserInDBBase):
    pass


class UserInDB(UserInDBBase):
    password_hash: str


# Schema cho đăng nhập
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# Schemas cho xác thực
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: str
    role: str
    exp: int


class PasswordChange(BaseModel):
    current_password: str
    new_password: str 