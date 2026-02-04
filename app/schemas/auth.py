from pydantic import BaseModel
from datetime import datetime


class UserCreate(BaseModel):
    """Schema for user registration"""
    email: str
    username: str
    password: str


class UserResponse(BaseModel):
    """Schema for user response (without password)"""
    id: int
    email: str
    username: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str


class LoginRequest(BaseModel):
    """Schema for login request"""
    email: str
    password: str
