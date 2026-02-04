from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production-12345"
    GROQ_API_KEY: str
    DATABASE_URL: str = "sqlite:///./jobalyze.db"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
