from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to activities
    activities = relationship("ResumeActivity", back_populates="user")


class ResumeActivity(Base):
    """Tracks each resume analysis/update activity"""
    __tablename__ = "resume_activities"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    original_filename = Column(String, nullable=False)
    modified_filename = Column(String, nullable=False)
    original_score = Column(Integer, nullable=False)
    optimized_score = Column(Integer, nullable=False)
    job_title = Column(String, nullable=True)  # Optional: extracted from JD
    created_at = Column(DateTime, default=datetime.utcnow)
    download_link = Column(String, nullable=True)
    
    # Relationship back to user
    user = relationship("User", back_populates="activities")
