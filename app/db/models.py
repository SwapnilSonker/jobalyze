from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    activities = relationship("ResumeActivity", back_populates="user")
    jobs = relationship("Job", back_populates="user")


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


class Job(Base):
    """Saved job descriptions for users"""
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    job_title = Column(String, nullable=False)
    company = Column(String, nullable=False)
    jd_text = Column(Text, nullable=False)
    job_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to user
    user = relationship("User", back_populates="jobs")


class Template(Base):
    """Resume templates for users to choose from"""
    __tablename__ = "templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)
    markdown_content = Column(Text, nullable=False)
    is_default = Column(Integer, default=0)  # SQLite Boolean workaround (0=False, 1=True)
    created_at = Column(DateTime, default=datetime.utcnow)
