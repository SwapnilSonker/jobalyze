from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class JobCreate(BaseModel):
    """Schema for creating a new saved job"""
    job_title: str
    company: str
    jd_text: str
    job_url: Optional[str] = None


class JobResponse(BaseModel):
    """Schema for job response"""
    id: int
    job_title: str
    company: str
    jd_text: str
    job_url: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
