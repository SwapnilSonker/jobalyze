from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from app.schemas.auth import UserResponse


class ActivityItem(BaseModel):
    """Schema for a single resume activity"""
    id: int
    original_filename: str
    modified_filename: str
    original_score: int
    optimized_score: int
    job_title: Optional[str]
    created_at: datetime
    download_link: Optional[str]
    
    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    """Schema for dashboard statistics"""
    total_resumes_updated: int
    average_score_improvement: float
    latest_activities: List[ActivityItem]


class DashboardResponse(BaseModel):
    """Schema for complete dashboard response"""
    user: UserResponse
    stats: DashboardStats
