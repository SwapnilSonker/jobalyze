from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.schemas.dashboard import DashboardResponse, DashboardStats, ActivityItem
from app.schemas.auth import UserResponse
from app.api.deps import get_db, get_current_user
from app.db import models

router = APIRouter()


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user dashboard with stats and recent activities.
    Requires authentication.
    """
    # Get recent activities
    activities = db.query(models.ResumeActivity).filter(
        models.ResumeActivity.user_id == current_user.id
    ).order_by(models.ResumeActivity.created_at.desc()).limit(10).all()
    
    # Calculate stats
    total = len(current_user.activities)
    avg_improvement = 0.0
    if total > 0:
        improvements = [a.optimized_score - a.original_score for a in current_user.activities]
        avg_improvement = sum(improvements) / total
    
    return DashboardResponse(
        user=UserResponse(
            id=current_user.id,
            email=current_user.email,
            username=current_user.username,
            created_at=current_user.created_at
        ),
        stats=DashboardStats(
            total_resumes_updated=total,
            average_score_improvement=round(avg_improvement, 1),
            latest_activities=[ActivityItem.model_validate(a) for a in activities]
        )
    )


@router.get("/dashboard/activities", response_model=List[ActivityItem])
async def get_all_activities(
    skip: int = 0,
    limit: int = 20,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of all user activities.
    Requires authentication.
    """
    activities = db.query(models.ResumeActivity).filter(
        models.ResumeActivity.user_id == current_user.id
    ).order_by(models.ResumeActivity.created_at.desc()).offset(skip).limit(limit).all()
    
    return [ActivityItem.model_validate(a) for a in activities]
