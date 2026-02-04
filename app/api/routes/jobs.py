from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List

from app.schemas.job import JobCreate, JobResponse
from app.api.deps import get_db, get_current_user
from app.db import models

router = APIRouter()


@router.post("/save", response_model=JobResponse)
async def save_job(
    job: JobCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Save a job description for later use.
    Requires authentication.
    """
    db_job = models.Job(
        user_id=current_user.id,
        job_title=job.job_title,
        company=job.company,
        jd_text=job.jd_text,
        job_url=job.job_url
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    return JobResponse.model_validate(db_job)


@router.get("/list", response_model=List[JobResponse])
async def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all saved jobs for the authenticated user.
    Supports pagination.
    """
    jobs = db.query(models.Job).filter(
        models.Job.user_id == current_user.id
    ).order_by(models.Job.created_at.desc()).offset(skip).limit(limit).all()
    
    return [JobResponse.model_validate(job) for job in jobs]


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific saved job by ID.
    Requires ownership.
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id,
        models.Job.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or access denied")
    
    return JobResponse.model_validate(job)


@router.delete("/{job_id}")
async def delete_job(
    job_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a saved job by ID.
    Requires ownership.
    """
    job = db.query(models.Job).filter(
        models.Job.id == job_id,
        models.Job.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or access denied")
    
    db.delete(job)
    db.commit()
    
    return {"message": "Job deleted successfully", "job_id": job_id}
