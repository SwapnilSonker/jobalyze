from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.schemas.template import TemplateResponse
from app.api.deps import get_db
from app.db import models

router = APIRouter()


@router.get("/list", response_model=List[TemplateResponse])
async def list_templates(db: Session = Depends(get_db)):
    """
    List all available resume templates.
    Public endpoint (no authentication required).
    """
    templates = db.query(models.Template).all()
    return [TemplateResponse.model_validate(t) for t in templates]


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific template by ID.
    Public endpoint.
    """
    template = db.query(models.Template).filter(models.Template.id == template_id).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return TemplateResponse.model_validate(template)
