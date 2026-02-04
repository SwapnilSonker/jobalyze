from pydantic import BaseModel
from typing import Optional


class TemplateResponse(BaseModel):
    """Schema for resume template response"""
    id: int
    name: str
    description: Optional[str]
    markdown_content: str
    is_default: int
    
    class Config:
        from_attributes = True
