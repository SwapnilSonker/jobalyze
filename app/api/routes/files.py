from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter()


@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a generated file (resume, cover letter, or translated document).
    """
    file_path = f"generated_resumes/{filename}"
    if os.path.exists(file_path):
        # Determine media type based on extension
        if filename.endswith('.pdf'):
            media_type = 'application/pdf'
        elif filename.endswith('.docx'):
            media_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        else:
            media_type = 'application/octet-stream'
        
        return FileResponse(
            file_path, 
            media_type=media_type, 
            filename=filename
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy", 
        "message": "Jobalyze API v2.0 is running",
        "features": [
            "Multi-step AI workflow",
            "Cover letter generation",
            "Resume translation",
            "ATS simulation",
            "Job management",
            "Template system"
        ]
    }
