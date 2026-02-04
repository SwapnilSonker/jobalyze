"""
Jobalyze API - AI-powered resume analyzer and optimizer

Main application entry point with route configuration.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.db.database import engine, Base
from app.api.routes import auth, dashboard, resume, jobs, templates, files

# Create database tables on startup
Base.metadata.create_all(bind=engine)

# Ensure generated_resumes directory exists
os.makedirs("generated_resumes", exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Jobalyze API",
    description="AI-powered resume analyzer and optimizer with multi-step workflows",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, tags=["Authentication"])
app.include_router(dashboard.router, tags=["Dashboard"])
app.include_router(resume.router, tags=["Resume Analysis"])
app.include_router(jobs.router, prefix="/jobs", tags=["Job Management"])
app.include_router(templates.router, prefix="/templates", tags=["Templates"])
app.include_router(files.router, tags=["Files"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
