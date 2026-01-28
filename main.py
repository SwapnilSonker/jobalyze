from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import shutil
import os
import uuid
from dotenv import load_dotenv

# Imports
from utils import extract_text_from_pdf, extract_text_from_image, convert_pdf_to_docx, update_word_resume
from vector_store import setup_vector_store, get_relevant_context
from ai_engine import run_agent_workflow
from schemas import (
    AgentResponse, UserCreate, UserResponse, Token, LoginRequest,
    DashboardResponse, DashboardStats, ActivityItem
)
from database import engine, get_db, Base
from auth import get_password_hash, verify_password, create_access_token, get_current_user
import models

load_dotenv()
app = FastAPI(title="Jobalyze API", description="AI-powered resume analyzer and optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables on startup
Base.metadata.create_all(bind=engine)

os.makedirs("generated_resumes", exist_ok=True)


# --- AUTH ENDPOINTS ---

@app.post("/signup", response_model=UserResponse, tags=["Authentication"])
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user account.
    """
    # Check if email already exists
    if db.query(models.User).filter(models.User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check if username already exists
    if db.query(models.User).filter(models.User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create new user
    db_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=get_password_hash(user.password)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.post("/login", response_model=Token, tags=["Authentication"])
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    Login and receive JWT access token.
    """
    user = db.query(models.User).filter(models.User.email == login_data.email).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


# --- DASHBOARD ENDPOINTS ---

@app.get("/dashboard", response_model=DashboardResponse, tags=["Dashboard"])
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


@app.get("/dashboard/activities", response_model=list[ActivityItem], tags=["Dashboard"])
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


# --- RESUME GENERATION ENDPOINT (Updated with auth) ---

@app.post("/generate-agent", response_model=AgentResponse, tags=["Resume Analysis"])
async def generate_agent(
    file: UploadFile,
    jd_text: str = Form(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze and optimize a resume against a job description.
    Requires authentication.
    """
    # 1. Save Uploaded File
    original_filename = file.filename
    temp_pdf_path = f"temp_{original_filename}"
    
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Text Extraction (AI ke liye raw text)
        raw_text = ""
        if file.filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(temp_pdf_path)
        elif file.filename.endswith((".png", ".jpg", ".jpeg")):
            raw_text = extract_text_from_image(temp_pdf_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")

        # 3. Conversion Strategy (PDF -> DOCX)
        temp_docx_filename = f"converted_{uuid.uuid4()}.docx"
        temp_docx_path = f"generated_resumes/{temp_docx_filename}"
        
        print("Converting PDF to DOCX...")
        if file.filename.endswith(".pdf"):
            conversion_success = convert_pdf_to_docx(temp_pdf_path, temp_docx_path)
            if not conversion_success:
                raise HTTPException(status_code=500, detail="Failed to convert PDF to Word")
        else:
            raise HTTPException(status_code=400, detail="Round-trip editing only supports PDF files.")

        # 4. AI Processing
        if len(raw_text) > 4000:
            vector_db = setup_vector_store(raw_text)
            context = get_relevant_context(vector_db, query=jd_text)
            print(f"📊 Using RAG: Retrieved {len(context)} chars from vector store")
            print(f"📄 Context preview (first 500 chars): {context[:500]}...")
        else:
            context = raw_text
            print(f"📊 Using full resume: {len(context)} chars")

        print("AI Analyzing & Generating Edits...")
        feedback, message = run_agent_workflow(context, jd_text)

        # 5. Apply Edits to DOCX
        print(f"✅ AI Generated {len(feedback.detailed_edits)} edits")
        print("Applying Edits to DOCX...")
        
        edits_list = []
        for edit in feedback.detailed_edits:
            edits_list.append({
                "original_text": edit.original_text,
                "new_text": edit.new_text
            })
            print(f"  - {edit.section}: {edit.change_type}")
            
        final_docx_path = update_word_resume(temp_docx_path, edits_list, f"final_{uuid.uuid4()}.docx")

        # 6. Generate Link
        final_filename = os.path.basename(final_docx_path)
        download_url = f"http://localhost:8000/download/{final_filename}"

        # 7. Save Activity to Database
        activity = models.ResumeActivity(
            user_id=current_user.id,
            original_filename=original_filename,
            modified_filename=final_filename,
            original_score=feedback.original_score,
            optimized_score=feedback.optimized_score,
            download_link=download_url
        )
        db.add(activity)
        db.commit()
        print(f"📝 Activity saved for user: {current_user.username}")

        return AgentResponse(
            feedback=feedback, 
            message=message,
            file_download_link=download_url 
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temp PDF only (Keep DOCX for download)
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


# --- DOWNLOAD ENDPOINT ---

@app.get("/download/{filename}", tags=["Resume Analysis"])
async def download_file(filename: str):
    """
    Download a generated resume file.
    """
    file_path = f"generated_resumes/{filename}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
            filename=filename
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")


# --- HEALTH CHECK ---

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "message": "Jobalyze API is running"}