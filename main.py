from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles # Import for static serving if needed
import shutil
import os
import uuid
from dotenv import load_dotenv

# Imports
from utils import extract_text_from_pdf, extract_text_from_image, save_resume_as_pdf
from vector_store import setup_vector_store, get_relevant_context
from ai_engine import run_agent_workflow
from schemas import AgentResponse

load_dotenv()
app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the generated_resumes folder exists
os.makedirs("generated_resumes", exist_ok=True)

@app.post("/generate-agent", response_model=AgentResponse)
async def generate_agent(
    file: UploadFile,
    jd_text: str = Form(...)
):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Extraction
        raw_text = ""
        if file.filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_location)
        elif file.filename.endswith((".png", ".jpg", ".jpeg")):
            raw_text = extract_text_from_image(file_location)
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")

        # 2. Context Logic
        if len(raw_text) > 4000:
            vector_db = setup_vector_store(raw_text)
            context = get_relevant_context(vector_db, query=jd_text)
        else:
            context = raw_text

        # 3. AI Workflow
        print("Running AI Workflow...")
        feedback, message = run_agent_workflow(context, jd_text)
        print("AI Workflow Complete.")

        # --- 4. PDF GENERATION (CRITICAL STEP) ---
        print("Generating PDF...")
        new_filename = f"optimized_resume_{uuid.uuid4()}.pdf"
        
        # Call the utils function to create the PDF file
        pdf_path = save_resume_as_pdf(feedback.rewritten_content, new_filename)
        
        if not pdf_path:
             # Fallback if PDF generation fails (prevents crash)
             print("PDF Generation Failed")
             download_url = ""
        else:
             # Construct the Download URL
             # Make sure to use http://localhost:8000 (or your ngrok url)
             download_url = f"http://localhost:8000/download/{new_filename}"
             print(f"PDF Generated: {download_url}")

        # 5. Return Response
        return AgentResponse(
            feedback=feedback, 
            message=message,
            pdf_download_link=download_url # <--- Sending the link back
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

# --- DOWNLOAD ENDPOINT ---
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"generated_resumes/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")