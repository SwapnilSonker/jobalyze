from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import uuid
from dotenv import load_dotenv

# Imports
# Note: 'save_resume_as_pdf' hata diya kyunki ab hum DOCX use kar rahe hain
from utils import extract_text_from_pdf, extract_text_from_image, convert_pdf_to_docx, update_word_resume
from vector_store import setup_vector_store, get_relevant_context
from ai_engine import run_agent_workflow
from schemas import AgentResponse

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("generated_resumes", exist_ok=True)

@app.post("/generate-agent", response_model=AgentResponse)
async def generate_agent(
    file: UploadFile,
    jd_text: str = Form(...)
):
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
            # Note: Image hui to convert_pdf_to_docx fail ho jayega. 
            # Isliye ye flow strictly PDF ke liye best hai.
            raw_text = extract_text_from_image(temp_pdf_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")

        # 3. Conversion Strategy (PDF -> DOCX)
        temp_docx_filename = f"converted_{uuid.uuid4()}.docx"
        temp_docx_path = f"generated_resumes/{temp_docx_filename}"
        
        print("Converting PDF to DOCX...")
        # Only try converting if it's a PDF
        if file.filename.endswith(".pdf"):
            conversion_success = convert_pdf_to_docx(temp_pdf_path, temp_docx_path)
            if not conversion_success:
                raise HTTPException(status_code=500, detail="Failed to convert PDF to Word")
        else:
            # Agar image hai to hum edit nahi kar sakte, sirf analyze kar sakte hain
            # Dummy file create karni padegi ya error dena padega
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

# --- DOWNLOAD ENDPOINT (Updated for DOCX) ---
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"generated_resumes/{filename}"
    if os.path.exists(file_path):
        # Media type for MS Word
        return FileResponse(
            file_path, 
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
            filename=filename
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")