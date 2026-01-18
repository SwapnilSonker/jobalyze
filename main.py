from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from dotenv import load_dotenv

# Imports
from utils import extract_text_from_pdf, extract_text_from_image # (Same as previous utils.py)
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

@app.post("/generate-agent", response_model=AgentResponse)
async def generate_agent(
    file: UploadFile,
    jd_text: str = Form(...)
):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Extraction (Utils same as before)
        raw_text = ""
        if file.filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_location)
        elif file.filename.endswith((".png", ".jpg", ".jpeg")):
            raw_text = extract_text_from_image(file_location) # Uses Groq Vision
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")

        # 2. RAG Retrieval (Optimization)
        # Agar resume bohot bada hai to RAG use karo
        if len(raw_text) > 4000:
            vector_db = setup_vector_store(raw_text)
            context = get_relevant_context(vector_db, query=jd_text)
        else:
            context = raw_text

        # 3. Run LangChain Agent
        feedback, message = run_agent_workflow(context, jd_text)

        return AgentResponse(feedback=feedback, message=message)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)