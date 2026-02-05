from fastapi import APIRouter, UploadFile, Form, HTTPException, Depends
from sqlalchemy.orm import Session
import shutil
import os
import uuid

from app.schemas.resume import (
    EnhancedAgentResponse, ResumeFeedback, LinkedInDraft,
    CoverLetterResponse, TranslationResponse
)
from app.api.deps import get_db, get_current_user
from app.db import models
from app.services.file_service import (
    extract_text_from_pdf, extract_text_from_image,
    save_resume_as_pdf, save_cover_letter_as_pdf, save_translated_resume_as_pdf
)
from app.services.vector_service import setup_vector_store, get_relevant_context
from app.services.ai_service import (
    run_enhanced_agent_workflow, cover_letter_chain, translation_chain,
    extract_company_job_title, keyword_extraction_prompt, llm
)

router = APIRouter()


@router.post("/generate-agent", response_model=EnhancedAgentResponse)
async def generate_agent(
    file: UploadFile,
    jd_text: str = Form(...),
    generate_cover_letter: bool = Form(True),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Enhanced AI workflow: Analyze and optimize resume with multi-step processing.
    Includes keyword extraction, honest scoring, rewriting, cover letter, and ATS simulation.
    Requires authentication.
    """
    # 1. Save Uploaded File
    original_filename = file.filename
    temp_file_path = f"temp_{uuid.uuid4()}_{original_filename}"
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Text Extraction
        raw_text = ""
        if file.filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(temp_file_path)
        elif file.filename.endswith((".png", ".jpg", ".jpeg")):
            raw_text = extract_text_from_image(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Use PDF or image files.")

        # 3. Use vector store for long resumes
        if len(raw_text) > 4000:
            vector_db = setup_vector_store(raw_text)
            context = get_relevant_context(vector_db, query=jd_text)
            print(f"📊 Using RAG: Retrieved {len(context)} chars from vector store")
        else:
            context = raw_text
            print(f"📊 Using full resume: {len(context)} chars")

        # 4. Run Enhanced Multi-Step AI Workflow
        print("🤖 Running enhanced AI workflow...")
        workflow_results = run_enhanced_agent_workflow(
            resume_text=context,
            jd_text=jd_text,
            generate_cover_letter=generate_cover_letter,
            translate_to=None
        )

        # 5. Save optimized resume as PDF
        resume_filename = f"resume_{uuid.uuid4()}.pdf"
        resume_pdf_path = save_resume_as_pdf(
            workflow_results['rewritten_resume'], 
            resume_filename
        )
        resume_download_url = f"http://localhost:8000/download/{resume_filename}"

        # 6. Save cover letter as PDF (if generated)
        cover_letter_url = None
        if workflow_results['cover_letter']:
            cover_letter_filename = f"cover_letter_{uuid.uuid4()}.pdf"
            cover_letter_pdf_path = save_cover_letter_as_pdf(
                workflow_results['cover_letter'],
                cover_letter_filename
            )
            cover_letter_url = f"http://localhost:8000/download/{cover_letter_filename}"
            print(f"✅ Cover letter saved: {cover_letter_filename}")

        # 7. Generate LinkedIn/Email Message using AI
        try:
            from app.services.ai_service import llm
            from langchain_core.prompts import PromptTemplate
            
            linkedin_prompt = PromptTemplate(
                template="""Write a brief professional LinkedIn message (100-150 words).

JOB DESCRIPTION:
{jd_text}

Write a message expressing interest. Return ONLY JSON:
{{
    "subject_line": "subject here",
    "message_body": "Dear Hiring Manager,\\n\\n[message]\\n\\nBest regards"
}}""",
                input_variables=["jd_text"]
            )
            
            linkedin_result = (linkedin_prompt | llm).invoke({"jd_text": jd_text[:500]})
            
            # Parse JSON
            import re
            import json as json_lib
            text = linkedin_result.content if hasattr(linkedin_result, 'content') else str(linkedin_result)
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                data = json_lib.loads(match.group(0))
                linkedin_message = LinkedInDraft(
                    subject_line=data.get("subject_line", "Application for Position"),
                    message_body=data.get("message_body", "Message generated")
                )
            else:
                raise ValueError("No JSON")
        except Exception as e:
            print(f"⚠️ LinkedIn message failed: {e}")
            linkedin_message = LinkedInDraft(
                subject_line="Application for Position",
                message_body="Dear Hiring Manager,\n\nI am interested in this opportunity.\n\nBest regards"
            )

        # 8. Create legacy ResumeFeedback for compatibility
        feedback = ResumeFeedback(
            missing_skills=workflow_results['missing_skills'],
            detailed_edits=[edit.model_dump() if hasattr(edit, 'model_dump') else edit for edit in workflow_results['detailed_edits']],
            original_score=workflow_results['original_score'].score,
            optimized_score=workflow_results['optimized_score'].score,
            rewritten_content=workflow_results['rewritten_resume']
        )

        # 9. Extract job title from JD and save Activity to Database
        from app.services.ai_service import extract_company_job_title
        company, job_title = extract_company_job_title(jd_text)
        
        activity = models.ResumeActivity(
            user_id=current_user.id,
            original_filename=original_filename,
            modified_filename=resume_filename,
            original_score=workflow_results['original_score'].score,
            optimized_score=workflow_results['optimized_score'].score,
            download_link=resume_download_url,
            job_title=job_title  # Now populated with extracted job title
        )
        db.add(activity)
        db.commit()
        print(f"📝 Activity saved for user: {current_user.username}")

        # 10. Return Enhanced Response
        return EnhancedAgentResponse(
            feedback=feedback,
            message=linkedin_message,
            file_download_link=resume_download_url,
            cover_letter=workflow_results['cover_letter'],
            cover_letter_download_link=cover_letter_url,
            translated_resumes=workflow_results['translations'],
            ats_simulation=workflow_results['ats_simulation'].model_dump() if hasattr(workflow_results['ats_simulation'], 'model_dump') else workflow_results['ats_simulation']
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/generate-cover-letter", response_model=CoverLetterResponse)
async def generate_cover_letter_endpoint(
    activity_id: int = Form(...),
    jd_text: str = Form(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a cover letter for a previously processed resume.
    Requires authentication.
    """
    # Fetch activity and verify ownership
    activity = db.query(models.ResumeActivity).filter(
        models.ResumeActivity.id == activity_id,
        models.ResumeActivity.user_id == current_user.id
    ).first()
    
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found or access denied")
    
    try:
        # Extract keywords from JD
        keywords_result = (keyword_extraction_prompt | llm).invoke({"jd_text": jd_text})
        
        # Extract company and job title
        company, job_title = extract_company_job_title(jd_text)
        
        # Generate cover letter
        cover_letter_result = cover_letter_chain.invoke({
            "resume_text": "Resume content placeholder",
            "jd_text": jd_text,
            "keywords_json": str(keywords_result.content),
            "company_name": company,
            "job_title": job_title
        })
        
        cover_letter_content = cover_letter_result.content
        
        # Save as PDF
        cover_letter_filename = f"cover_letter_activity_{activity_id}_{uuid.uuid4()}.pdf"
        pdf_path = save_cover_letter_as_pdf(cover_letter_content, cover_letter_filename)
        
        if not pdf_path:
            raise HTTPException(status_code=500, detail="Failed to generate PDF")
        
        download_url = f"http://localhost:8000/download/{cover_letter_filename}"
        
        return CoverLetterResponse(
            content=cover_letter_content,
            pdf_download_link=download_url
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Cover letter generation failed: {str(e)}")


@router.post("/translate-resume", response_model=TranslationResponse)
async def translate_resume_endpoint(
    activity_id: int = Form(...),
    target_language: str = Form(...),
   current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Translate a previously optimized resume to another language.
    Requires authentication.
    """
    # Fetch activity and verify ownership
    activity = db.query(models.ResumeActivity).filter(
        models.ResumeActivity.id == activity_id,
        models.ResumeActivity.user_id == current_user.id
    ).first()
    
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found or access denied")
    
    try:
        # For now, use placeholder content
        resume_content = "Resume content to translate"
        
        # Translate
        translated_result = translation_chain.invoke({
            "resume_text": resume_content,
            "target_language": target_language
        })
        
        translated_content = translated_result.content
        
        # Save as PDF
        translation_filename = f"resume_{target_language}_{uuid.uuid4()}.pdf"
        pdf_path = save_translated_resume_as_pdf(
            translated_content, 
            target_language, 
            translation_filename
        )
        
        if not pdf_path:
            raise HTTPException(status_code=500, detail="Failed to generate translated PDF")
        
        download_url = f"http://localhost:8000/download/{translation_filename}"
        
        return TranslationResponse(
            language=target_language,
            content=translated_content,
            pdf_download_link=download_url
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
