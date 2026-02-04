from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# --- Resume Edit and Feedback Schemas ---

class ResumeEdit(BaseModel):
    section: str = Field(description="The specific section name (e.g., 'Experience - Google', 'Skills', 'Summary')")
    change_type: str = Field(description="Type of edit: 'Modification', 'Addition', or 'Deletion')")
    original_text: str = Field(description="The exact text BEFORE the change (or 'N/A' if it's new)")
    new_text: str = Field(description="The exact text AFTER the change")
    keywords_added: List[str] = Field(description="List of specific keywords added in this line")


class ResumeFeedback(BaseModel):
    missing_skills: List[str] = Field(description="Critical skills present in JD but missing in Resume")
    detailed_edits: List[ResumeEdit] = Field(description="List of specific line-by-line edits made to the resume")
    original_score: int = Field(description="Fit score (0-100) of the ORIGINAL resume")
    optimized_score: int = Field(description="Projected score (0-100) after applying the changes")
    rewritten_content: str = Field(description="The FULL optimized resume content in Markdown format with keywords integrated.")


class LinkedInDraft(BaseModel):
    subject_line: str = Field(description="Professional and catchy subject")
    message_body: str = Field(description="The DM content, under 150 words")


class AgentResponse(BaseModel):
    feedback: ResumeFeedback
    message: LinkedInDraft
    file_download_link: str = Field(description="URL to download the updated resume PDF")


# --- Enhanced Multi-Step Workflow Schemas ---

class KeywordExtractionResult(BaseModel):
    """Step 0: Extracted keywords and requirements from JD"""
    hard_skills: List[str] = Field(description="Technical hard skills required")
    soft_skills: List[str] = Field(description="Soft skills and competencies")
    required_experience: List[str] = Field(description="Years of experience, specific roles")
    certifications: List[str] = Field(description="Required certifications or degrees")
    tools_technologies: List[str] = Field(description="Specific tools, frameworks, technologies")
    industry_keywords: List[str] = Field(description="Industry-specific terms and jargon")


class OriginalScoreResult(BaseModel):
    """Step 1: Original resume score and gap analysis"""
    score: int = Field(ge=0, le=100, description="Original ATS compatibility score")
    missing_keywords: List[str] = Field(description="Keywords present in JD but missing in resume")
    weak_sections: List[str] = Field(description="Resume sections that need improvement")
    reasoning: str = Field(description="Detailed explanation of the score")


class OptimizedScoreResult(BaseModel):
    """Step 3: Optimized score with edit details"""
    score: int = Field(ge=0, le=100, description="Optimized ATS compatibility score")
    detailed_edits: List[ResumeEdit] = Field(description="All changes made to the resume")
    improvements_summary: str = Field(description="Summary of key improvements")


class ATSSimulationResult(BaseModel):
    """ATS parsing simulation results"""
    pass_fail: str = Field(description="PASS or FAIL based on ATS parsing")
    issues_list: List[str] = Field(description="List of detected parsing issues")
    explanation: str = Field(description="Detailed explanation of ATS compatibility")


class CoverLetterResponse(BaseModel):
    """Response for cover letter generation"""
    content: str = Field(description="Markdown formatted cover letter")
    pdf_download_link: Optional[str] = Field(default=None, description="URL to download PDF")


class TranslationResponse(BaseModel):
    """Response for resume translation"""
    language: str = Field(description="Target language")
    content: str = Field(description="Translated resume in Markdown")
    pdf_download_link: Optional[str] = Field(default=None, description="URL to download PDF")


class EnhancedAgentResponse(BaseModel):
    """Enhanced response with all workflow outputs"""
    feedback: ResumeFeedback
    message: LinkedInDraft
    file_download_link: str = Field(description="URL to download the updated resume PDF")
    cover_letter: Optional[str] = Field(default=None, description="Generated cover letter content")
    cover_letter_download_link: Optional[str] = Field(default=None, description="Cover letter PDF link")
    translated_resumes: Optional[Dict[str, str]] = Field(default_factory=dict, description="Language -> markdown content")
    ats_simulation: Optional[ATSSimulationResult] = Field(default=None, description="ATS parsing simulation results")
