from pydantic import BaseModel, Field
from typing import List

# Isme ab purana score, naya score, aur naya resume teeno aayenge

class ResumeEdit(BaseModel):
    section: str = Field(description="The specific section name (e.g., 'Experience - Google', 'Skills', 'Summary')")
    change_type: str = Field(description="Type of edit: 'Modification', 'Addition', or 'Deletion'")
    original_text: str = Field(description="The exact text BEFORE the change (or 'N/A' if it's new)")
    new_text: str = Field(description="The exact text AFTER the change")
    keywords_added: List[str] = Field(description="List of specific keywords added in this line")

class ResumeFeedback(BaseModel):
    missing_skills: List[str] = Field(description="Critical skills present in JD but missing in Resume")
    # suggested_changes: List[str] = Field(description="Actionable bullet points to rewrite")
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