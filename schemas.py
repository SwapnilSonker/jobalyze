from pydantic import BaseModel, Field
from typing import List

# Isme ab purana score, naya score, aur naya resume teeno aayenge
class ResumeFeedback(BaseModel):
    missing_skills: List[str] = Field(description="Critical skills present in JD but missing in Resume")
    suggested_changes: List[str] = Field(description="Actionable bullet points to rewrite")
    original_score: int = Field(description="Fit score (0-100) of the ORIGINAL resume")
    optimized_score: int = Field(description="Projected score (0-100) after applying the changes")
    rewritten_content: str = Field(description="The FULL optimized resume content in Markdown format with keywords integrated.")

class LinkedInDraft(BaseModel):
    subject_line: str = Field(description="Professional and catchy subject")
    message_body: str = Field(description="The DM content, under 150 words")

class AgentResponse(BaseModel):
    feedback: ResumeFeedback
    message: LinkedInDraft