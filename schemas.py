from pydantic import BaseModel, Field
from typing import List

class ResumeFeedback(BaseModel):
    missing_skills: List[str] = Field(description="Critical skills present in JD but missing in Resume")
    suggested_changes: List[str] = Field(description="Actionable bullet points to rewrite")
    match_score: int = Field(description="Fit score out of 100")

class LinkedInDraft(BaseModel):
    subject_line: str = Field(description="Professional and catchy subject")
    message_body: str = Field(description="The DM content, under 150 words")

class AgentResponse(BaseModel):
    feedback: ResumeFeedback
    message: LinkedInDraft