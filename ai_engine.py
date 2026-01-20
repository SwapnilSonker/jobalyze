import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas import ResumeFeedback, LinkedInDraft, ResumeEdit  # <--- Make sure ResumeEdit is imported

# 1. Setup LLM
llm = ChatGroq(
    temperature=0.0, 
    model_name="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Define Parsers
feedback_parser = PydanticOutputParser(pydantic_object=ResumeFeedback)
message_parser = PydanticOutputParser(pydantic_object=LinkedInDraft)

# --- PROMPT 1: ANALYSIS, REWRITE & LOGGING (UPDATED) ---
analysis_prompt = PromptTemplate(
    template="""
    You are a strict ATS (Applicant Tracking System) Scanner and Resume Editor. Output ONLY JSON.
    
    TASK: 
    1. Analyze the Resume vs Job Description (JD).
    2. Calculate a REAL compatibility score (0-100).
    3. Rewrite the resume to improve the score.
    4. Log every specific edit.
    
    RESUME CONTEXT:
    {resume_context}
    
    JOB DESCRIPTION:
    {jd_text}
    
    SCORING RULES (CRITICAL):
    - **Original Score:** Calculate strictly based on how many JD keywords are currently present in the resume. 
      (Example: If JD has 10 skills and Resume has 2, Score = 20). DO NOT RETURN 50. CALCULATE IT.
    - **Optimized Score:** Predicted score after your rewrites. This should be significantly higher (85-100).
    
    INSTRUCTIONS FOR EDITING:
    - Analyze and edit ALL sections of the resume: Summary, Skills, Experience, Education, Projects, Certifications, etc.
    - For Experience: Rewrite bullet points to include missing keywords using strong action verbs.
    - For Skills: Add missing technical skills, tools, and technologies mentioned in the JD.
    - For Summary/Objective: Align with JD requirements and add relevant keywords.
    - For Education/Projects/Certifications: Highlight relevant coursework, projects, or certifications that match JD.
    - Preserve company names, dates, and educational institutions.
    - Populate 'detailed_edits' for EVERY change across ALL sections.

    OUTPUT FORMAT (Strict JSON):
    {{
        "missing_skills": ["List actual missing skills here"],
        "suggested_changes": [],
        "detailed_edits": [
            {{
                "section": "Experience",
                "change_type": "Modification",
                "original_text": "Old line...",
                "new_text": "New line with keywords...",
                "keywords_added": ["Java", "AWS"]
            }}
        ],
        "original_score": <CALCULATED_INTEGER_0_TO_100>, 
        "optimized_score": <CALCULATED_INTEGER_80_TO_100>,
        "rewritten_content": "# Markdown Resume Content..."
    }}

    CRITICAL OUTPUT RULES:
    - Return ONLY valid JSON.
    - Start with {{ and end with }}.
    - Do NOT copy the example scores. Calculate them based on the input data.
    """,
    input_variables=["resume_context", "jd_text"]
)

# --- PROMPT 2: DETAILED EMAIL ---
draft_prompt = PromptTemplate(
    template="""
    You are a Senior Career Coach. Write a detailed, high-impact cold email.

    JOB DESCRIPTION:
    {jd_text}

    CANDIDATE ANALYSIS:
    {analysis_json}

    INSTRUCTIONS:
    1. Write a comprehensive email (Subject + Body).
    2. The body must be professional, detailing WHY the candidate fits.
    3. Use placeholders like [Recruiter Name].
    
    OUTPUT FORMAT (Strict JSON):
    {{
        "subject_line": "Catchy Subject Here",
        "message_body": "Dear Hiring Manager,\\n\\nI am writing to express my interest... (Full email content)"
    }}
    
    CRITICAL OUTPUT RULES:
    - Return ONLY valid JSON.
    - Start with {{ and end with }}.
    - No markdown code blocks.
    """,
    input_variables=["jd_text", "analysis_json"]
)

# 4. Build Raw Chains
analysis_chain_raw = analysis_prompt | llm 
draft_chain_raw = draft_prompt | llm 

# --- HELPER FUNCTION TO CLEAN OUTPUT ---
def clean_and_parse(raw_content, parser):
    """
    Strips markdown code blocks and parses JSON safely.
    """
    content = raw_content.strip()
    
    # Remove markdown code blocks if present
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    
    content = content.strip()
    
    # Parse using Pydantic
    return parser.parse(content)

def run_agent_workflow(resume_text: str, jd_text: str):
    # --- Step 1: Analyze AND Rewrite ---
    try:
        raw_response_1 = analysis_chain_raw.invoke({
            "resume_context": resume_text,
            "jd_text": jd_text
        })
        # Use helper to clean and parse
        feedback_obj = clean_and_parse(raw_response_1.content, feedback_parser)
        
    except Exception as e:
        print(f"Analysis Parsing Failed. Raw Output: {raw_response_1.content[:200]}...") 
        raise e

    # --- Step 2: Draft Detailed Email ---
    try:
        raw_response_2 = draft_chain_raw.invoke({
            "jd_text": jd_text,
            "analysis_json": feedback_obj.json()
        })
        # Use helper to clean and parse
        message_obj = clean_and_parse(raw_response_2.content, message_parser)
        
    except Exception as e:
        print(f"Draft Parsing Failed. Raw Output: {raw_response_2.content[:200]}...")
        raise e
    
    return feedback_obj, message_obj