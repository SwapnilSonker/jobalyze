import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas import ResumeFeedback, LinkedInDraft, ResumeEdit  # <--- Make sure ResumeEdit is imported

# 1. Setup LLM
llm = ChatGroq(
    temperature=0.4, 
    model_name="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Define Parsers
feedback_parser = PydanticOutputParser(pydantic_object=ResumeFeedback)
message_parser = PydanticOutputParser(pydantic_object=LinkedInDraft)

# --- PROMPT 1: ANALYSIS, REWRITE & LOGGING (UPDATED) ---
analysis_prompt = PromptTemplate(
    template="""
    You are a strict backend data processor. Output ONLY JSON.
    
    TASK: Rewrite the resume to target the Job Description (JD) while preserving the truth, and LOG every specific edit made.
    
    RESUME CONTEXT:
    {resume_context}
    
    JOB DESCRIPTION:
    {jd_text}
    
    INSTRUCTIONS:
    1. **Rewriting:** - Rewrite bullet points to naturally include JD keywords using strong action verbs.
       - Preserve Facts (Company Names, Dates, Titles must remain exact).
       - Put the FULL rewritten markdown in 'rewritten_content'.
    
    2. **Detailed Logging (Critical):**
       - You MUST populate the 'detailed_edits' list.
       - For every bullet point or sentence you modify/add, create an entry explaining exactly what changed.
       - 'section': Where is this change? (e.g., "Experience - Google", "Skills").
       - 'change_type': "Modification", "Addition", or "Deletion".
       - 'original_text': The text BEFORE the change (or "N/A" if new).
       - 'new_text': The text AFTER the change.
       - 'keywords_added': List of JD keywords that justified this change.
    
    OUTPUT FORMAT (Strict JSON):
    {{
        "missing_skills": ["skill1", "skill2"],
        "detailed_edits": [
            {{
                "section": "Experience - Company A",
                "change_type": "Modification",
                "original_text": "Worked on backend.",
                "new_text": "Engineered scalable backend APIs using Python and AWS.",
                "keywords_added": ["Scalable", "AWS", "Python"]
            }}
        ],
        "original_score": 50,
        "optimized_score": 90,
        "rewritten_content": "# Name\\n## Experience..."
    }}

    CRITICAL OUTPUT RULES:
    - Return ONLY valid JSON.
    - Start with {{ and end with }}.
    - No markdown code blocks (```json).
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