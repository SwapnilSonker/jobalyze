import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas import ResumeFeedback, LinkedInDraft

# 1. Setup LLM
llm = ChatGroq(
    temperature=0.3, 
    model_name="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Define Parsers
feedback_parser = PydanticOutputParser(pydantic_object=ResumeFeedback)
message_parser = PydanticOutputParser(pydantic_object=LinkedInDraft)

# 3. Define Prompts (UPDATED: VERY STRICT)
analysis_prompt = PromptTemplate(
    template="""
    You are a strict backend data processor. You output ONLY JSON.
    You do NOT talk. You do NOT explain. You do NOT use Markdown code blocks.
    
    TASK: Analyze the Resume against the Job Description (JD) and rewrite the resume.
    
    RESUME CONTEXT:
    {resume_context}
    
    JOB DESCRIPTION:
    {jd_text}
    
    INSTRUCTIONS:
    1. Identify missing keywords.
    2. Calculate 'original_score' (0-100).
    3. REWRITE the full resume in Markdown format. You MUST put the ENTIRE Markdown string INSIDE the JSON field 'rewritten_content'.
    4. Calculate 'optimized_score' (0-100).
    
    CRITICAL OUTPUT RULES:
    - Your response must start with {{ and end with }}.
    - Do not wrap the output in ```json ... ```.
    - Do not add any text before or after the JSON.
    
    {format_instructions}
    """,
    input_variables=["resume_context", "jd_text"],
    partial_variables={"format_instructions": feedback_parser.get_format_instructions()}
)

draft_prompt = PromptTemplate(
    template="""
    You are a master networker. Write a LinkedIn connection message.
    
    JOB DESCRIPTION:
    {jd_text}
    
    CANDIDATE STRENGTHS (Based on optimized resume):
    {analysis_json}
    
    Write a short, professional, and personalized DM (under 150 words) highlighting the fit.
    
    OUTPUT RULES:
    - Return ONLY valid JSON.
    - No markdown formatting.
    
    {format_instructions}
    """,
    input_variables=["jd_text", "analysis_json"],
    partial_variables={"format_instructions": message_parser.get_format_instructions()}
)

# 4. Build Chains (Note: We removed the parser from analysis_chain to do manual cleaning)
analysis_chain_raw = analysis_prompt | llm 
draft_chain = draft_prompt | llm | message_parser

def run_agent_workflow(resume_text: str, jd_text: str):
    # --- Step 1: Analyze AND Rewrite (With Safety Cleaning) ---
    try:
        # Get raw string output from LLM
        raw_response = analysis_chain_raw.invoke({
            "resume_context": resume_text,
            "jd_text": jd_text
        })
        
        # CLEANUP LOGIC: Remove ```json and ``` if present
        content = raw_response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        # Now Parse manually
        feedback_obj = feedback_parser.parse(content)
        
    except Exception as e:
        print(f"Analysis Parsing Failed. Raw Content: {raw_response.content[:500]}...") # Print first 500 chars for debug
        raise e

    # --- Step 2: Draft Message ---
    # Draft chain usually works fine as it's shorter, but error handling is good
    try:
        message_obj = draft_chain.invoke({
            "jd_text": jd_text,
            "analysis_json": feedback_obj.json()
        })
    except Exception as e:
        print(f"Draft Parsing Failed.")
        raise e
    
    return feedback_obj, message_obj