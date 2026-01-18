import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas import ResumeFeedback, LinkedInDraft

# 1. Setup LLM
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant",  # Or 'mixtral-8x7b-32768'
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Define Parsers
feedback_parser = PydanticOutputParser(pydantic_object=ResumeFeedback)
message_parser = PydanticOutputParser(pydantic_object=LinkedInDraft)

# 3. Define Prompts with Format Instructions
analysis_prompt = PromptTemplate(
    template="""
    You are an expert technical recruiter. Analyze the following resume context against the Job Description.
    
    RESUME CONTEXT:
    {resume_context}
    
    JOB DESCRIPTION:
    {jd_text}
    
    Identify missing keywords, suggest improvements, and give a match score.
    {format_instructions}
    """,
    input_variables=["resume_context", "jd_text"],
    partial_variables={"format_instructions": feedback_parser.get_format_instructions()}
)

draft_prompt = PromptTemplate(
    template="""
    You are a master networker. Write a LinkedIn connection message based on the analysis.
    
    JOB DESCRIPTION:
    {jd_text}
    
    ANALYSIS SUMMARY:
    {analysis_json}
    
    Write a short, professional, and personalized DM (under 150 words).
    {format_instructions}
    """,
    input_variables=["jd_text", "analysis_json"],
    partial_variables={"format_instructions": message_parser.get_format_instructions()}
)

# 4. Build Chains (LCEL)
analysis_chain = analysis_prompt | llm | feedback_parser
draft_chain = draft_prompt | llm | message_parser

def run_agent_workflow(resume_text: str, jd_text: str):
    # Step 1: Analyze Resume
    # Note: Hum poora text bhej rahe hain agar chota hai, ya RAG context agar bada hai.
    # Yahan simplicity ke liye direct text pass kar rahe hain, lekin RAG integration main.py me hoga.
    
    feedback_obj = analysis_chain.invoke({
        "resume_context": resume_text,
        "jd_text": jd_text
    })
    
    # Step 2: Draft Message using the feedback
    # Hum feedback object ko dict mein convert karke pass karenge
    message_obj = draft_chain.invoke({
        "jd_text": jd_text,
        "analysis_json": feedback_obj.json()
    })
    
    return feedback_obj, message_obj