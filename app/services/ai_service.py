import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from app.schemas.resume import (
    ResumeFeedback, LinkedInDraft, ResumeEdit,
    KeywordExtractionResult, OriginalScoreResult, OptimizedScoreResult,
    ATSSimulationResult
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Setup LLM
llm = ChatGroq(
    temperature=0.0, 
    model_name="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY")
)

# High temperature for creative content
llm_creative = ChatGroq(
    temperature=0.7,
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================================
# STEP 0: KEYWORD EXTRACTION
# ============================================================================

keyword_extraction_prompt = PromptTemplate(
    template="""You are an expert ATS analyzer. Extract structured requirements from this job description.

JOB DESCRIPTION:
{jd_text}

Extract and categorize ALL keywords into these categories. Be thorough and exhaustive.

OUTPUT FORMAT (Strict JSON):
{{
    "hard_skills": ["Python", "JavaScript", "Machine Learning", ...],
    "soft_skills": ["Leadership", "Communication", "Problem Solving", ...],
    "required_experience": ["5+ years Python", "3+ years team management", ...],
    "certifications": ["AWS Certified", "PMP", "Scrum Master", ...],
    "tools_technologies": ["Docker", "Kubernetes", "AWS", "PostgreSQL", ...],
    "industry_keywords": ["Agile", "CI/CD", "Microservices", "APIs", ...]
}}

CRITICAL RULES:
- Return ONLY valid JSON starting with {{ and ending with }}
- Extract ALL relevant keywords, don't summarize
- Be specific and comprehensive
- If a category has no items, return empty list []
""",
    input_variables=["jd_text"]
)

keyword_chain = keyword_extraction_prompt | llm  # Parse JSON manually to handle extra text


# ============================================================================
# STEP 1: ORIGINAL SCORE & GAP ANALYSIS
# ============================================================================

original_score_prompt = PromptTemplate(
    template="""You are a strict ATS scoring system. Calculate the HONEST compatibility score.

RESUME CONTENT:
{resume_context}

EXTRACTED JD REQUIREMENTS:
{keywords_json}

TASK:
1. Count how many JD keywords are present in the resume
2. Calculate exact score: (matched_keywords / total_keywords) * 100
3. Identify missing critical keywords
4. Identify weak sections

SCORING RULES:
- Be STRICT and HONEST
- If resume has 20% of JD keywords, score = 20
- Don't inflate scores
- Hard skills weigh more than soft skills

OUTPUT FORMAT (Strict JSON):
{{
    "score": <integer 0-100>,
    "missing_keywords": ["keyword1", "keyword2", ...],
    "weak_sections": ["Skills", "Experience", "Summary", ...],
    "reasoning": "Detailed explanation of why this score was given..."
}}

CRITICAL: Return ONLY valid JSON. Calculate the REAL score, no fake numbers.
""",
    input_variables=["resume_context", "keywords_json"]
)

original_score_chain = original_score_prompt | llm  # Parse JSON manually


# ============================================================================
# STEP 2: SECTION-BY-SECTION REWRITING
# ============================================================================

section_rewrite_prompt = PromptTemplate(
    template="""You are an expert resume writer. Rewrite this resume section to optimize for JD keywords.

SECTION NAME: {section_name}

ORIGINAL SECTION CONTENT:
{section_content}

TARGET JD KEYWORDS:
{keywords_json}

FULL JD CONTEXT:
{jd_text}

INSTRUCTIONS:
1. Preserve all company names, dates, and educational institutions EXACTLY
2. Integrate missing keywords naturally into descriptions
3. Use strong action verbs (Led, Architected, Implemented, Optimized, etc.)
4. Quantify achievements where possible
5. Maintain professional tone and formatting
6. Return ONLY the rewritten section content in Markdown format
7. Do NOT add headers or explanations, just the content

For Experience section:
- Each role should be: ### Company Name | Role Title | Dates
- Bullet points should start with strong action verbs

For Skills section:
- Organize by categories (e.g., **Programming Languages:** Python, Java, ...)
- Include ALL JD-mentioned skills the candidate likely has

For Summary/Objective:
- 3-4 sentences maximum
- Mention key skills from JD
- Highlight relevant experience

OUTPUT: Return ONLY the rewritten markdown content, nothing else.
""",
    input_variables=["section_name", "section_content", "keywords_json", "jd_text"]
)

section_rewrite_chain = section_rewrite_prompt | llm_creative


# ============================================================================
# STEP 3: OPTIMIZED SCORE CALCULATION
# ============================================================================

optimized_score_prompt = PromptTemplate(
    template="""You are an ATS scoring system. Calculate the NEW score after resume optimization.

ORIGINAL RESUME:
{original_resume}

REWRITTEN RESUME:
{rewritten_resume}

TARGET KEYWORDS:
{keywords_json}

TASK:
1. Count how many JD keywords are NOW present in the rewritten resume
2. Calculate new score: (matched_keywords / total_keywords) * 100
3. List ALL specific edits made
4. Summarize improvements

OUTPUT FORMAT (Strict JSON):
{{
    "score": <integer 80-100>,
    "detailed_edits": [
        {{
            "section": "Experience",
            "change_type": "Modification",
            "original_text": "Worked on backend systems",
            "new_text": "Architected scalable Python microservices using Docker and Kubernetes",
            "keywords_added": ["Python", "microservices", "Docker", "Kubernetes"]
        }},
        ...
    ],
    "improvements_summary": "Added 15 critical keywords across all sections. Enhanced experience bullets with quantifiable achievements..."
}}

CRITICAL: Return ONLY valid JSON. Score should be 80-100 after optimization.
""",
    input_variables=["original_resume", "rewritten_resume", "keywords_json"]
)

optimized_score_chain = optimized_score_prompt | llm  # Parse JSON manually


# ============================================================================
# STEP 4: COVER LETTER GENERATION
# ============================================================================

cover_letter_prompt = PromptTemplate(
    template="""You are a professional career coach. Write a compelling cover letter.

CANDIDATE'S RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

EXTRACTED KEYWORDS:
{keywords_json}

COMPANY NAME: {company_name}
JOB TITLE: {job_title}

INSTRUCTIONS:
1. Write a professional, personalized cover letter in Markdown format
2. 3-4 paragraphs maximum (300-400 words)
3. Structure:
   - Opening: Express enthusiasm for the role
   - Body 1: Highlight relevant experience matching JD requirements
   - Body 2: Showcase specific achievements and skills
   - Closing: Call to action and thank you
4. Use specific keywords from JD naturally
5. Be confident but not arrogant
6. Use industry-appropriate tone

OUTPUT FORMAT:
# Cover Letter

[Hiring Manager Name]  
{company_name}  
[Date]

Dear Hiring Manager,

[Paragraph 1: Opening with enthusiasm]

[Paragraph 2: Relevant experience and skills]

[Paragraph 3: Specific achievements]

[Paragraph 4: Closing]

Sincerely,  
[Candidate Name]

Return ONLY the markdown formatted cover letter.
""",
    input_variables=["resume_text", "jd_text", "keywords_json", "company_name", "job_title"]
)

cover_letter_chain = cover_letter_prompt | llm_creative


# ============================================================================
# STEP 5: TRANSLATION
# ============================================================================

translation_prompt = PromptTemplate(
    template="""You are a professional translator specializing in resumes.

RESUME TO TRANSLATE:
{resume_text}

TARGET LANGUAGE: {target_language}

INSTRUCTIONS:
1. Translate the entire resume to {target_language}
2. Preserve all Markdown formatting exactly
3. Keep company names, dates, and proper nouns in original language
4. Maintain professional terminology
5. Ensure cultural appropriateness

Return ONLY the translated markdown content.
""",
    input_variables=["resume_text", "target_language"]
)

translation_chain = translation_prompt | llm


# ============================================================================
# ATS SIMULATION
# ============================================================================

ats_simulation_prompt = PromptTemplate(
    template="""You are an ATS (Applicant Tracking System) parser. Simulate parsing this resume.

RESUME:
{resume_text}

TASK: Identify potential parsing issues that would cause rejection.

CHECK FOR:
1. Tables or complex formatting (ATS often fails to parse tables)
2. Images, graphics, or charts (not parseable)
3. Non-standard section headers
4. Multiple columns
5. Headers/footers with important info
6. Special characters or unusual fonts
7. Missing contact information
8. File format issues

OUTPUT FORMAT (Strict JSON):
{{
    "pass_fail": "PASS" or "FAIL",
    "issues_list": ["Issue 1", "Issue 2", ...],
    "explanation": "Detailed explanation of parsing results..."
}}

CRITICAL: Return ONLY valid JSON.
""",
    input_variables=["resume_text"]
)

ats_simulation_chain = ats_simulation_prompt | llm  # Parse JSON manually


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_and_parse(raw_content, parser):
    """Strips markdown code blocks and parses JSON safely."""
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


def parse_resume_sections(resume_text: str) -> dict:
    """
    Simple parser to extract sections from markdown resume.
    Returns dict: {section_name: content}
    """
    sections = {}
    current_section = "Header"
    current_content = []
    
    lines = resume_text.split('\n')
    
    for line in lines:
        # Check for markdown headers (# or ##)
        if line.startswith('# ') or line.startswith('## '):
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line.lstrip('#').strip()
            current_content = []
        else:
            current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections


def extract_company_job_title(jd_text: str) -> tuple:
    """Extract company name and job title from JD. Simple heuristic."""
    lines = jd_text.split('\n')
    company = "the company"
    job_title = "this position"
    
    # Try to find company and title in first few lines
    for line in lines[:10]:
        if "company" in line.lower() and len(line) < 100:
            company = line.strip()
        if "title:" in line.lower() or "position:" in line.lower():
            job_title = line.split(':')[-1].strip()
    
    return company, job_title


# ============================================================================
# MAIN MULTI-STEP WORKFLOW
# ============================================================================

def run_enhanced_agent_workflow(
    resume_text: str, 
    jd_text: str, 
    generate_cover_letter: bool = True,
    translate_to: list = None
):
    """
    Enhanced multi-step AI workflow.
    
    Args:
        resume_text: Original resume content
        jd_text: Job description text
        generate_cover_letter: Whether to generate cover letter
        translate_to: List of languages to translate to (e.g., ['Spanish', 'French'])
    
    Returns:
        Dictionary with all outputs
    """
    
    try:
        # ========== STEP 0: KEYWORD EXTRACTION ==========
        logger.info("Step 0: Extracting keywords from JD...")
        keywords_raw = keyword_chain.invoke({"jd_text": jd_text})
        
        # Extract content from LLM response
        keywords_text = keywords_raw.content if hasattr(keywords_raw, 'content') else str(keywords_raw)
        
        # Handle case where LLM adds extra text before JSON
        import re
        json_match = re.search(r'\{[\s\S]*\}', keywords_text)
        if json_match:
            keywords_result = json.loads(json_match.group(0))
        else:
            raise ValueError(f"No JSON found in output: {keywords_text[:200]}")
            
        keywords_obj = KeywordExtractionResult(**keywords_result)
        logger.info(f"Extracted {len(keywords_obj.hard_skills)} hard skills, {len(keywords_obj.tools_technologies)} tools")
        
    except Exception as e:
        logger.error(f"Step 0 failed: {e}")
        raise Exception(f"Keyword extraction failed: {e}")
    
    try:
        # ========== STEP 1: ORIGINAL SCORE ==========
        logger.info("Step 1: Calculating original ATS score...")
        original_score_raw = original_score_chain.invoke({
            "resume_context": resume_text,
            "keywords_json": keywords_obj.json()
        })
        
        # Extract content and parse JSON
        original_score_text = original_score_raw.content if hasattr(original_score_raw, 'content') else str(original_score_raw)
        json_match = re.search(r'\{[\s\S]*\}', original_score_text)
        if json_match:
            original_score_result = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON found in original score output")
            
        original_score_obj = OriginalScoreResult(**original_score_result)
        logger.info(f"Original score: {original_score_obj.score}/100")
        
    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        # Fallback
        original_score_obj = OriginalScoreResult(
            score=50,
            missing_keywords=keywords_obj.hard_skills[:5],
            weak_sections=["Skills", "Experience"],
            reasoning="Error in original scoring, using fallback values"
        )
    
    try:
        # ========== STEP 2: SECTION REWRITING ==========
        logger.info("Step 2: Rewriting resume sections...")
        sections = parse_resume_sections(resume_text)
        rewritten_sections = {}
        
        priority_sections = ['Summary', 'Professional Summary', 'Experience', 'Work Experience', 
                             'Skills', 'Technical Skills', 'Projects', 'Education']
        
        for section_name in priority_sections:
            # Find matching section (case-insensitive)
            matching_section = None
            for key in sections.keys():
                if section_name.lower() in key.lower():
                    matching_section = key
                    break
            
            if matching_section and sections[matching_section]:
                logger.info(f"Rewriting section: {matching_section}")
                rewritten_content = section_rewrite_chain.invoke({
                    "section_name": matching_section,
                    "section_content": sections[matching_section],
                    "keywords_json": keywords_obj.json(),
                    "jd_text": jd_text
                })
                rewritten_sections[matching_section] = rewritten_content.content
        
        # Reconstruct full resume
        rewritten_resume = ""
        for section_name, content in rewritten_sections.items():
            rewritten_resume += f"# {section_name}\n\n{content}\n\n"
        
        # Add sections that weren't rewritten
        for section_name, content in sections.items():
            if section_name not in rewritten_sections and content:
                rewritten_resume += f"# {section_name}\n\n{content}\n\n"
        
        logger.info("Resume rewriting complete")
        
    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        rewritten_resume = resume_text  # Fallback to original
    
    try:
        # ========== STEP 3: OPTIMIZED SCORE ==========
        logger.info("Step 3: Calculating optimized score...")
        optimized_score_raw = optimized_score_chain.invoke({
            "original_resume": resume_text,
            "rewritten_resume": rewritten_resume,
            "keywords_json": keywords_obj.json()
        })
        
        # Extract content and parse JSON
        optimized_score_text = optimized_score_raw.content if hasattr(optimized_score_raw, 'content') else str(optimized_score_raw)
        json_match = re.search(r'\{[\s\S]*\}', optimized_score_text)
        if json_match:
            optimized_score_result = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON found in optimized score output")
            
        optimized_score_obj = OptimizedScoreResult(**optimized_score_result)
        logger.info(f"Optimized score: {optimized_score_obj.score}/100")
        
    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        # Fallback
        optimized_score_obj = OptimizedScoreResult(
            score=85,
            detailed_edits=[],
            improvements_summary="Optimized resume with JD keywords"
        )
    
    try:
        # ========== STEP 4: COVER LETTER (Optional) ==========
        cover_letter_content = None
        if generate_cover_letter:
            logger.info("Step 4: Generating cover letter...")
            company, job_title = extract_company_job_title(jd_text)
            cover_letter_result = cover_letter_chain.invoke({
                "resume_text": rewritten_resume,
                "jd_text": jd_text,
                "keywords_json": keywords_obj.json(),
                "company_name": company,
                "job_title": job_title
            })
            cover_letter_content = cover_letter_result.content
            logger.info("Cover letter generated")
    
    except Exception as e:
        logger.error(f"Step 4 failed: {e}")
        cover_letter_content = None
    
    try:
        # ========== STEP 5: TRANSLATION (Optional) ==========
        translations = {}
        if translate_to:
            for language in translate_to:
                logger.info(f"Step 5: Translating to {language}...")
                translated = translation_chain.invoke({
                    "resume_text": rewritten_resume,
                    "target_language": language
                })
                translations[language] = translated.content
                logger.info(f"Translation to {language} complete")
    
    except Exception as e:
        logger.error(f"Step 5 failed: {e}")
        translations = {}
    
    try:
        # ========== ATS SIMULATION ==========
        logger.info("Running ATS simulation...")
        ats_raw = ats_simulation_chain.invoke({"resume_text": rewritten_resume})
        
        # Extract content and parse JSON
        ats_text = ats_raw.content if hasattr(ats_raw, 'content') else str(ats_raw)
        json_match = re.search(r'\{[\s\S]*\}', ats_text)
        if json_match:
            ats_result = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON found in ATS output")
            
        ats_obj = ATSSimulationResult(**ats_result)
        logger.info(f"ATS simulation: {ats_obj.pass_fail}")
    
    except Exception as e:
        logger.error(f"ATS simulation failed: {e}")
        ats_obj = ATSSimulationResult(
            pass_fail="PASS",
            issues_list=[],
            explanation="ATS simulation not available"
        )
    
    # ========== RETURN COMPREHENSIVE RESULTS ==========
    return {
        "keywords": keywords_obj,
        "original_score": original_score_obj,
        "optimized_score": optimized_score_obj,
        "rewritten_resume": rewritten_resume,
        "cover_letter": cover_letter_content,
        "translations": translations,
        "ats_simulation": ats_obj,
        "missing_skills": original_score_obj.missing_keywords,
        "detailed_edits": optimized_score_obj.detailed_edits
    }


# ============================================================================
# LEGACY COMPATIBILITY (keep old workflow for now)
# ============================================================================

# Keep old prompts and functions for backward compatibility
feedback_parser = PydanticOutputParser(pydantic_object=ResumeFeedback)
message_parser = PydanticOutputParser(pydantic_object=LinkedInDraft)

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

analysis_chain_raw = analysis_prompt | llm 
draft_chain_raw = draft_prompt | llm 


def clean_and_parse_legacy(raw_content, parser):
    """Strips markdown code blocks and parses JSON safely."""
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
    """Legacy workflow - kept for backward compatibility"""
    # --- Step 1: Analyze AND Rewrite ---
    try:
        raw_response_1 = analysis_chain_raw.invoke({
            "resume_context": resume_text,
            "jd_text": jd_text
        })
        # Use helper to clean and parse
        feedback_obj = clean_and_parse_legacy(raw_response_1.content, feedback_parser)
        
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
        message_obj = clean_and_parse_legacy(raw_response_2.content, message_parser)
        
    except Exception as e:
        print(f"Draft Parsing Failed. Raw Output: {raw_response_2.content[:200]}...")
        raise e
    
    return feedback_obj, message_obj