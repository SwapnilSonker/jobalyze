import re
import json

def extract_json_from_llm_output(text: str) -> dict:
    """
    Robustly extract JSON from LLM output that may contain extra text.
    Handles cases where LLM adds explanatory text before/after the JSON.
    """
    text = text.strip()
    
    # Try to find JSON object using regex (find everything between first { and last })
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Last resort: try parsing the whole thing
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not extract valid JSON from LLM output. Text was: {text[:500]}...")
