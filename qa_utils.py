import json
import re
from typing import List, Dict, Union
from jsonschema import validate

# Define the expected schema
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "solution": {"type": "string"}
    },
    "required": ["question", "solution"]
}

def validate_response(qa_pair: Dict) -> bool:
    """Validate QA pair against schema"""
    try:
        validate(instance=qa_pair, schema=RESPONSE_SCHEMA)
        return True
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return False

def load_existing_qa(filepath: str) -> List[Dict]:
    """Load existing QA pairs from the JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def append_qa_to_file(filepath: str, qa_pair: Dict):
    """Append new QA pair to the JSON file"""
    qa_data = load_existing_qa(filepath)
    qa_data.append(qa_pair)
    with open(filepath, 'w') as f:
        json.dump(qa_data, f, indent=2)

def get_all_questions(qa_data: List[Dict]) -> List[str]:
    """Extract all questions from QA data"""
    return [qa['question'] for qa in qa_data]

def analyze_question_history(qa_data: List[Dict]) -> Dict:
    """Analyze question history to guide next question generation."""
    topics = {
        'calculus': ['derivative', 'integral', 'limit'],
        'algebra': ['matrix', 'polynomial', 'equation'],
        'geometry': ['triangle', 'circle', 'volume'],
        'probability': ['random', 'probability', 'distribution'],
        'number_theory': ['prime', 'divisor', 'modulo']
    }
    
    recent_topics = []
    for qa in qa_data[-5:]:  # Look at last 5 questions
        question = qa['question'].lower()
        for topic, keywords in topics.items():
            if any(keyword in question for keyword in keywords):
                recent_topics.append(topic)
                break
    
    return {
        'total_questions': len(qa_data),
        'recent_topics': recent_topics,
        'needs_attention': [topic for topic in topics.keys() 
                          if topic not in recent_topics]
    }

def clean_json_string(text: str) -> str:
    """Clean and format JSON string with mathematical notation."""
    # Remove any markdown code block markers
    text = re.sub(r'```json\s*|\s*```', '', text)
    
    # First pass: Handle LaTeX expressions
    def escape_latex(match):
        latex = match.group(0)
        # Keep $ signs but escape backslashes properly
        return latex.replace('\\', '\\\\').replace('"', '\\"')
    
    # Replace LaTeX expressions while preserving $ signs
    text = re.sub(r'\$(.*?)\$', escape_latex, text, flags=re.DOTALL)
    
    # Second pass: Handle special characters outside LaTeX
    text = text.replace('\n', '\\n')
    text = text.replace('"', '\\"')
    
    # Third pass: Fix double escapes and other issues
    text = re.sub(r'\\\\\\\\', r'\\\\', text)  # Fix quadruple backslashes
    text = re.sub(r'\\{3,}', r'\\\\', text)    # Fix triple or more backslashes
    text = re.sub(r'\\([^\\/"bfnrt])', r'\1', text)  # Remove invalid escapes
    
    return text

def fix_json_format(text: str) -> str:
    """Fix JSON formatting issues and ensure valid escape characters."""
    def try_parse_json(attempt: str) -> Union[Dict, None]:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            return None

    # First attempt: Try original text
    if result := try_parse_json(text):
        return text

    # Second attempt: Basic cleaning
    cleaned = clean_json_string(text)
    if result := try_parse_json(cleaned):
        return cleaned

    # Third attempt: Fix common LaTeX issues
    try:
        # Normalize LaTeX escapes
        cleaned = re.sub(r'\\\\([^\\])', r'\\\1', cleaned)
        # Fix consecutive backslashes
        cleaned = re.sub(r'\\{2,}', r'\\\\', cleaned)
        # Ensure proper quote escaping
        cleaned = re.sub(r'(?<!\\)"', '\\"', cleaned)
        
        if result := try_parse_json(cleaned):
            return cleaned

        # Final attempt: More aggressive cleaning
        cleaned = re.sub(r'\\([^\\/"bfnrt])', r'\1', cleaned)
        if result := try_parse_json(cleaned):
            return cleaned

    except Exception as e:
        print(f"Error during JSON cleaning: {str(e)}")

    # If all attempts fail, return original text
    print("Failed to fix JSON format")
    return text
