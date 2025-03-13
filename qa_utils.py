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
    
    # Handle LaTeX style math notation
    text = text.replace('\\\\', '\\')  # Fix double escapes
    text = text.replace('\n', '\\n')   # Properly escape newlines
    
    # Fix common mathematical symbols
    math_symbols = {
        '→': '\\to',
        '∫': '\\int',
        '∞': '\\infty',
        '≤': '\\leq',
        '≥': '\\geq',
        'π': '\\pi',
        '∈': '\\in',
        '⊆': '\\subseteq',
        '∑': '\\sum',
        '∏': '\\prod',
        '√': '\\sqrt'
    }
    
    for symbol, latex in math_symbols.items():
        text = text.replace(symbol, latex)
    
    # Fix any unescaped quotes within mathematical expressions
    text = re.sub(r'(?<!\\)"', '\\"', text)
    
    return text

def fix_json_format(text: str) -> str:
    """Fix JSON formatting issues."""
    try:
        # Try to parse as is first
        json.loads(text)
        return text
    except json.JSONDecodeError:
        # Clean and try to fix the JSON
        cleaned = clean_json_string(text)
        try:
            # Verify the cleaned version is valid JSON
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError as e:
            print(f"Failed to fix JSON format: {str(e)}")
            return text
