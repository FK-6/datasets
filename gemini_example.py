import os
from google import genai
from google.genai import types  # Add import
import sys
from dotenv import load_dotenv
from qa_utils import load_existing_qa, append_qa_to_file, get_all_questions, validate_response
import time
import json
from qa_utils import fix_json_format, clean_json_string

# Load environment variables from .env file
load_dotenv()

def create_gemini_client():
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error creating client: {str(e)}", file=sys.stderr)
        return None

def get_next_question(client, previous_questions):
    # Format previous questions for better context
    formatted_questions = "\n".join([
        f"{i+1}. {q}" for i, q in enumerate(previous_questions[-10:])  # Show last 10 questions
    ])
    
    context = f"""You are a mathematics question generator specializing in creating diverse and challenging questions.

PREVIOUS QUESTIONS (DO NOT REPEAT THESE):
{formatted_questions}

REQUIREMENTS:
1. Generate ONE new challenging mathematics question
2. Ensure the question is completely different from any previously asked questions
3. Vary the difficulty level: alternate between basic, intermediate, advanced, and PhD-level
4. Cover different areas of mathematics (calculus, algebra, topology, number theory, etc.)
5. Do not repeat topics from the last 3 questions
6. Make questions clear and well-defined

EXAMPLE FORMATS:
- "Prove that..."
- "Find the value of..."
- "Show that..."
- "Calculate the..."
- "Determine if..."

Generate only the question text, without any additional context or formatting."""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[context],
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.5  # Increased for more variety
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating question: {str(e)}", file=sys.stderr)
        return None

def get_answer(client, question):
    context = """You are a mathematics and science expert. Provide your response in the following exact JSON format:
{{
  "question": "What is the derivative of f(x) = sin(x)?",
  "solution": "To find the derivative of f(x) = sin(x), we use the standard derivative rule for sine. The derivative is f'(x) = cos(x). This is one of the fundamental derivatives in calculus."
}}
    
QUESTION: {0}

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the JSON object with question and solution fields
- Include the complete question text exactly as provided
- Give a detailed solution that explains the steps clearly
- Ensure the solution is accurate and well-explained
- Use proper mathematical notation and terminology
- Provide a complete and self-contained solution
- Avoid any unnecessary information or extra text
- Use exactly the format shown in the example above
- Ensure proper JSON formatting with indentation
- Do not add any text before or after the JSON
- use proper json escape characters fot latex and other areas to escape parse error leverage double escaping 
""".format(question)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[context],  # Pass as list
            config=types.GenerateContentConfig(
                max_output_tokens=8100,
                temperature=0.05
            )
        )
        answer = response.text.strip()
        
        # Debug print
        print("\nRaw Response:")
        print("-" * 40)
        print(answer)
        print("-" * 40)
        
        # Clean the response by removing JSON markers if present
        cleaned_answer = answer
        if answer.startswith('```json'):
            cleaned_answer = answer.split('```json')[1]
        if '```' in cleaned_answer:
            cleaned_answer = cleaned_answer.split('```')[0]
        cleaned_answer = cleaned_answer.strip()
        
        # Fix JSON formatting
        cleaned_answer = fix_json_format(cleaned_answer)
        
        # Parse JSON response
        try:
            qa_pair = json.loads(cleaned_answer)
            # Validate response format
            if not validate_response(qa_pair):
                raise ValueError("Response does not match expected format")
            return qa_pair
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response: {str(e)}")
            print(f"Cleaned response: {cleaned_answer}")
            return None
            
    except Exception as e:
        print(f"Error generating answer: {str(e)}", file=sys.stderr)
        return None

def main():
    qa_file = "math_dataset.json"
    questioner = create_gemini_client()
    answerer = create_gemini_client()
    
    if not questioner or not answerer:
        return
    
    try:
        qa_data = load_existing_qa(qa_file)
    except FileNotFoundError:
        # Initialize empty dataset if file doesn't exist
        qa_data = []
        with open(qa_file, 'w') as f:
            json.dump(qa_data, f, indent=2)
    
    while True:
        previous_questions = get_all_questions(qa_data)
        
        # Get next question
        question = get_next_question(questioner, previous_questions)
        if not question:
            print("Failed to generate question")
            break
            
        # Check if question is too similar to previous questions
        if any(calculate_similarity(question, prev_q) > 0.8 for prev_q in previous_questions):
            print("Question too similar to previous ones, generating new question...")
            continue
            
        print(f"\nNew Question: {question}")
        
        # Get answer
        qa_pair = get_answer(answerer, question)
        if not qa_pair:
            print("Failed to generate answer")
            continue
            
        # Append to dataset
        append_qa_to_file(qa_file, qa_pair)
        print("QA pair added to dataset")
        
        # Wait before next iteration
        time.sleep(5)  # Delay to respect API rate limits

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings using basic comparison."""
    # Convert to lowercase and split into words
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

if __name__ == "__main__":
    main()
