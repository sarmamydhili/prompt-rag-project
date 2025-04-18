from tqdm import tqdm
from pipeline.pipeline_utils.llm_connections import call_llm_api
import json
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_llm_response(response: str) -> List[Dict]:
    """
    Parse LLM response into structured questions
    Args:
        response: LLM response string
    Returns:
        List of structured questions
    """
    try:
        # Clean the response string - remove markdown code block if present
        if response.startswith('```json'):
            response = response[7:]  # Remove ```json
        if response.endswith('```'):
            response = response[:-3]  # Remove ```
        response = response.strip()
        
        # Try to parse as JSON
        try:
            questions = json.loads(response)
            if isinstance(questions, list):
                return questions
            elif isinstance(questions, dict):
                return [questions]
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response")
            return []

        # If not JSON, try to parse as text
        questions = []
        current_question = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(('Q:', 'Question:', 'q:')):
                if current_question:
                    questions.append(current_question)
                current_question = {"question": line.split(':', 1)[1].strip()}
            elif line.startswith(('A:', 'Answer:', 'a:')):
                current_question["correct_answer"] = line.split(':', 1)[1].strip()
            elif line.startswith(('Options:', 'Choices:', 'options:')):
                current_question["multiple_choices"] = [opt.strip() for opt in line.split(':', 1)[1].split(',')]
        
        if current_question:
            questions.append(current_question)
            
        return questions
        
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        return []

def structure_questions_from_chunk(text: str, diagram_flag: bool, model_name: str) -> List[Dict]:
    """
    Structure questions from a text chunk
    Args:
        text: Text to process
        diagram_flag: Whether a diagram is required
        model_name: Name of the model used
    Returns:
        List of structured questions
    """
    try:
        questions = parse_llm_response(text)
        
        # Add metadata to each question
        for question in questions:
            question.update({
                "diagram_required": diagram_flag,
                "model_used": model_name,
                "metadata": {
                    "source": "llm_generation",
                    "has_diagram": diagram_flag
                }
            })
            
        logger.info(f"Successfully structured {len(questions)} questions")
        return questions
        
    except Exception as e:
        logger.error(f"Error structuring questions: {str(e)}")
        return []

def structure(self, extracted_pages):
    # Remove this function as it's now handled in pipeline_manager.py
    pass
