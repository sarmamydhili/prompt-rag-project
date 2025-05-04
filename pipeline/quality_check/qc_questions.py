import csv
import logging
from datetime import datetime
from services.gen_ai_access import get_connection_doc_db    
from services.llm_connections import llm_connections
from services.gen_ai_access import get_question_by_id
import json
import re
import collections

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _process_single_question_document(question_data, question_id=None, models_to_include=None):
    """
    Core function to process a single question document by getting answers from different LLMs and comparing with correct answer.
    
    Parameters:
    - question_data (dict): The question document containing question, choices, and correct answer
    - question_id (str, optional): The ID of the question, if available
    - models_to_include (list, optional): List of model names to include in analysis. 
      If None, all models are included. Valid options: ['anthropic', 'openai', 'deepseek', 'gemini']
    
    Returns:
    - dict or None: Dictionary containing analysis data if successful, None if failed
    """
    try:
        # Extract question elements
        question = question_data.get('question', '')
        multiple_choices = question_data.get('multiple_choices', [])
        correct_answer = question_data.get('correct_answer', '')
        
        if not question or not multiple_choices or not correct_answer:
            logger.error(f"Missing required question data for document {question_id or 'unknown'}")
            return None

        # If models_to_include is None, include all models
        if models_to_include is None:
            models_to_include = ['anthropic', 'openai', 'deepseek', 'gemini']

        # Construct prompts for answer choice identification
        base_system_prompt = """You are an expert in analyzing physics problems and their solutions. 
        Your task is to identify the correct answer choice (A, B, C, or D) based on the step-by-step solution provided.
        Consider the multiple choice options carefully and select the one that matches the final answer in the solution."""

        # Claude-specific system prompt - more explicit about response format
        claude_system_prompt = """You are a precise answer selection system.
        Your ONLY task is to select the correct answer (A, B, C, or D) and respond with EXACTLY a JSON object.
        You must NOT provide explanations, steps, or any other text.
        Respond with ONLY this format: {"answer": "X"} where X is A, B, C, or D.
        Any other format or additional text is considered an error."""

        user_prompt = f"""Question: {question}

Multiple Choice Options:
A) {multiple_choices[0]}
B) {multiple_choices[1]}
C) {multiple_choices[2]}
D) {multiple_choices[3]}

Based on your analysis, which answer choice (A, B, C, or D) is correct? 
Respond with only the letter (A, B, C, or D) that corresponds to the correct answer.
Please respond with a JSON object in the following format:
{{"answer": "A"}}
Don't include any other text or characters in your response.
"""

        def parse_llm_response(response):
            """Helper function to parse and validate LLM responses"""
            if not response:
                return None
            
            try:
                # Clean the response - remove any markdown formatting if present
                cleaned_response = response.strip().strip('`').strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:-3].strip()
                elif cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:-3].strip()
                elif cleaned_response.startswith('json'):
                    cleaned_response = cleaned_response[4:].strip()
                
                # Try to find JSON object in the response
                json_match = re.search(r'(\{"answer":\s*"[A-D]"\})', cleaned_response)
                if json_match:
                    json_str = json_match.group(1)
                    json_response = json.loads(json_str)
                    answer = json_response.get('answer', '')
                    if answer and isinstance(answer, str) and answer.upper() in ['A', 'B', 'C', 'D']:
                        return answer.upper()
                    else:
                        logger.error(f"Invalid answer format in response: {json_str}")
                        return None
                else:
                    logger.error(f"No JSON answer found in response: {cleaned_response}")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return None
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                return None

        # Initialize answers as None
        anthropic_answer = None
        openai_answer = None
        deepseek_answer = None
        gemini_answer = None

        # Get AI's answers only for selected models
        if 'anthropic' in models_to_include:
            anthropic_response = llm_connections.get_content_from_anthropic(claude_system_prompt, user_prompt)
            anthropic_answer = parse_llm_response(anthropic_response)

        if 'openai' in models_to_include:
            openai_response = llm_connections.get_content_from_openai(base_system_prompt, user_prompt)
            openai_answer = parse_llm_response(openai_response)

        if 'deepseek' in models_to_include:
            deepseek_response = llm_connections.get_content_from_deepseek(base_system_prompt, user_prompt)
            deepseek_answer = parse_llm_response(deepseek_response)

        if 'gemini' in models_to_include:
            gemini_response = llm_connections.get_content_from_gemini(base_system_prompt, user_prompt)
            gemini_answer = parse_llm_response(gemini_response)
        
        # Determine Majority Answer and Need for Review
        responses = [ans for ans in [anthropic_answer, openai_answer, deepseek_answer, gemini_answer, correct_answer] if ans is not None]
        majority_answer = 'No Majority'
        majority_answer_value = ''
        majority_choice_text = ''
        if responses:
            counts = collections.Counter(responses)
            most_common = counts.most_common(1)
            if most_common and most_common[0][1] >= 3:  # Still require 3 votes for majority
                majority_answer = f"{most_common[0][0]} ({most_common[0][1]} votes)"
                majority_answer_value = most_common[0][0]
                # Get the actual choice text based on the majority answer
                choice_index = ord(majority_answer_value) - ord('A')  # Convert A,B,C,D to 0,1,2,3
                if 0 <= choice_index < len(multiple_choices):
                    majority_choice_text = multiple_choices[choice_index]
        
        need_review = (majority_answer == 'No Majority')

        # Prepare data for CSV
        csv_data = {
            'question_id': question_id or question_data.get('_id', 'unknown'),
            'anthropic_answer': anthropic_answer,
            'openai_answer': openai_answer,
            'deepseek_answer': deepseek_answer,
            'gemini_answer': gemini_answer,
            'correct_answer': correct_answer,
            'majority_answer': majority_answer,
            'majority_answer_value': majority_answer_value,
            'majority_choice_text': majority_choice_text,
            'review': 1 if need_review else 0
        }
        
        return csv_data

    except Exception as e:
        logger.error(f"Error processing question {question_id or 'unknown'}: {e}", exc_info=True)
        return None

def analyze_question_answer(question_id):
    """
    Analyzes a question by getting answers from Anthropic, OpenAI, and DeepSeek and comparing with correct answer.
    Returns the analysis data for CSV writing.

    Parameters:
    - question_id (str): The ID of the question to analyze.

    Returns:
    - dict or None: Dictionary containing analysis data if successful, None if failed.
    """
    try:
        # Get question details
        question_data = get_question_by_id(question_id)
        if not question_data:
            logger.error(f"No question found for ID: {question_id}")
            return None
            
        # Process the question document
        return _process_single_question_document(question_data, question_id)

    except Exception as e:
        logger.error(f"Error analyzing question {question_id}: {e}", exc_info=True)
        return None

def analyze_question_list(question_documents, models_to_include=None):
    """
    Analyzes a list of question documents by getting answers from different LLMs and comparing with correct answers.
    
    Parameters:
    - question_documents (list): List of question document dictionaries
    - models_to_include (list, optional): List of model names to include in analysis.
      If None, all models are included. Valid options: ['anthropic', 'openai', 'deepseek', 'gemini']
    
    Returns:
    - list: List of dictionaries containing analysis data for each question
    """
    try:
        results = []
        
        for doc in question_documents:
            # Extract question ID from document if available
            # Handle both ObjectId objects from MongoDB and dictionary representations
            if '_id' in doc:
                if hasattr(doc['_id'], 'get'):  # Dictionary representation with $oid
                    question_id = doc['_id'].get('$oid', 'unknown')
                else:  # ObjectId object
                    question_id = str(doc['_id'])
            else:
                question_id = 'unknown'
            
            # Process the question document
            csv_data = _process_single_question_document(doc, question_id, models_to_include)
            
            if csv_data:
                results.append(csv_data)
            else:
                logger.error(f"Failed to analyze question ID: {question_id}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in analyze_question_list: {e}", exc_info=True)
        return []

def analyze_questions_by_subject(subject, database_name='test_questions', skill=None, limit=None, models_to_include=None):
    """
    Retrieves and analyzes questions from the database that match the specified subject and optional skill.
    
    Parameters:
    - subject (str): The subject to filter questions by (e.g., "AP Calculus BC")
    - database_name (str): The name of the database collection to search in (default: 'test_questions')
    - skill (str, optional): The specific skill to filter questions by
    - limit (int, optional): Maximum number of documents to retrieve and analyze
    - models_to_include (list, optional): List of model names to include in analysis.
      If None, all models are included. Valid options: ['anthropic', 'openai', 'deepseek', 'gemini']
    
    Returns:
    - str: Path to the generated CSV file or None if the operation failed
    """
    try:
        # Get database connection
        db = get_connection_doc_db()
        questions_collection = db[database_name]
        
        # Build query based on provided parameters
        query = {"subject": subject}
        if skill:
            query["skill"] = skill
        
        # Print the MongoDB query
        logger.info(f"MongoDB Query: {json.dumps(query, indent=2)}")
        if limit:
            logger.info(f"Query Limit: {limit}")
        
        # Apply limit if specified
        if limit:
            cursor = questions_collection.find(query).limit(limit)
        else:
            cursor = questions_collection.find(query)
        
        # Convert cursor to list of documents
        question_documents = list(cursor)
        
        if not question_documents:
            logger.error(f"No questions found for subject: {subject}" + (f" and skill: {skill}" if skill else ""))
            return None
        
        logger.info(f"Retrieved {len(question_documents)} questions for subject: {subject}" + (f" and skill: {skill}" if skill else ""))
        
        # Analyze the documents
        analysis_results = analyze_question_list(question_documents, models_to_include)
        
        # Generate CSV report with the results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subject_slug = subject.lower().replace(' ', '_')
        skill_slug = f"_{skill.lower().replace(' ', '_')}" if skill else ""
        csv_filename = f'analysis_results_{subject_slug}{skill_slug}_{timestamp}.csv'
        
        return generate_analysis_report_csv(analysis_results, csv_filename)
        
    except Exception as e:
        logger.error(f"Error analyzing questions by subject '{subject}'" + (f" and skill '{skill}'" if skill else "") + f": {e}", exc_info=True)
        return None

def generate_analysis_report_csv(analysis_results, csv_filename=None):
    """
    Generates a CSV report from the analysis results including summary statistics.
    
    Parameters:
    - analysis_results (list): List of dictionaries containing analysis data
    - csv_filename (str, optional): Filename for the CSV report. If None, a timestamped name will be generated.
    
    Returns:
    - str: Path to the generated CSV file
    """
    try:
        if not analysis_results:
            logger.error("No analysis data to write to CSV file")
            return None
            
        # Initialize counters
        review_count = 0
        total_questions = len(analysis_results)
        
        # Initialize model match counters
        anthropic_matches = 0
        openai_matches = 0
        deepseek_matches = 0
        gemini_matches = 0
        
        # Calculate summary statistics
        for result in analysis_results:
            if result['review']:
                review_count += 1
                
            # Count matches with majority answer
            if result['majority_answer_value']:  # Only count if there is a majority
                if result['anthropic_answer'] == result['majority_answer_value']:
                    anthropic_matches += 1
                if result['openai_answer'] == result['majority_answer_value']:
                    openai_matches += 1
                if result['deepseek_answer'] == result['majority_answer_value']:
                    deepseek_matches += 1
                if result['gemini_answer'] == result['majority_answer_value']:
                    gemini_matches += 1
        
        # Calculate model performance percentages
        anthropic_performance = f"{anthropic_matches}/{total_questions} ({round(anthropic_matches/total_questions*100, 2)}%)"
        openai_performance = f"{openai_matches}/{total_questions} ({round(openai_matches/total_questions*100, 2)}%)"
        deepseek_performance = f"{deepseek_matches}/{total_questions} ({round(deepseek_matches/total_questions*100, 2)}%)"
        gemini_performance = f"{gemini_matches}/{total_questions} ({round(gemini_matches/total_questions*100, 2)}%)"
        
        # Add summary row
        summary_row = {
            'question_id': 'TOTAL',
            'anthropic_answer': anthropic_performance,
            'openai_answer': openai_performance,
            'deepseek_answer': deepseek_performance,
            'gemini_answer': gemini_performance,
            'correct_answer': '',
            'majority_answer': '',
            'majority_answer_value': '',
            'majority_choice_text': '',
            'review': f'{review_count}'
        }
        
        # Ensure analysis_results is not empty before appending summary
        if not analysis_results:
            logger.error("Cannot generate CSV report from empty analysis results after processing.")
            return None
            
        # Make a mutable copy to append summary row
        results_with_summary = list(analysis_results) 
        results_with_summary.append(summary_row)
        
        # Generate CSV filename if not provided
        if not csv_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f'analysis_results_{timestamp}.csv'
        
        # Write to CSV file
        fieldnames = analysis_results[0].keys()
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_with_summary)
        
        logger.info(f"Successfully wrote analysis results to {csv_filename}")
        return csv_filename
        
    except Exception as e:
        logger.error(f"Error generating CSV report: {e}", exc_info=True)
        return None

def analyze_test_questions():
    """
    Analyzes AP Calculus BC test questions in the database and writes results to a CSV file.
    """
    try:
        # Get database connection
        db = get_connection_doc_db()
        test_questions_collection = db['test_questions']
        
        # Use specific question IDs
        question_ids = [
            '67edcabeadab6b7882e51011',
            '67edcabeadab6b7882e5100d',
            '67edcabeadab6b7882e5100e',
            '67edcabeadab6b7882e5100f',
            '67edcabeadab6b7882e51010',
            '67edcabeadab6b7882e51012',
            '67edcabeadab6b7882e51013',
            '67edcabeadab6b7882e51014',
            '67edcabeadab6b7882e51015',
            '67edcabeadab6b7882e51016'
        ]
        
        # METHOD 1: Using question IDs
        # Initialize list to store all CSV data
        all_csv_data = []
        
        # Analyze each question
        for question_id in question_ids:
            csv_data = analyze_question_answer(question_id)
            if csv_data:
                all_csv_data.append(csv_data)
        
        # Generate CSV report for Method 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'ap_calculus_bc_analysis_results_method1_{timestamp}.csv'
        generate_analysis_report_csv(all_csv_data, csv_filename)
        
        # METHOD 2: Using list of question documents
        # Fetch all question documents at once
        question_documents = []
        for question_id in question_ids:
            question_data = get_question_by_id(question_id)
            if question_data:
                question_documents.append(question_data)
        
        # Analyze all question documents
        analysis_results = analyze_question_list(question_documents)
        
        # Generate CSV report for Method 2
        csv_filename = f'ap_calculus_bc_analysis_results_method2_{timestamp}.csv'
        generate_analysis_report_csv(analysis_results, csv_filename)
    
    except Exception as e:
        logger.error(f"Error in analyze_test_questions: {e}", exc_info=True)

def main():
    """
    Main function to run the test questions analysis.
    """
    try:
        # Analyze test questions using the predefined IDs approach
        #analyze_test_questions()
        
        # Alternatively, analyze questions by subject
        analyze_questions_by_subject(
            "AP Calculus AB", 
            database_name="test_questions",
            models_to_include=['anthropic', 'openai']  # Example: only use Anthropic and OpenAI
            #skill="Parametric Equations, Polar Coordinates, and Vector-Valued Functions"
        )
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)

if __name__ == '__main__':
    main() 