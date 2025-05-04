import logging
from services.gen_ai_access import get_question_by_id
from services.llm_connections import llm_connections
import json
import re
import sympy
from sympy.parsing.latex import parse_latex
from models.data_access import update_question_answer, get_connection_doc_db
from pymongo import MongoClient
from bson.objectid import ObjectId

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

model = 'deepseek'
database_name = "test_questions"


def extract_latex_value(text):
    """
    Extract LaTeX expressions from text and convert to numerical/computable form.
    
    Parameters:
    - text (str): Text containing LaTeX expressions
    
    Returns:
    - str: Extracted LaTeX value or original text if no LaTeX found
    """
    try:
        # Find LaTeX expressions (both inline $...$ and display $$...$$)
        latex_pattern = r'\$(.*?)\$'
        matches = re.findall(latex_pattern, text)
        
        if matches:
            # Take the first LaTeX expression found
            latex_expr = matches[0]
            try:
                # Parse LaTeX to SymPy expression
                sympy_expr = parse_latex(latex_expr)
                # Convert to numerical value if possible
                if sympy_expr.is_number:
                    return str(float(sympy_expr.evalf()))
                return str(sympy_expr)
            except Exception as e:
                logger.warning(f"Failed to parse LaTeX expression {latex_expr}: {e}")
                return text
        return text
    except Exception as e:
        logger.error(f"Error extracting LaTeX value: {e}")
        return text

def compare_answers(answer1, answer2):
    """
    Compare two answers, handling LaTeX expressions and numerical values.
    
    Parameters:
    - answer1 (str): First answer to compare
    - answer2 (str): Second answer to compare
    
    Returns:
    - bool: True if answers are equivalent, False otherwise
    """
    try:
        # Extract LaTeX values if present
        value1 = extract_latex_value(answer1)
        value2 = extract_latex_value(answer2)
        
        # Try numerical comparison
        try:
            num1 = float(value1)
            num2 = float(value2)
            return abs(num1 - num2) < 1e-10  # Allow for small floating point differences
        except ValueError:
            # If not numerical, do string comparison
            return value1.strip() == value2.strip()
    except Exception as e:
        logger.error(f"Error comparing answers: {e}")
        return False

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

def get_llm_answer(question, choices, model='gemini'):
    """
    Get answer from specified LLM model for a given question and choices
    
    Parameters:
    - question (str): The question text
    - choices (list): List of multiple choice options
    - model (str): The model to use ('gemini', 'anthropic', 'openai', 'deepseek')
    
    Returns:
    - str: The model's answer choice (A, B, C, or D) or None if failed
    """
    system_prompt = """You are a precise mathematical problem solver. Your task is to:
1. Carefully read and understand the mathematical problem
2. Solve the problem step by step, showing your reasoning
3. Compare your solution with each multiple choice option
4. Select the correct answer (A, B, C, or D)

For each problem:
- First, identify what mathematical concept is being tested
- Break down the problem into smaller, manageable steps
- Show your calculations and reasoning clearly
- If the problem involves Calculus (derivatives, concavity, critical points, maxima/minima, integrals), always explicitly compute the first and second derivatives (if needed), analyze their sign carefully, and DO NOT guess based on the shape of known graphs. Use rigorous mathematical reasoning.
- Verify your solution matches one of the given options
- If your solution doesn't match any option exactly, choose the closest one
- If multiple options seem correct, choose the most precise one

The question and answers will contain LaTeX expressions. Interpret these correctly as mathematical formulas.
After your step-by-step solution, respond with EXACTLY a JSON object in this format: {"answer": "X"} where X is A, B, C, or D.
Do NOT provide any other text after the JSON object."""

    user_prompt = f"""Question: {question}

Multiple Choice Options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Please solve this problem step by step:
1. Identify the mathematical concept being tested
2. Break down the problem into smaller steps
3. Show your calculations and reasoning
4. Compare your solution with each option
5. Select the correct answer (A, B, C, or D)

After your step-by-step solution, respond with only a JSON object in this format:
{{"answer": "X"}}
where X is the letter (A, B, C, or D) of the correct answer."""

    try:
        # Get response from specified model
        if model.lower() == 'gemini':
            response = llm_connections.get_content_from_gemini(system_prompt, user_prompt)
        elif model.lower() == 'anthropic':
            response = llm_connections.get_content_from_anthropic(system_prompt, user_prompt)
        elif model.lower() == 'openai':
            response = llm_connections.get_content_from_openai(system_prompt, user_prompt)
        elif model.lower() == 'deepseek':
            response = llm_connections.get_content_from_deepseek(system_prompt, user_prompt)
        else:
            logger.error(f"Unsupported model: {model}")
            return None

        # Log the full response for debugging
        #logger.info(f"Full {model} response: {response}")

        return parse_llm_response(response)
    except Exception as e:
        logger.error(f"Error getting {model} answer: {e}", exc_info=True)
        return None

def format_as_latex(value):
    """
    Format a value as LaTeX if it's a mathematical expression.
    
    Parameters:
    - value (str): The value to format
    
    Returns:
    - str: LaTeX formatted value
    """
    try:
        # Check if the value is already in LaTeX format
        if '$' in value:
            return value
            
        # Try to parse as a mathematical expression
        try:
            # If it's a simple number, format it directly
            float(value)
            return f"${value}$"
        except ValueError:
            # If it's not a simple number, try to parse as a mathematical expression
            try:
                sympy_expr = sympy.sympify(value)
                latex_expr = sympy.latex(sympy_expr)
                return f"${latex_expr}$"
            except:
                # If parsing fails, return the original value
                return value
    except Exception as e:
        logger.warning(f"Failed to format as LaTeX: {e}")
        return value

def fetch_documents(query, database_name='questions', limit=None):
    """
    Fetch documents from MongoDB based on query parameters.
    
    Parameters:
    - query (dict): MongoDB query to filter documents
    - database_name (str): Name of the database collection to search in (default: 'questions')
    - limit (int, optional): Maximum number of documents to retrieve
    
    Returns:
    - list: List of documents matching the query or None if failed
    """
    try:
        # Get database connection
        db = get_connection_doc_db()
        collection = db[database_name]
        
        # Convert ObjectId to string for logging
        def convert_objectid(obj):
            if isinstance(obj, ObjectId):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_objectid(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_objectid(item) for item in obj]
            return obj
            
        # Print the MongoDB query with ObjectIds converted to strings
        log_query = convert_objectid(query)
        #logger.info(f"MongoDB Query: {json.dumps(log_query, indent=2)}")
        if limit:
            logger.info(f"Query Limit: {limit}")
        
        # Apply limit if specified
        if limit:
            cursor = collection.find(query).limit(limit)
        else:
            cursor = collection.find(query)
        
        # Convert cursor to list of documents
        documents = list(cursor)
        
        if not documents:
            logger.warning(f"No documents found matching query: {log_query}")
            return None
        
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching documents: {e}", exc_info=True)
        return None

def fetch_question_documents_by_subject_and_skill(subject, database_name='questions', skill=None, limit=None):
    """
    Fetch question documents from the database that match the specified subject and optional skill.
    
    Parameters:
    - subject (str): The subject to filter questions by (e.g., "AP Calculus BC")
    - database_name (str): The name of the database collection to search in (default: 'questions')
    - skill (str, optional): The specific skill to filter questions by
    - limit (int, optional): Maximum number of documents to retrieve
    
    Returns:
    - list: List of question documents or None if failed
    """
    try:
        # Build query based on provided parameters
        query = {"subject": subject}
        if skill:
            query["skill"] = skill
        
        # Use fetch_documents to get the questions
        question_documents = fetch_documents(query, database_name, limit)
        
        if not question_documents:
            logger.error(f"No questions found for subject: {subject}" + (f" and skill: {skill}" if skill else ""))
            return None
        
        logger.info(f"Retrieved {len(question_documents)} questions for subject: {subject}" + (f" and skill: {skill}" if skill else ""))
        return question_documents
        
    except Exception as e:
        logger.error(f"Error fetching questions by subject and skill: {e}", exc_info=True)
        return None

def fetch_question_documents_by_ids(question_ids, database_name='questions'):
    """
    Fetch question documents from the database for given question IDs.
    
    Parameters:
    - question_ids (str or list): Single question ID or list of question IDs
    - database_name (str): Name of the database collection to search in (default: 'questions')
    
    Returns:
    - list: List of question documents or None if failed
    """
    # Convert single question_id to list for consistent processing
    if isinstance(question_ids, str):
        question_ids = [question_ids]
        
    try:
        logger.info(f"\nFetching documents for {len(question_ids)} question IDs")
        logger.info(f"Question IDs: {question_ids}")
        
        # Convert string IDs to ObjectId
        object_ids = [ObjectId(qid) for qid in question_ids]
        
        # Fetch all documents at once - use ObjectId directly in $in
        query = {"_id": {"$in": object_ids}}
        question_documents = fetch_documents(query, database_name)
        
        if not question_documents:
            logger.error(f"No questions found for IDs: {question_ids}")
            return None
            
        # Create a map of question_id to document for quick lookup
        question_map = {str(doc['_id']): doc for doc in question_documents}
        documents = [question_map.get(qid) for qid in question_ids if qid in question_map]
        
        if len(documents) != len(question_ids):
            missing_ids = set(question_ids) - set(question_map.keys())
            logger.warning(f"Found {len(documents)} documents out of {len(question_ids)} requested IDs")
            logger.warning(f"Missing IDs: {missing_ids}")
            
        logger.info(f"Successfully mapped {len(documents)} documents to their IDs")
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching documents by IDs: {e}", exc_info=True)
        return None

def update_test_question_answer(doc_id, choice, database_name='test_questions'):
    """
    Updates the correct_answer field in the test_questions collection for a given doc_id.
    
    Parameters:
    - doc_id (str): The document ID to update
    - choice (str): The new correct answer choice (A, B, C, or D)
    - database_name (str): The name of the database collection (default: 'test_questions')
    
    Returns:
    - str: Success or error message
    """
    logger.info(f"Updating test question answer for doc_id: {doc_id} with choice: {choice}")
    try:
        db = get_connection_doc_db()
        questions_collection = db[database_name]  # MongoDB collection

        # Convert string ID to ObjectId
        try:
            doc_id = ObjectId(doc_id)
            logger.debug(f"Converted doc_id to ObjectId: {doc_id}")
        except Exception as e:
            logger.error(f"Invalid doc_id format: {doc_id}, Error: {e}")
            return f"Invalid doc_id format: {e}"

        query = {"_id": doc_id}

        # Retrieve the document using the _id
        document = questions_collection.find_one(query)
        if not document:
            logger.warning(f"Document with doc_id {doc_id} not found.")
            return f"Document with doc_id {doc_id} not found."

        # Validate the choice and update the document if valid
        if choice in ["A", "B", "C", "D"]:
            result = questions_collection.update_one(
                query,
                {"$set": {"correct_answer": choice}}
            )
            if result.modified_count > 0:
                logger.info(f"Document with doc_id {doc_id} updated successfully with choice: {choice}")
                return f"Document with doc_id {doc_id} updated successfully with choice: {choice}"
            else:
                logger.warning(f"Document with doc_id {doc_id} was not updated. It may already have the selected choice.")
                return f"Document with doc_id {doc_id} was not updated. It may already have the selected choice."
        else:
            logger.warning(f"Invalid choice: {choice}. Must be one of A, B, C, or D.")
            return f"Invalid choice. Please select from A, B, C, or D."
    except Exception as e:
        logger.error(f"Error updating test question answer for doc_id {doc_id}: {e}")
        return f"An error occurred: {e}"

def review_and_correct_questions(question_documents, model, database_name='test_questions'):
    """
    Review and correct multiple questions using specified LLM model.
    
    Parameters:
    - question_documents (list): List of question documents to review
    - model (str): The model to use ('gemini', 'anthropic', 'openai', 'deepseek')
    - database_name (str): The name of the database collection to update (default: 'test_questions')
    
    Returns:
    - list: List of updated question data
    """
    if not isinstance(question_documents, list):
        logger.error("Input must be a list of question documents")
        return []
        
    results = []
    logger.info(f"\nStarting review and correction for {len(question_documents)} questions")
    
    for question_data in question_documents:
        if not question_data:
            logger.warning("Skipping empty question data")
            continue
            
        try:
            question_id = str(question_data['_id'])
            logger.info(f"\nProcessing question ID: {question_id}")
            
            # Extract question elements
            question = question_data.get('question', '')
            multiple_choices = question_data.get('multiple_choices', [])
            correct_answer = question_data.get('correct_answer', '')

            if not question:
                logger.error(f"Missing question text for ID {question_id}")
                continue
            if not multiple_choices:
                logger.error(f"Missing multiple choices for ID {question_id}")
                continue
            if not correct_answer:
                logger.error(f"Missing correct answer for ID {question_id}")
                continue

            logger.info(f"Question data validation passed for ID {question_id}")
            
            # Store original answer for tracking changes
            question_data['original_answer'] = correct_answer
            question_data['original_answer_text'] = multiple_choices[ord(correct_answer) - ord('A')]

            # Get model's answer
            logger.info(f"Getting {model} answer for question ID {question_id}")
            model_answer = get_llm_answer(question, multiple_choices, model)
            if not model_answer:
                logger.error(f"Failed to get {model} answer for question {question_id}")
                continue
            logger.info(f"Received {model} answer: {model_answer}")
            
            # Store model's answer in question_data for reporting
            question_data['model_answer'] = model_answer
            
            # Check if model's answer matches any of the choices
            choice_index = ord(model_answer) - ord('A')  # Convert A,B,C,D to 0,1,2,3
            if 0 <= choice_index < len(multiple_choices):
                # If model's answer matches a choice, store its text
                question_data['model_answer_text'] = multiple_choices[choice_index]
                
                # Update the correct answer if different
                if model_answer != correct_answer:
                    logger.info(f"Updating correct answer from {correct_answer} to {model_answer}")
                    # Call update_test_question_answer to update the correct answer in the database
                    update_result = update_test_question_answer(question_id, model_answer, database_name)
                    logger.info(f"Update result: {update_result}")
                    
                    # Verify the update
                    updated_doc = fetch_question_documents_by_ids(question_id, database_name)
                    if updated_doc and updated_doc[0].get('correct_answer') == model_answer:
                        logger.info(f"Update verification successful for ID {question_id}")
                        question_data['correct_answer'] = model_answer
                        question_data['new_answer_text'] = multiple_choices[choice_index]
                    else:
                        logger.error(f"Update verification failed for question {question_id}")
                        continue
                else:
                    question_data['new_answer_text'] = question_data['original_answer_text']
            else:
                # If model's answer doesn't match any choice, format it and update choice A
                formatted_answer = format_as_latex(model_answer)
                logger.info(f"Updating choice A with {model}'s answer: {formatted_answer}")
                question_data['model_answer_text'] = model_answer  # Store the raw model answer
                question_data['multiple_choices'][0] = formatted_answer
                
                # Call update_test_question_answer to update the correct answer to A in the database
                update_result = update_test_question_answer(question_id, 'A', database_name)
                logger.info(f"Update result: {update_result}")
                
                # Verify the update
                updated_doc = fetch_question_documents_by_ids(question_id, database_name)
                if updated_doc and updated_doc[0].get('correct_answer') == 'A':
                    logger.info(f"Update verification successful for ID {question_id}")
                    question_data['correct_answer'] = 'A'
                    question_data['new_answer_text'] = formatted_answer
                else:
                    logger.error(f"Update verification failed for question {question_id}")
                    continue

            results.append(question_data)
            logger.info(f"Successfully processed question ID {question_id}")

        except Exception as e:
            logger.error(f"Error reviewing question {question_id}: {e}", exc_info=True)
            continue
            
    logger.info(f"\nCompleted review and correction for {len(results)} questions")
    return results

def generate_correction_report(updated_questions, model):
    """
    Generate and print a detailed report of the correction process and create a CSV report.
    
    Parameters:
    - updated_questions (list): List of updated question documents
    - model (str): The model used for correction
    """
    if not updated_questions:
        logger.error("No questions to generate report for")
        return
        
    # Calculate statistics
    total_questions = len(updated_questions)
    answer_changes = 0
    choice_updates = 0
    
    # Create CSV report
    import csv
    from datetime import datetime
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"correction_report_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'Question Id', 
            'Original Answer Choice', 
            'New Answer Choice', 
            'Model Returned Answer',
            'Original Answer', 
            'New Answer',
            'Model Answer Text',
            'Choice Answer Changed'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Log detailed results
        logger.info("\n" + "="*80)
        logger.info(f"Review and Correction Summary using {model.upper()} model")
        logger.info("="*80)
        logger.info(f"Total questions processed: {total_questions}")
        logger.info("\nDetailed Results:")
        logger.info("-"*80)
        
        for question in updated_questions:
            question_id = question.get('_id')
            original_answer = question.get('original_answer', 'Unknown')
            new_answer = question.get('correct_answer')
            model_answer = question.get('model_answer', 'Unknown')
            original_answer_text = question.get('original_answer_text', 'Unknown')
            new_answer_text = question.get('new_answer_text', 'Unknown')
            model_answer_text = question.get('model_answer_text', 'Unknown')
            
            # Determine if choice answer changed
            choice_changed = original_answer != new_answer
            if choice_changed:
                answer_changes += 1
                if new_answer == 'A':  # If answer is A, it means we updated choice A
                    choice_updates += 1
            
            # Write to CSV
            writer.writerow({
                'Question Id': question_id,
                'Original Answer Choice': original_answer,
                'New Answer Choice': new_answer,
                'Model Returned Answer': model_answer,
                'Original Answer': original_answer_text,
                'New Answer': new_answer_text,
                'Model Answer Text': model_answer_text,
                'Choice Answer Changed': choice_changed
            })
            
            if original_answer != new_answer:
                if new_answer == 'A':  # If answer is A, it means we updated choice A
                    logger.info(f"Question ID: {question_id}")
                    logger.info(f"  - Model returned answer: {model_answer} ({model_answer_text})")
                    logger.info(f"  - Updated choice A with new answer")
                    logger.info(f"  - New correct answer: {new_answer} ({new_answer_text})")
                else:
                    logger.info(f"Question ID: {question_id}")
                    logger.info(f"  - Model returned answer: {model_answer} ({model_answer_text})")
                    logger.info(f"  - Changed correct answer from {original_answer} ({original_answer_text}) to {new_answer} ({new_answer_text})")
            else:
                logger.info(f"Question ID: {question_id}")
                logger.info(f"  - Model returned answer: {model_answer} ({model_answer_text})")
                logger.info(f"  - No changes needed (correct answer: {new_answer} ({new_answer_text}))")
        
        # Write summary statistics to CSV
        writer.writerow({})  # Empty row for separation
        writer.writerow({
            'Question Id': 'SUMMARY STATISTICS',
            'Original Answer Choice': '',
            'New Answer Choice': '',
            'Model Returned Answer': '',
            'Original Answer': '',
            'New Answer': '',
            'Model Answer Text': '',
            'Choice Answer Changed': ''
        })
        writer.writerow({
            'Question Id': 'Total questions processed',
            'Original Answer Choice': total_questions,
            'New Answer Choice': '',
            'Model Returned Answer': '',
            'Original Answer': '',
            'New Answer': '',
            'Model Answer Text': '',
            'Choice Answer Changed': ''
        })
        writer.writerow({
            'Question Id': 'Total answer changes',
            'Original Answer Choice': answer_changes,
            'New Answer Choice': '',
            'Model Returned Answer': '',
            'Original Answer': '',
            'New Answer': '',
            'Model Answer Text': '',
            'Choice Answer Changed': ''
        })
        writer.writerow({
            'Question Id': 'Total choice updates',
            'Original Answer Choice': choice_updates,
            'New Answer Choice': '',
            'Model Returned Answer': '',
            'Original Answer': '',
            'New Answer': '',
            'Model Answer Text': '',
            'Choice Answer Changed': ''
        })
        writer.writerow({
            'Question Id': 'Questions unchanged',
            'Original Answer Choice': total_questions - answer_changes,
            'New Answer Choice': '',
            'Model Returned Answer': '',
            'Original Answer': '',
            'New Answer': '',
            'Model Answer Text': '',
            'Choice Answer Changed': ''
        })
        
        logger.info("\n" + "="*80)
        logger.info("Correction Statistics:")
        logger.info(f"Total questions processed: {total_questions}")
        logger.info(f"Total answer changes: {answer_changes}")
        logger.info(f"Total choice updates: {choice_updates}")
        logger.info(f"Questions unchanged: {total_questions - answer_changes}")
        logger.info(f"CSV report generated: {csv_filename}")
        logger.info("="*80)

def make_corrections_for_question_ids():
    """
    Make corrections for questions with the given IDs using the specified model.
    """
    try:
        question_ids = [
        '67e9e1eb024ecb99b5c3d776',
        '67e9e1eb024ecb99b5c3d72f',
        '67e9e1eb024ecb99b5c3d734',
        '67e9e1eb024ecb99b5c3d740',
        '67e9e1eb024ecb99b5c3d74a',
        '67e9e1eb024ecb99b5c3d74f',
        '67e9e1eb024ecb99b5c3d759',
        '67e9e1eb024ecb99b5c3d75f',
        '67e9e1eb024ecb99b5c3d761',
        '67e9e1eb024ecb99b5c3d762',
        '67e9e1eb024ecb99b5c3d769',
        '67e9e1eb024ecb99b5c3d76a',
        '67e9e1eb024ecb99b5c3d77e',
        '67e9e1eb024ecb99b5c3d786',
        '67e9e1eb024ecb99b5c3d78c'
        ]
        
        logger.info(f"Starting processing for {len(question_ids)} question IDs")
        
       
        question_documents = fetch_question_documents_by_ids(question_ids, database_name)
        if not question_documents:
            logger.error("Failed to fetch question documents")
            return
            
        logger.info(f"Successfully fetched {len(question_documents)} documents from database")
        
        updated_questions = review_and_correct_questions(question_documents, model)
        logger.info(f"Completed review and correction for {len(updated_questions)} questions")
        
        generate_correction_report(updated_questions, model)
    
    except Exception as e:
        logger.error(f"Error in make_corrections_for_question_ids: {e}", exc_info=True)

def make_corrections_for_subject():
    """
    Make corrections for questions in a specific subject.
    """
    try:
        subject = "AP Calculus AB"
        question_documents = fetch_question_documents_by_subject_and_skill(subject, database_name="test_questions")
        updated_questions = review_and_correct_questions(question_documents, model)
        generate_correction_report(updated_questions, model)    
    except Exception as e:
        logger.error(f"Error in make_corrections_for_subject: {e}", exc_info=True)

def main():
    """
    Main function to run the review and correction process.
    """
    #make_corrections_for_question_ids()
    make_corrections_for_subject()
        
        
        

if __name__ == '__main__':
    main() 