import pytesseract
import json
import os
from PIL import Image
from pdf2image import convert_from_path
from datetime import datetime
from pipeline_utils.llm_connections import LLMConnections
import config
from bson import ObjectId
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts/similar")
INPUT_DIR = "data/input_questions"
OUTPUT_DIR = "generated_questions/diagrams"

# Diagram generation instruction
#DIAGRAM_NO_SOLUTION_INSTRUCTION = "IMPORTANT: The diagram should illustrate the problem/scenario but should NOT show the solution or answer."
DIAGRAM_NO_SOLUTION_INSTRUCTION = ""

# LLM Configuration - these can be parameterized
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_QUESTIONS = 3  # Default number of questions to generate

def load_prompt(filename):
    with open(os.path.join(PROMPT_DIR, filename), 'r') as f:
        return f.read()

def extract_single_question(filepath):
    if filepath.endswith('.pdf'):
        # Only use first page since one question per file
        image = convert_from_path(filepath, first_page=1, last_page=1)[0]
        return pytesseract.image_to_string(image, lang='eng')
    elif filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
        return pytesseract.image_to_string(Image.open(filepath), lang='eng')
    elif filepath.endswith('.txt'):
        with open(filepath, 'r') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format.")

def generate_question(sample_text, topic, subject, subject_area, 
                     provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL, 
                     temperature=DEFAULT_TEMPERATURE, num_questions=DEFAULT_NUM_QUESTIONS,
                     system_prompt_template=None, user_prompt_template=None):
    """
    Generate similar questions using the LLM connections module.
    
    Args:
        sample_text: The sample question text
        topic: The topic for the question
        subject: The subject area
        subject_area: The specific subject area
        provider: LLM provider (openai, anthropic, gemini, deepseek)
        model: The model to use
        temperature: Temperature for generation
        num_questions: Number of questions to generate
        system_prompt_template: Custom system prompt template (optional)
        user_prompt_template: Custom user prompt template (optional)
    """
    
    # Load prompts - use custom templates if provided, otherwise use defaults
    if system_prompt_template is None:
        system_prompt_template = load_prompt("system_prompt.txt")
    
    if user_prompt_template is None:
        user_prompt_template = load_prompt("user_prompt.txt")
    
    # Format the prompts
    system_prompt = system_prompt_template.format(
        model=model,
        topic=topic,
        subject=subject,
        subject_area=subject_area,
        num_questions=num_questions
    )

    user_prompt = user_prompt_template.format(
        topic=topic,
        subject=subject,
        subject_area=subject_area,
        sample_question=sample_text.strip(),
        model=model,
        num_questions=num_questions,
        learning_objectives_section="" # Kept for compatibility, but now empty
    )

    # Create configuration for LLM connections
    llm_config = {
        f"{provider}_llm_model": model
    }
    
    # Initialize LLM connections
    llm_connections = LLMConnections(llm_config)
    
    # Make the API call
    content = llm_connections.call_llm_api(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature
    )
    
    if content is None:
        raise Exception(f"Failed to generate content using {provider} API")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_output": content}

def save_output_json(data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_question_{timestamp}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved to {filepath}")

def get_connection_doc_db():
    """
    Get connection to the document database.
    This function should be implemented based on your database connection setup.
    """
    try:
        from pymongo import MongoClient
        # Update these connection details with your actual MongoDB setup
        client = MongoClient('mongodb://localhost:27017/')
        return client['adaptive_learning_docs']  # Update with your database name
    except ImportError:
        logger.error("pymongo not installed. Please install it with: pip install pymongo")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None

def get_question_by_id(doc_id):
    """
    Retrieves the complete question document from the questions collection for a given doc_id.
    
    Args:
        doc_id: The document ID (can be string or ObjectId)
        
    Returns:
        dict: The question document or None if not found
    """
    logger.info(f"Fetching question document for doc_id: {doc_id}")
    try:
        db = get_connection_doc_db()
        if db is None:
            logger.error("Database connection failed")
            return None
            
        questions_collection = db['dryrun_questions']  # MongoDB collection

        # Determine if the doc_id is an ObjectId or an integer
        if isinstance(doc_id, str):
            try:
                doc_id = ObjectId(doc_id)
                logger.debug(f"Converted doc_id to ObjectId: {doc_id}")
            except Exception as e:
                logger.error(f"Invalid doc_id format: {doc_id}, Error: {e}")
                return None

        query = {"_id": doc_id}

        # Retrieve the document using the _id
        document = questions_collection.find_one(query)
        if not document:
            logger.warning(f"Document with doc_id {doc_id} not found.")
            return None

        # Convert ObjectId to string for JSON serialization
        if "_id" in document:
            document["_id"] = str(document["_id"])

        logger.info(f"Question document retrieved successfully for doc_id: {doc_id}")
        return document

    except Exception as e:
        logger.error(f"Error fetching question document for doc_id {doc_id}: {e}")
        return None

def generate_and_save_diagrams(result_data, base_output_dir):
    """
    Checks for diagram generation steps in the result and calls GPT-1 to create them.
    This function is specifically for LLM results with multiple questions.
    Args:
        result_data: The JSON data returned from the LLM.
        base_output_dir: The root directory to save diagrams in.
    """
    # Add null check for result_data
    if result_data is None:
        logger.error("Result data is None, cannot generate diagrams")
        return
        
    if 'questions' not in result_data:
        logger.warning("No 'questions' key found in result_data")
        return

    llm_connections = LLMConnections(config={})  # Instantiate for GPT-1
    diagram_output_dir = os.path.join(base_output_dir)
    
    for i, question in enumerate(result_data['questions']):
        if question.get('requires_diagram') and question.get('diagram_gen_steps'):
            print(f"🎨 Found diagram instructions for question {i+1}. Generating with GPT-1...")
            
            # Concatenate question text with diagram generation steps
            question_text = question.get('question', '')
            #diagram_steps = " ".join(question['diagram_gen_steps'])
            diagram_steps = " "
            gpt1_prompt = f"Question: {question_text}\n\nDiagram Instructions: {diagram_steps}\n\n{DIAGRAM_NO_SOLUTION_INSTRUCTION}"
            #gpt1_prompt = f"Question: {question_text}"
            
            result = llm_connections.generate_diagram_openai(
                prompt=gpt1_prompt,
                output_dir=diagram_output_dir
            )
            
            if result is None:
                print(f"❌ Diagram generation failed for question {i+1} - no image was generated")
            else:
                print(f"✅ Diagram for question {i+1} saved to {diagram_output_dir}")
        elif question.get('requires_diagram') and not question.get('diagram_gen_steps'):
            print(f"⚠️  Question {i+1} requires diagram but no diagram generation steps provided")
        else:
            print(f"📝 Question {i+1} does not require a diagram")

def generate_diagram_for_question(question_doc, base_output_dir):
    """
    Process a single question document from database for diagram generation.
    This function is specifically for database documents.
    """
    logger.info(f"Processing question document: {question_doc.get('_id')}")
    
    # Print the document structure for debugging
    #print("📄 Document structure:")
    #for key, value in question_doc.items():
    #    if key == '_id':
    #        print(f"  {key}: {value}")
    #    elif isinstance(value, str) and len(value) > 100:
    #        print(f"  {key}: {value[:100]}...")
    #    else:
    #        print(f"  {key}: {value}")

    # Check if the question requires a diagram
    requires_diagram = question_doc.get('requires_diagram', False)
    diagram_steps = question_doc.get('diagram_gen_steps', [])
    question_text = question_doc.get('question_text', '') or question_doc.get('question', '')
    
    if requires_diagram:
        print(f"🎨 Question requires diagram. Generating with GPT-1...")
        
        llm_connections = LLMConnections(config={})
        diagram_output_dir = os.path.join(base_output_dir)
        
        # Build the prompt based on available information
        if diagram_steps:
            # Use both question text and diagram generation steps
            diagram_steps_text = " ".join(diagram_steps) if isinstance(diagram_steps, list) else str(diagram_steps)
            gpt1_prompt = f"Question: {question_text}\n\nDiagram Instructions: {diagram_steps_text}\n\n{DIAGRAM_NO_SOLUTION_INSTRUCTION}"
            print(f"📝 Using question text with diagram instructions")
        else:
            # Use only question text when no diagram steps are provided
            gpt1_prompt = f"Question: {question_text}\n\n{DIAGRAM_NO_SOLUTION_INSTRUCTION}"
            print(f"📝 Using question text only (no diagram instructions provided)")
        
        try:
            # Generate diagram with custom filename
            doc_id = str(question_doc.get('_id', ''))
            filename = f"diagram_{doc_id}.png"
            
            result = llm_connections.generate_diagram_openai(
                prompt=gpt1_prompt,
                output_dir=diagram_output_dir,
                filename=filename
            )
            
            if result is None:
                print(f"❌ Diagram generation failed for doc_id: {doc_id} - no image was generated")
            else:
                print(f"✅ Diagram saved as {filename} in {diagram_output_dir}")
        except Exception as e:
            print(f"❌ Failed to generate diagram: {e}")
    else:
        print(f"📝 Question does not require a diagram")

def generate_similar_questions_from_file():
    """
    Main method to generate similar questions from an input file
    """
    # Use the specified input file with proper path handling
    file_path = os.path.join(os.getcwd(), INPUT_DIR, "input_question.jpg")
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: Input file not found at {file_path}")
        exit(1)
        
    print(f"📄 Using input file: {file_path}")

    # Metadata to inject into the prompt - specific to the histogram question
    topic = "Kimematics"
    subject = "AP Physics"
    subject_area = "Physics"
    
    # LLM parameters - these can be easily modified
    provider = "openai"  # Options: openai, anthropic, gemini, deepseek
    model = "gpt-4"      # Model name for the selected provider
    temperature = 0.7   # Temperature for generation (0.0 to 1.0)
    num_questions = 1    # Number of questions to generate

    try:
        print("📄 Reading question from input file...")
        sample_question = extract_single_question(file_path)
        print('***sample_question***', sample_question)
        
        print(f"🤖 Generating {num_questions} similar questions using {provider} ({model})...")
  
        result = generate_question(
            sample_question, 
            topic, 
            subject, 
            subject_area,
            provider=provider,
            model=model,
            temperature=temperature,
            num_questions=num_questions
        )

        print("💾 Saving output...")
        save_output_json(result)

        # Generate diagrams for questions that require them
        generate_and_save_diagrams(result, "generated_diagrams")
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        exit(1)

def generate_diagram_for_question_id(question_id):
    question_document = get_question_by_id(question_id)
    
    if question_document:
        print("✅ Successfully retrieved question document:")
        print(f"ID: {question_document.get('_id')}")
        #print(f"Question: {question_document.get('question_text', 'No text')[:200]}...")
        
        # Use the centralized diagram output directory
        skill_output_dir = "generated_diagrams"
        
        # Ensure the output directory exists
        os.makedirs(skill_output_dir, exist_ok=True)
        
        # Check if diagram already exists
        doc_id = str(question_document.get('_id', ''))
        diagram_filename = f"diagram_{doc_id}.png"
        diagram_path = os.path.join(skill_output_dir, diagram_filename)
        
        if os.path.exists(diagram_path):
            print(f"📁 Diagram already exists: {diagram_filename}")
            print(f"⏭️  Skipping generation for question {doc_id}")
            return
        else:
            print(f"🆕 Diagram does not exist: {diagram_filename}")
            print(f"🎨 Generating diagram for question {doc_id}")
        
        # Direct call to process single question diagram for database documents
        generate_diagram_for_question(question_document, skill_output_dir)
    else:
        print("❌ Failed to retrieve question document")
        print("\n🔍 Debugging options:")
        print("1. Uncomment 'check_database_structure()' to see available collections")
        print("2. Uncomment 'list_available_documents(10)' to see available document IDs")
        print("3. Update the document ID to one that exists in your database")

def generate_diagrams_for_skill(skill_name=None):
    """
    Generate diagrams for all questions in a specific skill.
    
    Args:
        skill_name: The skill name to filter questions by. Use "*", "no", "all", or None to get all documents.
    """
    logger.info(f"Generating diagrams for skill: {skill_name}")
    
    try:
        db = get_connection_doc_db()
        if db is None:
            logger.error("Database connection failed")
            return
            
        questions_collection = db['dryrun_questions']
        #questions_collection = db['test_questions']
        
        # Build query based on skill_name parameter
        if skill_name is None or skill_name in ["*", "no", "all"]:
            # Get all documents that require diagrams, regardless of skill
            query = {"requires_diagram": True}
            logger.info("Using wildcard parameter or None - getting all documents that require diagrams")
        else:
            # Query for questions with the specified skill and requires_diagram = true
            query = {
                "skill": skill_name,
                "requires_diagram": True
            }
            logger.info(f"Filtering by specific skill: {skill_name}")
        
        # Find all matching documents
        documents = list(questions_collection.find(query))
        
        if not documents:
            if skill_name is None or skill_name in ["*", "no", "all"]:
                logger.warning("No questions found that require diagrams")
            else:
                logger.warning(f"No questions found for skill '{skill_name}' that require diagrams")
            return
            
        logger.info(f"Found {len(documents)} questions that require diagrams")
        
        # Set the output directory for skill-based diagrams
        skill_output_dir = "generated_diagrams"
        
        # Ensure the output directory exists
        os.makedirs(skill_output_dir, exist_ok=True)
        
        # Process each question
        for i, question_doc in enumerate(documents, 1):
            print(f"\n{'='*60}")
            print(f"Processing question {i}/{len(documents)}")
            print(f"{'='*60}")
            
            # Convert ObjectId to string for JSON serialization
            if "_id" in question_doc:
                question_doc["_id"] = str(question_doc["_id"])
            
            # Check if diagram already exists
            doc_id = str(question_doc.get('_id', ''))
            diagram_filename = f"diagram_{doc_id}.png"
            diagram_path = os.path.join(skill_output_dir, diagram_filename)
            
            if os.path.exists(diagram_path):
                print(f"📁 Diagram already exists: {diagram_filename}")
                print(f"⏭️  Skipping generation for question {doc_id}")
                continue
            else:
                print(f"🆕 Diagram does not exist: {diagram_filename}")
                print(f"🎨 Generating diagram for question {doc_id}")
            
            # Generate diagram for this question
            generate_diagram_for_question(question_doc, skill_output_dir)
            
        logger.info(f"✅ Completed diagram generation for {len(documents)} questions")
        
    except Exception as e:
        logger.error(f"Error generating diagrams for skill '{skill_name}': {e}")

def generate_diagrams_for_subject(subject=None, question_type=None):
    """
    Generate diagrams for all questions in a specific subject.
    
    Args:
        subject: The subject name to filter questions by. Use "*", "no", "all", or None to get all documents.
        question_type: Optional question type to filter by.
    """
    logger.info(f"Generating diagrams for subject: {subject}")
    
    try:
        db = get_connection_doc_db()
        if db is None:
            logger.error("Database connection failed")
            return
            
        questions_collection = db['dryrun_questions']
        
        # Build query based on subject parameter
        if subject is None or subject in ["*", "no", "all"]:
            # Get all documents that require diagrams, regardless of subject
            query = {"requires_diagram": True}
            logger.info("Using wildcard parameter or None - getting all documents that require diagrams")
        else:
            # Query for questions with the specified subject and requires_diagram = true
            query = {
                "subject": subject,
                "requires_diagram": True
            }
            # Add question_type filter if provided
            if question_type:
                query["question_type"] = question_type
            logger.info(f"Filtering by specific subject: {subject}")
        
        # Find all matching documents
        documents = list(questions_collection.find(query))
        
        if not documents:
            if subject is None or subject in ["*", "no", "all"]:
                logger.warning("No questions found that require diagrams")
            else:
                logger.warning(f"No questions found for subject '{subject}' that require diagrams")
            return
            
        logger.info(f"Found {len(documents)} questions that require diagrams")
        
        # Set the output directory for subject-based diagrams
        subject_output_dir = "generated_diagrams"
        
        # Ensure the output directory exists
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Process each question
        for i, question_doc in enumerate(documents, 1):
            print(f"\n{'='*60}")
            print(f"Processing question {i}/{len(documents)}")
            print(f"{'='*60}")
            
            # Convert ObjectId to string for JSON serialization
            if "_id" in question_doc:
                question_doc["_id"] = str(question_doc["_id"])
            
            # Check if diagram already exists
            doc_id = str(question_doc.get('_id', ''))
            diagram_filename = f"diagram_{doc_id}.png"
            diagram_path = os.path.join(subject_output_dir, diagram_filename)
            
            if os.path.exists(diagram_path):
                print(f"📁 Diagram already exists: {diagram_filename}")
                print(f"⏭️  Skipping generation for question {doc_id}")
                continue
            else:
                print(f"🆕 Diagram does not exist: {diagram_filename}")
                print(f"🎨 Generating diagram for question {doc_id}")
            
            # Generate diagram for this question
            generate_diagram_for_question(question_doc, subject_output_dir)
            
        logger.info(f"✅ Completed diagram generation for {len(documents)} questions")
        
    except Exception as e:
        logger.error(f"Error generating diagrams for subject '{subject}': {e}")

# === MAIN ===

def main():
    """
    Main function to control script execution.
    """
    # Uncomment the line below to debug database structure
    # check_database_structure()
    
    # Uncomment the line below to list available documents
    # list_available_documents(10)
    
    # Uncomment the line below to run the original file-based generation
    # generate_similar_questions_from_file()
    
    # Try to get a question document by ID
    #generate_diagram_for_question_id("6850d4dbfeff49e44a5324f8")
    generate_diagrams_for_skill("Vector Calculus")
    #generate_diagrams_for_subject("AP Physics C: Electricity and Magnetism", "tests")
    #generate_diagrams_for_subject("AP Microeconomics")
    #generate_diagrams_for_skill("Problem Solving and Data Analysis")
    #generate_diagrams_for_skill("Long-Run Consequences of Stabilization Policie")
    #generate_diagrams_for_skill("Basic Micro Economic Concepts")

if __name__ == "__main__":
    main()

