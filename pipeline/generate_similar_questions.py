import pytesseract
import json
import os
from PIL import Image
from pdf2image import convert_from_path
from datetime import datetime
from pipeline_utils.llm_connections import LLMConnections
import config

# === CONFIG ===
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts/similar")
INPUT_DIR = "data/input_questions"
OUTPUT_DIR = "generated_questions/diagrams"

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

def generate_and_save_diagrams(result_data, base_output_dir):
    """
    Checks for diagram generation steps in the result and calls GPT-1 to create them.
    Args:
        result_data: The JSON data returned from the LLM.
        base_output_dir: The root directory to save diagrams in.
    """
    if 'questions' not in result_data:
        return

    llm_connections = LLMConnections(config={})  # Instantiate for GPT-1
    diagram_output_dir = os.path.join(base_output_dir)
    
    for i, question in enumerate(result_data['questions']):
        if question.get('requires_diagram') and question.get('diagram_gen_steps'):
            print(f"🎨 Found diagram instructions for question {i+1}. Generating with GPT-1...")
            
            # Concatenate question text with diagram generation steps
            question_text = question.get('question', '')
            diagram_steps = " ".join(question['diagram_gen_steps'])
            gpt1_prompt = f"Question: {question_text}\n\nDiagram Instructions: {diagram_steps}"
            
            llm_connections.generate_diagram_openai(
                prompt=gpt1_prompt,
                output_dir=diagram_output_dir
            )
            print(f"✅ Diagram for question {i+1} saved to {diagram_output_dir}")
            
            

# === MAIN ===

if __name__ == "__main__":
    # Use the specified input file with proper path handling
    file_path = os.path.join(os.getcwd(), INPUT_DIR, "input_question.jpg")
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: Input file not found at {file_path}")
        exit(1)
        
    print(f"📄 Using input file: {file_path}")

    # Metadata to inject into the prompt - specific to the histogram question
    topic = "Data Interpretation"
    subject = "SAT Math"
    subject_area = "Statistics"
    
    # LLM parameters - these can be easily modified
    provider = "openai"  # Options: openai, anthropic, gemini, deepseek
    model = "gpt-4"      # Model name for the selected provider
    temperature = 0.7    # Temperature for generation (0.0 to 1.0)
    num_questions = 1    # Number of questions to generate

    try:
        print("📄 Reading question from input file...")
        sample_question = extract_single_question(file_path)
        
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
        generate_and_save_diagrams(result, OUTPUT_DIR)
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        exit(1)
