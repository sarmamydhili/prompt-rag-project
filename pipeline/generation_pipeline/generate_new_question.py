import os
import sys
import re
import configparser

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Standard library imports
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter
from dotenv import load_dotenv

# Third-party imports
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm

# Local imports
import config
from pipeline.pipeline_utils.db_connections import get_mysql_connection, get_mongo_connection, DBConfig
from pipeline.generation_pipeline.build_prompt import PromptBuilder
from pipeline.pipeline_utils.llm_connections import LLMConnections
from pipeline.pipeline_utils.mongo_operations import MongoOperations
from pipeline.pipeline_utils.sql_operations import SQLOperations

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# ------------------ Global Context ------------------
class GlobalContext:
    def __init__(self):
        # Database connections
        self.mysql_conn = None
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_operations = None
        self.sql_operations = None

        # Collection names will be loaded from task_config
        self.mongo_questions_collection = None  # For reading questions
        self.mongo_output_collection_name = None  # For writing enhanced/generated questions

        # Add new attributes for Bloom's level specific prompts
        self.bloom_prompt_paths = {
            'Remembering': {
                'system': None,
                'user': None
            },
            'Understanding': {
                'system': None,
                'user': None
            },
            'Applying': {
                'system': None,
                'user': None
            },
            'Analyzing': {
                'system': None,
                'user': None
            },
            'Evaluating': {
                'system': None,
                'user': None
            }
        }

        # Load application configurations from properties file
        self._load_app_config()
        
        # Initialize database connections after config is loaded
        self.initialize()

    def _load_app_config(self):
        """Load application configurations from task_config.properties"""
        try:
            # Get the pipeline directory
            pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(pipeline_dir, 'task_config.properties')
            print(f"Looking for config file at: {config_path}")
            
            # Check if file exists
            if not os.path.exists(config_path):
                print(f"Error: Config file not found at {config_path}")
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            # Load properties file with extended interpolation
            config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            config.read(config_path)

            # Load MongoDB collection names
            if 'mongodb' in config:
                self.mongo_questions_collection = config['mongodb'].get('mongo_questions_collection')
                self.mongo_output_collection_name = config['mongodb'].get('mongo_output_collection')
                print(f"✓ MongoDB collection names loaded:")
                print(f"  - Questions collection: {self.mongo_questions_collection}")
                print(f"  - Output collection: {self.mongo_output_collection_name}")

            # Load all other configurations directly
            for section in config.sections():
                for key, value in config.items(section):
                    if key not in ['mongo_questions_collection', 'mongo_output_collection']:  # Skip these as they're handled above
                        setattr(self, key, value)

            # Initialize DBConfig with the loaded context
            DBConfig.initialize_from_context(self)
            print("✓ DBConfig initialized with context")

            # Verify the properties were set
            required_props = ['task_name', 'skill_ids', 'num_questions', 'prompt_type', 'output_mode']
            for prop in required_props:
                if hasattr(self, prop):
                    print(f"✓ {prop}: {getattr(self, prop)}")
                else:
                    print(f"✗ {prop} not found in context")

            # Convert skill_ids from string to list of integers
            skill_ids_str = getattr(self, 'skill_ids', '')
            self.skill_ids = [int(sid.strip()) for sid in skill_ids_str.split(',') if sid.strip().isdigit()]

            # Convert temperature from string to float
            self.temperature = float(getattr(self, 'temperature', '0.2'))

            # Load Bloom's level specific prompt paths
            if 'bloom_prompts' in config:
                for level in ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating']:
                    system_key = f'{level.lower()}_system_prompt_path'
                    user_key = f'{level.lower()}_user_prompt_path'
                    
                    if system_key in config['bloom_prompts']:
                        self.bloom_prompt_paths[level]['system'] = config['bloom_prompts'][system_key]
                    if user_key in config['bloom_prompts']:
                        self.bloom_prompt_paths[level]['user'] = config['bloom_prompts'][user_key]

            # Set Bloom's Taxonomy levels from config or use default
            bloom_levels_str = getattr(self, 'bloom_levels', 'Remembering,Understanding,Applying,Analyzing,Evaluating')
            self.bloom_levels = [level.strip() for level in bloom_levels_str.split(',')]
            print(f"✓ Bloom's Taxonomy levels: {self.bloom_levels}")

        except FileNotFoundError as e:
            print(f"Warning: {str(e)}")
            raise
        except Exception as e:
            print(f"Error loading application configuration: {e}")
            raise

    def initialize(self):
        """Initialize all connections and services"""
        try:
            # Initialize database connections
            print("\nInitializing database connections...")
            self.mysql_conn = get_mysql_connection()
            print("✓ MySQL connection established")
            
            self.mongo_client, self.mongo_db = get_mongo_connection()
            print("✓ MongoDB connection established")
            
            self.mongo_operations = MongoOperations()
            print("✓ MongoOperations initialized")
            
            self.sql_operations = SQLOperations()
            print("✓ SQLOperations initialized")
            
            # Validate required properties
            if not hasattr(self, 'task_name') and not hasattr(self, 'skill_ids'):
                raise ValueError("Either task_name or skill_ids must be provided in task_config.json")

            # Extract LLM model parameters
            self.llm_model_params = {
                "llm_model": getattr(self, 'llm_model', None),
                "openai_llm_model": getattr(self, 'openai_llm_model', None),
                "gemini_llm_model": getattr(self, 'gemini_llm_model', None),
                "deepseek_llm_model": getattr(self, 'deepseek_llm_model', None),
                "anthropic_llm_model": getattr(self, 'anthropic_llm_model', None),
                "grok_llm_model": getattr(self, 'grok_llm_model', None),
                "temperature": getattr(self, 'temperature', 0.2)
            }
            print("✓ LLM model parameters loaded")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def resolve_skills_from_context(self):
        """Resolve skills from the context based on skill_ids or task_name"""
        print("\nResolving skills from context...")
        print(f"Task name: {getattr(self, 'task_name', None)}")
        print(f"Skill IDs: {getattr(self, 'skill_ids', None)}")
        
        try:
            # Priority 1: Check for skill_ids
            if hasattr(self, 'skill_ids') and self.skill_ids:
                print("\nUsing skill_ids as priority...")
                return self.sql_operations.get_skills_by_ids(self.skill_ids)

            # Priority 2: Check for task_name
            elif hasattr(self, 'task_name') and self.task_name:
                print("\nUsing task_name as fallback...")
                return self.sql_operations.get_skills_by_task_name(self.task_name)

            else:
                print("Error: Neither skill_ids nor task_name provided in context")
                raise ValueError("Either skill_ids or task_name must be provided in context.")

        except Exception as e:
            print(f"Error resolving skills: {str(e)}")
            raise

    def _format_learning_objectives(self, skill_data: dict) -> tuple[List[str], str]:
        """
        Get learning objectives from course framework using subject and skill_name from skill_data
        Args:
            skill_data: Dictionary containing skill information including subject and skill_name
        Returns:
            Tuple of (list of learning objectives, skill topic)
        """
        learning_objectives = []
        try:
            # Get objectives from course framework using subject and skill_name
            subject = skill_data.get('subject')
            skill_name = skill_data.get('skill_name')
            
            if subject and skill_name:
                learning_objectives = self.mongo_operations.get_unit_objectives(
                    subject=subject,
                    unit=skill_name
                )
                print(f"\nRetrieved {len(learning_objectives)} objectives for subject: {subject}, unit: {skill_name}")
            
            # Use skill_name as the topic
            skill = skill_name
            
            print(f"\nUsing skill as topic: {skill}")
            return learning_objectives, skill

        except Exception as e:
            print(f"Error getting learning objectives: {e}")
            return [], skill_data.get('skill_name', '')

    def get_skill_topic_parameters(self, skills_data):
        """
        Get skill and topic parameters for all skills.
        Args:
            skills_data: List of skill data dictionaries
        Returns:
            List of dictionaries containing skill and topic information
        """
        print("\n" + "="*50)
        print("Getting skill and topic parameters")
        print("="*50)
        
        skill_topic_params = []
        for skill_data in skills_data:
            print(f"\nProcessing skill: {skill_data['skill_name']}")
            
            # Format learning objectives using only skill_data
            learning_objectives, skill = self._format_learning_objectives(skill_data)
            
            skill_topic_params.append({
                'skill_id': skill_data["skill_id"],
                'skill': skill_data["skill_name"],
                'subject_area': skill_data["subject_area"],
                'subject': skill_data["subject"],
                'task_name': skill_data["task_name"],
                'learning_objectives': learning_objectives
            })
            
            print(f"\nAdded parameters for skill: {skill}")
            print(f"Number of suggested skills: {len(learning_objectives)}")
                
        print(f"\nTotal skill-topic parameters created: {len(skill_topic_params)}")
        print("="*50 + "\n")
        return skill_topic_params

    def prepare_llm_parameters(self, skill_topic_params, sample_questions):
        """
        Prepare LLM parameters from skill and topic parameters, including sample questions
        Args:
            skill_topic_params: List of dictionaries containing skill and topic information
            sample_questions: List of dictionaries containing sample questions
        Returns:
            List of dictionaries containing parameters for content generation
        """
        print("\n" + "="*50)
        print("Preparing LLM parameters")
        print("="*50)
        
        parameters_list = []
        for params in skill_topic_params:
            print(f"\nProcessing parameters for skill: {params['skill']}")
            
            llm_params = {
                'subject_id': params['skill_id'],
                'subject': params.get('subject', 'Mathematics'),  # Default to Mathematics if not present
                'subject_area_id': params.get('subject_area_id', None),
                'subject_area': params.get('subject_area', 'Calculus'),  # Default to Calculus if not present
                'skill_id': params['skill_id'],
                'skill': params['skill'],
                'learning_objectives': params['learning_objectives'],
                'num_questions': self.num_questions,
                'bloom_levels': params.get('bloom_levels', self.bloom_levels)  # Use from params or fallback to context
            }
            
            # Only add sample questions if they exist and are not empty
            if sample_questions and len(sample_questions) > 0:
                llm_params['sample_questions'] = sample_questions
                print(f"Added {len(sample_questions)} sample questions")
            else:
                print("No sample questions to add - ignoring empty sample_questions")
            
            parameters_list.append({
                'skill_id': params['skill_id'],
                'skill': params['skill'],
                'parameters': llm_params
            })
            
            print(f"Added parameters for topic: {params['skill']}")
        
        print(f"\nTotal parameter sets created: {len(parameters_list)}")
        print("="*50 + "\n")
        return parameters_list

    def get_prompts(self, parameters):
        """
        Get system and user prompts using the provided prompt builder
        Args:
            parameters (dict): Parameters for prompt generation
        Returns:
            tuple: (system_prompt, user_prompt) or (None, None) if failed
        """
        try:
            # Get the current Bloom's level from parameters
            bloom_level = parameters.get('bloom_levels', [self.bloom_levels[0]])[0]
            system_path, user_path = self.get_prompt_paths_for_bloom_level(bloom_level)
            
            # Create a temporary prompt builder for this request
            prompt_builder = PromptBuilder(
                system_prompt_template_path=system_path,
                user_prompt_template_path=user_path
            )
            
            # Get prompts from the temporary prompt builder
            system_prompt, user_prompt = prompt_builder.create_prompts(parameters)
            if system_prompt is None or user_prompt is None:
                print("Error: Could not create prompts.")
                return None, None
            return system_prompt, user_prompt
        except Exception as e:
            print(f"Error in get_prompts: {str(e)}")
            return None, None

    def fix_latex_escapes(self, text: str) -> str:
        """
        Escapes LaTeX-style backslashes properly for JSON parsing,
        without touching already escaped ones.
        """
        #print(f"Text before fixing escapes: {text}")
        # Fix all single-backslash LaTeX sequences (not like \n, \t, etc.)
        safe = re.sub(r'(?<!\\)\\(?![\\ntr"\/])', r'\\\\', text)
        # Specifically handle the \infty case
        safe = re.sub(r'(?<!\\)\\infty', r'\\\\infty', safe)
        #print(f"Text after fixing escapes: {safe}")
        return safe

    def generate_content_from_llm(self, system_prompt, user_prompt, llm_connections):
        """
        Generate content using LLM with the given prompts
        Args:
            system_prompt (str): System prompt for the LLM
            user_prompt (str): User prompt for the LLM
        Returns:
            str: Generated content or None if generation fails
        """
        try:
            # Retrieve LLM configuration from context
            llm_model = getattr(self, 'llm_model', 'openai')
            temperature = getattr(self, 'temperature', 0.2)

            # Call LLM API with configuration
            print("\nCalling LLM API with model: ", llm_model)
            ai_response_content = llm_connections.call_llm_api(provider=llm_model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
            #print(f"AI response content before parsing: {ai_response_content}")
            if not ai_response_content:
                print("Error: No response from LLM API")
                return None

            # Clean and validate the response
            try:
                # Fix LaTeX escapes before parsing
                ai_response_content = self.fix_latex_escapes(ai_response_content)

                # Remove any markdown code block markers if present
                ai_response_content = ai_response_content.strip('`')
                if ai_response_content.startswith('json'):
                    ai_response_content = ai_response_content[4:].strip()
                ai_response_content = ai_response_content.strip('`')
            
                return ai_response_content

            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON response from LLM: {e}")
                print("Raw response:", ai_response_content)
                return None

        except Exception as e:
            print(f"Error in generate_content_from_prompts: {str(e)}")
            return None

    def store_output_to_file(self, topic_name, skill_name, content):
        if content:
            llm_model = getattr(self, 'llm_model', 'default_model')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{llm_model}_questions_{skill_name}_{topic_name}_{timestamp}.txt".replace(" ", "_").replace("/", "_")
            os.makedirs("generated_questions", exist_ok=True)
            filepath = os.path.join("generated_questions", filename)
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Content written to file: {filepath}")

    def store_output_to_mongo(self, content):
        if content:
            try:
                # Validate collection name is set
                if not hasattr(self, 'mongo_output_collection_name') or not self.mongo_output_collection_name:
                    raise ValueError("MongoDB output collection name not set in configuration")
                
                parsed_json = json.loads(content)
                questions_collection = self.mongo_db[self.mongo_output_collection_name]
                
                # Handle both array of questions and single question object
                questions_to_process = []
                if isinstance(parsed_json, list):
                    questions_to_process = parsed_json
                elif isinstance(parsed_json, dict):
                    if 'questions' in parsed_json:
                        questions_to_process = parsed_json['questions']
                    else:
                        questions_to_process = [parsed_json]
                
                for question in questions_to_process:
                    if isinstance(question, dict):
                        question['created_at'] = datetime.utcnow()
                        print(f"Inserting question: {question['question']} in collection: {self.mongo_output_collection_name}")
                        questions_collection.insert_one(question)
                        print(f"Inserted question: {question['question']} in collection: {self.mongo_output_collection_name}")
                    else:
                        print(f"Skipped non-dictionary item: {question}")
                        
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed: {e}")
                print("Raw API response that caused the error:\n", content)
            except ValueError as e:
                print(f"Configuration error: {e}")
                raise
            except Exception as e:
                print(f"Error storing content to MongoDB: {e}")
                raise

    def _load_sample_questions(self, sample_questions_file):
        """
        Load and format sample questions from JSON file
        Args:
            sample_questions_file: Path to the sample questions JSON file
        Returns:
            str: Formatted sample questions section or empty string if loading fails
        """
        try:
            with open(sample_questions_file, 'r') as f:
                sample_questions = json.load(f)
            
            if sample_questions and isinstance(sample_questions, list):
                # Format sample questions as a string
                sample_questions_str = "\n".join([
                    f"Question {i+1}: {q.get('question', '')}"
                    for i, q in enumerate(sample_questions)
                ])
                return f"\n### Sample Questions:\n{sample_questions_str}"
            return ""
        except Exception as e:
            print(f"Error loading sample questions from {sample_questions_file}: {e}")
            return ""

    def get_prompt_paths_for_bloom_level(self, bloom_level):
        """
        Get the appropriate prompt paths for a given Bloom's level
        Args:
            bloom_level (str): The Bloom's taxonomy level
        Returns:
            tuple: (system_prompt_path, user_prompt_path)
        Raises:
            ValueError: If the Bloom's level is not found in the configuration
        """
        print(f"\nDEBUG: Getting prompt paths for Bloom's level: {bloom_level}")
        print(f"DEBUG: Available bloom_prompt_paths: {self.bloom_prompt_paths}")
        
        if bloom_level in self.bloom_prompt_paths:
            system_path = self.bloom_prompt_paths[bloom_level]['system']
            user_path = self.bloom_prompt_paths[bloom_level]['user']
            print(f"DEBUG: Found paths for {bloom_level}:")
            print(f"DEBUG: System path: {system_path}")
            print(f"DEBUG: User path: {user_path}")
            if system_path and user_path:
                return system_path, user_path
            else:
                print(f"DEBUG: Warning - One or both paths are None for {bloom_level}")
        
        raise ValueError(f"No prompt paths found for Bloom's level: {bloom_level}")


def generate_content_with_llm(context, skill_topic_params, sample_questions_section, llm_connections, prompt_builder=None):
    """
    Prepares LLM parameters and generates content using the LLM.
    Args:
        context: The global context
        skill_topic_params: Parameters for the skills
        sample_questions_section: Sample questions to include
        llm_connections: LLM connection handler
        prompt_builder: The prompt builder to use (required)
    """
    if not prompt_builder:
        raise ValueError("prompt_builder is required for generate_content_with_llm")
        
    print("Preparing LLM parameters...")
    llm_prompt_parameters_list = context.prepare_llm_parameters(skill_topic_params, [])
    
    print("Generating content...")
    all_contents = []
    for params in llm_prompt_parameters_list:
        # Add sample questions section to parameters
        params['parameters']['sample_questions_section'] = sample_questions_section
        print(f"Sample questions section: {sample_questions_section}")
        
        # Get prompts using the provided prompt builder
        system_prompt, user_prompt = prompt_builder.create_prompts(params['parameters'])
        if system_prompt is None or user_prompt is None:
            print(f"Failed to get prompts for skill: {params['skill']}")
            continue
        print(f"**System prompt: {system_prompt} \n\n **User prompt: {user_prompt}")
        
        # Generate content
        content = context.generate_content_from_llm(system_prompt, user_prompt, llm_connections)
        if content:
            all_contents.append(content)
    return all_contents

class BaseWorkflow:
    def __init__(self):
        print("Initializing workflow...")
        self.context = GlobalContext()  # This now handles both config loading and initialization
        
        # Validate context is properly loaded
        if not hasattr(self.context, 'task_name') and not hasattr(self.context, 'skill_ids'):
            raise ValueError("Either task_name or skill_ids must be provided in task_config.properties")
        
        print(f"Context loaded with task: {getattr(self.context, 'task_name', 'None')}")
        print(f"Skill IDs: {getattr(self.context, 'skill_ids', [])}")
        
        # Initialize LLM connections after context is ready
        self.llm_connections = LLMConnections(config=self.context.llm_model_params)
        print("✓ LLM connections initialized")

    def get_prompts(self, parameters):
        """Base method to be overridden by specific workflows"""
        raise NotImplementedError("Subclasses must implement get_prompts method")

    def write_content(self, contents):
        """Write generated content to a JSON file or MongoDB based on output_mode."""
        try:
            if self.context.output_mode == "file":
                # Create a single file for all questions
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Determine prefix based on workflow type
                prefix = "enhanced_questions" if isinstance(self, QuestionEnhanceWorkflow) else "generated_questions"
                filename = f"{prefix}_{timestamp}.json"
                
                os.makedirs("generated_questions", exist_ok=True)
                filepath = os.path.join("generated_questions", filename)
                
                # Combine all contents into a single JSON array
                all_questions = []
                for content in contents:
                    try:
                        parsed_content = json.loads(content)
                        if 'questions' in parsed_content:
                            all_questions.extend(parsed_content['questions'])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse content as JSON: {content}")
                        continue
                
                # Write all questions to a single file
                with open(filepath, 'w') as f:
                    json.dump({"questions": all_questions}, f, indent=2)
                #print(f"✅ All {prefix} written to file: {filepath}")

            elif self.context.output_mode == "mongo":
                print(f"Writing content to MongoDB")
                for content in contents:
                    parsed_content = json.loads(content)
                    if 'questions' in parsed_content:
                        for question in parsed_content['questions']:
                            self.context.store_output_to_mongo(json.dumps([question]))

            else:
                raise ValueError(f"Invalid output_mode: {self.context.output_mode}")

        except Exception as e:
            print(f"❌ Error writing content: {str(e)}")
            raise


class QuestionGenerationWorkflow(BaseWorkflow):
    def __init__(self):
        super().__init__()
        # Initialize prompt builder with first Bloom's level (will be updated per level)
        first_level = self.context.bloom_levels[0] if self.context.bloom_levels else 'Remembering'
        system_path, user_path = self.context.get_prompt_paths_for_bloom_level(first_level)
        self.prompt_builder = PromptBuilder(
            system_prompt_template_path=system_path,
            user_prompt_template_path=user_path
        )
        self.output_file = None
        self.all_questions = []

    def initialize_output_file(self):
        """Initialize the output file at the start of the run"""
        if self.context.output_mode == "file":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_questions_{timestamp}.json"
            os.makedirs("generated_questions", exist_ok=True)
            self.output_file = os.path.join("generated_questions", filename)
            # Initialize file with empty questions array
            with open(self.output_file, 'w') as f:
                json.dump({"questions": []}, f, indent=2)
            print(f"Initialized output file: {self.output_file}")

    def append_to_output_file(self, questions):
        """Append new questions to the output file"""
        if self.context.output_mode == "file" and self.output_file:
            try:
                # Read existing content
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                
                # Append new questions
                data['questions'].extend(questions)
                
                # Write back to file
                with open(self.output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✅ Appended {len(questions)} questions to {self.output_file}")
            except Exception as e:
                print(f"❌ Error appending to file: {str(e)}")

    def load_sample_questions(self):
        sample_questions_section = ""
        sample_questions_file = getattr(self.context, 'sample_questions_file', None)
        print(f"Sample questions file: {sample_questions_file}")
        if sample_questions_file:
            sample_questions_section = self.context._load_sample_questions(sample_questions_file)
        return sample_questions_section

    def run(self):
        try:
            # Initialize output file if in file mode
            if self.context.output_mode == "file":
                self.initialize_output_file()

            # Step 1: Get all skills data
            skills_data = self.context.resolve_skills_from_context()
            print(f"Skills data in run method: {skills_data}")
            if not skills_data:
                raise ValueError("No skills data found for the given context")

            # Step 2: Load sample questions once
            sample_questions_section = self.load_sample_questions()

            # Step 3: Outer Loop - Process each skill
            for skill_data in skills_data:
                print(f"\nProcessing skill: {skill_data['skill_name']}")
                
                # Get base parameters for this skill
                skill_params = self.context.get_skill_topic_parameters([skill_data])[0]
                
                # Step 4: Inner Loop - Process each Bloom's level
                print(f"\nDEBUG: Available Bloom's levels: {self.context.bloom_levels}")
                for bloom_level in self.context.bloom_levels:
                    print(f"\nDEBUG: Starting generation for Bloom's level: {bloom_level}")
                    
                    # Update prompt builder with Bloom's level specific prompts
                    system_path, user_path = self.context.get_prompt_paths_for_bloom_level(bloom_level)
                    print(f"DEBUG: Creating new PromptBuilder with paths:")
                    print(f"DEBUG: System path: {system_path}")
                    print(f"DEBUG: User path: {user_path}")
                    
                    self.prompt_builder = PromptBuilder(
                        system_prompt_template_path=system_path,
                        user_prompt_template_path=user_path
                    )
                    
                    # Add current Bloom's level to parameters
                    skill_params["bloom_levels"] = [bloom_level]
                    
                    # Generate questions for this skill and Bloom's level, passing the prompt builder
                    all_contents = generate_content_with_llm(
                        self.context,
                        [skill_params],
                        sample_questions_section,
                        self.llm_connections,
                        prompt_builder=self.prompt_builder  # Pass the prompt builder
                    )
                    
                    # Process and write output for this batch
                    if all_contents:
                        print(f"Processing content for skill: {skill_data['skill_name']}, Bloom's level: {bloom_level}")
                        self.process_and_write_content(all_contents)
                        print(f"✅ Generated questions for skill: {skill_data['skill_name']}, Bloom's level: {bloom_level}")
                    else:
                        print(f"❌ Failed to generate questions for skill: {skill_data['skill_name']}, Bloom's level: {bloom_level}")

            print("✅ Question Generation Completed for all skills and Bloom's levels")

        except Exception as e:
            print(f"❌ Error in main workflow: {str(e)}")
            raise

    def process_and_write_content(self, contents):
        """Process and write content based on output mode"""
        try:
            if self.context.output_mode == "file":
                # Process each content and append to file
                for content in contents:
                    try:
                        parsed_content = json.loads(content)
                        if 'questions' in parsed_content:
                            self.append_to_output_file(parsed_content['questions'])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse content as JSON: {content}")
                        continue

            elif self.context.output_mode == "mongo":
                print(f"Writing content to MongoDB")
                for content in contents:
                    try:
                        parsed_content = json.loads(content)
                        if 'questions' in parsed_content:
                            for question in parsed_content['questions']:
                                self.context.store_output_to_mongo(json.dumps([question]))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse content as JSON: {content}")
                        continue
                print("✅ Questions stored in MongoDB")

            else:
                raise ValueError(f"Invalid output_mode: {self.context.output_mode}")

        except Exception as e:
            print(f"❌ Error processing content: {str(e)}")
            raise


class QuestionEnhanceWorkflow(BaseWorkflow):
    def __init__(self):
        super().__init__()
        self.context = GlobalContext()
        self.context.initialize()
        # Initialize prompt builder with first Bloom's level (will be updated per level)
        first_level = self.context.bloom_levels[0] if self.context.bloom_levels else 'Remembering'
        system_path, user_path = self.context.get_prompt_paths_for_bloom_level(first_level)
        self.prompt_builder = PromptBuilder(
            system_prompt_template_path=system_path,
            user_prompt_template_path=user_path
        )

    def get_prompts(self, parameters):
        """
        Override get_prompts to use enhancement-specific prompt creation
        Args:
            parameters: Dictionary containing parameters for prompt enhancement
        Returns:
            Tuple of (system_prompt, user_prompt) or (None, None) if failed
        """
        try:
            return self.prompt_builder.create_enhance_prompts(parameters)
        except Exception as e:
            print(f"Error in get_prompts: {str(e)}")
            return None, None

    def enhance_question(self, question):
        """Enhance a single question"""
        try:
            # Get skill details from SQL database
            skill_data = self.context.sql_operations.get_skills_by_ids([question['skill_id']])
            if not skill_data:
                print(f"Warning: No skill data found for skill_id: {question['skill_id']}")
                return None
                
            skill = skill_data[0]  # Get the first (and should be only) skill
            
            # Format learning objectives from additional details
            learning_objectives, skill_topic = self.context._format_learning_objectives(skill)
            
            # Prepare parameters for the prompts
            parameters = {
                'question': question['question'],
                'subject': question['subject'],
                'subject_id': question.get('subject_id'),
                'subject_area': question['subject_area'],
                'subject_area_id': question.get('subject_area_id'),
                'skill': question['skill'],
                'skill_name': question['skill_name'],
                'skill_id': question['skill_id'],
                'multiple_choices': question['multiple_choices'],
                'correct_answer': question['correct_answer'],
                'level': question['level'],
                'level_num': question['level_num'],
                'requires_diagram': question.get('requires_diagram', False),
                'learning_objectives': learning_objectives,
                'skill_topic': skill_topic
            }
            
            # Get prompts using the overridden get_prompts method
            system_prompt, user_prompt = self.get_prompts(parameters)
            if system_prompt is None or user_prompt is None:
                print(f"Failed to get prompts for question: {question['question']}")
                return None

            # Generate enhanced content
            enhanced_content = self.context.generate_content_from_llm(system_prompt, user_prompt, self.llm_connections)
            if enhanced_content is None:
                print(f"Failed to enhance question: {question['question']}")
                return None

            return enhanced_content
            
        except Exception as e:
            print(f"Error enhancing question: {str(e)}")
            return None

    def run(self):
        try:
            print("Starting Question Enhancement Workflow...")
            
            # Get questions for specific skill
            pskill = "Infinite Sequences and Series"
            questions = self.context.mongo_operations.get_questions_by_skill(skill=pskill, limit=None)
            print(f"Found {len(questions)} questions for skill: {pskill}")
            
            # Process each question
            enhanced_contents = []
            for question in questions:
                enhanced_content = self.enhance_question(question)
                if enhanced_content:
                    #print(f"***Enhanced content: {enhanced_content}")
                    enhanced_contents.append(enhanced_content)
                    print(f"Successfully enhanced question: {question['question']}")
                else:
                    print(f"Failed to enhance question: {question['question']}")
            
            # Write enhanced content
            if enhanced_contents:
                print("\nWriting enhanced content...")
                self.write_content(enhanced_contents)
                print(f"✅ Successfully enhanced and wrote {len(enhanced_contents)} questions")
            else:
                print("❌ No questions were successfully enhanced")
                
            print("✅ Question Enhancement Completed.")

        except Exception as e:
            print(f"❌ Error in enhancement workflow: {str(e)}")
            raise


def main():
    # Initialize context to get workflow type
    context = GlobalContext()
    context.initialize()
    
    # Get workflow type from context
    workflow_type = getattr(context, 'workflow_type', 'generate')  # Default to 'generate' if not specified
    print(f"Running workflow type: {workflow_type}")
    
    if workflow_type == "generate":
        workflow = QuestionGenerationWorkflow()
    else:
        workflow = QuestionEnhanceWorkflow()
        
    workflow.run()

if __name__ == '__main__':
    main()
