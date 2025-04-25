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
from pipeline.pipeline_utils.db_connections import get_mysql_connection, get_mongo_connection, get_chroma_connection, DBConfig, save_to_mongodb, save_to_chroma
from pipeline.generation_pipeline.build_prompt import PromptBuilder
from pipeline.pipeline_utils.llm_connections import LLMConnections

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# ------------------ Global Context ------------------
class GlobalContext:
    def __init__(self):
        # Database connections
        self.mysql_conn = None
        self.mongo_client = None
        self.mongo_db = None
        self.chroma_client = None
        self.chroma_collection = None
        self.prompt_builder = None

        # Collection names
        self.mongo_collection_name = DBConfig.MONGO_QUESTIONS_COLLECTION
        self.chroma_collection_name = DBConfig.CHROMA_COLLECTION_NAME

        # Load application configurations from JSON
        self._load_app_config()

    def _load_app_config(self):
        """Load application configurations from task_config.properties"""
        try:
            # Get the pipeline directory
            pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(pipeline_dir, 'task_config.properties')
            print(f"Looking for config file at: {config_path}")
            print(f"Pipeline directory: {pipeline_dir}")
            
            # Check if file exists
            if not os.path.exists(config_path):
                print(f"Error: Config file not found at {config_path}")
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            # Load properties file with extended interpolation
            config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            config.read(config_path)

            # Load all configurations directly
            for section in config.sections():
                for key, value in config.items(section):
                    setattr(self, key, value)

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

            # Load template paths from configuration
            self.generation_system_prompt_path = getattr(self, 'generation_system_prompt_path', 'prompts/generation_system_prompt_template_mc.txt')
            self.generation_user_prompt_path = getattr(self, 'generation_user_prompt_path', 'prompts/generation_user_prompt_template_mc.txt')

        except FileNotFoundError as e:
            print(f"Warning: {str(e)}")
            raise
        except Exception as e:
            print(f"Error loading application configuration: {e}")
            raise

    def initialize(self):
        """Initialize all connections and services"""
        # Initialize database connections
        self.mysql_conn = get_mysql_connection()
        self.mongo_client, self.mongo_db = get_mongo_connection()
        self.chroma_client, self.chroma_collection = get_chroma_connection()
        
        # Initialize services
        self.prompt_builder = PromptBuilder(
            system_prompt_template_path=self.generation_system_prompt_path,
            user_prompt_template_path=self.generation_user_prompt_path
        )

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

    def resolve_skills_from_context(self):
        """Resolve skills from the context based on skill_ids or task_name"""
        print("\nResolving skills from context...")
        print(f"Task name: {getattr(self, 'task_name', None)}")
        print(f"Skill IDs: {getattr(self, 'skill_ids', None)}")
        
        cursor = self.mysql_conn.cursor()
        skills_data = []

        try:
            # Priority 1: Check for skill_ids
            if hasattr(self, 'skill_ids') and self.skill_ids:
                print("\nUsing skill_ids as priority...")
                query = """
                    SELECT
                        s.skill_id,
                        s.skill_name,
                        s.additional_details,
                        s.subject_area,
                        s.subject,
                        t.task_name
                    FROM
                        adaptive_skills s
                    LEFT JOIN
                        adaptive_task_skills t ON s.skill_id = t.skill_id
                    WHERE
                        s.skill_id = %s
                """
                for skill_id in self.skill_ids:
                    print(f"Querying for skill_id: {skill_id}")
                    cursor.execute(query, (skill_id,))
                    result = cursor.fetchone()
                    if result:
                        print(f"Found skill: {result[1]} (ID: {result[0]})")
                        skills_data.append({
                            "skill_id": result[0],
                            "skill_name": result[1],
                            "skill_additional_details": result[2],
                            "subject_area": result[3],
                            "subject": result[4],
                            "task_name": result[5]
                        })
                    else:
                        print(f"No skill found for skill_id: {skill_id}")
                    # Ensure all results are consumed
                    while cursor.fetchone() is not None:
                        pass
                print(f"\nTotal skills found from skill_ids: {len(skills_data)}")
                return skills_data

            # Priority 2: Check for task_name
            elif hasattr(self, 'task_name') and self.task_name:
                print("\nUsing task_name as fallback...")
                query = """
                    SELECT
                        s.skill_id,
                        s.skill_name,
                        s.additional_details,
                        s.subject_area,
                        s.subject,
                        t.task_name
                    FROM
                        adaptive_skills s
                    INNER JOIN
                        adaptive_task_skills t ON s.skill_id = t.skill_id
                    WHERE
                        t.task_name = %s
                """
                print(f"Querying for task: {self.task_name}")
                cursor.execute(query, (self.task_name,))
                results = cursor.fetchall()
                for result in results:
                    print(f"Found skill: {result[1]} (ID: {result[0]})")
                    skills_data.append({
                        "skill_id": result[0],
                        "skill_name": result[1],
                        "skill_additional_details": result[2],
                        "subject_area": result[3],
                        "subject": result[4],
                        "task_name": result[5]
                    })
                #print(f"\nTotal skills found from task: {len(skills_data)}")
                return skills_data

            else:
                print("Error: Neither skill_ids nor task_name provided in context")
                raise ValueError("Either skill_ids or task_name must be provided in context.")

        finally:
            cursor.close()
  
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
            
            # Initialize learning_objectives with empty list
            learning_objectives = []
            
            # Get the additional details
            additional_details = skill_data.get('skill_additional_details', {})
            if not additional_details:
                print("Warning: No additional details found for skill")
                continue
                
            try:
                # Parse the additional details
                details = json.loads(additional_details)
                print("\nParsed additional details:")
                print(json.dumps(details, indent=2))
                
                # Use unit as the topic
                skill = details.get('unit')
                if not skill:
                    print("Warning: No unit found in additional details")
                    continue
                    
                print(f"\nUsing unit as topic: {skill}")
                
                # Get objectives as suggested skills
                if 'objectives' in details:
                    for objective in details['objectives']:
                        if 'description' in objective:
                            learning_objectives.append(objective['description'])
                            print(f"Added suggested skill: {objective['description']}")
                
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
                
            except json.JSONDecodeError as e:
                print(f"Error parsing additional details: {e}")
                # Even if parsing fails, we still add the skill with empty learning_objectives
                skill_topic_params.append({
                    'skill_id': skill_data["skill_id"],
                    'skill': skill_data["skill_name"],
                    'subject_area': skill_data["subject_area"],
                    'subject': skill_data["subject"],
                    'task_name': skill_data["task_name"],
                    'learning_objectives': learning_objectives
                })
                continue
                
        print(f"\nTotal skill-topic parameters created: {len(skill_topic_params)}")
        print("="*50 + "\n")
        return skill_topic_params

    def calculate_question_similarity(self, question1: Dict, question2: Dict) -> float:
        """
        Calculate similarity score between two questions based on multiple factors
        Args:
            question1: First question dictionary
            question2: Second question dictionary
        Returns:
            Similarity score between 0 and 1
        """
        # Initialize weights for different similarity components
        weights = {
            'topic': 0.3,
            'skill': 0.3,
            'difficulty': 0.2,
            'structure': 0.2
        }
        
        # Calculate topic similarity
        topic_sim = 1.0 if question1.get('topic') == question2.get('topic') else 0.0
        
        # Calculate skill similarity
        skill_sim = 1.0 if question1.get('skill') == question2.get('skill') else 0.0
        
        # Calculate difficulty similarity
        diff1 = question1.get('difficulty', 'medium')
        diff2 = question2.get('difficulty', 'medium')
        difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
        diff_sim = 1.0 - abs(difficulty_map.get(diff1, 1) - difficulty_map.get(diff2, 1)) / 2.0
        
        # Calculate structure similarity
        struct_sim = 0.0
        if 'multiple_choices' in question1 and 'multiple_choices' in question2:
            # Compare number of choices
            struct_sim += 0.5 * (1.0 if len(question1['multiple_choices']) == len(question2['multiple_choices']) else 0.0)
            # Compare question length
            len1 = len(question1.get('question_text', ''))
            len2 = len(question2.get('question_text', ''))
            struct_sim += 0.5 * (1.0 - abs(len1 - len2) / max(len1, len2))
        
        # Calculate weighted similarity
        total_sim = (
            weights['topic'] * topic_sim +
            weights['skill'] * skill_sim +
            weights['difficulty'] * diff_sim +
            weights['structure'] * struct_sim
        )
        
        return total_sim

    

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
                'num_questions': self.num_questions
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
        Get system and user prompts from prompt builder
        Args:
            parameters (dict): Parameters for prompt generation
        Returns:
            tuple: (system_prompt, user_prompt) or (None, None) if failed
        """
        try:
            # Get prompts from prompt builder
            system_prompt, user_prompt = self.prompt_builder.create_prompts(parameters)
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
        print(f"Text after fixing escapes: {safe}")
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

                print(f"AI response content after cleaning: {ai_response_content}")
                # Try to parse as JSON to validate
                parsed_json = json.loads(ai_response_content)
                
                # Validate the structure
                if not isinstance(parsed_json, dict):
                    print("Error: Response is not a JSON object")
                    return None
                
                if 'questions' not in parsed_json:
                    print("Error: Response missing 'questions' key")
                    return None
                
                if not isinstance(parsed_json['questions'], list):
                    print("Error: 'questions' is not a list")
                    return None
                #print(f"******ai_response_content: {ai_response_content}")
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

    def store_output_to_mongo(self, content, skill_id):
        if content:
            try:
                parsed_json = json.loads(content)
                questions_collection = self.mongo_db[self.mongo_collection_name]
                if 'questions' in parsed_json and isinstance(parsed_json['questions'], list):
                    for question in parsed_json['questions']:
                        if isinstance(question, dict):
                            question['skill_id'] = skill_id
                            question['created_at'] = datetime.utcnow()
                            print(f"Inserting question: {question['question']}")
                            questions_collection.insert_one(question)
                        else:
                            print(f"Skipped non-dictionary item: {question}")
                else:
                    print(f"Unexpected JSON structure: {parsed_json}")
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed: {e}")
                print("Raw API response that caused the error:\n", content)

    def write_content(self, contents):
        """Write generated content to a JSON file or MongoDB based on output_mode."""
        try:
            if self.output_mode == "file":
                for index, content in enumerate(contents, 1):
                    topic_name = getattr(self, 'task_name', 'general')
                    skill_name = getattr(self, 'skill_ids', ['unknown'])[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.llm_model}_questions_{skill_name}_{topic_name}_{timestamp}_batch{index}.txt".replace(" ", "_").replace("/", "_")
                    os.makedirs("generated_questions", exist_ok=True)
                    filepath = os.path.join("generated_questions", filename)
                    with open(filepath, 'w') as f:
                        f.write(content)
                    print(f"✅ Content written to file: {filepath}")

            elif self.output_mode == "mongo":
                for content in contents:
                    parsed_content = json.loads(content)
                    if 'questions' in parsed_content:
                        for question in parsed_content['questions']:
                            if 'skill_id' in question:
                                self.store_output_to_mongo(json.dumps([question]), question['skill_id'])

            else:
                raise ValueError(f"Invalid output_mode: {self.output_mode}")

        except Exception as e:
            print(f"❌ Error writing content: {str(e)}")
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



def generate_content_with_llm(context, skill_topic_params, sample_questions_section, llm_connections):
    """
    Prepares LLM parameters and generates content using the LLM.
    """
    print("Preparing LLM parameters...")
    llm_prompt_parameters_list = context.prepare_llm_parameters(skill_topic_params, [])
    #print(f"LLM prompt parameters list: {llm_prompt_parameters_list}")
    
    print("Generating content...")
    all_contents = []
    for params in llm_prompt_parameters_list:
        # Add sample questions section to parameters
        params['parameters']['sample_questions_section'] = sample_questions_section
        print(f"Sample questions section: {sample_questions_section}")
        # Get prompts
        system_prompt, user_prompt = context.get_prompts(params['parameters'])
        if system_prompt is None or user_prompt is None:
            print(f"Failed to get prompts for skill: {params['skill']}")
            continue
        print(f"System prompt: {system_prompt} \n\n User prompt: {user_prompt}")
        # Generate content
        content = context.generate_content_from_llm(system_prompt, user_prompt, llm_connections)
        if content:
            all_contents.append(content)
    return all_contents

class BaseWorkflow:
    def __init__(self):
        self.context = self.initialize_context()
        self.llm_connections = LLMConnections(config=self.context.llm_model_params)

    def initialize_context(self):
        print("Initializing context...")
        context = GlobalContext()  # Creates context and loads task_config.properties
        context.initialize()       # Initializes connections and services
        
        # Validate context is properly loaded
        if not hasattr(context, 'task_name') and not hasattr(context, 'skill_ids'):
            raise ValueError("Either task_name or skill_ids must be provided in task_config.properties")
        
        print(f"Context loaded with task: {getattr(context, 'task_name', 'None')}")
        print(f"Skill IDs: {getattr(context, 'skill_ids', [])}")
        return context

    def fix_latex_escapes(self, text: str) -> str:
        """Escapes LaTeX-style backslashes properly for JSON parsing."""
        safe = re.sub(r'(?<!\\)\\(?![\\ntr"\/])', r'\\\\', text)
        safe = re.sub(r'(?<!\\)\\infty', r'\\\\infty', safe)
        print(f"Text after fixing escapes: {safe}")
        return safe

    def get_prompts(self, parameters):
        """Get system and user prompts from prompt builder"""
        try:
            system_prompt, user_prompt = self.context.prompt_builder.create_prompts(parameters)
            if system_prompt is None or user_prompt is None:
                print("Error: Could not create prompts.")
                return None, None
            return system_prompt, user_prompt
        except Exception as e:
            print(f"Error in get_prompts: {str(e)}")
            return None, None

    def generate_content_from_llm(self, system_prompt, user_prompt):
        """Generate content using LLM with the given prompts"""
        try:
            llm_model = getattr(self.context, 'llm_model', 'openai')
            temperature = getattr(self.context, 'temperature', 0.2)

            print("\nCalling LLM API with model: ", llm_model)
            ai_response_content = self.llm_connections.call_llm_api(
                provider=llm_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature
            )

            if not ai_response_content:
                print("Error: No response from LLM API")
                return None

            try:
                ai_response_content = self.fix_latex_escapes(ai_response_content)
                ai_response_content = ai_response_content.strip('`')
                if ai_response_content.startswith('json'):
                    ai_response_content = ai_response_content[4:].strip()
                ai_response_content = ai_response_content.strip('`')

                print(f"AI response content after cleaning: {ai_response_content}")
                parsed_json = json.loads(ai_response_content)
                
                if not isinstance(parsed_json, dict):
                    print("Error: Response is not a JSON object")
                    return None
                
                if 'questions' not in parsed_json:
                    print("Error: Response missing 'questions' key")
                    return None
                
                if not isinstance(parsed_json['questions'], list):
                    print("Error: 'questions' is not a list")
                    return None

                return ai_response_content

            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON response from LLM: {e}")
                print("Raw response:", ai_response_content)
                return None

        except Exception as e:
            print(f"Error in generate_content_from_prompts: {str(e)}")
            return None

    def write_content(self, contents):
        """Write generated content to a JSON file or MongoDB based on output_mode."""
        try:
            if self.context.output_mode == "file":
                for index, content in enumerate(contents, 1):
                    topic_name = getattr(self.context, 'task_name', 'general')
                    skill_name = getattr(self.context, 'skill_ids', ['unknown'])[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.context.llm_model}_questions_{skill_name}_{topic_name}_{timestamp}_batch{index}.txt".replace(" ", "_").replace("/", "_")
                    os.makedirs("generated_questions", exist_ok=True)
                    filepath = os.path.join("generated_questions", filename)
                    with open(filepath, 'w') as f:
                        f.write(content)
                    print(f"✅ Content written to file: {filepath}")

            elif self.context.output_mode == "mongo":
                for content in contents:
                    parsed_content = json.loads(content)
                    if 'questions' in parsed_content:
                        for question in parsed_content['questions']:
                            if 'skill_id' in question:
                                self.context.store_output_to_mongo(json.dumps([question]), question['skill_id'])

            else:
                raise ValueError(f"Invalid output_mode: {self.context.output_mode}")

        except Exception as e:
            print(f"❌ Error writing content: {str(e)}")
            raise


class QuestionGenerationWorkflow(BaseWorkflow):
    def resolve_skills(self):
        print("Resolving skills...")
        skills_data = self.context.resolve_skills_from_context()
        print(f"Skills data: {skills_data}")
        if not skills_data:
            raise ValueError("No skills data found for the given context")
        return skills_data

    def prepare_parameters(self, skills_data):
        print("Preparing parameters...")
        skill_topic_params = self.context.get_skill_topic_parameters(skills_data)
        print(f"Skill topic parameters: {skill_topic_params}")
        return skill_topic_params

    def load_sample_questions(self):
        sample_questions_section = ""
        sample_questions_file = getattr(self.context, 'sample_questions_file', None)
        print(f"Sample questions file: {sample_questions_file}")
        if sample_questions_file:
            sample_questions_section = self.context._load_sample_questions(sample_questions_file)
        return sample_questions_section

    def run(self):
        try:
            # Step 3: Resolve Skills
            skills_data = self.resolve_skills()

            # Step 4: Prepare Parameters
            skill_topic_params = self.prepare_parameters(skills_data)

            # Step 5: Load Sample Questions
            sample_questions_section = self.load_sample_questions()

            # Step 6: Generate Content
            all_contents = generate_content_with_llm(self.context, skill_topic_params, sample_questions_section, self.llm_connections)

            # Step 7: Write Content
            print("Writing content...")
            self.write_content(all_contents)

            print("✅ Question Generation Completed.")

        except Exception as e:
            print(f"❌ Error in main workflow: {str(e)}")
            raise


class QuestionEnhanceWorkflow(BaseWorkflow):
    def __init__(self):
        super().__init__()
        self.questions = self.get_questions_from_mongo()

    def get_questions_from_mongo(self, subject=None):
        """Read from MongoDB. Filter by subject if provided, return list of questions"""
        mongo_client = get_mongo_connection(DBConfig.MONGO_DB_NAME)
        questions = mongo_client.find_many(collection_name="questions", query={"subject": subject})
        return questions

    def enhance_question(self, question):
        """Enhance a single question"""
        # TODO: Implement question enhancement logic
        pass

    def run(self):
        try:
            print("Starting Question Enhancement Workflow...")
            
            for question in self.questions:
                enhanced_question = self.enhance_question(question)
                # TODO: Save enhanced question back to MongoDB
                
            print("✅ Question Enhancement Completed.")

        except Exception as e:
            print(f"❌ Error in enhancement workflow: {str(e)}")
            raise


def main():
    # Choose which workflow to run
    workflow_type = "generate"  # or "generate"
    
    if workflow_type == "generate":
        workflow = QuestionGenerationWorkflow()
    else:
        workflow = QuestionEnhanceWorkflow()
        
    workflow.run()

if __name__ == '__main__':
    main()
