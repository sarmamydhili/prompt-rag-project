import os
import sys

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
import mysql.connector
from tqdm import tqdm

# Local imports
import config
from pipeline.pipeline_utils.llm_connections import call_llm_api
from pipeline.generation_pipeline.build_prompt import PromptBuilder
from pipeline.generation_pipeline.retrieve_similar import QuestionRetriever
from pipeline.pipeline_utils.db_connections import get_mysql_connection, get_mongo_connection, get_chroma_connection, DBConfig, save_to_mongodb, save_to_chroma
from pipeline.pipeline_utils.embed_questions import embed_question

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
        self.question_retriever = None

        # Collection names
        self.mongo_collection_name = DBConfig.MONGO_QUESTIONS_COLLECTION
        self.chroma_collection_name = DBConfig.CHROMA_COLLECTION_NAME

        # Load application configurations from JSON
        self._load_app_config()

    def _load_app_config(self):
        """Load application configurations from task_config.json"""
        try:
            # Get the pipeline directory
            pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(pipeline_dir, 'task_config.json')
            print(f"Looking for config file at: {config_path}")
            print(f"Pipeline directory: {pipeline_dir}")
            
            # Check if file exists
            if not os.path.exists(config_path):
                print(f"Error: Config file not found at {config_path}")
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                #print("\nRaw config data:")
                print(json.dumps(config_data, indent=2))
                
                # Load all configurations directly
                for key, value in config_data.items():
                    #print(f"\nSetting {key} to: {value}")
                    setattr(self, key, value)
                
                # Verify the properties were set
                #print("\nVerifying context properties:")
                required_props = ['task_name', 'skill_ids', 'num_questions', 'prompt_type', 'output_mode']
                for prop in required_props:
                    if hasattr(self, prop):
                        print(f"✓ {prop}: {getattr(self, prop)}")
                    else:
                        print(f"✗ {prop} not found in context")
                
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
        self.prompt_builder = PromptBuilder()
        self.question_retriever = QuestionRetriever()

        # Validate required properties
        if not hasattr(self, 'task_name') and not hasattr(self, 'skill_ids'):
            raise ValueError("Either task_name or skill_ids must be provided in task_config.json")

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
                print(f"\nTotal skills found from task: {len(skills_data)}")
                return skills_data

            else:
                print("Error: Neither skill_ids nor task_name provided in context")
                raise ValueError("Either skill_ids or task_name must be provided in context.")

        finally:
            # Ensure cursor is closed
            cursor.close()

    def read_topics_for_skill(self, skill_data):
        """
        Extract topics from skill_additional_details.
        Args:
            skill_data: Dictionary containing skill information including skill_additional_details
        Returns:
            Tuple of (topics, additional_details) where:
            - topics is a list of objective descriptions
            - additional_details is the parsed JSON structure
        """
        print("\n" + "="*50)
        print("Starting read_topics_for_skill")
        print("="*50)
        
        print(f"\nInput skill_data:")
        print(json.dumps(skill_data, indent=2))
        
        # Get the skill_additional_details
        skill_details = skill_data.get('skill_additional_details', '')
        print(f"\nRaw skill_details:")
        print(skill_details)
        
        if not skill_details:
            print("Warning: No skill_additional_details found")
            return [], {}
            
        try:
            # Parse the JSON string
            additional_details = json.loads(skill_details)
            print("\nParsed additional_details:")
            print(json.dumps(additional_details, indent=2))
            
            # Extract topics from objectives
            topics = []
            if 'objectives' in additional_details:
                print("\nProcessing objectives:")
                for i, objective in enumerate(additional_details['objectives'], 1):
                    print(f"\nObjective {i}:")
                    print(json.dumps(objective, indent=2))
                    if 'description' in objective:
                        topics.append(objective['description'])
                        print(f"Added topic: {objective['description']}")
                    else:
                        print("No description found in objective")
            else:
                print("\nNo objectives found in additional_details")
            
            print("\nExtracted topics:")
            for i, topic in enumerate(topics, 1):
                print(f"{i}. {topic}")
            
            print(f"\nTotal topics found: {len(topics)}")
            print("="*50 + "\n")
            return topics, additional_details
            
        except json.JSONDecodeError as e:
            print(f"Error parsing additional_details as JSON: {e}")
            print("Falling back to string splitting...")
            # Fallback to string splitting if JSON parsing fails
            topics = [topic.strip() for topic in skill_details.split(';')]
            print("\nExtracted topics (fallback):")
            for i, topic in enumerate(topics, 1):
                print(f"{i}. {topic}")
            return topics, skill_details

    def extract_skills_for_topic(self, additional_details, topic_name):
        for detail in additional_details:
            if detail.get("Topic") == topic_name and "Suggested_Skills" in detail:
                return detail["Suggested_Skills"]
        return []

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
                topic_name = details.get('unit')
                if not topic_name:
                    print("Warning: No unit found in additional details")
                    continue
                    
                print(f"\nUsing unit as topic: {topic_name}")
                
                # Get objectives as suggested skills
                suggested_skills = []
                if 'objectives' in details:
                    for objective in details['objectives']:
                        if 'description' in objective:
                            suggested_skills.append(objective['description'])
                            print(f"Added suggested skill: {objective['description']}")
                
                skill_topic_params.append({
                    'skill_id': skill_data["skill_id"],
                    'skill_name': skill_data["skill_name"],
                    'topic_name': topic_name,
                    'suggested_skills': suggested_skills
                })
                
                print(f"\nAdded parameters for topic: {topic_name}")
                print(f"Number of suggested skills: {len(suggested_skills)}")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing additional details: {e}")
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

    def fetch_sample_question_embeddings(self, skill_topic_params):
        """
        Fetch sample questions using QuestionRetriever with enhanced similarity matching
        Args:
            skill_topic_params: List of dictionaries containing skill and topic information
        Returns:
            List of dictionaries containing skill, topic, and their corresponding sample questions
        """
        if not self.use_sample_questions:
            return [{
                'skill_id': params['skill_id'],
                'skill_name': params['skill_name'],
                'topic_name': params['topic_name'],
                'sample_questions': []
            } for params in skill_topic_params]

        retriever = QuestionRetriever()
        sample_questions = []
        
        for params in skill_topic_params:
            # Get similar questions using QuestionRetriever
            similar_questions = retriever.find_similar_questions(
                subject=params['skill_data']['subject'],
                skill=params['skill_name'],
                topic=params['topic_name'],
                n_results=self.num_sample_questions * 2  # Get more questions for filtering
            )
            
            # Format and enhance questions with additional metadata
            formatted_questions = []
            for question in similar_questions:
                # Calculate question complexity metrics
                question_text = question.question_text
                word_count = len(question_text.split())
                has_math = any(char in question_text for char in ['∫', '∑', '√', 'θ', 'π'])
                has_diagram = any(word in question_text.lower() for word in ['diagram', 'figure', 'graph', 'plot'])
                
                formatted_question = {
                    'question_text': question.question_text,
                    'multiple_choices': question.multiple_choices,
                    'correct_answer': question.correct_answer,
                    'skill': question.skill,
                    'topic': question.topic,
                    'metadata': {
                        'word_count': word_count,
                        'has_math_notation': has_math,
                        'requires_diagram': has_diagram,
                        'num_choices': len(question.multiple_choices),
                        'blooms_level': question.metadata.get('blooms_level', 'Analyzing'),
                        'difficulty': question.metadata.get('difficulty', 'medium')
                    }
                }
                formatted_questions.append(formatted_question)
            
            # Sort questions by similarity score
            if formatted_questions:
                # Calculate similarity scores for each question
                similarity_scores = []
                for q in formatted_questions:
                    score = self.calculate_question_similarity(q, {
                        'topic': params['topic_name'],
                        'skill': params['skill_name'],
                        'difficulty': params.get('difficulty', 'medium')
                    })
                    similarity_scores.append(score)
                
                # Sort questions by similarity score
                sorted_questions = [q for _, q in sorted(zip(similarity_scores, formatted_questions), reverse=True)]
                # Take top N questions
                top_questions = sorted_questions[:self.num_sample_questions]
            else:
                top_questions = []
            
            sample_questions.append({
                'skill_id': params['skill_id'],
                'skill_name': params['skill_name'],
                'topic_name': params['topic_name'],
                'sample_questions': top_questions
            })
        
        return sample_questions

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
            print(f"\nProcessing parameters for skill: {params['skill_name']}")
            
            llm_params = {
                'subject_id': None,
                'subject': params.get('subject', 'Mathematics'),  # Default to Mathematics if not present
                'subject_area_id': None,
                'subject_area': params.get('subject_area', 'Calculus'),  # Default to Calculus if not present
                'skill_id': params['skill_id'],
                'skill_details': params['topic_name'],
                'skill_name': params['skill_name'],
                'skills_list': params['suggested_skills'],
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
                'skill_name': params['skill_name'],
                'topic_name': params['topic_name'],
                'parameters': llm_params
            })
            
            print(f"Added parameters for topic: {params['topic_name']}")
        
        print(f"\nTotal parameter sets created: {len(parameters_list)}")
        print("="*50 + "\n")
        return parameters_list

    def generate_content(self, parameters):
        """
        Generate content using LLM with the given parameters
        Args:
            parameters (dict): Parameters for content generation
        Returns:
            str: Generated content or None if generation fails
        """
        try:
            # Get prompts from prompt builder
            system_prompt, user_prompt = self.prompt_builder.create_prompts(parameters)
            if system_prompt is None or user_prompt is None:
                print("Error: Could not create prompts.")
                return None

            # Call LLM API
            print("\nCalling LLM API...")
            ai_response_content = call_llm_api("openai", system_prompt, user_prompt)
            
            if not ai_response_content:
                print("Error: No response from LLM API")
                return None

            # Clean and validate the response
            try:
                # Remove any markdown code block markers if present
                if ai_response_content.startswith('```json'):
                    ai_response_content = ai_response_content[7:]
                if ai_response_content.endswith('```'):
                    ai_response_content = ai_response_content[:-3]
                ai_response_content = ai_response_content.strip()

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
                
                return ai_response_content

            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON response from LLM: {e}")
                print("Raw response:", ai_response_content)
                return None

        except Exception as e:
            print(f"Error in generate_content: {str(e)}")
            return None

    def store_output_to_file(self, topic_name, skill_name, content):
        if content:
            filename = f"questions_{skill_name}_{topic_name}.txt".replace(" ", "_").replace("/", "_")
            os.makedirs("generated_questions", exist_ok=True)
            filepath = os.path.join("generated_questions", filename)
            with open(filepath, 'w') as f:
                f.write(content)
            

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

    def generate_content_from_llm(self, llm_prompt_parameters_list, sample_questions):
        """
        Generate content using LLM for each set of parameters and sample questions
        Args:
            llm_prompt_parameters_list: List of dictionaries containing LLM parameters
            sample_questions: List of dictionaries containing sample questions
        Returns:
            List of tuples containing (skill_id, skill_name, topic_name, content)
        """
        all_contents = []
        
        # If sample_questions is empty, just use the parameters list
        if not sample_questions or len(sample_questions) == 0:
            print("No sample questions provided - generating content without samples")
            for params in llm_prompt_parameters_list:
                content = self.generate_content(params['parameters'])
                if content:
                    all_contents.append((
                        params['skill_id'],
                        params['skill_name'],
                        params['topic_name'],
                        content
                    ))
        else:
            # If we have sample questions, use them with the parameters
            for params, samples in zip(llm_prompt_parameters_list, sample_questions):
                content = self.generate_content(params['parameters'])
                if content:
                    all_contents.append((
                        params['skill_id'],
                        params['skill_name'],
                        params['topic_name'],
                        content
                    ))
        
        return all_contents

    def write_content(self, contents):
        for skill_id, skill_name, topic_name, content in contents:
            if self.output_mode == "file":
                self.store_output_to_file(topic_name, skill_name, content)
            elif self.output_mode == "mongo":
                self.store_output_to_mongo(content, skill_id)
            print(f"✅ Content generated and stored for topic: {topic_name}")

# ------------------ Main Workflow ------------------

def main():
    try:
        # Step 1: Initialize Context
        print("Initializing context...")
        context = GlobalContext()  # Creates context and loads task_config.json
        context.initialize()       # Initializes connections and services
       
        # Validate context is properly loaded
        if not hasattr(context, 'task_name') and not hasattr(context, 'skill_ids'):
            raise ValueError("Either task_name or skill_ids must be provided in task_config.json")
        
        print(f"Context loaded with task: {getattr(context, 'task_name', 'None')}")
        print(f"Skill IDs: {getattr(context, 'skill_ids', [])}")
       
        # Step 2: Resolve skills
        print("Resolving skills...")
        skills_data = context.resolve_skills_from_context()
        #print(f"Skills data: {skills_data}")
        if not skills_data:
            raise ValueError("No skills data found for the given context")
       
        # Step 3: Prepare parameters and generate content
        #print("Preparing parameters...")
        skill_topic_params = context.get_skill_topic_parameters(skills_data)
        print(f"Skill topic parameters: {skill_topic_params}")
        
        # Step 3.1: Fetch sample questions using QuestionRetriever
        #print("Fetching sample questions...")
        #sample_questions = context.fetch_sample_question_embeddings(skill_topic_params)
        sample_questions = []
        
        # Step 3.2: Prepare LLM parameters with sample questions
        print("Preparing LLM parameters...")
        llm_prompt_parameters_list = context.prepare_llm_parameters(skill_topic_params, sample_questions)
        print(f"LLM prompt parameters list: {llm_prompt_parameters_list}")
        '''
        # Step 3.3: Generate content using LLM with sample questions
        print("Generating content...")
        all_contents = context.generate_content_from_llm(llm_prompt_parameters_list, sample_questions)

        # Step 4: Write content
        print("Writing content...")
        context.write_content(all_contents)

        print("✅ Question Generation Completed.")
        '''
    except Exception as e:
        print(f"❌ Error in main workflow: {str(e)}")
        raise

if __name__ == '__main__':
    main()
