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
            config_path = os.path.join(os.path.dirname(__file__), 'task_config.json')
            print(f"Loading configuration from: {config_path}")
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                print(f"Configuration data loaded: {config_data}")
                
                # Load all properties from task_config.json into the context object
                for key, value in config_data.items():
                    print(f"Setting context.{key} = {value}")
                    setattr(self, key, value)
                
                # Verify the properties were set
                print("Verifying context properties:")
                for key in config_data.keys():
                    if hasattr(self, key):
                        print(f"✓ {key}: {getattr(self, key)}")
                    else:
                        print(f"✗ {key} not found in context")
                
        except FileNotFoundError:
            print("Warning: task_config.json not found. Using default values.")
            # Set default values for required properties
            self.task_name = None
            self.skill_ids = []
            self.num_questions = 12
            self.prompt_type = "Multiple Choice"
            self.output_mode = "file"
        except Exception as e:
            print(f"Error loading application configuration: {e}")
            print("Using default values.")
            # Set default values for required properties
            self.task_name = None
            self.skill_ids = []
            self.num_questions = 12
            self.prompt_type = "Multiple Choice"
            self.output_mode = "file"

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

# ------------------ Workflow Steps ------------------

def resolve_skills_from_context(self):
    cursor = self.mysql_conn.cursor()
    skills_data = []

    if self.task_name:
        # Priority: If task is provided, find all skill_ids for the task
        query = """
            SELECT
                s.skill_id,
                s.skill_name,
                s.skill_details,
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
        cursor.execute(query, (self.task_name,))
    elif self.skill_ids:
        # If only skill_ids provided, find details for each skill
        query = """
            SELECT
                s.skill_id,
                s.skill_name,
                s.skill_details,
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
            cursor.execute(query, (skill_id,))
            result = cursor.fetchone()
            if result:
                skills_data.append({
                    "skill_id": result[0],
                    "skill_name": result[1],
                    "skill_details": result[2],
                    "subject_area": result[3],
                    "subject": result[4],
                    "task_name": result[5]
                })
        cursor.close()
        return skills_data
    else:
        raise ValueError("Either skill_ids or task_name must be provided in context.")

    # Fetch all matching skills for task
    results = cursor.fetchall()
    for result in results:
        skills_data.append({
            "skill_id": result[0],
            "skill_name": result[1],
            "skill_details": result[2],
            "subject_area": result[3],
            "subject": result[4],
            "task_name": result[5]
        })
    cursor.close()
    return skills_data


def read_topics_for_skill(self, skill_data, prompt_type):
    prompt_collection = self.mongo_db['prompt_collection']
    mongo_doc = prompt_collection.find_one({
        "task_name": skill_data["task_name"],
        "prompt_type": prompt_type,
        "skill_details.skill_name": skill_data["skill_name"]
    })

    if not mongo_doc:
        print(f"⚠️ No prompt found for type '{prompt_type}'. Trying fallback without prompt_type...")
        mongo_doc = prompt_collection.find_one({
            "task_name": skill_data["task_name"],
            "skill_details.skill_name": skill_data["skill_name"]
        })

    if mongo_doc:
        skill_details = next((skill for skill in mongo_doc["skill_details"] if skill["skill_name"] == skill_data["skill_name"]), None)
        if skill_details:
            topics = [detail["Topic"] for detail in skill_details["additional_details"]]
            additional_details = skill_details.get("additional_details", [])
            return topics, additional_details
    return [], []


def extract_skills_for_topic(additional_details, topic_name):
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
    skill_topic_params = []
    for skill_data in skills_data:
        topics, additional_details = self.read_topics_for_skill(skill_data, self.prompt_type)
        for topic_name in topics:
            suggested_skills = extract_skills_for_topic(additional_details, topic_name)
            skill_topic_params.append({
                'skill_id': skill_data["skill_id"],
                'skill_name': skill_data["skill_name"],
                'topic_name': topic_name,
                'suggested_skills': suggested_skills,
                'skill_data': skill_data
            })
    return skill_topic_params


def calculate_question_similarity(question1: Dict, question2: Dict) -> float:
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
                score = calculate_question_similarity(q, {
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
    parameters_list = []
    for params, samples in zip(skill_topic_params, sample_questions):
        llm_params = {
            'subject_id': None,
            'subject': params['skill_data']['subject'],
            'subject_area_id': None,
            'subject_area': params['skill_data']['subject_area'],
            'skill_id': params['skill_data']['skill_id'],
            'skill_details': params['topic_name'],
            'skill_name': params['skill_data']['skill_name'],
            'skills_list': params['suggested_skills'],
            'num_questions': self.num_questions,
            'sample_questions': samples['sample_questions']  # Add sample questions to parameters
        }
        parameters_list.append({
            'skill_id': params['skill_id'],
            'skill_name': params['skill_name'],
            'topic_name': params['topic_name'],
            'parameters': llm_params
        })
    return parameters_list


def generate_content(self, parameters):
    system_prompt, user_prompt = self.prompt_builder.create_prompts(parameters)
    if system_prompt is None or user_prompt is None:
        print("Error: Could not create prompts.")
        return None
    ai_response_content = call_llm_api("anthropic", system_prompt, user_prompt)
    return ai_response_content


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
    for params, samples in zip(llm_prompt_parameters_list, sample_questions):
        # Generate content using the parameters that now include sample questions
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
        if not skills_data:
            raise ValueError("No skills data found for the given context")

        # Step 3: Prepare parameters and generate content
        print("Preparing parameters...")
        skill_topic_params = context.get_skill_topic_parameters(skills_data)
        
        # Step 3.1: Fetch sample questions using QuestionRetriever
        print("Fetching sample questions...")
        sample_questions = context.fetch_sample_question_embeddings(skill_topic_params)
        
        # Step 3.2: Prepare LLM parameters with sample questions
        print("Preparing LLM parameters...")
        llm_prompt_parameters_list = context.prepare_llm_parameters(skill_topic_params, sample_questions)
        
        # Step 3.3: Generate content using LLM with sample questions
        print("Generating content...")
        all_contents = context.generate_content_from_llm(llm_prompt_parameters_list, sample_questions)

        # Step 4: Write content
        print("Writing content...")
        context.write_content(all_contents)

        print("✅ Question Generation Completed.")
    except Exception as e:
        print(f"❌ Error in main workflow: {str(e)}")
        raise

if __name__ == '__main__':
    main()
