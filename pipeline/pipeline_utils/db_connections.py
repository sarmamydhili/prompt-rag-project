import os
import mysql.connector
from pymongo import MongoClient
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import config
import datetime
from typing import Tuple, Optional

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class DBConfig:
    """Database configuration class that combines environment variables and application config"""
    
    # Environment variables (from config.py)
    MONGODB_USER = config.MONGODB_USER
    MONGODB_PASSWORD = config.MONGODB_PASSWORD
    MYSQL_USER = config.MYSQL_USER
    MYSQL_PASSWORD = config.MYSQL_PASSWORD
    
    # Application config (from task_config.properties via GlobalContext)
    # These will be set by GlobalContext when initializing the application
    MONGO_SERVER = None  # Will be set from task_config
    MONGO_PORT = None    # Will be set from task_config
    MONGO_DB_NAME = None # Will be set from task_config
    MONGO_QUESTIONS_COLLECTION = None  # Will be set from task_config
    MONGO_COURSE_FRAMEWORK_COLLECTION = None  # Will be set from task_config
    MONGO_OUTPUT_COLLECTION = None  # Will be set from task_config
    MONGO_ADAPTIVE_DB_NAME = None  # Will be set from task_config
    
    MYSQL_HOST = None  # Will be set from task_config
    MYSQL_DATABASE = None  # Will be set from task_config
    
    CHROMA_COLLECTION_NAME = None  # Will be set from task_config
    CHROMA_PERSIST_DIRECTORY = None  # Will be set from task_config
    
    @classmethod
    def initialize_from_context(cls, context):
        """Initialize application config from GlobalContext"""
        # MongoDB settings
        cls.MONGO_SERVER = getattr(context, 'mongo_server', '127.0.0.1')
        cls.MONGO_PORT = getattr(context, 'mongo_port', '27017')
        cls.MONGO_DB_NAME = getattr(context, 'mongo_db_name', 'prompt_project')
        cls.MONGO_QUESTIONS_COLLECTION = getattr(context, 'mongo_questions_collection', 'questions')
        cls.MONGO_COURSE_FRAMEWORK_COLLECTION = getattr(context, 'mongo_course_framework_collection', 'course_framework')
        cls.MONGO_OUTPUT_COLLECTION = getattr(context, 'mongo_output_collection', 'output_questions_enhanced')
        cls.MONGO_ADAPTIVE_DB_NAME = getattr(context, 'mongo_adaptive_db_name', 'adaptive_learning_docs')
        
        # MySQL settings
        cls.MYSQL_HOST = getattr(context, 'mysql_host', 'localhost')
        cls.MYSQL_DATABASE = getattr(context, 'mysql_database', 'adaptive_learning')
        
        # Chroma settings
        cls.CHROMA_COLLECTION_NAME = getattr(context, 'chroma_collection_name', 'questions_collection')
        cls.CHROMA_PERSIST_DIRECTORY = getattr(context, 'chroma_persist_directory', 'chroma_db')
        
        # Construct MongoDB URI
        cls.MONGO_URI = f"mongodb://{cls.MONGO_SERVER}:{cls.MONGO_PORT}/"
        if cls.MONGODB_USER and cls.MONGODB_PASSWORD:
            cls.MONGO_URI = f"mongodb://{cls.MONGODB_USER}:{cls.MONGODB_PASSWORD}@{cls.MONGO_SERVER}:{cls.MONGO_PORT}/"

def get_mysql_connection():
    """
    Get MySQL database connection
    Returns:
        mysql.connector.connection.MySQLConnection: MySQL connection object
    """
    try:
        conn = mysql.connector.connect(
            host=DBConfig.MYSQL_HOST,
            user=DBConfig.MYSQL_USER,
            password=DBConfig.MYSQL_PASSWORD,
            database=DBConfig.MYSQL_DATABASE
        )
        return conn
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        raise

def get_mongo_connection() -> Tuple[MongoClient, any]:
    """
    Get MongoDB connection and database
    Returns:
        tuple: (MongoClient, database)
    """
    try:
        client = MongoClient(DBConfig.MONGO_URI)
        db = client[DBConfig.MONGO_DB_NAME]
        return client, db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise

def get_chroma_connection() -> Tuple[chromadb.Client, any]:
    """
    Get ChromaDB client and collection
    Returns:
        tuple: (chromadb.Client, collection)
    """
    try:
        client = chromadb.Client()
        collection = client.get_or_create_collection(DBConfig.CHROMA_COLLECTION_NAME)
        return client, collection
    except Exception as e:
        print(f"Error connecting to Chroma: {e}")
        raise

def save_to_chroma(question_text, question_id, metadata):
    """
    Save a question to Chroma with enhanced metadata for better searchability
    Args:
        question_text: The question text
        question_id: MongoDB document ID
        metadata: Dictionary containing question metadata
    """
    # Import here to avoid circular dependency
    from pipeline.pipeline_utils.embed_questions import embed_question
    
    try:
        # Prepare metadata for Chroma
        chroma_metadata = {
            "question_id": question_id,
            "topic": metadata.get("topic", ""),
            "keywords": ",".join(metadata.get("keywords", [])),
            "blooms_level": metadata.get("blooms_level", ""),
            "concepts": ",".join(metadata.get("concepts_tested", [])),
            "difficulty": metadata.get("difficulty", ""),
            "question_type": metadata.get("question_type", ""),
            "prerequisites": ",".join(metadata.get("prerequisites", [])),
            "common_misconceptions": ",".join(metadata.get("common_misconceptions", [])),
            "solution_strategy": metadata.get("solution_strategy", ""),
            "time_estimate": metadata.get("time_estimate", ""),
            "real_world_applications": ",".join(metadata.get("real_world_applications", [])),
            "cross_curricular_connections": ",".join(metadata.get("cross_curricular_connections", [])),
            "diagram_required": metadata.get("diagram_required", False),
            "source": metadata.get("source", "college_board"),
            "timestamp": metadata.get("timestamp", datetime.datetime.now().isoformat()),
            # New fields for enhanced searchability
            "question_pattern": metadata.get("question_pattern", ""),
            "mathematical_operations": ",".join(metadata.get("mathematical_operations", [])),
            "context_type": metadata.get("context_type", ""),
            "cognitive_demand": metadata.get("cognitive_demand", ""),
            "answer_format": metadata.get("answer_format", ""),
            "question_family": metadata.get("question_family", ""),
            # Combined fields for semantic search
            "semantic_context": f"{metadata.get('topic', '')} {metadata.get('solution_strategy', '')} {','.join(metadata.get('concepts_tested', []))} {','.join(metadata.get('real_world_applications', []))}"
        }
        
        # Embed and save to Chroma
        embed_question(
            question_text=question_text,
            question_id=question_id,
            metadata=chroma_metadata
        )
    except Exception as e:
        print(f"Error saving to Chroma: {e}")
        raise 