import os
import mysql.connector
from pymongo import MongoClient
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import config
import datetime
from typing import Tuple, Optional
from pipeline.pipeline_utils.embed_questions import embed_question

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Database configurations
class DBConfig:
    # MySQL configurations
    MYSQL_HOST = config.mysql_host
    MYSQL_USER = config.mysql_user
    MYSQL_PASSWORD = config.mysql_password
    MYSQL_DATABASE = config.mysql_database
    
    # MongoDB configurations
    MONGO_SERVER = config.MONGODB_SERVER
    MONGO_PORT = config.MONGODB_PORT
    MONGO_USER = config.MONGODB_USER
    MONGO_PASSWORD = config.MONGODB_PASSWORD
    MONGO_DB_NAME = config.MONGODB_DB_NAME
    MONGO_QUESTIONS_COLLECTION = config.MONGODB_QUESTIONS_COLLECTION
    MONGO_ADAPTIVE_DB_NAME = config.MONGODB_ADAPTIVE_DB_NAME
    MONGO_URI = config.MONGO_URI
    CHROMA_COLLECTION_NAME = config.CHROMA_COLLECTION_NAME

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
        db = client[DBConfig.MONGO_ADAPTIVE_DB_NAME]
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

def save_to_mongodb(question, mongo_db, collection_name):
    """
    Save a question with its metadata to MongoDB and return its ID
    Args:
        question: Dictionary containing question and metadata
        mongo_db: MongoDB database instance
        collection_name: Name of the collection to save to
    Returns:
        str: MongoDB document ID
    """
    try:
        # Add timestamp
        question["created_at"] = datetime.datetime.now()
        
        # Ensure metadata is properly structured
        if "metadata" not in question:
            question["metadata"] = {}
            
        # Add searchable fields for better querying
        question["searchable_fields"] = {
            "topics": question["metadata"].get("topic", ""),
            "keywords": question["metadata"].get("keywords", []),
            "blooms_level": question["metadata"].get("blooms_level", ""),
            "concepts": question["metadata"].get("concepts_tested", []),
            "difficulty": question["metadata"].get("difficulty", ""),
            "question_type": question["metadata"].get("question_type", ""),
            # Additional searchable fields
            "prerequisites": question["metadata"].get("prerequisites", []),
            "common_misconceptions": question["metadata"].get("common_misconceptions", []),
            "real_world_applications": question["metadata"].get("real_world_applications", []),
            "cross_curricular_connections": question["metadata"].get("cross_curricular_connections", []),
            "solution_strategy": question["metadata"].get("solution_strategy", ""),
            "time_estimate": question["metadata"].get("time_estimate", ""),
            # New fields for question generation
            "question_pattern": question["metadata"].get("question_pattern", ""),  # e.g., "word_problem", "proof", "calculation"
            "mathematical_operations": question["metadata"].get("mathematical_operations", []),  # e.g., ["differentiation", "integration"]
            "context_type": question["metadata"].get("context_type", ""),  # e.g., "theoretical", "applied", "experimental"
            "cognitive_demand": question["metadata"].get("cognitive_demand", ""),  # e.g., "low", "medium", "high"
            "answer_format": question["metadata"].get("answer_format", ""),  # e.g., "numerical", "symbolic", "verbal"
            "question_family": question["metadata"].get("question_family", "")  # Group of related questions
        }
        
        # Create indexes for efficient querying
        mongo_db[collection_name].create_index([
            ("question", "text"),
            ("searchable_fields.topics", "text"),
            ("searchable_fields.keywords", "text"),
            ("searchable_fields.concepts", "text"),
            ("searchable_fields.prerequisites", "text"),
            ("searchable_fields.common_misconceptions", "text"),
            ("searchable_fields.real_world_applications", "text"),
            ("searchable_fields.cross_curricular_connections", "text"),
            ("searchable_fields.solution_strategy", "text")
        ])
        
        # Create compound indexes for common query patterns
        mongo_db[collection_name].create_index([
            ("searchable_fields.difficulty", 1),
            ("searchable_fields.blooms_level", 1),
            ("searchable_fields.question_type", 1)
        ])
        
        mongo_db[collection_name].create_index([
            ("searchable_fields.topic", 1),
            ("searchable_fields.cognitive_demand", 1),
            ("searchable_fields.question_pattern", 1)
        ])
        
        # Insert the document
        mongo_id = mongo_db[collection_name].insert_one(question).inserted_id
        return str(mongo_id)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        raise

def save_to_chroma(question_text, question_id, metadata):
    """
    Save a question to Chroma with enhanced metadata for better searchability
    Args:
        question_text: The question text
        question_id: MongoDB document ID
        metadata: Dictionary containing question metadata
    """
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