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
    MONGO_SERVER = config.MONGO_SERVER
    MONGO_PORT = config.MONGO_PORT
    MONGO_USER = config.MONGO_USER
    MONGO_PASSWORD = config.MONGO_PASSWORD
    MONGO_DB_NAME = config.MONGO_DB_NAME
    MONGO_QUESTIONS_COLLECTION = config.MONGO_QUESTIONS_COLLECTION
    MONGO_ADAPTIVE_DB_NAME = config.MONGO_ADAPTIVE_DB_NAME
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
    """Save a question to MongoDB and return its ID"""
    question["created_at"] = datetime.datetime.now()
    mongo_id = mongo_db[collection_name].insert_one(question).inserted_id
    return str(mongo_id)

def save_to_chroma(question_text, question_id, metadata):
    """Save a question to Chroma with its metadata"""
    embed_question(
        question_text=question_text,
        question_id=question_id,
        metadata=metadata
    ) 