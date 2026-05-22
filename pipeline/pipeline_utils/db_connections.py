import os
import mysql.connector
from pymongo import MongoClient
from dotenv import load_dotenv
import config
from typing import Tuple

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