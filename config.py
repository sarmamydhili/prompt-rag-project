import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# === API Credentials ===
# OpenAI API credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Anthropic API credentials
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# DeepSeek API credentials
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Gemini API credentials
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# LangChain API credentials
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

# === Embedding Model Settings ===
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# === Chroma Settings ===
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "questions")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")

# === MongoDB Settings ===
MONGODB_SERVER = os.getenv("MONGODB_SERVER", "127.0.0.1")
MONGODB_PORT = os.getenv("MONGODB_PORT", "27017")
MONGODB_USER = os.getenv("MONGODB_USER", "")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "adaptive_learning")
MONGODB_QUESTIONS_COLLECTION = os.getenv("MONGODB_QUESTIONS_COLLECTION", "questions")
MONGODB_ADAPTIVE_DB_NAME = os.getenv("MONGODB_ADAPTIVE_DB_NAME", "adaptive_learning")

# Construct MongoDB URI
MONGO_URI = f"mongodb://{MONGODB_SERVER}:{MONGODB_PORT}/"
if MONGODB_USER and MONGODB_PASSWORD:
    MONGO_URI = f"mongodb://{MONGODB_USER}:{MONGODB_PASSWORD}@{MONGODB_SERVER}:{MONGODB_PORT}/"

# === MySQL Settings ===
mysql_host = os.getenv("MYSQL_HOST", "localhost") 
mysql_user = os.getenv("MYSQL_USER", "root")
mysql_password = os.getenv("MYSQL_PASSWORD", "")
mysql_database = os.getenv("MYSQL_DATABASE", "adaptive_learning")

# ======== Logging Configuration ========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s") 