import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === API Credentials ===
# OpenAI API credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# === Database Credentials ===
# MongoDB credentials
MONGODB_USER = os.getenv("MONGODB_USER")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")
MONGODB_SERVER = os.getenv("MONGODB_SERVER", "127.0.0.1")
MONGODB_PORT = os.getenv("MONGODB_PORT", "27017")

# MySQL credentials
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")

# === Logging Configuration ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s") 