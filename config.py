import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# === OpenAI Settings ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # From your .env
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4-turbo"  # or gpt-3.5-turbo if you want cheaper parsing


LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

# Anthropic API credentials
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# DeepSeek API credentials
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Gemini API credentials
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')



# === MongoDB Settings ===
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "prompt_project"
MONGO_COLLECTION_NAME = "input_questions"

# === Chroma Vector DB Settings ===
CHROMA_COLLECTION_NAME = "questions_collection"

# === PDF Data Settings ===
PDF_PATH = "data/questions.pdf"

# === Prompt Settings ===
PROMPT_TEMPLATE_PATH = "prompts/structure_prompt.txt"

# === General Settings ===
CHUNK_SIZE = 3000  # Characters per chunk when splitting PDF text
TEMPERATURE = 0.2  # LLM generation temperature (lower = more deterministic)
