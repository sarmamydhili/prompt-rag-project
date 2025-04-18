import chromadb
from chromadb.utils import embedding_functions
import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Chroma client with configuration
try:
    chroma_client = chromadb.Client()
    try:
        collection = chroma_client.get_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=config.OPENAI_API_KEY,
                model_name=config.EMBEDDING_MODEL
            )
        )
    except chromadb.errors.NotFoundError:
        # Create the collection if it doesn't exist
        collection = chroma_client.create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=config.OPENAI_API_KEY,
                model_name=config.EMBEDDING_MODEL
            )
        )
    logger.info(f"Successfully connected to Chroma collection: {config.CHROMA_COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize Chroma client: {str(e)}")
    raise

def embed_question(question_text: str, question_id: str, metadata: dict) -> None:
    """
    Embed a question and store it in ChromaDB
    Args:
        question_text: The text of the question
        question_id: Unique identifier for the question
        metadata: Dictionary containing question metadata
    """
    try:
        collection.add(
            documents=[question_text],
            metadatas=[metadata],
            ids=[question_id]
        )
        logger.info(f"Successfully embedded question with ID: {question_id}")
    except Exception as e:
        logger.error(f"Error embedding question {question_id}: {str(e)}")
        raise
