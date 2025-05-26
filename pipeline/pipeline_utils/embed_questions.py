import chromadb
from chromadb.utils import embedding_functions
import config
import logging
from pipeline.pipeline_utils.db_connections import DBConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global collection variable
_collection = None

def get_chroma_collection():
    """Get or initialize the Chroma collection"""
    global _collection
    if _collection is None:
        _collection = initialize_chroma_client()
    return _collection

def initialize_chroma_client():
    """Initialize Chroma client with configuration from DBConfig"""
    try:
        chroma_client = chromadb.Client()
        try:
            collection = chroma_client.get_collection(
                name=DBConfig.CHROMA_COLLECTION_NAME,
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=config.OPENAI_API_KEY,
                    model_name=config.EMBEDDING_MODEL
                )
            )
        except chromadb.errors.NotFoundError:
            # Create the collection if it doesn't exist
            collection = chroma_client.create_collection(
                name=DBConfig.CHROMA_COLLECTION_NAME,
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=config.OPENAI_API_KEY,
                    model_name=config.EMBEDDING_MODEL
                )
            )
        logger.info(f"Successfully connected to Chroma collection: {DBConfig.CHROMA_COLLECTION_NAME}")
        return collection
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
        collection = get_chroma_collection()
        collection.add(
            documents=[question_text],
            metadatas=[metadata],
            ids=[question_id]
        )
        logger.info(f"Successfully embedded question with ID: {question_id}")
    except Exception as e:
        logger.error(f"Error embedding question {question_id}: {str(e)}")
        raise
