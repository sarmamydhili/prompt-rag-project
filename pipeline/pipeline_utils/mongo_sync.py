import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, BulkWriteError
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hardcoded MongoDB credentials for production
mongo_db_user = "testuser"
mongo_db_password = "test0215"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoSync:
    def __init__(self, dev_uri: str = None, prod_uri: str = None, 
                 database: str = "adaptive_learning_docs", 
                 collection: str = "dryrun_questions"):
        """
        Initialize MongoDB synchronization utility.
        
        Args:
            dev_uri: Development MongoDB connection string
            prod_uri: Production MongoDB connection string  
            database: Database name
            collection: Collection name
        """
        # Build production URI with hardcoded credentials
        # Try with authentication database specified
        prod_uri_with_auth = f"mongodb://{mongo_db_user}:{mongo_db_password}@3.128.97.182:27017/adaptive_learning_docs"
            
        self.dev_uri = dev_uri or os.getenv('DEV_MONGO_URI', 'mongodb://localhost:27017')
        self.prod_uri = prod_uri or os.getenv('PROD_MONGO_URI', prod_uri_with_auth)
        self.database = "adaptive_learning_docs"
        self.collection = "dryrun_questions"
        self.dev_client = None
        self.prod_client = None
        
    def connect_databases(self) -> bool:
        """Establish connections to both development and production databases."""
        try:
            logger.info("Connecting to development database...")
            self.dev_client = MongoClient(self.dev_uri, serverSelectionTimeoutMS=5000)
            self.dev_client.admin.command('ping')  # Test connection
            
            logger.info("Connecting to production database...")
            self.prod_client = MongoClient(self.prod_uri, serverSelectionTimeoutMS=5000)
            self.prod_client.admin.command('ping')  # Test connection
            
            logger.info("Successfully connected to both databases")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            return False
    
    def get_production_ids(self) -> set:
        """Get all document IDs from production database."""
        try:
            prod_coll = self.prod_client[self.database][self.collection]
            # Use projection to only fetch _id field for memory efficiency
            prod_ids = set(doc["_id"] for doc in prod_coll.find({}, {"_id": 1}))
            logger.info(f"Found {len(prod_ids)} existing documents in production")
            return prod_ids
        except Exception as e:
            logger.error(f"Error fetching production IDs: {e}")
            return set()
    
    def sync_documents(self, batch_size: int = 1000) -> int:
        """
        Synchronize documents from development to production.
        
        Args:
            batch_size: Number of documents to process in each batch
            
        Returns:
            Number of documents successfully synced
        """
        if not self.connect_databases():
            return 0
            
        try:
            dev_coll = self.dev_client[self.database][self.collection]
            prod_coll = self.prod_client[self.database][self.collection]
            
            # Get existing production IDs
            prod_ids = self.get_production_ids()
            
            # Find new documents in development
            new_docs_query = {"_id": {"$nin": list(prod_ids)}}
            new_docs_count = dev_coll.count_documents(new_docs_query)
            
            if new_docs_count == 0:
                logger.info("No new documents to sync")
                return 0
            
            logger.info(f"Found {new_docs_count} new documents to sync")
            
            # Process documents in batches
            synced_count = 0
            new_docs_cursor = dev_coll.find(new_docs_query)
            
            while True:
                batch = list(new_docs_cursor.limit(batch_size))
                if not batch:
                    break
                    
                try:
                    result = prod_coll.insert_many(batch, ordered=False)
                    synced_count += len(result.inserted_ids)
                    logger.info(f"Synced batch of {len(batch)} documents. Total: {synced_count}")
                    
                except BulkWriteError as e:
                    # Handle partial failures
                    inserted_count = len(e.details.get('insertedIds', []))
                    synced_count += inserted_count
                    logger.warning(f"Batch had {len(batch) - inserted_count} failed insertions")
                    
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}")
                    continue
            
            logger.info(f"Sync completed. Total documents synced: {synced_count}")
            return synced_count
            
        except Exception as e:
            logger.error(f"Error during sync operation: {e}")
            return 0
        finally:
            self.close_connections()
    
    def close_connections(self):
        """Close database connections."""
        if self.dev_client:
            self.dev_client.close()
            logger.info("Development database connection closed")
        if self.prod_client:
            self.prod_client.close()
            logger.info("Production database connection closed")

def main():
    """Main function to run the sync operation."""
    try:
        sync_util = MongoSync()
        synced_count = sync_util.sync_documents()
        
        if synced_count > 0:
            logger.info(f"Successfully synced {synced_count} documents")
        else:
            logger.info("No documents were synced")
            
    except KeyboardInterrupt:
        logger.info("Sync operation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")

if __name__ == "__main__":
    main()
