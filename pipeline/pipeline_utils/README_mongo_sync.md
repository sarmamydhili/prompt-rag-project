# MongoDB Synchronization Utility

## Overview

The `mongo_sync.py` script provides a robust way to synchronize documents from a development MongoDB instance to a production instance. It includes comprehensive error handling, batch processing, and proper resource management.

## Key Improvements

### ✅ Fixed Issues
- **Error Handling**: Added comprehensive try-catch blocks for all database operations
- **Resource Management**: Proper connection cleanup with `close_connections()`
- **Fixed Bug**: Removed incorrect `new_docs.retrieved` attribute access
- **Security**: Moved hardcoded credentials to environment variables
- **Memory Efficiency**: Added batch processing for large datasets
- **Logging**: Replaced print statements with proper logging

### ✅ New Features
- **Class-based Design**: More maintainable and testable code structure
- **Configuration**: Environment variable support for database connections
- **Batch Processing**: Configurable batch size for memory efficiency
- **Progress Tracking**: Detailed logging of sync progress
- **Partial Failure Handling**: Continues sync even if some documents fail
- **Connection Testing**: Validates database connections before operations

## Usage

### 1. Environment Configuration

Create a `.env` file in your project root:

```bash
# Development MongoDB URI
DEV_MONGO_URI=mongodb://localhost:27017

# Production MongoDB URI (include credentials if needed)
PROD_MONGO_URI=mongodb://username:password@3.128.97.182:27017

# Optional: Override default database and collection
# MONGO_DATABASE=adaptive_learning_docs
# MONGO_COLLECTION=dryrun_questions
```

### 2. Basic Usage

```python
from pipeline.pipeline_utils.mongo_sync import MongoSync

# Use default configuration
sync_util = MongoSync()
synced_count = sync_util.sync_documents()
print(f"Synced {synced_count} documents")
```

### 3. Custom Configuration

```python
# Custom database connections
sync_util = MongoSync(
    dev_uri="mongodb://dev-server:27017",
    prod_uri="mongodb://prod-server:27017",
    database="my_database",
    collection="my_collection"
)

# Custom batch size
synced_count = sync_util.sync_documents(batch_size=500)
```

### 4. Command Line Usage

```bash
python pipeline/pipeline_utils/mongo_sync.py
```

## Error Handling

The script handles various error scenarios:

- **Connection Failures**: Logs error and exits gracefully
- **Authentication Errors**: Proper error messages for credential issues
- **Partial Insert Failures**: Continues sync even if some documents fail
- **Network Timeouts**: Configurable timeout settings
- **Keyboard Interrupt**: Graceful shutdown on Ctrl+C

## Logging

The script provides detailed logging with timestamps:

```
2024-01-15 10:30:15 - INFO - Connecting to development database...
2024-01-15 10:30:16 - INFO - Connecting to production database...
2024-01-15 10:30:16 - INFO - Successfully connected to both databases
2024-01-15 10:30:17 - INFO - Found 1500 existing documents in production
2024-01-15 10:30:18 - INFO - Found 250 new documents to sync
2024-01-15 10:30:19 - INFO - Synced batch of 250 documents. Total: 250
2024-01-15 10:30:19 - INFO - Sync completed. Total documents synced: 250
```

## Performance Considerations

- **Batch Processing**: Default batch size of 1000 documents
- **Memory Efficiency**: Only fetches `_id` fields for comparison
- **Connection Pooling**: Proper connection management
- **Indexed Queries**: Uses `_id` field for efficient lookups

## Security Best Practices

1. **Environment Variables**: Never hardcode credentials
2. **Network Security**: Use VPN or firewall rules for production access
3. **Authentication**: Always use authentication for production databases
4. **Connection Strings**: Use connection strings with proper encoding

## Troubleshooting

### Common Issues

1. **Connection Timeout**: Check network connectivity and firewall settings
2. **Authentication Failed**: Verify username/password in connection string
3. **Permission Denied**: Ensure user has read/write permissions
4. **Memory Issues**: Reduce batch size for large datasets

### Debug Mode

Enable debug logging by modifying the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Dependencies

- `pymongo>=4.8.0`
- `python-dotenv>=1.0.1`
- `typing-extensions>=4.12.2`

All dependencies are already included in the project's `requirements.txt`. 