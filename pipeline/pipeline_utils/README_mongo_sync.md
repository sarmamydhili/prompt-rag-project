# MongoDB Synchronization Utility

## Overview

The `mongo_sync.py` script syncs **delta** records from MongoDB **Staging** (source) to **Production** (target). Only documents whose `_id` is not already in the target are copied. It is generic over database and collection and supports env vars and CLI.

Use case: sync the `hints_and_answers` collection (or any collection) from Staging to Prod.

## Key Features

- **Delta sync**: Copies only documents missing in target (by `_id`)
- **Generic**: Any database and collection via constructor, env, or CLI
- **No hardcoded credentials**: Staging and Prod URIs from env (or CLI)
- **Batch processing**: Configurable batch size; continues on partial failures
- **Clear naming**: Source = Staging, Target = Production

## Usage

### 1. Environment Configuration

Create a `.env` file in your project root. Use **full URI** or **URL + user + password**:

```bash
# --- Staging (source) ---
# Option A: full URI
STAGING_MONGO_URI=mongodb://user:password@staging-host:27017/adaptive_learning_docs

# Option B: separate URL and credentials
# STAGING_MONGO_URL=staging-host:27017/adaptive_learning_docs
# STAGING_MONGO_USER=user
# STAGING_MONGO_PASSWORD=password

# --- Production (target) ---
# Option A: full URI (recommended; set Prod URL, user, password here)
PROD_MONGO_URI=mongodb://produser:prodpass@prod-host:27017/adaptive_learning_docs

# Option B: separate URL and credentials
# PROD_MONGO_URL=prod-host:27017/adaptive_learning_docs
# PROD_MONGO_USER=produser
# PROD_MONGO_PASSWORD=prodpass

# --- Optional: database and collection (defaults below) ---
# MONGO_DATABASE=adaptive_learning_docs
# MONGO_COLLECTION=hints_and_answers
```

### 2. Sync `hints_and_answers` (default collection)

```bash
# Uses MONGO_COLLECTION=hints_and_answers and MONGO_DATABASE from .env
python pipeline/pipeline_utils/mongo_sync.py
```

### 3. Command line options

```bash
# Explicit collection and database
python pipeline/pipeline_utils/mongo_sync.py --collection hints_and_answers --database adaptive_learning_docs

# Shorthand
python pipeline/pipeline_utils/mongo_sync.py -c hints_and_answers -d adaptive_learning_docs

# Custom batch size
python pipeline/pipeline_utils/mongo_sync.py -c hints_and_answers --batch-size 500

# Override URIs from command line (no env needed)
python pipeline/pipeline_utils/mongo_sync.py --staging-uri "mongodb://..." --prod-uri "mongodb://..."
```

### 4. Use as a module

```python
from pipeline.pipeline_utils.mongo_sync import MongoSync

# Default: collection=hints_and_answers, database=adaptive_learning_docs
sync_util = MongoSync()
synced_count = sync_util.sync_documents()

# Custom collection/database and URIs
sync_util = MongoSync(
    source_uri="mongodb://staging-host:27017",
    target_uri="mongodb://produser:prodpass@prod-host:27017/adaptive_learning_docs",
    database="adaptive_learning_docs",
    collection="hints_and_answers",
)
synced_count = sync_util.sync_documents(batch_size=500)
```

## Error Handling

The script handles:

- **Connection failures**: Logs and exits gracefully
- **Authentication errors**: Clear messages for credential issues
- **Partial insert failures**: Continues sync; logs failed count
- **Network timeouts**: 5s server selection timeout
- **Keyboard interrupt**: Graceful shutdown on Ctrl+C

## Logging

Example output:

```
... - INFO - Connecting to source (staging) database...
... - INFO - Connecting to target (production) database...
... - INFO - Successfully connected to both databases
... - INFO - Found 1500 existing documents in target (adaptive_learning_docs.hints_and_answers)
... - INFO - Found 250 new documents to sync (collection: hints_and_answers)
... - INFO - Synced batch of 250 documents. Total: 250
... - INFO - Sync completed. Total documents synced: 250 (collection: hints_and_answers)
```

## Performance Considerations

- **Batch Processing**: Default batch size of 1000 documents
- **Memory Efficiency**: Only fetches `_id` fields for comparison
- **Connection Pooling**: Proper connection management
- **Indexed Queries**: Uses `_id` field for efficient lookups

## Production URL, user and password

Do **not** put Production URL, user or password in code. Set them in `.env`:

- **Recommended**: `PROD_MONGO_URI=mongodb://USER:PASSWORD@HOST:PORT/DATABASE`
- Or: `PROD_MONGO_URL`, `PROD_MONGO_USER`, `PROD_MONGO_PASSWORD` (script builds the URI)

Keep `.env` out of version control (e.g. in `.gitignore`).

## Security

1. Use environment variables (or `--prod-uri` / `--staging-uri` at runtime); never hardcode credentials.
2. Use authentication for both Staging and Production.
3. Restrict network access to Prod (VPN/firewall) where possible.

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