# Prompt RAG Project

A pipeline for processing educational content, generating questions, and managing them in a vector database.

## Project Structure

```
.
├── pipeline/                  # Main pipeline code
│   ├── database/             # Database connections
│   │   ├── mysql_setup.py    # MySQL connection
│   │   ├── mongodb_setup.py  # MongoDB connection
│   │   ├── chroma_setup.py   # Chroma vector DB
│   │   └── database_manager.py # Unified DB manager
│   ├── llm/                  # LLM connections
│   │   └── llm_connections.py # LLM client management
│   ├── build_prompt.py       # Prompt building
│   └── search_profiles.py    # Search configurations
├── prompts/                  # Prompt templates
│   └── math_templates.json   # Math-specific prompts
├── data/                     # Data storage
├── main.py                   # Entry point
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
└── .env                      # Environment variables
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys and database credentials:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
MYSQL_HOST=localhost
MYSQL_USER=user
MYSQL_PASSWORD=password
MYSQL_DATABASE=db_name
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=db_name
CHROMA_COLLECTION=questions
```

## Usage

1. Run the pipeline:
```bash
python main.py
```

## Features

- PDF text extraction
- Question structuring
- Vector embeddings
- Question generation
- Similar question retrieval
- Database management (MySQL, MongoDB, Chroma)
- LLM integration (OpenAI, Anthropic)

## License

MIT 