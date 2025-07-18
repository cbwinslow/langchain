# Political Document Analysis RAG System

This project is a sophisticated, agent-based Retrieval-Augmented Generation (RAG) system designed to ingest, understand, and answer questions about political documents. It uses an ensemble of specialized agents coordinated by an orchestrator to manage a complex workflow involving NLP, embeddings, vector storage, and LLM inference.

## Architecture

The system is built around a multi-agent framework where each agent has a specific responsibility. This creates a modular and extensible system.

*   **Orchestrator Agent:** The central coordinator. It receives user requests and delegates tasks to the appropriate slave agents.
*   **Ingestion Agent:** Handles document processing. It loads files (PDF, TXT), chunks them, enriches them with spaCy-extracted metadata (entities, keywords), generates embeddings, and stores them in the vector database.
*   **Retrieval Agent:** Manages document retrieval. It can enrich user queries using spaCy, embed the query, and fetch the most relevant document chunks from the PGVector store.
*   **Answer Generation Agent:** Uses a Large Language Model (LLM) via OpenRouter to synthesize an answer based on the user's query and the context provided by the Retrieval Agent.
*   **Memory Agent:** Provides the system with memory. It maintains conversation history and can store learned "facts" or system logs in simple JSON files.

This entire system is exposed via a FastAPI server, which includes a "Mission Control Post" (MCP) for executing predefined utility scripts.

![Conceptual Architecture Diagram (Text-based)](https://via.placeholder.com/800x400.png?text=User+->+API+->+Orchestrator+->+[Retrieval,+AnswerGen,+Memory]+->+Response)
*(A proper diagram would show the detailed agent interactions)*

## Features

*   **Multi-Agent Framework:** A robust, coordinated system of specialized agents.
*   **Advanced NLP Ingestion:** Uses spaCy to extract named entities, noun chunks, and keywords during document ingestion, enriching the metadata for better retrieval.
*   **RAG Pipeline:**
    *   **Ingestion:** Supports PDF and TXT files.
    *   **Embedding:** Uses `sentence-transformers` for high-quality text embeddings.
    *   **Storage:** Leverages PostgreSQL with the `pgvector` extension for efficient vector storage and similarity search (using an HNSW index).
    *   **Retrieval:** Performs cosine similarity search to find the most relevant context.
    *   **Generation:** Connects to a wide variety of LLMs through `OpenRouter.ai`.
*   **Query Enhancement:** Optionally enriches user queries with spaCy-extracted key terms to improve retrieval accuracy.
*   **Conversational Memory:** Remembers past turns of a conversation for potential future context.
*   **Mission Control Post (MCP):** A FastAPI server that provides API endpoints for all core functionalities and includes a mechanism to list and execute predefined Python scripts for system maintenance and tasks.

## Setup and Installation

Follow these steps to get the RAG system running locally.

### 1. Prerequisites

*   Python 3.10+
*   PostgreSQL server (v12+) with the [pgvector extension](https://github.com/pgvector/pgvector) installed.
*   An API key from [OpenRouter.ai](https://openrouter.ai/).

### 2. Clone and Setup Environment

```bash
# Clone this repository (or use the provided project files)
# cd rag_political_analyzer

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# The application will automatically download the spaCy model on first run
# but you can also do it manually:
python -m spacy download en_core_web_sm
```

### 3. Configure Environment Variables

Create a `.env` file in the project root (`rag_political_analyzer/.env`). Use the `.env.example` file as a template.

```dotenv
# .env

# --- Database Configuration for PGVector ---
# Replace with your actual PostgreSQL connection details
DB_NAME="your_rag_db"
DB_USER="your_rag_user"
DB_PASSWORD="your_secure_password"
DB_HOST="localhost" # Or the hostname of your DB server
DB_PORT="5432"

# --- LLM API Keys ---
OPENROUTER_API_KEY="your_openrouter_api_key_here"

# --- (Optional) Other configurations ---
# Name of the default table/collection in the database
DB_DEFAULT_COLLECTION="political_documents"
```

### 4. Initialize the Database

The system is designed to create the necessary tables and extensions on startup. However, you must ensure the database (`your_rag_db`) and user (`your_rag_user`) exist in PostgreSQL and that the user has permissions to create tables and extensions.

Connect to your PostgreSQL instance and run:
```sql
CREATE DATABASE your_rag_db;
CREATE USER your_rag_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE your_rag_db TO your_rag_user;
-- Connect to the new database (\c your_rag_db in psql)
-- Then, the pgvector extension needs to be available. If it's installed server-wide,
-- the app will run `CREATE EXTENSION IF NOT EXISTS vector;` successfully.
```

### 5. Run the Application Server

Once the setup is complete, run the FastAPI server using Uvicorn:

```bash
# From the project root (rag_political_analyzer/)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start, initialize all agents and components, and will be accessible at `http://localhost:8000`.

## API Usage

You can interact with the API using tools like `curl`, Postman, or by visiting the interactive documentation at `http://localhost:8000/docs`.

### Upload a Document

Upload a `.txt` or `.pdf` file for ingestion.

```bash
curl -X POST -F "file=@/path/to/your/document.pdf" -F "user_id=test_user" http://localhost:8000/upload-document/
```

### Query the System

Ask a question about the ingested documents.

```bash
curl -X POST -F "query=What did the document say about fiscal policy?" \
             -F "user_id=test_user" \
             -F "k_retrieval=3" \
             -F "use_query_enrichment=true" \
             http://localhost:8000/query/
```

### List Available Scripts

See which utility scripts can be executed.

```bash
curl http://localhost:8000/scripts/
```

### Execute a Script

Run a script from the `app/scripts/` directory.

```bash
# Execute with no arguments
curl -X POST http://localhost:8000/scripts/execute/sample_script.py

# Execute with arguments
curl -X POST -F "script_args=arg1" -F "script_args=arg2" http://localhost:8000/scripts/execute/sample_script.py
```

## Running Tests

A conceptual unit test file is provided in `tests/test_units.py`. To run it:

```bash
# From the project root (rag_political_analyzer/)
python -m unittest discover -s ./tests -p 'test_*.py'
```
*Note: The test runner needs to support `asyncio` for the `MemoryAgent` tests. `pytest` with `pytest-asyncio` is recommended for a more robust test setup.*

An integration test is also available within the `orchestrator_agent.py` file and can be run directly (requires DB and API keys to be set up):
```bash
python app/agents/orchestrator_agent.py
```

## Future Work & Caveats

*   **Script Execution Security:** The script execution endpoint is **NOT SECURE** for production use. It lacks authentication, authorization, and sandboxing. Use with extreme caution.
*   **Dockerization:** The project would greatly benefit from a `docker-compose.yml` file to spin up the PostgreSQL/pgvector database and the FastAPI application in a containerized environment.
*   **Advanced Memory:** The `MemoryAgent` could be enhanced to use the vector database to find relevant past conversations, creating a more dynamic and context-aware system.
*   **Error Handling:** More granular error handling and reporting can be implemented across the agent framework.
*   **Agent-Zero:** The prompt mentioned "Agent-Zero". As this appears to be a private or non-public framework, this implementation interprets it as the foundational agent design principles embodied by the BaseAgent and Orchestrator. If a specific framework is intended, it could be integrated here.
```
