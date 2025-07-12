# app/main_v2.py
# A new main file for the Code Documentation RAG system
import os
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel
from typing import List, Optional

# Core components for the new RAG system
from app.ingestion_pipeline import IngestionPipeline
from app.core.vector_store import CodeDocVectorStore
from app.core.ingestion import get_embeddings_model
from app.core.llm_services import LLMService

# --- Application Setup ---
app = FastAPI(
    title="Code Documentation RAG API",
    description="An API to crawl, ingest, and query code documentation.",
    version="1.0.0"
)

# --- Global Components & State ---
# In a production app, this state might be managed differently (e.g., Redis)
# For now, a simple dict to track the status of background tasks.
background_tasks_status = {}

# Initialize core services on startup
# This is a simplified approach. Dependency injection frameworks would be better for larger apps.
try:
    print("Initializing core components...")
    embedding_model = get_embeddings_model()
    try:
        sample_emb = embedding_model.embed_query("test")
        emb_dim = len(sample_emb)
    except Exception:
        emb_dim = 384 # Fallback

    vector_store = CodeDocVectorStore(embedding_dimension=emb_dim)
    llm_service = LLMService() # Assumes .env is configured
    print("Core components initialized successfully.")
except Exception as e:
    print(f"FATAL: Failed to initialize core components: {e}")
    # Set to None so endpoints can fail gracefully
    vector_store = None
    llm_service = None
    embedding_model = None

# --- Helper Function for Background Ingestion ---
def run_ingestion_task(task_id: str, start_url: str, depth: int, max_pages: int):
    """The function that will be run in the background by FastAPI."""
    try:
        background_tasks_status[task_id] = "running"
        pipeline = IngestionPipeline() # Re-initialize pipeline components in the new process/thread
        asyncio.run(pipeline.run_for_url(start_url, depth, max_pages))
        background_tasks_status[task_id] = "completed"
    except Exception as e:
        print(f"Background ingestion task {task_id} failed: {e}")
        background_tasks_status[task_id] = f"failed: {str(e)}"

# --- Pydantic Models for API ---
class CrawlRequest(BaseModel):
    start_url: str
    crawl_depth: int = 1
    max_pages: int = 20

class TaskStatus(BaseModel):
    task_id: str
    status: str

class DataSource(BaseModel):
    id: int
    url: str
    name: Optional[str] = None
    created_at: Optional[str] = None

# --- API Endpoints ---

@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Code Documentation RAG API."}

# --- Ingestion and Data Management Endpoints ---

@app.post("/ingest/", status_code=202, response_model=TaskStatus, tags=["Ingestion & Data Management"])
async def start_ingestion(request: CrawlRequest, background_tasks: BackgroundTasks):
    """
    Starts a background task to crawl and ingest a documentation website.
    """
    task_id = f"ingest-{request.start_url}-{os.urandom(4).hex()}"
    background_tasks_status[task_id] = "pending"

    background_tasks.add_task(
        run_ingestion_task,
        task_id,
        request.start_url,
        request.crawl_depth,
        request.max_pages
    )

    return {"task_id": task_id, "status": "pending"}

@app.get("/ingest/status/{task_id}", response_model=TaskStatus, tags=["Ingestion & Data Management"])
async def get_ingestion_status(task_id: str):
    """
    Checks the status of a background ingestion task.
    """
    status = background_tasks_status.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": status}

@app.get("/sources/", response_model=List[DataSource], tags=["Ingestion & Data Management"])
async def list_data_sources():
    """
    Lists all the data sources that have been ingested.
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not available.")
    try:
        sources = vector_store.list_data_sources()
        # Convert datetime objects for JSON serialization if they are not already strings
        return [{"id": s["id"], "url": s["url"], "name": s["name"], "created_at": str(s["created_at"])} for s in sources]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sources/{source_id}", tags=["Ingestion & Data Management"])
async def delete_source(source_id: int):
    """
    Deletes a data source and all its associated content chunks.
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not available.")
    try:
        deleted_count = vector_store.delete_data_source(source_id)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Data source not found.")
        return {"message": f"Successfully deleted data source {source_id} and its content."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- RAG Query Endpoint ---

class QueryRequest(BaseModel):
    query: str
    k_text: int = 3  # Number of text chunks to retrieve
    k_code: int = 2  # Number of code chunks to retrieve
    llm_model: Optional[str] = None # e.g., "ollama/llama3" or "mistralai/mistral-7b-instruct"

class QueryResponse(BaseModel):
    answer: str
    retrieved_context: List[Dict[str, Any]]

@app.post("/query/", response_model=QueryResponse, tags=["RAG Query"])
async def query_rag_system(request: QueryRequest):
    """
    Receives a query and returns a synthesized answer based on retrieved context.
    """
    if not vector_store or not llm_service or not embedding_model:
        raise HTTPException(status_code=503, detail="Core RAG services not available.")

    try:
        # 1. Embed the query
        query_embedding = embedding_model.embed_query(request.query)

        # 2. Retrieve separate text and code chunks
        text_chunks = vector_store.similarity_search(query_embedding, k=request.k_text, chunk_type='text')
        code_chunks = vector_store.similarity_search(query_embedding, k=request.k_code, chunk_type='code')

        retrieved_context = text_chunks + code_chunks

        # De-duplicate context based on content if necessary (simple approach)
        unique_context = {item['content']: item for item in retrieved_context}.values()

        if not unique_context:
            # Fallback: If no context is found, ask the LLM to answer from general knowledge
            model = request.llm_model or "mistralai/mistral-7b-instruct" # Sensible default
            answer = llm_service.generate_response(
                prompt=request.query,
                model_name=model,
                system_message="You are a helpful assistant for code and documentation. The user's query did not match any stored documents, so answer from your general knowledge."
            )
            return {"answer": answer, "retrieved_context": []}

        # 3. Construct a detailed prompt for the LLM
        context_str = ""
        for item in unique_context:
            context_type = item.get('chunk_type', 'text').upper()
            source_url = item.get('metadata', {}).get('source_url', item.get('source_url', 'Unknown'))
            context_str += f"--- Source: {source_url} (Type: {context_type}) ---\n"
            context_str += item['content']
            context_str += "\n\n"

        prompt = f"""
        You are an expert programmer and documentation analyst. Answer the user's question based *only* on the provided context, which contains both explanatory text and code snippets.

        Combine information from the text and code to provide a comprehensive answer. When referencing code, explain what it does.

        Provided Context:
        {context_str}

        User's Question: {request.query}

        Answer:
        """

        # 4. Call the LLM to generate the final answer
        model_to_use = request.llm_model
        if not model_to_use:
            # Smart default: prefer OpenRouter for its power, fallback to Ollama if available
            if llm_service.openrouter_client:
                model_to_use = "mistralai/mistral-7b-instruct"
            elif llm_service.ollama_client:
                model_to_use = "ollama/llama3"
            else:
                raise HTTPException(status_code=503, detail="No LLM providers are configured.")

        final_answer = llm_service.generate_response(
            prompt=prompt,
            model_name=model_to_use,
            system_message="You are an expert programmer and documentation analyst." # System message for generate_response
        )

        return {
            "answer": final_answer,
            "retrieved_context": list(unique_context)
        }

    except Exception as e:
        print(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```
