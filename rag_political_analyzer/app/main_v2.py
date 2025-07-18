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
background_tasks_status = {}
orchestrator_instance: Optional[Any] = None # Will hold our main agent

@app.on_event("startup")
async def startup_event():
    """Initializes all agents and services when the API starts."""
    global orchestrator_instance
    print("API Startup: Initializing agent framework...")
    try:
        # Core Components
        embedding_model = get_embeddings_model()
        try:
            emb_dim = len(embedding_model.embed_query("test"))
        except Exception:
            emb_dim = 384

        vector_store = CodeDocVectorStore(embedding_dimension=emb_dim)
        llm_service = LLMService()

        # Agents
        # For simplicity, memory agent is still using the old file-based system
        from app.agents.memory_agent import MemoryAgent
        memory_agent = MemoryAgent()

        from app.agents.ingestion_agent import IngestionAgent
        ingestion_agent = IngestionAgent(vector_store=vector_store, embedding_model=embedding_model)

        # The other agents (Retrieval, AnswerGen) would be initialized here too
        # For now, let's assume they are part of a full orchestrator setup
        from app.agents.orchestrator_agent import OrchestratorAgent
        # A full orchestrator would need all agents, but for ingestion we only need the ingestion agent
        # This highlights a potential need to refactor the orchestrator or have multiple specialized orchestrators

        # For now, we will create a "dummy" orchestrator for the background task
        # that can call the ingestion agent. A better approach is a full orchestrator init.
        # Let's just initialize the IngestionAgent and call it directly from the background task
        # to simplify and avoid circular dependencies for this step.
        app.state.ingestion_agent = ingestion_agent
        print("Core components and IngestionAgent initialized successfully.")

    except Exception as e:
        print(f"FATAL: Failed to initialize components: {e}")
        app.state.ingestion_agent = None

# --- Helper Function for Background Ingestion ---
def run_ingestion_task(task_id: str, start_url: str, depth: int, max_pages: int):
    """The function that will be run in the background by FastAPI."""
    # This function runs in a separate thread, so it can't access app.state directly
    # A more robust solution involves a shared message queue (e.g., Redis, Celery)
    # For this project, we'll re-initialize the necessary components here.
    # This is less efficient but works for a simple background task model.
    print(f"Background task {task_id} started for URL: {start_url}")
    try:
        background_tasks_status[task_id] = "running"

        # Re-initialize components for this thread
        embedding_model = get_embeddings_model()
        try: emb_dim = len(embedding_model.embed_query("test"))
        except: emb_dim = 384
        vector_store = CodeDocVectorStore(embedding_dimension=emb_dim)
        from app.agents.ingestion_agent import IngestionAgent
        ingestion_agent = IngestionAgent(vector_store=vector_store, embedding_model=embedding_model)

        # Create and run the task
        task = {"type": "ingest_from_url", "data": {"start_url": start_url, "crawl_depth": depth, "max_pages": max_pages}}
        result = asyncio.run(ingestion_agent.execute(task))

        if result.get("status") == "success":
            background_tasks_status[task_id] = f"completed: {result.get('message')}"
        else:
            background_tasks_status[task_id] = f"failed: {result.get('message')}"

    except Exception as e:
        error_msg = f"failed: {str(e)}"
        print(f"Background ingestion task {task_id} failed: {e}")
        background_tasks_status[task_id] = error_msg

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
