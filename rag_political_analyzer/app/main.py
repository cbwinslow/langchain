# app/main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import List, Dict, Any, Optional
import shutil # For saving uploaded files

# Adjust import paths if needed based on your project structure
from app.core.ingestion import get_embeddings_model, get_spacy_model
from app.core.vector_store import PGVectorStore, DEFAULT_MEMORY_DIR
from app.core.retrieval import Retriever as CoreRetriever
from app.core.llm_services import LLMService

from app.agents.memory_agent import MemoryAgent, DEFAULT_MEMORY_FILE, DEFAULT_CONVERSATION_MEMORY_FILE
from app.agents.ingestion_agent import IngestionAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.answer_generation_agent import AnswerGenerationAgent
from app.agents.orchestrator_agent import OrchestratorAgent

import subprocess # For script execution
import sys # For script execution path

# --- Application Setup ---
app = FastAPI(
    title="Political Document RAG API with Agent Framework",
    description="API for ingesting and querying political documents using an agent-based RAG system.",
    version="0.2.0"
)

# --- Global Components (Initialize once) ---
from dotenv import load_dotenv
load_dotenv()

orchestrator_instance: Optional[OrchestratorAgent] = None

@app.on_event("startup")
async def startup_event():
    global orchestrator_instance
    print("Application startup: Initializing agents and core components...")
    try:
        # Core services first
        embedding_model_instance = get_embeddings_model()
        spacy_model_instance = get_spacy_model()

        try:
            sample_embedding_for_dim = embedding_model_instance.embed_query("test")
            EMBEDDING_DIMENSION = len(sample_embedding_for_dim)
        except Exception as e:
            print(f"Warning: Could not dynamically determine embedding dimension. Defaulting to 384. Error: {e}")
            EMBEDDING_DIMENSION = 384

        vector_store_instance = PGVectorStore(
            collection_name=os.getenv("DB_DEFAULT_COLLECTION", "political_documents"),
            embedding_dimension=EMBEDDING_DIMENSION
        )

        core_retriever_instance = CoreRetriever(
            vector_store=vector_store_instance,
            embedding_model=embedding_model_instance,
            spacy_model=spacy_model_instance
        )
        llm_service_instance = LLMService()

        # Agents
        # Adjust memory file paths to be relative to project root or ensure DEFAULT_MEMORY_DIR is correct
        # If main.py is in app/, and DEFAULT_MEMORY_DIR is "memory_system", path becomes "app/memory_system"
        # For simplicity, let's assume DEFAULT_MEMORY_DIR is at project root if not specified with absolute path.
        # MemoryAgent expects paths relative to where it's run or absolute.
        # Let's make paths relative to the project root (rag_political_analyzer)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # app -> project_root

        memory_agent = MemoryAgent(
            memory_file_path=os.path.join(project_root, DEFAULT_MEMORY_DIR, os.path.basename(DEFAULT_MEMORY_FILE)),
            conversation_memory_path=os.path.join(project_root, DEFAULT_MEMORY_DIR, os.path.basename(DEFAULT_CONVERSATION_MEMORY_FILE))
        )
        ingestion_agent = IngestionAgent(vector_store=vector_store_instance, embedding_model=embedding_model_instance)
        retrieval_agent = RetrievalAgent(core_retriever=core_retriever_instance)
        answer_generation_agent = AnswerGenerationAgent(llm_service=llm_service_instance)

        orchestrator_instance = OrchestratorAgent(
            memory_agent=memory_agent,
            ingestion_agent=ingestion_agent,
            retrieval_agent=retrieval_agent,
            answer_generation_agent=answer_generation_agent
        )
        print("OrchestratorAgent and all dependent components initialized successfully.")

    except Exception as e:
        print(f"FATAL: Could not initialize OrchestratorAgent or its components during startup: {e}")
        import traceback
        traceback.print_exc()
        orchestrator_instance = None # Ensure it's None if setup fails
        print("Application will run with non-functional RAG capabilities.")


# --- File Upload Directory ---
# Ensure UPLOAD_DIR is relative to the project root or an absolute path
project_root_for_uploads = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_DIR = os.path.join(project_root_for_uploads, "uploaded_files_api")
os.makedirs(UPLOAD_DIR, exist_ok=True)
SCRIPTS_DIR = os.path.join(project_root_for_uploads, "app", "scripts") # Assuming scripts are in app/scripts


# --- API Endpoints ---

@app.post("/upload-document/", tags=["Ingestion"])
async def upload_document_endpoint(file: UploadFile = File(...), user_id: str = Form("api_user")):
    if not orchestrator_instance:
        raise HTTPException(status_code=503, detail="Orchestrator service not available due to initialization error.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        task = {
            "type": "ingest_file",
            "data": {"file_path": file_path, "user_id": user_id}
        }
        result = await orchestrator_instance.execute(task)

        if result.get("status") == "success":
            return {"message": result.get("message", "Document processed."), "details": result}
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Failed to process document."))

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error processing document {file.filename} via API: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Optional: Decide on file retention policy
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        pass


@app.post("/query/", tags=["Querying"])
async def query_rag_system(
    query: str = Form(...),
    user_id: str = Form("api_user"),
    k_retrieval: int = Form(5),
    use_query_enrichment: bool = Form(True),
    llm_model_name: Optional[str] = Form(None)
):
    if not orchestrator_instance:
        raise HTTPException(status_code=503, detail="Orchestrator service not available due to initialization error.")

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        task_data = {
            "query": query,
            "user_id": user_id,
            "k_retrieval": k_retrieval,
            "use_query_enrichment": use_query_enrichment
        }
        if llm_model_name:
            task_data["llm_model_name"] = llm_model_name

        task = {"type": "user_query", "data": task_data}
        result = await orchestrator_instance.execute(task)

        if result.get("status") == "success":
            return result # Orchestrator already formats a good response
        else:
            # Try to provide more specific error from orchestrator if available
            error_message = result.get("message", "Failed to process query.")
            if "retrieval_details" in result and result["retrieval_details"].get("status") != "success":
                error_message += f" Retrieval error: {result['retrieval_details'].get('message')}"
            elif "answer_generation_details" in result and result.get("answer_generation_details",{}).get("status") != "success": # Assuming Orchestrator might add this
                 error_message += f" Answer generation error: {result['answer_generation_details'].get('message')}"
            raise HTTPException(status_code=500, detail=error_message)

    except Exception as e:
        print(f"Error during query processing via API: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# --- Script Execution Endpoints ---
@app.get("/scripts/", tags=["Scripts"])
async def list_scripts():
    """Lists available scripts in the app/scripts directory."""
    if not os.path.exists(SCRIPTS_DIR):
        return {"message": "Scripts directory not found.", "scripts": []}
    try:
        scripts = [f for f in os.listdir(SCRIPTS_DIR) if f.endswith(".py") and not f.startswith("_")]
        return {"scripts": scripts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing scripts: {str(e)}")

@app.post("/scripts/execute/{script_name}", tags=["Scripts"])
async def execute_script(script_name: str, script_args: List[str] = Form(None)):
    """
    Executes a predefined Python script from the app/scripts directory.
    WARNING: THIS IS A HUGE SECURITY RISK IF NOT PROPERLY SECURED AND VALIDATED.
    For this project, it assumes scripts are trusted and safe.
    In a real-world scenario, use extreme caution: whitelist scripts, sanitize inputs, run in isolated environments.
    """
    if not os.path.exists(SCRIPTS_DIR):
        raise HTTPException(status_code=404, detail="Scripts directory not found.")

    # Basic validation: ensure script_name is just a filename and ends with .py
    if ".." in script_name or not script_name.endswith(".py") or os.path.dirname(script_name):
        raise HTTPException(status_code=400, detail="Invalid script name.")

    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path) or not os.path.isfile(script_path):
        raise HTTPException(status_code=404, detail=f"Script '{script_name}' not found.")

    try:
        command = [sys.executable, script_path] # Use the same python interpreter
        if script_args:
            command.extend(script_args)

        print(f"Executing script: {' '.join(command)}")
        # Using subprocess.run for simplicity. For long-running scripts, consider Popen and streaming.
        # Timeout is important for API responsiveness.
        process = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False) # check=False to get output even on error

        if process.returncode == 0:
            return {
                "script": script_name,
                "status": "success",
                "return_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr
            }
        else:
            return {
                "script": script_name,
                "status": "error",
                "return_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr
            }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail=f"Script '{script_name}' timed out after 60 seconds.")
    except Exception as e:
        print(f"Error executing script {script_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing script: {str(e)}")


@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Political Document RAG API (Agentic Version). Use /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    print("Attempting to run Uvicorn server for Agentic RAG API...")
    print("Make sure your .env file is configured in the project root ('rag_political_analyzer').")
    print("Database tables should be initialized.")

    # Note: orchestrator_instance is initialized in startup_event, so no direct check here.
    # Uvicorn will run the startup event.

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
