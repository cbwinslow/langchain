"""Simple FastAPI service for querying a vector store."""

import os
from fastapi import FastAPI, Depends, Header, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.llms import HuggingFaceHub
try:
    from langchain_ollama import OllamaLLM
except Exception:  # pragma: no cover - optional dependency
    OllamaLLM = None
try:  # optional providers for troubleshooting
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional dependency
    ChatGroq = None
try:
    from langchain_localai import ChatLocalAI
except Exception:  # pragma: no cover - optional dependency
    ChatLocalAI = None
try:
    from langchain_openrouter import ChatOpenRouter
except Exception:  # pragma: no cover - optional dependency
    ChatOpenRouter = None
from .fake_llm import FakeLLM
from .ingest import ingest_text_content, ingest_pdf_file
from .langgraph_workflow import create_workflow
import langsmith

try:
    from langchain_elasticsearch import ElasticsearchStore
except Exception:  # pragma: no cover
    ElasticsearchStore = None

try:
    from langchain_community.vectorstores import Weaviate
except Exception:  # pragma: no cover
    Weaviate = None

app = FastAPI()


def configure_langsmith() -> None:
    """Enable LangSmith tracing if API key is set."""
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        langsmith.Client()


configure_langsmith()


def verify_token(x_api_token: str | None = Header(None)) -> str | None:
    expected = os.getenv("API_TOKEN")
    if expected and x_api_token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return x_api_token


def get_llm() -> "BaseLLM":
    """Return an LLM based on environment configuration."""
    use_fake = os.environ.get("USE_FAKE_LLM", "false").lower() == "true"
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if use_fake:
        return FakeLLM()
    if provider == "groq" and ChatGroq is not None:
        return ChatGroq()
    if provider == "ollama" and OllamaLLM is not None:
        return OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3"))
    if provider == "localai" and ChatLocalAI is not None:
        return ChatLocalAI()
    if provider == "openrouter" and ChatOpenRouter is not None:
        return ChatOpenRouter()
    return HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0})

db_dir = os.environ.get("PERSIST_DIR", "chroma_db")
embedding_model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
if embedding_model == "fake":
    embedding = FakeEmbeddings(size=10)
else:
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)

store_type = os.environ.get("VECTOR_STORE_TYPE", "chroma")
try:
    if store_type == "chroma":
        store = Chroma(persist_directory=db_dir, embedding_function=embedding)
    elif store_type == "elastic":
        if ElasticsearchStore is None:
            raise RuntimeError("Elasticsearch dependencies are missing")
        store = ElasticsearchStore(
            es_url=os.getenv("ES_URL", "http://localhost:9200"),
            index_name=os.getenv("ES_INDEX", "langchain_index"),
            embedding=embedding,
        )
    elif store_type == "weaviate":
        if Weaviate is None:
            raise RuntimeError("Weaviate dependencies are missing")
        store = Weaviate(
            url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            index_name=os.getenv("WEAVIATE_INDEX", "LangChain")
        )
    else:
        raise ValueError(f"Unsupported store type: {store_type}")
except Exception as exc:
    raise RuntimeError(f"Failed to initialize vector store: {exc}") from exc

# --- Global Configuration State ---
# These will be initialized from environment variables on startup,
# and can be updated via the /settings/update endpoint.

CURRENT_SETTINGS = {
    "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "ollama").lower(),
    "OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", "llama3"),
    "USE_FAKE_LLM": os.environ.get("USE_FAKE_LLM", "false").lower() == "true",
    "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "VECTOR_STORE_TYPE": os.environ.get("VECTOR_STORE_TYPE", "chroma"),
    "PERSIST_DIR": os.environ.get("PERSIST_DIR", "chroma_db"),
    "ES_URL": os.getenv("ES_URL", "http://localhost:9200"),
    "ES_INDEX": os.getenv("ES_INDEX", "langchain_index"),
    "WEAVIATE_URL": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
    "WEAVIATE_INDEX": os.getenv("WEAVIATE_INDEX", "LangChain"),
}

# --- LLM and Vector Store Initialization ---
# These will now be functions that can be called to get instances
# based on CURRENT_SETTINGS. We need to be careful about global instances
# if we want true dynamic updates without restarting.
# For simplicity in this iteration, we'll re-initialize.
# This means existing chains/agents might not immediately pick up changes
# unless they are also re-initialized. A more robust solution would involve
# a more sophisticated context management or dependency injection.

_llm_instance = None
_store_instance = None
_embedding_instance = None
_qa_chain_instance = None
_qa_graph_instance = None

def get_active_llm():
    global _llm_instance
    # This function will now re-initialize if settings have changed or not initialized.
    # A proper check for changed settings would be more robust.
    # For now, assume if called, it might need re-init based on CURRENT_SETTINGS.

    provider = CURRENT_SETTINGS["LLM_PROVIDER"]
    if CURRENT_SETTINGS["USE_FAKE_LLM"]:
        _llm_instance = FakeLLM()
        return _llm_instance
    if provider == "groq" and ChatGroq is not None:
        _llm_instance = ChatGroq()
    elif provider == "ollama" and OllamaLLM is not None:
        _llm_instance = OllamaLLM(model=CURRENT_SETTINGS["OLLAMA_MODEL"])
    elif provider == "localai" and ChatLocalAI is not None:
        _llm_instance = ChatLocalAI()
    elif provider == "openrouter" and ChatOpenRouter is not None:
        _llm_instance = ChatOpenRouter()
    else:
        # Fallback or default if specific provider not found/configured
        _llm_instance = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0})
    return _llm_instance

def get_active_embedding_model():
    global _embedding_instance
    model_name = CURRENT_SETTINGS["EMBEDDING_MODEL"]
    if model_name == "fake":
        _embedding_instance = FakeEmbeddings(size=10)
    else:
        _embedding_instance = HuggingFaceEmbeddings(model_name=model_name)
    return _embedding_instance

def get_active_vector_store():
    global _store_instance
    active_embedding = get_active_embedding_model()
    store_type = CURRENT_SETTINGS["VECTOR_STORE_TYPE"]

    try:
        if store_type == "chroma":
            _store_instance = Chroma(persist_directory=CURRENT_SETTINGS["PERSIST_DIR"], embedding_function=active_embedding)
        elif store_type == "elastic":
            if ElasticsearchStore is None:
                raise RuntimeError("Elasticsearch dependencies are missing")
            _store_instance = ElasticsearchStore(
                es_url=CURRENT_SETTINGS["ES_URL"],
                index_name=CURRENT_SETTINGS["ES_INDEX"],
                embedding=active_embedding,
            )
        elif store_type == "weaviate":
            if Weaviate is None:
                raise RuntimeError("Weaviate dependencies are missing")
            _store_instance = Weaviate(
                url=CURRENT_SETTINGS["WEAVIATE_URL"],
                index_name=CURRENT_SETTINGS["WEAVIATE_INDEX"],
                embedding=active_embedding # Weaviate might handle embeddings differently if client manages it
            )
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
    except Exception as exc:
        # Fallback to a default in-memory Chroma if specific store fails
        print(f"Failed to initialize vector store {store_type}: {exc}. Falling back to in-memory Chroma.")
        _store_instance = Chroma(embedding_function=active_embedding) # In-memory
    return _store_instance

# Initialize on first load
get_active_llm()
get_active_vector_store()


# --- Settings Endpoints ---
class SettingsModel(BaseModel):
    LLM_PROVIDER: Optional[str] = None
    OLLAMA_MODEL: Optional[str] = None
    USE_FAKE_LLM: Optional[bool] = None
    EMBEDDING_MODEL: Optional[str] = None
    VECTOR_STORE_TYPE: Optional[str] = None

@app.get("/settings/current")
async def get_current_settings(_: str | None = Depends(verify_token)):
    return CURRENT_SETTINGS

@app.post("/settings/update")
async def update_settings(settings: SettingsModel, _: str | None = Depends(verify_token)):
    global _llm_instance, _store_instance, _embedding_instance, _qa_chain_instance, _qa_graph_instance

    updated = False
    if settings.LLM_PROVIDER is not None and settings.LLM_PROVIDER != CURRENT_SETTINGS["LLM_PROVIDER"]:
        CURRENT_SETTINGS["LLM_PROVIDER"] = settings.LLM_PROVIDER.lower()
        updated = True
    if settings.OLLAMA_MODEL is not None and settings.OLLAMA_MODEL != CURRENT_SETTINGS["OLLAMA_MODEL"]:
        CURRENT_SETTINGS["OLLAMA_MODEL"] = settings.OLLAMA_MODEL
        updated = True
    if settings.USE_FAKE_LLM is not None and settings.USE_FAKE_LLM != CURRENT_SETTINGS["USE_FAKE_LLM"]:
        CURRENT_SETTINGS["USE_FAKE_LLM"] = settings.USE_FAKE_LLM
        updated = True
    if settings.EMBEDDING_MODEL is not None and settings.EMBEDDING_MODEL != CURRENT_SETTINGS["EMBEDDING_MODEL"]:
        CURRENT_SETTINGS["EMBEDDING_MODEL"] = settings.EMBEDDING_MODEL
        # Embedding model change implies vector store re-init
        updated = True
    if settings.VECTOR_STORE_TYPE is not None and settings.VECTOR_STORE_TYPE != CURRENT_SETTINGS["VECTOR_STORE_TYPE"]:
        CURRENT_SETTINGS["VECTOR_STORE_TYPE"] = settings.VECTOR_STORE_TYPE.lower()
        updated = True

    if updated:
        # Re-initialize LLM and Vector Store if relevant settings changed
        # This is a simplified approach. In a production system, you'd need to handle
        # this more carefully, potentially with locks or context managers if these
        # instances are used by concurrent requests.
        print("Settings updated, re-initializing LLM and Vector Store...")
        get_active_llm()
        get_active_vector_store()
        # Reset dependent chains so they pick up new instances on next use
        _qa_chain_instance = None
        _qa_graph_instance = None
        # Potentially re-initialize other global/cached agent executors too if they depend on these.

    return {"status": "Settings updated" if updated else "No changes detected", "new_settings": CURRENT_SETTINGS}


# --- Modified Endpoints to use dynamic getters ---

class Query(BaseModel):
    question: str

class IngestText(BaseModel):
    content: str

@app.post("/query")
async def query(q: Query, _: str | None = Depends(verify_token)):
    global _qa_chain_instance
    active_llm = get_active_llm()
    active_store = get_active_vector_store()
    if _qa_chain_instance is None: # Or if settings changed that affect it
        _qa_chain_instance = RetrievalQA.from_chain_type(llm=active_llm, retriever=active_store.as_retriever())
    result = _qa_chain_instance.invoke({"query": q.question})
    return {"answer": result}


@app.post("/query_graph")
async def query_graph(q: Query, _: str | None = Depends(verify_token)):
    global _qa_graph_instance
    active_llm = get_active_llm()
    active_store = get_active_vector_store()
    if _qa_graph_instance is None: # Or if settings changed
        _qa_graph_instance = create_workflow(active_store.as_retriever(), active_llm)
    result = _qa_graph_instance.invoke(q.question)
    return {"answer": result}


@app.post("/ingest_text")
async def ingest_text(q: IngestText, _: str | None = Depends(verify_token)):
    # Ingestion depends on the current vector store and embedding model
    # For simplicity, we assume the vector store is re-initialized if settings change.
    # A more robust implementation might involve invalidating/rebuilding indexes.
    active_store = get_active_vector_store() # Ensures store is using current embedding model

    # The ingest_text_content function internally creates its own Chroma instance based on env vars.
    # This needs to be modified or it won't respect dynamic settings.
    # For now, this endpoint might not fully respect dynamic changes without further mods to ingest_text_content
    ingest_text_content(q.content, CURRENT_SETTINGS["PERSIST_DIR"], CURRENT_SETTINGS["VECTOR_STORE_TYPE"], CURRENT_SETTINGS["EMBEDDING_MODEL"])
    return {"status": "ok"}


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...), _: str | None = Depends(verify_token)):
    # Similar to ingest_text, ingest_pdf_file needs to respect dynamic settings.
    active_store = get_active_vector_store()
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    ingest_pdf_file(tmp_path, CURRENT_SETTINGS["PERSIST_DIR"], CURRENT_SETTINGS["VECTOR_STORE_TYPE"], CURRENT_SETTINGS["EMBEDDING_MODEL"])
    os.remove(tmp_path)
    return {"status": "ok"}


# --- Agent with Wikipedia Tool ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate

# Define the prompt template for the agent
react_prompt = PromptTemplate.from_template("""\
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")


@app.post("/agent/wikipedia")
async def agent_wikipedia(q: Query, _: str | None = Depends(verify_token)):
    """Agent that uses Wikipedia to answer questions."""
    active_llm = get_active_llm() # Use dynamic LLM
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [wikipedia_tool]

    agent = create_react_agent(active_llm, tools, react_prompt)
    # This agent_executor is created on each call, so it will use the latest llm.
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    try:
        result = await agent_executor.ainvoke({"input": q.question})
        return {"answer": result.get("output", "No output found.")}
    except Exception as e:
        print(f"Error in Wikipedia Agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Cookbook Demos ---
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.pydantic_v1 import BaseModel as PydanticV1BaseModel
from typing import Optional

# --- SQL Agent Demo ---
DB_PATH = "sample_company.db"

@app.post("/cookbook/sql-agent")
async def cookbook_sql_agent(q: Query, _: str | None = Depends(verify_token)):
    """Agent that can query a sample SQL database."""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail=f"Database not found at {DB_PATH}")

    db = SQLDatabase(engine_args={"connect_args": {"check_same_thread": False}}, uri=f"sqlite:///{DB_PATH}")
    active_llm = get_active_llm() # Use dynamic LLM

    try:
        # SQL Agent executor created on each call
        sql_agent_executor = create_sql_agent(active_llm, db=db, agent_type="openai-tools", verbose=True)
        result = await sql_agent_executor.ainvoke({"input": q.question})
        return {"answer": result.get("output", "No output found.")}
    except Exception as e:
        print(f"Error in SQL Agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Structured Output Demo ---
class Person(PydanticV1BaseModel):
    """Information about a person."""
    name: str
    age: Optional[int] = None

class StructuredOutputQuery(BaseModel):
    text: str

@app.post("/cookbook/structured-output")
async def cookbook_structured_output(q: StructuredOutputQuery, _: str | None = Depends(verify_token)):
    """Extracts structured data from text."""
    active_llm = get_active_llm() # Use dynamic LLM
    try:
        structured_llm = active_llm.with_structured_output(Person)
        result = await structured_llm.ainvoke(q.text)
        return {"answer": result.dict()}
    except Exception as e:
        print(f"Error in Structured Output: {e}")
        raise HTTPException(status_code=500, detail=str(e))
