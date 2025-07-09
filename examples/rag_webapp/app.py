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

# RetrievalQA setup mirrors the pattern in LangChain docs
llm = get_llm()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=store.as_retriever())
qa_graph = create_workflow(store.as_retriever(), llm)

class Query(BaseModel):
    question: str


class IngestText(BaseModel):
    content: str

@app.post("/query")
async def query(q: Query, _: str | None = Depends(verify_token)):
    result = qa_chain.invoke({"query": q.question})
    return {"answer": result}


@app.post("/query_graph")
async def query_graph(q: Query, _: str | None = Depends(verify_token)):
    result = qa_graph.invoke(q.question)
    return {"answer": result}


@app.post("/ingest_text")
async def ingest_text(q: IngestText, _: str | None = Depends(verify_token)):
    ingest_text_content(q.content, db_dir, store_type, embedding_model)
    return {"status": "ok"}


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...), _: str | None = Depends(verify_token)):
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    ingest_pdf_file(tmp_path, db_dir, store_type, embedding_model)
    os.remove(tmp_path)
    return {"status": "ok"}

