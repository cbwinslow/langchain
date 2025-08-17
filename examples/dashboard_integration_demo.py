"""Example showing how to connect LangChain with various third-party services.

This demo stitches together several integrations often used in production
LLM applications:

* Vercel dashboard boilerplate (see docs/dashboard_integration_example.md)
* Langfuse for tracing and monitoring
* Neo4j for graph data
* n8n for workflow automation
* Weaviate and PGVector for vector storage
* Supabase for authentication and data storage

The code below is illustrative and uses placeholders for credentials and
connection details. It is not meant to be executed as-is but serves as a
starting point for developers integrating these services.
"""

from typing import Any

# Langfuse integrates as a tracing and analytics backend.
try:
    from langfuse import Langfuse  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Langfuse = None  # type: ignore

# Neo4j graph integration
try:
    from langchain.graphs import Neo4jGraph
except ImportError:  # pragma: no cover - optional dependency
    Neo4jGraph = None  # type: ignore

# Vector stores
try:
    from langchain.vectorstores import PGVector, Weaviate
except ImportError:  # pragma: no cover - optional dependency
    PGVector, Weaviate = None, None  # type: ignore

# Supabase client
try:
    from supabase import create_client, Client  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    create_client, Client = None, None  # type: ignore


def build_demo() -> dict[str, Any]:
    """Return initialized service clients for the dashboard demo.

    This function only demonstrates the configuration pattern. Replace the
    placeholder values with real credentials and hosts before executing.
    """

    services: dict[str, Any] = {}

    if Langfuse:
        services["langfuse"] = Langfuse(
            api_key="LF_API_KEY",
            host="https://cloud.langfuse.com",
        )

    if Neo4jGraph:
        services["neo4j"] = Neo4jGraph(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password",
        )

    if Weaviate:
        services["weaviate"] = Weaviate(
            url="http://localhost:8080",
            api_key="WEAVIATE_API_KEY",
        )

    if PGVector:
        services["pgvector"] = PGVector.from_collection_name(
            connection_string="postgresql+psycopg://user:password@localhost:5432/db",
            collection_name="langchain",
        )

    if create_client:
        services["supabase"] = create_client(
            "https://SUPABASE_PROJECT.supabase.co",
            "SUPABASE_ANON_KEY",
        )

    # n8n typically exposes HTTP endpoints for triggering workflows.
    services["n8n_webhook_url"] = "https://n8n.example.com/webhook/langchain"

    return services


if __name__ == "__main__":  # pragma: no cover
    demo_services = build_demo()
    for name, client in demo_services.items():
        print(f"Initialized {name}: {client}")
