# app/core/vector_store.py
import os
import psycopg2
from psycopg2.extras import execute_values, Json
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv(dotenv_path='../.env')

DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# --- Table Names ---
DATA_SOURCES_TABLE = "data_sources"
CONTENT_CHUNKS_TABLE = "content_chunks"

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    register_vector(conn)
    return conn

class CodeDocVectorStore:
    """
    Manages vector storage for code documentation, using a structured approach
    with separate tables for data sources and content chunks.
    """
    def __init__(self, embedding_dimension: int = 384):
        self.embedding_dimension = embedding_dimension
        self._ensure_schema_exists()

    def _ensure_schema_exists(self):
        """Ensures the necessary tables and indexes exist in the database."""
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # --- data_sources Table ---
                # Tracks the origin of ingested content (e.g., a documentation website URL)
                create_sources_table_query = f"""
                CREATE TABLE IF NOT EXISTS {DATA_SOURCES_TABLE} (
                    id SERIAL PRIMARY KEY,
                    source_url TEXT UNIQUE NOT NULL, -- The root URL or unique identifier for the source
                    source_name TEXT, -- A human-readable name for the source
                    last_crawled_at TIMESTAMPTZ,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
                cur.execute(create_sources_table_query)
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_source_url ON {DATA_SOURCES_TABLE} (source_url);")

                # --- content_chunks Table ---
                # Stores the actual text/code chunks and their embeddings
                create_chunks_table_query = f"""
                CREATE TABLE IF NOT EXISTS {CONTENT_CHUNKS_TABLE} (
                    id SERIAL PRIMARY KEY,
                    source_id INTEGER NOT NULL REFERENCES {DATA_SOURCES_TABLE}(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    chunk_type TEXT NOT NULL, -- 'text' or 'code'
                    embedding VECTOR({self.embedding_dimension}),
                    metadata JSONB, -- For storing page_url, language, etc.
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
                cur.execute(create_chunks_table_query)

                # Create HNSW index for efficient similarity search
                create_index_query = f"""
                CREATE INDEX IF NOT EXISTS idx_{CONTENT_CHUNKS_TABLE}_embedding
                ON {CONTENT_CHUNKS_TABLE}
                USING hnsw (embedding vector_cosine_ops);
                """
                cur.execute(create_index_query)
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_chunk_type ON {CONTENT_CHUNKS_TABLE} (chunk_type);")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_source_id ON {CONTENT_CHUNKS_TABLE} (source_id);")

                conn.commit()
            print("Database schema (data_sources, content_chunks) ensured successfully.")
        except psycopg2.Error as e:
            print(f"Database error ensuring schema: {e}")
            if conn: conn.rollback()
            raise
        finally:
            if conn: conn.close()

    def add_data_source(self, url: str, name: Optional[str] = None, metadata: Optional[Dict] = None) -> int:
        """Adds a new data source and returns its ID. If source exists, returns its ID."""
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                # Upsert logic: Insert or do nothing if the URL already exists, then return the ID.
                cur.execute(f"""
                    INSERT INTO {DATA_SOURCES_TABLE} (source_url, source_name, metadata)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_url) DO NOTHING;
                """, (url, name, Json(metadata) if metadata else None))

                cur.execute(f"SELECT id FROM {DATA_SOURCES_TABLE} WHERE source_url = %s;", (url,))
                source_id = cur.fetchone()[0]
                conn.commit()
                return source_id
        except psycopg2.Error as e:
            print(f"Database error adding data source: {e}")
            if conn: conn.rollback()
            raise
        finally:
            if conn: conn.close()

    def add_content_chunks(self, source_id: int, chunks: List[Dict[str, Any]]):
        """
        Adds multiple content chunks associated with a data source.
        Each chunk in the list should be a dict with 'content', 'embedding', 'metadata', and 'chunk_type'.
        """
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                data_to_insert = [
                    (source_id, doc['content'], doc.get('chunk_type', 'text'), doc['embedding'], Json(doc.get('metadata', {})))
                    for doc in chunks
                ]

                insert_query = f"""
                INSERT INTO {CONTENT_CHUNKS_TABLE} (source_id, content, chunk_type, embedding, metadata)
                VALUES %s;
                """
                execute_values(cur, insert_query, data_to_insert)
                conn.commit()
            print(f"Successfully added {len(chunks)} content chunks for source_id {source_id}.")
        except psycopg2.Error as e:
            print(f"Database error adding content chunks: {e}")
            if conn: conn.rollback()
            raise
        finally:
            if conn: conn.close()

    def similarity_search(self, query_embedding: List[float], k: int = 5, chunk_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search. Can optionally filter by chunk_type ('text' or 'code').
        """
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                sql_query = f"""
                SELECT
                    c.id,
                    c.content,
                    c.metadata,
                    c.chunk_type,
                    s.source_url,
                    s.source_name,
                    c.embedding <=> %s AS distance
                FROM {CONTENT_CHUNKS_TABLE} c
                JOIN {DATA_SOURCES_TABLE} s ON c.source_id = s.id
                """

                params = [query_embedding, query_embedding]

                if chunk_type:
                    sql_query += " WHERE c.chunk_type = %s"
                    params.append(chunk_type)

                sql_query += " ORDER BY distance LIMIT %s;"
                params.append(k)

                cur.execute(sql_query, tuple(params))
                rows = cur.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "chunk_id": row[0],
                        "content": row[1],
                        "metadata": row[2],
                        "chunk_type": row[3],
                        "source_url": row[4],
                        "source_name": row[5],
                        "similarity_score": 1 - row[6] # Convert cosine distance to similarity
                    })
                return results
        except psycopg2.Error as e:
            print(f"Database error during similarity search: {e}")
            raise
        finally:
            if conn: conn.close()

    def list_data_sources(self) -> List[Dict[str, Any]]:
        """Lists all ingested data sources."""
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute(f"SELECT id, source_url, source_name, created_at, last_crawled_at FROM {DATA_SOURCES_TABLE} ORDER BY created_at DESC;")
                rows = cur.fetchall()
                return [{"id": r[0], "url": r[1], "name": r[2], "created_at": r[3], "last_crawled": r[4]} for r in rows]
        finally:
            if conn: conn.close()

    def delete_data_source(self, source_id: int) -> int:
        """Deletes a data source and all its associated chunks."""
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {DATA_SOURCES_TABLE} WHERE id = %s;", (source_id,))
                deleted_rows = cur.rowcount
                conn.commit()
                return deleted_rows
        finally:
            if conn: conn.close()

if __name__ == '__main__':
    print("Testing CodeDocVectorStore...")
    try:
        # Assumes .env file is in project root, and this script is in app/core
        vector_store = CodeDocVectorStore(embedding_dimension=3) # Use low dim for test
        print("Initialization and schema check complete.")

        # Add a source
        source_url = "https://example-docs.com"
        source_id = vector_store.add_data_source(source_url, "Example Docs")
        print(f"Added/found data source '{source_url}' with ID: {source_id}")

        # Add chunks
        test_chunks = [
            {'content': 'This is a text chunk about installation.', 'chunk_type': 'text', 'embedding': [0.1, 0.2, 0.3], 'metadata': {'page': '/install'}},
            {'content': 'def setup():\n  print("Setup code")', 'chunk_type': 'code', 'embedding': [0.7, 0.8, 0.9], 'metadata': {'page': '/install', 'language': 'python'}}
        ]
        vector_store.add_content_chunks(source_id, test_chunks)

        # List sources
        sources = vector_store.list_data_sources()
        print("\nAvailable data sources:")
        for s in sources:
            print(f"- ID: {s['id']}, Name: {s['name']}, URL: {s['url']}")

        # Similarity search
        query_embedding = [0.1, 0.2, 0.4] # Similar to the 'text' chunk
        results = vector_store.similarity_search(query_embedding, k=1)
        print(f"\nSimilarity search results for query similar to 'text':")
        print(json.dumps(results, indent=2))

        # Filtered search
        results_code = vector_store.similarity_search(query_embedding, k=1, chunk_type='code')
        print(f"\nSimilarity search results for query similar to 'text' (filtered for 'code' chunks):")
        print(json.dumps(results_code, indent=2))

        # Delete source
        deleted_count = vector_store.delete_data_source(source_id)
        print(f"\nDeleted {deleted_count} data source(s).")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        print("Please ensure your PostgreSQL server is running and .env is configured correctly.")
```
