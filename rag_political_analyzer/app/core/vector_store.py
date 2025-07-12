# app/core/vector_store.py
import os
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='../.env') # Assuming .env is in rag_political_analyzer directory

# Database connection parameters - should be loaded from environment variables
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")
DB_HOST = os.getenv("DB_HOST", "localhost") # Or your Docker service name e.g. 'db'
DB_PORT = os.getenv("DB_PORT", "5432")

# Table and collection names
DEFAULT_COLLECTION_NAME = "political_documents" # Langchain uses this term, for pgvector it's a table

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    register_vector(conn) # Important: Register PGVector types
    return conn

class PGVectorStore:
    def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME, embedding_dimension: int = 384):
        """
        Initializes the PGVectorStore.
        :param collection_name: Name of the table to store documents.
        :param embedding_dimension: Dimension of the embeddings (e.g., 384 for all-MiniLM-L6-v2).
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Ensures the vector table exists in the database."""
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                # Enable pgvector extension if not already enabled
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create table if it doesn't exist
                # Using TEXT for content, JSONB for metadata
                # Storing source and original document ID in metadata could be useful
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding VECTOR({self.embedding_dimension})
                );
                """
                cur.execute(create_table_query)

                # Optional: Create an index for faster similarity search
                # Using HNSW index for cosine distance, suitable for many use cases
                # The choice of index and its parameters (m, ef_construction) can be tuned
                create_index_query = f"""
                CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_embedding
                ON {self.collection_name}
                USING hnsw (embedding vector_cosine_ops);
                """
                # For IVFFlat, it would be:
                # CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
                # Choose index based on dataset size, query speed needs, and accuracy trade-offs.
                # HNSW is generally a good default.
                cur.execute(create_index_query)
                conn.commit()
            print(f"Table '{self.collection_name}' ensured with HNSW index.")
        except psycopg2.Error as e:
            print(f"Database error ensuring table: {e}")
            if conn:
                conn.rollback() # Rollback on error
            # Consider raising the exception or handling it more gracefully
        finally:
            if conn:
                conn.close()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Adds multiple documents (chunks with their embeddings and metadata) to the store.
        Each document in the list should be a dictionary with 'content', 'embedding', and 'metadata'.
        """
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                # Prepare data for batch insertion
                # data_to_insert should be a list of tuples: (content, metadata_json, embedding_vector)
                data_to_insert = [
                    (doc['content'], psycopg2.extras.Json(doc['metadata']), doc['embedding'])
                    for doc in documents
                ]

                insert_query = f"""
                INSERT INTO {self.collection_name} (content, metadata, embedding)
                VALUES %s;
                """
                execute_values(cur, insert_query, data_to_insert)
                conn.commit()
            print(f"Successfully added {len(documents)} documents to '{self.collection_name}'.")
        except psycopg2.Error as e:
            print(f"Database error adding documents: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def similarity_search_with_scores(self, query_embedding: List[float], k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a similarity search against the vector store.
        Returns k most similar documents and their similarity scores.
        Note: pgvector uses distance (smaller is better). We convert to similarity score (0-1, larger is better).
        For cosine distance: similarity = 1 - distance.
        For L2 distance: similarity = 1 / (1 + distance). This needs careful scaling.
        Assuming HNSW with vector_cosine_ops, so distance is cosine distance.
        """
        conn = None
        results = []
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                # query_embedding needs to be formatted as a string for the SQL query
                # but psycopg2 handles list-to-array conversion for vector type if registered.

                # <-> is L2 distance
                # <=> is cosine distance
                # <#> is inner product (negative for similarity with normalized vectors)
                # We used vector_cosine_ops for HNSW, so we should use <=> for cosine distance.
                # Cosine distance is 0 for identical, 1 for orthogonal, 2 for opposite.
                # Similarity = 1 - cosine_distance (ranges from -1 to 1, but for positive embeddings usually 0 to 1)

                # If embeddings are normalized, (1 - cosine_distance) / 2 maps to [0,1]
                # Or, more simply, if using inner product on normalized vectors, it's directly cosine similarity.
                # Let's stick to cosine distance and convert.

                search_query = f"""
                SELECT id, content, metadata, embedding <=> %s AS distance
                FROM {self.collection_name}
                ORDER BY embedding <=> %s
                LIMIT %s;
                """
                cur.execute(search_query, (query_embedding, query_embedding, k))
                fetched_rows = cur.fetchall()

                for row in fetched_rows:
                    doc_id, content, metadata, distance = row
                    similarity_score = 1 - distance # For cosine distance
                    document_data = {
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata
                    }
                    results.append((document_data, similarity_score))

            print(f"Found {len(results)} similar documents.")
            return results
        except psycopg2.Error as e:
            print(f"Database error during similarity search: {e}")
            return [] # Return empty list on error
        finally:
            if conn:
                conn.close()

# Example usage (for testing this file directly)
if __name__ == '__main__':
    # This assumes you have a PostgreSQL server running with pgvector enabled,
    # and the database/user are set up as per .env or defaults.
    # You might need to run `CREATE EXTENSION vector;` manually in your DB if the script fails.

    print("Initializing PGVectorStore...")
    # Make sure your .env file is in the rag_political_analyzer directory, or set env vars.
    # Example: Create a .env file in rag_political_analyzer with:
    # DB_NAME=my_rag_db
    # DB_USER=my_user
    # DB_PASSWORD=my_password
    # DB_HOST=localhost
    # DB_PORT=5432

    try:
        vector_store = PGVectorStore(collection_name="test_collection", embedding_dimension=3) # Small dimension for test
        print("PGVectorStore initialized.")

        # Dummy data for testing
        dummy_docs = [
            {"content": "This is test document 1 about apples.", "embedding": [0.1, 0.2, 0.3], "metadata": {"source": "test1.txt", "page": 1}},
            {"content": "Another test document, this one about bananas.", "embedding": [0.4, 0.5, 0.6], "metadata": {"source": "test2.txt", "page": 1}},
            {"content": "Apples and oranges are fruits.", "embedding": [0.15, 0.25, 0.35], "metadata": {"source": "test3.txt", "page": 1}},
        ]

        print("Adding dummy documents...")
        vector_store.add_documents(dummy_docs)

        print("Performing similarity search for 'apples' (embedding [0.1, 0.2, 0.3])...")
        query_vec = [0.1, 0.2, 0.3]
        similar_docs = vector_store.similarity_search_with_scores(query_vec, k=2)

        if similar_docs:
            print("\nFound similar documents:")
            for doc, score in similar_docs:
                print(f"  Content: {doc['content'][:50]}..., Metadata: {doc['metadata']}, Score: {score:.4f}")
        else:
            print("No similar documents found or error occurred.")

    except psycopg2.OperationalError as e:
        print(f"Could not connect to database. Ensure PostgreSQL is running and configured: {e}")
        print("You might need to set up your .env file in the 'rag_political_analyzer' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # To clean up (optional, run manually in psql or a cleanup script):
    # DROP TABLE IF EXISTS test_collection;
    # DROP EXTENSION IF EXISTS vector; (if no other tables use it)
```

And the `.env.example` file.
