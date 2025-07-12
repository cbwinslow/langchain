# app/agents/ingestion_agent.py
import os
from typing import Dict, Any
from .base_agent import Agent
# Assuming core components are accessible. Adjust paths if necessary.
from app.core.ingestion import load_document, chunk_documents, get_embeddings_model
from app.core.vector_store import PGVectorStore # Or a generic VectorStore interface

class IngestionAgent(Agent):
    """
    Handles the ingestion of documents into the RAG system.
    """
    def __init__(
        self,
        name: str = "IngestionAgent",
        vector_store: PGVectorStore, # Expects an initialized vector store
        embedding_model: Any # Expects an initialized embedding model
    ):
        super().__init__(name)
        self.vector_store = vector_store
        self.embedding_model = embedding_model # From ingestion.get_embeddings_model()

    async def execute(self, task: Dict[str, Any], **kwargs) -> Any:
        """
        Executes an ingestion task.
        Task example: {"type": "ingest_document", "data": {"file_path": "/path/to/doc.pdf"}}
        """
        task_type = task.get("type")
        data = task.get("data")

        if task_type == "ingest_document":
            if not data or "file_path" not in data:
                return {"status": "error", "message": "File path missing for ingest_document task."}

            file_path = data["file_path"]
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"File not found: {file_path}"}

            try:
                print(f"{self.name}: Processing document: {file_path}")
                docs_data = load_document(file_path) # Returns List[LangchainDocument]
                if not docs_data:
                    return {"status": "error", "message": f"Could not load document or unsupported file type: {file_path}"}

                # chunk_documents from ingestion.py now includes spaCy feature extraction in metadata
                chunks = chunk_documents(docs_data)
                if not chunks:
                     return {"status": "error", "message": f"Failed to chunk document: {file_path}"}

                docs_for_db = []
                # Langchain Document objects in `chunks` have `page_content` and `metadata`
                chunk_contents = [chunk.page_content for chunk in chunks]

                if chunk_contents:
                    # Generate embeddings
                    chunk_embeddings = self.embedding_model.embed_documents(chunk_contents)

                    for i, chunk_doc in enumerate(chunks):
                        docs_for_db.append({
                            "content": chunk_doc.page_content,
                            "embedding": chunk_embeddings[i],
                            "metadata": chunk_doc.metadata
                        })

                    self.vector_store.add_documents(docs_for_db)
                    msg = f"Document '{os.path.basename(file_path)}' processed and {len(docs_for_db)} chunks stored successfully."
                    print(f"{self.name}: {msg}")
                    return {"status": "success", "message": msg, "chunks_stored": len(docs_for_db)}
                else:
                    msg = f"Document '{os.path.basename(file_path)}' resulted in no content chunks."
                    print(f"{self.name}: {msg}")
                    return {"status": "success", "message": msg, "chunks_stored": 0}

            except Exception as e:
                print(f"{self.name}: Error processing document {file_path}: {e}")
                return {"status": "error", "message": f"Error processing document: {str(e)}"}
        else:
            return {"status": "error", "message": f"Unknown task type for IngestionAgent: {task_type}"}

# Example Usage (for testing this file directly)
if __name__ == '__main__':
    import asyncio
    from dotenv import load_dotenv

    load_dotenv(dotenv_path='../../.env') # Adjust path to .env if running from app/agents

    # This requires a running PGVector instance and appropriate .env settings
    # Also assumes spacy model is downloaded.

    async def test_ingestion_agent():
        print("Testing IngestionAgent...")

        # Initialize components
        try:
            emb_model = get_embeddings_model()
            # Determine embedding dimension dynamically
            try:
                sample_emb = emb_model.embed_query("test")
                emb_dim = len(sample_emb)
            except Exception as e:
                print(f"Warning: Could not get embedding dim dynamically, defaulting to 384. Error: {e}")
                emb_dim = 384

            vec_store = PGVectorStore(
                collection_name="agent_test_ingestion",
                embedding_dimension=emb_dim
            )
            ingestion_agent = IngestionAgent(vector_store=vec_store, embedding_model=emb_model)
            print("IngestionAgent initialized.")

            # Create a dummy file for testing
            dummy_file_path = "dummy_test_doc_for_agent.txt"
            with open(dummy_file_path, "w", encoding="utf-8") as f:
                f.write("This is a test document for the Ingestion Agent.\n")
                f.write("It contains political discussions and policy matters.\n")
                f.write("The economy is a key focus in the upcoming election.")

            task = {"type": "ingest_document", "data": {"file_path": dummy_file_path}}
            result = await ingestion_agent.execute(task)
            print(f"\nIngestion Result: {result}")

            # Verify by trying to retrieve (basic check)
            if result.get("status") == "success" and result.get("chunks_stored", 0) > 0 :
                query_embedding = emb_model.embed_query("political discussions")
                search_results = vec_store.similarity_search_with_scores(query_embedding, k=1)
                print("\nVerification search results:")
                for doc, score in search_results:
                    print(f"  Content: {doc['content'][:50]}..., Score: {score:.4f}, Metadata: {doc['metadata']}")

            # Clean up
            if os.path.exists(dummy_file_path):
                os.remove(dummy_file_path)
            # Optional: Clean up the test table from DB
            # conn = vec_store.get_db_connection()
            # if conn:
            #     with conn.cursor() as cur:
            #         cur.execute(f"DROP TABLE IF EXISTS {vec_store.collection_name};")
            #     conn.commit()
            #     conn.close()
            #     print(f"Dropped table {vec_store.collection_name}")


        except Exception as e:
            print(f"Error during IngestionAgent test: {e}")
            print("Ensure DB is running, .env is correct, and spaCy models downloaded.")

    asyncio.run(test_ingestion_agent())
