# app/agents/retrieval_agent.py
from typing import Dict, Any, List
from .base_agent import Agent
from app.core.retrieval import Retriever as CoreRetriever # The class we built earlier
from app.core.vector_store import PGVectorStore
from app.core.ingestion import get_embeddings_model, get_spacy_model # For initialization

class RetrievalAgent(Agent):
    """
    Handles the retrieval of relevant documents from the vector store based on a query.
    It uses the core Retriever class.
    """
    def __init__(
        self,
        name: str = "RetrievalAgent",
        core_retriever: CoreRetriever # Expects an initialized core Retriever
    ):
        super().__init__(name)
        self.core_retriever = core_retriever

    async def execute(self, task: Dict[str, Any], **kwargs) -> Any:
        """
        Executes a retrieval task.
        Task example: {
            "type": "retrieve_documents",
            "data": {
                "query": "What are the economic policies?",
                "k": 5,
                "use_query_enrichment": True
            }
        }
        """
        task_type = task.get("type")
        data = task.get("data")

        if task_type == "retrieve_documents":
            if not data or "query" not in data:
                return {"status": "error", "message": "Query missing for retrieve_documents task."}

            query = data["query"]
            k = data.get("k", 5) # Default to 5 results
            use_enrichment = data.get("use_query_enrichment", True) # Default to using enrichment

            try:
                print(f"{self.name}: Retrieving documents for query: '{query}' (k={k}, enrichment={use_enrichment})")
                retrieved_docs = self.core_retriever.retrieve(
                    query=query,
                    k=k,
                    use_query_enrichment=use_enrichment
                )
                # retrieved_docs is List[Dict[str, Any]] with 'content' and 'metadata' (incl. score)

                actual_query_used_for_embedding = processed_query # This is the one we need to report
                return {
                    "status": "success",
                    "original_query": query,
                    "query_used_for_embedding": actual_query_used_for_embedding,
                    "retrieved_documents": retrieved_docs,
                    "count": len(retrieved_docs)
                }
            except Exception as e:
                print(f"{self.name}: Error during document retrieval: {e}")
                return {"status": "error", "message": f"Error during retrieval: {str(e)}"}
        else:
            return {"status": "error", "message": f"Unknown task type for RetrievalAgent: {task_type}"}

# Example Usage (for testing this file directly)
if __name__ == '__main__':
    import asyncio
    import os
    from dotenv import load_dotenv

    # Adjust path to .env if running from app/agents
    load_dotenv(dotenv_path='../../.env')

    # This requires a running PGVector instance, appropriate .env settings,
    # and some data previously ingested into the 'agent_test_retrieval' collection.
    # For a full test, run IngestionAgent's test first or ensure data exists.

    async def test_retrieval_agent():
        print("Testing RetrievalAgent...")

        # Initialize components
        try:
            emb_model = get_embeddings_model()
            spacy_nlp = get_spacy_model()

            # Determine embedding dimension
            try:
                sample_emb = emb_model.embed_query("test")
                emb_dim = len(sample_emb)
            except Exception as e:
                print(f"Warning: Could not get embedding dim dynamically, defaulting to 384. Error: {e}")
                emb_dim = 384

            # Use a specific collection for this test or one populated by IngestionAgent test
            test_collection_name = "agent_test_ingestion" # Same as IngestionAgent test for continuity
            vec_store = PGVectorStore(
                collection_name=test_collection_name,
                embedding_dimension=emb_dim
            )

            # Ensure there's some data if running standalone
            # This is just a quick check; ideally, data is there from IngestionAgent test.
            if not vec_store.similarity_search_with_scores(emb_model.embed_query("test"), k=1):
                print(f"Warning: Collection '{test_collection_name}' might be empty. Test results may be trivial.")
                # You could add a few dummy items here if needed for isolated testing
                # from app.agents.ingestion_agent import IngestionAgent
                # print("Adding sample data for retrieval test...")
                # ingestion_agent_for_test = IngestionAgent(vector_store=vec_store, embedding_model=emb_model)
                # sample_file = "retrieval_sample.txt"
                # with open(sample_file, "w") as f: f.write("Fiscal policy impacts economy.")
                # await ingestion_agent_for_test.execute({"type": "ingest_document", "data": {"file_path": sample_file}})
                # if os.path.exists(sample_file): os.remove(sample_file)


            core_retriever_instance = CoreRetriever(
                vector_store=vec_store,
                embedding_model=emb_model,
                spacy_model=spacy_nlp
            )
            retrieval_agent = RetrievalAgent(core_retriever=core_retriever_instance)
            print("RetrievalAgent initialized.")

            test_query = "What are political discussions about?"
            task = {
                "type": "retrieve_documents",
                "data": {
                    "query": test_query,
                    "k": 2,
                    "use_query_enrichment": True
                }
            }
            result = await retrieval_agent.execute(task)

            print(f"\nRetrieval Result for query '{test_query}':")
            if result.get("status") == "success":
                print(f"  Enriched query used: {result.get('enriched_query_if_used')}")
                print(f"  Found {result.get('count')} documents:")
                for i, doc in enumerate(result.get("retrieved_documents", [])):
                    print(f"    Rank {i+1}: Content: '{doc['content'][:60]}...', Score: {doc['metadata'].get('retrieval_score', 'N/A'):.4f}")
            else:
                print(f"  Error: {result.get('message')}")

        except Exception as e:
            print(f"Error during RetrievalAgent test: {e}")
            import traceback
            traceback.print_exc()
            print("Ensure DB is running, .env is correct, spaCy models downloaded, and data ingested.")

    asyncio.run(test_retrieval_agent())
