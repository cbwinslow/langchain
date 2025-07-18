# app/agents/orchestrator_agent.py
from typing import Dict, Any, Optional
from .base_agent import Agent
from .memory_agent import MemoryAgent
from .ingestion_agent import IngestionAgent
from .retrieval_agent import RetrievalAgent
from .answer_generation_agent import AnswerGenerationAgent

# For initializing the core components if not passed directly
from app.core.ingestion import get_embeddings_model, get_spacy_model
from app.core.vector_store import PGVectorStore
from app.core.retrieval import Retriever as CoreRetriever
from app.core.llm_services import LLMService
import os


class OrchestratorAgent(Agent):
    """
    Coordinates slave agents to process user requests for the RAG system.
    """
    def __init__(
        self,
        name: str = "OrchestratorAgent",
        memory_agent: MemoryAgent,
        ingestion_agent: IngestionAgent,
        retrieval_agent: RetrievalAgent,
        answer_generation_agent: AnswerGenerationAgent
    ):
        super().__init__(name)
        self.memory_agent = memory_agent
        self.ingestion_agent = ingestion_agent
        self.retrieval_agent = retrieval_agent
        self.answer_generation_agent = answer_generation_agent

    async def execute(self, task: Dict[str, Any], **kwargs) -> Any:
        """
        Executes a high-level task by orchestrating slave agents.
        Task examples:
        - {"type": "user_query", "data": {"query": "What are economic policies?", "user_id": "user123"}}
        - {"type": "ingest_file", "data": {"file_path": "/path/to/doc.pdf", "user_id": "admin"}}
        """
        task_type = task.get("type")
        data = task.get("data", {})
        user_id = data.get("user_id", "default_user") # For conversation history tracking

        print(f"{self.name}: Received task: {task_type} with data: {data}")

        if task_type == "user_query":
            query = data.get("query")
            if not query:
                return {"status": "error", "message": "Query missing for user_query task."}

            # 1. Store user query in conversation history (optional, could be part of MemoryAgent's role)
            await self.memory_agent.execute({
                "type": "store_conversation_turn",
                "data": {"role": "user", "content": query, "user_id": user_id}
            })

            # 2. Retrieve relevant documents
            # Default to using query enrichment, k=5
            k_retrieval = data.get("k_retrieval", 5)
            use_query_enrichment = data.get("use_query_enrichment", True)

            retrieval_task = {
                "type": "retrieve_documents",
                "data": {
                    "query": query,
                    "k": k_retrieval,
                    "use_query_enrichment": use_query_enrichment
                }
            }
            retrieval_result = await self.retrieval_agent.execute(retrieval_task)

            if retrieval_result.get("status") != "success":
                # Log error or handle, maybe try answering without context
                print(f"{self.name}: Retrieval failed. Proceeding without retrieved context.")
                retrieved_docs = []
            else:
                retrieved_docs = retrieval_result.get("retrieved_documents", [])

            # 3. Generate answer
            llm_model_name = data.get("llm_model_name") # Allow overriding default LLM model
            answer_gen_task = {
                "type": "generate_answer",
                "data": {
                    "query": query,
                    "retrieved_chunks": retrieved_docs,
                }
            }
            if llm_model_name:
                answer_gen_task["data"]["llm_model_name"] = llm_model_name

            answer_result = await self.answer_generation_agent.execute(answer_gen_task)

            if answer_result.get("status") != "success":
                final_answer = "I encountered an error trying to generate an answer."
                # Optionally store this error state in memory
            else:
                final_answer = answer_result.get("answer", "No answer could be generated.")

            # 4. Store assistant response in conversation history
            await self.memory_agent.execute({
                "type": "store_conversation_turn",
                "data": {"role": "assistant", "content": final_answer, "user_id": user_id}
            })

            # 5. (Optional) Store Q&A pair as a "fact" or "insight" in long-term memory
            # This would require more sophisticated logic in MemoryAgent or another agent
            # For now, we'll skip this explicit step for simplicity.

            return {
                "status": "success",
                "query": query,
                "answer": final_answer,
                "retrieved_context_count": len(retrieved_docs),
                "retrieval_details": retrieval_result # Pass along for transparency
            }

        elif task_type == "ingest_from_url":
            # The data here should be {"start_url": ..., "crawl_depth": ..., "max_pages": ...}
            if not data or "start_url" not in data:
                return {"status": "error", "message": "start_url missing for ingest_from_url task."}

            # The IngestionAgent now handles the full pipeline.
            # The task structure for the agent is the same as what we receive.
            ingestion_result = await self.ingestion_agent.execute(task)

            # Log the result of the ingestion task
            await self.memory_agent.execute({
                "type": "store_fact",
                "data": {
                    "key": f"ingestion_{data['start_url']}_{user_id}",
                    "value": ingestion_result,
                    "category": "system_logs"
                }
            })
            return ingestion_result # Return the direct result from the agent

        else:
            return {"status": "error", "message": f"Unknown task type for OrchestratorAgent: {task_type}"}


# Example Usage
if __name__ == '__main__':
    import asyncio
    from dotenv import load_dotenv

    # Load .env from project root (rag_political_analyzer)
    # This assumes this script is run from rag_political_analyzer/app/agents
    load_dotenv(dotenv_path='../../.env')

    async def test_orchestrator():
        print("Initializing components for Orchestrator test...")
        # This is a simplified setup. In a real app, these would be managed (e.g., by FastAPI app startup)
        try:
            # Core components
            emb_model = get_embeddings_model()
            spacy_nlp = get_spacy_model()

            # Determine embedding dimension
            try:
                sample_emb = emb_model.embed_query("test")
                emb_dim = len(sample_emb)
            except Exception as e:
                emb_dim = 384 # Default
                print(f"Could not get emb_dim, defaulting to {emb_dim}. Error: {e}")

            # Use a distinct collection for orchestrator tests to avoid conflicts
            db_collection_name = "orchestrator_test_collection"
            vector_db = PGVectorStore(collection_name=db_collection_name, embedding_dimension=emb_dim)

            core_retriever = CoreRetriever(vector_store=vector_db, embedding_model=emb_model, spacy_model=spacy_nlp)
            llm_service = LLMService() # API key from .env

            # Agents
            memory_agent = MemoryAgent(
                memory_file_path=os.path.join(DEFAULT_MEMORY_DIR, "orch_test_ltm.json"),
                conversation_memory_path=os.path.join(DEFAULT_MEMORY_DIR, "orch_test_conv.json")
            )
            ingestion_agent = IngestionAgent(vector_store=vector_db, embedding_model=emb_model)
            retrieval_agent = RetrievalAgent(core_retriever=core_retriever)
            answer_gen_agent = AnswerGenerationAgent(llm_service=llm_service)

            orchestrator = OrchestratorAgent(
                memory_agent=memory_agent,
                ingestion_agent=ingestion_agent,
                retrieval_agent=retrieval_agent,
                answer_generation_agent=answer_gen_agent
            )
            print("Orchestrator and all dependent agents initialized.")

            # Test Ingestion through Orchestrator
            print("\n--- Testing Ingestion via Orchestrator ---")
            dummy_ingest_file = "orchestrator_dummy_ingest.txt"
            with open(dummy_ingest_file, "w") as f:
                f.write("The latest bill discusses renewable energy credits and carbon taxes. It aims to reduce emissions by 20% in the next decade.")

            ingest_task = {"type": "ingest_file", "data": {"file_path": dummy_ingest_file, "user_id": "test_admin"}}
            ingest_result = await orchestrator.execute(ingest_task)
            print(f"Orchestrator Ingestion Result: {ingest_result}")
            if os.path.exists(dummy_ingest_file):
                os.remove(dummy_ingest_file)

            # Test Querying through Orchestrator
            print("\n--- Testing Querying via Orchestrator ---")
            query_task = {
                "type": "user_query",
                "data": {
                    "query": "What are the goals for emission reduction?",
                    "user_id": "test_user_001",
                    "k_retrieval": 1 # Ask for 1 relevant chunk
                }
            }
            query_result = await orchestrator.execute(query_task)
            print("\nOrchestrator Query Result:")
            print(f"  Query: {query_result.get('query')}")
            print(f"  Answer: {query_result.get('answer')}")
            print(f"  Retrieved Context Count: {query_result.get('retrieved_context_count')}")
            if query_result.get("retrieval_details", {}).get("retrieved_documents"):
                 print(f"  Top Retrieved Doc: {query_result['retrieval_details']['retrieved_documents'][0]['content'][:100]}...")

            # Test Querying without enrichment
            print("\n--- Testing Querying via Orchestrator (NO enrichment) ---")
            query_task_no_enrich = {
                "type": "user_query",
                "data": {
                    "query": "What are the goals for emission reduction?",
                    "user_id": "test_user_002",
                    "k_retrieval": 1,
                    "use_query_enrichment": False # Explicitly disable
                }
            }
            query_result_no_enrich = await orchestrator.execute(query_task_no_enrich)
            print("\nOrchestrator Query Result (No Enrichment):")
            print(f"  Query: {query_result_no_enrich.get('query')}")
            print(f"  Answer: {query_result_no_enrich.get('answer')}")
            # Manual check for now, or simple assert
            enriched_query_info = query_result_no_enrich.get("retrieval_details", {}).get("enriched_query_if_used", "")
            print(f"  Enriched query info for no-enrichment test: {enriched_query_info}")
            assert "N/A" in enriched_query_info or query_result_no_enrich.get('query') == enriched_query_info


            # Clean up memory files for test
            if os.path.exists(memory_agent.memory_file_path):
                os.remove(memory_agent.memory_file_path)
            if os.path.exists(memory_agent.conversation_memory_path):
                os.remove(memory_agent.conversation_memory_path)
            # Optional: Clean up DB table (manual or script)
            # conn = vector_db.get_db_connection()
            # if conn:
            #     with conn.cursor() as cur: cur.execute(f"DROP TABLE IF EXISTS {db_collection_name};")
            #     conn.commit(); conn.close()
            #     print(f"Dropped DB table {db_collection_name}")


        except ValueError as ve: # Catches API key issues from LLMService
            print(f"Setup Error (likely API key): {ve}")
        except Exception as e:
            print(f"An error occurred during OrchestratorAgent test: {e}")
            import traceback
            traceback.print_exc()
            print("Ensure DB is running, .env correct, spaCy models downloaded, OPENROUTER_API_KEY set.")

    asyncio.run(test_orchestrator())
