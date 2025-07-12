# app/agents/answer_generation_agent.py
from typing import Dict, Any, List
from .base_agent import Agent
from app.core.llm_services import LLMService # The class we built earlier

class AnswerGenerationAgent(Agent):
    """
    Generates answers based on a query and retrieved context using an LLM.
    It uses the core LLMService.
    """
    def __init__(
        self,
        name: str = "AnswerGenerationAgent",
        llm_service: LLMService # Expects an initialized LLMService
    ):
        super().__init__(name)
        self.llm_service = llm_service

    async def execute(self, task: Dict[str, Any], **kwargs) -> Any:
        """
        Executes an answer generation task.
        Task example: {
            "type": "generate_answer",
            "data": {
                "query": "What are the economic policies?",
                "retrieved_chunks": [
                    {"content": "Chunk 1 content...", "metadata": {...}},
                    {"content": "Chunk 2 content...", "metadata": {...}}
                ],
                "llm_model_name": "mistralai/mistral-7b-instruct" // Optional
            }
        }
        """
        task_type = task.get("type")
        data = task.get("data")

        if task_type == "generate_answer":
            if not data or "query" not in data or "retrieved_chunks" not in data:
                return {"status": "error", "message": "Query or retrieved_chunks missing for generate_answer task."}

            query = data["query"]
            retrieved_chunks = data["retrieved_chunks"] # List[Dict[str, Any]]
            model_name = data.get("llm_model_name") # Uses default from LLMService if None

            try:
                print(f"{self.name}: Generating answer for query: '{query}' using {len(retrieved_chunks)} context chunks.")

                if model_name:
                    answer = self.llm_service.generate_answer_from_context(
                        query=query,
                        retrieved_chunks=retrieved_chunks,
                        model_name=model_name
                    )
                else:
                    answer = self.llm_service.generate_answer_from_context(
                        query=query,
                        retrieved_chunks=retrieved_chunks
                    )

                return {
                    "status": "success",
                    "query": query,
                    "answer": answer,
                    "context_chunk_count": len(retrieved_chunks)
                }
            except Exception as e:
                print(f"{self.name}: Error during answer generation: {e}")
                return {"status": "error", "message": f"Error during answer generation: {str(e)}"}
        else:
            return {"status": "error", "message": f"Unknown task type for AnswerGenerationAgent: {task_type}"}

# Example Usage (for testing this file directly)
if __name__ == '__main__':
    import asyncio
    from dotenv import load_dotenv

    # Adjust path to .env if running from app/agents
    load_dotenv(dotenv_path='../../.env')
    # This requires OPENROUTER_API_KEY in .env

    async def test_answer_generation_agent():
        print("Testing AnswerGenerationAgent...")

        try:
            # Initialize LLMService
            # OPENROUTER_API_KEY should be loaded from .env by LLMService itself
            llm_service_instance = LLMService()

            answer_agent = AnswerGenerationAgent(llm_service=llm_service_instance)
            print("AnswerGenerationAgent initialized.")

            sample_query = "What did the trade agreement state about tariffs on electronics?"
            sample_context = [
                {"content": "The new trade agreement focuses on agricultural subsidies.", "metadata": {"source": "docA"}},
                {"content": "Regarding electronics, the agreement specifies a 5% reduction in tariffs over two years.", "metadata": {"source": "docB"}},
                {"content": "The previous agreement had higher tariffs on electronics.", "metadata": {"source": "docC"}}
            ]

            task = {
                "type": "generate_answer",
                "data": {
                    "query": sample_query,
                    "retrieved_chunks": sample_context
                    # "llm_model_name": "openai/gpt-3.5-turbo" # Optionally specify a model
                }
            }

            result = await answer_agent.execute(task)
            print(f"\nAnswer Generation Result for query '{sample_query}':")
            if result.get("status") == "success":
                print(f"  Query: {result.get('query')}")
                print(f"  Answer: {result.get('answer')}")
                print(f"  Context chunks used: {result.get('context_chunk_count')}")
            else:
                print(f"  Error: {result.get('message')}")

            # Test with no context
            sample_query_no_context = "What is the capital of Mars?"
            task_no_context = {
                "type": "generate_answer",
                "data": {
                    "query": sample_query_no_context,
                    "retrieved_chunks": []
                }
            }
            result_no_context = await answer_agent.execute(task_no_context)
            print(f"\nAnswer Generation Result for query '{sample_query_no_context}' (no context):")
            if result_no_context.get("status") == "success":
                print(f"  Answer: {result_no_context.get('answer')}")
            else:
                print(f"  Error: {result_no_context.get('message')}")


        except ValueError as ve: # Handles API key not found in LLMService init
            print(f"ValueError during setup: {ve}")
        except Exception as e:
            print(f"Error during AnswerGenerationAgent test: {e}")
            import traceback
            traceback.print_exc()
            print("Ensure OPENROUTER_API_KEY is set in .env.")

    asyncio.run(test_answer_generation_agent())
