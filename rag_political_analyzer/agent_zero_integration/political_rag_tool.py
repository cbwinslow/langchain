# This file should be placed in the `python/tools/` directory of an Agent Zero instance.
# It allows Agent Zero to use our RAG system as a specialized tool.

import requests
import json

# The agent's prompt needs to be updated to know about this tool and its arguments.
# Example prompt addition:
#
# **political_rag_query(query: str, k: int = 3, use_enrichment: bool = True, llm_provider: str = 'default')**
# Use this tool to answer questions about politics, fiscal policy, or specific political documents.
# - `query`: The question to ask the RAG system.
# - `k`: The number of context documents to retrieve. Default is 3.
# - `use_enrichment`: Whether to use spaCy to enrich the query. Default is True.
# - `llm_provider`: Optional. Specify 'ollama/model_name' to use a local Ollama model (e.g., 'ollama/llama3')
#   or an OpenRouter model name (e.g., 'mistralai/mistral-7b-instruct'). Defaults to the RAG system's default.
#
# Example of the agent thinking:
# I need to answer a question about fiscal policy. I will use the political_rag_query tool.
# TOOL: political_rag_query(query="What are the latest fiscal policies mentioned in the documents?")

# --- Tool Configuration ---
# The URL of your running RAG API server.
# This should be changed if your RAG API is not running on localhost:8000.
RAG_API_URL = "http://localhost:8000/query/"

def political_rag_query(query: str, k: int = 3, use_enrichment: bool = True, llm_provider: str = 'default'):
    """
    Queries the specialized political RAG system to get answers based on ingested documents.
    """
    if not query:
        return "Error: The 'query' argument cannot be empty."

    print(f"INFO: political_rag_query tool called with: query='{query}', k={k}, use_enrichment={use_enrichment}, llm_provider='{llm_provider}'")

    # Prepare the form data for the POST request
    form_data = {
        "query": query,
        "k_retrieval": str(k), # Form data values are typically strings
        "use_query_enrichment": str(use_enrichment),
        "user_id": "agent_zero_tool_user" # A hardcoded user for logging purposes
    }

    # Only add the model name if it's not the default, letting the RAG API choose its own default otherwise
    if llm_provider != 'default':
        form_data["llm_model_name"] = llm_provider

    try:
        response = requests.post(RAG_API_URL, data=form_data)

        # Check for successful response
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()

        # Format the response for the agent. We want to give it the final answer,
        # but also some context on what was retrieved, so it can reason about it.
        final_answer = response_data.get("answer", "No answer was generated.")
        retrieved_count = response_data.get("retrieved_context_count", 0)

        # Prepare a clean, readable summary for the agent
        summary = f"The specialized RAG system provided the following answer:\n\n"
        summary += f"Answer: {final_answer}\n\n"
        summary += f"This answer was generated based on {retrieved_count} relevant document chunks."

        # Optionally, include snippets of the retrieved context if the agent needs to see the source material
        # retrieval_details = response_data.get("retrieval_details", {})
        # retrieved_docs = retrieval_details.get("retrieved_documents", [])
        # if retrieved_docs:
        #     summary += "\n\nTop Retrieved Context Snippets:\n"
        #     for i, doc in enumerate(retrieved_docs[:2]): # Show top 2 snippets
        #         summary += f"- Snippet {i+1}: \"{doc.get('content', 'N/A')[:150]}...\"\n"

        return summary

    except requests.exceptions.RequestException as e:
        error_message = f"Error connecting to the RAG API at {RAG_API_URL}. Please ensure the RAG server is running. Details: {e}"
        print(f"ERROR: {error_message}")
        return error_message
    except json.JSONDecodeError:
        error_message = f"Error: Could not decode the JSON response from the RAG API. Response text: {response.text}"
        print(f"ERROR: {error_message}")
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred in the political_rag_query tool: {e}"
        print(f"ERROR: {error_message}")
        return error_message

# This is how Agent Zero discovers the tool.
# The function name is the tool name.
# The docstring is used by the agent to understand how to use the tool.
# The type hints are used to parse arguments.
tool = political_rag_query
```
