# Agent Zero Integration for Political RAG System

This directory contains the necessary components to allow an instance of the [Agent Zero](https://github.com/Cloud-Curio/agent-zero) framework to use our Political RAG system as a specialized tool.

## How it Works

We are treating our FastAPI-based RAG application as a dedicated "MCP Server" or a tool provider. An Agent Zero agent can be equipped with a custom tool that knows how to communicate with our RAG API.

## Files

1.  **`political_rag_tool.py`**: This is a Python script that defines a custom tool for Agent Zero. It makes a `POST` request to the `/query` endpoint of our RAG API.
2.  **`system_prompt_example.md`**: This is an example of how you might modify Agent Zero's `agent.system.md` prompt to make the agent aware of this new tool and how to use it effectively.

## Setup

1.  **Run the RAG API Server:** First, ensure the Political RAG API server is running as described in the main `README.md`. It should be accessible, for example, at `http://localhost:8000`.

2.  **Set up Agent Zero:** Follow the installation instructions for Agent Zero from its official repository.

3.  **Install the Custom Tool:**
    *   Copy the `political_rag_tool.py` file into the `python/tools/` directory of your Agent Zero instance.
    *   Agent Zero should automatically detect and load this new tool on startup.

4.  **Update the System Prompt:**
    *   Open the system prompt for your Agent Zero agent (e.g., `prompts/default/agent.system.md`).
    *   Add instructions telling the agent about its new capability. See `system_prompt_example.md` for inspiration. You should describe what the `political_rag_query` tool does and when to use it (e.g., "When you are asked a question about politics, fiscal policy, or specific political documents, use the `political_rag_query` tool to get an answer from a specialized system.").

## Usage

Once set up, you can interact with your Agent Zero agent (e.g., via its CLI or Web UI) and give it a prompt like:

> "Using your specialized political document analysis tool, tell me what the latest bill says about renewable energy credits."

Agent Zero's LLM, guided by its system prompt, should recognize that this task requires the `political_rag_query` tool. It will then call the tool with the query, which in turn calls our RAG API. The final answer from our RAG system will be returned to the Agent Zero agent, which will then present it to you.
```
