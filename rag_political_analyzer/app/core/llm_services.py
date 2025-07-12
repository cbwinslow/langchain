# app/core/llm_services.py
import os
from typing import List, Dict, Any, Optional
from openrouter import Client as OpenRouterClient
import ollama  # Import the ollama library
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='../.env')

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Default Ollama host

# --- Model Constants ---
DEFAULT_OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"
DEFAULT_OLLAMA_MODEL = "llama3"  # A common default local model

class LLMService:
    def __init__(self, openrouter_api_key: Optional[str] = None, ollama_host: Optional[str] = None):
        # OpenRouter Client
        self.openrouter_api_key = openrouter_api_key or OPENROUTER_API_KEY
        self.openrouter_client = None
        if self.openrouter_api_key:
            self.openrouter_client = OpenRouterClient(api_key=self.openrouter_api_key)
            print("OpenRouter client initialized.")
        else:
            print("Warning: OpenRouter API key not found. OpenRouter models will not be available.")

        # Ollama Client
        self.ollama_host = ollama_host or OLLAMA_HOST
        try:
            self.ollama_client = ollama.Client(host=self.ollama_host)
            # A light check to see if the host is reachable by listing models
            self.ollama_client.list()
            print(f"Successfully connected to Ollama at {self.ollama_host}")
        except Exception as e:
            self.ollama_client = None
            print(f"Warning: Could not connect to Ollama at {self.ollama_host}. Ollama models will not be available. Error: {e}")

    def _generate_with_openrouter(self, messages: List[Dict[str, str]], model_name: str, max_tokens: int, temperature: float) -> str:
        if not self.openrouter_client:
            raise ConnectionError("OpenRouter client not initialized. Check your OPENROUTER_API_KEY.")
        response = self.openrouter_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response.choices:
            return response.choices[0].message.content.strip()
        raise ValueError("No response choices received from OpenRouter.")

    def _generate_with_ollama(self, messages: List[Dict[str, str]], model_name: str, max_tokens: int, temperature: float) -> str:
        if not self.ollama_client:
            raise ConnectionError(f"Ollama client not initialized. Could not connect to {self.ollama_host}.")

        response = self.ollama_client.chat(
            model=model_name,
            messages=messages,
            options={
                "num_predict": max_tokens,
                "temperature": temperature
            }
        )
        return response['message']['content'].strip()

    def generate_response(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_message: Optional[str] = "You are a helpful assistant."
    ) -> str:
        """
        Generates a response from an LLM, dispatching to the correct service based on model_name.
        - For Ollama, use a prefix, e.g., "ollama/llama3".
        - Otherwise, it's assumed to be an OpenRouter model.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            if model_name.startswith("ollama/"):
                ollama_model = model_name.split("/", 1)[1]
                return self._generate_with_ollama(messages, ollama_model, max_tokens, temperature)
            else:
                # Default to OpenRouter
                return self._generate_with_openrouter(messages, model_name, max_tokens, temperature)
        except Exception as e:
            print(f"Error during LLM call to {model_name}: {e}")
            return f"Error generating response: {str(e)}"

    def generate_answer_from_context(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        model_name: str, # Make model_name mandatory here
        max_tokens: int = 1500,
        temperature: float = 0.5
    ) -> str:
        """
        Generates an answer to a query based on retrieved context chunks.
        Dispatches to the correct service via generate_response.
        """
        if not retrieved_chunks:
            prompt_for_llm = f"Question: {query}\n\nAnswer based on your general knowledge (no specific context was found for this query):"
            system_msg = "You are a helpful assistant. Answer the question to the best of your ability."
        else:
            context_str = "\n\n---\n\n".join([chunk['content'] for chunk in retrieved_chunks])
            prompt_for_llm = f"""
            Use ONLY the information from the text provided below to answer the question.
            Do not use any external knowledge. If the answer is not found in the context, state that clearly.

            Context from documents:
            ---
            {context_str}
            ---

            Question: {query}

            Answer:
            """
            system_msg = "You are an AI assistant specialized in analyzing provided text to answer questions. Stick to the provided context."

        return self.generate_response(
            prompt=prompt_for_llm,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system_message=system_msg
        )

if __name__ == '__main__':
    # This test assumes Ollama is running locally with the 'llama3' model
    # and/or an OPENROUTER_API_KEY is set in the .env file.

    async def test_llm_service():
        print("Testing LLMService...")
        llm_service = LLMService()

        # Test OpenRouter if available
        if llm_service.openrouter_client:
            print(f"\n--- Testing OpenRouter ({DEFAULT_OPENROUTER_MODEL}) ---")
            try:
                response = llm_service.generate_response("What is the capital of France?", model_name=DEFAULT_OPENROUTER_MODEL)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("\n--- Skipping OpenRouter Test (not configured) ---")

        # Test Ollama if available
        if llm_service.ollama_client:
            print(f"\n--- Testing Ollama ({DEFAULT_OLLAMA_MODEL}) ---")
            try:
                response = llm_service.generate_response("Why is the sky blue?", model_name=f"ollama/{DEFAULT_OLLAMA_MODEL}")
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("\n--- Skipping Ollama Test (not connected) ---")

    import asyncio
    asyncio.run(test_llm_service())
