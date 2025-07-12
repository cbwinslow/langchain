# app/agents/memory_agent.py
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from .base_agent import Agent

# Define the path for the memory file within the app/memory_system directory
DEFAULT_MEMORY_DIR = "memory_system"
DEFAULT_MEMORY_FILE = os.path.join(DEFAULT_MEMORY_DIR, "long_term_memory.json")
DEFAULT_CONVERSATION_MEMORY_FILE = os.path.join(DEFAULT_MEMORY_DIR, "conversation_history.json")

class MemoryAgent(Agent):
    """
    Manages the system's memory, including conversation history and learned facts/insights.
    """
    def __init__(
        self,
        name: str = "MemoryAgent",
        memory_file_path: str = DEFAULT_MEMORY_FILE,
        conversation_memory_path: str = DEFAULT_CONVERSATION_MEMORY_FILE
    ):
        super().__init__(name)
        self.memory_file_path = memory_file_path
        self.conversation_memory_path = conversation_memory_path

        # Ensure memory directory exists
        os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.conversation_memory_path), exist_ok=True)

        self.long_term_memory = self._load_memory(self.memory_file_path)
        self.conversation_history = self._load_memory(self.conversation_memory_path, default_type=list)

    def _load_memory(self, file_path: str, default_type: type = dict) -> Any:
        """Loads memory from a JSON file."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load memory from {file_path}. Initializing new memory. Error: {e}")
                return default_type() if default_type == dict else default_type([])
        return default_type() if default_type == dict else default_type([])

    def _save_memory(self, data: Any, file_path: str):
        """Saves memory to a JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving memory to {file_path}: {e}")

    async def execute(self, task: Dict[str, Any], **kwargs) -> Any:
        """
        Executes a memory-related task.
        Task types: "store_fact", "retrieve_fact", "store_conversation_turn", "get_conversation_history"
        """
        task_type = task.get("type")
        data = task.get("data")

        if task_type == "store_fact":
            # data = {"key": "some_fact_id", "value": "details_of_the_fact", "category": "general"}
            if data and "key" in data and "value" in data:
                category = data.get("category", "general")
                if category not in self.long_term_memory:
                    self.long_term_memory[category] = {}
                self.long_term_memory[category][data["key"]] = {
                    "value": data["value"],
                    "timestamp": datetime.utcnow().isoformat()
                }
                self._save_memory(self.long_term_memory, self.memory_file_path)
                return {"status": "success", "message": f"Fact '{data['key']}' stored in category '{category}'."}
            return {"status": "error", "message": "Invalid data for store_fact."}

        elif task_type == "retrieve_fact":
            # data = {"key": "some_fact_id", "category": "general"}
            if data and "key" in data:
                category = data.get("category", "general")
                fact = self.long_term_memory.get(category, {}).get(data["key"])
                if fact:
                    return {"status": "success", "fact": fact}
                return {"status": "not_found", "message": f"Fact '{data['key']}' not found in category '{category}'."}
            return {"status": "error", "message": "Invalid data for retrieve_fact."}

        elif task_type == "store_conversation_turn":
            # data = {"role": "user/assistant", "content": "message_text", "timestamp": (optional)}
            if data and "role" in data and "content" in data:
                turn = {
                    "role": data["role"],
                    "content": data["content"],
                    "timestamp": data.get("timestamp", datetime.utcnow().isoformat())
                }
                self.conversation_history.append(turn)
                self._save_memory(self.conversation_history, self.conversation_memory_path)
                return {"status": "success", "message": "Conversation turn stored."}
            return {"status": "error", "message": "Invalid data for store_conversation_turn."}

        elif task_type == "get_conversation_history":
            # data = {"last_n_turns": 5} (optional)
            last_n = data.get("last_n_turns") if data else None
            history_to_return = self.conversation_history
            if last_n is not None and isinstance(last_n, int) and last_n > 0:
                history_to_return = self.conversation_history[-last_n:]
            return {"status": "success", "history": history_to_return}

        elif task_type == "clear_conversation_history":
            self.conversation_history = []
            self._save_memory(self.conversation_history, self.conversation_memory_path)
            return {"status": "success", "message": "Conversation history cleared."}

        else:
            return {"status": "error", "message": f"Unknown task type for MemoryAgent: {task_type}"}

    def get_recent_conversation(self, k: int = 5) -> List[Dict[str, str]]:
        """Helper to get last k turns for internal use by other agents."""
        if k <= 0: return []
        return self.conversation_history[-k:]

# Example Usage (for testing this file directly)
if __name__ == '__main__':
    import asyncio

    # Create dummy memory directory if running standalone for the first time for test
    if not os.path.exists(DEFAULT_MEMORY_DIR):
        os.makedirs(DEFAULT_MEMORY_DIR)

    memory_agent = MemoryAgent(
        memory_file_path=os.path.join(DEFAULT_MEMORY_DIR, "test_ltm.json"),
        conversation_memory_path=os.path.join(DEFAULT_MEMORY_DIR, "test_conv_hist.json")
    )

    async def test_memory_agent():
        print("Testing MemoryAgent...")

        # Store conversation
        await memory_agent.execute({"type": "store_conversation_turn", "data": {"role": "user", "content": "Hello, RAG system!"}})
        await memory_agent.execute({"type": "store_conversation_turn", "data": {"role": "assistant", "content": "Hello! How can I help you today?"}})

        # Retrieve conversation history
        history_result = await memory_agent.execute({"type": "get_conversation_history", "data": {"last_n_turns": 1}})
        print("\nLast 1 turn of conversation:")
        if history_result['status'] == 'success':
            for turn in history_result['history']:
                print(f"  {turn['role']}: {turn['content']} ({turn['timestamp']})")

        # Store a fact
        fact_data = {"key": "capital_france", "value": "Paris", "category": "geography"}
        store_fact_result = await memory_agent.execute({"type": "store_fact", "data": fact_data})
        print(f"\nStore fact result: {store_fact_result}")

        # Retrieve a fact
        retrieve_fact_result = await memory_agent.execute({"type": "retrieve_fact", "data": {"key": "capital_france", "category": "geography"}})
        print(f"\nRetrieve fact result: {retrieve_fact_result}")

        retrieve_missing_fact = await memory_agent.execute({"type": "retrieve_fact", "data": {"key": "capital_germany", "category": "geography"}})
        print(f"\nRetrieve missing fact result: {retrieve_missing_fact}")

        # Clean up test files
        if os.path.exists(memory_agent.memory_file_path):
            os.remove(memory_agent.memory_file_path)
        if os.path.exists(memory_agent.conversation_memory_path):
            os.remove(memory_agent.conversation_memory_path)
        print("\nCleaned up test memory files.")

    asyncio.run(test_memory_agent())
