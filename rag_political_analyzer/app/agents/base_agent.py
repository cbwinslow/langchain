# app/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class Agent(ABC):
    """
    Abstract base class for all agents.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, task: Dict[str, Any], **kwargs) -> Any:
        """
        Executes a given task.

        :param task: A dictionary describing the task to be performed.
                     Example: {"type": "query", "data": "What is fiscal policy?"}
        :param kwargs: Additional arguments that might be needed by the agent.
        :return: The result of the task execution.
        """
        pass

    def __str__(self):
        return f"Agent({self.name})"
