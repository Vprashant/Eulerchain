from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, Type

class LLMBaseAgent(ABC):
    @abstractmethod
    def run(self, query: str) -> Any:
        pass

class LLMConfig(BaseModel):
    agent_type: str
    config: Dict[str, Any] = Field(default_factory=dict)
