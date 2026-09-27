"""
Every model wrapper (claude_wrapper.py, llama_wrapper.py, ...) must implement
this same interface. Nothing else in the pipeline is allowed to know which
model it's talking to — it only ever calls .query(prompt).

This is what makes adding GPT/Gemini later a drop-in, not a rewrite.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelResponse:
    text: str
    model_id: str
    prompt: str
    raw: dict | None = None   # full provider response, kept for debugging/audit trail


class BaseModel(ABC):
    def __init__(self, model_id: str, temperature: float = 0.0, timeout: int = 30,
                 max_tokens: int = 300):
        self.model_id = model_id
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens   # hard ceiling — backs up the concise prompt instruction

    @abstractmethod
    def query(self, prompt: str) -> ModelResponse:
        """Send one prompt, return one ModelResponse. Must raise on failure
        (retry logic lives in utils/retry.py, not here)."""
        raise NotImplementedError