"""
LLM Backend module for connecting to local engines (Ollama, LM Studio, vLLM).
"""

from .client import LocalLLMClient, clean_model_display_name
from .adapters import ReasoningTraceAdapter

__all__ = ["LocalLLMClient", "clean_model_display_name", "ReasoningTraceAdapter"]
