"""
llmfuel adapters — thin wrappers to drop llmfuel into existing model clients.
"""

from .ollama import wrap_ollama
from .ollama_stream import wrap_ollama_stream

__all__ = ["wrap_ollama", "wrap_ollama_stream"]
