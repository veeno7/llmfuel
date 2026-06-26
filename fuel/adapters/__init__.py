"""
llmfuel adapters — thin wrappers to drop llmfuel into existing model clients.
"""

from .ollama import wrap_ollama

__all__ = ["wrap_ollama"]
