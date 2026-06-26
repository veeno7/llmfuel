# fuel/cache.py
"""
llmfuel — cache module
Step-level cache for deduplicated CoT outputs.

Stub interface — Meta AI will implement.
"""

from __future__ import annotations
from typing import Any, Optional


class StepCache:
    """
    Local in-process cache keyed on step content hash.

    TODO Meta AI:
        - Persistence layer (SQLite or flat JSONL)
        - TTL / max-size eviction
        - Optional cross-run cache sharing (opt-in, still local)
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self.max_size:
            # Simple FIFO eviction placeholder — Meta AI: replace with LRU
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)
