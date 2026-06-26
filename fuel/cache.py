# fuel/cache.py
"""
llmfuel — cache module
Step-level cache for deduplicated CoT outputs.

Stub interface with SQLite persistence for v0.1.
"""

from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


class StepCache:
    """
    Local cache keyed on step content hash.

    Uses SQLite for persistence and keeps a simple capped-size store.
    """

    def __init__(self, max_size: int = 1000, db_path: Optional[Path] = None):
        self.max_size = max_size
        self.db_path = Path(db_path) if db_path is not None else Path(".llmfuel_stepcache.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                inserted_at INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[Any]:
        cursor = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        timestamp = int(time.time())
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO cache (key, value, inserted_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    inserted_at = excluded.inserted_at
                """,
                (key, payload, timestamp),
            )
        if len(self) > self.max_size:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        excess = len(self) - self.max_size
        if excess <= 0:
            return
        with self._conn:
            self._conn.execute(
                """
                DELETE FROM cache
                WHERE key IN (
                    SELECT key FROM cache ORDER BY inserted_at ASC LIMIT ?
                )
                """,
                (excess,),
            )

    def __len__(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM cache")
        return cursor.fetchone()[0]
