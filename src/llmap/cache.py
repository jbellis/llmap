import os
import sqlite3
import json
from pathlib import Path

class Cache:
    def __init__(self):
        cache_dir = Path.home() / ".cache" / "llmap"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "cache.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get(self, cache_key: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response FROM responses WHERE cache_key = ?",
                (cache_key,)
            )
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
        return None

    def set(self, cache_key: str, response: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses (cache_key, response) VALUES (?, ?)",
                (cache_key, json.dumps(response))
            )
            conn.commit()
