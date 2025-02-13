from contextlib import contextmanager, closing
import sqlite3
import json
from pathlib import Path
from dbutils.pooled_db import PooledDB

class Cache:
    """
    A cache implementation that uses a pooled SQLite connection.
    """
    def __init__(self):
        cache_dir = Path.home() / ".cache" / "llmap"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(cache_dir / "cache.db")
        # Create a pool with a maximum of 10 connections and disable thread check.
        self.pool = PooledDB(
            sqlite3,
            database=self.db_path,
            check_same_thread=False,
            maxconnections=10
        )
        self._init_db()

    def _init_db(self):
        """
        Initialize the cache database with the required table.
        """
        with self.get_conn() as conn:
            with closing(conn.cursor()) as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS responses (
                        cache_key TEXT PRIMARY KEY,
                        response TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()

    @contextmanager
    def get_conn(self):
        """
        Context manager to get a connection from the pool.
        """
        conn = self.pool.connection()
        try:
            yield conn
        finally:
            conn.close()  # returns connection to the pool

    def get(self, cache_key: str) -> dict | None:
        """
        Retrieve a cached response by key.
        """
        with self.get_conn() as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(
                    "SELECT response FROM responses WHERE cache_key = ?",
                    (cache_key,)
                )
                result = cur.fetchone()
            if result:
                return json.loads(result[0])
        return None

    def set(self, cache_key: str, response: dict):
        """
        Cache a response with the given key.
        """
        with self.get_conn() as conn:
            with closing(conn.cursor()) as cur:
                cur.execute(
                    "INSERT OR REPLACE INTO responses (cache_key, response) VALUES (?, ?)",
                    (cache_key, json.dumps(response))
                )
                conn.commit()

    def delete(self, cache_key: str):
        """
        Remove a cached response by key.
        """
        with self.get_conn() as conn:
            with closing(conn.cursor()) as cur:
                cur.execute("DELETE FROM responses WHERE cache_key = ?", (cache_key,))
                conn.commit()
