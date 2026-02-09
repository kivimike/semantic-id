from abc import ABC, abstractmethod
import sqlite3
import threading
from typing import Dict

class CollisionStore(ABC):
    """
    Abstract base class for collision storage.
    Manages the counter for each semantic ID.
    """
    
    @abstractmethod
    def next_suffix(self, key: str) -> int:
        """
        Increment the counter for the given key and return the NEW count.
        If key is new, return 0 (or 1? Logic says first collision gets suffix -1).
        
        Let's define:
        First occurrence: count = 0 (no suffix)
        Second occurrence: count = 1 (suffix -1)
        Third occurrence: count = 2 (suffix -2)
        
        So this method returns the current count value before incrementing, OR
        increments and returns.
        
        Let's stick to the prompt's example:
        1st time -> "12-5" (no suffix)
        2nd time -> "12-5-1"
        
        So `next_suffix` should return the index to be used as suffix.
        If it's the first time seeing the key, we should record it exists.
        
        Let's refine semantics: `get_and_increment(key)`
        Returns 0 for the first time.
        Returns 1 for the second time.
        
        If return value > 0, we append "-{value}".
        """
        pass

class InMemoryCollisionStore(CollisionStore):
    def __init__(self):
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {} # Key -> number of times seen so far

    def next_suffix(self, key: str) -> int:
        with self._lock:
            current = self._counts.get(key, 0)
            self._counts[key] = current + 1
            return current

class SQLiteCollisionStore(CollisionStore):
    def __init__(self, db_path: str = "collisions.db"):
        self.db_path = db_path
        self._init_db()
        self._lock = threading.Lock() # SQLite is thread-safe mostly, but let's be safe for connection sharing or just create connection per thread.
        # Actually standard sqlite3 connections are not thread-safe.
        # Ideally we open a new connection or use a pool.
        # For MVP simplicity, we'll open a connection for each operation or use a global lock around a single connection (low throughput).
        # Let's use a lock and a single connection with check_same_thread=False (careful usage) OR just open/close per call.
        # Open/close is safer for MVP.

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collisions (
                    key TEXT PRIMARY KEY,
                    count INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def next_suffix(self, key: str) -> int:
        # returns the index (0 for first, 1 for second...)
        # We need atomic increment.
        with sqlite3.connect(self.db_path) as conn:
            # We want to return the OLD value (0 if not exists) and set NEW value = OLD + 1.
            # OR we can just use RETURNING clause if sqlite version supports it (3.35+).
            # MacOS sqlite might be old. Let's try standard UPSERT.
            
            # Logic:
            # Insert 1 if not exists.
            # Update count = count + 1 if exists.
            # We want to know what was the state.
            
            # Using a transaction:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                cursor.execute("SELECT count FROM collisions WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row is None:
                    current = 0
                    cursor.execute("INSERT INTO collisions (key, count) VALUES (?, ?)", (key, 1))
                else:
                    current = row[0]
                    cursor.execute("UPDATE collisions SET count = ? WHERE key = ?", (current + 1, key))
                conn.commit()
                return current
            except Exception:
                conn.rollback()
                raise
