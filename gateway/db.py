import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(
  id TEXT PRIMARY KEY, path TEXT NOT NULL, kind TEXT NOT NULL,
  width INTEGER, height INTEGER, exif TEXT, rrdata TEXT,
  size INTEGER NOT NULL, created REAL NOT NULL, last_used REAL NOT NULL,
  evicted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, capability TEXT NOT NULL, source_id TEXT NOT NULL,
  params TEXT NOT NULL, priority TEXT NOT NULL, status TEXT NOT NULL,
  created REAL NOT NULL, started REAL, finished REAL,
  progress REAL, result TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority, created);
"""

class Db:
    """Tiny thread-safe sqlite wrapper. All operations are short; a single
    lock keeps things simple (the gateway is one process, low QPS)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def execute(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

def now() -> float:
    return time.time()

def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))

def loads(s):
    return json.loads(s) if s else None
