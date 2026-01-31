import sqlite3
from pathlib import Path
from contextlib import contextmanager


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_tables(self):
        with self.connect() as conn:
            c = conn.cursor()

            c.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                data TEXT,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS operations (
                operation_id TEXT PRIMARY KEY,
                type TEXT,
                status TEXT,
                project_id TEXT,
                data TEXT,
                progress REAL,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                completed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                steps TEXT,
                current_step INTEGER,
                data TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS system_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                component TEXT,
                data TEXT,
                severity TEXT,
                timestamp TIMESTAMP
            );
            """)
