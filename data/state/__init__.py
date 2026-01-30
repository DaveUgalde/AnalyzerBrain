# data/state/__init__.py
"""
Sistema de gestión de estado persistente.
"""

import json
import sqlite3
import threading
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class StateManager:
    """Gestor de estado persistente del sistema."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state_path.mkdir(parents=True, exist_ok=True)

        self._init_directories()
        self._init_databases()
        self._init_locks()

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #

    def _init_directories(self) -> None:
        for directory in [
            "sessions",
            "operations",
            "components",
            "workflows",
            "agents",
            "snapshots",
            "backups",
        ]:
            (self.state_path / directory).mkdir(parents=True, exist_ok=True)

    def _init_databases(self) -> None:
        self.db_path = self.state_path / "state.db"
        self._create_tables()

    def _init_locks(self) -> None:
        self.locks = {
            "sessions": threading.RLock(),
            "operations": threading.RLock(),
            "components": threading.RLock(),
            "workflows": threading.RLock(),
            "agents": threading.RLock(),
            "events": threading.RLock(),
        }

    # ------------------------------------------------------------------ #
    # DB TABLES
    # ------------------------------------------------------------------ #

    def _create_tables(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                data TEXT,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP
            )
        """)

        cursor.execute("""
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
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS components (
                component_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                health_data TEXT,
                last_check TIMESTAMP,
                created_at TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                steps TEXT,
                current_step INTEGER,
                data TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                capabilities TEXT,
                metrics TEXT,
                last_active TIMESTAMP,
                created_at TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                component TEXT,
                data TEXT,
                severity TEXT,
                timestamp TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    # SESSIONS
    # ------------------------------------------------------------------ #

    def save_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        with self.locks["sessions"]:
            now = datetime.now().isoformat()
            payload = json.dumps(data, ensure_ascii=False)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, data, last_accessed, created_at)
                VALUES (?, ?, ?, COALESCE(
                    (SELECT created_at FROM sessions WHERE session_id = ?),
                    ?
                ))
            """, (session_id, payload, now, session_id, now))

            conn.commit()
            conn.close()

            self._backup_json("sessions", session_id, data)
            return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.locks["sessions"]:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT data FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            self._touch_session(session_id)

            try:
                return json.loads(row[0])
            except Exception:
                return None

    def _touch_session(self, session_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
            (datetime.now().isoformat(), session_id),
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    # OPERATIONS
    # ------------------------------------------------------------------ #

    def create_operation(
        self,
        operation_id: str,
        operation_type: str,
        project_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self.locks["operations"]:
            now = datetime.now().isoformat()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO operations
                (operation_id, type, status, project_id, data, progress, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?, 0.0, ?, ?)
            """, (
                operation_id,
                operation_type,
                project_id,
                json.dumps(data or {}, ensure_ascii=False),
                now,
                now,
            ))

            conn.commit()
            conn.close()
            return True

    # ------------------------------------------------------------------ #
    # WORKFLOWS
    # ------------------------------------------------------------------ #

    def save_workflow_state(
        self,
        workflow_id: str,
        name: str,
        status: str,
        steps: List[Dict[str, Any]],
        current_step: int,
        data: Dict[str, Any],
    ) -> bool:
        with self.locks["workflows"]:
            now = datetime.now().isoformat()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO workflows
                (workflow_id, name, status, steps, current_step, data, updated_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM workflows WHERE workflow_id = ?),
                    ?
                ))
            """, (
                workflow_id,
                name,
                status,
                json.dumps(steps, ensure_ascii=False),
                current_step,
                json.dumps(data, ensure_ascii=False),
                now,
                workflow_id,
                now,
            ))

            conn.commit()
            conn.close()

            self._backup_json("workflows", workflow_id, {
                "name": name,
                "status": status,
                "steps": steps,
                "current_step": current_step,
                "data": data,
                "updated_at": now,
            })

            return True

    # ------------------------------------------------------------------ #
    # EVENTS
    # ------------------------------------------------------------------ #

    def log_system_event(
        self,
        event_type: str,
        component: str,
        data: Dict[str, Any],
        severity: str = "info",
    ) -> int:
        with self.locks["events"]:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO system_events
                (type, component, data, severity, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                event_type,
                component,
                json.dumps(data, ensure_ascii=False),
                severity,
                datetime.now().isoformat(),
            ))

            event_id = cursor.lastrowid
            conn.commit()
            conn.close()

            self._rotate_old_events()
            return event_id

    # ------------------------------------------------------------------ #
    # SNAPSHOTS
    # ------------------------------------------------------------------ #

    def create_snapshot(self, description: str = "") -> str:
        snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.state_path / "snapshots" / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(self.db_path, snapshot_dir / "state.db")

        for name in ["sessions", "workflows", "operations"]:
            src = self.state_path / name
            if src.exists():
                shutil.copytree(src, snapshot_dir / name)

        metadata = {
            "snapshot_id": snapshot_id,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "size": self._get_directory_size(snapshot_dir),
        }

        with open(snapshot_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self._cleanup_old_snapshots()
        return snapshot_id

    # ------------------------------------------------------------------ #
    # UTILITIES
    # ------------------------------------------------------------------ #

    def _backup_json(self, folder: str, name: str, data: Dict[str, Any]) -> None:
        try:
            path = self.state_path / folder / f"{name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _rotate_old_events(self, max_events: int = 10_000) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM system_events")
        count = cursor.fetchone()[0]

        if count > max_events:
            cursor.execute("""
                DELETE FROM system_events
                WHERE event_id IN (
                    SELECT event_id FROM system_events
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
            """, (count - max_events,))

        conn.commit()
        conn.close()

    def _cleanup_old_snapshots(self, max_snapshots: int = 10) -> None:
        snapshots = sorted(
            (d for d in (self.state_path / "snapshots").iterdir() if d.is_dir()),
            key=lambda p: p.name,
        )

        while len(snapshots) > max_snapshots:
            shutil.rmtree(snapshots.pop(0), ignore_errors=True)

    def _get_directory_size(self, path: Path) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total
