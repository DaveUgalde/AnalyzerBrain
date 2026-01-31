import json
from datetime import datetime
from typing import Dict, Any, Optional


class OperationStore:
    def __init__(self, db, lock):
        self.db = db
        self.lock = lock

    def create(
        self,
        operation_id: str,
        operation_type: str,
        project_id: Optional[str],
        data: Optional[Dict[str, Any]],
    ) -> bool:
        now = datetime.now().isoformat()

        with self.lock, self.db.connect() as conn:
            conn.execute("""
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

        return True

    def update_status(
        self,
        operation_id: str,
        status: str,
        progress: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        completed: bool = False,
    ) -> bool:
        now = datetime.now().isoformat()

        with self.lock, self.db.connect() as conn:
            conn.execute("""
                UPDATE operations
                SET status = ?,
                    progress = COALESCE(?, progress),
                    data = COALESCE(?, data),
                    updated_at = ?,
                    completed_at = COALESCE(completed_at, ?)
                WHERE operation_id = ?
            """, (
                status,
                progress,
                json.dumps(data, ensure_ascii=False) if data else None,
                now,
                now if completed else None,
                operation_id,
            ))

        return True
