import json
from datetime import datetime
from typing import Dict, Any, List


class WorkflowStore:
    def __init__(self, db, lock, backup_fn):
        self.db = db
        self.lock = lock
        self.backup = backup_fn

    def save(
        self,
        workflow_id: str,
        name: str,
        status: str,
        steps: List[Dict[str, Any]],
        current_step: int,
        data: Dict[str, Any],
    ) -> bool:
        now = datetime.now().isoformat()

        with self.lock, self.db.connect() as conn:
            conn.execute("""
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

        self.backup("workflows", workflow_id, {
            "name": name,
            "status": status,
            "steps": steps,
            "current_step": current_step,
            "data": data,
        })

        return True
