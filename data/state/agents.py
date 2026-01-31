import json
from datetime import datetime
from typing import Dict, Any


class AgentStore:
    def __init__(self, db, lock):
        self.db = db
        self.lock = lock

    def update(
        self,
        agent_id: str,
        name: str,
        status: str,
        capabilities: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> bool:
        now = datetime.now().isoformat()

        with self.lock, self.db.connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agents
                (agent_id, name, status, capabilities, metrics, last_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM agents WHERE agent_id = ?),
                    ?
                ))
            """, (
                agent_id,
                name,
                status,
                json.dumps(capabilities, ensure_ascii=False),
                json.dumps(metrics, ensure_ascii=False),
                now,
                agent_id,
                now,
            ))

        return True
