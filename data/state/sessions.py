import json
from datetime import datetime
from typing import Dict, Any, Optional

class SessionStore:
    def __init__(self, db, lock, backup_fn):
        self.db = db
        self.lock = lock
        self.backup = backup_fn

    def save(self, session_id: str, data: Dict[str, Any]) -> bool:
        now = datetime.now().isoformat()
        payload = json.dumps(data, ensure_ascii=False)

        with self.lock, self.db.connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, data, last_accessed, created_at)
                VALUES (?, ?, ?, COALESCE(
                    (SELECT created_at FROM sessions WHERE session_id = ?),
                    ?
                ))
            """, (session_id, payload, now, session_id, now))

        self.backup("sessions", session_id, data)
        return True

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.lock, self.db.connect() as conn:
            row = conn.execute(
                "SELECT data FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if not row:
                return None

            conn.execute(
                "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )

        try:
            return json.loads(row[0])
        except Exception:
            return None
