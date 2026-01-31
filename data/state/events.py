import json
from datetime import datetime


class EventLogger:
    def __init__(self, db, lock):
        self.db = db
        self.lock = lock

    def log(self, event_type, component, data, severity="info") -> int:
        with self.lock, self.db.connect() as conn:
            cur = conn.execute("""
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

            self._rotate(conn)
            return cur.lastrowid

    def _rotate(self, conn, max_events=10_000):
        count = conn.execute(
            "SELECT COUNT(*) FROM system_events"
        ).fetchone()[0]

        if count > max_events:
            conn.execute("""
                DELETE FROM system_events
                WHERE event_id IN (
                    SELECT event_id FROM system_events
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
            """, (count - max_events,))
