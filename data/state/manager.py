import threading
import json
from pathlib import Path

from .db import Database
from .sessions import SessionStore
from .operations import OperationStore
from .workflows import WorkflowStore
from .components import ComponentStore
from .agents import AgentStore
from .events import EventLogger
from .snapshots import SnapshotManager


class StateManager:
    """Gestor de estado persistente del sistema."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state_path.mkdir(parents=True, exist_ok=True)

        self.db = Database(self.state_path / "state.db")
        self.db.init_tables()

        self.locks = {
            "sessions": threading.RLock(),
            "operations": threading.RLock(),
            "workflows": threading.RLock(),
            "components": threading.RLock(),
            "agents": threading.RLock(),
            "events": threading.RLock(),
        }

        self.sessions = SessionStore(self.db, self.locks["sessions"], self._backup_json)
        self.operations = OperationStore(self.db, self.locks["operations"])
        self.workflows = WorkflowStore(self.db, self.locks["workflows"], self._backup_json)
        self.components = ComponentStore(self.db, self.locks["components"])
        self.agents = AgentStore(self.db, self.locks["agents"])
        self.events = EventLogger(self.db, self.locks["events"])
        self.snapshots = SnapshotManager(self.state_path, self.db.db_path)

    # === API pública preservada ===

    def save_session(self, session_id, data):
        return self.sessions.save(session_id, data)

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def create_operation(self, *args, **kwargs):
        return self.operations.create(*args, **kwargs)

    def update_operation(self, *args, **kwargs):
        return self.operations.update_status(*args, **kwargs)

    def save_workflow_state(self, *args, **kwargs):
        return self.workflows.save(*args, **kwargs)

    def log_system_event(self, *args, **kwargs):
        return self.events.log(*args, **kwargs)

    def create_snapshot(self, description=""):
        return self.snapshots.create(description)

    # === Utils ===

    def _backup_json(self, folder, name, data):
        try:
            path = self.state_path / folder
            path.mkdir(exist_ok=True)
            (path / f"{name}.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
