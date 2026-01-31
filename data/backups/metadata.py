import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional


class BackupMetadata:
    def __init__(self, base_path: Path):
        self.file = base_path / "backup_metadata.json"
        self.lock = threading.RLock()
        self.data = {
            "backups": [],
            "last_backup": None,
            "total_backups": 0,
            "successful_backups": 0,
            "failed_backups": 0,
            "total_size": 0,
        }
        self._load()

    def _load(self) -> None:
        if self.file.exists():
            try:
                self.data.update(json.loads(self.file.read_text()))
            except Exception:
                pass

    def save(self) -> None:
        with self.lock:
            self.file.write_text(json.dumps(self.data, indent=2))

    def register(self, info: Dict[str, Any]) -> None:
        with self.lock:
            self.data["backups"].append(info)
            self.data["total_backups"] += 1

            if info["status"] == "success":
                self.data["successful_backups"] += 1
                self.data["last_backup"] = info["created_at"]
                self.data["total_size"] += info.get("size", 0)
            else:
                self.data["failed_backups"] += 1

            self.data["backups"] = self.data["backups"][-1000:]
            self.save()

    def last_successful(self) -> Optional[Dict[str, Any]]:
        for b in reversed(self.data["backups"]):
            if b["status"] == "success":
                return b
        return None
