import json
import shutil
import os
from pathlib import Path
from datetime import datetime


class SnapshotManager:
    def __init__(self, state_path: Path, db_path: Path):
        self.state_path = state_path
        self.db_path = db_path

    def create(self, description: str = "") -> str:
        sid = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = self.state_path / "snapshots" / sid
        target.mkdir(parents=True, exist_ok=True)

        shutil.copy2(self.db_path, target / "state.db")

        meta = {
            "snapshot_id": sid,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "size": self._dir_size(target),
        }

        (target / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._cleanup()
        return sid

    def _cleanup(self, max_snapshots=10):
        snaps = sorted(
            (d for d in (self.state_path / "snapshots").iterdir() if d.is_dir()),
            key=lambda p: p.name,
        )
        while len(snaps) > max_snapshots:
            shutil.rmtree(snaps.pop(0), ignore_errors=True)

    def _dir_size(self, path: Path) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total
