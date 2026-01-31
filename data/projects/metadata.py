from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .utils import save_json, load_json


def init_metadata(path: Path, project_id: str, metadata: Dict[str, Any]) -> None:
    now = datetime.now().isoformat()
    meta = dict(metadata)
    meta.update({
        "id": project_id,
        "created_at": now,
        "updated_at": now,
        "status": "created",
    })
    save_json(path / "metadata.json", meta)


def update_metadata(path: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    meta = load_json(path / "metadata.json", {})
    meta.update(updates)
    meta["updated_at"] = datetime.now().isoformat()
    save_json(path / "metadata.json", meta)
    return meta
