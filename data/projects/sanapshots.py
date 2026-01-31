import shutil
from datetime import datetime
from pathlib import Path

from .utils import save_json


def create_snapshot(project_path: Path, project_id: str, description: str) -> str:
    snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = project_path / "snapshots" / snapshot_id
    snap_dir.mkdir(parents=True)

    for name in ("metadata.json", "files.json", "analysis.json"):
        src = project_path / name
        if src.exists():
            shutil.copy2(src, snap_dir / name)

    save_json(snap_dir / "snapshot_meta.json", {
        "id": snapshot_id,
        "project_id": project_id,
        "created_at": datetime.now().isoformat(),
        "description": description,
    })

    latest = project_path / "snapshots" / "latest"
    if latest.exists():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)

    try:
        latest.symlink_to(snapshot_id)
    except Exception:
        shutil.copytree(snap_dir, latest)

    return snapshot_id
