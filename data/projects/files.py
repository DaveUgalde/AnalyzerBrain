from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from .utils import save_json, load_json, sha256_text


def store_file(
    project_path: Path,
    logical_path: str,
    content: str,
) -> str:
    raw_dir = project_path / "raw_files"
    target = raw_dir / Path(logical_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(content, encoding="utf-8")

    content_hash = sha256_text(content)

    files_meta: List[Dict[str, Any]] = load_json(project_path / "files.json", [])

    info = {
        "path": logical_path,
        "hash": content_hash,
        "size_bytes": len(content.encode("utf-8")),
        "line_count": content.count("\n") + 1,
        "stored_at": datetime.now().isoformat(),
        "stored_path": str(target.relative_to(project_path)),
    }

    existing = next((f for f in files_meta if f["path"] == logical_path), None)
    if existing:
        existing.update(info)
    else:
        files_meta.append(info)

    save_json(project_path / "files.json", files_meta)
    return content_hash
