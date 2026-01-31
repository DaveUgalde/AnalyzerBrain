import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any


def checksum(file: Path) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def backup_full(source: str, src_dir: Path, dest_dir: Path) -> List[Dict[str, Any]]:
    files = []
    for file in src_dir.rglob("*"):
        if not file.is_file():
            continue

        rel = file.relative_to(src_dir)
        dest = dest_dir / source / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)

        files.append({
            "source": source,
            "path": str(rel),
            "size": file.stat().st_size,
            "modified": file.stat().st_mtime,
            "checksum": checksum(file),
        })
    return files


def backup_incremental(
    source: str,
    src_dir: Path,
    dest_dir: Path,
    base_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    files = []

    for file in src_dir.rglob("*"):
        if not file.is_file():
            continue

        rel = str(file.relative_to(src_dir))
        cs = checksum(file)
        mtime = file.stat().st_mtime

        prev = base_state.get(rel)
        if prev and prev["checksum"] == cs and prev["modified"] >= mtime:
            continue

        dest = dest_dir / source / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)

        files.append({
            "source": source,
            "path": rel,
            "size": file.stat().st_size,
            "modified": mtime,
            "checksum": cs,
        })

    return files
