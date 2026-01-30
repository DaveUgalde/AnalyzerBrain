# data/projects/__init__.py
"""
Sistema de almacenamiento para proyectos analizados.
"""

import json
import shutil
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...core.exceptions import ProjectException, ValidationError


class ProjectStorage:
    """Almacenamiento persistente para proyectos analizados."""

    def __init__(self, projects_base: Path):
        self.base = projects_base
        self.base.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # PROJECT CREATION
    # ======================================================

    def create_project(self, project_id: str, metadata: Dict[str, Any]) -> Path:
        project_path = self.base / project_id

        if project_path.exists():
            raise ProjectException(f"Project {project_id} already exists")

        dirs = [
            project_path,
            project_path / "entities",
            project_path / "dependencies",
            project_path / "issues",
            project_path / "snapshots",
            project_path / "embeddings",
            project_path / "raw_files",
            project_path / "processed",
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        now = datetime.now().isoformat()
        metadata.update({
            "id": project_id,
            "created_at": now,
            "updated_at": now,
            "status": "created",
        })

        self._save_json(project_path / "metadata.json", metadata)
        self._save_json(project_path / "files.json", [])
        self._save_json(project_path / "analysis.json", {})

        return project_path

    # ======================================================
    # ANALYSIS
    # ======================================================

    def save_analysis(self, project_id: str, analysis_data: Dict[str, Any]) -> bool:
        project_path = self._require_project(project_id)

        metadata = self._load_json(project_path / "metadata.json", default={})
        metadata.update({
            "updated_at": datetime.now().isoformat(),
            "last_analyzed": datetime.now().isoformat(),
            "analysis_version": analysis_data.get("version", "1.0.0"),
            "status": "analyzed",
        })

        self._save_json(project_path / "metadata.json", metadata)
        self._save_json(project_path / "analysis.json", analysis_data)

        for name, data in analysis_data.get("entities", {}).items():
            self._save_json(project_path / "entities" / f"{name}.json", data)

        for name, data in analysis_data.get("dependencies", {}).items():
            self._save_json(project_path / "dependencies" / f"{name}.json", data)

        for severity, data in analysis_data.get("issues", {}).items():
            self._save_json(project_path / "issues" / f"{severity}.json", data)

        return True

    # ======================================================
    # READ
    # ======================================================

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        project_path = self.base / project_id
        if not project_path.exists():
            return None

        return {
            "metadata": self._load_json(project_path / "metadata.json", {}),
            "analysis": self._load_json(project_path / "analysis.json", {}),
            "entities": self._load_group(project_path / "entities"),
            "dependencies": self._load_group(project_path / "dependencies"),
            "issues": self._load_group(project_path / "issues"),
        }

    # ======================================================
    # FILE STORAGE
    # ======================================================

    def save_file_content(self, project_id: str, file_path: str, content: str) -> str:
        project_path = self._require_project(project_id)

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        raw_dir = project_path / "raw_files"

        target = raw_dir / Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(content, encoding="utf-8")

        files_meta = self._load_json(project_path / "files.json", [])

        info = {
            "path": file_path,
            "hash": content_hash,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": content.count("\n") + 1,
            "stored_at": datetime.now().isoformat(),
            "stored_path": str(target.relative_to(project_path)),
        }

        existing = next((f for f in files_meta if f["path"] == file_path), None)
        if existing:
            existing.update(info)
        else:
            files_meta.append(info)

        self._save_json(project_path / "files.json", files_meta)
        return content_hash

    # ======================================================
    # SNAPSHOTS
    # ======================================================

    def create_snapshot(self, project_id: str, description: str = "") -> str:
        project_path = self._require_project(project_id)

        snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir = project_path / "snapshots" / snapshot_id
        snap_dir.mkdir(parents=True)

        for name in ("metadata.json", "files.json", "analysis.json"):
            src = project_path / name
            if src.exists():
                shutil.copy2(src, snap_dir / name)

        self._save_json(snap_dir / "snapshot_meta.json", {
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

    # ======================================================
    # EXPORT
    # ======================================================

    def export_project(self, project_id: str, export_format: str = "json") -> Dict[str, Any]:
        project_path = self._require_project(project_id)
        data = self.get_project(project_id)

        if export_format == "json":
            return {
                "format": "json",
                "exported_at": datetime.now().isoformat(),
                "project": data,
            }

        import tempfile
        import zipfile
        import tarfile

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{export_format}")
        tmp.close()

        if export_format == "zip":
            with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as z:
                for root, _, files in os.walk(project_path):
                    for f in files:
                        fp = Path(root) / f
                        z.write(fp, fp.relative_to(self.base))
        elif export_format == "tar":
            with tarfile.open(tmp.name, "w:gz") as tar:
                tar.add(project_path, arcname=project_id)
        else:
            raise ValidationError(f"Unsupported export format: {export_format}")

        return {
            "format": export_format,
            "path": tmp.name,
            "size": os.path.getsize(tmp.name),
        }

    # ======================================================
    # INTERNAL HELPERS
    # ======================================================

    def _require_project(self, project_id: str) -> Path:
        path = self.base / project_id
        if not path.exists():
            raise ProjectException(f"Project {project_id} not found")
        return path

    def _load_group(self, directory: Path) -> Dict[str, Any]:
        if not directory.exists():
            return {}
        return {f.stem: self._load_json(f, []) for f in directory.glob("*.json")}

    def _save_json(self, path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
