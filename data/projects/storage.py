from pathlib import Path
from typing import Dict, Any, Optional

from .filesystem import ensure_project_structure, require_project
from .metadata import init_metadata
from .analysis import save_analysis as _save_analysis
from .files import store_file
from .snapshots import create_snapshot as _create_snapshot
from .export import export_project
from .utils import load_json


class ProjectStorage:
    """Almacenamiento persistente para proyectos analizados."""

    def __init__(self, projects_base: Path):
        self.base = projects_base
        self.base.mkdir(parents=True, exist_ok=True)

    # ===== CREATION =====

    def create_project(self, project_id: str, metadata: Dict[str, Any]) -> Path:
        path = ensure_project_structure(self.base, project_id)
        init_metadata(path, project_id, metadata)

        from .utils import save_json
        save_json(path / "files.json", [])
        save_json(path / "analysis.json", {})

        return path

    # ===== ANALYSIS =====

    def save_analysis(self, project_id: str, analysis_data: Dict[str, Any]) -> bool:
        path = require_project(self.base, project_id)
        _save_analysis(path, analysis_data)
        return True

    # ===== READ =====

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        path = self.base / project_id
        if not path.exists():
            return None

        return {
            "metadata": load_json(path / "metadata.json", {}),
            "analysis": load_json(path / "analysis.json", {}),
            "entities": self._load_group(path / "entities"),
            "dependencies": self._load_group(path / "dependencies"),
            "issues": self._load_group(path / "issues"),
        }

    # ===== FILES =====

    def save_file_content(self, project_id: str, file_path: str, content: str) -> str:
        path = require_project(self.base, project_id)
        return store_file(path, file_path, content)

    # ===== SNAPSHOTS =====

    def create_snapshot(self, project_id: str, description: str = "") -> str:
        path = require_project(self.base, project_id)
        return _create_snapshot(path, project_id, description)

    # ===== EXPORT =====

    def export_project(self, project_id: str, export_format: str = "json"):
        path = require_project(self.base, project_id)
        data = self.get_project(project_id)
        return export_project(self.base, path, project_id, data, export_format)

    # ===== INTERNAL =====

    def _load_group(self, directory: Path) -> Dict[str, Any]:
        if not directory.exists():
            return {}
        return {
            f.stem: load_json(f, [])
            for f in directory.glob("*.json")
        }
