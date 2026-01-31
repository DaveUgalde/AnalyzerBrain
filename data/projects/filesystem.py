from pathlib import Path
from typing import Iterable

from ...core.exceptions import ProjectException


PROJECT_DIRS = (
    "entities",
    "dependencies",
    "issues",
    "snapshots",
    "embeddings",
    "raw_files",
    "processed",
)


def ensure_project_structure(base: Path, project_id: str) -> Path:
    project_path = base / project_id
    if project_path.exists():
        raise ProjectException(f"Project {project_id} already exists")

    project_path.mkdir(parents=True)
    for name in PROJECT_DIRS:
        (project_path / name).mkdir()

    return project_path


def require_project(base: Path, project_id: str) -> Path:
    path = base / project_id
    if not path.exists():
        raise ProjectException(f"Project {project_id} not found")
    return path
