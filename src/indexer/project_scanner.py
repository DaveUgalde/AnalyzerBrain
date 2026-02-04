"""
Escáner de proyectos: Recorre directorios y recopila archivos para análisis.
"""

import os
from pathlib import Path
from typing import List, Set
from loguru import logger

from ..core.config_manager import config
from ..core.exceptions import IndexerError
from ..utils.file_utils import safe_read_file
from .models import ProjectStructure


class ProjectScanner:
    """Escanea proyectos para análisis."""

    def __init__(self, config_manager=None):
        self.config = config_manager or config
        self.excluded_dirs: Set[str] = {
            '.git',
            '.venv',
            'venv',
            'node_modules',
            '__pycache__',
            '.pytest_cache',
            'dist',
            'build',
        }
        self.excluded_extensions: Set[str] = {
            '.pyc',
            '.pyo',
            '.pyd',
            '.so',
            '.dll',
            '.exe',
            '.bin',
            '.class',
            '.jar',
        }

    def scan(self, project_path: Path) -> ProjectStructure:
        """Escanea un proyecto y retorna su estructura."""
        if not project_path.exists():
            raise IndexerError(f"Proyecto no encontrado: {project_path}")

        files: List[Path] = []
        directories: List[Path] = []

        for root, dirs, file_names in os.walk(project_path):
            # Filtrar directorios excluidos
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

            for dir_name in dirs:
                directories.append(Path(root) / dir_name)

            for file_name in file_names:
                file_path = Path(root) / file_name
                if self._should_process(file_path):
                    files.append(file_path)

        return ProjectStructure(
            root=project_path,
            files=files,
            directories=directories,
            metadata={
                "total_files": len(files),
                "total_dirs": len(directories),
                "scanned_at": datetime.now().isoformat(),
            },
        )

    def _should_process(self, file_path: Path) -> bool:
        """Determina si un archivo debe ser procesado."""
        # Verificar extensión
        if file_path.suffix in self.excluded_extensions:
            return False

        # Verificar tamaño máximo
        max_size_mb = self.config.get("indexer.max_file_size_mb", 10)
        max_size_bytes = max_size_mb * 1024 * 1024
        try:
            file_size = file_path.stat().st_size
            if file_size > max_size_bytes:
                logger.warning(f"Archivo demasiado grande, omitiendo: {file_path}")
                return False
        except OSError:
            return False

        return True
