from pathlib import Path
from typing import Dict, Any, Callable
from datetime import datetime

from .exporters import EXPORTERS
from .utils import normalize_graph, count_graph, log_export
from ...core.exceptions import GraphException


class GraphExporter:
    """Sistema de exportación de grafos de conocimiento."""

    SUPPORTED_FORMATS = set(EXPORTERS.keys())

    def __init__(self, export_path: Path):
        self.export_path = export_path
        self.export_path.mkdir(parents=True, exist_ok=True)

    def export_graph(
        self,
        graph_data: Dict[str, Any],
        project_id: str,
        export_format: str = "json",
        **_,
    ) -> Dict[str, Any]:
        export_format = export_format.lower()

        if export_format not in self.SUPPORTED_FORMATS:
            raise GraphException(f"Unsupported format: {export_format}")

        graph = normalize_graph(graph_data)
        exporter = EXPORTERS[export_format]
        export_file = self._build_export_path(project_id, export_format)

        try:
            result = exporter(graph, export_file)
            log_export(self.export_path, project_id, export_format, export_file)

            return {
                "success": True,
                "format": export_format,
                "file_path": str(export_file),
                "file_size": export_file.stat().st_size if export_file.exists() else None,
                **count_graph(graph),
                **(result or {}),
            }

        except Exception as exc:
            raise GraphException(
                f"Export failed ({export_format}) → {export_file.name}: {exc}"
            ) from exc

    def _build_export_path(self, project_id: str, export_format: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.export_path / f"{project_id}_{ts}.{export_format}"
