# scripts/export_knowledge.py
"""
Script para exportar conocimiento de Project Brain.
Permite exportar grafos de conocimiento, embeddings, análisis, etc.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import json
import yaml
from datetime import datetime
import pickle
import csv

# -------------------------------------------------------------------
# Paths robustos
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -------------------------------------------------------------------
# Imports internos
# -------------------------------------------------------------------

from core.orchestrator import BrainOrchestrator
from core.config_manager import ConfigManager
from utils.logging_config import setup_logging

# -------------------------------------------------------------------

SUPPORTED_FORMATS = ("json", "yaml", "pickle", "csv")

# -------------------------------------------------------------------
# KnowledgeExporter
# -------------------------------------------------------------------

class KnowledgeExporter:
    """Clase para exportar conocimiento del sistema."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(BASE_DIR / "config" / "system.yaml")
        self.config: Optional[Dict[str, Any]] = None
        self.orchestrator: Optional[BrainOrchestrator] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        try:
            (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)

            setup_logging({
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": str(BASE_DIR / "logs" / "export_knowledge.log"),
            })

            self.config = ConfigManager(self.config_path).get_config()

            self.orchestrator = BrainOrchestrator(self.config_path)
            await self.orchestrator.initialize()

            self.logger.info("✅ Exportador inicializado correctamente")
            return True

        except Exception:
            self.logger.exception("❌ Error inicializando exportador")
            return False

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    async def export_project_knowledge(
        self,
        project_id: str,
        output_path: str,
        export_format: str = "json",
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        export_format = export_format.lower()
        if export_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Formato no soportado: {export_format}")

        if include is None or "all" in include:
            include = ["graph", "embeddings", "analysis", "metadata"]

        export_data: Dict[str, Any] = {
            "project_id": project_id,
            "export_timestamp": datetime.now().isoformat(),
            "format": export_format,
            "included": include,
            "data": {},
        }

        if "graph" in include:
            export_data["data"]["knowledge_graph"] = await self._export_knowledge_graph(project_id)

        if "embeddings" in include:
            export_data["data"]["embeddings"] = await self._export_embeddings(project_id)

        if "analysis" in include:
            export_data["data"]["analysis"] = await self._export_analysis(project_id)

        if "metadata" in include:
            export_data["data"]["metadata"] = await self._export_metadata(project_id)

        self._save_export(export_data, output_path, export_format)
        export_data["statistics"] = self._calculate_export_stats(export_data)

        self.logger.info(
            "📤 Exportación completada | proyecto=%s formato=%s",
            project_id,
            export_format,
        )

        return export_data

    # ------------------------------------------------------------------
    # EXPORTERS (placeholders compatibles)
    # ------------------------------------------------------------------

    async def _export_knowledge_graph(self, project_id: str) -> Dict[str, Any]:
        entities = [
            {"id": "func_001", "type": "function", "name": "calculate_total", "file": "main.py", "line": 42},
            {"id": "func_002", "type": "function", "name": "process_data", "file": "utils.py", "line": 15},
            {"id": "cls_001", "type": "class", "name": "Database", "file": "database.py", "line": 1},
        ]

        relationships = [
            {"source": "func_001", "target": "func_002", "type": "calls", "weight": 0.8},
            {"source": "func_001", "target": "cls_001", "type": "uses", "weight": 0.6},
        ]

        return {
            "entities": entities,
            "relationships": relationships,
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "export_timestamp": datetime.now().isoformat(),
        }

    async def _export_embeddings(self, project_id: str) -> Dict[str, Any]:
        embeddings = [
            {
                "entity_id": "func_001",
                "entity_type": "function",
                "embedding": [0.1, 0.2, 0.3] * 128,
                "model": "all-MiniLM-L6-v2",
            }
        ]

        return {
            "embeddings": embeddings,
            "total_embeddings": len(embeddings),
            "embedding_dimensions": 384,
        }

    async def _export_analysis(self, project_id: str) -> Dict[str, Any]:
        return {
            "analysis": {
                "quality_metrics": {
                    "maintainability_index": 78.5,
                    "cyclomatic_complexity_avg": 3.2,
                },
                "issues": [],
                "patterns_detected": [],
            },
            "issue_count": 0,
            "pattern_count": 0,
        }

    async def _export_metadata(self, project_id: str) -> Dict[str, Any]:
        return {
            "project_id": project_id,
            "export_timestamp": datetime.now().isoformat(),
            "system": self.config.get("system", {}) if self.config else {},
        }

    # ------------------------------------------------------------------
    # SAVE / SERIALIZATION
    # ------------------------------------------------------------------

    def _save_export(self, data: Dict[str, Any], output_path: str, export_format: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if export_format == "json":
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        elif export_format == "yaml":
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

        elif export_format == "pickle":
            with path.open("wb") as f:
                pickle.dump(data, f)

        elif export_format == "csv":
            self._save_as_csv(data, path)

    def _save_as_csv(self, data: Dict[str, Any], base_path: Path) -> None:
        graph = data.get("data", {}).get("knowledge_graph", {})
        entities = graph.get("entities", [])

        if not entities:
            self.logger.warning("⚠️ No hay entidades para exportar a CSV")
            return

        csv_path = base_path.with_suffix("").with_name(base_path.stem + "_entities.csv")

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entities[0].keys())
            writer.writeheader()
            writer.writerows(entities)

    # ------------------------------------------------------------------
    # STATS & LIFECYCLE
    # ------------------------------------------------------------------

    def _calculate_export_stats(self, export_data: Dict[str, Any]) -> Dict[str, Any]:
        raw = json.dumps(export_data, ensure_ascii=False).encode("utf-8")
        return {
            "total_entities": export_data.get("data", {}).get("knowledge_graph", {}).get("entity_count", 0),
            "total_embeddings": export_data.get("data", {}).get("embeddings", {}).get("total_embeddings", 0),
            "total_issues": export_data.get("data", {}).get("analysis", {}).get("issue_count", 0),
            "total_patterns": export_data.get("data", {}).get("analysis", {}).get("pattern_count", 0),
            "export_size_bytes": len(raw),
        }

    async def shutdown(self) -> None:
        if self.orchestrator:
            await self.orchestrator.shutdown()

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Exporta conocimiento de Project Brain")
    parser.add_argument("--config", default=str(BASE_DIR / "config" / "system.yaml"))

    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Exportar conocimiento de un proyecto")
    export.add_argument("project_id")
    export.add_argument("output")
    export.add_argument("--format", default="json", choices=SUPPORTED_FORMATS)
    export.add_argument("--include", help="Elementos a incluir (graph,embeddings,analysis,metadata,all)")

    args = parser.parse_args()

    async def run() -> int:
        exporter = KnowledgeExporter(args.config)
        if not await exporter.initialize():
            return 1

        try:
            include = args.include.split(",") if args.include else None

            await exporter.export_project_knowledge(
                project_id=args.project_id,
                output_path=args.output,
                export_format=args.format,
                include=include,
            )
            return 0

        finally:
            await exporter.shutdown()

    return asyncio.run(run())

if __name__ == "__main__":
    raise SystemExit(main())
