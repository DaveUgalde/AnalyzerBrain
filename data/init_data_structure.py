#!/usr/bin/env python3
"""
Script de inicialización de la estructura de datos de Project Brain.
"""

import sys
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime


# =====================================================
# UTILIDADES
# =====================================================

class FileWriter:
    @staticmethod
    def write_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def write_yaml(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    @staticmethod
    def write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


# =====================================================
# CREADORES
# =====================================================

class DirectoryCreator:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def create(self) -> None:
        structure = {
            "projects": ["files", "snapshots", "change_logs", "reports"],
            "embeddings": ["collections", "archive", "temp"],
            "graph_exports": ["templates", "system", "temp"],
            "cache": [f"shard_{i}" for i in range(8)] + ["metadata"],
            "state": [
                "agents_state", "learning_state", "knowledge_state",
                "workflow_state", "metrics_history", "snapshots"
            ],
            "backups": ["full", "incremental", "components", "snapshots", "metadata"],
        }

        for main, subs in structure.items():
            main_path = self.base_path / main
            main_path.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Creado: {main_path}")

            for sub in subs:
                sub_path = main_path / sub
                sub_path.mkdir(exist_ok=True)
                print(f"    └─ 📁 Creado: {sub_path}")

            (main_path / ".gitkeep").touch()


class ConfigCreator:
    def __init__(self, base_path: Path, timestamp: str):
        self.base_path = base_path
        self.timestamp = timestamp

    def create(self) -> None:
        FileWriter.write_yaml(
            self.base_path / "embeddings/chromadb_config.yaml",
            {
                "version": "1.0.0",
                "created_at": self.timestamp,
                "system": "project_brain",
                "persistence": {
                    "persist_directory": str(self.base_path / "embeddings"),
                    "migration_mode": "auto",
                    "allow_reset": False,
                },
            },
        )

        FileWriter.write_json(
            self.base_path / "cache/l3_cache_config.json",
            {
                "version": "1.0.0",
                "created_at": self.timestamp,
                "cache_level": 3,
                "storage": {
                    "type": "filesystem",
                    "base_directory": str(self.base_path / "cache"),
                    "shards": 8,
                },
            },
        )

        FileWriter.write_json(
            self.base_path / "backups/backup_manifest.json",
            {
                "manifest_version": "1.0.0",
                "created_at": self.timestamp,
                "strategy": "full_incremental_hybrid",
            },
        )


class TemplateCreator:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def create(self) -> None:
        FileWriter.write_text(
            self.base_path / "projects/project_template.json",
            json.dumps({
                "project_id": "{{PROJECT_ID}}",
                "name": "{{PROJECT_NAME}}",
                "created_at": "{{TIMESTAMP}}",
            }, indent=2),
        )

        FileWriter.write_text(
            self.base_path / "graph_exports/templates/export_template.cypher",
            "-- Cypher export template\n",
        )

        FileWriter.write_text(
            self.base_path / "graph_exports/templates/export_template.graphml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<graphml></graphml>",
        )

        FileWriter.write_text(
            self.base_path / "state/agents_state_template.json",
            json.dumps({
                "agent_id": "{{AGENT_ID}}",
                "state": "ready",
                "timestamp": "{{TIMESTAMP}}",
            }, indent=2),
        )


class StateCreator:
    def __init__(self, base_path: Path, timestamp: str):
        self.base_path = base_path
        self.timestamp = timestamp

    def create(self) -> None:
        FileWriter.write_json(
            self.base_path / "state/system_state.json",
            {
                "system": "Project Brain",
                "version": "1.0.0",
                "status": "initialized",
                "timestamp": self.timestamp,
            },
        )


class ReadmeCreator:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def create(self) -> None:
        readmes = {
            "projects/README.md": "# Proyectos\n\nProyectos analizados por Project Brain.\n",
            "embeddings/README.md": "# Embeddings\n\nBase de datos vectorial del sistema.\n",
            "graph_exports/README.md": "# Exportaciones de Grafo\n\nFormatos Cypher y GraphML.\n",
            "cache/README.md": "# Caché\n\nCaché persistente en disco.\n",
            "state/README.md": "# Estado\n\nEstado persistente del sistema.\n",
            "backups/README.md": "# Backups\n\nBackups automáticos del sistema.\n",
        }

        for rel, content in readmes.items():
            FileWriter.write_text(self.base_path / rel, content)


# =====================================================
# ORQUESTADOR
# =====================================================

class DataStructureInitializer:
    """Orquestador de inicialización de datos."""

    def __init__(self, base_path: str = "./data", force: bool = False):
        self.base_path = Path(base_path)
        self.force = force
        self.timestamp = datetime.now().isoformat()

    def initialize_all(self) -> bool:
        try:
            print("🚀 Inicializando estructura de datos de Project Brain...")

            if self.force and self.base_path.exists():
                print(f"⚠️  Eliminando estructura existente: {self.base_path}")
                shutil.rmtree(self.base_path)

            DirectoryCreator(self.base_path).create()
            ConfigCreator(self.base_path, self.timestamp).create()
            TemplateCreator(self.base_path).create()
            StateCreator(self.base_path, self.timestamp).create()
            ReadmeCreator(self.base_path).create()

            print("✅ Estructura de datos inicializada exitosamente!")
            return True

        except Exception as e:
            print(f"❌ Error inicializando estructura de datos: {e}")
            return False


# =====================================================
# CLI
# =====================================================

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Inicializa la estructura de datos de Project Brain"
    )
    parser.add_argument("--path", default="./data")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    initializer = DataStructureInitializer(args.path, args.force)
    return 0 if initializer.initialize_all() else 1


if __name__ == "__main__":
    sys.exit(main())
