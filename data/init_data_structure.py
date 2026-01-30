#!/usr/bin/env python3
"""
Script de inicialización de la estructura de datos de Project Brain.
Este script crea todos los directorios necesarios y archivos de configuración.
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime


class DataStructureInitializer:
    """Inicializador de la estructura de datos del sistema."""

    def __init__(self, base_path: str = "./data", force: bool = False):
        self.base_path = Path(base_path)
        self.force = force
        self.timestamp = datetime.now().isoformat()

    def initialize_all(self) -> bool:
        """Inicializa toda la estructura de datos."""
        try:
            print("🚀 Inicializando estructura de datos de Project Brain...")

            if self.base_path.exists() and self.force:
                print(f"⚠️  Eliminando estructura existente: {self.base_path}")
                for item in self.base_path.iterdir():
                    if item.is_dir():
                        for sub in item.rglob("*"):
                            if sub.is_file():
                                sub.unlink()
                        item.rmdir()
                    else:
                        item.unlink()

            self._create_main_directories()
            self._create_configuration_files()
            self._create_template_files()
            self._create_initial_state_files()

            if not self._verify_structure():
                raise RuntimeError("La verificación de la estructura falló")

            print("✅ Estructura de datos inicializada exitosamente!")
            return True

        except Exception as e:
            print(f"❌ Error inicializando estructura de datos: {e}")
            return False

    def _create_main_directories(self) -> None:
        """Crea los directorios principales."""
        directories = {
            "projects": ["files", "snapshots", "change_logs", "reports"],
            "embeddings": ["collections", "archive", "temp"],
            "graph_exports": ["templates", "system", "temp"],
            "cache": [
                "shard_0", "shard_1", "shard_2", "shard_3",
                "shard_4", "shard_5", "shard_6", "shard_7", "metadata"
            ],
            "state": [
                "agents_state", "learning_state", "knowledge_state",
                "workflow_state", "metrics_history", "snapshots"
            ],
            "backups": ["full", "incremental", "components", "snapshots", "metadata"]
        }

        for main_dir, subs in directories.items():
            main_path = self.base_path / main_dir
            main_path.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Creado: {main_path}")

            for sub in subs:
                sub_path = main_path / sub
                sub_path.mkdir(exist_ok=True)
                print(f"    └─ 📁 Creado: {sub_path}")

            (main_path / ".gitkeep").touch()

    def _create_configuration_files(self) -> None:
        """Crea archivos de configuración."""
        configs = {
            "embeddings/chromadb_config.yaml": self._get_chromadb_config(),
            "cache/l3_cache_config.json": self._get_cache_config(),
            "backups/backup_manifest.json": self._get_backup_manifest()
        }

        for rel_path, content in configs.items():
            path = self.base_path / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.suffix == ".yaml":
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(content, f, allow_unicode=True, sort_keys=False)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)

            print(f"  📄 Creado: {path}")

    def _create_template_files(self) -> None:
        """Crea archivos de plantilla."""
        templates = {
            "projects/project_template.json": self._get_project_template(),
            "graph_exports/templates/export_template.cypher": self._get_cypher_template(),
            "graph_exports/templates/export_template.graphml": self._get_graphml_template(),
            "state/agents_state_template.json": self._get_agent_state_template()
        }

        for rel_path, content in templates.items():
            path = self.base_path / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"  📄 Creado: {path}")

    def _create_initial_state_files(self) -> None:
        """Crea archivos de estado inicial."""
        state_path = self.base_path / "state" / "system_state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self._get_system_state(), f, indent=2, ensure_ascii=False)
        print(f"  📄 Creado: {state_path}")

        readmes = {
            "projects/README.md": self._get_projects_readme(),
            "embeddings/README.md": self._get_embeddings_readme(),
            "graph_exports/README.md": self._get_graph_exports_readme(),
            "cache/README.md": self._get_cache_readme(),
            "state/README.md": self._get_state_readme(),
            "backups/README.md": self._get_backups_readme()
        }

        for rel_path, content in readmes.items():
            path = self.base_path / rel_path
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  📄 Creado: {path}")

    def _verify_structure(self) -> bool:
        """Verifica que la estructura se creó correctamente."""
        print("\n🔍 Verificando estructura creada...")

        required = [
            "projects", "embeddings", "graph_exports",
            "cache", "state", "backups",
            "projects/project_template.json",
            "embeddings/chromadb_config.yaml",
            "graph_exports/templates/export_template.cypher",
            "cache/l3_cache_config.json",
            "state/system_state.json",
            "backups/backup_manifest.json"
        ]

        for item in required:
            path = self.base_path / item
            if not path.exists():
                print(f"  ❌ Faltante: {path}")
                return False
            print(f"  ✅ OK: {path}")

        return True

    # ---------- Configs & Templates ----------

    def _get_chromadb_config(self) -> dict:
        return {
            "version": "1.0.0",
            "created_at": self.timestamp,
            "system": "project_brain",
            "persistence": {
                "persist_directory": str(self.base_path / "embeddings"),
                "migration_mode": "auto",
                "allow_reset": False
            }
        }

    def _get_cache_config(self) -> dict:
        return {
            "version": "1.0.0",
            "created_at": self.timestamp,
            "cache_level": 3,
            "storage": {
                "type": "filesystem",
                "base_directory": str(self.base_path / "cache"),
                "shards": 8
            }
        }

    def _get_backup_manifest(self) -> dict:
        return {
            "manifest_version": "1.0.0",
            "created_at": self.timestamp,
            "strategy": "full_incremental_hybrid"
        }

    def _get_project_template(self) -> str:
        return json.dumps({
            "project_id": "{{PROJECT_ID}}",
            "name": "{{PROJECT_NAME}}",
            "created_at": "{{TIMESTAMP}}"
        }, indent=2)

    def _get_cypher_template(self) -> str:
        return "-- Cypher export template\n"

    def _get_graphml_template(self) -> str:
        return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<graphml></graphml>"

    def _get_agent_state_template(self) -> str:
        return json.dumps({
            "agent_id": "{{AGENT_ID}}",
            "state": "ready",
            "timestamp": "{{TIMESTAMP}}"
        }, indent=2)

    def _get_system_state(self) -> dict:
        return {
            "system": "Project Brain",
            "version": "1.0.0",
            "status": "initialized",
            "timestamp": self.timestamp
        }

    # ---------- README ----------

    def _get_projects_readme(self) -> str:
        return "# Proyectos\n\nProyectos analizados por Project Brain.\n"

    def _get_embeddings_readme(self) -> str:
        return "# Embeddings\n\nBase de datos vectorial del sistema.\n"

    def _get_graph_exports_readme(self) -> str:
        return "# Exportaciones de Grafo\n\nFormatos Cypher y GraphML.\n"

    def _get_cache_readme(self) -> str:
        return "# Caché\n\nCaché persistente en disco.\n"

    def _get_state_readme(self) -> str:
        return "# Estado\n\nEstado persistente del sistema.\n"

    def _get_backups_readme(self) -> str:
        return "# Backups\n\nBackups automáticos del sistema.\n"


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
