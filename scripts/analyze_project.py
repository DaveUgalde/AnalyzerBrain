"""
Script para análisis de proyectos en Project Brain.
Permite analizar proyectos completos, múltiples proyectos o archivos individuales.
"""

import sys
import argparse
import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# -------------------------------------------------------------------
# PYTHONPATH
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -------------------------------------------------------------------
# Imports core
# -------------------------------------------------------------------

from core.orchestrator import (
    BrainOrchestrator,
    OperationPriority,
    OperationRequest,
)
from core.config_manager import ConfigManager
from core.exceptions import BrainException
from utils.logging_config import setup_logging
from utils.file_utils import read_file_safely, find_files

# -------------------------------------------------------------------
# ProjectAnalyzer
# -------------------------------------------------------------------

class ProjectAnalyzer:
    """Maneja análisis de proyectos y archivos."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(BASE_DIR / "config" / "system.yaml")
        self.config: Optional[Dict[str, Any]] = None
        self.orchestrator: Optional[BrainOrchestrator] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    async def __aenter__(self):
        if not await self.initialize():
            raise RuntimeError("No se pudo inicializar el analizador")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        try:
            (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)

            setup_logging({
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": str(BASE_DIR / "logs" / "analyze_project.log"),
            })

            self.config = ConfigManager(self.config_path).get_config()
            self.orchestrator = BrainOrchestrator(self.config_path)
            await self.orchestrator.initialize()

            self.logger.info("✅ Analizador inicializado")
            return True

        except Exception:
            self.logger.exception("❌ Error inicializando analizador")
            return False

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    async def analyze_project(
        self,
        project_path: str,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:

        project_dir = Path(project_path)
        if not project_dir.exists():
            raise FileNotFoundError(project_path)

        if not self.orchestrator:
            raise RuntimeError("Orchestrator no inicializado")

        self.logger.info("📁 Analizando proyecto: %s", project_dir)
        start = datetime.now()

        result = await self.orchestrator.analyze_project(
            project_path=str(project_dir),
            options=options,
        )

        elapsed = (datetime.now() - start).total_seconds()
        self.logger.info("✅ Análisis completado en %.2fs", elapsed)

        return result

    async def analyze_file(self, file_path: str) -> Dict[str, Any]:
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(file_path)

        if not self.orchestrator:
            raise RuntimeError("Orchestrator no inicializado")

        language = self._detect_language(file_path)
        content = read_file_safely(file_path)

        request = OperationRequest(
            operation_type="analyze_file",
            priority=OperationPriority.HIGH,
            context={
                "file_path": str(file),
                "content": content,
                "language": language,
            },
        )

        result = await self.orchestrator.process_operation(request)

        if not result.success:
            raise BrainException(result.error or "Análisis fallido")

        return result.data

    # ------------------------------------------------------------------

    @staticmethod
    def _detect_language(file_path: str) -> str:
        return {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".go": "go",
            ".rs": "rust",
        }.get(Path(file_path).suffix.lower(), "text")

    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        if self.orchestrator:
            await self.orchestrator.shutdown()
            self.logger.info("🔌 Analizador apagado")

    # ------------------------------------------------------------------

    def print_analysis_summary(self, result: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("ANÁLISIS COMPLETADO")
        print("=" * 60)

        if not result:
            print("⚠ No hay resultados")
            return

        for key, value in result.items():
            if isinstance(value, (int, float, str)):
                print(f"{key}: {value}")

# =====================================================================
# Helpers CLI
# =====================================================================

def build_project_options(args) -> Dict[str, Any]:
    return {
        "mode": args.mode or "comprehensive",
        "include_tests": not args.no_tests,
        "include_docs": not args.no_docs,
        "timeout_minutes": args.timeout or 30,
    }

def write_report(path: str, data: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, default=str))
    print(f"📄 Reporte guardado en {path}")

# =====================================================================
# CLI commands
# =====================================================================

async def cmd_project(args) -> int:
    async with ProjectAnalyzer(args.config) as analyzer:
        result = await analyzer.analyze_project(
            args.project_path,
            build_project_options(args),
        )

        analyzer.print_analysis_summary(result)

        if args.output:
            write_report(args.output, result)

    return 0

async def cmd_multi(args) -> int:
    projects = [
        p for p in Path(args.projects_file).read_text().splitlines() if p.strip()
    ]

    results = []

    async with ProjectAnalyzer(args.config) as analyzer:
        for project in projects:
            try:
                await analyzer.analyze_project(
                    project, build_project_options(args)
                )
                results.append({"project": project, "success": True})
            except Exception as exc:
                results.append({
                    "project": project,
                    "success": False,
                    "error": str(exc),
                })

    report = {
        "timestamp": datetime.now().isoformat(),
        "total": len(projects),
        "successful": sum(r["success"] for r in results),
        "failed": sum(not r["success"] for r in results),
        "results": results,
    }

    write_report("analysis_report.json", report)
    return 0

async def cmd_directory(args) -> int:
    files = find_files(
        args.directory,
        extensions=args.extensions.split(",") if args.extensions else None,
        recursive=args.recursive,
    )

    results = []

    async with ProjectAnalyzer(args.config) as analyzer:
        for file in files:
            try:
                await analyzer.analyze_file(file)
                results.append({"file": file, "success": True})
            except Exception as exc:
                results.append({
                    "file": file,
                    "success": False,
                    "error": str(exc),
                })

    report = {
        "directory": args.directory,
        "timestamp": datetime.now().isoformat(),
        "total_files": len(files),
        "results": results,
    }

    write_report("directory_analysis_report.json", report)
    return 0

# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser("Analiza proyectos con Project Brain")
    parser.add_argument("--config")

    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("project")
    p.add_argument("project_path")
    p.add_argument("--mode")
    p.add_argument("--no-tests", action="store_true")
    p.add_argument("--no-docs", action="store_true")
    p.add_argument("--timeout", type=int)
    p.add_argument("--output")

    m = sub.add_parser("multi")
    m.add_argument("projects_file")

    d = sub.add_parser("directory")
    d.add_argument("directory")
    d.add_argument("--extensions")
    d.add_argument("--recursive", action="store_true")

    args = parser.parse_args()

    commands = {
        "project": lambda: asyncio.run(cmd_project(args)),
        "multi": lambda: asyncio.run(cmd_multi(args)),
        "directory": lambda: asyncio.run(cmd_directory(args)),
    }

    if args.command not in commands:
        parser.print_help()
        return 1

    return commands[args.command]()

if __name__ == "__main__":
    raise SystemExit(main())
