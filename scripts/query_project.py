"""
Script para consultar proyectos en Project Brain desde la línea de comandos.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio
import json
from datetime import datetime
import uuid

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data" / "projects"

sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------
# Imports internos
# ---------------------------------------------------------------------
from core.orchestrator import BrainOrchestrator
from core.config_manager import ConfigManager
from utils.logging_config import setup_logging


# ---------------------------------------------------------------------
# ProjectQuery
# ---------------------------------------------------------------------
class ProjectQuery:
    """Maneja consultas sobre proyectos."""

    DEFAULT_OPTIONS = {
        "detail_level": "normal",
        "include_code": True,
        "include_explanations": True,
        "include_sources": True,
    }

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.orchestrator: Optional[BrainOrchestrator] = None
        self.session_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)

    # ----------------------------
    # Lifecycle
    # ----------------------------

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

    async def initialize(self) -> None:
        setup_logging({
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": str(BASE_DIR / "logs" / "query_project.log"),
        })

        ConfigManager(self.config_path).get_config()
        self.orchestrator = BrainOrchestrator(self.config_path)
        await self.orchestrator.initialize()

        self.logger.info("Sistema de consultas inicializado")

    async def shutdown(self) -> None:
        if self.orchestrator:
            await self.orchestrator.shutdown()
            self.logger.info("Sistema de consultas apagado")

    # ----------------------------
    # Queries
    # ----------------------------

    async def ask(
        self,
        question: str,
        project_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        query_context = {
            "session_id": self.session_id,
            **(context or {}),
            **(options or self.DEFAULT_OPTIONS),
        }

        start = datetime.utcnow()
        result = await self.orchestrator.ask_question(
            question=question,
            project_id=project_id,
            context=query_context,
        )

        elapsed = (datetime.utcnow() - start).total_seconds()
        self.logger.info(
            "Pregunta procesada en %.2fs | Confianza %.1f%%",
            elapsed,
            result.get("confidence", 0) * 100,
        )
        return result

    # ----------------------------
    # Projects
    # ----------------------------

    async def list_projects(self) -> List[Dict[str, Any]]:
        if not DATA_DIR.exists():
            return []

        projects = [
            self._load_project_metadata(p)
            for p in DATA_DIR.iterdir()
            if p.is_dir()
        ]

        return sorted(
            projects,
            key=lambda p: p.get("last_analyzed") or "",
            reverse=True,
        )

    def _load_project_metadata(self, project_dir: Path) -> Dict[str, Any]:
        metadata = {}
        metadata_file = project_dir / "metadata.json"

        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
            except Exception:
                pass

        return {
            "id": project_dir.name,
            "name": metadata.get("name", project_dir.name),
            "path": str(project_dir),
            "language": metadata.get("language", "unknown"),
            "last_analyzed": metadata.get("last_analyzed"),
            "file_count": metadata.get("file_count", 0),
        }

    # ----------------------------
    # Formatting
    # ----------------------------

    def format_answer(self, answer: Dict[str, Any], verbose: bool) -> str:
        sections = [
            self._format_text(answer),
            self._format_confidence(answer),
        ]

        if verbose:
            sections.extend([
                self._format_sources(answer),
                self._format_reasoning(answer),
            ])

        return "\n".join(s for s in sections if s)

    def _format_text(self, answer: Dict[str, Any]) -> str:
        text = answer.get("answer", {}).get("text")
        if not text:
            return ""

        return "\n".join([
            "\n📝 RESPUESTA",
            "=" * 60,
            text,
            "=" * 60,
        ])

    def _format_confidence(self, answer: Dict[str, Any]) -> str:
        confidence = answer.get("confidence", 0)
        level = "Alta" if confidence > 0.8 else "Media" if confidence > 0.5 else "Baja"
        return f"\n📊 Confianza: {confidence:.1%} ({level})"

    def _format_sources(self, answer: Dict[str, Any]) -> str:
        sources = answer.get("answer", {}).get("sources", [])
        if not sources:
            return ""

        lines = ["\n📚 Fuentes:"]
        for i, src in enumerate(sources[:3], 1):
            lines.append(f"  {i}. {src.get('file_path', 'N/A')}")
        return "\n".join(lines)

    def _format_reasoning(self, answer: Dict[str, Any]) -> str:
        reasoning = answer.get("answer", {}).get("reasoning", [])
        if not reasoning:
            return ""

        lines = ["\n🤔 Razonamiento:"]
        for i, step in enumerate(reasoning[:5], 1):
            lines.append(f"  {i}. {step}")
        return "\n".join(lines)

    # ----------------------------
    # Interactive
    # ----------------------------

    async def interactive(self, project_id: Optional[str]) -> None:
        print("\n🔄 MODO INTERACTIVO - PROJECT BRAIN")
        print("=" * 60)

        history: List[Dict[str, Any]] = []

        while True:
            try:
                question = input("\n❓ Pregunta: ").strip()
                if question.lower() in {"exit", "quit", "salir"}:
                    break
                if not question:
                    continue

                context = {"conversation_history": history[-5:]}
                print("🧠 Procesando...", end="", flush=True)

                answer = await self.ask(question, project_id, context)
                print("\r" + " " * 40 + "\r", end="")
                print(self.format_answer(answer, verbose=True))

                history.append({
                    "question": question,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            except KeyboardInterrupt:
                break


# ---------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------

async def with_query(config: str, fn):
    async with ProjectQuery(config) as query:
        return await fn(query)


async def handle_ask(args) -> int:
    async def run(query: ProjectQuery):
        context = json.loads(Path(args.file).read_text()) if args.file else {}
        answer = await query.ask(args.question, args.project, context)
        print(query.format_answer(answer, args.verbose))

        if args.output:
            Path(args.output).write_text(json.dumps(answer, indent=2))
            print(f"\n💾 Respuesta guardada en {args.output}")
        return 0

    return await with_query(args.config, run)


async def handle_interactive(args) -> int:
    return await with_query(
        args.config,
        lambda q: q.interactive(args.project),
    )


async def handle_projects(args) -> int:
    async def run(query: ProjectQuery):
        projects = await query.list_projects()
        if not projects:
            print("No hay proyectos disponibles")
            return 0

        print("\n📁 Proyectos disponibles")
        print("=" * 80)
        for p in projects:
            print(f"{p['id']:36} {p['language']:10} {p['file_count']:5}")
        return 0

    return await with_query(args.config, run)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser("Consulta proyectos en Project Brain")
    parser.add_argument("--config", default=str(BASE_DIR / "config" / "system.yaml"))

    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--project")
    ask.add_argument("--file")
    ask.add_argument("--output")
    ask.add_argument("--verbose", action="store_true")

    inter = sub.add_parser("interactive")
    inter.add_argument("--project")

    sub.add_parser("projects")

    args = parser.parse_args()

    handlers = {
        "ask": handle_ask,
        "interactive": handle_interactive,
        "projects": handle_projects,
    }

    return asyncio.run(handlers[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
