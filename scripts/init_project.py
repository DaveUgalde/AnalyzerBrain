# scripts/init_project.py
"""
Script de inicialización del sistema Project Brain.
Configura bases de datos, valida dependencias y prepara el entorno.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# -------------------------------------------------
# Ajuste robusto del path del proyecto
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config_manager import ConfigManager
from core.orchestrator import BrainOrchestrator
from core.exceptions import ConfigurationError
from utils.logging_config import setup_logging

import yaml
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import redis
from neo4j import GraphDatabase
import chromadb


class SystemInitializer:
    """Inicializa todos los componentes del sistema Project Brain."""

    def __init__(self, config_path: Optional[str] = None, verbose: bool = False):
        self.config_path = Path(config_path or BASE_DIR / "config/system.yaml")
        self.verbose = verbose
        self.config: Optional[Dict[str, Any]] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    # -------------------------------------------------
    # ORQUESTACIÓN PRINCIPAL
    # -------------------------------------------------

    def initialize(self, skip_db: bool = False) -> bool:
        try:
            self._ensure_base_directories()
            self._setup_logging()

            self.logger.info("🚀 Iniciando inicialización del sistema Project Brain")

            self._load_configuration()
            self._setup_directories()

            if not skip_db:
                self._initialize_databases()
            else:
                self.logger.warning("⏭️ Inicialización de bases de datos omitida")

            self._check_external_dependencies()
            self._create_initial_data_structure()
            self._test_basic_orchestrator()

            self.logger.info("✅ Inicialización completada exitosamente")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error crítico durante la inicialización: {e}")
            return False

    # -------------------------------------------------
    # LOGGING & CONFIG
    # -------------------------------------------------

    def _ensure_base_directories(self) -> None:
        (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        log_config = {
            "level": "DEBUG" if self.verbose else os.getenv("LOG_LEVEL", "INFO"),
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "file": BASE_DIR / "logs/init_system.log",
        }
        setup_logging(log_config)

    def _load_configuration(self) -> None:
        manager = ConfigManager(str(self.config_path))
        self.config = manager.get_config()

        for section in ("system", "databases", "projects"):
            if section not in self.config:
                raise ConfigurationError(f"Falta sección requerida: {section}")

        self.logger.info(f"📄 Configuración cargada desde {self.config_path}")

    # -------------------------------------------------
    # DIRECTORIOS
    # -------------------------------------------------

    def _setup_directories(self) -> None:
        directories = [
            BASE_DIR / "data/projects",
            BASE_DIR / "data/embeddings",
            BASE_DIR / "data/graph_exports",
            BASE_DIR / "data/cache",
            BASE_DIR / "data/state",
            BASE_DIR / "data/backups",
            BASE_DIR / "logs",
            BASE_DIR / "config",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # BASES DE DATOS
    # -------------------------------------------------

    def _initialize_databases(self) -> None:
        dbs = self.config["databases"]

        if dbs.get("postgresql", {}).get("enabled", True):
            self._init_postgresql()

        if dbs.get("neo4j", {}).get("enabled", True):
            self._init_neo4j()

        if dbs.get("redis", {}).get("enabled", True):
            self._init_redis()

        if dbs.get("chromadb", {}).get("enabled", True):
            self._init_chromadb()

    # ---------- PostgreSQL ----------

    def _init_postgresql(self) -> None:
        cfg = self.config["databases"]["postgresql"]

        conn = psycopg2.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["username"],
            password=os.getenv("DB_PASSWORD", cfg.get("password")),
            database=cfg["database"],
        )
        conn.close()
        self.logger.info("✅ PostgreSQL conectado")

        init_script = BASE_DIR / "scripts/init-db.sql"
        if init_script.exists():
            self._run_postgres_init_script(init_script, cfg)

    def _run_postgres_init_script(self, script_path: Path, cfg: Dict[str, Any]) -> None:
        with psycopg2.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["username"],
            password=os.getenv("DB_PASSWORD", cfg.get("password")),
            database=cfg["database"],
        ) as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute(script_path.read_text())
        self.logger.info("📦 PostgreSQL schema inicializado")

    # ---------- Neo4j ----------

    def _init_neo4j(self) -> None:
        cfg = self.config["databases"]["neo4j"]

        driver = GraphDatabase.driver(
            cfg["uri"],
            auth=(cfg["username"], os.getenv("NEO4J_PASSWORD", cfg.get("password"))),
        )

        with driver.session() as session:
            session.run("RETURN 1")

        self._create_neo4j_indexes(driver)
        driver.close()
        self.logger.info("✅ Neo4j conectado")

    def _create_neo4j_indexes(self, driver) -> None:
        queries = [
            "CREATE INDEX project_id IF NOT EXISTS FOR (p:Project) ON (p.id)",
            "CREATE INDEX file_id IF NOT EXISTS FOR (f:File) ON (f.id)",
            "CREATE INDEX function_id IF NOT EXISTS FOR (f:Function) ON (f.id)",
            "CREATE INDEX class_id IF NOT EXISTS FOR (c:Class) ON (c.id)",
        ]
        with driver.session() as session:
            for q in queries:
                session.run(q)

    # ---------- Redis ----------

    def _init_redis(self) -> None:
        cfg = self.config["databases"]["redis"]

        r = redis.Redis(
            host=cfg["host"],
            port=cfg["port"],
            password=os.getenv("REDIS_PASSWORD", cfg.get("password")),
            db=cfg.get("db", 0),
            socket_timeout=5,
        )
        r.ping()
        self.logger.info("✅ Redis conectado")

    # ---------- ChromaDB ----------

    def _init_chromadb(self) -> None:
        cfg = self.config["databases"]["chromadb"]

        client = chromadb.PersistentClient(path=cfg["persist_directory"])
        name = cfg["collection_name"]

        try:
            client.get_collection(name)
            self.logger.info(f"📦 ChromaDB colección '{name}' existente")
        except Exception:
            client.create_collection(
                name=name,
                metadata={"hnsw:space": cfg.get("similarity_metric", "cosine")},
            )
            self.logger.info(f"📦 ChromaDB colección '{name}' creada")

    # -------------------------------------------------
    # OTROS
    # -------------------------------------------------

    def _check_external_dependencies(self) -> None:
        for pkg in (
            "tree_sitter",
            "transformers",
            "sentence_transformers",
            "networkx",
            "fastapi",
            "uvicorn",
        ):
            try:
                __import__(pkg)
                self.logger.debug(f"✔ {pkg}")
            except ImportError:
                self.logger.warning(f"⚠ Dependencia faltante: {pkg}")

    def _create_initial_data_structure(self) -> None:
        example_cfg = BASE_DIR / "config/system.yaml.example"
        if not example_cfg.exists():
            self._create_example_config(example_cfg)

        env_example = BASE_DIR / ".env.example"
        if not env_example.exists():
            self._create_env_example(env_example)

    def _create_example_config(self, path: Path) -> None:
        example = {
            "system": {"name": "Project Brain", "environment": "development"},
            "projects": {},
            "databases": {
                "postgresql": {"enabled": True},
                "neo4j": {"enabled": True},
                "redis": {"enabled": True},
                "chromadb": {"enabled": True},
            },
        }
        path.write_text(yaml.dump(example, sort_keys=False))

    def _create_env_example(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "DB_PASSWORD=changeme",
                    "NEO4J_PASSWORD=changeme",
                    "REDIS_PASSWORD=changeme",
                    "JWT_SECRET=changeme",
                    "DEBUG=false",
                ]
            )
        )

    def _test_basic_orchestrator(self) -> None:
        BrainOrchestrator(str(self.config_path))
        self.logger.info("🧠 BrainOrchestrator instanciado correctamente")

    # -------------------------------------------------

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("PROJECT BRAIN — SISTEMA LISTO")
        print("=" * 60)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Inicializa Project Brain")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-db", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    initializer = SystemInitializer(args.config, verbose=args.verbose)
    ok = initializer.initialize(skip_db=args.skip_db)

    if ok:
        initializer.print_summary()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
