"""
Script para migrar datos entre versiones del sistema Project Brain.
⚠️ Este script modifica datos en disco. Usar con precaución.
"""

import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Callable

# -------------------------------------------------
# Logging
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("DataMigrator")


class DataMigrator:
    """Migrador de datos entre versiones."""

    VERSION_CHAIN: List[str] = [
        "1.0.0",
        "1.1.0",
        "1.2.0",
        "2.0.0",
    ]

    LATEST_VERSION = "2.0.0"

    # -------------------------------------------------

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir.resolve()
        self.migration_log = self.data_dir / "migration_log.json"

        self.migrations: dict[tuple[str, str], Callable[[], None]] = {
            ("1.0.0", "1.1.0"): self._migrate_1_0_0_to_1_1_0,
            ("1.1.0", "1.2.0"): self._migrate_1_1_0_to_1_2_0,
            ("1.2.0", "2.0.0"): self._migrate_1_2_0_to_2_0_0,
        }

    # =================================================
    # VERSION
    # =================================================

    def detect_version(self) -> str:
        version_file = self.data_dir / "version.json"

        if version_file.exists():
            return self._read_version_file(version_file)

        if (self.data_dir / "projects").exists():
            return "1.0.0"

        return "unknown"

    def _read_version_file(self, path: Path) -> str:
        try:
            return json.loads(path.read_text()).get("version", "1.0.0")
        except Exception:
            logger.warning("version.json corrupto, asumiendo 1.0.0")
            return "1.0.0"

    # =================================================
    # MIGRATION ORCHESTRATION
    # =================================================

    def migrate_to_latest(self) -> bool:
        current = self.detect_version()

        if not self._validate_start_version(current):
            return False

        if current == self.LATEST_VERSION:
            logger.info("✓ Datos ya están en la última versión")
            return True

        logger.info("Migrando datos %s → %s", current, self.LATEST_VERSION)

        backup = self._create_backup()

        try:
            self._run_migration_chain(current)
            self._update_version_file(self.LATEST_VERSION)
            self._log_migration(current, self.LATEST_VERSION, success=True)

            logger.info("✅ Migración completada exitosamente")
            return True

        except Exception as e:
            logger.error("❌ Migración fallida: %s", e)
            self._rollback(backup)
            self._log_migration(current, self.LATEST_VERSION, success=False, error=str(e))
            return False

    # -------------------------------------------------

    def _validate_start_version(self, version: str) -> bool:
        if version == "unknown":
            logger.error("No se pudo detectar versión")
            return False

        if version not in self.VERSION_CHAIN:
            logger.error("Versión no soportada: %s", version)
            return False

        return True

    def _run_migration_chain(self, start_version: str) -> None:
        start = self.VERSION_CHAIN.index(start_version)
        end = self.VERSION_CHAIN.index(self.LATEST_VERSION)

        for i in range(start, end):
            from_v = self.VERSION_CHAIN[i]
            to_v = self.VERSION_CHAIN[i + 1]

            logger.info("→ Migración %s → %s", from_v, to_v)
            self._run_single_migration(from_v, to_v)

    def _run_single_migration(self, from_v: str, to_v: str) -> None:
        migration = self.migrations.get((from_v, to_v))
        if not migration:
            raise RuntimeError(f"No existe migración {from_v} → {to_v}")

        migration()

    # =================================================
    # MIGRATIONS
    # =================================================

    def _migrate_1_0_0_to_1_1_0(self) -> None:
        cache_dir = self.data_dir / "embeddings/cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        legacy = self.data_dir / "embeddings.json"
        if not legacy.exists():
            return

        data = json.loads(legacy.read_text())

        for key, value in data.items():
            payload = (
                value
                if isinstance(value, dict)
                else {"vector": value, "metadata": {}, "version": "1.0.0"}
            )
            (cache_dir / f"{key}.json").write_text(json.dumps(payload, indent=2))

        legacy.rename(legacy.with_suffix(".json.bak"))

    def _migrate_1_1_0_to_1_2_0(self) -> None:
        import sqlite3

        state_dir = self.data_dir / "state"
        state_dir.mkdir(exist_ok=True)

        db_path = state_dir / "state.db"
        conn = sqlite3.connect(db_path)

        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS migration_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

        old_sessions = self.data_dir / "sessions"
        if old_sessions.exists():
            for f in old_sessions.glob("*.json"):
                conn.execute(
                    "INSERT OR IGNORE INTO migration_state VALUES (?, ?)",
                    (f"session_{f.stem}", f.read_text()),
                )

        conn.close()

    def _migrate_1_2_0_to_2_0_0(self) -> None:
        temp = self.data_dir / "_migration_tmp"
        temp.mkdir(exist_ok=True)

        for item in self.data_dir.iterdir():
            if item.name not in {
                "_migration_tmp",
                "version.json",
                "migration_log.json",
            }:
                shutil.move(str(item), temp / item.name)

        sys.path.insert(0, str(self.data_dir.parent / "src"))
        from data.init_data_structure import DataManager

        DataManager(str(self.data_dir))

        for folder in ("projects", "embeddings", "state"):
            src = temp / folder
            dst = self.data_dir / folder
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)

        shutil.rmtree(temp)

    # =================================================
    # BACKUP & LOG
    # =================================================

    def _create_backup(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = self.data_dir.parent / f"data_backup_{ts}"

        logger.info("Creando backup en %s", backup)
        shutil.copytree(self.data_dir, backup)
        return backup

    def _rollback(self, backup: Path) -> None:
        logger.warning("Restaurando backup…")
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        shutil.copytree(backup, self.data_dir)
        logger.info("✓ Backup restaurado")

    def _update_version_file(self, version: str) -> None:
        payload = {
            "version": version,
            "migrated_at": datetime.now().isoformat(),
        }
        (self.data_dir / "version.json").write_text(json.dumps(payload, indent=2))

    def _log_migration(
        self,
        from_version: str,
        to_version: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        entry = {
            "from": from_version,
            "to": to_version,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "error": error,
        }

        log = []
        if self.migration_log.exists():
            log = json.loads(self.migration_log.read_text())

        log.append(entry)
        self.migration_log.write_text(json.dumps(log[-100:], indent=2))


# -------------------------------------------------
# CLI
# -------------------------------------------------

def main() -> int:
    print("=== MIGRADOR DE DATOS PROJECT BRAIN ===")

    data_dir = Path("./data").resolve()
    if not data_dir.exists():
        logger.error("Directorio de datos no encontrado")
        return 1

    migrator = DataMigrator(data_dir)
    logger.info("Versión detectada: %s", migrator.detect_version())

    return 0 if migrator.migrate_to_latest() else 1


if __name__ == "__main__":
    raise SystemExit(main())
