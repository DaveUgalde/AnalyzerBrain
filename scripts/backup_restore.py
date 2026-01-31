"""
Script para backup y restauración de Project Brain.
"""

import sys
import argparse
import logging
import json
import shutil
import tarfile
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -------------------------------------------------------------------
# Imports internos
# -------------------------------------------------------------------

from core.config_manager import ConfigManager
from core.exceptions import BrainException
from utils.logging_config import setup_logging

# -------------------------------------------------------------------
# Enums
# -------------------------------------------------------------------

class BackupType(Enum):
    FULL = "full"
    KNOWLEDGE = "knowledge"
    CONFIG = "config"
    DATABASE = "database"

# -------------------------------------------------------------------
# BackupManager
# -------------------------------------------------------------------

class BackupManager:
    """Gestor de backup y restauración."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(BASE_DIR / "config" / "system.yaml")
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.backup_dir: Optional[Path] = None

    # ----------------------------
    # Init
    # ----------------------------

    def initialize(self) -> bool:
        try:
            (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)

            setup_logging({
                "level": "INFO",
                "file": str(BASE_DIR / "logs" / "backup_restore.log"),
            })

            self.config = ConfigManager(self.config_path).get_config()

            self.backup_dir = Path(
                self.config.get("backup", {}).get(
                    "location",
                    BASE_DIR / "data" / "backups",
                )
            )
            self.backup_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("✅ Gestor de backup inicializado")
            return True

        except Exception:
            self.logger.exception("❌ Error inicializando gestor de backup")
            return False

    # ===============================================================
    # BACKUP
    # ===============================================================

    def create_backup(
        self,
        backup_type: BackupType,
        output_path: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        include = include or self._default_includes(backup_type)
        exclude_set: Set[str] = set(exclude or [])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = self._resolve_output_path(output_path, backup_type, timestamp)
        temp_dir = self._create_temp_dir(timestamp)

        metadata = self._init_metadata(backup_type, include, exclude_set)

        try:
            self._collect_files(include, exclude_set, temp_dir, metadata)
            self._write_metadata(temp_dir, metadata)
            self._pack_backup(temp_dir, output)
            self._finalize_metadata(metadata, output)

            self.logger.info("✅ Backup creado: %s", output)
            return metadata

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ----------------------------

    def _resolve_output_path(
        self,
        output_path: Optional[str],
        backup_type: BackupType,
        timestamp: str,
    ) -> Path:
        if output_path:
            return Path(output_path)

        assert self.backup_dir is not None
        return self.backup_dir / f"backup_{backup_type.value}_{timestamp}.tar.gz"

    def _create_temp_dir(self, timestamp: str) -> Path:
        temp_dir = BASE_DIR / ".tmp" / f"backup_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def _init_metadata(
        self,
        backup_type: BackupType,
        include: List[str],
        exclude: Set[str],
    ) -> Dict[str, Any]:
        return {
            "id": f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "type": backup_type.value,
            "timestamp": datetime.now().isoformat(),
            "included": include,
            "excluded": list(exclude),
            "files": [],
            "checksums": {},
        }

    def _collect_files(
        self,
        include: List[str],
        exclude: Set[str],
        temp_dir: Path,
        metadata: Dict[str, Any],
    ) -> None:
        for item in include:
            self._backup_item(Path(item), temp_dir, metadata, exclude)

    def _write_metadata(self, temp_dir: Path, metadata: Dict[str, Any]) -> None:
        (temp_dir / "backup_metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

    def _pack_backup(self, temp_dir: Path, output: Path) -> None:
        with tarfile.open(output, "w:gz") as tar:
            tar.add(temp_dir, arcname="backup")

    def _finalize_metadata(self, metadata: Dict[str, Any], output: Path) -> None:
        metadata["checksums"]["sha256"] = self._sha256(output)
        metadata["file_size_bytes"] = output.stat().st_size
        metadata["output_path"] = str(output)

    # ===============================================================
    # RESTORE
    # ===============================================================

    def restore_backup(
        self,
        backup_path: str,
        restore_path: Optional[str] = None,
        items: Optional[List[str]] = None,
        verify: bool = True,
    ) -> Dict[str, Any]:

        backup_file = self._validate_backup_file(backup_path)
        restore_dir = self._resolve_restore_dir(restore_path)
        metadata = self._load_backup_metadata(backup_file)

        if not metadata:
            raise BrainException("Metadata de backup no encontrada")

        if verify:
            self._verify_checksum(backup_file, metadata)

        restored = self._extract_backup(
            backup_file,
            restore_dir,
            items,
        )

        return {
            "backup_id": metadata.get("id"),
            "restore_path": str(restore_dir),
            "restored_items": restored,
        }

    # ----------------------------

    def _validate_backup_file(self, backup_path: str) -> Path:
        backup_file = Path(backup_path)
        if not backup_file.exists():
            raise FileNotFoundError(backup_path)
        return backup_file

    def _resolve_restore_dir(self, restore_path: Optional[str]) -> Path:
        restore_dir = Path(restore_path) if restore_path else BASE_DIR
        restore_dir.mkdir(parents=True, exist_ok=True)
        return restore_dir

    def _load_backup_metadata(self, backup_file: Path) -> Dict[str, Any]:
        with tarfile.open(backup_file, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name.endswith("backup_metadata.json"):
                    f = tar.extractfile(m)
                    if f:
                        return json.load(f)
        return {}

    def _verify_checksum(self, backup_file: Path, metadata: Dict[str, Any]) -> None:
        expected = metadata.get("checksums", {}).get("sha256")
        if not expected:
            return

        actual = self._sha256(backup_file)
        if expected != actual:
            raise BrainException("Checksum inválido")

    def _extract_backup(
        self,
        backup_file: Path,
        restore_dir: Path,
        items: Optional[List[str]],
    ) -> List[Dict[str, Any]]:

        restored: List[Dict[str, Any]] = []

        with tarfile.open(backup_file, "r:gz") as tar:
            for member in tar.getmembers():
                self._safe_extract_member(
                    tar,
                    member,
                    restore_dir,
                    items,
                    restored,
                )
        return restored

    # ===============================================================
    # Helpers
    # ===============================================================

    def _default_includes(self, backup_type: BackupType) -> List[str]:
        if backup_type == BackupType.CONFIG:
            return [str(BASE_DIR / "config")]
        if backup_type == BackupType.KNOWLEDGE:
            return [str(BASE_DIR / "data" / "projects")]
        return [
            str(BASE_DIR / "config"),
            str(BASE_DIR / "data"),
            str(BASE_DIR / "logs"),
        ]

    def _backup_item(
        self,
        path: Path,
        temp_dir: Path,
        metadata: Dict[str, Any],
        exclude: Set[str],
    ) -> None:

        if not path.exists():
            self.logger.warning("No existe: %s", path)
            return

        if any(e in str(path) for e in exclude):
            return

        if path.is_dir():
            dest = temp_dir / path.name
            shutil.copytree(path, dest, dirs_exist_ok=True)
            for f in dest.rglob("*"):
                if f.is_file():
                    metadata["files"].append(str(f.relative_to(temp_dir)))
        else:
            dest = temp_dir / path.name
            shutil.copy2(path, dest)
            metadata["files"].append(str(dest.relative_to(temp_dir)))

    def _sha256(self, file: Path) -> str:
        h = hashlib.sha256()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _safe_extract_member(
        self,
        tar: tarfile.TarFile,
        member: tarfile.TarInfo,
        target_dir: Path,
        items: Optional[List[str]],
        restored: List[Dict[str, Any]],
    ) -> None:

        member_path = Path(member.name)
        resolved = (target_dir / member_path).resolve()

        if not str(resolved).startswith(str(target_dir.resolve())):
            return

        if items and not any(i in member.name for i in items):
            return

        tar.extract(member, target_dir)

        if member.isfile():
            restored.append({
                "path": str(resolved),
                "size_bytes": member.size,
            })

    # ===============================================================
    # Listing
    # ===============================================================

    def list_backups(self) -> List[Dict[str, Any]]:
        assert self.backup_dir is not None
        backups = []

        for f in sorted(self.backup_dir.glob("backup_*.tar.gz")):
            backups.append({
                "file": str(f),
                "size_bytes": f.stat().st_size,
                "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })

        return backups

# =====================================================================
# CLI
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser("Backup y restauración Project Brain")
    parser.add_argument("--config", default=None)

    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create")
    c.add_argument("--type", choices=[t.value for t in BackupType], default="full")
    c.add_argument("--output")

    r = sub.add_parser("restore")
    r.add_argument("--backup-path", required=True)
    r.add_argument("--restore-path")

    sub.add_parser("list")

    args = parser.parse_args()
    mgr = BackupManager(args.config)

    if not mgr.initialize():
        return 1

    commands = {
        "create": lambda: mgr.create_backup(
            BackupType(args.type),
            args.output,
        ),
        "restore": lambda: mgr.restore_backup(
            args.backup_path,
            args.restore_path,
        ),
        "list": lambda: print(json.dumps(mgr.list_backups(), indent=2)),
    }

    commands[args.cmd]()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
