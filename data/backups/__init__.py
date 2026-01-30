# data/backups/__init__.py
"""
Sistema de backups automáticos para Project Brain.
"""

from __future__ import annotations

import json
import shutil
import zipfile
import tarfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading
import time

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from ...core.exceptions import BackupException


class BackupManager:
    """Gestor de backups automáticos."""

    def __init__(self, backup_path: Path, config: Optional[Dict[str, Any]] = None):
        self.backup_path = backup_path
        self.backup_path.mkdir(parents=True, exist_ok=True)

        self.config = config or {
            "enabled": True,
            "schedule": "daily",
            "retention_days": 30,
            "compression": True,
            "encryption": False,
            "verify_integrity": True,
            "backup_sources": ["projects", "embeddings", "state", "config"],
        }

        self._init_directories()
        self._init_encryption()
        self._load_metadata()

        if self.config["enabled"] and SCHEDULE_AVAILABLE:
            self._start_scheduler()

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def _init_directories(self) -> None:
        for name in ("full", "incremental", "logs", "temp", "restore_points"):
            (self.backup_path / name).mkdir(parents=True, exist_ok=True)

    def _init_encryption(self) -> None:
        self.cipher: Optional[Fernet] = None

        if self.config["encryption"] and CRYPTO_AVAILABLE:
            key_file = self.backup_path / "encryption.key"
            if key_file.exists():
                key = key_file.read_bytes()
            else:
                key = Fernet.generate_key()
                key_file.write_bytes(key)

            self.cipher = Fernet(key)

    def _load_metadata(self) -> None:
        self.metadata_file = self.backup_path / "backup_metadata.json"
        self.metadata = {
            "backups": [],
            "last_backup": None,
            "total_backups": 0,
            "successful_backups": 0,
            "failed_backups": 0,
            "total_size": 0,
        }

        if self.metadata_file.exists():
            try:
                self.metadata.update(json.loads(self.metadata_file.read_text()))
            except Exception:
                pass

    def _save_metadata(self) -> None:
        self.metadata_file.write_text(json.dumps(self.metadata, indent=2))

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _start_scheduler(self) -> None:
        if self.config["schedule"] == "daily":
            schedule.every().day.at("02:00").do(self.create_scheduled_backup)
        elif self.config["schedule"] == "weekly":
            schedule.every().monday.at("02:00").do(self.create_scheduled_backup)
        elif self.config["schedule"] == "monthly":
            schedule.every(30).days.at("02:00").do(self.create_scheduled_backup)

        threading.Thread(target=self._run_scheduler, daemon=True).start()

    def _run_scheduler(self) -> None:
        while True:
            schedule.run_pending()
            time.sleep(60)

    # ------------------------------------------------------------------
    # Backups
    # ------------------------------------------------------------------

    def create_scheduled_backup(self) -> Dict[str, Any]:
        backup_type = "full" if self._should_create_full_backup() else "incremental"
        result = self.create_backup(backup_type, f"Scheduled {backup_type} backup")
        self.rotate_old_backups()
        return result

    def _should_create_full_backup(self) -> bool:
        for backup in reversed(self.metadata["backups"]):
            if backup["type"] == "full" and backup["status"] == "success":
                last_date = datetime.fromisoformat(backup["created_at"])
                return (datetime.now() - last_date).days >= 7
        return True

    def create_backup(
        self,
        backup_type: str = "incremental",
        description: str = "",
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        staging_dir = self.backup_path / "temp" / backup_id
        staging_dir.mkdir(parents=True, exist_ok=True)

        sources = sources or self.config["backup_sources"]

        info = {
            "id": backup_id,
            "type": backup_type,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "sources": sources,
            "files": [],
            "status": "in_progress",
        }

        try:
            base_backup = self._get_last_successful_backup() if backup_type == "incremental" else None
            if base_backup:
                info["base_backup"] = base_backup["id"]

            for source in sources:
                info["files"].extend(
                    self._backup_source(source, staging_dir, info)
                )

            archive = self._create_backup_archive(staging_dir, info)

            if self.config["verify_integrity"] and not self._verify_backup_integrity(archive):
                raise BackupException("Integrity check failed")

            shutil.rmtree(staging_dir)

            info.update(
                {
                    "status": "success",
                    "backup_file": str(archive),
                    "size": archive.stat().st_size,
                    "checksum": self._calculate_checksum(archive),
                }
            )

        except Exception as exc:
            info["status"] = "failed"
            info["error"] = str(exc)
            raise

        finally:
            self._register_backup(info)

        return info

    # ------------------------------------------------------------------
    # Fuentes
    # ------------------------------------------------------------------

    def _backup_source(
        self, source: str, backup_dir: Path, info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        src_dir = Path("data") / source
        if not src_dir.exists():
            return []

        if info["type"] == "full":
            return self._backup_full(source, src_dir, backup_dir)

        base = info.get("base_backup")
        return self._backup_incremental(source, src_dir, backup_dir, base)

    def _backup_full(
        self, source: str, src_dir: Path, backup_dir: Path
    ) -> List[Dict[str, Any]]:
        files = []
        for file in src_dir.rglob("*"):
            if file.is_file():
                rel = file.relative_to(src_dir)
                dest = backup_dir / source / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, dest)

                files.append(
                    {
                        "source": source,
                        "path": str(rel),
                        "size": file.stat().st_size,
                        "modified": file.stat().st_mtime,
                        "checksum": self._calculate_checksum(file),
                    }
                )
        return files

    def _backup_incremental(
        self,
        source: str,
        src_dir: Path,
        backup_dir: Path,
        base_backup_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        base_state = self._get_backup_state(base_backup_id, source)
        files = []

        for file in src_dir.rglob("*"):
            if not file.is_file():
                continue

            rel = str(file.relative_to(src_dir))
            checksum = self._calculate_checksum(file)
            mtime = file.stat().st_mtime

            prev = base_state.get(rel)
            if prev and prev["checksum"] == checksum and prev["modified"] >= mtime:
                continue

            dest = backup_dir / source / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)

            files.append(
                {
                    "source": source,
                    "path": rel,
                    "size": file.stat().st_size,
                    "modified": mtime,
                    "checksum": checksum,
                }
            )

        return files

    def _get_backup_state(self, backup_id: Optional[str], source: str) -> Dict[str, Any]:
        if not backup_id:
            return {}

        for backup in self.metadata["backups"]:
            if backup["id"] == backup_id and backup["status"] == "success":
                return {
                    f["path"]: f
                    for f in backup.get("files", [])
                    if f.get("source") == source and "checksum" in f
                }
        return {}

    # ------------------------------------------------------------------
    # Archivos
    # ------------------------------------------------------------------

    def _create_backup_archive(self, staging_dir: Path, info: Dict[str, Any]) -> Path:
        name = f"{info['id']}_{info['type']}"
        archive = self.backup_path / f"{name}.tar.gz"

        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staging_dir, arcname=info["id"])

        if self.config["encryption"] and self.cipher:
            archive = self._encrypt_backup(archive)

        return archive

    def _encrypt_backup(self, file: Path) -> Path:
        encrypted = file.with_suffix(file.suffix + ".enc")
        encrypted.write_bytes(self.cipher.encrypt(file.read_bytes()))
        file.unlink()
        return encrypted

    def _verify_backup_integrity(self, file: Path) -> bool:
        try:
            data = (
                self.cipher.decrypt(file.read_bytes())
                if file.suffix.endswith(".enc")
                else file.read_bytes()
            )

            tmp = self.backup_path / "temp" / "verify.tar.gz"
            tmp.write_bytes(data)

            with tarfile.open(tmp, "r:gz") as tar:
                tar.getmembers()

            tmp.unlink()
            return True
        except Exception:
            return False

    def _calculate_checksum(self, file: Path) -> str:
        h = hashlib.sha256()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _register_backup(self, info: Dict[str, Any]) -> None:
        self.metadata["backups"].append(info)
        self.metadata["total_backups"] += 1

        if info["status"] == "success":
            self.metadata["successful_backups"] += 1
            self.metadata["last_backup"] = info["created_at"]
            self.metadata["total_size"] += info.get("size", 0)
        else:
            self.metadata["failed_backups"] += 1

        self.metadata["backups"] = self.metadata["backups"][-1000:]
        self._save_metadata()

    def _get_last_successful_backup(self) -> Optional[Dict[str, Any]]:
        for backup in reversed(self.metadata["backups"]):
            if backup["status"] == "success":
                return backup
        return None
