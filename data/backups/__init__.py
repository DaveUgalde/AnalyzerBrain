from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .crypto import CryptoManager
from .metadata import BackupMetadata
from .filesystem import backup_full, backup_incremental
from .archive import create_archive, verify_archive
from .scheduler import BackupScheduler
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

        for d in ("full", "incremental", "logs", "temp", "restore_points"):
            (self.backup_path / d).mkdir(parents=True, exist_ok=True)

        self.crypto = CryptoManager(self.backup_path, self.config["encryption"])
        self.metadata = BackupMetadata(self.backup_path)

        if self.config["enabled"]:
            BackupScheduler(self.config["schedule"], self.create_scheduled_backup)

    # -------------------------------------------------

    def create_scheduled_backup(self) -> Dict[str, Any]:
        full = self._should_full()
        return self.create_backup("full" if full else "incremental", "Scheduled backup")

    def _should_full(self) -> bool:
        last = self.metadata.last_successful()
        if not last or last["type"] != "full":
            return True
        return (datetime.now() - datetime.fromisoformat(last["created_at"])).days >= 7

    # -------------------------------------------------

    def create_backup(
        self,
        backup_type: str = "incremental",
        description: str = "",
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        staging = self.backup_path / "temp" / backup_id
        staging.mkdir(parents=True, exist_ok=True)

        info = {
            "id": backup_id,
            "type": backup_type,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "sources": sources or self.config["backup_sources"],
            "files": [],
            "status": "in_progress",
        }

        try:
            base = self.metadata.last_successful() if backup_type == "incremental" else None

            for src in info["sources"]:
                src_dir = Path("data") / src
                if not src_dir.exists():
                    continue

                if backup_type == "full":
                    info["files"] += backup_full(src, src_dir, staging)
                else:
                    state = {
                        f["path"]: f
                        for f in base.get("files", [])
                        if f["source"] == src
                    } if base else {}
                    info["files"] += backup_incremental(src, src_dir, staging, state)

            archive = self.backup_path / f"{backup_id}_{backup_type}.tar.gz"
            create_archive(staging, archive)

            if self.crypto.enabled:
                archive.write_bytes(self.crypto.encrypt(archive.read_bytes()))

            if self.config["verify_integrity"] and not verify_archive(archive):
                raise BackupException("Integrity check failed")

            info.update({
                "status": "success",
                "backup_file": str(archive),
                "size": archive.stat().st_size,
            })

        except Exception as exc:
            info["status"] = "failed"
            info["error"] = str(exc)
            raise
        finally:
            self.metadata.register(info)

        return info
