import tarfile
from pathlib import Path


def create_archive(staging_dir: Path, output: Path) -> Path:
    with tarfile.open(output, "w:gz") as tar:
        tar.add(staging_dir, arcname=staging_dir.name)
    return output


def verify_archive(file: Path) -> bool:
    try:
        with tarfile.open(file, "r:gz") as tar:
            tar.getmembers()
        return True
    except Exception:
        return False
