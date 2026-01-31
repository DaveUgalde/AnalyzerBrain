import os
import tempfile
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Any

from ...core.exceptions import ValidationError


def export_project(
    base: Path,
    project_path: Path,
    project_id: str,
    data: Dict[str, Any],
    export_format: str,
) -> Dict[str, Any]:

    if export_format == "json":
        return {
            "format": "json",
            "exported_at": data["metadata"].get("updated_at"),
            "project": data,
        }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{export_format}")
    tmp.close()

    if export_format == "zip":
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(project_path):
                for f in files:
                    fp = Path(root) / f
                    z.write(fp, fp.relative_to(base))

    elif export_format == "tar":
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(project_path, arcname=project_id)

    else:
        raise ValidationError(f"Unsupported export format: {export_format}")

    return {
        "format": export_format,
        "path": tmp.name,
        "size": os.path.getsize(tmp.name),
    }
