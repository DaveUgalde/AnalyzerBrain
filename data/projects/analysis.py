from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .utils import save_json
from .metadata import update_metadata


def save_analysis(path: Path, analysis: Dict[str, Any]) -> None:
    update_metadata(path, {
        "last_analyzed": datetime.now().isoformat(),
        "analysis_version": analysis.get("version", "1.0.0"),
        "status": "analyzed",
    })

    save_json(path / "analysis.json", analysis)

    for name, data in analysis.get("entities", {}).items():
        save_json(path / "entities" / f"{name}.json", data)

    for name, data in analysis.get("dependencies", {}).items():
        save_json(path / "dependencies" / f"{name}.json", data)

    for severity, data in analysis.get("issues", {}).items():
        save_json(path / "issues" / f"{severity}.json", data)
