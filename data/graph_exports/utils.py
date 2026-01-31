import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from ...core.exceptions import GraphException


def normalize_graph(graph: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(graph, dict):
        raise GraphException("graph_data must be a dict")

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise GraphException("nodes and edges must be lists")

    return {"nodes": nodes, "edges": edges}


def count_graph(graph: Dict[str, Any]) -> Dict[str, int]:
    return {
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
    }


def log_export(
    base_path: Path,
    project_id: str,
    fmt: str,
    file: Path,
) -> None:
    log_file = base_path / "export_log.json"
    data = json.loads(log_file.read_text()) if log_file.exists() else {"exports": []}

    data["exports"].append({
        "project_id": project_id,
        "format": fmt,
        "file": file.name,
        "size": file.stat().st_size if file.exists() else None,
        "timestamp": datetime.now().isoformat(),
    })

    data["exports"] = data["exports"][-1000:]
    log_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
