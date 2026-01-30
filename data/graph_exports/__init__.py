# data/graph_exports/__init__.py
"""
Sistema de exportación de grafos de conocimiento.

Formats soportados:
- JSON: Intercambio estándar
- GraphML: Para Gephi, Cytoscape
- Cypher: Para importar a Neo4j
- DOT: Para Graphviz
- GEXF: Para Gephi
- CSV: Para hojas de cálculo
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import networkx as nx

from ...core.exceptions import GraphException


class GraphExporter:
    """Sistema de exportación de grafos."""

    SUPPORTED_FORMATS = {"json", "graphml", "cypher", "dot", "csv", "gexf"}

    def __init__(self, export_path: Path):
        """
        Inicializa el exportador de grafos.

        Args:
            export_path: Directorio para exportaciones
        """
        self.export_path = export_path
        self.export_path.mkdir(parents=True, exist_ok=True)

    def export_graph(
        self,
        graph_data: Dict[str, Any],
        project_id: str,
        export_format: str = "json",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Exporta grafo en formato especificado.
        """
        export_format = export_format.lower()

        if export_format not in self.SUPPORTED_FORMATS:
            raise GraphException(f"Unsupported format: {export_format}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{project_id}_{timestamp}.{export_format}"
        export_file = self.export_path / filename

        try:
            if export_format == "json":
                result = self._export_json(graph_data, export_file)
            elif export_format == "graphml":
                result = self._export_graphml(graph_data, export_file)
            elif export_format == "cypher":
                result = self._export_cypher(graph_data, export_file)
            elif export_format == "dot":
                result = self._export_dot(graph_data, export_file)
            elif export_format == "csv":
                result = self._export_csv(graph_data, export_file)
            elif export_format == "gexf":
                result = self._export_gexf(graph_data, export_file)

            self._log_export(project_id, export_format, export_file)

            return {
                "success": True,
                "format": export_format,
                "file_path": str(export_file),
                "file_size": export_file.stat().st_size if export_file.exists() else 0,
                **result,
            }

        except Exception as e:
            raise GraphException(f"Export failed: {e}") from e

    # -------------------------------------------------
    # EXPORTADORES
    # -------------------------------------------------

    def _export_json(self, graph_data: Dict[str, Any], export_file: Path) -> Dict[str, Any]:
        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        return {
            "nodes": len(graph_data.get("nodes", [])),
            "edges": len(graph_data.get("edges", [])),
        }

    def _export_graphml(self, graph_data: Dict[str, Any], export_file: Path) -> Dict[str, Any]:
        G = nx.DiGraph()

        for node in graph_data.get("nodes", []):
            G.add_node(node["id"], **node.get("properties", {}))

        for edge in graph_data.get("edges", []):
            G.add_edge(edge["source"], edge["target"], **edge.get("properties", {}))

        nx.write_graphml(G, export_file)

        return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

    def _export_cypher(self, graph_data: Dict[str, Any], export_file: Path) -> Dict[str, Any]:
        lines: List[str] = []

        lines.append(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;"
        )

        for node in graph_data.get("nodes", []):
            node_id = node["id"]
            label = node.get("type", "Entity").replace(" ", "_")
            props = node.get("properties", {})

            prop_items = []
            for k, v in props.items():
                if isinstance(v, str):
                    prop_items.append(f"{k}: {json.dumps(v)}")
                else:
                    prop_items.append(f"{k}: {v}")

            props_str = ", " + ", ".join(prop_items) if prop_items else ""

            lines.append(
                f"CREATE (:{label} {{id: {json.dumps(node_id)}{props_str}}});"
            )

        for edge in graph_data.get("edges", []):
            rel = edge.get("type", "RELATES_TO").replace(" ", "_").upper()
            lines.append(
                f"MATCH (a {{id: {json.dumps(edge['source'])}}}), "
                f"(b {{id: {json.dumps(edge['target'])}}}) "
                f"CREATE (a)-[:{rel}]->(b);")

        with open(export_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return {
            "nodes": len(graph_data.get("nodes", [])),
            "edges": len(graph_data.get("edges", [])),
        }

    def _export_dot(self, graph_data: Dict[str, Any], export_file: Path) -> Dict[str, Any]:
        lines = [
            "digraph ProjectGraph {",
            "  rankdir=LR;",
            "  node [shape=box, style=filled, fillcolor=lightblue];",
        ]

        for node in graph_data.get("nodes", []):
            label = node.get("properties", {}).get("name", node["id"])
            lines.append(f'  "{node["id"]}" [label="{label}"];')

        for edge in graph_data.get("edges", []):
            lines.append(f'  "{edge["source"]}" -> "{edge["target"]}";')

        lines.append("}")

        with open(export_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return {"nodes": len(graph_data.get("nodes", [])), "edges": len(graph_data.get("edges", []))}

    def _export_csv(self, graph_data: Dict[str, Any], export_file: Path) -> Dict[str, Any]:
        nodes_file = export_file.with_suffix(".nodes.csv")
        edges_file = export_file.with_suffix(".edges.csv")

        with open(nodes_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "type", "properties"])
            for n in graph_data.get("nodes", []):
                writer.writerow([n["id"], n.get("type", ""), json.dumps(n.get("properties", {}))])

        with open(edges_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "type", "properties"])
            for e in graph_data.get("edges", []):
                writer.writerow([e["source"], e["target"], e.get("type", ""), json.dumps(e.get("properties", {}))])

        return {"nodes_file": str(nodes_file), "edges_file": str(edges_file)}

    def _export_gexf(self, graph_data: Dict[str, Any], export_file: Path) -> Dict[str, Any]:
        G = nx.DiGraph()

        for node in graph_data.get("nodes", []):
            attrs = node.get("properties", {})
            attrs["type"] = node.get("type", "entity")
            G.add_node(node["id"], **attrs)

        for edge in graph_data.get("edges", []):
            attrs = edge.get("properties", {})
            attrs["type"] = edge.get("type", "relates")
            G.add_edge(edge["source"], edge["target"], **attrs)

        nx.write_gexf(G, export_file)

        return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

    # -------------------------------------------------
    # UTILIDADES
    # -------------------------------------------------

    def _log_export(self, project_id: str, format: str, export_file: Path) -> None:
        log_file = self.export_path / "export_log.json"

        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        else:
            log_data = {"exports": []}

        log_data["exports"].append(
            {
                "project_id": project_id,
                "format": format,
                "file": export_file.name,
                "size": export_file.stat().st_size,
                "timestamp": datetime.now().isoformat(),
            }
        )

        log_data["exports"] = log_data["exports"][-1000:]

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

    def list_exports(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        exports: List[Dict[str, Any]] = []

        for file in self.export_path.iterdir():
            if not file.is_file() or file.name == "export_log.json":
                continue

            if project_id and not file.name.startswith(project_id):
                continue

            exports.append(
                {
                    "filename": file.name,
                    "size": file.stat().st_size,
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    "format": file.suffix.lstrip("."),
                }
            )

        return exports

    def cleanup_exports(self, max_age_days: int = 30) -> Dict[str, int]:
        cutoff = datetime.now().timestamp() - max_age_days * 86400
        stats = {"deleted": 0, "failed": 0, "space_freed": 0}

        for file in self.export_path.iterdir():
            if not file.is_file() or file.name == "export_log.json":
                continue

            if file.stat().st_mtime < cutoff:
                try:
                    size = file.stat().st_size
                    file.unlink()
                    stats["deleted"] += 1
                    stats["space_freed"] += size
                except Exception:
                    stats["failed"] += 1

        return stats
