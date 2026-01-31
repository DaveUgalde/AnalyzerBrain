import json
import csv
from pathlib import Path
from typing import Dict, Any

import networkx as nx

from .builders import build_nx_graph


def export_json(graph: Dict[str, Any], file: Path) -> Dict[str, Any]:
    file.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    return {}


def export_graphml(graph: Dict[str, Any], file: Path) -> Dict[str, Any]:
    nx.write_graphml(build_nx_graph(graph), file)
    return {}


def export_gexf(graph: Dict[str, Any], file: Path) -> Dict[str, Any]:
    nx.write_gexf(build_nx_graph(graph), file)
    return {}


def export_dot(graph: Dict[str, Any], file: Path) -> Dict[str, Any]:
    lines = [
        "digraph ProjectGraph {",
        "  rankdir=LR;",
        "  node [shape=box, style=filled, fillcolor=lightblue];",
    ]

    for n in graph["nodes"]:
        label = n.get("properties", {}).get("name", n["id"])
        lines.append(f'  "{n["id"]}" [label="{label}"];')

    for e in graph["edges"]:
        lines.append(f'  "{e["source"]}" -> "{e["target"]}";')

    lines.append("}")
    file.write_text("\n".join(lines), encoding="utf-8")
    return {}


def export_csv(graph: Dict[str, Any], file: Path) -> Dict[str, Any]:
    nodes = file.with_suffix(".nodes.csv")
    edges = file.with_suffix(".edges.csv")

    with nodes.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type", "properties"])
        for n in graph["nodes"]:
            writer.writerow([n["id"], n.get("type", ""), json.dumps(n.get("properties", {}))])

    with edges.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "type", "properties"])
        for e in graph["edges"]:
            writer.writerow([
                e["source"],
                e["target"],
                e.get("type", ""),
                json.dumps(e.get("properties", {})),
            ])

    return {
        "nodes_file": str(nodes),
        "edges_file": str(edges),
    }


def export_cypher(graph: Dict[str, Any], file: Path) -> Dict[str, Any]:
    lines = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;"
    ]

    for n in graph["nodes"]:
        label = n.get("type", "Entity").replace(" ", "_")
        props = n.get("properties", {})
        props_str = ", ".join(f"{k}: {json.dumps(v)}" for k, v in props.items())
        props_str = f", {props_str}" if props_str else ""
        lines.append(
            f"CREATE (:{label} {{id: {json.dumps(n['id'])}{props_str}}});"
        )

    for e in graph["edges"]:
        rel = e.get("type", "RELATES_TO").replace(" ", "_").upper()
        lines.append(
            f"MATCH (a {{id: {json.dumps(e['source'])}}}), "
            f"(b {{id: {json.dumps(e['target'])}}}) "
            f"CREATE (a)-[:{rel}]->(b);"
        )

    file.write_text("\n".join(lines), encoding="utf-8")
    return {}


EXPORTERS = {
    "json": export_json,
    "graphml": export_graphml,
    "gexf": export_gexf,
    "dot": export_dot,
    "csv": export_csv,
    "cypher": export_cypher,
}
