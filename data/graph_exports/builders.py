import networkx as nx
from typing import Dict, Any


def build_nx_graph(graph_data: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()

    for node in graph_data["nodes"]:
        G.add_node(
            node["id"],
            type=node.get("type", "entity"),
            **node.get("properties", {}),
        )

    for edge in graph_data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            type=edge.get("type", "relates"),
            **edge.get("properties", {}),
        )

    return G
