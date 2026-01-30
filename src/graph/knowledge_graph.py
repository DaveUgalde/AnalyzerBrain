"""
KnowledgeGraph - Grafo de conocimiento principal.
Almacena nodos y relaciones con propiedades, y proporciona operaciones básicas de grafo.
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
import networkx as nx
from pydantic import BaseModel, Field, validator
import json

class NodeType(Enum):
    """Tipos de nodos en el grafo."""
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    MODULE = "module"
    PACKAGE = "package"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    TRAIT = "trait"
    CONCEPT = "concept"
    PATTERN = "pattern"
    ISSUE = "issue"
    RECOMMENDATION = "recommendation"
    PERSON = "person"
    TEAM = "team"

class RelationshipType(Enum):
    """Tipos de relaciones en el grafo."""
    CONTAINS = "contains"
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    REFERENCES = "references"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    CREATES = "creates"
    MODIFIES = "modifies"
    TESTS = "tests"
    DOCUMENTS = "documents"
    AUTHORED_BY = "authored_by"
    ASSIGNED_TO = "assigned_to"
    RELATED_TO = "related_to"

@dataclass
class Node:
    """Nodo en el grafo de conocimiento."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    labels: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Inicialización posterior."""
        if not self.labels:
            self.labels = {self.type.value}
        else:
            self.labels.add(self.type.value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el nodo a diccionario."""
        return {
            "id": self.id,
            "type": self.type.value,
            "properties": self.properties,
            "labels": list(self.labels),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Crea un nodo desde diccionario."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=NodeType(data["type"]),
            properties=data.get("properties", {}),
            labels=set(data.get("labels", [])),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        )

@dataclass
class Relationship:
    """Relación entre nodos en el grafo."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la relación a diccionario."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "properties": self.properties,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Crea una relación desde diccionario."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=RelationshipType(data["type"]),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )

class KnowledgeGraph:
    """
    Grafo de conocimiento principal con operaciones CRUD y análisis básico.
    """
    
    def __init__(self, name: str = "knowledge_graph"):
        """
        Inicializa un grafo de conocimiento.
        
        Args:
            name: Nombre del grafo
        """
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.relationships: Dict[str, Relationship] = {}
        self._node_relationships: Dict[str, Dict[str, List[str]]] = {}
        self._reverse_relationships: Dict[str, Dict[str, List[str]]] = {}
        
    def add_node(self, node: Node) -> str:
        """
        Añade un nodo al grafo.
        
        Args:
            node: Nodo a añadir
            
        Returns:
            ID del nodo añadido
            
        Raises:
            ValueError: Si el nodo ya existe
        """
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists")
        
        self.nodes[node.id] = node
        self._node_relationships[node.id] = {"outgoing": [], "incoming": []}
        self._reverse_relationships[node.id] = {"outgoing": [], "incoming": []}
        return node.id
    
    def add_edge(self, relationship: Relationship) -> str:
        """
        Añade una relación (arista) al grafo.
        
        Args:
            relationship: Relación a añadir
            
        Returns:
            ID de la relación añadida
            
        Raises:
            ValueError: Si la relación ya existe o los nodos no existen
        """
        if relationship.id in self.relationships:
            raise ValueError(f"Relationship with id {relationship.id} already exists")
        
        if relationship.source_id not in self.nodes:
            raise ValueError(f"Source node {relationship.source_id} does not exist")
        
        if relationship.target_id not in self.nodes:
            raise ValueError(f"Target node {relationship.target_id} does not exist")
        
        self.relationships[relationship.id] = relationship
        
        # Actualizar índices
        self._node_relationships[relationship.source_id]["outgoing"].append(relationship.id)
        self._node_relationships[relationship.target_id]["incoming"].append(relationship.id)
        
        self._reverse_relationships[relationship.target_id]["outgoing"].append(relationship.id)
        self._reverse_relationships[relationship.source_id]["incoming"].append(relationship.id)
        
        return relationship.id
    
    def remove_node(self, node_id: str, cascade: bool = True) -> bool:
        """
        Elimina un nodo y opcionalmente sus relaciones.
        
        Args:
            node_id: ID del nodo a eliminar
            cascade: Si True, elimina también todas las relaciones del nodo
            
        Returns:
            True si se eliminó, False si no existía
        """
        if node_id not in self.nodes:
            return False
        
        if cascade:
            # Eliminar todas las relaciones asociadas
            rels_to_remove = (
                self._node_relationships[node_id]["outgoing"] +
                self._node_relationships[node_id]["incoming"]
            )
            for rel_id in rels_to_remove:
                self.remove_edge(rel_id)
        
        # Eliminar el nodo
        del self.nodes[node_id]
        del self._node_relationships[node_id]
        del self._reverse_relationships[node_id]
        
        return True
    
    def remove_edge(self, relationship_id: str) -> bool:
        """
        Elimina una relación.
        
        Args:
            relationship_id: ID de la relación a eliminar
            
        Returns:
            True si se eliminó, False si no existía
        """
        if relationship_id not in self.relationships:
            return False
        
        rel = self.relationships[relationship_id]
        
        # Actualizar índices
        if rel.source_id in self._node_relationships:
            self._node_relationships[rel.source_id]["outgoing"].remove(relationship_id)
            self._reverse_relationships[rel.target_id]["incoming"].remove(relationship_id)
        
        if rel.target_id in self._node_relationships:
            self._node_relationships[rel.target_id]["incoming"].remove(relationship_id)
            self._reverse_relationships[rel.source_id]["outgoing"].remove(relationship_id)
        
        # Eliminar relación
        del self.relationships[relationship_id]
        
        return True
    
    def find_node(self, node_id: str) -> Optional[Node]:
        """
        Encuentra un nodo por ID.
        
        Args:
            node_id: ID del nodo
            
        Returns:
            El nodo si existe, None en caso contrario
        """
        return self.nodes.get(node_id)
    
    def find_nodes(self, 
                  node_type: Optional[NodeType] = None,
                  properties: Optional[Dict[str, Any]] = None,
                  labels: Optional[List[str]] = None,
                  limit: int = 100) -> List[Node]:
        """
        Encuentra nodos por criterios.
        
        Args:
            node_type: Tipo de nodo a buscar
            properties: Propiedades que deben coincidir
            labels: Etiquetas que debe tener
            limit: Número máximo de resultados
            
        Returns:
            Lista de nodos que cumplen los criterios
        """
        results = []
        
        for node in self.nodes.values():
            # Filtrar por tipo
            if node_type and node.type != node_type:
                continue
            
            # Filtrar por propiedades
            if properties:
                match = True
                for key, value in properties.items():
                    if key not in node.properties or node.properties[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Filtrar por etiquetas
            if labels:
                if not all(label in node.labels for label in labels):
                    continue
            
            results.append(node)
            
            if len(results) >= limit:
                break
        
        return results
    
    def find_path(self, start_id: str, end_id: str, 
                 max_depth: int = 10,
                 relationship_types: Optional[List[RelationshipType]] = None) -> List[List[Node]]:
        """
        Encuentra todos los caminos entre dos nodos.
        
        Args:
            start_id: ID del nodo de inicio
            end_id: ID del nodo de fin
            max_depth: Profundidad máxima de búsqueda
            relationship_types: Tipos de relaciones a considerar
            
        Returns:
            Lista de caminos, donde cada camino es una lista de nodos
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return []
        
        visited = set()
        paths = []
        
        def dfs(current_id: str, path: List[Node], depth: int) -> None:
            if depth > max_depth:
                return
            
            current_node = self.nodes[current_id]
            path.append(current_node)
            visited.add(current_id)
            
            if current_id == end_id:
                paths.append(path.copy())
            else:
                # Explorar vecinos salientes
                for rel_id in self._node_relationships[current_id]["outgoing"]:
                    rel = self.relationships[rel_id]
                    
                    # Filtrar por tipo de relación
                    if relationship_types and rel.type not in relationship_types:
                        continue
                    
                    if rel.target_id not in visited:
                        dfs(rel.target_id, path, depth + 1)
            
            # Backtrack
            path.pop()
            visited.remove(current_id)
        
        dfs(start_id, [], 0)
        return paths
    
    def get_neighbors(self, node_id: str, 
                     direction: str = "outgoing",
                     relationship_types: Optional[List[RelationshipType]] = None) -> List[Tuple[Node, Relationship]]:
        """
        Obtiene vecinos de un nodo.
        
        Args:
            node_id: ID del nodo
            direction: 'outgoing', 'incoming', o 'both'
            relationship_types: Tipos de relaciones a considerar
            
        Returns:
            Lista de tuplas (nodo, relación)
        """
        if node_id not in self.nodes:
            return []
        
        neighbors = []
        
        if direction in ["outgoing", "both"]:
            for rel_id in self._node_relationships[node_id]["outgoing"]:
                rel = self.relationships[rel_id]
                if relationship_types and rel.type not in relationship_types:
                    continue
                target_node = self.nodes[rel.target_id]
                neighbors.append((target_node, rel))
        
        if direction in ["incoming", "both"]:
            for rel_id in self._node_relationships[node_id]["incoming"]:
                rel = self.relationships[rel_id]
                if relationship_types and rel.type not in relationship_types:
                    continue
                source_node = self.nodes[rel.source_id]
                neighbors.append((source_node, rel))
        
        return neighbors
    
    def calculate_graph_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas básicas del grafo.
        
        Returns:
            Diccionario con métricas del grafo
        """
        total_nodes = len(self.nodes)
        total_edges = len(self.relationships)
        
        # Calcular grados
        in_degrees = {}
        out_degrees = {}
        total_degrees = {}
        
        for node_id in self.nodes:
            in_deg = len(self._node_relationships[node_id]["incoming"])
            out_deg = len(self._node_relationships[node_id]["outgoing"])
            total_deg = in_deg + out_deg
            
            in_degrees[node_id] = in_deg
            out_degrees[node_id] = out_deg
            total_degrees[node_id] = total_deg
        
        # Calcular densidad (para grafo dirigido)
        density = 0.0
        if total_nodes > 1:
            max_possible_edges = total_nodes * (total_nodes - 1)
            density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Calcular distribución de tipos
        type_distribution = {}
        for node in self.nodes.values():
            type_name = node.type.value
            type_distribution[type_name] = type_distribution.get(type_name, 0) + 1
        
        # Calcular distribución de relaciones
        relationship_distribution = {}
        for rel in self.relationships.values():
            type_name = rel.type.value
            relationship_distribution[type_name] = relationship_distribution.get(type_name, 0) + 1
        
        # Encontrar nodos con mayor grado
        if total_degrees:
            max_degree_node = max(total_degrees.items(), key=lambda x: x[1])
            min_degree_node = min(total_degrees.items(), key=lambda x: x[1])
        else:
            max_degree_node = (None, 0)
            min_degree_node = (None, 0)
        
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "density": density,
            "average_in_degree": sum(in_degrees.values()) / total_nodes if total_nodes > 0 else 0,
            "average_out_degree": sum(out_degrees.values()) / total_nodes if total_nodes > 0 else 0,
            "average_total_degree": sum(total_degrees.values()) / total_nodes if total_nodes > 0 else 0,
            "max_degree": max_degree_node,
            "min_degree": min_degree_node,
            "type_distribution": type_distribution,
            "relationship_distribution": relationship_distribution,
            "connected_components": self._find_connected_components(),
        }
    
    def _find_connected_components(self) -> int:
        """Encuentra componentes conectados."""
        visited = set()
        components = 0
        
        for node_id in self.nodes:
            if node_id not in visited:
                components += 1
                self._bfs_component(node_id, visited)
        
        return components
    
    def _bfs_component(self, start_id: str, visited: Set[str]) -> None:
        """BFS para encontrar un componente conectado."""
        queue = [start_id]
        visited.add(start_id)
        
        while queue:
            current_id = queue.pop(0)
            
            # Vecinos salientes
            for rel_id in self._node_relationships[current_id]["outgoing"]:
                rel = self.relationships[rel_id]
                if rel.target_id not in visited:
                    visited.add(rel.target_id)
                    queue.append(rel.target_id)
            
            # Vecinos entrantes
            for rel_id in self._node_relationships[current_id]["incoming"]:
                rel = self.relationships[rel_id]
                if rel.source_id not in visited:
                    visited.add(rel.source_id)
                    queue.append(rel.source_id)
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Convierte el grafo a NetworkX para análisis avanzado.
        
        Returns:
            Grafo de NetworkX
        """
        nx_graph = nx.DiGraph()
        
        # Añadir nodos
        for node_id, node in self.nodes.items():
            nx_graph.add_node(
                node_id,
                type=node.type.value,
                labels=list(node.labels),
                properties=node.properties,
                created_at=node.created_at,
                updated_at=node.updated_at
            )
        
        # Añadir aristas
        for rel_id, rel in self.relationships.items():
            nx_graph.add_edge(
                rel.source_id,
                rel.target_id,
                id=rel_id,
                type=rel.type.value,
                properties=rel.properties,
                created_at=rel.created_at
            )
        
        return nx_graph
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Exporta el grafo completo a diccionario.
        
        Returns:
            Diccionario con todos los nodos y relaciones
        """
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "relationships": [rel.to_dict() for rel in self.relationships.values()],
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "node_count": len(self.nodes),
                "relationship_count": len(self.relationships)
            }
        }
    
    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Importa un grafo desde diccionario.
        
        Args:
            data: Datos del grafo
        """
        self.name = data.get("name", self.name)
        
        # Limpiar grafo actual
        self.nodes.clear()
        self.relationships.clear()
        self._node_relationships.clear()
        self._reverse_relationships.clear()
        
        # Importar nodos
        for node_data in data.get("nodes", []):
            node = Node.from_dict(node_data)
            self.add_node(node)
        
        # Importar relaciones
        for rel_data in data.get("relationships", []):
            rel = Relationship.from_dict(rel_data)
            self.add_edge(rel)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Guarda el grafo en un archivo JSON.
        
        Args:
            filepath: Ruta del archivo
        """
        data = self.export_to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str) -> None:
        """
        Carga el grafo desde un archivo JSON.
        
        Args:
            filepath: Ruta del archivo
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.import_from_dict(data)