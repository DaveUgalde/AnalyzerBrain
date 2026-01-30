"""
GraphBuilder - Constructor de grafos a partir de diferentes fuentes.
Transforma entidades y relaciones en un grafo de conocimiento estructurado.
"""

from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass
import uuid
from datetime import datetime
from .knowledge_graph import KnowledgeGraph, Node, NodeType, Relationship, RelationshipType
from ..indexer.entity_extractor import Entity

@dataclass
class BuildConfig:
    """Configuración para la construcción del grafo."""
    merge_duplicates: bool = True
    infer_relationships: bool = True
    max_nodes: int = 10000
    max_relationships: int = 50000
    validate_schema: bool = True
    batch_size: int = 1000

class GraphBuilder:
    """
    Constructor de grafos a partir de diferentes fuentes de datos.
    """
    
    def __init__(self, config: Optional[BuildConfig] = None):
        """
        Inicializa el constructor de grafos.
        
        Args:
            config: Configuración de construcción
        """
        self.config = config or BuildConfig()
        self.stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "duplicates_merged": 0,
            "errors": []
        }
    
    def build_from_entities(self, entities: List[Dict[str, Any]]) -> KnowledgeGraph:
        """
        Construye un grafo a partir de una lista de entidades.
        
        Args:
            entities: Lista de diccionarios con información de entidades
            
        Returns:
            Grafo construido
        """
        graph = KnowledgeGraph(name="entity_graph")
        
        # Primera pasada: crear nodos
        node_map = {}
        for entity in entities:
            try:
                node = self._entity_to_node(entity)
                if node.id in graph.nodes:
                    if self.config.merge_duplicates:
                        self._merge_nodes(graph.nodes[node.id], node)
                        self.stats["duplicates_merged"] += 1
                        continue
                    else:
                        raise ValueError(f"Duplicate node id: {node.id}")
                
                graph.add_node(node)
                node_map[entity.get("id", str(uuid.uuid4()))] = node.id
                self.stats["nodes_created"] += 1
                
            except Exception as e:
                self.stats["errors"].append({
                    "entity": entity.get("id", "unknown"),
                    "error": str(e),
                    "operation": "node_creation"
                })
        
        # Segunda pasada: crear relaciones
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id not in node_map:
                continue
            
            source_id = node_map[entity_id]
            
            # Procesar relaciones explícitas
            for rel_data in entity.get("relationships", []):
                try:
                    target_id = node_map.get(rel_data.get("target_id"))
                    if not target_id:
                        continue
                    
                    rel_type = self._parse_relationship_type(rel_data.get("type", "related_to"))
                    rel = Relationship(
                        source_id=source_id,
                        target_id=target_id,
                        type=rel_type,
                        properties=rel_data.get("properties", {})
                    )
                    
                    graph.add_edge(rel)
                    self.stats["relationships_created"] += 1
                    
                except Exception as e:
                    self.stats["errors"].append({
                        "entity": entity_id,
                        "error": str(e),
                        "operation": "relationship_creation"
                    })
        
        # Tercera pasada: inferir relaciones implícitas
        if self.config.infer_relationships:
            self._infer_implicit_relationships(graph, node_map)
        
        return graph
    
    def build_from_dependencies(self, dependencies: List[Dict[str, Any]]) -> KnowledgeGraph:
        """
        Construye un grafo de dependencias.
        
        Args:
            dependencies: Lista de dependencias con formato {source, target, type, properties}
            
        Returns:
            Grafo de dependencias
        """
        graph = KnowledgeGraph(name="dependency_graph")
        
        # Crear nodos únicos
        all_items = set()
        for dep in dependencies:
            all_items.add(dep["source"])
            all_items.add(dep["target"])
        
        node_map = {}
        for item in all_items:
            node = Node(
                type=NodeType.MODULE,
                properties={"name": item},
                labels={"module", "dependency"}
            )
            graph.add_node(node)
            node_map[item] = node.id
            self.stats["nodes_created"] += 1
        
        # Crear relaciones de dependencia
        for dep in dependencies:
            source_id = node_map.get(dep["source"])
            target_id = node_map.get(dep["target"])
            
            if not source_id or not target_id:
                continue
            
            rel_type = RelationshipType.DEPENDS_ON
            if "type" in dep:
                try:
                    rel_type = RelationshipType(dep["type"].lower())
                except ValueError:
                    pass
            
            rel = Relationship(
                source_id=source_id,
                target_id=target_id,
                type=rel_type,
                properties=dep.get("properties", {})
            )
            
            graph.add_edge(rel)
            self.stats["relationships_created"] += 1
        
        return graph
    
    def build_from_embeddings(self, 
                             embeddings: List[Dict[str, Any]], 
                             similarity_threshold: float = 0.8) -> KnowledgeGraph:
        """
        Construye un grafo a partir de embeddings, conectando elementos similares.
        
        Args:
            embeddings: Lista de diccionarios con id, embedding y metadata
            similarity_threshold: Umbral de similitud para crear relaciones
            
        Returns:
            Grafo de similitud
        """
        graph = KnowledgeGraph(name="similarity_graph")
        
        # Crear nodos
        node_map = {}
        for emb in embeddings:
            node = Node(
                type=NodeType.CONCEPT,
                properties={
                    "embedding": emb["embedding"],
                    "metadata": emb.get("metadata", {}),
                    "original_id": emb["id"]
                },
                labels={"concept", "embedding"}
            )
            graph.add_node(node)
            node_map[emb["id"]] = node.id
            self.stats["nodes_created"] += 1
        
        # Calcular similitudes y crear relaciones (versión simplificada)
        # En producción usar ANN (Approximate Nearest Neighbor)
        if len(embeddings) > 1000:
            # Para grandes conjuntos, usar sampling o ANN
            self._build_similarity_graph_approximate(graph, node_map, embeddings, similarity_threshold)
        else:
            # Para conjuntos pequeños, cálculo exacto
            self._build_similarity_graph_exact(graph, node_map, embeddings, similarity_threshold)
        
        return graph
    
    def merge_graphs(self, graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """
        Fusiona múltiples grafos en uno solo.
        
        Args:
            graphs: Lista de grafos a fusionar
            
        Returns:
            Grafo fusionado
        """
        merged = KnowledgeGraph(name="merged_graph")
        
        for graph in graphs:
            # Añadir nodos
            for node in graph.nodes.values():
                if node.id not in merged.nodes:
                    merged.add_node(node)
                    self.stats["nodes_created"] += 1
                elif self.config.merge_duplicates:
                    self._merge_nodes(merged.nodes[node.id], node)
                    self.stats["duplicates_merged"] += 1
            
            # Añadir relaciones
            for rel in graph.relationships.values():
                # Verificar que ambos nodos existan
                if (rel.source_id in merged.nodes and 
                    rel.target_id in merged.nodes and
                    rel.id not in merged.relationships):
                    
                    merged.add_edge(rel)
                    self.stats["relationships_created"] += 1
        
        return merged
    
    def optimize_graph_structure(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """
        Optimiza la estructura del grafo.
        
        Args:
            graph: Grafo a optimizar
            
        Returns:
            Grafo optimizado
        """
        # 1. Eliminar nodos aislados (opcional)
        if self.config.merge_duplicates:
            self._remove_isolated_nodes(graph)
        
        # 2. Fusionar nodos similares
        self._merge_similar_nodes(graph)
        
        # 3. Eliminar relaciones redundantes
        self._remove_redundant_relationships(graph)
        
        # 4. Comprimir caminos
        self._compress_paths(graph)
        
        return graph
    
    def validate_graph(self, graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """
        Valida la corrección del grafo.
        
        Args:
            graph: Grafo a validar
            
        Returns:
            Lista de problemas encontrados
        """
        issues = []
        
        # Verificar nodos huérfanos
        orphan_nodes = self._find_orphan_nodes(graph)
        if orphan_nodes:
            issues.append({
                "type": "orphan_nodes",
                "count": len(orphan_nodes),
                "nodes": list(orphan_nodes)[:10]  # Mostrar solo primeros 10
            })
        
        # Verificar relaciones inválidas
        invalid_rels = self._find_invalid_relationships(graph)
        if invalid_rels:
            issues.append({
                "type": "invalid_relationships",
                "count": len(invalid_rels),
                "relationships": invalid_rels[:10]
            })
        
        # Verificar ciclos en dependencias
        dependency_cycles = self._find_dependency_cycles(graph)
        if dependency_cycles:
            issues.append({
                "type": "dependency_cycles",
                "count": len(dependency_cycles),
                "cycles": dependency_cycles[:5]  # Mostrar solo primeros 5 ciclos
            })
        
        # Verificar consistencia de tipos
        type_issues = self._validate_node_types(graph)
        if type_issues:
            issues.append({
                "type": "type_inconsistencies",
                "issues": type_issues
            })
        
        return issues
    
    def export_graph(self, graph: KnowledgeGraph, format: str = "dict") -> Any:
        """
        Exporta el grafo en diferentes formatos.
        
        Args:
            graph: Grafo a exportar
            format: Formato de exportación
            
        Returns:
            Grafo en el formato especificado
        """
        if format == "dict":
            return graph.export_to_dict()
        elif format == "networkx":
            return graph.to_networkx()
        elif format == "adjacency_list":
            return self._to_adjacency_list(graph)
        elif format == "edge_list":
            return self._to_edge_list(graph)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    # Métodos auxiliares
    
    def _entity_to_node(self, entity: Dict[str, Any]) -> Node:
        """Convierte una entidad a nodo."""
        entity_type = entity.get("type", "concept").upper()
        
        try:
            node_type = NodeType(entity_type)
        except ValueError:
            # Mapear tipos desconocidos a CONCEPT
            node_type = NodeType.CONCEPT
        
        properties = entity.get("properties", {}).copy()
        if "type" in properties:
            del properties["type"]
        
        labels = set(entity.get("labels", []))
        labels.add(entity_type.lower())
        
        return Node(
            id=entity.get("id", str(uuid.uuid4())),
            type=node_type,
            properties=properties,
            labels=labels
        )
    
    def _parse_relationship_type(self, rel_type: str) -> RelationshipType:
        """Parsea un tipo de relación."""
        try:
            return RelationshipType(rel_type.upper())
        except ValueError:
            # Mapear tipos desconocidos a RELATED_TO
            return RelationshipType.RELATED_TO
    
    def _merge_nodes(self, existing: Node, new: Node) -> None:
        """Fusiona dos nodos."""
        # Combinar propiedades (nuevas sobreescriben antiguas)
        existing.properties.update(new.properties)
        
        # Combinar etiquetas
        existing.labels.update(new.labels)
        
        # Actualizar timestamp
        existing.updated_at = datetime.now()
    
    def _infer_implicit_relationships(self, graph: KnowledgeGraph, node_map: Dict[str, str]) -> None:
        """Infere relaciones implícitas basadas en propiedades comunes."""
        # Agrupar nodos por propiedades comunes
        property_groups = {}
        
        for node_id, node in graph.nodes.items():
            # Agrupar por tipo
            key = f"type:{node.type.value}"
            if key not in property_groups:
                property_groups[key] = []
            property_groups[key].append(node_id)
            
            # Agrupar por propiedades específicas
            for prop_name, prop_value in node.properties.items():
                if isinstance(prop_value, (str, int, float, bool)):
                    key = f"{prop_name}:{prop_value}"
                    if key not in property_groups:
                        property_groups[key] = []
                    property_groups[key].append(node_id)
        
        # Crear relaciones entre nodos en los mismos grupos
        for group_name, node_ids in property_groups.items():
            if len(node_ids) < 2:
                continue
            
            # Crear relaciones entre todos los pares (simplificado)
            # En producción usaríamos un límite
            for i in range(len(node_ids)):
                for j in range(i + 1, min(i + 5, len(node_ids))):  # Límite de 5 conexiones
                    rel = Relationship(
                        source_id=node_ids[i],
                        target_id=node_ids[j],
                        type=RelationshipType.SIMILAR_TO,
                        properties={"group": group_name, "inferred": True}
                    )
                    
                    try:
                        graph.add_edge(rel)
                        self.stats["relationships_created"] += 1
                    except ValueError:
                        pass  # Relación ya existe
    
    def _build_similarity_graph_exact(self, graph: KnowledgeGraph, node_map: Dict[str, str],
                                     embeddings: List[Dict[str, Any]], threshold: float) -> None:
        """Construye grafo de similitud con cálculo exacto."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Extraer embeddings
        emb_ids = [emb["id"] for emb in embeddings]
        emb_vectors = [emb["embedding"] for emb in embeddings]
        
        if not emb_vectors:
            return
        
        # Calcular matriz de similitud
        sim_matrix = cosine_similarity(emb_vectors)
        
        # Crear relaciones para similitudes altas
        for i in range(len(emb_ids)):
            for j in range(i + 1, len(emb_ids)):
                similarity = sim_matrix[i][j]
                if similarity > threshold:
                    source_id = node_map[emb_ids[i]]
                    target_id = node_map[emb_ids[j]]
                    
                    rel = Relationship(
                        source_id=source_id,
                        target_id=target_id,
                        type=RelationshipType.SIMILAR_TO,
                        properties={
                            "similarity": float(similarity),
                            "threshold": threshold,
                            "inferred": True
                        }
                    )
                    
                    try:
                        graph.add_edge(rel)
                        self.stats["relationships_created"] += 1
                    except ValueError:
                        pass
    
    def _build_similarity_graph_approximate(self, graph: KnowledgeGraph, node_map: Dict[str, str],
                                           embeddings: List[Dict[str, Any]], threshold: float) -> None:
        """Construye grafo de similitud con ANN aproximado."""
        # Implementación simplificada - usaría FAISS o similar en producción
        # Por ahora, usamos sampling
        import random
        import numpy as np
        
        sample_size = min(100, len(embeddings))
        sampled = random.sample(embeddings, sample_size)
        
        # Usar el método exacto en la muestra
        self._build_similarity_graph_exact(graph, node_map, sampled, threshold)
    
    def _remove_isolated_nodes(self, graph: KnowledgeGraph) -> None:
        """Elimina nodos aislados del grafo."""
        nodes_to_remove = []
        
        for node_id, rels in graph._node_relationships.items():
            if not rels["incoming"] and not rels["outgoing"]:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            graph.remove_node(node_id, cascade=False)
    
    def _merge_similar_nodes(self, graph: KnowledgeGraph) -> None:
        """Fusiona nodos similares."""
        # Agrupar nodos por propiedades clave
        groups = {}
        for node_id, node in graph.nodes.items():
            # Usar combinación de tipo y nombre como clave
            key = (node.type.value, node.properties.get("name", ""))
            if key not in groups:
                groups[key] = []
            groups[key].append(node_id)
        
        # Fusionar nodos en cada grupo
        for key, node_ids in groups.items():
            if len(node_ids) < 2:
                continue
            
            # Mantener el primero, fusionar los demás
            keep_id = node_ids[0]
            for merge_id in node_ids[1:]:
                self._merge_nodes(graph.nodes[keep_id], graph.nodes[merge_id])
                
                # Redirigir relaciones
                self._redirect_relationships(graph, merge_id, keep_id)
                
                # Eliminar nodo fusionado
                graph.remove_node(merge_id, cascade=True)
    
    def _redirect_relationships(self, graph: KnowledgeGraph, old_id: str, new_id: str) -> None:
        """Redirige relaciones de un nodo viejo a uno nuevo."""
        rels_to_update = []
        
        # Encontrar todas las relaciones que involucran al nodo viejo
        for rel_id, rel in graph.relationships.items():
            if rel.source_id == old_id or rel.target_id == old_id:
                rels_to_update.append(rel_id)
        
        # Actualizar relaciones
        for rel_id in rels_to_update:
            rel = graph.relationships[rel_id]
            
            # Crear nueva relación
            new_rel = Relationship(
                source_id=rel.source_id if rel.source_id != old_id else new_id,
                target_id=rel.target_id if rel.target_id != old_id else new_id,
                type=rel.type,
                properties=rel.properties.copy()
            )
            
            # Eliminar relación vieja y añadir nueva
            graph.remove_edge(rel_id)
            try:
                graph.add_edge(new_rel)
            except ValueError:
                pass  # Relación duplicada
    
    def _remove_redundant_relationships(self, graph: KnowledgeGraph) -> None:
        """Elimina relaciones redundantes."""
        seen = set()
        rels_to_remove = []
        
        for rel_id, rel in graph.relationships.items():
            key = (rel.source_id, rel.target_id, rel.type.value)
            if key in seen:
                rels_to_remove.append(rel_id)
            else:
                seen.add(key)
        
        for rel_id in rels_to_remove:
            graph.remove_edge(rel_id)
    
    def _compress_paths(self, graph: KnowledgeGraph) -> None:
        """Comprime caminos transitivos."""
        # Para cada par de nodos, si hay un camino directo e indirecto,
        # eliminar la relación indirecta si es redundante
        # (implementación simplificada)
        pass
    
    def _find_orphan_nodes(self, graph: KnowledgeGraph) -> Set[str]:
        """Encuentra nodos huérfanos."""
        orphans = set()
        for node_id, rels in graph._node_relationships.items():
            if not rels["incoming"] and not rels["outgoing"]:
                orphans.add(node_id)
        return orphans
    
    def _find_invalid_relationships(self, graph: KnowledgeGraph) -> List[str]:
        """Encuentra relaciones inválidas."""
        invalid = []
        for rel_id, rel in graph.relationships.items():
            if (rel.source_id not in graph.nodes or 
                rel.target_id not in graph.nodes):
                invalid.append(rel_id)
        return invalid
    
    def _find_dependency_cycles(self, graph: KnowledgeGraph) -> List[List[str]]:
        """Encuentra ciclos en relaciones de dependencia."""
        import networkx as nx
        
        nx_graph = nx.DiGraph()
        
        # Añadir solo relaciones de dependencia
        for rel in graph.relationships.values():
            if rel.type == RelationshipType.DEPENDS_ON:
                nx_graph.add_edge(rel.source_id, rel.target_id)
        
        # Encontrar ciclos
        try:
            return list(nx.simple_cycles(nx_graph))
        except:
            return []
    
    def _validate_node_types(self, graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Valida consistencia de tipos de nodos."""
        issues = []
        
        # Verificar que nodos FILE tengan propiedad 'path'
        for node_id, node in graph.nodes.items():
            if node.type == NodeType.FILE and "path" not in node.properties:
                issues.append({
                    "node_id": node_id,
                    "issue": "FILE node missing 'path' property",
                    "type": "missing_property"
                })
        
        return issues
    
    def _to_adjacency_list(self, graph: KnowledgeGraph) -> Dict[str, List[Dict[str, Any]]]:
        """Convierte a lista de adyacencia."""
        adjacency = {}
        
        for node_id in graph.nodes:
            neighbors = []
            for rel_id in graph._node_relationships[node_id]["outgoing"]:
                rel = graph.relationships[rel_id]
                neighbors.append({
                    "node_id": rel.target_id,
                    "relationship": rel.type.value,
                    "properties": rel.properties
                })
            
            adjacency[node_id] = neighbors
        
        return adjacency
    
    def _to_edge_list(self, graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Convierte a lista de aristas."""
        edges = []
        
        for rel in graph.relationships.values():
            edges.append({
                "source": rel.source_id,
                "target": rel.target_id,
                "type": rel.type.value,
                "properties": rel.properties
            })
        
        return edges