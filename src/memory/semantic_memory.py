"""
SemanticMemory - Memoria semántica para almacenar y recuperar conceptos y conocimientos.
Sistema basado en grafos semánticos con inferencia de relaciones.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from datetime import datetime
import json
from collections import defaultdict
import uuid
import hashlib
import networkx as nx
from pathlib import Path

from ..core.exceptions import MemoryException, ValidationError

class ConceptType(Enum):
    """Tipos de conceptos en memoria semántica."""
    ENTITY = "entity"           # Entidad concreta (función, clase, archivo)
    CONCEPT = "concept"         # Concepto abstracto (patrón, principio)
    CATEGORY = "category"       # Categoría o clasificación
    RELATION = "relation"       # Relación entre conceptos
    PROPERTY = "property"       # Propiedad o atributo
    EVENT = "event"             # Evento o acción
    RULE = "rule"              # Regla o heurística

class RelationType(Enum):
    """Tipos de relaciones entre conceptos."""
    IS_A = "is_a"              # Relación de tipo/clase
    PART_OF = "part_of"        # Relación parte-todo
    HAS_A = "has_a"            # Relación de posesión
    RELATED_TO = "related_to"  # Relación genérica
    USES = "uses"              # Usa o utiliza
    DEPENDS_ON = "depends_on"  # Dependencia
    SIMILAR_TO = "similar_to"  # Similitud
    OPPOSITE_OF = "opposite_of" # Opuesto
    EXAMPLE_OF = "example_of"  # Ejemplo de
    IMPLIES = "implies"        # Implicación lógica

@dataclass
class SemanticConcept:
    """Concepto en memoria semántica."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    concept_type: ConceptType = ConceptType.CONCEPT
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None  # Embedding vectorial del concepto
    
    # Metadata
    confidence: float = 1.0
    source: Optional[str] = None  # Origen del concepto
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Relaciones (guardadas en el grafo, no en el objeto)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            "id": self.id,
            "name": self.name,
            "concept_type": self.concept_type.value,
            "description": self.description,
            "properties": self.properties,
            "embeddings": self.embeddings,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticConcept':
        """Crea desde diccionario."""
        concept = cls(
            id=data["id"],
            name=data["name"],
            concept_type=ConceptType(data["concept_type"]),
            description=data["description"],
            properties=data["properties"],
            embeddings=data["embeddings"],
            confidence=data["confidence"],
            source=data["source"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            access_count=data["access_count"]
        )
        return concept
    
    def access(self) -> None:
        """Registra un acceso al concepto."""
        self.access_count += 1
        self.updated_at = datetime.now()
    
    def update_confidence(self, delta: float) -> None:
        """Actualiza la confianza del concepto."""
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        self.updated_at = datetime.now()
    
    def similarity_to(self, other: 'SemanticConcept', 
                     use_embeddings: bool = True) -> float:
        """Calcula similitud con otro concepto."""
        if use_embeddings and self.embeddings and other.embeddings:
            # Similitud coseno entre embeddings
            import numpy as np
            vec1 = np.array(self.embeddings)
            vec2 = np.array(other.embeddings)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                return float(np.dot(vec1, vec2) / (norm1 * norm2))
        
        # Similitud basada en nombre y propiedades (fallback)
        similarity = 0.0
        
        # Similitud de nombre
        if self.name.lower() == other.name.lower():
            similarity += 0.3
        elif self.name.lower() in other.name.lower() or other.name.lower() in self.name.lower():
            similarity += 0.15
        
        # Similitud de tipo
        if self.concept_type == other.concept_type:
            similarity += 0.2
        
        # Similitud de propiedades comunes
        common_props = set(self.properties.keys()) & set(other.properties.keys())
        if common_props:
            prop_similarity = 0.0
            for prop in common_props:
                if self.properties[prop] == other.properties[prop]:
                    prop_similarity += 1.0
                elif isinstance(self.properties[prop], str) and isinstance(other.properties[prop], str):
                    # Similitud parcial de strings
                    if self.properties[prop].lower() in other.properties[prop].lower():
                        prop_similarity += 0.5
            
            if common_props:
                similarity += 0.5 * (prop_similarity / len(common_props))
        
        return min(1.0, similarity)

@dataclass
class SemanticRelation:
    """Relación entre conceptos en memoria semántica."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: RelationType = RelationType.RELATED_TO
    weight: float = 1.0  # Peso de la relación (0.0-1.0)
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticRelation':
        """Crea desde diccionario."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            weight=data["weight"],
            confidence=data["confidence"],
            properties=data["properties"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )

class SemanticMemory:
    """
    Sistema de memoria semántica basado en grafos de conocimiento.
    
    Características:
    1. Almacenamiento de conceptos y relaciones semánticas
    2. Inferencia de nuevas relaciones a partir de existentes
    3. Búsqueda por similitud semántica (embeddings)
    4. Razonamiento sobre relaciones transitivas
    5. Consolidación y generalización de conceptos
    6. Validación de consistencia del grafo
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa la memoria semántica.
        
        Args:
            config: Configuración de la memoria
        """
        self.config = config or self._default_config()
        
        # Almacenamiento de conceptos y relaciones
        self.concepts: Dict[str, SemanticConcept] = {}
        self.relations: Dict[str, SemanticRelation] = {}
        
        # Índices para búsqueda rápida
        self.indices = {
            "by_name": defaultdict(list),
            "by_type": defaultdict(list),
            "by_property": defaultdict(lambda: defaultdict(list)),
            "outgoing_relations": defaultdict(list),
            "incoming_relations": defaultdict(list)
        }
        
        # Grafo NetworkX para análisis
        self.graph = nx.MultiDiGraph()
        
        # Estadísticas
        self.stats = {
            "total_concepts": 0,
            "total_relations": 0,
            "concepts_by_type": {ctype.value: 0 for ctype in ConceptType},
            "relations_by_type": {rtype.value: 0 for rtype in RelationType},
            "graph_density": 0.0,
            "avg_degree": 0.0
        }
        
        # Bloqueo para concurrencia
        self._lock = asyncio.Lock()
        
        # Tiempo de inicio
        self.start_time = datetime.now()
        
        # Cargar memoria existente si existe
        self._load_from_disk()
        
        # Inicializar grafo
        self._rebuild_graph()
    
    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto de memoria semántica."""
        return {
            "max_concepts": 100000,
            "max_relations": 1000000,
            "inference_enabled": True,
            "inference_depth": 3,
            "similarity_threshold": 0.7,
            "consolidation_threshold": 0.8,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "auto_infer_interval": 3600,  # Inferir cada hora
            "storage_path": "./data/semantic_memory",
            "enable_graph_analytics": True,
            "default_relation_confidence": 0.8
        }
    
    async def store_concept(
        self,
        name: str,
        concept_type: ConceptType,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        embeddings: Optional[List[float]] = None,
        confidence: float = 1.0,
        source: Optional[str] = None,
        update_existing: bool = True
    ) -> str:
        """
        Almacena un concepto en la memoria semántica.
        
        Args:
            name: Nombre del concepto
            concept_type: Tipo de concepto
            description: Descripción del concepto
            properties: Propiedades adicionales
            embeddings: Vector de embeddings
            confidence: Confianza (0.0-1.0)
            source: Origen del concepto
            update_existing: Si True, actualiza concepto existente con mismo nombre
            
        Returns:
            str: ID del concepto creado/actualizado
        """
        async with self._lock:
            # Verificar si ya existe un concepto con este nombre
            existing_id = None
            if update_existing and name in self.indices["by_name"]:
                existing_ids = self.indices["by_name"][name]
                for cid in existing_ids:
                    if cid in self.concepts:
                        concept = self.concepts[cid]
                        if concept.concept_type == concept_type:
                            existing_id = cid
                            break
            
            if existing_id:
                # Actualizar concepto existente
                concept = self.concepts[existing_id]
                concept.description = description or concept.description
                concept.properties.update(properties or {})
                if embeddings is not None:
                    concept.embeddings = embeddings
                concept.confidence = max(concept.confidence, confidence)
                if source:
                    concept.source = source
                concept.updated_at = datetime.now()
                
                return existing_id
            else:
                # Crear nuevo concepto
                concept = SemanticConcept(
                    name=name,
                    concept_type=concept_type,
                    description=description,
                    properties=properties or {},
                    embeddings=embeddings,
                    confidence=confidence,
                    source=source
                )
                
                # Verificar límite de conceptos
                if len(self.concepts) >= self.config["max_concepts"]:
                    await self._prune_concepts(100)
                
                # Almacenar concepto
                self.concepts[concept.id] = concept
                
                # Actualizar índices
                self._update_concept_indices(concept)
                
                # Actualizar estadísticas
                self.stats["total_concepts"] += 1
                self.stats["concepts_by_type"][concept_type.value] += 1
                
                # Actualizar grafo
                self.graph.add_node(concept.id, **concept.to_dict())
                
                # Guardar periódicamente
                if len(self.concepts) % 100 == 0:
                    self._save_to_disk()
                
                return concept.id
    
    async def retrieve_concept(
        self,
        concept_id: str,
        include_relations: bool = True,
        relation_depth: int = 1,
        include_properties: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Recupera un concepto específico por ID.
        
        Args:
            concept_id: ID del concepto
            include_relations: Si True, incluye relaciones
            relation_depth: Profundidad de relaciones a incluir
            include_properties: Si True, incluye propiedades
            
        Returns:
            Dict con concepto y metadatos, o None si no existe
        """
        async with self._lock:
            if concept_id not in self.concepts:
                return None
            
            concept = self.concepts[concept_id]
            concept.access()  # Registrar acceso
            
            # Preparar respuesta
            result = {
                "concept": {
                    "id": concept.id,
                    "name": concept.name,
                    "type": concept.concept_type.value,
                    "description": concept.description,
                    "confidence": concept.confidence,
                    "source": concept.source,
                    "created_at": concept.created_at.isoformat(),
                    "updated_at": concept.updated_at.isoformat(),
                    "access_count": concept.access_count
                }
            }
            
            # Incluir propiedades si se solicita
            if include_properties and concept.properties:
                result["concept"]["properties"] = concept.properties
            
            # Incluir relaciones si se solicita
            if include_relations:
                relations_info = await self._get_concept_relations(
                    concept_id, 
                    max_depth=relation_depth
                )
                result["relations"] = relations_info
            
            # Incluir estadísticas del concepto
            result["concept_stats"] = self._get_concept_stats(concept_id)
            
            return result
    
    async def link_concepts(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        confidence: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
        bidirectional: bool = False
    ) -> str:
        """
        Crea una relación entre dos conceptos.
        
        Args:
            source_id: ID del concepto origen
            target_id: ID del concepto destino
            relation_type: Tipo de relación
            weight: Peso de la relación (0.0-1.0)
            confidence: Confianza de la relación
            properties: Propiedades adicionales de la relación
            bidirectional: Si True, crea relación inversa también
            
        Returns:
            str: ID de la relación creada
        """
        async with self._lock:
            # Verificar que ambos conceptos existan
            if source_id not in self.concepts or target_id not in self.concepts:
                raise MemoryException("Source or target concept not found")
            
            # Usar confianza por defecto si no se especifica
            if confidence is None:
                confidence = self.config["default_relation_confidence"]
            
            # Crear relación
            relation = SemanticRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                weight=max(0.0, min(1.0, weight)),
                confidence=max(0.0, min(1.0, confidence)),
                properties=properties or {}
            )
            
            # Verificar límite de relaciones
            if len(self.relations) >= self.config["max_relations"]:
                await self._prune_relations(100)
            
            # Almacenar relación
            self.relations[relation.id] = relation
            
            # Actualizar índices
            self._update_relation_indices(relation)
            
            # Actualizar estadísticas
            self.stats["total_relations"] += 1
            self.stats["relations_by_type"][relation_type.value] += 1
            
            # Actualizar grafo
            self.graph.add_edge(
                source_id, 
                target_id, 
                key=relation.id,
                **relation.to_dict()
            )
            
            # Crear relación inversa si se solicita
            if bidirectional:
                inverse_type = self._get_inverse_relation_type(relation_type)
                if inverse_type:
                    inverse_relation = SemanticRelation(
                        source_id=target_id,
                        target_id=source_id,
                        relation_type=inverse_type,
                        weight=weight,
                        confidence=confidence,
                        properties=properties or {}
                    )
                    
                    self.relations[inverse_relation.id] = inverse_relation
                    self._update_relation_indices(inverse_relation)
                    
                    self.stats["total_relations"] += 1
                    self.stats["relations_by_type"][inverse_type.value] += 1
                    
                    self.graph.add_edge(
                        target_id,
                        source_id,
                        key=inverse_relation.id,
                        **inverse_relation.to_dict()
                    )
            
            # Inferir nuevas relaciones si está habilitado
            if self.config["inference_enabled"]:
                await self._infer_new_relations(relation)
            
            # Guardar periódicamente
            if len(self.relations) % 1000 == 0:
                self._save_to_disk()
            
            return relation.id
    
    async def infer_relationships(
        self,
        source_id: str,
        target_id: str,
        max_path_length: int = 3,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Infiere relaciones indirectas entre dos conceptos.
        
        Args:
            source_id: ID del concepto origen
            target_id: ID del concepto destino
            max_path_length: Longitud máxima del camino
            min_confidence: Confianza mínima para inferir
            
        Returns:
            Lista de relaciones inferidas con caminos y confianza
        """
        async with self._lock:
            if source_id not in self.concepts or target_id not in self.concepts:
                return []
            
            # Encontrar todos los caminos entre los conceptos
            try:
                paths = list(nx.all_simple_paths(
                    self.graph, 
                    source_id, 
                    target_id, 
                    cutoff=max_path_length
                ))
            except nx.NetworkXNoPath:
                return []
            
            inferred_relations = []
            
            for path in paths:
                if len(path) < 2:
                    continue
                
                # Calcular relación compuesta
                relation_info = self._compose_relations_along_path(path)
                if relation_info and relation_info["confidence"] >= min_confidence:
                    inferred_relations.append(relation_info)
            
            # Ordenar por confianza descendente
            inferred_relations.sort(key=lambda x: x["confidence"], reverse=True)
            
            return inferred_relations
    
    async def update_concept(
        self,
        concept_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        embeddings: Optional[List[float]] = None,
        confidence_delta: Optional[float] = None
    ) -> bool:
        """
        Actualiza un concepto existente.
        
        Args:
            concept_id: ID del concepto
            name: Nuevo nombre (opcional)
            description: Nueva descripción (opcional)
            properties: Nuevas propiedades (se fusionan con existentes)
            embeddings: Nuevos embeddings (reemplazan existentes)
            confidence_delta: Cambio en confianza (-1.0 a 1.0)
            
        Returns:
            bool: True si se actualizó exitosamente
        """
        async with self._lock:
            if concept_id not in self.concepts:
                return False
            
            concept = self.concepts[concept_id]
            
            # Actualizar campos si se proporcionan
            if name is not None and name != concept.name:
                # Actualizar índice de nombre
                old_name = concept.name
                concept.name = name
                
                if old_name in self.indices["by_name"]:
                    if concept_id in self.indices["by_name"][old_name]:
                        self.indices["by_name"][old_name].remove(concept_id)
                
                self.indices["by_name"][name].append(concept_id)
            
            if description is not None:
                concept.description = description
            
            if properties is not None:
                concept.properties.update(properties)
            
            if embeddings is not None:
                concept.embeddings = embeddings
            
            if confidence_delta is not None:
                concept.update_confidence(confidence_delta)
            
            concept.updated_at = datetime.now()
            
            # Actualizar grafo
            if concept_id in self.graph:
                self.graph.nodes[concept_id].update(concept.to_dict())
            
            return True
    
    async def forget_concept(self, concept_id: str, cascade: bool = True) -> bool:
        """
        Olvida (elimina) un concepto de la memoria.
        
        Args:
            concept_id: ID del concepto
            cascade: Si True, elimina también relaciones asociadas
            
        Returns:
            bool: True si se olvidó exitosamente
        """
        async with self._lock:
            if concept_id not in self.concepts:
                return False
            
            concept = self.concepts[concept_id]
            
            # Eliminar relaciones asociadas si se solicita
            if cascade:
                # Relaciones salientes
                outgoing = self.indices["outgoing_relations"].get(concept_id, [])
                for rel_id in outgoing[:]:  # Copia para modificar durante iteración
                    if rel_id in self.relations:
                        await self._forget_relation(rel_id)
                
                # Relaciones entrantes
                incoming = self.indices["incoming_relations"].get(concept_id, [])
                for rel_id in incoming[:]:
                    if rel_id in self.relations:
                        await self._forget_relation(rel_id)
            
            # Eliminar de índices
            self._remove_concept_from_indices(concept)
            
            # Eliminar concepto
            del self.concepts[concept_id]
            
            # Eliminar del grafo
            if concept_id in self.graph:
                self.graph.remove_node(concept_id)
            
            # Actualizar estadísticas
            self.stats["total_concepts"] -= 1
            self.stats["concepts_by_type"][concept.concept_type.value] -= 1
            
            # Guardar cambios
            self._save_to_disk()
            
            return True
    
    async def get_semantic_network(
        self,
        center_concept_id: Optional[str] = None,
        depth: int = 2,
        relation_types: Optional[List[RelationType]] = None,
        min_confidence: float = 0.3,
        max_concepts: int = 100
    ) -> Dict[str, Any]:
        """
        Obtiene una subred semántica alrededor de un concepto.
        
        Args:
            center_concept_id: ID del concepto central (None = red completa)
            depth: Profundidad máxima desde el centro
            relation_types: Tipos de relaciones a incluir
            min_confidence: Confianza mínima de relaciones
            max_concepts: Máximo de conceptos a incluir
            
        Returns:
            Dict con red semántica y metadatos
        """
        async with self._lock:
            if center_concept_id and center_concept_id not in self.concepts:
                return {"concepts": [], "relations": [], "stats": {}}
            
            # Obtener subgrafo
            if center_concept_id:
                # Encontrar conceptos dentro del radio
                if center_concept_id in self.graph:
                    # Usar BFS para encontrar conceptos cercanos
                    visited = {center_concept_id}
                    queue = [(center_concept_id, 0)]
                    concepts_in_radius = {center_concept_id}
                    
                    while queue:
                        current_id, current_depth = queue.pop(0)
                        
                        if current_depth >= depth:
                            continue
                        
                        # Vecinos salientes
                        for neighbor in self.graph.successors(current_id):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                concepts_in_radius.add(neighbor)
                                queue.append((neighbor, current_depth + 1))
                        
                        # Vecinos entrantes
                        for neighbor in self.graph.predecessors(current_id):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                concepts_in_radius.add(neighbor)
                                queue.append((neighbor, current_depth + 1))
                    
                    # Crear subgrafo
                    subgraph = self.graph.subgraph(concepts_in_radius).copy()
                else:
                    subgraph = nx.MultiDiGraph()
            else:
                # Red completa (limitada)
                all_nodes = list(self.graph.nodes())
                if len(all_nodes) > max_concepts:
                    # Muestra aleatoria de conceptos
                    import random
                    sampled_nodes = random.sample(all_nodes, max_concepts)
                    subgraph = self.graph.subgraph(sampled_nodes).copy()
                else:
                    subgraph = self.graph.copy()
            
            # Filtrar por tipo de relación y confianza
            edges_to_remove = []
            for u, v, key, data in subgraph.edges(data=True, keys=True):
                # Filtrar por tipo
                if relation_types:
                    rel_type = RelationType(data.get("relation_type", ""))
                    if rel_type not in relation_types:
                        edges_to_remove.append((u, v, key))
                        continue
                
                # Filtrar por confianza
                if data.get("confidence", 0.0) < min_confidence:
                    edges_to_remove.append((u, v, key))
            
            # Eliminar aristas que no cumplen los criterios
            for u, v, key in edges_to_remove:
                if subgraph.has_edge(u, v, key):
                    subgraph.remove_edge(u, v, key)
            
            # Eliminar nodos aislados
            isolated = [n for n in subgraph.nodes() if subgraph.degree(n) == 0]
            subgraph.remove_nodes_from(isolated)
            
            # Preparar respuesta
            concepts_data = []
            for node_id, node_data in subgraph.nodes(data=True):
                if node_id in self.concepts:
                    concept = self.concepts[node_id]
                    concepts_data.append({
                        "id": concept.id,
                        "name": concept.name,
                        "type": concept.concept_type.value,
                        "description": concept.description[:100] + "..." if len(concept.description) > 100 else concept.description,
                        "confidence": concept.confidence,
                        "degree": subgraph.degree(node_id)
                    })
            
            relations_data = []
            for u, v, key, data in subgraph.edges(data=True, keys=True):
                relations_data.append({
                    "id": key,
                    "source_id": u,
                    "target_id": v,
                    "type": data.get("relation_type", ""),
                    "weight": data.get("weight", 1.0),
                    "confidence": data.get("confidence", 1.0)
                })
            
            # Estadísticas de la red
            stats = {
                "concept_count": len(concepts_data),
                "relation_count": len(relations_data),
                "density": nx.density(subgraph) if len(subgraph) > 1 else 0.0,
                "avg_degree": sum(d for _, d in subgraph.degree()) / max(len(subgraph), 1),
                "connected_components": nx.number_weakly_connected_components(subgraph)
            }
            
            return {
                "concepts": concepts_data,
                "relations": relations_data,
                "stats": stats,
                "center_concept_id": center_concept_id,
                "depth": depth
            }
    
    async def search_concepts(
        self,
        query: Optional[str] = None,
        concept_type: Optional[ConceptType] = None,
        property_filters: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "confidence",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """
        Busca conceptos que coincidan con los criterios.
        
        Args:
            query: Texto para búsqueda en nombre/descripción
            concept_type: Tipo de concepto
            property_filters: Filtros por propiedades
            min_confidence: Confianza mínima
            limit: Máximo de resultados
            offset: Desplazamiento para paginación
            sort_by: Campo para ordenar (confidence, access_count, name)
            sort_order: Orden (asc, desc)
            
        Returns:
            Dict con resultados de búsqueda y metadatos
        """
        async with self._lock:
            # Filtrar conceptos
            filtered = []
            for concept in self.concepts.values():
                # Filtrar por tipo
                if concept_type and concept.concept_type != concept_type:
                    continue
                
                # Filtrar por confianza
                if concept.confidence < min_confidence:
                    continue
                
                # Filtrar por propiedades
                if property_filters:
                    matches = True
                    for prop_key, prop_value in property_filters.items():
                        if prop_key not in concept.properties:
                            matches = False
                            break
                        if concept.properties[prop_key] != prop_value:
                            matches = False
                            break
                    if not matches:
                        continue
                
                # Búsqueda de texto
                if query:
                    query_lower = query.lower()
                    name_match = query_lower in concept.name.lower()
                    desc_match = query_lower in concept.description.lower()
                    
                    if not name_match and not desc_match:
                        # Buscar en propiedades de texto
                        text_props = [
                            str(v).lower() 
                            for v in concept.properties.values() 
                            if isinstance(v, str)
                        ]
                        prop_match = any(query_lower in prop for prop in text_props)
                        
                        if not prop_match:
                            continue
                
                filtered.append(concept)
            
            # Ordenar resultados
            reverse = sort_order.lower() == "desc"
            if sort_by == "confidence":
                filtered.sort(key=lambda x: x.confidence, reverse=reverse)
            elif sort_by == "access_count":
                filtered.sort(key=lambda x: x.access_count, reverse=reverse)
            elif sort_by == "name":
                filtered.sort(key=lambda x: x.name.lower(), reverse=reverse)
            elif sort_by == "created_at":
                filtered.sort(key=lambda x: x.created_at, reverse=reverse)
            
            # Paginar resultados
            total = len(filtered)
            start_idx = offset
            end_idx = min(offset + limit, total)
            paginated = filtered[start_idx:end_idx]
            
            # Preparar resultados
            results = []
            for concept in paginated:
                results.append({
                    "id": concept.id,
                    "name": concept.name,
                    "type": concept.concept_type.value,
                    "description": concept.description[:200] + "..." if len(concept.description) > 200 else concept.description,
                    "confidence": concept.confidence,
                    "access_count": concept.access_count,
                    "relation_count": len(self.indices["outgoing_relations"].get(concept.id, [])),
                    "created_at": concept.created_at.isoformat()
                })
            
            return {
                "concepts": results,
                "pagination": {
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "has_more": end_idx < total
                },
                "filters_applied": {
                    "query": query,
                    "concept_type": concept_type.value if concept_type else None,
                    "property_filters": property_filters,
                    "min_confidence": min_confidence
                }
            }
    
    # Métodos auxiliares protegidos
    
    def _update_concept_indices(self, concept: SemanticConcept) -> None:
        """Actualiza todos los índices para un concepto."""
        # Índice por nombre
        self.indices["by_name"][concept.name].append(concept.id)
        
        # Índice por tipo
        self.indices["by_type"][concept.concept_type].append(concept.id)
        
        # Índice por propiedades
        for prop_key, prop_value in concept.properties.items():
            if isinstance(prop_value, (str, int, float, bool)):
                self.indices["by_property"][prop_key][str(prop_value)].append(concept.id)
    
    def _update_relation_indices(self, relation: SemanticRelation) -> None:
        """Actualiza todos los índices para una relación."""
        # Índice de relaciones salientes
        self.indices["outgoing_relations"][relation.source_id].append(relation.id)
        
        # Índice de relaciones entrantes
        self.indices["incoming_relations"][relation.target_id].append(relation.id)
    
    def _remove_concept_from_indices(self, concept: SemanticConcept) -> None:
        """Elimina un concepto de todos los índices."""
        # Índice por nombre
        if concept.name in self.indices["by_name"]:
            if concept.id in self.indices["by_name"][concept.name]:
                self.indices["by_name"][concept.name].remove(concept.id)
        
        # Índice por tipo
        if concept.id in self.indices["by_type"][concept.concept_type]:
            self.indices["by_type"][concept.concept_type].remove(concept.id)
        
        # Índice por propiedades
        for prop_key, prop_value in concept.properties.items():
            if isinstance(prop_value, (str, int, float, bool)):
                value_str = str(prop_value)
                if (prop_key in self.indices["by_property"] and 
                    value_str in self.indices["by_property"][prop_key]):
                    if concept.id in self.indices["by_property"][prop_key][value_str]:
                        self.indices["by_property"][prop_key][value_str].remove(concept.id)
    
    def _remove_relation_from_indices(self, relation: SemanticRelation) -> None:
        """Elimina una relación de todos los índices."""
        # Índice de relaciones salientes
        if relation.source_id in self.indices["outgoing_relations"]:
            if relation.id in self.indices["outgoing_relations"][relation.source_id]:
                self.indices["outgoing_relations"][relation.source_id].remove(relation.id)
        
        # Índice de relaciones entrantes
        if relation.target_id in self.indices["incoming_relations"]:
            if relation.id in self.indices["incoming_relations"][relation.target_id]:
                self.indices["incoming_relations"][relation.target_id].remove(relation.id)
    
    async def _get_concept_relations(
        self, 
        concept_id: str, 
        max_depth: int = 1
    ) -> Dict[str, Any]:
        """Obtiene relaciones de un concepto hasta cierta profundidad."""
        if concept_id not in self.concepts:
            return {}
        
        result = {
            "outgoing": [],
            "incoming": [],
            "related_concepts": set()
        }
        
        # Relaciones salientes (depth 1)
        outgoing_ids = self.indices["outgoing_relations"].get(concept_id, [])
        for rel_id in outgoing_ids:
            if rel_id in self.relations:
                relation = self.relations[rel_id]
                result["outgoing"].append(relation.to_dict())
                result["related_concepts"].add(relation.target_id)
        
        # Relaciones entrantes (depth 1)
        incoming_ids = self.indices["incoming_relations"].get(concept_id, [])
        for rel_id in incoming_ids:
            if rel_id in self.relations:
                relation = self.relations[rel_id]
                result["incoming"].append(relation.to_dict())
                result["related_concepts"].add(relation.source_id)
        
        # Profundidades mayores (recursivamente)
        if max_depth > 1:
            visited = {concept_id}
            for related_id in list(result["related_concepts"]):
                if related_id not in visited:
                    visited.add(related_id)
                    deeper_relations = await self._get_concept_relations(
                        related_id, 
                        max_depth - 1
                    )
                    
                    # Combinar resultados
                    result["outgoing"].extend(deeper_relations.get("outgoing", []))
                    result["incoming"].extend(deeper_relations.get("incoming", []))
                    result["related_concepts"].update(deeper_relations.get("related_concepts", set()))
        
        # Convertir set a lista
        result["related_concepts"] = list(result["related_concepts"])
        
        return result
    
    def _get_concept_stats(self, concept_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas de un concepto."""
        stats = {
            "outgoing_relations": 0,
            "incoming_relations": 0,
            "total_relations": 0,
            "centrality": 0.0
        }
        
        if concept_id in self.concepts:
            stats["outgoing_relations"] = len(self.indices["outgoing_relations"].get(concept_id, []))
            stats["incoming_relations"] = len(self.indices["incoming_relations"].get(concept_id, []))
            stats["total_relations"] = stats["outgoing_relations"] + stats["incoming_relations"]
            
            # Calcular centralidad aproximada
            if concept_id in self.graph:
                try:
                    centrality = nx.degree_centrality(self.graph)
                    stats["centrality"] = centrality.get(concept_id, 0.0)
                except:
                    pass
        
        return stats
    
    def _get_inverse_relation_type(self, relation_type: RelationType) -> Optional[RelationType]:
        """Obtiene el tipo inverso de una relación."""
        inverse_map = {
            RelationType.IS_A: None,  # No tiene inverso claro
            RelationType.PART_OF: RelationType.HAS_A,
            RelationType.HAS_A: RelationType.PART_OF,
            RelationType.USES: None,  # No tiene inverso claro
            RelationType.DEPENDS_ON: None,  # No tiene inverso claro
            RelationType.SIMILAR_TO: RelationType.SIMILAR_TO,  # Simétrico
            RelationType.OPPOSITE_OF: RelationType.OPPOSITE_OF,  # Simétrico
            RelationType.EXAMPLE_OF: None,  # No tiene inverso claro
            RelationType.IMPLIES: None  # No tiene inverso claro
        }
        
        return inverse_map.get(relation_type)
    
    async def _infer_new_relations(self, new_relation: SemanticRelation) -> None:
        """Infere nuevas relaciones a partir de una relación nueva."""
        if not self.config["inference_enabled"]:
            return
        
        inferences = []
        
        # Reglas de inferencia basadas en tipo de relación
        if new_relation.relation_type == RelationType.IS_A:
            # Transitividad de IS_A
            # Si A IS_A B y B IS_A C, entonces A IS_A C
            b_concept_id = new_relation.target_id
            
            # Buscar relaciones donde b_concept_id es fuente de IS_A
            for rel_id in self.indices["outgoing_relations"].get(b_concept_id, []):
                if rel_id in self.relations:
                    rel = self.relations[rel_id]
                    if rel.relation_type == RelationType.IS_A:
                        # Inferir nueva relación
                        inferred_conf = min(
                            new_relation.confidence,
                            rel.confidence,
                            0.9  # Límite superior para inferencias
                        )
                        
                        inferences.append({
                            "source_id": new_relation.source_id,
                            "target_id": rel.target_id,
                            "relation_type": RelationType.IS_A,
                            "confidence": inferred_conf * 0.8,  # Penalizar inferencias
                            "inferred_from": [new_relation.id, rel.id]
                        })
        
        elif new_relation.relation_type == RelationType.PART_OF:
            # Transitividad de PART_OF
            # Si A PART_OF B y B PART_OF C, entonces A PART_OF C
            b_concept_id = new_relation.target_id
            
            for rel_id in self.indices["outgoing_relations"].get(b_concept_id, []):
                if rel_id in self.relations:
                    rel = self.relations[rel_id]
                    if rel.relation_type == RelationType.PART_OF:
                        inferred_conf = min(
                            new_relation.confidence,
                            rel.confidence,
                            0.9
                        )
                        
                        inferences.append({
                            "source_id": new_relation.source_id,
                            "target_id": rel.target_id,
                            "relation_type": RelationType.PART_OF,
                            "confidence": inferred_conf * 0.7,
                            "inferred_from": [new_relation.id, rel.id]
                        })
        
        # Crear relaciones inferidas
        for inference in inferences:
            # Verificar si ya existe una relación directa
            existing = False
            source_id = inference["source_id"]
            target_id = inference["target_id"]
            rel_type = inference["relation_type"]
            
            for rel_id in self.indices["outgoing_relations"].get(source_id, []):
                if rel_id in self.relations:
                    rel = self.relations[rel_id]
                    if (rel.target_id == target_id and 
                        rel.relation_type == rel_type):
                        existing = True
                        break
            
            if not existing:
                # Crear relación inferida
                await self.link_concepts(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=rel_type,
                    weight=0.5,  # Peso reducido para inferencias
                    confidence=inference["confidence"],
                    properties={
                        "inferred": True,
                        "inferred_from": inference["inferred_from"],
                        "inference_rule": "transitivity"
                    }
                )
    
    def _compose_relations_along_path(self, path: List[str]) -> Optional[Dict[str, Any]]:
        """Compre relaciones a lo largo de un camino en el grafo."""
        if len(path) < 2:
            return None
        
        # Recolectar relaciones a lo largo del camino
        path_relations = []
        total_confidence = 1.0
        
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            # Buscar relación entre estos nodos
            relation_found = False
            for rel_id in self.indices["outgoing_relations"].get(source_id, []):
                if rel_id in self.relations:
                    rel = self.relations[rel_id]
                    if rel.target_id == target_id:
                        path_relations.append(rel)
                        total_confidence *= rel.confidence
                        relation_found = True
                        break
            
            if not relation_found:
                return None
        
        if not path_relations:
            return None
        
        # Calcular tipo de relación compuesta
        # Por simplicidad, usar el tipo más común o el primero
        rel_types = [rel.relation_type for rel in path_relations]
        from collections import Counter
        type_counts = Counter(rel_types)
        most_common_type = type_counts.most_common(1)[0][0]
        
        # Calcular confianza promedio geométrica
        avg_confidence = total_confidence ** (1.0 / len(path_relations))
        
        return {
            "source_id": path[0],
            "target_id": path[-1],
            "relation_type": most_common_type,
            "confidence": avg_confidence,
            "path_length": len(path) - 1,
            "path": path,
            "path_relations": [rel.id for rel in path_relations]
        }
    
    async def _prune_concepts(self, count: int = 100) -> List[str]:
        """Podar conceptos menos importantes."""
        # Calcular score de importancia para cada concepto
        concept_scores = []
        current_time = datetime.now()
        
        for concept in self.concepts.values():
            # Factor de acceso
            access_factor = concept.access_count / max((current_time - concept.created_at).total_seconds(), 1.0)
            
            # Factor de confianza
            confidence_factor = concept.confidence
            
            # Factor de conectividad
            out_degree = len(self.indices["outgoing_relations"].get(concept.id, []))
            in_degree = len(self.indices["incoming_relations"].get(concept.id, []))
            connectivity_factor = (out_degree + in_degree) / 10.0  # Normalizar
            
            # Score combinado (más alto = más importante)
            importance_score = (
                0.4 * confidence_factor +
                0.3 * access_factor +
                0.3 * min(connectivity_factor, 1.0)
            )
            
            concept_scores.append((concept.id, importance_score, concept))
        
        # Ordenar por score (más bajo primero)
        concept_scores.sort(key=lambda x: x[1])
        
        # Podar los primeros 'count' conceptos
        pruned = []
        for i in range(min(count, len(concept_scores))):
            concept_id, score, concept = concept_scores[i]
            if await self.forget_concept(concept_id, cascade=True):
                pruned.append(concept_id)
        
        return pruned
    
    async def _prune_relations(self, count: int = 100) -> List[str]:
        """Podar relaciones menos importantes."""
        # Calcular score de importancia para cada relación
        relation_scores = []
        
        for relation in self.relations.values():
            # Factor de confianza
            confidence_factor = relation.confidence
            
            # Factor de peso
            weight_factor = relation.weight
            
            # Factor de antigüedad
            age_days = (datetime.now() - relation.created_at).days
            age_factor = max(0.0, 1.0 - (age_days / 365.0))  # Decae en 1 año
            
            # Score combinado (más alto = más importante)
            importance_score = (
                0.5 * confidence_factor +
                0.3 * weight_factor +
                0.2 * age_factor
            )
            
            relation_scores.append((relation.id, importance_score, relation))
        
        # Ordenar por score (más bajo primero)
        relation_scores.sort(key=lambda x: x[1])
        
        # Podar las primeras 'count' relaciones
        pruned = []
        for i in range(min(count, len(relation_scores))):
            rel_id, score, relation = relation_scores[i]
            if await self._forget_relation(rel_id):
                pruned.append(rel_id)
        
        return pruned
    
    async def _forget_relation(self, relation_id: str) -> bool:
        """Olvida (elimina) una relación."""
        if relation_id not in self.relations:
            return False
        
        relation = self.relations[relation_id]
        
        # Eliminar de índices
        self._remove_relation_from_indices(relation)
        
        # Eliminar relación
        del self.relations[relation_id]
        
        # Eliminar del grafo
        if (relation.source_id in self.graph and 
            relation.target_id in self.graph):
            if self.graph.has_edge(relation.source_id, relation.target_id, relation_id):
                self.graph.remove_edge(relation.source_id, relation.target_id, relation_id)
        
        # Actualizar estadísticas
        self.stats["total_relations"] -= 1
        self.stats["relations_by_type"][relation.relation_type.value] -= 1
        
        return True
    
    def _rebuild_graph(self) -> None:
        """Reconstruye el grafo NetworkX desde conceptos y relaciones."""
        self.graph.clear()
        
        # Añadir nodos (conceptos)
        for concept_id, concept in self.concepts.items():
            self.graph.add_node(concept_id, **concept.to_dict())
        
        # Añadir aristas (relaciones)
        for relation_id, relation in self.relations.items():
            if (relation.source_id in self.graph and 
                relation.target_id in self.graph):
                self.graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    key=relation_id,
                    **relation.to_dict()
                )
        
        # Actualizar estadísticas del grafo
        if len(self.graph) > 0:
            self.stats["graph_density"] = nx.density(self.graph)
            degrees = [d for _, d in self.graph.degree()]
            self.stats["avg_degree"] = sum(degrees) / len(degrees)
    
    def _save_to_disk(self) -> None:
        """Guarda la memoria semántica en disco."""
        try:
            save_path = Path(self.config["storage_path"])
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Guardar conceptos
            concepts_file = save_path / "concepts.json"
            concepts_data = {
                concept_id: concept.to_dict()
                for concept_id, concept in self.concepts.items()
            }
            
            with open(concepts_file, 'w') as f:
                json.dump(concepts_data, f, indent=2)
            
            # Guardar relaciones
            relations_file = save_path / "relations.json"
            relations_data = {
                relation_id: relation.to_dict()
                for relation_id, relation in self.relations.items()
            }
            
            with open(relations_file, 'w') as f:
                json.dump(relations_data, f, indent=2)
            
            # Guardar índices
            indices_file = save_path / "indices.json"
            indices_data = {}
            for index_name, index_dict in self.indices.items():
                if isinstance(index_dict, defaultdict):
                    indices_data[index_name] = {
                        key: list(values) for key, values in index_dict.items()
                    }
                else:
                    indices_data[index_name] = dict(index_dict)
            
            with open(indices_file, 'w') as f:
                json.dump(indices_data, f, indent=2)
            
            # Guardar estadísticas
            stats_file = save_path / "stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            # Guardar grafo (formato GraphML)
            graph_file = save_path / "graph.graphml"
            nx.write_graphml(self.graph, graph_file)
            
        except Exception as e:
            print(f"Failed to save semantic memory: {e}")
    
    def _load_from_disk(self) -> None:
        """Carga la memoria semántica desde disco."""
        try:
            save_path = Path(self.config["storage_path"])
            
            # Cargar conceptos
            concepts_file = save_path / "concepts.json"
            if concepts_file.exists():
                with open(concepts_file, 'r') as f:
                    concepts_data = json.load(f)
                
                for concept_id, concept_dict in concepts_data.items():
                    concept = SemanticConcept.from_dict(concept_dict)
                    self.concepts[concept_id] = concept
            
            # Cargar relaciones
            relations_file = save_path / "relations.json"
            if relations_file.exists():
                with open(relations_file, 'r') as f:
                    relations_data = json.load(f)
                
                for relation_id, relation_dict in relations_data.items():
                    relation = SemanticRelation.from_dict(relation_dict)
                    self.relations[relation_id] = relation
            
            # Cargar índices
            indices_file = save_path / "indices.json"
            if indices_file.exists():
                with open(indices_file, 'r') as f:
                    indices_data = json.load(f)
                
                for index_name, index_dict in indices_data.items():
                    if index_name in self.indices:
                        self.indices[index_name].clear()
                        if isinstance(self.indices[index_name], defaultdict):
                            for key, values in index_dict.items():
                                self.indices[index_name][key] = values
                        else:
                            self.indices[index_name].update(index_dict)
            
            # Cargar estadísticas
            stats_file = save_path / "stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
            
            # Cargar grafo
            graph_file = save_path / "graph.graphml"
            if graph_file.exists():
                self.graph = nx.read_graphml(graph_file)
            else:
                self._rebuild_graph()
            
            # Recalcular estadísticas si es necesario
            if not self.stats.get("total_concepts"):
                self.stats["total_concepts"] = len(self.concepts)
                self.stats["total_relations"] = len(self.relations)
                
                # Recalcular distribuciones
                self.stats["concepts_by_type"] = {ctype.value: 0 for ctype in ConceptType}
                for concept in self.concepts.values():
                    self.stats["concepts_by_type"][concept.concept_type.value] += 1
                
                self.stats["relations_by_type"] = {rtype.value: 0 for rtype in RelationType}
                for relation in self.relations.values():
                    self.stats["relations_by_type"][relation.relation_type.value] += 1
                
                # Recalcular métricas del grafo
                self._rebuild_graph()
            
        except Exception as e:
            print(f"Failed to load semantic memory: {e}")
            # Inicializar estructuras vacías
            self.concepts.clear()
            self.relations.clear()
            self.indices.clear()
            self.graph.clear()
            self._rebuild_graph()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de la memoria semántica."""
        stats = self.stats.copy()
        
        # Agregar detalles adicionales
        stats["concept_type_distribution"] = stats["concepts_by_type"]
        stats["relation_type_distribution"] = stats["relations_by_type"]
        
        # Calcular métricas avanzadas del grafo
        if len(self.graph) > 0:
            try:
                # Centralidad
                degree_centrality = nx.degree_centrality(self.graph)
                if degree_centrality:
                    stats["max_degree_centrality"] = max(degree_centrality.values())
                    stats["avg_degree_centrality"] = sum(degree_centrality.values()) / len(degree_centrality)
                
                # Componentes conectados
                stats["weakly_connected_components"] = nx.number_weakly_connected_components(self.graph)
                stats["strongly_connected_components"] = nx.number_strongly_connected_components(self.graph)
                
                # Diámetro del grafo (aproximado para grafos grandes)
                if len(self.graph) < 1000:
                    try:
                        if nx.is_weakly_connected(self.graph):
                            stats["diameter"] = nx.diameter(self.graph.to_undirected())
                    except:
                        stats["diameter"] = "N/A (graph too large or disconnected)"
                else:
                    stats["diameter"] = "N/A (graph too large)"
                
            except Exception as e:
                stats["graph_metrics_error"] = str(e)
        
        # Distribución de confianza
        if self.concepts:
            confidence_values = [c.confidence for c in self.concepts.values()]
            stats["confidence_distribution"] = {
                "min": min(confidence_values),
                "max": max(confidence_values),
                "avg": sum(confidence_values) / len(confidence_values),
                "std": (
                    sum((c - stats["avg_confidence"]) ** 2 for c in confidence_values) / 
                    max(len(confidence_values), 1)
                ) ** 0.5
            }
        
        # Uptime
        stats["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()
        
        return stats