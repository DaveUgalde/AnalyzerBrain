"""
GraphTraverser - Algoritmos de recorrido de grafos.
Implementa BFS, DFS, Dijkstra, A*, y otros algoritmos de recorrido.
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass
import heapq
from collections import deque, defaultdict
from datetime import datetime
from .knowledge_graph import KnowledgeGraph, Node, Relationship, RelationshipType

@dataclass
class TraversalResult:
    """Resultado de un recorrido."""
    nodes_visited: List[Node]
    relationships_used: List[Relationship]
    total_distance: float = 0.0
    depth_reached: int = 0
    execution_time_ms: float = 0.0

@dataclass
class PathResult:
    """Resultado de búsqueda de camino."""
    path: List[Node]
    relationships: List[Relationship]
    total_cost: float = 0.0
    nodes_explored: int = 0
    found: bool = False

class GraphTraverser:
    """
    Algoritmos de recorrido y búsqueda en grafos.
    """
    
    def __init__(self, graph: KnowledgeGraph):
        """
        Inicializa el traverser.
        
        Args:
            graph: Grafo a recorrer
        """
        self.graph = graph
    
    def breadth_first_search(self, 
                            start_node_id: str,
                            max_depth: int = 10,
                            relationship_types: Optional[List[RelationshipType]] = None,
                            node_filter: Optional[Callable[[Node], bool]] = None) -> TraversalResult:
        """
        Recorrido en anchura (BFS) desde un nodo.
        
        Args:
            start_node_id: ID del nodo de inicio
            max_depth: Profundidad máxima
            relationship_types: Tipos de relaciones a considerar
            node_filter: Función para filtrar nodos
            
        Returns:
            Resultado del recorrido BFS
        """
        start_time = datetime.now()
        
        if start_node_id not in self.graph.nodes:
            return TraversalResult([], [], 0, 0, 0)
        
        visited = set()
        queue = deque([(start_node_id, 0)])
        nodes_visited = []
        relationships_used = []
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            current_node = self.graph.nodes[current_id]
            
            # Aplicar filtro si existe
            if node_filter and not node_filter(current_node):
                continue
            
            nodes_visited.append(current_node)
            
            # Obtener vecinos
            neighbors = self.graph.get_neighbors(
                current_id, 
                "outgoing", 
                relationship_types
            )
            
            for neighbor_node, rel in neighbors:
                if neighbor_node.id not in visited:
                    queue.append((neighbor_node.id, depth + 1))
                    relationships_used.append(rel)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return TraversalResult(
            nodes_visited=nodes_visited,
            relationships_used=relationships_used,
            depth_reached=max_depth,
            execution_time_ms=execution_time
        )
    
    def depth_first_search(self,
                          start_node_id: str,
                          max_depth: int = 10,
                          relationship_types: Optional[List[RelationshipType]] = None,
                          node_filter: Optional[Callable[[Node], bool]] = None) -> TraversalResult:
        """
        Recorrido en profundidad (DFS) desde un nodo.
        
        Args:
            start_node_id: ID del nodo de inicio
            max_depth: Profundidad máxima
            relationship_types: Tipos de relaciones a considerar
            node_filter: Función para filtrar nodos
            
        Returns:
            Resultado del recorrido DFS
        """
        start_time = datetime.now()
        
        if start_node_id not in self.graph.nodes:
            return TraversalResult([], [], 0, 0, 0)
        
        visited = set()
        stack = [(start_node_id, 0)]
        nodes_visited = []
        relationships_used = []
        
        while stack:
            current_id, depth = stack.pop()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            current_node = self.graph.nodes[current_id]
            
            # Aplicar filtro si existe
            if node_filter and not node_filter(current_node):
                continue
            
            nodes_visited.append(current_node)
            
            # Obtener vecinos (en orden inverso para mantener orden natural)
            neighbors = self.graph.get_neighbors(
                current_id, 
                "outgoing", 
                relationship_types
            )
            
            # Añadir en orden inverso para procesar en el orden correcto
            for neighbor_node, rel in reversed(neighbors):
                if neighbor_node.id not in visited:
                    stack.append((neighbor_node.id, depth + 1))
                    relationships_used.append(rel)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return TraversalResult(
            nodes_visited=nodes_visited,
            relationships_used=relationships_used,
            depth_reached=max_depth,
            execution_time_ms=execution_time
        )
    
    def dijkstra_shortest_path(self,
                              start_node_id: str,
                              end_node_id: str,
                              weight_property: str = "weight",
                              default_weight: float = 1.0) -> PathResult:
        """
        Camino más corto entre dos nodos usando el algoritmo de Dijkstra.
        
        Args:
            start_node_id: ID del nodo de inicio
            end_node_id: ID del nodo de destino
            weight_property: Propiedad de la relación que contiene el peso
            default_weight: Peso por defecto si no se especifica
            
        Returns:
            Resultado de la búsqueda de camino
        """
        start_time = datetime.now()
        
        if (start_node_id not in self.graph.nodes or 
            end_node_id not in self.graph.nodes):
            return PathResult([], [], 0, 0, False)
        
        # Inicializar distancias
        distances = {node_id: float('inf') for node_id in self.graph.nodes}
        distances[start_node_id] = 0
        
        # Predecesores para reconstruir el camino
        predecessors = {node_id: None for node_id in self.graph.nodes}
        
        # Cola de prioridad
        pq = [(0, start_node_id)]
        nodes_explored = 0
        
        while pq:
            current_dist, current_id = heapq.heappop(pq)
            nodes_explored += 1
            
            # Si ya encontramos un camino mejor, ignorar
            if current_dist > distances[current_id]:
                continue
            
            # Si llegamos al destino, reconstruir camino
            if current_id == end_node_id:
                # Reconstruir camino
                path = []
                relationships = []
                node_id = end_node_id
                
                while node_id is not None:
                    path.append(self.graph.nodes[node_id])
                    
                    prev_id = predecessors[node_id]
                    if prev_id is not None:
                        # Encontrar relación entre prev_id y node_id
                        rels = self._find_relationship(prev_id, node_id)
                        if rels:
                            relationships.append(rels[0])
                    
                    node_id = prev_id
                
                path.reverse()
                relationships.reverse()
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return PathResult(
                    path=path,
                    relationships=relationships,
                    total_cost=current_dist,
                    nodes_explored=nodes_explored,
                    found=True
                )
            
            # Explorar vecinos
            for rel_id in self.graph._node_relationships[current_id]["outgoing"]:
                rel = self.graph.relationships[rel_id]
                neighbor_id = rel.target_id
                
                # Obtener peso de la relación
                weight = rel.properties.get(weight_property, default_weight)
                if not isinstance(weight, (int, float)):
                    weight = default_weight
                
                # Calcular nueva distancia
                new_dist = current_dist + weight
                
                if new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    predecessors[neighbor_id] = current_id
                    heapq.heappush(pq, (new_dist, neighbor_id))
        
        # No se encontró camino
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PathResult(
            path=[],
            relationships=[],
            total_cost=0,
            nodes_explored=nodes_explored,
            found=False
        )
    
    def a_star_search(self,
                     start_node_id: str,
                     end_node_id: str,
                     heuristic_func: Callable[[str, str], float],
                     weight_property: str = "weight",
                     default_weight: float = 1.0) -> PathResult:
        """
        Búsqueda A* entre dos nodos.
        
        Args:
            start_node_id: ID del nodo de inicio
            end_node_id: ID del nodo de destino
            heuristic_func: Función heurística que estima la distancia entre dos nodos
            weight_property: Propiedad de la relación que contiene el peso
            default_weight: Peso por defecto si no se especifica
            
        Returns:
            Resultado de la búsqueda de camino
        """
        start_time = datetime.now()
        
        if (start_node_id not in self.graph.nodes or 
            end_node_id not in self.graph.nodes):
            return PathResult([], [], 0, 0, False)
        
        # Inicializar
        open_set = [(0, start_node_id)]  # (f_score, node_id)
        came_from = {}
        
        g_score = {node_id: float('inf') for node_id in self.graph.nodes}
        g_score[start_node_id] = 0
        
        f_score = {node_id: float('inf') for node_id in self.graph.nodes}
        f_score[start_node_id] = heuristic_func(start_node_id, end_node_id)
        
        nodes_explored = 0
        
        while open_set:
            _, current_id = heapq.heappop(open_set)
            nodes_explored += 1
            
            if current_id == end_node_id:
                # Reconstruir camino
                path = []
                relationships = []
                node_id = end_node_id
                
                while node_id in came_from:
                    path.append(self.graph.nodes[node_id])
                    
                    prev_id = came_from[node_id]
                    if prev_id is not None:
                        # Encontrar relación entre prev_id y node_id
                        rels = self._find_relationship(prev_id, node_id)
                        if rels:
                            relationships.append(rels[0])
                    
                    node_id = prev_id
                
                # Añadir nodo inicial
                path.append(self.graph.nodes[start_node_id])
                path.reverse()
                relationships.reverse()
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return PathResult(
                    path=path,
                    relationships=relationships,
                    total_cost=g_score[end_node_id],
                    nodes_explored=nodes_explored,
                    found=True
                )
            
            for rel_id in self.graph._node_relationships[current_id]["outgoing"]:
                rel = self.graph.relationships[rel_id]
                neighbor_id = rel.target_id
                
                # Calcular g_score tentativo
                weight = rel.properties.get(weight_property, default_weight)
                if not isinstance(weight, (int, float)):
                    weight = default_weight
                
                tentative_g_score = g_score[current_id] + weight
                
                if tentative_g_score < g_score[neighbor_id]:
                    came_from[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g_score
                    f_score[neighbor_id] = g_score[neighbor_id] + heuristic_func(neighbor_id, end_node_id)
                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
        
        # No se encontró camino
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PathResult(
            path=[],
            relationships=[],
            total_cost=0,
            nodes_explored=nodes_explored,
            found=False
        )
    
    def find_connected_components(self) -> List[List[Node]]:
        """
        Encuentra componentes conectados en el grafo.
        
        Returns:
            Lista de componentes, cada componente es una lista de nodos
        """
        visited = set()
        components = []
        
        for node_id in self.graph.nodes:
            if node_id not in visited:
                # BFS para encontrar el componente
                component_nodes = []
                queue = [node_id]
                
                while queue:
                    current_id = queue.pop(0)
                    if current_id in visited:
                        continue
                    
                    visited.add(current_id)
                    component_nodes.append(self.graph.nodes[current_id])
                    
                    # Añadir vecinos (tanto entrantes como salientes)
                    for rel_id in self.graph._node_relationships[current_id]["outgoing"]:
                        rel = self.graph.relationships[rel_id]
                        if rel.target_id not in visited:
                            queue.append(rel.target_id)
                    
                    for rel_id in self.graph._node_relationships[current_id]["incoming"]:
                        rel = self.graph.relationships[rel_id]
                        if rel.source_id not in visited:
                            queue.append(rel.source_id)
                
                if component_nodes:
                    components.append(component_nodes)
        
        return components
    
    def detect_cycles(self) -> List[List[Node]]:
        """
        Detecta ciclos en el grafo.
        
        Returns:
            Lista de ciclos, cada ciclo es una lista de nodos
        """
        # Usamos DFS para detectar ciclos
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs_cycle(current_id: str, path: List[Node]) -> None:
            visited.add(current_id)
            rec_stack.add(current_id)
            path.append(self.graph.nodes[current_id])
            
            for rel_id in self.graph._node_relationships[current_id]["outgoing"]:
                rel = self.graph.relationships[rel_id]
                neighbor_id = rel.target_id
                
                if neighbor_id not in visited:
                    dfs_cycle(neighbor_id, path)
                elif neighbor_id in rec_stack:
                    # Encontramos un ciclo
                    # Encontrar la posición del nodo en el camino
                    cycle_start = -1
                    for i, node in enumerate(path):
                        if node.id == neighbor_id:
                            cycle_start = i
                            break
                    
                    if cycle_start >= 0:
                        cycle = path[cycle_start:]
                        cycles.append(cycle.copy())
            
            # Backtrack
            rec_stack.remove(current_id)
            path.pop()
        
        for node_id in self.graph.nodes:
            if node_id not in visited:
                dfs_cycle(node_id, [])
        
        # Eliminar ciclos duplicados
        unique_cycles = []
        seen = set()
        
        for cycle in cycles:
            # Ordenar ciclo para comparación
            cycle_ids = tuple(sorted(node.id for node in cycle))
            if cycle_ids not in seen:
                seen.add(cycle_ids)
                unique_cycles.append(cycle)
        
        return unique_cycles
    
    def calculate_centrality(self, centrality_type: str = "degree") -> Dict[str, float]:
        """
        Calcula centralidad de los nodos.
        
        Args:
            centrality_type: Tipo de centralidad
            
        Returns:
            Diccionario con ID de nodo -> valor de centralidad
        """
        if centrality_type == "degree":
            return self._degree_centrality()
        elif centrality_type == "betweenness":
            return self._betweenness_centrality()
        elif centrality_type == "closeness":
            return self._closeness_centrality()
        elif centrality_type == "eigenvector":
            return self._eigenvector_centrality()
        else:
            raise ValueError(f"Unsupported centrality type: {centrality_type}")
    
    def _degree_centrality(self) -> Dict[str, float]:
        """Centralidad por grado."""
        centrality = {}
        total_nodes = len(self.graph.nodes)
        
        if total_nodes <= 1:
            return {node_id: 0 for node_id in self.graph.nodes}
        
        for node_id in self.graph.nodes:
            # Grado = número de conexiones (entrantes + salientes)
            degree = (
                len(self.graph._node_relationships[node_id]["incoming"]) +
                len(self.graph._node_relationships[node_id]["outgoing"])
            )
            centrality[node_id] = degree / (total_nodes - 1)
        
        return centrality
    
    def _betweenness_centrality(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """Centralidad de intermediación (betweenness)."""
        centrality = {node_id: 0.0 for node_id in self.graph.nodes}
        nodes = list(self.graph.nodes.keys())
        
        if sample_size and sample_size < len(nodes):
            import random
            nodes = random.sample(nodes, sample_size)
        
        # Para cada par de nodos (fuente, destino)
        for i, source_id in enumerate(nodes):
            # Dijkstra desde source_id a todos los demás nodos
            distances, predecessors = self._dijkstra_all(source_id)
            
            # Contar caminos más cortos que pasan por cada nodo
            for target_id in nodes:
                if source_id == target_id:
                    continue
                
                # Reconstruir camino más corto
                path = []
                current_id = target_id
                
                while current_id is not None and current_id in predecessors:
                    path.append(current_id)
                    current_id = predecessors[current_id]
                
                # Si hay un camino, contar nodos intermedios
                if path and path[-1] == source_id:
                    # Nodos intermedios (excluyendo source y target)
                    for node_id in path[1:-1]:
                        centrality[node_id] += 1
        
        # Normalizar
        if len(nodes) > 2:
            normalization = (len(nodes) - 1) * (len(nodes) - 2)
            for node_id in centrality:
                centrality[node_id] /= normalization
        
        return centrality
    
    def _closeness_centrality(self) -> Dict[str, float]:
        """Centralidad de cercanía (closeness)."""
        centrality = {}
        
        for source_id in self.graph.nodes:
            # Dijkstra desde source_id a todos los demás nodos
            distances, _ = self._dijkstra_all(source_id)
            
            # Sumar distancias a todos los demás nodos alcanzables
            total_distance = 0
            reachable_count = 0
            
            for target_id, dist in distances.items():
                if source_id != target_id and dist < float('inf'):
                    total_distance += dist
                    reachable_count += 1
            
            if reachable_count > 0 and total_distance > 0:
                centrality[source_id] = reachable_count / total_distance
            else:
                centrality[source_id] = 0
        
        return centrality
    
    def _eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
        """Centralidad de vector propio (eigenvector)."""
        import numpy as np
        
        # Crear matriz de adyacencia
        node_ids = list(self.graph.nodes.keys())
        node_index = {node_id: i for i, node_id in enumerate(node_ids)}
        n = len(node_ids)
        
        A = np.zeros((n, n))
        
        for rel in self.graph.relationships.values():
            i = node_index.get(rel.source_id)
            j = node_index.get(rel.target_id)
            if i is not None and j is not None:
                A[i, j] = 1  # Grafo dirigido
        
        # Algoritmo de potencia
        x = np.ones(n) / np.sqrt(n)
        
        for _ in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm == 0:
                break
            
            x_new = x_new / norm
            
            if np.linalg.norm(x_new - x) < tol:
                break
            
            x = x_new
        
        # Convertir a diccionario
        centrality = {}
        for node_id, idx in node_index.items():
            centrality[node_id] = float(x[idx])
        
        return centrality
    
    def _dijkstra_all(self, start_id: str, weight_property: str = "weight") -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        """Dijkstra desde un nodo a todos los demás."""
        distances = {node_id: float('inf') for node_id in self.graph.nodes}
        distances[start_id] = 0
        predecessors = {node_id: None for node_id in self.graph.nodes}
        
        pq = [(0, start_id)]
        
        while pq:
            current_dist, current_id = heapq.heappop(pq)
            
            if current_dist > distances[current_id]:
                continue
            
            for rel_id in self.graph._node_relationships[current_id]["outgoing"]:
                rel = self.graph.relationships[rel_id]
                neighbor_id = rel.target_id
                
                weight = rel.properties.get(weight_property, 1.0)
                if not isinstance(weight, (int, float)):
                    weight = 1.0
                
                new_dist = current_dist + weight
                
                if new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    predecessors[neighbor_id] = current_id
                    heapq.heappush(pq, (new_dist, neighbor_id))
        
        return distances, predecessors
    
    def _find_relationship(self, source_id: str, target_id: str) -> List[Relationship]:
        """Encuentra relaciones entre dos nodos."""
        relationships = []
        
        for rel_id in self.graph._node_relationships[source_id]["outgoing"]:
            rel = self.graph.relationships[rel_id]
            if rel.target_id == target_id:
                relationships.append(rel)
        
        return relationships
    
    # Métodos adicionales especificados en ArquiFunciones.txt
    
    def find_connected_components(self) -> List[List[Node]]:
        """
        Encuentra componentes conectados en el grafo.
        
        Returns:
            Lista de componentes, cada componente es una lista de nodos
        """
        return self.find_connected_components()
    
    def detect_cycles(self) -> List[List[Node]]:
        """
        Detecta ciclos en el grafo.
        
        Returns:
            Lista de ciclos, cada ciclo es una lista de nodos
        """
        return self.detect_cycles()
    
    def calculate_centrality(self) -> Dict[str, float]:
        """
        Calcula centralidad de los nodos.
        
        Returns:
            Diccionario con ID de nodo -> valor de centralidad
        """
        return self.calculate_centrality("degree")