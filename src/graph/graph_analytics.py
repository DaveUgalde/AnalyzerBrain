"""
GraphAnalytics - Análisis avanzado de grafos.
Implementa algoritmos de análisis de grafos como detección de comunidades, PageRank, clustering, etc.
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
import numpy as np
import networkx as nx
from datetime import datetime
from .knowledge_graph import KnowledgeGraph, Node, Relationship, NodeType, RelationshipType

@dataclass
class Community:
    """Comunidad detectada en el grafo."""
    id: int
    nodes: List[Node]
    size: int
    density: float
    modularity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalyticsResult:
    """Resultado de análisis de grafo."""
    communities: List[Community]
    pagerank: Dict[str, float]
    clusters: Dict[str, int]
    graph_density: float
    degree_distribution: Dict[int, int]
    bridges: List[Tuple[str, str]]
    influential_nodes: List[Node]
    execution_time_ms: float = 0.0

class GraphAnalytics:
    """
    Análisis avanzado de grafos de conocimiento.
    """
    
    def __init__(self, graph: KnowledgeGraph):
        """
        Inicializa el analizador de grafos.
        
        Args:
            graph: Grafo a analizar
        """
        self.graph = graph
        self.nx_graph = None
    
    def analyze_community_structure(self, algorithm: str = "louvain") -> List[Community]:
        """
        Analiza la estructura de comunidades del grafo.
        
        Args:
            algorithm: Algoritmo a usar ('louvain', 'label_propagation', 'girvan_newman')
            
        Returns:
            Lista de comunidades detectadas
        """
        start_time = datetime.now()
        
        # Convertir a NetworkX si es necesario
        self._ensure_nx_graph()
        
        if algorithm == "louvain":
            communities = self._louvain_communities()
        elif algorithm == "label_propagation":
            communities = self._label_propagation_communities()
        elif algorithm == "girvan_newman":
            communities = self._girvan_newman_communities()
        else:
            raise ValueError(f"Unsupported community detection algorithm: {algorithm}")
        
        # Calcular métricas para cada comunidad
        enriched_communities = []
        for i, community_nodes in enumerate(communities):
            community = Community(
                id=i,
                nodes=community_nodes,
                size=len(community_nodes),
                density=self._calculate_community_density(community_nodes),
                modularity_score=self._calculate_community_modularity(community_nodes)
            )
            enriched_communities.append(community)
        
        # Ordenar por tamaño (de mayor a menor)
        enriched_communities.sort(key=lambda c: c.size, reverse=True)
        
        return enriched_communities
    
    def calculate_pagerank(self, 
                          damping_factor: float = 0.85,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Calcula PageRank para todos los nodos del grafo.
        
        Args:
            damping_factor: Factor de amortiguación (usualmente 0.85)
            max_iterations: Máximo número de iteraciones
            tolerance: Tolerancia para convergencia
            
        Returns:
            Diccionario con ID de nodo -> valor de PageRank
        """
        start_time = datetime.now()
        
        # Convertir a NetworkX si es necesario
        self._ensure_nx_graph()
        
        # Calcular PageRank usando NetworkX
        pagerank_scores = nx.pagerank(
            self.nx_graph,
            alpha=damping_factor,
            max_iter=max_iterations,
            tol=tolerance
        )
        
        return pagerank_scores
    
    def detect_clusters(self, 
                       method: str = "k_cliques",
                       k: int = 3,
                       threshold: float = 0.5) -> Dict[str, int]:
        """
        Detecta clusters (agrupaciones densas) en el grafo.
        
        Args:
            method: Método de clustering ('k_cliques', 'modularity', 'spectral')
            k: Tamaño de clique para k-clique clustering
            threshold: Umbral para detección
            
        Returns:
            Diccionario con ID de nodo -> ID de cluster
        """
        start_time = datetime.now()
        
        self._ensure_nx_graph()
        
        if method == "k_cliques":
            clusters = self._k_clique_clustering(k)
        elif method == "modularity":
            clusters = self._modularity_clustering(threshold)
        elif method == "spectral":
            clusters = self._spectral_clustering()
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        return clusters
    
    def measure_graph_density(self) -> float:
        """
        Mide la densidad del grafo.
        
        Returns:
            Densidad del grafo (0.0 a 1.0)
        """
        total_nodes = len(self.graph.nodes)
        total_edges = len(self.graph.relationships)
        
        if total_nodes <= 1:
            return 0.0
        
        # Para grafo dirigido
        max_possible_edges = total_nodes * (total_nodes - 1)
        density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        return density
    
    def analyze_degree_distribution(self) -> Dict[int, int]:
        """
        Analiza la distribución de grados del grafo.
        
        Returns:
            Diccionario con grado -> número de nodos con ese grado
        """
        degree_dist = {}
        
        for node_id in self.graph.nodes:
            rels = self.graph._node_relationships[node_id]
            degree = len(rels["incoming"]) + len(rels["outgoing"])
            
            degree_dist[degree] = degree_dist.get(degree, 0) + 1
        
        return degree_dist
    
    def find_bridges(self) -> List[Tuple[str, str]]:
        """
        Encuentra puentes (aristas cuya remoción desconecta el grafo).
        
        Returns:
            Lista de tuplas (source_id, target_id) que son puentes
        """
        self._ensure_nx_graph()
        
        # Convertir a grafo no dirigido para análisis de puentes
        undirected_graph = self.nx_graph.to_undirected()
        
        # Encontrar puentes usando NetworkX
        nx_bridges = list(nx.bridges(undirected_graph))
        
        # Convertir a IDs de nuestro grafo
        bridges = []
        for source, target in nx_bridges:
            bridges.append((str(source), str(target)))
        
        return bridges
    
    def identify_influential_nodes(self, 
                                  method: str = "pagerank",
                                  top_k: int = 10) -> List[Node]:
        """
        Identifica nodos influyentes en el grafo.
        
        Args:
            method: Método para identificar influencia
            top_k: Número de nodos a retornar
            
        Returns:
            Lista de nodos más influyentes
        """
        if method == "pagerank":
            scores = self.calculate_pagerank()
        elif method == "degree":
            scores = self._calculate_degree_centrality()
        elif method == "betweenness":
            scores = self._calculate_betweenness_centrality()
        elif method == "closeness":
            scores = self._calculate_closeness_centrality()
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Ordenar nodos por score
        sorted_nodes = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Convertir a objetos Node
        influential_nodes = []
        for node_id, _ in sorted_nodes:
            if node_id in self.graph.nodes:
                influential_nodes.append(self.graph.nodes[node_id])
        
        return influential_nodes
    
    def comprehensive_analysis(self) -> AnalyticsResult:
        """
        Ejecuta análisis completo del grafo.
        
        Returns:
            Resultado completo del análisis
        """
        start_time = datetime.now()
        
        # Ejecutar todos los análisis
        communities = self.analyze_community_structure()
        pagerank = self.calculate_pagerank()
        clusters = self.detect_clusters()
        density = self.measure_graph_density()
        degree_dist = self.analyze_degree_distribution()
        bridges = self.find_bridges()
        influential_nodes = self.identify_influential_nodes()
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AnalyticsResult(
            communities=communities,
            pagerank=pagerank,
            clusters=clusters,
            graph_density=density,
            degree_distribution=degree_dist,
            bridges=bridges,
            influential_nodes=influential_nodes,
            execution_time_ms=execution_time
        )
    
    # Métodos de implementación
    
    def _ensure_nx_graph(self) -> None:
        """Asegura que exista el grafo de NetworkX."""
        if self.nx_graph is None:
            self.nx_graph = self.graph.to_networkx()
    
    def _louvain_communities(self) -> List[List[Node]]:
        """Detecta comunidades usando el algoritmo de Louvain."""
        try:
            import community as community_louvain
            
            # Usar la versión no dirigida para Louvain
            undirected_graph = self.nx_graph.to_undirected()
            
            # Detectar partición
            partition = community_louvain.best_partition(undirected_graph)
            
            # Agrupar nodos por comunidad
            communities_dict = {}
            for node_id, comm_id in partition.items():
                if comm_id not in communities_dict:
                    communities_dict[comm_id] = []
                
                # Convertir node_id de NetworkX a nuestro ID si es necesario
                actual_node_id = str(node_id)
                if actual_node_id in self.graph.nodes:
                    communities_dict[comm_id].append(self.graph.nodes[actual_node_id])
            
            return list(communities_dict.values())
            
        except ImportError:
            # Fallback a algoritmo simple si no está instalado community
            return self._simple_community_detection()
    
    def _label_propagation_communities(self) -> List[List[Node]]:
        """Detecta comunidades usando propagación de etiquetas."""
        self._ensure_nx_graph()
        
        # Usar el algoritmo de NetworkX
        communities = list(nx.algorithms.community.label_propagation_communities(self.nx_graph))
        
        # Convertir a objetos Node
        node_communities = []
        for comm in communities:
            nodes = []
            for node_id in comm:
                actual_node_id = str(node_id)
                if actual_node_id in self.graph.nodes:
                    nodes.append(self.graph.nodes[actual_node_id])
            if nodes:
                node_communities.append(nodes)
        
        return node_communities
    
    def _girvan_newman_communities(self) -> List[List[Node]]:
        """Detecta comunidades usando el algoritmo de Girvan-Newman."""
        self._ensure_nx_graph()
        
        # Usar versión no dirigida
        undirected_graph = self.nx_graph.to_undirected()
        
        # Ejecutar Girvan-Newman (limitado a 10 comunidades para rendimiento)
        comp = nx.algorithms.community.girvan_newman(undirected_graph)
        
        # Obtener el primer nivel de partición
        try:
            communities = next(comp)
        except StopIteration:
            # Si no hay partición, cada nodo es su propia comunidad
            communities = [{node} for node in undirected_graph.nodes()]
        
        # Convertir a objetos Node
        node_communities = []
        for comm in communities:
            nodes = []
            for node_id in comm:
                actual_node_id = str(node_id)
                if actual_node_id in self.graph.nodes:
                    nodes.append(self.graph.nodes[actual_node_id])
            if nodes:
                node_communities.append(nodes)
        
        return node_communities
    
    def _simple_community_detection(self) -> List[List[Node]]:
        """Detección simple de comunidades basada en conectividad."""
        # Usar componentes conectados como comunidades
        components = []
        visited = set()
        
        for node_id in self.graph.nodes:
            if node_id not in visited:
                # BFS para encontrar componente
                component = []
                queue = [node_id]
                
                while queue:
                    current_id = queue.pop(0)
                    if current_id in visited:
                        continue
                    
                    visited.add(current_id)
                    component.append(self.graph.nodes[current_id])
                    
                    # Añadir vecinos
                    for rel_id in self.graph._node_relationships[current_id]["outgoing"]:
                        rel = self.graph.relationships[rel_id]
                        if rel.target_id not in visited:
                            queue.append(rel.target_id)
                    
                    for rel_id in self.graph._node_relationships[current_id]["incoming"]:
                        rel = self.graph.relationships[rel_id]
                        if rel.source_id not in visited:
                            queue.append(rel.source_id)
                
                if component:
                    components.append(component)
        
        return components
    
    def _calculate_community_density(self, community_nodes: List[Node]) -> float:
        """Calcula la densidad de una comunidad."""
        if len(community_nodes) <= 1:
            return 0.0
        
        # Crear subgrafo de la comunidad
        node_ids = {node.id for node in community_nodes}
        
        # Contar aristas dentro de la comunidad
        internal_edges = 0
        
        for node in community_nodes:
            for rel_id in self.graph._node_relationships[node.id]["outgoing"]:
                rel = self.graph.relationships[rel_id]
                if rel.target_id in node_ids:
                    internal_edges += 1
        
        # Para grafo dirigido
        max_possible_edges = len(community_nodes) * (len(community_nodes) - 1)
        density = internal_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        return density
    
    def _calculate_community_modularity(self, community_nodes: List[Node]) -> float:
        """Calcula el score de modularidad de una comunidad."""
        # Implementación simplificada de modularidad
        if len(self.graph.nodes) == 0:
            return 0.0
        
        community_ids = {node.id for node in community_nodes}
        total_edges = len(self.graph.relationships)
        
        if total_edges == 0:
            return 0.0
        
        # Calcular L_c (aristas dentro de la comunidad)
        L_c = 0
        for node_id in community_ids:
            for rel_id in self.graph._node_relationships[node_id]["outgoing"]:
                rel = self.graph.relationships[rel_id]
                if rel.target_id in community_ids:
                    L_c += 1
        
        # Calcular k_c (grados totales de nodos en la comunidad)
        k_c = 0
        for node_id in community_ids:
            rels = self.graph._node_relationships[node_id]
            k_c += len(rels["incoming"]) + len(rels["outgoing"])
        
        # Fórmula de modularidad simplificada
        modularity = (L_c / total_edges) - ((k_c / (2 * total_edges)) ** 2)
        
        return modularity
    
    def _k_clique_clustering(self, k: int) -> Dict[str, int]:
        """Clustering basado en k-cliques."""
        self._ensure_nx_graph()
        
        # Encontrar todos los k-cliques
        cliques = list(nx.find_cliques(self.nx_graph.to_undirected()))
        k_cliques = [clique for clique in cliques if len(clique) >= k]
        
        # Asignar clusters basados en solapamiento de cliques
        clusters = {}
        current_cluster_id = 0
        
        for clique in k_cliques:
            # Verificar si algún nodo ya está en un cluster
            existing_clusters = set()
            for node_id in clique:
                if str(node_id) in clusters:
                    existing_clusters.add(clusters[str(node_id)])
            
            if not existing_clusters:
                # Nuevo cluster
                for node_id in clique:
                    clusters[str(node_id)] = current_cluster_id
                current_cluster_id += 1
            else:
                # Unir clusters existentes
                target_cluster = min(existing_clusters)
                for node_id in clique:
                    clusters[str(node_id)] = target_cluster
        
        # Asignar cluster 0 a nodos no asignados
        for node_id in self.graph.nodes:
            if node_id not in clusters:
                clusters[node_id] = -1  # No pertenece a ningún cluster
        
        return clusters
    
    def _modularity_clustering(self, threshold: float) -> Dict[str, int]:
        """Clustering basado en modularidad optimizada."""
        communities = self.analyze_community_structure()
        
        clusters = {}
        for i, community in enumerate(communities):
            for node in community.nodes:
                clusters[node.id] = i
        
        # Asignar -1 a nodos no en comunidades
        for node_id in self.graph.nodes:
            if node_id not in clusters:
                clusters[node_id] = -1
        
        return clusters
    
    def _spectral_clustering(self) -> Dict[str, int]:
        """Clustering espectral."""
        self._ensure_nx_graph()
        
        try:
            from sklearn.cluster import SpectralClustering
            
            # Crear matriz de adyacencia
            nodes = list(self.nx_graph.nodes())
            n = len(nodes)
            
            if n == 0:
                return {}
            
            # Determinar número de clusters (máximo 10 para rendimiento)
            n_clusters = min(10, max(2, n // 10))
            
            # Convertir grafo a matriz
            A = nx.to_numpy_array(self.nx_graph, nodelist=nodes)
            
            # Aplicar clustering espectral
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            
            labels = clustering.fit_predict(A)
            
            # Crear diccionario de clusters
            clusters = {}
            for i, node in enumerate(nodes):
                clusters[str(node)] = int(labels[i])
            
            return clusters
            
        except ImportError:
            # Fallback a k-means simple si no hay scikit-learn
            return self._simple_kmeans_clustering()
    
    def _simple_kmeans_clustering(self) -> Dict[str, int]:
        """Clustering simple basado en k-means en embeddings de nodos."""
        # Crear embeddings simples basados en vecinos
        embeddings = {}
        
        for node_id in self.graph.nodes:
            # Usar grado y vecinos como embedding simple
            rels = self.graph._node_relationships[node_id]
            degree = len(rels["incoming"]) + len(rels["outgoing"])
            
            # Añadir información de tipos de vecinos
            neighbor_types = {}
            neighbors = self.graph.get_neighbors(node_id, "both")
            for neighbor_node, _ in neighbors:
                neighbor_type = neighbor_node.type.value
                neighbor_types[neighbor_type] = neighbor_types.get(neighbor_type, 0) + 1
            
            # Embedding simple: [grado, count_type1, count_type2, ...]
            embedding = [degree]
            for node_type in NodeType:
                embedding.append(neighbor_types.get(node_type.value, 0))
            
            embeddings[node_id] = embedding
        
        # K-means manual simple (para pocos nodos)
        if not embeddings:
            return {}
        
        # Número de clusters (máximo 5)
        n_clusters = min(5, len(embeddings))
        
        # Inicializar centroides aleatorios
        import random
        centroids = random.sample(list(embeddings.values()), n_clusters)
        
        # Asignaciones
        clusters = {node_id: -1 for node_id in embeddings}
        
        # Simple iteración (solo una pasada para rendimiento)
        for node_id, embedding in embeddings.items():
            # Encontrar centroide más cercano
            min_dist = float('inf')
            closest_centroid = 0
            
            for i, centroid in enumerate(centroids):
                dist = self._euclidean_distance(embedding, centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_centroid = i
            
            clusters[node_id] = closest_centroid
        
        return clusters
    
    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Distancia euclidiana entre dos vectores."""
        if len(vec1) != len(vec2):
            # Padding con ceros
            max_len = max(len(vec1), len(vec2))
            vec1 = vec1 + [0] * (max_len - len(vec1))
            vec2 = vec2 + [0] * (max_len - len(vec2))
        
        return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5
    
    def _calculate_degree_centrality(self) -> Dict[str, float]:
        """Calcula centralidad por grado."""
        centrality = {}
        total_nodes = len(self.graph.nodes)
        
        if total_nodes <= 1:
            return {node_id: 0 for node_id in self.graph.nodes}
        
        for node_id in self.graph.nodes:
            rels = self.graph._node_relationships[node_id]
            degree = len(rels["incoming"]) + len(rels["outgoing"])
            centrality[node_id] = degree / (total_nodes - 1)
        
        return centrality
    
    def _calculate_betweenness_centrality(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """Calcula centralidad de intermediación."""
        self._ensure_nx_graph()
        
        if sample_size and len(self.nx_graph) > sample_size:
            centrality = nx.betweenness_centrality(self.nx_graph, k=sample_size)
        else:
            centrality = nx.betweenness_centrality(self.nx_graph)
        
        # Convertir keys a strings
        return {str(k): v for k, v in centrality.items()}
    
    def _calculate_closeness_centrality(self) -> Dict[str, float]:
        """Calcula centralidad de cercanía."""
        self._ensure_nx_graph()
        
        centrality = nx.closeness_centrality(self.nx_graph)
        
        # Convertir keys a strings
        return {str(k): v for k, v in centrality.items()}