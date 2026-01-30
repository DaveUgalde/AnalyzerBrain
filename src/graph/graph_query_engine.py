"""
GraphQueryEngine - Motor de consultas para el grafo de conocimiento.
Permite consultas complejas, recorridos y análisis de patrones.
"""

from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime
from .knowledge_graph import KnowledgeGraph, Node, NodeType, Relationship, RelationshipType

class QueryLanguage(Enum):
    """Lenguajes de consulta soportados."""
    CYPHER = "cypher"
    GREMLIN = "gremlin"
    PROPRIETARY = "proprietary"
    PATTERN = "pattern"

@dataclass
class QueryResult:
    """Resultado de una consulta."""
    success: bool
    data: Optional[List[Any]] = None
    columns: Optional[List[str]] = None
    execution_time_ms: float = 0.0
    nodes_scanned: int = 0
    relationships_scanned: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

@dataclass
class QueryPlan:
    """Plan de ejecución de una consulta."""
    steps: List[Dict[str, Any]]
    estimated_cost: float = 0.0
    estimated_rows: int = 0
    parallelizable: bool = False

class GraphQueryEngine:
    """
    Motor de consultas para grafos de conocimiento.
    """
    
    def __init__(self, graph: KnowledgeGraph):
        """
        Inicializa el motor de consultas.
        
        Args:
            graph: Grafo sobre el que se ejecutarán las consultas
        """
        self.graph = graph
        self.query_cache: Dict[str, QueryResult] = {}
        self.stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "total_execution_time_ms": 0.0
        }
    
    def execute_query(self, query: str, 
                     params: Optional[Dict[str, Any]] = None,
                     language: QueryLanguage = QueryLanguage.PROPRIETARY,
                     timeout_ms: int = 5000) -> QueryResult:
        """
        Ejecuta una consulta en el grafo.
        
        Args:
            query: Consulta a ejecutar
            params: Parámetros de la consulta
            language: Lenguaje de la consulta
            timeout_ms: Tiempo máximo de ejecución
            
        Returns:
            Resultado de la consulta
        """
        start_time = datetime.now()
        
        # Verificar caché
        cache_key = self._create_cache_key(query, params, language)
        if cache_key in self.query_cache:
            self.stats["cache_hits"] += 1
            result = self.query_cache[cache_key]
            result.execution_time_ms = 0.0  # Indicar que viene de caché
            return result
        
        try:
            # Parsear y ejecutar consulta según el lenguaje
            if language == QueryLanguage.CYPHER:
                result = self._execute_cypher(query, params)
            elif language == QueryLanguage.PROPRIETARY:
                result = self._execute_proprietary(query, params)
            elif language == QueryLanguage.PATTERN:
                result = self._execute_pattern_query(query, params)
            else:
                raise ValueError(f"Unsupported query language: {language}")
            
            # Calcular tiempo de ejecución
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Verificar timeout
            if execution_time > timeout_ms:
                result.success = False
                result.error = f"Query timeout after {execution_time}ms"
            
            result.execution_time_ms = execution_time
            
            # Actualizar estadísticas
            self.stats["queries_executed"] += 1
            self.stats["total_execution_time_ms"] += execution_time
            
            # Almacenar en caché (si fue exitosa y no es una mutación)
            if result.success and not self._is_mutation_query(query, language):
                self.query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return QueryResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    def find_patterns(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Encuentra patrones específicos en el grafo.
        
        Args:
            pattern: Definición del patrón
            
        Returns:
            Lista de patrones encontrados
        """
        # Extraer componentes del patrón
        nodes_pattern = pattern.get("nodes", [])
        relationships_pattern = pattern.get("relationships", [])
        constraints = pattern.get("constraints", {})
        
        # Plan de ejecución
        plan = self._create_pattern_plan(nodes_pattern, relationships_pattern, constraints)
        
        # Ejecutar plan
        results = self._execute_pattern_plan(plan)
        
        # Formatear resultados
        return self._format_pattern_results(results, pattern)
    
    def traverse_graph(self, 
                      start_node_id: str,
                      traversal_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recorre el grafo según una especificación.
        
        Args:
            start_node_id: ID del nodo de inicio
            traversal_spec: Especificación del recorrido
            
        Returns:
            Lista de nodos visitados en el recorrido
        """
        if start_node_id not in self.graph.nodes:
            return []
        
        # Configuración del recorrido
        direction = traversal_spec.get("direction", "outgoing")
        max_depth = traversal_spec.get("max_depth", 10)
        relationship_types = traversal_spec.get("relationship_types")
        node_filters = traversal_spec.get("node_filters", {})
        collect_paths = traversal_spec.get("collect_paths", False)
        
        # Convertir tipos de relación si es necesario
        if relationship_types:
            rel_types = [RelationshipType(rt.upper()) for rt in relationship_types]
        else:
            rel_types = None
        
        # Ejecutar recorrido
        if traversal_spec.get("algorithm", "dfs") == "bfs":
            return self._bfs_traversal(start_node_id, direction, max_depth, rel_types, node_filters, collect_paths)
        else:
            return self._dfs_traversal(start_node_id, direction, max_depth, rel_types, node_filters, collect_paths)
    
    def aggregate_graph_data(self, aggregation_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Agrega datos del grafo según una especificación.
        
        Args:
            aggregation_spec: Especificación de la agregación
            
        Returns:
            Resultados agregados
        """
        group_by = aggregation_spec.get("group_by", "type")
        metrics = aggregation_spec.get("metrics", ["count"])
        filters = aggregation_spec.get("filters", {})
        
        # Filtrar nodos
        filtered_nodes = self._filter_nodes(filters)
        
        # Agrupar
        groups = {}
        for node in filtered_nodes:
            key = self._extract_group_key(node, group_by)
            if key not in groups:
                groups[key] = []
            groups[key].append(node)
        
        # Calcular métricas
        results = []
        for key, nodes in groups.items():
            group_result = {"group": key, "count": len(nodes)}
            
            # Calcular métricas adicionales
            for metric in metrics:
                if metric == "avg_degree":
                    total_degree = 0
                    for node in nodes:
                        total_degree += (
                            len(self.graph._node_relationships[node.id]["incoming"]) +
                            len(self.graph._node_relationships[node.id]["outgoing"])
                        )
                    group_result["avg_degree"] = total_degree / len(nodes) if nodes else 0
                
                elif metric == "property_stats":
                    group_result["property_stats"] = self._calculate_property_stats(nodes)
                
                elif metric.startswith("avg_"):
                    prop_name = metric[4:]
                    values = [node.properties.get(prop_name) for node in nodes 
                             if prop_name in node.properties]
                    numeric_values = [v for v in values if isinstance(v, (int, float))]
                    if numeric_values:
                        group_result[metric] = sum(numeric_values) / len(numeric_values)
            
            results.append(group_result)
        
        # Ordenar resultados
        sort_by = aggregation_spec.get("sort_by", "count")
        sort_desc = aggregation_spec.get("sort_desc", True)
        
        results.sort(
            key=lambda x: x.get(sort_by, 0),
            reverse=sort_desc
        )
        
        return results
    
    def filter_graph(self, filter_spec: Dict[str, Any]) -> List[Node]:
        """
        Filtra nodos del grafo según criterios.
        
        Args:
            filter_spec: Especificación del filtro
            
        Returns:
            Nodos que cumplen los criterios
        """
        return self._filter_nodes(filter_spec)
    
    def optimize_query(self, query: str, 
                      language: QueryLanguage = QueryLanguage.PROPRIETARY) -> str:
        """
        Optimiza una consulta para mejorar el rendimiento.
        
        Args:
            query: Consulta a optimizar
            language: Lenguaje de la consulta
            
        Returns:
            Consulta optimizada
        """
        if language == QueryLanguage.CYPHER:
            return self._optimize_cypher(query)
        elif language == QueryLanguage.PROPRIETARY:
            return self._optimize_proprietary(query)
        else:
            return query
    
    def explain_query(self, query: str,
                     params: Optional[Dict[str, Any]] = None,
                     language: QueryLanguage = QueryLanguage.PROPRIETARY) -> QueryPlan:
        """
        Explica cómo se ejecutará una consulta.
        
        Args:
            query: Consulta a explicar
            params: Parámetros de la consulta
            language: Lenguaje de la consulta
            
        Returns:
            Plan de ejecución de la consulta
        """
        plan = QueryPlan(steps=[])
        
        if language == QueryLanguage.CYPHER:
            plan = self._explain_cypher(query, params)
        elif language == QueryLanguage.PROPRIETARY:
            plan = self._explain_proprietary(query, params)
        
        return plan
    
    # Métodos de implementación
    
    def _execute_cypher(self, query: str, params: Optional[Dict[str, Any]]) -> QueryResult:
        """Ejecuta una consulta Cypher (simplificada)."""
        # Implementación simplificada - en producción usaríamos Neo4j o similar
        # Por ahora, parseamos consultas básicas
        
        # Extraer tipo de consulta
        query_lower = query.lower().strip()
        
        if query_lower.startswith("match"):
            return self._execute_cypher_match(query, params)
        elif query_lower.startswith("return"):
            return self._execute_cypher_return(query, params)
        else:
            raise ValueError(f"Unsupported Cypher query: {query[:50]}...")
    
    def _execute_cypher_match(self, query: str, params: Optional[Dict[str, Any]]) -> QueryResult:
        """Ejecuta consulta MATCH de Cypher."""
        # Parsear patrones de MATCH (simplificado)
        pattern = self._parse_cypher_pattern(query)
        
        # Encontrar coincidencias
        matches = self._find_cypher_matches(pattern, params)
        
        # Aplicar WHERE si existe
        where_clause = self._extract_cypher_where(query)
        if where_clause:
            matches = self._apply_cypher_where(matches, where_clause, params)
        
        # Aplicar RETURN
        return_clause = self._extract_cypher_return(query)
        if return_clause:
            results = self._apply_cypher_return(matches, return_clause)
        else:
            results = matches
        
        return QueryResult(
            success=True,
            data=results,
            columns=list(results[0].keys()) if results else []
        )
    
    def _execute_proprietary(self, query: str, params: Optional[Dict[str, Any]]) -> QueryResult:
        """Ejecuta consulta en lenguaje propietario."""
        # Parsear consulta
        parsed = self._parse_proprietary_query(query, params)
        
        # Ejecutar según tipo
        if parsed["type"] == "find_nodes":
            nodes = self._find_nodes_by_criteria(parsed["criteria"])
            return QueryResult(
                success=True,
                data=[node.to_dict() for node in nodes],
                columns=["id", "type", "properties", "labels"]
            )
        
        elif parsed["type"] == "find_paths":
            paths = self.graph.find_path(
                parsed["start_id"],
                parsed["end_id"],
                parsed.get("max_depth", 10),
                parsed.get("relationship_types")
            )
            return QueryResult(
                success=True,
                data=[[node.to_dict() for node in path] for path in paths],
                columns=["path"]
            )
        
        elif parsed["type"] == "get_neighbors":
            neighbors = self.graph.get_neighbors(
                parsed["node_id"],
                parsed.get("direction", "outgoing"),
                parsed.get("relationship_types")
            )
            data = []
            for node, rel in neighbors:
                data.append({
                    "node": node.to_dict(),
                    "relationship": rel.to_dict()
                })
            return QueryResult(
                success=True,
                data=data,
                columns=["node", "relationship"]
            )
        
        else:
            raise ValueError(f"Unknown query type: {parsed['type']}")
    
    def _execute_pattern_query(self, query: str, params: Optional[Dict[str, Any]]) -> QueryResult:
        """Ejecuta consulta de patrón."""
        try:
            pattern = eval(query) if isinstance(query, str) else query
            
            if not isinstance(pattern, dict):
                raise ValueError("Pattern must be a dictionary")
            
            results = self.find_patterns(pattern)
            
            return QueryResult(
                success=True,
                data=results,
                columns=list(results[0].keys()) if results else []
            )
            
        except Exception as e:
            return QueryResult(
                success=False,
                error=f"Pattern query error: {str(e)}"
            )
    
    def _create_pattern_plan(self, nodes_pattern: List[Dict], 
                            relationships_pattern: List[Dict],
                            constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Crea plan de ejecución para búsqueda de patrones."""
        plan = {
            "steps": [],
            "estimated_cost": 0,
            "join_order": []
        }
        
        # Determinar orden de unión basado en selectividad
        # 1. Comenzar con nodos más selectivos
        selective_nodes = []
        for i, node_pattern in enumerate(nodes_pattern):
            selectivity = self._estimate_node_selectivity(node_pattern)
            selective_nodes.append((i, selectivity))
        
        selective_nodes.sort(key=lambda x: x[1])  # Menor selectividad primero
        
        # 2. Construir plan paso a paso
        plan["join_order"] = [idx for idx, _ in selective_nodes]
        
        for idx in plan["join_order"]:
            step = {
                "type": "node_scan",
                "pattern": nodes_pattern[idx],
                "estimated_rows": len(self.graph.nodes) * (1 - selective_nodes[idx][1]),
                "index_used": self._can_use_index(nodes_pattern[idx])
            }
            plan["steps"].append(step)
            plan["estimated_cost"] += step["estimated_rows"]
        
        # 3. Añadir uniones (joins)
        for rel_pattern in relationships_pattern:
            step = {
                "type": "relationship_join",
                "pattern": rel_pattern,
                "estimated_rows": plan["estimated_cost"] * 0.1,  # Estimación simplificada
                "join_type": "inner"
            }
            plan["steps"].append(step)
            plan["estimated_cost"] += step["estimated_rows"]
        
        # 4. Añadir filtros
        if constraints:
            step = {
                "type": "filter",
                "constraints": constraints,
                "estimated_rows": plan["estimated_cost"] * 0.5  # Asumir 50% de filtrado
            }
            plan["steps"].append(step)
            plan["estimated_cost"] *= 0.5
        
        return plan
    
    def _execute_pattern_plan(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ejecuta plan de búsqueda de patrones."""
        results = []
        
        # Implementación simplificada
        # En producción, esto sería mucho más sofisticado
        
        # Obtener nodos iniciales (primer paso del plan)
        if not plan["steps"]:
            return results
        
        first_step = plan["steps"][0]
        if first_step["type"] == "node_scan":
            initial_nodes = self._find_nodes_by_pattern(first_step["pattern"])
            
            # Para cada nodo inicial, buscar patrones completos
            for node in initial_nodes:
                pattern_match = self._expand_pattern(node, plan)
                if pattern_match:
                    results.append(pattern_match)
        
        return results
    
    def _estimate_node_selectivity(self, node_pattern: Dict[str, Any]) -> float:
        """Estima selectividad de un patrón de nodo."""
        total_nodes = len(self.graph.nodes)
        if total_nodes == 0:
            return 1.0
        
        # Contar nodos que coinciden con el patrón
        matching_nodes = self._find_nodes_by_pattern(node_pattern)
        
        return 1.0 - (len(matching_nodes) / total_nodes)
    
    def _can_use_index(self, node_pattern: Dict[str, Any]) -> bool:
        """Determina si se puede usar índice para el patrón."""
        # Verificar si hay propiedades indexables
        indexable_props = ["id", "type", "name", "path"]
        for prop in indexable_props:
            if prop in node_pattern.get("properties", {}):
                return True
        return False
    
    def _find_nodes_by_pattern(self, pattern: Dict[str, Any]) -> List[Node]:
        """Encuentra nodos que coinciden con un patrón."""
        node_type = pattern.get("type")
        properties = pattern.get("properties", {})
        labels = pattern.get("labels", [])
        
        # Convertir tipo si es necesario
        if node_type:
            try:
                node_type = NodeType(node_type.upper())
            except ValueError:
                node_type = None
        
        return self.graph.find_nodes(node_type, properties, labels, 1000)
    
    def _expand_pattern(self, start_node: Node, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Expande un patrón a partir de un nodo inicial."""
        # Implementación simplificada
        # En producción, esto seguiría el plan completo
        return {
            "node": start_node.to_dict(),
            "matched": True
        }
    
    def _bfs_traversal(self, start_id: str, direction: str, max_depth: int,
                      rel_types: Optional[List[RelationshipType]],
                      node_filters: Dict[str, Any],
                      collect_paths: bool) -> List[Dict[str, Any]]:
        """Recorrido BFS."""
        visited = set()
        queue = [(start_id, 0, [start_id] if collect_paths else None)]
        results = []
        
        while queue:
            current_id, depth, path = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            current_node = self.graph.nodes[current_id]
            
            # Aplicar filtros de nodo
            if not self._node_matches_filters(current_node, node_filters):
                continue
            
            # Añadir a resultados
            result = {
                "node": current_node.to_dict(),
                "depth": depth
            }
            if collect_paths:
                result["path"] = path
            results.append(result)
            
            # Obtener vecinos
            neighbors = self.graph.get_neighbors(current_id, direction, rel_types)
            for neighbor_node, rel in neighbors:
                if neighbor_node.id not in visited:
                    new_path = path + [neighbor_node.id] if collect_paths else None
                    queue.append((neighbor_node.id, depth + 1, new_path))
        
        return results
    
    def _dfs_traversal(self, start_id: str, direction: str, max_depth: int,
                      rel_types: Optional[List[RelationshipType]],
                      node_filters: Dict[str, Any],
                      collect_paths: bool) -> List[Dict[str, Any]]:
        """Recorrido DFS."""
        visited = set()
        results = []
        
        def dfs(current_id: str, depth: int, path: List[str]) -> None:
            if current_id in visited or depth > max_depth:
                return
            
            visited.add(current_id)
            current_node = self.graph.nodes[current_id]
            
            # Aplicar filtros de nodo
            if not self._node_matches_filters(current_node, node_filters):
                return
            
            # Añadir a resultados
            result = {
                "node": current_node.to_dict(),
                "depth": depth
            }
            if collect_paths:
                result["path"] = path.copy()
            results.append(result)
            
            # Explorar vecinos
            neighbors = self.graph.get_neighbors(current_id, direction, rel_types)
            for neighbor_node, rel in neighbors:
                if neighbor_node.id not in visited:
                    dfs(neighbor_node.id, depth + 1, path + [neighbor_node.id])
        
        dfs(start_id, 0, [start_id] if collect_paths else [])
        return results
    
    def _filter_nodes(self, filters: Dict[str, Any]) -> List[Node]:
        """Filtra nodos según criterios."""
        node_type = filters.get("node_type")
        properties = filters.get("properties", {})
        labels = filters.get("labels", [])
        min_degree = filters.get("min_degree", 0)
        max_degree = filters.get("max_degree", float('inf'))
        
        # Convertir tipo si es necesario
        if node_type:
            try:
                node_type = NodeType(node_type.upper())
            except ValueError:
                node_type = None
        
        # Filtrar por tipo, propiedades y etiquetas
        filtered = self.graph.find_nodes(node_type, properties, labels, 10000)
        
        # Filtrar por grado
        if min_degree > 0 or max_degree < float('inf'):
            filtered = [
                node for node in filtered
                if self._node_degree_in_range(node.id, min_degree, max_degree)
            ]
        
        return filtered
    
    def _node_degree_in_range(self, node_id: str, min_degree: int, max_degree: int) -> bool:
        """Verifica si el grado de un nodo está en el rango."""
        rels = self.graph._node_relationships.get(node_id, {"incoming": [], "outgoing": []})
        degree = len(rels["incoming"]) + len(rels["outgoing"])
        return min_degree <= degree <= max_degree
    
    def _node_matches_filters(self, node: Node, filters: Dict[str, Any]) -> bool:
        """Verifica si un nodo coincide con los filtros."""
        # Filtrar por tipo
        if "type" in filters and node.type.value != filters["type"]:
            return False
        
        # Filtrar por propiedades
        if "properties" in filters:
            for key, value in filters["properties"].items():
                if key not in node.properties or node.properties[key] != value:
                    return False
        
        # Filtrar por etiquetas
        if "labels" in filters:
            for label in filters["labels"]:
                if label not in node.labels:
                    return False
        
        return True
    
    def _extract_group_key(self, node: Node, group_by: Union[str, List[str]]) -> Any:
        """Extrae clave de agrupación de un nodo."""
        if isinstance(group_by, list):
            return tuple(node.properties.get(key, None) for key in group_by)
        else:
            if group_by == "type":
                return node.type.value
            else:
                return node.properties.get(group_by, None)
    
    def _calculate_property_stats(self, nodes: List[Node]) -> Dict[str, Any]:
        """Calcula estadísticas de propiedades."""
        if not nodes:
            return {}
        
        # Recolectar todas las propiedades
        all_props = {}
        for node in nodes:
            for key, value in node.properties.items():
                if key not in all_props:
                    all_props[key] = []
                all_props[key].append(value)
        
        # Calcular estadísticas
        stats = {}
        for key, values in all_props.items():
            # Determinar tipo de valores
            sample = values[0]
            
            if isinstance(sample, (int, float)):
                # Estadísticas numéricas
                numeric_vals = [v for v in values if isinstance(v, (int, float))]
                if numeric_vals:
                    stats[key] = {
                        "type": "numeric",
                        "count": len(numeric_vals),
                        "min": min(numeric_vals),
                        "max": max(numeric_vals),
                        "mean": sum(numeric_vals) / len(numeric_vals),
                        "unique": len(set(numeric_vals))
                    }
            
            elif isinstance(sample, str):
                # Estadísticas de texto
                stats[key] = {
                    "type": "string",
                    "count": len(values),
                    "unique": len(set(values)),
                    "min_length": min(len(str(v)) for v in values),
                    "max_length": max(len(str(v)) for v in values),
                    "sample": values[:5]  # Muestra de valores
                }
            
            elif isinstance(sample, bool):
                # Estadísticas booleanas
                true_count = sum(1 for v in values if v is True)
                stats[key] = {
                    "type": "boolean",
                    "count": len(values),
                    "true_count": true_count,
                    "false_count": len(values) - true_count,
                    "true_percentage": true_count / len(values) * 100 if values else 0
                }
        
        return stats
    
    def _create_cache_key(self, query: str, params: Optional[Dict[str, Any]], 
                         language: QueryLanguage) -> str:
        """Crea clave de caché para una consulta."""
        import hashlib
        import json
        
        key_data = {
            "query": query,
            "params": params or {},
            "language": language.value
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_mutation_query(self, query: str, language: QueryLanguage) -> bool:
        """Determina si una consulta modifica el grafo."""
        mutation_keywords = ["create", "merge", "set", "delete", "remove"]
        
        if language == QueryLanguage.CYPHER:
            query_lower = query.lower()
            return any(keyword in query_lower for keyword in mutation_keywords)
        
        return False
    
    def _parse_cypher_pattern(self, query: str) -> Dict[str, Any]:
        """Parsea patrón Cypher (simplificado)."""
        # Implementación simplificada
        # En producción usaríamos un parser real
        pattern = {
            "nodes": [],
            "relationships": []
        }
        
        # Extraer patrones de nodo básicos
        node_patterns = re.findall(r'\((\w+)(?::(\w+))?(?:\s*\{([^}]+)\})?\)', query)
        for var_name, label, props in node_patterns:
            node = {"variable": var_name}
            if label:
                node["labels"] = [label]
            if props:
                # Parsear propiedades básicas
                props_dict = {}
                for prop in props.split(','):
                    if ':' in prop:
                        key, value = prop.split(':', 1)
                        props_dict[key.strip()] = eval(value.strip())
                node["properties"] = props_dict
            pattern["nodes"].append(node)
        
        return pattern
    
    def _extract_cypher_where(self, query: str) -> Optional[str]:
        """Extrae cláusula WHERE de consulta Cypher."""
        match = re.search(r'WHERE\s+(.+?)(?:\s+(?:RETURN|ORDER|LIMIT|SKIP)|$)', query, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _extract_cypher_return(self, query: str) -> Optional[str]:
        """Extrae cláusula RETURN de consulta Cypher."""
        match = re.search(r'RETURN\s+(.+?)(?:\s+(?:ORDER|LIMIT|SKIP)|$)', query, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _find_cypher_matches(self, pattern: Dict[str, Any], params: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encuentra coincidencias para patrón Cypher."""
        # Implementación simplificada
        matches = []
        
        # Para cada nodo en el patrón
        for node_pattern in pattern.get("nodes", []):
            # Encontrar nodos coincidentes
            matching_nodes = self._find_nodes_by_pattern(node_pattern)
            
            # Crear resultados
            for node in matching_nodes:
                match = {
                    node_pattern.get("variable", "n"): node.to_dict()
                }
                matches.append(match)
        
        return matches
    
    def _apply_cypher_where(self, matches: List[Dict[str, Any]], 
                           where_clause: str, params: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aplica cláusula WHERE a coincidencias Cypher."""
        # Implementación simplificada
        filtered = []
        
        for match in matches:
            # Evaluar condiciones básicas
            try:
                # Reemplazar variables con valores reales
                eval_str = where_clause
                for var_name, var_value in match.items():
                    # Convertir a expresión evaluable (simplificado)
                    eval_str = eval_str.replace(f"{var_name}.", f"match['{var_name}']['")
                    eval_str = eval_str.replace(f"{var_name}", f"match['{var_name}']")
                
                # Evaluar (en producción usaríamos un evaluador seguro)
                # Por ahora, simplemente incluimos todos
                filtered.append(match)
            except:
                continue
        
        return filtered
    
    def _apply_cypher_return(self, matches: List[Dict[str, Any]], 
                            return_clause: str) -> List[Dict[str, Any]]:
        """Aplica cláusula RETURN a coincidencias Cypher."""
        # Implementación simplificada
        results = []
        
        for match in matches:
            result = {}
            
            # Parsear expresión de RETURN
            # Ej: "n.name, n.age" -> extraer propiedades
            parts = return_clause.split(',')
            for part in parts:
                part = part.strip()
                
                if '.' in part:
                    var_name, prop_name = part.split('.', 1)
                    if var_name in match:
                        value = match[var_name].get(prop_name)
                        result[part] = value
                else:
                    # Variable completa
                    if part in match:
                        result[part] = match[part]
            
            results.append(result)
        
        return results
    
    def _parse_proprietary_query(self, query: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Parsea consulta en lenguaje propietario."""
        # Formato: OPERATION(arg1=value1, arg2=value2, ...)
        match = re.match(r'(\w+)\((.*)\)', query.strip())
        if not match:
            raise ValueError(f"Invalid query format: {query}")
        
        operation = match.group(1).lower()
        args_str = match.group(2)
        
        # Parsear argumentos
        args = {}
        if args_str.strip():
            for arg in args_str.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Evaluar valor (simplificado)
                    try:
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        elif value.isdigit():
                            value = int(value)
                        elif value.replace('.', '', 1).isdigit():
                            value = float(value)
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        elif value.lower() == 'none':
                            value = None
                    except:
                        pass
                    
                    args[key] = value
        
        # Aplicar parámetros
        if params:
            args.update(params)
        
        return {
            "type": operation,
            **args
        }
    
    def _find_nodes_by_criteria(self, criteria: Dict[str, Any]) -> List[Node]:
        """Encuentra nodos por criterios específicos."""
        # Mapear criterios a parámetros de find_nodes
        node_type = criteria.get("type")
        if node_type:
            try:
                node_type = NodeType(node_type.upper())
            except ValueError:
                node_type = None
        
        properties = {}
        for key, value in criteria.items():
            if key not in ["type", "labels", "min_degree", "max_degree"]:
                properties[key] = value
        
        labels = criteria.get("labels", [])
        if isinstance(labels, str):
            labels = [labels]
        
        nodes = self.graph.find_nodes(node_type, properties, labels, 1000)
        
        # Filtrar por grado si es necesario
        min_degree = criteria.get("min_degree", 0)
        max_degree = criteria.get("max_degree", float('inf'))
        
        if min_degree > 0 or max_degree < float('inf'):
            nodes = [
                node for node in nodes
                if self._node_degree_in_range(node.id, min_degree, max_degree)
            ]
        
        return nodes
    
    def _format_pattern_results(self, results: List[Dict[str, Any]], 
                               pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Formatea resultados de búsqueda de patrones."""
        formatted = []
        
        for result in results:
            formatted_result = {
                "pattern_matched": True,
                "nodes": {},
                "relationships": {}
            }
            
            # Extraer nodos y relaciones del resultado
            # (depende de la estructura específica de los resultados)
            
            formatted.append(formatted_result)
        
        return formatted
    
    def _optimize_cypher(self, query: str) -> str:
        """Optimiza consulta Cypher."""
        # Reordenar MATCH clauses por selectividad
        # Añadir índices sugeridos
        # Eliminar cláusulas redundantes
        return query  # Por ahora, no hacemos optimización
    
    def _optimize_proprietary(self, query: str) -> str:
        """Optimiza consulta propietaria."""
        # Simplificar expresiones
        # Aplicar transformaciones equivalentes
        return query
    
    def _explain_cypher(self, query: str, params: Optional[Dict[str, Any]]) -> QueryPlan:
        """Explica consulta Cypher."""
        plan = QueryPlan(steps=[])
        
        # Analizar consulta para determinar plan
        if "MATCH" in query.upper():
            plan.steps.append({
                "operation": "MATCH",
                "description": "Find pattern matches",
                "estimated_rows": len(self.graph.nodes) * 0.1  # Estimación
            })
        
        if "WHERE" in query.upper():
            plan.steps.append({
                "operation": "FILTER",
                "description": "Apply WHERE conditions",
                "estimated_rows": len(self.graph.nodes) * 0.05
            })
        
        if "RETURN" in query.upper():
            plan.steps.append({
                "operation": "PROJECT",
                "description": "Project selected columns",
                "estimated_rows": len(self.graph.nodes) * 0.01
            })
        
        plan.estimated_cost = sum(step["estimated_rows"] for step in plan.steps)
        plan.estimated_rows = plan.steps[-1]["estimated_rows"] if plan.steps else 0
        
        return plan
    
    def _explain_proprietary(self, query: str, params: Optional[Dict[str, Any]]) -> QueryPlan:
        """Explica consulta propietaria."""
        parsed = self._parse_proprietary_query(query, params)
        plan = QueryPlan(steps=[])
        
        if parsed["type"] == "find_nodes":
            plan.steps.append({
                "operation": "NODE_SCAN",
                "description": f"Find nodes with criteria",
                "estimated_rows": len(self.graph.nodes) * 0.1,
                "index_used": True
            })
        
        plan.estimated_cost = sum(step["estimated_rows"] for step in plan.steps)
        plan.estimated_rows = plan.steps[-1]["estimated_rows"] if plan.steps else 0
        
        return plan