# src/graph/consistency_checker.py

from typing import Dict, List, Optional, Any, Set
from .knowledge_graph import KnowledgeGraph, Node, Relationship
from .schema_manager import SchemaManager

class ConsistencyChecker:
    """
    Verificador de consistencia del grafo.
    """
    
    def __init__(self, graph: KnowledgeGraph, schema_manager: Optional[SchemaManager] = None):
        """
        Inicializa el verificador de consistencia.
        
        Args:
            graph: Grafo a verificar.
            schema_manager: Gestor de esquemas (opcional).
        """
        self.graph = graph
        self.schema_manager = schema_manager or SchemaManager()
    
    def check_graph_consistency(self) -> Dict[str, Any]:
        """
        Verifica la consistencia general del grafo.
        
        Returns:
            Diccionario con resultados de la verificación.
        """
        results = {
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Verificar nodos huérfanos
        orphan_nodes = self.detect_orphan_nodes()
        if orphan_nodes:
            results["warnings"].append({
                "type": "orphan_nodes",
                "count": len(orphan_nodes),
                "nodes": orphan_nodes[:10]  # Mostrar solo los primeros 10
            })
        
        # Verificar referencias
        reference_errors = self.validate_references()
        results["errors"].extend(reference_errors)
        
        # Verificar tipos de propiedades
        type_errors = self.check_property_types()
        results["errors"].extend(type_errors)
        
        # Verificar inconsistencias
        inconsistencies = self.detect_inconsistencies()
        results["errors"].extend(inconsistencies)
        
        # Estadísticas
        results["stats"] = {
            "total_nodes": len(self.graph.nodes),
            "total_relationships": len(self.graph.relationships),
            "orphan_nodes": len(orphan_nodes),
            "errors_count": len(results["errors"]),
            "warnings_count": len(results["warnings"])
        }
        
        return results
    
    def detect_orphan_nodes(self) -> List[str]:
        """
        Detecta nodos huérfanos (sin relaciones).
        
        Returns:
            Lista de IDs de nodos huérfanos.
        """
        orphan_nodes = []
        
        for node_id, rels in self.graph._node_relationships.items():
            if not rels["incoming"] and not rels["outgoing"]:
                orphan_nodes.append(node_id)
        
        return orphan_nodes
    
    def validate_references(self) -> List[Dict[str, Any]]:
        """
        Valida que todas las referencias entre nodos sean válidas.
        
        Returns:
            Lista de errores de referencia.
        """
        errors = []
        
        # Verificar relaciones
        for rel_id, rel in self.graph.relationships.items():
            if rel.source_id not in self.graph.nodes:
                errors.append({
                    "type": "invalid_reference",
                    "message": f"Relationship {rel_id} references non-existent source node {rel.source_id}",
                    "relationship_id": rel_id,
                    "node_id": rel.source_id
                })
            
            if rel.target_id not in self.graph.nodes:
                errors.append({
                    "type": "invalid_reference",
                    "message": f"Relationship {rel_id} references non-existent target node {rel.target_id}",
                    "relationship_id": rel_id,
                    "node_id": rel.target_id
                })
        
        return errors
    
    def check_property_types(self) -> List[Dict[str, Any]]:
        """
        Verifica que los tipos de propiedades sean consistentes.
        
        Returns:
            Lista de errores de tipo.
        """
        errors = []
        
        # Verificar nodos
        for node_id, node in self.graph.nodes.items():
            # Validar contra esquema si está disponible
            if self.schema_manager:
                schema_errors = self.schema_manager.validate_node_schema(
                    node.type, node.properties, node.labels
                )
                
                for error in schema_errors:
                    errors.append({
                        "type": "property_type_error",
                        "message": f"Node {node_id}: {error}",
                        "node_id": node_id,
                        "node_type": node.type.value
                    })
        
        # Verificar relaciones
        for rel_id, rel in self.graph.relationships.items():
            # Obtener tipos de nodos origen y destino
            source_node = self.graph.find_node(rel.source_id)
            target_node = self.graph.find_node(rel.target_id)
            
            if source_node and target_node and self.schema_manager:
                schema_errors = self.schema_manager.validate_edge_schema(
                    rel.type, source_node.type, target_node.type, rel.properties
                )
                
                for error in schema_errors:
                    errors.append({
                        "type": "property_type_error",
                        "message": f"Relationship {rel_id}: {error}",
                        "relationship_id": rel_id,
                        "relationship_type": rel.type.value
                    })
        
        return errors
    
    def detect_inconsistencies(self) -> List[Dict[str, Any]]:
        """
        Detecta inconsistencias en el grafo.
        
        Returns:
            Lista de inconsistencias.
        """
        inconsistencies = []
        
        # Verificar ciclos en relaciones de dependencia
        from .graph_traverser import GraphTraverser
        traverser = GraphTraverser(self.graph)
        
        # Buscar ciclos en relaciones DEPENDS_ON
        depends_on_rels = [r for r in self.graph.relationships.values() 
                          if r.type.name == "DEPENDS_ON"]
        
        if depends_on_rels:
            # Crear subgrafo solo con relaciones DEPENDS_ON
            subgraph = KnowledgeGraph()
            for rel in depends_on_rels:
                # Añadir nodos si no existen
                source_node = self.graph.find_node(rel.source_id)
                target_node = self.graph.find_node(rel.target_id)
                
                if source_node and source_node.id not in subgraph.nodes:
                    subgraph.add_node(source_node)
                if target_node and target_node.id not in subgraph.nodes:
                    subgraph.add_node(target_node)
                
                # Añadir relación
                subgraph.add_edge(rel)
            
            # Buscar ciclos en el subgrafo
            sub_traverser = GraphTraverser(subgraph)
            cycles = sub_traverser.detect_cycles()
            
            for cycle in cycles:
                inconsistency = {
                    "type": "circular_dependency",
                    "message": f"Circular dependency detected with {len(cycle)} nodes",
                    "cycle": [node.id for node in cycle]
                }
                inconsistencies.append(inconsistency)
        
        # Verificar relaciones duplicadas
        seen_relationships = set()
        for rel in self.graph.relationships.values():
            key = (rel.source_id, rel.target_id, rel.type)
            if key in seen_relationships:
                inconsistencies.append({
                    "type": "duplicate_relationship",
                    "message": f"Duplicate relationship: {rel.source_id} -> {rel.target_id} ({rel.type})",
                    "relationship_id": rel.id
                })
            else:
                seen_relationships.add(key)
        
        return inconsistencies
    
    def generate_consistency_report(self, output_format: str = "json") -> Any:
        """
        Genera un reporte de consistencia.
        
        Args:
            output_format: Formato del reporte.
            
        Returns:
            Reporte en el formato especificado.
        """
        results = self.check_graph_consistency()
        
        if output_format == "json":
            import json
            return json.dumps(results, indent=2, default=str)
        elif output_format == "html":
            return self._generate_html_report(results)
        elif output_format == "text":
            return self._generate_text_report(results)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def auto_fix_inconsistencies(self) -> Dict[str, Any]:
        """
        Intenta corregir inconsistencias automáticamente.
        
        Returns:
            Diccionario con correcciones aplicadas.
        """
        fixes = {
            "fixed_orphan_nodes": [],
            "removed_invalid_references": [],
            "corrected_property_types": [],
            "removed_duplicate_relationships": []
        }
        
        # Corregir nodos huérfanos (eliminarlos)
        orphan_nodes = self.detect_orphan_nodes()
        for node_id in orphan_nodes:
            self.graph.remove_node(node_id)
            fixes["fixed_orphan_nodes"].append(node_id)
        
        # Corregir referencias inválidas (eliminar relaciones)
        for rel_id, rel in list(self.graph.relationships.items()):
            if (rel.source_id not in self.graph.nodes or 
                rel.target_id not in self.graph.nodes):
                self.graph.remove_edge(rel_id)
                fixes["removed_invalid_references"].append(rel_id)
        
        # Eliminar relaciones duplicadas
        seen_relationships = set()
        for rel in list(self.graph.relationships.values()):
            key = (rel.source_id, rel.target_id, rel.type)
            if key in seen_relationships:
                self.graph.remove_edge(rel.id)
                fixes["removed_duplicate_relationships"].append(rel.id)
            else:
                seen_relationships.add(key)
        
        # Corregir tipos de propiedades (conversión básica)
        # Por ahora, no hacemos correcciones de tipo automáticas.
        
        return fixes
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Genera reporte HTML."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Graph Consistency Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .error { color: #d32f2f; background-color: #ffcdd2; padding: 10px; margin: 5px 0; }
                .warning { color: #f57c00; background-color: #ffe0b2; padding: 10px; margin: 5px 0; }
                .stats { background-color: #e8f5e8; padding: 15px; margin: 10px 0; }
                .section { margin: 20px 0; }
                h1, h2, h3 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Graph Consistency Report</h1>
            <div class="stats">
                <h2>Statistics</h2>
                <p>Total Nodes: {total_nodes}</p>
                <p>Total Relationships: {total_relationships}</p>
                <p>Orphan Nodes: {orphan_nodes}</p>
                <p>Errors: {errors_count}</p>
                <p>Warnings: {warnings_count}</p>
            </div>
        """.format(**results["stats"])
        
        if results["errors"]:
            html += '<div class="section"><h2>Errors</h2>'
            for error in results["errors"]:
                html += f'<div class="error"><strong>{error.get("type", "Unknown")}:</strong> {error.get("message", "")}</div>'
            html += '</div>'
        
        if results["warnings"]:
            html += '<div class="section"><h2>Warnings</h2>'
            for warning in results["warnings"]:
                html += f'<div class="warning"><strong>{warning.get("type", "Unknown")}:</strong> {warning.get("message", "")}</div>'
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Genera reporte de texto."""
        lines = ["Graph Consistency Report", "=" * 30]
        
        # Estadísticas
        lines.append("\nStatistics:")
        lines.append(f"  Total Nodes: {results['stats']['total_nodes']}")
        lines.append(f"  Total Relationships: {results['stats']['total_relationships']}")
        lines.append(f"  Orphan Nodes: {results['stats']['orphan_nodes']}")
        lines.append(f"  Errors: {results['stats']['errors_count']}")
        lines.append(f"  Warnings: {results['stats']['warnings_count']}")
        
        # Errores
        if results["errors"]:
            lines.append("\nErrors:")
            for error in results["errors"]:
                lines.append(f"  [{error.get('type', 'Unknown')}] {error.get('message', '')}")
        
        # Advertencias
        if results["warnings"]:
            lines.append("\nWarnings:")
            for warning in results["warnings"]:
                lines.append(f"  [{warning.get('type', 'Unknown')}] {warning.get('message', '')}")
        
        return "\n".join(lines)