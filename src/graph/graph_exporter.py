"""
GraphExporter - Exportación de grafos a diferentes formatos.
Convierte el grafo de conocimiento a formatos estándar como GEXF, GraphML, JSON, etc.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import uuid
from datetime import datetime

# Importar desde los modelos definidos en las especificaciones
from ..models.entities import (
    KnowledgeGraph, 
    BaseEntity, 
    Relationship, 
    EntityType, 
    RelationshipType
)

class ExportFormat(Enum):
    """Formatos de exportación soportados."""
    GEXF = "gexf"
    GRAPHML = "graphml"
    JSON = "json"
    CYPHER = "cypher"
    DOT = "dot"
    NETWORKX = "networkx"
    CSV = "csv"
    YAML = "yaml"

@dataclass
class ExportOptions:
    """Opciones de exportación."""
    include_properties: bool = True
    include_timestamps: bool = True
    pretty_print: bool = True
    compress: bool = False
    max_nodes: Optional[int] = None
    max_relationships: Optional[int] = None
    node_filter: Optional[Dict[str, Any]] = None
    relationship_filter: Optional[Dict[str, Any]] = None

class GraphExporter:
    """
    Exportador de grafos a múltiples formatos.
    """
    
    def __init__(self, graph: KnowledgeGraph):
        """
        Inicializa el exportador.
        
        Args:
            graph: Grafo a exportar
        """
        self.graph = graph
        
    def export_to_gexf(self, options: Optional[ExportOptions] = None) -> str:
        """
        Exporta el grafo a formato GEXF.
        
        Args:
            options: Opciones de exportación
            
        Returns:
            String XML en formato GEXF
        """
        options = options or ExportOptions()
        
        # Crear elemento raíz
        gexf = ET.Element("gexf", xmlns="http://www.gexf.net/1.3")
        gexf.set("version", "1.3")
        
        # Metadatos
        meta = ET.SubElement(gexf, "meta", lastmodifieddate=datetime.now().strftime("%Y-%m-%d"))
        creator = ET.SubElement(meta, "creator")
        creator.text = "Project Brain Graph Exporter"
        description = ET.SubElement(meta, "description")
        description.text = f"Knowledge graph exported on {datetime.now()}"
        
        # Grafo
        graph_elem = ET.SubElement(gexf, "graph", mode="static", defaultedgetype="directed")
        
        attributes = ET.SubElement(graph_elem, "attributes", attrib={"class": "node"})
        
        # Atributos estándar
        for attr_name, attr_type in [
            ("type", "string"),
            ("created_at", "string"),
            ("updated_at", "string")
        ]:
            attr = ET.SubElement(attributes, "attribute")
            attr.set("id", attr_name)
            attr.set("title", attr_name)
            attr.set("type", attr_type)
        
        # Atributos dinámicos de propiedades
        if options.include_properties:
            all_props = self._collect_all_node_properties()
            for i, prop_name in enumerate(all_props):
                attr = ET.SubElement(attributes, "attribute")
                attr.set("id", f"prop_{i}")
                attr.set("title", prop_name)
                attr.set("type", "string")
        
        # Nodos
        nodes_elem = ET.SubElement(graph_elem, "nodes")
        
        filtered_nodes = self._filter_nodes(options)
        for i, (entity_id, entity) in enumerate(filtered_nodes):
            if options.max_nodes and i >= options.max_nodes:
                break
            
            node_elem = ET.SubElement(nodes_elem, "node")
            node_elem.set("id", entity_id)
            node_elem.set("label", entity.properties.get("name", entity_id) if hasattr(entity, 'properties') else entity_id)
            
            # Atributos
            attvalues = ET.SubElement(node_elem, "attvalues")
            
            # Atributos estándar
            self._add_attvalue(attvalues, "type", entity.type.value)
            
            if options.include_timestamps:
                self._add_attvalue(attvalues, "created_at", entity.created_at.isoformat())
                self._add_attvalue(attvalues, "updated_at", entity.updated_at.isoformat())
            
            # Propiedades
            if options.include_properties and hasattr(entity, 'properties'):
                for j, prop_name in enumerate(all_props):
                    if prop_name in entity.properties:
                        value = str(entity.properties[prop_name])
                        self._add_attvalue(attvalues, f"prop_{j}", value)
        
        # Aristas
        edges_elem = ET.SubElement(graph_elem, "edges")
        
        filtered_rels = self._filter_relationships(options, filtered_nodes)
        for i, (rel_id, rel) in enumerate(filtered_rels):
            if options.max_relationships and i >= options.max_relationships:
                break
            
            edge_elem = ET.SubElement(edges_elem, "edge")
            edge_elem.set("id", rel_id)
            edge_elem.set("source", rel.source_id)
            edge_elem.set("target", rel.target_id)
            edge_elem.set("label", rel.type.value)
            
            # Propiedades de la relación
            if options.include_properties and rel.properties:
                attvalues = ET.SubElement(edge_elem, "attvalues")
                for prop_name, prop_value in rel.properties.items():
                    self._add_attvalue(attvalues, prop_name, str(prop_value))
        
        # Convertir a string XML
        xml_string = ET.tostring(gexf, encoding='utf-8').decode('utf-8')
        
        if options.pretty_print:
            xml_string = self._prettify_xml(xml_string)
        
        return xml_string
    
    def export_to_graphml(self, options: Optional[ExportOptions] = None) -> str:
        """
        Exporta el grafo a formato GraphML.
        
        Args:
            options: Opciones de exportación
            
        Returns:
            String XML en formato GraphML
        """
        options = options or ExportOptions()
        
        # Crear elemento raíz
        graphml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
        graphml.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        graphml.set("xsi:schemaLocation", 
                   "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd")
        
        # Atributos de nodos
        node_attrs = {}
        all_node_props = self._collect_all_node_properties() if options.include_properties else []
        
        for i, attr_name in enumerate(["type", "created_at", "updated_at"] + all_node_props):
            key = ET.SubElement(graphml, "key")
            key_id = f"node_{attr_name}"
            key.set("id", key_id)
            key.set("for", "node")
            key.set("attr.name", attr_name)
            key.set("attr.type", "string")
            node_attrs[attr_name] = key_id
        
        # Atributos de aristas
        edge_attrs = {}
        all_edge_props = self._collect_all_relationship_properties() if options.include_properties else []
        
        for i, attr_name in enumerate(["type"] + all_edge_props):
            key = ET.SubElement(graphml, "key")
            key_id = f"edge_{attr_name}"
            key.set("id", key_id)
            key.set("for", "edge")
            key.set("attr.name", attr_name)
            key.set("attr.type", "string")
            edge_attrs[attr_name] = key_id
        
        # Grafo
        graph_elem = ET.SubElement(graphml, "graph")
        graph_elem.set("id", "knowledge_graph")
        graph_elem.set("edgedefault", "directed")
        
        # Nodos
        filtered_nodes = self._filter_nodes(options)
        for i, (entity_id, entity) in enumerate(filtered_nodes):
            if options.max_nodes and i >= options.max_nodes:
                break
            
            node_elem = ET.SubElement(graph_elem, "node")
            node_elem.set("id", entity_id)
            
            # Etiqueta
            label = entity.properties.get("name", entity_id) if hasattr(entity, 'properties') else entity_id
            ET.SubElement(node_elem, "data", key="d0").text = label
            
            # Atributos
            data = ET.SubElement(node_elem, "data", key=node_attrs["type"])
            data.text = entity.type.value
            
            if options.include_timestamps:
                data = ET.SubElement(node_elem, "data", key=node_attrs["created_at"])
                data.text = entity.created_at.isoformat()
                
                data = ET.SubElement(node_elem, "data", key=node_attrs["updated_at"])
                data.text = entity.updated_at.isoformat()
            
            # Propiedades
            if options.include_properties and hasattr(entity, 'properties'):
                for prop_name in all_node_props:
                    if prop_name in entity.properties:
                        data = ET.SubElement(node_elem, "data", key=node_attrs[prop_name])
                        data.text = str(entity.properties[prop_name])
        
        # Aristas
        filtered_rels = self._filter_relationships(options, filtered_nodes)
        for i, (rel_id, rel) in enumerate(filtered_rels):
            if options.max_relationships and i >= options.max_relationships:
                break
            
            edge_elem = ET.SubElement(graph_elem, "edge")
            edge_elem.set("id", rel_id)
            edge_elem.set("source", rel.source_id)
            edge_elem.set("target", rel.target_id)
            
            # Atributos
            data = ET.SubElement(edge_elem, "data", key=edge_attrs["type"])
            data.text = rel.type.value
            
            # Propiedades
            if options.include_properties:
                for prop_name in all_edge_props:
                    if prop_name in rel.properties:
                        data = ET.SubElement(edge_elem, "data", key=edge_attrs[prop_name])
                        data.text = str(rel.properties[prop_name])
        
        # Convertir a string XML
        xml_string = ET.tostring(graphml, encoding='utf-8').decode('utf-8')
        
        if options.pretty_print:
            xml_string = self._prettify_xml(xml_string)
        
        return xml_string
    
    def export_to_json(self, options: Optional[ExportOptions] = None) -> str:
        """
        Exporta el grafo a formato JSON.
        
        Args:
            options: Opciones de exportación
            
        Returns:
            String JSON
        """
        options = options or ExportOptions()
        
        # Construir estructura de datos
        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "format_version": "1.0",
                "source": "Project Brain Knowledge Graph"
            },
            "graph": {
                "name": self.graph.project_id if hasattr(self.graph, 'project_id') else "knowledge_graph",
                "nodes": [],
                "relationships": []
            }
        }
        
        # Nodos
        filtered_nodes = self._filter_nodes(options)
        for i, (entity_id, entity) in enumerate(filtered_nodes):
            if options.max_nodes and i >= options.max_nodes:
                break
            
            node_data = {
                "id": entity_id,
                "type": entity.type.value,
                "properties": {}
            }
            
            if options.include_timestamps:
                node_data["created_at"] = entity.created_at.isoformat()
                node_data["updated_at"] = entity.updated_at.isoformat()
            
            if options.include_properties and hasattr(entity, 'properties'):
                node_data["properties"] = entity.properties.copy()
            else:
                # Incluir solo propiedades básicas
                basic_props = ["name", "description", "path"]
                if hasattr(entity, 'properties'):
                    for prop in basic_props:
                        if prop in entity.properties:
                            node_data["properties"][prop] = entity.properties[prop]
            
            data["graph"]["nodes"].append(node_data)
        
        # Relaciones
        filtered_rels = self._filter_relationships(options, filtered_nodes)
        for i, (rel_id, rel) in enumerate(filtered_rels):
            if options.max_relationships and i >= options.max_relationships:
                break
            
            rel_data = {
                "id": rel_id,
                "source": rel.source_id,
                "target": rel.target_id,
                "type": rel.type.value,
                "properties": {}
            }
            
            if options.include_timestamps:
                rel_data["created_at"] = rel.created_at.isoformat()
            
            if options.include_properties:
                rel_data["properties"] = rel.properties.copy()
            
            data["graph"]["relationships"].append(rel_data)
        
        # Estadísticas
        data["statistics"] = {
            "total_nodes": len(data["graph"]["nodes"]),
            "total_relationships": len(data["graph"]["relationships"]),
            "node_types": self._count_node_types(data["graph"]["nodes"]),
            "relationship_types": self._count_relationship_types(data["graph"]["relationships"])
        }
        
        # Convertir a JSON
        if options.pretty_print:
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    
    def export_to_cypher(self, options: Optional[ExportOptions] = None) -> str:
        """
        Exporta el grafo a formato Cypher (Neo4j).
        
        Args:
            options: Opciones de exportación
            
        Returns:
            String con comandos Cypher
        """
        options = options or ExportOptions()
        
        cypher_commands = []
        
        # Nodos
        filtered_nodes = self._filter_nodes(options)
        for i, (entity_id, entity) in enumerate(filtered_nodes):
            if options.max_nodes and i >= options.max_nodes:
                break
            
            # Crear etiquetas Cypher (usar type como label principal)
            labels = f"{entity.type.value.capitalize()}"
            
            # Crear propiedades
            props = {"id": entity_id, "type": entity.type.value}
            
            if options.include_timestamps:
                props["created_at"] = entity.created_at.isoformat()
                props["updated_at"] = entity.updated_at.isoformat()
            
            if options.include_properties and hasattr(entity, 'properties'):
                props.update(entity.properties)
            else:
                # Propiedades básicas
                if hasattr(entity, 'properties'):
                    if "name" in entity.properties:
                        props["name"] = entity.properties["name"]
                    if "description" in entity.properties:
                        props["description"] = entity.properties["description"]
            
            # Formatear propiedades para Cypher
            props_str = self._format_cypher_properties(props)
            
            # Comando CREATE
            cypher_commands.append(f"CREATE (n{entity_id}:{labels} {props_str})")
        
        # Relaciones
        filtered_rels = self._filter_relationships(options, filtered_nodes)
        for i, (rel_id, rel) in enumerate(filtered_rels):
            if options.max_relationships and i >= options.max_relationships:
                break
            
            # Verificar que ambos nodos existen en los filtrados
            source_exists = any(entity_id == rel.source_id for entity_id, _ in filtered_nodes)
            target_exists = any(entity_id == rel.target_id for entity_id, _ in filtered_nodes)
            
            if not source_exists or not target_exists:
                continue
            
            # Crear propiedades
            props = {"id": rel_id}
            
            if options.include_timestamps:
                props["created_at"] = rel.created_at.isoformat()
            
            if options.include_properties:
                props.update(rel.properties)
            
            # Formatear propiedades para Cypher
            props_str = self._format_cypher_properties(props) if props else ""
            
            # Comando MATCH y CREATE
            rel_type = rel.type.value.upper().replace("_", "")
            cypher_commands.append(
                f"MATCH (a {{id: '{rel.source_id}'}}), (b {{id: '{rel.target_id}'}}) "
                f"CREATE (a)-[:{rel_type}{props_str}]->(b)"
            )
        
        return "\n".join(cypher_commands) + ";"
    
    def export_to_dot(self, options: Optional[ExportOptions] = None) -> str:
        """
        Exporta el grafo a formato DOT (Graphviz).
        
        Args:
            options: Opciones de exportación
            
        Returns:
            String en formato DOT
        """
        options = options or ExportOptions()
        
        dot_lines = [
            "digraph KnowledgeGraph {",
            "  rankdir=LR;",
            "  node [shape=record, fontname=Arial];",
            "  edge [fontname=Arial, fontsize=10];"
        ]
        
        # Agrupar nodos por tipo para diferentes estilos
        type_styles = {
            EntityType.FILE: "shape=note, style=filled, fillcolor=lightblue",
            EntityType.FUNCTION: "shape=ellipse, style=filled, fillcolor=lightgreen",
            EntityType.CLASS: "shape=box, style=filled, fillcolor=lightyellow",
            EntityType.MODULE: "shape=octagon, style=filled, fillcolor=orange",
            EntityType.CONCEPT: "shape=diamond, style=filled, fillcolor=pink"
        }
        
        default_style = "shape=ellipse, style=filled, fillcolor=gray"
        
        # Nodos
        filtered_nodes = self._filter_nodes(options)
        for i, (entity_id, entity) in enumerate(filtered_nodes):
            if options.max_nodes and i >= options.max_nodes:
                break
            
            # Obtener estilo por tipo
            style = type_styles.get(entity.type, default_style)
            
            # Crear etiqueta
            name = entity.properties.get("name", entity_id[:8]) if hasattr(entity, 'properties') else entity_id[:8]
            label = f"{name}\\n({entity.type.value})"
            
            # Propiedades adicionales
            if options.include_properties and hasattr(entity, 'properties'):
                extra_props = []
                for key, value in entity.properties.items():
                    if key != "name" and isinstance(value, str) and len(value) < 20:
                        extra_props.append(f"{key}: {value}")
                
                if extra_props:
                    label += "\\n" + "\\n".join(extra_props)
            
            # Escapar caracteres especiales
            label = label.replace('"', '\\"')
            
            # Línea de nodo
            dot_lines.append(f'  "{entity_id}" [{style}, label="{label}"];')
        
        # Relaciones
        filtered_rels = self._filter_relationships(options, filtered_nodes)
        for i, (rel_id, rel) in enumerate(filtered_rels):
            if options.max_relationships and i >= options.max_relationships:
                break
            
            # Verificar que ambos nodos existen
            source_exists = any(entity_id == rel.source_id for entity_id, _ in filtered_nodes)
            target_exists = any(entity_id == rel.target_id for entity_id, _ in filtered_nodes)
            
            if not source_exists or not target_exists:
                continue
            
            # Crear etiqueta de relación
            label = rel.type.value
            
            if options.include_properties:
                extra_props = []
                for key, value in rel.properties.items():
                    if isinstance(value, (str, int, float)) and len(str(value)) < 10:
                        extra_props.append(f"{key}: {value}")
                
                if extra_props:
                    label += f"\\n{', '.join(extra_props)}"
            
            # Línea de relación
            dot_lines.append(f'  "{rel.source_id}" -> "{rel.target_id}" [label="{label}"];')
        
        dot_lines.append("}")
        
        return "\n".join(dot_lines)
    
    def export_to_networkx(self, options: Optional[ExportOptions] = None) -> 'nx.DiGraph':
        """
        Exporta el grafo a objeto NetworkX.
        
        Args:
            options: Opciones de exportación
            
        Returns:
            Grafo de NetworkX
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required for this export format. Install with: pip install networkx")
        
        nx_graph = nx.DiGraph()
        
        # Añadir nodos
        filtered_nodes = self._filter_nodes(options)
        for entity_id, entity in filtered_nodes:
            node_data = {
                "type": entity.type.value,
            }
            
            if options and options.include_timestamps:
                node_data["created_at"] = entity.created_at
                node_data["updated_at"] = entity.updated_at
            
            if options and options.include_properties and hasattr(entity, 'properties'):
                node_data.update(entity.properties)
            
            nx_graph.add_node(entity_id, **node_data)
        
        # Añadir aristas
        filtered_rels = self._filter_relationships(options, filtered_nodes)
        for rel_id, rel in filtered_rels:
            rel_data = {
                "type": rel.type.value,
                "id": rel_id
            }
            
            if options and options.include_timestamps:
                rel_data["created_at"] = rel.created_at
            
            if options and options.include_properties:
                rel_data.update(rel.properties)
            
            nx_graph.add_edge(rel.source_id, rel.target_id, **rel_data)
        
        return nx_graph
    
    def export(self, format: ExportFormat, options: Optional[ExportOptions] = None) -> Union[str, 'nx.DiGraph']:
        """
        Exporta el grafo en el formato especificado.
        
        Args:
            format: Formato de exportación
            options: Opciones de exportación
            
        Returns:
            Grafo en el formato especificado
        """
        if format == ExportFormat.GEXF:
            return self.export_to_gexf(options)
        elif format == ExportFormat.GRAPHML:
            return self.export_to_graphml(options)
        elif format == ExportFormat.JSON:
            return self.export_to_json(options)
        elif format == ExportFormat.CYPHER:
            return self.export_to_cypher(options)
        elif format == ExportFormat.DOT:
            return self.export_to_dot(options)
        elif format == ExportFormat.NETWORKX:
            return self.export_to_networkx(options)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def validate_export(self, export_data: Any, format: ExportFormat) -> bool:
        """
        Valida que los datos exportados sean correctos.
        
        Args:
            export_data: Datos exportados
            format: Formato de los datos
            
        Returns:
            True si la exportación es válida
        """
        try:
            if format == ExportFormat.JSON:
                # Validar JSON
                if isinstance(export_data, str):
                    parsed = json.loads(export_data)
                else:
                    parsed = export_data
                
                required_keys = ["metadata", "graph"]
                if not all(key in parsed for key in required_keys):
                    return False
                
                graph = parsed["graph"]
                if "nodes" not in graph or "relationships" not in graph:
                    return False
                
                return True
            
            elif format in [ExportFormat.GEXF, ExportFormat.GRAPHML]:
                # Validar XML básico
                if not isinstance(export_data, str):
                    return False
                
                # Verificar que sea XML válido
                try:
                    ET.fromstring(export_data)
                    return True
                except ET.ParseError:
                    return False
            
            elif format == ExportFormat.CYPHER:
                # Validar Cypher básico
                if not isinstance(export_data, str):
                    return False
                
                lines = export_data.strip().split('\n')
                if len(lines) == 0:
                    return False
                
                # Verificar que termine con punto y coma
                return export_data.strip().endswith(';')
            
            else:
                # Para otros formatos, asumir válido
                return True
                
        except Exception:
            return False
    
    # Métodos auxiliares
    
    def _filter_nodes(self, options: ExportOptions) -> List[Tuple[str, BaseEntity]]:
        """Filtra nodos según las opciones."""
        entities = list(self.graph.entities.items())
        
        if not options.node_filter:
            return entities
        
        filtered = []
        for entity_id, entity in entities:
            # Aplicar filtros
            match = True
            
            if "type" in options.node_filter:
                if entity.type.value != options.node_filter["type"]:
                    match = False
            
            if "properties" in options.node_filter:
                for key, value in options.node_filter["properties"].items():
                    if not hasattr(entity, 'properties') or key not in entity.properties or entity.properties[key] != value:
                        match = False
                        break
            
            if match:
                filtered.append((entity_id, entity))
        
        return filtered
    
    def _filter_relationships(self, options: ExportOptions, 
                             filtered_nodes: List[Tuple[str, BaseEntity]]) -> List[Tuple[str, Relationship]]:
        """Filtra relaciones según las opciones."""
        # Obtener IDs de nodos filtrados
        filtered_node_ids = {entity_id for entity_id, _ in filtered_nodes}
        
        filtered_rels = []
        for rel in self.graph.relationships:
            # Verificar que ambos nodos estén en los filtrados
            if rel.source_id not in filtered_node_ids or rel.target_id not in filtered_node_ids:
                continue
            
            # Aplicar filtros de relación si existen
            if options.relationship_filter:
                match = True
                
                if "type" in options.relationship_filter:
                    if rel.type.value != options.relationship_filter["type"]:
                        match = False
                
                if "properties" in options.relationship_filter:
                    for key, value in options.relationship_filter["properties"].items():
                        if key not in rel.properties or rel.properties[key] != value:
                            match = False
                            break
                
                if not match:
                    continue
            
            filtered_rels.append((rel.id, rel))
        
        return filtered_rels
    
    def _collect_all_node_properties(self) -> List[str]:
        """Recolecta todas las propiedades de nodos únicas."""
        all_props = set()
        
        for entity in self.graph.entities.values():
            if hasattr(entity, 'properties'):
                all_props.update(entity.properties.keys())
        
        return sorted(all_props)
    
    def _collect_all_relationship_properties(self) -> List[str]:
        """Recolecta todas las propiedades de relaciones únicas."""
        all_props = set()
        
        for rel in self.graph.relationships:
            all_props.update(rel.properties.keys())
        
        return sorted(all_props)
    
    def _add_attvalue(self, parent, for_key: str, value: str) -> None:
        """Añade un elemento attvalue a XML."""
        attvalue = ET.SubElement(parent, "attvalue")
        attvalue.set("for", for_key)
        attvalue.set("value", value)
    
    def _prettify_xml(self, xml_string: str) -> str:
        """Formatea XML para que sea legible."""
        try:
            parsed = minidom.parseString(xml_string)
            return parsed.toprettyxml(indent="  ")
        except:
            return xml_string
    
    def _format_cypher_properties(self, props: Dict[str, Any]) -> str:
        """Formatea propiedades para Cypher."""
        if not props:
            return ""
        
        formatted_props = []
        for key, value in props.items():
            if value is None:
                formatted_props.append(f"{key}: null")
            elif isinstance(value, str):
                # Escapar comillas simples
                escaped = value.replace("'", "\\'")
                formatted_props.append(f"{key}: '{escaped}'")
            elif isinstance(value, bool):
                formatted_props.append(f"{key}: {str(value).lower()}")
            elif isinstance(value, (int, float)):
                formatted_props.append(f"{key}: {value}")
            else:
                # Convertir otros tipos a string
                formatted_props.append(f"{key}: '{str(value)}'")
        
        return " {" + ", ".join(formatted_props) + "}"
    
    def _count_node_types(self, nodes: List[Dict]) -> Dict[str, int]:
        """Cuenta tipos de nodos."""
        counts = {}
        for node in nodes:
            node_type = node.get("type", "unknown")
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _count_relationship_types(self, relationships: List[Dict]) -> Dict[str, int]:
        """Cuenta tipos de relaciones."""
        counts = {}
        for rel in relationships:
            rel_type = rel.get("type", "unknown")
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts