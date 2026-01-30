"""
SchemaManager - Gestión de esquemas para el grafo de conocimiento.
Define, valida y hace cumplir esquemas para nodos y relaciones.
"""

from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime
from .knowledge_graph import KnowledgeGraph, Node, Relationship, NodeType, RelationshipType

class PropertyType(Enum):
    """Tipos de propiedades soportados."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    LIST = "list"
    DICT = "dict"
    ANY = "any"

@dataclass
class PropertyConstraint:
    """Restricción para una propiedad."""
    required: bool = False
    default: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    unique: bool = False

@dataclass
class PropertySchema:
    """Esquema de una propiedad."""
    name: str
    type: PropertyType
    description: str = ""
    constraints: PropertyConstraint = field(default_factory=PropertyConstraint)

@dataclass
class NodeSchema:
    """Esquema para un tipo de nodo."""
    node_type: NodeType
    description: str = ""
    required_labels: Set[str] = field(default_factory=set)
    optional_labels: Set[str] = field(default_factory=set)
    required_properties: Dict[str, PropertySchema] = field(default_factory=dict)
    optional_properties: Dict[str, PropertySchema] = field(default_factory=dict)
    allowed_relationships: Dict[RelationshipType, List[NodeType]] = field(default_factory=dict)
    
    def get_all_properties(self) -> Dict[str, PropertySchema]:
        """Obtiene todas las propiedades."""
        all_props = self.required_properties.copy()
        all_props.update(self.optional_properties)
        return all_props

@dataclass
class RelationshipSchema:
    """Esquema para un tipo de relación."""
    relationship_type: RelationshipType
    description: str = ""
    allowed_source_types: List[NodeType] = field(default_factory=list)
    allowed_target_types: List[NodeType] = field(default_factory=list)
    required_properties: Dict[str, PropertySchema] = field(default_factory=dict)
    optional_properties: Dict[str, PropertySchema] = field(default_factory=dict)
    
    def get_all_properties(self) -> Dict[str, PropertySchema]:
        """Obtiene todas las propiedades."""
        all_props = self.required_properties.copy()
        all_props.update(self.optional_properties)
        return all_props

@dataclass
class SchemaValidationResult:
    """Resultado de validación de esquema."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)

class SchemaManager:
    """
    Gestor de esquemas para el grafo de conocimiento.
    """
    
    def __init__(self):
        """
        Inicializa el gestor de esquemas.
        """
        self.node_schemas: Dict[NodeType, NodeSchema] = {}
        self.relationship_schemas: Dict[RelationshipType, RelationshipSchema] = {}
        self.default_schema_enabled = True
        
        # Inicializar esquemas por defecto
        self._initialize_default_schemas()
    
    def define_schema(self, 
                     schema: Union[NodeSchema, RelationshipSchema]) -> bool:
        """
        Define un nuevo esquema.
        
        Args:
            schema: Esquema a definir
            
        Returns:
            True si se definió exitosamente
        """
        try:
            if isinstance(schema, NodeSchema):
                self.node_schemas[schema.node_type] = schema
            elif isinstance(schema, RelationshipSchema):
                self.relationship_schemas[schema.relationship_type] = schema
            else:
                raise ValueError("Schema must be NodeSchema or RelationshipSchema")
            
            return True
            
        except Exception as e:
            print(f"Error defining schema: {e}")
            return False
    
    def validate_node_schema(self, node: Node) -> SchemaValidationResult:
        """
        Valida un nodo contra su esquema.
        
        Args:
            node: Nodo a validar
            
        Returns:
            Resultado de la validación
        """
        result = SchemaValidationResult(valid=True)
        
        # Obtener esquema para este tipo de nodo
        schema = self.node_schemas.get(node.type)
        
        if not schema:
            if self.default_schema_enabled:
                # Usar esquema por defecto
                schema = self._get_default_node_schema(node.type)
            else:
                # Sin esquema definido, todo válido
                return result
        
        # Validar etiquetas
        if schema.required_labels:
            for label in schema.required_labels:
                if label not in node.labels:
                    result.valid = False
                    result.errors.append(f"Missing required label: {label}")
        
        # Validar propiedades requeridas
        for prop_name, prop_schema in schema.required_properties.items():
            if prop_name not in node.properties:
                result.valid = False
                result.errors.append(f"Missing required property: {prop_name}")
            else:
                # Validar tipo y restricciones
                prop_result = self._validate_property(
                    node.properties[prop_name],
                    prop_schema,
                    f"property '{prop_name}'"
                )
                if not prop_result.valid:
                    result.valid = False
                    result.errors.extend(prop_result.errors)
        
        # Validar propiedades opcionales (si están presentes)
        for prop_name, prop_schema in schema.optional_properties.items():
            if prop_name in node.properties:
                prop_result = self._validate_property(
                    node.properties[prop_name],
                    prop_schema,
                    f"property '{prop_name}'"
                )
                if not prop_result.valid:
                    result.valid = False
                    result.errors.extend(prop_result.errors)
        
        return result
    
    def validate_edge_schema(self, relationship: Relationship, 
                           source_node: Node, target_node: Node) -> SchemaValidationResult:
        """
        Valida una relación contra su esquema.
        
        Args:
            relationship: Relación a validar
            source_node: Nodo fuente
            target_node: Nodo destino
            
        Returns:
            Resultado de la validación
        """
        result = SchemaValidationResult(valid=True)
        
        # Obtener esquema para este tipo de relación
        schema = self.relationship_schemas.get(relationship.type)
        
        if not schema:
            if self.default_schema_enabled:
                # Usar esquema por defecto
                schema = self._get_default_relationship_schema(relationship.type)
            else:
                # Sin esquema definido, todo válido
                return result
        
        # Validar tipos de nodos
        if schema.allowed_source_types and source_node.type not in schema.allowed_source_types:
            result.valid = False
            result.errors.append(
                f"Source node type '{source_node.type.value}' not allowed for "
                f"relationship '{relationship.type.value}'"
            )
        
        if schema.allowed_target_types and target_node.type not in schema.allowed_target_types:
            result.valid = False
            result.errors.append(
                f"Target node type '{target_node.type.value}' not allowed for "
                f"relationship '{relationship.type.value}'"
            )
        
        # Validar propiedades requeridas
        for prop_name, prop_schema in schema.required_properties.items():
            if prop_name not in relationship.properties:
                result.valid = False
                result.errors.append(f"Missing required property: {prop_name}")
            else:
                # Validar tipo y restricciones
                prop_result = self._validate_property(
                    relationship.properties[prop_name],
                    prop_schema,
                    f"property '{prop_name}'"
                )
                if not prop_result.valid:
                    result.valid = False
                    result.errors.extend(prop_result.errors)
        
        # Validar propiedades opcionales (si están presentes)
        for prop_name, prop_schema in schema.optional_properties.items():
            if prop_name in relationship.properties:
                prop_result = self._validate_property(
                    relationship.properties[prop_name],
                    prop_schema,
                    f"property '{prop_name}'"
                )
                if not prop_result.valid:
                    result.valid = False
                    result.errors.extend(prop_result.errors)
        
        return result
    
    def migrate_schema(self, 
                      from_version: str, 
                      to_version: str,
                      graph: KnowledgeGraph) -> Tuple[bool, List[str]]:
        """
        Migra el grafo de una versión de esquema a otra.
        
        Args:
            from_version: Versión actual del esquema
            to_version: Versión objetivo del esquema
            graph: Grafo a migrar
            
        Returns:
            Tupla (éxito, lista de cambios aplicados)
        """
        changes_applied = []
        
        try:
            # Determinar cambios necesarios entre versiones
            migration_rules = self._get_migration_rules(from_version, to_version)
            
            # Aplicar reglas de migración
            for rule in migration_rules:
                if rule["type"] == "add_property":
                    self._apply_add_property_rule(graph, rule, changes_applied)
                elif rule["type"] == "remove_property":
                    self._apply_remove_property_rule(graph, rule, changes_applied)
                elif rule["type"] == "change_property_type":
                    self._apply_change_property_type_rule(graph, rule, changes_applied)
                elif rule["type"] == "add_label":
                    self._apply_add_label_rule(graph, rule, changes_applied)
                elif rule["type"] == "split_node_type":
                    self._apply_split_node_type_rule(graph, rule, changes_applied)
            
            return True, changes_applied
            
        except Exception as e:
            return False, [f"Migration failed: {str(e)}"]
    
    def export_schema(self, 
                     node_types: Optional[List[NodeType]] = None,
                     relationship_types: Optional[List[RelationshipType]] = None) -> Dict[str, Any]:
        """
        Exporta los esquemas definidos.
        
        Args:
            node_types: Tipos de nodo a exportar (None = todos)
            relationship_types: Tipos de relación a exportar (None = todos)
            
        Returns:
            Esquemas exportados en formato diccionario
        """
        export = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "schema_version": "1.0",
                "node_schema_count": 0,
                "relationship_schema_count": 0
            },
            "node_schemas": {},
            "relationship_schemas": {}
        }
        
        # Exportar esquemas de nodos
        if node_types is None:
            node_types = list(self.node_schemas.keys())
        
        for node_type in node_types:
            if node_type in self.node_schemas:
                schema = self.node_schemas[node_type]
                export["node_schemas"][node_type.value] = self._schema_to_dict(schema)
        
        # Exportar esquemas de relaciones
        if relationship_types is None:
            relationship_types = list(self.relationship_schemas.keys())
        
        for rel_type in relationship_types:
            if rel_type in self.relationship_schemas:
                schema = self.relationship_schemas[rel_type]
                export["relationship_schemas"][rel_type.value] = self._schema_to_dict(schema)
        
        # Actualizar conteos
        export["metadata"]["node_schema_count"] = len(export["node_schemas"])
        export["metadata"]["relationship_schema_count"] = len(export["relationship_schemas"])
        
        return export
    
    def infer_schema(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        Infiere esquemas a partir del grafo.
        
        Args:
            graph: Grafo para inferir esquemas
            
        Returns:
            Esquemas inferidos
        """
        inferred_schemas = {
            "node_schemas": {},
            "relationship_schemas": {}
        }
        
        # Inferir esquemas de nodos por tipo
        nodes_by_type = {}
        for node in graph.nodes.values():
            node_type = node.type.value
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
        
        for node_type, nodes in nodes_by_type.items():
            schema = self._infer_node_schema(nodes)
            inferred_schemas["node_schemas"][node_type] = schema
        
        # Inferir esquemas de relaciones por tipo
        relationships_by_type = {}
        for rel in graph.relationships.values():
            rel_type = rel.type.value
            if rel_type not in relationships_by_type:
                relationships_by_type[rel_type] = []
            relationships_by_type[rel_type].append(rel)
        
        for rel_type, relationships in relationships_by_type.items():
            schema = self._infer_relationship_schema(relationships, graph)
            inferred_schemas["relationship_schemas"][rel_type] = schema
        
        return inferred_schemas
    
    def enforce_schema(self, graph: KnowledgeGraph, strict: bool = True) -> Dict[str, Any]:
        """
        Hace cumplir esquemas en el grafo.
        
        Args:
            graph: Grafo en el que hacer cumplir esquemas
            strict: Si es True, elimina nodos/relaciones inválidas
            
        Returns:
            Reporte de aplicación
        """
        report = {
            "nodes_validated": 0,
            "nodes_fixed": 0,
            "nodes_removed": 0,
            "relationships_validated": 0,
            "relationships_fixed": 0,
            "relationships_removed": 0,
            "errors": [],
            "fixes_applied": []
        }
        
        # Validar y corregir nodos
        nodes_to_remove = []
        
        for node_id, node in graph.nodes.items():
            report["nodes_validated"] += 1
            validation = self.validate_node_schema(node)
            
            if not validation.valid:
                # Intentar corregir
                if self._can_fix_node(node, validation):
                    fixes = self._fix_node(node, validation)
                    report["nodes_fixed"] += 1
                    report["fixes_applied"].extend(fixes)
                elif strict:
                    nodes_to_remove.append(node_id)
                    report["nodes_removed"] += 1
                    report["errors"].append(f"Removed invalid node {node_id}")
        
        # Eliminar nodos inválidos
        for node_id in nodes_to_remove:
            graph.remove_node(node_id, cascade=True)
        
        # Validar y corregir relaciones
        relationships_to_remove = []
        
        for rel_id, rel in graph.relationships.items():
            # Obtener nodos fuente y destino
            source_node = graph.find_node(rel.source_id)
            target_node = graph.find_node(rel.target_id)
            
            if not source_node or not target_node:
                # Relación huérfana
                relationships_to_remove.append(rel_id)
                continue
            
            report["relationships_validated"] += 1
            validation = self.validate_edge_schema(rel, source_node, target_node)
            
            if not validation.valid:
                # Intentar corregir
                if self._can_fix_relationship(rel, validation):
                    fixes = self._fix_relationship(rel, validation)
                    report["relationships_fixed"] += 1
                    report["fixes_applied"].extend(fixes)
                elif strict:
                    relationships_to_remove.append(rel_id)
                    report["relationships_removed"] += 1
                    report["errors"].append(f"Removed invalid relationship {rel_id}")
        
        # Eliminar relaciones inválidas
        for rel_id in relationships_to_remove:
            graph.remove_edge(rel_id)
        
        return report
    
    # Métodos de implementación
    
    def _initialize_default_schemas(self) -> None:
        """Inicializa esquemas por defecto para tipos comunes."""
        # Esquema para nodos FILE
        file_schema = NodeSchema(
            node_type=NodeType.FILE,
            description="Archivo de código fuente",
            required_labels={"file"},
            required_properties={
                "path": PropertySchema(
                    name="path",
                    type=PropertyType.STRING,
                    description="Ruta del archivo",
                    constraints=PropertyConstraint(required=True)
                ),
                "name": PropertySchema(
                    name="name",
                    type=PropertyType.STRING,
                    description="Nombre del archivo",
                    constraints=PropertyConstraint(required=True)
                )
            },
            optional_properties={
                "extension": PropertySchema(
                    name="extension",
                    type=PropertyType.STRING,
                    description="Extensión del archivo"
                ),
                "language": PropertySchema(
                    name="language",
                    type=PropertyType.STRING,
                    description="Lenguaje de programación"
                )
            }
        )
        self.define_schema(file_schema)
        
        # Esquema para nodos FUNCTION
        function_schema = NodeSchema(
            node_type=NodeType.FUNCTION,
            description="Función o método",
            required_labels={"function"},
            required_properties={
                "name": PropertySchema(
                    name="name",
                    type=PropertyType.STRING,
                    description="Nombre de la función",
                    constraints=PropertyConstraint(required=True)
                ),
                "signature": PropertySchema(
                    name="signature",
                    type=PropertyType.STRING,
                    description="Firma de la función",
                    constraints=PropertyConstraint(required=True)
                )
            }
        )
        self.define_schema(function_schema)
        
        # Esquema para relación CALLS
        calls_schema = RelationshipSchema(
            relationship_type=RelationshipType.CALLS,
            description="Una función llama a otra",
            allowed_source_types=[NodeType.FUNCTION],
            allowed_target_types=[NodeType.FUNCTION]
        )
        self.define_schema(calls_schema)
        
        # Esquema para relación CONTAINS
        contains_schema = RelationshipSchema(
            relationship_type=RelationshipType.CONTAINS,
            description="Un archivo contiene una función/clase",
            allowed_source_types=[NodeType.FILE],
            allowed_target_types=[NodeType.FUNCTION, NodeType.CLASS]
        )
        self.define_schema(contains_schema)
    
    def _get_default_node_schema(self, node_type: NodeType) -> NodeSchema:
        """Obtiene esquema por defecto para un tipo de nodo."""
        return NodeSchema(
            node_type=node_type,
            description=f"Default schema for {node_type.value}",
            required_labels={node_type.value}
        )
    
    def _get_default_relationship_schema(self, rel_type: RelationshipType) -> RelationshipSchema:
        """Obtiene esquema por defecto para un tipo de relación."""
        return RelationshipSchema(
            relationship_type=rel_type,
            description=f"Default schema for {rel_type.value}"
        )
    
    def _validate_property(self, value: Any, schema: PropertySchema, context: str) -> SchemaValidationResult:
        """Valida una propiedad individual."""
        result = SchemaValidationResult(valid=True)
        
        # Validar tipo
        type_valid = self._validate_property_type(value, schema.type)
        if not type_valid:
            result.valid = False
            result.errors.append(f"{context}: Expected type {schema.type.value}, got {type(value).__name__}")
            return result
        
        # Validar restricciones
        constraints = schema.constraints
        
        # Validar longitud para strings
        if schema.type == PropertyType.STRING and isinstance(value, str):
            if constraints.min_length is not None and len(value) < constraints.min_length:
                result.valid = False
                result.errors.append(f"{context}: Minimum length is {constraints.min_length}")
            
            if constraints.max_length is not None and len(value) > constraints.max_length:
                result.valid = False
                result.errors.append(f"{context}: Maximum length is {constraints.max_length}")
            
            if constraints.pattern is not None:
                if not re.match(constraints.pattern, value):
                    result.valid = False
                    result.errors.append(f"{context}: Pattern '{constraints.pattern}' not matched")
        
        # Validar rango para números
        if schema.type in [PropertyType.INTEGER, PropertyType.FLOAT] and isinstance(value, (int, float)):
            if constraints.min_value is not None and value < constraints.min_value:
                result.valid = False
                result.errors.append(f"{context}: Minimum value is {constraints.min_value}")
            
            if constraints.max_value is not None and value > constraints.max_value:
                result.valid = False
                result.errors.append(f"{context}: Maximum value is {constraints.max_value}")
        
        # Validar valores permitidos
        if constraints.allowed_values is not None and value not in constraints.allowed_values:
            result.valid = False
            result.errors.append(f"{context}: Value not in allowed values {constraints.allowed_values}")
        
        return result
    
    def _validate_property_type(self, value: Any, expected_type: PropertyType) -> bool:
        """Valida que un valor sea del tipo esperado."""
        if expected_type == PropertyType.ANY:
            return True
        
        elif expected_type == PropertyType.STRING:
            return isinstance(value, str)
        
        elif expected_type == PropertyType.INTEGER:
            return isinstance(value, int)
        
        elif expected_type == PropertyType.FLOAT:
            return isinstance(value, (int, float))
        
        elif expected_type == PropertyType.BOOLEAN:
            return isinstance(value, bool)
        
        elif expected_type == PropertyType.DATETIME:
            return isinstance(value, (datetime, str))  # Strings también válidos para datetime
        
        elif expected_type == PropertyType.LIST:
            return isinstance(value, list)
        
        elif expected_type == PropertyType.DICT:
            return isinstance(value, dict)
        
        return False
    
    def _get_migration_rules(self, from_version: str, to_version: str) -> List[Dict[str, Any]]:
        """Obtiene reglas de migración entre versiones."""
        # En una implementación real, esto vendría de una base de datos o archivo
        # Por ahora, devolvemos reglas de ejemplo
        
        if from_version == "1.0" and to_version == "1.1":
            return [
                {
                    "type": "add_property",
                    "node_type": NodeType.FILE,
                    "property": PropertySchema(
                        name="encoding",
                        type=PropertyType.STRING,
                        description="Codificación del archivo",
                        constraints=PropertyConstraint(
                            required=False,
                            default="utf-8"
                        )
                    )
                },
                {
                    "type": "change_property_type",
                    "node_type": NodeType.FUNCTION,
                    "property_name": "line_count",
                    "new_type": PropertyType.INTEGER,
                    "conversion": "int"
                }
            ]
        
        return []
    
    def _apply_add_property_rule(self, graph: KnowledgeGraph, rule: Dict[str, Any], changes: List[str]) -> None:
        """Aplica regla de añadir propiedad."""
        node_type = rule["node_type"]
        property_schema = rule["property"]
        
        for node in graph.nodes.values():
            if node.type == node_type and property_schema.name not in node.properties:
                node.properties[property_schema.name] = property_schema.constraints.default
                changes.append(f"Added property '{property_schema.name}' to node {node.id}")
    
    def _apply_remove_property_rule(self, graph: KnowledgeGraph, rule: Dict[str, Any], changes: List[str]) -> None:
        """Aplica regla de eliminar propiedad."""
        node_type = rule["node_type"]
        property_name = rule["property_name"]
        
        for node in graph.nodes.values():
            if node.type == node_type and property_name in node.properties:
                del node.properties[property_name]
                changes.append(f"Removed property '{property_name}' from node {node.id}")
    
    def _apply_change_property_type_rule(self, graph: KnowledgeGraph, rule: Dict[str, Any], changes: List[str]) -> None:
        """Aplica regla de cambiar tipo de propiedad."""
        node_type = rule["node_type"]
        property_name = rule["property_name"]
        conversion = rule.get("conversion")
        
        for node in graph.nodes.values():
            if node.type == node_type and property_name in node.properties:
                old_value = node.properties[property_name]
                
                try:
                    if conversion == "int":
                        new_value = int(old_value)
                    elif conversion == "float":
                        new_value = float(old_value)
                    elif conversion == "str":
                        new_value = str(old_value)
                    else:
                        # Conversión por defecto
                        new_value = old_value
                    
                    node.properties[property_name] = new_value
                    changes.append(f"Converted property '{property_name}' for node {node.id}")
                    
                except (ValueError, TypeError):
                    # Si no se puede convertir, eliminar la propiedad
                    del node.properties[property_name]
                    changes.append(f"Removed invalid property '{property_name}' from node {node.id}")
    
    def _apply_add_label_rule(self, graph: KnowledgeGraph, rule: Dict[str, Any], changes: List[str]) -> None:
        """Aplica regla de añadir etiqueta."""
        node_type = rule["node_type"]
        label = rule["label"]
        
        for node in graph.nodes.values():
            if node.type == node_type and label not in node.labels:
                node.labels.add(label)
                changes.append(f"Added label '{label}' to node {node.id}")
    
    def _apply_split_node_type_rule(self, graph: KnowledgeGraph, rule: Dict[str, Any], changes: List[str]) -> None:
        """Aplica regla de dividir tipo de nodo."""
        old_type = rule["old_type"]
        new_type = rule["new_type"]
        condition = rule.get("condition")
        
        for node in graph.nodes.values():
            if node.type == old_type:
                # Evaluar condición si existe
                should_split = True
                if condition:
                    # Condición simple basada en propiedades
                    prop_name = condition.get("property")
                    expected_value = condition.get("value")
                    
                    if prop_name and expected_value:
                        should_split = node.properties.get(prop_name) == expected_value
                
                if should_split:
                    node.type = new_type
                    changes.append(f"Changed node {node.id} from {old_type.value} to {new_type.value}")
    
    def _schema_to_dict(self, schema: Union[NodeSchema, RelationshipSchema]) -> Dict[str, Any]:
        """Convierte un esquema a diccionario."""
        if isinstance(schema, NodeSchema):
            return {
                "type": "node_schema",
                "node_type": schema.node_type.value,
                "description": schema.description,
                "required_labels": list(schema.required_labels),
                "optional_labels": list(schema.optional_labels),
                "required_properties": {
                    name: self._property_schema_to_dict(prop)
                    for name, prop in schema.required_properties.items()
                },
                "optional_properties": {
                    name: self._property_schema_to_dict(prop)
                    for name, prop in schema.optional_properties.items()
                },
                "allowed_relationships": {
                    rel_type.value: [nt.value for nt in node_types]
                    for rel_type, node_types in schema.allowed_relationships.items()
                }
            }
        else:
            return {
                "type": "relationship_schema",
                "relationship_type": schema.relationship_type.value,
                "description": schema.description,
                "allowed_source_types": [nt.value for nt in schema.allowed_source_types],
                "allowed_target_types": [nt.value for nt in schema.allowed_target_types],
                "required_properties": {
                    name: self._property_schema_to_dict(prop)
                    for name, prop in schema.required_properties.items()
                },
                "optional_properties": {
                    name: self._property_schema_to_dict(prop)
                    for name, prop in schema.optional_properties.items()
                }
            }
    
    def _property_schema_to_dict(self, prop_schema: PropertySchema) -> Dict[str, Any]:
        """Convierte un esquema de propiedad a diccionario."""
        return {
            "name": prop_schema.name,
            "type": prop_schema.type.value,
            "description": prop_schema.description,
            "constraints": {
                "required": prop_schema.constraints.required,
                "default": prop_schema.constraints.default,
                "min_value": prop_schema.constraints.min_value,
                "max_value": prop_schema.constraints.max_value,
                "min_length": prop_schema.constraints.min_length,
                "max_length": prop_schema.constraints.max_length,
                "pattern": prop_schema.constraints.pattern,
                "allowed_values": prop_schema.constraints.allowed_values,
                "unique": prop_schema.constraints.unique
            }
        }
    
    def _infer_node_schema(self, nodes: List[Node]) -> Dict[str, Any]:
        """Infere esquema a partir de una lista de nodos del mismo tipo."""
        if not nodes:
            return {}
        
        # Recolectar todas las propiedades
        all_properties = {}
        all_labels = set()
        
        for node in nodes:
            all_labels.update(node.labels)
            
            for prop_name, prop_value in node.properties.items():
                if prop_name not in all_properties:
                    all_properties[prop_name] = {
                        "values": [],
                        "types": set()
                    }
                
                all_properties[prop_name]["values"].append(prop_value)
                all_properties[prop_name]["types"].add(type(prop_value).__name__)
        
        # Determinar qué propiedades son requeridas
        required_props = {}
        optional_props = {}
        
        for prop_name, prop_data in all_properties.items():
            # Propiedad presente en todos los nodos = requerida
            if len(prop_data["values"]) == len(nodes):
                required_props[prop_name] = self._infer_property_schema(prop_name, prop_data)
            else:
                optional_props[prop_name] = self._infer_property_schema(prop_name, prop_data)
        
        return {
            "node_type": nodes[0].type.value,
            "sample_size": len(nodes),
            "required_labels": list(all_labels),
            "required_properties": required_props,
            "optional_properties": optional_props
        }
    
    def _infer_relationship_schema(self, relationships: List[Relationship], 
                                 graph: KnowledgeGraph) -> Dict[str, Any]:
        """Infere esquema a partir de una lista de relaciones del mismo tipo."""
        if not relationships:
            return {}
        
        # Recolectar información
        all_properties = {}
        source_types = set()
        target_types = set()
        
        for rel in relationships:
            source_node = graph.find_node(rel.source_id)
            target_node = graph.find_node(rel.target_id)
            
            if source_node:
                source_types.add(source_node.type.value)
            if target_node:
                target_types.add(target_node.type.value)
            
            for prop_name, prop_value in rel.properties.items():
                if prop_name not in all_properties:
                    all_properties[prop_name] = {
                        "values": [],
                        "types": set()
                    }
                
                all_properties[prop_name]["values"].append(prop_value)
                all_properties[prop_name]["types"].add(type(prop_value).__name__)
        
        # Determinar propiedades
        required_props = {}
        optional_props = {}
        
        for prop_name, prop_data in all_properties.items():
            if len(prop_data["values"]) == len(relationships):
                required_props[prop_name] = self._infer_property_schema(prop_name, prop_data)
            else:
                optional_props[prop_name] = self._infer_property_schema(prop_name, prop_data)
        
        return {
            "relationship_type": relationships[0].type.value,
            "sample_size": len(relationships),
            "allowed_source_types": list(source_types),
            "allowed_target_types": list(target_types),
            "required_properties": required_props,
            "optional_properties": optional_props
        }
    
    def _infer_property_schema(self, prop_name: str, prop_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infere esquema para una propiedad individual."""
        # Determinar tipo
        types = prop_data["types"]
        values = prop_data["values"]
        
        if len(types) == 1:
            prop_type = next(iter(types))
        else:
            prop_type = "mixed"
        
        # Mapear tipo de Python a PropertyType
        type_mapping = {
            "str": PropertyType.STRING.value,
            "int": PropertyType.INTEGER.value,
            "float": PropertyType.FLOAT.value,
            "bool": PropertyType.BOOLEAN.value,
            "list": PropertyType.LIST.value,
            "dict": PropertyType.DICT.value
        }
        
        schema_type = type_mapping.get(prop_type, PropertyType.ANY.value)
        
        # Calcular estadísticas para strings y números
        constraints = {}
        
        if "str" in types:
            string_lengths = [len(v) for v in values if isinstance(v, str)]
            if string_lengths:
                constraints["min_length"] = min(string_lengths)
                constraints["max_length"] = max(string_lengths)
        
        if "int" in types or "float" in types:
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                constraints["min_value"] = min(numeric_values)
                constraints["max_value"] = max(numeric_values)
        
        return {
            "name": prop_name,
            "type": schema_type,
            "sample_count": len(values),
            "unique_count": len(set(values)),
            "constraints": constraints
        }
    
    def _can_fix_node(self, node: Node, validation: SchemaValidationResult) -> bool:
        """Determina si un nodo puede ser corregido."""
        # Por ahora, solo corregimos propiedades faltantes con valores por defecto
        for error in validation.errors:
            if error.startswith("Missing required property"):
                # Extraer nombre de propiedad
                match = re.search(r"property '(.+)'", error)
                if match:
                    prop_name = match.group(1)
                    # Verificar si tenemos un valor por defecto
                    schema = self.node_schemas.get(node.type)
                    if schema and prop_name in schema.required_properties:
                        if schema.required_properties[prop_name].constraints.default is not None:
                            return True
        
        return False
    
    def _fix_node(self, node: Node, validation: SchemaValidationResult) -> List[str]:
        """Corrige un nodo según los errores de validación."""
        fixes = []
        schema = self.node_schemas.get(node.type)
        
        if not schema:
            return fixes
        
        for error in validation.errors:
            if error.startswith("Missing required property"):
                match = re.search(r"property '(.+)'", error)
                if match:
                    prop_name = match.group(1)
                    if prop_name in schema.required_properties:
                        default_value = schema.required_properties[prop_name].constraints.default
                        if default_value is not None:
                            node.properties[prop_name] = default_value
                            fixes.append(f"Added missing property '{prop_name}' with default value")
        
        return fixes
    
    def _can_fix_relationship(self, relationship: Relationship, validation: SchemaValidationResult) -> bool:
        """Determina si una relación puede ser corregida."""
        # Similar a _can_fix_node
        for error in validation.errors:
            if error.startswith("Missing required property"):
                match = re.search(r"property '(.+)'", error)
                if match:
                    prop_name = match.group(1)
                    schema = self.relationship_schemas.get(relationship.type)
                    if schema and prop_name in schema.required_properties:
                        if schema.required_properties[prop_name].constraints.default is not None:
                            return True
        
        return False
    
    def _fix_relationship(self, relationship: Relationship, validation: SchemaValidationResult) -> List[str]:
        """Corrige una relación según los errores de validación."""
        fixes = []
        schema = self.relationship_schemas.get(relationship.type)
        
        if not schema:
            return fixes
        
        for error in validation.errors:
            if error.startswith("Missing required property"):
                match = re.search(r"property '(.+)'", error)
                if match:
                    prop_name = match.group(1)
                    if prop_name in schema.required_properties:
                        default_value = schema.required_properties[prop_name].constraints.default
                        if default_value is not None:
                            relationship.properties[prop_name] = default_value
                            fixes.append(f"Added missing property '{prop_name}' with default value")
        
        return fixes