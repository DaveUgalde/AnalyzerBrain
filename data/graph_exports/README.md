# Directorio de Exportaciones de Grafo

Este directorio almacena exportaciones del grafo de conocimiento en varios formatos.

## Formatos Soportados

1. **Cypher (.cypher)**: Para importar en Neo4j
2. **GraphML (.graphml)**: Formato estándar XML para grafos
3. **JSON (.json)**: Formato nativo de Project Brain
4. **DOT (.dot)**: Para visualización con Graphviz
5. **GEXF (.gexf)**: Para Gephi y otras herramientas

## Estructura de Archivos
5. ARCHIVO: data/graph_exports/README.md

markdown
# Directorio de Exportaciones de Grafo

Este directorio almacena exportaciones del grafo de conocimiento en varios formatos.

## Formatos Soportados

1. **Cypher (.cypher)**: Para importar en Neo4j
2. **GraphML (.graphml)**: Formato estándar XML para grafos
3. **JSON (.json)**: Formato nativo de Project Brain
4. **DOT (.dot)**: Para visualización con Graphviz
5. **GEXF (.gexf)**: Para Gephi y otras herramientas

## Estructura de Archivos
graph_exports/
├── {project_id}/ # Exportaciones por proyecto
│ ├── {timestamp}_full.cypher
│ ├── {timestamp}_full.graphml
│ ├── {timestamp}_summary.json
│ └── {timestamp}_visualization.dot
├── system/ # Exportaciones del sistema
│ ├── knowledge_graph_full.cypher
│ └── agents_network.graphml
└── templates/ # Plantillas de exportación
├── export_template.cypher
└── export_template.graphml

text

## Ejemplo de Uso

### Exportar a Cypher
```python
from src.graph.graph_exporter import GraphExporter

exporter = GraphExporter()
cypher_script = exporter.export_to_cypher(
    graph=knowledge_graph,
    include_properties=True,
    include_indexes=True
)

with open(f"data/graph_exports/{project_id}/export.cypher", "w") as f:
    f.write(cypher_script)
Importar en Neo4j

bash
# Importar desde archivo Cypher
cat data/graph_exports/project_123/export.cypher | cypher-shell -u neo4j -p password

# Importar desde archivo GraphML
neo4j-admin import --database=project_brain --nodes=data/graph_exports/export.graphml
Plantillas de Exportación

Las plantillas en templates/ definen la estructura de exportación:

export_template.cypher: Estructura Cypher con placeholders
export_template.graphml: Esquema GraphML con tipos de nodo/edge
export_template.json: Esquema JSON para exportaciones nativas
Mantenimiento

Retención: Las exportaciones se mantienen por 90 días
Compresión: Archivos grandes se comprimen automáticamente
Validación: Todas las exportaciones se validan antes de guardar
Metadata: Cada exportación incluye metadata de creación