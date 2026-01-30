# Directorio de Proyectos Analizados

Este directorio almacena todos los proyectos analizados por el sistema Project Brain. Cada proyecto se organiza en subdirectorios con la siguiente estructura:

## Estructura de Proyecto
ESTRUCTURA COMPLETA DE DATA/

text
data/
├── projects/
│   ├── README.md
│   ├── project_template.json
│   └── .gitkeep
├── embeddings/
│   ├── README.md
│   ├── chromadb_config.yaml
│   └── .gitkeep
├── graph_exports/
│   ├── README.md
│   ├── export_template.cypher
│   ├── export_template.graphml
│   └── .gitkeep
├── cache/
│   ├── README.md
│   ├── l3_cache_config.json
│   └── .gitkeep
├── state/
│   ├── README.md
│   ├── system_state.json
│   ├── agents_state_template.json
│   └── .gitkeep
├── backups/
│   ├── README.md
│   ├── backup_manifest.json
│   └── .gitkeep
└── init_data_structure.py
1. ARCHIVO: data/projects/README.md

markdown
# Directorio de Proyectos Analizados

Este directorio almacena todos los proyectos analizados por el sistema Project Brain. Cada proyecto se organiza en subdirectorios con la siguiente estructura:

## Estructura de Proyecto
{project_id}/
├── metadata.json # Metadatos del proyecto
├── analysis_summary.json # Resumen del análisis
├── files/ # Copia de seguridad de archivos analizados
│ ├── {file1}.py
│ ├── {file2}.js
│ └── ...
├── snapshots/ # Snapshots de versiones anteriores
│ ├── snapshot_20240101_120000.json
│ └── ...
├── change_logs/ # Registro de cambios detectados
│ ├── changes_20240101.json
│ └── ...
└── reports/ # Reportes generados
├── quality_report.json
├── security_report.json
└── ...

text

## Formatos de Archivo

### metadata.json
```json
{
  "project_id": "uuid",
  "name": "Nombre del Proyecto",
  "path": "/ruta/original/del/proyecto",
  "language": "python",
  "created_at": "2024-01-01T12:00:00Z",
  "last_analyzed": "2024-01-01T12:00:00Z",
  "analysis_status": "completed",
  "file_count": 150,
  "total_lines": 12000,
  "metadata": {
    "team": "Equipo de Desarrollo",
    "repository_url": "https://github.com/usuario/proyecto",
    "branch": "main"
  }
}
analysis_summary.json

json
{
  "analysis_id": "uuid",
  "project_id": "uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "summary": {
    "files_analyzed": 150,
    "entities_extracted": 1200,
    "issues_found": 45,
    "patterns_detected": 12,
    "analysis_time_seconds": 45.2
  },
  "quality_metrics": {
    "maintainability_index": 85.5,
    "test_coverage": 78.3,
    "avg_complexity": 4.2
  }
}
Políticas de Almacenamiento

Retención: Los proyectos se mantienen indefinidamente
Compresión: Archivos grandes (>1MB) se comprimen automáticamente
Encriptación: Datos sensibles se encriptan en reposo
Backup: Incluido en el sistema de backup automático
Scripts Relacionados

scripts/init_project.py: Inicializa estructura de proyecto
scripts/analyze_project.py: Ejecuta análisis
scripts/export_knowledge.py: Exporta conocimiento a otros formatos
text

## 2. ARCHIVO: `data/projects/project_template.json`

```json
{
  "project_id": "{{PROJECT_ID}}",
  "name": "{{PROJECT_NAME}}",
  "path": "{{PROJECT_PATH}}",
  "description": "{{PROJECT_DESCRIPTION}}",
  "language": "{{MAIN_LANGUAGE}}",
  "created_at": "{{TIMESTAMP}}",
  "updated_at": "{{TIMESTAMP}}",
  "last_analyzed": null,
  "analysis_status": "pending",
  "metadata": {
    "team": "{{TEAM_NAME}}",
    "repository_url": "{{REPO_URL}}",
    "branch": "main",
    "contact_email": "{{EMAIL}}",
    "tags": ["{{TAG1}}", "{{TAG2}}"]
  },
  "stats": {
    "file_count": 0,
    "total_lines": 0,
    "function_count": 0,
    "class_count": 0,
    "issue_count": 0
  },
  "config": {
    "analysis_level": "comprehensive",
    "include_tests": true,
    "include_docs": true,
    "max_file_size_mb": 10,
    "exclude_patterns": ["**/node_modules/**", "**/.git/**"]
  },
  "permissions": {
    "read": ["user:{{USER_ID}}"],
    "write": ["user:{{USER_ID}}"],
    "admin": ["user:{{USER_ID}}"]
  }
}
