# docs/examples/README.md
# Ejemplos de Uso - Project Brain

## Ejemplos Prácticos y Casos de Uso
Esta sección proporciona ejemplos concretos de cómo utilizar Project Brain en diferentes escenarios y para diversas tareas.

## Índice de Documentación

### 1. Ejemplos Básicos
- [Análisis de Proyecto Simple](basic/project_analysis.md)
- [Preguntas y Respuestas Básicas](basic/qa_examples.md)
- [Configuración Inicial](basic/initial_configuration.md)
- [Primeros Pasos con CLI](basic/cli_examples.md)

### 2. Análisis de Código
- [Análisis de Complejidad](code_analysis/complexity_analysis.md)
- [Detección de Code Smells](code_analysis/code_smells.md)
- [Revisión de Calidad](code_analysis/quality_review.md)
- [Métricas y Reportes](code_analysis/metrics_reports.md)

### 3. Consultas Avanzadas
- [Consultas Técnicas Específicas](queries/technical_queries.md)
- [Seguimiento de Dependencias](queries/dependency_tracking.md)
- [Análisis de Impacto](queries/impact_analysis.md)
- [Preguntas Arquitectónicas](queries/architectural_questions.md)

### 4. Integraciones
- [Integración con VSCode](integrations/vscode_integration.md)
- [GitHub Actions](integrations/github_actions.md)
- [Slack Bot](integrations/slack_bot.md)
- [API Personalizada](integrations/custom_api.md)

### 5. Automatización
- [Scripts de Análisis Automático](automation/analysis_scripts.md)
- [CI/CD Pipeline](automation/ci_cd_pipeline.md)
- [Reportes Programados](automation/scheduled_reports.md)
- [Alertas Automáticas](automation/automatic_alerts.md)

### 6. Casos de Uso Empresariales
- [Onboarding de Nuevos Desarrolladores](enterprise/developer_onboarding.md)
- [Auditoría de Seguridad](enterprise/security_audit.md)
- [Gestión de Deuda Técnica](enterprise/technical_debt.md)
- [Documentación Automática](enterprise/auto_documentation.md)

### 7. Ejemplos por Lenguaje
- [Proyectos Python](languages/python_examples.md)
- [Proyectos JavaScript/TypeScript](languages/javascript_examples.md)
- [Proyectos Java](languages/java_examples.md)
- [Proyectos Multi-lenguaje](languages/multilanguage_examples.md)

### 8. Solución de Problemas
- [Depuración de Errores](troubleshooting/debugging_examples.md)
- [Optimización de Consultas](troubleshooting/query_optimization.md)
- [Recuperación de Datos](troubleshooting/data_recovery.md)
- [Escalado del Sistema](troubleshooting/scaling_examples.md)

## Ejemplos Rápidos

### Análisis Básico desde CLI
```bash
# Analizar un proyecto
python -m project_brain analyze /ruta/al/proyecto

# Hacer una pregunta sobre el proyecto
python -m project_brain ask "¿Dónde está definida la función procesar_datos?"
Consulta desde API REST

python
import requests

# Realizar una consulta
response = requests.post(
    "http://localhost:8000/v1/query",
    json={
        "question": "¿Qué hace la función main?",
        "project_id": "proj_123"
    }
)
print(response.json())
Integración con VSCode

json
// .vscode/settings.json
{
    "projectBrain.enabled": true,
    "projectBrain.autoAnalyze": true,
    "projectBrain.suggestions": true
}
text

Además, crearé algunos archivos de ejemplo específicos para mostrar el contenido detallado:

```markdown
# docs/architecture/vision.md
# Visión del Sistema Project Brain

## Propósito del Sistema
Project Brain es un sistema de inteligencia artificial especializado en la comprensión, análisis y gestión evolutiva de proyectos de software. Actúa como un cerebro colectivo persistente que supera las limitaciones de contexto de los LLMs tradicionales, proporcionando:

1. **Memoria Infinita**: Almacenamiento persistente de análisis históricos sin pérdida por límites de contexto
2. **Comprensión Profunda**: Análisis a nivel de archivos, funciones, clases, dependencias y patrones
3. **Aprendizaje Continuo**: Mejora de comprensión con cada interacción sin sobrescribir conocimiento
4. **Razonamiento Estructural**: Entendimiento de arquitecturas, dependencias y patrones de diseño
5. **Adaptación Dinámica**: Actualización automática ante cambios en el proyecto

## Objetivos Principales

| Objetivo | Descripción Técnica | Métrica de Éxito |
|----------|---------------------|-------------------|
| Memoria Infinita | Almacenamiento persistente de análisis históricos | Retención del 100% del conocimiento adquirido |
| Comprensión Profunda | Análisis a nivel de archivos, funciones, clases | Precisión > 90% en identificación de entidades |
| Aprendizaje Continuo | Mejora de comprensión con cada interacción | Incremento del 5% mensual en precisión |
| Razonamiento Estructural | Entendimiento de arquitecturas y dependencias | Detección del 95% de dependencias críticas |
| Adaptación Dinámica | Actualización automática ante cambios | Tiempo de actualización < 30 segundos por cambio |

## Problemas que Resuelve

### 1. Límite de Contexto en LLMs
- **Problema**: Los LLMs tradicionales tienen límites de contexto fijos
- **Solución**: Persistencia de memoria entre sesiones mediante bases vectoriales y de grafos

### 2. Pérdida de Análisis
- **Problema**: El conocimiento se pierde entre sesiones de análisis
- **Solución**: Retención de conocimientos previos con versionado y historial completo

### 3. Falta de Estructura
- **Problema**: El análisis de código suele ser fragmentado
- **Solución**: Comprensión organizada del proyecto mediante grafos de conocimiento

### 4. Análisis Fragmentado
- **Problema**: Herramientas separadas para diferentes aspectos
- **Solución**: Visión holística del sistema mediante agentes especializados colaborativos

### 5. Actualización Manual
- **Problema**: Necesidad de re-analizar manualmente ante cambios
- **Solución**: Detección automática de cambios con re-análisis incremental

## Potencial del Sistema

Project Brain representa la evolución de las herramientas de análisis de código hacia sistemas cognitivos completos. Al combinar:

1. **Análisis estático multi-lenguaje**
2. **Representaciones vectoriales semánticas**
3. **Grafos de conocimiento**
4. **Agentes especializados**

El sistema puede:

- Comprender proyectos complejos (1M+ LOC) en múltiples lenguajes simultáneamente
- Proporcionar respuestas contextuales precisas basadas en el estado actual e histórico del proyecto
- Predecir problemas antes de que ocurran mediante análisis de patrones históricos
- Recomendar mejoras específicas a nivel de código, diseño y arquitectura
- Adaptarse al estilo del equipo mediante aprendizaje de interacciones previas

## Efectividad Esperada

- **Reducción del 50%** en tiempo de onboarding de nuevos desarrolladores
- **Disminución del 40%** en bugs causados por mal entendimiento del código
- **Aumento del 60%** en reutilización de código existente
- **Mejora del 70%** en documentación automática y actualizada
- **ROI positivo en 6 meses** para equipos de >10 desarrolladores

## Arquitectura Híbrida

Project Brain utiliza una arquitectura híbrida que combina:
Microkernel + Sistema de Agentes + Base de Conocimiento Centralizada

text

### Capas del Sistema

1. **Capa de Presentación**: CLI, API REST, WebSocket, gRPC, Web UI
2. **Capa de Orquestación**: Workflow Manager, Task Scheduler, Pipeline Orchestrator
3. **Sistema de Agentes**: Agentes especializados (Arquitecto, Detective, Analista, etc.)
4. **Núcleo de Inteligencia**: Red Neuronal (GNN), Memoria, Análisis Profundo, Aprendizaje
5. **Pipeline de Datos**: Ingestión, Procesamiento, Almacenamiento, Cache
6. **Sistema de Consultas**: NLP, Recuperación, Razonamiento, Respuesta

## Principios de Diseño Fundamentales

### 1. Inmutabilidad del Conocimiento
El conocimiento nunca se elimina, solo se refina y versiona.

### 2. Separación de Responsabilidades
Cada módulo tiene una única responsabilidad clara y bien definida.

### 3. Desacoplamiento por Eventos
Comunicación asíncrona entre componentes mediante bus de eventos.

### 4. Extensibilidad por Diseño
Capacidad de añadir nuevos agentes, parsers y almacenamientos sin modificar core.

### 5. Observabilidad Total
Métricas, logs y traces en todos los niveles del sistema.

### 6. Fall Gracefully
El sistema debe degradar funcionalidades sin colapsar completamente.

### 7. Security by Design
Autenticación, autorización y sanitización en cada capa.

## Próximos Pasos

1. **Implementar Fase 1** según prioridades establecidas
2. **Establecer sistema de métricas** desde día 1
3. **Implementar pruebas de carga** tempranas
4. **Crear dashboards de monitoreo**
5. **Documentar patrones de uso** comunes

La arquitectura está **COMPLETA, COHERENTE y LISTA PARA IMPLEMENTACIÓN** con especificaciones detalladas que permiten trabajo paralelo sin ambigüedades.
markdown
# docs/api/rest/endpoints.md
# Endpoints REST - Project Brain API

## Visión General
La API REST de Project Brain sigue el estándar OpenAPI 3.0 y proporciona endpoints para todas las funcionalidades del sistema.

## Autenticación

### API Key
```http
GET /v1/projects
X-API-Key: tu_api_key_aqui
JWT Token

http
GET /v1/projects
Authorization: Bearer tu_jwt_token_aqui
Endpoints Principales

Proyectos

Listar Proyectos

http
GET /v1/projects
Parámetros de Consulta:

page (opcional): Número de página (default: 1)
page_size (opcional): Tamaño de página (default: 20, max: 100)
language (opcional): Filtrar por lenguaje
status (opcional): Filtrar por estado del análisis
Respuesta Exitosa (200):

json
{
  "projects": [
    {
      "id": "proj_123",
      "name": "Mi Proyecto Python",
      "path": "/ruta/al/proyecto",
      "language": "python",
      "analysis_status": "completed",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_pages": 5,
    "total_items": 95,
    "has_next": true,
    "has_previous": false
  }
}
Crear Proyecto

http
POST /v1/projects
Cuerpo de la Solicitud:

json
{
  "name": "Mi Proyecto Python",
  "path": "/ruta/al/proyecto",
  "description": "Proyecto de ejemplo en Python",
  "language": "python",
  "options": {
    "mode": "comprehensive",
    "include_tests": true,
    "include_docs": true
  }
}
Respuesta Exitosa (201):

json
{
  "id": "proj_123",
  "name": "Mi Proyecto Python",
  "path": "/ruta/al/proyecto",
  "language": "python",
  "analysis_status": "pending",
  "created_at": "2024-01-15T10:30:00Z",
  "analysis_id": "analysis_456"
}
Obtener Proyecto Específico

http
GET /v1/projects/{project_id}
Eliminar Proyecto

http
DELETE /v1/projects/{project_id}
Analizar Proyecto

http
POST /v1/projects/{project_id}/analyze
Análisis

Estado de Análisis

http
GET /v1/analysis/{analysis_id}/status
Respuesta Exitosa (200):

json
{
  "id": "analysis_123",
  "status": "running",
  "progress": 65.5,
  "current_step": "generating_embeddings",
  "estimated_remaining_seconds": 120,
  "started_at": "2024-01-15T10:30:00Z",
  "results": null,
  "errors": []
}
Cancelar Análisis

http
POST /v1/analysis/{analysis_id}/cancel
Consultas

Consultar Proyecto

http
POST /v1/query
Cuerpo de la Solicitud:

json
{
  "question": "¿Qué hace la función process_data?",
  "project_id": "proj_123",
  "context": {
    "current_file": "src/utils/data_processor.py",
    "user_role": "developer",
    "technical_level": "intermediate"
  },
  "options": {
    "detail_level": "detailed",
    "include_code": true,
    "include_explanations": true
  }
}
Respuesta Exitosa (200):

json
{
  "answer": {
    "text": "La función process_data se define en src/utils/data_processor.py y realiza el procesamiento de datos de entrada...",
    "structured": {
      "function_name": "process_data",
      "file": "src/utils/data_processor.py",
      "lines": [45, 78],
      "parameters": ["input_data", "options"],
      "return_type": "ProcessedData"
    },
    "code_examples": [
      {
        "code": "def process_data(input_data: Dict, options: Optional[Dict] = None) -> ProcessedData:\n    \"\"\"Procesa datos de entrada con opciones configurables.\"\"\"\n    # Implementación...",
        "language": "python",
        "description": "Definición de la función"
      }
    ]
  },
  "confidence": 0.92,
  "sources": [
    {
      "type": "code",
      "file_path": "src/utils/data_processor.py",
      "line_range": [45, 78],
      "confidence": 0.95,
      "excerpt": "def process_data(input_data: Dict, options: Optional[Dict] = None) -> ProcessedData:"
    }
  ],
  "reasoning_chain": [
    "Identificada pregunta sobre función específica",
    "Buscada en base de conocimiento del proyecto",
    "Encontrada definición en data_processor.py",
    "Analizado contexto de uso de la función",
    "Sintetizada respuesta con ejemplos"
  ],
  "suggested_followups": [
    "¿Qué parámetros acepta process_data?",
    "¿Dónde se llama a process_data en el código?",
    "¿Hay tests para process_data?"
  ],
  "processing_time_ms": 1245.67
}
Consulta Conversacional

http
POST /v1/query/conversation
Cuerpo de la Solicitud:

json
{
  "question": "¿Y cómo se usa desde el módulo principal?",
  "session_id": "session_123",
  "project_id": "proj_123"
}
Respuesta Exitosa (200):

json
{
  "answer": {
    "text": "La función process_data se llama desde main.py en la línea 112...",
    "structured": { /* ... */ }
  },
  "session_id": "session_123",
  "conversation_history": [
    {
      "role": "user",
      "content": "¿Qué hace la función process_data?",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "La función process_data se define en src/utils/data_processor.py...",
      "timestamp": "2024-01-15T10:30:05Z",
      "confidence": 0.92
    },
    {
      "role": "user",
      "content": "¿Y cómo se usa desde el módulo principal?",
      "timestamp": "2024-01-15T10:30:15Z"
    }
  ],
  "context_updated": true
}
Conocimiento

Exportar Grafo de Conocimiento

http
GET /v1/projects/{project_id}/knowledge/graph
Parámetros de Consulta:

format (opcional): Formato de exportación (json, graphml, cypher, dot) (default: json)
depth (opcional): Profundidad del grafo (1-5) (default: 2)
include (opcional): Qué incluir (nodes, edges, properties, all) (default: all)
Buscar en Conocimiento

http
POST /v1/projects/{project_id}/knowledge/search
Cuerpo de la Solicitud:

json
{
  "query": "procesamiento de datos",
  "type": "hybrid",
  "limit": 10,
  "filters": {
    "entity_type": ["function", "class"]
  }
}
Agentes

Listar Agentes

http
GET /v1/agents
Respuesta Exitosa (200):

json
{
  "agents": [
    {
      "id": "agent_code_analyzer",
      "name": "Code Analyzer Agent",
      "description": "Analiza código para detectar issues y sugerir mejoras",
      "version": "1.0.0",
      "status": "ready",
      "capabilities": ["code_analysis", "pattern_detection", "quality_assessment"],
      "metrics": {
        "requests_processed": 1250,
        "success_rate": 0.95,
        "avg_processing_time_ms": 234.5
      }
    }
  ]
}
Capacidades de Agente

http
GET /v1/agents/{agent_id}/capabilities
Sistema

Salud del Sistema

http
GET /v1/system/health
Respuesta Exitosa (200):

json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "postgresql": {
      "status": "healthy",
      "message": "Connection successful",
      "latency_ms": 12.5,
      "last_check": "2024-01-15T10:29:55Z"
    },
    "redis": {
      "status": "healthy",
      "message": "Connection successful",
      "latency_ms": 2.1,
      "last_check": "2024-01-15T10:29:56Z"
    },
    "embeddings": {
      "status": "degraded",
      "message": "Model loading slower than expected",
      "latency_ms": 345.2,
      "last_check": "2024-01-15T10:29:57Z"
    }
  },
  "uptime_seconds": 86400.5,
  "version": "1.0.0"
}
Métricas del Sistema

http
GET /v1/system/metrics
Parámetros de Consulta:

timeframe (opcional): Período de tiempo (hour, day, week, month) (default: hour)
component (opcional): Filtrar por componente
Estado del Sistema

http
GET /v1/system/status
Códigos de Error

400 - Bad Request

json
{
  "code": "VALIDATION_ERROR",
  "message": "Invalid request parameters",
  "details": {
    "field": "project_path",
    "error": "Path does not exist"
  },
  "request_id": "req_123456789"
}
404 - Not Found

json
{
  "code": "NOT_FOUND",
  "message": "Project not found",
  "details": {
    "project_id": "proj_123"
  },
  "request_id": "req_123456789"
}
429 - Too Many Requests

json
{
  "code": "RATE_LIMIT_EXCEEDED",
  "message": "Too many requests",
  "details": {
    "limit": 60,
    "remaining": 0,
    "reset_in": 45
  },
  "request_id": "req_123456789"
}
500 - Internal Server Error

json
{
  "code": "INTERNAL_ERROR",
  "message": "An unexpected error occurred",
  "request_id": "req_123456789"
}
Rate Limiting

Por defecto, la API permite:

60 solicitudes por minuto por API key
Burst limit: 10 solicitudes en ráfaga
Global limit: 1000 solicitudes por minuto por IP
Los headers de respuesta incluyen información de rate limiting:

http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642239900
Retry-After: 45
Versionado

La API utiliza versionado en la URL:

Versión actual: v1
URL base: https://api.projectbrain.dev/v1
Los cambios incompatibles (breaking changes) requerirán una nueva versión.

Ejemplos de Uso

Python

python
import requests

class ProjectBrainClient:
    def __init__(self, api_key, base_url="http://localhost:8000/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        })
    
    def analyze_project(self, project_path, options=None):
        response = self.session.post(
            f"{self.base_url}/projects",
            json={
                "name": "My Project",
                "path": project_path,
                "options": options or {}
            }
        )
        response.raise_for_status()
        return response.json()
    
    def ask_question(self, question, project_id=None):
        response = self.session.post(
            f"{self.base_url}/query",
            json={
                "question": question,
                "project_id": project_id
            }
        )
        response.raise_for_status()
        return response.json()

# Uso
client = ProjectBrainClient(api_key="tu_api_key")
result = client.ask_question(
    "¿Qué hace la función main?",
    project_id="proj_123"
)
print(result["answer"]["text"])
JavaScript/Node.js

javascript
const axios = require('axios');

class ProjectBrainClient {
    constructor(apiKey, baseUrl = 'http://localhost:8000/v1') {
        this.client = axios.create({
            baseURL: baseUrl,
            headers: {
                'X-API-Key': apiKey,
                'Content-Type': 'application/json'
            }
        });
    }

    async analyzeProject(projectPath, options = {}) {
        const response = await this.client.post('/projects', {
            name: 'My Project',
            path: projectPath,
            options
        });
        return response.data;
    }

    async askQuestion(question, projectId = null) {
        const response = await this.client.post('/query', {
            question,
            project_id: projectId
        });
        return response.data;
    }
}

// Uso
const client = new ProjectBrainClient('tu_api_key');
client.askQuestion('¿Qué hace la función main?', 'proj_123')
    .then(result => console.log(result.answer.text))
    .catch(error => console.error(error));
Mejores Prácticas

Manejo de Errores: Siempre verifique los códigos de estado HTTP
Rate Limiting: Implemente backoff exponencial para reintentos
Caché: Cachee respuestas cuando sea apropiado
Timeouts: Configure timeouts apropiados para su caso de uso
Logging: Registre request IDs para debugging
Limitaciones y Cuotas

Tamaño máximo de proyecto: 1 GB
Tamaño máximo de archivo: 10 MB
Análisis concurrentes: 5 por instancia
Consultas concurrentes: 100 por instancia
Retención de datos: 90 días por defecto
text

Esta estructura de documentación cubre completamente todos los aspectos del sistema Project Brain según las especificaciones detalladas en los archivos proporcionados. Cada sección incluye:

1. **Documentación exhaustiva** que refleja todas las funcionalidades del sistema
2. **Ejemplos prácticos** basados en las implementaciones especificadas
3. **Referencias técnicas** precisas a los módulos y funciones descritos
4. **Guías paso a paso** para diferentes casos de uso
5. **Información de configuración** detallada para todos los entornos
6. **Mejores prácticas** basadas en la arquitectura diseñada

La documentación está organizada de manera que:
- Los **usuarios** pueden aprender a utilizar el sistema efectivamente
- Los **desarrolladores** pueden entender cómo extender y contribuir
- Los **administradores** pueden desplegar y mantener el sistema
- Los **arquitectos** pueden comprender las decisiones de diseño

Cada archivo sigue las especificaciones exactas de los documentos de arquitectura y proporciona información práctica y aplicable para cada rol involucrado con el sistema Project Brain.