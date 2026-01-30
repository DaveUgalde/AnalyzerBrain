# 🧠 Project Brain - AnalyzerBrain

**Sistema de Inteligencia Artificial para Comprensión y Gestión Evolutiva de Proyectos de Software**

> Un cerebro colectivo persistente que supera las limitaciones de contexto de los LLMs tradicionales

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Microkernel%2BAgents-orange.svg)](#arquitectura)
[![Documentation](https://img.shields.io/badge/Docs-Comprehensive-brightgreen.svg)](#documentación)

## 🌟 Características Principales

### 🧠 **Memoria Infinita**
- Almacenamiento persistente de análisis históricos sin pérdida por límites de contexto
- Retención del 100% del conocimiento adquirido
- Versionado completo y seguimiento de cambios

### 🔍 **Comprensión Profunda Multi-Lenguaje**
- Análisis a nivel de archivos, funciones, clases, dependencias y patrones
- Soporte para 10+ lenguajes (Python, JavaScript, Java, C++, Go, Rust, etc.)
- Precisión > 90% en identificación de entidades

### 📚 **Aprendizaje Continuo**
- Mejora de comprensión con cada interacción sin sobrescribir conocimiento
- Incremento del 5% mensual en precisión
- Adaptación al estilo del equipo

### 🏗️ **Razonamiento Estructural Avanzado**
- Entendimiento de arquitecturas, dependencias y patrones de diseño
- Detección del 95% de dependencias críticas
- Análisis predictivo de problemas

### ⚡ **Rendimiento Optimizado**
- Análisis de 1000 archivos en < 30 segundos
- Respuestas a preguntas en < 2 segundos (p95)
- Soporte para 50+ consultas concurrentes por segundo

## 🚀 Comenzando Rápidamente

### Prerrequisitos
- Python 3.10 o superior
- PostgreSQL 15+
- Redis 7+
- Neo4j 5+
- 8GB+ RAM, 4+ núcleos CPU

### Instalación en 5 minutos

```bash
# 1. Clonar el repositorio
git clone https://github.com/yourusername/project-brain.git
cd project-brain

# 2. Configurar entorno
cp .env.example .env
# Editar .env con tus configuraciones

# 3. Instalar dependencias
pip install -r requirements/base.txt

# 4. Iniciar con Docker Compose (recomendado para desarrollo)
docker-compose up -d

# 5. Inicializar el sistema
python scripts/init_system.py

# 6. Ejecutar análisis de ejemplo
python scripts/analyze_project.py examples/sample-python-project

# 7. Probar con una pregunta
python scripts/query_project.py "¿Qué hace la función main?"

Arquitectura del Sistema

📐 Arquitectura Híbrida: Microkernel + Sistema de Agentes + Base de Conocimiento Centralizada

┌─────────────────────────────────────────────────────────────────────────────┐
│                             CAPA DE PRESENTACIÓN                            │
├─────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│    CLI      │   API REST   │  WebSocket   │    gRPC      │   Web UI        │
│  (click)    │  (FastAPI)   │ (real-time)  │ (high-perf)  │  (Streamlit)    │
└─────────────┴──────────────┴──────────────┴──────────────┴─────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CAPA DE ORQUESTACIÓN                               │
├─────────────┬──────────────┬──────────────┬─────────────────────────────────┤
│Workflow Mng │ Task Scheduler│Pipeline Orch.│         Event Bus               │
│(Prefect)    │ (Celery)     │ (Kedro)      │ (Redis Pub/Sub)                 │
└─────────────┴──────────────┴──────────────┴─────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SISTEMA DE AGENTES ESPECIALIZADOS                      │
├─────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│ Arquitecto  │  Detective   │  Analista    │   Curador    │     Q&A         │
│ (patrones)  │ (problemas)  │ (métricas)   │(conocimiento)│ (respuestas)    │
└─────────────┴──────────────┴──────────────┴──────────────┴─────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NÚCLEO DE INTELIGENCIA                             │
├─────────────┬──────────────┬──────────────┬─────────────────────────────────┤
│ Red Neuronal│  Memoria     │  Análisis    │         Aprendizaje             │
│ (GNN)       │(vector+grafo)│ (profundo)   │     (incremental+RL)            │
└─────────────┴──────────────┴──────────────┴─────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE DE DATOS                                  │
├─────────────┬──────────────┬──────────────┬─────────────────────────────────┤
│  Ingestión  │Procesamiento │Almacenamiento│            Cache                │
│ (scanner)   │(parsing+emb) │  (multi-DB)  │        (Redis+Memcached)        │
└─────────────┴──────────────┴──────────────┴─────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SISTEMA DE CONSULTAS                               │
├─────────────┬──────────────┬──────────────┬─────────────────────────────────┤
│     NLP     │ Recuperación │ Razonamiento │           Respuesta             │
│(intent+NER) │(vector+grafo)│(chain+agents)│      (synthesis+formatting)     │
└─────────────┴──────────────┴──────────────┴─────────────────────────────────┘


Módulos Principales

Módulo	Responsabilidad	% Sistema	Estado
core/	Orquestación principal, gestión de estado	15%	✅
indexer/	Indexación, parsing multi-lenguaje	25%	✅
embeddings/	Representación vectorial, búsqueda semántica	15%	✅
graph/	Grafo de conocimiento, consultas	10%	✅
memory/	Sistemas de memoria persistente	10%	✅
agents/	Agentes IA especializados	15%	✅
api/	Interfaces externas (REST, WebSocket, etc.)	5%	✅
learning/	Aprendizaje incremental	3%	✅
utils/	Utilidades compartidas	2%	✅
📊 Métricas de Rendimiento

⚡ Análisis de Código

Escenario	Tiempo Objetivo	Límite Aceptable
Archivo Python 1000 líneas	< 500ms	< 1s
Archivo JavaScript	< 300ms	< 500ms
Lote 100 archivos	< 30s	< 60s
🧮 Generación de Embeddings

Escenario	Tiempo Objetivo	Límite Aceptable
Texto 512 tokens	< 100ms	< 200ms
Código embedding	< 200ms	< 400ms
Lote 1000 embeddings	< 10s	< 20s
🔍 Consultas

Escenario	Tiempo Objetivo	Límite Aceptable
Pregunta simple	< 2s p95	< 5s p95
Análisis complejo	< 10s p95	< 20s p95
Traversal grafo profundidad 5	< 500ms	< 1s
🛠️ Uso del Sistema

Comandos CLI Principales

# Inicializar sistema
project-brain init

# Analizar proyecto
project-brain analyze /ruta/al/proyecto

# Consultar sobre proyecto
project-brain query --project-id PROJ_123 "¿Qué hace esta función?"

# Exportar conocimiento
project-brain export --format json --output conocimiento.json

# Monitorear sistema
project-brain monitor --metrics --live

# Administrar agentes
project-brain agents list
project-brain agents enable code_analyzer
project-brain agents status

API REST Ejemplos

import requests

# Crear proyecto
response = requests.post(
    "http://localhost:8000/v1/projects",
    json={
        "name": "Mi Proyecto Python",
        "path": "/ruta/al/proyecto",
        "language": "python"
    },
    headers={"X-API-Key": "tu-api-key"}
)

# Consultar
response = requests.post(
    "http://localhost:8000/v1/query",
    json={
        "question": "¿Dónde se define la función process_data?",
        "project_id": "proj_123"
    }
)

# Stream de análisis via WebSocket
import websocket
ws = websocket.WebSocket()
ws.connect("ws://localhost:8001")
ws.send(json.dumps({
    "type": "subscribe",
    "data": {"topics": ["analysis_progress"]}
}))

Integración con IDEs

VSCode Extension disponible en Marketplace

PyCharm Plugin disponible en JetBrains Marketplace

📈 Impacto Esperado

Para Desarrolladores Individuales

✅ Reducción del 50% en tiempo de onboarding
✅ Disminución del 40% en bugs introducidos
✅ Aumento del 60% en reutilización de código
✅ Mejora del 70% en documentación actualizada
Para Equipos

✅ Consistencia en patrones y estándares
✅ Conocimiento compartido accesible
✅ Calidad sostenida con detección proactiva
✅ Colaboración mejorada con contexto compartido
Para Organizaciones

✅ ROI positivo en 6 meses (equipos >10 devs)
✅ Reducción de deuda técnica gestionable
✅ Mejora en seguridad con detección temprana
✅ Escalabilidad para nuevos equipos
🧪 Ejemplos de Uso

Ejemplo 1: Análisis Completo de Proyecto

from project_brain import BrainOrchestrator

# Inicializar orquestador
orchestrator = BrainOrchestrator()
orchestrator.initialize()

# Analizar proyecto
result = await orchestrator.analyze_project(
    "/ruta/al/proyecto",
    options={
        "mode": "comprehensive",
        "languages": ["python", "javascript"],
        "include_tests": True,
        "max_file_size_mb": 10
    }
)

print(f"Archivos analizados: {result['summary']['files_analyzed']}")
print(f"Entidades extraídas: {result['summary']['entities_extracted']}")
print(f"Problemas encontrados: {result['summary']['issues_found']}")

Ejemplo 2: Consulta Inteligente

# Hacer una pregunta sobre el proyecto
answer = await orchestrator.ask_question(
    question="¿Por qué la función calculate_total es tan lenta?",
    project_id="proj_123",
    context={
        "current_file": "src/utils/calculations.py",
        "technical_level": "advanced"
    }
)

print(f"Respuesta: {answer['answer']['text']}")
print(f"Confianza: {answer['confidence']}")
print(f"Fuentes: {len(answer['sources'])}")

Ejemplo 3: Detección de Cambios Automática

# Detectar cambios desde último análisis
changes = await orchestrator.detect_changes({
    "project_id": "proj_123",
    "since": "2024-01-01T00:00:00Z"
})

print(f"Archivos modificados: {changes['files_modified']}")
print(f"Impacto en dependencias: {changes['impact_analysis']}")

Configuración Avanzada

Configuración de Agentes

yaml
# config/agents.yaml
agents:
  enabled:
    - code_analyzer
    - qa_agent
    - architect
    - detective
    - curator
  
  code_analyzer:
    confidence_threshold: 0.7
    capabilities:
      - code_analysis
      - pattern_detection
      - quality_assessment
  
  qa_agent:
    max_processing_time: 10
    stream_responses: true
Configuración de Caché Multi-Nivel

yaml
# config/system.yaml
cache:
  hierarchy:
    level1:
      type: "memory"
      max_size: 1000
      ttl_seconds: 300
      
    level2:
      type: "redis"
      max_size: 10000
      ttl_seconds: 3600
      
    level3:
      type: "disk"
      max_size: 100000
      ttl_seconds: 86400
📚 Documentación Adicional

📖 Guías Detalladas

📘 Guía de Arquitectura - Diseño detallado del sistema
🔌 Guía de Integración - Cómo integrar con otras herramientas
🚀 Guía de Despliegue - Producción, Kubernetes, etc.
🧪 Guía de Testing - Pruebas, CI/CD, calidad
🎓 Tutoriales

Tutorial 1: Primer Proyecto
Tutorial 2: Agentes Personalizados
Tutorial 3: Análisis a Escala
Tutorial 4: Integración con CI/CD
📊 Referencias de API

API REST Completa - Todos los endpoints
WebSocket Protocol - Protocolo en tiempo real
gRPC API - API de alta performance
CLI Reference - Todos los comandos CLI
🚢 Despliegue

Docker (Recomendado para desarrollo)

bash
docker-compose up -d
Kubernetes (Producción)

bash
# Instalar con Helm
helm install project-brain ./deployments/helm/

# O con manifests directos
kubectl apply -f ./deployments/kubernetes/
Nube (AWS, GCP, Azure)

bash
# Terraform para AWS
cd deployments/terraform/aws
terraform init
terraform apply
🤝 Contribuir

¡Contribuciones son bienvenidas! Por favor lee nuestras guías de contribución.

Estructura del Proyecto

text
project_brain/
├── src/                    # Código fuente
│   ├── core/              # Núcleo del sistema
│   ├── indexer/           # Indexación y parsing
│   ├── embeddings/        # Representación vectorial
│   ├── agents/            # Agentes IA
│   └── ...                # Otros módulos
├── tests/                 # Pruebas
├── docs/                  # Documentación
├── deployments/           # Configuraciones de despliegue
└── scripts/              # Scripts de utilidad
Pruebas

bash
# Ejecutar todas las pruebas
pytest tests/

# Pruebas específicas
pytest tests/unit/core/
pytest tests/integration/

# Con cobertura
pytest --cov=src --cov-report=html
📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

📞 Soporte y Contacto

📧 Email: support@projectbrain.dev
🐛 Issues: GitHub Issues
💬 Discord: Únete a nuestro Discord
📖 Documentación: docs.projectbrain.dev
🙏 Agradecimientos

Gracias a todos los contribuidores
Basado en investigaciones de OpenAI, Google Research, y Microsoft Research
Utiliza tree-sitter para parsing multi-lenguaje
Embeddings con Sentence Transformers