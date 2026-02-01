Distribución de archivos:

ANALYZERBRAIN/
├── 📁 .github/                  # CI/CD y automatización
├── 📁 .vscode/                  # Configuración de IDE
├── 📁 architecture/             # Documentación arquitectónica
├── 📁 config/                  # Configuraciones YAML del sistema
├── 📁 data/                    # Datos persistentes (SOLO datos, NO código)
├── 📁 deployments/             # Configuraciones de despliegue
├── 📁 docs/                    # Documentación completa
├── 📁 logs/                    # Logs del sistema
├── 📁 monitoring/              # Monitoreo y métricas
├── 📁 requirements/            # Dependencias categorizadas
├── 📁 scripts/                 # Scripts de utilidad
├── 📁 src/                     # CÓDIGO FUENTE PRINCIPAL
├── 📁 tests/                   # Pruebas y fixtures
├── 📁 venv/                    # Entorno virtual
├── 📄 .env                     # Variables de entorno
├── 📄 .env.example             # Plantilla variables de entorno
├── 📄 .gitignore               # Archivos ignorados por git
├── 📄 Dockerfile               # Imagen Docker
├── 📄 LICENSE                  # Licencia MIT
├── 📄 pyproject.toml          # Configuración de paquete Python moderno
└── 📄 README.md               # Documentación principal
🔧 SRC/ - ESTRUCTURA DETALLADA DEL CÓDIGO FUENTE

text
src/
├── __init__.py                 # Paquete raíz
├── main.py                     # Punto de entrada principal
│
├── 📁 api/                     # CAPA DE PRESENTACIÓN
│   ├── __init__.py
│   ├── authentication.py       # Autenticación JWT/API Key
│   ├── cli_interface.py       # Interfaz línea de comandos
│   ├── grpc_api.py            # API gRPC (alta performance)
│   ├── rate_limiter.py        # Limitación de tasa
│   ├── request_validator.py   # Validación de peticiones
│   ├── rest_api.py            # Endpoints REST
│   ├── server.py              # Servidor principal FastAPI
│   ├── web_ui.py              # Interfaz web (Streamlit)
│   └── websocket_api.py       # WebSockets (tiempo real)
│
├── 📁 agents/                  # SISTEMA DE AGENTES
│   ├── __init__.py
│   ├── agent_factory.py       # Fábrica de agentes
│   ├── agent_orchestrator.py  # Orquestación de agentes
│   ├── analyst_agent.py       # Análisis de métricas
│   ├── architect_agent.py     # Análisis arquitectónico
│   ├── base_agent.py          # Clase base abstracta
│   ├── code_analyzer_agent.py # Análisis de código
│   ├── collaboration_protocol.py # Protocolo colaborativo
│   ├── curator_agent.py       # Curación de conocimiento
│   ├── detective_agent.py     # Investigación de problemas
│   ├── learning_agent.py      # Agente de aprendizaje
│   ├── qa_agent.py           # Preguntas y respuestas
│   └── security_agent.py     # Análisis de seguridad
│
├── 📁 core/                   # NÚCLEO DEL SISTEMA
│   ├── __init__.py
│   ├── config_manager.py     # Gestión de configuración
│   ├── dependency_injector.py # Inyección de dependencias
│   ├── event_bus.py          # Bus de eventos
│   ├── exceptions.py         # Excepciones personalizadas
│   ├── health_check.py       # Verificación de salud
│   ├── orchestrator.py       # BrainOrchestrator principal
│   ├── plugin_manager.py     # Gestión de plugins
│   ├── system_state.py       # Gestión de estado del sistema
│   └── workflow_manager.py   # Orquestación de flujos
│
├── 📁 embeddings/            # REPRESENTACIÓN VECTORIAL
│   ├── __init__.py
│   ├── dimensionality_reducer.py # Reducción dimensional
│   ├── embedding_cache.py    # Caché de embeddings
│   ├── embedding_generator.py # Generación de embeddings
│   ├── embedding_models.py   # Modelos de embeddings
│   ├── semantic_search.py    # Búsqueda semántica
│   ├── similarity_calculator.py # Cálculo de similitudes
│   └── vector_store.py       # Almacenamiento vectorial
│
├── 📁 graph/                 # GRAFO DE CONOCIMIENTO
│   ├── __init__.py
│   ├── consistency_checker.py # Verificación de consistencia
│   ├── graph_analytics.py    # Análisis de grafos
│   ├── graph_builder.py      # Construcción de grafos
│   ├── graph_exporter.py     # Exportación de grafos
│   ├── graph_query_engine.py # Motor de consultas
│   ├── graph_traverser.py    # Navegación de grafos
│   ├── knowledge_graph.py    # Grafo de conocimiento principal
│   └── schema_manager.py     # Gestión de esquemas
│
├── 📁 indexer/               # INDEXACIÓN Y PARSING
│   ├── __init__.py
│   ├── change_detector.py    # Detección de cambios
│   ├── dependency_mapper.py  # Mapeo de dependencias
│   ├── entity_extractor.py   # Extracción de entidades
│   ├── file_processor.py     # Procesamiento de archivos
│   ├── multi_language_parser.py # Parser multi-lenguaje
│   ├── pattern_detector.py   # Detección de patrones
│   ├── project_scanner.py    # Escaneo de proyectos
│   ├── quality_analyzer.py   # Análisis de calidad
│   └── version_tracker.py    # Seguimiento de versiones
│
├── 📁 learning/              # APRENDIZAJE AUTOMÁTICO
│   ├── __init__.py
│   ├── adaptation_engine.py  # Adaptación a nuevos dominios
│   ├── feedback_loop.py      # Bucle de retroalimentación
│   ├── forgetting_mechanism.py # Mecanismo de olvido
│   ├── incremental_learner.py # Aprendizaje incremental
│   ├── knowledge_refiner.py  # Refinamiento de conocimiento
│   ├── learning_evaluator.py # Evaluación de aprendizaje
│   └── reinforcement_learner.py # Aprendizaje por refuerzo
│
├── 📁 memory/               # SISTEMA DE MEMORIA
│   ├── __init__.py
│   ├── cache_manager.py     # Gestión de caché
│   ├── episodic_memory.py   # Memoria episódica
│   ├── memory_cleaner.py    # Limpieza de memoria
│   ├── memory_consolidator.py # Consolidación de memoria
│   ├── memory_hierarchy.py  # Jerarquía de memoria
│   ├── memory_retriever.py  # Recuperación de memoria
│   ├── semantic_memory.py   # Memoria semántica
│   └── working_memory.py    # Memoria de trabajo
│
└── 📁 utils/                # UTILIDADES COMPARTIDAS
    ├── __init__.py
    ├── file_utils.py        # Operaciones de archivos
    ├── logging_config.py    # Configuración de logging
    ├── metrics_collector.py # Colección de métricas
    ├── parallel_processing.py # Procesamiento paralelo
    ├── security_utils.py    # Utilidades de seguridad
    ├── serialization.py     # Serialización de datos
    ├── text_processing.py   # Procesamiento de texto
    └── validation.py        # Validación de datos
📁 DATA/ - ESTRUCTURA DE DATOS PERSISTENTES

text
data/
├── .gitkeep                  # Mantener carpeta en git
├── init_data_structure.py    # Script de inicialización de estructura
│
├── 📁 backups/              # Backups automáticos
│   ├── .gitkeep
│   ├── backups_manifest.json # Metadatos de backups
│   └── README.md
│
├── 📁 cache/               # Caché persistente (L3)
│   ├── .gitkeep
│   ├── L3_cache_config.json # Configuración de caché en disco
│   └── README.md
│
├── 📁 embeddings/          # Base vectorial ChromaDB
│   ├── .gitkeep
│   ├── chroma.json        # Configuración ChromaDB
│   ├── chromadb_config.yaml # Configuración avanzada
│   └── README.md
│
├── 📁 graph_exports/      # Exportaciones de grafos
│   ├── .gitkeep
│   ├── export_template.cypher   # Plantilla Cypher
│   ├── export_template.graphml  # Plantilla GraphML
│   └── README.md
│
├── 📁 projects/           # Proyectos analizados
│   ├── .gitkeep
│   ├── project_template.json # Plantilla de proyecto
│   └── README.md
│
└── 📁 state/             # Estado del sistema
    ├── .gitkeep
    ├── agents_state_template.json # Plantilla estado agentes
    ├── system_state.json          # Estado del sistema
    └── README.md
📁 DEPLOYMENTS/ - CONFIGURACIÓN DE DESPLIEGUE

text
deployments/
│
├── 📁 docker/            # Configuración Docker
│   ├── Dockerfile        # Para producción
│   ├── Dockerfile.dev    # Para desarrollo
│   ├── .dockerignore
│   ├── backup.sh         # Scripts de backup
│   ├── health-check.sh   # Health checks
│   ├── init-db.sh        # Inicialización de BD
│   └── nginx.conf        # Configuración nginx
│
├── 📁 helm/             # Charts Helm para Kubernetes
│   ├── Chart.yaml
│   ├── values.yaml
│   └── 📁 templates/    # Plantillas Kubernetes
│       ├── 📁 api/      # Despliegue API
│       │   ├── deployment.yaml
│       │   ├── ingress.yaml
│       │   └── service.yaml
│       └── _helpers.tpl # Helpers
│
├── 📁 kubernetes/       # Configuraciones K8s nativas
│   ├── api-deployment.yaml
│   ├── configmap.yaml
│   ├── hpa.yaml         # Auto-scaling
│   ├── kustomization.yaml
│   ├── monitoring.yaml
│   ├── namespace.yaml
│   ├── neo4j.yaml
│   ├── nginx-ingress.yaml
│   ├── postgresql.yaml
│   ├── redis.yaml
│   ├── secrets.yaml
│   └── serviceaccount.yaml
│
├── docker-compose.yml        # Desarrollo local
└── docker-compose.prod.yml   # Producción
📁 SCRIPTS/ - UTILIDADES DE SISTEMA

text
scripts/
├── analyze_project.py        # Análisis de proyectos
├── backup_restore.py         # Backup y restauración
├── exhaustive_project_analyzer.py # Análisis exhaustivo
├── export_knowledge.py       # Exportación de conocimiento
├── init_data_system.py       # Inicialización de sistema de datos
├── init_db.sql              # SQL inicial para PostgreSQL
├── init_project.py          # Inicialización de proyecto
├── migrate_data.py          # Migración de datos
├── monitor_system.py        # Monitoreo del sistema
├── query_project.py         # Consulta de proyectos
├── setup_data_permissions.sh # Permisos de datos
└── verify_data_integrity.py  # Verificación de integridad
📁 REQUIREMENTS/ - DEPENDENCIAS

text
requirements/
├── agents.txt       # Dependencias para agentes
├── api.txt          # Dependencias para API
├── base.txt         # Dependencias base obligatorias
├── core.txt         # Dependencias del núcleo
├── databases.txt    # Bases de datos (PostgreSQL, Neo4j, Redis)
├── dev.txt          # Desarrollo (testing, debugging)
├── ml.txt           # Machine Learning (transformers, embeddings)
├── nlp.txt          # Procesamiento de lenguaje natural
└── prod.txt         # Producción (optimizaciones, seguridad)
📁 GITHUB/ - CI/CD

text
.github/
├── dependabot.yml           # Actualizaciones automáticas
│
└── 📁 workflows/
    ├── ci.yml              # Integración continua
    ├── cd.yml              # Despliegue continuo
    ├── tests.yml           # Ejecución de tests
    └── security.yml        # Escaneo de seguridad
📁 TESTS/ - PRUEBAS

text
tests/
├── conftest.py             # Configuración pytest
│
├── 📁 analyzer_code/       # Utilidades de análisis (¿Mover a scripts/?)
│   ├── analyzer_completo.py
│   ├── config_analyzer.yaml
│   ├── requerements.txt
│   ├── run_analyzer.txt
│   └── workflow_discovery.txt
│
├── 📁 e2e/                # Pruebas end-to-end
│   ├── test_analysis_workflow.py
│   ├── test_query_workflow.py
│   └── test_system_workflow.py
│
├── 📁 fixtures/           # Datos de prueba
│   ├── sample_code/      # Código de ejemplo
│   ├── sample_project/   # Proyecto de prueba
│   └── test_data.json    # Datos estructurados
│
├── 📁 integration/        # Pruebas de integración
│   └── test_core_integration.py
│
├── 📁 performance/        # Pruebas de rendimiento
│   ├── test_analysis_performance.py
│   ├── test_concurrent_performance.py
│   └── test_query_performance.py
│
└── 📁 unit/              # Pruebas unitarias
    ├── test_agents_base.py
    ├── test_embeddings_generator.py
    └── test_indexer_parser.py
📁 DOCS/ - DOCUMENTACIÓN

text
docs/
│
├── 📁 api/                # Documentación de API
│   ├── cli_reference.md
│   ├── grpc_api.md
│   ├── README.md
│   ├── rest_api.md
│   └── websocket_api.md
│
├── 📁 architecture/       # Arquitectura del sistema
│   ├── architecture_overview.md
│   ├── cohesion_coupling.md
│   ├── implementation_plan.md
│   ├── modules_details.md
│   ├── performance_analysis.md
│   ├── README.md
│   └── system_vision.md
│
├── 📁 deployment/        # Despliegue
│   ├── docker_deployment.md
│   ├── kubernetes_deployment.md
│   ├── local_deployment.md
│   ├── monitoring.md
│   ├── production_configuration.md
│   └── README.md
│
├── 📁 developer/         # Desarrolladores
│   ├── adding_parsers.md
│   ├── contributing.md
│   ├── extending_agents.md
│   ├── README.md
│   ├── setup_development.md
│   └── testing.md
│
├── 📁 examples/          # Ejemplos de uso
│   └── README.md
│
└── 📁 user_guide/       # Guía de usuario
    ├── advanced_features.md
    ├── asking_questions.md
    ├── getting_started.md
    ├── project_analysis.md
    ├── README.md
    └── Troubleshooting.md
📁 MONITORING/ - MONITOREO

text
monitoring/
│
├── 📁 alerts/            # Reglas de alertas
│   └── alerts.yml
│
├── 📁 grafana/          # Dashboards Grafana
│   ├── 📁 dashboards/
│   │   ├── dashboards.yaml
│   │   └── project_brain.json
│   └── 📁 datasources/
│       └── prometheus.yml
│
├── 📁 loki/            # Log aggregation
│   └── loki-config.yaml
│
└── 📁 prometheus/      # Métricas
    └── prometheus.yaml