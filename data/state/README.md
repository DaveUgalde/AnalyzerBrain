10. ARCHIVO: data/state/README.md

markdown
# Directorio de Estado del Sistema

Este directorio almacena el estado persistente del sistema Project Brain, incluyendo configuración, estado de agentes, métricas y conocimiento aprendido.

## Estructura de Archivos
state/
├── system_state.json # Estado general del sistema
├── orchestrator_state.json # Estado del orquestador
├── agents_state/ # Estado individual de agentes
│ ├── code_analyzer.json
│ ├── qa_agent.json
│ ├── architect_agent.json
│ └── ...
├── learning_state/ # Estado de aprendizaje
│ ├── incremental_learner.json
│ ├── feedback_loop.json
│ ├── reinforcement_state.json
│ └── ...
├── knowledge_state/ # Estado del conocimiento
│ ├── semantic_memory.json
│ ├── episodic_memory.json
│ └── working_memory.json
├── workflow_state/ # Estado de flujos de trabajo
│ ├── active_workflows.json
│ ├── completed_workflows.json
│ └── workflow_templates.json
├── metrics_history/ # Histórico de métricas
│ ├── system_metrics.json
│ ├── performance_metrics.json
│ ├── business_metrics.json
│ └── ...
└── snapshots/ # Snapshots de estado
├── snapshot_20240101_120000.tar.gz
└── ...

text

## Archivos Principales

### system_state.json
Estado global del sistema, incluyendo:
- Configuración activa
- Estado de componentes
- Métricas en tiempo real
- Información de sesiones activas

### agents_state/
Cada agente mantiene su propio estado en archivos separados:
- Memoria del agente
- Conocimiento aprendido
- Métricas de desempeño
- Configuración específica

### learning_state/
Estado de los sistemas de aprendizaje:
- Modelos entrenados
- Feedback procesado
- Conocimiento refinado
- Parámetros de adaptación

## Formato de Estado

### Estado de Agente
```json
{
  "agent_id": "agent_123",
  "agent_type": "code_analyzer",
  "state": "ready",
  "memory": {
    "short_term": [...],
    "long_term": [...],
    "episodic": [...],
    "semantic": [...]
  },
  "knowledge": {
    "patterns_learned": [...],
    "rules_inferred": [...],
    "examples_stored": [...]
  },
  "metrics": {
    "requests_processed": 1500,
    "success_rate": 0.95,
    "avg_processing_time_ms": 45.2,
    "confidence_distribution": [...]
  },
  "configuration": {
    "confidence_threshold": 0.7,
    "learning_rate": 0.1,
    "capabilities": ["code_analysis", "pattern_detection"]
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
Estado de Aprendizaje

json
{
  "learner_id": "incremental_learner",
  "state": "active",
  "knowledge_base": {
    "total_concepts": 1250,
    "concepts_by_domain": {...},
    "confidence_scores": {...}
  },
  "learning_history": [
    {
      "timestamp": "2024-01-01T12:00:00Z",
      "event_type": "concept_learned",
      "concept": "singleton_pattern",
      "confidence_delta": 0.1
    }
  ],
  "adaptation_state": {
    "current_domain": "python_web",
    "adaptation_rate": 0.3,
    "domain_shift": 0.15
  },
  "forgetting_state": {
    "forgetting_applied": true,
    "concepts_forgotten": [...],
    "retention_rate": 0.95
  }
}
Persistencia y Recuperación

Guardar Estado

python
from src.core.system_state import SystemStateManager

state_manager = SystemStateManager()
state_manager.save_state(
    state_type="full",
    backup=True,
    compress=True
)
Cargar Estado

python
state_manager.load_state(
    snapshot_id="latest",
    restore_agents=True,
    restore_learning=True
)
Crear Snapshot

python
snapshot_id = state_manager.create_snapshot(
    name="pre_update_snapshot",
    description="Estado antes de actualización",
    include=["agents", "learning", "knowledge"]
)
Políticas de Retención

Estado actual: Siempre disponible en memoria
Snapshots diarios: Mantenidos por 7 días
Snapshots semanales: Mantenidos por 30 días
Snapshots mensuales: Mantenidos por 365 días
Estado de agentes: Persistido en cada cierre limpio
Recuperación de Fallos

Recuperación Automática

bash
# Recuperar desde último estado bueno
python scripts/recover_state.py --strategy=auto

# Recuperar desde snapshot específico
python scripts/recover_state.py --snapshot=snapshot_20240101_120000

# Validar integridad del estado
python scripts/validate_state.py --check=all
Recuperación Manual

Detener sistema
Restaurar snapshot desde backups/
Validar integridad
Iniciar sistema con estado restaurado
Monitoreo del Estado

Métricas Clave

Tamaño del estado
Tiempo de carga/guardado
Integridad del estado
Consistencia entre componentes
Alertas

Estado corrupto detectado
Tiempo de guardado excesivo
Inconsistencias en el estado
Espacio en disco insuficiente