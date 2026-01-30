"""
Módulo core - Núcleo del sistema Project Brain.
Contiene la orquestación principal, gestión de estado y componentes base.
"""

from .orchestrator import BrainOrchestrator, OperationRequest, OperationResult
from .system_state import SystemState, SystemStateManager
from .workflow_manager import WorkflowOrchestrator, WorkflowContext
from .event_bus import EventBus, Event, EventType
from .config_manager import ConfigManager
from .dependency_injector import DependencyInjector, DependencyScope
from .plugin_manager import PluginManager, PluginInfo
from .health_check import HealthCheck, HealthCheckResult, HealthStatus
from .exceptions import (
    BrainException,
    ValidationError,
    ConfigurationError,
    PluginError,
    WorkflowError,
    HealthCheckError,
    TimeoutError,
    DependencyError,
    ResourceError,
    AnalysisError,
    AgentError,
    LearningError
)

__all__ = [
    # Clases principales
    "BrainOrchestrator",
    "SystemStateManager",
    "WorkflowOrchestrator",
    "EventBus",
    "ConfigManager",
    "DependencyInjector",
    "PluginManager",
    "HealthCheck",
    
    # Modelos de datos
    "OperationRequest",
    "OperationResult",
    "SystemState",
    "WorkflowContext",
    "Event",
    "PluginInfo",
    "HealthCheckResult",
    
    # Enums y tipos
    "EventType",
    "HealthStatus",
    "DependencyScope",
    
    # Excepciones
    "BrainException",
    "ValidationError",
    "ConfigurationError",
    "PluginError",
    "WorkflowError",
    "HealthCheckError",
    "TimeoutError",
    "DependencyError",
    "ResourceError",
    "AnalysisError",
    "AgentError",
    "LearningError"
]