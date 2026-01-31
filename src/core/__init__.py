"""
Módulo core - Núcleo del sistema Project Brain.

Contiene la orquestación principal, gestión de estado y componentes base.
Este módulo expone la API pública del núcleo del sistema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "1.0.0"
__author__ = "Project Brain Team"
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
    "LearningError",
]

# Imports principales (runtime)
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
    LearningError,
)

# Soporte para type checkers (mypy, pyright, IDEs)
if TYPE_CHECKING:
    from .orchestrator import BrainOrchestrator as _BrainOrchestrator
    from .workflow_manager import WorkflowOrchestrator as _WorkflowOrchestrator
