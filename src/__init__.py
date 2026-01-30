"""
Project Brain - Sistema de inteligencia artificial para análisis y comprensión de código.
Módulo principal del sistema.
"""

__version__ = "1.0.0"
__author__ = "Project Brain Team"
__description__ = "Sistema de IA para análisis y comprensión de código con memoria infinita"

from src.core import (
    BrainOrchestrator,
    SystemStateManager,
    WorkflowOrchestrator,
    EventBus,
    ConfigManager,
    DependencyInjector,
    PluginManager,
    HealthCheck
)

__all__ = [
    "BrainOrchestrator",
    "SystemStateManager", 
    "WorkflowOrchestrator",
    "EventBus",
    "ConfigManager",
    "DependencyInjector",
    "PluginManager",
    "HealthCheck",
]