"""
Módulo para gestionar el estado global del sistema.
"""

from datetime import datetime
from typing import Dict, Any, List
from loguru import logger

from .config_manager import config
from .exceptions import AnalyzerBrainError


class SystemState:
    """Mantiene el estado global del sistema."""

    def __init__(self):
        self.start_time = datetime.now()
        self.status = "initializing"
        self.metrics: Dict[str, Any] = {
            "projects_analyzed": 0,
            "errors": 0,
            "uptime": 0,
        }
        self.components: Dict[str, Any] = {}

    def register_component(self, name: str, component: Any) -> None:
        """Registra un componente del sistema."""
        self.components[name] = component
        logger.debug(f"Componente registrado: {name}")

    def update_metric(self, name: str, value: Any) -> None:
        """Actualiza una métrica del sistema."""
        self.metrics[name] = value

    def get_health_report(self) -> Dict[str, Any]:
        """Genera un reporte de salud del sistema."""
        self.metrics['uptime'] = (datetime.now() - self.start_time).total_seconds()
        return {
            "status": self.status,
            "metrics": self.metrics,
            "components": list(self.components.keys()),
            "timestamp": datetime.now().isoformat(),
        }
