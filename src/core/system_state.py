"""
SystemStateManager - Gestión de estado del sistema.
"""

from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
import json
from .exceptions import BrainException


class SystemStatus(str, Enum):
    """Estados del sistema."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


class ComponentStatus(str, Enum):
    """Estados de componentes."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class SystemMetrics(BaseModel):
    """Métricas del sistema."""
    uptime_seconds: float = 0.0
    operations_completed: int = 0
    operations_failed: int = 0
    avg_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    class Config:
        arbitrary_types_allowed = True


class ComponentInfo(BaseModel):
    """Información de un componente."""
    name: str
    status: ComponentStatus = ComponentStatus.OFFLINE
    last_heartbeat: Optional[datetime] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class SystemState:
    """Estado completo del sistema."""

    def __init__(self):
        self.status: SystemStatus = SystemStatus.INITIALIZING
        self.start_time: datetime = datetime.now()
        self.components: Dict[str, ComponentInfo] = {}
        self.metrics: SystemMetrics = SystemMetrics()
        self.custom_state: Dict[str, Any] = {}

    # -----------------
    # Estado general
    # -----------------

    def set_status(self, status: SystemStatus) -> None:
        self.status = status

    def get_status(self) -> SystemStatus:
        return self.status

    # -----------------
    # Componentes
    # -----------------

    def register_component(self, name: str) -> None:
        if name in self.components:
            raise BrainException(f"Component {name} already registered")

        self.components[name] = ComponentInfo(name=name)

    def update_component_status(self, name: str, status: ComponentStatus) -> None:
        component = self.components.get(name)
        if not component:
            raise BrainException(f"Component {name} not registered")

        component.status = status
        component.last_heartbeat = datetime.now()

    def update_component_metrics(self, name: str, metrics: Dict[str, Any]) -> None:
        component = self.components.get(name)
        if not component:
            raise BrainException(f"Component {name} not registered")

        component.metrics.update(metrics)

    # -----------------
    # Métricas
    # -----------------

    def update_system_metrics(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

    def get_uptime(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    # -----------------
    # Serialización
    # -----------------

    def to_dict(self) -> Dict[str, Any]:
        uptime = self.get_uptime()
        self.metrics.uptime_seconds = uptime

        return {
            "status": self.status.value,
            "uptime_seconds": uptime,
            "components": {
                name: {
                    "status": comp.status.value,
                    "last_heartbeat": comp.last_heartbeat.isoformat() if comp.last_heartbeat else None,
                    "metrics": comp.metrics,
                }
                for name, comp in self.components.items()
            },
            "metrics": self.metrics.dict(),
            "custom_state": self.custom_state,
        }

    async def save(self, filepath: Optional[str] = None) -> bool:
        try:
            state_json = json.dumps(self.to_dict(), indent=2, default=str)

            if filepath:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(state_json)

            return True
        except Exception as e:
            raise BrainException(f"Failed to save system state: {e}")

    async def load(self, filepath: str) -> bool:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state_dict = json.load(f)

            self.status = SystemStatus(state_dict.get("status", SystemStatus.INITIALIZING))

            # Componentes
            for name, comp_data in state_dict.get("components", {}).items():
                component = self.components.get(name) or ComponentInfo(name=name)

                component.status = ComponentStatus(
                    comp_data.get("status", ComponentStatus.OFFLINE)
                )

                if comp_data.get("last_heartbeat"):
                    component.last_heartbeat = datetime.fromisoformat(
                        comp_data["last_heartbeat"]
                    )

                component.metrics = comp_data.get("metrics", {})
                self.components[name] = component

            # Métricas
            self.metrics = SystemMetrics(**state_dict.get("metrics", {}))

            # Estado custom
            self.custom_state = state_dict.get("custom_state", {})

            return True
        except Exception as e:
            raise BrainException(f"Failed to load system state: {e}")

    # -----------------
    # Helpers de estado
    # -----------------

    def set_ready(self) -> None:
        self.set_status(SystemStatus.READY)

    def set_running(self) -> None:
        self.set_status(SystemStatus.RUNNING)

    def set_degraded(self) -> None:
        self.set_status(SystemStatus.DEGRADED)

    def set_error(self) -> None:
        self.set_status(SystemStatus.ERROR)


class SystemStateManager:
    """Gestor de estado del sistema (facade)."""

    def __init__(self):
        self._state = SystemState()

    def get_state(self) -> SystemState:
        return self._state

    def set_state(self, status: SystemStatus) -> None:
        self._state.set_status(status)

    async def save_state(self, filepath: Optional[str] = None) -> bool:
        return await self._state.save(filepath)

    async def load_state(self, filepath: str) -> bool:
        return await self._state.load(filepath)

    def reset_state(self) -> None:
        components = self._state.components.copy()
        self._state = SystemState()
        self._state.components = components

    # -----------------
    # Operaciones
    # -----------------

    def track_operation(self, operation_id: str, operation_type: str) -> None:
        ops = self._state.custom_state.setdefault("operations", {})
        ops[operation_id] = {
            "type": operation_type,
            "start_time": datetime.now().isoformat(),
            "status": "running",
        }

    def complete_operation(self, operation_id: str, success: bool) -> None:
        ops = self._state.custom_state.get("operations", {})
        if operation_id in ops:
            ops[operation_id]["end_time"] = datetime.now().isoformat()
            ops[operation_id]["status"] = "completed" if success else "failed"

            if success:
                self._state.metrics.operations_completed += 1
            else:
                self._state.metrics.operations_failed += 1

    def get_metrics(self) -> Dict[str, Any]:
        return self._state.to_dict()
