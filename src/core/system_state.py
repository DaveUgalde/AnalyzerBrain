"""
SystemStateManager - Gestión de estado del sistema.
"""

from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
import json
from .exceptions import BrainException

class SystemStatus(Enum):
    """Estados del sistema."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"

class ComponentStatus(Enum):
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
    
    def set_status(self, status: SystemStatus) -> None:
        """Establece el estado del sistema."""
        self.status = status
    
    def get_status(self) -> SystemStatus:
        """Obtiene el estado del sistema."""
        return self.status
    
    def register_component(self, name: str) -> None:
        """Registra un componente."""
        if name in self.components:
            raise BrainException(f"Component {name} already registered")
        
        self.components[name] = ComponentInfo(name=name)
    
    def update_component_status(self, name: str, status: ComponentStatus) -> None:
        """Actualiza estado de un componente."""
        if name not in self.components:
            raise BrainException(f"Component {name} not registered")
        
        self.components[name].status = status
        self.components[name].last_heartbeat = datetime.now()
    
    def update_component_metrics(self, name: str, metrics: Dict[str, Any]) -> None:
        """Actualiza métricas de un componente."""
        if name not in self.components:
            raise BrainException(f"Component {name} not registered")
        
        self.components[name].metrics.update(metrics)
    
    def update_system_metrics(self, **kwargs) -> None:
        """Actualiza métricas del sistema."""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
    
    def get_uptime(self) -> float:
        """Obtiene tiempo de actividad en segundos."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el estado a dict."""
        return {
            "status": self.status.value,
            "uptime_seconds": self.get_uptime(),
            "components": {
                name: {
                    "status": comp.status.value,
                    "last_heartbeat": comp.last_heartbeat.isoformat() if comp.last_heartbeat else None,
                    "metrics": comp.metrics
                }
                for name, comp in self.components.items()
            },
            "metrics": self.metrics.dict(),
            "custom_state": self.custom_state
        }
    
    async def save(self, filepath: Optional[str] = None) -> bool:
        """Guarda el estado a disco."""
        try:
            state_dict = self.to_dict()
            state_json = json.dumps(state_dict, indent=2, default=str)
            
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(state_json)
            
            return True
        except Exception as e:
            raise BrainException(f"Failed to save system state: {e}")
    
    async def load(self, filepath: str) -> bool:
        """Carga el estado desde disco."""
        try:
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
            
            # Actualizar estado desde dict
            self.status = SystemStatus(state_dict.get("status", SystemStatus.INITIALIZING.value))
            # Nota: start_time no se carga desde archivo
            
            components = state_dict.get("components", {})
            for name, comp_data in components.items():
                if name not in self.components:
                    self.components[name] = ComponentInfo(name=name)
                
                self.components[name].status = ComponentStatus(comp_data.get("status", ComponentStatus.OFFLINE.value))
                
                last_heartbeat = comp_data.get("last_heartbeat")
                if last_heartbeat:
                    self.components[name].last_heartbeat = datetime.fromisoformat(last_heartbeat)
                
                self.components[name].metrics = comp_data.get("metrics", {})
            
            metrics_data = state_dict.get("metrics", {})
            self.metrics = SystemMetrics(**metrics_data)
            
            self.custom_state = state_dict.get("custom_state", {})
            
            return True
        except Exception as e:
            raise BrainException(f"Failed to load system state: {e}")
    
    def set_ready(self) -> None:
        """Marca el sistema como listo."""
        self.set_status(SystemStatus.READY)
    
    def set_running(self) -> None:
        """Marca el sistema como en ejecución."""
        self.set_status(SystemStatus.RUNNING)
    
    def set_degraded(self) -> None:
        """Marca el sistema como degradado."""
        self.set_status(SystemStatus.DEGRADED)
    
    def set_error(self) -> None:
        """Marca el sistema como en error."""
        self.set_status(SystemStatus.ERROR)

class SystemStateManager:
    """Gestor de estado del sistema (facade)."""
    
    def __init__(self):
        self._state = SystemState()
    
    def get_state(self) -> SystemState:
        """Obtiene el estado del sistema."""
        return self._state
    
    def set_state(self, status: SystemStatus) -> None:
        """Establece el estado del sistema."""
        self._state.set_status(status)
    
    async def save_state(self, filepath: Optional[str] = None) -> bool:
        """Guarda el estado."""
        return await self._state.save(filepath)
    
    async def load_state(self, filepath: str) -> bool:
        """Carga el estado."""
        return await self._state.load(filepath)
    
    def reset_state(self) -> None:
        """Reinicia el estado (manteniendo solo componentes registrados)."""
        components = self._state.components.copy()
        self._state = SystemState()
        self._state.components = components
    
    def track_operation(self, operation_id: str, operation_type: str) -> None:
        """Registra una operación en curso."""
        if "operations" not in self._state.custom_state:
            self._state.custom_state["operations"] = {}
        
        self._state.custom_state["operations"][operation_id] = {
            "type": operation_type,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
    
    def complete_operation(self, operation_id: str, success: bool) -> None:
        """Marca una operación como completada."""
        if "operations" in self._state.custom_state and operation_id in self._state.custom_state["operations"]:
            self._state.custom_state["operations"][operation_id]["end_time"] = datetime.now().isoformat()
            self._state.custom_state["operations"][operation_id]["status"] = "completed" if success else "failed"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema."""
        return self._state.to_dict()