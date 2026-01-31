"""
BrainOrchestrator - Coordinador principal del sistema.
Responsable de orquestar todos los componentes, gestionar flujos de trabajo
y mantener el estado global del sistema.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator

from .system_state import SystemState
from .event_bus import EventBus, EventType
from .exceptions import BrainException, ValidationError, TimeoutError


class SystemMode(Enum):
    """Modos de operación del sistema."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"


class OperationPriority(Enum):
    """Prioridades de operación."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class OrchestratorConfig:
    """Configuración del orquestador."""
    system_mode: SystemMode = SystemMode.DEVELOPMENT
    max_concurrent_operations: int = 10
    operation_timeout_seconds: Dict[OperationPriority, int] = field(
        default_factory=lambda: {
            OperationPriority.CRITICAL: 30,
            OperationPriority.HIGH: 300,
            OperationPriority.MEDIUM: 1800,
            OperationPriority.LOW: 3600,
        }
    )
    enable_learning: bool = True
    enable_monitoring: bool = True
    enable_backup: bool = True
    plugins_directory: str = "./plugins"


class ProjectContext(BaseModel):
    """Contexto de un proyecto para operaciones."""
    project_id: str
    project_path: str
    language: Optional[str] = None
    analysis_depth: str = "comprehensive"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


class OperationRequest(BaseModel):
    """Solicitud de operación al orquestador."""
    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str
    priority: OperationPriority = OperationPriority.MEDIUM
    context: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    callback_url: Optional[str] = None

    @validator("operation_type")
    def validate_operation_type(cls, v: str) -> str:
        allowed_types = {
            "analyze_project",
            "process_question",
            "detect_changes",
            "learn_from_feedback",
            "export_knowledge",
            "system_status",
        }
        if v not in allowed_types:
            raise ValueError(f"Operation type must be one of {allowed_types}")
        return v


class OperationResult(BaseModel):
    """Resultado de una operación."""
    operation_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class BrainOrchestrator:
    """Clase principal de orquestación del sistema."""

    def __init__(self, config_path: Optional[str] = None):
        self._config = self._load_config(config_path)
        self._state = SystemState()
        self._event_bus = EventBus()

        self._operations_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._active_operations: Dict[str, asyncio.Task] = {}

        self._components: Dict[str, Any] = {}
        self._plugins: Dict[str, Any] = {}

        self._is_running: bool = False
        self._start_time: datetime = datetime.now()

        self._metrics: Dict[str, Any] = {
            "operations_completed": 0,
            "operations_failed": 0,
            "total_processing_time_ms": 0.0,
            "avg_response_time_ms": 0.0,
            "concurrent_operations": 0,
        }

    async def initialize(self) -> bool:
        try:
            await self._event_bus.initialize()

            components_order = [
                "config_manager",
                "dependency_injector",
                "plugin_manager",
                "health_check",
                "workflow_manager",
            ]

            for name in components_order:
                self._components[name] = await self._initialize_component(name)

            await self._load_plugins()
            await self._start_workers()

            if self._config.enable_monitoring:
                await self._start_monitoring()

            self._is_running = True
            self._state.set_ready()

            await self._event_bus.publish(
                EventType.SYSTEM_STARTED,
                {"timestamp": datetime.now(), "version": "1.0.0"},
            )
            return True

        except Exception as e:
            await self._event_bus.publish(
                EventType.SYSTEM_ERROR,
                {"error": str(e), "component": "orchestrator"},
            )
            raise BrainException(f"Failed to initialize orchestrator: {e}") from e

    async def process_operation(self, request: OperationRequest) -> OperationResult:
        start_time = datetime.now()
        operation_id = request.operation_id

        try:
            self._validate_operation_request(request)

            await self._event_bus.publish(
                EventType.OPERATION_STARTED,
                {"operation_id": operation_id, "type": request.operation_type},
            )

            if request.operation_type == "analyze_project":
                data = await self._analyze_project(request.context)
            elif request.operation_type == "process_question":
                data = await self._process_question(request.context)
            elif request.operation_type == "detect_changes":
                data = await self._detect_changes(request.context)
            elif request.operation_type == "learn_from_feedback":
                data = await self._learn_from_feedback(request.context)
            elif request.operation_type == "export_knowledge":
                data = await self._export_knowledge(request.context)
            elif request.operation_type == "system_status":
                data = await self._get_system_status()
            else:
                raise ValidationError(
                    f"Unknown operation type: {request.operation_type}"
                )

            processing_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            self._update_metrics(True, processing_time)

            await self._event_bus.publish(
                EventType.OPERATION_COMPLETED,
                {
                    "operation_id": operation_id,
                    "success": True,
                    "processing_time_ms": processing_time,
                },
            )

            return OperationResult(
                operation_id=operation_id,
                success=True,
                data=data,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            processing_time = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            self._update_metrics(False, processing_time)

            await self._event_bus.publish(
                EventType.OPERATION_FAILED,
                {
                    "operation_id": operation_id,
                    "error": str(e),
                    "processing_time_ms": processing_time,
                },
            )

            return OperationResult(
                operation_id=operation_id,
                success=False,
                error=str(e),
                processing_time_ms=processing_time,
            )

    async def shutdown(self, force: bool = False) -> bool:
        try:
            await self._event_bus.publish(
                EventType.SYSTEM_SHUTDOWN_STARTED,
                {"timestamp": datetime.now(), "force": force},
            )

            self._is_running = False

            if not force and self._active_operations:
                await asyncio.sleep(5)

            for component in reversed(list(self._components.values())):
                if hasattr(component, "shutdown"):
                    await component.shutdown()

            await self._event_bus.shutdown()
            await self._state.save()

            await self._event_bus.publish(
                EventType.SYSTEM_SHUTDOWN_COMPLETED,
                {"timestamp": datetime.now()},
            )
            return True

        except Exception as e:
            await self._event_bus.publish(
                EventType.SYSTEM_ERROR,
                {"error": str(e), "context": "shutdown"},
            )
            if force:
                return True
            raise BrainException(f"Error during shutdown: {e}") from e

    async def get_metrics(self) -> Dict[str, Any]:
        uptime = (datetime.now() - self._start_time).total_seconds()
        return {
            **self._metrics,
            "uptime_seconds": uptime,
            "active_operations": len(self._active_operations),
            "components_initialized": list(self._components.keys()),
            "plugins_loaded": list(self._plugins.keys()),
        }

    # -----------------------
    # Métodos privados
    # -----------------------

    def _validate_operation_request(self, request: OperationRequest) -> None:
        if not request.operation_type:
            raise ValidationError("Operation type is required")
        if not self._is_running:
            raise BrainException("System is not running")

    def _update_metrics(self, success: bool, processing_time: float) -> None:
        self._metrics["operations_completed"] += 1
        if not success:
            self._metrics["operations_failed"] += 1

        self._metrics["total_processing_time_ms"] += processing_time
        completed = self._metrics["operations_completed"] or 1
        self._metrics["avg_response_time_ms"] = (
            self._metrics["total_processing_time_ms"] / completed
        )

    async def _initialize_component(self, component_name: str) -> Any:
        class MockComponent:
            async def shutdown(self) -> None:
                pass
        return MockComponent()

    async def _load_plugins(self) -> None:
        pass

    async def _start_workers(self) -> None:
        pass

    async def _start_monitoring(self) -> None:
        pass

    def _load_config(self, config_path: Optional[str]) -> OrchestratorConfig:
        return OrchestratorConfig()

    async def _analyze_project(self, context: Dict) -> Dict[str, Any]:
        return {
            "project_id": context.get("project_id", "temp_id"),
            "status": "completed",
            "files_analyzed": 0,
            "entities_extracted": 0,
            "analysis_time_seconds": 0.0,
            "findings": [],
            "recommendations": [],
        }

    async def _process_question(self, context: Dict) -> Dict[str, Any]:
        return {
            "answer": "Respuesta de ejemplo",
            "confidence": 0.9,
            "sources": [],
            "reasoning_chain": [],
            "suggested_followups": [],
        }

    async def _detect_changes(self, context: Dict) -> Dict[str, Any]:
        return {
            "changes_detected": [],
            "files_modified": 0,
            "impact_analysis": {},
        }

    async def _learn_from_feedback(self, context: Dict) -> Dict[str, Any]:
        return {
            "learning_applied": True,
            "confidence_updates": 0,
            "new_patterns": [],
        }

    async def _export_knowledge(self, context: Dict) -> Dict[str, Any]:
        return {
            "export_format": context.get("format", "json"),
            "size_bytes": 0,
            "entities_exported": 0,
        }

    async def _get_system_status(self) -> Dict[str, Any]:
        return {
            "status": "running" if self._is_running else "stopped",
            "uptime_seconds": (
                datetime.now() - self._start_time
            ).total_seconds(),
            "components": {name: "healthy" for name in self._components},
            "metrics": self._metrics,
            "active_operations": len(self._active_operations),
            "system_mode": self._config.system_mode.value,
        }
