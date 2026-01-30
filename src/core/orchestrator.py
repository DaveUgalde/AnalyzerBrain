"""
BrainOrchestrator - Coordinador principal del sistema.
Responsable de orquestar todos los componentes, gestionar flujos de trabajo
y mantener el estado global del sistema.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
import time
from datetime import datetime
from pydantic import BaseModel, Field, validator
from .system_state import SystemState, SystemStateManager
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
    CRITICAL = 0    # Análisis en tiempo real, respuestas a usuarios
    HIGH = 1        # Procesamiento de cambios, aprendizaje
    MEDIUM = 2      # Análisis programado, mantenimiento
    LOW = 3         # Tareas en background, limpieza

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
            OperationPriority.LOW: 3600
        }
    )
    enable_learning: bool = True
    enable_monitoring: bool = True
    enable_backup: bool = True
    plugins_directory: str = "./plugins"

class ProjectContext(BaseModel):
    """Contexto de un proyecto para operaciones."""
    project_id: str = Field(..., description="ID único del proyecto")
    project_path: str = Field(..., description="Ruta al proyecto")
    language: Optional[str] = Field(None, description="Lenguaje principal")
    analysis_depth: str = Field("comprehensive", description="Profundidad de análisis")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True

class OperationRequest(BaseModel):
    """Solicitud de operación al orquestador."""
    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str = Field(..., description="Tipo de operación")
    priority: OperationPriority = Field(OperationPriority.MEDIUM)
    context: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = Field(None)
    callback_url: Optional[str] = Field(None)
    
    @validator('operation_type')
    def validate_operation_type(cls, v):
        allowed_types = {'analyze_project', 'process_question', 'detect_changes', 
                        'learn_from_feedback', 'export_knowledge', 'system_status'}
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
    """
    Clase principal de orquestación del sistema.
    
    Responsabilidades:
    1. Coordinar todos los componentes del sistema
    2. Gestionar flujos de trabajo complejos
    3. Manejar concurrencia y prioridades
    4. Proporcionar interfaz unificada para operaciones
    5. Gestionar estado y recuperación de errores
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el orquestador del sistema.
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
            
        Raises:
            ValidationError: Si la configuración es inválida
            BrainException: Si hay errores de inicialización
        """
        self._config = self._load_config(config_path)
        self._state = SystemState()
        self._event_bus = EventBus()
        self._operations_queue = asyncio.PriorityQueue()
        self._active_operations: Dict[str, asyncio.Task] = {}
        self._components: Dict[str, Any] = {}
        self._plugins: Dict[str, Any] = {}
        self._is_running = False
        self._start_time = datetime.now()
        
        # Métricas
        self._metrics = {
            "operations_completed": 0,
            "operations_failed": 0,
            "total_processing_time_ms": 0.0,
            "avg_response_time_ms": 0.0,
            "concurrent_operations": 0
        }
        
    async def initialize(self) -> bool:
        """
        Inicializa todos los componentes del sistema.
        
        Returns:
            bool: True si la inicialización fue exitosa
            
        Raises:
            BrainException: Si algún componente falla al inicializar
        """
        try:
            # 1. Inicializar sistema de eventos
            await self._event_bus.initialize()
            
            # 2. Inicializar componentes core
            components_order = [
                "config_manager",
                "dependency_injector", 
                "plugin_manager",
                "health_check",
                "workflow_manager"
            ]
            
            for component_name in components_order:
                component = await self._initialize_component(component_name)
                self._components[component_name] = component
            
            # 3. Cargar plugins
            await self._load_plugins()
            
            # 4. Iniciar workers de procesamiento
            await self._start_workers()
            
            # 5. Iniciar monitoreo
            if self._config.enable_monitoring:
                await self._start_monitoring()
            
            self._is_running = True
            self._state.set_ready()
            
            await self._event_bus.publish(
                EventType.SYSTEM_STARTED,
                {"timestamp": datetime.now(), "version": "1.0.0"}
            )
            
            return True
            
        except Exception as e:
            await self._event_bus.publish(
                EventType.SYSTEM_ERROR,
                {"error": str(e), "component": "orchestrator"}
            )
            raise BrainException(f"Failed to initialize orchestrator: {e}")
    
    async def process_operation(self, request: OperationRequest) -> OperationResult:
        """
        Procesa una operación en el sistema.
        
        Args:
            request: Solicitud de operación
            
        Returns:
            OperationResult: Resultado de la operación
            
        Raises:
            ValidationError: Si la solicitud es inválida
            TimeoutError: Si la operación excede el timeout
        """
        start_time = datetime.now()
        operation_id = request.operation_id
        
        try:
            # Validar solicitud
            self._validate_operation_request(request)
            
            # Publicar evento de inicio
            await self._event_bus.publish(
                EventType.OPERATION_STARTED,
                {"operation_id": operation_id, "type": request.operation_type}
            )
            
            # Procesar según tipo
            if request.operation_type == "analyze_project":
                result = await self._analyze_project(request.context)
            elif request.operation_type == "process_question":
                result = await self._process_question(request.context)
            elif request.operation_type == "detect_changes":
                result = await self._detect_changes(request.context)
            elif request.operation_type == "learn_from_feedback":
                result = await self._learn_from_feedback(request.context)
            elif request.operation_type == "export_knowledge":
                result = await self._export_knowledge(request.context)
            elif request.operation_type == "system_status":
                result = await self._get_system_status()
            else:
                raise ValidationError(f"Unknown operation type: {request.operation_type}")
            
            # Calcular métricas
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(success=True, processing_time=processing_time)
            
            # Publicar evento de completado
            await self._event_bus.publish(
                EventType.OPERATION_COMPLETED,
                {
                    "operation_id": operation_id,
                    "success": True,
                    "processing_time_ms": processing_time
                }
            )
            
            return OperationResult(
                operation_id=operation_id,
                success=True,
                data=result,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(success=False, processing_time=processing_time)
            
            await self._event_bus.publish(
                EventType.OPERATION_FAILED,
                {
                    "operation_id": operation_id,
                    "error": str(e),
                    "processing_time_ms": processing_time
                }
            )
            
            return OperationResult(
                operation_id=operation_id,
                success=False,
                error=str(e),
                processing_time_ms=processing_time
            )
    
    async def analyze_project(self, project_path: str, 
                            options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Ejecuta análisis completo de un proyecto.
        
        Args:
            project_path: Ruta al proyecto a analizar
            options: Opciones de análisis (profundidad, lenguajes, etc.)
            
        Returns:
            Dict con resultados del análisis
            
        Example:
            {
                "project_id": "proj_123",
                "status": "completed",
                "files_analyzed": 150,
                "entities_extracted": 1200,
                "analysis_time_seconds": 45.2,
                "findings": [...],
                "recommendations": [...]
            }
        """
        request = OperationRequest(
            operation_type="analyze_project",
            priority=OperationPriority.HIGH,
            context={
                "project_path": project_path,
                "options": options or {},
                "timestamp": datetime.now().isoformat()
            }
        )
        
        result = await self.process_operation(request)
        
        if not result.success:
            raise BrainException(f"Project analysis failed: {result.error}")
        
        return result.data
    
    async def ask_question(self, question: str, 
                          project_id: Optional[str] = None,
                          context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Procesa una pregunta sobre un proyecto.
        
        Args:
            question: Pregunta en lenguaje natural
            project_id: ID del proyecto (opcional si ya hay contexto)
            context: Contexto adicional
            
        Returns:
            Dict con respuesta estructurada
            
        Example:
            {
                "answer": "La función process_data se define en...",
                "confidence": 0.92,
                "sources": [...],
                "reasoning_chain": [...],
                "suggested_followups": [...]
            }
        """
        request = OperationRequest(
            operation_type="process_question",
            priority=OperationPriority.CRITICAL,
            context={
                "question": question,
                "project_id": project_id,
                "additional_context": context or {},
                "session_id": str(uuid.uuid4())
            },
            timeout_seconds=30  # Máximo 30 segundos para respuestas
        )
        
        result = await self.process_operation(request)
        
        if not result.success:
            # Fallback a respuesta básica si hay error
            return {
                "answer": "Lo siento, hubo un error procesando tu pregunta.",
                "confidence": 0.0,
                "sources": [],
                "error": result.error
            }
        
        return result.data
    
    async def shutdown(self, force: bool = False) -> bool:
        """
        Apaga el sistema de manera controlada.
        
        Args:
            force: Si True, fuerza el apagado sin esperar operaciones
            
        Returns:
            bool: True si el apagado fue exitoso
        """
        try:
            # Publicar evento de apagado
            await self._event_bus.publish(
                EventType.SYSTEM_SHUTDOWN_STARTED,
                {"timestamp": datetime.now(), "force": force}
            )
            
            # Detener aceptación de nuevas operaciones
            self._is_running = False
            
            # Esperar operaciones pendientes (si no es forzado)
            if not force and self._active_operations:
                await asyncio.sleep(5)  # Dar tiempo para completar
            
            # Apagar componentes en orden inverso
            shutdown_order = reversed(list(self._components.keys()))
            
            for component_name in shutdown_order:
                component = self._components[component_name]
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
            
            # Apagar event bus
            await self._event_bus.shutdown()
            
            # Guardar estado
            await self._state.save()
            
            # Publicar evento de apagado completado
            await self._event_bus.publish(
                EventType.SYSTEM_SHUTDOWN_COMPLETED,
                {"timestamp": datetime.now()}
            )
            
            return True
            
        except Exception as e:
            # En caso de error, loguear pero continuar con apagado
            await self._event_bus.publish(
                EventType.SYSTEM_ERROR,
                {"error": str(e), "context": "shutdown"}
            )
            
            if force:
                return True
            raise BrainException(f"Error during shutdown: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas del sistema.
        
        Returns:
            Dict con métricas actualizadas
        """
        return {
            **self._metrics,
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
            "active_operations": len(self._active_operations),
            "components_initialized": list(self._components.keys()),
            "plugins_loaded": list(self._plugins.keys())
        }
    
    async def list_operations(self, status: Optional[str] = None) -> List[Dict]:
        """
        Lista operaciones según su estado.
        
        Args:
            status: Estado a filtrar (opcional)
            
        Returns:
            Lista de operaciones con su información
        """
        # Esta es una implementación simplificada
        # En una implementación real, se consultaría una base de datos
        operations = []
        
        for op_id, task in self._active_operations.items():
            op_info = {
                "operation_id": op_id,
                "status": "running" if not task.done() else "completed",
                "task": str(task)
            }
            
            if status is None or op_info["status"] == status:
                operations.append(op_info)
        
        return operations
    
    async def cancel_operation(self, operation_id: str) -> bool:
        """
        Cancela una operación en curso.
        
        Args:
            operation_id: ID de la operación a cancelar
            
        Returns:
            bool: True si se canceló exitosamente
        """
        task = self._active_operations.get(operation_id)
        
        if task is None:
            return False
        
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        del self._active_operations[operation_id]
        
        await self._event_bus.publish(
            EventType.OPERATION_CANCELLED,
            {"operation_id": operation_id}
        )
        
        return True
    
    async def get_operation_result(self, operation_id: str) -> Optional[OperationResult]:
        """
        Obtiene el resultado de una operación completada.
        
        Args:
            operation_id: ID de la operación
            
        Returns:
            OperationResult si existe, None en caso contrario
        """
        # Esta es una implementación simplificada
        # En una implementación real, se consultaría una base de datos
        return None
    
    # Métodos privados de implementación
    
    async def _analyze_project(self, context: Dict) -> Dict[str, Any]:
        """Implementación específica de análisis de proyecto."""
        # Este método sería implementado con la lógica real
        # Por ahora retornamos estructura esperada
        return {
            "project_id": context.get("project_id", "temp_id"),
            "status": "completed",
            "files_analyzed": 0,
            "entities_extracted": 0,
            "analysis_time_seconds": 0.0,
            "findings": [],
            "recommendations": []
        }
    
    async def _process_question(self, context: Dict) -> Dict[str, Any]:
        """Implementación específica de procesamiento de preguntas."""
        return {
            "answer": "Respuesta de ejemplo",
            "confidence": 0.9,
            "sources": [],
            "reasoning_chain": [],
            "suggested_followups": []
        }
    
    async def _detect_changes(self, context: Dict) -> Dict[str, Any]:
        """Implementación específica de detección de cambios."""
        return {
            "changes_detected": [],
            "files_modified": 0,
            "impact_analysis": {}
        }
    
    async def _learn_from_feedback(self, context: Dict) -> Dict[str, Any]:
        """Implementación específica de aprendizaje de feedback."""
        return {
            "learning_applied": True,
            "confidence_updates": 0,
            "new_patterns": []
        }
    
    async def _export_knowledge(self, context: Dict) -> Dict[str, Any]:
        """Implementación específica de exportación de conocimiento."""
        return {
            "export_format": context.get("format", "json"),
            "size_bytes": 0,
            "entities_exported": 0
        }
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema."""
        return {
            "status": "running" if self._is_running else "stopped",
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
            "components": {name: "healthy" for name in self._components},
            "metrics": self._metrics,
            "active_operations": len(self._active_operations),
            "system_mode": self._config.system_mode.value
        }
    
    def _validate_operation_request(self, request: OperationRequest) -> None:
        """Valida una solicitud de operación."""
        if not request.operation_type:
            raise ValidationError("Operation type is required")
        
        if not self._is_running:
            raise BrainException("System is not running")
    
    def _update_metrics(self, success: bool, processing_time: float) -> None:
        """Actualiza métricas del sistema."""
        self._metrics["operations_completed"] += 1
        if not success:
            self._metrics["operations_failed"] += 1
        
        self._metrics["total_processing_time_ms"] += processing_time
        self._metrics["avg_response_time_ms"] = (
            self._metrics["total_processing_time_ms"] / self._metrics["operations_completed"]
        )
    
    async def _initialize_component(self, component_name: str) -> Any:
        """Inicializa un componente del sistema."""
        # Implementación real cargaría el componente dinámicamente
        # Por ahora retornamos un mock
        class MockComponent:
            async def shutdown(self):
                pass
        
        return MockComponent()
    
    async def _load_plugins(self) -> None:
        """Carga plugins del sistema."""
        pass
    
    async def _start_workers(self) -> None:
        """Inicia workers para procesamiento en background."""
        pass
    
    async def _start_monitoring(self) -> None:
        """Inicia sistema de monitoreo."""
        pass
    
    def _load_config(self, config_path: Optional[str]) -> OrchestratorConfig:
        """Carga configuración del orquestador."""
        # Implementación real leería de archivo
        return OrchestratorConfig()
    
    async def _handle_operation_timeout(self, operation_id: str) -> None:
        """Maneja timeout de una operación."""
        pass
    
    async def _cleanup_completed_operations(self) -> None:
        """Limpia operaciones completadas del registro."""
        pass
    
    def _balance_workload(self) -> Dict[str, int]:
        """Balancea carga de trabajo entre componentes."""
        return {}
    
    async def _recover_from_failure(self, error: Exception) -> bool:
        """Recupera el sistema de un fallo."""
        return True
    
    async def _notify_operation_progress(self, operation_id: str, progress: float, message: str) -> None:
        """Notifica progreso de una operación."""
        await self._event_bus.publish(
            EventType.OPERATION_PROGRESS,
            {
                "operation_id": operation_id,
                "progress": progress,
                "message": message,
                "timestamp": datetime.now()
            }
        )