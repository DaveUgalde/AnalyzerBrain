"""
Workflow Manager for ANALYZERBRAIN.

Provides a flexible, event-driven workflow system with step dependencies,
retry logic, error handling, and comprehensive monitoring.

Dependencies:
    - event_bus: For workflow event publishing and subscriptions
    - system_state: For component registration and health monitoring
    - exceptions: For structured error handling
    - asyncio: For asynchronous workflow execution
    - loguru: For structured logging

Author: ANALYZERBRAIN Team
Date: 2024
Version: 1.0.0
"""

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

from loguru import logger

from .event_bus import EventBus, Event, EventPriority, EventDeliveryMode
from .system_state import SystemState, ComponentType, HealthStatus
from .exceptions import AnalyzerBrainError, ErrorCode, ErrorSeverity


# -------------------------------------------------------------------
# WORKFLOW EXCEPTIONS
# -------------------------------------------------------------------
class WorkflowError(AnalyzerBrainError):
    """Base exception for workflow-related errors."""

    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        step_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        error_details = details or {}
        if workflow_name:
            error_details["workflow_name"] = workflow_name
        if step_name:
            error_details["step_name"] = step_name
        if workflow_id:
            error_details["workflow_id"] = workflow_id

        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            severity=ErrorSeverity.HIGH,
            details=error_details,
            cause=cause,
        )


class StepExecutionError(WorkflowError):
    """Exception raised when a workflow step fails."""

    def __init__(
        self,
        message: str,
        workflow_name: str,
        step_name: str,
        workflow_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            workflow_name=workflow_name,
            step_name=step_name,
            workflow_id=workflow_id,
            details=details,
            cause=cause,
        )


class WorkflowValidationError(WorkflowError):
    """Exception raised when workflow validation fails."""

    def __init__(
        self,
        message: str,
        workflow_name: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            workflow_name=workflow_name,
            details=details,
            cause=cause,
        )


# -------------------------------------------------------------------
# ENUMS AND DATA STRUCTURES
# -------------------------------------------------------------------
class WorkflowStatus(Enum):
    """Workflow execution status."""

    DRAFT = "draft"  # Workflow defined but not validated
    VALIDATED = "validated"  # Workflow validated and ready
    PENDING = "pending"  # Waiting to start
    RUNNING = "running"  # Currently executing
    PAUSED = "paused"  # Execution paused
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Execution failed
    CANCELLED = "cancelled"  # Manually cancelled
    TIMEOUT = "time out"  # Execution timed out


class StepStatus(Enum):
    """Workflow step execution status."""

    PENDING = "pending"  # Waiting to execute
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Execution failed
    SKIPPED = "skipped"  # Skipped due to condition
    RETRYING = "retrying"  # Retrying after failure


class StepCondition(Enum):
    """Conditions for step execution."""

    ALWAYS = "always"  # Always execute
    ON_SUCCESS = "on_success"  # Only if previous step succeeded
    ON_FAILURE = "on_failure"  # Only if previous step failed
    CONDITIONAL = "conditional"  # Based on custom condition


@dataclass
class WorkflowStep:
    """Definition of a single workflow step."""

    name: str
    action: Callable
    description: str = ""

    # Dependencies
    requires: List[str] = field(default_factory=list)  # Steps that must complete first
    provides: List[str] = field(default_factory=list)  # Data keys this step provides

    # Execution control
    condition: StepCondition = StepCondition.ALWAYS
    condition_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    timeout: int = 300  # seconds
    retry_attempts: int = 0
    retry_delay: int = 5  # seconds

    # Metadata
    tags: List[str] = field(default_factory=list)
    critical: bool = False  # If True, failure stops entire workflow

    def __post_init__(self):
        """Validate step configuration."""
        # Ensure requires and provides are unique
        self.requires = list(set(self.requires))
        self.provides = list(set(self.provides))


@dataclass
class StepExecutionResult:
    """Result of a step execution."""

    step_name: str
    status: StepStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None  # seconds
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    output_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "error": self.error,
            "retry_count": self.retry_count,
            "output_data_keys": list(self.output_data.keys()),
        }


@dataclass
class WorkflowExecutionContext:
    """Context for workflow execution."""

    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    created_at: datetime = field(default_factory=datetime.now)

    # Execution tracking
    current_step: Optional[str] = None
    step_results: Dict[str, StepExecutionResult] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)

    # Data context
    input_data: Dict[str, Any] = field(default_factory=dict)
    context_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None

    # Metadata
    priority: int = 50
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_duration(self):
        """Update duration if workflow has ended."""
        if self.start_time and self.end_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        self.update_duration()

        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "current_step": self.current_step,
            "execution_order": self.execution_order,
            "step_count": len(self.step_results),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "input_data_keys": list(self.input_data.keys()),
            "output_data_keys": list(self.output_data.keys()),
            "priority": self.priority,
            "created_by": self.created_by,
            "metadata": self.metadata,
        }

    @property
    def completed_steps(self):
        """Get set of completed step names."""
        return {
            name
            for name, result in self.step_results.items()
            if result.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
        }


# -------------------------------------------------------------------
# WORKFLOW MANAGER
# -------------------------------------------------------------------
class WorkflowManager:
    """
    Manages complex workflows with dependencies, retries, and monitoring.

    Features:
    - Directed acyclic graph (DAG) based workflow execution
    - Step dependencies and data passing
    - Conditional step execution
    - Automatic retries with backoff
    - Comprehensive event publishing
    - Real-time progress tracking
    - Timeout and cancellation support
    """

    def __init__(self, event_bus: EventBus, system_state: SystemState):
        """
        Initialize workflow manager.

        Args:
            event_bus: Event bus for workflow events
            system_state: System state for component registration
        """
        self.event_bus = event_bus
        self.system_state = system_state

        # Workflow registry
        self.workflow_definitions: Dict[str, List[WorkflowStep]] = {}
        self.workflow_dependencies: Dict[str, Dict[str, Set[str]]] = {}

        # Active workflows
        self.active_workflows: Dict[str, WorkflowExecutionContext] = {}
        self.workflow_tasks: Dict[str, asyncio.Task] = {}

        # History and statistics
        self.workflow_history: Dict[str, List[WorkflowExecutionContext]] = {}
        self.max_history_per_workflow = 100

        # Configuration
        self.default_timeout = 3600  # 1 hour
        self.max_concurrent_workflows = 10

        # Component registration
        self.system_state.register_component(
            name="workflow_manager",
            component_type=ComponentType.CORE,
            health_check=self._health_check,
        )

        # Event subscriptions
        self._setup_event_subscriptions()

        logger.info("Workflow Manager initialized")

    def _setup_event_subscriptions(self):
        """Subscribe to workflow-related events."""
        # Workflow control events
        self.event_bus.subscribe("workflow.execute", self._on_workflow_execute)
        self.event_bus.subscribe("workflow.cancel", self._on_workflow_cancel)
        self.event_bus.subscribe("workflow.pause", self._on_workflow_pause)
        self.event_bus.subscribe("workflow.resume", self._on_workflow_resume)

        # System events
        self.event_bus.subscribe("system.shutdown", self._on_system_shutdown)
        self.event_bus.subscribe("system.state_changed", self._on_system_state_changed)

    # -------------------------------------------------------------------
    # WORKFLOW REGISTRATION AND VALIDATION
    # -------------------------------------------------------------------
    # En la función register_workflow, asegurarnos del orden correcto:
    def register_workflow(self, name: str, steps: List[WorkflowStep]) -> None:
        """
        Register a new workflow definition.

        Args:
            name: Unique workflow name
            steps: List of workflow steps

        Raises:
            WorkflowValidationError: If workflow is invalid
        """
        if name in self.workflow_definitions:
            raise WorkflowValidationError(
                f"Workflow already registered: {name}",
                workflow_name=name,
                details={"existing_workflows": list(self.workflow_definitions.keys())},
            )

        # 1. Validar pasos básicos (sin validar dependencias)
        self._validate_workflow_steps(name, steps)

        # 2. Construir grafo de dependencias (valida que los pasos requeridos existan)
        dependency_graph = self._build_dependency_graph(steps, name)

        # 3. Verificar dependencias circulares
        if self._has_circular_dependencies(dependency_graph):
            raise WorkflowValidationError(
                f"Workflow '{name}' has circular dependencies",
                workflow_name=name,
                details={"dependency_graph": {k: list(v) for k, v in dependency_graph.items()}},
            )

        # 4. Almacenar definición de workflow
        self.workflow_definitions[name] = steps
        self.workflow_dependencies[name] = dependency_graph

        logger.info(f"Workflow registered: {name} ({len(steps)} steps)")

    # En la función _validate_workflow_steps:
    def _validate_workflow_steps(self, workflow_name: str, steps: List[WorkflowStep]) -> None:
        """Validate workflow steps for consistency."""
        step_names = set()

        for step in steps:
            # Check for duplicate step names
            if step.name in step_names:
                raise WorkflowValidationError(
                    f"Duplicate step name: {step.name}",
                    workflow_name=workflow_name,
                    details={"duplicate_name": step.name, "step_names": list(step_names)},
                )
            step_names.add(step.name)

            # Validate step configuration
            if not step.name.strip():
                raise WorkflowValidationError(
                    "Step name cannot be empty",
                    workflow_name=workflow_name,
                    details={"step_index": steps.index(step)},
                )

            # Validate step configuration
            if not callable(step.action):
                raise WorkflowValidationError(
                    f"Step action must be callable: {step.name}",
                    workflow_name=workflow_name,
                    details={"step_name": step.name, "action_type": type(step.action).__name__},
                )

            # Validate conditional step has condition_func
            if step.condition == StepCondition.CONDITIONAL and step.condition_func is None:
                raise WorkflowValidationError(
                    f"Conditional step '{step.name}' requires condition_func",
                    workflow_name=workflow_name,
                    details={"step_name": step.name, "condition": step.condition.value},
                )

            # NOTA: No validamos aquí si los pasos requeridos existen
            # Esto se hará en _build_dependency_graph y la detección de ciclos
            # La validación de requerimientos desconocidos se manejará en _build_dependency_graph

    # En la función _build_dependency_graph:
    def _build_dependency_graph(
        self,
        steps: List[WorkflowStep],
        workflow_name: str,
    ) -> Dict[str, Set[str]]:
        """
        Build a dependency graph for the workflow.

        The graph maps each step name to the set of step names it depends on.
        """
        # Primero crear un mapa de nombres de pasos
        step_names = {step.name for step in steps}
        
        # Inicializar grafo
        graph: Dict[str, Set[str]] = {step.name: set() for step in steps}

        for step in steps:
            for requirement in step.requires:
                # Validar que el paso requerido existe
                if requirement not in step_names:
                    raise WorkflowValidationError(
                        f"Step '{step.name}' requires unknown step '{requirement}'",
                        workflow_name=workflow_name,
                        details={
                            "step_name": step.name,
                            "unknown_requirement": requirement,
                            "available_steps": list(step_names),
                        },
                    )
                graph[step.name].add(requirement)

        return graph

    def _has_circular_dependencies(self, graph: Dict[str, Set[str]]) -> bool:
        """
        Detect circular dependencies in a dependency graph using DFS.

        Returns True if a cycle is found, otherwise False.
        """
        visited: Set[str] = set()
        recursion_stack: Set[str] = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            recursion_stack.add(node)

            for dependency in graph.get(node, ()):
                if dependency not in visited:
                    if has_cycle(dependency):
                        return True
                elif dependency in recursion_stack:
                    return True

            recursion_stack.discard(node)
            return False

        for node in graph:
            if node not in visited and has_cycle(node):
                return True

        return False

    #  En workflow_manager.py, reemplazar get_available_steps con:

    def get_available_steps(
        self, workflow_name: str, context: WorkflowExecutionContext
    ) -> List[str]:
        """Get steps that are ready to execute based on dependencies and conditions."""
        if workflow_name not in self.workflow_definitions:
            return []

        steps = self.workflow_definitions[workflow_name]
        dependency_graph = self.workflow_dependencies[workflow_name]

        # Steps that finished successfully or were skipped
        finished_steps = {
            step_name
            for step_name, result in context.step_results.items()
            if result.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
        }

        available_steps: List[str] = []

        # Buscar todos los pasos disponibles, no solo críticos
        for step in steps:
            step_name = step.name

            # Skip steps already processed
            if step_name in context.step_results:
                continue

            # Dependencies must be COMPLETED or SKIPPED
            dependencies = dependency_graph.get(step_name, set())
            if not dependencies.issubset(finished_steps):
                continue

            # Evaluate execution condition
            if not self._should_execute_step(step, context):
                # Mark as skipped
                context.step_results[step_name] = StepExecutionResult(
                    step_name=step_name,
                    status=StepStatus.SKIPPED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration=0.0,
                )
                logger.debug(f"Step skipped: {step_name} in workflow {workflow_name}")
                continue

            available_steps.append(step_name)

        return available_steps

    def _should_execute_step(self, step: WorkflowStep, context: WorkflowExecutionContext) -> bool:
        """Determine if a step should be executed based on conditions."""
        if step.condition == StepCondition.ALWAYS:
            return True

        elif step.condition == StepCondition.ON_SUCCESS:
            # Check if ALL dependencies succeeded
            for dep in step.requires:
                if dep in context.step_results:
                    if context.step_results[dep].status != StepStatus.COMPLETED:
                        return False
                else:
                    # Dependency hasn't executed yet
                    return False
            return True

        elif step.condition == StepCondition.ON_FAILURE:
            # Check if ANY dependency failed
            for dep in step.requires:
                if dep in context.step_results:
                    if context.step_results[dep].status == StepStatus.FAILED:
                        return True
            return False

        elif step.condition == StepCondition.CONDITIONAL:
            # Use custom condition function
            if step.condition_func:
                try:
                    return step.condition_func(context.context_data)
                except Exception as e:
                    logger.error(f"Condition function failed for step {step.name}: {e}")
                    return False
            return False

        return True

    # -------------------------------------------------------------------
    # WORKFLOW EXECUTION
    # -------------------------------------------------------------------
    async def execute_workflow(
        self,
        workflow_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        priority: int = 50,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecutionContext:
        """
        Execute a registered workflow.

        Args:
            workflow_name: Name of the workflow to execute
            input_data: Initial input data for the workflow
            workflow_id: Optional workflow ID (generated if not provided)
            priority: Execution priority (higher = more important)
            created_by: Who initiated the workflow
            metadata: Additional metadata

        Returns:
            Workflow execution context

        Raises:
            WorkflowError: If workflow execution fails
        """
        if workflow_name not in self.workflow_definitions:
            raise WorkflowError(
                f"Workflow not found: {workflow_name}",
                workflow_name=workflow_name,
                details={"available_workflows": list(self.workflow_definitions.keys())},
            )

        # Check concurrent workflow limit
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            raise WorkflowError(
                f"Maximum concurrent workflows reached: {self.max_concurrent_workflows}",
                workflow_name=workflow_name,
                details={
                    "active_workflows": len(self.active_workflows),
                    "max_concurrent": self.max_concurrent_workflows,
                },
            )

        # Generate workflow ID if not provided
        workflow_id = workflow_id or f"{workflow_name}_{uuid4().hex[:8]}"

        # Create execution context
        context = WorkflowExecutionContext(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            status=WorkflowStatus.PENDING,
            input_data=input_data or {},
            context_data=input_data.copy() if input_data else {},
            priority=priority,
            created_by=created_by,
            metadata=metadata or {},
        )

        # Store active workflow
        self.active_workflows[workflow_id] = context

        # Start execution as background task
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._execute_workflow_internal(context), name=f"workflow_{workflow_id}"
        )
        self.workflow_tasks[workflow_id] = task

        # Publish workflow started event
        loop.create_task(
            self.event_bus.publish(
                "workflow.started",
                {
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_name,
                    "priority": priority,
                    "created_by": created_by,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        )

        logger.info(f"Workflow execution started: {workflow_id} ({workflow_name})")

        return context
    
    async def _execute_workflow_internal(self, context: WorkflowExecutionContext) -> None:
        workflow_id = context.workflow_id
        workflow_name = context.workflow_name

        try:
            context.status = WorkflowStatus.RUNNING
            context.start_time = datetime.now()

            steps = self.workflow_definitions[workflow_name]
            step_map = {step.name: step for step in steps}

            while True:
                # -------------------------
                # PAUSE HANDLING
                # -------------------------
                if context.status == WorkflowStatus.PAUSED:
                    await asyncio.sleep(0.2)
                    continue

                # -------------------------
                # CANCELLATION CHECK
                # -------------------------
                if context.status == WorkflowStatus.CANCELLED:
                    break

                # -------------------------
                # GET AVAILABLE STEPS
                # -------------------------
                available_steps = self.get_available_steps(workflow_name, context)

                # -------------------------
                # COMPLETION CHECK
                # -------------------------
                if not available_steps:
                    all_steps = set(step_map.keys())

                    # Get steps in final states
                    completed_steps = set()
                    for step_name, result in context.step_results.items():
                        if result.status in [
                            StepStatus.COMPLETED,
                            StepStatus.FAILED,
                            StepStatus.SKIPPED,
                        ]:
                            completed_steps.add(step_name)

                    # workflow done
                    if all_steps.issubset(completed_steps):
                        # Check if any step failed and workflow should fail
                        failed_steps = {
                            name
                            for name, result in context.step_results.items()
                            if result.status == StepStatus.FAILED
                        }

                        # If any critical step failed, workflow fails
                        for step_name in failed_steps:
                            step = step_map.get(step_name)
                            if step and step.critical:
                                context.status = WorkflowStatus.FAILED
                                break
                        else:
                            # No critical failures
                            context.status = WorkflowStatus.COMPLETED

                        break

                    # true deadlock - check if any steps can still run
                    pending = all_steps - completed_steps
                    can_still_run = False
                    for step_name in pending:
                        step = step_map[step_name]
                        dependencies = self.workflow_dependencies[workflow_name].get(
                            step_name, set()
                        )
                        if dependencies.issubset(completed_steps) and self._should_execute_step(
                            step, context
                        ):
                            can_still_run = True
                            break

                    if not can_still_run:
                        # Check if workflow should fail due to failed steps
                        failed_steps = {
                            name
                            for name, result in context.step_results.items()
                            if result.status == StepStatus.FAILED
                        }
                        if failed_steps:
                            context.status = WorkflowStatus.FAILED
                        else:
                            context.status = WorkflowStatus.COMPLETED
                        break

                    # Steps still running → wait
                    await asyncio.sleep(0.05)
                    continue

                # -------------------------
                # EXECUTE STEPS
                # -------------------------
                tasks = []
                for step_name in available_steps:
                    step = step_map[step_name]
                    context.current_step = step_name

                    task = asyncio.create_task(
                        self._execute_step(step, context),
                        name=f"step_{step_name}_{workflow_id}",
                    )
                    tasks.append(task)

                # -------------------------
                # WAIT FOR STEPS
                # -------------------------
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Check for critical failures
                    critical_failure_occurred = False
                    for step_name, result in zip(available_steps, results):
                        if isinstance(result, Exception):
                            step = step_map[step_name]
                            
                            if step.critical:
                                critical_failure_occurred = True
                                # Cancelar tareas pendientes si hay un fallo crítico
                                for task in tasks:
                                    if not task.done():
                                        task.cancel()
                                
                                # Publicar evento de fallo
                                loop = asyncio.get_running_loop()
                                loop.create_task(
                                    self.event_bus.publish(
                                        "workflow.failed",
                                        {
                                            "workflow_id": workflow_id,
                                            "workflow_name": workflow_name,
                                            "error": str(result),
                                            "current_step": step_name,
                                            "timestamp": datetime.now().isoformat(),
                                        },
                                    )
                                )
                                break  # Salir del loop de verificación

                    if critical_failure_occurred:
                        # Si hubo un fallo crítico, marcar workflow como fallido y salir
                        context.status = WorkflowStatus.FAILED
                        break

                await asyncio.sleep(0.01)

            # -------------------------
            # COMPLETION
            # -------------------------
            context.end_time = datetime.now()
            context.update_duration()

            # Solo publicar evento de completado si no es un fallo
            if context.status != WorkflowStatus.FAILED:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.event_bus.publish(
                        "workflow.completed",
                        {
                            "workflow_id": workflow_id,
                            "workflow_name": workflow_name,
                            "status": context.status.value,
                            "duration": context.duration,
                            "step_count": len(context.step_results),
                            "timestamp": context.end_time.isoformat(),
                        },
                    )
                )

            logger.info(
                f"Workflow completed: {workflow_id} ({context.duration:.2f}s) - Status: {context.status.value}"
            )

        except asyncio.CancelledError:
            context.status = WorkflowStatus.CANCELLED
            context.end_time = datetime.now()
            context.update_duration()

            # Mark current step as cancelled if it exists
            if context.current_step and context.current_step not in context.step_results:
                context.step_results[context.current_step] = StepExecutionResult(
                    step_name=context.current_step,
                    status=StepStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration=0.0,
                    error="Cancelled",
                )

            loop = asyncio.get_running_loop()
            loop.create_task(
                self.event_bus.publish(
                    "workflow.cancelled",
                    {
                        "workflow_id": workflow_id,
                        "workflow_name": workflow_name,
                        "timestamp": context.end_time.isoformat(),
                    },
                )
            )

            logger.info(f"Workflow cancelled: {workflow_id}")

        except Exception as e:
            context.status = WorkflowStatus.FAILED
            context.end_time = datetime.now()
            context.update_duration()

            loop = asyncio.get_running_loop()
            loop.create_task(
                self.event_bus.publish(
                    "workflow.failed",
                    {
                        "workflow_id": workflow_id,
                        "workflow_name": workflow_name,
                        "error": str(e),
                        "current_step": context.current_step,
                        "timestamp": context.end_time.isoformat(),
                    },
                )
            )

            logger.error(f"Workflow failed: {workflow_id} - {e}")

            if not isinstance(e, WorkflowError):
                raise WorkflowError(
                    f"Workflow execution failed: {e}",
                    workflow_name=workflow_name,
                    workflow_id=workflow_id,
                    cause=e,
                ) from e
            raise

        finally:
            self.active_workflows.pop(workflow_id, None)
            self.workflow_tasks.pop(workflow_id, None)
            self._record_workflow_history(context)

    async def _execute_step(
        self, step: WorkflowStep, context: WorkflowExecutionContext
    ) -> StepExecutionResult:
        """
        Execute a single workflow step with retry logic.

        Args:
            step: Workflow step to execute
            context: Workflow execution context

        Returns:
            Step execution result
        """
        step_name = step.name
        workflow_id = context.workflow_id

        # Create result object
        result = StepExecutionResult(
            step_name=step_name,
            status=StepStatus.PENDING,
            start_time=datetime.now(),
            retry_count=0,
        )

        # Update context
        context.current_step = step_name

        # Publish step started event
        loop = asyncio.get_running_loop()
        loop.create_task(
            self.event_bus.publish(
                "workflow.step_started",
                {
                    "workflow_id": workflow_id,
                    "workflow_name": context.workflow_name,
                    "step_name": step_name,
                    "timestamp": result.start_time.isoformat(),
                },
            )
        )

        # Execute with retries
        last_exception = None

        for attempt in range(step.retry_attempts + 1):
            try:
                result.retry_count = attempt

                if attempt > 0:
                    result.status = StepStatus.RETRYING
                    logger.info(f"Retry {attempt}/{step.retry_attempts} for step {step_name}")

                    # Delay before retry
                    await asyncio.sleep(step.retry_delay * attempt)  # Exponential backoff

                # Prepare arguments
                args = self._prepare_step_arguments(step, context)

                # Execute step with timeout
                if asyncio.iscoroutinefunction(step.action):
                    # Async function
                    step_result = await asyncio.wait_for(step.action(**args), timeout=step.timeout)
                else:
                    # Sync function - run in executor
                    loop = asyncio.get_event_loop()
                    step_result = await loop.run_in_executor(None, lambda: step.action(**args))

                # Step succeeded
                result.status = StepStatus.COMPLETED
                result.result = step_result

                # Extract output data if step provides it
                if step.provides:
                    if isinstance(step_result, dict):
                        for key in step.provides:
                            if key in step_result:
                                result.output_data[key] = step_result[key]
                                context.context_data[key] = step_result[key]
                                context.output_data[key] = step_result[key]
                    else:
                        # If step_result is not a dict, use step name as key
                        for key in step.provides:
                            result.output_data[key] = step_result
                            context.context_data[key] = step_result
                            context.output_data[key] = step_result

                break

            except asyncio.TimeoutError as e:
                last_exception = e
                result.error = f"Step timed out after {step.timeout} seconds"
                result.status = StepStatus.FAILED
                logger.error(f"Step {step_name} timed out (attempt {attempt + 1})")

            except Exception as e:
                last_exception = e
                result.error = str(e)
                result.status = StepStatus.FAILED
                logger.error(f"Step {step_name} failed (attempt {attempt + 1}): {e}")

        # Finalize result
        result.end_time = datetime.now()
        result.duration = (result.end_time - result.start_time).total_seconds()

        # Store result in context
        context.step_results[step_name] = result
        context.execution_order.append(step_name)

        # Publish step completed event
        loop = asyncio.get_running_loop()
        loop.create_task(
            self.event_bus.publish(
                "workflow.step_completed",
                {
                    "workflow_id": workflow_id,
                    "workflow_name": context.workflow_name,
                    "step_name": step_name,
                    "status": result.status.value,
                    "duration": result.duration,
                    "retry_count": result.retry_count,
                    "error": result.error,
                    "timestamp": result.end_time.isoformat(),
                },
            )
        )

        # If step failed after all retries, raise exception
        if result.status == StepStatus.FAILED:
            raise StepExecutionError(
                f"Step failed after {step.retry_attempts + 1} attempts: {result.error}",
                workflow_name=context.workflow_name,
                step_name=step_name,
                workflow_id=workflow_id,
                cause=last_exception,
                details={
                    "retry_attempts": step.retry_attempts,
                    "timed out": step.timeout,
                    "duration": result.duration,
                },
            )

        return result

    def _prepare_step_arguments(
        self, step: WorkflowStep, context: WorkflowExecutionContext
    ) -> Dict[str, Any]:
        """
        Prepare arguments for a workflow step.

        Args:
            step: Workflow step
            context: Workflow execution context

        Returns:
            Dictionary of arguments

        Raises:
            WorkflowError: If required arguments are missing
        """
        args = {}

        # Add workflow context
        args["workflow_context"] = context

        # Add data from context based on step requirements
        for requirement in step.requires:
            # PRIMERO: Buscar si el requirement es una clave de datos directa
            if requirement in context.context_data:
                args[requirement] = context.context_data[requirement]
            else:
                # SEGUNDO: Si no es clave directa, buscar en los resultados de pasos
                # Esto es un HACK temporal - el diseño necesita ser reconsiderado
                found = False
                for step_name, step_result in context.step_results.items():
                    if requirement == step_name:
                        # Si el paso proporcionó datos específicos, pasarlos
                        if step_result.output_data:
                            # Pasar todos los datos de output
                            args.update(step_result.output_data)
                        else:
                            # Pasar el resultado completo
                            args[requirement] = step_result.result
                        found = True
                        break

                if not found:
                    raise WorkflowError(
                        f"Step {step.name} requires '{requirement}' but it's not available",
                        workflow_name=context.workflow_name,
                        step_name=step.name,
                        workflow_id=context.workflow_id,
                        details={
                            "required": requirement,
                            "available_keys": list(context.context_data.keys()),
                            "completed_steps": list(context.step_results.keys()),
                        },
                    )

        return args

    # -------------------------------------------------------------------
    # WORKFLOW CONTROL
    # -------------------------------------------------------------------
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            workflow_id: ID of the workflow to cancel

        Returns:
            True if workflow was cancelled, False if not found
        """
        if workflow_id not in self.active_workflows:
            logger.warning(f"Cannot cancel workflow: {workflow_id} not found")
            return False

        context = self.active_workflows[workflow_id]
        context.status = WorkflowStatus.CANCELLED

        # Cancel the task if it exists
        if workflow_id in self.workflow_tasks:
            task = self.workflow_tasks[workflow_id]
            if not task.done():
                task.cancel()

        logger.info(f"Workflow cancelled: {workflow_id}")
        return True

    async def pause_workflow(self, workflow_id: str) -> bool:
        """
        Pause a running workflow.

        Args:
            workflow_id: ID of the workflow to pause

        Returns:
            True if workflow was paused, False if not found
        """
        if workflow_id not in self.active_workflows:
            logger.warning(f"Cannot pause workflow: {workflow_id} not found")
            return False

        context = self.active_workflows[workflow_id]
        if context.status == WorkflowStatus.RUNNING:
            context.status = WorkflowStatus.PAUSED
            logger.info(f"Workflow paused: {workflow_id}")
            return True

        return False

    async def resume_workflow(self, workflow_id: str) -> bool:
        """
        Resume a paused workflow.

        Args:
            workflow_id: ID of the workflow to resume

        Returns:
            True if workflow was resumed, False if not found
        """
        if workflow_id not in self.active_workflows:
            logger.warning(f"Cannot resume workflow: {workflow_id} not found")
            return False

        context = self.active_workflows[workflow_id]
        if context.status == WorkflowStatus.PAUSED:
            context.status = WorkflowStatus.RUNNING
            logger.info(f"Workflow resumed: {workflow_id}")
            return True

        return False

    # -------------------------------------------------------------------
    # QUERY AND MONITORING
    # -------------------------------------------------------------------
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowExecutionContext]:
        """
        Get the current status of a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow execution context if found, None otherwise
        """
        return self.active_workflows.get(workflow_id)

    def get_registered_workflows(self) -> List[str]:
        """Get list of registered workflow names."""
        return list(self.workflow_definitions.keys())

    def get_active_workflows(self) -> Dict[str, WorkflowExecutionContext]:
        """Get all active workflows."""
        return self.active_workflows.copy()

    def get_workflow_history(
        self, workflow_name: str, limit: int = 10
    ) -> List[WorkflowExecutionContext]:
        """
        Get execution history for a workflow.

        Args:
            workflow_name: Name of the workflow
            limit: Maximum number of history entries to return

        Returns:
            List of workflow execution contexts
        """
        history = self.workflow_history.get(workflow_name, [])
        return history[-limit:] if limit else history

    def _record_workflow_history(self, context: WorkflowExecutionContext) -> None:
        """Record workflow execution in history."""
        workflow_name = context.workflow_name

        if workflow_name not in self.workflow_history:
            self.workflow_history[workflow_name] = []

        self.workflow_history[workflow_name].append(context)

        # Limit history size
        if len(self.workflow_history[workflow_name]) > self.max_history_per_workflow:
            self.workflow_history[workflow_name] = self.workflow_history[workflow_name][
                -self.max_history_per_workflow :
            ]

    # -------------------------------------------------------------------
    # HEALTH AND METRICS
    # -------------------------------------------------------------------
    async def _health_check(self) -> bool:
        """Health check for the workflow manager."""
        try:
            # Check internal structures
            assert isinstance(self.workflow_definitions, dict)
            assert isinstance(self.active_workflows, dict)
            assert isinstance(self.workflow_tasks, dict)

            # Check for stalled workflows
            now = datetime.now()
            for workflow_id, context in self.active_workflows.items():
                if (
                    context.start_time
                    and (now - context.start_time).total_seconds() > self.default_timeout
                ):
                    logger.warning(f"Workflow {workflow_id} may be stalled")

            return True

        except Exception as e:
            logger.error(f"Workflow manager health check failed: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get workflow manager metrics."""
        completed_steps = 0
        failed_steps = 0
        skipped_steps = 0

        # Count from active workflows
        for context in self.active_workflows.values():
            for result in context.step_results.values():
                if result.status == StepStatus.COMPLETED:
                    completed_steps += 1
                elif result.status == StepStatus.FAILED:
                    failed_steps += 1
                elif result.status == StepStatus.SKIPPED:
                    skipped_steps += 1

        # Count from history
        for history_list in self.workflow_history.values():
            for context in history_list:
                for result in context.step_results.values():
                    if result.status == StepStatus.COMPLETED:
                        completed_steps += 1
                    elif result.status == StepStatus.FAILED:
                        failed_steps += 1
                    elif result.status == StepStatus.SKIPPED:
                        skipped_steps += 1

        return {
            "workflows_registered": len(self.workflow_definitions),
            "workflows_active": len(self.active_workflows),
            "workflow_tasks_running": len(self.workflow_tasks),
            "steps_completed_total": completed_steps,
            "steps_failed_total": failed_steps,
            "steps_skipped_total": skipped_steps,
            "workflow_history_size": sum(len(h) for h in self.workflow_history.values()),
            "max_concurrent_workflows": self.max_concurrent_workflows,
            "timestamp": datetime.now().isoformat(),
        }

    # -------------------------------------------------------------------
    # EVENT HANDLERS
    # -------------------------------------------------------------------
    async def _on_workflow_execute(self, event: Event) -> None:
        """Handle workflow execution request event."""
        data = event.data
        workflow_name = data["workflow_name"]

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self.execute_workflow(
                    workflow_name=workflow_name,
                    input_data=data.get("input_data"),
                    workflow_id=data.get("workflow_id"),
                    priority=data.get("priority", 50),
                    created_by=data.get("created_by"),
                    metadata=data.get("metadata"),
                )
            )
        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow_name}: {e}")

    async def _on_workflow_cancel(self, event: Event) -> None:
        """Handle workflow cancellation request event."""
        data = event.data
        workflow_id = data["workflow_id"]
        await self.cancel_workflow(workflow_id)

    async def _on_workflow_pause(self, event: Event) -> None:
        """Handle workflow pause request event."""
        data = event.data
        workflow_id = data["workflow_id"]
        await self.pause_workflow(workflow_id)

    async def _on_workflow_resume(self, event: Event) -> None:
        """Handle workflow resume request event."""
        data = event.data
        workflow_id = data["workflow_id"]
        await self.resume_workflow(workflow_id)

    async def _on_system_shutdown(self, event: Event) -> None:
        """Handle system shutdown event."""
        # Cancel all active workflows
        for workflow_id in list(self.active_workflows.keys()):
            await self.cancel_workflow(workflow_id)

        logger.info("Workflow Manager shutdown initiated")

    async def _on_system_state_changed(self, event: Event) -> None:
        """Handle system state changes."""
        data = event.data
        new_status = data.get("new_status")

        # If system is stopping, begin graceful shutdown
        if new_status in ["stopping", "error"]:
            await self._on_system_shutdown(event)

    # -------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------
    async def shutdown(self) -> None:
        """Gracefully shutdown workflow manager."""
        logger.info("Shutting down Workflow Manager...")

        # Cancel all active workflows
        for workflow_id in list(self.active_workflows.keys()):
            await self.cancel_workflow(workflow_id)

        # Wait for tasks to complete
        tasks = list(self.workflow_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Unregister component
        self.system_state.unregister_component("workflow_manager")

        logger.info("Workflow Manager shutdown complete")

    def __str__(self) -> str:
        """String representation of workflow manager."""
        return (
            f"WorkflowManager("
            f"workflows={len(self.workflow_definitions)}, "
            f"active={len(self.active_workflows)}, "
            f"history={sum(len(h) for h in self.workflow_history.values())})"
        )


def _dependencies_completed(self, step, ctx) -> bool:
    for dep in step.requires:
        if dep not in ctx.step_results:
            return False

        status = ctx.step_results[dep].status
        if status not in {StepStatus.COMPLETED, StepStatus.SKIPPED}:
            return False

    return True


# Singleton instance
_workflow_manager_instance: Optional[WorkflowManager] = None


def get_workflow_manager(
    event_bus: Optional[EventBus] = None, system_state: Optional[SystemState] = None
) -> WorkflowManager:
    """
    Get or create singleton workflow manager instance.

    Args:
        event_bus: Event bus instance (required on first call)
        system_state: System state instance (required on first call)

    Returns:
        WorkflowManager instance

    Raises:
        WorkflowError: If dependencies not provided on first call
    """
    global _workflow_manager_instance

    if _workflow_manager_instance is None:
        if event_bus is None or system_state is None:
            raise WorkflowError(
                "EventBus and SystemState required for WorkflowManager initialization"
            )
        _workflow_manager_instance = WorkflowManager(event_bus, system_state)

    return _workflow_manager_instance
