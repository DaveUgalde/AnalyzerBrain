"""
WorkflowOrchestrator - Orquestación de flujos de trabajo complejos.
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import uuid

from pydantic import BaseModel, Field

from .exceptions import BrainException, WorkflowError


# -------------------------------------------------
# ENUMS
# -------------------------------------------------

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# -------------------------------------------------
# DEFINICIONES
# -------------------------------------------------

@dataclass
class StepDefinition:
    """Definición de un paso en el flujo de trabajo."""
    name: str
    function: Callable
    description: str = ""
    timeout: int = 300
    retries: int = 3
    retry_delay: int = 5
    dependencies: List[str] = field(default_factory=list)
    required: bool = True


class StepResult(BaseModel):
    """Resultado de un paso."""
    step_name: str
    status: StepStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    attempts: int = 0

    class Config:
        arbitrary_types_allowed = True


class WorkflowContext(BaseModel):
    """Contexto del flujo de trabajo."""
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    data: Dict[str, Any] = Field(default_factory=dict)
    step_results: Dict[str, StepResult] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# -------------------------------------------------
# ORQUESTADOR
# -------------------------------------------------

class WorkflowOrchestrator:
    """Orquestador de flujos de trabajo."""

    def __init__(self):
        self.workflows: Dict[str, List[StepDefinition]] = {}
        self.active_workflows: Dict[str, asyncio.Task] = {}
        self.workflow_contexts: Dict[str, WorkflowContext] = {}

    # ----------------------------
    # REGISTRO
    # ----------------------------

    def register_workflow(self, workflow_name: str, steps: List[StepDefinition]) -> None:
        if workflow_name in self.workflows:
            raise BrainException(f"Workflow {workflow_name} already registered")

        step_names = {step.name for step in steps}
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    raise WorkflowError(
                        f"Dependency {dep} not found in workflow {workflow_name}"
                    )

        self.workflows[workflow_name] = steps

    # ----------------------------
    # EJECUCIÓN
    # ----------------------------

    async def execute_workflow(
        self,
        workflow_name: str,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> str:
        if workflow_name not in self.workflows:
            raise WorkflowError(f"Workflow {workflow_name} not found")

        workflow_id = str(uuid.uuid4())

        context = WorkflowContext(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            data=initial_data or {}
        )

        self.workflow_contexts[workflow_id] = context

        task = asyncio.create_task(
            self._run_workflow(context)
        )
        self.active_workflows[workflow_id] = task

        return workflow_id

    async def _run_workflow(self, context: WorkflowContext) -> None:
        steps = self.workflows[context.workflow_name]
        context.status = WorkflowStatus.RUNNING

        for step in steps:
            # Verificar dependencias
            deps_met = all(
                dep in context.step_results and
                context.step_results[dep].status == StepStatus.SUCCESS
                for dep in step.dependencies
            )

            if not deps_met:
                msg = f"Dependencies not met for step {step.name}"
                context.errors.append(msg)

                if step.required:
                    context.status = WorkflowStatus.FAILED
                    return
                else:
                    context.step_results[step.name] = StepResult(
                        step_name=step.name,
                        status=StepStatus.SKIPPED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        attempts=0
                    )
                    continue

            result = await self._execute_step(step, context)
            context.step_results[step.name] = result

            if result.status == StepStatus.FAILED and step.required:
                context.status = WorkflowStatus.FAILED
                return

        context.status = WorkflowStatus.COMPLETED

    async def _execute_step(
        self,
        step: StepDefinition,
        context: WorkflowContext
    ) -> StepResult:

        result = StepResult(
            step_name=step.name,
            status=StepStatus.RUNNING,
            start_time=datetime.now()
        )

        for attempt in range(1, step.retries + 1):
            result.attempts = attempt
            try:
                if asyncio.iscoroutinefunction(step.function):
                    output = await asyncio.wait_for(
                        step.function(context.data),
                        timeout=step.timeout
                    )
                else:
                    loop = asyncio.get_running_loop()
                    output = await loop.run_in_executor(
                        None, step.function, context.data
                    )

                result.status = StepStatus.SUCCESS
                result.output = output
                break

            except asyncio.TimeoutError:
                result.status = StepStatus.FAILED
                result.error = f"Timeout after {step.timeout} seconds"

            except Exception as e:
                result.status = StepStatus.FAILED
                result.error = str(e)

            if attempt < step.retries:
                await asyncio.sleep(step.retry_delay)

        result.end_time = datetime.now()
        return result

    # ----------------------------
    # CONTROL
    # ----------------------------

    async def cancel_workflow(self, workflow_id: str) -> bool:
        task = self.active_workflows.get(workflow_id)

        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        context = self.workflow_contexts.get(workflow_id)
        if context:
            context.status = WorkflowStatus.CANCELLED
            context.errors.append("Workflow cancelled by user")

        return True

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        context = self.workflow_contexts.get(workflow_id)

        if not context:
            return {"error": "Workflow not found"}

        steps = self.workflows.get(context.workflow_name, [])
        completed_steps = [
            name for name, result in context.step_results.items()
            if result.status in {
                StepStatus.SUCCESS,
                StepStatus.FAILED,
                StepStatus.SKIPPED
            }
        ]

        return {
            "workflow_id": workflow_id,
            "workflow_name": context.workflow_name,
            "status": context.status.value,
            "progress": f"{len(completed_steps)}/{len(steps)}",
            "step_results": {
                name: {
                    "status": result.status.value,
                    "error": result.error,
                    "start_time": result.start_time.isoformat(),
                    "end_time": result.end_time.isoformat() if result.end_time else None,
                    "attempts": result.attempts
                }
                for name, result in context.step_results.items()
            },
            "errors": context.errors,
        }

    # ----------------------------
    # VALIDACIÓN
    # ----------------------------

    def validate_workflow(self, workflow_name: str) -> List[str]:
        errors: List[str] = []

        if workflow_name not in self.workflows:
            return [f"Workflow {workflow_name} not found"]

        steps = self.workflows[workflow_name]
        step_names = [step.name for step in steps]

        if len(step_names) != len(set(step_names)):
            errors.append("Step names must be unique")

        for step in steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    errors.append(
                        f"Step {step.name}: dependency {dep} not found"
                    )

        return errors
