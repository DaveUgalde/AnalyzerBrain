# tests/unit/helpers/workflow_helpers.py
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock

from src.core.workflow_manager import (
    WorkflowManager,
    WorkflowExecutionContext,
    WorkflowStatus,
    WorkflowStep,
    StepCondition,
    StepStatus
)


# ------------------------------------------------------------------
# ASYNC HELPERS
# ------------------------------------------------------------------
async def wait_for_workflow_completion(
    manager: WorkflowManager,
    workflow_id: str,
    timeout: float = 5.0,
) -> WorkflowExecutionContext:
    """
    Wait until the workflow reaches a terminal state or the timeout expires.
    """
    loop = asyncio.get_running_loop()
    start_time = loop.time()

    terminal_states = {
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
        WorkflowStatus.TIMEOUT,
    }

    while True:
        ctx = manager.get_workflow_status(workflow_id)
        if ctx and ctx.status in terminal_states:
            return ctx

        # Fallback: workflow may already be archived in history
        for history in manager.workflow_history.values():
            for execution in history:
                if execution.workflow_id == workflow_id:
                    return execution

        if loop.time() - start_time > timeout:
            raise TimeoutError(
                f"Workflow '{workflow_id}' did not finish within {timeout:.2f}s"
            )

        await asyncio.sleep(0.01)

async def wait_for_workflow_status_with_timeout(
    manager: WorkflowManager,
    workflow_id: str,
    status: WorkflowStatus,
    timeout: float = 5.0,
    poll_interval: float = 0.01
) -> WorkflowExecutionContext:
    """
    Wait for workflow to reach specific status with timeout.
    """
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    
    while True:
        ctx = manager.get_workflow_status(workflow_id)
        if ctx and ctx.status == status:
            return ctx
            
        # También revisar el historial
        for history_list in manager.workflow_history.values():
            for execution in history_list:
                if execution.workflow_id == workflow_id and execution.status == status:
                    return execution
        
        if loop.time() - start_time > timeout:
            raise TimeoutError(
                f"Workflow '{workflow_id}' did not reach status {status} within {timeout}s"
            )
        
        await asyncio.sleep(poll_interval)

async def wait_for_step_execution(
    manager: WorkflowManager,
    workflow_id: str,
    step: str,
    timeout: float = 5.0,
) -> None:
    """
    Wait until a specific step is executed (completed, failed, or skipped).
    """
    loop = asyncio.get_running_loop()
    start_time = loop.time()

    while True:
        ctx = manager.get_workflow_status(workflow_id)
        if ctx and step in ctx.step_results:
            result = ctx.step_results[step]
            if result.status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]:
                return
        
        if loop.time() - start_time > timeout:
            raise TimeoutError(
                f"Step '{step}' in workflow '{workflow_id}' did not execute within {timeout:.2f}s"
            )
        
        await asyncio.sleep(0.01)


async def wait_for_workflow_status(
    manager: WorkflowManager,
    workflow_id: str,
    status: WorkflowStatus,
    timeout: float = 5.0,
) -> WorkflowExecutionContext:
    """
    Wait until workflow reaches a specific status.
    """
    loop = asyncio.get_running_loop()
    start_time = loop.time()

    while True:
        ctx = manager.get_workflow_status(workflow_id)
        if ctx and ctx.status == status:
            return ctx

        # Check history
        for history in manager.workflow_history.values():
            for execution in history:
                if execution.workflow_id == workflow_id and execution.status == status:
                    return execution

        if loop.time() - start_time > timeout:
            raise TimeoutError(
                f"Workflow '{workflow_id}' did not reach status {status} within {timeout:.2f}s"
            )

        await asyncio.sleep(0.01)


async def execute_and_wait(
    manager: WorkflowManager,
    workflow_name: str,
    input_data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> WorkflowExecutionContext:
    """Execute workflow and wait for completion."""
    ctx = await manager.execute_workflow(workflow_name, input_data=input_data, **kwargs)
    return await wait_for_workflow_completion(manager, ctx.workflow_id)


# ------------------------------------------------------------------
# STEP FACTORIES
# ------------------------------------------------------------------
def create_simple_step(name: str, result: Any = None, async_step: bool = False, **kwargs) -> WorkflowStep:
    """Create a simple step that can be sync or async."""
    if async_step:
        async def action(**kwargs):
            return result
    else:
        def action(**kwargs):
            return result
    
    return WorkflowStep(name=name, action=action, **kwargs)


def create_async_step(name: str, result: Any = None, delay: float = 0.01, **kwargs):
    """Create an asynchronous step with optional delay."""
    async def action(**kwargs):
        await asyncio.sleep(delay)
        return result
    return WorkflowStep(name=name, action=action, **kwargs)


def create_failing_step(name: str, message: str = "fail", **kwargs):
    """Create a step that always fails."""
    def action(**kwargs):
        raise ValueError(message)
    return WorkflowStep(name=name, action=action, **kwargs)


def create_conditional_step(
    name: str,
    condition_func: Callable[[Dict[str, Any]], bool],
    result: Any = None,
    **kwargs
):
    """Create a conditional step."""
    def action(**kwargs):
        return result

    return WorkflowStep(
        name=name,
        action=action,
        condition=StepCondition.CONDITIONAL,
        condition_func=condition_func,
        **kwargs
    )


# ------------------------------------------------------------------
# ASSERT HELPERS
# ------------------------------------------------------------------
def assert_workflow_completed(ctx: WorkflowExecutionContext):
    """Assert workflow completed successfully."""
    assert ctx.status == WorkflowStatus.COMPLETED


def assert_workflow_failed(ctx: WorkflowExecutionContext):
    """Assert workflow failed."""
    assert ctx.status == WorkflowStatus.FAILED


def assert_step_executed(ctx: WorkflowExecutionContext, step: str):
    """Assert step executed successfully."""
    assert step in ctx.step_results
    assert ctx.step_results[step].status == StepStatus.COMPLETED


def assert_step_skipped(ctx: WorkflowExecutionContext, step: str):
    """Assert step was skipped."""
    assert step in ctx.step_results
    assert ctx.step_results[step].status == StepStatus.SKIPPED


def assert_step_failed(ctx: WorkflowExecutionContext, step: str):
    """Assert step failed."""
    assert step in ctx.step_results
    assert ctx.step_results[step].status == StepStatus.FAILED


def assert_step_not_executed(ctx: WorkflowExecutionContext, step: str):
    """Assert step was not executed."""
    assert step not in ctx.step_results


# ------------------------------------------------------------------
# MOCK EVENT BUS
# ------------------------------------------------------------------
class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.events = []
        self.subscriptions = {}
    
    def subscribe(self, event_type: str, handler, priority=None, delivery_mode=None):
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        self.subscriptions[event_type].append(handler)
    
    async def publish(self, event_type: str, data: Dict[str, Any], priority=None, delivery_mode=None):
        self.events.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow(),
            "priority": priority,
            "delivery_mode": delivery_mode
        })
        # Execute handlers if any
        if event_type in self.subscriptions:
            for handler in self.subscriptions[event_type]:
                try:
                    # Create mock event object
                    event = Mock()
                    event.type = event_type
                    event.data = data
                    # Execute handler
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    print(f"Error in handler for {event_type}: {e}")
    
    async def start(self):
        pass
    
    async def stop(self):
        pass

def assert_step_executed_with_result(ctx: WorkflowExecutionContext, step: str, expected_result: Any = None):
    """Assert step executed successfully and optionally check result."""
    assert step in ctx.step_results
    assert ctx.step_results[step].status == StepStatus.COMPLETED
    if expected_result is not None:
        assert ctx.step_results[step].result == expected_result

# En test_workflow_manager.py, puedes actualizar el test para verificar mejor:
def test_register_workflow_with_circular_dependencies_fails(self):
    """Test that workflow with circular dependencies fails validation."""
    manager = create_test_workflow_manager()

    step1 = WorkflowStep(name="step1", action=lambda **kwargs: "mock", requires=["step2"])
    step2 = WorkflowStep(name="step2", action=lambda **kwargs: "mock", requires=["step1"])

    with pytest.raises(WorkflowValidationError) as exc_info:
        manager.register_workflow("circular_workflow", [step1, step2])

    error_message = str(exc_info.value).lower()
    # Verificar que el error sea sobre dependencias circulares
    # No sobre "unknown step" 
    assert "circular" in error_message or "cycle" in error_message

# ------------------------------------------------------------------
# TEST SETUP
# ------------------------------------------------------------------
def create_test_workflow_manager() -> WorkflowManager:
    """Create a test workflow manager with mock dependencies."""
    bus = MockEventBus()
    state = Mock()
    state.register_component = Mock()
    state.unregister_component = Mock()

    manager = WorkflowManager(bus, state)
    manager.workflow_definitions.clear()
    manager.workflow_history.clear()
    manager.active_workflows.clear()
    manager.workflow_tasks.clear()
    return manager


def reset_singleton():
    """Reset workflow manager singleton for testing."""
    import sys
    if "src.core.workflow_manager" in sys.modules:
        del sys.modules["src.core.workflow_manager"]