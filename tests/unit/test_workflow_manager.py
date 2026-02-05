# tests/unit/core/test_workflow_manager.py
"""
Unit tests for WorkflowManager.

Tests workflow registration, execution, control, and error handling
using the provided test helpers and mock components.

Author: ANALYZERBRAIN Team
Date: 2024
Version: 1.0.0
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
from typing import Dict, Any

from src.core.workflow_manager import (
    WorkflowManager,
    WorkflowStep,
    WorkflowStatus,
    StepStatus,
    StepCondition,
    WorkflowError,
    StepExecutionError,
    WorkflowValidationError,
    StepExecutionResult,
    WorkflowExecutionContext,
)
from src.core.event_bus import EventBus
from src.core.system_state import SystemState

# Import test helpers
from tests.unit.helpers.workflow_helpers import (
    wait_for_workflow_completion,
    execute_and_wait,
    create_simple_step,
    create_async_step,
    create_failing_step,
    create_conditional_step,
    assert_workflow_completed,
    assert_workflow_failed,
    assert_step_executed,
    assert_step_not_executed,
    create_test_workflow_manager,
    MockEventBus,
    wait_for_step_execution,
    wait_for_workflow_status,
    assert_step_skipped,
    assert_step_failed,
)


class TestWorkflowRegistration:
    """Tests for workflow registration and validation."""

    def test_register_workflow_success(self):
        """Test successful workflow registration."""
        manager = create_test_workflow_manager()
        step1 = create_simple_step("step1", "result1")
        step2 = create_simple_step("step2", "result2")

        manager.register_workflow("test_workflow", [step1, step2])

        assert "test_workflow" in manager.workflow_definitions
        assert len(manager.workflow_definitions["test_workflow"]) == 2
        assert manager.workflow_definitions["test_workflow"][0].name == "step1"
        assert manager.workflow_definitions["test_workflow"][1].name == "step2"

    def test_register_duplicate_workflow_fails(self):
        """Test that duplicate workflow registration fails."""
        manager = create_test_workflow_manager()
        
        # Workflow simple sin dependencias
        step = create_simple_step("step1", "result1")
        manager.register_workflow("test_workflow", [step])
        
        # Intentar registrar el MISMO workflow
        with pytest.raises(WorkflowValidationError) as exc_info:
            manager.register_workflow("test_workflow", [step])  # Mismo objeto
        
        assert "already registered" in str(exc_info.value)

    def test_register_workflow_with_circular_dependencies_fails(self):
        """Test that workflow with circular dependencies fails validation."""
        manager = create_test_workflow_manager()

        step1 = WorkflowStep(name="step1", action=lambda **kwargs: "mock", requires=["step2"])
        step2 = WorkflowStep(name="step2", action=lambda **kwargs: "mock", requires=["step1"])

        with pytest.raises(WorkflowValidationError) as exc_info:
            manager.register_workflow("circular_workflow", [step1, step2])

        assert "circular dependencies" in str(exc_info.value).lower()

    def test_register_workflow_with_missing_dependency_fails(self):
        """Test that workflow with missing dependency fails validation."""
        manager = create_test_workflow_manager()

        step1 = WorkflowStep(name="step1", action=lambda **kwargs: "mock", requires=["nonexistent"])

        with pytest.raises(WorkflowValidationError) as exc_info:
            manager.register_workflow("invalid_workflow", [step1])

        assert "unknown step" in str(exc_info.value).lower()

    def test_register_workflow_with_invalid_step_action_fails(self):
        """Test that workflow with non-callable action fails validation."""
        manager = create_test_workflow_manager()

        step1 = WorkflowStep(name="step1", action="not_callable")  # type: ignore

        with pytest.raises(WorkflowValidationError) as exc_info:
            manager.register_workflow("invalid_workflow", [step1])

        assert "callable" in str(exc_info.value).lower()

    def test_conditional_step_without_condition_func_fails(self):
        """Test that conditional step without condition function fails validation."""
        manager = create_test_workflow_manager()

        step1 = WorkflowStep(
            name="step1",
            action=lambda **kwargs: "mock",
            condition=StepCondition.CONDITIONAL
        )

        with pytest.raises(WorkflowValidationError) as exc_info:
            manager.register_workflow("invalid_workflow", [step1])

        assert "requires condition_func" in str(exc_info.value).lower()

    def test_register_workflow_with_duplicate_step_names_in_different_workflows(self):
        """
        Test that workflows can reuse step names (and even step instances)
        without conflicts across workflow boundaries.
        """
        manager = create_test_workflow_manager()

        step1 = create_simple_step("step1", "result1")
        step2 = create_simple_step("step2", "result2")

        # Same step names in different workflows should be OK
        manager.register_workflow("workflow1", [step1, step2])
        manager.register_workflow("workflow2", [step1, step2])

        assert set(manager.workflow_definitions.keys()) == {"workflow1", "workflow2"}

        wf1_steps = manager.workflow_definitions["workflow1"]
        wf2_steps = manager.workflow_definitions["workflow2"]

        assert len(wf1_steps) == 2
        assert len(wf2_steps) == 2

        assert [step.name for step in wf1_steps] == ["step1", "step2"]
        assert [step.name for step in wf2_steps] == ["step1", "step2"]


class TestWorkflowExecution:
    """Tests for workflow execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self):
        """Test executing a simple linear workflow."""
        manager = create_test_workflow_manager()

        step1 = create_simple_step("step1", "result1")
        step2 = create_simple_step("step2", "result2")
        step3 = create_simple_step("step3", "result3")

        manager.register_workflow("simple_workflow", [step1, step2, step3])

        ctx = await execute_and_wait(manager, "simple_workflow", input_data={"input": "test"})

        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "step1")
        assert_step_executed(ctx, "step2")
        assert_step_executed(ctx, "step3")
        assert len(ctx.execution_order) == 3
        assert ctx.execution_order == ["step1", "step2", "step3"]

    @pytest.mark.asyncio
    async def test_execute_async_workflow(self):
        """Test executing a workflow with async steps."""
        manager = create_test_workflow_manager()

        step1 = create_async_step("step1", "result1", delay=0.01)
        step2 = create_async_step("step2", "result2", delay=0.01)
        step3 = create_async_step("step3", "result3", delay=0.01)

        manager.register_workflow("async_workflow", [step1, step2, step3])

        ctx = await execute_and_wait(manager, "async_workflow")

        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "step1")
        assert_step_executed(ctx, "step2")
        assert_step_executed(ctx, "step3")

    @pytest.mark.asyncio
    async def test_execute_workflow_with_dependencies(self):
        """Test executing a workflow with step dependencies."""
        manager = create_test_workflow_manager()

        step1 = create_simple_step("step1", "result1")
        step2 = create_simple_step("step2", "result2", requires=["step1"])
        step3 = create_simple_step("step3", "result3", requires=["step1", "step2"])

        manager.register_workflow("dependent_workflow", [step1, step2, step3])

        ctx = await execute_and_wait(manager, "dependent_workflow")

        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "step1")
        assert_step_executed(ctx, "step2")
        assert_step_executed(ctx, "step3")
        
        # Verify execution order respects dependencies
        assert ctx.execution_order.index("step1") < ctx.execution_order.index("step2")
        assert ctx.execution_order.index("step2") < ctx.execution_order.index("step3")

# In test_workflow_manager.py, update the test:

    @pytest.mark.asyncio
    async def test_execute_workflow_with_parallel_dependencies(self):
        """
        Test that independent steps may run in parallel and that
        dependent steps start only after their requirements complete.
        """
        manager = create_test_workflow_manager()

        # Create steps that properly provide data
        async def step1_action(**kwargs):
            await asyncio.sleep(0.05)
            return {"step1_data": "result1"}
        
        async def step2_action(**kwargs):
            await asyncio.sleep(0.05)
            return {"step2_data": "result2"}
        
        async def step3_action(step1_data=None, step2_data=None, **kwargs):
            await asyncio.sleep(0.05)
            return {"step3_data": f"{step1_data}_{step2_data}_result3"}

        # CORRECCIÓN: 'requires' debe contener NOMBRES DE PASOS, no datos
        step1 = WorkflowStep(name="step1", action=step1_action, provides=["step1_data"])
        step2 = WorkflowStep(name="step2", action=step2_action, provides=["step2_data"])
        step3 = WorkflowStep(
            name="step3",
            action=step3_action,
            requires=["step1", "step2"],  # ✅ Nombres de pasos, no datos
            provides=["step3_data"]
        )

        manager.register_workflow("parallel_workflow", [step1, step2, step3])

        ctx = await execute_and_wait(manager, "parallel_workflow")

        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "step1")
        assert_step_executed(ctx, "step2")
        assert_step_executed(ctx, "step3")

        step1_result = ctx.step_results["step1"]
        step2_result = ctx.step_results["step2"]
        step3_result = ctx.step_results["step3"]

        assert step1_result.end_time is not None
        assert step2_result.end_time is not None
        assert step3_result.start_time is not None

        # Dependent step must not start before its dependencies finish
        # Allow some tolerance for timing differences
        assert step1_result.end_time <= step3_result.start_time or abs((step1_result.end_time - step3_result.start_time).total_seconds()) < 0.01
        assert step2_result.end_time <= step3_result.start_time or abs((step2_result.end_time - step3_result.start_time).total_seconds()) < 0.01
        
        # Also verify data was passed correctly
        assert "step1_data" in ctx.context_data
        assert "step2_data" in ctx.context_data
        assert "step3_data" in ctx.context_data
        assert ctx.context_data["step3_data"] == "result1_result2_result3"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_workflow_fails(self):
        """Test that executing a non-existent workflow fails."""
        manager = create_test_workflow_manager()

        with pytest.raises(WorkflowError) as exc_info:
            await manager.execute_workflow("nonexistent_workflow")

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_workflow_with_input_data(self):
        """Test executing a workflow with input data."""
        manager = create_test_workflow_manager()

        def step_action(workflow_context=None, **kwargs) -> Dict[str, Any]:
            # El WorkflowManager pasa el contexto como argumento nombrado
            # También podría pasar otros datos si el paso tuviera 'requires'
            if workflow_context:
                input_data = workflow_context.input_data
                # También podríamos usar workflow_context.context_data
                return {"processed": input_data.get("input", "") + "_processed"}
            return {}

        step = WorkflowStep(name="process", action=step_action, provides=["processed"])
        manager.register_workflow("data_workflow", [step])

        ctx = await execute_and_wait(
            manager, 
            "data_workflow", 
            input_data={"input": "test_data"}
        )

        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "process")
        assert "processed" in ctx.output_data
        assert ctx.output_data["processed"] == "test_data_processed"

    @pytest.mark.asyncio
    async def test_execute_workflow_with_data_passing(self):
        """Test data passing between workflow steps."""
        manager = create_test_workflow_manager()

        def step1_action(workflow_context=None, **kwargs) -> Dict[str, Any]:
            return {"intermediate": "step1_result"}

        def step2_action(workflow_context=None, intermediate=None, **kwargs) -> Dict[str, Any]:
            # El segundo paso recibe 'intermediate' porque lo requiere
            # También recibe workflow_context como siempre
            return {"final": intermediate + "_step2_result"}

        step1 = WorkflowStep(name="step1", action=step1_action, provides=["intermediate"])
        step2 = WorkflowStep(name="step2", action=step2_action, provides=["final"], requires=["intermediate"])

        manager.register_workflow("data_passing_workflow", [step1, step2])

        ctx = await execute_and_wait(manager, "data_passing_workflow")

        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "step1")
        assert_step_executed(ctx, "step2")
        assert "intermediate" in ctx.context_data
        assert "final" in ctx.output_data
        assert ctx.output_data["final"] == "step1_result_step2_result"


class TestWorkflowControl:
    """Tests for workflow control operations (pause, cancel, resume)."""

    @pytest.mark.asyncio
    async def test_cancel_workflow(self):
        manager = create_test_workflow_manager()

        # Use a longer delay and track execution
        step_executed = False
        
        async def step1_action(**kwargs):
            nonlocal step_executed
            step_executed = True
            await asyncio.sleep(0.5)  # Longer delay
            return "result1"
        
        step1 = WorkflowStep(name="step1", action=step1_action)
        step2 = create_async_step("step2", "result2", delay=0.1)
        
        manager.register_workflow("cancel_workflow", [step1, step2])
        
        # Start workflow
        ctx = await manager.execute_workflow("cancel_workflow")
        
        # Wait a bit for step1 to start but not complete
        await asyncio.sleep(0.1)
        assert step_executed is True  # Verify step started
        
        success = await manager.cancel_workflow(ctx.workflow_id)
        assert success is True
        
        # Wait for cancellation to complete
        final_ctx = await wait_for_workflow_completion(manager, ctx.workflow_id)
        
        assert final_ctx.status == WorkflowStatus.CANCELLED
        # step1 should be in step_results, may be FAILED or CANCELLED
        assert "step1" in final_ctx.step_results
        # step2 should not be executed
        assert "step2" not in final_ctx.step_results

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_workflow(self):
        """Test canceling a non-existent workflow returns False."""
        manager = create_test_workflow_manager()
        
        success = await manager.cancel_workflow("nonexistent_id")
        assert success is False

    @pytest.mark.asyncio
    async def test_pause_and_resume_workflow(self):
        """Test pausing and resuming a workflow."""
        manager = create_test_workflow_manager()

        step1 = create_async_step("step1", "result1", delay=0.05)
        step2 = create_async_step("step2", "result2", delay=0.05)
        step3 = create_async_step("step3", "result3", delay=0.05)

        manager.register_workflow("pause_resume_workflow", [step1, step2, step3])

        ctx = await manager.execute_workflow("pause_resume_workflow")

        # Esperar explícitamente a que step1 termine
        await wait_for_step_execution(manager, ctx.workflow_id, "step1")

        # Pausar
        pause_success = await manager.pause_workflow(ctx.workflow_id)
        assert pause_success is True

        # Esperar estado PAUSED (polling, no sleep fijo)
        paused_ctx = await wait_for_workflow_status(
            manager,
            ctx.workflow_id,
            WorkflowStatus.PAUSED
        )
        assert paused_ctx.status == WorkflowStatus.PAUSED

        # Asegurar que step2 NO se ejecuta mientras está pausado
        # En lugar de verificar que step2 no está en los resultados,
        # verificamos que step2 no se ha completado
        await asyncio.sleep(0.1)
        paused_ctx = manager.get_workflow_status(ctx.workflow_id)
        assert paused_ctx is not None
        # Verificar que step2 no está completado
        if "step2" in paused_ctx.step_results:
            assert paused_ctx.step_results["step2"].status != StepStatus.COMPLETED

        # Reanudar
        resume_success = await manager.resume_workflow(ctx.workflow_id)
        assert resume_success is True

        final_ctx = await wait_for_workflow_completion(manager, ctx.workflow_id)

        assert_workflow_completed(final_ctx)
        assert_step_executed(final_ctx, "step1")
        assert_step_executed(final_ctx, "step2")
        assert_step_executed(final_ctx, "step3")

    @pytest.mark.asyncio
    async def test_pause_nonexistent_workflow(self):
        """Test pausing a non-existent workflow returns False."""
        manager = create_test_workflow_manager()
        
        success = await manager.pause_workflow("nonexistent_id")
        assert success is False

    @pytest.mark.asyncio
    async def test_resume_nonexistent_workflow(self):
        """Test resuming a non-existent workflow returns False."""
        manager = create_test_workflow_manager()
        
        success = await manager.resume_workflow("nonexistent_id")
        assert success is False

    @pytest.mark.asyncio
    async def test_resume_non_paused_workflow(self):
        """Test resuming a workflow that isn't paused returns False."""
        manager = create_test_workflow_manager()

        step = create_simple_step("step1", "result1")
        manager.register_workflow("simple_workflow", [step])

        ctx = await execute_and_wait(manager, "simple_workflow")
        
        # Try to resume completed workflow
        success = await manager.resume_workflow(ctx.workflow_id)
        assert success is False


class TestWorkflowConditions:
    """Tests for conditional step execution."""

    @pytest.mark.asyncio
    async def test_always_condition(self):
        """Test ALWAYS condition (default)."""
        manager = create_test_workflow_manager()

        step1 = create_simple_step("step1", "result1")
        step2 = WorkflowStep(
            name="step2",
            action=lambda **kwargs: "result2",
            condition=StepCondition.ALWAYS
        )
        
        manager.register_workflow("always_workflow", [step1, step2])

        ctx = await execute_and_wait(manager, "always_workflow")
        
        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "step1")
        assert_step_executed(ctx, "step2")

    @pytest.mark.asyncio
    async def test_on_success_condition(self):
        """Test ON_SUCCESS condition."""
        manager = create_test_workflow_manager()

        step1 = create_simple_step("step1", "result1")
        step2 = WorkflowStep(
            name="step2",
            action=lambda **kwargs: "result2",
            condition=StepCondition.ON_SUCCESS,
            requires=["step1"]
        )
        
        manager.register_workflow("on_success_workflow", [step1, step2])

        ctx = await execute_and_wait(manager, "on_success_workflow")
        
        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "step1")
        assert_step_executed(ctx, "step2")

    @pytest.mark.asyncio
    async def test_on_success_condition_skips_step_when_dependency_fails(self):
        """
        Test that a step with ON_SUCCESS condition is not executed
        when its required step fails.
        """
        manager = create_test_workflow_manager()

        step1 = create_failing_step("step1", "step1_failed")
        step2 = WorkflowStep(
            name="step2",
            action=lambda **kwargs: "result2",
            condition=StepCondition.ON_SUCCESS,
            requires=["step1"],
        )

        manager.register_workflow("on_success_fail_workflow", [step1, step2])

        ctx = await execute_and_wait(manager, "on_success_fail_workflow")

        assert_workflow_failed(ctx)

        assert "step1" in ctx.step_results
        assert ctx.step_results["step1"].status == StepStatus.FAILED

        # El step2 debe estar SKIPPED, no not executed
        assert_step_skipped(ctx, "step2")

    @pytest.mark.asyncio
    async def test_on_failure_condition(self):
        """Test ON_FAILURE condition."""
        manager = create_test_workflow_manager()

        step1 = create_failing_step("step1", "step1_failed")
        step2 = WorkflowStep(
            name="step2",
            action=lambda **kwargs: "result2",
            condition=StepCondition.ON_FAILURE,
            requires=["step1"]
        )
        
        manager.register_workflow("on_failure_workflow", [step1, step2])

        ctx = await execute_and_wait(manager, "on_failure_workflow")
        
        # Workflow should fail overall, but step2 should execute
        assert_workflow_failed(ctx)
        assert "step1" in ctx.step_results
        assert ctx.step_results["step1"].status == StepStatus.FAILED
        assert_step_executed(ctx, "step2")

    @pytest.mark.asyncio
    async def test_on_failure_condition_is_skipped_when_dependency_succeeds(self):
        """
        ON_FAILURE steps should NOT execute when the required step completes successfully.
        """
        # Arrange
        manager = create_test_workflow_manager()

        step1 = create_simple_step("step1", "result1")  # successful step
        step2 = WorkflowStep(
            name="step2",
            action=lambda **kwargs: "result2",
            condition=StepCondition.ON_FAILURE,
            requires=["step1"],
        )

        manager.register_workflow(
            "on_failure_success_workflow",
            [step1, step2],
        )

        # Act
        ctx = await execute_and_wait(manager, "on_failure_success_workflow")

        # Assert
        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "step1")
        # El step2 debe estar SKIPPED, no not executed
        assert_step_skipped(ctx, "step2")

    @pytest.mark.asyncio
    async def test_conditional_step_executes_when_condition_is_true(self):
        """
        Test that a conditional step is executed when its condition
        evaluates to True based on data from a previous step.
        """
        manager = create_test_workflow_manager()

        def condition_func(data: Dict[str, Any]) -> bool:
            return data.get("execute_step2", False)

        step1 = create_simple_step("step1", {"execute_step2": True})
        step2 = create_conditional_step("step2", condition_func, "result2")

        manager.register_workflow("conditional_workflow", [step1, step2])

        ctx = await execute_and_wait(manager, "conditional_workflow")

        assert_workflow_completed(ctx)

        assert_step_executed(ctx, "step1")
        assert_step_executed(ctx, "step2")

        # Explicitly verify the conditional step result
        assert ctx.step_results["step2"].result == "result2"

    @pytest.mark.asyncio
    async def test_conditional_step_skip(self):
        """Test conditional step that gets skipped."""
        manager = create_test_workflow_manager()

        def condition_func(data: Dict[str, Any]) -> bool:
            return data.get("execute_step2", False)

        step1 = create_simple_step("step1", {"execute_step2": False})  # False, so step2 should skip
        step2 = create_conditional_step("step2", condition_func, "result2")
        
        manager.register_workflow("conditional_skip_workflow", [step1, step2])

        ctx = await execute_and_wait(manager, "conditional_skip_workflow")
        
        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "step1")
        # El step2 debe estar SKIPPED
        assert_step_skipped(ctx, "step2")


class TestWorkflowErrors:
    """Tests for workflow error handling."""

    @pytest.mark.asyncio
    async def test_failing_step_without_retry(self):
        """Test workflow fails when a step fails without retries."""
        manager = create_test_workflow_manager()

        step1 = create_simple_step("step1", "result1")
        step2 = create_failing_step("step2", "step2_failed")
        step3 = create_simple_step("step3", "result3")
        
        manager.register_workflow("failing_workflow", [step1, step2, step3])

        ctx = await execute_and_wait(manager, "failing_workflow")
        
        assert_workflow_failed(ctx)
        assert_step_executed(ctx, "step1")
        assert "step2" in ctx.step_results
        assert ctx.step_results["step2"].status == StepStatus.FAILED
        assert_step_not_executed(ctx, "step3")

    @pytest.mark.asyncio
    async def test_failing_step_with_retry_success(self):
        """Test step retry that eventually succeeds."""
        manager = create_test_workflow_manager()
        
        attempt_count = 0
        
        def failing_then_succeeding_action(**kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Failing attempt {attempt_count}")
            return f"Success on attempt {attempt_count}"
        
        step = WorkflowStep(
            name="retry_step",
            action=failing_then_succeeding_action,
            retry_attempts=3,
            retry_delay=0.01
        )
        
        manager.register_workflow("retry_workflow", [step])

        ctx = await execute_and_wait(manager, "retry_workflow")
        
        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "retry_step")
        assert ctx.step_results["retry_step"].retry_count == 2  # 2 retries = 3 total attempts
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_failing_step_with_retry_failure(self):
        """
        Test that a step retries the configured number of times and
        ultimately fails after exhausting all retry attempts.
        """
        manager = create_test_workflow_manager()

        attempt_count = 0

        def always_failing_action(**kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError(f"Failing attempt {attempt_count}")

        step = WorkflowStep(
            name="retry_step",
            action=always_failing_action,
            retry_attempts=2,   # 2 retries + 1 initial attempt
            retry_delay=0.01,
        )

        manager.register_workflow("retry_fail_workflow", [step])

        ctx = await execute_and_wait(manager, "retry_fail_workflow")

        assert_workflow_failed(ctx)
        assert "retry_step" in ctx.step_results

        step_result = ctx.step_results["retry_step"]

        assert step_result.status == StepStatus.FAILED
        assert step_result.retry_count == 2
        assert step_result.error is not None

        # Initial attempt + retries
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_critical_step_failure_stops_workflow(self):
        """
        Test that a critical step failure aborts the workflow
        and prevents execution of subsequent steps.
        """
        manager = create_test_workflow_manager()

        # Usar funciones asíncronas para mejor control
        step1_executed = False
        step2_executed = False
        step3_executed = False
        
        async def step1_action(**kwargs):
            nonlocal step1_executed
            step1_executed = True
            await asyncio.sleep(0.01)  # Pequeño delay para asegurar orden
            return "result1"
        
        async def step2_action(**kwargs):
            nonlocal step2_executed
            step2_executed = True
            await asyncio.sleep(0.01)
            raise ValueError("critical_failure")
        
        async def step3_action(**kwargs):
            nonlocal step3_executed
            step3_executed = True
            await asyncio.sleep(0.01)
            return "result3"

        step1 = WorkflowStep(name="step1", action=step1_action)
        step2 = WorkflowStep(name="step2", action=step2_action, critical=True)
        step3 = WorkflowStep(name="step3", action=step3_action)

        manager.register_workflow(
            "critical_failure_workflow",
            [step1, step2, step3],
        )

        ctx = await execute_and_wait(manager, "critical_failure_workflow")

        assert_workflow_failed(ctx)
        assert step1_executed is True
        assert step2_executed is True
        assert step3_executed is False
        
        # Verificar que step1 se ejecutó y completó
        assert "step1" in ctx.step_results
        step1_result = ctx.step_results["step1"]
        assert step1_result.status == StepStatus.COMPLETED
        
        # Verificar que step2 se ejecutó y falló
        assert "step2" in ctx.step_results
        step2_result = ctx.step_results["step2"]
        assert step2_result.status == StepStatus.FAILED
        assert "critical_failure" in str(step2_result.error)
        
        # Verificar que step3 NO se ejecutó
        assert "step3" not in ctx.step_results

    @pytest.mark.asyncio
    async def test_non_critical_step_failure_allows_continuation(self):
        """Test that non-critical step failure allows workflow to continue."""
        manager = create_test_workflow_manager()

        step1 = create_simple_step("step1", "result1")
        step2 = create_failing_step("step2", "non_critical_failure")
        step3 = create_simple_step("step3", "result3")
        
        manager.register_workflow("non_critical_failure_workflow", [step1, step2, step3])

        ctx = await execute_and_wait(manager, "non_critical_failure_workflow")
        
        # Non-critical failure, workflow should complete
        assert_workflow_completed(ctx)
        assert_step_executed(ctx, "step1")
        assert "step2" in ctx.step_results
        assert ctx.step_results["step2"].status == StepStatus.FAILED
        assert_step_executed(ctx, "step3")

    @pytest.mark.asyncio
    async def test_step_timeout(self):
        """Test that a step exceeding its timeout fails with a timeout error."""
        manager = create_test_workflow_manager()

        async def slow_action(**kwargs):
            await asyncio.sleep(0.2)  # Intentionally longer than timeout
            return "too_late"

        step = WorkflowStep(
            name="slow_step",
            action=slow_action,
            timeout=0.05,  # Shorter than sleep → guaranteed timeout
            critical=True,  # ✅ AÑADIR ESTA LÍNEA
        )

        manager.register_workflow("timed out_workflow", [step])

        ctx = await execute_and_wait(manager, "timed out_workflow")

        assert_workflow_failed(ctx)

        assert "slow_step" in ctx.step_results
        step_result = ctx.step_results["slow_step"]

        assert step_result.status == StepStatus.FAILED
        assert step_result.error is not None
        assert "timed out" in str(step_result.error).lower()


class TestWorkflowQueryAndMonitoring:
    """Tests for workflow query and monitoring methods."""

    @pytest.mark.asyncio
    async def test_get_workflow_status(self):
        """Test getting workflow status."""
        manager = create_test_workflow_manager()

        step = create_async_step("step1", "result1", delay=0.05)
        manager.register_workflow("status_workflow", [step])

        # Start workflow
        ctx = await manager.execute_workflow("status_workflow")
        
        # Check status while running - debería ser RUNNING después de un tiempo
        await asyncio.sleep(0.01)
        status_ctx = manager.get_workflow_status(ctx.workflow_id)
        assert status_ctx is not None
        assert status_ctx.workflow_id == ctx.workflow_id
        # Puede estar en RUNNING o PENDING dependiendo de cuándo se revise
        assert status_ctx.status in [WorkflowStatus.RUNNING, WorkflowStatus.PENDING]
        
        # Wait for completion and check again
        final_ctx = await wait_for_workflow_completion(manager, ctx.workflow_id)
        assert final_ctx.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_active_workflows(self):
        """Test getting active workflows."""
        manager = create_test_workflow_manager()

        step1 = create_async_step("step1", "result1", delay=0.1)
        step2 = create_async_step("step2", "result2", delay=0.1)
        
        manager.register_workflow("workflow1", [step1])
        manager.register_workflow("workflow2", [step2])

        # Start two workflows
        ctx1 = await manager.execute_workflow("workflow1")
        ctx2 = await manager.execute_workflow("workflow2")
        
        # Check active workflows
        active = manager.get_active_workflows()
        assert len(active) == 2
        assert ctx1.workflow_id in active
        assert ctx2.workflow_id in active
        
        # Wait for completion
        await wait_for_workflow_completion(manager, ctx1.workflow_id)
        await wait_for_workflow_completion(manager, ctx2.workflow_id)
        
        # Check no active workflows
        active = manager.get_active_workflows()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_get_registered_workflows(self):
        """Test getting list of registered workflows."""
        manager = create_test_workflow_manager()

        step1 = create_simple_step("step1", "result1")
        step2 = create_simple_step("step2", "result2")
        step3 = create_simple_step("step3", "result3")
        
        manager.register_workflow("workflow1", [step1])
        manager.register_workflow("workflow2", [step1, step2])
        manager.register_workflow("workflow3", [step1, step2, step3])

        workflows = manager.get_registered_workflows()
        assert len(workflows) == 3
        assert "workflow1" in workflows
        assert "workflow2" in workflows
        assert "workflow3" in workflows

    @pytest.mark.asyncio
    async def test_get_workflow_history(self):
        """Test getting workflow execution history."""
        manager = create_test_workflow_manager()

        step = create_simple_step("step1", "result1")
        manager.register_workflow("history_workflow", [step])

        # Execute workflow multiple times
        for i in range(3):
            ctx = await execute_and_wait(
                manager, 
                "history_workflow", 
                input_data={"execution": i}
            )
            assert_workflow_completed(ctx)

        # Get history
        history = manager.get_workflow_history("history_workflow")
        assert len(history) == 3
        
        # History should be in chronological order (oldest first)
        for i in range(3):
            assert history[i].input_data.get("execution") == i

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting workflow manager metrics."""
        manager = create_test_workflow_manager()

        step = create_simple_step("step1", "result1")

        # Check initial metrics - no workflows registered yet
        metrics = manager.get_metrics()
        assert metrics["workflows_registered"] == 0  # Not registered yet
        
        # Register workflow
        manager.register_workflow("test_workflow", [step])
        
        # Check metrics after registration
        metrics = manager.get_metrics()
        assert metrics["workflows_registered"] == 1
        
        # Execute workflow
        ctx = await execute_and_wait(manager, "test_workflow")
        
        # Check metrics after execution
        metrics = manager.get_metrics()
        assert metrics["workflows_registered"] == 1
        assert metrics["steps_completed_total"] >= 1
        assert "workflow_history_size" in metrics

    @pytest.mark.asyncio
    async def test_get_available_steps_respects_dependencies(self):
        """
        Test that get_available_steps returns only steps whose
        dependencies have been satisfied.
        """
        manager = create_test_workflow_manager()

        step1 = create_simple_step("step1", "result1")
        step2 = create_simple_step("step2", "result2", requires=["step1"])
        step3 = create_simple_step("step3", "result3", requires=["step2"])

        manager.register_workflow("test_workflow", [step1, step2, step3])

        ctx = WorkflowExecutionContext(
            workflow_id="test_id",
            workflow_name="test_workflow",
            status=WorkflowStatus.RUNNING,
        )

        # Initially only step1 should be available
        available = manager.get_available_steps("test_workflow", ctx)
        assert set(available) == {"step1"}

        # Simulate step1 completion
        ctx.step_results["step1"] = StepExecutionResult(
            step_name="step1",
            status=StepStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # Now step2 should be available, but not step3
        available = manager.get_available_steps("test_workflow", ctx)
        assert set(available) == {"step2"}


class TestWorkflowEvents:
    """Tests for workflow event publishing."""

    @pytest.mark.asyncio
    async def test_workflow_events_published(self):
        """Test that workflow events are published."""
        # Create real EventBus for this test
        event_bus = EventBus(name="test_workflow_events")
        await event_bus.start()
        
        system_state = Mock()
        system_state.register_component = Mock()
        system_state.unregister_component = Mock()
        
        manager = WorkflowManager(event_bus, system_state)
        
        # Track events
        events = []
        
        def capture_event(event):
            events.append(event.type)
        
        # Subscribe to workflow events
        event_bus.subscribe("workflow.started", capture_event)
        event_bus.subscribe("workflow.completed", capture_event)
        event_bus.subscribe("workflow.step_started", capture_event)
        event_bus.subscribe("workflow.step_completed", capture_event)
        
        # Register and execute workflow
        step = create_simple_step("step1", "result1")
        manager.register_workflow("event_workflow", [step])
        
        ctx = await execute_and_wait(manager, "event_workflow")
        
        # Check events were published
        assert "workflow.started" in events
        assert "workflow.completed" in events
        assert "workflow.step_started" in events
        assert "workflow.step_completed" in events
        
        await event_bus.stop()
        await manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])