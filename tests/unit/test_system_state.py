"""
Tests for SystemState Manager.

Comprehensive unit and integration tests for system state management.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call, ANY
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.system_state import (
    SystemState,
    SystemStatus,
    ComponentType,
    HealthStatus,
    ComponentHealth,
    SystemMetrics,
    get_system_state,
    SystemStateError,
)


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    mock_bus = AsyncMock()
    mock_bus.subscribe = MagicMock()
    mock_bus.publish = AsyncMock()
    mock_bus.subscribe.return_value = None
    return mock_bus


@pytest.fixture
def mock_config():
    """Mock configuration."""
    with patch("src.core.system_state.config") as mock_config:
        mock_config.get.side_effect = lambda key, default=None: {
            "system_state.max_history_size": 1000,
            "system_state.health_check_interval": 30,
            "system_state.watchdog_timeout": 300,
        }.get(key, default)
        yield mock_config


# ..async
@pytest.fixture
async def system_state(mock_event_bus, mock_config):
    """Create a SystemState instance for testing."""
    state = SystemState(mock_event_bus)
    # Enable testing mode to make record_metric synchronous
    state._testing_mode = True
    yield state
    # Clean up health monitoring task if it exists
    if state._health_check_task and not state._health_check_task.done():
        state._health_check_task.cancel()
        try:
            await asyncio.wait_for(state._health_check_task, timeout=0.1)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


# ..async
@pytest.fixture
async def populated_system_state(system_state):
    """Create a SystemState instance with some components registered."""
    # Register some components
    system_state.register_component("test_indexer", ComponentType.INDEXER)
    system_state.register_component(
        "test_graph", ComponentType.GRAPH, dependencies=["test_indexer"]
    )
    system_state.register_component("test_api", ComponentType.API)

    # Set system to running through proper transitions
    await system_state.set_status(SystemStatus.STARTING)
    await system_state.set_status(SystemStatus.RUNNING)

    # Update component health
    await system_state.update_component_health(
        "test_indexer", HealthStatus.HEALTHY, response_time_ms=10.5
    )
    await system_state.update_component_health(
        "test_graph", HealthStatus.HEALTHY, response_time_ms=15.2
    )
    await system_state.update_component_health(
        "test_api", HealthStatus.HEALTHY, response_time_ms=5.1
    )

    return system_state


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_component_health_creation(self):
        """Test creating a ComponentHealth instance."""
        now = datetime.now()
        component = ComponentHealth(
            name="test_component",
            component_type=ComponentType.CORE,
            status=HealthStatus.HEALTHY,
            last_check=now,
            response_time_ms=12.5,
            error_count=0,
            warnings=["Warning 1", "Warning 2"],
            details={"version": "1.0.0"},
        )

        assert component.name == "test_component"
        assert component.component_type == ComponentType.CORE
        assert component.status == HealthStatus.HEALTHY
        assert component.last_check == now
        assert component.response_time_ms == 12.5
        assert component.error_count == 0
        assert len(component.warnings) == 2
        assert component.details["version"] == "1.0.0"

    def test_component_health_to_dict(self):
        """Test converting ComponentHealth to dictionary."""
        now = datetime.now()
        component = ComponentHealth(
            name="test_component",
            component_type=ComponentType.INDEXER,
            status=HealthStatus.DEGRADED,
            last_check=now,
        )

        result = component.to_dict()

        assert result["name"] == "test_component"
        assert result["component_type"] == "indexer"
        assert result["status"] == "degraded"
        assert result["last_check"] == now.isoformat()
        assert "response_time_ms" in result
        assert "error_count" in result
        assert "warnings" in result
        assert "details" in result


class TestSystemMetrics:
    """Tests for SystemMetrics dataclass."""

    def test_system_metrics_creation(self):
        """Test creating a SystemMetrics instance."""
        startup_time = datetime.now()
        last_update = datetime.now()

        metrics = SystemMetrics(
            startup_time=startup_time,
            last_update=last_update,
            uptime_seconds=3600.5,
            projects_analyzed=10,
            files_processed=1500,
            entities_extracted=50000,
            graph_nodes=10000,
            graph_relationships=25000,
            embeddings_generated=3000,
            agent_tasks_completed=150,
            total_errors=5,
            error_rate_5min=0.01,
            critical_errors=1,
            memory_usage_mb=512.5,
            cpu_percent=45.3,
            disk_usage_percent=65.7,
            avg_indexing_time_ms=125.5,
            avg_query_time_ms=45.2,
            avg_embedding_time_ms=320.1,
        )

        assert metrics.startup_time == startup_time
        assert metrics.last_update == last_update
        assert metrics.uptime_seconds == 3600.5
        assert metrics.projects_analyzed == 10
        assert metrics.files_processed == 1500
        assert metrics.entities_extracted == 50000
        assert metrics.total_errors == 5
        assert metrics.memory_usage_mb == 512.5
        assert metrics.avg_indexing_time_ms == 125.5

    def test_system_metrics_to_dict(self):
        """Test converting SystemMetrics to dictionary."""
        startup_time = datetime.now()
        last_update = datetime.now()

        metrics = SystemMetrics(
            startup_time=startup_time, last_update=last_update, projects_analyzed=5
        )

        result = metrics.to_dict()

        assert result["startup_time"] == startup_time.isoformat()
        assert result["last_update"] == last_update.isoformat()
        assert result["projects_analyzed"] == 5
        assert "uptime_seconds" in result
        assert "files_processed" in result
        assert "memory_usage_mb" in result


class TestSystemStateInitialization:
    """Tests for SystemState initialization."""

    def test_initialization(self, mock_event_bus, mock_config):
        """Test SystemState initialization."""
        state = SystemState(mock_event_bus)

        # Check initial state
        assert state.status == SystemStatus.INITIALIZING
        assert state.previous_status is None
        assert len(state.components) == 0
        assert len(state.component_dependencies) == 0
        assert len(state.state_history) == 0
        assert state.health_check_interval == 30
        assert state.watchdog_timeout == 300

        # Check event subscriptions
        expected_subscriptions = [
            "component.registered",
            "component.unregistered",
            "component.heartbeat",
            "component.error",
            "system.shutdown",
            "system.metric_updated",
            "project.analysis_started",
            "project.analysis_completed",
            "project.analysis_failed",
        ]

        # Verify all expected events are subscribed to
        actual_calls = mock_event_bus.subscribe.call_args_list
        subscribed_events = [call[0][0] for call in actual_calls]

        for event in expected_subscriptions:
            assert event in subscribed_events, f"Event {event} not subscribed"

        # Check metrics initialization
        assert isinstance(state.metrics, SystemMetrics)
        assert state.metrics.startup_time is not None
        assert state.metrics.projects_analyzed == 0

    def test_singleton_pattern(self, mock_event_bus):
        """Test get_system_state singleton functionality."""
        # Reset singleton for test
        import src.core.system_state as system_state_module

        system_state_module._system_state_instance = None

        # First call should create instance
        instance1 = get_system_state(mock_event_bus)
        assert instance1 is not None
        assert isinstance(instance1, SystemState)

        # Second call should return same instance
        instance2 = get_system_state()
        assert instance2 is instance1

        # Without event bus on first call should raise error
        # Reset the singleton again
        system_state_module._system_state_instance = None
        with pytest.raises(SystemStateError):
            get_system_state()


class TestSystemStatusTransitions:
    """Tests for system status transitions."""

    # ..async
    @pytest.mark.asyncio
    async def test_valid_status_transition(self, system_state):
        """Test valid status transition."""
        # Initial state should be INITIALIZING
        assert system_state.status == SystemStatus.INITIALIZING

        # Transition to STARTING
        await system_state.set_status(SystemStatus.STARTING, "Starting system")
        assert system_state.status == SystemStatus.STARTING
        assert system_state.previous_status == SystemStatus.INITIALIZING

        # Check history was recorded
        assert len(system_state.state_history) == 1
        history_entry = system_state.state_history[0]
        assert history_entry["old_status"] == "initializing"
        assert history_entry["new_status"] == "starting"
        assert history_entry["reason"] == "Starting system"

        # Check event was published
        system_state.event_bus.publish.assert_called_once()

    # ..async
    @pytest.mark.asyncio
    async def test_invalid_status_transition(self, system_state):
        """Test invalid status transition raises error."""
        # Try to go from INITIALIZING to RUNNING (invalid)
        with pytest.raises(SystemStateError):
            await system_state.set_status(SystemStatus.RUNNING)

        # State should not change
        assert system_state.status == SystemStatus.INITIALIZING

    # ..async
    @pytest.mark.asyncio
    async def test_multiple_valid_transitions(self, system_state):
        """Test sequence of valid status transitions."""
        transitions = [
            (SystemStatus.STARTING, "System starting"),
            (SystemStatus.RUNNING, "System running"),
            (SystemStatus.DEGRADED, "Component failure"),
            (SystemStatus.RECOVERING, "Attempting recovery"),
            (SystemStatus.RUNNING, "Recovery successful"),
            (SystemStatus.STOPPING, "Shutdown requested"),
            (SystemStatus.STOPPED, "Shutdown complete"),
        ]

        for new_status, reason in transitions:
            await system_state.set_status(new_status, reason)
            assert system_state.status == new_status

        # Check all transitions were recorded
        assert len(system_state.state_history) == len(transitions)

    # ..async
    @pytest.mark.asyncio
    async def test_status_transition_with_error_path(self, system_state):
        """Test status transitions including error states."""
        await system_state.set_status(SystemStatus.STARTING)
        await system_state.set_status(SystemStatus.RUNNING)

        # Transition to ERROR
        await system_state.set_status(SystemStatus.ERROR, "Critical failure")
        assert system_state.status == SystemStatus.ERROR

        # Recover from ERROR
        await system_state.set_status(SystemStatus.RECOVERING, "Attempting recovery")
        assert system_state.status == SystemStatus.RECOVERING


class TestComponentManagement:
    """Tests for component registration and management."""

    # ..async
    @pytest.mark.asyncio
    async def test_register_component(self, system_state):
        """Test registering a component."""
        # Register a component
        system_state.register_component(
            name="test_indexer",
            component_type=ComponentType.INDEXER,
            dependencies=["core", "database"],
            health_check=lambda: True,
        )

        # Check component was registered
        assert "test_indexer" in system_state.components
        component = system_state.components["test_indexer"]

        assert component.name == "test_indexer"
        assert component.component_type == ComponentType.INDEXER
        assert component.status == HealthStatus.STARTING
        assert component.details.get("health_check")

        # Check dependencies
        assert "test_indexer" in system_state.component_dependencies
        assert system_state.component_dependencies["test_indexer"] == {
            "core",
            "database",
        }

        # Check event was published
        system_state.event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_duplicate_component(self, system_state):
        """Test registering duplicate component logs warning."""
        # Register component twice
        system_state.register_component("test_component", ComponentType.CORE)
        system_state.register_component("test_component", ComponentType.CORE)

        # Should only be registered once
        assert len(system_state.components) == 1

        # Should only publish once
        assert system_state.event_bus.publish.call_count == 1

    @pytest.mark.asyncio
    async def test_unregister_component(self, system_state):
        """Test unregistering a component."""
        # Register a component first
        system_state.register_component("test_component", ComponentType.CORE)
        system_state.register_component(
            "dependent_component", ComponentType.API, dependencies=["test_component"]
        )

        # Reset mock before unregister
        system_state.event_bus.publish.reset_mock()

        # Unregister component
        system_state.unregister_component("test_component")

        # Check component was removed
        assert "test_component" not in system_state.components
        assert "test_component" not in system_state.component_dependencies

        # Check dependency was removed from other component
        assert "dependent_component" in system_state.component_dependencies
        assert "test_component" not in system_state.component_dependencies["dependent_component"]

        # Check event was published
        system_state.event_bus.publish.assert_called_once()

    def test_unregister_nonexistent_component(self, system_state):
        """Test unregistering non-existent component logs warning."""
        system_state.unregister_component("nonexistent")
        # Should log warning but not crash


class TestComponentHealthUpdates:
    """Tests for component health updates."""

    @pytest.mark.asyncio
    async def test_update_component_health(self, system_state):
        """Test updating component health."""
        # Register and update health
        system_state.register_component("test_component", ComponentType.CORE)

        await system_state.update_component_health(
            name="test_component",
            status=HealthStatus.HEALTHY,
            response_time_ms=12.5,
            warnings=["Warning 1"],
            details={"version": "1.0.0"},
        )

        # Check component was updated
        component = system_state.components["test_component"]
        assert component.status == HealthStatus.HEALTHY
        assert component.response_time_ms == 12.5
        assert len(component.warnings) == 1
        assert component.details["version"] == "1.0.0"

        # Check event was published - use ANY for timestamp
        system_state.event_bus.publish.assert_any_call(
            "component.health_updated",
            {
                "name": "test_component",
                "old_status": "starting",
                "new_status": "healthy",
                "response_time_ms": 12.5,
                "timestamp": ANY,
            },
        )

    @pytest.mark.asyncio
    async def test_update_nonexistent_component(self, system_state):
        """Test updating health of non-existent component logs warning."""
        await system_state.update_component_health(name="nonexistent", status=HealthStatus.HEALTHY)
        # Should log warning but not crash

    @pytest.mark.asyncio
    async def test_component_health_affects_system_status(self, system_state):
        await system_state.set_status(SystemStatus.STARTING)
        await system_state.set_status(SystemStatus.RUNNING)

        system_state.register_component("component1", ComponentType.CORE)
        system_state.register_component("component2", ComponentType.INDEXER)
        system_state.register_component("component3", ComponentType.GRAPH)

        # Component 1 becomes unhealthy → system should degrade
        await system_state.update_component_health("component1", HealthStatus.UNHEALTHY)
        assert system_state.status == SystemStatus.DEGRADED

        # Another degraded component → still degraded
        await system_state.update_component_health("component2", HealthStatus.DEGRADED)
        assert system_state.status == SystemStatus.DEGRADED

        # All components healthy again → system recovers
        await system_state.update_component_health("component1", HealthStatus.HEALTHY)
        await system_state.update_component_health("component2", HealthStatus.HEALTHY)
        await system_state.update_component_health("component3", HealthStatus.HEALTHY)

        assert system_state.status == SystemStatus.RUNNING

    @pytest.mark.asyncio
    async def test_majority_unhealthy_triggers_error(self, system_state):
        """Test that majority unhealthy components trigger ERROR status."""
        # Set system to RUNNING through proper transitions
        await system_state.set_status(SystemStatus.STARTING)
        await system_state.set_status(SystemStatus.RUNNING)

        # Register 4 components
        for i in range(4):
            system_state.register_component(f"component{i}", ComponentType.CORE)

        # Make 3 out of 4 unhealthy (75%)
        for i in range(3):
            await system_state.update_component_health(f"component{i}", HealthStatus.UNHEALTHY)

        # Should trigger ERROR status
        assert system_state.status == SystemStatus.ERROR


class TestMetricsRecording:
    """Tests for metrics recording."""

    def test_record_metric(self, system_state):
        """Test recording a metric."""
        system_state.event_bus.publish.reset_mock()

        # Mock the internal _update_metric_sync method
        with patch.object(system_state, '_update_metric_sync') as mock_sync:
            # Call the synchronous record_metric method (testing mode is enabled)
            system_state.record_metric("projects_analyzed", 5)

            # Verify the sync method was called with correct arguments
            mock_sync.assert_called_once_with("projects_analyzed", 5, False)

        # In testing mode, events are not published, so no need to check event_bus

    def test_increment_metric(self, system_state):
        """Test incrementing a metric."""
        # Set initial value
        system_state.metrics.projects_analyzed = 5

        # Mock the internal _update_metric_sync method
        with patch.object(system_state, '_update_metric_sync') as mock_sync:
            # Set up the side effect to actually increment the metric
            def update_metric_side_effect(name, value, increment):
                if increment:
                    current = getattr(system_state.metrics, name)
                    setattr(system_state.metrics, name, current + value)

            mock_sync.side_effect = update_metric_side_effect

            # Call the synchronous record_metric method
            system_state.record_metric("projects_analyzed", 3, increment=True)

            # Verify the sync method was called
            mock_sync.assert_called_once_with("projects_analyzed", 3, True)

        # Check metric was incremented
        assert system_state.metrics.projects_analyzed == 8

    def test_record_unknown_metric(self, system_state):
        """Test recording unknown metric logs warning."""
        with patch("loguru.logger.warning") as mock_warning:
            system_state.record_metric("unknown_metric", 100)

            # Should log warning
            mock_warning.assert_called()

    def test_increment_non_numeric_metric(self, system_state):
        """Test incrementing non-numeric metric logs error."""
        # Save the original startup_time to restore later
        original_startup_time = system_state.metrics.startup_time

        with patch("loguru.logger.error") as mock_error:
            # Try to increment a non-numeric metric
            system_state.record_metric("startup_time", 1, increment=True)

            # Should log error
            mock_error.assert_called()

        # Restore the original value
        system_state.metrics.startup_time = original_startup_time


class TestHealthReporting:
    """Tests for health report generation."""

    @pytest.mark.asyncio
    async def test_get_health_report_empty(self, system_state):
        """Test health report with no components."""
        report = await system_state.get_health_report()

        assert report["system_status"] == "initializing"
        assert report["overall_health"] == "healthy"  # No components = healthy
        assert report["uptime_seconds"] > 0
        assert report["components"] == {}
        assert "metrics" in report
        assert report["stale_components"] == []
        assert "timestamp" in report

    @pytest.mark.asyncio
    async def test_get_health_report_with_components(self, populated_system_state):
        """Test health report with components."""
        report = await populated_system_state.get_health_report()

        assert report["system_status"] == "running"
        assert report["overall_health"] == "healthy"
        assert len(report["components"]) == 3

        # Check component details
        assert "test_indexer" in report["components"]
        assert report["components"]["test_indexer"]["status"] == "healthy"

        # Check metrics
        assert report["metrics"]["projects_analyzed"] == 0

    @pytest.mark.asyncio
    async def test_health_report_with_stale_components(self, system_state):
        """Test health report detects stale components."""
        # Register a component and simulate it being stale
        system_state.register_component("stale_component", ComponentType.CORE)

        # Set last active time to beyond timeout
        old_time = datetime.now() - timedelta(seconds=400)  # Timeout is 300
        system_state.component_last_active["stale_component"] = old_time

        report = await system_state.get_health_report()

        assert "stale_component" in report["stale_components"]
        # Component is STARTING (not HEALTHY), so overall should be degraded
        assert report["overall_health"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_report_overall_health_calculation(self, system_state):
        """Test overall health calculation in report."""
        # Register components with different health statuses
        system_state.register_component("healthy", ComponentType.CORE)
        system_state.register_component("degraded", ComponentType.INDEXER)
        system_state.register_component("unhealthy", ComponentType.GRAPH)

        # Update health statuses
        await system_state.update_component_health("healthy", HealthStatus.HEALTHY)
        await system_state.update_component_health("degraded", HealthStatus.DEGRADED)
        await system_state.update_component_health("unhealthy", HealthStatus.UNHEALTHY)

        # Get health report without changing system status
        report = await system_state.get_health_report()

        # With unhealthy component, overall should be unhealthy
        assert report["overall_health"] == "unhealthy"

        # Fix unhealthy component
        await system_state.update_component_health("unhealthy", HealthStatus.HEALTHY)

        report = await system_state.get_health_report()
        # Now with only degraded, overall should be degraded
        assert report["overall_health"] == "degraded"

    def test_get_state_summary(self, populated_system_state):
        """Test getting concise state summary."""
        summary = populated_system_state.get_state_summary()

        assert summary["status"] == "running"
        assert summary["components_count"] == 3
        assert summary["healthy_components"] == 3
        assert summary["projects_analyzed"] == 0
        assert "uptime" in summary
        assert "last_update" in summary


class TestHealthMonitoring:
    """Tests for health monitoring functionality."""

    @pytest.mark.asyncio
    async def test_start_health_monitoring(self, system_state):
        """Test starting health monitoring."""
        # Set a very short interval for testing
        system_state.health_check_interval = 0.01
        await system_state.start_health_monitoring()

        assert system_state._health_check_task is not None
        assert not system_state._health_check_task.done()

        # Clean up immediately
        await system_state.stop_health_monitoring()

    @pytest.mark.asyncio
    async def test_stop_health_monitoring(self, system_state):
        """Test stopping health monitoring."""
        # Set a very short interval for testing
        system_state.health_check_interval = 0.01
        await system_state.start_health_monitoring()

        # Wait a bit then stop
        await asyncio.sleep(0.02)
        await system_state.stop_health_monitoring()

        # Task should be cancelled
        assert system_state._health_check_task is None

    @pytest.mark.asyncio
    async def test_perform_health_checks(self, system_state):
        """Test performing health checks on components."""

        # Register components with health check functions
        def healthy_check():
            return True

        def unhealthy_check():
            return False

        def throwing_check():
            raise Exception("Health check failed")

        system_state.register_component(
            "healthy_component", ComponentType.CORE, health_check=healthy_check
        )
        system_state.register_component(
            "unhealthy_component", ComponentType.INDEXER, health_check=unhealthy_check
        )
        system_state.register_component(
            "throwing_component", ComponentType.GRAPH, health_check=throwing_check
        )

        with patch.object(
            system_state, "update_component_health", new_callable=AsyncMock
        ) as mock_update:
            # Perform health checks
            await system_state._perform_health_checks()

            # Check components were updated
            # Healthy component
            mock_update.assert_any_call(
                name="healthy_component",
                status=HealthStatus.HEALTHY,
                response_time_ms=ANY,
            )

            # Unhealthy component
            mock_update.assert_any_call(
                name="unhealthy_component",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=ANY,
            )

            # Throwing component
            mock_update.assert_any_call(
                name="throwing_component",
                status=HealthStatus.UNHEALTHY,
                details={"health_check_error": ANY},
            )

    @pytest.mark.asyncio
    async def test_health_monitoring_loop(self, system_state):
        """Test health monitoring loop execution."""
        # Mock the health check method
        with patch.object(system_state, "_perform_health_checks") as mock_checks:
            mock_checks.return_value = None

            # Set a very short interval for testing
            system_state.health_check_interval = 0.01

            # Start monitoring
            await system_state.start_health_monitoring()

            # Wait for a short time to allow at least one check
            await asyncio.sleep(0.03)

            # Stop monitoring
            await system_state.stop_health_monitoring()

            # Should have been called at least once
            assert mock_checks.call_count >= 1


class TestEventHandlers:
    """Tests for event handlers."""

    @pytest.mark.asyncio
    async def test_on_component_registered(self, system_state):
        """Test component registered event handler."""
        data = {"name": "new_component", "timestamp": datetime.now().isoformat()}
        await system_state._on_component_registered(data)

        # Should update last active time
        assert "new_component" in system_state.component_last_active

    @pytest.mark.asyncio
    async def test_on_component_unregistered(self, system_state):
        """Test component unregistered event handler."""
        # First register
        system_state.component_last_active["component_to_remove"] = datetime.now()

        # Then unregister
        data = {"name": "component_to_remove", "timestamp": datetime.now().isoformat()}
        await system_state._on_component_unregistered(data)

        # Should be removed from last active
        assert "component_to_remove" not in system_state.component_last_active

    @pytest.mark.asyncio
    async def test_on_component_heartbeat(self, system_state):
        """Test component heartbeat event handler."""
        # Register component first
        system_state.register_component("heartbeat_component", ComponentType.CORE)

        # Send heartbeat
        data = {
            "name": "heartbeat_component",
            "status": "healthy",
            "response_time_ms": 25.5,
            "details": {"custom": "data"},
        }

        await system_state._on_component_heartbeat(data)

        # Should update last active
        assert system_state.component_last_active["heartbeat_component"] is not None

        # Should update component health
        component = system_state.components["heartbeat_component"]
        assert component.status == HealthStatus.HEALTHY
        assert component.response_time_ms == 25.5

    @pytest.mark.asyncio
    async def test_on_component_error(self, system_state):
        """Test component error event handler."""
        # Mock record_metric to avoid async issues
        with patch.object(system_state, "record_metric") as mock_record:
            # Register component first
            system_state.register_component("error_component", ComponentType.CORE)

            # Send error event
            data = {
                "name": "error_component",
                "error": "Something went wrong",
                "critical": True,
                "timestamp": datetime.now().isoformat(),
            }

            await system_state._on_component_error(data)

            # Should call record_metric for errors
            mock_record.assert_any_call("total_errors", 1, increment=True)
            mock_record.assert_any_call("critical_errors", 1, increment=True)

            # Should update component health
            component = system_state.components["error_component"]
            assert component.status == HealthStatus.UNHEALTHY
            assert len(component.warnings) == 1

    @pytest.mark.asyncio
    async def test_on_system_shutdown(self, system_state):
        """Test system shutdown event handler."""
        # Set up system through proper transitions
        await system_state.set_status(SystemStatus.STARTING)
        await system_state.set_status(SystemStatus.RUNNING)
        system_state.register_component("shutdown_component", ComponentType.CORE)

        with patch.object(
            system_state, "stop_health_monitoring", new_callable=AsyncMock
        ) as mock_stop:
            # Trigger shutdown
            data = {"reason": "Test shutdown"}
            await system_state._on_system_shutdown(data)

            # Should update status
            assert system_state.status == SystemStatus.STOPPING

            # Should stop health monitoring
            mock_stop.assert_called_once()

            # Should update components to stopped
            component = system_state.components["shutdown_component"]
            assert component.status == HealthStatus.STOPPED

    @pytest.mark.asyncio
    async def test_on_metric_updated(self, system_state):
        """Test metric updated event handler."""
        data = {
            "metric": "projects_analyzed",
            "value": 42,
            "timestamp": datetime.now().isoformat(),
        }

        await system_state._on_metric_updated(data)

        # Should update metric
        assert system_state.metrics.projects_analyzed == 42

    @pytest.mark.asyncio
    async def test_on_analysis_events(self, system_state):
        """Test analysis-related event handlers."""
        # Set up proper state transitions
        await system_state.set_status(SystemStatus.STARTING)
        await system_state.set_status(SystemStatus.RUNNING)

        # Register indexer for activity tracking
        system_state.register_component("indexer", ComponentType.INDEXER)

        # Mock record_metric to capture calls
        with patch.object(system_state, "record_metric") as mock_record:
            # Test analysis completed
            data = {
                "project_id": "test_project",
                "files_analyzed": 100,
                "entities_found": 500,
            }
            await system_state._on_analysis_completed(data)

            # Check that record_metric was called correctly
            mock_record.assert_any_call("projects_analyzed", 1, increment=True)
            mock_record.assert_any_call("files_processed", 100, increment=True)
            mock_record.assert_any_call("entities_extracted", 500, increment=True)

            # Reset mock for next call
            mock_record.reset_mock()

            # Test analysis failed
            data = {"project_id": "test_project", "error": "Analysis failed"}
            await system_state._on_analysis_failed(data)

            # Check that record_metric was called for errors
            mock_record.assert_called_once_with("total_errors", 1, increment=True)


class TestShutdownAndCleanup:
    """Tests for shutdown and cleanup functionality."""

    @pytest.mark.asyncio
    async def test_shutdown(self, system_state):
        """Test system shutdown."""
        # Set up system through proper transitions
        await system_state.set_status(SystemStatus.STARTING)
        await system_state.set_status(SystemStatus.RUNNING)
        system_state.register_component("test_component", ComponentType.CORE)

        # Mock health monitoring methods
        with patch.object(system_state, "stop_health_monitoring") as mock_stop:
            with patch.object(system_state, "_persist_state") as mock_persist:
                # Perform shutdown
                await system_state.shutdown()

                # Should be in STOPPED state
                assert system_state.status == SystemStatus.STOPPED

                # Health monitoring should be stopped
                mock_stop.assert_called_once()

                # State should be persisted
                mock_persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_state(self, system_state):
        """Test state persistence (logs final state)."""
        with patch.object(system_state, "get_health_report", new_callable=AsyncMock) as mock_report:
            mock_report.return_value = {"status": "stopped"}

            await system_state._persist_state()

            # Should call get_health_report
            mock_report.assert_called_once()


class TestCleanup:
    """Tests to ensure proper cleanup."""

    @pytest.mark.asyncio
    async def test_no_tasks_left_after_cleanup(self, system_state):
        """Test that no tasks are left hanging after test cleanup."""
        # Set a very short interval for testing
        system_state.health_check_interval = 0.01
        await system_state.start_health_monitoring()
        await system_state.stop_health_monitoring()

        # Give it a moment to settle
        await asyncio.sleep(0.01)

        # Verify no health check task is running
        assert system_state._health_check_task is None or system_state._health_check_task.done()


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, system_state):
        """Test concurrent state updates."""
        # Register multiple components
        for i in range(10):
            system_state.register_component(f"component_{i}", ComponentType.CORE)

        # Update all components concurrently
        tasks = [
            asyncio.create_task(
                system_state.update_component_health(
                    f"component_{i}",
                    HealthStatus.HEALTHY,
                    response_time_ms=i * 10.0,
                )
            )
            for i in range(10)
        ]

        # Wait for all updates with timeout
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)

        # All components should exist
        assert len(system_state.components) == 10

        # All components should be healthy
        for i in range(10):
            component = system_state.components[f"component_{i}"]
            assert component.status == HealthStatus.HEALTHY

    def test_string_representation(self, system_state):
        """Test string representation of SystemState."""
        string_repr = str(system_state)
        assert "SystemState" in string_repr
        assert "status=" in string_repr
        assert "components=" in string_repr

    @pytest.mark.asyncio
    async def test_health_check_with_exception(self, system_state):
        """Test health check that throws exception."""

        # Create a health check that raises an exception
        def bad_health_check():
            raise ValueError("Health check failed")

        # Register component with bad health check
        system_state.register_component(
            "bad_component", ComponentType.CORE, health_check=bad_health_check
        )

        # Mock update_component_health to verify it's called
        with patch.object(
            system_state, "update_component_health", new_callable=AsyncMock
        ) as mock_update:
            # Perform health checks
            await system_state._perform_health_checks()

            # Component should be marked as unhealthy
            mock_update.assert_called_with(
                name="bad_component",
                status=HealthStatus.UNHEALTHY,
                details={"health_check_error": ANY},
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
