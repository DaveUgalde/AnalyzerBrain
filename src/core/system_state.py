"""
System State Manager for ANALYZERBRAIN.

Manages global system state, health metrics, and component coordination.
Provides centralized state management with event-driven updates.

Dependencies:
    - config_manager: For system configuration
    - event_bus: For publishing state changes
    - logging_config: For structured logging

Author: ANALYZERBRAIN Team
Date: 2024
Version: 1.0.0
"""

import asyncio
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable

from loguru import logger

from .config_manager import config
from .event_bus import EventBus
from .exceptions import SystemStateError


class SystemStatus(Enum):
    """System operational status."""

    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"  # Some components failing but system operational
    RECOVERING = "recovering"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ComponentType(Enum):
    """Types of system components."""

    CORE = "core"
    INDEXER = "indexer"
    GRAPH = "graph"
    EMBEDDINGS = "embeddings"
    AGENT = "agent"
    MEMORY = "memory"
    LEARNING = "learning"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    MONITORING = "monitoring"


class HealthStatus(Enum):
    """Health status of individual components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPED = "stopped"


@dataclass
class ComponentHealth:
    """Health information for a system component."""

    name: str
    component_type: ComponentType
    status: HealthStatus
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["component_type"] = self.component_type.value
        data["status"] = self.status.value
        data["last_check"] = self.last_check.isoformat()
        return data


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""

    # Timestamps
    startup_time: datetime
    last_update: datetime

    # Performance metrics
    uptime_seconds: float = 0.0
    projects_analyzed: int = 0
    files_processed: int = 0
    entities_extracted: int = 0
    graph_nodes: int = 0
    graph_relationships: int = 0
    embeddings_generated: int = 0
    agent_tasks_completed: int = 0

    # Error metrics
    total_errors: int = 0
    error_rate_5min: float = 0.0
    critical_errors: int = 0

    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    disk_usage_percent: float = 0.0

    # Latency metrics (in milliseconds)
    avg_indexing_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    avg_embedding_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["startup_time"] = self.startup_time.isoformat()
        data["last_update"] = self.last_update.isoformat()
        return data


class SystemState:
    """
    Centralized system state manager.

    Manages:
    - System operational status
    - Component health monitoring
    - Performance metrics collection
    - State persistence and recovery
    - Event-driven state updates
    """

    def __init__(self, event_bus: EventBus):
        """
        Initialize system state manager.

        Args:
            event_bus: Event bus for state change notifications
        """
        self.event_bus = event_bus
        self._state_lock = asyncio.Lock()

        # System status
        self.status = SystemStatus.INITIALIZING
        self.previous_status = None

        # Component registry
        self.components: Dict[str, ComponentHealth] = {}
        self.component_dependencies: Dict[str, Set[str]] = {}

        # Metrics
        self.metrics = SystemMetrics(startup_time=datetime.now(), last_update=datetime.now())

        # State history for recovery and debugging
        self.state_history: List[Dict[str, Any]] = []
        self.max_history_size = config.get("system_state.max_history_size", 1000)

        # Health check intervals (seconds)
        self.health_check_interval = config.get("system_state.health_check_interval", 30)
        self._health_check_task: Optional[asyncio.Task] = None

        # Watchdog for stuck components
        self.watchdog_timeout = config.get("system_state.watchdog_timeout", 300)
        self.component_last_active: Dict[str, datetime] = {}

        # Modo testing
        self._testing_mode = False

        # Event subscriptions
        self._setup_event_subscriptions()

        logger.info("System State Manager initialized")

    def _setup_event_subscriptions(self):
        """Subscribe to relevant system events."""
        # Component events
        self.event_bus.subscribe("component.registered", self._on_component_registered)
        self.event_bus.subscribe("component.unregistered", self._on_component_unregistered)
        self.event_bus.subscribe("component.heartbeat", self._on_component_heartbeat)
        self.event_bus.subscribe("component.error", self._on_component_error)

        # System events
        self.event_bus.subscribe("system.shutdown", self._on_system_shutdown)
        self.event_bus.subscribe("system.metric_updated", self._on_metric_updated)

        # Analysis events
        self.event_bus.subscribe("project.analysis_started", self._on_analysis_started)
        self.event_bus.subscribe("project.analysis_completed", self._on_analysis_completed)
        self.event_bus.subscribe("project.analysis_failed", self._on_analysis_failed)

    async def set_status(self, new_status: SystemStatus, reason: Optional[str] = None) -> None:
        """
        Update system status with proper state transitions.

        Args:
            new_status: New system status
            reason: Optional reason for status change

        Raises:
            SystemStateError: If invalid state transition
        """
        async with self._state_lock:
            old_status = self.status

            # Validate state transition
            if not self._is_valid_transition(old_status, new_status):
                raise SystemStateError(
                    f"Invalid state transition: {old_status.value} -> {new_status.value}"
                )

            # Update state
            self.previous_status = old_status
            self.status = new_status

            # Record in history
            self._record_state_change(old_status, new_status, reason)

            # Publish event
            await self.event_bus.publish(
                "system.status_changed",
                {
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info(
                f"System status changed: {old_status.value} -> {new_status.value}"
                + (f" (Reason: {reason})" if reason else "")
            )

    def _is_valid_transition(self, old_status: SystemStatus, new_status: SystemStatus) -> bool:
        """
        Validate state transition logic.

        Args:
            old_status: Current status
            new_status: Proposed new status

        Returns:
            True if transition is valid
        """
        valid_transitions = {
            SystemStatus.INITIALIZING: [SystemStatus.STARTING, SystemStatus.ERROR],
            SystemStatus.STARTING: [SystemStatus.RUNNING, SystemStatus.ERROR],
            SystemStatus.RUNNING: [
                SystemStatus.DEGRADED,
                SystemStatus.RECOVERING,
                SystemStatus.STOPPING,
                SystemStatus.ERROR,
                SystemStatus.MAINTENANCE,
            ],
            SystemStatus.DEGRADED: [
                SystemStatus.RUNNING,
                SystemStatus.RECOVERING,
                SystemStatus.STOPPING,
                SystemStatus.ERROR,
            ],
            SystemStatus.RECOVERING: [
                SystemStatus.RUNNING,
                SystemStatus.DEGRADED,
                SystemStatus.ERROR,
                SystemStatus.STOPPING,
            ],
            SystemStatus.STOPPING: [SystemStatus.STOPPED, SystemStatus.ERROR],
            SystemStatus.STOPPED: [SystemStatus.STARTING],
            SystemStatus.ERROR: [SystemStatus.RECOVERING, SystemStatus.STOPPED],
            SystemStatus.MAINTENANCE: [SystemStatus.RUNNING, SystemStatus.STOPPING],
        }

        return new_status in valid_transitions.get(old_status, [])

    def register_component(
        self,
        name: str,
        component_type: ComponentType,
        dependencies: Optional[List[str]] = None,
        health_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Register a system component for health monitoring.

        Args:
            name: Unique component identifier
            component_type: Type of component
            dependencies: List of component names this component depends on
            health_check: Optional health check function
        """
        if name in self.components:
            logger.warning(f"Component already registered: {name}")
            return

        component_health = ComponentHealth(
            name=name,
            component_type=component_type,
            status=HealthStatus.STARTING,
            last_check=datetime.now(),
            details={"health_check": health_check is not None},
        )

        # Store health check function if provided
        if health_check:
            component_health.details["health_check_fn"] = health_check

        self.components[name] = component_health

        if dependencies:
            self.component_dependencies[name] = set(dependencies)

        # Update last active time
        self.component_last_active[name] = datetime.now()

        # Publish component registration event
        try:
            asyncio.create_task(
                self.event_bus.publish(
                    "component.registered",
                    {
                        "name": name,
                        "type": component_type.value,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            )
        except RuntimeError:
            # If no event loop, log warning and continue
            logger.warning(f"No event loop for publishing component registration: {name}")

        logger.info(f"Component registered: {name} ({component_type.value})")

    def unregister_component(self, name: str) -> None:
        """
        Unregister a system component.

        Args:
            name: Component identifier
        """
        if name not in self.components:
            logger.warning(f"Component not found for unregistration: {name}")
            return

        del self.components[name]

        # Remove from dependencies
        for deps in self.component_dependencies.values():
            deps.discard(name)

        if name in self.component_dependencies:
            del self.component_dependencies[name]

        # Remove from last active
        if name in self.component_last_active:
            del self.component_last_active[name]

        # Publish component unregistration event
        asyncio.create_task(
            self.event_bus.publish(
                "component.unregistered",
                {"name": name, "timestamp": datetime.now().isoformat()},
            )
        )

        logger.info(f"Component unregistered: {name}")

    async def update_component_health(
        self,
        name: str,
        status: HealthStatus,
        response_time_ms: Optional[float] = None,
        warnings: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if name not in self.components:
            logger.warning(f"Cannot update health for unregistered component: {name}")
            return

        event_payload = None
        should_recalculate = False

        async with self._state_lock:
            component = self.components[name]

            old_status = component.status
            status_changed = old_status != status

            component.status = status
            component.last_check = datetime.now()

            if response_time_ms is not None:
                component.response_time_ms = response_time_ms

            if warnings is not None:
                component.warnings = warnings

            if details is not None:
                component.details.update(details)

            if status == HealthStatus.UNHEALTHY:
                component.error_count += 1
            elif status == HealthStatus.HEALTHY:
                component.error_count = 0

            self.component_last_active[name] = datetime.now()

            if status_changed:
                should_recalculate = True
                event_payload = {
                    "name": name,
                    "old_status": old_status.value,
                    "new_status": status.value,
                    "response_time_ms": response_time_ms,
                    "timestamp": datetime.now().isoformat(),
                }

        # ⬇️ FUERA DEL LOCK (CRÍTICO)
        if should_recalculate:
            await self._update_system_status_based_on_components()
            await self.event_bus.publish("component.health_updated", event_payload)

            logger.info(f"Component health updated: {name} {old_status.value} -> {status.value}")

    async def _update_system_status_based_on_components(self) -> None:
        """
        Update system status based on component health.
        """
        if not self.components:
            return

        # Only update from certain states
        if self.status not in [
            SystemStatus.RUNNING,
            SystemStatus.DEGRADED,
            SystemStatus.RECOVERING,
        ]:
            return

        # Count component statuses
        status_counts = {}
        for component in self.components.values():
            status_counts[component.status] = status_counts.get(component.status, 0) + 1

        # Determine system status
        total_components = len(self.components)
        unhealthy_count = status_counts.get(HealthStatus.UNHEALTHY, 0)
        degraded_count = status_counts.get(HealthStatus.DEGRADED, 0)

        # If any component is unhealthy and system is running/degraded, mark as degraded
        if self.status in [SystemStatus.RUNNING, SystemStatus.DEGRADED]:
            if unhealthy_count > 0:
                if unhealthy_count / total_components > 0.5:  # More than 50% unhealthy
                    # Don't set status if we're already in the same state
                    if self.status != SystemStatus.ERROR:
                        await self.set_status(
                            SystemStatus.ERROR, "Majority of components unhealthy"
                        )
                else:
                    if self.status != SystemStatus.DEGRADED:
                        await self.set_status(SystemStatus.DEGRADED, "Some components unhealthy")
            elif degraded_count > 0:
                if self.status != SystemStatus.DEGRADED:
                    await self.set_status(SystemStatus.DEGRADED, "Some components degraded")
            else:
                # All components healthy, ensure system is running
                if self.status == SystemStatus.DEGRADED:
                    await self.set_status(SystemStatus.RUNNING, "All components recovered")

    def record_metric(self, metric_name: str, value: Any, increment: bool = False) -> None:
        """
        Record a system metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            increment: If True, increment the existing value
        """
        if not hasattr(self.metrics, metric_name):
            logger.warning(f"Unknown metric: {metric_name}")
            return

        # Si estamos en modo testing, ejecutar sincrónicamente
        if self._testing_mode:
            self._update_metric_sync(metric_name, value, increment)
            return

        async def _update():
            async with self._state_lock:
                self._update_metric_sync(metric_name, value, increment)

                # Publish metric update
                await self.event_bus.publish(
                    "system.metric_updated",
                    {
                        "metric": metric_name,
                        "value": getattr(self.metrics, metric_name),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        # Try to schedule the task
        try:
            asyncio.create_task(_update())
        except RuntimeError:
            # Fallback: run sync update without publishing event
            self._update_metric_sync(metric_name, value, increment)

    def _update_metric_sync(self, metric_name: str, value: Any, increment: bool) -> None:
        """Synchronous metric update for testing and fallback."""
        try:
            current_value = getattr(self.metrics, metric_name)

            if increment:
                if isinstance(current_value, (int, float)):
                    setattr(self.metrics, metric_name, current_value + value)
                else:
                    logger.error(f"Cannot increment non-numeric metric: {metric_name}")
                    return
            else:
                setattr(self.metrics, metric_name, value)

            self.metrics.last_update = datetime.now()
        except Exception as e:
            logger.warning(f"Failed to update metric {metric_name}: {e}")

    async def get_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health report.

        Returns:
            Health report dictionary
        """
        async with self._state_lock:
            # Check for stale components
            stale_components = []
            now = datetime.now()
            for name, last_active in self.component_last_active.items():
                if (now - last_active).total_seconds() > self.watchdog_timeout:
                    stale_components.append(name)

            # Calculate overall health based on component statuses and stale components
            component_statuses = [comp.status for comp in self.components.values()]
            overall_health = HealthStatus.HEALTHY

            # If any component is unhealthy, overall is unhealthy
            if HealthStatus.UNHEALTHY in component_statuses:
                overall_health = HealthStatus.UNHEALTHY
            # If any component is degraded OR there are stale components, overall is degraded
            elif HealthStatus.DEGRADED in component_statuses or stale_components:
                overall_health = HealthStatus.DEGRADED
            elif HealthStatus.UNKNOWN in component_statuses:
                overall_health = HealthStatus.UNKNOWN
            # If no components, still consider it healthy
            elif not self.components:
                overall_health = HealthStatus.HEALTHY

            # Update uptime
            self.metrics.uptime_seconds = (
                datetime.now() - self.metrics.startup_time
            ).total_seconds()

            return {
                "system_status": self.status.value,
                "overall_health": overall_health.value,
                "uptime_seconds": self.metrics.uptime_seconds,
                "components": {name: comp.to_dict() for name, comp in self.components.items()},
                "metrics": self.metrics.to_dict(),
                "stale_components": stale_components,
                "timestamp": datetime.now().isoformat(),
            }

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get concise state summary for quick monitoring.

        Returns:
            Summary dictionary
        """
        self.metrics.uptime_seconds = (datetime.now() - self.metrics.startup_time).total_seconds()

        healthy_count = sum(1 for c in self.components.values() if c.status == HealthStatus.HEALTHY)

        return {
            "status": self.status.value,
            "components_count": len(self.components),
            "healthy_components": healthy_count,
            "projects_analyzed": self.metrics.projects_analyzed,
            "uptime": str(timedelta(seconds=int(self.metrics.uptime_seconds))),
            "last_update": self.metrics.last_update.isoformat(),
        }

    async def start_health_monitoring(self) -> None:
        """Start periodic health monitoring of all components."""
        if self._health_check_task and not self._health_check_task.done():
            logger.warning("Health monitoring already running")
            return

        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("Health monitoring started")

    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Health monitoring stopped")

    async def _health_monitoring_loop(self) -> None:
        """Periodic health check loop."""
        try:
            while True:
                try:
                    await self._perform_health_checks()
                except Exception as e:
                    logger.error(f"Health check error: {e}")

                await asyncio.sleep(self.health_check_interval)
        except asyncio.CancelledError:
            logger.info("Health monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Health monitoring loop critical error: {e}")
            # No auto-restart

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all registered components."""
        for name, component in self.components.items():
            try:
                # Check if component has a health check function
                health_check_func = component.details.get("health_check_fn")

                if health_check_func and callable(health_check_func):
                    # Execute health check with timeout
                    start_time = time.time()
                    try:
                        # Timeout de 10 segundos para health checks
                        is_healthy = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(None, health_check_func),
                            timeout=10.0,
                        )
                        response_time = (time.time() - start_time) * 1000  # Convert to ms

                        # Update component health
                        status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                        await self.update_component_health(
                            name=name, status=status, response_time_ms=response_time
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"Health check timeout for component {name}")
                        await self.update_component_health(
                            name=name,
                            status=HealthStatus.UNHEALTHY,
                            details={"health_check_error": "Timeout"},
                        )
                else:
                    # No health check function, just mark as checked
                    component.last_check = datetime.now()

            except Exception as e:
                logger.error(f"Health check failed for component {name}: {e}")
                await self.update_component_health(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    details={"health_check_error": str(e)},
                )

    def _record_state_change(
        self, old_status: SystemStatus, new_status: SystemStatus, reason: Optional[str]
    ) -> None:
        """Record state change in history."""
        self.state_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "old_status": old_status.value,
                "new_status": new_status.value,
                "reason": reason,
                "components_count": len(self.components),
            }
        )

        # Limit history size
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size :]

    # Event Handlers
    async def _on_component_registered(self, data: Dict[str, Any]) -> None:
        """Handle component registration event."""
        component_name = data["name"]
        self.component_last_active[component_name] = datetime.now()

    async def _on_component_unregistered(self, data: Dict[str, Any]) -> None:
        """Handle component unregistration event."""
        component_name = data["name"]
        if component_name in self.component_last_active:
            del self.component_last_active[component_name]

    async def _on_component_heartbeat(self, data: Dict[str, Any]) -> None:
        """Handle component heartbeat event."""
        component_name = data["name"]
        self.component_last_active[component_name] = datetime.now()

        # Update component health if provided
        if "status" in data:
            try:
                status = HealthStatus(data["status"])
                await self.update_component_health(
                    name=component_name,
                    status=status,
                    response_time_ms=data.get("response_time_ms"),
                    details=data.get("details"),
                )
            except ValueError:
                logger.warning(f"Invalid health status: {data['status']}")

    async def _on_component_error(self, data: Dict[str, Any]) -> None:
        """Handle component error event."""
        component_name = data["name"]

        # Record error in metrics
        self.record_metric("total_errors", 1, increment=True)

        if data.get("critical", False):
            self.record_metric("critical_errors", 1, increment=True)

        # Update component health
        await self.update_component_health(
            name=component_name,
            status=HealthStatus.UNHEALTHY,
            warnings=[data.get("error", "Unknown error")],
            details={"last_error": data},
        )

    async def _on_system_shutdown(self, data: Dict[str, Any]) -> None:
        """Handle system shutdown event."""
        await self.set_status(SystemStatus.STOPPING, "System shutdown requested")

        # Stop health monitoring
        await self.stop_health_monitoring()

        # Update all components to stopped
        for component in self.components.values():
            component.status = HealthStatus.STOPPED
            component.last_check = datetime.now()

    async def _on_metric_updated(self, data: Dict[str, Any]) -> None:
        """Handle external metric updates."""
        # This allows other components to update metrics via events
        metric_name = data["metric"]
        value = data["value"]

        if hasattr(self.metrics, metric_name):
            setattr(self.metrics, metric_name, value)
            self.metrics.last_update = datetime.now()

    async def _on_analysis_started(self, data: Dict[str, Any]) -> None:
        """Handle project analysis started event."""
        # Update component activity
        if "indexer" in self.components:
            self.component_last_active["indexer"] = datetime.now()

    async def _on_analysis_completed(self, data: Dict[str, Any]) -> None:
        """Handle project analysis completed event."""
        # Update metrics
        self.record_metric("projects_analyzed", 1, increment=True)

        if "files_analyzed" in data:
            self.record_metric("files_processed", data["files_analyzed"], increment=True)

        if "entities_found" in data:
            self.record_metric("entities_extracted", data["entities_found"], increment=True)

    async def _on_analysis_failed(self, data: Dict[str, Any]) -> None:
        """Handle project analysis failed event."""
        self.record_metric("total_errors", 1, increment=True)

    async def shutdown(self) -> None:
        """Gracefully shutdown system state manager."""
        await self.set_status(SystemStatus.STOPPING, "Shutting down")
        await self.stop_health_monitoring()
        await self.set_status(SystemStatus.STOPPED, "System shutdown complete")

        # Save final state
        await self._persist_state()

        logger.info("System State Manager shutdown complete")

    async def _persist_state(self) -> None:
        """Persist current state to disk (optional)."""
        # This would save state to disk for recovery
        # For now, just log the final state
        health_report = await self.get_health_report()
        logger.info(f"Final system state: {health_report}")

    def __str__(self) -> str:
        """String representation of system state."""
        return (
            f"SystemState(status={self.status.value}, "
            f"components={len(self.components)}, "
            f"projects_analyzed={self.metrics.projects_analyzed})"
        )


# Singleton instance
_system_state_instance: Optional[SystemState] = None


def get_system_state(event_bus: Optional[EventBus] = None) -> SystemState:
    """
    Get or create singleton system state instance.

    Args:
        event_bus: Event bus instance (required on first call)

    Returns:
        SystemState instance

    Raises:
        SystemStateError: If event_bus not provided on first call
    """
    global _system_state_instance

    if _system_state_instance is None:
        if event_bus is None:
            raise SystemStateError("EventBus required for SystemState initialization")
        _system_state_instance = SystemState(event_bus)

    return _system_state_instance
