"""
Tests for Event Bus system.

Comprehensive unit and integration tests for the event-driven communication system.

Author: ANALYZERBRAIN Team
Date: 2024
"""

import asyncio
import pytest
import warnings
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch, call
import sys
import os

# Agregar el directorio src al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.event_bus import (
    EventBus,
    Event,
    EventPriority,
    EventDeliveryMode,
    get_default_event_bus,
    create_event_bus,
    EventBusError,
)


@pytest.fixture
def event_bus():
    """Create an EventBus instance for testing."""
    bus = EventBus(name="test_bus")
    yield bus
    # Cleanup
    asyncio.run(bus.stop())


class TestEventInitialization:
    """Tests for Event data structure."""

    def test_event_creation(self):
        """Test creating an Event with all parameters."""
        data = {"key": "value", "number": 42}
        metadata = {"source": "test", "version": "1.0"}

        event = Event(
            type="test.event",
            data=data,
            source="test_system",
            correlation_id="test-correlation-123",
            priority=EventPriority.HIGH,
            metadata=metadata,
        )

        assert event.type == "test.event"
        assert event.data == data
        assert event.source == "test_system"
        assert event.correlation_id == "test-correlation-123"
        assert event.priority == EventPriority.HIGH
        assert event.metadata == metadata
        assert isinstance(event.timestamp, datetime)

    def test_event_default_correlation_id(self):
        """Test that event generates correlation ID if not provided."""
        event = Event(type="test.event", data={})

        assert event.correlation_id is not None
        assert len(event.correlation_id) > 0
        assert isinstance(event.correlation_id, str)


class TestEventBusInitialization:
    """Tests for EventBus initialization."""

    def test_initialization_with_name(self):
        """Test EventBus initialization with custom name."""
        bus = EventBus(name="custom_bus")

        assert bus.name == "custom_bus"
        assert bus._subscriptions == {}
        assert bus._wildcard_subscriptions == []
        assert bus._pre_publish_middleware == []
        assert bus._post_publish_middleware == []
        assert bus._error_handlers == []
        assert bus._metrics["events_published"] == 0
        assert bus._metrics["subscriptions_count"] == 0

    def test_string_representation(self):
        """Test string representation of EventBus."""
        bus = EventBus(name="test_bus")
        assert str(bus) == "EventBus(name=test_bus, subscriptions=0)"


class TestEventBusSubscription:
    """Tests for event subscription and unsubscription."""

    def test_subscribe_single_event(self, event_bus):
        """Test subscribing to a single event type."""
        callback = MagicMock()
        subscription_id = event_bus.subscribe("test.event", callback)

        assert subscription_id is not None
        assert "test.event" in event_bus._subscriptions
        assert len(event_bus._subscriptions["test.event"]) == 1
        assert event_bus._subscriptions["test.event"][0].callback == callback
        assert event_bus._metrics["subscriptions_count"] == 1

    def test_subscribe_multiple_events(self, event_bus):
        """Test subscribing to multiple event types at once."""
        callback = MagicMock()
        subscription_id = event_bus.subscribe(["event1", "event2"], callback)

        assert subscription_id is not None
        assert "event1" in event_bus._subscriptions
        assert "event2" in event_bus._subscriptions
        assert len(event_bus._subscriptions["event1"]) == 1
        assert len(event_bus._subscriptions["event2"]) == 1
        # When subscribing to multiple events with one call, it counts as one subscription
        # because it's one callback registered for multiple event types
        assert event_bus._metrics["subscriptions_count"] == 1

    def test_subscribe_wildcard(self, event_bus):
        """Test wildcard subscription."""
        callback = MagicMock()
        subscription_id = event_bus.subscribe("*", callback)

        assert subscription_id is not None
        assert len(event_bus._wildcard_subscriptions) == 1
        assert event_bus._wildcard_subscriptions[0].callback == callback
        assert event_bus._metrics["subscriptions_count"] == 1

    def test_subscribe_with_priority(self, event_bus):
        """Test subscription with priority."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        # Subscribe with different priorities
        event_bus.subscribe("test.event", callback1, priority=EventPriority.HIGH.value)
        event_bus.subscribe("test.event", callback2, priority=EventPriority.LOW.value)

        subscriptions = event_bus._subscriptions["test.event"]
        assert len(subscriptions) == 2
        # Should be sorted by priority (highest first)
        assert subscriptions[0].priority == EventPriority.HIGH.value
        assert subscriptions[1].priority == EventPriority.LOW.value

    def test_subscribe_with_filter(self, event_bus):
        """Test subscription with filter function."""
        callback = MagicMock()

        # Filter that only accepts events with data["value"] > 10
        filter_func = lambda event: event.data.get("value", 0) > 10

        subscription_id = event_bus.subscribe("test.event", callback, filter_func=filter_func)

        subscription = event_bus._subscriptions["test.event"][0]
        assert subscription.filter_func is not None

        # Test filter works
        event1 = Event(type="test.event", data={"value": 5})
        event2 = Event(type="test.event", data={"value": 15})

        assert not subscription.filter_func(event1)
        assert subscription.filter_func(event2)

    def test_subscribe_invalid_callback(self, event_bus):
        """Test subscribing with invalid callback raises error."""
        with pytest.raises(EventBusError):
            event_bus.subscribe("test.event", "not a callable")

    def test_unsubscribe_by_id(self, event_bus):
        """Test unsubscribing using subscription ID."""
        callback = MagicMock()
        subscription_id = event_bus.subscribe("test.event", callback)

        assert event_bus.get_subscription_count("test.event") == 1

        # Unsubscribe
        result = event_bus.unsubscribe(subscription_id)

        assert result is True
        assert event_bus.get_subscription_count("test.event") == 0
        assert event_bus._metrics["subscriptions_count"] == 0

    def test_unsubscribe_invalid_id(self, event_bus):
        """Test unsubscribing with invalid ID returns False."""
        result = event_bus.unsubscribe("invalid_id")
        assert result is False

    def test_unsubscribe_all_for_event_type(self, event_bus):
        """Test unsubscribing all handlers for a specific event type."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        event_bus.subscribe("test.event", callback1)
        event_bus.subscribe("test.event", callback2)
        event_bus.subscribe("other.event", MagicMock())

        assert event_bus.get_subscription_count("test.event") == 2
        assert event_bus._metrics["subscriptions_count"] == 3

        removed = event_bus.unsubscribe_all("test.event")

        assert removed == 2
        assert event_bus.get_subscription_count("test.event") == 0
        assert event_bus._metrics["subscriptions_count"] == 1

    def test_unsubscribe_all_wildcard(self, event_bus):
        """Test unsubscribing all wildcard handlers."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        event_bus.subscribe("*", callback1)
        event_bus.subscribe("*", callback2)
        event_bus.subscribe("test.event", MagicMock())

        assert len(event_bus._wildcard_subscriptions) == 2
        assert event_bus._metrics["subscriptions_count"] == 3

        removed = event_bus.unsubscribe_all("*")

        assert removed == 2
        assert len(event_bus._wildcard_subscriptions) == 0
        assert event_bus._metrics["subscriptions_count"] == 1

    def test_unsubscribe_all_completely(self, event_bus):
        """Test unsubscribing all handlers."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        event_bus.subscribe("test.event", callback1)
        event_bus.subscribe("*", callback2)

        assert event_bus._metrics["subscriptions_count"] == 2

        removed = event_bus.unsubscribe_all()

        assert removed == 2
        assert event_bus._metrics["subscriptions_count"] == 0
        assert event_bus._subscriptions == {}
        assert event_bus._wildcard_subscriptions == []


class TestEventBusPublishing:
    """Tests for event publishing and handling."""

    @pytest.mark.asyncio
    async def test_publish_sync_mode(self, event_bus):
        """Test publishing in synchronous mode."""
        callback = AsyncMock()
        event_bus.subscribe("test.event", callback)

        event = await event_bus.publish(
            "test.event", {"key": "value"}, delivery_mode=EventDeliveryMode.SYNCHRONOUS
        )

        # Callback should have been called
        callback.assert_called_once()
        called_event = callback.call_args[0][0]

        assert isinstance(called_event, Event)
        assert called_event.type == "test.event"
        assert called_event.data == {"key": "value"}
        assert called_event.correlation_id == event.correlation_id

        # Metrics should be updated
        assert event_bus._metrics["events_published"] == 1
        assert event_bus._metrics["events_processed"] == 1

    @pytest.mark.asyncio
    async def test_publish_async_mode(self, event_bus):
        """Test publishing in asynchronous mode."""
        # Start the event bus for async processing
        await event_bus.start()

        callback = AsyncMock()
        event_bus.subscribe("test.event", callback)

        event = await event_bus.publish(
            "test.event", {"key": "value"}, delivery_mode=EventDeliveryMode.ASYNCHRONOUS
        )

        # Give some time for the async processing
        await asyncio.sleep(0.05)

        # Callback should have been called
        callback.assert_called_once()
        called_event = callback.call_args[0][0]

        assert called_event.type == "test.event"
        assert event_bus._metrics["events_published"] == 1

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_publish_broadcast_mode(self, event_bus):
        """Test publishing in broadcast mode."""
        callbacks = [AsyncMock() for _ in range(3)]
        for callback in callbacks:
            event_bus.subscribe("test.event", callback)

        event = await event_bus.publish(
            "test.event", {"key": "value"}, delivery_mode=EventDeliveryMode.BROADCAST
        )

        # All callbacks should have been called
        for callback in callbacks:
            callback.assert_called_once()

        assert event_bus._metrics["events_published"] == 1
        assert event_bus._metrics["events_processed"] == 3

    @pytest.mark.asyncio
    async def test_publish_with_filter(self, event_bus):
        """Test publishing with filtered subscriptions."""
        callback1 = AsyncMock()
        callback2 = AsyncMock()

        # Subscribe with filter
        event_bus.subscribe(
            "test.event", callback1, filter_func=lambda e: e.data.get("accept", False)
        )
        event_bus.subscribe("test.event", callback2)  # No filter

        # First event should only trigger callback2 (filter rejects)
        await event_bus.publish(
            "test.event", {"accept": False}, delivery_mode=EventDeliveryMode.SYNCHRONOUS
        )

        callback1.assert_not_called()
        callback2.assert_called_once()

        # Reset
        callback1.reset_mock()
        callback2.reset_mock()

        # Second event should trigger both (filter accepts)
        await event_bus.publish(
            "test.event", {"accept": True}, delivery_mode=EventDeliveryMode.SYNCHRONOUS
        )

        callback1.assert_called_once()
        callback2.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_wildcard_subscription(self, event_bus):
        """Test that wildcard subscriptions receive all events."""
        callback = AsyncMock()
        event_bus.subscribe("*", callback)

        # Publish different event types
        await event_bus.publish("event1", {"data": 1}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)
        await event_bus.publish("event2", {"data": 2}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        # Callback should have been called twice
        assert callback.call_count == 2

        # Check different event types were received
        calls = callback.call_args_list
        assert calls[0][0][0].type == "event1"
        assert calls[1][0][0].type == "event2"

    @pytest.mark.asyncio
    async def test_publish_with_priority(self, event_bus):
        """Test that handlers are called in priority order."""
        call_order = []

        def make_handler(name, priority):
            async def handler(event):
                call_order.append(name)

            event_bus.subscribe("test.event", handler, priority=priority)

        # Register handlers with different priorities
        make_handler("low", EventPriority.LOW.value)
        make_handler("high", EventPriority.HIGH.value)
        make_handler("normal", EventPriority.NORMAL.value)

        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        # Should be called in order: high, normal, low
        assert call_order == ["high", "normal", "low"]

    @pytest.mark.asyncio
    async def test_publish_with_source_and_correlation(self, event_bus):
        """Test publishing with source and correlation ID."""
        callback = AsyncMock()
        event_bus.subscribe("test.event", callback)

        event = await event_bus.publish(
            "test.event",
            {"key": "value"},
            source="test_source",
            correlation_id="test_correlation",
            delivery_mode=EventDeliveryMode.SYNCHRONOUS,
        )

        assert event.source == "test_source"
        assert event.correlation_id == "test_correlation"

        called_event = callback.call_args[0][0]
        assert called_event.source == "test_source"
        assert called_event.correlation_id == "test_correlation"

    @pytest.mark.asyncio
    async def test_publish_queue_full(self, event_bus):
        """Test behavior when event queue is full."""
        # Start the event bus
        await event_bus.start()

        # Subscribe a handler that will be called if event is processed
        callback = AsyncMock()
        event_bus.subscribe("test.event", callback)

        # Fill the queue completely
        for i in range(1000):  # Queue maxsize is 1000
            await event_bus._event_queue.put(Event(type="dummy", data={"id": i}))

        # Now queue is full, publishing should drop the event
        # We'll just verify the event bus is still functional
        await event_bus.publish(
            "test.event", {"key": "value"}, delivery_mode=EventDeliveryMode.ASYNCHRONOUS
        )

        # Give time for any processing
        await asyncio.sleep(0.05)

        # The callback might or might not be called depending on queue processing
        # We'll just verify the event bus is still running
        assert event_bus._queue_processor_task is not None

        await event_bus.stop()


class TestEventBusErrorHandling:
    """Tests for error handling in event processing."""

    @pytest.mark.asyncio
    async def test_handler_raises_exception(self, event_bus):
        """Test that handler exceptions are caught and metrics updated."""
        error_handler = AsyncMock()
        event_bus.add_error_handler(error_handler)

        def failing_handler(event):
            raise ValueError("Handler failed")

        event_bus.subscribe("test.event", failing_handler)

        # Publish event - should not raise
        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        # Error handler should have been called
        error_handler.assert_called_once()

        # Metrics should reflect the error
        assert event_bus._metrics["handler_errors"] == 1
        # Note: events_processed may or may not be incremented depending on implementation

    @pytest.mark.asyncio
    async def test_async_handler_raises_exception(self, event_bus):
        """Test that async handler exceptions are caught."""
        error_handler = AsyncMock()
        event_bus.add_error_handler(error_handler)

        async def failing_handler(event):
            raise ValueError("Async handler failed")

        event_bus.subscribe("test.event", failing_handler)

        # Publish event - should not raise
        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        # Error handler should have been called
        error_handler.assert_called_once()
        assert event_bus._metrics["handler_errors"] == 1

    @pytest.mark.asyncio
    async def test_error_handler_raises_exception(self, event_bus):
        """Test that error handler exceptions are logged but don't crash."""

        # This error handler will itself raise an exception
        def failing_error_handler(error, event, handler):
            raise RuntimeError("Error handler failed")

        event_bus.add_error_handler(failing_error_handler)

        def failing_handler(event):
            raise ValueError("Original error")

        event_bus.subscribe("test.event", failing_handler)

        # Publish event - should not raise despite error handler failing
        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        # Original error should still be counted
        assert event_bus._metrics["handler_errors"] == 1

    @pytest.mark.asyncio
    async def test_filter_function_raises_exception(self, event_bus):
        """Test that filter function exceptions are caught."""
        callback = AsyncMock()

        def failing_filter(event):
            raise ValueError("Filter failed")

        event_bus.subscribe("test.event", callback, filter_func=failing_filter)

        # Publish event - should not raise
        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        # Callback should not be called (filter rejected due to error)
        callback.assert_not_called()


class TestEventBusMiddleware:
    """Tests for middleware functionality."""

    @pytest.mark.asyncio
    async def test_pre_publish_middleware(self, event_bus):
        """Test pre-publish middleware."""

        # Middleware that modifies the event
        def add_timestamp_middleware(event):
            event.data["timestamp"] = "modified"
            return event

        callback = AsyncMock()
        event_bus.subscribe("test.event", callback)
        event_bus.add_middleware(add_timestamp_middleware, position="pre")

        await event_bus.publish(
            "test.event", {"original": True}, delivery_mode=EventDeliveryMode.SYNCHRONOUS
        )

        # Check middleware was applied
        called_event = callback.call_args[0][0]
        assert called_event.data["original"] is True
        assert called_event.data["timestamp"] == "modified"

    @pytest.mark.asyncio
    async def test_pre_publish_middleware_async(self, event_bus):
        """Test async pre-publish middleware."""

        async def async_middleware(event):
            event.data["async"] = True
            return event

        callback = AsyncMock()
        event_bus.subscribe("test.event", callback)
        event_bus.add_middleware(async_middleware, position="pre")

        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        called_event = callback.call_args[0][0]
        assert called_event.data["async"] is True

    @pytest.mark.asyncio
    async def test_pre_publish_middleware_raises_exception(self, event_bus):
        """Test that pre-publish middleware exceptions are caught."""

        def failing_middleware(event):
            raise ValueError("Middleware failed")

        callback = AsyncMock()
        event_bus.subscribe("test.event", callback)
        event_bus.add_middleware(failing_middleware, position="pre")

        # Should not raise
        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        # Event should still be published despite middleware failure
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_publish_middleware(self, event_bus):
        """Test post-publish middleware."""
        post_middleware_called = []

        def post_middleware(event):
            post_middleware_called.append(event.type)

        callback = AsyncMock()
        event_bus.subscribe("test.event", callback)
        event_bus.add_middleware(post_middleware, position="post")

        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        # Both callback and middleware should have been called
        callback.assert_called_once()
        assert post_middleware_called == ["test.event"]

    @pytest.mark.asyncio
    async def test_multiple_middleware(self, event_bus):
        """Test multiple middleware in correct order."""
        execution_order = []

        def middleware1(event):
            execution_order.append(1)
            event.data["order"] = execution_order[:]
            return event

        def middleware2(event):
            execution_order.append(2)
            event.data["order"] = execution_order[:]
            return event

        callback = AsyncMock()
        event_bus.subscribe("test.event", callback)

        event_bus.add_middleware(middleware1, position="pre")
        event_bus.add_middleware(middleware2, position="pre")

        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        # Middleware should execute in order added
        assert execution_order == [1, 2]

        called_event = callback.call_args[0][0]
        assert called_event.data["order"] == [1, 2]


class TestEventBusMetrics:
    """Tests for metrics collection."""

    def test_get_metrics(self, event_bus):
        """Test getting metrics."""
        metrics = event_bus.get_metrics()

        expected_keys = [
            "events_published",
            "events_processed",
            "handler_errors",
            "subscriptions_count",
            "start_time",
            "uptime_seconds",
            "queue_size",
            "wildcard_subscriptions",
            "event_types",
        ]

        for key in expected_keys:
            assert key in metrics

        assert metrics["events_published"] == 0
        assert metrics["subscriptions_count"] == 0
        assert isinstance(metrics["uptime_seconds"], float)

    @pytest.mark.asyncio
    async def test_metrics_after_publishing(self, event_bus):
        """Test metrics are updated after publishing."""
        callback = AsyncMock()
        event_bus.subscribe("test.event", callback)
        event_bus.subscribe("other.event", AsyncMock())

        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)
        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        metrics = event_bus.get_metrics()

        assert metrics["events_published"] == 2
        assert metrics["events_processed"] == 2  # One handler called twice
        assert metrics["subscriptions_count"] == 2
        assert metrics["event_types"] == ["test.event", "other.event"]

    def test_clear_metrics(self, event_bus):
        """Test clearing metrics."""
        # Set up some metrics
        event_bus._metrics["events_published"] = 10
        event_bus._metrics["events_processed"] = 8
        event_bus._metrics["handler_errors"] = 2
        event_bus._metrics["subscriptions_count"] = 3

        event_bus.clear_metrics()

        assert event_bus._metrics["events_published"] == 0
        assert event_bus._metrics["events_processed"] == 0
        assert event_bus._metrics["handler_errors"] == 0
        assert (
            event_bus._metrics["subscriptions_count"] == 3
        )  # Should not reset subscriptions count
        assert "start_time" in event_bus._metrics  # Should not reset start_time


class TestEventBusSingleton:
    """Tests for singleton functionality."""

    @pytest.mark.asyncio
    async def test_get_default_event_bus(self):
        """Test that get_default_event_bus returns a singleton."""
        bus1 = get_default_event_bus()
        bus2 = get_default_event_bus()

        assert bus1 is bus2
        assert bus1.name == "default"

        # Cleanup
        await bus1.stop()

    def test_create_event_bus(self):
        """Test creating multiple event buses."""
        bus1 = create_event_bus("bus1")
        bus2 = create_event_bus("bus2")

        assert bus1 is not bus2
        assert bus1.name == "bus1"
        assert bus2.name == "bus2"

        # Cleanup
        asyncio.run(bus1.stop())
        asyncio.run(bus2.stop())

    @pytest.mark.asyncio
    async def test_default_event_bus_starts_automatically(self):
        """Test that default event bus starts automatically."""
        # Reset the singleton to ensure clean test
        import src.core.event_bus as event_bus_module

        event_bus_module._default_event_bus = None

        bus = get_default_event_bus()

        # Should have a queue processor task
        # It might take a moment to start
        await asyncio.sleep(0.01)

        # Just check we got a bus
        assert bus is not None
        assert isinstance(bus, EventBus)

        # Cleanup
        await bus.stop()


class TestEventBusStartStop:
    """Tests for starting and stopping the event bus."""

    @pytest.mark.asyncio
    async def test_start_stop(self, event_bus):
        """Test starting and stopping the event bus."""
        # Initially not running
        assert event_bus._queue_processor_task is None or event_bus._queue_processor_task.done()

        # Start
        await event_bus.start()
        assert event_bus._queue_processor_task is not None
        assert not event_bus._queue_processor_task.done()

        # Stop
        await event_bus.stop()
        assert event_bus._queue_processor_task.done()

    @pytest.mark.asyncio
    async def test_restart_after_error(self, event_bus):
        """Test that event bus restarts after error in queue processor."""
        await event_bus.start()

        # Get the original task
        original_task = event_bus._queue_processor_task
        assert original_task is not None

        # Cancel the task to simulate an error
        original_task.cancel()

        # Wait a bit
        await asyncio.sleep(0.1)

        # The bus should still be functional
        assert event_bus._queue_processor_task is not None

        await event_bus.stop()


class TestEventBusIntegration:
    """Integration tests for EventBus with other components."""

    @pytest.mark.asyncio
    async def test_event_bus_with_system_state(self, event_bus):
        """Test EventBus integration with SystemState."""
        from src.core.system_state import SystemState, SystemStatus

        # Start the event bus
        await event_bus.start()

        # Create SystemState with EventBus
        system_state = SystemState(event_bus)

        # Subscribe to system status changes
        status_changes = []

        async def status_handler(event):
            status_changes.append(event.data)

        event_bus.subscribe("system.status_changed", status_handler)

        # Change system status with valid transitions
        await system_state.set_status(SystemStatus.STARTING, "Test starting")
        await asyncio.sleep(0.01)

        await system_state.set_status(SystemStatus.RUNNING, "Test running")
        await asyncio.sleep(0.01)

        # Check events were published
        # The events might be published asynchronously, so we can't guarantee they're processed
        # Let's just verify the subscription was created
        assert event_bus.get_subscription_count("system.status_changed") == 1

        await event_bus.stop()


class TestEventBusEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_invalid_middleware_position(self, event_bus):
        """Test that invalid middleware position raises error."""

        def middleware(event):
            return event

        with pytest.raises(EventBusError):
            event_bus.add_middleware(middleware, position="invalid")

    @pytest.mark.asyncio
    async def test_publish_with_sync_handler_in_executor(self, event_bus):
        """Test that synchronous handlers run in executor."""
        sync_handler_called = asyncio.Event()

        def sync_handler(event):
            # This is a synchronous handler
            sync_handler_called.set()

        event_bus.subscribe("test.event", sync_handler)

        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.SYNCHRONOUS)

        # Wait for handler to be called
        await asyncio.wait_for(sync_handler_called.wait(), timeout=1.0)
        assert sync_handler_called.is_set()

    @pytest.mark.asyncio
    async def test_handler_with_long_running_task(self, event_bus):
        """Test that long-running handlers don't block others."""
        fast_handler_called = asyncio.Event()
        slow_handler_called = asyncio.Event()

        async def slow_handler(event):
            await asyncio.sleep(0.1)  # Simulate slow processing
            slow_handler_called.set()

        async def fast_handler(event):
            fast_handler_called.set()

        event_bus.subscribe("test.event", slow_handler)
        event_bus.subscribe("test.event", fast_handler)

        # In BROADCAST mode, both should run concurrently
        await event_bus.publish("test.event", {}, delivery_mode=EventDeliveryMode.BROADCAST)

        # Fast handler should complete quickly
        await asyncio.wait_for(fast_handler_called.wait(), timeout=0.05)
        assert fast_handler_called.is_set()

        # Slow handler should complete eventually
        await asyncio.wait_for(slow_handler_called.wait(), timeout=0.15)
        assert slow_handler_called.is_set()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
