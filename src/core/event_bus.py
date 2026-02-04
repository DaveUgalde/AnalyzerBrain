"""
Event Bus for ANALYZERBRAIN.

Provides an asynchronous event-driven communication system for decoupled component interaction.
Supports event publishing, subscription, middleware, and filtering.

Dependencies:
    - asyncio: For asynchronous operations
    - loguru: For structured logging
    - typing: For type hints

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

from .exceptions import EventBusError


class EventPriority(Enum):
    """Priority levels for event processing."""

    HIGH = 100
    NORMAL = 50
    LOW = 10
    BACKGROUND = 1


class EventDeliveryMode(Enum):
    """Event delivery modes."""

    SYNCHRONOUS = "synchronous"  # Wait for all handlers
    ASYNCHRONOUS = "asynchronous"  # Fire and forget
    BROADCAST = "broadcast"  # Send to all handlers concurrently


@dataclass
class Event:
    """Event data structure."""

    type: str
    data: Dict[str, Any]
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate correlation ID if not provided."""
        if not self.correlation_id:
            self.correlation_id = str(uuid4())


@dataclass
class Subscription:
    """Subscription information."""

    event_type: str
    callback: Callable
    priority: int
    filter_func: Optional[Callable[[Event], bool]] = None
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)


class EventBus:
    """
    Asynchronous Event Bus for decoupled component communication.

    Features:
    - Type-safe event subscriptions
    - Priority-based handler execution
    - Event filtering
    - Middleware support
    - Error handling and recovery
    - Metrics collection
    """

    def __init__(self, name: str = "default"):
        """
        Initialize EventBus.

        Args:
            name: Name identifier for this event bus instance
        """
        self.name = name

        # Subscription storage
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._wildcard_subscriptions: List[Subscription] = []

        # Middleware
        self._pre_publish_middleware: List[Callable] = []
        self._post_publish_middleware: List[Callable] = []

        # Statistics
        self._metrics: Dict[str, Any] = {
            "events_published": 0,
            "events_processed": 0,
            "handler_errors": 0,
            "subscriptions_count": 0,
            "start_time": datetime.now(),
        }

        # Event queue for async processing
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._queue_processor_task: Optional[asyncio.Task] = None

        # Error handlers
        self._error_handlers: List[Callable] = []

        logger.info(f"EventBus '{name}' initialized")

    async def start(self) -> None:
        """Start the event bus queue processor."""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(self._process_event_queue())
            logger.info(f"EventBus '{self.name}' queue processor started")

    async def stop(self) -> None:
        """Stop the event bus queue processor."""
        if self._queue_processor_task and not self._queue_processor_task.done():
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
            logger.info(f"EventBus '{self.name}' queue processor stopped")

    def subscribe(
        self,
        event_type: Union[str, List[str]],
        callback: Callable,
        priority: int = EventPriority.NORMAL.value,
        filter_func: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """
        Subscribe to one or more event types.

        Args:
            event_type: Event type or list of event types to subscribe to.
                        Use "*" for wildcard subscription.
            callback: Callback function to invoke when event is published.
            priority: Handler priority (higher = executed earlier).
            filter_func: Optional filter function to accept/reject events.

        Returns:
            Subscription ID for unsubscription.

        Raises:
            EventBusError: If callback is not callable.
        """
        if not callable(callback):
            raise EventBusError("Callback must be callable")

        subscription = Subscription(
            event_type=event_type if isinstance(event_type, str) else ",".join(event_type),
            callback=callback,
            priority=priority,
            filter_func=filter_func,
        )

        if event_type == "*":
            self._wildcard_subscriptions.append(subscription)
        elif isinstance(event_type, str):
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []
            self._subscriptions[event_type].append(subscription)
        elif isinstance(event_type, list):
            for et in event_type:
                if et not in self._subscriptions:
                    self._subscriptions[et] = []
                self._subscriptions[et].append(subscription)

        # Sort by priority (highest first)
        if event_type != "*" and isinstance(event_type, str):
            self._subscriptions[event_type].sort(key=lambda s: s.priority, reverse=True)

        self._metrics["subscriptions_count"] += 1
        logger.debug(f"Subscription created: {event_type} (priority: {priority})")

        return subscription.id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe using subscription ID.

        Args:
            subscription_id: ID returned by subscribe().

        Returns:
            True if unsubscribed, False if not found.
        """
        # Search in wildcard subscriptions
        for i, sub in enumerate(self._wildcard_subscriptions):
            if sub.id == subscription_id:
                self._wildcard_subscriptions.pop(i)
                self._metrics["subscriptions_count"] -= 1
                logger.debug(f"Wildcard subscription removed: {subscription_id}")
                return True

        # Search in typed subscriptions
        for event_type, subscriptions in self._subscriptions.items():
            for i, sub in enumerate(subscriptions):
                if sub.id == subscription_id:
                    subscriptions.pop(i)
                    self._metrics["subscriptions_count"] -= 1

                    # Clean up empty event type
                    if not subscriptions:
                        del self._subscriptions[event_type]

                    logger.debug(f"Subscription removed: {event_type} - {subscription_id}")
                    return True

        logger.warning(f"Subscription not found: {subscription_id}")
        return False

    def unsubscribe_all(self, event_type: Optional[str] = None) -> int:
        """
        Unsubscribe all handlers for an event type, or all handlers if no type specified.

        Args:
            event_type: Event type to unsubscribe from. If None, unsubscribe all.

        Returns:
            Number of subscriptions removed.
        """
        count = 0

        if event_type is None:
            # Unsubscribe all
            count = self._metrics["subscriptions_count"]
            self._subscriptions.clear()
            self._wildcard_subscriptions.clear()
            self._metrics["subscriptions_count"] = 0
            logger.info(f"All subscriptions removed: {count} total")
        elif event_type == "*":
            # Unsubscribe wildcard only
            count = len(self._wildcard_subscriptions)
            self._wildcard_subscriptions.clear()
            self._metrics["subscriptions_count"] -= count
            logger.info(f"Wildcard subscriptions removed: {count}")
        elif event_type in self._subscriptions:
            # Unsubscribe specific event type
            count = len(self._subscriptions[event_type])
            del self._subscriptions[event_type]
            self._metrics["subscriptions_count"] -= count
            logger.info(f"Subscriptions removed for {event_type}: {count}")

        return count

    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
        delivery_mode: EventDeliveryMode = EventDeliveryMode.ASYNCHRONOUS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event to publish.
            data: Event payload data.
            source: Optional source identifier.
            correlation_id: Optional correlation ID for tracking.
            priority: Event priority level.
            delivery_mode: How to deliver the event.
            metadata: Optional additional metadata.

        Returns:
            The published Event object.
        """
        # Create event
        event = Event(
            type=event_type,
            data=data,
            source=source,
            correlation_id=correlation_id,
            priority=priority,
            metadata=metadata or {},
        )

        # Apply pre-publish middleware
        for middleware in self._pre_publish_middleware:
            try:
                event = (
                    await middleware(event)
                    if asyncio.iscoroutinefunction(middleware)
                    else middleware(event)
                )
            except Exception as e:
                logger.error(f"Pre-publish middleware error: {e}")

        # Update metrics
        self._metrics["events_published"] += 1

        # Process based on delivery mode
        if delivery_mode == EventDeliveryMode.SYNCHRONOUS:
            await self._process_event_sync(event)
        elif delivery_mode == EventDeliveryMode.ASYNCHRONOUS:
            asyncio.create_task(self._process_event_async(event))
        elif delivery_mode == EventDeliveryMode.BROADCAST:
            await self._process_event_broadcast(event)

        # Apply post-publish middleware
        for middleware in self._post_publish_middleware:
            try:
                (
                    await middleware(event)
                    if asyncio.iscoroutinefunction(middleware)
                    else middleware(event)
                )
            except Exception as e:
                logger.error(f"Post-publish middleware error: {e}")

        logger.debug(f"Event published: {event_type} (correlation: {event.correlation_id})")

        return event

    async def _process_event_sync(self, event: Event) -> None:
        """Process event synchronously."""
        handlers = self._get_handlers_for_event(event)

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler.callback):
                    await handler.callback(event)
                else:
                    handler.callback(event)
                self._metrics["events_processed"] += 1
            except Exception as e:
                self._metrics["handler_errors"] += 1
                await self._handle_error(e, event, handler)

    async def _process_event_async(self, event: Event) -> None:
        """Process event asynchronously (fire and forget)."""
        try:
            await self._event_queue.put(event)
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event.type}")

    async def _process_event_broadcast(self, event: Event) -> None:
        """Process event with all handlers concurrently."""
        handlers = self._get_handlers_for_event(event)

        tasks = []
        for handler in handlers:
            tasks.append(self._execute_handler(handler, event))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_handler(self, handler: Subscription, event: Event) -> None:
        """Execute a single handler with error handling."""
        try:
            if asyncio.iscoroutinefunction(handler.callback):
                await handler.callback(event)
            else:
                # Run sync handler in executor to avoid blocking
                await asyncio.get_event_loop().run_in_executor(None, handler.callback, event)
            self._metrics["events_processed"] += 1
        except Exception as e:
            self._metrics["handler_errors"] += 1
            await self._handle_error(e, event, handler)

    def _get_handlers_for_event(self, event: Event) -> List[Subscription]:
        """Get all handlers for an event, applying filters."""
        handlers = []

        # Get typed handlers
        if event.type in self._subscriptions:
            for subscription in self._subscriptions[event.type]:
                if self._filter_matches(subscription, event):
                    handlers.append(subscription)

        # Get wildcard handlers
        for subscription in self._wildcard_subscriptions:
            if self._filter_matches(subscription, event):
                handlers.append(subscription)

        # Sort by priority (highest first)
        handlers.sort(key=lambda h: h.priority, reverse=True)

        return handlers

    def _filter_matches(self, subscription: Subscription, event: Event) -> bool:
        """Check if event passes subscription filter."""
        if subscription.filter_func:
            try:
                return subscription.filter_func(event)
            except Exception as e:
                logger.error(f"Filter function error: {e}")
                return False
        return True

    async def _handle_error(self, error: Exception, event: Event, handler: Subscription) -> None:
        """Handle handler execution error."""
        error_msg = f"Handler error for event {event.type}: {error}"
        logger.error(error_msg)

        # Call error handlers
        for error_handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(error_handler):
                    await error_handler(error, event, handler)
                else:
                    error_handler(error, event, handler)
            except Exception as e:
                logger.error(f"Error handler failed: {e}")

    async def _process_event_queue(self) -> None:
        """Process events from the queue."""
        logger.info("Event queue processor started")

        try:
            while True:
                try:
                    event = await self._event_queue.get()

                    handlers = self._get_handlers_for_event(event)

                    # Execute handlers sequentially
                    for handler in handlers:
                        await self._execute_handler(handler, event)

                    self._event_queue.task_done()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Queue processor error: {e}")

        except asyncio.CancelledError:
            logger.info("Event queue processor cancelled")
        except Exception as e:
            logger.error(f"Queue processor fatal error: {e}")
            # Restart processor
            await self.start()

    def add_middleware(self, middleware: Callable, position: str = "pre") -> None:
        """
        Add middleware to event processing pipeline.

        Args:
            middleware: Middleware function taking Event and returning Event (pre) or None (post).
            position: "pre" (before handlers) or "post" (after handlers).
        """
        if position == "pre":
            self._pre_publish_middleware.append(middleware)
        elif position == "post":
            self._post_publish_middleware.append(middleware)
        else:
            raise EventBusError(f"Invalid middleware position: {position}")

    def add_error_handler(self, error_handler: Callable) -> None:
        """Add error handler for failed event processing."""
        self._error_handlers.append(error_handler)

    def get_subscription_count(self, event_type: Optional[str] = None) -> int:
        """Get number of subscriptions for an event type or total."""
        if event_type is None:
            return self._metrics["subscriptions_count"]
        elif event_type == "*":
            return len(self._wildcard_subscriptions)
        else:
            return len(self._subscriptions.get(event_type, []))

    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        metrics = self._metrics.copy()
        metrics["uptime_seconds"] = (datetime.now() - metrics["start_time"]).total_seconds()
        metrics["queue_size"] = self._event_queue.qsize()
        metrics["wildcard_subscriptions"] = len(self._wildcard_subscriptions)
        metrics["event_types"] = list(self._subscriptions.keys())

        return metrics

    def clear_metrics(self) -> None:
        """Clear all metrics (except start_time)."""
        self._metrics = {
            "events_published": 0,
            "events_processed": 0,
            "handler_errors": 0,
            "subscriptions_count": self._metrics["subscriptions_count"],
            "start_time": self._metrics["start_time"],
        }

    def __str__(self) -> str:
        """String representation."""
        return f"EventBus(name={self.name}, subscriptions={self._metrics['subscriptions_count']})"


# Singleton instance for default event bus
_default_event_bus: Optional[EventBus] = None


def get_default_event_bus() -> EventBus:
    """
    Get or create the default singleton event bus.

    Returns:
        Default EventBus instance.
    """
    global _default_event_bus

    if _default_event_bus is None:
        _default_event_bus = EventBus(name="default")
        asyncio.create_task(_default_event_bus.start())

    return _default_event_bus


def create_event_bus(name: str) -> EventBus:
    """
    Create a new named event bus.

    Args:
        name: Unique name for the event bus.

    Returns:
        New EventBus instance.
    """
    return EventBus(name=name)
