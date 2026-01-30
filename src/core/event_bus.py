"""
EventBus - Sistema de eventos para comunicación desacoplada entre componentes.
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import asyncio
from dataclasses import dataclass
from datetime import datetime
from .exceptions import BrainException

class EventType(Enum):
    """Tipos de eventos del sistema."""
    # Sistema
    SYSTEM_STARTED = "system_started"
    SYSTEM_SHUTDOWN_STARTED = "system_shutdown_started"
    SYSTEM_SHUTDOWN_COMPLETED = "system_shutdown_completed"
    SYSTEM_ERROR = "system_error"
    
    # Operaciones
    OPERATION_STARTED = "operation_started"
    OPERATION_COMPLETED = "operation_completed"
    OPERATION_FAILED = "operation_failed"
    OPERATION_CANCELLED = "operation_cancelled"
    OPERATION_PROGRESS = "operation_progress"
    
    # Análisis
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    
    # Agentes
    AGENT_CREATED = "agent_created"
    AGENT_DESTROYED = "agent_destroyed"
    AGENT_ERROR = "agent_error"
    
    # Aprendizaje
    LEARNING_STARTED = "learning_started"
    LEARNING_COMPLETED = "learning_completed"
    KNOWLEDGE_UPDATED = "knowledge_updated"

@dataclass
class Event:
    """Evento del sistema."""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    source: Optional[str] = None

class EventBus:
    """Bus de eventos para comunicación entre componentes."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._is_running = False
        self._processor_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Inicializa el bus de eventos."""
        self._is_running = True
        self._processor_task = asyncio.create_task(self._process_events())
    
    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Suscribe un callback a un tipo de evento."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Desuscribe un callback de un tipo de evento."""
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
    
    async def publish(self, event_type: EventType, data: Dict[str, Any], 
                     source: Optional[str] = None) -> None:
        """Publica un evento."""
        event = Event(
            type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source
        )
        
        await self._event_queue.put(event)
    
    async def publish_async(self, event_type: EventType, data: Dict[str, Any],
                           source: Optional[str] = None) -> None:
        """Publica un evento de manera asíncrona (sin esperar)."""
        asyncio.create_task(self.publish(event_type, data, source))
    
    def get_subscribers(self, event_type: EventType) -> List[Callable]:
        """Obtiene los suscriptores de un tipo de evento."""
        return self._subscribers.get(event_type, [])
    
    def clear_subscriptions(self, event_type: Optional[EventType] = None) -> None:
        """Limpia suscripciones."""
        if event_type is None:
            self._subscribers.clear()
        elif event_type in self._subscribers:
            self._subscribers[event_type].clear()
    
    async def shutdown(self) -> None:
        """Apaga el bus de eventos."""
        self._is_running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
    
    async def _process_events(self) -> None:
        """Procesa eventos de la cola."""
        while self._is_running:
            try:
                event = await self._event_queue.get()
                
                # Notificar a suscriptores
                subscribers = self._subscribers.get(event.type, [])
                
                for callback in subscribers:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        # Log error but don't crash
                        print(f"Error in event subscriber: {e}")
                
                self._event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing event: {e}")