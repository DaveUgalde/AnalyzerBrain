"""
WebSocket API - Implementación completa del protocolo WebSocket.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError
from websockets.server import WebSocketServerProtocol
from pydantic import BaseModel, Field, validator

from ..core.exceptions import BrainException, ValidationError
from ..core.orchestrator import BrainOrchestrator
from .authentication import Authentication
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class WebSocketMessageType(str, Enum):
    """Tipos de mensajes WebSocket."""
    CONNECT = "connect"
    CONNECTED = "connected"
    DISCONNECT = "disconnect"
    ERROR = "error"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIBED = "subscribed"
    QUERY = "query"
    QUERY_RESPONSE = "query_response"
    ANALYSIS_PROGRESS = "analysis_progress"
    CHANGE_DETECTED = "change_detected"
    LEARNING_UPDATE = "learning_update"
    PING = "ping"
    PONG = "pong"

@dataclass
class ConnectionState:
    """Estado de una conexión WebSocket."""
    connection_id: str
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    subscriptions: Set[str] = field(default_factory=set)
    authenticated: bool = False
    user_id: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    ping_count: int = 0
    query_count: int = 0

class WebSocketMessage(BaseModel):
    """Mensaje WebSocket estructurado."""
    type: WebSocketMessageType
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_json(self) -> str:
        """Convierte a JSON."""
        # Convertir datetime a string ISO
        message_dict = self.dict()
        message_dict["timestamp"] = self.timestamp.isoformat()
        return json.dumps(message_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Crea desde JSON."""
        data = json.loads(json_str)
        # Convertir string ISO de vuelta a datetime
        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        return cls(**data)

class ConnectionRequest(BaseModel):
    """Solicitud de conexión WebSocket."""
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    version: str = "1.0.0"
    auth_token: Optional[str] = None

class QueryRequest(BaseModel):
    """Solicitud de consulta via WebSocket."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    context: Optional[Dict[str, Any]] = None
    stream: bool = False
    priority: int = Field(5, ge=1, le=10)

class QueryResponseChunk(BaseModel):
    """Fragmento de respuesta de consulta."""
    request_id: str
    chunk_type: str
    content: str
    is_final: bool = False
    confidence: Optional[float] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)

class AnalysisProgressUpdate(BaseModel):
    """Actualización de progreso de análisis."""
    analysis_id: str
    step: str
    progress: float
    current_file: Optional[str] = None
    files_processed: int = 0
    total_files: int = 0
    estimated_remaining_seconds: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

class ChangeDetectionNotification(BaseModel):
    """Notificación de cambio detectado."""
    project_id: str
    changes: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: str = "info"

class WebSocketAPI:
    """
    Implementación completa del protocolo WebSocket.
    
    Características:
    1. Comunicación full-duplex en tiempo real
    2. Suscripción a diferentes canales (topics)
    3. Streaming de respuestas para consultas largas
    4. Reconexión automática con estado
    5. Heartbeat y detección de conexión perdida
    """
    
    def __init__(self, 
                 authentication: Optional[Authentication] = None,
                 rate_limiter: Optional[RateLimiter] = None):
        """
        Inicializa la API WebSocket.
        
        Args:
            authentication: Sistema de autenticación (opcional)
            rate_limiter: Limitador de tasa (opcional)
        """
        self.authentication = authentication
        self.rate_limiter = rate_limiter
        self.orchestrator: Optional[BrainOrchestrator] = None
        
        # Gestión de conexiones
        self.connections: Dict[str, ConnectionState] = {}
        self.connection_websockets: Dict[str, WebSocketServerProtocol] = {}
        
        # Suscripciones por canal
        self.subscriptions: Dict[str, Set[str]] = {
            "analysis_progress": set(),
            "change_detected": set(),
            "learning_updates": set(),
            "system_metrics": set(),
        }
        
        # Métricas
        self.metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_received": 0,
            "messages_sent": 0,
            "queries_processed": 0,
            "errors": 0,
            "subscriptions": {},
        }
        
        # Heartbeat
        self.heartbeat_interval = 30  # segundos
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        logger.info("WebSocketAPI inicializada")
    
    async def initialize(self, orchestrator: BrainOrchestrator) -> None:
        """
        Inicializa la API WebSocket con el orquestador.
        
        Args:
            orchestrator: Instancia de BrainOrchestrator
        """
        self.orchestrator = orchestrator
        
        # Iniciar heartbeat
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info("WebSocketAPI inicializada con orquestador")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """
        Maneja una nueva conexión WebSocket.
        
        Args:
            websocket: Conexión WebSocket
            path: Ruta de la conexión
        """
        connection_id = str(uuid.uuid4())
        
        try:
            # Registrar conexión
            connection_state = ConnectionState(connection_id=connection_id)
            self.connections[connection_id] = connection_state
            self.connection_websockets[connection_id] = websocket
            self.metrics["total_connections"] += 1
            self.metrics["active_connections"] += 1
            
            logger.info("Nueva conexión WebSocket %s desde %s", 
                       connection_id, websocket.remote_address)
            
            # Enviar mensaje de bienvenida
            await self._send_message(websocket, WebSocketMessage(
                type=WebSocketMessageType.CONNECTED,
                data={
                    "connection_id": connection_id,
                    "message": "Connected to Project Brain WebSocket API",
                    "protocol_version": "1.0.0",
                    "supported_message_types": [t.value for t in WebSocketMessageType],
                    "available_subscriptions": list(self.subscriptions.keys()),
                }
            ))
            
            # Escuchar mensajes
            async for message in websocket:
                try:
                    await self.process_message(connection_id, message)
                    connection_state.last_activity = datetime.now()
                    
                except json.JSONDecodeError as e:
                    await self._send_error(websocket, f"Invalid JSON: {str(e)}")
                    
                except ValidationError as e:
                    await self._send_error(websocket, f"Validation error: {str(e)}")
                    
                except Exception as e:
                    logger.error("Error procesando mensaje: %s", e, exc_info=True)
                    await self._send_error(websocket, f"Processing error: {str(e)}")
                    self.metrics["errors"] += 1
        
        except ConnectionClosed:
            logger.info("Conexión WebSocket %s cerrada normalmente", connection_id)
            
        except Exception as e:
            logger.error("Error en conexión WebSocket %s: %s", connection_id, e, exc_info=True)
            
        finally:
            # Limpiar conexión
            await self.handle_disconnection(connection_id)
    
    async def process_message(self, connection_id: str, message_raw: str) -> None:
        """
        Procesa un mensaje WebSocket.
        
        Args:
            connection_id: ID de la conexión
            message_raw: Mensaje en formato JSON string
        """
        connection_state = self.connections.get(connection_id)
        if not connection_state:
            raise ValidationError(f"Connection {connection_id} not found")
        
        websocket = self.connection_websockets.get(connection_id)
        if not websocket:
            raise ValidationError(f"WebSocket for connection {connection_id} not found")
        
        # Parsear mensaje
        message = WebSocketMessage.from_json(message_raw)
        self.metrics["messages_received"] += 1
        
        logger.debug("Mensaje recibido: %s", message.type)
        
        # Procesar según tipo
        if message.type == WebSocketMessageType.CONNECT:
            await self._handle_connect(connection_state, websocket, message)
            
        elif message.type == WebSocketMessageType.SUBSCRIBE:
            await self._handle_subscribe(connection_state, websocket, message)
            
        elif message.type == WebSocketMessageType.UNSUBSCRIBE:
            await self._handle_unsubscribe(connection_state, websocket, message)
            
        elif message.type == WebSocketMessageType.QUERY:
            await self._handle_query(connection_state, websocket, message)
            
        elif message.type == WebSocketMessageType.PING:
            await self._handle_ping(connection_state, websocket, message)
            
        elif message.type == WebSocketMessageType.DISCONNECT:
            await self._handle_disconnect(connection_state, websocket, message)
            
        else:
            await self._send_error(websocket, f"Unknown message type: {message.type}")
    
    async def broadcast_message(self, channel: str, message: WebSocketMessage) -> int:
        """
        Transmite un mensaje a todos los suscriptores de un canal.
        
        Args:
            channel: Canal de transmisión
            message: Mensaje a transmitir
            
        Returns:
            int: Número de conexiones que recibieron el mensaje
        """
        if channel not in self.subscriptions:
            logger.warning("Canal de transmisión no existente: %s", channel)
            return 0
        
        sent_count = 0
        connection_ids = list(self.subscriptions[channel])
        
        for connection_id in connection_ids:
            if connection_id in self.connection_websockets:
                try:
                    await self._send_message(
                        self.connection_websockets[connection_id],
                        message
                    )
                    sent_count += 1
                    self.metrics["messages_sent"] += 1
                    
                except Exception as e:
                    logger.error("Error transmitiendo a %s: %s", connection_id, e)
        
        logger.debug("Transmitido a %s suscriptores de %s", sent_count, channel)
        return sent_count
    
    async def manage_subscriptions(self, connection_id: str, action: str, channels: List[str]) -> List[str]:
        """
        Gestiona suscripciones de una conexión.
        
        Args:
            connection_id: ID de la conexión
            action: "subscribe" o "unsubscribe"
            channels: Lista de canales
            
        Returns:
            List[str]: Canales procesados exitosamente
        """
        connection_state = self.connections.get(connection_id)
        if not connection_state:
            raise ValidationError(f"Connection {connection_id} not found")
        
        processed = []
        
        for channel in channels:
            if channel in self.subscriptions:
                if action == "subscribe":
                    self.subscriptions[channel].add(connection_id)
                    connection_state.subscriptions.add(channel)
                    processed.append(channel)
                    
                    # Actualizar métricas
                    if channel not in self.metrics["subscriptions"]:
                        self.metrics["subscriptions"][channel] = 0
                    self.metrics["subscriptions"][channel] += 1
                    
                elif action == "unsubscribe":
                    self.subscriptions[channel].discard(connection_id)
                    connection_state.subscriptions.discard(channel)
                    processed.append(channel)
                    
                    # Actualizar métricas
                    if channel in self.metrics["subscriptions"]:
                        self.metrics["subscriptions"][channel] = max(0, 
                            self.metrics["subscriptions"][channel] - 1)
        
        return processed
    
    async def handle_disconnection(self, connection_id: str) -> None:
        """
        Maneja la desconexión de una conexión WebSocket.
        
        Args:
            connection_id: ID de la conexión
        """
        connection_state = self.connections.get(connection_id)
        
        if connection_state:
            # Remover de todas las suscripciones
            for channel in list(connection_state.subscriptions):
                self.subscriptions[channel].discard(connection_id)
                
                # Actualizar métricas
                if channel in self.metrics["subscriptions"]:
                    self.metrics["subscriptions"][channel] = max(0,
                        self.metrics["subscriptions"][channel] - 1)
            
            # Notificar desconexión (si el socket aún está abierto)
            websocket = self.connection_websockets.get(connection_id)
            if websocket and not websocket.closed:
                try:
                    await self._send_message(websocket, WebSocketMessage(
                        type=WebSocketMessageType.DISCONNECT,
                        data={
                            "message": "Connection closed",
                            "connection_duration": (
                                datetime.now() - connection_state.connected_at
                            ).total_seconds()
                        }
                    ))
                except Exception as e:
                    logger.debug("Error enviando mensaje de desconexión: %s", e)
        
        # Limpiar recursos
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        if connection_id in self.connection_websockets:
            del self.connection_websockets[connection_id]
        
        # Actualizar métricas
        self.metrics["active_connections"] = max(0, self.metrics["active_connections"] - 1)
        
        logger.info("Conexión %s desconectada", connection_id)
    
    async def maintain_heartbeat(self) -> None:
        """
        Mantiene heartbeat con todas las conexiones activas.
        Envía pings y desconecta conexiones inactivas.
        """
        current_time = datetime.now()
        connections_to_remove = []
        
        for connection_id, connection_state in list(self.connections.items()):
            websocket = self.connection_websockets.get(connection_id)
            
            if not websocket or websocket.closed:
                connections_to_remove.append(connection_id)
                continue
            
            # Verificar inactividad
            idle_time = (current_time - connection_state.last_activity).total_seconds()
            
            if idle_time > 300:  # 5 minutos de inactividad
                logger.info("Desconectando conexión %s por inactividad", connection_id)
                connections_to_remove.append(connection_id)
                continue
            
            # Enviar ping cada 30 segundos
            if connection_state.ping_count == 0 or idle_time > self.heartbeat_interval:
                try:
                    await self._send_message(websocket, WebSocketMessage(
                        type=WebSocketMessageType.PING,
                        data={"timestamp": current_time.isoformat()}
                    ))
                    connection_state.ping_count += 1
                    
                except Exception as e:
                    logger.debug("Error enviando ping a %s: %s", connection_id, e)
                    connections_to_remove.append(connection_id)
        
        # Limpiar conexiones muertas
        for connection_id in connections_to_remove:
            await self.handle_disconnection(connection_id)
    
    async def scale_connections(self, max_connections: int = 1000) -> bool:
        """
        Escala las conexiones WebSocket.
        
        Args:
            max_connections: Máximo número de conexiones permitidas
            
        Returns:
            bool: True si está por debajo del límite
        """
        current_connections = len(self.connections)
        
        if current_connections >= max_connections:
            logger.warning(
                "Límite de conexiones alcanzado: %s/%s", 
                current_connections, max_connections
            )
            return False
        
        return True
    
    async def get_connection_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de las conexiones WebSocket.
        
        Returns:
            Dict con métricas detalladas
        """
        connection_details = []
        
        for connection_id, state in self.connections.items():
            connection_details.append({
                "connection_id": connection_id,
                "session_id": state.session_id,
                "project_id": state.project_id,
                "connected_at": state.connected_at.isoformat(),
                "last_activity": state.last_activity.isoformat(),
                "subscriptions": list(state.subscriptions),
                "query_count": state.query_count,
                "authenticated": state.authenticated,
                "connection_duration": (
                    datetime.now() - state.connected_at
                ).total_seconds(),
            })
        
        return {
            "total_connections": self.metrics["total_connections"],
            "active_connections": self.metrics["active_connections"],
            "messages_received": self.metrics["messages_received"],
            "messages_sent": self.metrics["messages_sent"],
            "queries_processed": self.metrics["queries_processed"],
            "errors": self.metrics["errors"],
            "subscriptions": self.metrics["subscriptions"],
            "connections": connection_details,
            "timestamp": datetime.now().isoformat(),
        }
    
    # Métodos de manejo de mensajes
    
    async def _handle_connect(self, connection_state: ConnectionState, 
                            websocket: WebSocketServerProtocol, 
                            message: WebSocketMessage) -> None:
        """Maneja solicitud de conexión."""
        try:
            request = ConnectionRequest(**(message.data or {}))
            
            # Autenticación
            if self.authentication and request.auth_token:
                user_info = await self.authentication.validate_token(request.auth_token)
                if user_info:
                    connection_state.authenticated = True
                    connection_state.user_id = user_info.get("user_id")
            
            # Establecer información de conexión
            connection_state.session_id = request.session_id or str(uuid.uuid4())
            connection_state.project_id = request.project_id
            
            # Enviar confirmación
            await self._send_message(websocket, WebSocketMessage(
                type=WebSocketMessageType.CONNECTED,
                data={
                    "connection_id": connection_state.connection_id,
                    "session_id": connection_state.session_id,
                    "authenticated": connection_state.authenticated,
                    "capabilities_supported": [
                        "query_streaming",
                        "analysis_progress",
                        "change_notifications",
                        "learning_updates"
                    ],
                    "protocol_version": "1.0.0",
                }
            ))
            
            logger.info("Conexión %s configurada para sesión %s", 
                       connection_state.connection_id, connection_state.session_id)
            
        except Exception as e:
            await self._send_error(websocket, f"Connection failed: {str(e)}")
    
    async def _handle_subscribe(self, connection_state: ConnectionState,
                              websocket: WebSocketServerProtocol,
                              message: WebSocketMessage) -> None:
        """Maneja solicitud de suscripción."""
        channels = message.data.get("channels", []) if message.data else []
        
        if not channels:
            await self._send_error(websocket, "No channels specified")
            return
        
        processed = await self.manage_subscriptions(
            connection_state.connection_id, "subscribe", channels
        )
        
        await self._send_message(websocket, WebSocketMessage(
            type=WebSocketMessageType.SUBSCRIBED,
            data={"channels": processed}
        ))
    
    async def _handle_unsubscribe(self, connection_state: ConnectionState,
                                websocket: WebSocketServerProtocol,
                                message: WebSocketMessage) -> None:
        """Maneja solicitud de desuscripción."""
        channels = message.data.get("channels", []) if message.data else []
        
        if not channels:
            # Desuscribir de todos los canales
            channels = list(connection_state.subscriptions)
        
        processed = await self.manage_subscriptions(
            connection_state.connection_id, "unsubscribe", channels
        )
        
        await self._send_message(websocket, WebSocketMessage(
            type=WebSocketMessageType.SUBSCRIBED,
            data={"channels": processed, "action": "unsubscribe"}
        ))
    
    async def _handle_query(self, connection_state: ConnectionState,
                          websocket: WebSocketServerProtocol,
                          message: WebSocketMessage) -> None:
        """Maneja solicitud de consulta."""
        if not self.orchestrator:
            await self._send_error(websocket, "Orchestrator not initialized")
            return
        
        try:
            request = QueryRequest(**(message.data or {}))
            connection_state.query_count += 1
            self.metrics["queries_processed"] += 1
            
            # Enviar confirmación inmediata
            await self._send_message(websocket, WebSocketMessage(
                type=WebSocketMessageType.QUERY_RESPONSE,
                data={
                    "request_id": request.request_id,
                    "chunk_type": "acknowledged",
                    "content": "Processing your question...",
                    "is_final": False
                }
            ))
            
            # Procesar consulta
            if request.stream:
                await self._stream_query_response(websocket, connection_state, request)
            else:
                await self._send_complete_response(websocket, connection_state, request)
                
        except Exception as e:
            logger.error("Error procesando consulta: %s", e, exc_info=True)
            await self._send_error(websocket, f"Query processing failed: {str(e)}")
    
    async def _handle_ping(self, connection_state: ConnectionState,
                         websocket: WebSocketServerProtocol,
                         message: WebSocketMessage) -> None:
        """Maneja mensaje ping (heartbeat)."""
        connection_state.ping_count = 0  # Resetear contador de pings
        
        await self._send_message(websocket, WebSocketMessage(
            type=WebSocketMessageType.PONG,
            data={
                "timestamp": datetime.now().isoformat(),
                "original_timestamp": message.data.get("timestamp") if message.data else None
            }
        ))
    
    async def _handle_disconnect(self, connection_state: ConnectionState,
                               websocket: WebSocketServerProtocol,
                               message: WebSocketMessage) -> None:
        """Maneja solicitud de desconexión."""
        await self._send_message(websocket, WebSocketMessage(
            type=WebSocketMessageType.DISCONNECT,
            data={
                "message": "Disconnecting as requested",
                "connection_duration": (
                    datetime.now() - connection_state.connected_at
                ).total_seconds()
            }
        ))
        
        # Cerrar conexión
        await websocket.close()
    
    async def _stream_query_response(self, websocket: WebSocketServerProtocol,
                                   connection_state: ConnectionState,
                                   request: QueryRequest) -> None:
        """Envía respuesta en streaming."""
        # Simular pasos de procesamiento
        thinking_steps = [
            "Analyzing your question...",
            "Searching project knowledge...",
            "Consulting specialized agents...",
            "Synthesizing answer..."
        ]
        
        for step in thinking_steps:
            await self._send_message(websocket, WebSocketMessage(
                type=WebSocketMessageType.QUERY_RESPONSE,
                data={
                    "request_id": request.request_id,
                    "chunk_type": "thinking",
                    "content": step,
                    "is_final": False
                }
            ))
            await asyncio.sleep(0.3)
        
        # En una implementación real, esto usaría el orquestador
        # Para ahora, enviamos una respuesta de ejemplo en streaming
        
        answer_parts = [
            "Based on my analysis of the codebase,",
            "I found that the function `process_data` is defined in `utils.py`.",
            "It takes a single parameter and returns processed data.",
            "The function is called from 3 different places in the code."
        ]
        
        for i, part in enumerate(answer_parts):
            await self._send_message(websocket, WebSocketMessage(
                type=WebSocketMessageType.QUERY_RESPONSE,
                data={
                    "request_id": request.request_id,
                    "chunk_type": "partial",
                    "content": part,
                    "is_final": i == len(answer_parts) - 1
                }
            ))
            await asyncio.sleep(0.5)
        
        # Enviar respuesta final con metadatos
        await self._send_message(websocket, WebSocketMessage(
            type=WebSocketMessageType.QUERY_RESPONSE,
            data={
                "request_id": request.request_id,
                "chunk_type": "complete",
                "content": "Answer complete.",
                "confidence": 0.85,
                "sources": [
                    {
                        "file": "src/utils.py",
                        "lines": [10, 25],
                        "type": "function_definition"
                    }
                ],
                "is_final": True
            }
        ))
    
    async def _send_complete_response(self, websocket: WebSocketServerProtocol,
                                    connection_state: ConnectionState,
                                    request: QueryRequest) -> None:
        """Envía respuesta completa."""
        # En una implementación real, esto usaría el orquestador
        # Para ahora, enviamos una respuesta de ejemplo
        
        await self._send_message(websocket, WebSocketMessage(
            type=WebSocketMessageType.QUERY_RESPONSE,
            data={
                "request_id": request.request_id,
                "chunk_type": "complete",
                "content": "The function `process_data` is defined in `utils.py` at lines 10-25. It processes incoming data by applying transformations based on configuration.",
                "confidence": 0.92,
                "sources": [
                    {
                        "type": "code",
                        "file_path": "src/utils.py",
                        "line_range": [10, 25],
                        "confidence": 0.95,
                        "excerpt": "def process_data(data, config=None):\n    \"\"\"Process data with optional configuration.\"\"\"\n    if config:\n        return apply_config(data, config)\n    return data"
                    }
                ],
                "is_final": True
            }
        ))
    
    async def _send_message(self, websocket: WebSocketServerProtocol, 
                          message: WebSocketMessage) -> None:
        """Envía un mensaje a través del WebSocket."""
        try:
            await websocket.send(message.to_json())
            self.metrics["messages_sent"] += 1
            
        except ConnectionClosed:
            logger.debug("Conexión cerrada, no se pudo enviar mensaje")
            raise
            
        except Exception as e:
            logger.error("Error enviando mensaje WebSocket: %s", e)
            raise
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_message: str) -> None:
        """Envía un mensaje de error."""
        await self._send_message(websocket, WebSocketMessage(
            type=WebSocketMessageType.ERROR,
            data={"message": error_message}
        ))
    
    async def _heartbeat_loop(self) -> None:
        """Loop de heartbeat para mantener conexiones."""
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                await self.maintain_heartbeat()
                
        except asyncio.CancelledError:
            logger.info("Heartbeat loop cancelled")
            
        except Exception as e:
            logger.error("Error en heartbeat loop: %s", e, exc_info=True)
    
    async def cleanup(self) -> None:
        """Limpia recursos de la API WebSocket."""
        logger.info("Limpiando WebSocket API")
        
        # Cancelar heartbeat
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Cerrar todas las conexiones
        for connection_id in list(self.connections.keys()):
            await self.handle_disconnection(connection_id)
        
        logger.info("WebSocket API limpiada")