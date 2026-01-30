"""
Protocolo de colaboración para agentes multi-agente.
Define los mecanismos de comunicación, coordinación y negociación entre agentes.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
import asyncio

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Tipos de mensajes en el protocolo de colaboración."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_RESULT = "task_result"
    TASK_FAILURE = "task_failure"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    NEGOTIATION_PROPOSAL = "negotiation_proposal"
    NEGOTIATION_ACCEPT = "negotiation_accept"
    NEGOTIATION_REJECT = "negotiation_reject"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_GRANT = "resource_grant"
    RESOURCE_DENY = "resource_deny"
    BROADCAST = "broadcast"


class AgentRole(Enum):
    """Roles que pueden asumir los agentes en la colaboración."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    NEGOTIATOR = "negotiator"
    MONITOR = "monitor"
    SPECIALIST = "specialist"
    GENERALIST = "generalist"


class CollaborationState(Enum):
    """Estados posibles de una sesión de colaboración."""
    INITIALIZING = "initializing"
    NEGOTIATING = "negotiating"
    COORDINATING = "coordinating"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class Message:
    """Estructura de mensaje para comunicación entre agentes."""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    priority: int = 1
    ttl: int = 3600  # Time to live en segundos
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el mensaje a diccionario para serialización."""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "priority": self.priority,
            "ttl": self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Crea un mensaje desde un diccionario."""
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data.get("session_id"),
            priority=data.get("priority", 1),
            ttl=data.get("ttl", 3600)
        )


@dataclass
class CollaborationSession:
    """Sesión de colaboración entre múltiples agentes."""
    session_id: str
    initiator_id: str
    participants: Set[str] = field(default_factory=set)
    state: CollaborationState = CollaborationState.INITIALIZING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_queue: List[Dict[str, Any]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    
    def add_participant(self, agent_id: str) -> None:
        """Agrega un agente a la sesión de colaboración."""
        self.participants.add(agent_id)
        self.updated_at = datetime.now()
    
    def remove_participant(self, agent_id: str) -> None:
        """Remueve un agente de la sesión de colaboración."""
        self.participants.discard(agent_id)
        self.updated_at = datetime.now()
    
    def update_state(self, new_state: CollaborationState) -> None:
        """Actualiza el estado de la sesión."""
        self.state = new_state
        self.updated_at = datetime.now()
    
    def add_task(self, task: Dict[str, Any]) -> None:
        """Agrega una tarea a la cola de la sesión."""
        self.task_queue.append(task)
        self.updated_at = datetime.now()
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Obtiene la siguiente tarea de la cola."""
        if self.task_queue:
            return self.task_queue.pop(0)
        return None


class CollaborationProtocol:
    """
    Protocolo principal para la colaboración entre agentes.
    Maneja la comunicación, coordinación y negociación.
    """
    
    def __init__(self, agent_id: str, message_handler: Callable):
        """
        Inicializa el protocolo de colaboración.
        
        Args:
            agent_id: Identificador único del agente
            message_handler: Función para manejar mensajes recibidos
        """
        self.agent_id = agent_id
        self.message_handler = message_handler
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.message_queue: List[Message] = []
        self.negotiation_strategies: Dict[str, Callable] = {}
        self.coordination_rules: Dict[str, Dict[str, Any]] = {}
        self.registered_services: Dict[str, List[str]] = {}  # servicio -> [agentes]
        
        # Configuración del protocolo
        self.max_retries = 3
        self.timeout_seconds = 30
        self.heartbeat_interval = 10
        
        # Estadísticas
        self.messages_sent = 0
        self.messages_received = 0
        self.sessions_completed = 0
        
        logger.info(f"Protocolo de colaboración inicializado para agente {agent_id}")
    
    async def send_message(self, receiver_id: str, message_type: MessageType,
                          content: Dict[str, Any], session_id: Optional[str] = None,
                          priority: int = 1) -> str:
        """
        Envía un mensaje a otro agente.
        
        Args:
            receiver_id: ID del agente receptor
            message_type: Tipo de mensaje
            content: Contenido del mensaje
            session_id: ID de sesión (opcional)
            priority: Prioridad del mensaje
            
        Returns:
            ID del mensaje enviado
        """
        message_id = str(uuid.uuid4())
        message = Message(
            message_id=message_id,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            session_id=session_id,
            priority=priority
        )
        
        # Aquí normalmente se enviaría a través de un sistema de mensajería
        # Para este ejemplo, lo agregamos a una cola simulada
        self.message_queue.append(message)
        self.messages_sent += 1
        
        logger.debug(f"Mensaje {message_id} enviado a {receiver_id}: {message_type}")
        
        # Simular envío asíncrono
        await asyncio.sleep(0.01)
        
        return message_id
    
    async def broadcast_message(self, message_type: MessageType,
                               content: Dict[str, Any],
                               session_id: Optional[str] = None,
                               exclude_self: bool = True) -> List[str]:
        """
        Envía un mensaje a todos los agentes conocidos.
        
        Args:
            message_type: Tipo de mensaje
            content: Contenido del mensaje
            session_id: ID de sesión (opcional)
            exclude_self: Excluir al propio agente del broadcast
            
        Returns:
            Lista de IDs de mensaje enviados
        """
        # En una implementación real, esto obtendría la lista de agentes
        # de un registro o discovery service
        all_agents = ["agent_1", "agent_2", "agent_3"]  # Ejemplo
        
        message_ids = []
        for agent_id in all_agents:
            if exclude_self and agent_id == self.agent_id:
                continue
            
            message_id = await self.send_message(
                receiver_id=agent_id,
                message_type=message_type,
                content=content,
                session_id=session_id,
                priority=1
            )
            message_ids.append(message_id)
        
        logger.info(f"Broadcast enviado a {len(message_ids)} agentes: {message_type}")
        return message_ids
    
    async def initiate_collaboration(self, task_description: Dict[str, Any],
                                    required_roles: List[AgentRole],
                                    participants: Optional[List[str]] = None) -> str:
        """
        Inicia una nueva sesión de colaboración.
        
        Args:
            task_description: Descripción de la tarea a realizar
            required_roles: Roles necesarios para la colaboración
            participants: Lista de participantes específicos (opcional)
            
        Returns:
            ID de la sesión creada
        """
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        session = CollaborationSession(
            session_id=session_id,
            initiator_id=self.agent_id,
            state=CollaborationState.NEGOTIATING
        )
        
        session.metadata = {
            "task_description": task_description,
            "required_roles": [role.value for role in required_roles],
            "initiated_at": datetime.now().isoformat()
        }
        
        # Agregar al iniciador
        session.add_participant(self.agent_id)
        
        # Agregar participantes específicos si se proporcionan
        if participants:
            for participant in participants:
                session.add_participant(participant)
        
        self.active_sessions[session_id] = session
        
        logger.info(f"Sesión de colaboración {session_id} iniciada")
        
        # Si no hay participantes específicos, buscar agentes para los roles requeridos
        if not participants:
            await self._recruit_agents(session_id, required_roles)
        
        return session_id
    
    async def _recruit_agents(self, session_id: str,
                             required_roles: List[AgentRole]) -> None:
        """
        Busca y recluta agentes para roles específicos.
        
        Args:
            session_id: ID de la sesión
            required_roles: Roles requeridos
        """
        session = self.active_sessions.get(session_id)
        if not session:
            logger.error(f"Sesión {session_id} no encontrada")
            return
        
        for role in required_roles:
            # En una implementación real, se buscarían agentes disponibles
            # con las capacidades requeridas
            await self.send_message(
                receiver_id="coordinator_service",  # Ejemplo
                message_type=MessageType.COORDINATION_REQUEST,
                content={
                    "session_id": session_id,
                    "required_role": role.value,
                    "task_description": session.metadata["task_description"]
                },
                session_id=session_id
            )
    
    async def handle_incoming_message(self, message: Message) -> Dict[str, Any]:
        """
        Procesa un mensaje entrante.
        
        Args:
            message: Mensaje recibido
            
        Returns:
            Respuesta del procesamiento
        """
        self.messages_received += 1
        
        logger.debug(f"Mensaje recibido de {message.sender_id}: {message.message_type}")
        
        # Verificar si el mensaje ha expirado
        message_age = (datetime.now() - message.timestamp).total_seconds()
        if message_age > message.ttl:
            logger.warning(f"Mensaje {message.message_id} expirado ({message_age}s)")
            return {"status": "error", "reason": "message_expired"}
        
        # Manejar según el tipo de mensaje
        handler_name = f"_handle_{message.message_type.value}"
        handler = getattr(self, handler_name, self._handle_unknown_message)
        
        response = await handler(message)
        
        # Actualizar sesión si aplica
        if message.session_id and message.session_id in self.active_sessions:
            session = self.active_sessions[message.session_id]
            session.updated_at = datetime.now()
        
        return response
    
    async def _handle_task_request(self, message: Message) -> Dict[str, Any]:
        """Maneja solicitudes de tarea."""
        session_id = message.session_id or f"task_{uuid.uuid4().hex[:8]}"
        
        # Evaluar si se puede aceptar la tarea
        can_accept = await self._evaluate_task(message.content)
        
        if can_accept:
            response_content = {
                "status": "accepted",
                "session_id": session_id,
                "capabilities": self._get_capabilities(),
                "estimated_completion_time": self._estimate_completion_time(message.content)
            }
            
            # Crear o actualizar sesión
            if session_id not in self.active_sessions:
                session = CollaborationSession(
                    session_id=session_id,
                    initiator_id=message.sender_id,
                    state=CollaborationState.EXECUTING
                )
                session.add_participant(self.agent_id)
                self.active_sessions[session_id] = session
            else:
                self.active_sessions[session_id].add_participant(self.agent_id)
            
            await self.send_message(
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content=response_content,
                session_id=session_id
            )
            
            # Ejecutar la tarea
            asyncio.create_task(self._execute_task(message.content, session_id))
            
            return {"status": "task_accepted", "session_id": session_id}
        else:
            response_content = {
                "status": "rejected",
                "reason": "unavailable",
                "session_id": session_id
            }
            
            await self.send_message(
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content=response_content,
                session_id=session_id
            )
            
            return {"status": "task_rejected"}
    
    async def _handle_task_response(self, message: Message) -> Dict[str, Any]:
        """Maneja respuestas a solicitudes de tarea."""
        session = self.active_sessions.get(message.session_id or "")
        
        if not session:
            logger.warning(f"Sesión no encontrada para respuesta de tarea")
            return {"status": "session_not_found"}
        
        if message.content.get("status") == "accepted":
            logger.info(f"Tarea aceptada por {message.sender_id}")
            session.add_participant(message.sender_id)
        else:
            logger.info(f"Tarea rechazada por {message.sender_id}: {message.content.get('reason')}")
        
        return {"status": "response_processed"}
    
    async def _handle_coordination_request(self, message: Message) -> Dict[str, Any]:
        """Maneja solicitudes de coordinación."""
        # Evaluar si se puede coordinar
        coordination_ability = self._evaluate_coordination_ability(message.content)
        
        response_content = {
            "can_coordinate": coordination_ability,
            "agent_id": self.agent_id,
            "capabilities": self._get_coordination_capabilities()
        }
        
        await self.send_message(
            receiver_id=message.sender_id,
            message_type=MessageType.COORDINATION_RESPONSE,
            content=response_content,
            session_id=message.session_id
        )
        
        return {"status": "coordination_response_sent"}
    
    async def _handle_negotiation_proposal(self, message: Message) -> Dict[str, Any]:
        """Maneja propuestas de negociación."""
        session = self.active_sessions.get(message.session_id or "")
        
        if not session:
            logger.warning(f"Sesión no encontrada para propuesta de negociación")
            return {"status": "session_not_found"}
        
        # Evaluar la propuesta
        evaluation = await self._evaluate_negotiation_proposal(
            message.content,
            message.sender_id
        )
        
        if evaluation["acceptable"]:
            response_type = MessageType.NEGOTIATION_ACCEPT
            session.update_state(CollaborationState.COORDINATING)
        else:
            response_type = MessageType.NEGOTIATION_REJECT
        
        response_content = {
            "proposal_id": message.content.get("proposal_id"),
            "evaluation": evaluation,
            "counter_proposal": evaluation.get("counter_proposal")
        }
        
        await self.send_message(
            receiver_id=message.sender_id,
            message_type=response_type,
            content=response_content,
            session_id=message.session_id
        )
        
        return {"status": "negotiation_processed"}
    
    async def _handle_heartbeat(self, message: Message) -> Dict[str, Any]:
        """Maneja mensajes de heartbeat."""
        # Simplemente responder para confirmar que el agente está vivo
        await self.send_message(
            receiver_id=message.sender_id,
            message_type=MessageType.HEARTBEAT,
            content={"status": "alive", "agent_id": self.agent_id},
            session_id=message.session_id,
            priority=5  # Baja prioridad para heartbeats
        )
        
        return {"status": "heartbeat_responded"}
    
    async def _handle_unknown_message(self, message: Message) -> Dict[str, Any]:
        """Maneja tipos de mensaje desconocidos."""
        logger.warning(f"Tipo de mensaje desconocido: {message.message_type}")
        return {"status": "unknown_message_type"}
    
    async def _evaluate_task(self, task_description: Dict[str, Any]) -> bool:
        """Evalúa si se puede aceptar una tarea."""
        # Implementación básica - siempre acepta
        # En una implementación real, se verificarían recursos, capacidades, etc.
        return True
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Obtiene las capacidades del agente."""
        return {
            "agent_id": self.agent_id,
            "roles": [AgentRole.WORKER.value, AgentRole.SPECIALIST.value],
            "skills": ["processing", "analysis", "decision_making"],
            "available_resources": {"cpu": 80, "memory": 60, "storage": 90}
        }
    
    def _estimate_completion_time(self, task: Dict[str, Any]) -> float:
        """Estima el tiempo de completado para una tarea."""
        # Estimación básica
        complexity = task.get("complexity", 1)
        return complexity * 10.0  # segundos
    
    def _evaluate_coordination_ability(self, request: Dict[str, Any]) -> bool:
        """Evalúa la capacidad de coordinar."""
        # Por defecto, cualquier agente puede coordinar
        # En una implementación real, se verificarían permisos y capacidades
        return True
    
    def _get_coordination_capabilities(self) -> Dict[str, Any]:
        """Obtiene capacidades de coordinación."""
        return {
            "max_agents": 10,
            "supported_protocols": ["task_distribution", "consensus", "auction"],
            "monitoring_capabilities": True
        }
    
    async def _evaluate_negotiation_proposal(self, proposal: Dict[str, Any],
                                            sender_id: str) -> Dict[str, Any]:
        """Evalúa una propuesta de negociación."""
        # Evaluación básica - siempre acepta
        # En una implementación real, se evaluarían términos, condiciones, etc.
        return {
            "acceptable": True,
            "reason": "proposal_accepted",
            "terms": proposal.get("terms", {})
        }
    
    async def _execute_task(self, task: Dict[str, Any], session_id: str) -> None:
        """Ejecuta una tarea asignada."""
        try:
            logger.info(f"Ejecutando tarea para sesión {session_id}")
            
            # Simular ejecución
            await asyncio.sleep(2)
            
            # Generar resultado
            result = {
                "task_id": task.get("task_id", "unknown"),
                "status": "completed",
                "result": "task_executed_successfully",
                "execution_time": 2.0,
                "agent_id": self.agent_id
            }
            
            # Enviar resultado
            session = self.active_sessions.get(session_id)
            if session:
                session.results[task.get("task_id", "unknown")] = result
                
                await self.send_message(
                    receiver_id=session.initiator_id,
                    message_type=MessageType.TASK_RESULT,
                    content=result,
                    session_id=session_id
                )
            
        except Exception as e:
            logger.error(f"Error ejecutando tarea: {e}")
            
            # Enviar notificación de fallo
            failure_result = {
                "task_id": task.get("task_id", "unknown"),
                "status": "failed",
                "error": str(e),
                "agent_id": self.agent_id
            }
            
            await self.send_message(
                receiver_id=session.initiator_id if session else "unknown",
                message_type=MessageType.TASK_FAILURE,
                content=failure_result,
                session_id=session_id
            )
    
    async def register_service(self, service_name: str,
                              capabilities: Dict[str, Any]) -> bool:
        """
        Registra un servicio que ofrece el agente.
        
        Args:
            service_name: Nombre del servicio
            capabilities: Capacidades del servicio
            
        Returns:
            True si el registro fue exitoso
        """
        if service_name not in self.registered_services:
            self.registered_services[service_name] = []
        
        # Registrar el agente como proveedor del servicio
        self.registered_services[service_name].append(self.agent_id)
        
        # Notificar a otros agentes (broadcast)
        await self.broadcast_message(
            message_type=MessageType.BROADCAST,
            content={
                "type": "service_registration",
                "service_name": service_name,
                "provider": self.agent_id,
                "capabilities": capabilities
            }
        )
        
        logger.info(f"Servicio {service_name} registrado por {self.agent_id}")
        return True
    
    async def discover_services(self, service_name: str) -> List[str]:
        """
        Descubre agentes que ofrecen un servicio específico.
        
        Args:
            service_name: Nombre del servicio a descubrir
            
        Returns:
            Lista de IDs de agentes que ofrecen el servicio
        """
        # En una implementación real, se consultaría un servicio de descubrimiento
        # Aquí usamos el registro local
        return self.registered_services.get(service_name, [])
    
    async def start_heartbeat(self) -> None:
        """Inicia el envío periódico de heartbeats."""
        async def heartbeat_loop():
            while True:
                try:
                    await self.broadcast_message(
                        message_type=MessageType.HEARTBEAT,
                        content={"agent_id": self.agent_id, "timestamp": datetime.now().isoformat()},
                        exclude_self=True
                    )
                    await asyncio.sleep(self.heartbeat_interval)
                except Exception as e:
                    logger.error(f"Error en heartbeat: {e}")
                    await asyncio.sleep(5)
        
        asyncio.create_task(heartbeat_loop())
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de una sesión de colaboración.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Estado de la sesión o None si no existe
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "participants": list(session.participants),
            "initiator": session.initiator_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "task_count": len(session.task_queue),
            "results_count": len(session.results)
        }
    
    async def terminate_session(self, session_id: str,
                               reason: str = "normal_termination") -> bool:
        """
        Termina una sesión de colaboración.
        
        Args:
            session_id: ID de la sesión
            reason: Razón de la terminación
            
        Returns:
            True si la sesión fue terminada exitosamente
        """
        session = self.active_sessions.get(session_id)
        if not session:
            logger.warning(f"Sesión {session_id} no encontrada para terminación")
            return False
        
        # Notificar a todos los participantes
        for participant in session.participants:
            if participant != self.agent_id:
                await self.send_message(
                    receiver_id=participant,
                    message_type=MessageType.BROADCAST,
                    content={
                        "type": "session_termination",
                        "session_id": session_id,
                        "reason": reason,
                        "terminated_by": self.agent_id
                    },
                    session_id=session_id
                )
        
        # Actualizar estado
        session.update_state(CollaborationState.TERMINATED)
        
        # Registrar métricas
        self.sessions_completed += 1
        
        # Limpiar recursos (en una implementación real)
        logger.info(f"Sesión {session_id} terminada: {reason}")
        
        return True
    
    def get_protocol_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del protocolo."""
        return {
            "agent_id": self.agent_id,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "active_sessions": len(self.active_sessions),
            "sessions_completed": self.sessions_completed,
            "registered_services": len(self.registered_services),
            "timestamp": datetime.now().isoformat()
        }