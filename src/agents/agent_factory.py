"""
agent_factory.py - Fábrica para crear y gestionar agentes.

Versión actualizada manteniendo compatibilidad total con la API original.
Se han aplicado mejoras de tipado, robustez y legibilidad sin romper contratos.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from ..core.exceptions import BrainException, ValidationError
from .base_agent import BaseAgent
from .code_analyzer_agent import CodeAnalyzerAgent
from .architect_agent import ArchitectAgent
from .detective_agent import DetectiveAgent
from .qa_agent import QuestionAnsweringAgent
from .curator_agent import CuratorAgent
from .analyst_agent import AnalystAgent
from .security_agent import SecurityAgent
from .learning_agent import LearningAgent

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Tipos de agentes disponibles."""

    CODE_ANALYZER = "code_analyzer"
    ARCHITECT = "architect"
    DETECTIVE = "detective"
    QA = "qa"
    CURATOR = "curator"
    ANALYST = "analyst"
    SECURITY = "security"
    LEARNING = "learning"


@dataclass(slots=True)
class AgentConfig:
    """Configuración para crear un agente."""

    agent_type: AgentType
    agent_id: str
    capabilities: List[str]
    memory_size: int = 1000
    learning_enabled: bool = True
    collaboration_enabled: bool = True
    custom_params: Optional[Dict[str, Any]] = None


class AgentFactory:
    """
    Fábrica para crear y gestionar instancias de agentes.
    """

    def __init__(self, orchestrator: Optional[Any] = None) -> None:
        self.orchestrator = orchestrator
        self.agent_pool: Dict[str, BaseAgent] = {}

        # Registro de tipos conocidos (API estable)
        self.registered_types: Dict[AgentType, Type[BaseAgent]] = {
            AgentType.CODE_ANALYZER: CodeAnalyzerAgent,
            AgentType.ARCHITECT: ArchitectAgent,
            AgentType.DETECTIVE: DetectiveAgent,
            AgentType.QA: QuestionAnsweringAgent,
            AgentType.CURATOR: CuratorAgent,
            AgentType.ANALYST: AnalystAgent,
            AgentType.SECURITY: SecurityAgent,
            AgentType.LEARNING: LearningAgent,
        }

        # Configuraciones por defecto (extensibles)
        self.default_configs: Dict[AgentType, Dict[str, Any]] = {
            AgentType.CODE_ANALYZER: {
                "memory_size": 2000,
                "analysis_depth": "comprehensive",
                "code_languages": ["python", "javascript", "java"],
            },
            AgentType.QA: {
                "memory_size": 5000,
                "context_window": 10,
                "confidence_threshold": 0.7,
            },
            AgentType.ARCHITECT: {
                "memory_size": 3000,
                "arch_patterns": ["microservices", "layered", "event-driven"],
            },
        }

        logger.info(
            "AgentFactory inicializado con %d tipos de agentes",
            len(self.registered_types),
        )

    # ------------------------------------------------------------------
    # Creación y registro
    # ------------------------------------------------------------------

    def create_agent(self, config: AgentConfig) -> BaseAgent:
        errors = self.validate_agent_creation(config)
        if errors:
            raise ValidationError("; ".join(errors))

        agent_class = self.registered_types.get(config.agent_type)
        if agent_class is None:
            raise ValidationError(f"Tipo de agente no registrado: {config.agent_type}")

        try:
            full_config = self.get_agent_config(config.agent_type)
            full_config.update(
                {
                    "memory_size": config.memory_size,
                    "learning_enabled": config.learning_enabled,
                    "collaboration_enabled": config.collaboration_enabled,
                }
            )

            if config.custom_params:
                full_config.update(config.custom_params)

            agent = agent_class(
                agent_id=config.agent_id,
                capabilities=config.capabilities,
                config=full_config,
                orchestrator=self.orchestrator,
            )

            agent.initialize()
            self.agent_pool[config.agent_id] = agent

            logger.info(
                "Agente creado: %s (%s)",
                config.agent_id,
                config.agent_type.value,
            )

            return agent

        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Error creando agente %s", config.agent_id)
            raise BrainException(f"Error creando agente: {exc}") from exc

    def register_agent_type(
        self,
        agent_type: AgentType,
        agent_class: Type[BaseAgent],
        default_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if agent_type in self.registered_types:
            raise ValidationError(f"Tipo de agente ya registrado: {agent_type}")

        if not issubclass(agent_class, BaseAgent):
            raise ValidationError("La clase debe heredar de BaseAgent")

        self.registered_types[agent_type] = agent_class
        if default_config:
            self.default_configs[agent_type] = default_config

        logger.info("Tipo de agente registrado: %s", agent_type.value)

    # ------------------------------------------------------------------
    # Información
    # ------------------------------------------------------------------

    def list_available_agents(self) -> List[Dict[str, Any]]:
        agents_info: List[Dict[str, Any]] = []

        for agent_type, agent_class in self.registered_types.items():
            capabilities = (
                agent_class.get_capabilities()
                if hasattr(agent_class, "get_capabilities")
                else []
            )

            is_abstract = bool(getattr(agent_class, "__abstractmethods__", False))

            agents_info.append(
                {
                    "type": agent_type.value,
                    "class_name": agent_class.__name__,
                    "module": agent_class.__module__,
                    "description": getattr(
                        agent_class,
                        "DESCRIPTION",
                        "No description available",
                    ),
                    "capabilities": capabilities,
                    "default_config": self.default_configs.get(agent_type, {}),
                    "is_abstract": is_abstract,
                }
            )

        return agents_info

    def get_agent_config(self, agent_type: AgentType) -> Dict[str, Any]:
        if agent_type not in self.registered_types:
            raise ValidationError(f"Tipo de agente no registrado: {agent_type}")

        config: Dict[str, Any] = {
            "agent_type": agent_type.value,
            "memory_size": 1000,
            "learning_enabled": True,
            "collaboration_enabled": True,
        }

        config.update(self.default_configs.get(agent_type, {}))
        return config

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    def validate_agent_creation(self, config: AgentConfig) -> List[str]:
        errors: List[str] = []

        if config.agent_type not in self.registered_types:
            errors.append(f"Tipo de agente no registrado: {config.agent_type}")

        if config.agent_id in self.agent_pool:
            errors.append(f"ID de agente ya existe: {config.agent_id}")

        if not config.capabilities:
            errors.append("El agente debe tener al menos una capacidad")

        if config.memory_size <= 0:
            errors.append("El tamaño de memoria debe ser mayor a 0")
        elif config.memory_size > 100_000:
            errors.append("El tamaño de memoria no puede exceder 100000")

        return errors

    # ------------------------------------------------------------------
    # Gestión del pool
    # ------------------------------------------------------------------

    def optimize_agent_pool(
        self,
        max_inactive_time: int = 300,
        min_memory_usage: float = 0.3,
    ) -> Dict[str, Any]:
        stats = {
            "agents_before": len(self.agent_pool),
            "agents_terminated": 0,
            "agents_optimized": 0,
            "memory_freed": 0,
        }

        now = time.time()

        for agent_id, agent in list(self.agent_pool.items()):
            try:
                memory = agent.get_memory_stats()
                last_active = memory.get("last_active_time", now)
                memory_usage = memory.get("memory_usage_percent", 1.0)

                if now - last_active > max_inactive_time:
                    agent.shutdown()
                    del self.agent_pool[agent_id]
                    stats["agents_terminated"] += 1
                    stats["memory_freed"] += memory.get("memory_used", 0)

                elif memory_usage < min_memory_usage:
                    agent.optimize_performance()
                    stats["agents_optimized"] += 1

            except Exception:
                logger.exception("Error optimizando agente %s", agent_id)

        stats["agents_after"] = len(self.agent_pool)
        return stats

    def cleanup_agents(self, force: bool = False) -> int:
        removed = 0

        for agent_id, agent in list(self.agent_pool.items()):
            try:
                state = agent.get_state()
                should_remove = (
                    force
                    or state.get("status") == "error"
                    or state.get("health_status") == "unhealthy"
                    or not state.get("is_initialized", False)
                )

                if should_remove:
                    agent.shutdown()
                    del self.agent_pool[agent_id]
                    removed += 1

            except Exception:
                logger.exception("Error limpiando agente %s", agent_id)
                if force:
                    del self.agent_pool[agent_id]
                    removed += 1

        return removed

    # ------------------------------------------------------------------
    # Acceso directo
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self.agent_pool.get(agent_id)

    def list_active_agents(self) -> List[Dict[str, Any]]:
        active: List[Dict[str, Any]] = []

        for agent_id, agent in self.agent_pool.items():
            try:
                state = agent.get_state()
                config = agent.get_config()

                active.append(
                    {
                        "agent_id": agent_id,
                        "type": config.get("agent_type"),
                        "status": state.get("status"),
                        "capabilities": config.get("capabilities", []),
                        "memory_stats": agent.get_memory_stats(),
                        "created_at": state.get("created_at"),
                        "last_active": state.get("last_active"),
                    }
                )
            except Exception:
                logger.exception("Error leyendo agente %s", agent_id)

        return active

    def shutdown_all(self) -> bool:
        success = True

        for agent_id, agent in list(self.agent_pool.items()):
            try:
                agent.shutdown()
                del self.agent_pool[agent_id]
            except Exception:
                logger.exception("Error apagando agente %s", agent_id)
                success = False

        return success
