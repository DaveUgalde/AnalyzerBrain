"""
Agents module for Project Brain.

Provides specialized AI agents for code analysis, architecture review,
security assessment, QA, learning, and collaboration.

This module preserves backward compatibility with earlier public exports
while improving readability and maintainability.
"""

from __future__ import annotations

# Base primitives
from .base_agent import (
    BaseAgent,
    AgentConfig,
    AgentInput,
    AgentOutput,
    AgentState,
    AgentCapability,
    AgentMemoryType,
)

# Core infrastructure
from .agent_factory import AgentFactory
from .agent_orchestrator import AgentOrchestrator
from .collaboration_protocol import CollaborationProtocol

# Agentes especializados
from .analyst_agent import AnalystAgent
from .architect_agent import ArchitectAgent
from .code_analyzer_agent import CodeAnalyzerAgent
from .curator_agent import CuratorAgent
from .detective_agent import DetectiveAgent
from .learning_agent import LearningAgent
from .qa_agent import QuestionAnsweringAgent
from .security_agent import SecurityAgent

# Public API (kept stable for compatibility)
__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentInput",
    "AgentOutput",
    "AgentState",
    "AgentCapability",
    "AgentMemoryType",
    "AgentFactory",
    "AgentOrchestrator",
    "CollaborationProtocol",
    "AnalystAgent",
    "ArchitectAgent",
    "CodeAnalyzerAgent",
    "CuratorAgent",
    "DetectiveAgent",
    "LearningAgent",
    "QuestionAnsweringAgent",
    "SecurityAgent",
]
