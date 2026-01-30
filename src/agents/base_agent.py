"""
BaseAgent - Abstract base class for all AI agents.
Defines common interface and base behavior for specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
import warnings
from datetime import datetime
from pydantic import BaseModel, Field, validator
import json
import os
from pathlib import Path

from ..core.exceptions import AgentException, ValidationError


class AgentState(Enum):
    """Possible states of an agent."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    LEARNING = "learning"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentCapability(Enum):
    """Capabilities an agent can have."""
    CODE_ANALYSIS = "code_analysis"
    PATTERN_DETECTION = "pattern_detection"
    QUESTION_ANSWERING = "question_answering"
    ARCHITECTURE_REVIEW = "architecture_review"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    CODE_GENERATION = "code_generation"
    DOCUMENTATION_GENERATION = "documentation_generation"
    TEST_GENERATION = "test_generation"
    REFACTORING_SUGGESTION = "refactoring_suggestion"


class AgentMemoryType(Enum):
    """Memory types for agents."""
    SHORT_TERM = "short_term"      # Immediate memory (minutes)
    LONG_TERM = "long_term"        # Persistent memory (days/months)
    EPISODIC = "episodic"          # Memory of specific experiences
    SEMANTIC = "semantic"          # Memory of general concepts


@dataclass
class AgentConfig:
    """Base configuration for agents."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "BaseAgent"
    version: str = "1.0.0"
    description: str = "Base agent for Project Brain"
    capabilities: List[AgentCapability] = field(default_factory=list)
    max_processing_time: int = 30  # seconds
    confidence_threshold: float = 0.7
    learning_rate: float = 0.1
    memory_size: Dict[AgentMemoryType, int] = field(
        default_factory=lambda: {
            AgentMemoryType.SHORT_TERM: 100,
            AgentMemoryType.LONG_TERM: 1000,
            AgentMemoryType.EPISODIC: 500,
            AgentMemoryType.SEMANTIC: 10000
        }
    )
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True


class AgentInput(BaseModel):
    """Standardized input for agents."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(..., description="Input data")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    priority: int = Field(1, ge=1, le=10, description="Priority (1-10)")
    
    class Config:
        arbitrary_types_allowed = True


class AgentOutput(BaseModel):
    """Standardized output from agents."""
    request_id: str
    agent_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class AgentMemory:
    """Memory system for agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.memories: Dict[AgentMemoryType, List[Dict]] = {
            AgentMemoryType.SHORT_TERM: [],
            AgentMemoryType.LONG_TERM: [],
            AgentMemoryType.EPISODIC: [],
            AgentMemoryType.SEMANTIC: []
        }
    
    def store(self, memory_type: AgentMemoryType, content: Dict) -> str:
        """Store a memory in memory."""
        memory_id = str(uuid.uuid4())
        memory = {
            "id": memory_id,
            "type": memory_type.value,
            "content": content,
            "timestamp": datetime.now(),
            "access_count": 0
        }
        
        self.memories[memory_type].append(memory)
        
        # Maintain maximum size
        max_size = self.config.memory_size.get(memory_type, 100)
        if len(self.memories[memory_type]) > max_size:
            self.memories[memory_type].pop(0)
        
        return memory_id
    
    def retrieve(self, memory_type: AgentMemoryType, 
                query: Optional[Dict] = None, 
                limit: int = 10) -> List[Dict]:
        """Retrieve memories from memory."""
        memories = self.memories[memory_type]
        
        if query is None:
            # Return most recent
            return sorted(memories, key=lambda x: x["timestamp"], reverse=True)[:limit]
        
        # Simple content search (basic implementation)
        results = []
        for memory in memories:
            if self._matches_query(memory["content"], query):
                results.append(memory)
        
        # Sort by relevance (simplified)
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def consolidate(self) -> None:
        """Consolidate memories (move from short to long term)."""
        # Move old memories from short to long term
        old_threshold = datetime.now().timestamp() - 3600  # 1 hour
        
        short_term = self.memories[AgentMemoryType.SHORT_TERM]
        long_term = self.memories[AgentMemoryType.LONG_TERM]
        
        to_move = []
        to_keep = []
        
        for memory in short_term:
            if memory["timestamp"].timestamp() < old_threshold:
                to_move.append(memory)
            else:
                to_keep.append(memory)
        
        # Move
        long_term.extend(to_move)
        self.memories[AgentMemoryType.SHORT_TERM] = to_keep
        
        # Maintain size
        max_long_term = self.config.memory_size.get(AgentMemoryType.LONG_TERM, 1000)
        if len(long_term) > max_long_term:
            self.memories[AgentMemoryType.LONG_TERM] = long_term[-max_long_term:]
    
    def clear(self, memory_type: Optional[AgentMemoryType] = None) -> None:
        """Clear memories."""
        if memory_type:
            self.memories[memory_type] = []
        else:
            for mt in AgentMemoryType:
                self.memories[mt] = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            mem_type.value: len(self.memories[mem_type])
            for mem_type in AgentMemoryType
        }
    
    def _matches_query(self, content: Dict, query: Dict) -> bool:
        """Check if content matches query."""
        # Basic implementation - in real agents would use embeddings
        for key, value in query.items():
            if key in content:
                if isinstance(value, str) and isinstance(content[key], str):
                    if value.lower() in content[key].lower():
                        return True
                elif content[key] == value:
                    return True
        return False


class BaseAgent(ABC):
    """
    Abstract base class for all Project Brain agents.
    
    All agents must inherit from this class and implement:
    1. process() - Main processing
    2. learn() - Learning from feedback
    3. evaluate() - Self-evaluation
    
    Common features:
    - Hierarchical memory system
    - State and error handling
    - Performance metrics
    - Learning capability
    - Collaboration with other agents
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self.state = AgentState.INITIALIZING
        self.memory = AgentMemory(config)
        self.metrics: Dict[str, Any] = {
            "requests_processed": 0,
            "success_rate": 1.0,
            "avg_processing_time_ms": 0.0,
            "total_learning_events": 0,
            "confidence_distribution": [],
            "error_types": {}
        }
        self.dependencies: Dict[str, Any] = {}
        self._initialized = False
        
    async def initialize(self, dependencies: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the agent and its dependencies.
        
        Args:
            dependencies: Injected dependencies (other agents, services)
            
        Returns:
            bool: True if initialization was successful
            
        Raises:
            AgentException: If there are initialization errors
        """
        try:
            self.state = AgentState.INITIALIZING
            
            # Check required dependencies
            if dependencies:
                self.dependencies = dependencies
                missing_deps = [
                    dep for dep in self.config.dependencies 
                    if dep not in dependencies
                ]
                if missing_deps:
                    raise AgentException(f"Missing dependencies: {missing_deps}")
            
            # Agent-specific initialization
            success = await self._initialize_internal()
            
            if success:
                self.state = AgentState.READY
                self._initialized = True
                
                # Consolidate initial memory
                self.memory.consolidate()
                
                return True
            else:
                self.state = AgentState.ERROR
                return False
                
        except Exception as e:
            self.state = AgentState.ERROR
            self.metrics["error_types"]["initialization"] = \
                self.metrics["error_types"].get("initialization", 0) + 1
            raise AgentException(f"Failed to initialize agent {self.config.name}: {e}")
    
    @abstractmethod
    async def _initialize_internal(self) -> bool:
        """
        Agent-specific initialization.
        Must be implemented by each concrete agent.
        
        Returns:
            bool: True if initialization was successful
        """
        pass
    
    async def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process an input and produce an output.
        
        Args:
            input_data: Standardized input data
            
        Returns:
            AgentOutput: Processing result
            
        Raises:
            AgentException: If agent is not initialized
            ValidationError: If input is invalid
            TimeoutError: If exceeds maximum processing time
        """
        if not self._initialized:
            raise AgentException(f"Agent {self.config.name} is not initialized")
        
        if self.state != AgentState.READY:
            raise AgentException(f"Agent {self.config.name} is not ready (state: {self.state})")
        
        start_time = datetime.now()
        self.state = AgentState.PROCESSING
        
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Store in short-term memory
            self.memory.store(
                AgentMemoryType.SHORT_TERM,
                {
                    "type": "input",
                    "data": input_data.dict(),
                    "timestamp": input_data.timestamp
                }
            )
            
            # Process (abstract method)
            result = await self._process_internal(input_data)
            
            # Validate output
            self._validate_output(result)
            
            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(result, processing_time)
            
            # Store in episodic memory
            self.memory.store(
                AgentMemoryType.EPISODIC,
                {
                    "type": "processing",
                    "input": input_data.dict(),
                    "output": result.dict(),
                    "processing_time_ms": processing_time,
                    "success": result.success
                }
            )
            
            # Consolidate memory periodically
            if self.metrics["requests_processed"] % 10 == 0:
                self.memory.consolidate()
            
            self.state = AgentState.READY
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record error
            error_type = type(e).__name__
            self.metrics["error_types"][error_type] = \
                self.metrics["error_types"].get(error_type, 0) + 1
            
            # Create error output
            output = AgentOutput(
                request_id=input_data.request_id,
                agent_id=self.config.agent_id,
                success=False,
                confidence=0.0,
                errors=[str(e)],
                processing_time_ms=processing_time
            )
            
            self.state = AgentState.READY
            return output
    
    @abstractmethod
    async def _process_internal(self, input_data: AgentInput) -> AgentOutput:
        """
        Agent-specific processing.
        Must be implemented by each concrete agent.
        
        Args:
            input_data: Input data
            
        Returns:
            AgentOutput: Processing result
        """
        pass
    
    async def learn(self, feedback: Dict[str, Any]) -> bool:
        """
        Learn from feedback or experiences.
        
        Args:
            feedback: Feedback data for learning
            
        Returns:
            bool: True if learning was successful
            
        Example feedback:
            {
                "type": "correction",
                "original_input": {...},
                "expected_output": {...},
                "actual_output": {...},
                "confidence_impact": 0.1
            }
        """
        if not self._initialized:
            return False
        
        self.state = AgentState.LEARNING
        
        try:
            # Validate feedback
            if not self._validate_feedback(feedback):
                return False
            
            # Specific learning
            success = await self._learn_internal(feedback)
            
            if success:
                # Store in semantic memory
                self.memory.store(
                    AgentMemoryType.SEMANTIC,
                    {
                        "type": "learning",
                        "feedback": feedback,
                        "timestamp": datetime.now(),
                        "agent_version": self.config.version
                    }
                )
                
                self.metrics["total_learning_events"] += 1
                
                # Adjust confidence if needed
                if "confidence_impact" in feedback:
                    self.config.confidence_threshold = max(0.1, min(0.95,
                        self.config.confidence_threshold + feedback["confidence_impact"]
                    ))
            
            self.state = AgentState.READY
            return success
            
        except Exception as e:
            self.state = AgentState.READY
            self.metrics["error_types"]["learning"] = \
                self.metrics["error_types"].get("learning", 0) + 1
            return False
    
    @abstractmethod
    async def _learn_internal(self, feedback: Dict[str, Any]) -> bool:
        """
        Agent-specific learning.
        Must be implemented by each concrete agent.
        
        Args:
            feedback: Feedback data
            
        Returns:
            bool: True if learning was successful
        """
        pass
    
    async def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate current agent performance.
        
        Returns:
            Dict with evaluation metrics
        """
        evaluation = {
            "agent_id": self.config.agent_id,
            "agent_name": self.config.name,
            "state": self.state.value,
            "initialized": self._initialized,
            "metrics": self.metrics.copy(),
            "memory_stats": self.memory.get_stats(),
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "learning_rate": self.config.learning_rate,
                "capabilities": [c.value for c in self.config.capabilities]
            },
            "timestamp": datetime.now()
        }
        
        # Calculate additional metrics
        if self.metrics["requests_processed"] > 0:
            evaluation["metrics"]["success_rate"] = (
                1 - (sum(self.metrics["error_types"].values()) / 
                     self.metrics["requests_processed"])
            )
        
        return evaluation
    
    async def get_capabilities(self) -> List[Dict[str, Any]]:
        """
        Get detailed agent capabilities.
        
        Returns:
            List of capabilities with description
        """
        capabilities = []
        
        for capability in self.config.capabilities:
            cap_info = {
                "name": capability.value,
                "description": self._get_capability_description(capability),
                "confidence_threshold": self.config.confidence_threshold,
                "supported_languages": self._get_supported_languages(capability),
                "examples": self._get_capability_examples(capability)
            }
            capabilities.append(cap_info)
        
        return capabilities
    
    async def shutdown(self) -> bool:
        """
        Shut down agent in controlled manner.
        
        Returns:
            bool: True if shutdown was successful
        """
        try:
            # Save state if needed
            await self._save_state()
            
            # Final memory consolidation
            self.memory.consolidate()
            
            self.state = AgentState.MAINTENANCE
            self._initialized = False
            
            return True
            
        except Exception as e:
            self.state = AgentState.ERROR
            return False
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state
    
    def get_config(self) -> AgentConfig:
        """Get agent configuration."""
        return self.config
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.memory.get_stats()
    
    def reset(self) -> bool:
        """Reset agent to initial state."""
        try:
            self.state = AgentState.INITIALIZING
            self.memory.clear()
            self.metrics = {
                "requests_processed": 0,
                "success_rate": 1.0,
                "avg_processing_time_ms": 0.0,
                "total_learning_events": 0,
                "confidence_distribution": [],
                "error_types": {}
            }
            self._initialized = False
            return True
        except Exception as e:
            self.state = AgentState.ERROR
            return False
    
    # Protected helper methods
    
    def _validate_input(self, input_data: AgentInput) -> None:
        """Validate agent input."""
        if not input_data.data:
            raise ValidationError("Input data cannot be empty")
        
        # Specific validations by agent
        self._validate_input_specific(input_data)
    
    def _validate_output(self, output: AgentOutput) -> None:
        """Validate agent output."""
        if output.confidence < 0.0 or output.confidence > 1.0:
            raise ValidationError(f"Invalid confidence value: {output.confidence}")
        
        if output.success and not output.data:
            raise ValidationError("Successful output must contain data")
    
    def _validate_feedback(self, feedback: Dict) -> bool:
        """Validate feedback data."""
        required_fields = ["type", "timestamp"]
        
        for field in required_fields:
            if field not in feedback:
                return False
        
        return True
    
    def _update_metrics(self, output: AgentOutput, processing_time: float) -> None:
        """Update agent metrics."""
        self.metrics["requests_processed"] += 1
        
        # Update success rate
        if output.success:
            current_success = self.metrics["success_rate"]
            new_count = self.metrics["requests_processed"]
            self.metrics["success_rate"] = (
                (current_success * (new_count - 1) + 1) / new_count
            )
        else:
            current_success = self.metrics["success_rate"]
            new_count = self.metrics["requests_processed"]
            self.metrics["success_rate"] = (
                (current_success * (new_count - 1)) / new_count
            )
        
        # Update average processing time
        current_avg = self.metrics["avg_processing_time_ms"]
        new_count = self.metrics["requests_processed"]
        self.metrics["avg_processing_time_ms"] = (
            (current_avg * (new_count - 1) + processing_time) / new_count
        )
        
        # Record confidence distribution
        self.metrics["confidence_distribution"].append(output.confidence)
        if len(self.metrics["confidence_distribution"]) > 1000:
            self.metrics["confidence_distribution"].pop(0)
    
    def _get_capability_description(self, capability: AgentCapability) -> str:
        """Get description of a capability."""
        descriptions = {
            AgentCapability.CODE_ANALYSIS: "Analyzes code structure, complexity, and patterns",
            AgentCapability.PATTERN_DETECTION: "Detects design patterns and anti-patterns",
            AgentCapability.QUESTION_ANSWERING: "Answers questions about code and projects",
            AgentCapability.ARCHITECTURE_REVIEW: "Reviews and suggests architectural improvements",
            AgentCapability.SECURITY_ANALYSIS: "Analyzes code for security vulnerabilities",
            AgentCapability.PERFORMANCE_ANALYSIS: "Analyzes and suggests performance improvements",
            AgentCapability.CODE_GENERATION: "Generates code based on specifications",
            AgentCapability.DOCUMENTATION_GENERATION: "Generates documentation for code",
            AgentCapability.TEST_GENERATION: "Generates tests for code",
            AgentCapability.REFACTORING_SUGGESTION: "Suggests refactoring opportunities"
        }
        return descriptions.get(capability, "No description available")
    
    def _get_supported_languages(self, capability: AgentCapability) -> List[str]:
        """Get supported languages for a capability."""
        # By default, all languages
        # Specific agents can override
        return ["python", "javascript", "typescript", "java", "cpp", "go", "rust"]
    
    def _get_capability_examples(self, capability: AgentCapability) -> List[Dict]:
        """Get usage examples of a capability."""
        # Generic examples - specific agents can extend
        return []
    
    # Abstract methods for specific validation
    
    @abstractmethod
    def _validate_input_specific(self, input_data: AgentInput) -> None:
        """Specific input validation for agent."""
        pass
    
    @abstractmethod
    async def _save_state(self) -> None:
        """Save agent state (for persistence)."""
        pass
    
    # Utility functions
    
    def _log_activity(self, activity_type: str, details: Dict) -> None:
        """Log agent activity."""
        log_entry = {
            "agent_id": self.config.agent_id,
            "activity_type": activity_type,
            "details": details,
            "timestamp": datetime.now()
        }
        # In real implementation, would log to system logger
        print(f"[Agent Log] {log_entry}")
    
    def _check_health(self) -> Dict[str, Any]:
        """Check agent health."""
        return {
            "agent_id": self.config.agent_id,
            "state": self.state.value,
            "initialized": self._initialized,
            "memory_usage": self.memory.get_stats(),
            "metrics": self.metrics
        }
    
    def _backup_state(self, backup_path: str) -> bool:
        """Backup agent state."""
        try:
            state_data = {
                "config": self.config,
                "metrics": self.metrics,
                "state": self.state.value,
                "memory_stats": self.memory.get_stats(),
                "timestamp": datetime.now()
            }
            
            backup_file = Path(backup_path) / f"agent_{self.config.agent_id}_backup.json"
            with open(backup_file, 'w') as f:
                json.dump(state_data, f, default=str)
            
            return True
        except Exception as e:
            print(f"Failed to backup agent state: {e}")
            return False
    
    def _restore_state(self, backup_path: str) -> bool:
        """Restore agent state from backup."""
        try:
            backup_file = Path(backup_path) / f"agent_{self.config.agent_id}_backup.json"
            
            if not backup_file.exists():
                return False
            
            with open(backup_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore metrics and state
            self.metrics = state_data.get("metrics", self.metrics)
            state_str = state_data.get("state", "INITIALIZING")
            self.state = AgentState(state_str)
            
            return True
        except Exception as e:
            print(f"Failed to restore agent state: {e}")
            return False
    
    def _optimize_performance(self) -> None:
        """Optimize agent performance."""
        # Memory consolidation
        self.memory.consolidate()
        
        # Clean old error records
        if len(self.metrics["error_types"]) > 100:
            # Keep only most frequent errors
            sorted_errors = sorted(
                self.metrics["error_types"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:50]
            self.metrics["error_types"] = dict(sorted_errors)
    
    def _handle_error(self, error: Exception, context: Dict) -> AgentOutput:
        """Handle error and create error output."""
        error_id = str(uuid.uuid4())
        
        # Log error
        self._log_activity("error", {
            "error_id": error_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        })
        
        # Record in metrics
        error_type = type(error).__name__
        self.metrics["error_types"][error_type] = \
            self.metrics["error_types"].get(error_type, 0) + 1
        
        # Create error output
        return AgentOutput(
            request_id=context.get("request_id", str(uuid.uuid4())),
            agent_id=self.config.agent_id,
            success=False,
            confidence=0.0,
            errors=[f"{type(error).__name__}: {str(error)}"],
            warnings=["Error occurred during processing"],
            processing_time_ms=context.get("processing_time_ms", 0.0)
        )
    
    # Memory functions
    
    def store_memory(self, memory_type: AgentMemoryType, content: Dict) -> str:
        """Store content in agent memory."""
        return self.memory.store(memory_type, content)
    
    def retrieve_memory(self, memory_type: AgentMemoryType, 
                       query: Optional[Dict] = None, 
                       limit: int = 10) -> List[Dict]:
        """Retrieve memories from agent memory."""
        return self.memory.retrieve(memory_type, query, limit)
    
    def consolidate_memory(self) -> None:
        """Consolidate agent memories."""
        self.memory.consolidate()
    
    def clear_memory(self, memory_type: Optional[AgentMemoryType] = None) -> None:
        """Clear agent memories."""
        self.memory.clear(memory_type)