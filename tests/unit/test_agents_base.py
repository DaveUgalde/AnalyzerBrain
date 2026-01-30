"""
Pruebas unitarias para el módulo agents/base_agent.py
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentInput,
    AgentOutput,
    AgentState,
    AgentCapability,
    AgentMemoryType,
    AgentMemory,
    AgentException,
    ValidationError
)


class TestAgentConfig:
    """Pruebas para AgentConfig."""
    
    def test_default_config(self):
        """Test configuración por defecto."""
        config = AgentConfig()
        
        assert config.name == "BaseAgent"
        assert config.version == "1.0.0"
        assert config.description == "Base agent for Project Brain"
        assert config.max_processing_time == 30
        assert config.confidence_threshold == 0.7
        assert config.learning_rate == 0.1
        assert config.enabled is True
        
        # Verificar memoria
        assert config.memory_size[AgentMemoryType.SHORT_TERM] == 100
        assert config.memory_size[AgentMemoryType.LONG_TERM] == 1000
        assert config.memory_size[AgentMemoryType.EPISODIC] == 500
        assert config.memory_size[AgentMemoryType.SEMANTIC] == 10000
        
        # Debería generarse un ID automático
        assert config.agent_id is not None
    
    def test_custom_config(self):
        """Test configuración personalizada."""
        config = AgentConfig(
            name="TestAgent",
            version="2.0.0",
            description="Test agent",
            capabilities=[AgentCapability.CODE_ANALYSIS, AgentCapability.QUESTION_ANSWERING],
            max_processing_time=60,
            confidence_threshold=0.8,
            learning_rate=0.2,
            enabled=False
        )
        
        assert config.name == "TestAgent"
        assert config.version == "2.0.0"
        assert config.description == "Test agent"
        assert len(config.capabilities) == 2
        assert config.max_processing_time == 60
        assert config.confidence_threshold == 0.8
        assert config.learning_rate == 0.2
        assert config.enabled is False
    
    def test_config_with_dependencies(self):
        """Test configuración con dependencias."""
        config = AgentConfig(
            dependencies=["agent1", "agent2", "service1"]
        )
        
        assert "agent1" in config.dependencies
        assert "agent2" in config.dependencies
        assert "service1" in config.dependencies


class TestAgentInput:
    """Pruebas para AgentInput."""
    
    def test_input_creation(self):
        """Test creación de entrada."""
        input_data = AgentInput(
            data={"type": "test", "content": "hello"},
            context={"project_id": "123"},
            priority=5
        )
        
        assert input_data.data == {"type": "test", "content": "hello"}
        assert input_data.context == {"project_id": "123"}
        assert input_data.priority == 5
        assert input_data.request_id is not None
        assert input_data.timestamp <= datetime.now()
    
    def test_input_defaults(self):
        """Test valores por defecto."""
        input_data = AgentInput(data={"test": "data"})
        
        assert input_data.priority == 1
        assert input_data.context is None
        assert input_data.request_id is not None
    
    def test_input_validation(self):
        """Test validación de entrada."""
        # Prioridad fuera de rango
        with pytest.raises(ValueError):
            AgentInput(data={}, priority=0)  # Menor que 1
        
        with pytest.raises(ValueError):
            AgentInput(data={}, priority=11)  # Mayor que 10
        
        # Prioridad válida
        for priority in [1, 5, 10]:
            input_data = AgentInput(data={}, priority=priority)
            assert input_data.priority == priority


class TestAgentOutput:
    """Pruebas para AgentOutput."""
    
    def test_output_success(self):
        """Test salida exitosa."""
        output = AgentOutput(
            request_id="req_123",
            agent_id="agent_456",
            success=True,
            data={"result": "ok"},
            confidence=0.9,
            reasoning=["step1", "step2"],
            processing_time_ms=150.5
        )
        
        assert output.success is True
        assert output.data == {"result": "ok"}
        assert output.confidence == 0.9
        assert output.reasoning == ["step1", "step2"]
        assert output.processing_time_ms == 150.5
        assert output.errors == []
        assert output.warnings == []
    
    def test_output_failure(self):
        """Test salida fallida."""
        output = AgentOutput(
            request_id="req_123",
            agent_id="agent_456",
            success=False,
            error="Processing failed",
            warnings=["warning1"],
            confidence=0.1
        )
        
        assert output.success is False
        assert output.error == "Processing failed"
        assert output.warnings == ["warning1"]
        assert output.confidence == 0.1
        assert output.data is None
    
    def test_output_validation(self):
        """Test validación de salida."""
        # Confianza fuera de rango
        with pytest.raises(ValueError):
            AgentOutput(
                request_id="req",
                agent_id="agent",
                success=True,
                confidence=1.5  # Mayor que 1.0
            )
        
        with pytest.raises(ValueError):
            AgentOutput(
                request_id="req",
                agent_id="agent",
                success=True,
                confidence=-0.1  # Menor que 0.0
            )
        
        # Confianza válida
        for confidence in [0.0, 0.5, 1.0]:
            output = AgentOutput(
                request_id="req",
                agent_id="agent",
                success=True,
                confidence=confidence
            )
            assert output.confidence == confidence


class TestAgentMemory:
    """Pruebas para AgentMemory."""
    
    @pytest.fixture
    def memory(self):
        """Crear memoria para pruebas."""
        config = AgentConfig()
        return AgentMemory(config)
    
    def test_store_memory(self, memory):
        """Test almacenamiento en memoria."""
        content = {"type": "fact", "value": "test"}
        
        memory_id = memory.store(AgentMemoryType.SHORT_TERM, content)
        
        assert memory_id is not None
        assert len(memory.memories[AgentMemoryType.SHORT_TERM]) == 1
        
        # Verificar estructura del recuerdo
        memory_item = memory.memories[AgentMemoryType.SHORT_TERM][0]
        assert memory_item["id"] == memory_id
        assert memory_item["content"] == content
        assert memory_item["type"] == "short_term"
        assert "timestamp" in memory_item
        assert memory_item["access_count"] == 0
    
    def test_store_memory_overflow(self, memory):
        """Test almacenamiento con sobreflujo de memoria."""
        # Configurar tamaño pequeño
        memory.config.memory_size[AgentMemoryType.SHORT_TERM] = 3
        
        # Almacenar 5 items
        for i in range(5):
            memory.store(AgentMemoryType.SHORT_TERM, {"index": i})
        
        # Solo deberían quedar los 3 más recientes
        assert len(memory.memories[AgentMemoryType.SHORT_TERM]) == 3
        
        # Los primeros deberían haberse eliminado
        remaining_indices = [item["content"]["index"] 
                           for item in memory.memories[AgentMemoryType.SHORT_TERM]]
        assert set(remaining_indices) == {2, 3, 4}
    
    def test_retrieve_memory_no_query(self, memory):
        """Test recuperación sin query."""
        # Almacenar algunos recuerdos
        for i in range(5):
            memory.store(AgentMemoryType.SHORT_TERM, {"index": i})
        
        # Recuperar sin query
        retrieved = memory.retrieve(AgentMemoryType.SHORT_TERM, limit=3)
        
        assert len(retrieved) == 3
        # Deberían estar ordenados por timestamp (más recientes primero)
        timestamps = [item["timestamp"] for item in retrieved]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_retrieve_memory_with_query(self, memory):
        """Test recuperación con query."""
        # Almacenar recuerdos diferentes
        memory.store(AgentMemoryType.SHORT_TERM, {"type": "fact", "value": "A"})
        memory.store(AgentMemoryType.SHORT_TERM, {"type": "fact", "value": "B"})
        memory.store(AgentMemoryType.SHORT_TERM, {"type": "rule", "value": "C"})
        
        # Query por tipo
        retrieved = memory.retrieve(
            AgentMemoryType.SHORT_TERM,
            query={"type": "fact"},
            limit=10
        )
        
        assert len(retrieved) == 2
        for item in retrieved:
            assert item["content"]["type"] == "fact"
    
    def test_consolidate_memory(self, memory):
        """Test consolidación de memoria."""
        # Crear recuerdos viejos y nuevos
        old_time = datetime.now() - timedelta(hours=2)
        new_time = datetime.now() - timedelta(minutes=30)
        
        # Simular recuerdos con diferentes timestamps
        memory.memories[AgentMemoryType.SHORT_TERM] = [
            {"id": "old1", "content": {"type": "old"}, "timestamp": old_time, "access_count": 0},
            {"id": "new1", "content": {"type": "new"}, "timestamp": new_time, "access_count": 0},
            {"id": "old2", "content": {"type": "old"}, "timestamp": old_time, "access_count": 0},
        ]
        
        # Consolidar
        memory.consolidate()
        
        # Los viejos deberían haberse movido a largo plazo
        assert len(memory.memories[AgentMemoryType.SHORT_TERM]) == 1
        assert len(memory.memories[AgentMemoryType.LONG_TERM]) == 2
        
        # Verificar que se mantuvo el nuevo
        assert memory.memories[AgentMemoryType.SHORT_TERM][0]["id"] == "new1"
    
    def test_matches_query(self, memory):
        """Test coincidencia de query."""
        content = {"name": "test", "value": 123, "tags": ["a", "b"]}
        
        # Query que coincide
        assert memory._matches_query(content, {"name": "test"}) is True
        assert memory._matches_query(content, {"value": 123}) is True
        assert memory._matches_query(content, {"name": "test", "value": 123}) is True
        
        # Query que no coincide
        assert memory._matches_query(content, {"name": "different"}) is False
        assert memory._matches_query(content, {"value": 999}) is False
        
        # Query con substring
        assert memory._matches_query(content, {"name": "tes"}) is True  # "tes" in "test"
        
        # Query vacío
        assert memory._matches_query(content, {}) is False


class TestBaseAgent:
    """Pruebas para BaseAgent."""
    
    @pytest.fixture
    def agent_config(self):
        """Configuración de prueba para agente."""
        return AgentConfig(
            name="TestAgent",
            capabilities=[AgentCapability.CODE_ANALYSIS],
            dependencies=["dep1", "dep2"]
        )
    
    @pytest.fixture
    def concrete_agent(self, agent_config):
        """Crear agente concreto para pruebas."""
        class ConcreteAgent(BaseAgent):
            async def _initialize_internal(self):
                return True
            
            async def _process_internal(self, input_data):
                return AgentOutput(
                    request_id=input_data.request_id,
                    agent_id=self.config.agent_id,
                    success=True,
                    data={"processed": True},
                    confidence=0.9
                )
            
            async def _learn_internal(self, feedback):
                return True
            
            def _validate_input_specific(self, input_data):
                if "type" not in input_data.data:
                    raise ValidationError("Missing type field")
            
            async def _save_state(self):
                pass
        
        return ConcreteAgent(agent_config)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Dependencias mockeadas."""
        return {
            "dep1": Mock(),
            "dep2": Mock()
        }
    
    @pytest.mark.asyncio
    async def test_agent_initialization_success(self, concrete_agent, mock_dependencies):
        """Test inicialización exitosa."""
        success = await concrete_agent.initialize(mock_dependencies)
        
        assert success is True
        assert concrete_agent.state == AgentState.READY
        assert concrete_agent._initialized is True
        assert concrete_agent.dependencies == mock_dependencies
    
    @pytest.mark.asyncio
    async def test_agent_initialization_missing_dependencies(self, concrete_agent):
        """Test inicialización con dependencias faltantes."""
        # Dependencias faltantes
        partial_deps = {"dep1": Mock()}  # Falta dep2
        
        with pytest.raises(AgentException):
            await concrete_agent.initialize(partial_deps)
        
        assert concrete_agent.state == AgentState.ERROR
        assert concrete_agent._initialized is False
    
    @pytest.mark.asyncio
    async def test_agent_initialization_internal_failure(self, agent_config):
        """Test fallo en inicialización interna."""
        class FailingAgent(BaseAgent):
            async def _initialize_internal(self):
                return False
            
            # Métodos abstractos requeridos
            async def _process_internal(self, input_data):
                pass
            
            async def _learn_internal(self, feedback):
                pass
            
            def _validate_input_specific(self, input_data):
                pass
            
            async def _save_state(self):
                pass
        
        agent = FailingAgent(agent_config)
        success = await agent.initialize({})
        
        assert success is False
        assert agent.state == AgentState.ERROR
    
    @pytest.mark.asyncio
    async def test_process_success(self, concrete_agent):
        """Test procesamiento exitoso."""
        await concrete_agent.initialize({})
        
        input_data = AgentInput(data={"type": "test", "content": "hello"})
        output = await concrete_agent.process(input_data)
        
        assert output.success is True
        assert output.data["processed"] is True
        assert output.confidence == 0.9
        assert output.agent_id == concrete_agent.config.agent_id
        assert output.request_id == input_data.request_id
        
        # Verificar métricas
        assert concrete_agent.metrics["requests_processed"] == 1
        assert concrete_agent.metrics["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_process_not_initialized(self, concrete_agent):
        """Test procesamiento sin inicializar."""
        input_data = AgentInput(data={"type": "test"})
        
        with pytest.raises(AgentException):
            await concrete_agent.process(input_data)
    
    @pytest.mark.asyncio
    async def test_process_not_ready(self, concrete_agent):
        """Test procesamiento cuando no está listo."""
        await concrete_agent.initialize({})
        concrete_agent.state = AgentState.ERROR
        
        input_data = AgentInput(data={"type": "test"})
        
        with pytest.raises(AgentException):
            await concrete_agent.process(input_data)
    
    @pytest.mark.asyncio
    async def test_process_validation_error(self, concrete_agent):
        """Test error de validación en procesamiento."""
        await concrete_agent.initialize({})
        
        # Entrada sin campo "type" requerido
        input_data = AgentInput(data={"content": "hello"})
        
        output = await concrete_agent.process(input_data)
        
        assert output.success is False
        assert "ValidationError" in output.errors[0] or "Missing type" in output.errors[0]
    
    @pytest.mark.asyncio
    async def test_process_internal_error(self, agent_config):
        """Test error interno en procesamiento."""
        class ErrorAgent(BaseAgent):
            async def _initialize_internal(self):
                return True
            
            async def _process_internal(self, input_data):
                raise Exception("Internal processing error")
            
            async def _learn_internal(self, feedback):
                return True
            
            def _validate_input_specific(self, input_data):
                pass
            
            async def _save_state(self):
                pass
        
        agent = ErrorAgent(agent_config)
        await agent.initialize({})
        
        input_data = AgentInput(data={"test": "data"})
        output = await agent.process(input_data)
        
        assert output.success is False
        assert "Internal processing error" in output.errors[0]
        
        # Verificar que se registró el error
        assert "Exception" in agent.metrics["error_types"]
        assert agent.metrics["error_types"]["Exception"] == 1
    
    @pytest.mark.asyncio
    async def test_learn_success(self, concrete_agent):
        """Test aprendizaje exitoso."""
        await concrete_agent.initialize({})
        
        feedback = {
            "type": "correction",
            "original": "wrong",
            "corrected": "right",
            "timestamp": datetime.now().isoformat()
        }
        
        success = await concrete_agent.learn(feedback)
        
        assert success is True
        assert concrete_agent.metrics["total_learning_events"] == 1
    
    @pytest.mark.asyncio
    async def test_learn_not_initialized(self, concrete_agent):
        """Test aprendizaje sin inicializar."""
        feedback = {"type": "test"}
        
        success = await concrete_agent.learn(feedback)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_learn_invalid_feedback(self, concrete_agent):
        """Test aprendizaje con feedback inválido."""
        await concrete_agent.initialize({})
        
        # Feedback sin campo "type" requerido
        feedback = {"message": "test"}
        
        success = await concrete_agent.learn(feedback)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_learn_with_confidence_impact(self, concrete_agent):
        """Test aprendizaje que impacta la confianza."""
        await concrete_agent.initialize({})
        
        initial_threshold = concrete_agent.config.confidence_threshold
        
        feedback = {
            "type": "reinforcement",
            "confidence_impact": 0.1,
            "timestamp": datetime.now().isoformat()
        }
        
        success = await concrete_agent.learn(feedback)
        
        assert success is True
        # La confianza debería haber aumentado
        assert concrete_agent.config.confidence_threshold == initial_threshold + 0.1
        
        # Test con impacto negativo (pero que no baje del mínimo)
        feedback["confidence_impact"] = -0.5
        await concrete_agent.learn(feedback)
        
        # Debería limitarse a mínimo 0.1
        assert concrete_agent.config.confidence_threshold >= 0.1
    
    @pytest.mark.asyncio
    async def test_evaluate(self, concrete_agent):
        """Test evaluación del agente."""
        await concrete_agent.initialize({})
        
        # Procesar algunas entradas
        for i in range(3):
            input_data = AgentInput(data={"type": "test", "index": i})
            await concrete_agent.process(input_data)
        
        evaluation = await concrete_agent.evaluate()
        
        assert evaluation["agent_id"] == concrete_agent.config.agent_id
        assert evaluation["agent_name"] == "TestAgent"
        assert evaluation["state"] == "ready"
        assert evaluation["initialized"] is True
        
        # Verificar métricas
        assert evaluation["metrics"]["requests_processed"] == 3
        assert "success_rate" in evaluation["metrics"]
        
        # Verificar memoria
        assert "memory_stats" in evaluation
        assert "short_term" in evaluation["memory_stats"]
        
        # Verificar configuración
        assert "config" in evaluation
        assert evaluation["config"]["confidence_threshold"] == 0.7
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, concrete_agent):
        """Test obtención de capacidades."""
        await concrete_agent.initialize({})
        
        capabilities = await concrete_agent.get_capabilities()
        
        assert len(capabilities) == 1
        assert capabilities[0]["name"] == "code_analysis"
        assert "description" in capabilities[0]
        assert "supported_languages" in capabilities[0]
        assert "examples" in capabilities[0]
    
    @pytest.mark.asyncio
    async def test_shutdown(self, concrete_agent):
        """Test apagado del agente."""
        await concrete_agent.initialize({})
        
        success = await concrete_agent.shutdown()
        
        assert success is True
        assert concrete_agent.state == AgentState.MAINTENANCE
        assert concrete_agent._initialized is False
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_memory(self, concrete_agent):
        """Test almacenamiento y recuperación de memoria."""
        await concrete_agent.initialize({})
        
        # Almacenar en diferentes tipos de memoria
        short_term_id = concrete_agent.store_memory(
            AgentMemoryType.SHORT_TERM,
            {"type": "fact", "value": "test1"}
        )
        
        episodic_id = concrete_agent.store_memory(
            AgentMemoryType.EPISODIC,
            {"event": "processing", "result": "success"}
        )
        
        # Recuperar
        short_term_memories = concrete_agent.retrieve_memory(
            AgentMemoryType.SHORT_TERM,
            query={"type": "fact"}
        )
        
        assert len(short_term_memories) == 1
        assert short_term_memories[0]["content"]["value"] == "test1"
        
        # Recuperar sin query
        all_episodic = concrete_agent.retrieve_memory(
            AgentMemoryType.EPISODIC,
            limit=10
        )
        
        assert len(all_episodic) == 1
    
    def test_validate_input_base(self, concrete_agent):
        """Test validación base de entrada."""
        # Entrada vacía
        input_data = AgentInput(data={})
        
        with pytest.raises(ValidationError):
            concrete_agent._validate_input(input_data)
        
        # Entrada válida
        input_data = AgentInput(data={"type": "test"})
        # No debería lanzar excepción
        concrete_agent._validate_input(input_data)
    
    def test_validate_output(self, concrete_agent):
        """Test validación de salida."""
        # Salida con confianza inválida
        output = AgentOutput(
            request_id="req",
            agent_id="agent",
            success=True,
            confidence=1.5  # Inválida
        )
        
        with pytest.raises(ValidationError):
            concrete_agent._validate_output(output)
        
        # Salida exitosa sin datos
        output = AgentOutput(
            request_id="req",
            agent_id="agent",
            success=True,
            data=None  # Debería tener datos
        )
        
        with pytest.raises(ValidationError):
            concrete_agent._validate_output(output)
        
        # Salida válida
        output = AgentOutput(
            request_id="req",
            agent_id="agent",
            success=True,
            data={"result": "ok"},
            confidence=0.8
        )
        
        # No debería lanzar excepción
        concrete_agent._validate_output(output)
    
    def test_validate_feedback(self, concrete_agent):
        """Test validación de feedback."""
        # Feedback inválido
        assert concrete_agent._validate_feedback({}) is False
        assert concrete_agent._validate_feedback({"type": "test"}) is False  # Sin timestamp
        
        # Feedback válido
        feedback = {
            "type": "correction",
            "timestamp": datetime.now().isoformat()
        }
        
        assert concrete_agent._validate_feedback(feedback) is True
    
    def test_update_metrics(self, concrete_agent):
        """Test actualización de métricas."""
        # Configurar estado inicial
        concrete_agent.metrics = {
            "requests_processed": 10,
            "success_rate": 1.0,
            "avg_processing_time_ms": 100.0
        }
        
        # Salida exitosa
        output = AgentOutput(
            request_id="req",
            agent_id="agent",
            success=True,
            confidence=0.9
        )
        
        concrete_agent._update_metrics(output, processing_time=50.0)
        
        assert concrete_agent.metrics["requests_processed"] == 11
        # success_rate debería seguir siendo 1.0 (10/10 -> 11/11)
        assert abs(concrete_agent.metrics["success_rate"] - 1.0) < 0.001
        # avg_processing_time: (100*10 + 50)/11 ≈ 95.45
        assert abs(concrete_agent.metrics["avg_processing_time_ms"] - 95.45) < 0.1
        
        # Salida fallida
        output.success = False
        concrete_agent._update_metrics(output, processing_time=30.0)
        
        assert concrete_agent.metrics["requests_processed"] == 12
        # success_rate: 11/12 ≈ 0.9167
        assert abs(concrete_agent.metrics["success_rate"] - 0.9167) < 0.001
    
    def test_get_capability_description(self, concrete_agent):
        """Test obtención de descripción de capacidad."""
        description = concrete_agent._get_capability_description(
            AgentCapability.CODE_ANALYSIS
        )
        
        assert isinstance(description, str)
        assert len(description) > 0
        
        # Capacidad desconocida
        unknown_cap = AgentCapability("unknown")
        description = concrete_agent._get_capability_description(unknown_cap)
        
        assert "No description" in description
    
    def test_get_supported_languages(self, concrete_agent):
        """Test obtención de lenguajes soportados."""
        languages = concrete_agent._get_supported_languages(
            AgentCapability.CODE_ANALYSIS
        )
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "python" in languages
    
    def test_get_capability_examples(self, concrete_agent):
        """Test obtención de ejemplos de capacidad."""
        examples = concrete_agent._get_capability_examples(
            AgentCapability.CODE_ANALYSIS
        )
        
        assert isinstance(examples, list)


class TestErrorHandling:
    """Pruebas de manejo de errores."""
    
    def test_agent_exception(self):
        """Test excepción de agente."""
        ex = AgentException("Test error")
        assert str(ex) == "Test error"
        
        # Debería ser subclase de BrainException
        from core.exceptions import BrainException
        assert issubclass(AgentException, BrainException)
    
    @pytest.mark.asyncio
    async def test_agent_handles_unexpected_errors(self, agent_config):
        """Test que el agente maneja errores inesperados."""
        class UnstableAgent(BaseAgent):
            async def _initialize_internal(self):
                return True
            
            async def _process_internal(self, input_data):
                # Error inesperado
                raise ValueError("Unexpected error")
            
            async def _learn_internal(self, feedback):
                return True
            
            def _validate_input_specific(self, input_data):
                pass
            
            async def _save_state(self):
                pass
        
        agent = UnstableAgent(agent_config)
        await agent.initialize({})
        
        input_data = AgentInput(data={"test": "data"})
        output = await agent.process(input_data)
        
        # Debería retornar output de error en lugar de lanzar excepción
        assert output.success is False
        assert "Unexpected error" in output.errors[0]
        
        # El agente debería seguir en estado READY
        assert agent.state == AgentState.READY


class TestMemoryManagement:
    """Pruebas de gestión de memoria."""
    
    @pytest.mark.asyncio
    async def test_memory_consolidation(self, concrete_agent):
        """Test consolidación automática de memoria."""
        await concrete_agent.initialize({})
        
        # Almacenar muchas memorias para trigger consolidación
        for i in range(15):
            concrete_agent.store_memory(
                AgentMemoryType.SHORT_TERM,
                {"index": i}
            )
            input_data = AgentInput(data={"type": "test", "index": i})
            await concrete_agent.process(input_data)
        
        # Después de 15 requests, debería haberse consolidado al menos una vez
        # (consolidación ocurre cada 10 requests)
        short_term_count = len(
            concrete_agent.memory.memories[AgentMemoryType.SHORT_TERM]
        )
        long_term_count = len(
            concrete_agent.memory.memories[AgentMemoryType.LONG_TERM]
        )
        
        # Algo de memoria debería haberse movido a largo plazo
        assert long_term_count > 0 or short_term_count < 15
    
    @pytest.mark.asyncio
    async def test_clear_memory(self, concrete_agent):
        """Test limpieza de memoria."""
        await concrete_agent.initialize({})
        
        # Almacenar en diferentes tipos
        concrete_agent.store_memory(AgentMemoryType.SHORT_TERM, {"test": "1"})
        concrete_agent.store_memory(AgentMemoryType.LONG_TERM, {"test": "2"})
        concrete_agent.store_memory(AgentMemoryType.EPISODIC, {"test": "3"})
        
        # Limpiar solo memoria de corto plazo
        concrete_agent.clear_memory(AgentMemoryType.SHORT_TERM)
        
        assert len(concrete_agent.memory.memories[AgentMemoryType.SHORT_TERM]) == 0
        assert len(concrete_agent.memory.memories[AgentMemoryType.LONG_TERM]) == 1
        assert len(concrete_agent.memory.memories[AgentMemoryType.EPISODIC]) == 1
        
        # Limpiar toda la memoria
        concrete_agent.clear_memory()
        
        for mem_type in AgentMemoryType:
            assert len(concrete_agent.memory.memories[mem_type]) == 0


class TestPerformance:
    """Pruebas de performance."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, agent_config):
        """Test procesamiento concurrente."""
        import time
        
        class FastAgent(BaseAgent):
            async def _initialize_internal(self):
                return True
            
            async def _process_internal(self, input_data):
                await asyncio.sleep(0.01)  # 10ms de procesamiento
                return AgentOutput(
                    request_id=input_data.request_id,
                    agent_id=self.config.agent_id,
                    success=True,
                    data={"index": input_data.data.get("index")},
                    confidence=0.9
                )
            
            async def _learn_internal(self, feedback):
                return True
            
            def _validate_input_specific(self, input_data):
                pass
            
            async def _save_state(self):
                pass
        
        agent = FastAgent(agent_config)
        await agent.initialize({})
        
        # Crear 100 solicitudes
        requests = [
            AgentInput(data={"type": "test", "index": i})
            for i in range(100)
        ]
        
        # Procesar concurrentemente
        start_time = time.time()
        tasks = [agent.process(req) for req in requests]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        # Verificar resultados
        assert len(results) == 100
        successful = [r for r in results if r.success]
        assert len(successful) == 100
        
        # Performance: 100 requests con 10ms cada una
        # Concurrentemente debería tomar ~100ms, pero con overhead
        assert elapsed < 2.0, f"100 requests concurrentes tomaron {elapsed:.3f}s"
        
        print(f"\nConcurrent processing performance: "
              f"100 requests en {elapsed:.3f}s "
              f"({100/elapsed:.1f} requests/segundo)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])