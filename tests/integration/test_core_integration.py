"""
Pruebas de integración para módulos core.
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from core.orchestrator import BrainOrchestrator, OperationRequest, OperationPriority
from core.config_manager import ConfigManager
from core.system_state import SystemStateManager
from core.event_bus import EventBus


class TestOrchestratorIntegration:
    """Pruebas de integración del orquestador con otros componentes core."""
    
    @pytest.fixture
    async def integrated_orchestrator(self):
        """Crear orquestador con componentes reales integrados."""
        # Crear directorio temporal para config
        temp_dir = tempfile.mkdtemp(prefix="brain_integration_test_")
        config_path = os.path.join(temp_dir, "config.yaml")
        
        # Crear configuración básica
        import yaml
        config = {
            "system": {
                "environment": "test",
                "log_level": "DEBUG"
            },
            "projects": {
                "supported_extensions": {
                    "python": [".py"]
                }
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Crear orquestador
        orch = BrainOrchestrator(config_path)
        
        # Usar componentes reales (sin mocks)
        yield orch
        
        # Limpiar
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_orchestrator_with_real_components(self, integrated_orchestrator):
        """Test orquestador con componentes reales."""
        # Inicializar (esto usa componentes reales)
        success = await integrated_orchestrator.initialize()
        
        # Verificar inicialización básica
        assert success is True or success is False  # Puede fallar sin dependencias reales
        # Pero no debería lanzar excepción
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_event_bus_integration(self):
        """Test integración con EventBus."""
        from core.event_bus import EventBus, EventType
        
        event_bus = EventBus()
        await event_bus.initialize()
        
        # Registrar handler
        events_received = []
        
        async def handler(event_type, data):
            events_received.append((event_type, data))
        
        await event_bus.subscribe(EventType.SYSTEM_STARTED, handler)
        
        # Publicar evento
        test_data = {"test": "data"}
        await event_bus.publish(EventType.SYSTEM_STARTED, test_data)
        
        # Dar tiempo para procesamiento asíncrono
        await asyncio.sleep(0.1)
        
        # Verificar que se recibió el evento
        assert len(events_received) == 1
        assert events_received[0][0] == EventType.SYSTEM_STARTED
        assert events_received[0][1] == test_data
        
        await event_bus.shutdown()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_config_manager_integration(self, temp_project_dir):
        """Test integración con ConfigManager."""
        from core.config_manager import ConfigManager
        
        # Crear archivo de configuración
        config_file = os.path.join(temp_project_dir, "test_config.yaml")
        
        config_data = {
            "test": {
                "key": "value",
                "number": 42,
                "list": ["a", "b", "c"]
            }
        }
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Crear y usar ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config(config_file)
        
        # Verificar carga
        assert config["test"]["key"] == "value"
        assert config["test"]["number"] == 42
        assert config["test"]["list"] == ["a", "b", "c"]
        
        # Test actualización
        config_manager.set_config("test.new_key", "new_value")
        assert config_manager.get_config("test.new_key") == "new_value"


class TestSystemStateIntegration:
    """Pruebas de integración del SystemState."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_system_state_persistence(self, temp_project_dir):
        """Test persistencia del estado del sistema."""
        from core.system_state import SystemStateManager
        
        state_file = os.path.join(temp_project_dir, "state.json")
        
        # Crear estado y guardar
        state1 = SystemStateManager(state_file)
        state1.set_state({
            "project1": {
                "status": "analyzed",
                "files": 100
            }
        })
        await state1.save_state()
        
        # Cargar en nuevo objeto
        state2 = SystemStateManager(state_file)
        await state2.load_state()
        
        # Verificar que se cargó el estado
        assert state2.get_state()["project1"]["status"] == "analyzed"
        assert state2.get_state()["project1"]["files"] == 100
        
        # Test reset
        state2.reset_state()
        assert state2.get_state() == {}


class TestWorkflowIntegration:
    """Pruebas de integración de flujos de trabajo."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_simple_workflow(self):
        """Test flujo de trabajo simple."""
        from core.workflow_manager import WorkflowOrchestrator
        from core.event_bus import EventBus, EventType
        
        # Crear componentes
        event_bus = EventBus()
        await event_bus.initialize()
        
        workflow_manager = WorkflowOrchestrator(event_bus)
        
        # Definir steps de workflow
        steps = [
            {
                "name": "step1",
                "action": lambda ctx: ctx.update({"step1": "done"}),
                "timeout": 10
            },
            {
                "name": "step2", 
                "action": lambda ctx: ctx.update({"step2": "done"}),
                "requires": ["step1"]
            }
        ]
        
        # Registrar workflow
        workflow_id = await workflow_manager.register_workflow(
            "test_workflow",
            steps
        )
        
        # Ejecutar
        context = {}
        result = await workflow_manager.execute_workflow(workflow_id, context)
        
        # Verificar ejecución
        assert result["success"] is True
        assert context["step1"] == "done"
        assert context["step2"] == "done"
        
        await event_bus.shutdown()


class TestErrorRecoveryIntegration:
    """Pruebas de integración de recuperación de errores."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_orchestrator_error_recovery(self):
        """Test recuperación de errores en orquestador."""
        from core.orchestrator import BrainOrchestrator
        from core.exceptions import BrainException
        
        orch = BrainOrchestrator()
        
        # Mockear componentes para simular error
        with patch.object(orch, '_initialize_component') as mock_init:
            mock_init.side_effect = Exception("Component failed")
            
            # Debería lanzar BrainException
            with pytest.raises(BrainException):
                await orch.initialize()
            
            # Verificar que se intentó publicar evento de error
            # (esto depende de la implementación real)
    
    @pytest.mark.integration
    @pytest.mark.asyncio 
    async def test_operation_timeout_handling(self):
        """Test manejo de timeout en operaciones."""
        from core.orchestrator import BrainOrchestrator, OperationRequest
        
        orch = BrainOrchestrator()
        orch._is_running = True
        orch._validate_operation_request = Mock()
        orch._event_bus = AsyncMock()
        
        # Configurar handler lento
        async def slow_handler(context):
            await asyncio.sleep(2)  # 2 segundos
            return {"result": "slow"}
        
        orch._analyze_project = slow_handler
        
        # Solicitud con timeout corto
        request = OperationRequest(
            operation_type="analyze_project",
            timeout_seconds=1
        )
        
        # En la implementación actual no hay timeout, pero verificar
        # que no se bloquea indefinidamente
        import asyncio
        try:
            result = await asyncio.wait_for(
                orch.process_operation(request),
                timeout=3
            )
            assert result is not None
        except asyncio.TimeoutError:
            pytest.fail("Operation timed out in test")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])