#!/usr/bin/env python3
"""
Tests unitarios para el sistema de health check de ANALYZERBRAIN.

Este módulo prueba:
- HealthCheckResult y su serialización
- SystemHealthChecker síncrono y asíncrono
- Checks individuales (ambiente, recursos, configuración, etc.)
- Manejo de errores y edge cases

Dependencias:
- pytest
- pytest-asyncio
- pytest-mock
- unittest.mock

Autor: ANALYZERBRAIN Team
Versión: 1.0.0
"""

import sys
import pytest
import socket
import asyncio
import platform
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List

import psutil

from src.core.health_check import (
    HealthStatus,
    HealthCheckResult,
    SystemHealthChecker,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Configuración mock para tests."""
    config = Mock()
    config.environment = "test"
    config.is_development = True
    config.is_production = False
    config.get = Mock(side_effect=lambda key, default=None: {
        "system.name": "TestSystem",
        "system.version": "1.0.0",
        "environment": "test",
        "storage.data_dir": "/test/data",
        "storage.log_dir": "/test/logs",
    }.get(key, default))
    return config


@pytest.fixture
def health_checker(mock_config):
    """Instancia de SystemHealthChecker con config mock."""
    return SystemHealthChecker(mock_config)


@pytest.fixture
def sample_healthy_result():
    """Resultado de health check saludable."""
    return HealthCheckResult(
        name="test_check",
        status=HealthStatus.HEALTHY,
        message="Todo está bien",
        details={"key": "value"}
    )


@pytest.fixture
def sample_unhealthy_result():
    """Resultado de health check no saludable."""
    return HealthCheckResult(
        name="test_check",
        status=HealthStatus.UNHEALTHY,
        message="Algo salió mal",
        details={"error": "detalles del error"},
        critical=True
    )


# -------------------------------------------------------------------
# Tests de HealthStatus Enum
# -------------------------------------------------------------------

def test_health_status_enum():
    """Verifica que HealthStatus tenga los valores correctos."""
    assert HealthStatus.HEALTHY.value == "healthy"
    assert HealthStatus.UNHEALTHY.value == "unhealthy"
    assert HealthStatus.WARNING.value == "warning"
    assert HealthStatus.ERROR.value == "error"


def test_health_status_membership():
    """Verifica que los miembros del enum sean accesibles."""
    assert HealthStatus.HEALTHY in HealthStatus
    assert HealthStatus.UNHEALTHY in HealthStatus
    assert HealthStatus.WARNING in HealthStatus
    assert HealthStatus.ERROR in HealthStatus


# -------------------------------------------------------------------
# Tests de HealthCheckResult
# -------------------------------------------------------------------

def test_health_check_result_creation(sample_healthy_result):
    """Verifica la creación básica de HealthCheckResult."""
    assert sample_healthy_result.name == "test_check"
    assert sample_healthy_result.status == HealthStatus.HEALTHY
    assert sample_healthy_result.message == "Todo está bien"
    assert sample_healthy_result.details == {"key": "value"}
    assert sample_healthy_result.critical is True
    assert isinstance(sample_healthy_result.timestamp, datetime)


def test_health_check_result_non_critical():
    """Verifica HealthCheckResult con critical=False."""
    result = HealthCheckResult(
        name="non_critical_check",
        status=HealthStatus.WARNING,
        message="Advertencia no crítica",
        details={},
        critical=False
    )
    assert result.critical is False


def test_health_check_result_to_dict(sample_healthy_result):
    """Verifica la serialización a diccionario."""
    result_dict = sample_healthy_result.to_dict()
    
    assert result_dict["name"] == "test_check"
    assert result_dict["status"] == "healthy"
    assert result_dict["message"] == "Todo está bien"
    assert result_dict["critical"] is True
    assert isinstance(result_dict["timestamp"], str)
    assert result_dict["details"] == {"key": "value"}


def test_health_check_result_timestamp_iso_format():
    """Verifica que el timestamp esté en formato ISO."""
    result = HealthCheckResult(
        name="test",
        status=HealthStatus.HEALTHY,
        message="test",
        details={}
    )
    
    # Verificar que se puede parsear como datetime ISO
    parsed_time = datetime.fromisoformat(result.timestamp.isoformat())
    assert isinstance(parsed_time, datetime)


# -------------------------------------------------------------------
# Tests de SystemHealthChecker - Inicialización
# -------------------------------------------------------------------

def test_system_health_checker_initialization():
    """Verifica la inicialización de SystemHealthChecker."""
    config_mock = Mock()
    checker = SystemHealthChecker(config_mock)
    
    assert checker.config == config_mock
    assert checker.results == {}
    assert checker._component_status == {}


def test_system_health_checker_default_config():
    """Verifica que SystemHealthChecker use config por defecto si no se proporciona."""
    # Necesitamos mockear la configuración global
    with patch('src.core.health_check.config') as mock_global_config:
        checker = SystemHealthChecker()
        assert checker.config == mock_global_config


# -------------------------------------------------------------------
# Tests de SystemHealthChecker - Métodos síncronos
# -------------------------------------------------------------------

def test_check_python_environment_sync_healthy(health_checker):
    """Verifica check_python_environment_sync con Python 3.9+."""
    # Mock sys.version_info para simular Python 3.9
    with patch('sys.version_info', Mock(major=3, minor=9, micro=0)):
        result = health_checker._check_python_environment_sync()
        
        assert result.name == "python_environment"
        assert result.status == HealthStatus.HEALTHY
        assert "Python 3.9 compatible" in result.message
        assert "version" in result.details


def test_check_python_environment_sync_unhealthy(health_checker):
    """Verifica check_python_environment_sync con Python antiguo."""
    # Mock sys.version_info para simular Python 3.6
    with patch('sys.version_info', Mock(major=3, minor=6, micro=0)):
        result = health_checker._check_python_environment_sync()
        
        assert result.name == "python_environment"
        assert result.status == HealthStatus.UNHEALTHY
        assert "no compatible" in result.message


def test_check_system_resources_sync_healthy(health_checker):
    """Verifica check_system_resources_sync con recursos saludables."""
    mock_cpu = 50.0
    mock_memory = Mock(percent=60.0, available=4 * 1024**3)  # 4GB disponibles
    
    with patch('psutil.cpu_percent', return_value=mock_cpu), \
         patch('psutil.virtual_memory', return_value=mock_memory):
        result = health_checker._check_system_resources_sync()
        
        assert result.name == "system_resources"
        assert result.status == HealthStatus.HEALTHY
        assert "OK" in result.message
        assert result.details["cpu_percent"] == mock_cpu
        assert result.details["memory_percent"] == mock_memory.percent


def test_check_system_resources_sync_warning(health_checker):
    """Verifica check_system_resources_sync con recursos altos."""
    mock_cpu = 95.0  # CPU alta
    mock_memory = Mock(percent=95.0, available=0.5 * 1024**3)  # 0.5GB disponibles
    
    with patch('psutil.cpu_percent', return_value=mock_cpu), \
         patch('psutil.virtual_memory', return_value=mock_memory):
        result = health_checker._check_system_resources_sync()
        
        assert result.name == "system_resources"
        assert result.status == HealthStatus.WARNING
        assert "altos" in result.message.lower()


def test_check_system_resources_sync_error(health_checker):
    """Verifica check_system_resources_sync cuando psutil falla."""
    with patch('psutil.cpu_percent', side_effect=Exception("Error de psutil")):
        result = health_checker._check_system_resources_sync()
        
        assert result.name == "system_resources"
        assert result.status == HealthStatus.WARNING
        assert "Error" in result.message
        assert "error" in result.details


def test_check_configuration_sync(health_checker):
    """Verifica check_configuration_sync."""
    result = health_checker._check_configuration_sync()
    
    assert result.name == "configuration"
    assert result.status == HealthStatus.HEALTHY
    assert health_checker.config.environment in result.message
    assert result.details["environment"] == health_checker.config.environment


def test_check_file_system_sync_healthy(health_checker):
    """Verifica check_file_system_sync exitoso."""
    with patch.object(health_checker.config, 'get') as mock_get:
        mock_get.side_effect = lambda key, default=None: {
            "storage.data_dir": "/test/data",
            "storage.log_dir": "/test/logs",
        }.get(key, default)
        
        # Mock específico para Path
        with patch('src.core.health_check.Path') as mock_path_class:
            mock_dir_instance = Mock()
            mock_dir_instance.mkdir = Mock()
            mock_path_class.return_value = mock_dir_instance
            
            result = health_checker._check_file_system_sync()
            
            assert result.name == "file_system"
            assert result.status == HealthStatus.HEALTHY
            assert "creados/verificados" in result.message


def test_check_file_system_sync_unhealthy(health_checker):
    """Verifica check_file_system_sync con error."""
    with patch.object(health_checker.config, 'get') as mock_get:
        mock_get.side_effect = lambda key, default=None: {
            "storage.data_dir": "/test/data",
            "storage.log_dir": "/test/logs",
        }.get(key, default)
        
        # Mock específico para Path
        with patch('src.core.health_check.Path') as mock_path_class:
            mock_dir_instance = Mock()
            mock_dir_instance.mkdir = Mock(side_effect=PermissionError("Permiso denegado"))
            mock_path_class.return_value = mock_dir_instance
            
            result = health_checker._check_file_system_sync()
            
            assert result.name == "file_system"
            assert result.status == HealthStatus.UNHEALTHY
            assert "Error" in result.message
            assert "error" in result.details


def test_check_dependencies_sync_healthy(health_checker):
    """Verifica check_dependencies_sync con todas las dependencias instaladas."""
    # Mock find_spec para simular dependencias instaladas
    with patch('importlib.util.find_spec', return_value=True):
        result = health_checker._check_dependencies_sync()
        
        assert result.name == "dependencies"
        assert result.status == HealthStatus.HEALTHY
        assert "instalada" in result.message.lower()


def test_check_dependencies_sync_unhealthy(health_checker):
    """Verifica check_dependencies_sync con dependencias faltantes."""
    # Mock find_spec para simular dependencia faltante
    with patch('importlib.util.find_spec', side_effect=lambda x: None if x == "pydantic" else True):
        result = health_checker._check_dependencies_sync()
        
        assert result.name == "dependencies"
        assert result.status == HealthStatus.UNHEALTHY
        assert "faltantes" in result.message.lower()
        assert "missing" in result.details


# -------------------------------------------------------------------
# Tests de SystemHealthChecker - check_all_sync
# -------------------------------------------------------------------

def test_check_all_sync_success(health_checker):
    """Verifica check_all_sync exitoso."""
    # Mock todos los checks individuales para devolver resultados saludables
    with patch.object(health_checker, '_check_python_environment_sync') as mock_py, \
         patch.object(health_checker, '_check_system_resources_sync') as mock_res, \
         patch.object(health_checker, '_check_configuration_sync') as mock_conf, \
         patch.object(health_checker, '_check_file_system_sync') as mock_fs, \
         patch.object(health_checker, '_check_dependencies_sync') as mock_deps:
        
        # Configurar mocks para devolver resultados saludables
        mock_py.return_value = HealthCheckResult(
            name="python_environment", 
            status=HealthStatus.HEALTHY, 
            message="OK", 
            details={}
        )
        mock_res.return_value = HealthCheckResult(
            name="system_resources", 
            status=HealthStatus.HEALTHY, 
            message="OK", 
            details={}
        )
        mock_conf.return_value = HealthCheckResult(
            name="configuration", 
            status=HealthStatus.HEALTHY, 
            message="OK", 
            details={}
        )
        mock_fs.return_value = HealthCheckResult(
            name="file_system", 
            status=HealthStatus.HEALTHY, 
            message="OK", 
            details={}
        )
        mock_deps.return_value = HealthCheckResult(
            name="dependencies", 
            status=HealthStatus.HEALTHY, 
            message="OK", 
            details={}
        )
        
        result = health_checker.check_all_sync()
        
        assert result["overall"] is True
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert len(result["checks"]) == 5  # Todos los checks síncronos
        assert "summary" in result


def test_check_all_sync_with_failure(health_checker):
    """Verifica check_all_sync con fallo crítico."""
    with patch.object(health_checker, '_check_python_environment_sync') as mock_py, \
         patch.object(health_checker, '_check_dependencies_sync') as mock_deps:
        
        # Python check falla (crítico)
        mock_py.return_value = HealthCheckResult(
            name="python_environment", 
            status=HealthStatus.UNHEALTHY, 
            message="Python incompatible", 
            details={},
            critical=True
        )
        
        # Dependencies check pasa
        mock_deps.return_value = HealthCheckResult(
            name="dependencies", 
            status=HealthStatus.HEALTHY, 
            message="OK", 
            details={}
        )
        
        # Mock otros checks para evitar errores
        health_checker._check_system_resources_sync = Mock(return_value=HealthCheckResult(
            name="system_resources", status=HealthStatus.HEALTHY, message="OK", details={}))
        health_checker._check_configuration_sync = Mock(return_value=HealthCheckResult(
            name="configuration", status=HealthStatus.HEALTHY, message="OK", details={}))
        health_checker._check_file_system_sync = Mock(return_value=HealthCheckResult(
            name="file_system", status=HealthStatus.HEALTHY, message="OK", details={}))
        
        result = health_checker.check_all_sync()
        
        assert result["overall"] is False
        assert result["status"] == "unhealthy"
        assert any(check["name"] == "python_environment" and check["status"] == "unhealthy" 
                   for check in result["checks"])


def test_check_all_sync_exception_handling(health_checker):
    """Verifica que check_all_sync maneje excepciones correctamente."""
    with patch.object(health_checker, '_check_python_environment_sync', 
                     side_effect=Exception("Error inesperado")):
        
        result = health_checker.check_all_sync()
        
        # Debería tener un check con error
        error_checks = [c for c in result["checks"] if c["name"] == "python_environment" and c["status"] == "error"]
        assert len(error_checks) == 1
        assert result["overall"] is False  # El check es crítico


# -------------------------------------------------------------------
# Tests de SystemHealthChecker - Métodos asíncronos
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_python_environment_async_healthy(health_checker):
    """Verifica check_python_environment con Python 3.9+."""
    # Mock sys.version_info para simular Python 3.11
    with patch('sys.version_info', Mock(major=3, minor=11, micro=0)):
        result = await health_checker._check_python_environment()
        
        assert result.name == "python_environment"
        assert result.status == HealthStatus.HEALTHY
        assert "3.11" in result.details["version"]


@pytest.mark.asyncio
async def test_check_system_resources_async_healthy(health_checker):
    """Verifica check_system_resources con recursos saludables."""
    mock_cpu_percent = 50.0
    mock_memory = Mock(
        total=16 * 1024**3,  # 16GB
        available=8 * 1024**3,  # 8GB
        percent=50.0,
        used=8 * 1024**3
    )
    mock_disk = Mock(
        total=500 * 1024**3,  # 500GB
        free=250 * 1024**3,   # 250GB
        percent=50.0
    )
    
    with patch('psutil.cpu_percent', return_value=mock_cpu_percent), \
         patch('psutil.virtual_memory', return_value=mock_memory), \
         patch('psutil.disk_usage', return_value=mock_disk), \
         patch('psutil.cpu_count', return_value=8):
        
        result = await health_checker._check_system_resources()
        
        assert result.name == "system_resources"
        assert result.status == HealthStatus.HEALTHY
        assert "OK" in result.message
        assert "cpu" in result.details
        assert "memory" in result.details
        assert "disk" in result.details


@pytest.mark.asyncio
async def test_check_configuration_async_complete(health_checker):
    """Verifica check_configuration con configuración completa."""
    result = await health_checker._check_configuration()
    
    assert result.name == "configuration"
    assert result.status == HealthStatus.HEALTHY
    assert "completa" in result.message


@pytest.mark.asyncio
async def test_check_configuration_async_missing_keys(health_checker):
    """Verifica check_configuration con claves faltantes."""
    # Configurar mock para devolver None para una clave requerida
    health_checker.config.get = Mock(return_value=None)
    
    result = await health_checker._check_configuration()
    
    assert result.name == "configuration"
    assert result.status == HealthStatus.UNHEALTHY
    assert "incompleta" in result.message
    assert "missing_keys" in result.details


@pytest.mark.asyncio
async def test_check_file_system_async_healthy(health_checker):
    """Verifica check_file_system exitoso."""
    # Mock config.get para devolver strings en lugar de Path objects
    with patch.object(health_checker.config, 'get') as mock_get:
        mock_get.side_effect = lambda key, default=None: {
            "storage.data_dir": "/test/data",
            "storage.log_dir": "/test/logs",
        }.get(key, default)
        
        # Mock más específico para evitar problemas con Path
        with patch('src.core.health_check.Path') as mock_path_class:
            # Crear instancias mock para Path
            mock_dir_instance = Mock()
            mock_file_instance = Mock()
            
            # Configurar el comportamiento de las instancias
            mock_dir_instance.mkdir = Mock()
            mock_dir_instance.name = "test"  # Para dirs_status[dir_path.name]
            mock_dir_instance.__truediv__ = Mock(return_value=mock_file_instance)
            
            mock_file_instance.write_text = Mock(return_value=10)
            mock_file_instance.unlink = Mock()
            
            # Cuando se llama a Path() debe devolver nuestra instancia mock
            mock_path_class.return_value = mock_dir_instance
            
            result = await health_checker._check_file_system()
            
            assert result.name == "file_system"
            assert result.status == HealthStatus.HEALTHY
            assert "OK" in result.message
            
            # Verificar que mkdir fue llamado
            assert mock_dir_instance.mkdir.called


@pytest.mark.asyncio
async def test_check_file_system_async_permission_error(health_checker):
    """Verifica check_file_system con error de permisos."""
    with patch.object(health_checker.config, 'get') as mock_get:
        mock_get.side_effect = lambda key, default=None: {
            "storage.data_dir": "/test/data",
            "storage.log_dir": "/test/logs",
        }.get(key, default)
        
        # Mock específico para Path
        with patch('src.core.health_check.Path') as mock_path_class:
            mock_dir_instance = Mock()
            mock_dir_instance.mkdir = Mock(side_effect=PermissionError("Permiso denegado"))
            mock_dir_instance.name = "test"
            mock_path_class.return_value = mock_dir_instance
            
            result = await health_checker._check_file_system()
            
            assert result.name == "file_system"
            assert result.status == HealthStatus.UNHEALTHY
            assert "Permiso" in result.message or "error" in result.message.lower()


@pytest.mark.asyncio
async def test_check_dependencies_async_healthy(health_checker):
    """Verifica check_dependencies con todas las dependencias instaladas."""
    # Mock __import__ para simular módulos instalados
    mock_module = Mock(__version__="2.5.0")
    
    with patch('builtins.__import__', return_value=mock_module):
        result = await health_checker._check_dependencies()
        
        assert result.name == "dependencies"
        assert result.status == HealthStatus.HEALTHY
        assert "instaladas" in result.message


@pytest.mark.asyncio
async def test_check_dependencies_async_missing(health_checker):
    """Verifica check_dependencies con dependencias faltantes."""
    # Mock __import__ para lanzar ImportError para pydantic
    def mock_import(name):
        if name == "pydantic":
            raise ImportError("No module named 'pydantic'")
        mock_module = Mock(__version__="1.0.0")
        return mock_module
    
    with patch('builtins.__import__', side_effect=mock_import):
        result = await health_checker._check_dependencies()
        
        assert result.name == "dependencies"
        assert result.status == HealthStatus.UNHEALTHY
        assert "faltantes" in result.message
        assert "pydantic" in result.details["missing"]


@pytest.mark.asyncio
async def test_check_network_basic_async_healthy(health_checker):
    """Verifica check_network_basic exitoso."""
    with patch('socket.gethostbyname', return_value="127.0.0.1"):
        result = await health_checker._check_network_basic()
        
        assert result.name == "network_basic"
        assert result.status == HealthStatus.HEALTHY
        assert "OK" in result.message


@pytest.mark.asyncio
async def test_check_network_basic_async_error(health_checker):
    """Verifica check_network_basic con error."""
    with patch('socket.gethostbyname', side_effect=socket.gaierror("Error de resolución")):
        result = await health_checker._check_network_basic()
        
        assert result.name == "network_basic"
        assert result.status == HealthStatus.WARNING
        assert "Problema" in result.message or "error" in result.message.lower()


@pytest.mark.asyncio
async def test_check_all_async_comprehensive(health_checker):
    """Verifica check_all con múltiples resultados."""
    # Mock todos los métodos async
    async_mocks = {
        '_check_python_environment': HealthCheckResult(
            name="python_environment", status=HealthStatus.HEALTHY, message="OK", details={}
        ),
        '_check_system_resources': HealthCheckResult(
            name="system_resources", status=HealthStatus.WARNING, message="CPU alta", details={}
        ),
        '_check_configuration': HealthCheckResult(
            name="configuration", status=HealthStatus.HEALTHY, message="OK", details={}
        ),
        '_check_file_system': HealthCheckResult(
            name="file_system", status=HealthStatus.HEALTHY, message="OK", details={}
        ),
        '_check_dependencies': HealthCheckResult(
            name="dependencies", status=HealthStatus.HEALTHY, message="OK", details={}
        ),
        '_check_network_basic': HealthCheckResult(
            name="network_basic", status=HealthStatus.HEALTHY, message="OK", details={}
        ),
    }
    
    for method_name, return_value in async_mocks.items():
        setattr(health_checker, method_name, AsyncMock(return_value=return_value))
    
    result = await health_checker.check_all()
    
    assert result["status"] == "warning"  # Por el warning en system_resources
    assert len(result["checks"]) == 6  # Todos los checks async
    assert "summary" in result
    assert result["summary"]["warnings"] >= 1


# -------------------------------------------------------------------
# Tests de SystemHealthChecker - Métodos auxiliares
# -------------------------------------------------------------------

def test_get_config_files(health_checker):
    """Verifica _get_config_files."""
    with patch('pathlib.Path.exists', side_effect=lambda: True):
        files = health_checker._get_config_files()
        
        assert isinstance(files, list)
        assert len(files) == 3  # system_config.yaml, agent_config.yaml, .env


def test_generate_summary_empty(health_checker):
    """Verifica _generate_summary sin resultados."""
    summary = health_checker._generate_summary()
    assert summary == {}


def test_generate_summary_with_results(health_checker):
    """Verifica _generate_summary con resultados."""
    health_checker.results = {
        "check1": HealthCheckResult(
            name="check1", status=HealthStatus.HEALTHY, message="OK", details={}
        ),
        "check2": HealthCheckResult(
            name="check2", status=HealthStatus.WARNING, message="Warning", details={}
        ),
        "check3": HealthCheckResult(
            name="check3", status=HealthStatus.UNHEALTHY, message="Error", details={}, critical=True
        ),
    }
    
    summary = health_checker._generate_summary()
    
    assert summary["total_checks"] == 3
    assert summary["healthy"] == 1
    assert summary["warnings"] == 1
    assert summary["unhealthy"] == 1
    assert summary["critical_failed"] is True
    assert 0 <= summary["success_rate"] <= 100


def test_get_status_not_checked(health_checker):
    """Verifica get_status cuando no se han ejecutados checks."""
    status = health_checker.get_status()
    assert status["status"] == "not_checked"
    assert "no ejecutado" in status["message"]


def test_get_status_after_check(health_checker):
    """Verifica get_status después de ejecutar checks."""
    # Agregar resultados simulados
    from datetime import datetime, timedelta
    timestamp = datetime.now() - timedelta(minutes=5)
    
    health_checker.results = {
        "test": HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            details={},
            timestamp=timestamp
        )
    }
    
    status = health_checker.get_status()
    assert status["status"] == "healthy"
    assert "summary" in status
    assert status["checks_count"] == 1
    assert status["last_check"] == timestamp.isoformat()


def test_print_detailed_report_empty(health_checker):
    """Verifica print_detailed_report sin resultados."""
    report = health_checker.print_detailed_report()
    assert "No hay resultados" in report


def test_print_detailed_report_with_results(health_checker):
    """Verifica print_detailed_report con resultados."""
    health_checker.results = {
        "critical_healthy": HealthCheckResult(
            name="critical_healthy", 
            status=HealthStatus.HEALTHY, 
            message="Critical OK", 
            details={},
            critical=True
        ),
        "non_critical_warning": HealthCheckResult(
            name="non_critical_warning", 
            status=HealthStatus.WARNING, 
            message="Warning", 
            details={},
            critical=False
        ),
        "critical_error": HealthCheckResult(
            name="critical_error", 
            status=HealthStatus.ERROR, 
            message="Big error", 
            details={"error": "details"},
            critical=True
        ),
    }
    
    report = health_checker.print_detailed_report()
    
    # Verificar contenido básico del reporte
    assert "REPORTE DETALLADO" in report
    assert "critical_healthy" in report
    assert "non_critical_warning" in report
    assert "critical_error" in report
    assert "Resumen:" in report
    assert "✅" in report  # Icono healthy
    assert "⚠️" in report   # Icono warning
    assert "💥" in report   # Icono error


# -------------------------------------------------------------------
# Tests de edge cases y manejo de errores
# -------------------------------------------------------------------

def test_check_all_sync_partial_failure(health_checker):
    """Verifica check_all_sync con fallo parcial (warning)."""
    with patch.object(health_checker, '_check_system_resources_sync') as mock_res:
        # Solo un warning, no crítico
        mock_res.return_value = HealthCheckResult(
            name="system_resources", 
            status=HealthStatus.WARNING, 
            message="CPU un poco alta", 
            details={},
            critical=True  # Pero es warning, no unhealthy
        )
        
        # Mock otros checks como healthy
        health_checker._check_python_environment_sync = Mock(return_value=HealthCheckResult(
            name="python_environment", status=HealthStatus.HEALTHY, message="OK", details={}))
        health_checker._check_configuration_sync = Mock(return_value=HealthCheckResult(
            name="configuration", status=HealthStatus.HEALTHY, message="OK", details={}))
        health_checker._check_file_system_sync = Mock(return_value=HealthCheckResult(
            name="file_system", status=HealthStatus.HEALTHY, message="OK", details={}))
        health_checker._check_dependencies_sync = Mock(return_value=HealthCheckResult(
            name="dependencies", status=HealthStatus.HEALTHY, message="OK", details={}))
        
        result = health_checker.check_all_sync()
        
        assert result["status"] == "warning"
        assert result["overall"] is True  # Warning no afecta overall


def test_critical_failed_logic(health_checker):
    """Verifica la lógica de critical_failed en _generate_summary."""
    # Solo un error no crítico no debería marcar critical_failed
    health_checker.results = {
        "test": HealthCheckResult(
            name="test", 
            status=HealthStatus.UNHEALTHY, 
            message="Error no crítico", 
            details={},
            critical=False  # No es crítico
        )
    }
    
    summary = health_checker._generate_summary()
    assert summary["critical_failed"] is False
    
    # Error crítico sí debería marcar critical_failed
    health_checker.results = {
        "test": HealthCheckResult(
            name="test", 
            status=HealthStatus.UNHEALTHY, 
            message="Error crítico", 
            details={},
            critical=True  # Es crítico
        )
    }
    
    summary = health_checker._generate_summary()
    assert summary["critical_failed"] is True


# -------------------------------------------------------------------
# Tests de integración mínima
# -------------------------------------------------------------------

def test_health_checker_integration_minimal():
    """Test de integración mínima del health checker."""
    # Crear instancia real sin mocks (dependerá del entorno)
    checker = SystemHealthChecker()
    
    # Solo verificar que se puede crear y tiene los métodos esperados
    assert hasattr(checker, 'check_all_sync')
    assert hasattr(checker, 'check_all')
    assert hasattr(checker, 'get_status')
    assert hasattr(checker, 'print_detailed_report')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])