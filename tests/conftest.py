"""
Modulo: tests/conftest.py
Autor: ANALYZERBRAIN Team
Fecha: 2026-02-01

Descripcion:
    Archivo central de configuración para pytest.
    Define fixtures compartidas entre todos los tests del proyecto.

Caracteristicas:
    - Creación de directorios temporales aislados
    - Configuración de event loop para pruebas asíncronas
    - Configuración YAML de ejemplo para tests
    - Utilidades comunes para mocking

Uso:
    Pytest carga automáticamente este archivo.
    Las fixtures definidas aquí están disponibles en todos los tests.

Dependencias:
    - pytest
    - pytest-asyncio
    - asyncio
    - tempfile
    - pathlib

Excepciones:
    Ninguna (las excepciones se gestionan a nivel de test).
"""

import asyncio
import tempfile
from pathlib import Path
from collections.abc import Iterator
from unittest.mock import Mock, AsyncMock

import pytest


# -------------------------
# Fixtures de filesystem
# -------------------------

@pytest.fixture
def temp_config_dir() -> Iterator[Path]:  # type: ignore
    """
    Directorio temporal para configuraciones y archivos de test.
    Se elimina automáticamente al finalizar el test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_yaml_config() -> str:
    """
    Configuración YAML de ejemplo para tests.
    """
    return """
system:
  name: "TEST_ANALYZER"
  version: "1.0.0"
  max_workers: 2

storage:
  data_dir: "./test_data"
  log_dir: "./test_logs"
"""


@pytest.fixture
def yaml_config_file(temp_config_dir: Path, sample_yaml_config: str) -> Path:
    """
    Archivo YAML temporal escrito en disco.
    """
    config_path = temp_config_dir / "config.yaml"
    config_path.write_text(sample_yaml_config, encoding="utf-8")
    return config_path


# -------------------------
# Fixtures async / event loop
# -------------------------

@pytest.fixture(scope="session")
def event_loop():
    """
    Event loop dedicado para toda la sesión de tests.
    Evita conflictos con pytest-asyncio.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# -------------------------
# Fixtures de mocking
# -------------------------

@pytest.fixture
def mock_logger():
    """
    Logger mockeado para evitar escritura real en logs.
    """
    return Mock()


@pytest.fixture
def mock_async_function():
    """
    Función async mockeada para pruebas asíncronas.
    """
    return AsyncMock()
# PENDING - Implementar imports reales
# from src.config.config_manager import ConfigManager


@pytest.fixture(autouse=True)
def cleanup_config_manager():
    """Fixture que limpia ConfigManager antes y después de cada test."""
    # PENDING: Implementar cleanup
    # ConfigManager._instance = None
    # ConfigManager._settings = None
    # ConfigManager._custom_config = {}
    yield
    # PENDING: Implementar cleanup
    # ConfigManager._instance = None
    # ConfigManager._settings = None
    # ConfigManager._custom_config = {}


@pytest.fixture
def temp_config_dir():
    """Fixture que crea directorio temporal para archivos de configuración."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars():
    """Fixture que mockea variables de entorno."""
    with patch.dict(os.environ, {  # type: ignore
        'ENVIRONMENT': 'testing',
        'LOG_LEVEL': 'DEBUG',
        'SYSTEM__MAX_WORKERS': '8',
        'API__PORT': '9000'
    }):
        yield


@pytest.fixture
def mock_yaml_files():
    """Fixture que mockea archivos YAML."""
    yaml_content = { # type: ignore
        'system': {'max_workers': 10, 'name': 'YAML_CONFIG'},
        'api': {'port': 8080, 'cors_origins': ['http://test.com']}
    }
    
    with patch('pathlib.Path.exists', return_value=True): # type: ignore
        with patch('builtins.open'): # type: ignore
            with patch('yaml.safe_load', return_value=yaml_content): # type: ignore
                yield


@pytest.fixture
def config_instance():
    """Fixture que retorna una instancia limpia de ConfigManager."""
    # PENDING: Implementar
    # ConfigManager._instance = None
    # return ConfigManager()
    pass


@pytest.fixture(params=['development', 'production', 'testing'])
def environment_param(request): # type: ignore
    """Fixture paramétrica para diferentes entornos."""
    return request.param # type: ignore


@pytest.fixture(params=[
    {'max_workers': 1},  # Mínimo
    {'max_workers': 32},  # Máximo
    {'max_workers': 16},  # Intermedio
])
def system_config_param(request): # type: ignore
    """Fixture paramétrica para configuraciones de sistema."""
    return request.param # type: ignore