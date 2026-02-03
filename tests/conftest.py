"""
Configuración global de pytest para ANALYZERBRAIN.

Este archivo define fixtures y configuraciones que se pueden
utilizar en todos los tests del proyecto.

Autor: ANALYZERBRAIN Team
Fecha: 2024
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

import pytest
from loguru import logger

# Agregar el directorio src al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config_manager import ConfigManager
from core.exceptions import AnalyzerBrainError


def pytest_configure(config):
    """Configuración inicial de pytest."""
    # Desactivar logging en los tests para mayor claridad
    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def pytest_sessionstart(session):
    """Se ejecuta al inicio de la sesión de pruebas."""
    print("\n" + "="*60)
    print("🚀 Iniciando pruebas de ANALYZERBRAIN")
    print("="*60)


def pytest_sessionfinish(session, exitstatus):
    """Se ejecuta al final de la sesión de pruebas."""
    print("\n" + "="*60)
    print(f"📊 Estado de pruebas: {exitstatus}")
    print("="*60)


@pytest.fixture(scope="session")
def event_loop():
    """Fixture para el event loop asyncio (sesión)."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def root_dir() -> Path:
    """Devuelve el directorio raíz del proyecto."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(root_dir: Path) -> Path:
    """Devuelve el directorio de datos de prueba."""
    test_data = root_dir / "tests" / "fixtures"
    test_data.mkdir(exist_ok=True)
    return test_data


@pytest.fixture(scope="session")
def sample_project_dir(test_data_dir: Path) -> Path:
    """Devuelve el directorio de proyecto de prueba."""
    sample_project = test_data_dir / "sample_project"
    sample_project.mkdir(exist_ok=True)
    return sample_project


@pytest.fixture(scope="session")
def config_manager() -> ConfigManager:
    """
    Fixture que devuelve una instancia de ConfigManager
    configurada para pruebas.
    """
    # Forzar una nueva instancia (limpiar singleton)
    ConfigManager._instance = None
    
    # Crear una configuración de prueba
    config = ConfigManager()
    
    # Sobrescribir configuraciones para pruebas
    config.set("environment", "testing")
    config.set("log_level", "WARNING")
    config.set("data_dir", Path("./data_test"))
    
    return config


@pytest.fixture(scope="function")
def clean_config_manager(config_manager: ConfigManager) -> ConfigManager:
    """
    Fixture que devuelve un ConfigManager limpio por cada test.
    """
    # Limpiar la configuración existente
    config_manager._config = {}
    config_manager._load_config()
    
    # Establecer entorno de prueba
    config_manager.set("environment", "testing")
    
    return config_manager


@pytest.fixture(scope="function")
def mock_config() -> Dict[str, Any]:
    """Devuelve un diccionario de configuración de prueba."""
    return {
        "system": {
            "name": "ANALYZERBRAIN-TEST",
            "version": "0.1.0",
            "max_workers": 2,
            "timeout_seconds": 60
        },
        "logging": {
            "level": "INFO",
            "format": "test",
            "rotation": "1 day"
        },
        "storage": {
            "data_dir": "./data_test",
            "cache_dir": "./data_test/cache",
            "max_cache_size_mb": 100
        }
    }


@pytest.fixture(scope="function")
def sample_code_file(sample_project_dir: Path) -> Path:
    """
    Crea un archivo de código de ejemplo para pruebas y devuelve su ruta.
    """
    code_file = sample_project_dir / "example.py"
    
    code_content = '''
"""
Este es un archivo de ejemplo para pruebas.
"""

def hello_world():
    """Función que imprime un salude."""
    print("Hello, World!")
    return True

class Calculator:
    """Clase de ejemplo para una calculadora simple."""
    
    def add(self, a: int, b: int) -> int:
        """Suma dos números."""
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiplica dos números."""
        return a * b

if __name__ == "__main__":
    hello_world()
    calc = Calculator()
    print(calc.add(2, 3))
'''
    
    code_file.write_text(code_content, encoding="utf-8")
    return code_file


@pytest.fixture(scope="function")
def sample_text_file(sample_project_dir: Path) -> Path:
    """Crea un archivo de texto de ejemplo para pruebas."""
    text_file = sample_project_dir / "example.txt"
    text_file.write_text("Este es un archivo de texto de ejemplo.\nLínea 2.\nLínea 3.", encoding="utf-8")
    return text_file


@pytest.fixture(scope="function")
def sample_json_file(sample_project_dir: Path) -> Path:
    """Crea un archivo JSON de ejemplo para pruebas."""
    import json
    
    json_file = sample_project_dir / "example.json"
    data = {
        "name": "Test Project",
        "version": "1.0.0",
        "dependencies": ["pytest", "loguru"],
        "config": {
            "debug": True,
            "max_items": 100
        }
    }
    
    json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return json_file


@pytest.fixture(scope="function")
def sample_yaml_file(sample_project_dir: Path) -> Path:
    """Crea un archivo YAML de ejemplo para pruebas."""
    import yaml
    
    yaml_file = sample_project_dir / "example.yaml"
    data = {
        "project": {
            "name": "Test YAML",
            "settings": {
                "debug": True,
                "level": "info"
            }
        },
        "services": ["api", "database", "cache"]
    }
    
    yaml_file.write_text(yaml.dump(data), encoding="utf-8")
    return yaml_file


@pytest.fixture(scope="function")
def temp_dir(tmp_path: Path) -> Path:
    """Devuelve un directorio temporal para pruebas."""
    return tmp_path


@pytest.fixture(scope="function")
def clean_test_dirs(config_manager: ConfigManager):
    """
    Fixture que limpia los directorios de prueba después de cada test.
    """
    yield
    
    # Limpiar directorios de prueba
    import shutil
    
    test_dirs = [
        config_manager.get("data_dir"),
        Path("./logs"),
        Path("./data_test")
    ]
    
    for dir_path in test_dirs:
        if dir_path and dir_path.exists():
            shutil.rmtree(dir_path, ignore_errors=True)


# Fixture para excepciones
@pytest.fixture
def analyzer_brain_error():
    """Fixture que devuelve la clase base de excepciones."""
    return AnalyzerBrainError


# Fixture para crear un proyecto de prueba completo
@pytest.fixture(scope="function")
def test_project_structure(temp_dir: Path) -> Dict[str, Path]:
    """Crea una estructura completa de proyecto de prueba."""
    project_structure = {
        "root": temp_dir / "test_project",
        "src": temp_dir / "test_project" / "src",
        "tests": temp_dir / "test_project" / "tests",
        "config": temp_dir / "test_project" / "config",
        "data": temp_dir / "test_project" / "data"
    }
    
    for path in project_structure.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Crear archivos de ejemplo
    (project_structure["src"] / "main.py").write_text("print('Hello')")
    (project_structure["tests"] / "test_example.py").write_text("def test_example(): pass")
    (project_structure["config"] / "settings.yaml").write_text("debug: true")
    
    return project_structure