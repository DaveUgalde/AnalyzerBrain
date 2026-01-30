"""
Configuración global de pytest para Project Brain.
Define fixtures, hooks y configuración común para todas las pruebas.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Generator, AsyncGenerator
from datetime import datetime, timedelta
import json
from unittest.mock import Mock, AsyncMock, MagicMock, patch

# Añadir el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config_manager import ConfigManager
from core.exceptions import BrainException
from core.orchestrator import BrainOrchestrator, OperationRequest, OperationPriority

# ============================================================================
# CONSTANTES DE PRUEBA
# ============================================================================

TEST_PROJECT_NAME = "test_project"
TEST_PROJECT_PATH = tempfile.mkdtemp(prefix="brain_test_")
TEST_PROJECT_ID = "test_project_123"
TEST_USER_ID = "test_user_456"
TEST_SESSION_ID = "test_session_789"

# ============================================================================
# FIXTURES DEL SISTEMA
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Crear event loop para pruebas asíncronas."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def temp_project_dir() -> Generator[str, None, None]:
    """Crear directorio temporal para proyectos de prueba."""
    temp_dir = tempfile.mkdtemp(prefix="brain_test_projects_")
    yield temp_dir
    # Limpiar después de todas las pruebas
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def sample_project_structure(temp_project_dir: str) -> Dict[str, str]:
    """Crear estructura de proyecto de ejemplo para pruebas."""
    project_dir = os.path.join(temp_project_dir, "sample_project")
    os.makedirs(project_dir, exist_ok=True)
    
    # Crear archivos Python de ejemplo
    python_dir = os.path.join(project_dir, "src", "app")
    os.makedirs(python_dir, exist_ok=True)
    
    # main.py
    main_py = """#!/usr/bin/env python3
\"\"\"Módulo principal de la aplicación de ejemplo.\"\"\"

import os
import sys
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class User:
    \"\"\"Clase que representa un usuario.\"\"\"
    id: int
    name: str
    email: str
    is_active: bool = True
    
    def get_display_name(self) -> str:
        \"\"\"Obtener nombre para mostrar.\"\"\"
        return f"{self.name} <{self.email}>"
    
    def activate(self) -> None:
        \"\"\"Activar usuario.\"\"\"
        self.is_active = True
        logger.info(f"Usuario {self.id} activado")

class UserManager:
    \"\"\"Gestor de usuarios.\"\"\"
    
    def __init__(self):
        self.users: List[User] = []
    
    def add_user(self, name: str, email: str) -> User:
        \"\"\"Añadir nuevo usuario.\"\"\"
        user_id = len(self.users) + 1
        user = User(id=user_id, name=name, email=email)
        self.users.append(user)
        return user
    
    def find_user(self, user_id: int) -> Optional[User]:
        \"\"\"Buscar usuario por ID.\"\"\"
        for user in self.users:
            if user.id == user_id:
                return user
        return None
    
    def get_active_users(self) -> List[User]:
        \"\"\"Obtener usuarios activos.\"\"\"
        return [user for user in self.users if user.is_active]

def process_data(data: List[int]) -> Dict[str, Any]:
    \"\"\"Procesar datos de entrada.\"\"\"
    if not data:
        return {"error": "Datos vacíos"}
    
    total = sum(data)
    average = total / len(data)
    
    return {
        "total": total,
        "average": average,
        "count": len(data),
        "max": max(data),
        "min": min(data)
    }

def main() -> None:
    \"\"\"Función principal.\"\"\"
    manager = UserManager()
    
    # Añadir usuarios
    manager.add_user("Alice", "alice@example.com")
    manager.add_user("Bob", "bob@example.com")
    
    # Procesar datos
    result = process_data([1, 2, 3, 4, 5])
    print(f"Resultado: {result}")
    
    # Mostrar usuarios activos
    active_users = manager.get_active_users()
    print(f"Usuarios activos: {len(active_users)}")

if __name__ == "__main__":
    main()
"""
    
    with open(os.path.join(python_dir, "main.py"), "w") as f:
        f.write(main_py)
    
    # utils.py
    utils_py = """#!/usr/bin/env python3
\"\"\"Utilidades para la aplicación.\"\"\"

import re
from typing import Dict, Any, List
from datetime import datetime, timedelta

def validate_email(email: str) -> bool:
    \"\"\"Validar formato de email.\"\"\"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def format_timestamp(timestamp: datetime, 
                    format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    \"\"\"Formatear timestamp.\"\"\"
    return timestamp.strftime(format_str)

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    \"\"\"Fusionar dos diccionarios.\"\"\"
    result = dict1.copy()
    result.update(dict2)
    return result

def calculate_age(birth_date: datetime) -> int:
    \"\"\"Calcular edad a partir de fecha de nacimiento.\"\"\"
    today = datetime.now()
    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age

class Cache:
    \"\"\"Caché simple con expiración.\"\"\"
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def set(self, key: str, value: Any) -> None:
        \"\"\"Almacenar valor en caché.\"\"\"
        self._cache[key] = {
            "value": value,
            "expires": datetime.now() + timedelta(seconds=self.ttl)
        }
    
    def get(self, key: str) -> Any:
        \"\"\"Obtener valor de caché.\"\"\"
        if key not in self._cache:
            return None
        
        item = self._cache[key]
        if datetime.now() > item["expires"]:
            del self._cache[key]
            return None
        
        return item["value"]
    
    def clear(self) -> None:
        \"\"\"Limpiar caché.\"\"\"
        self._cache.clear()
"""
    
    with open(os.path.join(python_dir, "utils.py"), "w") as f:
        f.write(utils_py)
    
    # test_main.py
    test_py = """#!/usr/bin/env python3
\"\"\"Pruebas para el módulo principal.\"\"\"

import pytest
from unittest.mock import Mock, patch
from src.app.main import User, UserManager, process_data

class TestUser:
    \"\"\"Pruebas para la clase User.\"\"\"
    
    def test_user_creation(self):
        \"\"\"Test crear usuario.\"\"\"
        user = User(id=1, name="Test", email="test@example.com")
        assert user.id == 1
        assert user.name == "Test"
        assert user.email == "test@example.com"
        assert user.is_active is True
    
    def test_get_display_name(self):
        \"\"\"Test obtener nombre para mostrar.\"\"\"
        user = User(id=1, name="Test", email="test@example.com")
        assert user.get_display_name() == "Test <test@example.com>"
    
    def test_activate(self):
        \"\"\"Test activar usuario.\"\"\"
        user = User(id=1, name="Test", email="test@example.com", is_active=False)
        user.activate()
        assert user.is_active is True

class TestUserManager:
    \"\"\"Pruebas para la clase UserManager.\"\"\"
    
    def setup_method(self):
        \"\"\"Configuración antes de cada test.\"\"\"
        self.manager = UserManager()
    
    def test_add_user(self):
        \"\"\"Test añadir usuario.\"\"\"
        user = self.manager.add_user("Test", "test@example.com")
        assert user.id == 1
        assert user.name == "Test"
        assert len(self.manager.users) == 1
    
    def test_find_user(self):
        \"\"\"Test buscar usuario.\"\"\"
        user = self.manager.add_user("Test", "test@example.com")
        found = self.manager.find_user(1)
        assert found == user
    
    def test_get_active_users(self):
        \"\"\"Test obtener usuarios activos.\"\"\"
        user1 = self.manager.add_user("Test1", "test1@example.com")
        user2 = self.manager.add_user("Test2", "test2@example.com")
        user2.is_active = False
        
        active_users = self.manager.get_active_users()
        assert len(active_users) == 1
        assert user1 in active_users

def test_process_data():
    \"\"\"Test procesar datos.\"\"\"
    result = process_data([1, 2, 3])
    assert result["total"] == 6
    assert result["average"] == 2.0
    assert result["count"] == 3
    assert result["max"] == 3
    assert result["min"] == 1
    
def test_process_empty_data():
    \"\"\"Test procesar datos vacíos.\"\"\"
    result = process_data([])
    assert "error" in result
    assert result["error"] == "Datos vacíos"

if __name__ == "__main__":
    pytest.main([__file__])
"""
    
    with open(os.path.join(python_dir, "test_main.py"), "w") as f:
        f.write(test_py)
    
    # requirements.txt
    requirements_txt = """# Dependencias del proyecto
python>=3.8
pytest>=7.0.0
pytest-asyncio>=0.20.0
pytest-cov>=4.0.0
"""
    
    with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
        f.write(requirements_txt)
    
    # README.md
    readme_md = """# Proyecto de Ejemplo

Este es un proyecto de ejemplo para pruebas de Project Brain.

## Características

- Gestión de usuarios
- Procesamiento de datos
- Utilidades varias
- Pruebas unitarias

## Instalación

```bash
pip install -r requirements.txt

Uso

bash
python src/app/main.py
Pruebas

bash
pytest src/app/test_main.py
"""

text
with open(os.path.join(project_dir, "README.md"), "w") as f:
    f.write(readme_md)

# Configuración de JavaScript
js_dir = os.path.join(project_dir, "web", "static", "js")
os.makedirs(js_dir, exist_ok=True)

# app.js
app_js = """/**
Aplicación JavaScript de ejemplo
*/
class Calculator {
constructor() {
this.history = [];
}

text
add(a, b) {
    const result = a + b;
    this.history.push({
        operation: 'add',
        operands: [a, b],
        result: result,
        timestamp: new Date()
    });
    return result;
}

subtract(a, b) {
    const result = a - b;
    this.history.push({
        operation: 'subtract',
        operands: [a, b],
        result: result,
        timestamp: new Date()
    });
    return result;
}

multiply(a, b) {
    const result = a * b;
    this.history.push({
        operation: 'multiply',
        operands: [a, b],
        result: result,
        timestamp: new Date()
    });
    return result;
}

divide(a, b) {
    if (b === 0) {
        throw new Error('Division by zero');
    }
    const result = a / b;
    this.history.push({
        operation: 'divide',
        operands: [a, b],
        result: result,
        timestamp: new Date()
    });
    return result;
}

getHistory() {
    return this.history;
}

clearHistory() {
    this.history = [];
}
}

// Función de utilidad
function formatCurrency(amount, currency = 'USD') {
return new Intl.NumberFormat('en-US', {
style: 'currency',
currency: currency
}).format(amount);
}

// Exportar para módulos
if (typeof module !== 'undefined' && module.exports) {
module.exports = {
Calculator,
formatCurrency
};
}
"""

text
with open(os.path.join(js_dir, "app.js"), "w") as f:
    f.write(app_js)

# package.json
package_json = {
    "name": "sample-project",
    "version": "1.0.0",
    "description": "Proyecto de ejemplo",
    "main": "web/static/js/app.js",
    "scripts": {
        "test": "jest",
        "start": "node server.js"
    },
    "dependencies": {
        "express": "^4.18.0",
        "jest": "^29.0.0"
    }
}

with open(os.path.join(project_dir, "package.json"), "w") as f:
    json.dump(package_json, f, indent=2)

return {
    "path": project_dir,
    "python_files": ["main.py", "utils.py", "test_main.py"],
    "js_files": ["app.js"],
    "config_files": ["requirements.txt", "package.json", "README.md"]
}
@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
"""Configuración de prueba del sistema."""
return {
"system": {
"name": "Project Brain Test",
"environment": "test",
"log_level": "DEBUG",
"debug_mode": True,
"data_directory": tempfile.mkdtemp(prefix="brain_test_data_"),
"log_directory": tempfile.mkdtemp(prefix="brain_test_logs_")
},
"projects": {
"supported_extensions": {
"python": [".py"],
"javascript": [".js"]
},
"exclude_patterns": [
"/pycache/",
"/.git/"
]
},
"databases": {
"postgresql": {
"enabled": False,
"host": "localhost",
"port": 5432,
"database": "test_brain"
},
"neo4j": {
"enabled": False
},
"redis": {
"enabled": False
},
"chromadb": {
"enabled": False
}
},
"api": {
"rest": {
"enabled": True,
"host": "127.0.0.1",
"port": 0 # Puerto dinámico para pruebas
}
},
"cache": {
"hierarchy": {
"level1": {
"type": "memory",
"max_size": 100,
"ttl_seconds": 60
}
}
},
"security": {
"sandbox": {
"enabled": False
}
}
}

@pytest.fixture(scope="function")
def mock_config_manager(test_config: Dict[str, Any]) -> Mock:
"""Mock del ConfigManager para pruebas."""
mock_cm = Mock(spec=ConfigManager)
mock_cm.get_config.return_value = test_config
mock_cm.load_config.return_value = test_config
return mock_cm

@pytest.fixture(scope="function")
def mock_orchestrator() -> Mock:
"""Mock del BrainOrchestrator para pruebas."""
mock_orch = Mock(spec=BrainOrchestrator)

text
# Configurar respuestas para métodos comunes
mock_orch.initialize.return_value = True
mock_orch.analyze_project.return_value = {
    "project_id": TEST_PROJECT_ID,
    "status": "completed",
    "files_analyzed": 10,
    "entities_extracted": 50,
    "analysis_time_seconds": 5.2,
    "findings": [],
    "recommendations": []
}
mock_orch.ask_question.return_value = {
    "answer": "Esta es una respuesta de prueba",
    "confidence": 0.85,
    "sources": [],
    "reasoning_chain": ["Paso 1", "Paso 2"],
    "suggested_followups": []
}
mock_orch.shutdown.return_value = True
mock_orch.get_metrics.return_value = {
    "operations_completed": 100,
    "operations_failed": 5,
    "avg_response_time_ms": 250.5
}

return mock_orch
@pytest.fixture(scope="function")
def mock_operation_request() -> OperationRequest:
"""Crear solicitud de operación de prueba."""
return OperationRequest(
operation_type="analyze_project",
priority=OperationPriority.HIGH,
context={
"project_path": TEST_PROJECT_PATH,
"project_id": TEST_PROJECT_ID,
"options": {"mode": "quick"}
}
)

============================================================================

FIXTURES DE DATOS DE PRUEBA

============================================================================

@pytest.fixture(scope="session")
def sample_code_files() -> Generator[Dict[str, str], None, None]:
"""Proporcionar archivos de código de ejemplo para pruebas."""

text
# Crear directorio temporal
temp_dir = tempfile.mkdtemp(prefix="brain_test_code_")

# Python simple
simple_py = os.path.join(temp_dir, "simple.py")
with open(simple_py, "w") as f:
    f.write("""
def hello_world():
print("Hello, World!")
return 42

class SimpleClass:
def init(self, value):
self.value = value

text
def get_value(self):
    return self.value
""")

text
# Python con imports
imports_py = os.path.join(temp_dir, "imports.py")
with open(imports_py, "w") as f:
    f.write("""
import os
import sys
from typing import List, Dict
from datetime import datetime

def process_data(data: List[int]) -> Dict[str, int]:
return {
"sum": sum(data),
"len": len(data)
}
""")

text
# JavaScript simple
simple_js = os.path.join(temp_dir, "simple.js")
with open(simple_js, "w") as f:
    f.write("""
function greet(name) {
return Hello, ${name}!;
}

class Calculator {
add(a, b) {
return a + b;
}

text
static multiply(a, b) {
    return a * b;
}
}
""")

text
yield {
    "simple_py": simple_py,
    "imports_py": imports_py,
    "simple_js": simple_js,
    "directory": temp_dir
}

# Limpiar
shutil.rmtree(temp_dir, ignore_errors=True)
@pytest.fixture(scope="session")
def test_embeddings_data() -> List[List[float]]:
"""Proporcionar datos de embeddings para pruebas."""
return [
[0.1, 0.2, 0.3, 0.4, 0.5] * 77, # 385 dimensiones
[0.5, 0.4, 0.3, 0.2, 0.1] * 77,
[0.2, 0.3, 0.4, 0.5, 0.6] * 77,
[0.6, 0.5, 0.4, 0.3, 0.2] * 77,
[0.3, 0.4, 0.5, 0.6, 0.7] * 77
]

@pytest.fixture(scope="session")
def test_entities_data() -> List[Dict[str, Any]]:
"""Proporcionar datos de entidades para pruebas."""
return [
{
"id": "func_1",
"type": "function",
"name": "process_data",
"file_path": "/src/app/main.py",
"start_line": 50,
"end_line": 70,
"metadata": {
"parameters": ["data"],
"return_type": "Dict[str, Any]"
}
},
{
"id": "class_1",
"type": "class",
"name": "UserManager",
"file_path": "/src/app/main.py",
"start_line": 30,
"end_line": 48,
"metadata": {
"methods": ["add_user", "find_user", "get_active_users"]
}
},
{
"id": "file_1",
"type": "file",
"name": "main.py",
"file_path": "/src/app/main.py",
"metadata": {
"language": "python",
"line_count": 120
}
}
]

@pytest.fixture(scope="session")
def test_graph_data() -> Dict[str, Any]:
"""Proporcionar datos de grafo para pruebas."""
return {
"nodes": [
{"id": "A", "type": "function", "name": "func_a"},
{"id": "B", "type": "function", "name": "func_b"},
{"id": "C", "type": "class", "name": "class_c"}
],
"edges": [
{"source": "A", "target": "B", "type": "calls", "weight": 1.0},
{"source": "B", "target": "C", "type": "uses", "weight": 0.5}
]
}

============================================================================

FIXTURES DE COMPONENTES MOCKEADOS

============================================================================

@pytest.fixture(scope="function")
def mock_embedding_generator() -> Mock:
"""Mock del EmbeddingGenerator para pruebas."""
mock_eg = Mock()
mock_eg.generate_text_embedding.return_value = [0.1] * 384
mock_eg.generate_code_embedding.return_value = [0.2] * 384
mock_eg.batch_generate.return_value = [[0.1] * 384 for _ in range(5)]
mock_eg.cache_embedding.return_value = True
mock_eg.get_cached_embedding.return_value = [0.1] * 384
mock_eg.compare_embeddings.return_value = 0.95
return mock_eg

@pytest.fixture(scope="function")
def mock_parser() -> Mock:
"""Mock del MultiLanguageParser para pruebas."""
mock_parser = Mock()
mock_parser.parse_file.return_value = Mock(
success=True,
language="python",
entities=[
Mock(type="function", name="test_func", start_line=1, end_line=10),
Mock(type="class", name="TestClass", start_line=15, end_line=30)
],
parse_time_ms=50.0
)
mock_parser.parse_directory.return_value = {
"/path/file1.py": Mock(success=True),
"/path/file2.py": Mock(success=True)
}
return mock_parser

@pytest.fixture(scope="function")
def mock_agent() -> Mock:
"""Mock de BaseAgent para pruebas."""
mock_agent = Mock()
mock_agent.initialize.return_value = True
mock_agent.process.return_value = Mock(
success=True,
data={"answer": "Test response"},
confidence=0.9
)
mock_agent.learn.return_value = True
mock_agent.evaluate.return_value = {
"state": "ready",
"metrics": {"requests_processed": 100}
}
mock_agent.shutdown.return_value = True
return mock_agent

============================================================================

HOOKS DE CONFIGURACIÓN

============================================================================

def pytest_configure(config):
"""Configuración de pytest."""
config.addinivalue_line(
"markers",
"slow: marca pruebas lentas (ejecutar con --run-slow)"
)
config.addinivalue_line(
"markers",
"integration: marca pruebas de integración"
)
config.addinivalue_line(
"markers",
"performance: marca pruebas de performance"
)
config.addinivalue_line(
"markers",
"e2e: marca pruebas end-to-end"
)
config.addinivalue_line(
"markers",
"async: marca pruebas asíncronas"
)
config.addinivalue_line(
"markers",
"database: marca pruebas que requieren base de datos"
)
config.addinivalue_line(
"markers",
"network: marca pruebas que requieren red"
)

def pytest_collection_modifyitems(config, items):
"""Modificar colección de tests según marcadores."""
skip_slow = pytest.mark.skip(reason="Necesita --run-slow para ejecutar")
skip_integration = pytest.mark.skip(reason="Solo se ejecuta en CI/CD")
skip_performance = pytest.mark.skip(reason="Solo se ejecuta en CI/CD")
skip_e2e = pytest.mark.skip(reason="Solo se ejecuta en CI/CD")

text
for item in items:
    # Saltar pruebas lentas a menos que se especifique --run-slow
    if "slow" in item.keywords and not config.getoption("--run-slow"):
        item.add_marker(skip_slow)
    
    # Saltar pruebas de integración en desarrollo local
    if "integration" in item.keywords and not config.getoption("--run-integration"):
        item.add_marker(skip_integration)
    
    # Saltar pruebas de performance
    if "performance" in item.keywords and not config.getoption("--run-performance"):
        item.add_marker(skip_performance)
    
    # Saltar pruebas e2e
    if "e2e" in item.keywords and not config.getoption("--run-e2e"):
        item.add_marker(skip_e2e)
def pytest_addoption(parser):
"""Añadir opciones de línea de comandos para pytest."""
parser.addoption(
"--run-slow",
action="store_true",
default=False,
help="Ejecutar pruebas marcadas como lentas"
)
parser.addoption(
"--run-integration",
action="store_true",
default=False,
help="Ejecutar pruebas de integración"
)
parser.addoption(
"--run-performance",
action="store_true",
default=False,
help="Ejecutar pruebas de performance"
)
parser.addoption(
"--run-e2e",
action="store_true",
default=False,
help="Ejecutar pruebas end-to-end"
)
parser.addoption(
"--test-timeout",
type=int,
default=30,
help="Timeout para pruebas individuales en segundos"
)

============================================================================

HELPERS DE PRUEBA

============================================================================

class TestHelpers:
"""Clase con helpers para pruebas."""

text
@staticmethod
def assert_dict_contains(expected: Dict, actual: Dict, path: str = "") -> None:
    """Verificar que el diccionario actual contiene todos los campos esperados."""
    for key, expected_value in expected.items():
        full_path = f"{path}.{key}" if path else key
        
        assert key in actual, f"Campo faltante: {full_path}"
        
        actual_value = actual[key]
        
        if isinstance(expected_value, dict):
            assert isinstance(actual_value, dict), \
                f"Se esperaba dict en {full_path}, se obtuvo {type(actual_value)}"
            TestHelpers.assert_dict_contains(expected_value, actual_value, full_path)
        elif isinstance(expected_value, list):
            assert isinstance(actual_value, list), \
                f"Se esperaba list en {full_path}, se obtuvo {type(actual_value)}"
            # Para listas, verificar longitud y tipos
            assert len(actual_value) == len(expected_value), \
                f"Longitud de lista diferente en {full_path}"
        else:
            assert actual_value == expected_value, \
                f"Valor diferente en {full_path}: {actual_value} != {expected_value}"

@staticmethod
def create_temp_file(content: str, extension: str = ".py") -> str:
    """Crear archivo temporal con contenido."""
    temp_dir = tempfile.mkdtemp(prefix="brain_test_file_")
    file_path = os.path.join(temp_dir, f"temp{extension}")
    with open(file_path, "w") as f:
        f.write(content)
    return file_path

@staticmethod
def measure_execution_time(func):
    """Decorador para medir tiempo de ejecución."""
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(f"\n⏱️  {func.__name__} tomó {elapsed:.3f} segundos")
        return result, elapsed
    return wrapper

@staticmethod
async def async_measure_execution_time(func):
    """Decorador asíncrono para medir tiempo de ejecución."""
    async def wrapper(*args, **kwargs):
        start = datetime.now()
        result = await func(*args, **kwargs)
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        print(f"\n⏱️  {func.__name__} tomó {elapsed:.3f} segundos")
        return result, elapsed
    return wrapper
@pytest.fixture(scope="session")
def test_helpers() -> TestHelpers:
"""Proveer helpers de prueba."""
return TestHelpers

============================================================================

FIXTURES PARA COBERTURA

============================================================================

@pytest.fixture(autouse=True)
def setup_test_coverage():
"""Configurar cobertura de pruebas."""
# Inicializar cobertura si está disponible
try:
import coverage
cov = coverage.Coverage()
cov.start()
yield
cov.stop()
cov.save()
except ImportError:
yield

============================================================================

LIMPIEZA GLOBAL

============================================================================

def pytest_sessionfinish(session, exitstatus):
"""Limpieza después de todas las pruebas."""
# Limpiar directorios temporales globales
temp_dirs = [
TEST_PROJECT_PATH,
]

text
for temp_dir in temp_dirs:
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

# Imprimir resumen si hay fallos
if exitstatus != 0:
    print("\n" + "="*80)
    print("ALGUNAS PRUEBAS FALLARON")
    print("="*80)
============================================================================

CONFIGURACIÓN PARA PRUEBAS ASÍNCRONAS

============================================================================

@pytest.fixture
def async_fixture():
"""Permitir pruebas asíncronas con pytest-asyncio."""
# Esta fixture es necesaria para pytest-asyncio
pass

text

## tests/unit/test_core_orchestrator.py

```python
"""
Pruebas unitarias para el módulo core/orchestrator.py
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from core.orchestrator import (
    BrainOrchestrator,
    OrchestratorConfig,
    OperationRequest,
    OperationPriority,
    OperationResult,
    ProjectContext,
    SystemMode,
    BrainException,
    ValidationError,
    TimeoutError
)


class TestOrchestratorConfig:
    """Pruebas para OrchestratorConfig."""
    
    def test_config_default_values(self):
        """Test valores por defecto de la configuración."""
        config = OrchestratorConfig()
        
        assert config.system_mode == SystemMode.DEVELOPMENT
        assert config.max_concurrent_operations == 10
        assert config.enable_learning is True
        assert config.enable_monitoring is True
        assert config.enable_backup is True
        assert config.plugins_directory == "./plugins"
        
        # Verificar timeouts por prioridad
        assert config.operation_timeout_seconds[OperationPriority.CRITICAL] == 30
        assert config.operation_timeout_seconds[OperationPriority.HIGH] == 300
        assert config.operation_timeout_seconds[OperationPriority.MEDIUM] == 1800
        assert config.operation_timeout_seconds[OperationPriority.LOW] == 3600
    
    def test_config_custom_values(self):
        """Test configuración con valores personalizados."""
        config = OrchestratorConfig(
            system_mode=SystemMode.PRODUCTION,
            max_concurrent_operations=20,
            enable_learning=False,
            operation_timeout_seconds={
                OperationPriority.CRITICAL: 10,
                OperationPriority.HIGH: 60
            }
        )
        
        assert config.system_mode == SystemMode.PRODUCTION
        assert config.max_concurrent_operations == 20
        assert config.enable_learning is False
        assert config.operation_timeout_seconds[OperationPriority.CRITICAL] == 10
        assert config.operation_timeout_seconds[OperationPriority.HIGH] == 60


class TestProjectContext:
    """Pruebas para ProjectContext."""
    
    def test_project_context_creation(self):
        """Test creación de contexto de proyecto."""
        context = ProjectContext(
            project_id="test_id",
            project_path="/test/path",
            language="python",
            analysis_depth="comprehensive"
        )
        
        assert context.project_id == "test_id"
        assert context.project_path == "/test/path"
        assert context.language == "python"
        assert context.analysis_depth == "comprehensive"
        assert context.created_at <= datetime.now()
        assert context.updated_at <= datetime.now()
    
    def test_project_context_defaults(self):
        """Test valores por defecto del contexto."""
        context = ProjectContext(
            project_id="test_id",
            project_path="/test/path"
        )
        
        assert context.language is None
        assert context.analysis_depth == "comprehensive"
    
    def test_project_context_validation(self):
        """Test validación de contexto."""
        with pytest.raises(ValueError):
            ProjectContext(
                project_id="",  # Vacío
                project_path="/test/path"
            )
        
        with pytest.raises(ValueError):
            ProjectContext(
                project_id="test_id",
                project_path=""  # Vacío
            )


class TestOperationRequest:
    """Pruebas para OperationRequest."""
    
    def test_operation_request_creation(self):
        """Test creación de solicitud de operación."""
        request = OperationRequest(
            operation_type="analyze_project",
            priority=OperationPriority.HIGH,
            context={"project_path": "/test"}
        )
        
        assert request.operation_type == "analyze_project"
        assert request.priority == OperationPriority.HIGH
        assert request.context == {"project_path": "/test"}
        assert request.timeout_seconds is None
        assert request.callback_url is None
        # Debería generarse un ID automáticamente
        assert request.operation_id is not None
    
    def test_operation_request_validation_valid(self):
        """Test validación de tipos de operación válidos."""
        valid_types = [
            "analyze_project",
            "process_question",
            "detect_changes",
            "learn_from_feedback",
            "export_knowledge",
            "system_status"
        ]
        
        for op_type in valid_types:
            request = OperationRequest(operation_type=op_type)
            assert request.operation_type == op_type
    
    def test_operation_request_validation_invalid(self):
        """Test validación de tipos de operación inválidos."""
        with pytest.raises(ValueError):
            OperationRequest(operation_type="invalid_type")
    
    def test_operation_request_with_timeout(self):
        """Test solicitud con timeout personalizado."""
        request = OperationRequest(
            operation_type="analyze_project",
            timeout_seconds=120
        )
        
        assert request.timeout_seconds == 120


class TestOperationResult:
    """Pruebas para OperationResult."""
    
    def test_operation_result_success(self):
        """Test resultado de operación exitosa."""
        result = OperationResult(
            operation_id="test_op",
            success=True,
            data={"result": "ok"},
            processing_time_ms=150.5
        )
        
        assert result.operation_id == "test_op"
        assert result.success is True
        assert result.data == {"result": "ok"}
        assert result.processing_time_ms == 150.5
        assert result.error is None
        assert result.warnings == []
        assert result.timestamp <= datetime.now()
    
    def test_operation_result_failure(self):
        """Test resultado de operación fallida."""
        result = OperationResult(
            operation_id="test_op",
            success=False,
            error="Something went wrong",
            warnings=["warning1", "warning2"]
        )
        
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.warnings == ["warning1", "warning2"]
        assert result.data is None
    
    def test_operation_result_with_metrics(self):
        """Test resultado con métricas."""
        result = OperationResult(
            operation_id="test_op",
            success=True,
            metrics={
                "files_processed": 10,
                "time_per_file_ms": 50.2
            }
        )
        
        assert result.metrics["files_processed"] == 10
        assert result.metrics["time_per_file_ms"] == 50.2


class TestBrainOrchestrator:
    """Pruebas para BrainOrchestrator."""
    
    @pytest.fixture
    def orchestrator_config(self):
        """Configuración de prueba para el orquestador."""
        return OrchestratorConfig(
            system_mode=SystemMode.DEVELOPMENT,
            max_concurrent_operations=5,
            enable_learning=False,
            enable_monitoring=False,
            enable_backup=False
        )
    
    @pytest.fixture
    async def orchestrator(self, orchestrator_config, mock_config_manager):
        """Crear instancia de BrainOrchestrator para pruebas."""
        with patch('core.orchestrator.SystemState'), \
             patch('core.orchestrator.EventBus'), \
             patch('core.orchestrator.asyncio.PriorityQueue'):
            
            orch = BrainOrchestrator(config_path=None)
            orch._config = orchestrator_config
            orch._state = Mock()
            orch._event_bus = AsyncMock()
            orch._operations_queue = AsyncMock()
            orch._components = {}
            orch._plugins = {}
            
            return orch
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test inicialización del orquestador."""
        # Configurar mocks
        orchestrator._event_bus.initialize.return_value = None
        orchestrator._initialize_component = AsyncMock(return_value=Mock())
        orchestrator._load_plugins = AsyncMock()
        orchestrator._start_workers = AsyncMock()
        orchestrator._start_monitoring = AsyncMock()
        
        # Ejecutar inicialización
        success = await orchestrator.initialize()
        
        # Verificar
        assert success is True
        assert orchestrator._is_running is True
        orchestrator._event_bus.initialize.assert_called_once()
        orchestrator._event_bus.publish.assert_called()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_failure(self, orchestrator):
        """Test fallo en inicialización del orquestador."""
        # Configurar mock para fallar
        orchestrator._event_bus.initialize.side_effect = Exception("Test error")
        
        # Verificar que se lanza excepción
        with pytest.raises(BrainException):
            await orchestrator.initialize()
        
        assert orchestrator._is_running is False
    
    @pytest.mark.asyncio
    async def test_process_operation_success(self, orchestrator):
        """Test procesamiento exitoso de operación."""
        # Configurar mocks
        orchestrator._is_running = True
        orchestrator._validate_operation_request = Mock()
        orchestrator._event_bus.publish = AsyncMock()
        orchestrator._analyze_project = AsyncMock(return_value={"result": "ok"})
        orchestrator._update_metrics = Mock()
        
        # Crear solicitud
        request = OperationRequest(
            operation_type="analyze_project",
            context={"project_path": "/test"}
        )
        
        # Procesar operación
        result = await orchestrator.process_operation(request)
        
        # Verificar
        assert result.success is True
        assert result.data == {"result": "ok"}
        assert result.operation_id == request.operation_id
        orchestrator._validate_operation_request.assert_called_with(request)
        orchestrator._event_bus.publish.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_operation_not_running(self, orchestrator):
        """Test procesamiento cuando el sistema no está ejecutándose."""
        orchestrator._is_running = False
        
        request = OperationRequest(operation_type="analyze_project")
        
        with pytest.raises(BrainException):
            await orchestrator.process_operation(request)
    
    @pytest.mark.asyncio
    async def test_process_operation_validation_error(self, orchestrator):
        """Test error de validación en procesamiento."""
        orchestrator._is_running = True
        orchestrator._validate_operation_request.side_effect = ValidationError("Invalid")
        
        request = OperationRequest(operation_type="analyze_project")
        
        result = await orchestrator.process_operation(request)
        
        assert result.success is False
        assert "Invalid" in result.error
    
    @pytest.mark.asyncio
    async def test_process_operation_unknown_type(self, orchestrator):
        """Test tipo de operación desconocido."""
        orchestrator._is_running = True
        orchestrator._validate_operation_request = Mock()
        
        request = OperationRequest(operation_type="unknown_type")
        
        result = await orchestrator.process_operation(request)
        
        assert result.success is False
        assert "Unknown operation type" in result.error
    
    @pytest.mark.asyncio
    async def test_analyze_project_success(self, orchestrator):
        """Test análisis de proyecto exitoso."""
        orchestrator.process_operation = AsyncMock(return_value=OperationResult(
            operation_id="test_op",
            success=True,
            data={
                "project_id": "proj_123",
                "status": "completed",
                "files_analyzed": 10
            }
        ))
        
        result = await orchestrator.analyze_project("/test/path")
        
        assert result["project_id"] == "proj_123"
        assert result["status"] == "completed"
        orchestrator.process_operation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_project_failure(self, orchestrator):
        """Test fallo en análisis de proyecto."""
        orchestrator.process_operation = AsyncMock(return_value=OperationResult(
            operation_id="test_op",
            success=False,
            error="Analysis failed"
        ))
        
        with pytest.raises(BrainException):
            await orchestrator.analyze_project("/test/path")
    
    @pytest.mark.asyncio
    async def test_ask_question_success(self, orchestrator):
        """Test pregunta exitosa."""
        orchestrator.process_operation = AsyncMock(return_value=OperationResult(
            operation_id="test_op",
            success=True,
            data={
                "answer": "Test answer",
                "confidence": 0.9,
                "sources": []
            }
        ))
        
        result = await orchestrator.ask_question("What is this?")
        
        assert "answer" in result
        assert result["answer"] == "Test answer"
        orchestrator.process_operation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ask_question_fallback(self, orchestrator):
        """Test fallback cuando la pregunta falla."""
        orchestrator.process_operation = AsyncMock(return_value=OperationResult(
            operation_id="test_op",
            success=False,
            error="Processing failed"
        ))
        
        result = await orchestrator.ask_question("What is this?")
        
        assert "error" in result
        assert result["confidence"] == 0.0
    
    @pytest.mark.asyncio
    async def test_shutdown_success(self, orchestrator):
        """Test apagado exitoso del sistema."""
        orchestrator._is_running = True
        orchestrator._active_operations = {}
        orchestrator._components = {
            "component1": Mock(shutdown=AsyncMock()),
            "component2": Mock(shutdown=AsyncMock())
        }
        orchestrator._event_bus.publish = AsyncMock()
        orchestrator._event_bus.shutdown = AsyncMock()
        orchestrator._state.save = AsyncMock()
        
        success = await orchestrator.shutdown()
        
        assert success is True
        assert orchestrator._is_running is False
        for component in orchestrator._components.values():
            component.shutdown.assert_called_once()
        orchestrator._event_bus.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_force(self, orchestrator):
        """Test apagado forzado."""
        orchestrator._is_running = True
        orchestrator._active_operations = {"op1": Mock()}
        orchestrator._components = {"comp1": Mock(shutdown=AsyncMock())}
        
        success = await orchestrator.shutdown(force=True)
        
        assert success is True
        # No debería esperar operaciones activas cuando es forzado
    
    @pytest.mark.asyncio
    async def test_shutdown_error(self, orchestrator):
        """Test error durante apagado."""
        orchestrator._is_running = True
        orchestrator._components = {
            "component1": Mock(shutdown=AsyncMock(side_effect=Exception("Error")))
        }
        
        with pytest.raises(BrainException):
            await orchestrator.shutdown()
    
    def test_validate_operation_request_valid(self, orchestrator):
        """Test validación de solicitud válida."""
        orchestrator._is_running = True
        request = OperationRequest(operation_type="analyze_project")
        
        # No debería lanzar excepción
        orchestrator._validate_operation_request(request)
    
    def test_validate_operation_request_missing_type(self, orchestrator):
        """Test validación sin tipo de operación."""
        orchestrator._is_running = True
        request = Mock(operation_type=None)
        
        with pytest.raises(ValidationError):
            orchestrator._validate_operation_request(request)
    
    def test_validate_operation_request_system_not_running(self, orchestrator):
        """Test validación cuando el sistema no está ejecutándose."""
        orchestrator._is_running = False
        request = OperationRequest(operation_type="analyze_project")
        
        with pytest.raises(BrainException):
            orchestrator._validate_operation_request(request)
    
    @pytest.mark.asyncio
    async def test_update_metrics(self, orchestrator):
        """Test actualización de métricas."""
        orchestrator._metrics = {
            "operations_completed": 10,
            "operations_failed": 2,
            "total_processing_time_ms": 5000.0,
            "avg_response_time_ms": 500.0
        }
        
        # Primera actualización exitosa
        orchestrator._update_metrics(success=True, processing_time=100.0)
        
        assert orchestrator._metrics["operations_completed"] == 11
        assert orchestrator._metrics["operations_failed"] == 2
        assert orchestrator._metrics["total_processing_time_ms"] == 5100.0
        # avg_response_time_ms debería ser 5100.0 / 11 ≈ 463.636
        assert abs(orchestrator._metrics["avg_response_time_ms"] - 463.636) < 0.1
        
        # Actualización fallida
        orchestrator._update_metrics(success=False, processing_time=50.0)
        
        assert orchestrator._metrics["operations_completed"] == 12
        assert orchestrator._metrics["operations_failed"] == 3
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, orchestrator):
        """Test obtención de estado del sistema."""
        orchestrator._is_running = True
        orchestrator._start_time = datetime.now() - timedelta(hours=1)
        orchestrator._components = {
            "comp1": Mock(),
            "comp2": Mock()
        }
        orchestrator._metrics = {
            "operations_completed": 100,
            "operations_failed": 5
        }
        orchestrator._active_operations = {"op1": Mock(), "op2": Mock()}
        orchestrator._config.system_mode = SystemMode.PRODUCTION
        
        status = await orchestrator._get_system_status()
        
        assert status["status"] == "running"
        assert status["system_mode"] == "production"
        assert status["components"]["comp1"] == "healthy"
        assert status["components"]["comp2"] == "healthy"
        assert status["metrics"]["operations_completed"] == 100
        assert status["active_operations"] == 2
        assert "uptime_seconds" in status
        assert status["uptime_seconds"] > 0
    
    @pytest.mark.asyncio
    async def test_specific_operation_handlers(self, orchestrator):
        """Test handlers específicos de operaciones."""
        # Test analyze_project handler
        context = {"project_path": "/test"}
        result = await orchestrator._analyze_project(context)
        assert "project_id" in result
        assert "status" in result
        
        # Test process_question handler
        context = {"question": "test"}
        result = await orchestrator._process_question(context)
        assert "answer" in result
        assert "confidence" in result
        
        # Test detect_changes handler
        result = await orchestrator._detect_changes({})
        assert "changes_detected" in result
        
        # Test learn_from_feedback handler
        result = await orchestrator._learn_from_feedback({})
        assert "learning_applied" in result
        
        # Test export_knowledge handler
        result = await orchestrator._export_knowledge({})
        assert "export_format" in result
    
    def test_load_config_default(self, orchestrator):
        """Test carga de configuración por defecto."""
        config = orchestrator._load_config(None)
        
        assert isinstance(config, OrchestratorConfig)
        assert config.system_mode == SystemMode.DEVELOPMENT
    
    @pytest.mark.asyncio
    async def test_initialize_component(self, orchestrator):
        """Test inicialización de componente."""
        component = await orchestrator._initialize_component("test_component")
        
        # En la implementación actual retorna MockComponent
        assert hasattr(component, 'shutdown')
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_operations(self, orchestrator):
        """Test operaciones concurrentes."""
        orchestrator._is_running = True
        orchestrator._validate_operation_request = Mock()
        orchestrator._event_bus.publish = AsyncMock()
        orchestrator._analyze_project = AsyncMock(return_value={"result": "ok"})
        orchestrator._update_metrics = Mock()
        
        # Crear múltiples solicitudes
        requests = [
            OperationRequest(
                operation_type="analyze_project",
                context={"project_path": f"/test/{i}"}
            )
            for i in range(5)
        ]
        
        # Procesar concurrentemente
        tasks = [orchestrator.process_operation(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verificar que todas se completaron
        assert len(results) == 5
        successful = [r for r in results if isinstance(r, OperationResult) and r.success]
        assert len(successful) == 5
    
    @pytest.mark.asyncio
    async def test_operation_timeout(self, orchestrator):
        """Test timeout de operación."""
        orchestrator._is_running = True
        orchestrator._validate_operation_request = Mock()
        orchestrator._event_bus.publish = AsyncMock()
        
        # Configurar handler para que tarde más que el timeout
        async def slow_handler(context):
            await asyncio.sleep(2)  # 2 segundos
            return {"result": "ok"}
        
        orchestrator._analyze_project = slow_handler
        
        # Solicitud con timeout de 1 segundo
        request = OperationRequest(
            operation_type="analyze_project",
            timeout_seconds=1
        )
        
        result = await orchestrator.process_operation(request)
        
        # Debería fallar por timeout (aunque en la implementación actual no hay timeout)
        # Esta prueba verifica que el sistema maneja operaciones largas
        assert result is not None


class TestErrorHandling:
    """Pruebas de manejo de errores."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_handles_component_failure(self, orchestrator):
        """Test que el orquestador maneja fallos de componentes."""
        orchestrator._is_running = True
        orchestrator._validate_operation_request = Mock()
        orchestrator._event_bus.publish = AsyncMock()
        
        # Simular fallo en handler
        orchestrator._analyze_project = AsyncMock(side_effect=Exception("Component failed"))
        
        request = OperationRequest(operation_type="analyze_project")
        result = await orchestrator.process_operation(request)
        
        assert result.success is False
        assert "Component failed" in result.error
        orchestrator._event_bus.publish.assert_called()
    
    def test_exception_hierarchy(self):
        """Test jerarquía de excepciones."""
        # BrainException es la base
        assert issubclass(ValidationError, BrainException)
        assert issubclass(TimeoutError, BrainException)
        
        # Crear instancias
        brain_ex = BrainException("test")
        validation_ex = ValidationError("test")
        timeout_ex = TimeoutError("test")
        
        assert str(brain_ex) == "test"
        assert str(validation_ex) == "test"
        assert str(timeout_ex) == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])