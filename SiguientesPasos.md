que archivos necesita para poder implementar:
1. system_state.py          # Estado global
2. project_scanner.py       # Escaneo de proyectos
3. multi_language_parser.py # Parser Python básico
4. entity_extractor.py      # Extracción de entidades
5. orchestrator.py          # Coordinación básica
Vamos a analizar las dependencias entre estos archivos y qué otros archivos necesitamos para implementarlos.
system_state.py:
Dependencias: config_manager, exceptions
Necesita: src/core/__init__.py para exportar la clase.
No requiere otros módulos del sistema.
project_scanner.py:
Dependencias: config_manager, exceptions, file_utils
Necesita: src/indexer/__init__.py y posiblemente un modelo de datos para representar la estructura del proyecto (por ejemplo, una clase ProjectStructure).
También puede necesitar src/utils/logging_config para logging.
multi_language_parser.py:
Dependencias: config_manager, exceptions, file_utils
Necesita: src/indexer/__init__.py y probablemente una clase base para los parsers o un modelo de entidades (como CodeEntity).
Dependencias externas: tree-sitter y tree-sitter-languages (ya en requirements/indexer.txt)
entity_extractor.py:
Dependencias: config_manager, exceptions, y posiblemente los modelos de entidades.
Necesita: src/indexer/__init__.py y los modelos de datos para las entidades (clases, funciones, etc.).
orchestrator.py:
Dependencias: config_manager, exceptions, system_state, project_scanner, multi_language_parser, entity_extractor, y también graph y agents (pero para la coordinación básica, podríamos posponer estas dependencias).
Para una versión básica, podríamos hacer que el orquestador coordine el escaneo, parsing y extracción, y luego genere un reporte simple sin necesidad de grafo o agentes.
Dado que estamos en una fase temprana, podemos construir una versión mínima del orquestador que use los módulos de indexación y produzca un reporte básico.

Además, necesitamos crear los archivos __init__.py en cada subpaquete para que los módulos sean importables.

Por lo tanto, los archivos que necesitamos crear (además de los 5 mencionados) son:

src/core/__init__.py
src/indexer/__init__.py
src/utils/__init__.py (ya debería existir, pero verificar)
src/__init__.py (ya existe, pero verificar)
También necesitamos definir algunos modelos de datos (dataclasses) para representar:

Estructura del proyecto (en project_scanner.py o en un archivo separado como models.py)
Entidades de código (en entity_extractor.py o en un archivo separado)
Para mantener la simplicidad, podemos definir estas clases dentro de los archivos que las necesitan, o crear un archivo src/indexer/models.py.

Dado que el proyecto es modular, sugiero crear un archivo src/indexer/models.py para los modelos relacionados con indexación.

Resumen de archivos a crear:
src/core/__init__.py (para exportar las clases del core)
src/indexer/__init__.py (para exportar las clases del indexer)
src/indexer/models.py (para las clases ProjectStructure, CodeEntity, etc.)
src/core/system_state.py
src/indexer/project_scanner.py
src/indexer/multi_language_parser.py
src/indexer/entity_extractor.py
src/core/orchestrator.py (versión básica)
Además, debemos actualizar src/main.py para usar el orquestador.

También, para el parser multi-lenguaje, necesitamos instalar las dependencias de indexer. Ya tenemos requirements/indexer.txt, pero debemos asegurarnos de que estén instaladas. Podemos actualizar el setup.sh para instalar también requirements/indexer.txt.

Plan de implementación:
Crear los archivos __init__.py y models.py.
Implementar system_state.py.
Implementar project_scanner.py.
Implementar multi_language_parser.py (solo para Python inicialmente).
Implementar entity_extractor.py.
Implementar orchestrator.py (que use los módulos 2,3,4).
Actualizar main.py para usar el orquestador.
Además, debemos escribir tests para cada nuevo módulo.

Vamos a crear una lista detallada de los archivos con una breve descripción de lo que deben contener.

Archivo: src/core/__init__.py

python
from .config_manager import ConfigManager, config
from .exceptions import AnalyzerBrainError, ConfigurationError, IndexerError, GraphError, AgentError, APIError, ValidationError
from .health_check import HealthCheck
from .system_state import SystemState

__all__ = [
    'ConfigManager',
    'config',
    'AnalyzerBrainError',
    'ConfigurationError',
    'IndexerError',
    'GraphError',
    'AgentError',
    'APIError',
    'ValidationError',
    'HealthCheck',
    'SystemState',
]
Archivo: src/indexer/__init__.py

python
from .project_scanner import ProjectScanner
from .multi_language_parser import MultiLanguageParser
from .entity_extractor import EntityExtractor

__all__ = [
    'ProjectScanner',
    'MultiLanguageParser',
    'EntityExtractor',
]
Archivo: src/indexer/models.py

python
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

@dataclass
class ProjectStructure:
    """Estructura de un proyecto."""
    root: Path
    files: List[Path]
    directories: List[Path]
    metadata: Dict[str, Any]

@dataclass
class CodeEntity:
    """Entidad de código extraída."""
    type: str  # 'class', 'function', 'variable', 'import', etc.
    name: str
    location: Path
    line_start: int
    line_end: int
    properties: Dict[str, Any]
    parent: Optional['CodeEntity'] = None
Archivo: src/core/system_state.py

python
"""
Módulo para gestionar el estado global del sistema.
"""
from datetime import datetime
from typing import Dict, Any, List
from loguru import logger

from .config_manager import config
from .exceptions import AnalyzerBrainError

class SystemState:
    """Mantiene el estado global del sistema."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.status = "initializing"
        self.metrics: Dict[str, Any] = {
            "projects_analyzed": 0,
            "errors": 0,
            "uptime": 0,
        }
        self.components: Dict[str, Any] = {}
    
    def register_component(self, name: str, component: Any) -> None:
        """Registra un componente del sistema."""
        self.components[name] = component
        logger.debug(f"Componente registrado: {name}")
    
    def update_metric(self, name: str, value: Any) -> None:
        """Actualiza una métrica del sistema."""
        self.metrics[name] = value
    
    def get_health_report(self) -> Dict[str, Any]:
        """Genera un reporte de salud del sistema."""
        self.metrics['uptime'] = (datetime.now() - self.start_time).total_seconds()
        return {
            "status": self.status,
            "metrics": self.metrics,
            "components": list(self.components.keys()),
            "timestamp": datetime.now().isoformat()
        }
Archivo: src/indexer/project_scanner.py

python
"""
Escáner de proyectos: Recorre directorios y recopila archivos para análisis.
"""
import os
from pathlib import Path
from typing import List, Set
from loguru import logger

from ..core.config_manager import config
from ..core.exceptions import IndexerError
from ..utils.file_utils import safe_read_file
from .models import ProjectStructure

class ProjectScanner:
    """Escanea proyectos para análisis."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager or config
        self.excluded_dirs: Set[str] = {
            '.git', '.venv', 'venv', 'node_modules',
            '__pycache__', '.pytest_cache', 'dist', 'build'
        }
        self.excluded_extensions: Set[str] = {
            '.pyc', '.pyo', '.pyd', '.so', '.dll',
            '.exe', '.bin', '.class', '.jar'
        }
    
    def scan(self, project_path: Path) -> ProjectStructure:
        """Escanea un proyecto y retorna su estructura."""
        if not project_path.exists():
            raise IndexerError(f"Proyecto no encontrado: {project_path}")
        
        files: List[Path] = []
        directories: List[Path] = []
        
        for root, dirs, file_names in os.walk(project_path):
            # Filtrar directorios excluidos
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for dir_name in dirs:
                directories.append(Path(root) / dir_name)
            
            for file_name in file_names:
                file_path = Path(root) / file_name
                if self._should_process(file_path):
                    files.append(file_path)
        
        return ProjectStructure(
            root=project_path,
            files=files,
            directories=directories,
            metadata={
                "total_files": len(files),
                "total_dirs": len(directories),
                "scanned_at": datetime.now().isoformat()
            }
        )
    
    def _should_process(self, file_path: Path) -> bool:
        """Determina si un archivo debe ser procesado."""
        # Verificar extensión
        if file_path.suffix in self.excluded_extensions:
            return False
        
        # Verificar tamaño máximo
        max_size_mb = self.config.get("indexer.max_file_size_mb", 10)
        max_size_bytes = max_size_mb * 1024 * 1024
        try:
            file_size = file_path.stat().st_size
            if file_size > max_size_bytes:
                logger.warning(f"Archivo demasiado grande, omitiendo: {file_path}")
                return False
        except OSError:
            return False
        
        return True
Archivo: src/indexer/multi_language_parser.py

python
"""
Parser para múltiples lenguajes de programación.
Inicialmente solo soporta Python usando el módulo ast.
"""
import ast
from pathlib import Path
from typing import List
from loguru import logger

from ..core.exceptions import IndexerError
from ..utils.file_utils import safe_read_file
from .models import CodeEntity

class MultiLanguageParser:
    """Parser para múltiples lenguajes (comenzando con Python)."""
    
    def __init__(self):
        self.supported_extensions = {'.py'}
    
    def can_parse(self, file_path: Path) -> bool:
        """Determina si el parser puede manejar un archivo."""
        return file_path.suffix in self.supported_extensions
    
    def parse(self, file_path: Path) -> List[CodeEntity]:
        """Parsea un archivo y extrae entidades de código."""
        if not self.can_parse(file_path):
            return []
        
        try:
            content = safe_read_file(file_path)
            if file_path.suffix == '.py':
                return self._parse_python(content, file_path)
            else:
                return []
        except Exception as e:
            logger.error(f"Error parseando {file_path}: {e}")
            raise IndexerError(f"Error parseando {file_path}: {e}")
    
    def _parse_python(self, content: str, file_path: Path) -> List[CodeEntity]:
        """Parsea código Python usando ast."""
        entities = []
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                entity = CodeEntity(
                    type='class',
                    name=node.name,
                    location=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    properties={
                        'bases': [ast.unparse(base) for base in node.bases],
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    }
                )
                entities.append(entity)
            elif isinstance(node, ast.FunctionDef):
                entity = CodeEntity(
                    type='function',
                    name=node.name,
                    location=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    properties={
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [ast.unparse(d) for d in node.decorator_list]
                    }
                )
                entities.append(entity)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # Para imports, podemos extraer los nombres
                for alias in node.names:
                    entity = CodeEntity(
                        type='import',
                        name=alias.name,
                        location=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                        properties={
                            'module': node.module if isinstance(node, ast.ImportFrom) else None,
                            'alias': alias.asname
                        }
                    )
                    entities.append(entity)
        
        return entities
Archivo: src/indexer/entity_extractor.py

python
"""
Extractor de entidades: Procesa las entidades parseadas y las enriquece.
"""
from typing import List
from .models import CodeEntity

class EntityExtractor:
    """Extrae y enriquece entidades de código."""
    
    def __init__(self):
        pass
    
    def extract(self, entities: List[CodeEntity]) -> List[CodeEntity]:
        """Procesa una lista de entidades y las enriquece con información adicional."""
        # Por ahora, simplemente retornamos las entidades sin cambios.
        # En el futuro, aquí se podrían añadir más análisis.
        return entities
    
    def _enrich_entity(self, entity: CodeEntity) -> CodeEntity:
        """Enriquece una entidad con información adicional."""
        # Por ejemplo, calcular complejidad ciclomática, etc.
        return entity
Archivo: src/core/orchestrator.py

python
"""
Orquestador principal: Coordina el análisis de proyectos.
"""
from pathlib import Path
from typing import Dict, Any
from loguru import logger

from .config_manager import config
from .system_state import SystemState
from ..indexer.project_scanner import ProjectScanner
from ..indexer.multi_language_parser import MultiLanguageParser
from ..indexer.entity_extractor import EntityExtractor

class BrainOrchestrator:
    """Orquestador principal del sistema."""
    
    def __init__(self, system_state: SystemState):
        self.system_state = system_state
        self.project_scanner = ProjectScanner()
        self.parser = MultiLanguageParser()
        self.entity_extractor = EntityExtractor()
        
        # Registrar componentes en el estado del sistema
        self.system_state.register_component('orchestrator', self)
        self.system_state.register_component('project_scanner', self.project_scanner)
        self.system_state.register_component('parser', self.parser)
        self.system_state.register_component('entity_extractor', self.entity_extractor)
        
        logger.info("Orquestador inicializado")
    
    def analyze_project(self, project_path: Path) -> Dict[str, Any]:
        """Analiza un proyecto completo."""
        logger.info(f"Iniciando análisis del proyecto: {project_path}")
        
        # 1. Escanear el proyecto
        structure = self.project_scanner.scan(project_path)
        logger.info(f"Proyecto escaneado: {len(structure.files)} archivos encontrados")
        
        # 2. Parsear cada archivo
        all_entities = []
        for file_path in structure.files:
            if self.parser.can_parse(file_path):
                try:
                    entities = self.parser.parse(file_path)
                    all_entities.extend(entities)
                    logger.debug(f"Archivo parseado: {file_path} -> {len(entities)} entidades")
                except Exception as e:
                    logger.error(f"Error parseando {file_path}: {e}")
                    self.system_state.update_metric('errors', self.system_state.metrics['errors'] + 1)
        
        # 3. Extraer y enriquecer entidades
        enriched_entities = self.entity_extractor.extract(all_entities)
        
        # 4. Generar reporte
        report = {
            'project': str(project_path),
            'structure': {
                'total_files': len(structure.files),
                'total_dirs': len(structure.directories),
            },
            'entities_found': len(enriched_entities),
            'entities_by_type': self._count_entities_by_type(enriched_entities),
            'metadata': structure.metadata
        }
        
        # Actualizar métricas
        self.system_state.update_metric('projects_analyzed', self.system_state.metrics['projects_analyzed'] + 1)
        
        logger.info(f"Análisis completado: {report['entities_found']} entidades encontradas")
        return report
    
    def _count_entities_by_type(self, entities: List[CodeEntity]) -> Dict[str, int]:
        """Cuenta las entidades por tipo."""
        counts = {}
        for entity in entities:
            counts[entity.type] = counts.get(entity.type, 0) + 1
        return counts
Archivo: src/main.py (actualizar)

python
"""
Punto de entrada principal del sistema ANALYZERBRAIN.
"""
import click
from pathlib import Path
from loguru import logger

from .core.config_manager import config
from .core.system_state import SystemState
from .core.orchestrator import BrainOrchestrator
from .utils.logging_config import setup_logging

@click.group()
def cli():
    """ANALYZERBRAIN - Sistema inteligente de análisis de código."""
    pass

@cli.command()
@click.argument('project_path', type=click.Path(exists=True))
def analyze(project_path):
    """Analiza un proyecto de código."""
    setup_logging(config)
    logger.info("Iniciando ANALYZERBRAIN")
    
    # Inicializar estado del sistema
    system_state = SystemState()
    system_state.status = "running"
    
    # Crear orquestador
    orchestrator = BrainOrchestrator(system_state)
    
    # Analizar proyecto
    report = orchestrator.analyze_project(Path(project_path))
    
    # Mostrar reporte
    click.echo(f"✅ Análisis completado para: {project_path}")
    click.echo(f"   Archivos analizados: {report['structure']['total_files']}")
    click.echo(f"   Entidades encontradas: {report['entities_found']}")
    for entity_type, count in report['entities_by_type'].items():
        click.echo(f"   - {entity_type}: {count}")
    
    # Mostrar estado del sistema
    health = system_state.get_health_report()
    click.echo(f"\n📊 Estado del sistema: {health['status']}")
    click.echo(f"   Proyectos analizados: {health['metrics']['projects_analyzed']}")
    click.echo(f"   Errores: {health['metrics']['errors']}")

@cli.command()
def health():
    """Muestra el estado de salud del sistema."""
    setup_logging(config)
    system_state = SystemState()
    system_state.status = "running"
    health_report = system_state.get_health_report()
    click.echo(health_report)

if __name__ == "__main__":
    cli()
Tests:
Debemos crear tests para cada nuevo módulo:

tests/unit/test_system_state.py
tests/unit/test_project_scanner.py
tests/unit/test_multi_language_parser.py
tests/unit/test_entity_extractor.py
tests/unit/test_orchestrator.py
Además, actualizar setup.sh para instalar las dependencias de indexer (tree-sitter, etc.) si no lo está haciendo ya.

Este plan te dará una base sólida para continuar con el desarrollo. Una vez que estos archivos estén implementados y probados, podrás pasar a la siguiente fase (grafo de conocimiento).
Para implementar los 5 archivos que mencionas, necesitas crear los siguientes archivos y dependencias:

📁 ESTRUCTURA NECESARIA PARA LA IMPLEMENTACIÓN

1. src/core/system_state.py

Archivos necesarios previos:

text
✅ src/core/config_manager.py  (ya existe)
✅ src/core/exceptions.py       (ya existe)
✅ src/core/__init__.py         (PENDIENTE - para importaciones)
Dependencias externas:

python
# Necesita en requirements/core.txt
loguru>=0.7.0,<0.8.0
pydantic>=2.0.0,<3.0.0  # para validación del estado
2. src/indexer/project_scanner.py

Archivos necesarios previos:

text
✅ src/core/config_manager.py    (ya existe)
✅ src/core/exceptions.py         (ya existe)
✅ src/utils/file_utils.py        (ya existe)
❌ src/indexer/__init__.py       (PENDIENTE - para estructura de paquete)
❌ src/indexer/models.py         (PENDIENTE - para tipos de datos)
Dependencias externas:

python
# En requirements/indexer.txt
python-magic>=0.4.27,<0.5.0  # para detección de tipos de archivo
chardet>=5.2.0,<6.0.0        # para detección de encoding
3. src/indexer/multi_language_parser.py

Archivos necesarios previos:

text
✅ src/core/exceptions.py        (ya existe)
❌ src/indexer/__init__.py      (PENDIENTE)
❌ src/indexer/models.py        (PENDIENTE)
❌ src/indexer/file_processor.py (OPCIONAL pero recomendado)
Dependencias externas (CRÍTICAS):

python
# En requirements/indexer.txt
tree-sitter>=0.20.1,<0.21.0        # parser principal
tree-sitter-languages>=1.5.0,<2.0.0 # gramáticas para múltiples lenguajes
pygments>=2.16.1,<3.0.0           # resaltado de sintaxis (opcional)
4. src/indexer/entity_extractor.py

Archivos necesarios previos:

text
❌ src/indexer/__init__.py       (PENDIENTE)
❌ src/indexer/models.py         (PENDIENTE)
✅ src/utils/validation.py       (ya existe)
Dependencias externas:

python
# En requirements/indexer.txt (las mismas que multi_language_parser)
# Además, puede necesitar:
radon>=6.0.0,<7.0.0  # para análisis de complejidad (opcional)
5. src/core/orchestrator.py

Archivos necesarios previos:

text
✅ src/core/config_manager.py    (ya existe)
✅ src/core/exceptions.py         (ya existe)
❌ src/core/system_state.py      (PENDIENTE - pero es #1 en tu lista)
❌ src/indexer/project_scanner.py (PENDIENTE - pero es #2 en tu lista)
❌ src/indexer/multi_language_parser.py (PENDIENTE - pero es #3)
❌ src/indexer/entity_extractor.py (PENDIENTE - pero es #4)
❌ src/core/__init__.py          (PENDIENTE)
📋 ARCHIVOS QUE NECESITAS CREAR (en orden)

FASE A: Estructura de Paquetes

src/core/__init__.py
python
"""
Paquete core de ANALYZERBRAIN.
"""

from .config_manager import ConfigManager, config
from .exceptions import *
from .health_check import HealthCheck
from .system_state import SystemState

__all__ = [
    'ConfigManager',
    'config',
    'HealthCheck',
    'SystemState',
    # ... y todas las excepciones
]
src/indexer/__init__.py
python
"""
Paquete indexer de ANALYZERBRAIN.
"""

from .project_scanner import ProjectScanner
from .multi_language_parser import MultiLanguageParser
from .entity_extractor import EntityExtractor

__all__ = [
    'ProjectScanner',
    'MultiLanguageParser',
    'EntityExtractor',
]
src/indexer/models.py (OPCIONAL pero RECOMENDADO)
python
"""
Modelos de datos para el módulo indexer.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

@dataclass
class ProjectStructure:
    """Estructura de un proyecto escaneado."""
    root: Path
    files: List[Path]
    directories: List[Path]
    metadata: Dict[str, Any]

@dataclass
class CodeEntity:
    """Entidad de código extraída."""
    type: str  # 'class', 'function', 'import', etc.
    name: str
    location: Path
    line_start: int
    line_end: int
    properties: Dict[str, Any]
FASE B: Archivos Principales (en tu lista)

src/core/system_state.py (Tu #1)
src/indexer/project_scanner.py (Tu #2)
src/indexer/multi_language_parser.py (Tu #3)
src/indexer/entity_extractor.py (Tu #4)
src/core/orchestrator.py (Tu #5)
FASE C: Archivos Complementarios (Recomendados)

tests/unit/test_system_state.py
tests/unit/test_project_scanner.py
tests/unit/test_multi_language_parser.py
tests/unit/test_entity_extractor.py
tests/unit/test_orchestrator.py
tests/fixtures/sample_project/ (Carpeta con código de prueba)
tests/fixtures/sample_code/ (Archivos individuales de prueba)
🛠️ PASOS DE IMPLEMENTACIÓN RECOMENDADOS

Paso 1: Crear estructura de paquetes

bash
# 1. Crear __init__.py files
touch src/core/__init__.py
touch src/indexer/__init__.py
touch src/indexer/models.py

# 2. Instalar dependencias del indexer
pip install -r requirements/indexer.txt
Paso 2: Implementar system_state.py

python
# src/core/system_state.py
"""
Gestión del estado global del sistema.
"""
from datetime import datetime
from typing import Dict, Any, List
from loguru import logger

from .config_manager import config
from .exceptions import AnalyzerBrainError

class SystemState:
    """Mantiene el estado global del sistema."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.status = "initializing"
        self.metrics: Dict[str, Any] = {
            "projects_analyzed": 0,
            "errors": 0,
            "uptime": 0,
        }
        self.components: Dict[str, Any] = {}
    
    def register_component(self, name: str, component: Any) -> None:
        """Registra un componente del sistema."""
        self.components[name] = component
        logger.debug(f"Componente registrado: {name}")
    
    def update_metric(self, name: str, value: Any) -> None:
        """Actualiza una métrica del sistema."""
        self.metrics[name] = value
    
    def get_health_report(self) -> Dict[str, Any]:
        """Genera un reporte de salud del sistema."""
        self.metrics['uptime'] = (datetime.now() - self.start_time).total_seconds()
        return {
            "status": self.status,
            "metrics": self.metrics,
            "components": list(self.components.keys()),
            "timestamp": datetime.now().isoformat()
        }
Paso 3: Crear proyecto de prueba

bash
# Crear estructura de prueba
mkdir -p tests/fixtures/sample_project
cat > tests/fixtures/sample_project/hello.py << 'EOF'
"""Módulo de ejemplo para pruebas."""

def saludar(nombre: str) -> str:
    """Saluda a una persona."""
    return f"Hola, {nombre}!"

class Persona:
    """Clase de ejemplo."""
    
    def __init__(self, nombre: str):
        self.nombre = nombre
    
    def presentarse(self) -> str:
        """La persona se presenta."""
        return f"Soy {self.nombre}"
EOF
Paso 4: Implementar project_scanner.py

python
# src/indexer/project_scanner.py
"""
Escáner de proyectos: Recorre directorios y recopila archivos para análisis.
"""
import os
from pathlib import Path
from typing import List, Set
from loguru import logger

from ..core.config_manager import config
from ..core.exceptions import IndexerError
from ..utils.file_utils import safe_read_file
from .models import ProjectStructure

class ProjectScanner:
    """Escanea proyectos para análisis."""
    
    def __init__(self):
        self.excluded_dirs: Set[str] = {
            '.git', '.venv', 'venv', 'node_modules',
            '__pycache__', '.pytest_cache', 'dist', 'build'
        }
    
    def scan(self, project_path: Path) -> ProjectStructure:
        """Escanea un proyecto y retorna su estructura."""
        if not project_path.exists():
            raise IndexerError(f"Proyecto no encontrado: {project_path}")
        
        files: List[Path] = []
        directories: List[Path] = []
        
        for root, dirs, file_names in os.walk(project_path):
            # Filtrar directorios excluidos
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file_name in file_names:
                file_path = Path(root) / file_name
                files.append(file_path)
        
        return ProjectStructure(
            root=project_path,
            files=files,
            directories=directories,
            metadata={
                "total_files": len(files),
                "scanned_at": datetime.now().isoformat()
            }
        )
Paso 5: Continuar con los demás archivos...

📦 DEPENDENCIAS QUE NECESITAS INSTALAR

Actualiza tu setup.sh para incluir:

bash
# En setup.sh, después de pip install -r requirements/base.txt
echo "📥 Instalando dependencias del indexer..."
pip install -r requirements/indexer.txt
O ejecuta manualmente:

bash
pip install tree-sitter tree-sitter-languages chardet python-magic
🔍 VERIFICACIÓN DE ESTADO ACTUAL

Para verificar que tienes todo listo, ejecuta:

bash
# 1. Verificar que los paquetes existen
ls src/core/__init__.py 2>/dev/null && echo "✅ core/__init__.py existe" || echo "❌ FALTA core/__init__.py"
ls src/indexer/__init__.py 2>/dev/null && echo "✅ indexer/__init__.py existe" || echo "❌ FALTA indexer/__init__.py"

# 2. Verificar dependencias
python -c "import tree_sitter" 2>/dev/null && echo "✅ tree-sitter instalado" || echo "❌ tree-sitter NO instalado"

# 3. Verificar estructura de tests
ls tests/fixtures/sample_project/ 2>/dev/null && echo "✅ fixtures existe" || echo "❌ FALTA fixtures"
📊 TIEMPO ESTIMADO DE IMPLEMENTACIÓN

Archivo	Complejidad	Tiempo estimado
system_state.py	Baja	1-2 horas
project_scanner.py	Media	2-3 horas
multi_language_parser.py	Alta	4-6 horas
entity_extractor.py	Media	2-3 horas
orchestrator.py	Media	3-4 horas
Tests	Media	4-5 horas
Total	-	16-23 horas
🚨 PUNTOS CRÍTICOS

tree-sitter installation: Puede tener problemas en algunos sistemas
Python version: Asegúrate de usar Python 3.9+
Path handling: Usa pathlib.Path para compatibilidad cross-platform
✅ CHECKLIST DE COMPLETITUD

src/core/__init__.py creado
src/indexer/__init__.py creado
src/indexer/models.py creado (opcional)
Dependencias de indexer instaladas
Proyecto de prueba creado en tests/fixtures/
Los 5 archivos principales implementados
Tests correspondientes creados
Todo funciona con python -m src.main analyze tests/fixtures/sample_project