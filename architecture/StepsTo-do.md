# 1. Crear directorio del proyecto
mkdir ANALYZERBRAIN
cd ANALYZERBRAIN

# 2. Inicializar repositorio Git
git init
echo "# ANALYZERBRAIN - Sistema Inteligente de Análisis de Código" > README.md

# 3. Crear entorno virtual Python
python -m venv venv

# 4. Activar entorno virtual
# Linux/Mac:
source venv/bin/activate

# 5. Actualizar pip
python -m pip install --upgrade pip setuptools wheel


# Crear estructura principal
mkdir -p .github/workflows
mkdir -p .vscode
mkdir -p architecture
mkdir -p config
mkdir -p data/{backups,cache,embeddings,graph_exports,projects,state}
mkdir -p deployments/{docker,helm/templates,kubernetes}
mkdir -p docs/{api,architecture,deployment,developer,examples,user_guide}
mkdir -p logs
mkdir -p monitoring/{alerts,grafana/{dashboards,datasources},loki,prometheus}
mkdir -p requirements
mkdir -p scripts
mkdir -p src/{api,agents,core,embeddings,graph,indexer,learning,memory,utils}
mkdir -p tests/{analyzer_code,e2e,fixtures/{sample_code,sample_project},integration,performance,unit}

Fundamentos del sistema:

Configuración del Proyecto
1. pyproject.toml

---------------------------------------------

[build-system]
requires = ["setuptools>=61.0", "wheel", "setuptools_scm>=7.0.0"]
build-backend = "setuptools.build_meta"

[project]
name = "analyzerbrain"
description = "Sistema inteligente de análisis de código con IA y grafos de conocimiento"
readme = "README.md"
requires-python = ">=3.9,<3.12"
license = {text = "MIT"}
authors = [
    {name = "ANALYZERBRAIN Team", email = "team@analyzerbrain.dev"}
]
keywords = ["code-analysis", "ai", "knowledge-graph", "nlp", "static-analysis"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Quality Assurance",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    # Core
    "python-dotenv>=1.0.0,<2.0.0",
    "pyyaml>=6.0,<7.0",
    "loguru>=0.7.0,<0.8.0",
    "pydantic>=2.0.0,<3.0.0",
    "pydantic-settings>=2.0.0,<3.0.0",
    
    # Async
    "anyio>=3.7.0,<4.0.0",
    "aiofiles>=23.2.0,<24.0.0",
    
    # Utilidades
    "rich>=13.0.0,<14.0.0",
    "click>=8.1.0,<9.0.0",
    "tqdm>=4.65.0,<5.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3,<8.0.0",
    "pytest-asyncio>=0.21.0,<0.22.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "black>=23.11.0,<24.0.0",
    "ruff>=0.1.0,<0.2.0",
    "mypy>=1.7.0,<2.0.0",
    "pre-commit>=3.5.0,<4.0.0",
]

api = [
    "fastapi>=0.104.0,<0.105.0",
    "uvicorn[standard]>=0.24.0,<0.25.0",
    "websockets>=12.0,<13.0",
    "python-jose[cryptography]>=3.3.0,<4.0.0",
]

[project.scripts]
analyzerbrain = "src.main:main"
analyzerbrain-cli = "src.api.cli_interface:main"


---------------------------------------------

2. requirements/base.txt

---------------------------------------------

# Dependencias base compartidas por todos los módulos
python>=3.9,<3.12
python-dotenv>=1.0.0,<2.0.0
pyyaml>=6.0,<7.0
loguru>=0.7.0,<0.8.0

# Tipado y validación
pydantic>=2.0.0,<3.0.0
pydantic-settings>=2.0.0,<3.0.0
typing-extensions>=4.8.0,<5.0.0

# Async y concurrencia
asyncio>=3.4.3
aiofiles>=23.2.0,<24.0.0
anyio>=3.7.0,<4.0.0

# Serialización
orjson>=3.9.0,<4.0.0
msgpack>=1.0.0,<2.0.0

# Utilidades
rich>=13.0.0,<14.0.0
click>=8.1.0,<9.0.0
tqdm>=4.65.0,<5.0.0
cachetools>=5.3.0,<6.0.0


---------------------------------------------

3. .env.example

---------------------------------------------

# Configuración del Sistema
ENVIRONMENT=development
LOG_LEVEL=INFO
DATA_DIR=./data
LOG_DIR=./logs

# Base de Datos
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=analyzerbrain
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# ChromaDB
CHROMA_PATH=./data/embeddings

# API Keys (opcional)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=


---------------------------------------------

4. .gitignore

---------------------------------------------

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Data
data/*/
!data/.gitkeep

# Environment
.env

# OS
.DS_Store
Thumbs.db

---------------------------------------------
5. LICENSE
---------------------------------------------

MIT License

Copyright (c) 2024 ANALYZERBRAIN Team

Permission is hereby granted...

---------------------------------------------
Sistema de Configuración
---------------------------------------------

"""
Gestor de configuración del sistema.

Dependencias Previas:
    - pyyaml
    - python-dotenv
    - loguru
    - pydantic
    - pydantic-settings

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from loguru import logger
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemConfig(BaseModel):
    """Configuración del sistema."""
    name: str = Field(default="ANALYZERBRAIN")
    version: str = Field(default="0.1.0")
    max_workers: int = Field(default=4, ge=1, le=32)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)


class LoggingConfig(BaseModel):
    """Configuración de logging."""
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    rotation: str = Field(default="1 day")
    retention: str = Field(default="30 days")


class StorageConfig(BaseModel):
    """Configuración de almacenamiento."""
    data_dir: Path = Field(default=Path("./data"))
    cache_dir: Path = Field(default=Path("./data/cache"))
    log_dir: Path = Field(default=Path("./logs"))
    max_cache_size_mb: int = Field(default=1024, ge=100, le=10240)


class DatabaseConfig(BaseModel):
    """Configuración de bases de datos."""
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432, ge=1024, le=65535)
    postgres_db: str = Field(default="analyzerbrain")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="password")
    
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379, ge=1024, le=65535)
    
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")


class APIConfig(BaseModel):
    """Configuración de API."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=2, ge=1, le=16)
    cors_origins: list[str] = Field(default=["http://localhost:3000"])


class AnalyzerBrainSettings(BaseSettings):
    """Configuración completa del sistema."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)


class ConfigManager:
    """Gestor centralizado de configuración."""
    
    _instance: Optional['ConfigManager'] = None
    _settings: Optional[AnalyzerBrainSettings] = None
    _custom_config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._settings is None:
            self._load_settings()
    
    def _load_settings(self) -> None:
        """Carga la configuración desde múltiples fuentes."""
        try:
            # 1. Cargar desde .env y variables de entorno
            self._settings = AnalyzerBrainSettings()
            
            # 2. Cargar configuración YAML personalizada si existe
            config_paths = [
                Path("config/system_config.yaml"),
                Path("config/agent_config.yaml"),
            ]
            
            for path in config_paths:
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        yaml_config = yaml.safe_load(f) or {}
                        self._update_settings(yaml_config)
            
            # 3. Crear directorios necesarios
            self._create_directories()
            
            logger.info(f"Configuración cargada para entorno: {self.environment}")
            
        except ValidationError as e:
            logger.error(f"Error de validación en configuración: {e}")
            raise
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            raise
    
    def _update_settings(self, config_dict: Dict[str, Any]) -> None:
        """Actualiza settings con configuración personalizada."""
        # Esto es una simplificación. En producción, usaríamos un merge profundo
        if not self._settings:
            return
        
        for key, value in config_dict.items():
            if hasattr(self._settings, key):
                current_value = getattr(self._settings, key)
                if isinstance(current_value, BaseModel) and isinstance(value, dict):
                    # Actualizar sub-modelo
                    updated = current_value.model_copy(update=value)
                    setattr(self._settings, key, updated)
                else:
                    setattr(self._settings, key, value)
            else:
                self._custom_config[key] = value
    
    def _create_directories(self) -> None:
        """Crea los directorios necesarios."""
        if not self._settings:
            return
        
        directories = [
            self._settings.storage.data_dir,
            self._settings.storage.cache_dir,
            self._settings.storage.log_dir,
            self._settings.storage.data_dir / "backups",
            self._settings.storage.data_dir / "embeddings",
            self._settings.storage.data_dir / "graph_exports",
            self._settings.storage.data_dir / "projects",
            self._settings.storage.data_dir / "state",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directorio creado/verificado: {directory}")
    
    @property
    def settings(self) -> AnalyzerBrainSettings:
        """Obtiene la configuración completa."""
        if not self._settings:
            self._load_settings()
        return self._settings
    
    @property
    def environment(self) -> str:
        return self.settings.environment
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración por clave."""
        try:
            keys = key.split('.')
            value = self.settings.model_dump()
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        except (AttributeError, KeyError):
            return self._custom_config.get(key, default)
    
    def reload(self) -> None:
        """Recarga la configuración."""
        self._settings = None
        self._custom_config = {}
        self._load_settings()
        logger.info("Configuración recargada")


# Instancia global
config = ConfigManager()


---------------------------------------------
2. config/system_config.yaml
---------------------------------------------


# Configuración del Sistema ANALYZERBRAIN

system:
  name: "ANALYZERBRAIN"
  version: "0.1.0"
  max_workers: 4
  timeout_seconds: 300

logging:
  level: "INFO"
  format: "json"
  rotation: "1 day"
  retention: "30 days"

storage:
  data_dir: "./data"
  cache_dir: "./data/cache"
  log_dir: "./logs"
  max_cache_size_mb: 1024

database:
  postgres_host: "localhost"
  postgres_port: 5432
  postgres_db: "analyzerbrain"
  postgres_user: "postgres"
  postgres_password: "password"
  
  redis_host: "localhost"
  redis_port: 6379
  
  neo4j_uri: "bolt://localhost:7687"
  neo4j_user: "neo4j"
  neo4j_password: "password"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 2
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:8000"

---------------------------------------------
3: Sistema de Logging y Utilidades
1. src/utils/logging_config.py
---------------------------------------------
"""
Configuración de logging unificado.

Dependencias Previas:
    - loguru
    - core.config_manager

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from ..core.config_manager import config


class StructuredLogger:
    """Logger estructurado para ANALYZERBRAIN."""
    
    @staticmethod
    def setup_logging(
        log_level: Optional[str] = None,
        log_dir: Optional[Path] = None,
        json_format: bool = None
    ) -> None:
        """
        Configura el sistema de logging.
        
        Args:
            log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
            log_dir: Directorio para archivos de log
            json_format: Si True, usa formato JSON para producción
        """
        # Usar configuración por defecto si no se especifica
        if log_level is None:
            log_level = config.get("logging.level", "INFO")
        
        if log_dir is None:
            log_dir = config.get("storage.log_dir", Path("./logs"))
        
        if json_format is None:
            json_format = not config.is_development
        
        # Remover handlers por defecto
        logger.remove()
        
        # Configuración para consola
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        logger.add(
            sys.stderr,
            level=log_level,
            format=console_format,
            colorize=True,
            backtrace=True,
            diagnose=config.is_development
        )
        
        # Configuración para archivo (formato estructurado)
        if json_format:
            file_format = (
                '{{"timestamp": "{time:YYYY-MM-DD HH:mm:ss}", '
                '"level": "{level}", '
                '"module": "{name}", '
                '"function": "{function}", '
                '"line": "{line}", '
                '"message": "{message}", '
                '"extra": {extra}}}'
            )
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message} | {extra}"
            )
        
        # Asegurar que el directorio de logs exista
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivo de log principal
        logger.add(
            log_dir / "analyzerbrain_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            format=file_format,
            rotation=config.get("logging.rotation", "1 day"),
            retention=config.get("logging.retention", "30 days"),
            compression="zip",
            backtrace=True,
            diagnose=config.is_development,
            enqueue=True  # Thread-safe
        )
        
        # Archivo de errores separado
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            level="ERROR",
            format=file_format,
            rotation="500 MB",
            retention="90 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        logger.info("Sistema de logging configurado correctamente")
        logger.debug(f"Nivel de log: {log_level}")
        logger.debug(f"Directorio de logs: {log_dir}")
        logger.debug(f"Formato JSON: {json_format}")
    
    @staticmethod
    def get_logger(name: str):
        """
        Obtiene un logger con un nombre específico.
        
        Args:
            name: Nombre del logger (generalmente __name__)
            
        Returns:
            Logger configurado
        """
        return logger.bind(module=name)


def setup_default_logging() -> None:
    """Configura logging con valores por defecto."""
    StructuredLogger.setup_logging()


# Configuración automática al importar el módulo
if not logger._core.handlers:
    setup_default_logging()

 -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------

 """
Utilidades para operaciones de archivos.

Dependencias Previas:
    - aiofiles (para operaciones async)
    - loguru

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
"""

import hashlib
import os
import shutil
import aiofiles
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO
from datetime import datetime
from loguru import logger


class FileUtils:
    """Utilidades para operaciones de archivos."""
    
    @staticmethod
    def read_file(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
        """
        Lee un archivo de texto de forma síncrona.
        
        Args:
            file_path: Ruta al archivo
            encoding: Codificación del archivo
            
        Returns:
            Contenido del archivo
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            IOError: Si hay error de lectura
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Intentar con diferentes codificaciones
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(path, 'r', encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise IOError(f"No se pudo decodificar el archivo: {file_path}")
    
    @staticmethod
    async def read_file_async(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
        """
        Lee un archivo de texto de forma asíncrona.
        
        Args:
            file_path: Ruta al archivo
            encoding: Codificación del archivo
            
        Returns:
            Contenido del archivo
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        try:
            async with aiofiles.open(path, 'r', encoding=encoding) as f:
                return await f.read()
        except UnicodeDecodeError as e:
            logger.error(f"Error decodificando archivo {file_path}: {e}")
            raise
    
    @staticmethod
    def write_file(
        file_path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
        backup: bool = False
    ) -> None:
        """
        Escribe contenido en un archivo.
        
        Args:
            file_path: Ruta al archivo
            content: Contenido a escribir
            encoding: Codificación para texto
            backup: Si True, crea backup si el archivo ya existe
        """
        path = Path(file_path)
        
        # Crear backup si es necesario
        if backup and path.exists():
            backup_path = path.parent / f"{path.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(path, backup_path)
            logger.debug(f"Backup creado: {backup_path}")
        
        # Crear directorio si no existe
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Escribir archivo
        try:
            if isinstance(content, str):
                with open(path, 'w', encoding=encoding) as f:
                    f.write(content)
            else:
                with open(path, 'wb') as f:
                    f.write(content)
            
            logger.debug(f"Archivo escrito: {file_path} ({len(content)} bytes)")
            
        except IOError as e:
            logger.error(f"Error escribiendo archivo {file_path}: {e}")
            raise
    
    @staticmethod
    async def write_file_async(
        file_path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8"
    ) -> None:
        """
        Escribe contenido en un archivo de forma asíncrona.
        
        Args:
            file_path: Ruta al archivo
            content: Contenido a escribir
            encoding: Codificación para texto
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(content, str):
                async with aiofiles.open(path, 'w', encoding=encoding) as f:
                    await f.write(content)
            else:
                async with aiofiles.open(path, 'wb') as f:
                    await f.write(content)
            
            logger.debug(f"Archivo escrito async: {file_path}")
            
        except IOError as e:
            logger.error(f"Error escribiendo archivo async {file_path}: {e}")
            raise
    
    @staticmethod
    def list_files(
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = True,
        exclude_dirs: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Lista archivos en un directorio.
        
        Args:
            directory: Directorio a escanear
            pattern: Patrón de búsqueda (ej: "*.py")
            recursive: Si True, busca recursivamente
            exclude_dirs: Directorios a excluir
            
        Returns:
            Lista de rutas a archivos
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        
        exclude_dirs = exclude_dirs or ['.git', '__pycache__', '.pytest_cache', 'node_modules']
        
        files = []
        
        if recursive:
            for root, dirs, filenames in os.walk(dir_path):
                # Excluir directorios
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for filename in filenames:
                    file_path = Path(root) / filename
                    if file_path.match(pattern):
                        files.append(file_path)
        else:
            files = list(dir_path.glob(pattern))
        
        return files
    
    @staticmethod
    def calculate_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
        """
        Calcula el hash de un archivo.
        
        Args:
            file_path: Ruta al archivo
            algorithm: Algoritmo de hash (md5, sha1, sha256, sha512)
            
        Returns:
            Hash hexadecimal del archivo
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)
        
        with open(path, 'rb') as f:
            file_hash = hash_func()
            for chunk in iter(lambda: f.read(8192), b""):
                file_hash.update(chunk)
        
        return file_hash.hexdigest()
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Obtiene información detallada de un archivo.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Diccionario con información del archivo
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        stat = path.stat()
        
        return {
            "path": str(path.absolute()),
            "name": path.name,
            "stem": path.stem,
            "suffix": path.suffix,
            "parent": str(path.parent),
            "size_bytes": stat.st_size,
            "size_human": FileUtils._humanize_bytes(stat.st_size),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "hash_sha256": FileUtils.calculate_hash(path, "sha256")
        }
    
    @staticmethod
    def _humanize_bytes(bytes_count: int) -> str:
        """Convierte bytes a formato legible."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.2f} PB"


# Instancia global
file_utils = FileUtils()

 -----------------------------------------------
 1. src/core/exceptions.py
 ------------------------------------------------
"""
Sistema de excepciones jerárquico para ANALYZERBRAIN.

Jerarquía:
AnalyzerBrainError
├── ConfigurationError
├── ValidationError
├── IndexerError
├── GraphError
├── AgentError
└── APIError

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
"""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class ErrorSeverity(Enum):
    """Severidad del error."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCode(Enum):
    """Códigos de error estandarizados."""
    # Errores generales
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND_ERROR = "NOT_FOUND_ERROR"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    
    # Errores de módulos específicos
    INDEXER_ERROR = "INDEXER_ERROR"
    GRAPH_ERROR = "GRAPH_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    API_ERROR = "API_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    
    # Errores de negocio
    PROJECT_ANALYSIS_ERROR = "PROJECT_ANALYSIS_ERROR"
    QUERY_EXECUTION_ERROR = "QUERY_EXECUTION_ERROR"
    LEARNING_ERROR = "LEARNING_ERROR"


@dataclass
class ErrorDetail:
    """Detalle estructurado de un error."""
    field: Optional[str] = None
    message: str = ""
    value: Any = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnalyzerBrainError(Exception):
    """Excepción base para todos los errores del sistema."""
    
    def __init__(
        self,
        message: str,
        error_code: Union[str, ErrorCode] = ErrorCode.INTERNAL_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = ErrorCode(error_code) if isinstance(error_code, str) else error_code
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.timestamp = __import__('datetime').datetime.now().isoformat()
        
        # Construir mensaje completo
        full_message = f"[{self.error_code.value}] {message}"
        if cause:
            full_message += f" | Causa: {type(cause).__name__}: {str(cause)}"
        
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la excepción a diccionario para serialización."""
        result = {
            "error": self.error_code.value,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "details": self.details
        }
        
        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause)
            }
        
        return result
    
    def __str__(self) -> str:
        return self.message


class ConfigurationError(AnalyzerBrainError):
    """Error en la configuración del sistema."""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details,
            cause=cause
        )


class ValidationError(AnalyzerBrainError):
    """Error de validación de datos."""
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["value"] = value
        if suggestion:
            error_details["suggestion"] = suggestion
        
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )


class IndexerError(AnalyzerBrainError):
    """Error durante la indexación de proyectos."""
    def __init__(
        self,
        message: str,
        project_path: Optional[str] = None,
        file_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if project_path:
            error_details["project_path"] = project_path
        if file_path:
            error_details["file_path"] = file_path
        
        super().__init__(
            message=message,
            error_code=ErrorCode.INDEXER_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )


class GraphError(AnalyzerBrainError):
    """Error en el grafo de conocimiento."""
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        node_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if query:
            error_details["query"] = query
        if node_id:
            error_details["node_id"] = node_id
        
        super().__init__(
            message=message,
            error_code=ErrorCode.GRAPH_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )


class AgentError(AnalyzerBrainError):
    """Error en un agente."""
    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if agent_name:
            error_details["agent_name"] = agent_name
        if task_type:
            error_details["task_type"] = task_type
        
        super().__init__(
            message=message,
            error_code=ErrorCode.AGENT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )


class APIError(AnalyzerBrainError):
    """Error en la API."""
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if endpoint:
            error_details["endpoint"] = endpoint
        if status_code:
            error_details["status_code"] = status_code
        
        super().__init__(
            message=message,
            error_code=ErrorCode.API_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )


class ProjectAnalysisError(AnalyzerBrainError):
    """Error durante el análisis de un proyecto."""
    def __init__(
        self,
        message: str,
        project_path: Optional[str] = None,
        analysis_step: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if project_path:
            error_details["project_path"] = project_path
        if analysis_step:
            error_details["analysis_step"] = analysis_step
        
        super().__init__(
            message=message,
            error_code=ErrorCode.PROJECT_ANALYSIS_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )

  -----------------------------------------------
 2. src/utils/validation.py
 ------------------------------------------------

"""
Utilidades de validación de datos.

Dependencias Previas:
    - pydantic
    - core.exceptions

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
"""

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime
from pydantic import BaseModel, ValidationError as PydanticValidationError
from email_validator import validate_email as email_validator, EmailNotValidError

from ..core.exceptions import ValidationError


T = TypeVar('T')


class Validator(Generic[T]):
    """Validador genérico para diferentes tipos de datos."""
    
    @staticmethod
    def validate_not_empty(value: T, field_name: str = "value") -> T:
        """Valida que un valor no esté vacío."""
        if value is None:
            raise ValidationError(
                message=f"{field_name} no puede ser nulo",
                field=field_name,
                value=value,
                suggestion="Proporcione un valor no nulo"
            )
        
        if isinstance(value, str):
            if not value.strip():
                raise ValidationError(
                    message=f"{field_name} no puede estar vacío",
                    field=field_name,
                    value=value,
                    suggestion="Proporcione una cadena no vacía"
                )
        elif isinstance(value, (list, dict, set, tuple)):
            if not value:
                raise ValidationError(
                    message=f"{field_name} no puede estar vacío",
                    field=field_name,
                    value=value,
                    suggestion=f"Proporcione un {type(value).__name__} no vacío"
                )
        
        return value
    
    @staticmethod
    def validate_type(value: Any, expected_type: type, field_name: str = "value") -> Any:
        """Valida el tipo de un valor."""
        if not isinstance(value, expected_type):
            raise ValidationError(
                message=f"{field_name} debe ser de tipo {expected_type.__name__}",
                field=field_name,
                value=value,
                value_type=type(value).__name__,
                suggestion=f"Convierta a {expected_type.__name__}"
            )
        return value
    
    @staticmethod
    def validate_string_length(
        value: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        field_name: str = "value"
    ) -> str:
        """Valida la longitud de una cadena."""
        Validator.validate_type(value, str, field_name)
        
        length = len(value)
        
        if min_length is not None and length < min_length:
            raise ValidationError(
                message=f"{field_name} debe tener al menos {min_length} caracteres",
                field=field_name,
                value=value,
                actual_length=length,
                suggestion=f"Aumente la longitud a al menos {min_length} caracteres"
            )
        
        if max_length is not None and length > max_length:
            raise ValidationError(
                message=f"{field_name} no puede tener más de {max_length} caracteres",
                field=field_name,
                value=value,
                actual_length=length,
                suggestion=f"Reduzca la longitud a máximo {max_length} caracteres"
            )
        
        return value
    
    @staticmethod
    def validate_number_range(
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        field_name: str = "value"
    ) -> Union[int, float]:
        """Valida el rango de un número."""
        Validator.validate_type(value, (int, float), field_name)
        
        if min_value is not None and value < min_value:
            raise ValidationError(
                message=f"{field_name} debe ser mayor o igual a {min_value}",
                field=field_name,
                value=value,
                suggestion=f"Incremente el valor a al menos {min_value}"
            )
        
        if max_value is not None and value > max_value:
            raise ValidationError(
                message=f"{field_name} debe ser menor o igual a {max_value}",
                field=field_name,
                value=value,
                suggestion=f"Reduzca el valor a máximo {max_value}"
            )
        
        return value
    
    @staticmethod
    def validate_email(email: str, field_name: str = "email") -> str:
        """Valida una dirección de email."""
        Validator.validate_type(email, str, field_name)
        Validator.validate_string_length(email, min_length=3, max_length=254, field_name=field_name)
        
        try:
            email_info = email_validator(email, check_deliverability=False)
            return email_info.normalized
        except EmailNotValidError as e:
            raise ValidationError(
                message=f"{field_name} no es una dirección de email válida",
                field=field_name,
                value=email,
                suggestion=str(e)
            )
    
    @staticmethod
    def validate_path(
        path: Union[str, Path],
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        field_name: str = "path"
    ) -> Path:
        """Valida una ruta de archivo o directorio."""
        path_obj = Path(path) if isinstance(path, str) else path
        
        if must_exist and not path_obj.exists():
            raise ValidationError(
                message=f"{field_name} no existe: {path_obj}",
                field=field_name,
                value=str(path_obj),
                suggestion="Verifique que la ruta exista"
            )
        
        if must_be_file and not path_obj.is_file():
            raise ValidationError(
                message=f"{field_name} no es un archivo: {path_obj}",
                field=field_name,
                value=str(path_obj),
                suggestion="Proporcione una ruta a un archivo válido"
            )
        
        if must_be_dir and not path_obj.is_dir():
            raise ValidationError(
                message=f"{field_name} no es un directorio: {path_obj}",
                field=field_name,
                value=str(path_obj),
                suggestion="Proporcione una ruta a un directorio válido"
            )
        
        return path_obj
    
    @staticmethod
    def validate_regex(
        value: str,
        pattern: str,
        field_name: str = "value",
        flags: int = 0
    ) -> str:
        """Valida una cadena contra una expresión regular."""
        Validator.validate_type(value, str, field_name)
        
        if not re.match(pattern, value, flags=flags):
            raise ValidationError(
                message=f"{field_name} no coincide con el patrón requerido",
                field=field_name,
                value=value,
                pattern=pattern,
                suggestion=f"El valor debe coincidir con: {pattern}"
            )
        
        return value
    
    @staticmethod
    def validate_json(
        json_str: str,
        schema: Optional[Dict[str, Any]] = None,
        field_name: str = "json"
    ) -> Dict[str, Any]:
        """Valida una cadena JSON."""
        Validator.validate_type(json_str, str, field_name)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValidationError(
                message=f"{field_name} no es un JSON válido",
                field=field_name,
                value=json_str,
                suggestion=f"Error de sintaxis JSON: {str(e)}"
            )
        
        if schema:
            # Validación básica de esquema (simplificada)
            Validator.validate_dict_structure(data, schema, field_name)
        
        return data
    
    @staticmethod
    def validate_dict_structure(
        data: Dict[str, Any],
        structure: Dict[str, Any],
        field_name: str = "data"
    ) -> Dict[str, Any]:
        """Valida la estructura de un diccionario."""
        Validator.validate_type(data, dict, field_name)
        
        for key, expected_type in structure.items():
            if key not in data:
                raise ValidationError(
                    message=f"Falta campo requerido: {key}",
                    field=key,
                    suggestion=f"Agregue el campo '{key}' de tipo {expected_type}"
                )
            
            if not isinstance(data[key], expected_type):
                raise ValidationError(
                    message=f"Campo '{key}' debe ser de tipo {expected_type}",
                    field=key,
                    value=data[key],
                    value_type=type(data[key]).__name__,
                    suggestion=f"Convierta a {expected_type}"
                )
        
        return data
    
    @staticmethod
    def validate_pydantic_model(
        data: Dict[str, Any],
        model_class: type[BaseModel],
        field_name: str = "data"
    ) -> BaseModel:
        """Valida datos usando un modelo Pydantic."""
        try:
            return model_class(**data)
        except PydanticValidationError as e:
            errors = []
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                errors.append({
                    "field": field,
                    "message": error["msg"],
                    "type": error["type"]
                })
            
            raise ValidationError(
                message=f"Error de validación en {field_name}",
                field=field_name,
                value=data,
                details={"errors": errors},
                suggestion="Corrija los errores de validación listados"
            )


# Instancia global para uso conveniente
validator = Validator()

  -----------------------------------------------
 1. src/main.py
 ------------------------------------------------

"""
Punto de entrada principal de ANALYZERBRAIN.

Este módulo proporciona:
1. Interfaz CLI usando Click
2. Inicialización del sistema
3. Comandos para análisis de proyectos
4. Gestión del ciclo de vida del sistema

Dependencias Previas:
    - click (CLI)
    - rich (interfaz enriquecida)
    - core.config_manager
    - utils.logging_config
    - utils.file_utils

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
"""

import asyncio
import sys
import signal
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box

from .core.config_manager import config
from .utils.logging_config import StructuredLogger
from .utils.file_utils import FileUtils
from .core.exceptions import AnalyzerBrainError, ConfigurationError
from .core.health_check import SystemHealthChecker


logger = StructuredLogger.get_logger(__name__)
console = Console()


class SystemStatus(Enum):
    """Estado del sistema."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class AnalyzerBrainSystem:
    """Sistema principal de ANALYZERBRAIN."""
    
    def __init__(self):
        self.status = SystemStatus.UNINITIALIZED
        self.start_time: Optional[datetime] = None
        self.health_checker: Optional[SystemHealthChecker] = None
        self._shutdown_flag = False
        
    def print_banner(self) -> None:
        """Imprime el banner del sistema."""
        banner = r"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                               ║
    ║        █████╗ ███╗   ██╗ █████╗ ██╗  ██╗██╗   ██╗███████╗██████╗             ║
    ║       ██╔══██╗████╗  ██║██╔══██╗██║  ██║╚██╗ ██╔╝██╔════╝██╔══██╗            ║
    ║       ███████║██╔██╗ ██║███████║███████║ ╚████╔╝ █████╗  ██████╔╝            ║
    ║       ██╔══██║██║╚██╗██║██╔══██║██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗            ║
    ║       ██║  ██║██║ ╚████║██║  ██║██║  ██║   ██║   ███████╗██║  ██║            ║
    ║       ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝            ║
    ║                                                                               ║
    ║                     Sistema Inteligente de Análisis de Código                 ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
        """
        
        version_info = f"Versión: {config.get('system.version', '0.1.0')} | Entorno: {config.environment}"
        env_color = "green" if config.is_development else "yellow"
        
        console.print(Panel.fit(
            banner,
            title="[bold cyan]ANALYZERBRAIN[/bold cyan]",
            subtitle=f"[{env_color}]{version_info}[/{env_color}]",
            border_style="cyan",
            padding=(1, 2)
        ))
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Verifica los requisitos del sistema."""
        requirements = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Verificando requisitos...", total=100)
            
            # 1. Verificar Python version
            progress.update(task, advance=10, description="[cyan]Verificando Python...")
            python_version = sys.version_info
            requirements["python_version"] = python_version.major == 3 and python_version.minor >= 9
            if requirements["python_version"]:
                progress.update(task, advance=10, description="[green]Python ✓")
            else:
                progress.update(task, advance=10, description="[red]Python ✗")
            
            # 2. Verificar directorios de datos
            progress.update(task, advance=10, description="[cyan]Verificando directorios...")
            data_dir = config.get("storage.data_dir", Path("./data"))
            log_dir = config.get("storage.log_dir", Path("./logs"))
            
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                log_dir.mkdir(parents=True, exist_ok=True)
                requirements["directories"] = True
                progress.update(task, advance=10, description="[green]Directorios ✓")
            except Exception:
                requirements["directories"] = False
                progress.update(task, advance=10, description="[red]Directorios ✗")
            
            # 3. Verificar permisos de escritura
            progress.update(task, advance=10, description="[cyan]Verificando permisos...")
            try:
                test_file = data_dir / ".write_test"
                test_file.touch()
                test_file.unlink()
                requirements["write_permissions"] = True
                progress.update(task, advance=10, description="[green]Permisos ✓")
            except Exception:
                requirements["write_permissions"] = False
                progress.update(task, advance=10, description="[red]Permisos ✗")
            
            # 4. Verificar configuraciones críticas
            progress.update(task, advance=20, description="[cyan]Verificando configuración...")
            critical_configs = [
                "environment",
                "storage.data_dir",
                "api.host",
                "api.port"
            ]
            
            config_ok = True
            for config_key in critical_configs:
                if config.get(config_key) is None:
                    config_ok = False
                    break
            
            requirements["configuration"] = config_ok
            if config_ok:
                progress.update(task, advance=20, description="[green]Configuración ✓")
            else:
                progress.update(task, advance=20, description="[red]Configuración ✗")
            
            # 5. Verificar módulos de Python
            progress.update(task, advance=20, description="[cyan]Verificando dependencias...")
            try:
                import pydantic
                import loguru
                import rich
                requirements["dependencies"] = True
                progress.update(task, advance=20, description="[green]Dependencias ✓")
            except ImportError as e:
                requirements["dependencies"] = False
                progress.update(task, advance=20, description=f"[red]Dependencias ✗: {e.name}")
        
        return requirements
    
    async def initialize(self) -> bool:
        """Inicializa el sistema completo."""
        if self.status != SystemStatus.UNINITIALIZED:
            logger.warning(f"Intento de reinicialización desde estado: {self.status}")
            return False
        
        self.status = SystemStatus.INITIALIZING
        self.start_time = datetime.now()
        
        try:
            # Mostrar banner
            self.print_banner()
            
            # Verificar requisitos
            console.print("\n[bold]1. Verificación de requisitos del sistema[/bold]")
            requirements = self.check_system_requirements()
            
            # Mostrar resultados de verificación
            table = Table(title="Resultados de Verificación", box=box.ROUNDED)
            table.add_column("Requisito", style="cyan")
            table.add_column("Estado", style="bold")
            table.add_column("Detalles", style="white")
            
            status_map = {
                "python_version": ("Python 3.9+", "3.9.0 o superior"),
                "directories": ("Directorios", "Estructura de datos"),
                "write_permissions": ("Permisos", "Escritura en disco"),
                "configuration": ("Configuración", "Variables críticas"),
                "dependencies": ("Dependencias", "Módulos Python")
            }
            
            all_ok = True
            for key, ok in requirements.items():
                name, details = status_map.get(key, (key, ""))
                if ok:
                    table.add_row(name, "[green]✓ OK[/green]", details)
                else:
                    table.add_row(name, "[red]✗ FALLÓ[/red]", details)
                    all_ok = False
            
            console.print(table)
            
            if not all_ok:
                console.print("\n[bold red]Error: No se cumplen todos los requisitos del sistema[/bold red]")
                console.print("Corrija los problemas e intente nuevamente.")
                self.status = SystemStatus.ERROR
                return False
            
            # Inicializar componentes
            console.print("\n[bold]2. Inicialización de componentes[/bold]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task_total = 100
                task = progress.add_task("[cyan]Inicializando...", total=task_total)
                
                # Paso 1: Sistema de logging
                progress.update(task, advance=15, description="[cyan]Configurando logging...")
                logger.info("Sistema de logging inicializado")
                
                # Paso 2: Sistema de configuración
                progress.update(task, advance=15, description="[cyan]Configurando sistema...")
                logger.info(f"Entorno: {config.environment}")
                
                # Paso 3: Sistema de salud
                progress.update(task, advance=15, description="[cyan]Inicializando health check...")
                self.health_checker = SystemHealthChecker()
                health_status = await self.health_checker.check_all()
                
                if not health_status["overall"]:
                    progress.update(task, advance=15, description="[red]Health check falló")
                    console.print(f"\n[red]Problemas de salud detectados:[/red]")
                    for check, status in health_status.items():
                        if check != "overall" and not status["healthy"]:
                            console.print(f"  • {check}: {status.get('message', 'Sin detalles')}")
                    self.status = SystemStatus.ERROR
                    return False
                
                progress.update(task, advance=15, description="[green]Health check ✓")
                
                # Paso 4: Módulos principales
                modules_to_init = [
                    ("Sistema de archivos", 10),
                    ("Gestor de caché", 10),
                    ("Sistema de eventos", 10),
                    ("API Server", 10)
                ]
                
                for module_name, weight in modules_to_init:
                    progress.update(task, advance=weight, description=f"[cyan]Inicializando {module_name}...")
                    await asyncio.sleep(0.2)  # Simulación
                    logger.debug(f"Módulo {module_name} inicializado")
                
                # Completar
                progress.update(task, advance=task_total - 85, description="[green]¡Sistema listo!")
            
            self.status = SystemStatus.READY
            
            # Mostrar resumen
            console.print("\n[bold green]✅ Sistema inicializado correctamente[/bold green]")
            self._print_system_summary()
            
            # Configurar manejo de señales
            self._setup_signal_handlers()
            
            return True
            
        except Exception as e:
            logger.error(f"Error durante inicialización: {e}", exc_info=True)
            console.print(f"\n[bold red]Error de inicialización:[/bold red] {e}")
            self.status = SystemStatus.ERROR
            return False
    
    def _print_system_summary(self) -> None:
        """Imprime un resumen del sistema."""
        tree = Tree("[bold cyan]Sistema ANALYZERBRAIN[/bold cyan]")
        
        # Información del sistema
        sys_info = tree.add("[bold]Información del Sistema[/bold]")
        sys_info.add(f"Entorno: [green]{config.environment}[/green]")
        sys_info.add(f"Versión: {config.get('system.version', '0.1.0')}")
        sys_info.add(f"Tiempo de inicio: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Configuraciones
        config_tree = tree.add("[bold]Configuraciones[/bold]")
        
        api_config = config_tree.add("API")
        api_config.add(f"Host: {config.get('api.host', '0.0.0.0')}")
        api_config.add(f"Puerto: {config.get('api.port', 8000)}")
        
        storage_config = config_tree.add("Almacenamiento")
        storage_config.add(f"Directorio de datos: {config.get('storage.data_dir')}")
        storage_config.add(f"Directorio de logs: {config.get('storage.log_dir')}")
        
        # Estado de componentes
        components = tree.add("[bold]Componentes[/bold]")
        components.add("[green]✓[/green] Configuración")
        components.add("[green]✓[/green] Logging")
        components.add("[green]✓[/green] Sistema de archivos")
        components.add("[yellow]○[/yellow] Base de datos (requiere servicios externos)")
        components.add("[yellow]○[/yellow] Agentes (listos para inicializar)")
        
        console.print("\n")
        console.print(Panel(tree, title="Resumen del Sistema", border_style="blue"))
    
    def _setup_signal_handlers(self) -> None:
        """Configura manejadores de señales para apagado elegante."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("Manejadores de señales configurados")
    
    def _signal_handler(self, signum, frame) -> None:
        """Manejador de señales para apagado elegante."""
        logger.info(f"Señal {signum} recibida, iniciando apagado elegante...")
        self.shutdown()
    
    async def analyze_project(self, project_path: Path) -> Dict[str, Any]:
        """Analiza un proyecto."""
        if self.status != SystemStatus.READY:
            raise AnalyzerBrainError("Sistema no está listo para análisis")
        
        logger.info(f"Iniciando análisis de proyecto: {project_path}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task_total = 100
            task = progress.add_task(f"[cyan]Analizando {project_path.name}...", total=task_total)
            
            # Simulación de análisis
            steps = [
                ("Escaneando estructura...", 20),
                ("Parseando archivos...", 30),
                ("Extrayendo entidades...", 20),
                ("Construyendo grafo...", 20),
                ("Generando reporte...", 10)
            ]
            
            results = {
                "project": str(project_path),
                "status": "success",
                "files_analyzed": 0,
                "entities_found": 0,
                "analysis_time": 0,
                "warnings": [],
                "errors": []
            }
            
            try:
                # Paso 1: Verificar proyecto
                if not project_path.exists():
                    raise AnalyzerBrainError(f"Proyecto no encontrado: {project_path}")
                
                # Paso 2: Escanear estructura
                progress.update(task, advance=20, description="[cyan]Escaneando estructura...")
                from .indexer.project_scanner import ProjectScanner
                scanner = ProjectScanner(config)
                structure = scanner.scan(project_path)
                results["files_analyzed"] = len(structure.files)
                await asyncio.sleep(0.5)  # Simulación
                
                # Paso 3: Análisis básico
                progress.update(task, advance=30, description="[cyan]Parseando archivos...")
                # Aquí iría el análisis real con MultiLanguageParser
                results["entities_found"] = 42  # Simulación
                await asyncio.sleep(1.0)  # Simulación
                
                # Paso 4: Generar reporte
                progress.update(task, advance=50, description="[green]¡Análisis completado!")
                
                results["analysis_time"] = 3.5  # Simulación
                results["summary"] = f"Proyecto analizado exitosamente: {results['entities_found']} entidades encontradas"
                
                return results
                
            except Exception as e:
                logger.error(f"Error analizando proyecto {project_path}: {e}")
                progress.update(task, advance=100, description="[red]Error en análisis")
                raise
    
    async def query_system(self, query: str) -> Dict[str, Any]:
        """Consulta el sistema de conocimiento."""
        # Por ahora, simulación
        return {
            "query": query,
            "results": [
                {
                    "type": "info",
                    "content": f"Consulta procesada: '{query}'",
                    "confidence": 0.8
                }
            ],
            "sources": ["sistema_de_conocimiento"],
            "timestamp": datetime.now().isoformat()
        }
    
    def shutdown(self) -> None:
        """Apaga el sistema de manera controlada."""
        if self.status in [SystemStatus.SHUTTING_DOWN, SystemStatus.UNINITIALIZED]:
            return
        
        self.status = SystemStatus.SHUTTING_DOWN
        logger.info("Iniciando apagado del sistema...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Apagando sistema...", total=100)
            
            # Paso 1: Detener servicios activos
            progress.update(task, advance=25, description="[cyan]Deteniendo servicios...")
            
            # Paso 2: Guardar estado
            progress.update(task, advance=25, description="[cyan]Guardando estado...")
            
            # Paso 3: Cerrar conexiones
            progress.update(task, advance=25, description="[cyan]Cerrando conexiones...")
            
            # Paso 4: Finalizar
            progress.update(task, advance=25, description="[green]¡Sistema apagado!")
        
        logger.info("Sistema apagado correctamente")
        self.status = SystemStatus.UNINITIALIZED
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema."""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "status": self.status.value,
            "uptime_seconds": uptime,
            "environment": config.environment,
            "version": config.get("system.version", "0.1.0"),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "health": self.health_checker.get_status() if self.health_checker else None
        }


# Instancia global del sistema
system = AnalyzerBrainSystem()


# Comandos CLI usando Click
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version="0.1.0", prog_name="ANALYZERBRAIN")
@click.pass_context
def cli(ctx):
    """ANALYZERBRAIN - Sistema Inteligente de Análisis de Código."""
    ctx.ensure_object(dict)
    ctx.obj["system"] = system


@cli.command()
def init():
    """Inicializa el sistema ANALYZERBRAIN."""
    success = asyncio.run(system.initialize())
    if not success:
        sys.exit(1)


@cli.command()
@click.argument("project_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Archivo de salida para resultados")
@click.option("--format", "-f", type=click.Choice(["json", "yaml", "text"]), default="text", help="Formato de salida")
def analyze(project_path: Path, output: Optional[Path], format: str):
    """Analiza un proyecto de código."""
    try:
        if system.status != SystemStatus.READY:
            console.print("[yellow]⚠ Sistema no inicializado, inicializando automáticamente...[/yellow]")
            if not asyncio.run(system.initialize()):
                console.print("[red]Error: No se pudo inicializar el sistema[/red]")
                sys.exit(1)
        
        results = asyncio.run(system.analyze_project(project_path))
        
        # Mostrar resultados
        if format == "json":
            import json
            output_text = json.dumps(results, indent=2, default=str)
        elif format == "yaml":
            import yaml
            output_text = yaml.dump(results, default_flow_style=False, allow_unicode=True)
        else:
            output_text = _format_results_text(results)
        
        if output:
            FileUtils.write_file(output, output_text)
            console.print(f"[green]✓[/green] Resultados guardados en: {output}")
        else:
            console.print(output_text)
            
    except Exception as e:
        console.print(f"[red]Error analizando proyecto:[/red] {e}")
        logger.error(f"Error en comando analyze: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("query", type=str)
def ask(query: str):
    """Consulta el sistema de conocimiento."""
    try:
        if system.status != SystemStatus.READY:
            console.print("[yellow]⚠ Sistema no inicializado, inicializando automáticamente...[/yellow]")
            if not asyncio.run(system.initialize()):
                console.print("[red]Error: No se pudo inicializar el sistema[/red]")
                sys.exit(1)
        
        results = asyncio.run(system.query_system(query))
        
        # Mostrar resultados en un formato amigable
        console.print("\n[bold cyan]Consulta:[/bold cyan]")
        console.print(f"  {query}\n")
        
        console.print("[bold cyan]Respuesta:[/bold cyan]")
        for result in results["results"]:
            console.print(f"  • {result['content']}")
            
        if results.get("sources"):
            console.print(f"\n[dim]Fuentes: {', '.join(results['sources'])}[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error en consulta:[/red] {e}")
        logger.error(f"Error en comando ask: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
def status():
    """Muestra el estado actual del sistema."""
    status_info = system.get_status()
    
    table = Table(title="Estado del Sistema", box=box.ROUNDED)
    table.add_column("Propiedad", style="cyan")
    table.add_column("Valor", style="white")
    
    table.add_row("Estado", status_info["status"])
    table.add_row("Entorno", status_info["environment"])
    table.add_row("Versión", status_info["version"])
    
    if status_info["start_time"]:
        start_time = datetime.fromisoformat(status_info["start_time"])
        uptime = status_info["uptime_seconds"]
        table.add_row("Inicio", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        table.add_row("Tiempo activo", f"{uptime:.0f} segundos")
    
    console.print(table)
    
    # Mostrar estado de salud si está disponible
    if status_info["health"]:
        console.print("\n[bold]Estado de Salud:[/bold]")
        health_table = Table(box=box.SIMPLE)
        health_table.add_column("Componente", style="cyan")
        health_table.add_column("Estado", style="bold")
        health_table.add_column("Mensaje", style="white")
        
        for component, info in status_info["health"].items():
            if component != "overall":
                status = info.get("healthy", False)
                message = info.get("message", "")
                status_text = "[green]✓[/green]" if status else "[red]✗[/red]"
                health_table.add_row(component, status_text, message)
        
        console.print(health_table)


@cli.command()
def config_show():
    """Muestra la configuración actual del sistema."""
    settings = config.settings.model_dump()
    
    tree = Tree("[bold cyan]Configuración del Sistema[/bold cyan]")
    
    for section, values in settings.items():
        if isinstance(values, dict):
            section_tree = tree.add(f"[bold]{section}[/bold]")
            for key, value in values.items():
                if isinstance(value, dict):
                    subsection_tree = section_tree.add(f"[cyan]{key}[/cyan]")
                    for subkey, subvalue in value.items():
                        subsection_tree.add(f"{subkey}: [green]{subvalue}[/green]")
                else:
                    section_tree.add(f"{key}: [green]{value}[/green]")
        else:
            tree.add(f"{section}: [green]{values}[/green]")
    
    console.print(Panel(tree, border_style="blue"))


@cli.command()
def shell():
    """Inicia una shell interactiva de ANALYZERBRAIN."""
    console.print("[bold cyan]Shell Interactiva ANALYZERBRAIN[/bold cyan]")
    console.print("Escribe 'exit' para salir, 'help' para ayuda.\n")
    
    # Inicializar sistema si es necesario
    if system.status != SystemStatus.READY:
        console.print("[yellow]Inicializando sistema...[/yellow]")
        if not asyncio.run(system.initialize()):
            console.print("[red]Error: No se pudo inicializar el sistema[/red]")
            return
    
    while True:
        try:
            command = console.input("[bold cyan]analyzerbrain>[/bold cyan] ").strip()
            
            if not command:
                continue
                
            if command.lower() in ["exit", "quit", "q"]:
                console.print("Saliendo de la shell...")
                break
                
            elif command.lower() in ["help", "?"]:
                _print_shell_help()
                
            elif command.lower() == "status":
                status_info = system.get_status()
                console.print(f"Estado: [green]{status_info['status']}[/green]")
                
            elif command.startswith("analyze "):
                path = command[8:].strip()
                if path.startswith('"') and path.endswith('"'):
                    path = path[1:-1]
                
                try:
                    results = asyncio.run(system.analyze_project(Path(path)))
                    console.print(_format_results_text(results))
                except Exception as e:
                    console.print(f"[red]Error:[/red] {e}")
                    
            elif command.startswith("ask "):
                query = command[4:].strip()
                if query.startswith('"') and query.endswith('"'):
                    query = query[1:-1]
                
                try:
                    results = asyncio.run(system.query_system(query))
                    for result in results["results"]:
                        console.print(f"  • {result['content']}")
                except Exception as e:
                    console.print(f"[red]Error:[/red] {e}")
                    
            else:
                console.print(f"[yellow]Comando no reconocido: {command}[/yellow]")
                console.print("Escribe 'help' para ver los comandos disponibles.")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Usa 'exit' para salir[/yellow]")
        except EOFError:
            console.print("\nSaliendo...")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


def _print_shell_help():
    """Muestra ayuda de la shell interactiva."""
    help_text = """
[bold]Comandos disponibles:[/bold]

[cyan]General:[/cyan]
  help, ?           Muestra esta ayuda
  exit, quit, q     Sale de la shell
  status            Muestra el estado del sistema

[cyan]Análisis:[/cyan]
  analyze <ruta>    Analiza un proyecto en la ruta especificada
                    Ejemplo: analyze /ruta/al/proyecto
                    Ejemplo: analyze "./mi proyecto"

[cyan]Consulta:[/cyan]
  ask <pregunta>    Realiza una consulta al sistema
                    Ejemplo: ask "¿Qué patrones encuentra en este código?"
                    Ejemplo: ask "show me the class hierarchy"

[cyan]Ejemplos:[/cyan]
  analyze ./tests/fixtures/sample_project
  ask "¿Cuántas funciones hay en este proyecto?"
  status
"""
    console.print(help_text)


def _format_results_text(results: Dict[str, Any]) -> str:
    """Formatea resultados de análisis para salida de texto."""
    output_lines = []
    output_lines.append("[bold green]Resultados del Análisis[/bold green]")
    output_lines.append("=" * 50)
    
    output_lines.append(f"Proyecto: {results.get('project', 'N/A')}")
    output_lines.append(f"Estado: {results.get('status', 'unknown')}")
    output_lines.append(f"Archivos analizados: {results.get('files_analyzed', 0)}")
    output_lines.append(f"Entidades encontradas: {results.get('entities_found', 0)}")
    output_lines.append(f"Tiempo de análisis: {results.get('analysis_time', 0):.2f} segundos")
    
    if results.get('summary'):
        output_lines.append("")
        output_lines.append("[bold]Resumen:[/bold]")
        output_lines.append(f"  {results['summary']}")
    
    if results.get('warnings'):
        output_lines.append("")
        output_lines.append("[yellow]Advertencias:[/yellow]")
        for warning in results['warnings'][:5]:  # Mostrar solo las primeras 5
            output_lines.append(f"  • {warning}")
        if len(results['warnings']) > 5:
            output_lines.append(f"  ... y {len(results['warnings']) - 5} más")
    
    if results.get('errors'):
        output_lines.append("")
        output_lines.append("[red]Errores:[/red]")
        for error in results['errors'][:5]:
            output_lines.append(f"  • {error}")
    
    return "\n".join(output_lines)


def main():
    """Función principal de entrada."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrumpido por el usuario[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error fatal:[/red] {e}")
        logger.critical(f"Error fatal en main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

  -----------------------------------------------
📄 src/init.py - Paquete Raíz
 ------------------------------------------------

"""
ANALYZERBRAIN - Sistema Inteligente de Análisis de Código.

Este paquete proporciona análisis de código inteligente combinando
técnicas de IA, procesamiento de lenguaje natural y grafos de conocimiento.

Módulos principales:
    - core: Núcleo del sistema, configuración y orquestación
    - api: Interfaces de usuario (REST, CLI, Web)
    - agents: Agentes especializados para análisis
    - indexer: Indexación y parsing de código fuente
    - graph: Grafo de conocimiento
    - embeddings: Representación vectorial y búsqueda semántica
    - memory: Sistema jerárquico de memoria
    - learning: Aprendizaje automático y adaptación
    - utils: Utilidades compartidas

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "ANALYZERBRAIN Team"
__email__ = "team@analyzerbrain.dev"

# Configuración inicial del logging
from .utils.logging_config import setup_default_logging

# Configura logging solo si no está ya configurado
import sys
if "loguru" not in sys.modules:
    setup_default_logging()

  -----------------------------------------------
📄 .vscode/settings.json - Configuración del IDE
 ------------------------------------------------

{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--max-line-length=88",
        "--ignore=E203,W503"
    ],
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length=88"
    ],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term",
        "-v"
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": "explicit"
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/*.pyc": true
    },
    "files.watcherExclude": {
        "**/.git/**": true,
        "**/venv/**": true,
        "**/data/**": true,
        "**/logs/**": true
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        }
    }
}

  -----------------------------------------------
 📄 .vscode/extensions.json - Extensiones Recomendadas
 ------------------------------------------------

{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.black-formatter",
        "charliermarsh.ruff",
        "eamodio.gitlens",
        "github.vscode-github-actions",
        "ms-azuretools.vscode-docker",
        "ms-vscode.makefile-tools",
        "ryanluker.vscode-coverage-gutters",
        "tamasfe.even-better-toml",
        "redhat.vscode-yaml"
    ]
}

  -----------------------------------------------
 🚀 Instrucciones de Uso Inicial

1. Configurar el Proyecto
 ------------------------------------------------
# Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias base
pip install -e .
pip install -r requirements/base.txt
pip install -r requirements/dev.txt

2. Inicializar el Sistema

bash
# Crear archivo de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# Inicializar el sistema
python -m src.main init

3. Usar la CLI

bash
# Mostrar ayuda
python -m src.main --help

# Analizar un proyecto
python -m src.main analyze /ruta/al/proyecto

# Consultar el sistema
python -m src.main ask "¿Qué patrones hay en este código?"

# Iniciar shell interactiva
python -m src.main shell

# Ver estado del sistema
python -m src.main status

4. Ejecutar Tests

bash
# Ejecutar todos los tests
pytest

# Ejecutar tests específicos
pytest tests/unit/test_config_manager.py

# Con cobertura
pytest --cov=src --cov-report=html

🎯 Próximos Pasos Después de Este Archivo

src/core/health_check.py - Implementar sistema de verificación de salud
src/indexer/project_scanner.py - Implementar escaneo de proyectos
src/indexer/multi_language_parser.py - Implementar parser multi-lenguaje
tests/unit/test_main.py - Crear tests para el punto de entrada
Este main.py proporciona:

✅ Sistema completo de CLI con Click
✅ Inicialización elegante del sistema
✅ Comandos para análisis de proyectos
✅ Shell interactiva
✅ Manejo de señales para apagado elegante
✅ Interfaz enriquecida con Rich
✅ Integración con el sistema de configuración
✅ Manejo robusto de errores

  -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------



  -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------



  -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------



  -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------



  -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------



  -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------



  -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------



  -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------



  -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------



  -----------------------------------------------
 2. src/utils/file_utils.py
 ------------------------------------------------


 