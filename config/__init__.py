"""
Configuración centralizada del sistema Project Brain.
Refactorizado para mejor rendimiento, mantenibilidad y testabilidad.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, TypeVar, Generic, Type
from functools import lru_cache, cached_property
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading

from pydantic import BaseModel, Field, ConfigDict, field_validator

logger = logging.getLogger(__name__)

# =========================
# CONSTANTES Y TIPOS
# =========================

T = TypeVar("T", bound=BaseModel)

CONFIG_FILES = {
    "system": "system.yaml",
    "models": "models.yaml", 
    "agents": "agents.yaml",
    "databases": "databases.yaml",
    "api": "api.yaml",
}

ENV_PREFIXES = {
    "BRAIN_": "system",
    "POSTGRES_": "databases.postgresql",
    "NEO4J_": "databases.neo4j", 
    "REDIS_": "databases.redis",
    "API_": "api.rest",
    "JWT_": "api.authentication.jwt",
}

# =========================
# EXCEPCIONES ESPECIALIZADAS
# =========================

class ConfigError(Exception):
    """Error base para problemas de configuración."""
    pass

class ConfigLoadError(ConfigError):
    """Error al cargar configuración desde archivo."""
    pass

class ConfigValidationError(ConfigError):
    """Error al validar configuración."""
    pass

class ConfigResolveError(ConfigError):
    """Error al resolver variables de entorno."""
    pass

# =========================
# INTERFACES Y COMPONENTES
# =========================

class ConfigLoader(ABC):
    """Interfaz para cargar configuración."""
    
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Carga la configuración completa."""
        pass
    
    @abstractmethod
    def get_raw_value(self, key: str, default: Any = None) -> Any:
        """Obtiene valor raw sin validación."""
        pass

class EnvResolver(ABC):
    """Interfaz para resolver variables de entorno."""
    
    @abstractmethod
    def resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resuelve variables de entorno en la configuración."""
        pass

class ConfigValidator(ABC):
    """Interfaz para validar configuración."""
    
    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> None:
        """Valida la configuración."""
        pass

# =========================
# MODELOS DE CONFIGURACIÓN REFACTORIZADOS
# =========================

class SystemConfig(BaseModel):
    """Configuración del sistema."""
    model_config = ConfigDict(extra="allow")
    
    name: str = Field(default="Project Brain")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    debug_mode: bool = Field(default=False)
    data_directory: Path = Field(default=Path("./data"))
    log_directory: Path = Field(default=Path("./logs"))
    
    @field_validator("data_directory", "log_directory", mode="before")
    @classmethod
    def validate_paths(cls, v):
        """Convierte strings a Path y crea directorios."""
        if isinstance(v, str):
            v = Path(v)
        if isinstance(v, Path):
            v.mkdir(parents=True, exist_ok=True)
        return v

class DatabaseConfig(BaseModel):
    """Configuración de base de datos."""
    
    enabled: bool = Field(default=True)
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="project_brain")
    username: str = Field(default="brain_user")
    password: str = Field(default="")
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=10)
    echo: bool = Field(default=False)
    
    @field_validator("password", mode="before")
    @classmethod
    def resolve_password(cls, v: Any) -> str:
        """Resuelve contraseñas desde variables de entorno."""
        if isinstance(v, str):
            if v.startswith("${") and v.endswith("}"):
                env_var = v[2:-1]
                resolved = os.getenv(env_var)
                if resolved is None:
                    raise ValueError(f"Variable de entorno {env_var} no definida")
                return resolved
        return str(v)

class APIConfig(BaseModel):
    """Configuración de API."""
    
    enabled: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    cors_origins: List[str] = Field(default_factory=list)

# =========================
# IMPLEMENTACIONES CONCRETAS
# =========================

@dataclass
class YamlConfigLoader(ConfigLoader):
    """Cargador de configuración desde archivos YAML."""
    
    config_dir: Path
    config_files: Dict[str, str] = field(default_factory=lambda: CONFIG_FILES)
    
    def load(self) -> Dict[str, Any]:
        """Carga configuración desde archivos YAML."""
        config = {}
        
        for key, filename in self.config_files.items():
            filepath = self.config_dir / filename
            config[key] = self._load_yaml_file(filepath)
        
        return config
    
    def get_raw_value(self, key: str, default: Any = None) -> Any:
        """Obtiene valor raw para una clave específica."""
        parts = key.split(".")
        config = self.load()
        
        for part in parts:
            if isinstance(config, dict):
                config = config.get(part, {})
            else:
                return default
        
        return config if config != {} else default
    
    @staticmethod
    def _load_yaml_file(filepath: Path) -> Dict[str, Any]:
        """Carga un archivo YAML."""
        if not filepath.exists():
            logger.warning(f"Archivo de configuración no encontrado: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"Error parseando YAML {filepath}: {e}")
            raise ConfigLoadError(f"Error parseando {filepath}: {e}") from e
        except Exception as e:
            logger.error(f"Error leyendo {filepath}: {e}")
            raise ConfigLoadError(f"Error leyendo {filepath}: {e}") from e

@dataclass
class EnvironmentResolver(EnvResolver):
    """Resolvedor de variables de entorno."""
    
    env_prefixes: Dict[str, str] = field(default_factory=lambda: ENV_PREFIXES)
    
    def resolve(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resuelve variables de entorno en la configuración."""
        import copy
        resolved_config = copy.deepcopy(config)
        
        for env_prefix, config_path in self.env_prefixes.items():
            for env_key, env_value in os.environ.items():
                if env_key.startswith(env_prefix):
                    # Convertir ENV_VAR_NAME a env.var.name
                    config_key = self._env_to_config_key(env_key, env_prefix, config_path)
                    self._set_nested_value(resolved_config, config_key, env_value)
        
        return resolved_config
    
    @staticmethod
    def _env_to_config_key(env_key: str, prefix: str, base_path: str) -> str:
        """Convierte nombre de variable de entorno a clave de configuración."""
        # Ejemplo: BRAIN_LOG_LEVEL -> system.log_level
        suffix = env_key[len(prefix):].lower()
        parts = suffix.split('_')
        config_key = '.'.join(parts)
        return f"{base_path}.{config_key}" if base_path else config_key
    
    @staticmethod
    def _set_nested_value(config: Dict[str, Any], key: str, value: Any) -> None:
        """Establece valor en diccionario anidado usando dot notation."""
        parts = key.split('.')
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Convertir tipos básicos
        if isinstance(current.get(parts[-1]), bool):
            value = str(value).lower() in ('true', '1', 'yes')
        elif isinstance(current.get(parts[-1]), int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                pass
        elif isinstance(current.get(parts[-1]), float):
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
        
        current[parts[-1]] = value

@dataclass
class ConfigValidatorImpl(ConfigValidator):
    """Validador de configuración."""
    
    def validate(self, config: Dict[str, Any]) -> None:
        """Valida la configuración completa."""
        errors = []
        
        # Validar directorios
        system_config = config.get("system", {})
        if data_dir := system_config.get("data_directory"):
            try:
                Path(data_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"No se puede crear directorio {data_dir}: {e}")
        
        # Validar puertos
        if api_config := config.get("api", {}).get("rest", {}):
            if port := api_config.get("port"):
                if not (1024 <= port <= 65535):
                    errors.append(f"Puerto API inválido: {port}")
        
        # Validar bases de datos habilitadas
        if databases := config.get("databases", {}):
            for db_name, db_config in databases.items():
                if db_config.get("enabled", False):
                    if not db_config.get("host"):
                        errors.append(f"Base de datos {db_name} habilitada sin host")
        
        if errors:
            raise ConfigValidationError("; ".join(errors))

# =========================
# CACHÉ Y GESTIÓN DE ESTADO
# =========================

class ConfigCache:
    """Caché para configuración validada."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._config_hash: Optional[str] = None
    
    def get(self, model_type: Type[T], config_dict: Dict[str, Any]) -> T:
        """Obtiene configuración validada desde caché o crea nueva."""
        cache_key = f"{model_type.__name__}:{self._dict_hash(config_dict)}"
        
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            try:
                instance = model_type(**config_dict)
                self._cache[cache_key] = instance
                return instance
            except Exception as e:
                raise ConfigValidationError(f"Error validando {model_type.__name__}: {e}") from e
    
    def clear(self) -> None:
        """Limpia la caché."""
        with self._lock:
            self._cache.clear()
            self._config_hash = None
    
    @staticmethod
    def _dict_hash(data: Dict[str, Any]) -> str:
        """Crea hash simple para un diccionario."""
        import hashlib
        import json
        
        try:
            serialized = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(serialized.encode()).hexdigest()
        except:
            return "unknown"

# =========================
# FACHADA PRINCIPAL (SINGLETON MEJORADO)
# =========================

class BrainConfig:
    """
    Fachada principal para manejo de configuración.
    Patrón Singleton thread-safe con carga perezosa.
    """
    
    _instance: Optional["BrainConfig"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicialización perezosa."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._loader = YamlConfigLoader(Path(__file__).parent)
                    self._resolver = EnvironmentResolver()
                    self._validator = ConfigValidatorImpl()
                    self._cache = ConfigCache()
                    self._raw_config: Optional[Dict[str, Any]] = None
                    self._initialized = True
    
    # -------------------------
    # PROPIEDADES CACHED
    # -------------------------
    
    @cached_property
    def _config(self) -> Dict[str, Any]:
        """Carga y resuelve configuración (cached)."""
        if self._raw_config is None:
            raw = self._loader.load()
            resolved = self._resolver.resolve(raw)
            self._validator.validate(resolved)
            self._raw_config = resolved
        return self._raw_config
    
    # -------------------------
    # API PÚBLICA
    # -------------------------
    
    @lru_cache(maxsize=128)
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene valor de configuración con dot notation.
        Usa caché LRU para accesos frecuentes.
        
        Args:
            key: Clave en formato dot notation (ej: "api.rest.port")
            default: Valor por defecto si no se encuentra
        
        Returns:
            Valor de configuración o default
        """
        value = self._config
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return default
            if value is None:
                return default
        return value
    
    def get_model(self, model_type: Type[T], key: str) -> T:
        """
        Obtiene configuración como modelo Pydantic validado.
        
        Args:
            model_type: Tipo de modelo Pydantic
            key: Clave de configuración
        
        Returns:
            Instancia del modelo validada
        """
        config_dict = self.get(key, {})
        return self._cache.get(model_type, config_dict)
    
    @property
    def system(self) -> SystemConfig:
        """Configuración del sistema (cached)."""
        return self.get_model(SystemConfig, "system")
    
    @property
    def databases(self) -> Dict[str, DatabaseConfig]:
        """Configuraciones de bases de datos (cached)."""
        raw_dbs = self.get("databases", {})
        return {
            name: self._cache.get(DatabaseConfig, config)
            for name, config in raw_dbs.items()
            if isinstance(config, dict)
        }
    
    @property
    def api(self) -> APIConfig:
        """Configuración de API (cached)."""
        return self.get_model(APIConfig, "api.rest")
    
    def reload(self) -> None:
        """Recarga la configuración desde disco."""
        with self._lock:
            self._raw_config = None
            self._cache.clear()
            
            # Limpiar cachés
            try:
                del self._config
            except AttributeError:
                pass
            
            # Recargar
            _ = self._config  # Esto disparará la recarga
    
    def export(self, format: str = "yaml") -> str:
        """Exporta configuración en formato especificado."""
        if format == "yaml":
            return yaml.dump(self._config, sort_keys=False, default_flow_style=False)
        elif format == "json":
            import json
            return json.dumps(self._config, indent=2, default=str)
        else:
            raise ValueError(f"Formato no soportado: {format}")

# =========================
# INSTANCIA GLOBAL Y HELPERS
# =========================

_config_instance: Optional[BrainConfig] = None

def get_config() -> BrainConfig:
    """Obtiene instancia singleton de configuración."""
    global _config_instance
    if _config_instance is None:
        _config_instance = BrainConfig()
    return _config_instance

@lru_cache(maxsize=1)
def get_system_config() -> SystemConfig:
    """Obtiene configuración del sistema (cached global)."""
    return get_config().system

@lru_cache(maxsize=32)
def get_database_config(db_name: str) -> Optional[DatabaseConfig]:
    """Obtiene configuración de base de datos específica (cached)."""
    return get_config().databases.get(db_name)

@lru_cache(maxsize=1)
def get_api_config() -> APIConfig:
    """Obtiene configuración de API (cached global)."""
    return get_config().api

# =========================
# INICIALIZACIÓN DIFERIDA
# =========================

def init_config(config_dir: Optional[Path] = None) -> BrainConfig:
    """
    Inicializa configuración con directorio personalizado.
    Útil para testing o scripts.
    """
    global _config_instance
    
    with BrainConfig._lock:
        if _config_instance is not None:
            logger.warning("Configuración ya inicializada, ignorando init_config")
            return _config_instance
        
        # Reset singleton
        BrainConfig._instance = None
        _config_instance = BrainConfig()
        
        if config_dir is not None:
            # Sobreescribir loader con directorio personalizado
            _config_instance._loader = YamlConfigLoader(config_dir)
            _config_instance.reload()
        
        return _config_instance

# =========================
# MÓDULO
# =========================

__all__ = [
    # Instancia principal
    "get_config",
    "init_config",
    
    # Funciones helper
    "get_system_config", 
    "get_database_config",
    "get_api_config",
    
    # Excepciones
    "ConfigError",
    "ConfigLoadError",
    "ConfigValidationError",
    "ConfigResolveError",
    
    # Modelos
    "SystemConfig",
    "DatabaseConfig", 
    "APIConfig",
    
    # Componentes (para testing/mocking)
    "YamlConfigLoader",
    "EnvironmentResolver",
    "ConfigValidatorImpl",
    "ConfigCache",
]