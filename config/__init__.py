"""
Configuración centralizada del sistema Project Brain.

Este módulo proporciona acceso unificado a toda la configuración del sistema,
cargando desde archivos YAML, variables de entorno, y proporcionando validación.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Excepción para errores de configuración."""
    pass


# =========================
# MODELOS DE CONFIGURACIÓN
# =========================

class SystemConfig(BaseModel):
    name: str = Field(default="Project Brain")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")  # development, staging, production
    log_level: str = Field(default="INFO")
    debug_mode: bool = Field(default=False)
    data_directory: str = Field(default="./data")
    log_directory: str = Field(default="./logs")

    model_config = {"extra": "allow"}


class DatabaseConfig(BaseModel):
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
    def resolve_password(cls, v):
        """Resuelve contraseñas desde variables de entorno."""
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            return os.getenv(v[2:-1], "")
        return v


class EmbeddingModelConfig(BaseModel):
    default: str = Field(default="all-MiniLM-L6-v2")
    alternatives: List[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_processing_time: int = Field(default=30, ge=1)
    capabilities: List[str] = Field(default_factory=list)


class APIConfig(BaseModel):
    enabled: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    cors_origins: List[str] = Field(default_factory=list)


# =========================
# CLASE PRINCIPAL
# =========================

class BrainConfig:
    """Clase principal de configuración del sistema (Singleton)."""

    _instance: Optional["BrainConfig"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = {}
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_config()
            self._initialized = True

    # -------------------------
    # CARGA DE CONFIGURACIÓN
    # -------------------------

    def _load_config(self) -> None:
        try:
            config_dir = Path(__file__).parent

            self._config["system"] = self._load_yaml(config_dir / "system.yaml")
            self._config["models"] = self._load_yaml(config_dir / "models.yaml")
            self._config["agents"] = self._load_yaml(config_dir / "agents.yaml")
            self._config["databases"] = self._load_yaml(config_dir / "databases.yaml")
            self._config["api"] = self._load_yaml(config_dir / "api.yaml")

            self._override_with_env_vars()
            self._validate_config()

            logger.info("Configuración cargada exitosamente")

        except Exception as e:
            logger.exception("Error cargando configuración")
            raise ConfigError(f"Error cargando configuración: {e}") from e

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            logger.warning(f"{path.name} no encontrado")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # -------------------------
    # VARIABLES DE ENTORNO
    # -------------------------

    def _override_with_env_vars(self) -> None:
        system = self._config.setdefault("system", {})
        databases = self._config.setdefault("databases", {})
        api = self._config.setdefault("api", {})

        if env := os.getenv("BRAIN_ENVIRONMENT"):
            system["environment"] = env

        if log_level := os.getenv("BRAIN_LOG_LEVEL"):
            system["log_level"] = log_level

        postgres = databases.setdefault("postgresql", {})
        if host := os.getenv("POSTGRES_HOST"):
            postgres["host"] = host
        if password := os.getenv("POSTGRES_PASSWORD"):
            postgres["password"] = password

        api_rest = api.setdefault("rest", {})
        if port := os.getenv("API_PORT"):
            api_rest["port"] = int(port)

    # -------------------------
    # VALIDACIÓN
    # -------------------------

    def _validate_config(self) -> None:
        system = self._config.get("system")
        if not system:
            raise ConfigError("Configuración del sistema no encontrada")

        data_dir = system.get("data_directory", "./data")
        try:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ConfigError(f"No se puede crear el directorio {data_dir}: {e}") from e

    # -------------------------
    # API PÚBLICA
    # -------------------------

    def get(self, key: str, default: Any = None) -> Any:
        value = self._config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        config = self._config
        parts = key.split(".")
        for part in parts[:-1]:
            config = config.setdefault(part, {})
        config[parts[-1]] = value

    def reload(self) -> None:
        self._config.clear()
        self._initialized = False
        self._load_config()
        self._initialized = True

    def export(self, format: str = "yaml") -> str:
        import json
        if format == "yaml":
            return yaml.dump(self._config, sort_keys=False)
        if format == "json":
            return json.dumps(self._config, indent=2)
        raise ValueError(f"Formato no soportado: {format}")

    # -------------------------
    # PROPIEDADES VALIDADAS
    # -------------------------

    @property
    def system(self) -> SystemConfig:
        return SystemConfig(**self.get("system", {}))

    @property
    def databases(self) -> Dict[str, DatabaseConfig]:
        raw = self.get("databases", {})
        return {
            name: DatabaseConfig(**cfg)
            for name, cfg in raw.items()
            if isinstance(cfg, dict)
        }

    @property
    def api(self) -> APIConfig:
        return APIConfig(**self.get("api.rest", {}))


# =========================
# INSTANCIA GLOBAL
# =========================

config = BrainConfig()


# =========================
# FUNCIONES DE CONVENIENCIA
# =========================

def get_config() -> BrainConfig:
    return config


def get_system_config() -> SystemConfig:
    return config.system


def get_database_config(db_name: str) -> Optional[DatabaseConfig]:
    return config.databases.get(db_name)


def get_api_config() -> APIConfig:
    return config.api


__all__ = [
    "config",
    "get_config",
    "get_system_config",
    "get_database_config",
    "get_api_config",
    "ConfigError",
    "SystemConfig",
    "DatabaseConfig",
    "APIConfig",
]
