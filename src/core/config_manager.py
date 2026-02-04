#!/usr/bin/env python3
"""
Gestor centralizado de configuración de ANALYZERBRAIN.

Este módulo define el sistema de configuración del proyecto utilizando
Pydantic y Pydantic Settings. Permite cargar, validar y unificar la
configuración del sistema desde múltiples fuentes, siguiendo el
siguiente orden de prioridad:

1. Variables de entorno (incluyendo archivo .env)
2. Archivos YAML de configuración (system_config.yaml, agent_config.yaml)
3. Valores por defecto definidos en los modelos

Características principales:
- Validación estricta mediante Pydantic
- Soporte para configuración anidada
- Carga automática de variables de entorno
- Creación automática de directorios requeridos
- Acceso centralizado mediante singleton

Fuentes de configuración soportadas:
- .env
- config/system_config.yaml
- config/agent_config.yaml

Dependencias:
- pyyaml
- python-dotenv
- loguru
- pydantic
- pydantic-settings

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, cast
import yaml
from loguru import logger
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
        extra="ignore",
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
    _settings: AnalyzerBrainSettings | None = None
    _custom_config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._settings is None:
            self._load_settings()

    # ------------------------------------------------------------------
    # MÉTODOS PÚBLICOS PARA TESTING (NUEVOS)
    # ------------------------------------------------------------------

    @classmethod
    def reset_for_tests(cls) -> None:
        """
        Resetea el singleton para tests.

        USO EXCLUSIVO PARA TESTS. No usar en producción.
        """
        cls._instance = None
        cls._settings = None
        cls._custom_config = {}

    def get_internal_state(self) -> Dict[str, Any]:
        """
        Retorna el estado interno para inspección en tests.

        Returns:
            Dict con: {'settings': ..., 'custom_config': ...}
        """
        return {'settings': self._settings, 'custom_config': self._custom_config}

    def update_settings_for_test(self, config_dict: Dict[str, Any]) -> None:
        """
        Actualiza configuración para tests.

        Args:
            config_dict: Diccionario con configuración a actualizar
        """
        if not self._settings:
            return

        for key, value in config_dict.items():
            if not hasattr(self._settings, key):
                self._custom_config[key] = value
                continue

            current_value = getattr(self._settings, key)

            if isinstance(current_value, BaseModel) and isinstance(value, dict):
                updated = current_value.model_copy(update=cast(Dict[str, Any], value))
                setattr(self._settings, key, updated)
            else:
                setattr(self._settings, key, value)

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS (EXISTENTES)
    # ------------------------------------------------------------------

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
                        yaml_config: Dict[str, Any] = yaml.safe_load(f) or {}
                        # Esto lanzará ValidationError si hay datos inválidos
                        self._update_settings(yaml_config)

            # 3. Crear directorios necesarios
            self._create_directories()

            logger.info(f"Configuración cargada para entorno: {self._settings.environment}")

        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            raise

    def _update_settings(self, config_dict: Dict[str, Any]) -> None:
        """Actualiza settings con configuración personalizada."""
        if not self._settings:
            return

        for key, value in config_dict.items():
            if not hasattr(self._settings, key):
                self._custom_config[key] = value
                continue

            current_value = getattr(self._settings, key)

            if isinstance(current_value, BaseModel) and isinstance(value, dict):
                # Usar model_validate para validación explícita
                model_type = type(current_value)
                # Obtener los valores actuales como dict
                current_dict = current_value.model_dump()
                # Actualizar con nuevos valores
                current_dict.update(value)
                # Crear nueva instancia validada
                updated = model_type(**current_dict)
                setattr(self._settings, key, updated)
            else:
                setattr(self._settings, key, value)

    def _create_directories(self) -> None:
        """Crea los directorios necesarios."""
        if not self._settings:
            return

        storage = self._settings.storage
        data_dir = Path(storage.data_dir)

        directories = [
            data_dir,
            Path(storage.cache_dir),
            Path(storage.log_dir),
            data_dir / "backups",
            data_dir / "embeddings",
            data_dir / "graph_exports",
            data_dir / "projects",
            data_dir / "state",
        ]

        errors = []
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directorio creado/verificado: {directory}")
            except Exception as e:
                error_msg = f"No se pudo crear el directorio {directory}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Continuamos intentando crear los demás directorios

        # Si hay errores y estamos en producción, lanzamos excepción
        if errors and self._settings.environment == "production":
            raise RuntimeError(
                f"Errores creando directorios: {', '.join(errors[:1])}"
            )  # Mostrar solo el primer error
        elif errors:
            # En desarrollo/testing, solo logueamos los errores
            logger.warning(f"{len(errors)} directorios no se pudieron crear")

    # ------------------------------------------------------------------
    # PROPIEDADES PÚBLICAS (EXISTENTES)
    # ------------------------------------------------------------------

    @property
    def settings(self) -> AnalyzerBrainSettings:
        """Obtiene la configuración completa."""
        if self._settings is None:
            self._load_settings()
        assert self._settings is not None
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
        """Obtiene un valor de configuración por clave, con soporte para custom_config."""
        # Primero buscar en _custom_config (para tests/overrides)
        if hasattr(self, '_custom_config') and self._custom_config:
            # Manejar claves anidadas en _custom_config
            if '.' in key:
                parts = key.split('.')
                current = self._custom_config
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        # No encontrado en _custom_config, romper y buscar en settings
                        break
                else:
                    # Si recorrió todas las partes exitosamente
                    return current
            elif key in self._custom_config:
                # Clave simple en _custom_config
                return self._custom_config[key]

        # Si no se encontró en _custom_config, buscar en settings
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
            return default

    def reload(self) -> None:
        """Recarga la configuración."""
        self._settings = None
        self._custom_config = {}
        self._load_settings()
        logger.info("Configuración recargada")


# Instancia global
config = ConfigManager()
