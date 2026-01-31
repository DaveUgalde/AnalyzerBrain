"""
ConfigManager - Gestión centralizada de configuración del sistema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

import yaml
from pydantic import BaseModel, ValidationError

from .exceptions import ConfigurationError


class ConfigManager:
    """Gestor de configuración."""

    def __init__(self, config_dir: str = "./config"):
        self.config_dir: Path = Path(config_dir)
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.watchers: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = {}

    def load_config(
        self,
        config_name: str,
        config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Carga una configuración desde archivo.
        """
        path = (
            Path(config_path)
            if config_path is not None
            else self.config_dir / f"{config_name}.yaml"
        )

        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                if path.suffix in (".yaml", ".yml"):
                    config = yaml.safe_load(f)
                elif path.suffix == ".json":
                    config = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config format: {path.suffix}"
                    )

            if config is None:
                config = {}

            if not isinstance(config, dict):
                raise ConfigurationError("Config must be a dictionary")

            self.configs[config_name] = config
            return config

        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML error in {path}: {e}") from e
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"JSON error in {path}: {e}") from e
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Error loading config {config_name}: {e}"
            ) from e

    def get_config(
        self,
        config_name: str,
        default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Obtiene una configuración cargada.
        """
        if config_name not in self.configs:
            if default is not None:
                return default
            raise ConfigurationError(f"Config {config_name} not loaded")

        return self.configs[config_name]

    def set_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """
        Establece una configuración y notifica watchers.
        """
        self.configs[config_name] = config

        for callback in self.watchers.get(config_name, []):
            try:
                callback(config_name, config)
            except Exception:
                # Se ignoran errores de watchers para no romper el flujo
                pass

    def reload_config(
        self,
        config_name: str,
        config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recarga una configuración desde archivo.
        """
        return self.load_config(config_name, config_path)

    def validate_config(self, config_name: str, schema: BaseModel) -> bool:
        """
        Valida una configuración contra un esquema Pydantic.
        """
        if config_name not in self.configs:
            raise ConfigurationError(f"Config {config_name} not loaded")

        try:
            schema(**self.configs[config_name])
            return True
        except ValidationError as e:
            raise ConfigurationError(f"Config validation failed: {e}") from e

    def export_config(
        self,
        config_name: str,
        format: str = "yaml",
        filepath: Optional[str] = None
    ) -> str:
        """
        Exporta una configuración a string o archivo.
        """
        if config_name not in self.configs:
            raise ConfigurationError(f"Config {config_name} not loaded")

        config = self.configs[config_name]

        if format == "yaml":
            output = yaml.dump(
                config,
                default_flow_style=False,
                sort_keys=False
            )
        elif format == "json":
            output = json.dumps(config, indent=2)
        else:
            raise ConfigurationError(
                f"Unsupported export format: {format}"
            )

        if filepath:
            Path(filepath).write_text(output, encoding="utf-8")

        return output

    def watch_config_changes(
        self,
        config_name: str,
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """
        Registra un watcher para cambios en configuración.
        """
        self.watchers.setdefault(config_name, [])

        if callback not in self.watchers[config_name]:
            self.watchers[config_name].append(callback)

    def unwatch_config_changes(
        self,
        config_name: str,
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """
        Elimina un watcher de cambios en configuración.
        """
        if config_name in self.watchers:
            try:
                self.watchers[config_name].remove(callback)
            except ValueError:
                pass
