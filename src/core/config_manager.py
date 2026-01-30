"""
ConfigManager - Gestión centralizada de configuración del sistema.
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, ValidationError
from .exceptions import BrainException, ConfigurationError

class ConfigManager:
    """Gestor de configuración."""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
        self.watchers: Dict[str, list] = {}
    
    def load_config(self, config_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Carga una configuración desde archivo.
        
        Args:
            config_name: Nombre de la configuración
            config_path: Ruta al archivo (opcional)
            
        Returns:
            Dict con la configuración cargada
        """
        if config_path is None:
            config_path = self.config_dir / f"{config_name}.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config format: {config_path.suffix}")
            
            # Validar estructura básica
            if not isinstance(config, dict):
                raise ConfigurationError("Config must be a dictionary")
            
            self.configs[config_name] = config
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML error in {config_path}: {e}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"JSON error in {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading config {config_name}: {e}")
    
    def get_config(self, config_name: str, default: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Obtiene una configuración cargada.
        
        Args:
            config_name: Nombre de la configuración
            default: Valor por defecto si no existe
            
        Returns:
            Dict con la configuración
        """
        if config_name not in self.configs:
            if default is not None:
                return default
            raise ConfigurationError(f"Config {config_name} not loaded")
        
        return self.configs[config_name]
    
    def set_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """
        Establece una configuración.
        
        Args:
            config_name: Nombre de la configuración
            config: Configuración a establecer
        """
        self.configs[config_name] = config
        
        # Notificar watchers
        if config_name in self.watchers:
            for callback in self.watchers[config_name]:
                try:
                    callback(config_name, config)
                except Exception as e:
                    print(f"Error in config watcher: {e}")
    
    def reload_config(self, config_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Recarga una configuración desde archivo.
        
        Args:
            config_name: Nombre de la configuración
            config_path: Ruta al archivo (opcional)
            
        Returns:
            Dict con la configuración recargada
        """
        return self.load_config(config_name, config_path)
    
    def validate_config(self, config_name: str, schema: BaseModel) -> bool:
        """
        Valida una configuración contra un esquema Pydantic.
        
        Args:
            config_name: Nombre de la configuración
            schema: Esquema Pydantic para validación
            
        Returns:
            bool: True si la configuración es válida
        """
        if config_name not in self.configs:
            raise ConfigurationError(f"Config {config_name} not loaded")
        
        try:
            schema(**self.configs[config_name])
            return True
        except ValidationError as e:
            raise ConfigurationError(f"Config validation failed: {e}")
    
    def export_config(self, config_name: str, format: str = "yaml", 
                     filepath: Optional[str] = None) -> str:
        """
        Exporta una configuración a string o archivo.
        
        Args:
            config_name: Nombre de la configuración
            format: Formato de exportación (yaml, json)
            filepath: Ruta para guardar (opcional)
            
        Returns:
            str: Configuración serializada
        """
        if config_name not in self.configs:
            raise ConfigurationError(f"Config {config_name} not loaded")
        
        config = self.configs[config_name]
        
        if format == "yaml":
            output = yaml.dump(config, default_flow_style=False)
        elif format == "json":
            output = json.dumps(config, indent=2)
        else:
            raise ConfigurationError(f"Unsupported export format: {format}")
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(output)
        
        return output
    
    def watch_config_changes(self, config_name: str, callback: callable) -> None:
        """
        Registra un watcher para cambios en configuración.
        
        Args:
            config_name: Nombre de la configuración
            callback: Función a llamar cuando cambie la configuración
        """
        if config_name not in self.watchers:
            self.watchers[config_name] = []
        
        if callback not in self.watchers[config_name]:
            self.watchers[config_name].append(callback)
    
    def unwatch_config_changes(self, config_name: str, callback: callable) -> None:
        """
        Elimina un watcher de cambios en configuración.
        
        Args:
            config_name: Nombre de la configuración
            callback: Función a eliminar
        """
        if config_name in self.watchers and callback in self.watchers[config_name]:
            self.watchers[config_name].remove(callback)