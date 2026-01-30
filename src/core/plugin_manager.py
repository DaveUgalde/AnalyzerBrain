"""
PluginManager - Gestión de plugins del sistema.
"""

import importlib
import inspect
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
from .exceptions import BrainException, PluginError

class PluginInfo:
    """Información de un plugin."""
    
    def __init__(self, name: str, module_path: str, enabled: bool = True):
        self.name = name
        self.module_path = module_path
        self.enabled = enabled
        self.loaded = False
        self.instance: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}

class PluginManager:
    """Gestor de plugins."""
    
    def __init__(self, plugins_directory: str = "./plugins"):
        self.plugins_directory = Path(plugins_directory)
        self.plugins: Dict[str, PluginInfo] = {}
    
    def load_plugin(self, name: str, module_path: str) -> PluginInfo:
        """
        Carga un plugin.
        
        Args:
            name: Nombre del plugin
            module_path: Ruta al módulo (ej: "plugins.my_plugin")
            
        Returns:
            PluginInfo: Información del plugin cargado
        """
        if name in self.plugins:
            raise PluginError(f"Plugin {name} already loaded")
        
        try:
            # Importar módulo
            module = importlib.import_module(module_path)
            
            # Buscar clase principal del plugin (que tenga atributo PLUGIN_NAME)
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if (inspect.isclass(attr) and 
                    hasattr(attr, 'PLUGIN_NAME') and 
                    attr.PLUGIN_NAME == name):
                    plugin_class = attr
                    break
            
            if plugin_class is None:
                raise PluginError(f"Plugin class not found in {module_path}")
            
            # Instanciar plugin
            plugin_instance = plugin_class()
            
            # Crear información del plugin
            plugin_info = PluginInfo(name, module_path)
            plugin_info.loaded = True
            plugin_info.instance = plugin_instance
            plugin_info.metadata = getattr(plugin_instance, 'metadata', {})
            
            self.plugins[name] = plugin_info
            
            return plugin_info
            
        except ImportError as e:
            raise PluginError(f"Failed to import plugin {name}: {e}")
        except Exception as e:
            raise PluginError(f"Failed to load plugin {name}: {e}")
    
    def unload_plugin(self, name: str) -> bool:
        """
        Descarga un plugin.
        
        Args:
            name: Nombre del plugin
            
        Returns:
            bool: True si se descargó exitosamente
        """
        if name not in self.plugins:
            return False
        
        plugin_info = self.plugins[name]
        
        try:
            # Llamar método de limpieza si existe
            if (plugin_info.instance and 
                hasattr(plugin_info.instance, 'cleanup')):
                plugin_info.instance.cleanup()
            
            # Eliminar de la lista
            del self.plugins[name]
            
            # TODO: Eliminar módulo de sys.modules
            # Esto es complejo y puede no ser necesario
            
            return True
            
        except Exception as e:
            raise PluginError(f"Failed to unload plugin {name}: {e}")
    
    def enable_plugin(self, name: str) -> bool:
        """
        Habilita un plugin.
        
        Args:
            name: Nombre del plugin
            
        Returns:
            bool: True si se habilitó exitosamente
        """
        if name not in self.plugins:
            return False
        
        self.plugins[name].enabled = True
        return True
    
    def disable_plugin(self, name: str) -> bool:
        """
        Deshabilita un plugin.
        
        Args:
            name: Nombre del plugin
            
        Returns:
            bool: True si se deshabilitó exitosamente
        """
        if name not in self.plugins:
            return False
        
        self.plugins[name].enabled = False
        return True
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        Lista todos los plugins.
        
        Returns:
            Lista de información de plugins
        """
        plugins_list = []
        
        for name, plugin_info in self.plugins.items():
            plugins_list.append({
                "name": name,
                "module_path": plugin_info.module_path,
                "enabled": plugin_info.enabled,
                "loaded": plugin_info.loaded,
                "metadata": plugin_info.metadata
            })
        
        return plugins_list
    
    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un plugin específico.
        
        Args:
            name: Nombre del plugin
            
        Returns:
            Dict con información del plugin o None si no existe
        """
        if name not in self.plugins:
            return None
        
        plugin_info = self.plugins[name]
        
        return {
            "name": name,
            "module_path": plugin_info.module_path,
            "enabled": plugin_info.enabled,
            "loaded": plugin_info.loaded,
            "metadata": plugin_info.metadata,
            "instance": plugin_info.instance
        }
    
    def validate_plugin(self, name: str) -> List[str]:
        """
        Valida un plugin.
        
        Args:
            name: Nombre del plugin
            
        Returns:
            Lista de errores de validación
        """
        errors = []
        
        if name not in self.plugins:
            errors.append(f"Plugin {name} not found")
            return errors
        
        plugin_info = self.plugins[name]
        
        if not plugin_info.loaded:
            errors.append(f"Plugin {name} not loaded")
        
        if plugin_info.instance:
            # Validar interfaz mínima
            required_methods = ['initialize', 'process']
            
            for method in required_methods:
                if not hasattr(plugin_info.instance, method):
                    errors.append(f"Plugin {name} missing required method: {method}")
        
        return errors
    
    def scan_plugins_directory(self) -> List[str]:
        """
        Escanea el directorio de plugins en busca de nuevos plugins.
        
        Returns:
            Lista de nombres de plugins encontrados
        """
        if not self.plugins_directory.exists():
            return []
        
        plugins_found = []
        
        for item in self.plugins_directory.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                # Es un paquete Python
                plugins_found.append(item.name)
            elif item.suffix == '.py' and item.name != "__init__.py":
                # Es un módulo Python
                plugins_found.append(item.stem)
        
        return plugins_found
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """
        Carga todos los plugins del directorio.
        
        Returns:
            Dict con resultado de carga por plugin
        """
        results = {}
        
        for plugin_name in self.scan_plugins_directory():
            try:
                module_path = f"plugins.{plugin_name}"
                self.load_plugin(plugin_name, module_path)
                results[plugin_name] = True
            except Exception as e:
                results[plugin_name] = False
                print(f"Failed to load plugin {plugin_name}: {e}")
        
        return results