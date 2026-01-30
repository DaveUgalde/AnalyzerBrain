"""
DependencyInjector - Sistema de inyección de dependencias.
"""

from typing import Dict, Any, Type, Optional, Callable
from .exceptions import BrainException

class DependencyInjector:
    """Inyector de dependencias."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._scoped_services: Dict[str, Dict[str, Any]] = {}
    
    def register_service(self, name: str, service: Any, 
                        factory: bool = False) -> None:
        """
        Registra un servicio.
        
        Args:
            name: Nombre del servicio
            service: Instancia del servicio o clase/fábrica
            factory: Si True, service es una fábrica (callable)
        """
        if name in self._services:
            raise BrainException(f"Service {name} already registered")
        
        if factory:
            self._factories[name] = service
        else:
            self._services[name] = service
    
    def get_service(self, name: str, scope: Optional[str] = None) -> Any:
        """
        Obtiene un servicio.
        
        Args:
            name: Nombre del servicio
            scope: Ámbito de la dependencia (opcional)
            
        Returns:
            Instancia del servicio
        """
        # Buscar en servicios con ámbito
        if scope is not None:
            if scope in self._scoped_services and name in self._scoped_services[scope]:
                return self._scoped_services[scope][name]
        
        # Buscar en servicios globales
        if name in self._services:
            return self._services[name]
        
        # Buscar en fábricas
        if name in self._factories:
            factory = self._factories[name]
            service = factory()
            
            # Cachear si hay ámbito
            if scope is not None:
                if scope not in self._scoped_services:
                    self._scoped_services[scope] = {}
                self._scoped_services[scope][name] = service
            
            return service
        
        raise BrainException(f"Service {name} not found")
    
    def create_scope(self, scope_id: str) -> 'DependencyScope':
        """
        Crea un nuevo ámbito de dependencias.
        
        Args:
            scope_id: Identificador del ámbito
            
        Returns:
            DependencyScope: Ámbito de dependencias
        """
        return DependencyScope(self, scope_id)
    
    def resolve_dependencies(self, cls: Type, scope: Optional[str] = None) -> Any:
        """
        Resuelve dependencias de una clase e instancia.
        
        Args:
            cls: Clase a instanciar
            scope: Ámbito (opcional)
            
        Returns:
            Instancia de la clase con dependencias inyectadas
        """
        # Obtener signature del constructor
        import inspect
        
        signature = inspect.signature(cls.__init__)
        parameters = {}
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
            
            # Buscar servicio por nombre del parámetro
            try:
                parameters[param_name] = self.get_service(param_name, scope)
            except BrainException:
                # Si no hay servicio registrado, usar valor por defecto
                if param.default != inspect.Parameter.empty:
                    parameters[param_name] = param.default
                else:
                    raise BrainException(f"Cannot resolve dependency {param_name} for {cls.__name__}")
        
        return cls(**parameters)
    
    def clear_services(self, scope: Optional[str] = None) -> None:
        """
        Limpia servicios registrados.
        
        Args:
            scope: Si se especifica, solo limpia servicios del ámbito
        """
        if scope is None:
            self._services.clear()
            self._factories.clear()
            self._scoped_services.clear()
        elif scope in self._scoped_services:
            del self._scoped_services[scope]
    
    def validate_dependencies(self, required_services: list) -> list:
        """
        Valida que las dependencias requeridas estén registradas.
        
        Args:
            required_services: Lista de nombres de servicios requeridos
            
        Returns:
            Lista de servicios faltantes
        """
        missing = []
        
        for service_name in required_services:
            if (service_name not in self._services and 
                service_name not in self._factories):
                missing.append(service_name)
        
        return missing

class DependencyScope:
    """Ámbito de dependencias."""
    
    def __init__(self, injector: DependencyInjector, scope_id: str):
        self._injector = injector
        self.scope_id = scope_id
    
    def get_service(self, name: str) -> Any:
        """Obtiene un servicio dentro del ámbito."""
        return self._injector.get_service(name, self.scope_id)
    
    def resolve(self, cls: Type) -> Any:
        """Resuelve dependencias de una clase dentro del ámbito."""
        return self._injector.resolve_dependencies(cls, self.scope_id)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Limpiar servicios del ámbito al salir
        self._injector.clear_services(self.scope_id)