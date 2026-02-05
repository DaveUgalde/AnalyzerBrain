"""
Inyector de dependencias para ANALYZERBRAIN.

Dependencias Previas:
1. core.config_manager
2. core.exceptions
3. utils.validation

Propósito:
- Gestión centralizada de dependencias del sistema
- Inyección automática basada en anotaciones de tipo
- Soporte para singleton y transient lifecycle
- Integración con configuración del sistema

Autor: ANALYZERBRAIN Team
Fecha: 2026
Versión: 1.0.2 (corregida - manejo de forward references y scoped)
"""

import inspect
import sys
from typing import Any, Dict, Type, Callable, Optional, Union, List, get_type_hints
from dataclasses import dataclass, field
from functools import wraps
from loguru import logger

from core.config_manager import config
from core.exceptions import AnalyzerBrainError
from src.utils.validation import validate_type, validate_not_none


class DependencyInjectionError(AnalyzerBrainError):
    """Error en la inyección de dependencias."""
    error_code = "DEPENDENCY_INJECTION_ERROR"


@dataclass
class DependencyInfo:
    """Información de una dependencia registrada."""
    abstract_type: Type
    concrete_type: Type
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifecycle: str = "singleton"  # "singleton" | "transient" | "scoped"
    dependencies: List[Type] = field(default_factory=list)
    
    def __post_init__(self):
        """Validación post-inicialización."""
        validate_not_none(self.abstract_type, "abstract_type")
        validate_not_none(self.concrete_type, "concrete_type")
        
        if self.lifecycle not in {"singleton", "transient", "scoped"}:
            raise ValueError(f"Lifecycle inválido: {self.lifecycle}")


class DependencyInjector:
    """
    Contenedor de inyección de dependencias con soporte para:
    - Inyección por constructor
    - Inyección por propiedad
    - Lifecycle management (singleton/transient/scoped)
    - Decoradores de inyección
    """
    
    def __init__(self, auto_register: bool = True):
        """
        Inicializa el inyector de dependencias.
        
        Args:
            auto_register: Si True, auto-registra dependencias del core.
        """
        self._registry: Dict[Type, DependencyInfo] = {}
        self._singleton_instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None
        self._resolving_stack: List[Type] = []
        self._initialized = False
        
        if auto_register:
            self._auto_register_core()
    
    def _get_type_hints(self, obj, localns=None):
        """Obtiene type hints con resolución de forward references."""
        try:
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                # Para funciones y métodos
                return get_type_hints(
                    obj,
                    globalns=vars(sys.modules[obj.__module__]) if obj.__module__ in sys.modules else {},
                    localns=localns or {}
                )
            elif inspect.isclass(obj):
                # Para clases
                return get_type_hints(
                    obj.__init__,
                    globalns=vars(sys.modules[obj.__module__]) if obj.__module__ in sys.modules else {},
                    localns=localns or {}
                )
            else:
                return {}
        except (NameError, TypeError, AttributeError):
            return {}
    
    def _auto_register_core(self) -> None:
        """Auto-registro de dependencias del core."""
        try:
            from .config_manager import ConfigManager, config
            from .event_bus import EventBus
            from .system_state import SystemState
            
            self.register_singleton(ConfigManager, lambda di=None: config)
            self.register_singleton(EventBus, EventBus)
            
            self.register_singleton(
                SystemState,
                lambda di: SystemState(
                    config=di.resolve(ConfigManager),
                    event_bus=di.resolve(EventBus)
                )
            )
            
            try:
                from .health_check import HealthChecker
                self.register_singleton(
                    HealthChecker,
                    lambda di: HealthChecker(
                        config=di.resolve(ConfigManager),
                        system_state=di.resolve(SystemState)
                    )
                )
                logger.info("HealthChecker registrado")
            except ImportError:
                logger.warning("HealthChecker no encontrado, omitiendo registro")
            
            logger.info("Dependencias del core auto-registradas")
            
        except ImportError as e:
            logger.warning(f"No se pudieron auto-registrar todas las dependencias del core: {e}")
    
    def register(
        self,
        abstract_type: Type,
        concrete_type: Union[Type, Callable],
        lifecycle: str = "singleton"
    ) -> None:
        """
        Registra una dependencia en el contenedor.
        
        Args:
            abstract_type: Tipo abstracto o interfaz
            concrete_type: Tipo concreto o factory function
            lifecycle: "singleton", "transient", o "scoped"
            
        Raises:
            DependencyInjectionError: Si el registro falla
        """
        try:
            validate_type(abstract_type, Type, "abstract_type")
            
            if abstract_type in self._registry:
                logger.warning(f"Dependencia ya registrada: {abstract_type}")
                return
            
            if inspect.isclass(concrete_type):
                factory = self._create_factory(concrete_type)
            elif callable(concrete_type):
                factory = concrete_type
            else:
                raise TypeError(f"concrete_type debe ser clase o factory: {concrete_type}")
            
            dependencies = self._analyze_dependencies(concrete_type)
            
            self._registry[abstract_type] = DependencyInfo(
                abstract_type=abstract_type,
                concrete_type=(
                    concrete_type if inspect.isclass(concrete_type) 
                    else concrete_type.__annotations__.get('return', type(None))
                ),
                factory=factory,
                lifecycle=lifecycle,
                dependencies=dependencies
            )
            
            logger.debug(f"Dependencia registrada: {abstract_type} -> {concrete_type} ({lifecycle})")
            
        except Exception as e:
            raise DependencyInjectionError(
                f"Error registrando dependencia {abstract_type}: {e}",
                details={"abstract_type": str(abstract_type)}
            ) from e
    
    def register_singleton(
        self, 
        abstract_type: Type, 
        concrete_type: Union[Type, Callable]
    ) -> None:
        """Registra una dependencia como singleton."""
        self.register(abstract_type, concrete_type, "singleton")
    
    def register_transient(
        self, 
        abstract_type: Type, 
        concrete_type: Union[Type, Callable]
    ) -> None:
        """Registra una dependencia como transient."""
        self.register(abstract_type, concrete_type, "transient")
    
    def register_scoped(
        self, 
        abstract_type: Type, 
        concrete_type: Union[Type, Callable]
    ) -> None:
        """Registra una dependencia como scoped."""
        self.register(abstract_type, concrete_type, "scoped")
    
    def register_instance(self, abstract_type: Type, instance: Any) -> None:
        """
        Registra una instancia ya creada.
        
        Args:
            abstract_type: Tipo abstracto
            instance: Instancia concreta
        """
        validate_not_none(instance, "instance")
        
        if not isinstance(instance, abstract_type):
            raise TypeError(
                f"Instancia {type(instance)} no es compatible con {abstract_type}"
            )
        
        self._registry[abstract_type] = DependencyInfo(
            abstract_type=abstract_type,
            concrete_type=type(instance),
            factory=lambda di=None: instance,
            instance=instance,
            lifecycle="singleton",
            dependencies=[]
        )
        
        self._singleton_instances[abstract_type] = instance
        
        logger.debug(f"Instancia registrada: {abstract_type} -> {instance}")
    
    def _create_factory(self, concrete_type: Type) -> Callable:
        """
        Crea una factory function para un tipo concreto.
        
        Args:
            concrete_type: Tipo concreto a instanciar
            
        Returns:
            Factory function que retorna una instancia
        """
        def factory(di: 'DependencyInjector' = None) -> Any:
            injector = di or self
            
            if not hasattr(concrete_type, '__init__'):
                return concrete_type()
            
            # Usar el método mejorado para obtener type hints
            type_hints = self._get_type_hints(concrete_type.__init__, vars(concrete_type))
            
            signature = inspect.signature(concrete_type.__init__)
            parameters = signature.parameters
            
            kwargs = {}
            
            for param_name, param in parameters.items():
                if param_name == 'self':
                    continue
                
                # Respetar valores por defecto
                if param.default != inspect.Parameter.empty:
                    kwargs[param_name] = param.default
                    continue
                
                param_type = type_hints.get(param_name, param.annotation)
                
                if param_type == inspect.Parameter.empty:
                    logger.warning(
                        f"Parámetro sin tipo en {concrete_type}.{param_name}"
                    )
                    continue
                
                try:
                    kwargs[param_name] = injector.resolve(param_type)
                except DependencyInjectionError as e:
                    raise DependencyInjectionError(
                        f"No se puede resolver parámetro '{param_name}' "
                        f"de tipo {param_type} para {concrete_type}",
                        cause=e
                    )
            
            return concrete_type(**kwargs)
        
        return factory
    
    def _create_instance(self, concrete_type: Type) -> Any:
        """
        Crea una instancia de una clase concreta no registrada (auto-resolución).
        
        Args:
            concrete_type: Tipo concreto a instanciar
            
        Returns:
            Instancia del tipo
        """
        factory = self._create_factory(concrete_type)
        return factory(self)
    
    def _analyze_dependencies(self, concrete_type: Union[Type, Callable]) -> List[Type]:
        """
        Analiza las dependencias de un tipo o factory.
        
        Args:
            concrete_type: Tipo o factory a analizar
            
        Returns:
            Lista de tipos de dependencias
        """
        dependencies = []
        
        if inspect.isclass(concrete_type):
            if hasattr(concrete_type, '__init__'):
                type_hints = self._get_type_hints(concrete_type.__init__, vars(concrete_type))
                
                signature = inspect.signature(concrete_type.__init__)
                for param_name, param in signature.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    param_type = type_hints.get(param_name, param.annotation)
                    if param_type != inspect.Parameter.empty:
                        dependencies.append(param_type)
        
        elif callable(concrete_type):
            type_hints = self._get_type_hints(concrete_type, {})
            
            signature = inspect.signature(concrete_type)
            for param_name, param in signature.parameters.items():
                param_type = type_hints.get(param_name, param.annotation)
                if param_type != inspect.Parameter.empty:
                    dependencies.append(param_type)
        
        return dependencies
    
    def resolve(self, abstract_type: Type) -> Any:
        """
        Resuelve y retorna una instancia del tipo solicitado.
        
        Args:
            abstract_type: Tipo a resolver
            
        Returns:
            Instancia del tipo solicitado
            
        Raises:
            DependencyInjectionError: Si no se puede resolver la dependencia
        """
        # 1. Verificar ciclos de dependencia
        if abstract_type in self._resolving_stack:
            cycle_path = ' -> '.join(
                [t.__name__ for t in self._resolving_stack] + [abstract_type.__name__]
            )
            raise DependencyInjectionError(
                f"Ciclo de dependencia detectado: {cycle_path}"
            )
        
        self._resolving_stack.append(abstract_type)
        
        try:
            # 2. Verificar si está registrado
            if abstract_type not in self._registry:
                raise DependencyInjectionError(
                    f"Dependencia no registrada: {abstract_type}"
                )
            
            dep_info = self._registry[abstract_type]
            
            # 3. Verificar instancia existente según lifecycle
            if dep_info.lifecycle == "singleton":
                if abstract_type in self._singleton_instances:
                    return self._singleton_instances[abstract_type]
            
            elif dep_info.lifecycle == "scoped":
                if self._current_scope is None:
                    raise DependencyInjectionError(
                        "No hay ámbito activo para resolver una dependencia scoped"
                    )
                if (self._current_scope in self._scoped_instances and
                    abstract_type in self._scoped_instances[self._current_scope]):
                    return self._scoped_instances[self._current_scope][abstract_type]
            
            # 4. Crear nueva instancia
            instance = dep_info.factory(self)
            
            # 5. Almacenar según lifecycle
            if dep_info.lifecycle == "singleton":
                self._singleton_instances[abstract_type] = instance
            elif dep_info.lifecycle == "scoped":
                if self._current_scope is None:
                    raise DependencyInjectionError(
                        "No hay ámbito activo para almacenar una dependencia scoped"
                    )
                if self._current_scope not in self._scoped_instances:
                    self._scoped_instances[self._current_scope] = {}
                self._scoped_instances[self._current_scope][abstract_type] = instance
            
            # 6. Actualizar info de dependencia
            dep_info.instance = instance
            
            logger.debug(f"Dependencia resuelta: {abstract_type}")
            
            return instance
            
        except Exception as e:
            if not isinstance(e, DependencyInjectionError):
                raise DependencyInjectionError(
                    f"Error resolviendo {abstract_type}: {e}",
                    cause=e
                ) from e
            raise
            
        finally:
            # Siempre remover del stack
            if self._resolving_stack and self._resolving_stack[-1] == abstract_type:
                self._resolving_stack.pop()
    
    def create_scope(self, scope_name: str) -> 'DependencyScope':
        """
        Crea un nuevo ámbito de dependencias.
        
        Args:
            scope_name: Nombre del ámbito
            
        Returns:
            Context manager para el ámbito
        """
        return DependencyScope(self, scope_name)
    
    def inject(self, target: Union[Type, Callable]) -> Any:
        """
        Decorador para inyectar dependencias en clases o funciones.
        
        Args:
            target: Clase o función a decorar
            
        Returns:
            Clase o función decorada
        """
        if inspect.isclass(target):
            return self._inject_class(target)
        elif callable(target):
            return self._inject_function(target)
        else:
            raise TypeError(f"Target debe ser clase o función: {target}")
    
    def _inject_class(self, cls: Type) -> Type:
        """
        Inyecta dependencias en una clase.
        
        Args:
            cls: Clase a decorar
            
        Returns:
            Clase decorada
        """
        original_init = cls.__init__
        injector = self
        
        type_hints = self._get_type_hints(cls.__init__, vars(cls))
        
        @wraps(original_init)
        def new_init(self_instance, *args, **kwargs):
            signature = inspect.signature(original_init)
            parameters = signature.parameters
            
            # Obtener nombres de parámetros que ya están siendo proporcionados
            bound_args = signature.bind_partial(*args, **kwargs)
            provided_args = set(bound_args.arguments.keys())
            
            for param_name, param in parameters.items():
                if param_name in ['self', 'args', 'kwargs']:
                    continue
                
                # SOLO inyectar si:
                # 1. Tiene tipo anotado
                # 2. NO fue proporcionado por el usuario (no está en args/kwargs)
                # 3. NO tiene valor por defecto (si tiene, Python lo manejará)
                param_type = type_hints.get(param_name, param.annotation)
                if (param_type != inspect.Parameter.empty and 
                    param_name not in provided_args and
                    param.default == inspect.Parameter.empty):
                    
                    try:
                        kwargs[param_name] = injector.resolve(param_type)
                    except DependencyInjectionError:
                        # Si no se puede resolver, dejar que falle normalmente
                        pass
            
            original_init(self_instance, *args, **kwargs)
        
        cls.__init__ = new_init
        
        return cls
    
    def _inject_function(self, func: Callable) -> Callable:
        """
        Inyecta dependencias en una función.
        
        Args:
            func: Función a decorar
            
        Returns:
            Función decorada
        """
        injector = self
        
        type_hints = self._get_type_hints(func, {})
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            parameters = signature.parameters
            
            for param_name, param in parameters.items():
                param_type = type_hints.get(param_name, param.annotation)
                if (param_type != inspect.Parameter.empty and 
                    param_name not in kwargs and
                    param.default == inspect.Parameter.empty):
                    
                    try:
                        kwargs[param_name] = injector.resolve(param_type)
                    except DependencyInjectionError:
                        if param.default == inspect.Parameter.empty:
                            raise
            
            return func(*args, **kwargs)
        
        return wrapper
    
    def get_all_registered(self) -> Dict[Type, DependencyInfo]:
        """Obtiene todas las dependencias registradas."""
        return self._registry.copy()
    
    def clear_scope(self, scope_name: str) -> None:
        """Limpia completamente un ámbito."""
        if scope_name in self._scoped_instances:
            for dep_type in list(self._scoped_instances[scope_name].keys()):
                if dep_type in self._registry:
                    self._registry[dep_type].instance = None
            
            del self._scoped_instances[scope_name]
            logger.debug(f"Ámbito limpiado: {scope_name}")
    
    def clear_all(self) -> None:
        """Limpia todas las instancias (excepto singletons globales)."""
        self._scoped_instances.clear()
        self._resolving_stack.clear()
        
        for abstract_type, dep_info in self._registry.items():
            if (dep_info.lifecycle != "singleton" or 
                abstract_type not in self._singleton_instances):
                dep_info.instance = None
        
        logger.info("Todas las instancias no-singleton limpiadas")


class DependencyScope:
    """Context manager para ámbitos de dependencias."""
    
    def __init__(self, injector: DependencyInjector, scope_name: str):
        self.injector = injector
        self.scope_name = scope_name
        self.previous_scope = None
    
    def __enter__(self):
        """Entra en el ámbito."""
        self.previous_scope = self.injector._current_scope
        self.injector._current_scope = self.scope_name
        
        if self.scope_name not in self.injector._scoped_instances:
            self.injector._scoped_instances[self.scope_name] = {}
        
        logger.debug(f"Entrando en ámbito: {self.scope_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sale del ámbito, limpiando TODAS las instancias scoped."""
        self.injector._current_scope = self.previous_scope
        
        # Limpiar todas las instancias de este ámbito, independientemente de excepciones
        if self.scope_name in self.injector._scoped_instances:
            self.injector._scoped_instances[self.scope_name].clear()
            del self.injector._scoped_instances[self.scope_name]
            
            # Limpiar referencia en registry para dependencias scoped
            for dep_type, dep_info in self.injector._registry.items():
                if dep_info.lifecycle == "scoped":
                    dep_info.instance = None
        
        logger.debug(f"Saliendo de ámbito: {self.scope_name}")
        return False


# Instancia global del inyector de dependencias
_injector: Optional[DependencyInjector] = None


def get_injector() -> DependencyInjector:
    """
    Obtiene la instancia global del inyector.
    
    Returns:
        DependencyInjector: Inyector global
    """
    global _injector
    if _injector is None:
        _injector = DependencyInjector()
        logger.info("Inyector de dependencias global inicializado")
    return _injector


def resolve(abstract_type: Type) -> Any:
    """
    Resuelve una dependencia usando el inyector global.
    
    Args:
        abstract_type: Tipo a resolver
        
    Returns:
        Instancia del tipo
    """
    return get_injector().resolve(abstract_type)


def register(
    abstract_type: Type,
    concrete_type: Union[Type, Callable],
    lifecycle: str = "singleton"
) -> None:
    """
    Registra una dependencia en el inyector global.
    
    Args:
        abstract_type: Tipo abstracto
        concrete_type: Tipo concreto o factory
        lifecycle: Lifecycle de la dependencia
    """
    get_injector().register(abstract_type, concrete_type, lifecycle)


def inject(target: Union[Type, Callable]) -> Any:
    """
    Decorador para inyectar dependencias usando el inyector global.
    
    Args:
        target: Clase o función a decorar
        
    Returns:
        Clase o función decorada
    """
    return get_injector().inject(target)


def create_scope(scope_name: str) -> DependencyScope:
    """
    Crea un nuevo ámbito en el inyector global.
    
    Args:
        scope_name: Nombre del ámbito
        
    Returns:
        DependencyScope: Context manager del ámbito
    """
    return get_injector().create_scope(scope_name)


# Decoradores de conveniencia
def singleton(cls: Type) -> Type:
    """
    Decorador para registrar una clase como singleton.
    
    Args:
        cls: Clase a registrar
        
    Returns:
        Clase registrada
    """
    register(cls, cls, "singleton")
    return cls


def transient(cls: Type) -> Type:
    """
    Decorador para registrar una clase como transient.
    
    Args:
        cls: Clase a registrar
        
    Returns:
        Clase registrada
    """
    register(cls, cls, "transient")
    return cls


def scoped(cls: Type) -> Type:
    """
    Decorador para registrar una clase como scoped.
    
    Args:
        cls: Clase a registrar
        
    Returns:
        Clase registrada
    """
    register(cls, cls, "scoped")
    return cls