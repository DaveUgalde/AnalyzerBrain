"""
DependencyInjector - Sistema de inyección de dependencias.
"""

from __future__ import annotations

import inspect
from typing import Dict, Any, Type, Optional, Callable

from .exceptions import BrainException


class DependencyInjector:
    """Inyector de dependencias."""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._scoped_services: Dict[str, Dict[str, Any]] = {}

    def register_service(
        self,
        name: str,
        service: Any,
        factory: bool = False
    ) -> None:
        """
        Registra un servicio.
        """
        if name in self._services or name in self._factories:
            raise BrainException(f"Service {name} already registered")

        if factory:
            if not callable(service):
                raise BrainException(
                    f"Factory for service {name} must be callable"
                )
            self._factories[name] = service
        else:
            self._services[name] = service

    def get_service(
        self,
        name: str,
        scope: Optional[str] = None
    ) -> Any:
        """
        Obtiene un servicio.
        """
        # 1. Servicios con ámbito
        if scope is not None:
            scoped = self._scoped_services.get(scope)
            if scoped and name in scoped:
                return scoped[name]

        # 2. Servicios singleton/globales
        if name in self._services:
            return self._services[name]

        # 3. Fábricas
        if name in self._factories:
            service = self._factories[name]()

            if scope is not None:
                self._scoped_services.setdefault(scope, {})[name] = service

            return service

        raise BrainException(f"Service {name} not found")

    def create_scope(self, scope_id: str) -> DependencyScope:
        """
        Crea un nuevo ámbito de dependencias.
        """
        return DependencyScope(self, scope_id)

    def resolve_dependencies(
        self,
        cls: Type,
        scope: Optional[str] = None
    ) -> Any:
        """
        Resuelve dependencias de una clase e instancia.
        """
        try:
            signature = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            raise BrainException(
                f"Cannot inspect constructor for {cls}"
            )

        parameters: Dict[str, Any] = {}

        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            try:
                parameters[param_name] = self.get_service(
                    param_name,
                    scope
                )
            except BrainException:
                if param.default is not inspect.Parameter.empty:
                    parameters[param_name] = param.default
                else:
                    raise BrainException(
                        f"Cannot resolve dependency "
                        f"{param_name} for {cls.__name__}"
                    )

        return cls(**parameters)

    def clear_services(self, scope: Optional[str] = None) -> None:
        """
        Limpia servicios registrados.
        """
        if scope is None:
            self._services.clear()
            self._factories.clear()
            self._scoped_services.clear()
        else:
            self._scoped_services.pop(scope, None)

    def validate_dependencies(self, required_services: list) -> list:
        """
        Valida que las dependencias requeridas estén registradas.
        """
        missing = []

        for name in required_services:
            if name not in self._services and name not in self._factories:
                missing.append(name)

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

    def __enter__(self) -> DependencyScope:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._injector.clear_services(self.scope_id)
