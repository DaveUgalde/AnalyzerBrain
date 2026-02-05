

1. Registro basico

from src.core.dependency_injector import register, resolve

# Registrar un singleton
register(IService, ConcreteService)

# Registrar con factory
register(IService, lambda: ConcreteService(config="custom"))

# Registrar instancia
service = ConcreteService()
register(IService, service)


2. Uso con decoradores

from src.core.dependency_injector import inject, singleton

@singleton
class DatabaseService:
    def __init__(self, connection_string: str):
        self.conn = create_connection(connection_string)

@inject
class UserRepository:
    def __init__(self, db: DatabaseService):
        self.db = db


3. Resolucion:

from src.core.dependency_injector import resolve

# Resolver dependencia
service = resolve(IService)
repo = resolve(UserRepository)


4. Ambitos:

from src.core.dependency_injector import create_scope

with create_scope("request_scope") as scope:
    # Todas las dependencias scoped en este bloque comparten instancia
    service1 = resolve(IService)
    service2 = resolve(IService)
    assert service1 is service2  # True dentro del ámbito

5. Integracion con el orchestrator:

# En src/core/orchestrator.py
from .dependency_injector import inject

@inject
class BrainOrchestrator:
    def __init__(
        self,
        config_manager: ConfigManager,
        event_bus: EventBus,
        system_state: SystemState,
        health_checker: HealthChecker
    ):
        self.config = config_manager
        self.event_bus = event_bus
        self.state = system_state
        self.health = health_checker
        
    # Las dependencias se inyectan automáticamente