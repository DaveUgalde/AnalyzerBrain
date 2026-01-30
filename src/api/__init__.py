"""
Módulo API - Interfaces de comunicación externa.
Proporciona REST API, WebSocket, gRPC, CLI e interfaces web.
"""

__version__ = "1.0.0"
__all__ = [
    # Módulos principales
    "server",
    "rest_api",
    "websocket_api",
    "grpc_api",
    "cli_interface",
    "web_ui",
    
    # Utilidades
    "authentication",
    "rate_limiter",
    "request_validator",
    
    # Clases principales
    "APIServer",
    "RESTAPI",
    "WebSocketAPI",
    "GRPCAPI",
    "CLIInterface",
    "WebUI",
    "Authentication",
    "RateLimiter",
    "RequestValidator",
]

# Inicialización del módulo API
from .server import APIServer
from .rest_api import RESTAPI
from .websocket_api import WebSocketAPI
from .grpc_api import GRPCAPI
from .cli_interface import CLIInterface
from .web_ui import WebUI
from .authentication import Authentication
from .rate_limiter import RateLimiter
from .request_validator import RequestValidator

# Configuración global
API_CONFIG = {
    "version": "1.0.0",
    "protocols": ["rest", "websocket", "grpc", "cli", "web"],
    "authentication_required": False,
    "rate_limiting_enabled": True,
    "cors_enabled": True,
}

def initialize_api(config: dict = None):
    """
    Inicializa todos los componentes del módulo API.
    
    Args:
        config: Configuración personalizada para la API
        
    Returns:
        APIServer: Servidor principal de la API
    """
    from .server import APIServer
    return APIServer(config)