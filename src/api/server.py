"""
APIServer - Servidor principal de la API.
Coordina todos los protocolos de comunicación (REST, WebSocket, gRPC).
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import websockets
import grpc
from concurrent import futures
from contextlib import asynccontextmanager
from datetime import datetime
import json
from pathlib import Path

from ..core.exceptions import BrainException, ValidationError
from ..core.config_manager import ConfigManager
from ..core.health_check import HealthCheck
from ..utils.logging_config import setup_logging
from .rest_api import RESTAPI
from .websocket_api import WebSocketAPI
from .grpc_api import GRPCAPI
from .authentication import Authentication
from .rate_limiter import RateLimiter
from .request_validator import RequestValidator

logger = logging.getLogger(__name__)

class ServerState(Enum):
    """Estados del servidor."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"

@dataclass
class ServerConfig:
    """Configuración del servidor API."""
    host: str = "0.0.0.0"
    rest_port: int = 8000
    websocket_port: int = 8001
    grpc_port: int = 50051
    workers: int = 4
    max_connections: int = 1000
    timeout: int = 30
    enable_rest: bool = True
    enable_websocket: bool = True
    enable_grpc: bool = False
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_compression: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ServerConfig':
        """Crea configuración desde diccionario."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

class APIServer:
    """
    Servidor principal que coordina todos los protocolos de API.
    
    Responsabilidades:
    1. Iniciar/detener todos los servidores (REST, WebSocket, gRPC)
    2. Configurar middleware global
    3. Gestionar ciclo de vida de la aplicación
    4. Coordinar entre diferentes protocolos
    5. Proporcionar interfaces unificadas
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa el servidor API.
        
        Args:
            config: Configuración del servidor (opcional)
        """
        self.config = ServerConfig.from_dict(config or {})
        self.state = ServerState.STOPPED
        
        # Componentes principales
        self.rest_api: Optional[RESTAPI] = None
        self.websocket_api: Optional[WebSocketAPI] = None
        self.grpc_api: Optional[GRPCAPI] = None
        self.authentication: Optional[Authentication] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.request_validator: Optional[RequestValidator] = None
        
        # Servidores
        self.fastapi_app: Optional[FastAPI] = None
        self.uvicorn_server: Optional[uvicorn.Server] = None
        self.websocket_server: Optional[websockets.Server] = None
        self.grpc_server: Optional[grpc.Server] = None
        
        # Tasks asíncronos
        self._tasks: List[asyncio.Task] = []
        
        # Métricas
        self.metrics = {
            "start_time": None,
            "requests_processed": 0,
            "websocket_connections": 0,
            "grpc_calls": 0,
            "errors": 0,
        }
        
        # Configurar logging
        if self.config.enable_logging:
            setup_logging(level=self.config.log_level)
        
        logger.info("APIServer inicializado con configuración: %s", self.config)
    
    async def start_server(self) -> bool:
        """
        Inicia todos los servidores de API.
        
        Returns:
            bool: True si todos los servidores se iniciaron exitosamente
            
        Raises:
            BrainException: Si hay errores al iniciar los servidores
        """
        try:
            self.state = ServerState.STARTING
            self.metrics["start_time"] = datetime.now()
            
            logger.info("Iniciando servidores API...")
            
            # Inicializar componentes
            await self._initialize_components()
            
            # Iniciar servidores según configuración
            tasks = []
            
            if self.config.enable_rest:
                tasks.append(self._start_rest_server())
            
            if self.config.enable_websocket:
                tasks.append(self._start_websocket_server())
            
            if self.config.enable_grpc:
                tasks.append(self._start_grpc_server())
            
            # Esperar que todos inicien
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verificar resultados
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("Error al iniciar servidor %s: %s", 
                               ["REST", "WebSocket", "gRPC"][i], result)
                    raise BrainException(f"Failed to start server: {result}")
            
            self.state = ServerState.RUNNING
            logger.info("Todos los servidores API iniciados exitosamente")
            
            # Registrar handlers de señal para shutdown graceful
            self._register_signal_handlers()
            
            return True
            
        except Exception as e:
            self.state = ServerState.STOPPED
            logger.error("Error al iniciar servidores API: %s", e, exc_info=True)
            await self.stop_server()
            raise BrainException(f"Failed to start API server: {e}")
    
    async def stop_server(self) -> bool:
        """
        Detiene todos los servidores de manera controlada.
        
        Returns:
            bool: True si todos los servidores se detuvieron exitosamente
        """
        try:
            self.state = ServerState.STOPPING
            logger.info("Deteniendo servidores API...")
            
            # Cancelar tasks pendientes
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            # Detener servidores en orden inverso al inicio
            stop_tasks = []
            
            if self.grpc_server:
                stop_tasks.append(self._stop_grpc_server())
            
            if self.websocket_server:
                stop_tasks.append(self._stop_websocket_server())
            
            if self.uvicorn_server:
                stop_tasks.append(self._stop_rest_server())
            
            # Esperar que todos se detengan
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
            
            # Limpiar componentes
            await self._cleanup_components()
            
            self.state = ServerState.STOPPED
            logger.info("Todos los servidores API detenidos exitosamente")
            
            return True
            
        except Exception as e:
            logger.error("Error al detener servidores API: %s", e, exc_info=True)
            return False
    
    def configure_routes(self) -> None:
        """Configura todas las rutas de la API."""
        if not self.fastapi_app:
            raise BrainException("FastAPI app no inicializada")
        
        # Configurar rutas REST
        if self.rest_api:
            self.rest_api.register_endpoints(self.fastapi_app)
        
        # Configurar rutas WebSocket
        if self.websocket_api and self.fastapi_app:
            # WebSocket endpoints se configuran directamente
            pass
        
        # Configurar rutas de salud y métricas
        self._configure_system_routes()
        
        logger.info("Rutas API configuradas")
    
    def setup_middleware(self) -> None:
        """Configura middleware global."""
        if not self.fastapi_app:
            raise BrainException("FastAPI app no inicializada")
        
        # CORS middleware
        if self.config.enable_cors:
            self.fastapi_app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Trusted Host middleware
        self.fastapi_app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"] if self.config.enable_cors else ["localhost", "127.0.0.1"]
        )
        
        # Middleware de autenticación
        if self.authentication:
            @self.fastapi_app.middleware("http")
            async def auth_middleware(request: Request, call_next):
                return await self.authentication.authenticate_user(request, call_next)
        
        # Middleware de rate limiting
        if self.rate_limiter:
            @self.fastapi_app.middleware("http")
            async def rate_limit_middleware(request: Request, call_next):
                return await self.rate_limiter.check_rate_limit(request, call_next)
        
        # Middleware de validación
        if self.request_validator:
            @self.fastapi_app.middleware("http")
            async def validation_middleware(request: Request, call_next):
                return await self.request_validator.validate_request_structure(request, call_next)
        
        # Middleware de logging
        @self.fastapi_app.middleware("http")
        async def log_middleware(request: Request, call_next):
            start_time = datetime.now()
            
            # Procesar request
            response = await call_next(request)
            
            # Calcular tiempo de procesamiento
            process_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Loggear
            logger.info(
                "Request: %s %s - Status: %s - Time: %.2fms",
                request.method,
                request.url.path,
                response.status_code,
                process_time
            )
            
            # Actualizar métricas
            self.metrics["requests_processed"] += 1
            
            return response
        
        logger.info("Middleware configurado")
    
    async def handle_errors(self) -> None:
        """Configura handlers globales de errores."""
        if not self.fastapi_app:
            raise BrainException("FastAPI app no inicializada")
        
        @self.fastapi_app.exception_handler(BrainException)
        async def brain_exception_handler(request: Request, exc: BrainException):
            logger.error("BrainException: %s", exc, exc_info=True)
            self.metrics["errors"] += 1
            
            return Response(
                status_code=500,
                content=json.dumps({
                    "error": "Internal server error",
                    "message": str(exc),
                    "request_id": request.state.get("request_id", "unknown")
                }),
                media_type="application/json"
            )
        
        @self.fastapi_app.exception_handler(ValidationError)
        async def validation_error_handler(request: Request, exc: ValidationError):
            logger.warning("ValidationError: %s", exc)
            
            return Response(
                status_code=400,
                content=json.dumps({
                    "error": "Validation error",
                    "message": str(exc),
                    "request_id": request.state.get("request_id", "unknown")
                }),
                media_type="application/json"
            )
        
        @self.fastapi_app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception):
            logger.error("Unhandled exception: %s", exc, exc_info=True)
            self.metrics["errors"] += 1
            
            return Response(
                status_code=500,
                content=json.dumps({
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "request_id": request.state.get("request_id", "unknown")
                }),
                media_type="application/json"
            )
        
        logger.info("Handlers de error configurados")
    
    async def monitor_performance(self) -> Dict[str, Any]:
        """
        Monitorea el rendimiento de la API.
        
        Returns:
            Dict con métricas de rendimiento
        """
        metrics = {
            "state": self.state.value,
            "uptime_seconds": (
                (datetime.now() - self.metrics["start_time"]).total_seconds() 
                if self.metrics["start_time"] else 0
            ),
            "requests_processed": self.metrics["requests_processed"],
            "websocket_connections": self.metrics["websocket_connections"],
            "grpc_calls": self.metrics["grpc_calls"],
            "errors": self.metrics["errors"],
            "active_tasks": len([t for t in self._tasks if not t.done()]),
        }
        
        # Métricas específicas por protocolo
        if self.rest_api:
            metrics["rest"] = await self.rest_api.get_api_metrics()
        
        if self.websocket_api:
            metrics["websocket"] = await self.websocket_api.get_connection_metrics()
        
        return metrics
    
    async def reload_configuration(self, new_config: Dict) -> bool:
        """
        Recarga la configuración sin reiniciar el servidor.
        
        Args:
            new_config: Nueva configuración
            
        Returns:
            bool: True si la recarga fue exitosa
        """
        try:
            old_config = self.config
            self.config = ServerConfig.from_dict(new_config)
            
            # Aplicar cambios dinámicos
            if old_config.enable_cors != self.config.enable_cors:
                await self._reconfigure_cors()
            
            if old_config.log_level != self.config.log_level:
                setup_logging(level=self.config.log_level)
            
            logger.info("Configuración recargada exitosamente")
            return True
            
        except Exception as e:
            logger.error("Error al recargar configuración: %s", e, exc_info=True)
            # Revertir cambios
            self.config = old_config
            return False
    
    # Métodos privados de implementación
    
    async def _initialize_components(self) -> None:
        """Inicializa todos los componentes de la API."""
        # Crear app FastAPI
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            yield
            # Shutdown
        
        self.fastapi_app = FastAPI(
            title="Project Brain API",
            version="1.0.0",
            description="API completa para el sistema Project Brain",
            lifespan=lifespan
        )
        
        # Inicializar componentes
        self.authentication = Authentication()
        self.rate_limiter = RateLimiter()
        self.request_validator = RequestValidator()
        
        # Inicializar APIs específicas
        self.rest_api = RESTAPI(
            authentication=self.authentication,
            rate_limiter=self.rate_limiter,
            request_validator=self.request_validator
        )
        
        self.websocket_api = WebSocketAPI(
            authentication=self.authentication,
            rate_limiter=self.rate_limiter
        )
        
        if self.config.enable_grpc:
            self.grpc_api = GRPCAPI()
        
        # Configurar middleware y rutas
        self.setup_middleware()
        self.configure_routes()
        await self.handle_errors()
        
        logger.info("Componentes de API inicializados")
    
    async def _start_rest_server(self) -> None:
        """Inicia el servidor REST (FastAPI + Uvicorn)."""
        config = uvicorn.Config(
            app=self.fastapi_app,
            host=self.config.host,
            port=self.config.rest_port,
            workers=self.config.workers,
            timeout_keep_alive=self.config.timeout,
            log_level=self.config.log_level.lower(),
            access_log=self.config.enable_logging,
        )
        
        self.uvicorn_server = uvicorn.Server(config)
        
        # Iniciar en background
        server_task = asyncio.create_task(self.uvicorn_server.serve())
        self._tasks.append(server_task)
        
        # Esperar que inicie
        await asyncio.sleep(1)
        
        logger.info("Servidor REST iniciado en http://%s:%s", 
                   self.config.host, self.config.rest_port)
    
    async def _start_websocket_server(self) -> None:
        """Inicia el servidor WebSocket."""
        # El servidor WebSocket se maneja a través de FastAPI normalmente
        # Para un servidor independiente:
        
        async def websocket_handler(websocket, path):
            await self.websocket_api.handle_connection(websocket, path)
        
        start_server = websockets.serve(
            websocket_handler,
            self.config.host,
            self.config.websocket_port,
            max_size=2**20,  # 1MB
            ping_interval=30,
            ping_timeout=90,
            close_timeout=10,
        )
        
        self.websocket_server = await start_server
        
        logger.info("Servidor WebSocket iniciado en ws://%s:%s", 
                   self.config.host, self.config.websocket_port)
    
    async def _start_grpc_server(self) -> None:
        """Inicia el servidor gRPC."""
        if not self.grpc_api:
            return
        
        self.grpc_server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=self.config.workers),
            options=[
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 10000),
            ]
        )
        
        # Añadir servicios
        await self.grpc_api.define_services(self.grpc_server)
        
        # Escuchar en puerto
        listen_addr = f'{self.config.host}:{self.config.grpc_port}'
        self.grpc_server.add_insecure_port(listen_addr)
        
        # Iniciar servidor
        await self.grpc_server.start()
        
        logger.info("Servidor gRPC iniciado en %s", listen_addr)
    
    async def _stop_rest_server(self) -> None:
        """Detiene el servidor REST."""
        if self.uvicorn_server:
            self.uvicorn_server.should_exit = True
            await self.uvicorn_server.shutdown()
            self.uvicorn_server = None
            logger.info("Servidor REST detenido")
    
    async def _stop_websocket_server(self) -> None:
        """Detiene el servidor WebSocket."""
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            self.websocket_server = None
            logger.info("Servidor WebSocket detenido")
    
    async def _stop_grpc_server(self) -> None:
        """Detiene el servidor gRPC."""
        if self.grpc_server:
            await self.grpc_server.stop(grace=5)  # 5 segundos de gracia
            self.grpc_server = None
            logger.info("Servidor gRPC detenido")
    
    async def _cleanup_components(self) -> None:
        """Limpia todos los componentes."""
        if self.rest_api:
            await self.rest_api.cleanup()
            self.rest_api = None
        
        if self.websocket_api:
            await self.websocket_api.cleanup()
            self.websocket_api = None
        
        if self.grpc_api:
            await self.grpc_api.cleanup()
            self.grpc_api = None
        
        self.authentication = None
        self.rate_limiter = None
        self.request_validator = None
        self.fastapi_app = None
        
        # Limpiar tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()
        
        logger.info("Componentes de API limpiados")
    
    def _configure_system_routes(self) -> None:
        """Configura rutas del sistema (salud, métricas, etc.)."""
        if not self.fastapi_app:
            return
        
        @self.fastapi_app.get("/health")
        async def health_check():
            """Endpoint de salud del sistema."""
            from ..core.health_check import HealthCheck
            health = HealthCheck()
            status = await health.check_system_health()
            return {
                "status": "healthy" if status["overall"] == "healthy" else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "components": status["components"],
                "version": "1.0.0"
            }
        
        @self.fastapi_app.get("/metrics")
        async def get_metrics():
            """Endpoint de métricas del sistema."""
            metrics = await self.monitor_performance()
            return {
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.fastapi_app.get("/")
        async def root():
            """Endpoint raíz."""
            return {
                "message": "Project Brain API",
                "version": "1.0.0",
                "documentation": "/docs",
                "endpoints": {
                    "rest": f"http://{self.config.host}:{self.config.rest_port}",
                    "websocket": f"ws://{self.config.host}:{self.config.websocket_port}",
                    "grpc": f"{self.config.host}:{self.config.grpc_port}" if self.config.enable_grpc else None,
                }
            }
    
    def _register_signal_handlers(self) -> None:
        """Registra handlers para señales del sistema."""
        def signal_handler(sig, frame):
            logger.info("Recibida señal %s, deteniendo servidor...", sig)
            asyncio.create_task(self.stop_server())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _reconfigure_cors(self) -> None:
        """Reconfigura CORS dinámicamente."""
        # En producción, esto requeriría reiniciar el servidor o usar middleware dinámico
        logger.warning("Reconfiguración de CORS requiere reinicio del servidor")

# Función principal para ejecutar el servidor
def run_server(config_path: Optional[str] = None):
    """
    Función principal para ejecutar el servidor API.
    
    Args:
        config_path: Ruta al archivo de configuración (opcional)
    """
    import asyncio
    
    async def main():
        # Cargar configuración
        config_manager = ConfigManager()
        if config_path:
            config_manager.load_config(config_path)
        
        api_config = config_manager.get_config().get("api", {})
        
        # Crear e iniciar servidor
        server = APIServer(api_config)
        
        try:
            await server.start_server()
            
            # Mantener el servidor corriendo
            while server.state == ServerState.RUNNING:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Interrupción por teclado, deteniendo servidor...")
            await server.stop_server()
        
        except Exception as e:
            logger.error("Error en servidor: %s", e, exc_info=True)
            await server.stop_server()
    
    # Ejecutar
    asyncio.run(main())

if __name__ == "__main__":
    run_server()