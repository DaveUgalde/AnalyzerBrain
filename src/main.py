"""
Punto de entrada principal del sistema Project Brain.
"""

import asyncio
import sys
import logging
import signal
from pathlib import Path
from typing import Optional, Dict, Any

# Configurar logging antes de cualquier import
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('project_brain.log')
    ]
)

logger = logging.getLogger(__name__)

from src.core.orchestrator import BrainOrchestrator
from src.core.exceptions import BrainException, ValidationError
from src.core.config_manager import ConfigManager

class ProjectBrain:
    """Clase principal que maneja el ciclo de vida del sistema."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa Project Brain.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path
        self.orchestrator: Optional[BrainOrchestrator] = None
        self.config_manager: Optional[ConfigManager] = None
        self.is_running = False
        
        # Configurar manejo de señales
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
    
    async def initialize(self) -> bool:
        """
        Inicializa todos los componentes del sistema.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            logger.info("🚀 Inicializando Project Brain v%s", __import__('src').__version__)
            
            # 1. Inicializar gestor de configuración
            self.config_manager = ConfigManager()
            if self.config_path:
                config = self.config_manager.load_config("system", self.config_path)
                logger.info("Configuración cargada desde: %s", self.config_path)
            else:
                logger.info("Usando configuración por defecto")
            
            # 2. Inicializar orquestador principal
            self.orchestrator = BrainOrchestrator(self.config_path)
            success = await self.orchestrator.initialize()
            
            if not success:
                logger.error("❌ Falló la inicialización del orquestador")
                return False
            
            self.is_running = True
            logger.info("✅ Sistema inicializado exitosamente")
            
            # 3. Mostrar información del sistema
            await self._show_system_info()
            
            return True
            
        except Exception as e:
            logger.error("❌ Error durante la inicialización: %s", str(e), exc_info=True)
            return False
    
    async def run(self) -> None:
        """
        Ejecuta el bucle principal del sistema.
        """
        if not self.is_running or not self.orchestrator:
            logger.error("Sistema no inicializado")
            return
        
        try:
            logger.info("⚡ Sistema en ejecución. Presiona Ctrl+C para salir.")
            
            # Mantener el sistema corriendo
            # En producción, esto manejaría peticiones, colas, etc.
            while self.is_running:
                # Verificar estado del sistema periódicamente
                status = await self.orchestrator._get_system_status()
                
                if status.get("status") != "running":
                    logger.warning("Estado del sistema: %s", status.get("status"))
                
                # Sleep para no consumir CPU innecesariamente
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("🛑 Señal de interrupción recibida")
        except Exception as e:
            logger.error("❌ Error en el bucle principal: %s", str(e))
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """
        Apaga el sistema de manera controlada.
        """
        logger.info("🔽 Iniciando apagado del sistema...")
        
        try:
            if self.orchestrator:
                await self.orchestrator.shutdown(force=False)
            
            self.is_running = False
            logger.info("✅ Sistema apagado exitosamente")
            
        except Exception as e:
            logger.error("❌ Error durante el apagado: %s", str(e))
            # Forzar apagado si hay error
            if self.orchestrator:
                await self.orchestrator.shutdown(force=True)
    
    async def _show_system_info(self) -> None:
        """Muestra información del sistema."""
        if not self.orchestrator:
            return
        
        try:
            status = await self.orchestrator._get_system_status()
            
            logger.info("📊 Estado del sistema:")
            logger.info("  - Estado: %s", status.get("status"))
            logger.info("  - Modo: %s", status.get("system_mode"))
            logger.info("  - Componentes: %s", len(status.get("components", {})))
            logger.info("  - Operaciones activas: %s", status.get("active_operations"))
            
        except Exception as e:
            logger.warning("No se pudo obtener información del sistema: %s", str(e))
    
    def _handle_shutdown_signal(self, signum, frame) -> None:
        """Maneja señales de apagado."""
        logger.info("📡 Señal %s recibida, iniciando apagado...", signal.Signals(signum).name)
        
        # Crear tarea de apagado asíncrona
        if self.is_running:
            asyncio.create_task(self.shutdown())

async def main() -> int:
    """
    Función principal del sistema.
    
    Returns:
        int: Código de salida (0 = éxito, >0 = error)
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Project Brain - Sistema de IA para análisis de código")
    parser.add_argument(
        "--config", 
        "-c", 
        type=str, 
        help="Ruta al archivo de configuración"
    )
    parser.add_argument(
        "--log-level", 
        "-l", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nivel de logging"
    )
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Crear y ejecutar sistema
    brain = ProjectBrain(args.config)
    
    try:
        # Inicializar sistema
        if not await brain.initialize():
            return 1
        
        # Ejecutar bucle principal
        await brain.run()
        
        return 0
        
    except BrainException as e:
        logger.error("Error del sistema: %s", str(e))
        return 1
    except ValidationError as e:
        logger.error("Error de validación: %s", str(e))
        return 2
    except Exception as e:
        logger.error("Error inesperado: %s", str(e), exc_info=True)
        return 3

if __name__ == "__main__":
    # Ejecutar sistema
    exit_code = asyncio.run(main())
    sys.exit(exit_code)