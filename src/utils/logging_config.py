"""
Configuración unificada y estructurada del sistema de logging de ANALYZERBRAIN.

Este módulo centraliza la inicialización y configuración del logging del sistema,
utilizando Loguru como backend principal. Proporciona una configuración consistente
para consola y archivos, con soporte para formatos legibles en desarrollo y
formatos estructurados (JSON) para entornos de producción.

El sistema de logging está diseñado para:
- Proveer trazabilidad clara de ejecución (módulo, función y línea).
- Separar logs generales y logs de error.
- Soportar rotación, retención y compresión de archivos.
- Ser thread-safe y apto para entornos concurrentes.
- Integrarse automáticamente con la configuración global del sistema.

Características principales:
- Logging en consola con formato enriquecido y colores.
- Logging a archivos con rotación y retención configurables.
- Soporte opcional para formato JSON estructurado.
- Archivo dedicado para errores críticos.
- Configuración automática al importar el módulo.

Dependencias:
- loguru (sistema de logging)
- src.core.config_manager (configuración global del sistema)

Clases:
- StructuredLogger: Encapsula la configuración y obtención de loggers estructurados.

Funciones:
- setup_default_logging(): Inicializa el logging con valores por defecto.
- init_logging(): Alias para inicialización automática del logging.

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from src.core.config_manager import config


class StructuredLogger:
    """Logger estructurado para ANALYZERBRAIN."""

    @staticmethod
    def setup_logging(
        log_level: Optional[str] = None,
        log_dir: Optional[Path] | None = None,
        json_format: bool | None = False,
    ) -> None:
        """
        Configura el sistema de logging.

        Args:
            log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
            log_dir: Directorio para archivos de log
            json_format: Si True, usa formato JSON para producción
        """
        # Usar configuración por defecto si no se especifica
        log_level = log_level or "INFO"
        log_dir = log_dir or Path("logs")

        log_dir.mkdir(parents=True, exist_ok=True)

        if json_format is None:
            json_format = not config.is_development

        # Remover handlers por defecto
        logger.remove()

        # Configuración para consola
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stderr,
            level=log_level,
            format=console_format,
            colorize=True,
            backtrace=True,
            diagnose=config.is_development,
        )

        # Configuración para archivo (formato estructurado)
        if json_format:
            file_format = (
                '{{"timestamp": "{time:YYYY-MM-DD HH:mm:ss}", '
                '"level": "{level}", '
                '"module": "{name}", '
                '"function": "{function}", '
                '"line": "{line}", '
                '"message": "{message}", '
                '"extra": {extra}}}'
            )
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message} | {extra}"
            )

        # Asegurar que el directorio de logs exista
        log_dir.mkdir(parents=True, exist_ok=True)

        # Archivo de log principal
        logger.add(
            log_dir / "analyzerbrain_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            format=file_format,
            rotation=config.get("logging.rotation", "1 day"),
            retention=config.get("logging.retention", "30 days"),
            compression="zip",
            backtrace=True,
            diagnose=config.is_development,
            enqueue=True,  # Thread-safe
        )

        # Archivo de errores separado
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            level="ERROR",
            format=file_format,
            rotation="500 MB",
            retention="90 days",
            compression="zip",
            backtrace=True,
            diagnose=False,
        )

        logger.info("Sistema de logging configurado correctamente")
        logger.debug(f"Nivel de log: {log_level}")
        logger.debug(f"Directorio de logs: {log_dir}")
        logger.debug(f"Formato JSON: {json_format}")

    @staticmethod
    def get_logger(name: str):
        """
        Obtiene un logger con un nombre específico.

        Args:
            name: Nombre del logger (generalmente __name__)

        Returns:
            Logger configurado
        """
        return logger.bind(module=name)


def setup_default_logging() -> None:
    """Configura logging con valores por defecto."""
    StructuredLogger.setup_logging()


# Configuración automática al importar el módulo
def init_logging() -> None:
    setup_default_logging()
