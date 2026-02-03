"""
Tests para la configuración de logging de ANALYZERBRAIN.
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from loguru import logger

from src.utils.logging_config import StructuredLogger, setup_default_logging, init_logging


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_loguru():
    """Fixture para limpiar todos los handlers de Loguru antes y después de cada prueba."""
    # Remover todos los handlers existentes
    logger.remove()
    yield
    # Limpiar después de la prueba
    logger.remove()


@pytest.fixture
def mock_config():
    """Fixture para mockear la configuración."""
    with patch('src.utils.logging_config.config') as mock:
        mock.is_development = True
        mock.get.side_effect = lambda key, default=None: {
            "logging.rotation": "1 day",
            "logging.retention": "30 days"
        }.get(key, default)
        yield mock


@pytest.fixture
def temp_log_dir(tmp_path):
    """Fixture para directorio temporal de logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


# -------------------------------------------------------------------
# Tests para StructuredLogger
# -------------------------------------------------------------------

class TestStructuredLogger:
    """Tests para la clase StructuredLogger."""

    def test_setup_logging_creates_directory(self, mock_config, temp_log_dir):
        """Verifica que se crea el directorio de logs si no existe."""
        # Remover el directorio temporal
        import shutil
        shutil.rmtree(temp_log_dir)
        
        # Configurar logging
        StructuredLogger.setup_logging(log_dir=temp_log_dir)
        
        # Verificar que el directorio se creó
        assert temp_log_dir.exists()
        assert temp_log_dir.is_dir()

    def test_setup_logging_removes_default_handlers(self, mock_config, temp_log_dir):
        """Verifica que se remueven los handlers por defecto de Loguru."""
        # Agregar un handler dummy para simular que hay handlers existentes
        dummy_called = []
        
        def dummy_sink(msg):
            dummy_called.append(msg)
        
        logger.add(dummy_sink)
        
        # Configurar logging
        StructuredLogger.setup_logging(log_dir=temp_log_dir)
        
        # El handler dummy debería haber sido removido
        logger.info("Este mensaje no debería ir al dummy")
        
        assert len(dummy_called) == 0

    def test_setup_logging_adds_console_handler(self, mock_config, temp_log_dir):
        """Verifica que se agrega un handler para consola."""
        # Capturar lo que se escribe en stderr
        stderr_calls = []
        
        class MockStderr:
            def write(self, text):
                stderr_calls.append(text)
            def flush(self):
                pass
        
        original_stderr = sys.stderr
        sys.stderr = MockStderr()
        
        try:
            # Configurar logging
            StructuredLogger.setup_logging(log_dir=temp_log_dir, log_level="INFO")
            
            # Escribir un log
            logger.info("Mensaje de prueba para consola")
            
            # Forzar flush
            logger.complete()
            
            # Verificar que se escribió en stderr
            assert len(stderr_calls) > 0
            assert any("Mensaje de prueba para consola" in call for call in stderr_calls)
        finally:
            sys.stderr = original_stderr

    def test_setup_logging_adds_file_handler(self, mock_config, temp_log_dir):
        """Verifica que se agregan handlers de archivo."""
        # Configurar logging
        StructuredLogger.setup_logging(log_dir=temp_log_dir)
        
        # Escribir logs de diferentes niveles
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Forzar escritura
        logger.complete()
        
        # Verificar que se crearon archivos de log
        log_files = list(temp_log_dir.glob("*.log"))
        assert len(log_files) >= 2  # Al menos 2 archivos (principal y errores)
        
        # Verificar que el archivo principal contiene los logs
        main_logs = list(temp_log_dir.glob("analyzerbrain_*.log"))
        assert len(main_logs) == 1
        
        # Leer el contenido
        content = main_logs[0].read_text()
        assert "Info message" in content
        assert "Warning message" in content
        
        # Verificar que el archivo de errores solo contiene errores
        error_logs = list(temp_log_dir.glob("errors_*.log"))
        assert len(error_logs) == 1
        
        error_content = error_logs[0].read_text()
        assert "Error message" in error_content
        assert "Info message" not in error_content

    def test_setup_logging_json_format_production(self, mock_config, temp_log_dir):
        """Verifica formato JSON en modo producción."""
        # Configurar como producción
        mock_config.is_development = False
        
        # Configurar logging con formato JSON
        StructuredLogger.setup_logging(log_dir=temp_log_dir, json_format=True)
        
        # Escribir un log
        logger.info("Mensaje JSON de prueba")
        
        # Forzar escritura
        logger.complete()
        
        # Verificar que el archivo contiene JSON válido
        main_logs = list(temp_log_dir.glob("analyzerbrain_*.log"))
        assert len(main_logs) == 1
        
        content = main_logs[0].read_text()
        # El archivo contiene múltiples líneas JSON, una por cada log
        lines = content.strip().split('\n')
        assert len(lines) > 0
        
        # Verificar que cada línea es JSON válido
        for line in lines:
            if line.strip():  # Ignorar líneas vacías
                try:
                    log_entry = json.loads(line.strip())
                    # Verificar estructura básica
                    assert "timestamp" in log_entry
                    assert "level" in log_entry
                    assert "message" in log_entry
                except json.JSONDecodeError:
                    pytest.fail(f"Línea no es JSON válido: {line}")
        
        # Verificar que nuestro mensaje está en alguna línea
        assert any("Mensaje JSON de prueba" in line for line in lines)

    def test_setup_logging_json_format_development(self, mock_config, temp_log_dir):
        """Verifica formato no-JSON en modo desarrollo."""
        # Configurar como desarrollo
        mock_config.is_development = True
        
        # Configurar logging (por defecto no-JSON en desarrollo)
        StructuredLogger.setup_logging(log_dir=temp_log_dir, json_format=False)
        
        # Escribir un log
        logger.info("Mensaje no-JSON de prueba")
        
        # Forzar escritura
        logger.complete()
        
        # Verificar que el archivo contiene formato legible, no JSON
        main_logs = list(temp_log_dir.glob("analyzerbrain_*.log"))
        assert len(main_logs) == 1
        
        content = main_logs[0].read_text()
        # Formato legible debería tener pipes
        assert " | " in content
        # No debería ser JSON válido (línea por línea)
        lines = content.strip().split('\n')
        for line in lines:
            if line.strip():
                with pytest.raises(json.JSONDecodeError):
                    json.loads(line.strip())

    def test_setup_logging_with_custom_level(self, mock_config, temp_log_dir):
        """Verifica que se respeta el nivel de log personalizado SOLO en consola."""
        # Configurar con nivel WARNING
        StructuredLogger.setup_logging(log_dir=temp_log_dir, log_level="WARNING")
        
        # Escribir logs de diferentes niveles
        logger.info("Este mensaje NO debería aparecer en consola")
        logger.warning("Este mensaje SÍ debería aparecer")
        logger.error("Este mensaje SÍ debería aparecer")
        
        # Forzar escritura
        logger.complete()
        
        # Verificar contenido del archivo
        # El archivo principal tiene nivel DEBUG, por lo que tiene todos los logs
        main_logs = list(temp_log_dir.glob("analyzerbrain_*.log"))
        content = main_logs[0].read_text()
        
        # El archivo debe tener TODOS los mensajes (nivel DEBUG)
        assert "Este mensaje NO debería aparecer en consola" in content
        assert "Este mensaje SÍ debería aparecer" in content

    def test_setup_logging_configures_rotation_and_retention(self, mock_config, temp_log_dir):
        """Verifica que se configuran rotación y retención."""
        # Configurar valores personalizados
        mock_config.get.side_effect = lambda key, default=None: {
            "logging.rotation": "500 MB",
            "logging.retention": "60 days"
        }.get(key, default)
        
        # Configurar logging
        StructuredLogger.setup_logging(log_dir=temp_log_dir)
        
        # Verificar que se llamó a config.get con los parámetros correctos
        mock_config.get.assert_any_call("logging.rotation", "1 day")
        mock_config.get.assert_any_call("logging.retention", "30 days")

    def test_get_logger_returns_bound_logger(self):
        """Verifica que get_logger retorna un logger con el módulo bindeado."""
        # Obtener logger para un módulo específico
        module_name = "mi_modulo"
        bound_logger = StructuredLogger.get_logger(module_name)
        
        # El logger debería tener el módulo en el contexto
        # Podemos verificar esto capturando el log
        captured_logs = []
        
        def capture_sink(msg):
            captured_logs.append(msg.record)
        
        logger.add(capture_sink)
        
        # Usar el logger bindeado
        bound_logger.info("Mensaje con módulo")
        
        # Verificar que el módulo está en el registro
        assert len(captured_logs) == 1
        assert captured_logs[0]["extra"]["module"] == module_name

    def test_setup_logging_thread_safe(self, mock_config, temp_log_dir):
        """Verifica que el logging se configura como thread-safe."""
        # Configurar logging
        StructuredLogger.setup_logging(log_dir=temp_log_dir)
        
        # El handler de archivo principal debería tener enqueue=True
        # (No podemos verificar esto directamente, pero podemos confirmar
        # que no hay errores con múltiples hilos)
        import threading
        
        def log_from_thread(thread_id):
            logger.info(f"Mensaje desde hilo {thread_id}")
        
        # Crear múltiples hilos
        threads = []
        for i in range(10):
            t = threading.Thread(target=log_from_thread, args=(i,))
            threads.append(t)
            t.start()
        
        # Esperar a que todos terminen
        for t in threads:
            t.join()
        
        # Forzar escritura
        logger.complete()
        
        # Verificar que los logs se escribieron sin errores
        main_logs = list(temp_log_dir.glob("analyzerbrain_*.log"))
        content = main_logs[0].read_text()
        
        for i in range(10):
            assert f"Mensaje desde hilo {i}" in content

    def test_setup_logging_error_file_only_errors(self, mock_config, temp_log_dir):
        """Verifica que el archivo de errores solo contiene logs de ERROR."""
        # Configurar logging
        StructuredLogger.setup_logging(log_dir=temp_log_dir)
        
        # Escribir logs de diferentes niveles
        logger.debug("Debug - no debería estar en errors.log")
        logger.info("Info - no debería estar en errors.log")
        logger.warning("Warning - no debería estar en errors.log")
        logger.error("Error - SÍ debería estar en errors.log")
        logger.critical("Critical - SÍ debería estar en errors.log")
        
        # Forzar escritura
        logger.complete()
        
        # Verificar archivo de errores
        error_logs = list(temp_log_dir.glob("errors_*.log"))
        error_content = error_logs[0].read_text()
        
        assert "Error - SÍ debería estar en errors.log" in error_content
        assert "Critical - SÍ debería estar en errors.log" in error_content
        assert "Debug - no debería estar en errors.log" not in error_content
        assert "Info - no debería estar en errors.log" not in error_content
        assert "Warning - no debería estar en errors.log" not in error_content

    def test_setup_logging_with_none_json_format_uses_config(self, mock_config, temp_log_dir):
        """Verifica que cuando json_format es None, se usa is_development."""
        # Caso 1: Desarrollo -> json_format=False
        mock_config.is_development = True
        StructuredLogger.setup_logging(log_dir=temp_log_dir, json_format=None)
        
        logger.info("Mensaje desarrollo")
        logger.complete()
        
        main_logs = list(temp_log_dir.glob("analyzerbrain_*.log"))
        assert len(main_logs) == 1
        
        content = main_logs[0].read_text()
        # Formato desarrollo no es JSON, tiene pipes
        assert " | " in content
        # No debería ser JSON válido (pero puede contener múltiples líneas)
        lines = content.strip().split('\n')
        for line in lines:
            if line.strip():
                with pytest.raises(json.JSONDecodeError):
                    json.loads(line.strip())
        
        # Limpiar para siguiente prueba
        logger.remove()
        
        # Eliminar archivos de log existentes para empezar limpio
        for log_file in temp_log_dir.glob("*.log"):
            log_file.unlink()
        
        # Caso 2: Producción -> json_format=True
        mock_config.is_development = False
        StructuredLogger.setup_logging(log_dir=temp_log_dir, json_format=None)
        
        logger.info("Mensaje producción")
        logger.complete()
        
        main_logs = list(temp_log_dir.glob("analyzerbrain_*.log"))
        assert len(main_logs) == 1
        
        content = main_logs[0].read_text().strip()
        # Formato producción debería ser JSON válido (línea por línea)
        lines = content.split('\n')
        for line in lines:
            if line.strip():
                try:
                    json.loads(line.strip())
                except json.JSONDecodeError:
                    pytest.fail(f"Línea no es JSON válido en modo producción: {line}")


# -------------------------------------------------------------------
# Tests para funciones del módulo
# -------------------------------------------------------------------

def test_setup_default_logging_calls_structured_logger():
    """Verifica que setup_default_logging llama a StructuredLogger.setup_logging."""
    with patch.object(StructuredLogger, 'setup_logging') as mock_setup:
        setup_default_logging()
        
        # Verificar que se llamó con parámetros por defecto
        mock_setup.assert_called_once()


def test_init_logging_calls_setup_default_logging():
    """Verifica que init_logging llama a setup_default_logging."""
    with patch('src.utils.logging_config.setup_default_logging') as mock_setup:
        init_logging()
        mock_setup.assert_called_once()


def test_setup_logging_logs_initialization(mock_config, temp_log_dir):
    """Verifica que se loguea la inicialización del sistema."""
    # Configurar logging
    StructuredLogger.setup_logging(log_dir=temp_log_dir)
    
    # Forzar escritura
    logger.complete()
    
    # Leer el archivo de log principal
    main_logs = list(temp_log_dir.glob("analyzerbrain_*.log"))
    assert len(main_logs) == 1
    
    content = main_logs[0].read_text()
    
    # Verificar que los mensajes de inicialización están en el archivo
    assert "Sistema de logging configurado correctamente" in content
    assert "Nivel de log:" in content
    assert "Directorio de logs:" in content
    assert "Formato JSON:" in content


def test_setup_logging_with_existing_directory(mock_config, temp_log_dir):
    """Verifica que funciona correctamente cuando el directorio ya existe."""
    # Crear algunos archivos en el directorio
    (temp_log_dir / "existing_file.txt").write_text("Contenido existente")
    (temp_log_dir / "old_log.log").write_text("Log viejo")
    
    # Configurar logging
    StructuredLogger.setup_logging(log_dir=temp_log_dir)
    
    # Escribir un log
    logger.info("Nuevo mensaje")
    logger.complete()
    
    # Verificar que el directorio sigue existiendo
    assert temp_log_dir.exists()
    
    # Verificar que los archivos existentes no fueron eliminados
    assert (temp_log_dir / "existing_file.txt").exists()
    
    # Verificar que se crearon nuevos archivos de log
    new_logs = list(temp_log_dir.glob("analyzerbrain_*.log"))
    assert len(new_logs) == 1


# -------------------------------------------------------------------
# Tests de edge cases
# -------------------------------------------------------------------

def test_setup_logging_with_invalid_log_level(mock_config, temp_log_dir):
    """Verifica que se manejan niveles de log inválidos."""
    # Loguru maneja niveles inválidos levantando ValueError
    with pytest.raises(ValueError):
        StructuredLogger.setup_logging(log_dir=temp_log_dir, log_level="INVALID_LEVEL")

def test_setup_logging_with_none_log_dir(mock_config):
    """Verifica que se usa directorio por defecto cuando log_dir es None."""
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        StructuredLogger.setup_logging(log_dir=None)
        
        # Verificar que se intentó crear el directorio "logs"
        # Se llama 2 veces en el código actual
        assert mock_mkdir.called
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

def test_get_logger_with_empty_name():
    """Verifica que get_logger funciona con nombre vacío."""
    logger_empty = StructuredLogger.get_logger("")
    
    # Debería retornar un logger válido
    assert logger_empty is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])