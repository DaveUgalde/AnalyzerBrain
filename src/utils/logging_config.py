"""
LoggingConfig - Configuración avanzada de logging para Project Brain.
Incluye configuración multi-formato, rotación, análisis y exportación de logs.
"""

import logging
import logging.config
import logging.handlers
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
import json
import yaml
import time
import gzip
import re
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
from core.exceptions import BrainException

class LogLevel(str, Enum):
    """Niveles de log soportados."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogFormat(str, Enum):
    """Formatos de log soportados."""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    GELF = "gelf"  # Graylog Extended Log Format

@dataclass
class LogEntry:
    """Entrada de log estructurada."""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_name: Optional[str] = None
    process_id: Optional[int] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'logger': self.logger_name,
            'module': self.module,
            'function': self.function,
            'line': self.line_number,
            'thread': self.thread_name,
            'process': self.process_id,
            **self.extra_fields
        }

class LoggingConfig:
    """
    Configuración avanzada de logging con soporte para múltiples formatos,
    rotación automática y análisis en tiempo real.
    """
    
    # Configuración global
    _initialized = False
    _default_config = None
    _log_buffer = deque(maxlen=10000)  # Buffer circular para logs recientes
    _log_analyzers = []
    _metrics = defaultdict(int)
    
    @staticmethod
    def setup_logging(
        config_file: Optional[Union[str, Path]] = None,
        default_level: LogLevel = LogLevel.INFO,
        log_dir: Union[str, Path] = "./logs",
        enable_console: bool = True,
        enable_file: bool = True,
        enable_metrics: bool = True,
        log_format: LogFormat = LogFormat.TEXT,
        **kwargs
    ) -> logging.Logger:
        """
        Configura el sistema de logging completo.
        
        Args:
            config_file: Archivo de configuración YAML/JSON (opcional)
            default_level: Nivel de log por defecto
            log_dir: Directorio para archivos de log
            enable_console: Habilitar logging a consola
            enable_file: Habilitar logging a archivo
            enable_metrics: Habilitar recolección de métricas
            log_format: Formato de los logs
            **kwargs: Configuración adicional
            
        Returns:
            Logger raíz configurado
            
        Raises:
            BrainException: Si hay error en la configuración
        """
        try:
            # Evitar configuración múltiple
            if LoggingConfig._initialized:
                root_logger = logging.getLogger()
                root_logger.info("Logging already initialized, returning existing config")
                return root_logger
            
            # Crear directorio de logs
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Cargar configuración desde archivo si existe
            if config_file:
                config = LoggingConfig._load_config_file(config_file)
            else:
                # Configuración por defecto
                config = LoggingConfig._create_default_config(
                    log_dir=log_dir,
                    default_level=default_level,
                    enable_console=enable_console,
                    enable_file=enable_file,
                    log_format=log_format,
                    **kwargs
                )
            
            # Aplicar configuración
            logging.config.dictConfig(config)
            
            # Configurar logger raíz
            root_logger = logging.getLogger()
            
            # Agregar filtro personalizado para capturar métricas
            if enable_metrics:
                metrics_filter = LoggingConfig._create_metrics_filter()
                for handler in root_logger.handlers:
                    handler.addFilter(metrics_filter)
            
            # Registrar handlers personalizados
            LoggingConfig._setup_custom_handlers(root_logger, config)
            
            # Marcar como inicializado
            LoggingConfig._initialized = True
            LoggingConfig._default_config = config
            
            root_logger.info(f"Logging initialized with level {default_level}")
            root_logger.info(f"Log directory: {log_dir.absolute()}")
            
            return root_logger
            
        except Exception as e:
            # Fallback a configuración básica
            logging.basicConfig(
                level=getattr(logging, default_level.value),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger()
            logger.error(f"Failed to setup advanced logging: {e}")
            raise BrainException(f"Logging setup failed: {e}")
    
    @staticmethod
    def configure_log_levels(
        log_levels: Dict[str, LogLevel],
        propagate: bool = True
    ) -> None:
        """
        Configura niveles de log específicos para diferentes loggers.
        
        Args:
            log_levels: Diccionario logger_name -> LogLevel
            propagate: Propagar logs al logger padre
        """
        if not LoggingConfig._initialized:
            raise BrainException("Logging not initialized. Call setup_logging first.")
        
        for logger_name, level in log_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, level.value))
            logger.propagate = propagate
            
            logging.getLogger(__name__).debug(
                f"Set log level for '{logger_name}' to {level.value}"
            )
    
    @staticmethod
    def setup_log_handlers(
        handlers: List[Dict[str, Any]],
        clear_existing: bool = False
    ) -> None:
        """
        Configura handlers de log personalizados.
        
        Args:
            handlers: Lista de configuraciones de handlers
            clear_existing: Eliminar handlers existentes
        """
        if not LoggingConfig._initialized:
            raise BrainException("Logging not initialized. Call setup_logging first.")
        
        root_logger = logging.getLogger()
        
        if clear_existing:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
        
        for handler_config in handlers:
            try:
                handler = LoggingConfig._create_handler_from_config(handler_config)
                if handler:
                    root_logger.addHandler(handler)
                    logging.getLogger(__name__).info(
                        f"Added handler: {handler_config.get('class', 'Unknown')}"
                    )
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Failed to create handler from config: {e}"
                )
    
    @staticmethod
    def format_log_messages(
        format_template: Optional[str] = None,
        date_format: str = "%Y-%m-%d %H:%M:%S",
        extra_fields: Optional[Dict[str, str]] = None,
        as_json: bool = False
    ) -> Callable[[logging.LogRecord], str]:
        """
        Crea un formateador personalizado para mensajes de log.
        
        Args:
            format_template: Template de formato (usando atributos de LogRecord)
            date_format: Formato de fecha
            extra_fields: Campos extra a incluir
            as_json: Formatear como JSON
            
        Returns:
            Función formateadora
        """
        if as_json:
            def json_formatter(record: logging.LogRecord) -> str:
                log_data = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'logger': record.name,
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                    'thread': record.threadName,
                    'process': record.processName if hasattr(record, 'processName') else record.process,
                }
                
                # Agregar campos extra del record
                if hasattr(record, 'extra_fields'):
                    log_data.update(record.extra_fields)
                
                if extra_fields:
                    log_data.update(extra_fields)
                
                # Agregar excepción si existe
                if record.exc_info:
                    log_data['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': logging.Formatter().formatException(record.exc_info)
                    }
                
                return json.dumps(log_data)
            
            return json_formatter
        
        else:
            # Template por defecto
            if format_template is None:
                format_template = (
                    '%(asctime)s | %(levelname)-8s | %(name)-20s | '
                    '%(module)s.%(funcName)s:%(lineno)d | %(message)s'
                )
            
            formatter = logging.Formatter(
                fmt=format_template,
                datefmt=date_format
            )
            
            return formatter.format
    
    @staticmethod
    def rotate_logs(
        log_dir: Union[str, Path],
        keep_days: int = 30,
        max_size_mb: int = 100,
        compress_old: bool = True,
        backup_count: int = 10
    ) -> Dict[str, Any]:
        """
        Rota y limpia archivos de log antiguos.
        
        Args:
            log_dir: Directorio con archivos de log
            keep_days: Mantener logs de los últimos N días
            max_size_mb: Tamaño máximo por archivo de log
            compress_old: Comprimir logs antiguos
            backup_count: Número máximo de archivos de backup
            
        Returns:
            Estadísticas de rotación
        """
        log_dir = Path(log_dir)
        stats = {
            'total_files': 0,
            'rotated': 0,
            'compressed': 0,
            'deleted': 0,
            'errors': 0
        }
        
        if not log_dir.exists():
            return stats
        
        cutoff_time = time.time() - (keep_days * 86400)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        try:
            for log_file in log_dir.glob("*.log*"):
                stats['total_files'] += 1
                
                try:
                    # Rotar archivos grandes
                    if log_file.stat().st_size > max_size_bytes:
                        LoggingConfig._rotate_single_file(log_file, backup_count)
                        stats['rotated'] += 1
                    
                    # Comprimir archivos antiguos
                    file_age = time.time() - log_file.stat().st_mtime
                    if compress_old and file_age > (keep_days / 2 * 86400):
                        if not log_file.name.endswith('.gz'):
                            compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
                            with open(log_file, 'rb') as f_in:
                                with gzip.open(compressed_file, 'wb') as f_out:
                                    f_out.write(f_in.read())
                            log_file.unlink()
                            stats['compressed'] += 1
                    
                    # Eliminar archivos muy antiguos
                    if file_age > cutoff_time:
                        log_file.unlink()
                        stats['deleted'] += 1
                        
                except Exception as e:
                    stats['errors'] += 1
                    logging.getLogger(__name__).warning(
                        f"Failed to process log file {log_file}: {e}"
                    )
            
            # Loggear estadísticas
            logging.getLogger(__name__).info(
                f"Log rotation completed: {stats}"
            )
            
            return stats
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Log rotation failed: {e}")
            raise BrainException(f"Log rotation failed: {e}")
    
    @staticmethod
    def analyze_logs(
        log_file: Union[str, Path],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        level_filter: Optional[List[LogLevel]] = None,
        pattern: Optional[str] = None,
        max_entries: int = 1000
    ) -> Dict[str, Any]:
        """
        Analiza archivos de log para extraer información útil.
        
        Args:
            log_file: Archivo de log a analizar
            time_range: Rango de tiempo para filtrar
            level_filter: Filtrar por niveles de log
            pattern: Patrón regex para buscar
            max_entries: Máximo de entradas a analizar
            
        Returns:
            Resultados del análisis
        """
        log_file = Path(log_file)
        
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        results = {
            'total_entries': 0,
            'by_level': defaultdict(int),
            'by_module': defaultdict(int),
            'by_hour': defaultdict(int),
            'errors': [],
            'warnings': [],
            'pattern_matches': [],
            'time_range': None,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Determinar si es JSON o texto
            is_json = log_file.suffix == '.json' or '.json.' in log_file.name
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num > max_entries:
                        break
                    
                    try:
                        if is_json:
                            log_entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                            level = log_entry.get('level', 'INFO')
                            message = log_entry.get('message', '')
                            module = log_entry.get('module', 'unknown')
                        else:
                            # Parsear formato de texto (simplificado)
                            log_entry = LoggingConfig._parse_log_line(line)
                            entry_time = log_entry.get('timestamp')
                            level = log_entry.get('level', 'INFO')
                            message = log_entry.get('message', '')
                            module = log_entry.get('module', 'unknown')
                        
                        # Filtrar por tiempo
                        if time_range and entry_time:
                            if not (time_range[0] <= entry_time <= time_range[1]):
                                continue
                        
                        # Filtrar por nivel
                        if level_filter and level not in [l.value for l in level_filter]:
                            continue
                        
                        # Filtrar por patrón
                        if pattern and not re.search(pattern, message, re.IGNORECASE):
                            continue
                        
                        # Actualizar estadísticas
                        results['total_entries'] += 1
                        results['by_level'][level] += 1
                        results['by_module'][module] += 1
                        
                        if entry_time:
                            hour = entry_time.hour
                            results['by_hour'][hour] += 1
                        
                        # Capturar errores y warnings
                        if level in ['ERROR', 'CRITICAL']:
                            results['errors'].append({
                                'line': line_num,
                                'message': message[:200],
                                'timestamp': entry_time.isoformat() if entry_time else None,
                                'module': module
                            })
                        elif level == 'WARNING':
                            results['warnings'].append({
                                'line': line_num,
                                'message': message[:200],
                                'timestamp': entry_time.isoformat() if entry_time else None,
                                'module': module
                            })
                        
                        # Capturar coincidencias de patrón
                        if pattern and re.search(pattern, message, re.IGNORECASE):
                            results['pattern_matches'].append({
                                'line': line_num,
                                'message': message[:200],
                                'timestamp': entry_time.isoformat() if entry_time else None
                            })
                            
                    except (json.JSONDecodeError, ValueError) as e:
                        # Ignorar líneas inválidas
                        continue
            
            # Calcular tiempo de procesamiento
            results['processing_time'] = time.time() - start_time
            
            # Ordenar resultados
            results['by_level'] = dict(sorted(results['by_level'].items()))
            results['by_module'] = dict(
                sorted(results['by_module'].items(), 
                      key=lambda x: x[1], reverse=True)[:10]
            )
            results['by_hour'] = dict(sorted(results['by_hour'].items()))
            
            # Limitar listas grandes
            results['errors'] = results['errors'][:100]
            results['warnings'] = results['warnings'][:100]
            results['pattern_matches'] = results['pattern_matches'][:100]
            
            return results
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Log analysis failed: {e}")
            raise BrainException(f"Log analysis failed: {e}")
    
    @staticmethod
    def export_logs(
        source: Union[str, Path, List[Dict]],
        output_file: Union[str, Path],
        format: str = 'json',
        time_range: Optional[Tuple[datetime, datetime]] = None,
        level_filter: Optional[List[LogLevel]] = None,
        include_fields: Optional[List[str]] = None
    ) -> Path:
        """
        Exporta logs a diferentes formatos.
        
        Args:
            source: Archivo de log o lista de entradas
            output_file: Archivo destino
            format: Formato de exportación ('json', 'csv', 'html')
            time_range: Filtrar por rango de tiempo
            level_filter: Filtrar por niveles
            include_fields: Campos a incluir
            
        Returns:
            Ruta al archivo exportado
        """
        output_file = Path(output_file)
        
        try:
            # Cargar datos
            if isinstance(source, (str, Path)):
                analysis = LoggingConfig.analyze_logs(
                    source, time_range, level_filter, max_entries=10000
                )
                entries = analysis.get('raw_entries', [])
            else:
                entries = source
            
            # Exportar según formato
            if format == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(entries, f, indent=2, default=str)
            
            elif format == 'csv':
                import csv
                
                # Determinar campos
                if entries and include_fields:
                    fieldnames = include_fields
                elif entries:
                    fieldnames = list(entries[0].keys())
                else:
                    fieldnames = ['timestamp', 'level', 'message', 'module']
                
                with open(output_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(entries)
            
            elif format == 'html':
                LoggingConfig._export_logs_html(entries, output_file, include_fields)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logging.getLogger(__name__).info(
                f"Exported {len(entries)} log entries to {output_file}"
            )
            
            return output_file
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Log export failed: {e}")
            raise BrainException(f"Log export failed: {e}")
    
    # ========== MÉTODOS PRIVADOS ==========
    
    @staticmethod
    def _load_config_file(config_file: Union[str, Path]) -> Dict[str, Any]:
        """Carga configuración de logging desde archivo."""
        config_file = Path(config_file)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix in ['.yaml', '.yml']:
                import yaml
                config = yaml.safe_load(f)
            elif config_file.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        
        return config
    
    @staticmethod
    def _create_default_config(
        log_dir: Path,
        default_level: LogLevel,
        enable_console: bool,
        enable_file: bool,
        log_format: LogFormat,
        **kwargs
    ) -> Dict[str, Any]:
        """Crea configuración por defecto."""
        # Formatters
        formatters = {
            'standard': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)-20s | %(module)s.%(funcName)s:%(lineno)d | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                'format': '%(timestamp)s %(level)s %(name)s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        }
        
        # Handlers
        handlers = {}
        
        if enable_console:
            handlers['console'] = {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'json' if log_format == LogFormat.JSON else 'standard',
                'stream': 'ext://sys.stdout'
            }
        
        if enable_file:
            # Handler principal
            handlers['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': default_level.value,
                'formatter': 'json' if log_format == LogFormat.JSON else 'detailed',
                'filename': str(log_dir / 'project_brain.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10,
                'encoding': 'utf-8'
            }
            
            # Handler de errores
            handlers['error_file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': str(log_dir / 'errors.log'),
                'maxBytes': 5242880,  # 5MB
                'backupCount': 5,
                'encoding': 'utf-8'
            }
        
        # Loggers
        loggers = {
            '': {  # Root logger
                'handlers': list(handlers.keys()),
                'level': default_level.value,
                'propagate': True
            },
            'project_brain': {
                'handlers': list(handlers.keys()),
                'level': 'INFO',
                'propagate': False
            },
            '__main__': {
                'handlers': list(handlers.keys()),
                'level': 'DEBUG',
                'propagate': False
            }
        }
        
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': formatters,
            'handlers': handlers,
            'loggers': loggers,
            'root': {
                'level': default_level.value,
                'handlers': list(handlers.keys())
            }
        }
    
    @staticmethod
    def _create_metrics_filter():
        """Crea filtro para capturar métricas de logging."""
        class MetricsFilter(logging.Filter):
            def filter(self, record):
                # Capturar en buffer
                log_entry = LogEntry(
                    timestamp=datetime.fromtimestamp(record.created),
                    level=LogLevel(record.levelname),
                    message=record.getMessage(),
                    logger_name=record.name,
                    module=record.module,
                    function=record.funcName,
                    line_number=record.lineno,
                    thread_name=record.threadName,
                    process_id=record.process
                )
                
                LoggingConfig._log_buffer.append(log_entry)
                
                # Actualizar métricas
                LoggingConfig._metrics[f'logs_{record.levelname.lower()}'] += 1
                LoggingConfig._metrics['logs_total'] += 1
                
                # Ejecutar analizadores
                for analyzer in LoggingConfig._log_analyzers:
                    try:
                        analyzer(log_entry)
                    except Exception as e:
                        pass  # No fallar por errores en analizadores
                
                return True
        
        return MetricsFilter()
    
    @staticmethod
    def _setup_custom_handlers(root_logger: logging.Logger, config: Dict[str, Any]) -> None:
        """Configura handlers personalizados adicionales."""
        # Handler para métricas en memoria
        metrics_handler = logging.handlers.MemoryHandler(
            capacity=1000,
            flushLevel=logging.ERROR,
            target=root_logger
        )
        root_logger.addHandler(metrics_handler)
        
        # Handler para syslog si está disponible
        try:
            syslog_handler = logging.handlers.SysLogHandler(
                address='/dev/log'
            )
            syslog_handler.setLevel(logging.WARNING)
            root_logger.addHandler(syslog_handler)
        except Exception:
            pass  # Syslog no disponible
    
    @staticmethod
    def _create_handler_from_config(handler_config: Dict[str, Any]) -> Optional[logging.Handler]:
        """Crea handler de logging desde configuración."""
        handler_class = handler_config.get('class')
        
        if not handler_class:
            return None
        
        try:
            # Importar clase dinámicamente
            if '.' in handler_class:
                module_name, class_name = handler_config.rsplit('.', 1)
                module = __import__(module_name, fromlist=[class_name])
                handler_class = getattr(module, class_name)
            else:
                handler_class = getattr(logging.handlers, handler_class)
            
            # Crear instancia
            handler_kwargs = {k: v for k, v in handler_config.items() 
                             if k not in ['class', 'formatter']}
            
            # Manejar valores especiales
            if 'filename' in handler_kwargs:
                handler_kwargs['filename'] = str(handler_kwargs['filename'])
            
            handler = handler_class(**handler_kwargs)
            
            # Configurar formatter
            if 'formatter' in handler_config:
                formatter_config = handler_config['formatter']
                if isinstance(formatter_config, dict):
                    formatter = logging.Formatter(**formatter_config)
                else:
                    formatter = logging.Formatter(formatter_config)
                handler.setFormatter(formatter)
            
            # Configurar nivel
            if 'level' in handler_config:
                handler.setLevel(getattr(logging, handler_config['level']))
            
            return handler
            
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to create handler {handler_class}: {e}"
            )
            return None
    
    @staticmethod
    def _rotate_single_file(log_file: Path, backup_count: int) -> None:
        """Rota un archivo de log individual."""
        if backup_count > 0:
            # Eliminar el backup más antiguo
            oldest_backup = log_file.with_suffix(f'.{backup_count}')
            if oldest_backup.exists():
                oldest_backup.unlink()
            
            # Renombrar backups existentes
            for i in range(backup_count - 1, 0, -1):
                old_backup = log_file.with_suffix(f'.{i}')
                new_backup = log_file.with_suffix(f'.{i + 1}')
                if old_backup.exists():
                    old_backup.rename(new_backup)
            
            # Renombrar archivo actual
            log_file.rename(log_file.with_suffix('.1'))
    
    @staticmethod
    def _parse_log_line(line: str) -> Dict[str, Any]:
        """Parse una línea de log en formato texto."""
        # Patrones comunes de formato de log
        patterns = [
            # Formato estándar: timestamp | level | name | message
            r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| '
            r'(?P<level>\w+)\s*\| '
            r'(?P<name>\S+)\s*\| '
            r'(?P<message>.*)',
            
            # Formato simple: timestamp - level - message
            r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - '
            r'(?P<level>\w+) - '
            r'(?P<message>.*)'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                result = match.groupdict()
                
                # Parsear timestamp
                try:
                    result['timestamp'] = datetime.strptime(
                        result['timestamp'], '%Y-%m-%d %H:%M:%S'
                    )
                except ValueError:
                    result['timestamp'] = None
                
                # Extraer módulo si está en el mensaje
                module_match = re.search(r'\((?P<module>\w+)\)', result.get('message', ''))
                if module_match:
                    result['module'] = module_match.group('module')
                else:
                    result['module'] = 'unknown'
                
                return result
        
        # Si no coincide con ningún patrón
        return {
            'timestamp': None,
            'level': 'INFO',
            'message': line.strip(),
            'module': 'unknown'
        }
    
    @staticmethod
    def _export_logs_html(entries: List[Dict], output_file: Path, include_fields: List[str]) -> None:
        """Exporta logs a formato HTML."""
        # Determinar campos
        if entries and include_fields:
            fields = include_fields
        elif entries:
            fields = list(entries[0].keys())
        else:
            fields = ['timestamp', 'level', 'message', 'module']
        
        # Crear HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Project Brain Log Export</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .level-DEBUG {{ color: #666; }}
                .level-INFO {{ color: #333; }}
                .level-WARNING {{ color: #ff9900; font-weight: bold; }}
                .level-ERROR {{ color: #ff0000; font-weight: bold; }}
                .level-CRITICAL {{ color: #cc0000; font-weight: bold; background-color: #ffe6e6; }}
                .timestamp {{ font-family: monospace; }}
                .search {{ margin-bottom: 20px; }}
                .stats {{ margin: 20px 0; padding: 10px; background-color: #f0f0f0; }}
            </style>
        </head>
        <body>
            <h1>Project Brain Log Export</h1>
            <div class="stats">
                <p>Total entries: {len(entries)}</p>
                <p>Generated: {datetime.now().isoformat()}</p>
            </div>
            <div class="search">
                <input type="text" id="searchInput" placeholder="Search logs..." style="width: 300px; padding: 5px;">
                <button onclick="searchLogs()">Search</button>
                <button onclick="clearSearch()">Clear</button>
            </div>
            <table id="logTable">
                <thead>
                    <tr>
                        {' '.join(f'<th>{field}</th>' for field in fields)}
                    </tr>
                </thead>
                <tbody>
        """
        
        for entry in entries:
            html_content += '<tr>'
            for field in fields:
                value = entry.get(field, '')
                if field == 'level':
                    html_content += f'<td class="level-{value}">{value}</td>'
                elif field == 'timestamp':
                    html_content += f'<td class="timestamp">{value}</td>'
                else:
                    # Escapar HTML en valores
                    escaped_value = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    html_content += f'<td>{escaped_value}</td>'
            html_content += '</tr>\n'
        
        html_content += """
                </tbody>
            </table>
            <script>
                function searchLogs() {
                    var input = document.getElementById("searchInput");
                    var filter = input.value.toUpperCase();
                    var table = document.getElementById("logTable");
                    var tr = table.getElementsByTagName("tr");
                    
                    for (var i = 1; i < tr.length; i++) {
                        var td = tr[i].getElementsByTagName("td");
                        var found = false;
                        for (var j = 0; j < td.length; j++) {
                            if (td[j].innerHTML.toUpperCase().indexOf(filter) > -1) {
                                found = true;
                                break;
                            }
                        }
                        tr[i].style.display = found ? "" : "none";
                    }
                }
                
                function clearSearch() {
                    document.getElementById("searchInput").value = "";
                    searchLogs();
                }
                
                // Search on Enter key
                document.getElementById("searchInput").addEventListener("keyup", function(event) {
                    if (event.key === "Enter") {
                        searchLogs();
                    }
                });
            </script>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)