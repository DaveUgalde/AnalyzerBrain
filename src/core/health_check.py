"""
Health check system for ANALYZERBRAIN.

This module provides synchronous and asynchronous health checks to verify
the operational status of the system, including environment, configuration,
resources, filesystem, dependencies, and basic network connectivity.

It is designed to be compatible with the current ANALYZERBRAIN core and can
be used both programmatically and as a diagnostic utility.

Requirements:
    - psutil
    - loguru
    - core.config_manager
    - core.exceptions

Author: ANALYZERBRAIN Team
Version: 1.0.0
"""


import sys
import platform
import socket
from datetime import datetime
from pathlib import Path
import importlib.util
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Awaitable, TypeAlias

import psutil
from loguru import logger

AsyncHealthCheck: TypeAlias = Callable[[], Awaitable["HealthCheckResult"]]

if TYPE_CHECKING:
    from .config_manager import ConfigManager

from .config_manager import config
#from .exceptions import AnalyzerBrainError


class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class HealthCheckResult:
    """Resultado de un health check individual."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    critical: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "critical": self.critical,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }


class SystemHealthChecker:
    """Checker de salud del sistema compatible con el main actual."""
    
    def __init__(self, config_manager: Optional["ConfigManager"]=None):
        """
        Inicializa el sistema de health check.
        
        Args:
            config_manager: Instancia de ConfigManager (opcional)
        """
        self.config = config_manager or config
        self.results: Dict[str, HealthCheckResult] = {}
        self._component_status: Dict[str, HealthStatus] = {}
    
    async def check_all(self) -> Dict[str, Any]:
        """
        Ejecuta todos los checks de salud (versión async).
        
        Returns:
            Dict con los resultados de todos los checks
        """
        logger.info("🧪 Ejecutando health checks del sistema (async)...")
        
        checks = [
            ("python_environment", self._check_python_environment, True),
            ("system_resources", self._check_system_resources, True),
            ("configuration", self._check_configuration, True),
            ("file_system", self._check_file_system, True),
            ("dependencies", self._check_dependencies, True),
            ("network_basic", self._check_network_basic, False),
        ]
        
        all_healthy = True
        any_warning = False
        
        for name, check_func, critical in checks:
            try:
                result = await self._run_check(name, check_func, critical)
                self.results[name] = result
                
                if result.status == HealthStatus.UNHEALTHY:
                    all_healthy = False
                    logger.error(f"❌ {name}: {result.message}")
                elif result.status == HealthStatus.WARNING:
                    any_warning = True
                    logger.warning(f"⚠️  {name}: {result.message}")
                else:
                    logger.debug(f"✅ {name}: {result.message}")
                    
            except Exception as e:
                error_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.ERROR,
                    message=f"Error durante el check: {str(e)}",
                    details={"error": str(e), "traceback": str(e.__traceback__)},
                    critical=critical
                )
                self.results[name] = error_result
                all_healthy = False
                logger.error(f"💥 {name}: Error - {e}")
        
        overall_status = HealthStatus.HEALTHY
        if not all_healthy:
            overall_status = HealthStatus.UNHEALTHY
        elif any_warning:
            overall_status = HealthStatus.WARNING
        
        return {
            "overall": overall_status == HealthStatus.HEALTHY,
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": [result.to_dict() for result in self.results.values()],
            "summary": self._generate_summary()
        }
    
    def check_all_sync(self) -> Dict[str, Any]:
        """
        Versión síncrona de check_all.
        
        Returns:
            Dict con los resultados de todos los checks
        """
        logger.info("🧪 Ejecutando health checks del sistema (sync)...")
        
        checks = [
            ("python_environment", self._check_python_environment_sync, True),
            ("system_resources", self._check_system_resources_sync, True),
            ("configuration", self._check_configuration_sync, True),
            ("file_system", self._check_file_system_sync, True),
            ("dependencies", self._check_dependencies_sync, True),
        ]
        
        all_healthy = True
        any_warning = False
        
        for name, check_func, critical in checks:
            try:
                result = check_func()
                self.results[name] = result
                
                if result.status == HealthStatus.UNHEALTHY:
                    all_healthy = False
                    logger.error(f"❌ {name}: {result.message}")
                elif result.status == HealthStatus.WARNING:
                    any_warning = True
                    logger.warning(f"⚠️  {name}: {result.message}")
                else:
                    logger.debug(f"✅ {name}: {result.message}")
                    
            except Exception as e:
                error_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.ERROR,
                    message=f"Error durante el check: {str(e)}",
                    details={"error": str(e)},
                    critical=critical
                )
                self.results[name] = error_result
                all_healthy = False
                logger.error(f"💥 {name}: Error - {e}")
        
        overall_status = HealthStatus.HEALTHY
        if not all_healthy:
            overall_status = HealthStatus.UNHEALTHY
        elif any_warning:
            overall_status = HealthStatus.WARNING
        
        return {
            "overall": all_healthy,
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": [result.to_dict() for result in self.results.values()],
            "summary": self._generate_summary()
        }
    
    async def _run_check(
        self,
        name: str,
        check_func: AsyncHealthCheck,
        critical: bool
    ) -> HealthCheckResult:
        """Ejecuta un check individual de forma asíncrona."""
        result = await check_func()
        result.critical = critical
        return result
    
    # Métodos de check asíncronos
    async def _check_python_environment(self) -> HealthCheckResult:
        """Verifica el entorno de Python."""
        python_version = sys.version_info
        
        details: Dict[str, Any] = {
            "version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "interpreter": sys.executable,
            "python_path": sys.path[:3]  # Primeros 3 elementos
        }
        
        # Python 3.9+ requerido
        if python_version.major == 3 and python_version.minor >= 9:
            return HealthCheckResult(
                name="python_environment",
                status=HealthStatus.HEALTHY,
                message=f"Python {python_version.major}.{python_version.minor}.{python_version.micro} compatible",
                details=details
            )
        else:
            return HealthCheckResult(
                name="python_environment",
                status=HealthStatus.UNHEALTHY,
                message=f"Python {python_version.major}.{python_version.minor} no compatible (requerido 3.9+)",
                details=details
            )
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Verifica los recursos del sistema."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details: Dict[str, Any] = {
                "cpu": {
                    "percent": cpu_percent,
                    "cores": psutil.cpu_count(),
                    "cores_logical": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent": memory.percent,
                    "used_gb": round(memory.used / (1024**3), 2)
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": disk.percent
                }
            }
            
            # Verificar límites
            warnings: List[str] = []
            if cpu_percent > 85:
                warnings.append(f"CPU alta: {cpu_percent}%")
            if memory.percent > 85:
                warnings.append(f"Memoria alta: {memory.percent}%")
            if disk.percent > 90:
                warnings.append(f"Disco casi lleno: {disk.percent}%")
            
            if warnings:
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.WARNING,
                    message=f"Recursos del sistema con advertencias: {', '.join(warnings)}",
                    details=details
                )
            
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.HEALTHY,
                message="Recursos del sistema OK",
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.WARNING,
                message=f"No se pudieron verificar recursos: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _check_configuration(self) -> HealthCheckResult:
        """Verifica la configuración del sistema."""
        required_keys: List[str] = [
            "system.name",
            "system.version",
            "environment",
            "storage.data_dir",
        ]
        
        missing:List[str] = []
        for key in required_keys:
            if self.config.get(key) is None:
                missing.append(key)
        
        details: Dict[str, Any] = {
            "environment": self.config.environment,
            "is_development": self.config.is_development,
            "is_production": self.config.is_production,
            "config_files": self._get_config_files(),
            "missing_keys": missing
        }
        
        if missing:
            return HealthCheckResult(
                name="configuration",
                status=HealthStatus.UNHEALTHY,
                message=f"Configuración incompleta. Faltan: {', '.join(missing)}",
                details=details
            )
        
        return HealthCheckResult(
            name="configuration",
            status=HealthStatus.HEALTHY,
            message="Configuración válida y completa",
            details=details
        )
    
    async def _check_file_system(self) -> HealthCheckResult:
        """Verifica el sistema de archivos y permisos."""
        required_dirs: List[Path] = [
           Path(self.config.get("storage.data_dir", "./data")),  # Changed from Path("./data") to "./data"
            Path(self.config.get("storage.log_dir", "./logs")),  # Changed from Path("./logs") to "./logs"
        ]
        
        dirs_status: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []
        
        for dir_path in required_dirs:
            dir_status: Dict[str, Any] = {
                "path": str(dir_path),
                "exists": False,
                "writable": False
            }
            
            try:
                # Verificar o crear directorio
                dir_path.mkdir(parents=True, exist_ok=True)
                dir_status["exists"] = True
                
                # Verificar permisos de escritura
                test_file = dir_path / ".health_check.tmp"
                test_file.write_text(f"Test {datetime.now().isoformat()}")
                test_file.unlink()
                dir_status["writable"] = True
                
            except PermissionError as e:
                errors.append(f"Permiso denegado en {dir_path}: {e}")
                dir_status["error"] = str(e)
            except Exception as e:
                errors.append(f"Error en {dir_path}: {e}")
                dir_status["error"] = str(e)
            
            dirs_status[dir_path.name] = dir_status
        
        details: Dict[str, Any] = {
            "directories": dirs_status,
            "errors": errors
        }
        
        if errors:
            return HealthCheckResult(
                name="file_system",
                status=HealthStatus.UNHEALTHY,
                message=f"Problemas con sistema de archivos: {len(errors)} errores",
                details=details
            )
        
        return HealthCheckResult(
            name="file_system",
            status=HealthStatus.HEALTHY,
            message="Sistema de archivos OK",
            details=details
        )
    
    async def _check_dependencies(self) -> HealthCheckResult:
        """Verifica dependencias críticas."""
        critical_deps: List[Tuple[str, str]] = [
            ("pydantic", "2.0.0"),
            ("loguru", "0.7.0"),
            ("pyyaml", "6.0"),
            ("python-dotenv", "1.0.0"),
            ("rich", "13.0.0"),
            ("click", "8.1.0"),
        ]
        
        deps_status: Dict[str, Dict[str, str]] = {}
        missing: List[str] = []
        outdated: List[str] = []
        
        for dep_name, min_version in critical_deps:
            try:
                module = __import__(dep_name.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                
                status = "ok" if version != "unknown" else "unknown"
                
                deps_status[dep_name] = {
                    "version": version,
                    "required": min_version,
                    "status": status
                }
                
                if status == "unknown":
                    outdated.append(dep_name)
                    
            except ImportError:
                missing.append(dep_name)
                deps_status[dep_name] = {
                    "version": "missing",
                    "required": min_version,
                    "status": "missing"
                }
        
        details: Dict[str, Any] = {
            "dependencies": deps_status,
            "missing": missing,
            "outdated": outdated
        }
        
        if missing:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"Dependencias faltantes: {', '.join(missing)}",
                details=details
            )
        
        if outdated:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.WARNING,
                message=f"Dependencias con versión desconocida: {', '.join(outdated)}",
                details=details
            )
        
        return HealthCheckResult(
            name="dependencies",
            status=HealthStatus.HEALTHY,
            message="Todas las dependencias críticas están instaladas",
            details=details
        )
    
    async def _check_network_basic(self) -> HealthCheckResult:
        """Verificación básica de red."""
        try:
            # Verificar que podemos resolver localhost
            socket.gethostbyname("localhost")
            
            details: Dict[str, Any] = {
                "localhost_resolvable": True,
                "hostname": socket.gethostname(),
                "note": "Verificación básica de red completada"
            }
            
            return HealthCheckResult(
                name="network_basic",
                status=HealthStatus.HEALTHY,
                message="Conectividad de red básica OK",
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="network_basic",
                status=HealthStatus.WARNING,
                message=f"Problema con conectividad de red: {str(e)}",
                details={"error": str(e)}
            )
    
    # Métodos de check síncronos (para compatibilidad)
    def _check_python_environment_sync(self) -> HealthCheckResult:
        """Versión síncrona de check_python_environment."""
        python_version = sys.version_info
        
        details: Dict[str, Any] = {
            "version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "platform": platform.platform(),
            "interpreter": sys.executable
        }
        
        if python_version.major == 3 and python_version.minor >= 9:
            return HealthCheckResult(
                name="python_environment",
                status=HealthStatus.HEALTHY,
                message=f"Python {python_version.major}.{python_version.minor} compatible",
                details=details
            )
        else:
            return HealthCheckResult(
                name="python_environment",
                status=HealthStatus.UNHEALTHY,
                message=f"Python {python_version.major}.{python_version.minor} no compatible",
                details=details
            )
    
    def _check_system_resources_sync(self) -> HealthCheckResult:
        """Versión síncrona de check_system_resources."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            details: Dict[str, Any] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2)
            }
            
            if cpu_percent > 90 or memory.percent > 90:
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.WARNING,
                    message=f"Recursos altos (CPU: {cpu_percent}%, Mem: {memory.percent}%)",
                    details=details
                )
            
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.HEALTHY,
                message=f"Recursos OK (CPU: {cpu_percent}%, Mem: {memory.percent}%)",
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.WARNING,
                message=f"Error verificando recursos: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_configuration_sync(self) -> HealthCheckResult:
        """Versión síncrona de check_configuration."""
        return HealthCheckResult(
            name="configuration",
            status=HealthStatus.HEALTHY,
            message=f"Configuración cargada para entorno: {self.config.environment}",
            details={"environment": self.config.environment}
        )
    
    def _check_file_system_sync(self) -> HealthCheckResult:
        """Versión síncrona de check_file_system."""
        try:
            data_dir = Path(self.config.get("storage.data_dir", "./data"))
            log_dir = Path(self.config.get("storage.log_dir", "./logs"))
            
            data_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            return HealthCheckResult(
                name="file_system",
                status=HealthStatus.HEALTHY,
                message="Directorios creados/verificados",
                details={
                    "data_dir": str(data_dir),
                    "log_dir": str(log_dir)
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name="file_system",
                status=HealthStatus.UNHEALTHY,
                message=f"Error con directorios: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_dependencies_sync(self) -> HealthCheckResult:
        """Versión síncrona de check_dependencies."""
        
        deps = ["pydantic", "loguru", "rich", "click"]
        missing: List[str] = []
        
        for dep in deps:
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        if missing:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"Dependencias faltantes: {', '.join(missing)}",  # Fixed: removed extra { }
                details={"missing": missing},
            )
        return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.HEALTHY,
                message="Dependencias básicas instaladas",  # Fixed typo
                details={"checked": deps}
            )
    
    def _get_config_files(self) -> List[str]:
        """Obtiene lista de archivos de configuración cargados."""
        config_files: List[Path] = [
            Path("config/system_config.yaml"),
            Path("config/agent_config.yaml"),
            Path(".env"),
        ]
        
        loaded: List[str] = []
        for file_path in config_files:
            if file_path.exists():
                loaded.append(str(file_path))
        
        return loaded
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Genera un resumen de los checks."""
        if not self.results:
            return {}
        
        total = len(self.results)
        healthy = sum(1 for r in self.results.values() if r.status == HealthStatus.HEALTHY)
        warning = sum(1 for r in self.results.values() if r.status == HealthStatus.WARNING)
        unhealthy = sum(1 for r in self.results.values() if r.status == HealthStatus.UNHEALTHY)
        error = sum(1 for r in self.results.values() if r.status == HealthStatus.ERROR)
        
        critical_failed = any(
            r.critical and r.status in [HealthStatus.UNHEALTHY, HealthStatus.ERROR]
            for r in self.results.values()
        )
        
        return {
            "total_checks": total,
            "healthy": healthy,
            "warnings": warning,
            "unhealthy": unhealthy,
            "errors": error,
            "critical_failed": critical_failed,
            "success_rate": round((healthy / total) * 100, 2) if total > 0 else 0
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del health checker.
        
        Returns:
            Dict con el estado actual
        """
        if not self.results:
            return {"status": "not_checked", "message": "Health check no ejecutado"}
        
        summary = self._generate_summary()
        
        overall_status = HealthStatus.HEALTHY
        if summary.get("critical_failed", False):
            overall_status = HealthStatus.UNHEALTHY
        elif summary.get("warnings", 0) > 0:
            overall_status = HealthStatus.WARNING
        
        return {
            "status": overall_status.value,
            "summary": summary,
            "last_check": max(r.timestamp for r in self.results.values()).isoformat() if self.results else None,
            "checks_count": len(self.results)
        }
    
    def print_detailed_report(self) -> str:
        """Genera un reporte detallado de los health checks."""
        if not self.results:
            return "No hay resultados de health check disponibles."
        
        report_lines: List[str] = []
        report_lines.append("=" * 80)
        report_lines.append("REPORTE DETALLADO DE HEALTH CHECK")
        report_lines.append("=" * 80)
        
        summary = self._generate_summary()
        report_lines.append(f"Resumen: {summary['healthy']}✅ {summary['warnings']}⚠️ {summary['unhealthy']}❌ {summary['errors']}💥")
        report_lines.append(f"Checks totales: {summary['total_checks']} | Tasa éxito: {summary['success_rate']}%")
        report_lines.append("-" * 80)
        
        for result in sorted(self.results.values(), key=lambda x: (not x.critical, x.name)):
            status_icon = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.WARNING: "⚠️",
                HealthStatus.UNHEALTHY: "❌",
                HealthStatus.ERROR: "💥"
            }.get(result.status, "❓")
            
            critical_marker = "⚡" if result.critical else " "
            report_lines.append(f"{status_icon}{critical_marker} {result.name}: {result.message}")
            if result.details and "error" in result.details:
                report_lines.append(f"    Error: {result.details['error']}")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


# Instancia global para uso conveniente
health_checker = SystemHealthChecker()