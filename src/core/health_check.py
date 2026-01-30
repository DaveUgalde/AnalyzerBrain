"""
HealthCheck - Sistema de verificación de salud del sistema.
"""

from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime, timedelta
from .exceptions import BrainException, HealthCheckError

class HealthStatus(Enum):
    """Estados de salud."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheckResult:
    """Resultado de una verificación de salud."""
    
    def __init__(self, component: str, status: HealthStatus, 
                 message: str = "", details: Dict[str, Any] = None):
        self.component = component
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a dict."""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

class HealthCheck:
    """Sistema de verificación de salud."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.required_components: List[str] = []
        self.last_full_check: Optional[datetime] = None
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Verifica la salud de todo el sistema.
        
        Returns:
            Dict con estado de salud del sistema
        """
        all_results = {}
        unhealthy_count = 0
        degraded_count = 0
        
        for component_name, check_func in self.checks.items():
            try:
                result = check_func()
                
                if isinstance(result, HealthCheckResult):
                    health_result = result
                elif isinstance(result, dict):
                    health_result = HealthCheckResult(
                        component=component_name,
                        status=HealthStatus(result.get("status", "unhealthy")),
                        message=result.get("message", ""),
                        details=result.get("details", {})
                    )
                else:
                    health_result = HealthCheckResult(
                        component=component_name,
                        status=HealthStatus.UNHEALTHY,
                        message="Invalid check result format"
                    )
                
            except Exception as e:
                health_result = HealthCheckResult(
                    component=component_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}"
                )
            
            self.results[component_name] = health_result
            all_results[component_name] = health_result.to_dict()
            
            if health_result.status == HealthStatus.UNHEALTHY:
                unhealthy_count += 1
            elif health_result.status == HealthStatus.DEGRADED:
                degraded_count += 1
        
        # Determinar estado general
        if unhealthy_count > 0:
            system_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            system_status = HealthStatus.DEGRADED
        else:
            system_status = HealthStatus.HEALTHY
        
        self.last_full_check = datetime.now()
        
        return {
            "status": system_status.value,
            "timestamp": self.last_full_check.isoformat(),
            "checks": all_results,
            "summary": {
                "total_checks": len(self.checks),
                "healthy": len(self.checks) - unhealthy_count - degraded_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            }
        }
    
    def check_component_health(self, component_name: str) -> HealthCheckResult:
        """
        Verifica la salud de un componente específico.
        
        Args:
            component_name: Nombre del componente
            
        Returns:
            HealthCheckResult: Resultado de la verificación
        """
        if component_name not in self.checks:
            return HealthCheckResult(
                component=component_name,
                status=HealthStatus.UNHEALTHY,
                message="Component not registered for health checks"
            )
        
        try:
            result = self.checks[component_name]()
            
            if isinstance(result, HealthCheckResult):
                health_result = result
            elif isinstance(result, dict):
                health_result = HealthCheckResult(
                    component=component_name,
                    status=HealthStatus(result.get("status", "unhealthy")),
                    message=result.get("message", ""),
                    details=result.get("details", {})
                )
            else:
                health_result = HealthCheckResult(
                    component=component_name,
                    status=HealthStatus.UNHEALTHY,
                    message="Invalid check result format"
                )
            
        except Exception as e:
            health_result = HealthCheckResult(
                component=component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}"
            )
        
        self.results[component_name] = health_result
        return health_result
    
    def register_health_check(self, component_name: str, 
                            check_func: Callable) -> None:
        """
        Registra una verificación de salud.
        
        Args:
            component_name: Nombre del componente
            check_func: Función que ejecuta la verificación
        """
        if component_name in self.checks:
            raise HealthCheckError(f"Health check for {component_name} already registered")
        
        self.checks[component_name] = check_func
    
    def run_all_checks(self) -> List[HealthCheckResult]:
        """
        Ejecuta todas las verificaciones registradas.
        
        Returns:
            Lista de resultados
        """
        results = []
        
        for component_name in self.checks:
            result = self.check_component_health(component_name)
            results.append(result)
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado de salud actual.
        
        Returns:
            Dict con estado de salud
        """
        if not self.results:
            return {
                "status": "unknown",
                "message": "No health checks performed yet",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calcular estado basado en últimos resultados
        unhealthy_count = 0
        degraded_count = 0
        
        for result in self.results.values():
            if result.status == HealthStatus.UNHEALTHY:
                unhealthy_count += 1
            elif result.status == HealthStatus.DEGRADED:
                degraded_count += 1
        
        if unhealthy_count > 0:
            system_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            system_status = HealthStatus.DEGRADED
        else:
            system_status = HealthStatus.HEALTHY
        
        # Verificar si los resultados están desactualizados
        is_stale = False
        if self.last_full_check:
            stale_threshold = datetime.now() - timedelta(minutes=5)
            if self.last_full_check < stale_threshold:
                is_stale = True
        
        return {
            "status": system_status.value,
            "stale": is_stale,
            "last_check": self.last_full_check.isoformat() if self.last_full_check else None,
            "component_count": len(self.results),
            "unhealthy_count": unhealthy_count,
            "degraded_count": degraded_count
        }
    
    def set_healthy(self, component_name: str, message: str = "") -> None:
        """
        Marca un componente como saludable manualmente.
        
        Args:
            component_name: Nombre del componente
            message: Mensaje opcional
        """
        self.results[component_name] = HealthCheckResult(
            component=component_name,
            status=HealthStatus.HEALTHY,
            message=message or "Manually set to healthy"
        )
    
    def set_unhealthy(self, component_name: str, message: str = "") -> None:
        """
        Marca un componente como no saludable manualmente.
        
        Args:
            component_name: Nombre del componente
            message: Mensaje opcional
        """
        self.results[component_name] = HealthCheckResult(
            component=component_name,
            status=HealthStatus.UNHEALTHY,
            message=message or "Manually set to unhealthy"
        )
    
    def add_required_component(self, component_name: str) -> None:
        """Añade un componente requerido."""
        if component_name not in self.required_components:
            self.required_components.append(component_name)
    
    def validate_required_components(self) -> List[str]:
        """
        Valida que todos los componentes requeridos estén saludables.
        
        Returns:
            Lista de componentes requeridos no saludables
        """
        unhealthy = []
        
        for component in self.required_components:
            if component in self.results:
                result = self.results[component]
                if result.status != HealthStatus.HEALTHY:
                    unhealthy.append(component)
            else:
                unhealthy.append(component)
        
        return unhealthy