"""
Excepciones personalizadas del sistema Project Brain.
"""

class BrainException(Exception):
    """Excepción base para todos los errores del sistema."""
    
    def __init__(self, message: str, code: str = "BRAIN_ERROR", details: dict = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}
    
    def __str__(self):
        return f"[{self.code}] {super().__str__()}"

class ValidationError(BrainException):
    """Error de validación de datos."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        details = {}
        if field:
            details['field'] = field
        if value:
            details['value'] = value
        
        super().__init__(message, "VALIDATION_ERROR", details)

class ConfigurationError(BrainException):
    """Error de configuración."""
    
    def __init__(self, message: str, config_file: str = None):
        details = {}
        if config_file:
            details['config_file'] = config_file
        
        super().__init__(message, "CONFIGURATION_ERROR", details)

class PluginError(BrainException):
    """Error relacionado con plugins."""
    
    def __init__(self, message: str, plugin_name: str = None):
        details = {}
        if plugin_name:
            details['plugin_name'] = plugin_name
        
        super().__init__(message, "PLUGIN_ERROR", details)

class WorkflowError(BrainException):
    """Error en flujo de trabajo."""
    
    def __init__(self, message: str, workflow_id: str = None, step: str = None):
        details = {}
        if workflow_id:
            details['workflow_id'] = workflow_id
        if step:
            details['step'] = step
        
        super().__init__(message, "WORKFLOW_ERROR", details)

class HealthCheckError(BrainException):
    """Error en verificación de salud."""
    
    def __init__(self, message: str, component: str = None):
        details = {}
        if component:
            details['component'] = component
        
        super().__init__(message, "HEALTH_CHECK_ERROR", details)

class TimeoutError(BrainException):
    """Error de timeout."""
    
    def __init__(self, message: str, operation_id: str = None, timeout_seconds: int = None):
        details = {}
        if operation_id:
            details['operation_id'] = operation_id
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        
        super().__init__(message, "TIMEOUT_ERROR", details)

class DependencyError(BrainException):
    """Error de dependencia."""
    
    def __init__(self, message: str, dependency: str = None):
        details = {}
        if dependency:
            details['dependency'] = dependency
        
        super().__init__(message, "DEPENDENCY_ERROR", details)

class ResourceError(BrainException):
    """Error de recursos (memoria, CPU, etc.)."""
    
    def __init__(self, message: str, resource_type: str = None, limit: Any = None):
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if limit:
            details['limit'] = limit
        
        super().__init__(message, "RESOURCE_ERROR", details)

class AnalysisError(BrainException):
    """Error en análisis de código."""
    
    def __init__(self, message: str, file_path: str = None, language: str = None):
        details = {}
        if file_path:
            details['file_path'] = file_path
        if language:
            details['language'] = language
        
        super().__init__(message, "ANALYSIS_ERROR", details)

class AgentError(BrainException):
    """Error en agente."""
    
    def __init__(self, message: str, agent_id: str = None, agent_type: str = None):
        details = {}
        if agent_id:
            details['agent_id'] = agent_id
        if agent_type:
            details['agent_type'] = agent_type
        
        super().__init__(message, "AGENT_ERROR", details)

class LearningError(BrainException):
    """Error en aprendizaje."""
    
    def __init__(self, message: str, learning_type: str = None):
        details = {}
        if learning_type:
            details['learning_type'] = learning_type
        
        super().__init__(message, "LEARNING_ERROR", details)