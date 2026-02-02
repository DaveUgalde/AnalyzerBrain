#!/usr/bin/env python3
"""
Sistema jerárquico de excepciones de ANALYZERBRAIN.

Este módulo define un sistema de excepciones estructurado y tipado para
todo el proyecto ANALYZERBRAIN. Proporciona:

- Códigos de error estandarizados
- Severidad del error
- Detalles estructurados y serializables
- Encadenamiento de causas (exception chaining)
- Representación lista para APIs, logs y observabilidad

Jerarquía principal:

    AnalyzerBrainError
    ├── ConfigurationError
    ├── ValidationError
    ├── IndexerError
    ├── GraphError
    ├── AgentError
    ├── APIError
    └── ProjectAnalysisError

Diseño:
- Todas las excepciones heredan de AnalyzerBrainError
- Todas pueden serializarse a dict (to_dict)
- Pensadas para uso en backend, API y pipelines async

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import datetime


class ErrorSeverity(Enum):
    """Severidad del error."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCode(Enum):
    """Códigos de error estandarizados."""
    # Errores generales
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND_ERROR = "NOT_FOUND_ERROR"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    
    # Errores de módulos específicos
    INDEXER_ERROR = "INDEXER_ERROR"
    GRAPH_ERROR = "GRAPH_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    API_ERROR = "API_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    
    # Errores de negocio
    PROJECT_ANALYSIS_ERROR = "PROJECT_ANALYSIS_ERROR"
    QUERY_EXECUTION_ERROR = "QUERY_EXECUTION_ERROR"
    LEARNING_ERROR = "LEARNING_ERROR"


@dataclass
class ErrorDetail:
    """Detalle estructurado de un error."""
    field: Optional[str] = None
    message: str = ""
    value: Any = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnalyzerBrainError(Exception):
    """Excepción base para todos los errores del sistema."""
    
    def __init__(
        self,
        message: str,
        error_code: Union[str, ErrorCode] = ErrorCode.INTERNAL_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = ErrorCode(error_code) if isinstance(error_code, str) else error_code
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.datetime.now().isoformat()
        
        # Construir mensaje completo
        full_message = f"[{self.error_code.value}] {message}"
        if cause:
            full_message += f" | Causa: {type(cause).__name__}: {str(cause)}"
        
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la excepción a diccionario para serialización."""
        result: Dict[str, Any] = {
            "error": self.error_code.value,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "details": self.details
        }
        
        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause)
            }
        
        return result
    
    def __str__(self) -> str:
        return self.message


class ConfigurationError(AnalyzerBrainError):
    """Error en la configuración del sistema."""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details,
            cause=cause
        )


class ValidationError(AnalyzerBrainError):
    """Error de validación de datos."""
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        value_type: Optional[str] = None,
        actual_length: Optional[float|int] = None,
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["value"] = value
        if value_type:
            error_details["value_type"] = value_type
        if actual_length:
            error_details["actual_length"] = value_type
        if suggestion:
            error_details["suggestion"] = suggestion
        
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )


class IndexerError(AnalyzerBrainError):
    """Error durante la indexación de proyectos."""
    def __init__(
        self,
        message: str,
        project_path: Optional[str] = None,
        file_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if project_path:
            error_details["project_path"] = project_path
        if file_path:
            error_details["file_path"] = file_path
        
        super().__init__(
            message=message,
            error_code=ErrorCode.INDEXER_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )


class GraphError(AnalyzerBrainError):
    """Error en el grafo de conocimiento."""
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        node_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if query:
            error_details["query"] = query
        if node_id:
            error_details["node_id"] = node_id
        
        super().__init__(
            message=message,
            error_code=ErrorCode.GRAPH_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )


class AgentError(AnalyzerBrainError):
    """Error en un agente."""
    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if agent_name:
            error_details["agent_name"] = agent_name
        if task_type:
            error_details["task_type"] = task_type
        
        super().__init__(
            message=message,
            error_code=ErrorCode.AGENT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )


class APIError(AnalyzerBrainError):
    """Error en la API."""
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if endpoint:
            error_details["endpoint"] = endpoint
        if status_code:
            error_details["status_code"] = status_code
        
        super().__init__(
            message=message,
            error_code=ErrorCode.API_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )


class ProjectAnalysisError(AnalyzerBrainError):
    """Error durante el análisis de un proyecto."""
    def __init__(
        self,
        message: str,
        project_path: Optional[str] = None,
        analysis_step: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        error_details = details or {}
        if project_path:
            error_details["project_path"] = project_path
        if analysis_step:
            error_details["analysis_step"] = analysis_step
        
        super().__init__(
            message=message,
            error_code=ErrorCode.PROJECT_ANALYSIS_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=error_details,
            cause=cause
        )