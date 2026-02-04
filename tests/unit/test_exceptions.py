#!/usr/bin/env python3
"""
Tests unitarios para el sistema de excepciones de ANALYZERBRAIN.
"""

import pytest
from datetime import datetime
from typing import Any, Dict

from src.core.exceptions import (
    AnalyzerBrainError,
    ConfigurationError,
    ValidationError,
    IndexerError,
    GraphError,
    AgentError,
    APIError,
    ProjectAnalysisError,
    ErrorSeverity,
    ErrorCode,
    ErrorDetail,
)

# -------------------------------------------------------------------
# Tests de ErrorSeverity
# -------------------------------------------------------------------


def test_error_severity_enum():
    """Verifica que el enum ErrorSeverity tenga los valores correctos."""
    assert ErrorSeverity.LOW.value == "low"
    assert ErrorSeverity.MEDIUM.value == "medium"
    assert ErrorSeverity.HIGH.value == "high"
    assert ErrorSeverity.CRITICAL.value == "critical"


def test_error_severity_membership():
    """Verifica que los miembros del enum sean accesibles."""
    assert ErrorSeverity.LOW in ErrorSeverity
    assert ErrorSeverity.MEDIUM in ErrorSeverity
    assert ErrorSeverity.HIGH in ErrorSeverity
    assert ErrorSeverity.CRITICAL in ErrorSeverity


# -------------------------------------------------------------------
# Tests de ErrorCode
# -------------------------------------------------------------------


def test_error_code_enum():
    """Verifica que el enum ErrorCode tenga los códigos principales."""
    expected_codes = {
        "INTERNAL_ERROR",
        "CONFIGURATION_ERROR",
        "VALIDATION_ERROR",
        "NOT_FOUND_ERROR",
        "PERMISSION_ERROR",
        "INDEXER_ERROR",
        "GRAPH_ERROR",
        "AGENT_ERROR",
        "API_ERROR",
        "EMBEDDING_ERROR",
        "MEMORY_ERROR",
        "PROJECT_ANALYSIS_ERROR",
        "QUERY_EXECUTION_ERROR",
        "LEARNING_ERROR",
        "SYSTEM_STATE_ERROR",
        "EVENT_BUS_ERROR",
    }

    actual_codes = {code.value for code in ErrorCode}
    assert actual_codes == expected_codes


# -------------------------------------------------------------------
# Tests de ErrorDetail
# -------------------------------------------------------------------


def test_error_detail_creation():
    """Verifica la creación de ErrorDetail."""
    detail = ErrorDetail(
        field="username",
        message="El nombre de usuario es requerido",
        value=None,
        suggestion="Proporcione un nombre de usuario válido",
    )

    assert detail.field == "username"
    assert detail.message == "El nombre de usuario es requerido"
    assert detail.value is None
    assert detail.suggestion == "Proporcione un nombre de usuario válido"


def test_error_detail_to_dict():
    """Verifica la conversión de ErrorDetail a diccionario."""
    detail = ErrorDetail(
        field="email",
        message="Email inválido",
        value="not-an-email",
        suggestion="Ingrese un email válido",
    )

    result = detail.to_dict()
    expected = {
        "field": "email",
        "message": "Email inválido",
        "value": "not-an-email",
        "suggestion": "Ingrese un email válido",
    }

    assert result == expected


# -------------------------------------------------------------------
# Tests de AnalyzerBrainError (excepción base)
# -------------------------------------------------------------------


def test_analyzer_brain_error_creation():
    """Verifica la creación básica de AnalyzerBrainError."""
    error = AnalyzerBrainError(
        message="Error interno del sistema",
        error_code=ErrorCode.INTERNAL_ERROR,
        severity=ErrorSeverity.CRITICAL,
    )

    assert error.message == "Error interno del sistema"
    assert error.error_code == ErrorCode.INTERNAL_ERROR
    assert error.severity == ErrorSeverity.CRITICAL
    assert isinstance(error.timestamp, str)
    assert error.cause is None
    assert error.details == {}


def test_analyzer_brain_error_with_details_and_cause():
    """Verifica AnalyzerBrainError con detalles y causa."""
    cause = ValueError("Valor inválido")
    details = {"context": "procesando archivo", "line": 42}

    # Usar un código de error existente
    error = AnalyzerBrainError(
        message="Error al procesar datos",
        error_code=ErrorCode.INTERNAL_ERROR,  # Cambiado de "PROCESSING_ERROR"
        severity=ErrorSeverity.HIGH,
        details=details,
        cause=cause,
    )

    assert error.message == "Error al procesar datos"
    assert error.error_code == ErrorCode.INTERNAL_ERROR
    assert error.severity == ErrorSeverity.HIGH
    assert error.details == details
    assert error.cause == cause


def test_analyzer_brain_error_to_dict():
    """Verifica la serialización a diccionario."""
    cause = TypeError("Tipo incorrecto")
    details = {"step": "validación", "file": "config.yaml"}

    error = AnalyzerBrainError(
        message="Fallo en la validación",
        error_code=ErrorCode.VALIDATION_ERROR,
        severity=ErrorSeverity.MEDIUM,
        details=details,
        cause=cause,
    )

    result = error.to_dict()

    assert result["error"] == "VALIDATION_ERROR"
    assert result["message"] == "Fallo en la validación"
    assert result["severity"] == "medium"
    assert result["timestamp"] == error.timestamp
    assert result["details"] == details
    assert result["cause"]["type"] == "TypeError"
    assert result["cause"]["message"] == "Tipo incorrecto"


def test_analyzer_brain_error_str_representation():
    """Verifica la representación en string de AnalyzerBrainError."""
    error = AnalyzerBrainError(message="Mensaje de error", error_code=ErrorCode.INTERNAL_ERROR)

    # __str__ ahora devuelve el mensaje completo que incluye el código
    assert "[INTERNAL_ERROR] Mensaje de error" in str(error)

    # Con causa
    cause = RuntimeError("Causa raíz")
    error_with_cause = AnalyzerBrainError(
        message="Error con causa", error_code=ErrorCode.INTERNAL_ERROR, cause=cause
    )

    # Verificar que el string incluye la causa
    error_str = str(error_with_cause)
    assert "Error con causa" in error_str
    assert "RuntimeError" in error_str
    assert "Causa raíz" in error_str


def test_analyzer_brain_error_chain():
    """Verifica que se pueda capturar la cadena de excepciones."""
    try:
        try:
            raise ValueError("Error interno")
        except ValueError as e:
            raise AnalyzerBrainError(message="Error envuelto", cause=e) from e
    except AnalyzerBrainError as e:
        assert e.cause is not None
        assert isinstance(e.cause, ValueError)
        assert str(e.cause) == "Error interno"


# -------------------------------------------------------------------
# Tests de ConfigurationError
# -------------------------------------------------------------------


def test_configuration_error_creation():
    """Verifica la creación de ConfigurationError."""
    error = ConfigurationError(
        message="Configuración inválida",
        details={"file": "config.yaml", "key": "database.url"},
        cause=FileNotFoundError("Archivo no encontrado"),
    )

    assert error.message == "Configuración inválida"
    assert error.error_code == ErrorCode.CONFIGURATION_ERROR
    assert error.severity == ErrorSeverity.HIGH
    assert error.details["file"] == "config.yaml"
    assert error.details["key"] == "database.url"
    assert isinstance(error.cause, FileNotFoundError)


def test_configuration_error_inheritance():
    """Verifica que ConfigurationError herede de AnalyzerBrainError."""
    error = ConfigurationError("Error de configuración")

    assert isinstance(error, AnalyzerBrainError)
    assert issubclass(ConfigurationError, AnalyzerBrainError)


# -------------------------------------------------------------------
# Tests de ValidationError
# -------------------------------------------------------------------


def test_validation_error_creation():
    """Verifica la creación de ValidationError con todos los parámetros."""
    error = ValidationError(
        message="El campo es requerido",
        field="username",
        value="",
        value_type="string",
        actual_length=0,
        suggestion="Ingrese un nombre de usuario",
        details={"min_length": 3},
    )

    assert error.message == "El campo es requerido"
    assert error.error_code == ErrorCode.VALIDATION_ERROR
    assert error.severity == ErrorSeverity.MEDIUM
    assert error.details["field"] == "username"
    assert error.details["value"] == ""
    assert error.details["value_type"] == "string"
    assert error.details["actual_length"] == 0  # ¡Ahora debería estar correcto!
    assert error.details["suggestion"] == "Ingrese un nombre de usuario"
    assert error.details["min_length"] == 3


def test_validation_error_inheritance():
    """Verifica que ValidationError herede de AnalyzerBrainError."""
    error = ValidationError("Error de validación")

    assert isinstance(error, AnalyzerBrainError)
    assert issubclass(ValidationError, AnalyzerBrainError)


def test_validation_error_minimal_creation():
    """Verifica la creación mínima de ValidationError."""
    error = ValidationError("Valor inválido")

    assert error.message == "Valor inválido"
    assert error.error_code == ErrorCode.VALIDATION_ERROR
    assert error.severity == ErrorSeverity.MEDIUM
    assert error.details == {}


# -------------------------------------------------------------------
# Tests de IndexerError
# -------------------------------------------------------------------


def test_indexer_error_creation():
    """Verifica la creación de IndexerError."""
    # Nota: UnicodeDecodeError necesita argumentos específicos
    try:
        b'\x80'.decode('utf-8')
    except UnicodeDecodeError as e:
        cause = e

    error = IndexerError(
        message="No se pudo indexar el archivo",
        project_path="/proyectos/mi_proyecto",
        file_path="/proyectos/mi_proyecto/src/main.py",
        details={"reason": "encoding error"},
        cause=cause,
    )

    assert error.message == "No se pudo indexar el archivo"
    assert error.error_code == ErrorCode.INDEXER_ERROR
    assert error.severity == ErrorSeverity.MEDIUM
    assert error.details["project_path"] == "/proyectos/mi_proyecto"
    assert error.details["file_path"] == "/proyectos/mi_proyecto/src/main.py"
    assert error.details["reason"] == "encoding error"
    assert error.cause is not None


def test_indexer_error_inheritance():
    """Verifica que IndexerError herede de AnalyzerBrainError."""
    error = IndexerError("Error de indexación")

    assert isinstance(error, AnalyzerBrainError)
    assert issubclass(IndexerError, AnalyzerBrainError)


# -------------------------------------------------------------------
# Tests de GraphError
# -------------------------------------------------------------------


def test_graph_error_creation():
    """Verifica la creación de GraphError."""
    error = GraphError(
        message="Consulta de grafo fallida",
        query="MATCH (n) RETURN n",
        node_id="node_123",
        details={"cypher_error": "syntax error"},
        cause=ValueError("Consulta inválida"),
    )

    assert error.message == "Consulta de grafo fallida"
    assert error.error_code == ErrorCode.GRAPH_ERROR
    assert error.severity == ErrorSeverity.MEDIUM
    assert error.details["query"] == "MATCH (n) RETURN n"
    assert error.details["node_id"] == "node_123"
    assert error.details["cypher_error"] == "syntax error"
    assert isinstance(error.cause, ValueError)


def test_graph_error_inheritance():
    """Verifica que GraphError herede de AnalyzerBrainError."""
    error = GraphError("Error de grafo")

    assert isinstance(error, AnalyzerBrainError)
    assert issubclass(GraphError, AnalyzerBrainError)


# -------------------------------------------------------------------
# Tests de AgentError
# -------------------------------------------------------------------


def test_agent_error_creation():
    """Verifica la creación de AgentError."""
    error = AgentError(
        message="El agente falló en la tarea",
        agent_name="AnalyzerAgent",
        task_type="code_analysis",
        details={"reason": "timeout"},
        cause=TimeoutError("Tiempo agotado"),
    )

    assert error.message == "El agente falló en la tarea"
    assert error.error_code == ErrorCode.AGENT_ERROR
    assert error.severity == ErrorSeverity.MEDIUM
    assert error.details["agent_name"] == "AnalyzerAgent"
    assert error.details["task_type"] == "code_analysis"
    assert error.details["reason"] == "timeout"
    assert isinstance(error.cause, TimeoutError)


def test_agent_error_inheritance():
    """Verifica que AgentError herede de AnalyzerBrainError."""
    error = AgentError("Error de agente")

    assert isinstance(error, AnalyzerBrainError)
    assert issubclass(AgentError, AnalyzerBrainError)


# -------------------------------------------------------------------
# Tests de APIError
# -------------------------------------------------------------------


def test_api_error_creation():
    """Verifica la creación de APIError."""
    error = APIError(
        message="Endpoint no encontrado",
        endpoint="/api/v1/projects",
        status_code=404,
        details={"method": "GET", "client_ip": "192.168.1.1"},
        cause=ValueError("Ruta no existe"),
    )

    assert error.message == "Endpoint no encontrado"
    assert error.error_code == ErrorCode.API_ERROR
    assert error.severity == ErrorSeverity.MEDIUM
    assert error.details["endpoint"] == "/api/v1/projects"
    assert error.details["status_code"] == 404
    assert error.details["method"] == "GET"
    assert error.details["client_ip"] == "192.168.1.1"
    assert isinstance(error.cause, ValueError)


def test_api_error_inheritance():
    """Verifica que APIError herede de AnalyzerBrainError."""
    error = APIError("Error de API")

    assert isinstance(error, AnalyzerBrainError)
    assert issubclass(APIError, AnalyzerBrainError)


# -------------------------------------------------------------------
# Tests de ProjectAnalysisError
# -------------------------------------------------------------------


def test_project_analysis_error_creation():
    """Verifica la creación de ProjectAnalysisError."""
    error = ProjectAnalysisError(
        message="Análisis de proyecto fallido",
        project_path="/proyectos/complejo",
        analysis_step="dependency_analysis",
        details={"reason": "circular dependency"},
        cause=RecursionError("Dependencia circular detectada"),
    )

    assert error.message == "Análisis de proyecto fallido"
    assert error.error_code == ErrorCode.PROJECT_ANALYSIS_ERROR
    assert error.severity == ErrorSeverity.MEDIUM
    assert error.details["project_path"] == "/proyectos/complejo"
    assert error.details["analysis_step"] == "dependency_analysis"
    assert error.details["reason"] == "circular dependency"
    assert isinstance(error.cause, RecursionError)


def test_project_analysis_error_inheritance():
    """Verifica que ProjectAnalysisError herede de AnalyzerBrainError."""
    error = ProjectAnalysisError("Error de análisis")

    assert isinstance(error, AnalyzerBrainError)
    assert issubclass(ProjectAnalysisError, AnalyzerBrainError)


# -------------------------------------------------------------------
# Tests de jerarquía y polimorfismo
# -------------------------------------------------------------------


def test_exception_hierarchy():
    """Verifica la jerarquía completa de excepciones."""
    exceptions = [
        ConfigurationError("test"),
        ValidationError("test"),
        IndexerError("test"),
        GraphError("test"),
        AgentError("test"),
        APIError("test"),
        ProjectAnalysisError("test"),
    ]

    for exc in exceptions:
        assert isinstance(exc, AnalyzerBrainError)
        assert isinstance(exc, Exception)
        assert hasattr(exc, 'to_dict')
        assert hasattr(exc, 'error_code')
        assert hasattr(exc, 'severity')
        assert hasattr(exc, 'timestamp')


def test_exception_catching():
    """Verifica que se puedan capturar excepciones por tipo base."""
    try:
        raise ConfigurationError("Error de configuración")
    except AnalyzerBrainError as e:
        assert e.message == "Error de configuración"
        assert e.error_code == ErrorCode.CONFIGURATION_ERROR

    try:
        raise ValidationError("Error de validación")
    except AnalyzerBrainError as e:
        assert e.message == "Error de validación"
        assert e.error_code == ErrorCode.VALIDATION_ERROR


# -------------------------------------------------------------------
# Tests de serialización completa
# -------------------------------------------------------------------


def test_complete_serialization():
    """Verifica serialización completa de una excepción compleja."""
    root_cause = ValueError("Valor fuera de rango")
    details = {"min": 0, "max": 100, "actual": 150, "unit": "porcentaje"}

    error = ValidationError(
        message="Valor fuera del rango permitido",
        field="threshold",
        value=150,
        value_type="integer",
        actual_length=3,
        suggestion="Use un valor entre 0 y 100",
        details=details,
        cause=root_cause,
    )

    serialized = error.to_dict()

    assert serialized["error"] == "VALIDATION_ERROR"
    assert serialized["message"] == "Valor fuera del rango permitido"
    assert serialized["severity"] == "medium"
    assert serialized["details"]["field"] == "threshold"
    assert serialized["details"]["value"] == 150
    assert serialized["details"]["value_type"] == "integer"
    assert serialized["details"]["actual_length"] == 3  # ¡Ahora correcto!
    assert serialized["details"]["suggestion"] == "Use un valor entre 0 y 100"
    assert serialized["details"]["min"] == 0
    assert serialized["details"]["max"] == 100
    assert serialized["details"]["actual"] == 150
    assert serialized["details"]["unit"] == "porcentaje"
    assert serialized["cause"]["type"] == "ValueError"
    assert serialized["cause"]["message"] == "Valor fuera de rango"


# -------------------------------------------------------------------
# Tests de edge cases
# -------------------------------------------------------------------


def test_error_with_string_error_code():
    """Verifica que NO se pueda usar string como código de error si no está definido."""
    # El enum no tiene CUSTOM_ERROR_CODE, debería fallar
    with pytest.raises(ValueError, match="is not a valid ErrorCode"):
        AnalyzerBrainError(
            message="Error personalizado",
            error_code="CUSTOM_ERROR_CODE",  # Este código no existe
            severity=ErrorSeverity.LOW,
        )


def test_error_with_empty_details():
    """Verifica creación de error con detalles vacíos."""
    error = AnalyzerBrainError(message="Error simple", error_code=ErrorCode.INTERNAL_ERROR)

    assert error.details == {}
    serialized = error.to_dict()
    assert serialized["details"] == {}
    assert "cause" not in serialized


def test_error_timestamp_format():
    """Verifica que el timestamp tenga formato ISO."""
    error = AnalyzerBrainError("test")

    # Intentar parsear el timestamp como fecha ISO
    try:
        datetime.fromisoformat(error.timestamp)
        is_valid = True
    except ValueError:
        # Si falla, intentar sin la Z al final (Python 3.10+)
        try:
            datetime.fromisoformat(error.timestamp.replace('Z', '+00:00'))
            is_valid = True
        except ValueError:
            is_valid = False

    assert is_valid, f"Timestamp inválido: {error.timestamp}"


# -------------------------------------------------------------------
# Tests de comparación
# -------------------------------------------------------------------


def test_error_equality_by_message():
    """Verifica que dos errores con el mismo mensaje sean diferentes objetos."""
    error1 = AnalyzerBrainError("Mismo mensaje")
    error2 = AnalyzerBrainError("Mismo mensaje")

    assert error1 != error2  # Son objetos diferentes
    assert error1.message == error2.message  # Pero tienen el mismo mensaje


def test_error_with_same_cause():
    """Verifica manejo de la misma causa en diferentes errores."""
    cause = RuntimeError("Causa compartida")

    error1 = AnalyzerBrainError("Error 1", cause=cause)
    error2 = AnalyzerBrainError("Error 2", cause=cause)

    assert error1.cause is error2.cause  # Mismo objeto de causa


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
