"""
RequestValidator - Sistema completo de validación de requests.
"""

import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import jsonschema
from email_validator import validate_email, EmailNotValidError
from fastapi import Request, HTTPException
from pydantic import BaseModel, ValidationError as PydanticValidationError
import bleach

from ..core.exceptions import BrainException, ValidationError
from ..utils.validation import Validation as GenericValidation

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Niveles de validación."""
    BASIC = "basic"      # Validaciones básicas (presencia, tipo)
    STRICT = "strict"    # Validaciones estrictas (formato, rangos)
    SECURE = "secure"    # Validaciones de seguridad (inyección, XSS)

@dataclass
class ValidationRule:
    """Regla de validación."""
    field: str
    field_type: str
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[str] = None

class RequestValidator:
    """
    Sistema completo de validación de requests.
    
    Características:
    1. Validación de estructura de requests
    2. Validación de tipos de datos
    3. Sanitización de entrada
    4. Validación de formatos específicos
    5. Validación de consistencia
    6. Generación de errores detallados
    7. Sugerencias de corrección
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa el validador de requests.
        
        Args:
            config: Configuración del validador (opcional)
        """
        self.config = config or {
            "enabled": True,
            "default_validation_level": ValidationLevel.STRICT,
            "max_request_size_mb": 10,
            "max_array_size": 1000,
            "max_nesting_depth": 10,
            "enable_sanitization": True,
            "enable_schema_validation": True,
            "log_validation_errors": True,
            "reject_unknown_fields": False,
        }
        
        # Esquemas de validación predefinidos
        self.schemas: Dict[str, Dict] = {}
        self._initialize_default_schemas()
        
        # Utilidad de validación genérica
        self.generic_validator = GenericValidation()
        
        logger.info("RequestValidator inicializado")
    
    async def validate_request_structure(self, request: Request, call_next) -> Any:
        """
        Middleware de validación de estructura para FastAPI.
        
        Args:
            request: Request HTTP
            call_next: Función para continuar el pipeline
            
        Returns:
            Response: Respuesta HTTP
        """
        if not self.config["enabled"]:
            return await call_next(request)
        
        try:
            # Validar tamaño de request
            content_length = request.headers.get("content-length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > self.config["max_request_size_mb"]:
                    raise ValidationError(
                        f"Request too large: {size_mb:.1f}MB > {self.config['max_request_size_mb']}MB"
                    )
            
            # Validar content-type para requests con body
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "")
                
                if not content_type:
                    raise ValidationError("Content-Type header is required")
                
                # Aceptar JSON y form-data
                if "application/json" not in content_type and "multipart/form-data" not in content_type:
                    raise ValidationError(
                        f"Unsupported content-type: {content_type}. "
                        "Use application/json or multipart/form-data"
                    )
            
            # Continuar con el pipeline
            return await call_next(request)
            
        except ValidationError as e:
            logger.warning("Request structure validation failed: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
            
        except Exception as e:
            logger.error("Request structure validation error: %s", e)
            raise HTTPException(status_code=400, detail="Invalid request structure")
    
    async def validate_request_data(self, 
                                  data: Dict[str, Any], 
                                  schema_name: Optional[str] = None,
                                  validation_level: ValidationLevel = None) -> Tuple[bool, List[Dict]]:
        """
        Valida datos de request contra un esquema.
        
        Args:
            data: Datos a validar
            schema_name: Nombre del esquema (opcional)
            validation_level: Nivel de validación (opcional)
            
        Returns:
            Tuple (success, errors)
        """
        errors = []
        
        try:
            # Aplicar nivel de validación
            level = validation_level or self.config["default_validation_level"]
            
            # Validación básica de estructura
            if not isinstance(data, dict):
                errors.append({
                    "field": "root",
                    "error": "Data must be a JSON object",
                    "code": "INVALID_TYPE"
                })
                return False, errors
            
            # Validar contra esquema si se especifica
            if schema_name and schema_name in self.schemas:
                schema_errors = await self._validate_against_schema(data, schema_name)
                errors.extend(schema_errors)
            
            # Validaciones adicionales según nivel
            if level == ValidationLevel.STRICT:
                strict_errors = await self._apply_strict_validations(data)
                errors.extend(strict_errors)
            
            elif level == ValidationLevel.SECURE:
                secure_errors = await self._apply_secure_validations(data)
                errors.extend(secure_errors)
            
            # Sanitización si está habilitada
            if self.config["enable_sanitization"] and not errors:
                sanitized_data = await self.sanitize_input(data)
                data.clear()
                data.update(sanitized_data)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error("Request data validation error: %s", e)
            errors.append({
                "field": "root",
                "error": f"Validation error: {str(e)}",
                "code": "VALIDATION_ERROR"
            })
            return False, errors
    
    async def sanitize_input(self, data: Any, depth: int = 0) -> Any:
        """
        Sanitiza entrada para prevenir XSS y otros ataques.
        
        Args:
            data: Datos a sanitizar
            depth: Profundidad actual de recursión
            
        Returns:
            Datos sanitizados
        """
        if depth > self.config["max_nesting_depth"]:
            raise ValidationError(f"Maximum nesting depth exceeded: {depth}")
        
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Sanitizar clave
                sanitized_key = bleach.clean(str(key), strip=True)
                
                # Sanitizar valor recursivamente
                sanitized_value = await self.sanitize_input(value, depth + 1)
                sanitized[sanitized_key] = sanitized_value
            
            return sanitized
        
        elif isinstance(data, list):
            return [await self.sanitize_input(item, depth + 1) for item in data]
        
        elif isinstance(data, str):
            # Sanitizar strings
            sanitized = bleach.clean(data, strip=True)
            
            # Remover caracteres de control
            sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\r\n\t')
            
            return sanitized
        
        else:
            # Otros tipos (números, booleanos, None) no necesitan sanitización
            return data
    
    async def check_required_fields(self, data: Dict, required_fields: List[str]) -> List[Dict]:
        """
        Verifica campos requeridos.
        
        Args:
            data: Datos a validar
            required_fields: Lista de campos requeridos
            
        Returns:
            Lista de errores
        """
        errors = []
        
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append({
                    "field": field,
                    "error": "This field is required",
                    "code": "REQUIRED_FIELD"
                })
        
        return errors
    
    async def validate_data_types(self, data: Dict, type_spec: Dict[str, str]) -> List[Dict]:
        """
        Valida tipos de datos.
        
        Args:
            data: Datos a validar
            type_spec: Especificación de tipos {field: type}
            
        Returns:
            Lista de errores
        """
        errors = []
        
        for field, expected_type in type_spec.items():
            if field not in data:
                continue
            
            value = data[field]
            
            # Verificar tipo
            type_ok = False
            
            if expected_type == "string":
                type_ok = isinstance(value, str)
            elif expected_type == "integer":
                type_ok = isinstance(value, int) and not isinstance(value, bool)
            elif expected_type == "number":
                type_ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif expected_type == "boolean":
                type_ok = isinstance(value, bool)
            elif expected_type == "array":
                type_ok = isinstance(value, list)
            elif expected_type == "object":
                type_ok = isinstance(value, dict)
            elif expected_type == "null":
                type_ok = value is None
            elif expected_type.startswith("string|"):
                # Tipo string o algo específico
                type_ok = isinstance(value, str)
            else:
                # Tipo custom
                type_ok = True  # Asumir OK y validar después
            
            if not type_ok:
                errors.append({
                    "field": field,
                    "error": f"Expected type {expected_type}, got {type(value).__name__}",
                    "code": "INVALID_TYPE",
                    "expected": expected_type,
                    "actual": type(value).__name__,
                })
        
        return errors
    
    async def handle_validation_errors(self, errors: List[Dict]) -> Dict[str, Any]:
        """
        Maneja errores de validación formateando respuesta.
        
        Args:
            errors: Lista de errores
            
        Returns:
            Dict con respuesta formateada
        """
        if not errors:
            return {"valid": True}
        
        # Agrupar errores por campo
        field_errors = {}
        global_errors = []
        
        for error in errors:
            field = error.get("field", "global")
            
            if field == "global":
                global_errors.append(error)
            else:
                if field not in field_errors:
                    field_errors[field] = []
                field_errors[field].append(error)
        
        # Generar sugerencias de corrección
        suggestions = await self.suggest_corrections(errors)
        
        response = {
            "valid": False,
            "error_count": len(errors),
            "errors": {
                "fields": field_errors,
                "global": global_errors,
            },
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat(),
        }
        
        if self.config["log_validation_errors"]:
            logger.warning("Validation failed: %s errors, suggestions: %s", 
                         len(errors), suggestions)
        
        return response
    
    async def generate_validation_report(self, 
                                       data: Dict, 
                                       schema_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Genera reporte completo de validación.
        
        Args:
            data: Datos validados
            schema_name: Nombre del esquema usado
            
        Returns:
            Dict con reporte de validación
        """
        # Ejecutar validación completa
        success, errors = await self.validate_request_data(data, schema_name, ValidationLevel.SECURE)
        
        # Obtener estadísticas
        field_count = len(data) if isinstance(data, dict) else 0
        nested_count = self._count_nested_elements(data)
        estimated_size = len(json.dumps(data)) if data else 0
        
        report = {
            "valid": success,
            "schema_used": schema_name,
            "validation_level": "secure",
            "statistics": {
                "field_count": field_count,
                "nested_element_count": nested_count,
                "estimated_size_bytes": estimated_size,
                "validation_time_ms": 0,  # Se calcularía en implementación real
            },
            "issues": {
                "critical": len([e for e in errors if e.get("code") in ["SECURITY", "INJECTION"]]),
                "errors": len([e for e in errors if e.get("code") not in ["WARNING", "INFO"]]),
                "warnings": len([e for e in errors if e.get("code") in ["WARNING", "INFO"]]),
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        if not success:
            report["errors"] = errors[:10]  # Limitar a 10 errores
            report["suggestions"] = await self.suggest_corrections(errors)
        
        return report
    
    async def suggest_corrections(self, errors: List[Dict]) -> List[Dict]:
        """
        Sugiere correcciones para errores de validación.
        
        Args:
            errors: Lista de errores
            
        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        for error in errors:
            field = error.get("field")
            code = error.get("code")
            actual = error.get("actual")
            expected = error.get("expected")
            
            suggestion = {
                "field": field,
                "issue": error.get("error"),
                "code": code,
            }
            
            # Sugerencias específicas por código de error
            if code == "REQUIRED_FIELD":
                suggestion["fix"] = f"Add field '{field}' with appropriate value"
                
            elif code == "INVALID_TYPE":
                if expected and actual:
                    suggestion["fix"] = f"Change type from {actual} to {expected}"
                    
            elif code == "INVALID_FORMAT":
                if field and "email" in str(error.get("error", "")).lower():
                    suggestion["fix"] = "Use a valid email format: user@example.com"
                elif field and "url" in str(error.get("error", "")).lower():
                    suggestion["fix"] = "Use a valid URL format: https://example.com"
                    
            elif code == "VALUE_TOO_SHORT":
                min_length = error.get("min_length")
                if min_length:
                    suggestion["fix"] = f"Increase length to at least {min_length} characters"
                    
            elif code == "VALUE_TOO_LONG":
                max_length = error.get("max_length")
                if max_length:
                    suggestion["fix"] = f"Decrease length to at most {max_length} characters"
                    
            elif code == "VALUE_TOO_SMALL":
                min_value = error.get("min_value")
                if min_value:
                    suggestion["fix"] = f"Increase value to at least {min_value}"
                    
            elif code == "VALUE_TOO_LARGE":
                max_value = error.get("max_value")
                if max_value:
                    suggestion["fix"] = f"Decrease value to at most {max_value}"
                    
            elif code == "INVALID_PATTERN":
                pattern = error.get("pattern")
                if pattern:
                    suggestion["fix"] = f"Match pattern: {pattern}"
                    
            elif code == "UNKNOWN_FIELD":
                suggestion["fix"] = f"Remove field '{field}' or check spelling"
                
            elif code == "INVALID_CHARACTERS":
                suggestion["fix"] = "Remove special characters or escape them properly"
            
            suggestions.append(suggestion)
        
        return suggestions
    
    # Métodos de implementación
    
    def _initialize_default_schemas(self):
        """Inicializa esquemas de validación por defecto."""
        # Esquema para creación de proyecto
        self.schemas["create_project"] = {
            "type": "object",
            "required": ["name", "path"],
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 255,
                    "pattern": "^[a-zA-Z0-9 _-]+$",
                },
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1000,
                },
                "description": {
                    "type": "string",
                    "maxLength": 1000,
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "java", "cpp", "go", "rust"],
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "analysis_mode": {
                            "type": "string",
                            "enum": ["quick", "standard", "comprehensive", "deep"],
                        },
                        "include_tests": {"type": "boolean"},
                        "include_docs": {"type": "boolean"},
                    },
                },
            },
            "additionalProperties": self.config["reject_unknown_fields"],
        }
        
        # Esquema para consulta
        self.schemas["query"] = {
            "type": "object",
            "required": ["question"],
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 5000,
                },
                "project_id": {
                    "type": "string",
                    "pattern": "^[a-fA-F0-9-]+$",  # UUID pattern simplified
                },
                "context": {
                    "type": "object",
                    "maxProperties": 20,
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "detail_level": {
                            "type": "string",
                            "enum": ["brief", "normal", "detailed"],
                        },
                        "include_code": {"type": "boolean"},
                        "include_sources": {"type": "boolean"},
                    },
                },
            },
            "additionalProperties": self.config["reject_unknown_fields"],
        }
        
        # Esquema para análisis
        self.schemas["analysis"] = {
            "type": "object",
            "required": ["project_id"],
            "properties": {
                "project_id": {
                    "type": "string",
                    "pattern": "^[a-fA-F0-9-]+$",
                },
                "mode": {
                    "type": "string",
                    "enum": ["quick", "standard", "comprehensive", "deep"],
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "timeout_minutes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 120,
                        },
                        "max_file_size_mb": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                },
            },
            "additionalProperties": self.config["reject_unknown_fields"],
        }
    
    async def _validate_against_schema(self, data: Dict, schema_name: str) -> List[Dict]:
        """
        Valida datos contra un esquema JSON Schema.
        
        Args:
            data: Datos a validar
            schema_name: Nombre del esquema
            
        Returns:
            Lista de errores
        """
        errors = []
        schema = self.schemas.get(schema_name)
        
        if not schema:
            errors.append({
                "field": "root",
                "error": f"Unknown schema: {schema_name}",
                "code": "UNKNOWN_SCHEMA"
            })
            return errors
        
        try:
            # Validar con JSON Schema
            jsonschema.validate(instance=data, schema=schema)
            
            # Validaciones adicionales específicas del esquema
            schema_specific_errors = await self._validate_schema_specific(data, schema_name)
            errors.extend(schema_specific_errors)
            
        except jsonschema.ValidationError as e:
            # Convertir error de jsonschema a formato estándar
            error_path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            
            error_info = {
                "field": error_path,
                "error": e.message,
                "code": "SCHEMA_VALIDATION",
                "schema_path": e.json_path,
            }
            
            # Añadir contexto adicional
            if e.validator:
                error_info["validator"] = e.validator
            if e.validator_value:
                error_info["validator_value"] = e.validator_value
            
            errors.append(error_info)
        
        return errors
    
    async def _validate_schema_specific(self, data: Dict, schema_name: str) -> List[Dict]:
        """Validaciones específicas por esquema."""
        errors = []
        
        if schema_name == "create_project":
            # Validar que la ruta del proyecto sea válida
            path = data.get("path")
            if path:
                if not self._is_valid_path(path):
                    errors.append({
                        "field": "path",
                        "error": "Invalid project path",
                        "code": "INVALID_PATH",
                        "suggestion": "Use an absolute path or a valid relative path",
                    })
        
        elif schema_name == "query":
            # Validar que project_id tenga formato UUID si está presente
            project_id = data.get("project_id")
            if project_id and not self._is_valid_uuid(project_id):
                errors.append({
                    "field": "project_id",
                    "error": "Invalid project ID format",
                    "code": "INVALID_FORMAT",
                    "suggestion": "Use a valid UUID format",
                })
            
            # Validar que la pregunta no sea solo espacios
            question = data.get("question", "").strip()
            if not question:
                errors.append({
                    "field": "question",
                    "error": "Question cannot be empty or only whitespace",
                    "code": "EMPTY_VALUE",
                })
        
        return errors
    
    async def _apply_strict_validations(self, data: Dict) -> List[Dict]:
        """Aplica validaciones estrictas."""
        errors = []
        
        # Validar formatos de campos específicos
        for field, value in data.items():
            if isinstance(value, str):
                field_errors = await self._validate_string_format(field, value)
                errors.extend(field_errors)
            
            elif isinstance(value, list):
                # Validar tamaño máximo de array
                if len(value) > self.config["max_array_size"]:
                    errors.append({
                        "field": field,
                        "error": f"Array too large: {len(value)} > {self.config['max_array_size']}",
                        "code": "ARRAY_TOO_LARGE",
                    })
                
                # Validar elementos del array
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        nested_errors = await self._apply_strict_validations(item)
                        for nested_error in nested_errors:
                            nested_error["field"] = f"{field}[{i}].{nested_error['field']}"
                            errors.append(nested_error)
        
        return errors
    
    async def _apply_secure_validations(self, data: Dict) -> List[Dict]:
        """Aplica validaciones de seguridad."""
        errors = []
        
        for field, value in data.items():
            if isinstance(value, str):
                # Detectar posibles inyecciones SQL
                if self._detect_sql_injection(value):
                    errors.append({
                        "field": field,
                        "error": "Potential SQL injection detected",
                        "code": "SECURITY",
                        "severity": "high",
                    })
                
                # Detectar posibles XSS
                if self._detect_xss(value):
                    errors.append({
                        "field": field,
                        "error": "Potential XSS attack detected",
                        "code": "SECURITY",
                        "severity": "high",
                    })
                
                # Detectar path traversal
                if self._detect_path_traversal(value):
                    errors.append({
                        "field": field,
                        "error": "Potential path traversal attack detected",
                        "code": "SECURITY",
                        "severity": "medium",
                    })
            
            elif isinstance(value, dict):
                # Validar recursivamente
                nested_errors = await self._apply_secure_validations(value)
                for nested_error in nested_errors:
                    nested_error["field"] = f"{field}.{nested_error['field']}"
                    errors.append(nested_error)
        
        return errors
    
    async def _validate_string_format(self, field: str, value: str) -> List[Dict]:
        """Valida formato de strings."""
        errors = []
        
        # Validar email
        if field.lower().endswith("email") or "email" in field.lower():
            try:
                validate_email(value, check_deliverability=False)
            except EmailNotValidError as e:
                errors.append({
                    "field": field,
                    "error": f"Invalid email format: {str(e)}",
                    "code": "INVALID_FORMAT",
                })
        
        # Validar URL
        elif field.lower().endswith("url") or "url" in field.lower():
            if not self._is_valid_url(value):
                errors.append({
                    "field": field,
                    "error": "Invalid URL format",
                    "code": "INVALID_FORMAT",
                })
        
        # Validar UUID
        elif field.lower().endswith("_id") and ("uuid" in field.lower() or "id" == field.lower()):
            if not self._is_valid_uuid(value):
                errors.append({
                    "field": field,
                    "error": "Invalid UUID format",
                    "code": "INVALID_FORMAT",
                })
        
        # Validar fecha ISO
        elif field.lower().endswith("date") or field.lower().endswith("_at"):
            if not self._is_valid_iso_date(value):
                errors.append({
                    "field": field,
                    "error": "Invalid date format. Use ISO 8601: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS",
                    "code": "INVALID_FORMAT",
                })
        
        return errors
    
    # Métodos auxiliares de validación
    
    def _is_valid_path(self, path: str) -> bool:
        """Verifica si una ruta es válida."""
        try:
            # Verificar caracteres peligrosos
            dangerous_patterns = [
                "../",  # Path traversal
                "..\\",  # Path traversal (Windows)
                "~/",   # Home directory
                "|",    # Pipe
                "&",    # Background process
                ";",    # Command separator
                "`",    # Command substitution
                "$(",   # Command substitution
            ]
            
            for pattern in dangerous_patterns:
                if pattern in path:
                    return False
            
            # Verificar longitud mínima
            if len(path.strip()) == 0:
                return False
            
            return True
            
        except:
            return False
    
    def _is_valid_uuid(self, uuid_str: str) -> bool:
        """Verifica si un string es un UUID válido."""
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        return bool(uuid_pattern.match(uuid_str))
    
    def _is_valid_url(self, url: str) -> bool:
        """Verifica si un string es una URL válida."""
        url_pattern = re.compile(
            r'^(https?|ftp)://'  # Protocolo
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # Dominio
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # Puerto
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        return bool(url_pattern.match(url))
    
    def _is_valid_iso_date(self, date_str: str) -> bool:
        """Verifica si un string es una fecha ISO válida."""
        try:
            # Intentar parsear como fecha ISO
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return True
        except ValueError:
            # Intentar con formato simple YYYY-MM-DD
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return True
            except ValueError:
                return False
    
    def _detect_sql_injection(self, value: str) -> bool:
        """Detecta posibles inyecciones SQL."""
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|EXEC)\b)",
            r"(--|\#)",  # Comentarios SQL
            r"(\b(OR|AND)\s+[\w']+\s*=\s*[\w']+)",
            r"(\b(SLEEP|WAITFOR|BENCHMARK)\s*\()",
        ]
        
        value_upper = value.upper()
        for pattern in sql_patterns:
            if re.search(pattern, value_upper, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_xss(self, value: str) -> bool:
        """Detecta posibles ataques XSS."""
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",  # Protocolo javascript:
            r"on\w+\s*=",    # Event handlers
            r"eval\s*\(",
            r"alert\s*\(",
            r"document\.(cookie|location|write)",
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_path_traversal(self, value: str) -> bool:
        """Detecta posibles path traversal."""
        traversal_patterns = [
            r"\.\./",      # Unix path traversal
            r"\.\.\\",     # Windows path traversal
            r"\.\.%2f",    # URL encoded
            r"\.\.%5c",    # URL encoded
            r"\.\.%255c",  # Double URL encoded
        ]
        
        for pattern in traversal_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        return False
    
    def _count_nested_elements(self, data: Any, depth: int = 0) -> int:
        """Cuenta elementos anidados en datos."""
        if depth > self.config["max_nesting_depth"]:
            return 0
        
        count = 0
        
        if isinstance(data, dict):
            count += len(data)
            for value in data.values():
                count += self._count_nested_elements(value, depth + 1)
        
        elif isinstance(data, list):
            count += len(data)
            for item in data:
                count += self._count_nested_elements(item, depth + 1)
        
        return count

# Modelos Pydantic para validación adicional (opcional)
class ProjectCreateRequest(BaseModel):
    """Modelo Pydantic para creación de proyecto."""
    name: str
    path: str
    description: Optional[str] = None
    language: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "forbid"  # No permitir campos adicionales

class QueryRequest(BaseModel):
    """Modelo Pydantic para consultas."""
    question: str
    project_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "forbid"