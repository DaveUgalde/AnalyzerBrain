"""
Validation - Utilidades para validación de datos y estructuras.
Incluye validación de tipos, valores, formatos, consistencia y sugerencias de corrección.
"""

import re
import json
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Set, Type, get_type_hints
from datetime import datetime, date, time
from decimal import Decimal
from email.utils import parseaddr
from pathlib import Path
import ipaddress
import uuid
import inspect
from enum import Enum
import logging
from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)

class ValidationRule(str, Enum):
    """Reglas de validación predefinidas."""
    REQUIRED = "required"
    TYPE = "type"
    MIN = "min"
    MAX = "max"
    LENGTH = "length"
    PATTERN = "pattern"
    EMAIL = "email"
    URL = "url"
    IP_ADDRESS = "ip_address"
    UUID = "uuid"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    CHOICE = "choice"
    RANGE = "range"
    CUSTOM = "custom"

@dataclass
class ValidationResult:
    """Resultado de una validación."""
    valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_error(self, field: str, rule: str, message: str, value: Any = None) -> None:
        """Agrega un error al resultado."""
        self.errors.append({
            'field': field,
            'rule': rule,
            'message': message,
            'value': value,
            'severity': 'error'
        })
        self.valid = False
    
    def add_warning(self, field: str, rule: str, message: str, value: Any = None) -> None:
        """Agrega una advertencia al resultado."""
        self.warnings.append({
            'field': field,
            'rule': rule,
            'message': message,
            'value': value,
            'severity': 'warning'
        })
    
    def add_correction(self, field: str, current: Any, suggested: Any, reason: str) -> None:
        """Agrega una sugerencia de corrección."""
        self.corrections.append({
            'field': field,
            'current': current,
            'suggested': suggested,
            'reason': reason
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'valid': self.valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'corrections': self.corrections,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }

class Validation:
    """
    Utilidades avanzadas para validación de datos.
    
    Características:
    1. Validación de estructuras complejas
    2. Validación de tipos con soporte para tipos complejos
    3. Validación de valores y rangos
    4. Validación de formatos (email, URL, etc.)
    5. Validación de consistencia entre campos
    6. Generación de errores y sugerencias de corrección
    """
    
    # Patrones precompilados para validación común
    _patterns = {
        'email': re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        ),
        'url': re.compile(
            r'^https?://'  # http:// o https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # dominio
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # puerto
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        ),
        'phone': re.compile(
            r'^(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}$'
        ),
        'zip_code': re.compile(
            r'^\d{5}(-\d{4})?$'  # US zip code
        ),
        'ssn': re.compile(
            r'^\d{3}-\d{2}-\d{4}$'  # US Social Security Number
        ),
        'credit_card': re.compile(
            r'^(?:4[0-9]{12}(?:[0-9]{3})?|'  # Visa
            r'5[1-5][0-9]{14}|'  # MasterCard
            r'3[47][0-9]{13}|'  # American Express
            r'3(?:0[0-5]|[68][0-9])[0-9]{11}|'  # Diners Club
            r'6(?:011|5[0-9]{2})[0-9]{12}|'  # Discover
            r'(?:2131|1800|35\d{3})\d{11})$'  # JCB
        ),
        'hex_color': re.compile(
            r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
        ),
        'slug': re.compile(
            r'^[a-z0-9]+(?:-[a-z0-9]+)*$'
        ),
        'username': re.compile(
            r'^[a-zA-Z0-9_]{3,20}$'
        ),
        'password': re.compile(
            r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
        )
    }
    
    @staticmethod
    def validate_structure(
        data: Any,
        schema: Dict[str, Any],
        strict: bool = False,
        allow_extra: bool = True
    ) -> ValidationResult:
        """
        Valida la estructura de datos contra un esquema.
        
        Args:
            data: Datos a validar
            schema: Esquema de validación
            strict: Modo estricto (no permite tipos diferentes)
            allow_extra: Permitir campos adicionales no en el esquema
            
        Returns:
            ValidationResult con resultados de validación
        """
        result = ValidationResult(valid=True)
        
        try:
            if isinstance(schema, dict) and 'type' in schema:
                # Esquema simple
                Validation._validate_value(
                    data, '', schema, result, strict, allow_extra
                )
            elif isinstance(schema, dict):
                # Esquema complejo (diccionario de campos)
                if not isinstance(data, dict):
                    result.add_error(
                        field='',
                        rule=ValidationRule.TYPE.value,
                        message=f"Expected dict, got {type(data).__name__}",
                        value=data
                    )
                    return result
                
                # Validar campos requeridos
                for field, field_schema in schema.items():
                    if isinstance(field_schema, dict) and field_schema.get('required', False):
                        if field not in data:
                            result.add_error(
                                field=field,
                                rule=ValidationRule.REQUIRED.value,
                                message=f"Field '{field}' is required",
                                value=None
                            )
                
                # Validar cada campo
                for field, value in data.items():
                    if field in schema:
                        field_schema = schema[field]
                        Validation._validate_value(
                            value, field, field_schema, result, strict, allow_extra
                        )
                    elif not allow_extra:
                        result.add_error(
                            field=field,
                            rule=ValidationRule.TYPE.value,
                            message=f"Extra field '{field}' not allowed in strict mode",
                            value=value
                        )
                
            elif isinstance(schema, list):
                # Esquema para listas
                if not isinstance(data, list):
                    result.add_error(
                        field='',
                        rule=ValidationRule.TYPE.value,
                        message=f"Expected list, got {type(data).__name__}",
                        value=data
                    )
                    return result
                
                # Validar cada elemento
                for i, item in enumerate(data):
                    if len(schema) > 0:
                        # Usar primer esquema para todos los elementos
                        Validation._validate_value(
                            item, f'[{i}]', schema[0], result, strict, allow_extra
                        )
            
            else:
                # Tipo simple
                expected_type = schema
                Validation._validate_type(data, '', expected_type, result, strict)
            
            return result
            
        except Exception as e:
            logger.error(f"Structure validation failed: {e}")
            result.add_error(
                field='',
                rule=ValidationRule.TYPE.value,
                message=f"Validation error: {str(e)}",
                value=data
            )
            return result
    
    @staticmethod
    def validate_types(
        data: Any,
        expected_type: Union[Type, Tuple[Type, ...]],
        allow_none: bool = False,
        strict: bool = False
    ) -> ValidationResult:
        """
        Valida tipos de datos.
        
        Args:
            data: Datos a validar
            expected_type: Tipo o tupla de tipos esperados
            allow_none: Permitir None
            strict: Modo estricto (sin conversiones implícitas)
            
        Returns:
            ValidationResult con resultados de validación
        """
        result = ValidationResult(valid=True)
        
        try:
            # Manejar None
            if data is None:
                if allow_none:
                    return result
                else:
                    result.add_error(
                        field='',
                        rule=ValidationRule.TYPE.value,
                        message=f"Expected {expected_type}, got None",
                        value=data
                    )
                    return result
            
            # Validar tipo
            Validation._validate_type(data, '', expected_type, result, strict)
            
            return result
            
        except Exception as e:
            logger.error(f"Type validation failed: {e}")
            result.add_error(
                field='',
                rule=ValidationRule.TYPE.value,
                message=f"Type validation error: {str(e)}",
                value=data
            )
            return result
    
    @staticmethod
    def validate_values(
        data: Any,
        rules: Dict[str, Any],
        field_name: str = ''
    ) -> ValidationResult:
        """
        Valida valores según reglas específicas.
        
        Args:
            data: Datos a validar
            rules: Diccionario de reglas de validación
            field_name: Nombre del campo (para mensajes de error)
            
        Returns:
            ValidationResult con resultados de validación
        """
        result = ValidationResult(valid=True)
        
        try:
            # Aplicar cada regla
            for rule_name, rule_value in rules.items():
                try:
                    Validation._apply_validation_rule(
                        data, field_name, rule_name, rule_value, result
                    )
                except Exception as e:
                    result.add_error(
                        field=field_name,
                        rule=rule_name,
                        message=f"Rule '{rule_name}' failed: {str(e)}",
                        value=data
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Value validation failed: {e}")
            result.add_error(
                field=field_name,
                rule='validation',
                message=f"Value validation error: {str(e)}",
                value=data
            )
            return result
    
    @staticmethod
    def validate_format(
        data: str,
        format_type: str,
        strict: bool = False,
        **kwargs
    ) -> ValidationResult:
        """
        Valida formatos específicos (email, URL, etc.).
        
        Args:
            data: String a validar
            format_type: Tipo de formato ('email', 'url', 'phone', etc.)
            strict: Modo estricto
            **kwargs: Argumentos adicionales para validación específica
            
        Returns:
            ValidationResult con resultados de validación
        """
        result = ValidationResult(valid=True)
        
        if not isinstance(data, str):
            result.add_error(
                field='',
                rule=ValidationRule.TYPE.value,
                message=f"Expected string for format validation, got {type(data).__name__}",
                value=data
            )
            return result
        
        try:
            if format_type == 'email':
                Validation._validate_email(data, result, **kwargs)
            elif format_type == 'url':
                Validation._validate_url(data, result, **kwargs)
            elif format_type == 'ip_address':
                Validation._validate_ip_address(data, result, **kwargs)
            elif format_type == 'uuid':
                Validation._validate_uuid(data, result, **kwargs)
            elif format_type == 'date':
                Validation._validate_date(data, result, **kwargs)
            elif format_type == 'time':
                Validation._validate_time(data, result, **kwargs)
            elif format_type == 'datetime':
                Validation._validate_datetime(data, result, **kwargs)
            elif format_type in Validation._patterns:
                Validation._validate_pattern(data, format_type, result, **kwargs)
            else:
                result.add_error(
                    field='',
                    rule=ValidationRule.PATTERN.value,
                    message=f"Unknown format type: {format_type}",
                    value=data
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Format validation failed: {e}")
            result.add_error(
                field='',
                rule=ValidationRule.PATTERN.value,
                message=f"Format validation error: {str(e)}",
                value=data
            )
            return result
    
    @staticmethod
    def validate_consistency(
        data: Dict[str, Any],
        rules: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Valida consistencia entre campos (dependencias, rangos relativos, etc.).
        
        Args:
            data: Diccionario de datos
            rules: Lista de reglas de consistencia
            
        Returns:
            ValidationResult con resultados de validación
        """
        result = ValidationResult(valid=True)
        
        try:
            for rule in rules:
                rule_type = rule.get('type')
                
                if rule_type == 'dependency':
                    # Campo B requiere Campo A
                    field = rule['field']
                    depends_on = rule['depends_on']
                    
                    if field in data and data[field] is not None:
                        if depends_on not in data or data[depends_on] is None:
                            result.add_error(
                                field=field,
                                rule='dependency',
                                message=f"Field '{field}' requires field '{depends_on}' to be set",
                                value=data[field]
                            )
                
                elif rule_type == 'exclusive':
                    # Solo uno de los campos puede estar presente
                    fields = rule['fields']
                    present_fields = [f for f in fields if f in data and data[f] is not None]
                    
                    if len(present_fields) > 1:
                        result.add_error(
                            field=','.join(present_fields),
                            rule='exclusive',
                            message=f"Fields {present_fields} are mutually exclusive",
                            value={f: data[f] for f in present_fields}
                        )
                
                elif rule_type == 'required_if':
                    # Campo requerido si otro campo tiene cierto valor
                    field = rule['field']
                    other_field = rule['other_field']
                    other_value = rule['other_value']
                    
                    if (other_field in data and 
                        data[other_field] == other_value and 
                        (field not in data or data[field] is None)):
                        
                        result.add_error(
                            field=field,
                            rule='required_if',
                            message=f"Field '{field}' is required when '{other_field}' is '{other_value}'",
                            value=None
                        )
                
                elif rule_type == 'range_relation':
                    # Validar que min <= max
                    min_field = rule['min_field']
                    max_field = rule['max_field']
                    
                    if (min_field in data and max_field in data and
                        data[min_field] is not None and data[max_field] is not None):
                        
                        try:
                            min_val = float(data[min_field])
                            max_val = float(data[max_field])
                            
                            if min_val > max_val:
                                result.add_error(
                                    field=f"{min_field},{max_field}",
                                    rule='range_relation',
                                    message=f"'{min_field}' ({min_val}) must be <= '{max_field}' ({max_val})",
                                    value={'min': min_val, 'max': max_val}
                                )
                        except (ValueError, TypeError):
                            pass
                
                elif rule_type == 'custom':
                    # Regla personalizada
                    validator = rule['validator']
                    if callable(validator):
                        try:
                            if not validator(data):
                                result.add_error(
                                    field=rule.get('field', ''),
                                    rule='custom',
                                    message=rule.get('message', 'Custom validation failed'),
                                    value=data
                                )
                        except Exception as e:
                            result.add_error(
                                field=rule.get('field', ''),
                                rule='custom',
                                message=f"Custom validator error: {str(e)}",
                                value=data
                            )
            
            return result
            
        except Exception as e:
            logger.error(f"Consistency validation failed: {e}")
            result.add_error(
                field='',
                rule='consistency',
                message=f"Consistency validation error: {str(e)}",
                value=data
            )
            return result
    
    @staticmethod
    def generate_validation_errors(
        validation_result: ValidationResult,
        include_warnings: bool = False,
        include_corrections: bool = False
    ) -> Dict[str, Any]:
        """
        Genera un reporte estructurado de errores de validación.
        
        Args:
            validation_result: Resultado de validación
            include_warnings: Incluir advertencias
            include_corrections: Incluir sugerencias de corrección
            
        Returns:
            Diccionario con reporte de errores
        """
        report = {
            'valid': validation_result.valid,
            'error_count': len(validation_result.errors),
            'errors': validation_result.errors
        }
        
        if include_warnings:
            report['warning_count'] = len(validation_result.warnings)
            report['warnings'] = validation_result.warnings
        
        if include_corrections:
            report['correction_count'] = len(validation_result.corrections)
            report['corrections'] = validation_result.corrections
        
        # Agrupar errores por campo
        field_errors = {}
        for error in validation_result.errors:
            field = error['field']
            if field not in field_errors:
                field_errors[field] = []
            field_errors[field].append(error)
        
        report['field_errors'] = field_errors
        
        return report
    
    @staticmethod
    def suggest_corrections(
        data: Any,
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """
        Sugiere correcciones automáticas basadas en errores de validación.
        
        Args:
            data: Datos originales
            validation_result: Resultado de validación con errores
            
        Returns:
            Diccionario con datos corregidos y cambios realizados
        """
        corrected_data = data.copy() if isinstance(data, dict) else data
        changes = []
        
        for correction in validation_result.corrections:
            field = correction['field']
            suggested = correction['suggested']
            
            if field:
                # Campo específico
                if isinstance(corrected_data, dict):
                    corrected_data[field] = suggested
                    changes.append({
                        'field': field,
                        'from': correction['current'],
                        'to': suggested,
                        'reason': correction['reason']
                    })
            else:
                # Datos completos
                corrected_data = suggested
                changes.append({
                    'field': '',
                    'from': correction['current'],
                    'to': suggested,
                    'reason': correction['reason']
                })
        
        # Intentar correcciones automáticas para errores comunes
        for error in validation_result.errors:
            field = error['field']
            rule = error['rule']
            value = error['value']
            
            if rule == ValidationRule.TYPE.value:
                # Conversión de tipos
                try:
                    if 'int' in str(error.get('message', '')) and field:
                        if isinstance(corrected_data, dict) and field in corrected_data:
                            try:
                                corrected_data[field] = int(float(corrected_data[field]))
                                changes.append({
                                    'field': field,
                                    'from': value,
                                    'to': corrected_data[field],
                                    'reason': 'Auto-converted to integer'
                                })
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    pass
        
        return {
            'corrected_data': corrected_data,
            'changes': changes,
            'original_valid': validation_result.valid,
            'correction_count': len(changes)
        }
    
    # ========== MÉTODOS PRIVADOS ==========
    
    @staticmethod
    def _validate_value(
        value: Any,
        field: str,
        schema: Any,
        result: ValidationResult,
        strict: bool,
        allow_extra: bool
    ) -> None:
        """Valida un valor individual contra un esquema."""
        if isinstance(schema, dict):
            # Esquema detallado
            if 'type' in schema:
                # Validar tipo
                Validation._validate_type(
                    value, field, schema['type'], result, strict
                )
                
                # Solo validar más reglas si el tipo es correcto
                if (not result.errors or 
                    all(e['field'] != field for e in result.errors)):
                    
                    # Aplicar reglas adicionales
                    for rule_name, rule_value in schema.items():
                        if rule_name != 'type' and rule_name != 'required':
                            try:
                                Validation._apply_validation_rule(
                                    value, field, rule_name, rule_value, result
                                )
                            except Exception as e:
                                result.add_error(
                                    field=field,
                                    rule=rule_name,
                                    message=f"Rule '{rule_name}' failed: {str(e)}",
                                    value=value
                                )
            
            elif 'properties' in schema:
                # Esquema de objeto
                Validation.validate_structure(
                    value, schema['properties'], strict, allow_extra
                )
            
            elif 'items' in schema:
                # Esquema de array
                if not isinstance(value, list):
                    result.add_error(
                        field=field,
                        rule=ValidationRule.TYPE.value,
                        message=f"Expected list, got {type(value).__name__}",
                        value=value
                    )
                else:
                    for i, item in enumerate(value):
                        Validation._validate_value(
                            item, f"{field}[{i}]", schema['items'], 
                            result, strict, allow_extra
                        )
        
        elif isinstance(schema, type) or isinstance(schema, tuple):
            # Tipo simple
            Validation._validate_type(value, field, schema, result, strict)
        
        elif callable(schema):
            # Función de validación
            try:
                if not schema(value):
                    result.add_error(
                        field=field,
                        rule=ValidationRule.CUSTOM.value,
                        message=f"Custom validation failed",
                        value=value
                    )
            except Exception as e:
                result.add_error(
                    field=field,
                    rule=ValidationRule.CUSTOM.value,
                    message=f"Custom validator error: {str(e)}",
                    value=value
                )
    
    @staticmethod
    def _validate_type(
        value: Any,
        field: str,
        expected_type: Union[Type, Tuple[Type, ...]],
        result: ValidationResult,
        strict: bool
    ) -> None:
        """Valida el tipo de un valor."""
        # Manejar tipos especiales
        type_name = str(expected_type)
        
        if '|' in type_name or 'Union' in type_name:
            # Tipo union (e.g., str | int, Union[str, int])
            types = Validation._parse_union_type(expected_type)
            if not any(Validation._check_type(value, t, strict) for t in types):
                result.add_error(
                    field=field,
                    rule=ValidationRule.TYPE.value,
                    message=f"Expected one of {types}, got {type(value).__name__}",
                    value=value
                )
        
        elif 'List' in type_name or 'list' in type_name:
            # Lista
            if not isinstance(value, list):
                result.add_error(
                    field=field,
                    rule=ValidationRule.TYPE.value,
                    message=f"Expected list, got {type(value).__name__}",
                    value=value
                )
        
        elif 'Dict' in type_name or 'dict' in type_name:
            # Diccionario
            if not isinstance(value, dict):
                result.add_error(
                    field=field,
                    rule=ValidationRule.TYPE.value,
                    message=f"Expected dict, got {type(value).__name__}",
                    value=value
                )
        
        elif not Validation._check_type(value, expected_type, strict):
            result.add_error(
                field=field,
                rule=ValidationRule.TYPE.value,
                message=f"Expected {expected_type}, got {type(value).__name__}",
                value=value
            )
    
    @staticmethod
    def _check_type(value: Any, expected_type: Type, strict: bool) -> bool:
        """Verifica si un valor es de un tipo específico."""
        # None siempre es especial
        if value is None:
            return expected_type is type(None)
        
        # Tipo exacto
        if strict:
            return type(value) == expected_type
        
        # Tipo compatible
        if expected_type == Any:
            return True
        
        # Manejar tipos de Python
        if expected_type == str:
            return isinstance(value, str)
        elif expected_type == int:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == bool:
            return isinstance(value, bool)
        elif expected_type == list:
            return isinstance(value, list)
        elif expected_type == dict:
            return isinstance(value, dict)
        elif expected_type == datetime:
            return isinstance(value, datetime)
        elif expected_type == date:
            return isinstance(value, date)
        elif expected_type == time:
            return isinstance(value, time)
        elif expected_type == Decimal:
            return isinstance(value, Decimal)
        
        # Para otros tipos, usar isinstance
        try:
            return isinstance(value, expected_type)
        except TypeError:
            # Para tipos genéricos, verificar estructura
            return True
    
    @staticmethod
    def _parse_union_type(union_type) -> List[Type]:
        """Parsea tipos union a lista de tipos."""
        if hasattr(union_type, '__args__'):
            return list(union_type.__args__)
        elif isinstance(union_type, str) and '|' in union_type:
            types = []
            for part in union_type.split('|'):
                part = part.strip()
                if part == 'str':
                    types.append(str)
                elif part == 'int':
                    types.append(int)
                elif part == 'float':
                    types.append(float)
                elif part == 'bool':
                    types.append(bool)
                elif part == 'list':
                    types.append(list)
                elif part == 'dict':
                    types.append(dict)
                elif part == 'None':
                    types.append(type(None))
            return types
        else:
            return [union_type]
    
    @staticmethod
    def _apply_validation_rule(
        value: Any,
        field: str,
        rule_name: str,
        rule_value: Any,
        result: ValidationResult
    ) -> None:
        """Aplica una regla de validación específica."""
        if rule_name == 'required':
            if rule_value and value is None:
                result.add_error(
                    field=field,
                    rule=ValidationRule.REQUIRED.value,
                    message="Field is required",
                    value=value
                )
        
        elif rule_name == 'min':
            if value is not None:
                if isinstance(value, (int, float, Decimal)):
                    if value < rule_value:
                        result.add_error(
                            field=field,
                            rule=ValidationRule.MIN.value,
                            message=f"Value must be >= {rule_value}",
                            value=value
                        )
                        # Sugerir corrección
                        result.add_correction(
                            field=field,
                            current=value,
                            suggested=rule_value,
                            reason=f"Value below minimum {rule_value}"
                        )
                elif isinstance(value, str):
                    if len(value) < rule_value:
                        result.add_error(
                            field=field,
                            rule=ValidationRule.LENGTH.value,
                            message=f"Length must be >= {rule_value} characters",
                            value=value
                        )
                elif isinstance(value, (list, dict, set, tuple)):
                    if len(value) < rule_value:
                        result.add_error(
                            field=field,
                            rule=ValidationRule.LENGTH.value,
                            message=f"Must have at least {rule_value} items",
                            value=value
                        )
        
        elif rule_name == 'max':
            if value is not None:
                if isinstance(value, (int, float, Decimal)):
                    if value > rule_value:
                        result.add_error(
                            field=field,
                            rule=ValidationRule.MAX.value,
                            message=f"Value must be <= {rule_value}",
                            value=value
                        )
                        # Sugerir corrección
                        result.add_correction(
                            field=field,
                            current=value,
                            suggested=rule_value,
                            reason=f"Value above maximum {rule_value}"
                        )
                elif isinstance(value, str):
                    if len(value) > rule_value:
                        result.add_error(
                            field=field,
                            rule=ValidationRule.LENGTH.value,
                            message=f"Length must be <= {rule_value} characters",
                            value=value
                        )
                        # Sugerir truncar
                        result.add_correction(
                            field=field,
                            current=value,
                            suggested=value[:rule_value],
                            reason=f"String truncated to {rule_value} characters"
                        )
                elif isinstance(value, (list, dict, set, tuple)):
                    if len(value) > rule_value:
                        result.add_error(
                            field=field,
                            rule=ValidationRule.LENGTH.value,
                            message=f"Must have at most {rule_value} items",
                            value=value
                        )
        
        elif rule_name == 'pattern':
            if value is not None and isinstance(value, str):
                pattern = re.compile(rule_value)
                if not pattern.match(value):
                    result.add_error(
                        field=field,
                        rule=ValidationRule.PATTERN.value,
                        message=f"Value must match pattern: {rule_value}",
                        value=value
                    )
        
        elif rule_name == 'enum' or rule_name == 'choice':
            if value is not None and value not in rule_value:
                result.add_error(
                    field=field,
                    rule=ValidationRule.CHOICE.value,
                    message=f"Value must be one of {rule_value}",
                    value=value
                )
                # Sugerir el valor más cercano si es string
                if isinstance(value, str) and isinstance(rule_value, (list, tuple, set)):
                    closest = Validation._find_closest_match(value, rule_value)
                    if closest:
                        result.add_correction(
                            field=field,
                            current=value,
                            suggested=closest,
                            reason=f"Closest match to '{value}'"
                        )
        
        elif rule_name == 'range':
            if value is not None and isinstance(value, (int, float, Decimal)):
                min_val, max_val = rule_value
                if not (min_val <= value <= max_val):
                    result.add_error(
                        field=field,
                        rule=ValidationRule.RANGE.value,
                        message=f"Value must be between {min_val} and {max_val}",
                        value=value
                    )
                    # Sugerir valor dentro del rango
                    if value < min_val:
                        suggested = min_val
                    else:
                        suggested = max_val
                    result.add_correction(
                        field=field,
                        current=value,
                        suggested=suggested,
                        reason=f"Value clamped to range [{min_val}, {max_val}]"
                    )
        
        elif rule_name == 'custom':
            if callable(rule_value):
                try:
                    if not rule_value(value):
                        result.add_error(
                            field=field,
                            rule=ValidationRule.CUSTOM.value,
                            message="Custom validation failed",
                            value=value
                        )
                except Exception as e:
                    result.add_error(
                        field=field,
                        rule=ValidationRule.CUSTOM.value,
                        message=f"Custom validator error: {str(e)}",
                        value=value
                    )
    
    @staticmethod
    def _validate_email(email: str, result: ValidationResult, **kwargs) -> None:
        """Valida formato de email."""
        if not email or not isinstance(email, str):
            result.add_error(
                field='',
                rule=ValidationRule.EMAIL.value,
                message="Email must be a non-empty string",
                value=email
            )
            return
        
        # Validar formato básico
        if not Validation._patterns['email'].match(email):
            result.add_error(
                field='',
                rule=ValidationRule.EMAIL.value,
                message="Invalid email format",
                value=email
            )
            
            # Intentar corrección
            corrected = email.strip().lower()
            if '@' in corrected:
                parts = corrected.split('@')
                if len(parts) == 2:
                    local, domain = parts
                    # Remover espacios y caracteres extraños
                    local = re.sub(r'[^\w.%+-]', '', local)
                    domain = re.sub(r'[^\w.-]', '', domain)
                    if '.' in domain:
                        suggested = f"{local}@{domain}"
                        if suggested != email:
                            result.add_correction(
                                field='',
                                current=email,
                                suggested=suggested,
                                reason="Cleaned email format"
                            )
    
    @staticmethod
    def _validate_url(url: str, result: ValidationResult, **kwargs) -> None:
        """Valida formato de URL."""
        if not url or not isinstance(url, str):
            result.add_error(
                field='',
                rule=ValidationRule.URL.value,
                message="URL must be a non-empty string",
                value=url
            )
            return
        
        # Validar formato básico
        if not Validation._patterns['url'].match(url):
            # Intentar agregar https:// si falta
            if not url.startswith(('http://', 'https://')):
                corrected = f"https://{url}"
                if Validation._patterns['url'].match(corrected):
                    result.add_correction(
                        field='',
                        current=url,
                        suggested=corrected,
                        reason="Added https:// prefix"
                    )
                else:
                    result.add_error(
                        field='',
                        rule=ValidationRule.URL.value,
                        message="Invalid URL format",
                        value=url
                    )
            else:
                result.add_error(
                    field='',
                    rule=ValidationRule.URL.value,
                    message="Invalid URL format",
                    value=url
                )
    
    @staticmethod
    def _validate_ip_address(ip: str, result: ValidationResult, **kwargs) -> None:
        """Valida dirección IP."""
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            result.add_error(
                field='',
                rule=ValidationRule.IP_ADDRESS.value,
                message="Invalid IP address",
                value=ip
            )
    
    @staticmethod
    def _validate_uuid(uuid_str: str, result: ValidationResult, **kwargs) -> None:
        """Valida UUID."""
        try:
            uuid.UUID(uuid_str)
        except ValueError:
            result.add_error(
                field='',
                rule=ValidationRule.UUID.value,
                message="Invalid UUID format",
                value=uuid_str
            )
    
    @staticmethod
    def _validate_date(date_str: str, result: ValidationResult, **kwargs) -> None:
        """Valida fecha."""
        formats = kwargs.get('formats', ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'])
        
        for fmt in formats:
            try:
                datetime.strptime(date_str, fmt)
                return
            except ValueError:
                continue
        
        result.add_error(
            field='',
            rule=ValidationRule.DATE.value,
            message=f"Invalid date format. Expected one of: {formats}",
            value=date_str
        )
    
    @staticmethod
    def _validate_time(time_str: str, result: ValidationResult, **kwargs) -> None:
        """Valida hora."""
        formats = kwargs.get('formats', ['%H:%M:%S', '%H:%M'])
        
        for fmt in formats:
            try:
                datetime.strptime(time_str, fmt)
                return
            except ValueError:
                continue
        
        result.add_error(
            field='',
            rule=ValidationRule.TIME.value,
            message=f"Invalid time format. Expected one of: {formats}",
            value=time_str
        )
    
    @staticmethod
    def _validate_datetime(datetime_str: str, result: ValidationResult, **kwargs) -> None:
        """Valida fecha y hora."""
        formats = kwargs.get('formats', [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ])
        
        for fmt in formats:
            try:
                datetime.strptime(datetime_str, fmt)
                return
            except ValueError:
                continue
        
        result.add_error(
            field='',
            rule=ValidationRule.DATETIME.value,
            message=f"Invalid datetime format. Expected one of: {formats}",
            value=datetime_str
        )
    
    @staticmethod
    def _validate_pattern(
        value: str, 
        pattern_name: str, 
        result: ValidationResult, 
        **kwargs
    ) -> None:
        """Valida contra un patrón predefinido."""
        pattern = Validation._patterns.get(pattern_name)
        if pattern and not pattern.match(value):
            result.add_error(
                field='',
                rule=ValidationRule.PATTERN.value,
                message=f"Invalid {pattern_name} format",
                value=value
            )
    
    @staticmethod
    def _find_closest_match(value: str, choices: List[str]) -> Optional[str]:
        """Encuentra la coincidencia más cercana usando distancia de Levenshtein."""
        if not value or not choices:
            return None
        
        from rapidfuzz import fuzz
        
        best_match = None
        best_score = 0
        
        for choice in choices:
            if isinstance(choice, str):
                score = fuzz.ratio(value.lower(), choice.lower())
                if score > best_score:
                    best_score = score
                    best_match = choice
        
        # Solo retornar si la similitud es razonable
        return best_match if best_score > 70 else None