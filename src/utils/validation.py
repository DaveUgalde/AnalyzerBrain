"""
Utilidades centralizadas de validación de datos para ANALYZERBRAIN.

Este módulo define un conjunto de validadores reutilizables y consistentes
para verificar la integridad, el tipo, el formato y la estructura de datos
a lo largo de todo el sistema. Actúa como una capa de validación previa
antes de la lógica de negocio, reduciendo errores en tiempo de ejecución
y mejorando la calidad de los mensajes de error.

El sistema de validación está diseñado para:
- Validar tipos, rangos y longitudes de datos primitivos.
- Verificar formatos comunes (emails, JSON, expresiones regulares).
- Validar rutas de archivos y directorios.
- Validar estructuras de diccionarios y esquemas simples.
- Integrarse con modelos Pydantic para validaciones complejas.
- Proveer errores estructurados y consistentes mediante excepciones propias.

Todas las validaciones lanzan excepciones de tipo `ValidationError`
(definida en `src.core.exceptions`) con información detallada, sugerencias
y metadatos útiles para depuración y respuesta en APIs.

Características principales:
- API estática y genérica mediante la clase `Validator`.
- Mensajes de error claros y orientados a corrección.
- Compatibilidad con validaciones síncronas y modelos Pydantic.
- Reutilizable en agentes, APIs, configuración y lógica de dominio.

Dependencias:
- pydantic (validación avanzada de modelos)
- email-validator (validación de correos electrónicos)
- src.core.exceptions (sistema de excepciones del proyecto)
- Librería estándar de Python (re, json, pathlib, datetime, typing)

Clases:
- Validator[T]: Validador genérico con métodos estáticos para múltiples tipos de datos.

Instancias globales:
- validator: Instancia compartida de `Validator` para uso conveniente.

Autor: ANALYZERBRAIN Team
Fecha: 2026
Versión: 1.0.0
"""


import re
import json
from pathlib import Path
from typing import Any, Dict, Type, Optional, Union, TypeVar, Generic, Tuple
from typing import Collection, List
from pydantic import BaseModel, ValidationError as PydanticValidationError
from email_validator import validate_email as email_validator, EmailNotValidError

from src.core.exceptions import ValidationError


T = TypeVar('T')
Number = Union[int, float]
ExpectedType = Union[Type[Any], Tuple[Type[Any], ...]]


class Validator(Generic[T]):
    """Validador genérico para diferentes tipos de datos."""
    
    @staticmethod
    def validate_not_empty(
        value: Optional[str | Collection[Any]],
        field_name: str = "value"
    ) -> str | Collection[Any]:
        """Valida que un valor no esté vacío."""

        if value is None:
            raise ValidationError(
                message=f"{field_name} no puede ser nulo",
                field=field_name,
                value=value,
                suggestion="Proporcione un valor no nulo"
            )

        if isinstance(value, str):
            if not value.strip():
                raise ValidationError(
                    message=f"{field_name} no puede estar vacío",
                    field=field_name,
                    value=value,
                    suggestion="Proporcione una cadena no vacía"
                )
        else:
            if not value:
                raise ValidationError(
                    message=f"{field_name} no puede estar vacío",
                    field=field_name,
                    value=value,
                    suggestion=f"Proporcione un {type(value).__name__} no vacío"
                )

        return value

    
    @staticmethod
    def validate_type(
        value: Any, 
        expected_type: ExpectedType, 
        field_name: str = "value"
    ) -> Any:
        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                type_name = ", ".join(t.__name__ for t in expected_type)
            else:
                type_name = expected_type.__name__

            raise ValidationError(
                message=f"{field_name} debe ser de tipo {type_name}",
                field=field_name,
                value=value,
                value_type=type(value).__name__,
                suggestion=f"Convierta a {type_name}",
            )

        return value
    
    @staticmethod
    def validate_string_length(
        value: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        field_name: str = "value"
    ) -> str:
        """Valida la longitud de una cadena."""
        Validator.validate_type(value, str, field_name)
        
        length = len(value)
        
        if min_length is not None and length < min_length:
            raise ValidationError(
                message=f"{field_name} debe tener al menos {min_length} caracteres",
                field=field_name,
                value=value,
                actual_length=length,
                suggestion=f"Aumente la longitud a al menos {min_length} caracteres"
            )
        
        if max_length is not None and length > max_length:
            raise ValidationError(
                message=f"{field_name} no puede tener más de {max_length} caracteres",
                field=field_name,
                value=value,
                actual_length=length,
                suggestion=f"Reduzca la longitud a máximo {max_length} caracteres"
            )
        
        return value
   
    
    @staticmethod
    def validate_number_range(
        value: Number,
        min_value: Optional[Number] = None,
        max_value: Optional[Number] = None,
        field_name: str = "value"
    ) -> Number:
        Validator.validate_type(value, (int, float), field_name)

        if min_value is not None and value < min_value:
            raise ValidationError(
                message=f"{field_name} debe ser mayor o igual a {min_value}",
                field=field_name,
                value=value,
                value_type=type(value).__name__,
                suggestion=f"Incremente el valor a al menos {min_value}",
            )

        if max_value is not None and value > max_value:
            raise ValidationError(
                message=f"{field_name} debe ser menor o igual a {max_value}",
                field=field_name,
                value=value,
                value_type=type(value).__name__,
                suggestion=f"Reduzca el valor a máximo {max_value}",
            )

        return value
    
    @staticmethod
    def validate_email(email: str, field_name: str = "email") -> str:
        """Valida una dirección de email."""
        Validator.validate_type(email, str, field_name)
        Validator.validate_string_length(email, min_length=3, max_length=254, field_name=field_name)
        
        try:
            email_info = email_validator(email, check_deliverability=False)
            return email_info.normalized
        except EmailNotValidError as e:
            raise ValidationError(
                message=f"{field_name} no es una dirección de email válida",
                field=field_name,
                value=email,
                suggestion=str(e)
            )
    
    @staticmethod
    def validate_path(
        path: Union[str, Path],
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        field_name: str = "path"
    ) -> Path:
        """Valida una ruta de archivo o directorio."""
        path_obj = Path(path) if isinstance(path, str) else path
        
        if must_exist and not path_obj.exists():
            raise ValidationError(
                message=f"{field_name} no existe: {path_obj}",
                field=field_name,
                value=str(path_obj),
                suggestion="Verifique que la ruta exista"
            )
        
        if must_be_file and not path_obj.is_file():
            raise ValidationError(
                message=f"{field_name} no es un archivo: {path_obj}",
                field=field_name,
                value=str(path_obj),
                suggestion="Proporcione una ruta a un archivo válido"
            )
        
        if must_be_dir and not path_obj.is_dir():
            raise ValidationError(
                message=f"{field_name} no es un directorio: {path_obj}",
                field=field_name,
                value=str(path_obj),
                suggestion="Proporcione una ruta a un directorio válido"
            )
        
        return path_obj
    
    @staticmethod
    def validate_regex(
        value: str,
        pattern: str,
        field_name: str = "value",
        flags: int = 0
    ) -> str:
        """Valida una cadena contra una expresión regular."""
        Validator.validate_type(value, str, field_name)
        
        if not re.match(pattern, value, flags=flags):
            raise ValidationError(
                message=f"{field_name} no coincide con el patrón requerido",
                field=field_name,
                value=value,
                details={"pattern": pattern},
                suggestion=f"El valor debe coincidir con: {pattern}"
            )
        
        return value
    
    @staticmethod
    def validate_json(
        json_str: str,
        schema: Optional[Dict[str, Any]] = None,
        field_name: str = "json"
    ) -> Dict[str, Any]:
        """Valida una cadena JSON."""
        Validator.validate_type(json_str, str, field_name)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValidationError(
                message=f"{field_name} no es un JSON válido",
                field=field_name,
                value=json_str,
                suggestion=f"Error de sintaxis JSON: {str(e)}"
            )
        
        if schema:
            # Validación básica de esquema (simplificada)
            Validator.validate_dict_structure(data, schema, field_name)
        
        return data
    
    @staticmethod
    def validate_dict_structure(
        data: Dict[str, Any],
        structure: Dict[str, Any],
        field_name: str = "data"
    ) -> Dict[str, Any]:
        """Valida la estructura de un diccionario."""
        Validator.validate_type(data, dict, field_name)
        
        for key, expected_type in structure.items():
            if key not in data:
                raise ValidationError(
                    message=f"Falta campo requerido: {key}",
                    field=key,
                    suggestion=f"Agregue el campo '{key}' de tipo {expected_type}"
                )
            
            if not isinstance(data[key], expected_type):
                raise ValidationError(
                    message=f"Campo '{key}' debe ser de tipo {expected_type}",
                    field=key,
                    value=data[key],
                    value_type=type(data[key]).__name__,
                    suggestion=f"Convierta a {expected_type}"
                )
        
        return data
    
    @staticmethod
    def validate_pydantic_model(
        data: Dict[str, Any],
        model_class: type[BaseModel],
        field_name: str = "data"
    ) -> BaseModel:
        """Valida datos usando un modelo Pydantic."""
        try:
            return model_class(**data)
        except PydanticValidationError as e:
            errors: List[dict[str,Any]] = []
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                errors.append({
                    "field": field,
                    "message": error["msg"],
                    "type": error["type"]
                })
            
            raise ValidationError(
                message=f"Error de validación en {field_name}",
                field=field_name,
                value=data,
                details={"errors": errors},
                suggestion="Corrija los errores de validación listados"
            )


# Instancia global para uso convenientae
validator = Validator[Any]()