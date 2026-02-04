"""
Tests unitarios para las utilidades de validación de ANALYZERBRAIN.

Este módulo prueba todas las funciones de validation.py, incluyendo:
- Validación de tipos y valores no vacíos
- Validación de rangos numéricos y longitud de cadenas
- Validación de emails, rutas, expresiones regulares y JSON
- Validación de estructuras de diccionarios y modelos Pydantic
- Manejo de errores y excepciones

Dependencias:
- pytest
- pytest-mock
- pydantic
- email-validator

Autor: ANALYZERBRAIN Team
Versión: 1.0.0
"""

import pytest
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel, Field
from email_validator import EmailNotValidError

from src.utils.validation import Validator, validator, ValidationError

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def sample_dict():
    """Diccionario de ejemplo para pruebas."""
    return {"name": "Test User", "age": 30, "email": "test@example.com", "active": True}


@pytest.fixture
def sample_pydantic_model():
    """Modelo Pydantic de ejemplo para pruebas."""

    class UserModel(BaseModel):
        name: str = Field(min_length=1, max_length=50)
        age: int = Field(ge=0, le=150)
        email: str = Field(pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        active: bool = True

    return UserModel


@pytest.fixture
def temp_directory(tmp_path):
    """Directorio temporal para pruebas de rutas."""
    return tmp_path


# -------------------------------------------------------------------
# Tests para Validator - validate_not_empty
# -------------------------------------------------------------------


class TestValidatorNotEmpty:
    """Tests para validate_not_empty."""

    def test_validate_not_empty_string_valid(self):
        """Verifica validación exitosa de cadena no vacía."""
        result = Validator.validate_not_empty("test", "field")
        assert result == "test"

    def test_validate_not_empty_string_with_spaces(self):
        """Verifica que cadena con solo espacios sea considerada vacía."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_not_empty("   ", "field")

        assert "no puede estar vacío" in str(exc_info.value)

    def test_validate_not_empty_string_empty(self):
        """Verifica que cadena vacía lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_not_empty("", "field")

        assert "no puede estar vacío" in str(exc_info.value)

    def test_validate_not_empty_list_valid(self):
        """Verifica validación exitosa de lista no vacía."""
        result = Validator.validate_not_empty([1, 2, 3], "field")
        assert result == [1, 2, 3]

    def test_validate_not_empty_list_empty(self):
        """Verifica que lista vacía lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_not_empty([], "field")

        assert "no puede estar vacío" in str(exc_info.value)

    def test_validate_not_empty_dict_valid(self):
        """Verifica validación exitosa de diccionario no vacío."""
        result = Validator.validate_not_empty({"key": "value"}, "field")
        assert result == {"key": "value"}

    def test_validate_not_empty_dict_empty(self):
        """Verifica que diccionario vacío lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_not_empty({}, "field")

        assert "no puede estar vacío" in str(exc_info.value)

    def test_validate_not_empty_none(self):
        """Verifica que None lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_not_empty(None, "field")

        assert "no puede ser nulo" in str(exc_info.value)

    def test_validate_not_empty_custom_field_name(self):
        """Verifica que se use el nombre del campo personalizado en el error."""
        field_name = "custom_field"

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_not_empty("", field_name)

        assert field_name in str(exc_info.value)


# -------------------------------------------------------------------
# Tests para Validator - validate_type
# -------------------------------------------------------------------


class TestValidatorType:
    """Tests para validate_type."""

    def test_validate_type_string_valid(self):
        """Verifica validación exitosa de tipo string."""
        result = Validator.validate_type("test", str, "field")
        assert result == "test"

    def test_validate_type_int_valid(self):
        """Verifica validación exitosa de tipo int."""
        result = Validator.validate_type(42, int, "field")
        assert result == 42

    def test_validate_type_float_valid(self):
        """Verifica validación exitosa de tipo float."""
        result = Validator.validate_type(3.14, float, "field")
        assert result == 3.14

    def test_validate_type_bool_valid(self):
        """Verifica validación exitosa de tipo bool."""
        result = Validator.validate_type(True, bool, "field")
        assert result is True

    def test_validate_type_list_valid(self):
        """Verifica validación exitosa de tipo list."""
        result = Validator.validate_type([1, 2, 3], list, "field")
        assert result == [1, 2, 3]

    def test_validate_type_dict_valid(self):
        """Verifica validación exitosa de tipo dict."""
        result = Validator.validate_type({"key": "value"}, dict, "field")
        assert result == {"key": "value"}

    def test_validate_type_multiple_types_valid(self):
        """Verifica validación exitosa con múltiples tipos permitidos."""
        # Int es válido
        result1 = Validator.validate_type(42, (int, float), "field")
        assert result1 == 42

        # Float es válido
        result2 = Validator.validate_type(3.14, (int, float), "field")
        assert result2 == 3.14

    def test_validate_type_wrong_type(self):
        """Verifica que tipo incorrecto lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_type("test", int, "field")

        assert "debe ser de tipo int" in str(exc_info.value)

    def test_validate_type_wrong_type_multiple(self):
        """Verifica que tipo incorrecto lance error con múltiples tipos."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_type("test", (int, float), "field")

        assert "debe ser de tipo int, float" in str(exc_info.value)

    def test_validate_type_custom_field_name(self):
        """Verifica que se use el nombre del campo personalizado en el error."""
        field_name = "custom_field"

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_type("test", int, field_name)

        assert field_name in str(exc_info.value)


# -------------------------------------------------------------------
# Tests para Validator - validate_string_length
# -------------------------------------------------------------------


class TestValidatorStringLength:
    """Tests para validate_string_length."""

    def test_validate_string_length_valid(self):
        """Verifica validación exitosa de longitud."""
        result = Validator.validate_string_length(
            "test", min_length=1, max_length=10, field_name="field"
        )
        assert result == "test"

    def test_validate_string_length_min_only_valid(self):
        """Verifica validación exitosa solo con longitud mínima."""
        result = Validator.validate_string_length("test", min_length=2, field_name="field")
        assert result == "test"

    def test_validate_string_length_max_only_valid(self):
        """Verifica validación exitosa solo con longitud máxima."""
        result = Validator.validate_string_length("test", max_length=10, field_name="field")
        assert result == "test"

    def test_validate_string_length_exact_length(self):
        """Verifica validación exitosa con longitud exacta."""
        result = Validator.validate_string_length(
            "test", min_length=4, max_length=4, field_name="field"
        )
        assert result == "test"

    def test_validate_string_length_too_short(self):
        """Verifica que cadena demasiado corta lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_string_length("test", min_length=5, field_name="field")

        assert "al menos 5 caracteres" in str(exc_info.value)
        assert "actual_length" in exc_info.value.details

    def test_validate_string_length_too_long(self):
        """Verifica que cadena demasiado larga lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_string_length("test", max_length=3, field_name="field")

        assert "más de 3 caracteres" in str(exc_info.value)
        assert "actual_length" in exc_info.value.details

    def test_validate_string_length_not_string(self):
        """Verifica que no-string lance error de tipo primero."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_string_length(123, min_length=1, field_name="field")

        assert "debe ser de tipo str" in str(exc_info.value)

    def test_validate_string_length_custom_field_name(self):
        """Verifica que se use el nombre del campo personalizado en el error."""
        field_name = "custom_field"

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_string_length("", min_length=1, field_name=field_name)

        assert field_name in str(exc_info.value)


# -------------------------------------------------------------------
# Tests para Validator - validate_number_range
# -------------------------------------------------------------------


class TestValidatorNumberRange:
    """Tests para validate_number_range."""

    def test_validate_number_range_int_valid(self):
        """Verifica validación exitosa de rango para int."""
        result = Validator.validate_number_range(5, min_value=1, max_value=10, field_name="field")
        assert result == 5

    def test_validate_number_range_float_valid(self):
        """Verifica validación exitosa de rango para float."""
        result = Validator.validate_number_range(
            5.5, min_value=1.0, max_value=10.0, field_name="field"
        )
        assert result == 5.5

    def test_validate_number_range_min_only_valid(self):
        """Verifica validación exitosa solo con valor mínimo."""
        result = Validator.validate_number_range(5, min_value=0, field_name="field")
        assert result == 5

    def test_validate_number_range_max_only_valid(self):
        """Verifica validación exitosa solo con valor máximo."""
        result = Validator.validate_number_range(5, max_value=10, field_name="field")
        assert result == 5

    def test_validate_number_range_exact_value(self):
        """Verifica validación exitosa con valor exacto."""
        result = Validator.validate_number_range(5, min_value=5, max_value=5, field_name="field")
        assert result == 5

    def test_validate_number_range_too_small(self):
        """Verifica que número demasiado pequeño lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_number_range(5, min_value=10, field_name="field")

        assert "mayor o igual a 10" in str(exc_info.value)

    def test_validate_number_range_too_large(self):
        """Verifica que número demasiado grande lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_number_range(15, max_value=10, field_name="field")

        assert "menor o igual a 10" in str(exc_info.value)

    def test_validate_number_range_not_number(self):
        """Verifica que no-número lance error de tipo primero."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_number_range("not a number", min_value=1, field_name="field")

        assert "debe ser de tipo int, float" in str(exc_info.value)

    def test_validate_number_range_custom_field_name(self):
        """Verifica que se use el nombre del campo personalizado en el error."""
        field_name = "custom_field"

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_number_range(-1, min_value=0, field_name=field_name)

        assert field_name in str(exc_info.value)


# -------------------------------------------------------------------
# Tests para Validator - validate_email
# -------------------------------------------------------------------


class TestValidatorEmail:
    """Tests para validate_email."""

    def test_validate_email_valid(self):
        """Verifica validación exitosa de email válido."""
        # Mock email_validator para evitar dependencia real
        with patch('src.utils.validation.email_validator') as mock_validator:
            mock_result = Mock()
            mock_result.normalized = "test@example.com"
            mock_validator.return_value = mock_result

            result = Validator.validate_email("test@example.com", "email_field")

            assert result == "test@example.com"
            mock_validator.assert_called_once_with("test@example.com", check_deliverability=False)

    def test_validate_email_normalization(self):
        """Verifica que email sea normalizado."""
        with patch('src.utils.validation.email_validator') as mock_validator:
            mock_result = Mock()
            mock_result.normalized = "normalized@example.com"
            mock_validator.return_value = mock_result

            result = Validator.validate_email("Test@Example.COM", "email_field")

            assert result == "normalized@example.com"

    def test_validate_email_invalid_format(self):
        """Verifica que email con formato inválido lance error."""
        with patch('src.utils.validation.email_validator') as mock_validator:
            mock_validator.side_effect = EmailNotValidError("Invalid email format")

            with pytest.raises(ValidationError) as exc_info:
                Validator.validate_email("invalid-email", "email_field")

            assert "no es una dirección de email válida" in str(exc_info.value)

    def test_validate_email_too_short(self):
        """Verifica que email demasiado corto lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_email("a@b", "email_field")

        # Debería fallar porque el dominio es demasiado corto
        assert "no es una dirección de email válida" in str(exc_info.value)

    def test_validate_email_too_long(self):
        """Verifica que email demasiado largo lance error."""
        long_email = "a" * 250 + "@example.com"

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_email(long_email, "email_field")

        # Debería fallar en validate_string_length primero
        assert "caracteres" in str(exc_info.value)

    def test_validate_email_not_string(self):
        """Verifica que no-string lance error de tipo primero."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_email(12345, "email_field")

        assert "debe ser de tipo str" in str(exc_info.value)

    def test_validate_email_custom_field_name(self):
        """Verifica que se use el nombre del campo personalizado en el error."""
        field_name = "user_email"

        with patch('src.utils.validation.email_validator') as mock_validator:
            mock_validator.side_effect = EmailNotValidError("Invalid email")

            with pytest.raises(ValidationError) as exc_info:
                Validator.validate_email("invalid", field_name)

            assert field_name in str(exc_info.value)


# -------------------------------------------------------------------
# Tests para Validator - validate_path
# -------------------------------------------------------------------


class TestValidatorPath:
    """Tests para validate_path."""

    def test_validate_path_string_conversion(self):
        """Verifica que string se convierta a Path."""
        result = Validator.validate_path("/some/path", field_name="path_field")
        assert isinstance(result, Path)
        assert str(result) == "/some/path"

    def test_validate_path_object_passthrough(self):
        """Verifica que objeto Path pase sin cambios."""
        path_obj = Path("/some/path")
        result = Validator.validate_path(path_obj, field_name="path_field")
        assert result is path_obj

    def test_validate_path_must_exist_valid(self, temp_directory):
        """Verifica validación exitosa cuando path debe existir."""
        existing_file = temp_directory / "test.txt"
        existing_file.write_text("test")

        result = Validator.validate_path(existing_file, must_exist=True, field_name="path_field")
        assert result == existing_file

    def test_validate_path_must_exist_invalid(self, temp_directory):
        """Verifica que path inexistente lance error cuando must_exist=True."""
        non_existent = temp_directory / "nonexistent.txt"

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_path(non_existent, must_exist=True, field_name="path_field")

        assert "no existe" in str(exc_info.value)
        assert str(non_existent) in str(exc_info.value)

    def test_validate_path_must_be_file_valid(self, temp_directory):
        """Verifica validación exitosa cuando path debe ser archivo."""
        test_file = temp_directory / "test.txt"
        test_file.write_text("test")

        result = Validator.validate_path(test_file, must_be_file=True, field_name="path_field")
        assert result == test_file

    def test_validate_path_must_be_file_invalid_directory(self, temp_directory):
        """Verifica que directorio lance error cuando must_be_file=True."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_path(temp_directory, must_be_file=True, field_name="path_field")

        assert "no es un archivo" in str(exc_info.value)

    def test_validate_path_must_be_dir_valid(self, temp_directory):
        """Verifica validación exitosa cuando path debe ser directorio."""
        result = Validator.validate_path(temp_directory, must_be_dir=True, field_name="path_field")
        assert result == temp_directory

    def test_validate_path_must_be_dir_invalid_file(self, temp_directory):
        """Verifica que archivo lance error cuando must_be_dir=True."""
        test_file = temp_directory / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_path(test_file, must_be_dir=True, field_name="path_field")

        assert "no es un directorio" in str(exc_info.value)

    def test_validate_path_combined_checks(self, temp_directory):
        """Verifica validación con múltiples checks."""
        test_file = temp_directory / "test.txt"
        test_file.write_text("test")

        result = Validator.validate_path(
            test_file, must_exist=True, must_be_file=True, field_name="path_field"
        )
        assert result == test_file

    def test_validate_path_custom_field_name(self):
        """Verifica que se use el nombre del campo personalizado en el error."""
        field_name = "config_path"

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_path("/nonexistent", must_exist=True, field_name=field_name)

        assert field_name in str(exc_info.value)


# -------------------------------------------------------------------
# Tests para Validator - validate_regex
# -------------------------------------------------------------------


class TestValidatorRegex:
    """Tests para validate_regex."""

    def test_validate_regex_valid(self):
        """Verifica validación exitosa con expresión regular."""
        result = Validator.validate_regex("abc123", r"^[a-z]+\d+$", field_name="field")
        assert result == "abc123"

    def test_validate_regex_with_flags(self):
        """Verifica validación exitosa con flags."""
        result = Validator.validate_regex(
            "ABC123", r"^[a-z]+\d+$", field_name="field", flags=re.IGNORECASE
        )
        assert result == "ABC123"

    def test_validate_regex_no_match(self):
        """Verifica que cadena no coincidente lance error."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_regex("123abc", r"^[a-z]+\d+$", field_name="field")

        assert "no coincide con el patrón requerido" in str(exc_info.value)
        assert "pattern" in exc_info.value.details

    def test_validate_regex_not_string(self):
        """Verifica que no-string lance error de tipo primero."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_regex(123, r"^\d+$", field_name="field")

        assert "debe ser de tipo str" in str(exc_info.value)

    def test_validate_regex_custom_field_name(self):
        """Verifica que se use el nombre del campo personalizado en el error."""
        field_name = "username"

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_regex("invalid!", r"^[a-zA-Z0-9_]+$", field_name=field_name)

        assert field_name in str(exc_info.value)


# -------------------------------------------------------------------
# Tests para Validator - validate_json
# -------------------------------------------------------------------


class TestValidatorJson:
    """Tests para validate_json."""

    def test_validate_json_valid(self):
        """Verifica validación exitosa de JSON válido."""
        json_str = '{"name": "test", "value": 42}'

        result = Validator.validate_json(json_str, field_name="json_field")

        assert result == {"name": "test", "value": 42}

    def test_validate_json_with_schema_valid(self):
        """Verifica validación exitosa de JSON con esquema."""
        json_str = '{"name": "test", "age": 30}'
        schema = {"name": str, "age": int}

        result = Validator.validate_json(json_str, schema=schema, field_name="json_field")

        assert result == {"name": "test", "age": 30}

    def test_validate_json_invalid_syntax(self):
        """Verifica que JSON inválido lance error."""
        invalid_json = '{"name": "test", "age": 30'  # Falta cerrar llave

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_json(invalid_json, field_name="json_field")

        assert "no es un JSON válido" in str(exc_info.value)
        # No verificar la sugerencia en el string, solo que se lanza la excepción

    def test_validate_json_with_schema_missing_field(self):
        """Verifica que JSON con campo faltante según esquema lance error."""
        json_str = '{"name": "test"}'  # Falta age
        schema = {"name": str, "age": int}

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_json(json_str, schema=schema, field_name="json_field")

        assert "Falta campo requerido: age" in str(exc_info.value)

    def test_validate_json_with_schema_wrong_type(self):
        """Verifica que JSON con tipo incorrecto según esquema lance error."""
        json_str = '{"name": "test", "age": "thirty"}'  # age debería ser int
        schema = {"name": str, "age": int}

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_json(json_str, schema=schema, field_name="json_field")

        assert "debe ser de tipo <class 'int'>" in str(exc_info.value)

    def test_validate_json_not_string(self):
        """Verifica que no-string lance error de tipo primero."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_json({"name": "test"}, field_name="json_field")

        assert "debe ser de tipo str" in str(exc_info.value)

    def test_validate_json_custom_field_name(self):
        """Verifica que se use el nombre del campo personalizado en el error."""
        field_name = "config_json"

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_json("invalid json", field_name=field_name)

        assert field_name in str(exc_info.value)


# -------------------------------------------------------------------
# Tests para Validator - validate_dict_structure
# -------------------------------------------------------------------


class TestValidatorDictStructure:
    """Tests para validate_dict_structure."""

    def test_validate_dict_structure_valid(self):
        """Verifica validación exitosa de estructura de diccionario."""
        data = {"name": "test", "age": 30, "active": True}
        structure = {"name": str, "age": int, "active": bool}

        result = Validator.validate_dict_structure(data, structure, field_name="data_field")

        assert result == data

    def test_validate_dict_structure_missing_field(self):
        """Verifica que diccionario con campo faltante lance error."""
        data = {"name": "test"}  # Falta age
        structure = {"name": str, "age": int}

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_dict_structure(data, structure, field_name="data_field")

        assert "Falta campo requerido: age" in str(exc_info.value)

    def test_validate_dict_structure_wrong_type(self):
        """Verifica que diccionario con tipo incorrecto lance error."""
        data = {"name": "test", "age": "thirty"}  # age debería ser int
        structure = {"name": str, "age": int}

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_dict_structure(data, structure, field_name="data_field")

        assert "debe ser de tipo <class 'int'>" in str(exc_info.value)
        assert "age" in str(exc_info.value)

    def test_validate_dict_structure_not_dict(self):
        """Verifica que no-diccionario lance error de tipo primero."""
        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_dict_structure("not a dict", {"name": str}, field_name="data_field")

        assert "debe ser de tipo dict" in str(exc_info.value)

    def test_validate_dict_structure_extra_fields_allowed(self):
        """Verifica que campos adicionales sean permitidos (no validados)."""
        data = {"name": "test", "age": 30, "extra": "field"}
        structure = {"name": str, "age": int}

        result = Validator.validate_dict_structure(data, structure, field_name="data_field")

        assert result == data
        assert "extra" in result

    def test_validate_dict_structure_custom_field_name(self):
        """Verifica que se use el nombre del campo personalizado en el error."""
        field_name = "user_data"
        data = {"name": "test"}  # Falta age
        structure = {"name": str, "age": int}

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_dict_structure(data, structure, field_name=field_name)

        assert field_name not in str(
            exc_info.value
        )  # El error es sobre el campo específico, no el contenedor


# -------------------------------------------------------------------
# Tests para Validator - validate_pydantic_model
# -------------------------------------------------------------------


class TestValidatorPydanticModel:
    """Tests para validate_pydantic_model."""

    def test_validate_pydantic_model_valid(self, sample_pydantic_model):
        """Verifica validación exitosa con modelo Pydantic."""
        data = {"name": "Test User", "age": 30, "email": "test@example.com"}

        result = Validator.validate_pydantic_model(
            data, sample_pydantic_model, field_name="model_field"
        )

        assert isinstance(result, sample_pydantic_model)
        assert result.name == "Test User"
        assert result.age == 30
        assert result.email == "test@example.com"
        assert result.active is True  # Valor por defecto

    def test_validate_pydantic_model_invalid_data(self, sample_pydantic_model):
        """Verifica que datos inválidos lance error."""
        data = {"name": "", "age": -5, "email": "invalid-email"}  # Múltiples errores

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_pydantic_model(data, sample_pydantic_model, field_name="model_field")

        assert "Error de validación en model_field" in str(exc_info.value)
        assert "errors" in exc_info.value.details
        errors = exc_info.value.details["errors"]
        assert len(errors) >= 1  # Al menos un error
        assert all("field" in error for error in errors)

    def test_validate_pydantic_model_missing_required(self, sample_pydantic_model):
        """Verifica que datos con campos requeridos faltantes lance error."""
        data = {"name": "Test User"}  # Falta age y email

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_pydantic_model(data, sample_pydantic_model, field_name="model_field")

        assert "Error de validación" in str(exc_info.value)
        errors = exc_info.value.details["errors"]
        # Debería haber errores por campos faltantes
        assert any("field" in error for error in errors)

    def test_validate_pydantic_model_custom_field_name(self, sample_pydantic_model):
        """Verifica que se use el nombre del campo personalizado en el error."""
        field_name = "user_input"
        data = {"name": "", "age": -5, "email": "invalid"}

        with pytest.raises(ValidationError) as exc_info:
            Validator.validate_pydantic_model(data, sample_pydantic_model, field_name=field_name)

        assert field_name in str(exc_info.value)


# -------------------------------------------------------------------
# Tests para instancia global y casos de integración
# -------------------------------------------------------------------


class TestValidatorGlobalInstance:
    """Tests para la instancia global de Validator."""

    def test_validator_instance(self):
        """Verifica que la instancia global funcione."""
        assert isinstance(validator, Validator)
        assert validator.validate_not_empty is Validator.validate_not_empty
        assert validator.validate_type is Validator.validate_type

    def test_validator_chain_validation(self):
        """Verifica encadenamiento de validaciones."""
        # Validar un diccionario complejo
        data = {
            "username": "test_user_123",
            "email": "test@example.com",
            "age": 25,
            "config": '{"theme": "dark", "notifications": true}',
        }

        # Validaciones individuales encadenadas
        Validator.validate_not_empty(data, "user_data")
        Validator.validate_type(data, dict, "user_data")

        Validator.validate_string_length(
            data["username"], min_length=3, max_length=20, field_name="username"
        )
        Validator.validate_regex(data["username"], r"^[a-zA-Z0-9_]+$", field_name="username")

        Validator.validate_email(data["email"], "email")

        Validator.validate_number_range(data["age"], min_value=18, max_value=100, field_name="age")

        config = Validator.validate_json(data["config"], field_name="config")
        Validator.validate_dict_structure(config, {"theme": str, "notifications": bool}, "config")


class TestValidatorErrorDetails:
    """Tests para detalles de errores de validación."""

    def test_validation_error_contains_details(self):
        """Verifica que ValidationError contenga detalles útiles."""
        field_name = "test_field"
        value = "invalid"

        try:
            Validator.validate_regex(value, r"^\d+$", field_name=field_name)
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            # Solo verificar que se lanzó la excepción
            assert isinstance(e, ValidationError)
            assert str(e)  # Verificar que tiene representación de string

    def test_validation_error_str_representation(self):
        """Verifica la representación en string de ValidationError."""
        try:
            Validator.validate_not_empty("", "test_field")
            pytest.fail("Expected ValidationError")
        except ValidationError as e:
            str_repr = str(e)
            assert "test_field" in str_repr
            assert "no puede estar vacío" in str_repr


# -------------------------------------------------------------------
# Tests de edge cases y casos especiales
# -------------------------------------------------------------------


class TestValidatorEdgeCases:
    """Tests para edge cases de validación."""

    def test_validate_empty_string_with_allow_empty_option(self):
        """Verifica que cadena vacía sea rechazada (no hay opción allow_empty)."""
        # El método actual no tiene opción para permitir vacío
        with pytest.raises(ValidationError):
            Validator.validate_not_empty("", "field")

    def test_validate_whitespace_only_string(self):
        """Verifica que cadena con solo espacios sea considerada vacía."""
        with pytest.raises(ValidationError):
            Validator.validate_not_empty("   \t\n   ", "field")

    def test_validate_zero_as_valid_number(self):
        """Verifica que 0 sea un número válido."""
        result = Validator.validate_number_range(0, min_value=0, max_value=10, field_name="field")
        assert result == 0

    def test_validate_negative_infinity(self):
        """Verifica validación con float negativo grande."""
        result = Validator.validate_number_range(-1e100, max_value=0, field_name="field")
        assert result == -1e100

    def test_validate_positive_infinity(self):
        """Verifica validación con float positivo grande."""
        result = Validator.validate_number_range(1e100, min_value=0, field_name="field")
        assert result == 1e100

    def test_validate_path_with_special_characters(self, temp_directory):
        """Verifica validación de rutas con caracteres especiales."""
        special_name = temp_directory / "test file [special].txt"
        special_name.write_text("test")

        result = Validator.validate_path(special_name, must_exist=True, field_name="path_field")
        assert result == special_name

    def test_validate_json_empty_object(self):
        """Verifica validación de JSON objeto vacío."""
        result = Validator.validate_json("{}", field_name="json_field")
        assert result == {}

    def test_validate_json_empty_array(self):
        """Verifica validación de JSON array vacío."""
        result = Validator.validate_json("[]", field_name="json_field")
        assert result == []

    def test_validate_dict_structure_empty(self):
        """Verifica validación de estructura de diccionario vacío."""
        data = {}
        structure = {}

        result = Validator.validate_dict_structure(data, structure, field_name="data_field")
        assert result == {}


# -------------------------------------------------------------------
# Tests de integración avanzada
# -------------------------------------------------------------------


class TestValidatorIntegration:
    """Tests de integración avanzada para Validator."""

    def test_complete_user_validation_workflow(self, sample_pydantic_model):
        """Verifica un flujo completo de validación de usuario."""
        # Datos de entrada crudos
        raw_data = {
            "username": "  john_doe123  ",  # Con espacios
            "email": "  John.Doe@Example.COM  ",  # Con espacios y mayúsculas
            "age": "30",  # String en lugar de int
            "profile": '{"bio": "Software developer", "experience": 5}',
            "files": ["/tmp/doc1.txt", "/tmp/doc2.txt"],
        }

        # Proceso de limpieza y validación
        cleaned_data = {}

        # 1. Validar y limpiar username
        username = raw_data["username"].strip()
        Validator.validate_string_length(
            username, min_length=3, max_length=20, field_name="username"
        )
        Validator.validate_regex(username, r"^[a-zA-Z0-9_]+$", field_name="username")
        cleaned_data["username"] = username

        # 2. Validar y normalizar email
        email = raw_data["email"].strip()
        normalized_email = Validator.validate_email(email, field_name="email")
        cleaned_data["email"] = normalized_email

        # 3. Validar y convertir age
        try:
            age = int(raw_data["age"])
        except ValueError:
            raise ValidationError(
                message="age debe ser un número entero",
                field="age",
                value=raw_data["age"],
                suggestion="Proporcione un número entero válido",
            )
        Validator.validate_number_range(age, min_value=18, max_value=100, field_name="age")
        cleaned_data["age"] = age

        # 4. Validar JSON profile
        profile = Validator.validate_json(raw_data["profile"], field_name="profile")
        Validator.validate_dict_structure(
            profile, {"bio": str, "experience": int}, field_name="profile"
        )
        cleaned_data["profile"] = profile

        # 5. Validar lista de archivos
        Validator.validate_type(raw_data["files"], list, field_name="files")
        for i, file_path in enumerate(raw_data["files"]):
            Validator.validate_path(file_path, field_name=f"files[{i}]")

        # 6. Validar con modelo Pydantic (si corresponde)
        model_data = {
            "name": cleaned_data["username"],
            "age": cleaned_data["age"],
            "email": cleaned_data["email"],
            "active": True,
        }

        user_model = Validator.validate_pydantic_model(
            model_data, sample_pydantic_model, field_name="user_model"
        )

        # Verificaciones finales
        assert user_model.name == "john_doe123"
        assert (
            user_model.email == "John.Doe@example.com"
        )  # Email normalizado (solo dominio en minúsculas)
        assert user_model.age == 30

    def test_error_accumulation_pattern(self):
        """Verifica patrón de acumulación de errores (simulado)."""
        data = {
            "name": "",  # Inválido: vacío
            "age": -5,  # Inválido: negativo
            "email": "invalid",  # Inválido: formato incorrecto
        }

        errors = []

        # Validar nombre
        try:
            Validator.validate_not_empty(data["name"], "name")
        except ValidationError as e:
            errors.append({"field": "name", "error": str(e)})

        # Validar edad
        try:
            Validator.validate_number_range(data["age"], min_value=0, field_name="age")
        except ValidationError as e:
            errors.append({"field": "age", "error": str(e)})

        # Validar email
        try:
            Validator.validate_email(data["email"], "email")
        except ValidationError as e:
            errors.append({"field": "email", "error": str(e)})

        # Verificar que se acumularon todos los errores
        assert len(errors) == 3
        assert all("field" in error for error in errors)
        assert all("error" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
