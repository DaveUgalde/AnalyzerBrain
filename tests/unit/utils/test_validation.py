# tests/unit/test_validation.py
import pytest
from src.core.exceptions import ValidationError
from src.utils.validation import validator

def test_validate_not_empty():
    # Caso válido
    assert validator.validate_not_empty("test", "field") == "test"
    
    # Casos inválidos
    with pytest.raises(ValidationError):
        validator.validate_not_empty("", "field")
    
    with pytest.raises(ValidationError):
        validator.validate_not_empty(None, "field")

def test_validate_type():
    assert validator.validate_type("test", str, "field") == "test"
    
    with pytest.raises(ValidationError):
        validator.validate_type("test", int, "field")

def test_validate_string_length():
    # Longitud correcta
    assert validator.validate_string_length("abc", min_length=2, max_length=5) == "abc"
    
    # Demasiado corto
    with pytest.raises(ValidationError):
        validator.validate_string_length("a", min_length=2)
    
    # Demasiado largo
    with pytest.raises(ValidationError):
        validator.validate_string_length("abcdef", max_length=5)

def test_validate_email():
    # Email válido
    assert validator.validate_email("test@example.com") == "test@example.com"
    
    # Email inválido
    with pytest.raises(ValidationError):
        validator.validate_email("not-an-email")

# Más tests para cada método...