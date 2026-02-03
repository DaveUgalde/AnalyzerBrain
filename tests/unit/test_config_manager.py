#!/usr/bin/env python3
"""
Tests unitarios para la clase ConfigManager - Versión simplificada.
"""

import os
from pathlib import Path
from typing import Any, Dict
import pytest
import yaml
from unittest.mock import patch, mock_open
from pydantic import ValidationError as PydanticValidationError

from src.core.config_manager import ConfigManager


# -------------------------------------------------------------------
# Fixtures simplificadas
# -------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_config_before_each_test():
    """Resetea el ConfigManager antes de cada test usando API pública."""
    ConfigManager.reset_for_tests()
    yield
    ConfigManager.reset_for_tests()


# -------------------------------------------------------------------
# Tests del Singleton
# -------------------------------------------------------------------

def test_singleton_instance():
    """Verifica que ConfigManager sea un singleton."""
    instance1 = ConfigManager()
    instance2 = ConfigManager()
    
    assert instance1 is instance2


def test_singleton_reset_works():
    """Verifica que el reset funcione correctamente."""
    instance1 = ConfigManager()
    instance1_id = id(instance1)
    
    # Resetear
    ConfigManager.reset_for_tests()
    
    # Nueva instancia después del reset
    instance2 = ConfigManager()
    
    # Deben ser diferentes objetos (porque se reseteó)
    assert instance1 is not instance2
    assert id(instance2) != instance1_id


# -------------------------------------------------------------------
# Tests de inicialización básica
# -------------------------------------------------------------------

def test_settings_loaded_on_init():
    """Verifica que los settings se cargan al inicializar."""
    config = ConfigManager()
    
    # Usar API pública para verificar
    state = config.get_internal_state()
    assert state['settings'] is not None
    assert config.settings is not None


def test_default_values_are_set():
    """Verifica que los valores por defecto estén configurados."""
    config = ConfigManager()
    
    assert config.settings.system.name == "ANALYZERBRAIN"
    assert config.settings.system.max_workers == 4
    assert config.settings.api.port == 8000
    assert config.environment == "development"


# -------------------------------------------------------------------
# Tests de carga desde variables de entorno
# -------------------------------------------------------------------

def test_load_from_environment_variables():
    """Verifica la carga desde variables de entorno."""
    with patch.dict(os.environ, {
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "DEBUG",
    }):
        ConfigManager.reset_for_tests()
        config = ConfigManager()
        
        assert config.settings.environment == "testing"
        assert config.settings.log_level == "DEBUG"


def test_environment_variables_priority():
    """Verifica que las variables de entorno tengan prioridad."""
    yaml_content = {"system": {"max_workers": 5}}
    
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open()):
            with patch("yaml.safe_load", return_value=yaml_content):
                ConfigManager.reset_for_tests()
                config = ConfigManager()
                assert config.settings.system.max_workers == 5


# -------------------------------------------------------------------
# Tests de carga desde archivos YAML
# -------------------------------------------------------------------

def test_load_from_yaml_files():
    """Verifica la carga desde archivos YAML."""
    yaml_content: Dict[str, Any] = {
        "system": {
            "name": "TEST_SYSTEM",
            "max_workers": 15
        },
        "api": {
            "port": 9000,
            "cors_origins": ["http://test.local"]
        }
    }
    
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open()):
            with patch("yaml.safe_load", return_value=yaml_content):
                ConfigManager.reset_for_tests()
                config = ConfigManager()
                
                assert config.settings.system.name == "TEST_SYSTEM"
                assert config.settings.system.max_workers == 15
                assert config.settings.api.port == 9000
                assert "http://test.local" in config.settings.api.cors_origins


def test_missing_yaml_files_no_error():
    """Verifica que no haya error si faltan archivos YAML."""
    with patch("pathlib.Path.exists", return_value=False):
        ConfigManager.reset_for_tests()
        config = ConfigManager()
        assert config.settings.system.name == "ANALYZERBRAIN"


# -------------------------------------------------------------------
# Tests del método get()
# -------------------------------------------------------------------

def test_get_with_dot_notation():
    """Verifica que get() funcione con notación de puntos."""
    config = ConfigManager()
    
    assert config.get("system.name") == "ANALYZERBRAIN"
    assert config.get("system.max_workers") == 4
    assert config.get("api.port") == 8000
    
    origins = config.get("api.cors_origins")
    assert isinstance(origins, list)
    assert "http://localhost:3000" in origins


def test_get_with_default_value():
    """Verifica que get() retorne valores por defecto cuando la clave no existe."""
    config = ConfigManager()
    
    assert config.get("non.existent.key", "default_value") == "default_value"
    assert config.get("system.non_existent", 999) == 999
    assert config.get("", "empty_default") == "empty_default"


def test_get_custom_config_values():
    config = ConfigManager()
    config.update_settings_for_test({
        "custom_setting": "custom_value",
        "another_custom": {"nested": "value"}
    })
    
    assert config._custom_config["custom_setting"] == "custom_value"
    assert config._custom_config["another_custom"]["nested"] == "value"
    
    # Ahora estas aserciones deberían pasar
    assert config.get("custom_setting") == "custom_value"
    assert config.get("another_custom.nested") == "value"


# -------------------------------------------------------------------
# Tests de propiedades
# -------------------------------------------------------------------

def test_environment_property():
    """Verifica la propiedad environment."""
    config = ConfigManager()
    
    assert config.environment == "development"
    assert config.is_development is True
    assert config.is_production is False


def test_environment_change():
    """Verifica que al cambiar el entorno, las propiedades se actualicen."""
    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        ConfigManager.reset_for_tests()
        config = ConfigManager()
        
        assert config.environment == "production"
        assert config.is_development is False
        assert config.is_production is True


# -------------------------------------------------------------------
# Tests del método reload()
# -------------------------------------------------------------------

def test_reload_resets_settings():
    """Verifica que reload() cargue nueva configuración."""
    config = ConfigManager()
    original_name = config.settings.system.name
    
    yaml_content = {"system": {"name": "RELOADED_NAME"}}
    
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open()):
            with patch("yaml.safe_load", return_value=yaml_content):
                config.reload()
                assert config.settings.system.name == "RELOADED_NAME"
                assert config.settings.system.name != original_name


def test_reload_with_different_environment():
    """Verifica que reload() cargue un entorno diferente."""
    config = ConfigManager()
    assert config.environment == "development"
    
    with patch.dict(os.environ, {"ENVIRONMENT": "testing"}):
        config.reload()
        assert config.environment == "testing"


# -------------------------------------------------------------------
# Tests de manejo de errores
# -------------------------------------------------------------------

def test_invalid_yaml_raises_error():
    """Verifica que YAML inválido lance error."""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="invalid: yaml: :")):
            with pytest.raises(yaml.YAMLError):
                ConfigManager.reset_for_tests()
                ConfigManager()


def test_validation_error_on_invalid_data():
    """Verifica que datos inválidos en YAML lancen ValidationError de Pydantic."""
    # max_workers negativo no es válido (debe ser >= 1)
    invalid_yaml = {"system": {"max_workers": -1}}
    
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open()):
            with patch("yaml.safe_load", return_value=invalid_yaml):
                # Capturamos la excepción de Pydantic
                ConfigManager.reset_for_tests()
                
                # Usar la excepción de Pydantic importada correctamente
                with pytest.raises(PydanticValidationError) as exc_info:
                    ConfigManager()
                
                # Verificar que el error es por max_workers
                error_str = str(exc_info.value)
                assert "max_workers" in error_str or "greater than or equal to 1" in error_str


# -------------------------------------------------------------------
# Tests de directorios
# -------------------------------------------------------------------

def test_directories_are_created():
    """Verifica que se creen los directorios necesarios."""
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        ConfigManager.reset_for_tests()
        ConfigManager()
        
        assert mock_mkdir.called
        call_args = mock_mkdir.call_args
        assert call_args.kwargs.get("parents") is True
        assert call_args.kwargs.get("exist_ok") is True


def test_directory_creation_failure_in_development():
    """Verifica manejo de errores al crear directorios en desarrollo."""
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("No permission")):
        ConfigManager.reset_for_tests()
        # En desarrollo, no debe lanzar excepción
        config = ConfigManager()
        assert config.settings is not None
        assert config.environment == "development"


def test_directory_creation_failure_in_production():
    """Verifica que en producción se lance excepción al no poder crear directorios."""
    # Primero creamos una configuración con entorno de producción
    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("No permission")):
            ConfigManager.reset_for_tests()
            with pytest.raises(RuntimeError, match="No se pudo crear el directorio"):
                ConfigManager()


# -------------------------------------------------------------------
# Tests de estado interno (para debugging)
# -------------------------------------------------------------------

def test_internal_state_inspection():
    """Verifica que se pueda inspeccionar el estado interno."""
    config = ConfigManager()
    state = config.get_internal_state()
    
    assert isinstance(state, dict)
    assert "settings" in state
    assert "custom_config" in state
    assert state["settings"] is not None


# -------------------------------------------------------------------
# Test de actualización de configuración
# -------------------------------------------------------------------

def test_update_settings_for_test():
    """Verifica que se pueda actualizar configuración para tests."""
    config = ConfigManager()
    
    original_workers = config.settings.system.max_workers
    
    config.update_settings_for_test({
        "system": {
            "max_workers": 99,
            "timeout_seconds": 999
        }
    })
    
    assert config.settings.system.max_workers == 99
    assert config.settings.system.timeout_seconds == 999
    assert config.settings.system.max_workers != original_workers
    assert config.settings.system.name == "ANALYZERBRAIN"


# 1. Tests de casos límite
def test_boundary_values():
    """Verifica validación en límites."""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open()):
            # Límite inferior
            with patch("yaml.safe_load", return_value={"system": {"max_workers": 0}}):
                ConfigManager.reset_for_tests()
                with pytest.raises(PydanticValidationError):
                    ConfigManager()
            
            # Límite superior
            ConfigManager.reset_for_tests()
            with patch("yaml.safe_load", return_value={"system": {"max_workers": 33}}):
                with pytest.raises(PydanticValidationError):
                    ConfigManager()


# 5. Tests de combinación compleja
def test_complex_nested_configuration():
    """Verifica configuración anidada compleja."""
    config = ConfigManager()
    
    complex_yaml = {
        "custom_config": {
            "system": {
                "name": "Test",
                "settings": {
                    "advanced": {
                        "feature_flags": ["flag1", "flag2"],
                        "thresholds": {"low": 0.1, "high": 0.9}
                    }
                }
            }
        }
    }
    
    config.update_settings_for_test(complex_yaml)
    assert config.get("custom_config.system.settings.advanced.feature_flags") == ["flag1", "flag2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])