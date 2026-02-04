"""
Configuration tests for SystemState.
Tests configuration loading and defaults.
"""

import pytest
from unittest.mock import patch, MagicMock
from core.system_state import SystemState


def test_config_defaults():
    """Test that SystemState uses config defaults."""
    mock_event_bus = MagicMock()
    mock_event_bus.subscribe = MagicMock()
    mock_event_bus.publish = MagicMock()

    # Mock config with test values
    with patch('core.system_state.config') as mock_config:
        mock_config.get.side_effect = lambda key, default=None: {
            "system_state.max_history_size": 500,
            "system_state.health_check_interval": 60,
            "system_state.watchdog_timeout": 600,
        }.get(key, default)

        state = SystemState(mock_event_bus)

        assert state.max_history_size == 500
        assert state.health_check_interval == 60
        assert state.watchdog_timeout == 600

    def test_config_fallbacks():
        """Test that SystemState falls back to defaults when config missing."""
        mock_event_bus = MagicMock()
        mock_event_bus.subscribe = MagicMock()
        mock_event_bus.publish = MagicMock()

        # Mock config que retorna None pero respeta valores por defecto
        with patch('core.system_state.config') as mock_config:
            # side_effect que retorna el valor por defecto si la clave no está
            def mock_get(key, default=None):
                # Para estas claves específicas, retorna None (como si no existieran)
                if key in [
                    "system_state.max_history_size",
                    "system_state.health_check_interval",
                    "system_state.watchdog_timeout",
                ]:
                    return None  # Simula que la clave no existe
                return default  # Para otras claves, usa el valor por defecto

            mock_config.get.side_effect = mock_get

            state = SystemState(mock_event_bus)

            # Debería usar los valores por defecto
            assert state.max_history_size == 1000  # From class default
            assert state.health_check_interval == 30  # From class default
            assert state.watchdog_timeout == 300  # From class default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
