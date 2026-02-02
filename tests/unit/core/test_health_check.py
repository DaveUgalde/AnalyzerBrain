# tests/unit/test_health_check.py
import pytest
from unittest.mock import patch
##from unittest.mock import MagicMock

from src.core.health_check import SystemHealthChecker, HealthStatus

def test_health_check_result_to_dict():
    from src.core.health_check import HealthCheckResult
    result = HealthCheckResult(
        name="test_check",
        status=HealthStatus.HEALTHY,
        message="Test message",
        details={"key": "value"}
    )
    
    dict_result = result.to_dict()
    assert dict_result["name"] == "test_check"
    assert dict_result["status"] == "healthy"
    assert dict_result["message"] == "Test message"

@pytest.mark.asyncio
async def test_health_check_all():
    checker = SystemHealthChecker()
    
    with patch('psutil.cpu_percent', return_value=50):
        with patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value.percent = 60
            mock_mem.return_value.total = 1024**3  # 1GB
            mock_mem.return_value.available = 512**3  # 0.5GB
            
            result = await checker.check_all()
            
            assert "overall" in result
            assert "checks" in result
            assert "summary" in result
            assert len(result["checks"]) > 0

def test_health_check_sync():
    checker = SystemHealthChecker()
    
    result = checker.check_all_sync()
    assert "overall" in result
    assert result["checks"] is not None

def test_print_detailed_report():
    checker = SystemHealthChecker()
    
    # Ejecutar checks primero
    checker.check_all_sync()
    
    report = checker.print_detailed_report()
    assert isinstance(report, str)
    assert "REPORTE DETALLADO" in report
    assert "Resumen:" in report