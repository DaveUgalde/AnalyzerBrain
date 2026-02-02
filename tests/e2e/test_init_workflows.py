# tests/e2e/test_basic_workflows.py
##import pytest
from click.testing import CliRunner
from src.main import cli
import tempfile
from pathlib import Path

def test_e2e_init_workflow():
    """Flujo completo: init -> status -> health"""
    runner = CliRunner()
    
    # 1. Inicializar
    result = runner.invoke(cli, ['init'])
    assert result.exit_code == 0
    
    # 2. Verificar estado
    result = runner.invoke(cli, ['status'])
    assert result.exit_code == 0
    assert "ESTADO DEL SISTEMA" in result.output
    
    # 3. Health check
    result = runner.invoke(cli, ['health'])
    # Puede fallar si no hay servicios, pero debe ejecutarse
    assert "HEALTH CHECK" in result.output

def test_e2e_project_analysis_workflow():
    """Flujo de análisis básico"""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear proyecto de prueba simple
        project_path = Path(tmpdir)
        (project_path / "test.py").write_text("print('Hello')")
        (project_path / "README.md").write_text("# Test Project")
        
        # Ejecutar análisis
        result = runner.invoke(cli, ['analyze', str(project_path)])
        assert result.exit_code == 0
        assert "RESULTADOS DEL ANÁLISIS" in result.output
        assert "simulado completado" in result.output  # Del placeholder actual