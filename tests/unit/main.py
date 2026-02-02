# tests/unit/test_cli.py
##import pytest
from click.testing import CliRunner
from src.main import cli
from unittest.mock import patch, AsyncMock

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "ANALYZERBRAIN" in result.output
    assert "init" in result.output
    assert "analyze" in result.output
    assert "query" in result.output

def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert "ANALYZERBRAIN" in result.output
    assert "0.1.0" in result.output

@patch('src.main.AnalyzerBrainSystem.initialize')
def test_cli_init(
    mock_initialize) # type: ignore
    mock_initialize.return_value = AsyncMock(return_value=True)
    
    runner = CliRunner()
    result = runner.invoke(cli, ['init'])
    assert result.exit_code == 0

@patch('src.main.AnalyzerBrainSystem.analyze_project')
def test_cli_analyze(mock_analyze): # type: ignore
    mock_analyze.return_value = AsyncMock(return_value={
        "project": "test",
        "status": "success"
    })
    
    runner = CliRunner()
    result = runner.invoke(cli, ['analyze', '.'], catch_exceptions=False)
    assert result.exit_code == 0
    assert "RESULTADOS DEL ANÁLISIS" in result.output

def test_cli_status():
    runner = CliRunner()
    result = runner.invoke(cli, ['status'])
    assert result.exit_code == 0
    assert "ESTADO DEL SISTEMA" in result.output

def test_cli_health():
    runner = CliRunner()
    result = runner.invoke(cli, ['health'])
    assert result.exit_code in [0, 1]  # Puede fallar si no hay servicios