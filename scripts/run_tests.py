#!/usr/bin/env python3
"""
Script para ejecutar pruebas de ANALYZERBRAIN.

Uso:
    python scripts/run_tests.py [opciones]

Opciones:
    --unit          Ejecuta solo pruebas unitarias
    --integration   Ejecuta solo pruebas de integración
    --e2e           Ejecuta solo pruebas end-to-end
    --coverage      Genera reporte de cobertura
    --all           Ejecuta todas las pruebas (default)
    --verbose       Modo verboso
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, verbose=False):
    """Ejecuta un comando y muestra la salida."""
    print(f"Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    
    if result.returncode != 0 and not verbose:
        print(f"Error en comando:\n{result.stderr}")
    
    return result.returncode

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ejecuta pruebas de ANALYZERBRAIN")
    parser.add_argument("--unit", action="store_true", help="Ejecuta solo pruebas unitarias")
    parser.add_argument("--integration", action="store_true", help="Ejecuta solo pruebas de integración")
    parser.add_argument("--e2e", action="store_true", help="Ejecuta solo pruebas end-to-end")
    parser.add_argument("--coverage", action="store_true", help="Genera reporte de cobertura")
    parser.add_argument("--all", action="store_true", default=True, help="Ejecuta todas las pruebas (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Modo verboso")
    
    args = parser.parse_args()
    
    # Directorio base
    base_dir = Path(__file__).parent.parent
    
    # Comando base de pytest
    cmd = [sys.executable, "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    # Seleccionar tipo de pruebas
    if args.unit:
        test_path = base_dir / "tests" / "unit"
    elif args.integration:
        test_path = base_dir / "tests" / "integration"
    elif args.e2e:
        test_path = base_dir / "tests" / "e2e"
    else:
        test_path = base_dir / "tests"
    
    cmd.append(str(test_path))
    
    # Agregar cobertura si se solicita
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term",
            "--cov-report=html:coverage_report",
            "--cov-report=xml:coverage.xml"
        ])
    
    # Ejecutar pruebas
    return_code = run_command(cmd, args.verbose)
    
    if return_code == 0:
        print("✅ Todas las pruebas pasaron exitosamente")
    else:
        print("❌ Algunas pruebas fallaron")
    
    sys.exit(return_code)

if __name__ == "__main__":
    main()