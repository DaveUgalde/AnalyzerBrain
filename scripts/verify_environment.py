#!/usr/bin/env python3
"""
Verifica que el entorno de desarrollo esté correctamente configurado.

Uso:
    python scripts/verify_environment.py
"""

import sys
import subprocess
from pathlib import Path

def check_python():
    """Verifica la versión de Python."""
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (se requiere 3.9+)")
        return False

def check_venv():
    """Verifica que estemos en un entorno virtual."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Entorno virtual detectado")
        return True
    else:
        print("⚠️  No estás en un entorno virtual (recomendado)")
        return False

def check_requirements():
    """Verifica que las dependencias estén instaladas."""
    try:
        import pkg_resources
        
        requirements_file = Path(__file__).parent.parent / "requirements" / "base.txt"
        
        if not requirements_file.exists():
            print("❌ No se encuentra requirements/base.txt")
            return False
        
        with open(requirements_file, 'r') as f:
            required = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-r'):
                    # Extraer nombre del paquete (sin versión)
                    pkg = line.split('==')[0].split('>=')[0].split('<=')[0]
                    required.append(pkg)
        
        missing = []
        for pkg in required:
            try:
                pkg_resources.get_distribution(pkg)
            except pkg_resources.DistributionNotFound:
                missing.append(pkg)
        
        if not missing:
            print(f"✅ Todas las {len(required)} dependencias base instaladas")
            return True
        else:
            print(f"❌ Faltan dependencias: {', '.join(missing)}")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando dependencias: {e}")
        return False

def check_directories():
    """Verifica la estructura de directorios."""
    base_dir = Path(__file__).parent.parent
    
    required_dirs = [
        "src",
        "tests",
        "config",
        "data",
        "logs",
        "requirements"
    ]
    
    missing = []
    for dir_name in required_dirs:
        if not (base_dir / dir_name).exists():
            missing.append(dir_name)
    
    if not missing:
        print("✅ Estructura de directorios correcta")
        return True
    else:
        print(f"❌ Directorios faltantes: {', '.join(missing)}")
        return False

def check_config_files():
    """Verifica archivos de configuración."""
    base_dir = Path(__file__).parent.parent
    
    config_files = [
        ".env.example",
        "pyproject.toml",
        ".gitignore"
    ]
    
    missing = []
    for file_name in config_files:
        if not (base_dir / file_name).exists():
            missing.append(file_name)
    
    if not missing:
        print("✅ Archivos de configuración presentes")
        return True
    else:
        print(f"⚠️  Archivos de configuración faltantes: {', '.join(missing)}")
        return False

def run_tests():
    """Ejecuta pruebas básicas."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Pruebas unitarias pasan correctamente")
            return True
        else:
            print("❌ Algunas pruebas unitarias fallaron")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ Error ejecutando pruebas: {e}")
        return False

def main():
    """Función principal."""
    print("🔍 Verificando entorno de desarrollo de ANALYZERBRAIN")
    print("=" * 60)
    
    checks = [
        ("Python 3.9+", check_python()),
        ("Entorno virtual", check_venv()),
        ("Dependencias", check_requirements()),
        ("Estructura de directorios", check_directories()),
        ("Archivos de configuración", check_config_files()),
        ("Pruebas unitarias", run_tests()),
    ]
    
    print("=" * 60)
    
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    if passed == total:
        print(f"🎉 ¡Entorno verificado correctamente! ({passed}/{total})")
        return 0
    else:
        print(f"⚠️  Entorno con problemas ({passed}/{total} checks pasados)")
        return 1

if __name__ == "__main__":
    sys.exit(main())