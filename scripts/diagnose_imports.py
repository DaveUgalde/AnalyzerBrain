#!/usr/bin/env python3
"""Diagnóstica problemas de importación."""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al sys.path
project_root = Path(__file__).parent.parent  # Subir dos niveles desde scripts/
sys.path.insert(0, str(project_root))

print("🔍 Diagnosticando importaciones de ANALYZERBRAIN...")
print(f"Python path: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'No establecido')}")

# Crear __init__.py faltante automáticamente
utils_init = project_root / "src/utils/__init__.py"
if not utils_init.exists():
    print(f"\n⚠️  Creando {utils_init.relative_to(project_root)}...")
    utils_init.touch()
    print(f"✅ {utils_init.relative_to(project_root)} creado")

# Verificar estructura de directorios
print(f"\n📁 Verificando estructura de directorios:")
dirs_to_check = [
    project_root,
    project_root / "src",
    project_root / "src/utils",
    project_root / "src/core",
    project_root / "scripts",
]

for d in dirs_to_check:
    exists = d.exists()
    print(f"  {d.relative_to(project_root) if exists else d}: {'✅ Existe' if exists else '❌ No existe'}")

# Verificar archivos __init__.py
print(f"\n📄 Verificando archivos __init__.py:")
init_files = [
    project_root / "src" / "__init__.py",
    project_root / "src/utils" / "__init__.py",
    project_root / "src/core" / "__init__.py",
]

for init_file in init_files:
    exists = init_file.exists()
    print(f"  {init_file.relative_to(project_root) if exists else init_file}: {'✅ Existe' if exists else '❌ FALTANTE'}")

# Verificar paths de módulos específicos
print(f"\n🔍 Verificando módulos específicos:")
paths = [
    project_root / "src/utils/logging_config.py",
    project_root / "src/core/config_manager.py",
    project_root / "src/core/exceptions.py"
]

for p in paths:
    exists = p.exists()
    print(f"  {p.relative_to(project_root) if exists else p}: {'✅ Existe' if exists else '❌ No existe'}")

# Intentar importaciones básicas
print(f"\n🧪 Probando importaciones básicas...")
try:
    import loguru
    print(f"✅ loguru importado: {loguru.__version__}")
except ImportError as e:
    print(f"❌ Error importando loguru: {e}")

try:
    import pydantic
    print(f"✅ pydantic importado: {pydantic.__version__}")
except ImportError as e:
    print(f"❌ Error importando pydantic: {e}")

# Intentar importaciones del proyecto
print(f"\n🚀 Intentando importaciones de ANALYZERBRAIN...")

try:
    import src
    print(f"✅ Paquete 'src' importado correctamente")
    print(f"  Ubicación: {src.__file__}")
except ImportError as e:
    print(f"❌ Error importando 'src': {e}")

# CORREGIDO: Importar funciones correctas de logging_config
try:
    print(f"\nIntentando importar StructuredLogger de logging_config...")
    from src.utils.logging_config import StructuredLogger
    print("✅ StructuredLogger importado correctamente")
    
    # Probar el método setup_logging
    print("  Probando StructuredLogger.setup_logging()...")
    StructuredLogger.setup_logging()
    print("  ✅ StructuredLogger.setup_logging() ejecutado sin errores")
    
except ImportError as e:
    print(f"❌ Error importando StructuredLogger: {e}")
    import traceback
    traceback.print_exc()

# También probar las funciones de nivel superior
try:
    print(f"\nIntentando importar setup_default_logging...")
    from src.utils.logging_config import setup_default_logging
    print("✅ setup_default_logging importado correctamente")
    
    print("  Probando setup_default_logging()...")
    setup_default_logging()
    print("  ✅ setup_default_logging ejecutado sin errores")
    
except ImportError as e:
    print(f"❌ Error importando setup_default_logging: {e}")
    import traceback
    traceback.print_exc()

try:
    print(f"\nIntentando importar init_logging...")
    from src.utils.logging_config import init_logging
    print("✅ init_logging importado correctamente")
    
    print("  Probando init_logging()...")
    init_logging()
    print("  ✅ init_logging ejecutado sin errores")
    
except ImportError as e:
    print(f"❌ Error importando init_logging: {e}")
    import traceback
    traceback.print_exc()

# Intentar importar ConfigManager
try:
    print(f"\nIntentando importar ConfigManager...")
    from src.core.config_manager import ConfigManager, config
    
    print("✅ ConfigManager importado correctamente")
    print(f"  ConfigManager ubicación: {ConfigManager.__module__}")
    
    # Probar la instancia config
    print("  Probando instancia 'config'...")
    print(f"    Entorno: {config.environment}")
    print(f"    Es desarrollo: {config.is_development}")
    
except ImportError as e:
    print(f"❌ Error importando ConfigManager: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*50}")
print("🎯 RESUMEN DE DIAGNÓSTICO")
print("="*50)

# Verificación final
if utils_init.exists():
    print("✅ src/utils/__init__.py creado exitosamente")
else:
    print("❌ src/utils/__init__.py NO creado - ejecuta manualmente: touch src/utils/__init__.py")

print(f"\n📋 Para instalar el proyecto en modo desarrollo:")
print("   pip install -e .")
print(f"\n📋 Para ejecutar ANALYZERBRAIN:")
print("   python -m src.main --help")