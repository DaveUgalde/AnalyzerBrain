# scripts/init_data_system.py
"""
Script de inicialización del sistema de datos.

Este script:
1. Crea toda la estructura de directorios
2. Inicializa bases de datos
3. Configura permisos
4. Valida que todo esté listo
"""

import sys
from pathlib import Path
import traceback

# Añadir src al path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


def initialize_data_system() -> bool:
    """Inicializa todo el sistema de datos."""
    print("\n=== INICIALIZANDO SISTEMA DE DATOS ===\n")

    # Directorio base de datos
    data_dir = (BASE_DIR / "data").resolve()

    try:
        if not data_dir.exists():
            print(f"📁 Creando directorio de datos: {data_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"📁 Directorio de datos existente: {data_dir}")

        # Importar DataManager aquí para asegurar path correcto
        from data.init_data_structure import DataManager

        # Inicializar DataManager
        print("\n🚀 Inicializando DataManager...")
        data_manager = DataManager(str(data_dir))

        # Obtener estadísticas
        stats = data_manager.get_storage_stats()

        def gb(value: int) -> float:
            return value / (1024 ** 3)

        def mb(value: int) -> float:
            return value / (1024 ** 2)

        print("\n=== 📊 ESTADÍSTICAS DE ALMACENAMIENTO ===")
        print(f"Espacio total usado: {gb(stats.get('total_size', 0)):.2f} GB")
        print(f"Número de proyectos: {stats.get('project_count', 0)}")
        print(f"Número de embeddings: {stats.get('embedding_count', 0)}")
        print(f"Número de grafos: {stats.get('graph_count', 0)}")
        print(f"Tamaño de caché: {mb(stats.get('cache_size', 0)):.2f} MB")
        print(f"Tamaño de backups: {mb(stats.get('backup_size', 0)):.2f} MB")
        print(f"Espacio libre: {gb(stats.get('free_space', 0)):.2f} GB")

        # Validar integridad
        print("\n=== 🔍 VALIDANDO INTEGRIDAD ===")
        integrity = data_manager.validate_data_integrity()

        valid_total = 0
        invalid_total = 0

        for category, results in integrity.items():
            valid = results.get("valid", 0)
            invalid = results.get("invalid", 0)

            print(f"{category}: {valid} válidos, {invalid} inválidos")
            valid_total += valid
            invalid_total += invalid

        print(f"\nTotal: {valid_total} válidos, {invalid_total} inválidos")

        if invalid_total == 0:
            print("✅ Todos los datos son válidos")
        else:
            print(f"⚠️  {invalid_total} elementos inválidos encontrados")

        # Limpiar archivos temporales
        print("\n=== 🧹 LIMPIANDO ARCHIVOS TEMPORALES ===")
        removed = data_manager.cleanup_temp_files()
        print(f"Eliminados {removed} archivos temporales")

        print("\n=== ✅ SISTEMA DE DATOS INICIALIZADO CORRECTAMENTE ===\n")
        return True

    except Exception as e:
        print("\n❌ Error inicializando sistema de datos")
        print(f"Motivo: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = initialize_data_system()
    sys.exit(0 if success else 1)
