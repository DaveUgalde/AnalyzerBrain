#!/usr/bin/env python3
"""
Script de ejecución simplificado para el analizador
Versión: 2.0 (Completamente integrado con YAML)
"""
import sys
import yaml
from pathlib import Path

def cargar_configuracion(ruta_config=None):
    """Carga configuración desde YAML o usa valores por defecto"""
    config_por_defecto = {
        'extensiones_analizar': ['.py', '.js', '.ts', '.java', '.cpp', '.cs'],
        'limites_calidad': {
            'lineas_por_archivo': 300,
            'complejidad_ciclomatica_max': 10,
            'lineas_por_funcion': 50,
            'metodos_por_clase': 10,
            'parametros_por_funcion': 7,
        },
        'patrones_nombres': {
            'getters': ['get_', 'fetch_', 'retrieve_'],
            'setters': ['set_', 'update_', 'modify_'],
            'validadores': ['validate_', 'check_', 'verify_'],
            'booleanos': ['is_', 'has_', 'can_', 'should_'],
        },
        'excluir_directorios': [
            '__pycache__', '.git', 'node_modules', 'venv',
            '.venv', '.pytest_cache', '.vscode', 'dist', 'build'
        ],
        'prioridad_pruebas': {
            'alta': ['validate', 'auth', 'security', 'payment', 'critical'],
            'media': ['process', 'calculate', 'transform', 'api'],
            'baja': ['helper', 'util', 'format'],
        }
    }
    
    if ruta_config and Path(ruta_config).exists():
        try:
            with open(ruta_config, 'r', encoding='utf-8') as f:
                config_yaml = yaml.safe_load(f)
                # Fusionar configuración (YAML sobreescribe defaults)
                if config_yaml and 'configuracion' in config_yaml:
                    config_por_defecto.update(config_yaml['configuracion'])
                print(f"✅ Configuración cargada desde: {ruta_config}")
                return config_por_defecto
        except Exception as e:
            print(f"⚠️  Error cargando configuración: {e}")
            print("   Usando valores por defecto")
    
    return config_por_defecto

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizador de Módulos')
    parser.add_argument('ruta_modulo', help='Ruta al módulo a analizar')
    parser.add_argument('-o', '--output', default='analisis_resultados',
                       help='Directorio de salida para reportes')
    parser.add_argument('-c', '--config', default='config_analyzer.yaml',
                       help='Ruta al archivo de configuración YAML')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Modo verbose con más detalles')
    
    args = parser.parse_args()
    
    print(f"🔍 Analizando módulo: {args.ruta_modulo}")
    print(f"📁 Salida: {args.output}")
    print(f"⚙️  Configuración: {args.config}")
    
    # VERIFICAR SI module_analyzer.py EXISTE
    try:
        from tests.analyzer_code.module_analyzer import AnalizadorModulo
    except ImportError:
        print("❌ ERROR: No se encuentra 'module_analyzer.py'")
        print("   Asegúrate de tener el archivo en el mismo directorio")
        return 1
    
    # Cargar configuración
    config = cargar_configuracion(args.config)
    
    # Crear analizador con configuración
    analizador = AnalizadorModulo(
        ruta_raiz=args.ruta_modulo,
        output_dir=args.output,
        config=config,
        verbose=args.verbose
    )
    
    analizador.ejecutar_analisis_completo()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())