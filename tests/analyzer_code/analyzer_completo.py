#!/usr/bin/env python3
"""
DEPENDENCY ANALYZER PRO - v1.0
Analiza todas las dependencias de un proyecto Python y compara con requirements.
Genera reportes detallados de dependencias faltantes y conflictos.
"""

import ast
import re
import sys
import json
import toml
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import subprocess
import pkg_resources
from datetime import datetime
import argparse

# Mapeo de imports a paquetes PyPI (casos especiales)
IMPORT_TO_PACKAGE = {
    'sklearn': 'scikit-learn',
    'PIL': 'Pillow',
    'yaml': 'PyYAML',
    'bs4': 'beautifulsoup4',
    'dateutil': 'python-dateutil',
    'cv2': 'opencv-python',
    'Bio': 'biopython',
    'MySQLdb': 'mysqlclient',
    'psycopg2': 'psycopg2-binary',
    'dotenv': 'python-dotenv',
    'tensorflow.keras': 'tensorflow',
    'torch': 'torch',
    'pydantic_ai': 'pydantic-ai',
    'sqlalchemy': 'SQLAlchemy',
    'aiohttp': 'aiohttp',
    'httpx': 'httpx',
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn[standard]',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'langchain': 'langchain',
    'openai': 'openai',
    'anthropic': 'anthropic',
    'cohere': 'cohere',
    'transformers': 'transformers',
    'sentence_transformers': 'sentence-transformers',
    'chromadb': 'chromadb',
    'crewai': 'crewai',
    'pyautogen': 'pyautogen',
    'instructor': 'instructor',
    'marvin': 'marvin',
    'nltk': 'nltk',
    'gensim': 'gensim',
    'spacy': 'spacy',
    'rapidfuzz': 'rapidfuzz',
    'langdetect': 'langdetect',
    'jellyfish': 'jellyfish',
    'textblob': 'textblob',
    'pdfminer': 'pdfminer.six',
    'docx': 'python-docx',
    'redis': 'redis',
    'pymongo': 'pymongo',
    'elasticsearch': 'elasticsearch',
    'neo4j': 'neo4j-driver',
    'asyncpg': 'asyncpg',
    'psycopg2': 'psycopg2-binary',
    'sqlalchemy': 'SQLAlchemy',
    'alembic': 'alembic',
    'pytest': 'pytest',
    'black': 'black',
    'flake8': 'flake8',
    'mypy': 'mypy',
    'pre_commit': 'pre-commit',
    'jupyter': 'jupyter',
    'jupyterlab': 'jupyterlab',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'plotly': 'plotly',
    'scikit_learn': 'scikit-learn',
    'networkx': 'networkx',
    'scipy': 'scipy',
    'cryptography': 'cryptography',
    'psutil': 'psutil',
    'tqdm': 'tqdm',
    'structlog': 'structlog',
    'prometheus_client': 'prometheus-client',
    'typer': 'typer',
    'rich': 'rich',
    'typing_extensions': 'typing-extensions',
    'pydantic': 'pydantic',
    'pydantic_settings': 'pydantic-settings',
    'anyio': 'anyio',
    'aiofiles': 'aiofiles',
    'jinja2': 'Jinja2',
    'platformdirs': 'platformdirs',
    'filelock': 'filelock',
    'python_multipart': 'python-multipart',
}

# Módulos estándar de Python (no necesitan instalación)
PYTHON_STDLIB_MODULES = {
    'os', 'sys', 're', 'json', 'time', 'datetime', 'math', 'hashlib',
    'collections', 'itertools', 'functools', 'typing', 'pathlib',
    'argparse', 'dataclasses', 'enum', 'abc', 'copy', 'csv', 'decimal',
    'fractions', 'random', 'statistics', 'string', 'textwrap', 'unicodedata',
    'bisect', 'heapq', 'array', 'weakref', 'types', 'pdb', 'pickle',
    'shelve', 'marshal', 'sqlite3', 'hashlib', 'hmac', 'secrets',
    'doctest', 'unittest', 'logging', 'getpass', 'curses', 'platform',
    'errno', 'ctypes', 'threading', 'multiprocessing', 'concurrent',
    'asyncio', 'socket', 'ssl', 'select', 'shutil', 'tempfile', 'glob',
    'fnmatch', 'linecache', 'codecs', 'locale', 'gettext', 'stringprep',
    'readline', 'rlcompleter', 'atexit', 'signal', 'subprocess', 'os.path',
    'io', 'zipfile', 'tarfile', 'lzma', 'bz2', 'zlib', 'gzip', 'importlib',
    'pkgutil', 'modulefinder', 'runpy', 'sysconfig', 'builtins', 'warnings',
    'contextlib', 'abc', 'atexit', 'traceback', '__future__', 'future',
    'functools', 'inspect', 'site', 'code', 'codeop', 'py_compile',
    'compileall', 'imp', 'zipimport', 'pkg_resources', 'setuptools'
}

class DependencyExtractor(ast.NodeVisitor):
    """Extrae imports de archivos Python usando AST."""
    
    def __init__(self):
        self.imports = set()
        self.import_from = defaultdict(set)
        
    def visit_Import(self, node):
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self.imports.add(module_name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module.split('.')[0]
            if node.level == 0:  # No es import relativo
                self.imports.add(module_name)
                # También registrar imports específicos
                for alias in node.names:
                    self.import_from[module_name].add(alias.name)
        self.generic_visit(node)

class ProjectDependencyAnalyzer:
    """Analizador completo de dependencias del proyecto."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.all_imports = set()
        self.import_details = defaultdict(list)
        self.requirements_files = []
        self.parsed_requirements = defaultdict(dict)  # {file: {package: version}}
        
    def discover_files(self) -> List[Path]:
        """Descubre todos los archivos Python del proyecto."""
        exclude_dirs = {
            '.git', '__pycache__', '.pytest_cache', '.mypy_cache',
            '.venv', 'venv', 'env', 'node_modules', 'build', 'dist',
            'htmlcov', '.coverage', '.tox', '.hypothesis'
        }
        
        exclude_patterns = ['*.pyc', '*.pyo', '*.pyd', '*.so']
        
        python_files = []
        for file in self.project_path.rglob('*.py'):
            # Excluir directorios no deseados
            if any(excluded in file.parts for excluded in exclude_dirs):
                continue
            
            # Excluir archivos de test si se quiere (opcional)
            if 'test' in file.name.lower() or 'tests' in file.parts:
                continue  # O comentar esta línea para incluir tests
            
            try:
                if file.stat().st_size > 10 * 1024 * 1024:  # 10MB
                    continue
            except OSError:
                continue
            
            python_files.append(file)
        
        return python_files
    
    def discover_requirements_files(self) -> List[Path]:
        """Encuentra todos los archivos de requirements."""
        req_patterns = ['requirements*.txt', 'requirements/*.txt', 'setup.py', 
                       'setup.cfg', 'pyproject.toml', 'Pipfile', 'poetry.lock']
        
        req_files = []
        
        # Buscar archivos de requirements
        for pattern in ['requirements*.txt', 'requirements/*.txt']:
            for file in self.project_path.rglob(pattern):
                req_files.append(file)
        
        # Buscar archivos de configuración
        config_files = ['pyproject.toml', 'setup.py', 'setup.cfg', 
                       'Pipfile', 'poetry.lock']
        
        for config_file in config_files:
            file_path = self.project_path / config_file
            if file_path.exists():
                req_files.append(file_path)
        
        return req_files
    
    def extract_imports_from_file(self, file_path: Path) -> Set[str]:
        """Extrae todos los imports de un archivo Python."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            extractor = DependencyExtractor()
            extractor.visit(tree)
            
            # También buscar imports con regex (por si falla AST)
            regex_imports = self._find_imports_with_regex(content)
            
            all_imports = extractor.imports.union(regex_imports)
            
            # Guardar detalles para reporte
            for imp in all_imports:
                self.import_details[imp].append(str(file_path.relative_to(self.project_path)))
            
            return all_imports
            
        except SyntaxError as e:
            print(f"⚠️  Error de sintaxis en {file_path.name}: {e}")
            return set()
        except Exception as e:
            print(f"⚠️  Error leyendo {file_path.name}: {e}")
            return set()
    
    def _find_imports_with_regex(self, content: str) -> Set[str]:
        """Encuentra imports usando regex (backup)."""
        imports = set()
        
        # Patrones para import X y from X import Y
        import_pattern = r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        from_pattern = r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import'
        
        for line in content.splitlines():
            # Buscar import simple
            import_match = re.match(import_pattern, line)
            if import_match:
                module = import_match.group(1).split('.')[0]
                imports.add(module)
            
            # Buscar from ... import
            from_match = re.match(from_pattern, line)
            if from_match:
                module = from_match.group(1).split('.')[0]
                imports.add(module)
        
        return imports
    
    def parse_requirements_file(self, file_path: Path) -> Dict[str, str]:
        """Parsea un archivo de requirements y extrae paquetes."""
        packages = {}
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            if file_path.name == 'pyproject.toml':
                # Parsear toml
                data = toml.loads(content)
                if 'tool' in data and 'poetry' in data['tool']:
                    if 'dependencies' in data['tool']['poetry']:
                        for pkg, version in data['tool']['poetry']['dependencies'].items():
                            if pkg != 'python':
                                packages[pkg] = str(version)
                elif 'project' in data and 'dependencies' in data['project']:
                    for dep in data['project']['dependencies']:
                        # Parsear dependency string
                        pkg = re.match(r'([a-zA-Z0-9_-]+)', dep).group(1)
                        packages[pkg] = dep
                        
            elif file_path.name in ['setup.py', 'setup.cfg']:
                # Parsear setup.py o setup.cfg (simplificado)
                # Usar regex para encontrar install_requires
                pattern = r'install_requires\s*=\s*\[(.*?)\]'
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    deps_text = match.group(1)
                    deps = re.findall(r"['\"]([^'\"]+)['\"]", deps_text)
                    for dep in deps:
                        pkg = re.match(r'([a-zA-Z0-9_-]+)', dep).group(1)
                        packages[pkg] = dep
            
            elif file_path.suffix == '.txt':
                # Parsear requirements.txt
                for line in content.splitlines():
                    line = line.strip()
                    
                    # Ignorar comentarios y líneas vacías
                    if not line or line.startswith('#') or line.startswith('-'):
                        continue
                    
                    # Extraer nombre del paquete (antes del primer operador)
                    # Formato: package==1.0.0, package>=1.0.0, etc.
                    match = re.match(r'([a-zA-Z0-9._-]+)(?:\[.*\])?', line)
                    if match:
                        pkg_name = match.group(1)
                        packages[pkg_name] = line
            
        except Exception as e:
            print(f"⚠️  Error parseando {file_path}: {e}")
        
        return packages
    
    def map_import_to_package(self, import_name: str) -> str:
        """Mapea un nombre de import al nombre del paquete en PyPI."""
        # Casos especiales
        if import_name in IMPORT_TO_PACKAGE:
            return IMPORT_TO_PACKAGE[import_name]
        
        # Convertir guiones bajos a guiones
        if '_' in import_name:
            return import_name.replace('_', '-')
        
        return import_name
    
    def analyze(self) -> Dict[str, Any]:
        """Ejecuta el análisis completo."""
        print(f"🔍 Analizando proyecto: {self.project_path}")
        print("=" * 60)
        
        # 1. Descubrir archivos
        python_files = self.discover_files()
        print(f"📁 Encontrados {len(python_files)} archivos Python")
        
        if not python_files:
            return {"error": "No se encontraron archivos Python"}
        
        # 2. Extraer imports de todos los archivos
        print("\n📦 Extrayendo imports...")
        for i, file_path in enumerate(python_files, 1):
            imports = self.extract_imports_from_file(file_path)
            self.all_imports.update(imports)
            
            if i % 10 == 0:
                print(f"  📄 Procesados {i}/{len(python_files)} archivos")
        
        # 3. Descubrir y parsear requirements
        self.requirements_files = self.discover_requirements_files()
        print(f"\n📋 Encontrados {len(self.requirements_files)} archivos de dependencias:")
        
        for req_file in self.requirements_files:
            rel_path = req_file.relative_to(self.project_path)
            packages = self.parse_requirements_file(req_file)
            self.parsed_requirements[str(rel_path)] = packages
            print(f"  • {rel_path}: {len(packages)} paquetes")
        
        # 4. Filtrar imports que no son stdlib
        external_imports = self.all_imports - PYTHON_STDLIB_MODULES
        
        # 5. Mapear imports a nombres de paquetes
        needed_packages = set()
        import_to_package_map = {}
        
        for imp in external_imports:
            pkg = self.map_import_to_package(imp)
            needed_packages.add(pkg)
            import_to_package_map[imp] = pkg
        
        # 6. Combinar todos los requirements
        all_required_packages = set()
        all_requirements = {}
        
        for file_path, packages in self.parsed_requirements.items():
            for pkg, version in packages.items():
                all_required_packages.add(pkg)
                all_requirements[pkg] = {
                    'version': version,
                    'source': file_path
                }
        
        # 7. Comparar y encontrar diferencias
        missing_packages = needed_packages - all_required_packages
        extra_packages = all_required_packages - needed_packages
        
        # 8. Generar reporte
        report = {
            'project': str(self.project_path),
            'timestamp': datetime.now().isoformat(),
            'files_analyzed': len(python_files),
            'total_imports_found': len(self.all_imports),
            'external_imports': sorted(external_imports),
            'needed_packages': sorted(needed_packages),
            'requirements_files': list(self.parsed_requirements.keys()),
            'total_requirements': len(all_required_packages),
            'missing_packages': sorted(missing_packages),
            'extra_packages': sorted(extra_packages),
            'import_details': {
                imp: files[:5]  # Solo primeros 5 archivos por brevedad
                for imp, files in self.import_details.items()
                if imp not in PYTHON_STDLIB_MODULES
            },
            'import_to_package_map': import_to_package_map,
            'all_requirements': all_requirements
        }
        
        return report
    
    def generate_requirements_report(self, report: Dict[str, Any]) -> str:
        """Genera un reporte legible de dependencias."""
        lines = []
        
        lines.append("=" * 80)
        lines.append("📊 REPORTE COMPLETO DE DEPENDENCIAS")
        lines.append("=" * 80)
        lines.append(f"Proyecto: {report['project']}")
        lines.append(f"Fecha: {datetime.fromisoformat(report['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Estadísticas
        lines.append("📈 ESTADÍSTICAS")
        lines.append("-" * 40)
        lines.append(f"• Archivos Python analizados: {report['files_analyzed']}")
        lines.append(f"• Imports encontrados: {report['total_imports_found']}")
        lines.append(f"• Imports externos: {len(report['external_imports'])}")
        lines.append(f"• Paquetes necesarios: {len(report['needed_packages'])}")
        lines.append(f"• Requirements encontrados: {len(report['all_requirements'])}")
        lines.append("")
        
        # Imports externos
        lines.append("📦 IMPORTS EXTERNOS ENCONTRADOS")
        lines.append("-" * 40)
        for imp in sorted(report['external_imports']):
            pkg = report['import_to_package_map'].get(imp, imp)
            lines.append(f"• {imp} → {pkg}")
        lines.append("")
        
        # Paquetes faltantes
        if report['missing_packages']:
            lines.append("❌ PAQUETES FALTANTES (Agregar a requirements)")
            lines.append("-" * 40)
            for pkg in sorted(report['missing_packages']):
                lines.append(f"• {pkg}")
            lines.append("")
        else:
            lines.append("✅ Todos los paquetes necesarios están en requirements")
            lines.append("")
        
        # Paquetes extras (posiblemente innecesarios)
        if report['extra_packages']:
            lines.append("⚠️  PAQUETES EXTRA (Revisar si son necesarios)")
            lines.append("-" * 40)
            for pkg in sorted(report['extra_packages']):
                source = report['all_requirements'].get(pkg, {}).get('source', 'desconocido')
                lines.append(f"• {pkg} (de {source})")
            lines.append("")
        
        # Archivos de requirements
        lines.append("📋 ARCHIVOS DE REQUIREMENTS")
        lines.append("-" * 40)
        for req_file in sorted(report['requirements_files']):
            lines.append(f"• {req_file}")
        lines.append("")
        
        # Recomendaciones para agregar
        if report['missing_packages']:
            lines.append("🚀 COMANDOS PARA INSTALAR PAQUETES FALTANTES")
            lines.append("-" * 40)
            lines.append("# Agregar estos paquetes a requirements/core.txt o agents.txt:")
            lines.append("")
            for pkg in sorted(report['missing_packages']):
                lines.append(f"{pkg}>=1.0.0  # Versión por determinar")
            lines.append("")
            
            lines.append("# O instalar directamente:")
            lines.append(f"pip install {' '.join(sorted(report['missing_packages']))}")
            lines.append("")
        
        # Detalles por import
        lines.append("🔍 DETALLES POR IMPORT (primeros 5 archivos)")
        lines.append("-" * 40)
        for imp, files in sorted(report['import_details'].items()):
            lines.append(f"\n{imp}:")
            for file in files[:5]:
                lines.append(f"  • {file}")
            if len(files) > 5:
                lines.append(f"  ... y {len(files) - 5} más")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def generate_requirements_files(self, report: Dict[str, Any], output_dir: Path):
        """Genera archivos de requirements actualizados."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Archivo completo con todas las dependencias
        all_deps = sorted(report['needed_packages'])
        
        requirements_all = output_dir / "requirements_all.txt"
        with open(requirements_all, 'w', encoding='utf-8') as f:
            f.write("# DEPENDENCIAS COMPLETAS DEL PROYECTO\n")
            f.write("# Generado automáticamente por Dependency Analyzer\n")
            f.write(f"# Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for pkg in all_deps:
                # Buscar versión si ya existe en algún requirement
                existing_version = report['all_requirements'].get(pkg, {}).get('version', '>=1.0.0')
                f.write(f"{pkg}{existing_version}\n")
        
        # 2. Archivo con solo dependencias faltantes
        missing_deps = sorted(report['missing_packages'])
        
        if missing_deps:
            requirements_missing = output_dir / "requirements_missing.txt"
            with open(requirements_missing, 'w', encoding='utf-8') as f:
                f.write("# DEPENDENCIAS FALTANTES\n")
                f.write("# Agregar estas dependencias a los requirements existentes\n\n")
                
                for pkg in missing_deps:
                    f.write(f"{pkg}>=1.0.0  # TODO: Especificar versión exacta\n")
        
        # 3. Archivo de reporte JSON
        report_json = output_dir / "dependencies_report.json"
        with open(report_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return {
            'all_deps': str(requirements_all),
            'missing_deps': str(requirements_missing) if missing_deps else None,
            'json_report': str(report_json)
        }

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Analizador de Dependencias - Detecta imports y compara con requirements'
    )
    
    parser.add_argument(
        '--path',
        default='.',
        help='Ruta del proyecto a analizar (default: directorio actual)'
    )
    
    parser.add_argument(
        '--output',
        default='./dependency_reports',
        help='Directorio de salida para reportes (default: ./dependency_reports)'
    )
    
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generar archivos de requirements actualizados'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'all'],
        default='text',
        help='Formato de salida (default: text)'
    )
    
    args = parser.parse_args()
    
    project_path = Path(args.path).resolve()
    output_dir = Path(args.output).resolve()
    
    if not project_path.exists():
        print(f"❌ Error: La ruta {project_path} no existe")
        return 1
    
    print(f"🚀 Iniciando análisis de dependencias...")
    print(f"📂 Proyecto: {project_path}")
    print(f"📁 Salida: {output_dir}")
    print("=" * 60)
    
    try:
        # Crear analizador
        analyzer = ProjectDependencyAnalyzer(project_path)
        
        # Ejecutar análisis
        report = analyzer.analyze()
        
        if 'error' in report:
            print(f"❌ Error: {report['error']}")
            return 1
        
        # Mostrar resumen inmediato
        print("\n" + "=" * 60)
        print("📊 RESUMEN RÁPIDO")
        print("-" * 40)
        print(f"Paquetes necesarios: {len(report['needed_packages'])}")
        print(f"Paquetes faltantes: {len(report['missing_packages'])}")
        print(f"Paquetes extras: {len(report['extra_packages'])}")
        
        if report['missing_packages']:
            print("\n❌ PAQUETES FALTANTES:")
            for pkg in sorted(report['missing_packages'])[:10]:  # Mostrar primeros 10
                print(f"  • {pkg}")
            if len(report['missing_packages']) > 10:
                print(f"  ... y {len(report['missing_packages']) - 10} más")
        
        # Generar reportes según formato
        if args.format in ['text', 'all']:
            text_report = analyzer.generate_requirements_report(report)
            
            # Guardar reporte de texto
            report_file = output_dir / "dependencies_report.txt"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(text_report, encoding='utf-8')
            
            # Mostrar en consola
            print("\n" + "=" * 60)
            print(text_report)
        
        if args.format in ['json', 'all']:
            # Guardar reporte JSON
            json_file = output_dir / "dependencies_report.json"
            json_file.parent.mkdir(parents=True, exist_ok=True)
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Reporte JSON guardado en: {json_file}")
        
        # Generar archivos de requirements si se solicita
        if args.generate:
            generated_files = analyzer.generate_requirements_files(report, output_dir)
            print(f"\n📄 Archivos generados:")
            print(f"  • {generated_files['all_deps']}")
            if generated_files['missing_deps']:
                print(f"  • {generated_files['missing_deps']}")
            print(f"  • {generated_files['json_report']}")
        
        print(f"\n✅ Análisis completado exitosamente!")
        
        # Sugerencia final
        if report['missing_packages']:
            print(f"\n💡 SUGERENCIA: Agrega estos paquetes a tus requirements:")
            for pkg in sorted(report['missing_packages']):
                print(f"  - {pkg}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Análisis interrumpido por el usuario")
        return 130
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())