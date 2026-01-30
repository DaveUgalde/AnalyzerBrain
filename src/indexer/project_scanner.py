"""
Módulo ProjectScanner - Escaneo y descubrimiento de proyectos
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import fnmatch
import json

@dataclass
class ProjectStructure:
    """Estructura de proyecto detectada"""
    root_path: Path
    files: List[Path] = field(default_factory=list)
    directories: List[Path] = field(default_factory=list)
    file_types: Dict[str, int] = field(default_factory=dict)
    total_size: int = 0
    scan_timestamp: datetime = field(default_factory=datetime.now)

class ProjectScanner:
    """Escáner de proyectos para descubrir y analizar estructura de código"""
    
    # Patrones de exclusión por defecto
    DEFAULT_EXCLUDE_PATTERNS = [
        '*.pyc', '*.pyo', '*.pyd', '__pycache__',
        '*.class', '*.jar', '*.war', '*.ear',
        'node_modules', '.git', '.svn', '.hg',
        '.DS_Store', 'Thumbs.db', '.idea', '.vscode',
        '*.log', '*.tmp', '*.temp', 'dist', 'build',
        '.env', '.venv', 'venv', 'env', 'virtualenv',
        '*.egg-info', '.pytest_cache', '.coverage'
    ]
    
    # Extensiones de archivo de código soportadas
    CODE_EXTENSIONS = {
        '.py', '.java', '.js', '.jsx', '.ts', '.tsx',
        '.cpp', '.c', '.h', '.hpp', '.cs', '.go',
        '.rb', '.php', '.swift', '.kt', '.rs',
        '.html', '.css', '.scss', '.less', '.xml',
        '.json', '.yaml', '.yml', '.toml', '.md',
        '.sql', '.sh', '.bat', '.ps1', '.dockerfile'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar el escáner con configuración"""
        self.config = config or {}
        self.exclude_patterns = self.DEFAULT_EXCLUDE_PATTERNS.copy()
        self.exclude_patterns.extend(self.config.get('exclude_patterns', []))
        
    def scan_project(self, project_path: str) -> ProjectStructure:
        """
        Escanear proyecto completo y retornar estructura
        
        Args:
            project_path: Ruta al directorio raíz del proyecto
            
        Returns:
            ProjectStructure: Estructura detectada del proyecto
        """
        root_path = Path(project_path).resolve()
        
        if not root_path.exists():
            raise FileNotFoundError(f"La ruta del proyecto no existe: {project_path}")
            
        if not root_path.is_dir():
            raise ValueError(f"La ruta del proyecto debe ser un directorio: {project_path}")
        
        structure = ProjectStructure(root_path=root_path)
        total_size = 0
        
        for root, dirs, files in os.walk(root_path):
            # Filtrar directorios excluidos
            dirs[:] = [d for d in dirs if not self._should_exclude(Path(root) / d)]
            
            current_dir = Path(root)
            
            # Agregar directorio a la estructura
            if current_dir != root_path:  # No incluir el directorio raíz
                structure.directories.append(current_dir.relative_to(root_path))
            
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(root_path)
                
                # Verificar si el archivo debe ser excluido
                if self._should_exclude(file_path):
                    continue
                    
                # Solo procesar archivos de código
                if file_path.suffix.lower() in self.CODE_EXTENSIONS:
                    structure.files.append(rel_path)
                    
                    # Actualizar contadores de tipos de archivo
                    ext = file_path.suffix.lower()
                    structure.file_types[ext] = structure.file_types.get(ext, 0) + 1
                    
                    # Calcular tamaño
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, IOError):
                        pass
        
        structure.total_size = total_size
        return structure
    
    def _should_exclude(self, path: Path) -> bool:
        """Determinar si una ruta debe ser excluida del escaneo"""
        path_str = str(path)
        
        # Verificar patrones de exclusión
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path.name, pattern) or pattern in path_str:
                return True
        
        # Excluir archivos ocultos (comenzando con .)
        if path.name.startswith('.'):
            return True
            
        return False
    
    def detect_project_type(self, structure: ProjectStructure) -> str:
        """
        Detectar tipo de proyecto basado en archivos presentes
        
        Args:
            structure: Estructura del proyecto escaneada
            
        Returns:
            str: Tipo de proyecto detectado
        """
        file_extensions = set(structure.file_types.keys())
        
        # Detectar Python
        if '.py' in file_extensions:
            if 'requirements.txt' in [f.name for f in structure.files]:
                return "python-django" if self._has_django(structure) else "python-flask" if self._has_flask(structure) else "python"
            return "python"
        
        # Detectar JavaScript/TypeScript
        if any(ext in file_extensions for ext in ['.js', '.jsx', '.ts', '.tsx']):
            if 'package.json' in [f.name for f in structure.files]:
                return "nodejs"
            return "javascript"
        
        # Detectar Java
        if '.java' in file_extensions:
            if 'pom.xml' in [f.name for f in structure.files]:
                return "java-maven"
            elif 'build.gradle' in [f.name for f in structure.files]:
                return "java-gradle"
            return "java"
        
        # Detectar C/C++
        if any(ext in file_extensions for ext in ['.c', '.cpp', '.h', '.hpp']):
            if 'CMakeLists.txt' in [f.name for f in structure.files]:
                return "cpp-cmake"
            elif 'Makefile' in [f.name for f in structure.files]:
                return "cpp-make"
            return "cpp"
        
        # Por defecto
        return "multi-language"
    
    def _has_django(self, structure: ProjectStructure) -> bool:
        """Verificar si es un proyecto Django"""
        django_files = ['manage.py', 'settings.py', 'urls.py']
        found_files = [f.name for f in structure.files]
        return any(df in found_files for df in django_files)
    
    def _has_flask(self, structure: ProjectStructure) -> bool:
        """Verificar si es un proyecto Flask"""
        flask_files = ['app.py', 'flask_app.py', 'application.py']
        found_files = [f.name for f in structure.files]
        return any(ff in found_files for ff in flask_files) or \
               any('flask' in str(f).lower() for f in structure.files)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calcular hash MD5 de un archivo
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            str: Hash MD5 del archivo
        """
        hasher = hashlib.md5()
        
        try:
            with open(file_path, 'rb') as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
        except (IOError, OSError):
            return ""
        
        return hasher.hexdigest()
    
    def get_project_metadata(self, structure: ProjectStructure) -> Dict[str, Any]:
        """
        Extraer metadatos del proyecto
        
        Args:
            structure: Estructura del proyecto
            
        Returns:
            Dict: Metadatos del proyecto
        """
        metadata = {
            'project_root': str(structure.root_path),
            'total_files': len(structure.files),
            'total_directories': len(structure.directories),
            'total_size_bytes': structure.total_size,
            'total_size_human': self._format_size(structure.total_size),
            'file_types': structure.file_types,
            'project_type': self.detect_project_type(structure),
            'scan_timestamp': structure.scan_timestamp.isoformat()
        }
        
        # Detectar archivos de configuración
        config_files = []
        for file in structure.files:
            if any(config in str(file).lower() for config in 
                  ['package.json', 'requirements.txt', 'pom.xml', 
                   'build.gradle', 'dockerfile', '.gitignore', 
                   'readme.md', 'makefile', 'cmakelists.txt']):
                config_files.append(str(file))
        
        metadata['config_files'] = config_files
        
        return metadata
    
    def _format_size(self, size_bytes: int) -> str:
        """Formatear tamaño en bytes a formato legible"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def find_duplicate_files(self, structure: ProjectStructure) -> Dict[str, List[str]]:
        """
        Encontrar archivos duplicados por hash
        
        Args:
            structure: Estructura del proyecto
            
        Returns:
            Dict: Hash a lista de archivos duplicados
        """
        hashes = {}
        
        for file in structure.files:
            full_path = structure.root_path / file
            file_hash = self.calculate_file_hash(full_path)
            
            if file_hash:  # Solo si se pudo calcular el hash
                if file_hash not in hashes:
                    hashes[file_hash] = []
                hashes[file_hash].append(str(file))
        
        # Filtrar solo los hashes con múltiples archivos
        duplicates = {h: files for h, files in hashes.items() if len(files) > 1}
        
        return duplicates
    
    def export_structure(self, structure: ProjectStructure, 
                        output_format: str = 'json') -> str:
        """
        Exportar estructura del proyecto en diferentes formatos
        
        Args:
            structure: Estructura del proyecto
            output_format: Formato de salida (json, text, csv)
            
        Returns:
            str: Estructura formateada
        """
        metadata = self.get_project_metadata(structure)
        
        if output_format == 'json':
            return json.dumps(metadata, indent=2, ensure_ascii=False)
        
        elif output_format == 'text':
            output = []
            output.append(f"PROYECTO: {metadata['project_root']}")
            output.append(f"Tipo: {metadata['project_type']}")
            output.append(f"Archivos: {metadata['total_files']}")
            output.append(f"Tamaño: {metadata['total_size_human']}")
            output.append("\nTipos de archivo:")
            for ext, count in metadata['file_types'].items():
                output.append(f"  {ext}: {count}")
            
            if metadata['config_files']:
                output.append("\nArchivos de configuración:")
                for config in metadata['config_files']:
                    output.append(f"  {config}")
            
            return "\n".join(output)
        
        elif output_format == 'csv':
            output = []
            output.append("Extension,Cantidad")
            for ext, count in metadata['file_types'].items():
                output.append(f"{ext},{count}")
            return "\n".join(output)
        
        else:
            raise ValueError(f"Formato no soportado: {output_format}")