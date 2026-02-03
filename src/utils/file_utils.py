"""
Utilidades para operaciones de archivos en ANALYZERBRAIN.

Este módulo centraliza funciones comunes para la manipulación de archivos,
incluyendo lectura y escritura síncrona y asíncrona, listado de archivos,
cálculo de hashes, y obtención de metadatos detallados.

Está diseñado para ser reutilizable por múltiples componentes del sistema
(indexadores, agentes, analizadores y utilidades internas), garantizando
manejo consistente de errores, logging y compatibilidad multiplataforma.

Características principales:
- Lectura y escritura de archivos (sync / async).
- Creación automática de directorios.
- Backups opcionales al sobrescribir archivos.
- Listado de archivos con filtrado y exclusión de directorios.
- Cálculo de hashes criptográficos (sha256 por defecto).
- Obtención de metadatos enriquecidos de archivos.
- Conversión de tamaños de archivo a formato legible.

Dependencias:
- aiofiles (operaciones asíncronas de archivos)
- loguru (logging estructurado)

Clases:
- FileUtils: Conjunto de métodos estáticos para operaciones de archivos.

Instancias globales:
- file_utils: Instancia conveniente de FileUtils.

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 1.0.0
"""

import hashlib
import os
import shutil
import aiofiles
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from loguru import logger


class FileUtils:
    """Utilidades para operaciones de archivos."""
    
    @staticmethod
    def read_file(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
        """
        Lee un archivo de texto de forma síncrona.
        
        Args:
            file_path: Ruta al archivo
            encoding: Codificación del archivo
            
        Returns:
            Contenido del archivo
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            IOError: Si hay error de lectura
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Intentar con diferentes codificaciones
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(path, 'r', encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise IOError(f"No se pudo decodificar el archivo: {file_path}")
    
    @staticmethod
    async def read_file_async(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
        """
        Lee un archivo de texto de forma asíncrona.
        
        Args:
            file_path: Ruta al archivo
            encoding: Codificación del archivo
            
        Returns:
            Contenido del archivo
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        try:
            async with aiofiles.open(path, 'r', encoding=encoding) as f:
                return await f.read()
        except UnicodeDecodeError as e:
            logger.error(f"Error decodificando archivo {file_path}: {e}")
            raise
    
    @staticmethod
    def write_file(
        file_path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
        backup: bool = False
    ) -> None:
        """
        Escribe contenido en un archivo.
        
        Args:
            file_path: Ruta al archivo
            content: Contenido a escribir
            encoding: Codificación para texto
            backup: Si True, crea backup si el archivo ya existe
        """
        path = Path(file_path)
        
        # Crear backup si es necesario
        if backup and path.exists():
            backup_path = path.parent / f"{path.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(path, backup_path)
            logger.debug(f"Backup creado: {backup_path}")
        
        # Crear directorio si no existe
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Escribir archivo
        try:
            if isinstance(content, str):
                with open(path, 'w', encoding=encoding) as f:
                    f.write(content)
            else:
                with open(path, 'wb') as f:
                    f.write(content)
            
            logger.debug(f"Archivo escrito: {file_path} ({len(content)} bytes)")
            
        except IOError as e:
            logger.error(f"Error escribiendo archivo {file_path}: {e}")
            raise
    
    @staticmethod
    async def write_file_async(
        file_path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8"
    ) -> None:
        """
        Escribe contenido en un archivo de forma asíncrona.
        
        Args:
            file_path: Ruta al archivo
            content: Contenido a escribir
            encoding: Codificación para texto
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(content, str):
                async with aiofiles.open(path, 'w', encoding=encoding) as f:
                    await f.write(content)
            else:
                async with aiofiles.open(path, 'wb') as f:
                    await f.write(content)
            
            logger.debug(f"Archivo escrito async: {file_path}")
            
        except IOError as e:
            logger.error(f"Error escribiendo archivo async {file_path}: {e}")
            raise
    
    @staticmethod
    def list_files(
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = True,
        exclude_dirs: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Lista archivos en un directorio.
        
        Args:
            directory: Directorio a escanear
            pattern: Patrón de búsqueda (ej: "*.py")
            recursive: Si True, busca recursivamente
            exclude_dirs: Directorios a excluir
            
        Returns:
            Lista de rutas a archivos
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        
        exclude_dirs = exclude_dirs or ['.git', '__pycache__', '.pytest_cache', 'node_modules']
        
        files: List[Path] = []
        
        if recursive:
            for root, dirs, filenames in os.walk(dir_path):
                # Excluir directorios
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for filename in filenames:
                    file_path = Path(root) / filename
                    if file_path.match(pattern):
                        files.append(file_path)
        else:
            files = list(dir_path.glob(pattern))
        
        return files
    
    @staticmethod
    def calculate_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
        """
        Calcula el hash de un archivo.
        
        Args:
            file_path: Ruta al archivo
            algorithm: Algoritmo de hash (md5, sha1, sha256, sha512)
            
        Returns:
            Hash hexadecimal del archivo
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)
        
        with open(path, 'rb') as f:
            file_hash = hash_func()
            for chunk in iter(lambda: f.read(8192), b""):
                file_hash.update(chunk)
        
        return file_hash.hexdigest()
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Obtiene información detallada de un archivo.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Diccionario con información del archivo
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        stat = path.stat()
        
        result = {
            "path": str(path.absolute()),
            "name": path.name,
            "stem": path.stem,
            "suffix": path.suffix,
            "parent": str(path.parent),
            "size_bytes": stat.st_size,
            "size_human": FileUtils._humanize_bytes(stat.st_size),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
        }
        
        # Solo calcular hash para archivos regulares, no para directorios
        if path.is_file():
            result["hash_sha256"] = FileUtils.calculate_hash(path, "sha256")
        else:
            result["hash_sha256"] = None
        
        return result
    
    @staticmethod
    def _humanize_bytes(bytes_count: int | float) -> str:
        """Convierte bytes a formato legible."""
        if bytes_count < 0:
            raise ValueError("bytes_count no puede ser negativo")
        units = ("B", "KB", "MB", "GB", "TB", "PB")
        
        for unit in units:
            if bytes_count < 1024:
                if unit == "B":
                    return f"{int(bytes_count)} {unit}"
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024

        return f"{bytes_count:.2f} PB"


# Instancia global
file_utils = FileUtils()