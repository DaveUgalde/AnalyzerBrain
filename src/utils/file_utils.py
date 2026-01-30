"""
FileUtils - Utilidades para manejo seguro y eficiente de archivos.
Incluye operaciones de lectura/escritura, copiado, movimiento y cálculo de estadísticas.
"""

import os
import shutil
import hashlib
import tempfile
import mimetypes
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO, Callable
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import logging
from ..core.exceptions import BrainException, ValidationError

logger = logging.getLogger(__name__)

class FileUtils:
    """
    Utilidades para operaciones seguras y eficientes con archivos.
    
    Características:
    1. Operaciones atómicas con rollback en error
    2. Validación de rutas y permisos
    3. Cálculo de hashes y verificación de integridad
    4. Operaciones en lote con manejo de errores
    5. Caché de metadatos para operaciones frecuentes
    """
    
    # Cache para metadatos de archivos (ruta -> metadata)
    _metadata_cache = {}
    _CACHE_SIZE = 1000
    
    @staticmethod
    def read_file_safely(
        file_path: Union[str, Path],
        encoding: str = 'utf-8',
        mode: str = 'r',
        binary_chunk_size: int = 8192,
        max_size_mb: int = 10
    ) -> Union[str, bytes]:
        """
        Lee un archivo de manera segura con validaciones.
        
        Args:
            file_path: Ruta al archivo
            encoding: Encoding para archivos de texto
            mode: 'r' para texto, 'rb' para binario
            binary_chunk_size: Tamaño de chunk para lectura binaria
            max_size_mb: Tamaño máximo permitido en MB
            
        Returns:
            Contenido del archivo como string o bytes
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            PermissionError: Si no hay permisos de lectura
            ValidationError: Si el archivo excede el tamaño máximo
            UnicodeDecodeError: Si hay error de decodificación
        """
        file_path = Path(file_path)
        
        # Validaciones iniciales
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"No read permission for: {file_path}")
        
        # Verificar tamaño
        file_size = file_path.stat().st_size
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            raise ValidationError(
                f"File too large: {file_size} bytes > {max_size_bytes} bytes "
                f"({max_size_mb} MB limit)"
            )
        
        try:
            if 'b' in mode:
                # Lectura binaria
                content = b''
                with open(file_path, 'rb') as f:
                    while chunk := f.read(binary_chunk_size):
                        content += chunk
                return content
            else:
                # Lectura de texto
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
                    
        except UnicodeDecodeError as e:
            # Intentar detectar encoding
            detected_encoding = FileUtils._detect_encoding(file_path)
            if detected_encoding and detected_encoding != encoding:
                try:
                    with open(file_path, 'r', encoding=detected_encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    raise
            else:
                raise
        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise BrainException(f"Failed to read file {file_path}: {e}")
    
    @staticmethod
    def write_file_safely(
        file_path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = 'utf-8',
        mode: str = 'w',
        backup: bool = False,
        atomic: bool = True
    ) -> bool:
        """
        Escribe en un archivo de manera segura con backup y atomicidad.
        
        Args:
            file_path: Ruta destino
            content: Contenido a escribir
            encoding: Encoding para archivos de texto
            mode: 'w' para texto, 'wb' para binario
            backup: Crear backup del archivo existente
            atomic: Usar escritura atómica (temp file + rename)
            
        Returns:
            True si la escritura fue exitosa
            
        Raises:
            PermissionError: Si no hay permisos de escritura
            BrainException: Si hay error durante la escritura
        """
        file_path = Path(file_path)
        temp_file = None
        
        try:
            # Crear directorio padre si no existe
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Verificar permisos de escritura
            if file_path.exists() and not os.access(file_path, os.W_OK):
                raise PermissionError(f"No write permission for: {file_path}")
            
            # Crear backup si es necesario y existe
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                shutil.copy2(file_path, backup_path)
            
            if atomic:
                # Escritura atómica
                with tempfile.NamedTemporaryFile(
                    mode=mode,
                    encoding=None if 'b' in mode else encoding,
                    delete=False,
                    dir=file_path.parent,
                    prefix=f".tmp_{file_path.name}."
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    
                    if isinstance(content, str) and 'b' not in mode:
                        temp_file.write(content)
                    elif isinstance(content, bytes) and 'b' in mode:
                        temp_file.write(content)
                    else:
                        raise ValueError(
                            f"Content type {type(content)} doesn't match mode {mode}"
                        )
                
                # Reemplazar archivo original
                shutil.move(temp_path, file_path)
            else:
                # Escritura directa
                with open(file_path, mode, encoding=None if 'b' in mode else encoding) as f:
                    f.write(content)
            
            # Limpiar cache de metadatos
            FileUtils._clear_metadata_cache(str(file_path))
            
            logger.debug(f"Successfully wrote file: {file_path}")
            return True
            
        except Exception as e:
            # Si hay error en escritura atómica, eliminar temp file
            if atomic and temp_file and Path(temp_file.name).exists():
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
            logger.error(f"Error writing file {file_path}: {e}")
            raise BrainException(f"Failed to write file {file_path}: {e}")
    
    @staticmethod
    def copy_file(
        source: Union[str, Path],
        destination: Union[str, Path],
        overwrite: bool = False,
        preserve_metadata: bool = True,
        buffer_size: int = 16384
    ) -> bool:
        """
        Copia un archivo con manejo de errores y opciones avanzadas.
        
        Args:
            source: Ruta origen
            destination: Ruta destino
            overwrite: Sobrescribir si existe
            preserve_metadata: Preservar metadatos (timestamps, permisos)
            buffer_size: Tamaño del buffer de copia
            
        Returns:
            True si la copia fue exitosa
            
        Raises:
            FileNotFoundError: Si el archivo origen no existe
            FileExistsError: Si el destino existe y overwrite=False
            BrainException: Si hay error durante la copia
        """
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination file exists: {destination}")
        
        try:
            # Copiar con shutil para mejor manejo de metadatos
            if preserve_metadata:
                shutil.copy2(source, destination)
            else:
                # Copia manual con buffer
                with open(source, 'rb') as src_file:
                    with open(destination, 'wb') as dst_file:
                        while chunk := src_file.read(buffer_size):
                            dst_file.write(chunk)
                
                # Copiar permisos básicos
                shutil.copymode(source, destination)
            
            logger.debug(f"Copied {source} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error copying file {source} to {destination}: {e}")
            raise BrainException(f"Failed to copy file: {e}")
    
    @staticmethod
    def move_file(
        source: Union[str, Path],
        destination: Union[str, Path],
        overwrite: bool = False
    ) -> bool:
        """
        Mueve/renombra un archivo de manera segura.
        
        Args:
            source: Ruta origen
            destination: Ruta destino
            overwrite: Sobrescribir si existe
            
        Returns:
            True si el movimiento fue exitoso
            
        Raises:
            FileNotFoundError: Si el archivo origen no existe
            FileExistsError: Si el destino existe y overwrite=False
            BrainException: Si hay error durante el movimiento
        """
        source = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination file exists: {destination}")
        
        try:
            # Si overwrite=True y destino existe, eliminarlo primero
            if destination.exists() and overwrite:
                destination.unlink()
            
            # Mover archivo
            shutil.move(str(source), str(destination))
            
            # Limpiar caches
            FileUtils._clear_metadata_cache(str(source))
            FileUtils._clear_metadata_cache(str(destination))
            
            logger.debug(f"Moved {source} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving file {source} to {destination}: {e}")
            raise BrainException(f"Failed to move file: {e}")
    
    @staticmethod
    def delete_file(
        file_path: Union[str, Path],
        secure_delete: bool = False,
        passes: int = 3
    ) -> bool:
        """
        Elimina un archivo de manera segura.
        
        Args:
            file_path: Ruta al archivo
            secure_delete: Sobrescribir con datos aleatorios antes de eliminar
            passes: Número de pases de sobrescritura (solo si secure_delete=True)
            
        Returns:
            True si la eliminación fue exitosa
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            PermissionError: Si no hay permisos de eliminación
            BrainException: Si hay error durante la eliminación
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.access(file_path, os.W_OK):
            raise PermissionError(f"No write permission for: {file_path}")
        
        try:
            if secure_delete:
                # Sobrescribir con datos aleatorios antes de eliminar
                file_size = file_path.stat().st_size
                
                with open(file_path, 'wb') as f:
                    for _ in range(passes):
                        f.seek(0)
                        # Generar datos aleatorios
                        random_data = os.urandom(file_size)
                        f.write(random_data)
                        f.flush()
                        os.fsync(f.fileno())
            
            # Eliminar archivo
            file_path.unlink()
            
            # Limpiar cache
            FileUtils._clear_metadata_cache(str(file_path))
            
            logger.debug(f"Deleted file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            raise BrainException(f"Failed to delete file: {e}")
    
    @staticmethod
    def find_files(
        directory: Union[str, Path],
        pattern: str = "**/*",
        recursive: bool = True,
        include_dirs: bool = False,
        file_types: Optional[List[str]] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        modified_after: Optional[datetime] = None,
        modified_before: Optional[datetime] = None
    ) -> List[Path]:
        """
        Encuentra archivos que cumplan con criterios específicos.
        
        Args:
            directory: Directorio raíz para búsqueda
            pattern: Patrón glob para coincidencia
            recursive: Búsqueda recursiva en subdirectorios
            include_dirs: Incluir directorios en los resultados
            file_types: Lista de extensiones permitidas (ej: ['.py', '.js'])
            min_size: Tamaño mínimo en bytes
            max_size: Tamaño máximo en bytes
            modified_after: Fecha mínima de modificación
            modified_before: Fecha máxima de modificación
            
        Returns:
            Lista de Paths que cumplen los criterios
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        results = []
        
        # Construir patrón de búsqueda
        if recursive:
            glob_pattern = pattern
        else:
            # Para no recursivo, buscar solo en el directorio actual
            glob_pattern = pattern.split('/')[-1]
        
        # Buscar archivos
        for item in directory.glob(glob_pattern):
            # Saltar si es directorio y no queremos incluirlos
            if item.is_dir() and not include_dirs:
                continue
            
            # Filtrar por tipo de archivo
            if file_types and item.is_file():
                if item.suffix.lower() not in [ext.lower() for ext in file_types]:
                    continue
            
            # Filtrar por tamaño
            if min_size is not None and item.is_file():
                if item.stat().st_size < min_size:
                    continue
            
            if max_size is not None and item.is_file():
                if item.stat().st_size > max_size:
                    continue
            
            # Filtrar por fecha de modificación
            if modified_after is not None:
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < modified_after:
                    continue
            
            if modified_before is not None:
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime > modified_before:
                    continue
            
            results.append(item)
        
        return results
    
    @staticmethod
    def calculate_file_stats(
        file_path: Union[str, Path],
        include_hash: bool = True,
        hash_algorithm: str = 'sha256'
    ) -> Dict[str, any]:
        """
        Calcula estadísticas detalladas de un archivo.
        
        Args:
            file_path: Ruta al archivo
            include_hash: Calcular hash del contenido
            hash_algorithm: Algoritmo de hash a usar
            
        Returns:
            Diccionario con estadísticas del archivo
        """
        file_path = Path(file_path)
        cache_key = f"{file_path}:{include_hash}:{hash_algorithm}"
        
        # Verificar cache
        if cache_key in FileUtils._metadata_cache:
            return FileUtils._metadata_cache[cache_key].copy()
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stats = os.stat(file_path)
        
        file_stats = {
            'path': str(file_path),
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix,
            'parent': str(file_path.parent),
            'exists': True,
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir(),
            'is_symlink': file_path.is_symlink(),
            
            # Tamaños
            'size_bytes': stats.st_size,
            'size_kb': stats.st_size / 1024,
            'size_mb': stats.st_size / (1024 * 1024),
            
            # Fechas
            'created': datetime.fromtimestamp(stats.st_ctime),
            'modified': datetime.fromtimestamp(stats.st_mtime),
            'accessed': datetime.fromtimestamp(stats.st_atime),
            
            # Permisos
            'permissions_octal': oct(stats.st_mode)[-3:],
            'permissions_human': FileUtils._format_permissions(stats.st_mode),
            'owner_uid': stats.st_uid,
            'group_gid': stats.st_gid,
            
            # Metadata adicional
            'inode': stats.st_ino,
            'device': stats.st_dev,
            'hard_links': stats.st_nlink,
            
            # Tipo MIME
            'mime_type': FileUtils._get_mime_type(file_path),
            'encoding_guess': None
        }
        
        # Calcular hash si se solicita
        if include_hash and file_path.is_file():
            try:
                file_stats['hash'] = FileUtils._calculate_file_hash(
                    file_path, hash_algorithm
                )
                file_stats['hash_algorithm'] = hash_algorithm
            except Exception as e:
                file_stats['hash'] = None
                file_stats['hash_error'] = str(e)
        
        # Intentar detectar encoding para archivos de texto
        if file_path.is_file() and file_stats['mime_type'] and \
           'text' in file_stats['mime_type']:
            try:
                file_stats['encoding_guess'] = FileUtils._detect_encoding(file_path)
            except:
                file_stats['encoding_guess'] = None
        
        # Actualizar cache
        FileUtils._update_metadata_cache(cache_key, file_stats)
        
        return file_stats.copy()
    
    # ========== MÉTODOS PRIVADOS ==========
    
    @staticmethod
    def _detect_encoding(file_path: Path, sample_size: int = 4096) -> Optional[str]:
        """Detecta encoding de un archivo de texto."""
        try:
            import chardet
            
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
            
            if not raw_data:
                return 'utf-8'
            
            result = chardet.detect(raw_data)
            return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
            
        except ImportError:
            # Fallback simple si chardet no está disponible
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1024)
                return 'utf-8'
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        f.read(1024)
                    return 'latin-1'
                except UnicodeDecodeError:
                    return None
        except Exception:
            return None
    
    @staticmethod
    def _calculate_file_hash(file_path: Path, algorithm: str) -> str:
        """Calcula hash de un archivo."""
        hasher = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    @staticmethod
    def _get_mime_type(file_path: Path) -> Optional[str]:
        """Obtiene tipo MIME de un archivo."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        if not mime_type and file_path.is_file():
            # Fallback basado en extensión
            ext = file_path.suffix.lower()
            mime_map = {
                '.py': 'text/x-python',
                '.js': 'text/javascript',
                '.jsx': 'text/javascript',
                '.ts': 'text/typescript',
                '.tsx': 'text/typescript',
                '.java': 'text/x-java-source',
                '.cpp': 'text/x-c++src',
                '.c': 'text/x-csrc',
                '.h': 'text/x-chdr',
                '.go': 'text/x-go',
                '.rs': 'text/rust',
                '.cs': 'text/x-csharp',
                '.php': 'text/x-php',
                '.rb': 'text/x-ruby',
                '.md': 'text/markdown',
                '.json': 'application/json',
                '.yaml': 'application/x-yaml',
                '.yml': 'application/x-yaml',
                '.xml': 'application/xml',
                '.html': 'text/html',
                '.css': 'text/css',
            }
            mime_type = mime_map.get(ext)
        
        return mime_type
    
    @staticmethod
    def _format_permissions(mode: int) -> str:
        """Formatea permisos en formato humano (rwx)."""
        permissions = []
        
        # Owner
        permissions.append('r' if mode & 0o400 else '-')
        permissions.append('w' if mode & 0o200 else '-')
        permissions.append('x' if mode & 0o100 else '-')
        
        # Group
        permissions.append('r' if mode & 0o040 else '-')
        permissions.append('w' if mode & 0o020 else '-')
        permissions.append('x' if mode & 0o010 else '-')
        
        # Others
        permissions.append('r' if mode & 0o004 else '-')
        permissions.append('w' if mode & 0o002 else '-')
        permissions.append('x' if mode & 0o001 else '-')
        
        return ''.join(permissions)
    
    @staticmethod
    def _update_metadata_cache(key: str, metadata: Dict) -> None:
        """Actualiza cache de metadatos con LRU."""
        if len(FileUtils._metadata_cache) >= FileUtils._CACHE_SIZE:
            # Eliminar el más antiguo (primer elemento)
            oldest_key = next(iter(FileUtils._metadata_cache))
            FileUtils._metadata_cache.pop(oldest_key)
        
        FileUtils._metadata_cache[key] = metadata
    
    @staticmethod
    def _clear_metadata_cache(file_path: Optional[str] = None) -> None:
        """Limpia cache de metadatos."""
        if file_path:
            # Eliminar todas las entradas para este archivo
            keys_to_remove = [
                key for key in FileUtils._metadata_cache.keys()
                if key.startswith(file_path)
            ]
            for key in keys_to_remove:
                FileUtils._metadata_cache.pop(key, None)
        else:
            # Limpiar todo el cache
            FileUtils._metadata_cache.clear()