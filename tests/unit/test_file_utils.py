"""
Tests unitarios para las utilidades de archivos de ANALYZERBRAIN.

Este módulo prueba todas las funciones de file_utils.py, incluyendo:
- Lectura y escritura de archivos (sync/async)
- Listado de archivos
- Cálculo de hashes
- Obtención de metadatos
- Funciones auxiliares

Dependencias:
- pytest
- pytest-asyncio
- pytest-mock
- aiofiles

Autor: ANALYZERBRAIN Team
Versión: 1.0.0
"""

import pytest
import tempfile
import asyncio
import os
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

from src.utils.file_utils import FileUtils, file_utils


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def temp_directory():
    """Crea un directorio temporal para pruebas."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Limpiar después de la prueba
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_text_file(temp_directory):
    """Crea un archivo de texto de ejemplo."""
    file_path = temp_directory / "sample.txt"
    file_path.write_text("Hello, World!\nThis is a test file.")
    return file_path


@pytest.fixture
def sample_binary_file(temp_directory):
    """Crea un archivo binario de ejemplo."""
    file_path = temp_directory / "sample.bin"
    file_path.write_bytes(b"\x00\x01\x02\x03\x04\x05")
    return file_path


@pytest.fixture
def nested_directory_structure(temp_directory):
    """Crea una estructura de directorios anidados con archivos."""
    # Crear directorios
    dirs = [
        temp_directory / "dir1",
        temp_directory / "dir2",
        temp_directory / "dir1" / "subdir1",
        temp_directory / ".hidden_dir",  # Directorio oculto
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Crear archivos
    files = [
        temp_directory / "file1.txt",
        temp_directory / "file2.py",
        temp_directory / "file3.json",
        temp_directory / "dir1" / "file4.txt",
        temp_directory / "dir1" / "file5.py",
        temp_directory / "dir1" / "subdir1" / "file6.txt",
        temp_directory / "dir2" / "file7.txt",
        temp_directory / ".hidden_dir" / ".hidden_file",
        temp_directory / "__pycache__" / "cache_file.pyc",  # Directorio a excluir
    ]
    
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"Content of {f.name}")
    
    return temp_directory


# -------------------------------------------------------------------
# Tests para FileUtils - Métodos síncronos
# -------------------------------------------------------------------

class TestFileUtilsSync:
    """Tests para métodos síncronos de FileUtils."""
    
    def test_read_file_success(self, sample_text_file):
        """Verifica lectura exitosa de archivo de texto."""
        content = FileUtils.read_file(sample_text_file)
        assert content == "Hello, World!\nThis is a test file."
    
    def test_read_file_with_path_object(self, sample_text_file):
        """Verifica lectura con objeto Path."""
        content = FileUtils.read_file(Path(sample_text_file))
        assert "Hello, World!" in content
    
    def test_read_file_file_not_found(self, temp_directory):
        """Verifica que FileNotFoundError se lanza cuando el archivo no existe."""
        non_existent = temp_directory / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            FileUtils.read_file(non_existent)
        
        assert "no encontrado" in str(exc_info.value).lower()
    
    def test_read_file_with_different_encodings(self, temp_directory):
        """Verifica lectura con diferentes codificaciones."""
        # Crear archivo con encoding específico
        file_path = temp_directory / "latin1.txt"
        latin1_text = "café mañana"  # Caracteres especiales
        
        # Escribir en latin-1
        with open(file_path, 'w', encoding='latin-1') as f:
            f.write(latin1_text)
        
        # Leer con auto-detección de encoding
        content = FileUtils.read_file(file_path, encoding="utf-8")
        # El método intentará diferentes encodings, debería funcionar
        assert len(content) > 0
    
    def test_read_file_io_error(self, temp_directory):
        """Verifica manejo de errores de IO."""
        file_path = temp_directory / "test.txt"
        
        # PRIMERO: Crear el archivo
        file_path.write_text("test content")
        
        # Hacer el archivo no legible (simular error de permisos)
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with pytest.raises(IOError) as exc_info:
                FileUtils.read_file(file_path)
        
        assert "error" in str(exc_info.value).lower() or "permission" in str(exc_info.value).lower()
    
    def test_write_file_text(self, temp_directory):
        """Verifica escritura de archivo de texto."""
        file_path = temp_directory / "output.txt"
        content = "This is test content\nwith multiple lines."
        
        FileUtils.write_file(file_path, content)
        
        assert file_path.exists()
        assert file_path.read_text() == content
    
    def test_write_file_binary(self, temp_directory):
        """Verifica escritura de archivo binario."""
        file_path = temp_directory / "output.bin"
        content = b"\x00\x01\x02\x03\x04\x05"
        
        FileUtils.write_file(file_path, content)
        
        assert file_path.exists()
        assert file_path.read_bytes() == content
    
    def test_write_file_with_backup(self, temp_directory):
        """Verifica creación de backup al sobrescribir archivo."""
        file_path = temp_directory / "test.txt"
        original_content = "Original content"
        new_content = "New content"
        
        # Escribir archivo original
        FileUtils.write_file(file_path, original_content)
        
        # Sobrescribir con backup
        FileUtils.write_file(file_path, new_content, backup=True)
        
        # Verificar que el nuevo archivo existe con el nuevo contenido
        assert file_path.exists()
        assert file_path.read_text() == new_content
        
        # Verificar que se creó un backup
        backup_files = list(temp_directory.glob("test.txt.backup_*"))
        assert len(backup_files) == 1
        
        # Verificar que el backup tiene el contenido original
        assert backup_files[0].read_text() == original_content
    
    def test_write_file_creates_directory(self, temp_directory):
        """Verifica que se crean directorios padres si no existen."""
        file_path = temp_directory / "nested" / "deep" / "file.txt"
        content = "Test content"
        
        FileUtils.write_file(file_path, content)
        
        assert file_path.exists()
        assert file_path.read_text() == content
    
    def test_list_files_non_recursive(self, nested_directory_structure):
        """Verifica listado de archivos no recursivo."""
        files = FileUtils.list_files(nested_directory_structure, pattern="*.txt", recursive=False)
        
        # Solo debería encontrar archivos en el directorio raíz
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        assert "file2.py" not in file_names  # No es .txt
        assert "file4.txt" not in file_names  # Está en subdirectorio
    
    def test_list_files_recursive(self, nested_directory_structure):
        """Verifica listado de archivos recursivo."""
        files = FileUtils.list_files(nested_directory_structure, pattern="*.txt", recursive=True)
        
        # Debería encontrar todos los archivos .txt
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        assert "file4.txt" in file_names
        assert "file6.txt" in file_names
        assert "file7.txt" in file_names
        assert ".hidden_file" not in file_names  # Oculto y no .txt
    
    def test_list_files_with_exclude_dirs(self, nested_directory_structure):
        """Verifica que se excluyen directorios específicos."""
        files = FileUtils.list_files(
            nested_directory_structure,
            pattern="*.py",
            recursive=True,
            exclude_dirs=[".git", "__pycache__", ".pytest_cache"]
        )
        
        # No debería incluir archivos en __pycache__
        file_paths = [str(f) for f in files]
        assert any("file2.py" in p for p in file_paths)
        assert any("file5.py" in p for p in file_paths)
        assert not any("__pycache__" in p for p in file_paths)
        assert not any("cache_file.pyc" in p for p in file_paths)
    
    def test_list_files_nonexistent_directory(self, temp_directory):
        """Verifica listado en directorio inexistente."""
        non_existent = temp_directory / "nonexistent"
        files = FileUtils.list_files(non_existent)
        
        assert files == []  # Debería devolver lista vacía
    
    def test_calculate_hash_sha256(self, sample_text_file):
        """Verifica cálculo de hash SHA256."""
        hash_value = FileUtils.calculate_hash(sample_text_file, "sha256")
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 produce 64 caracteres hex
        # El hash debería ser consistente
        assert hash_value == FileUtils.calculate_hash(sample_text_file, "sha256")
    
    def test_calculate_hash_md5(self, sample_text_file):
        """Verifica cálculo de hash MD5."""
        hash_value = FileUtils.calculate_hash(sample_text_file, "md5")
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 32  # MD5 produce 32 caracteres hex
    
    def test_calculate_hash_file_not_found(self, temp_directory):
        """Verifica que FileNotFoundError se lanza al calcular hash de archivo inexistente."""
        non_existent = temp_directory / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            FileUtils.calculate_hash(non_existent)
    
    def test_calculate_hash_unknown_algorithm(self, sample_text_file):
        """Verifica que usa sha256 por defecto para algoritmo desconocido."""
        # Algoritmo inexistente debería usar sha256 por defecto
        with patch('hashlib.sha256') as mock_sha256:
            mock_hash = Mock()
            mock_hash.hexdigest.return_value = "mock_hash"
            mock_sha256.return_value = mock_hash
            
            hash_value = FileUtils.calculate_hash(sample_text_file, "unknown_algorithm")
            
            assert hash_value == "mock_hash"
            mock_sha256.assert_called_once()
    
    def test_get_file_info_text_file(self, sample_text_file):
        """Verifica obtención de información de archivo de texto."""
        info = FileUtils.get_file_info(sample_text_file)
        
        assert info["path"] == str(sample_text_file.absolute())
        assert info["name"] == "sample.txt"
        assert info["stem"] == "sample"
        assert info["suffix"] == ".txt"
        # Verificar que es un directorio temporal (contiene 'tmp' o 'temp')
        assert any(x in info["parent"].lower() for x in ["tmp", "temp"])
        assert info["size_bytes"] > 0
        assert "B" in info["size_human"]  # Formato humano
        assert isinstance(info["created"], str) and "T" in info["created"]  # ISO format
        assert isinstance(info["modified"], str) and "T" in info["modified"]
        assert isinstance(info["accessed"], str) and "T" in info["accessed"]
        assert info["is_file"] is True
        assert info["is_dir"] is False
        # Para archivos, el hash debe ser una cadena de 64 caracteres (SHA256)
        assert info["hash_sha256"] is not None
        assert len(info["hash_sha256"]) == 64
    
    def test_get_file_info_directory(self, temp_directory):
        """Verifica obtención de información de directorio."""
        info = FileUtils.get_file_info(temp_directory)

        assert info["name"] == temp_directory.name
        assert info["is_file"] is False
        assert info["is_dir"] is True
        assert info["size_bytes"] >= 0
        # El hash debe ser None para directorios
        assert info["hash_sha256"] is None
    
    def test_get_file_info_file_not_found(self, temp_directory):
        """Verifica que FileNotFoundError se lanza al obtener info de archivo inexistente."""
        non_existent = temp_directory / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            FileUtils.get_file_info(non_existent)
    
    def test_humanize_bytes(self):
        """Verifica conversión de bytes a formato legible."""
        test_cases = [
            (0, "0 B"),
            (100, "100 B"),
            (1023, "1023 B"),
            (1024, "1.00 KB"),
            (1024 * 1024, "1.00 MB"),
            (1024 * 1024 * 1024, "1.00 GB"),
            (1024 * 1024 * 1024 * 1024, "1.00 TB"),
            (1500, "1.46 KB"),
            (1500000, "1.43 MB"),
        ]
        
        for bytes_count, expected in test_cases:
            result = FileUtils._humanize_bytes(bytes_count)
            assert result == expected, f"Failed for {bytes_count} bytes"
    
    def test_humanize_bytes_negative(self):
        """Verifica que ValueError se lanza para bytes negativos."""
        with pytest.raises(ValueError):
            FileUtils._humanize_bytes(-1)
    
    def test_humanize_bytes_large(self):
        """Verifica conversión de valores muy grandes."""
        # 1 PB
        pb = 1024 * 1024 * 1024 * 1024 * 1024
        result = FileUtils._humanize_bytes(pb)
        assert "PB" in result
    
    def test_file_utils_instance(self):
        """Verifica que la instancia global funciona."""
        assert isinstance(file_utils, FileUtils)
        assert file_utils.read_file is FileUtils.read_file
        assert file_utils.write_file is FileUtils.write_file


# -------------------------------------------------------------------
# Tests para FileUtils - Métodos asíncronos
# -------------------------------------------------------------------

class TestFileUtilsAsync:
    """Tests para métodos asíncronos de FileUtils."""
    
    @pytest.mark.asyncio
    async def test_read_file_async_success(self, sample_text_file):
        """Verifica lectura asíncrona exitosa de archivo de texto."""
        content = await FileUtils.read_file_async(sample_text_file)
        assert content == "Hello, World!\nThis is a test file."
    
    @pytest.mark.asyncio
    async def test_read_file_async_file_not_found(self, temp_directory):
        """Verifica que FileNotFoundError se lanza en lectura async."""
        non_existent = temp_directory / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            await FileUtils.read_file_async(non_existent)
    
    @pytest.mark.asyncio
    async def test_read_file_async_unicode_error(self, temp_directory):
        """Verifica manejo de errores de Unicode en lectura async."""
        file_path = temp_directory / "test.txt"
        
        # PRIMERO: Crear el archivo para que exista
        file_path.write_text("test content")
        
        # Mock aiofiles.open para lanzar UnicodeDecodeError
        with patch('aiofiles.open', side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'test')):
            with pytest.raises(UnicodeDecodeError):
                await FileUtils.read_file_async(file_path)
    
    @pytest.mark.asyncio
    async def test_write_file_async_text(self, temp_directory):
        """Verifica escritura asíncrona de archivo de texto."""
        file_path = temp_directory / "async_output.txt"
        content = "Async test content"
        
        await FileUtils.write_file_async(file_path, content)
        
        assert file_path.exists()
        assert file_path.read_text() == content
    
    @pytest.mark.asyncio
    async def test_write_file_async_binary(self, temp_directory):
        """Verifica escritura asíncrona de archivo binario."""
        file_path = temp_directory / "async_output.bin"
        content = b"\x00\x01\x02\x03"
        
        await FileUtils.write_file_async(file_path, content)
        
        assert file_path.exists()
        assert file_path.read_bytes() == content
    
    @pytest.mark.asyncio
    async def test_write_file_async_creates_directory(self, temp_directory):
        """Verifica que escritura async crea directorios padres."""
        file_path = temp_directory / "async" / "nested" / "file.txt"
        content = "Test content"
        
        await FileUtils.write_file_async(file_path, content)
        
        assert file_path.exists()
        assert file_path.read_text() == content
    
    @pytest.mark.asyncio
    async def test_write_file_async_io_error(self, temp_directory):
        """Verifica manejo de errores de IO en escritura async."""
        file_path = temp_directory / "test.txt"
        
        # Mock aiofiles.open para lanzar IOError
        with patch('aiofiles.open', side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                await FileUtils.write_file_async(file_path, "content")


# -------------------------------------------------------------------
# Tests para edge cases y casos especiales
# -------------------------------------------------------------------

class TestFileUtilsEdgeCases:
    """Tests para edge cases y casos especiales."""
    
    def test_read_file_empty(self, temp_directory):
        """Verifica lectura de archivo vacío."""
        file_path = temp_directory / "empty.txt"
        file_path.write_text("")  # Archivo vacío
        
        content = FileUtils.read_file(file_path)
        assert content == ""
    
    def test_write_file_empty_content(self, temp_directory):
        """Verifica escritura de contenido vacío."""
        file_path = temp_directory / "empty.txt"
        
        # Texto vacío
        FileUtils.write_file(file_path, "")
        assert file_path.exists()
        assert file_path.read_text() == ""
        
        # Bytes vacíos
        file_path2 = temp_directory / "empty.bin"
        FileUtils.write_file(file_path2, b"")
        assert file_path2.exists()
        assert file_path2.read_bytes() == b""
    
    def test_list_files_empty_directory(self, temp_directory):
        """Verifica listado en directorio vacío."""
        empty_dir = temp_directory / "empty"
        empty_dir.mkdir()
        
        files = FileUtils.list_files(empty_dir)
        assert files == []
    
    def test_list_files_pattern_with_dot(self, nested_directory_structure):
        """Verifica listado con patrón que incluye punto."""
        # Buscar archivos con extensión específica
        py_files = FileUtils.list_files(nested_directory_structure, pattern="*.py")
        txt_files = FileUtils.list_files(nested_directory_structure, pattern="*.txt")
        
        # Verificar que se encontraron los tipos correctos
        assert all(f.suffix == ".py" for f in py_files)
        assert all(f.suffix == ".txt" for f in txt_files)
    
    def test_calculate_hash_large_file(self, temp_directory):
        """Verifica cálculo de hash para archivo grande (simulado)."""
        # Crear archivo "grande" (64KB para pruebas)
        file_path = temp_directory / "large.bin"
        large_content = b"X" * (64 * 1024)  # 64KB
        
        with open(file_path, 'wb') as f:
            f.write(large_content)
        
        hash_value = FileUtils.calculate_hash(file_path)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256
    
    def test_get_file_info_symlink(self, temp_directory):
        """Verifica obtención de información de symlink (si el SO lo soporta)."""
        target = temp_directory / "target.txt"
        target.write_text("Target content")
        
        symlink = temp_directory / "link.txt"
        
        try:
            symlink.symlink_to(target)
            
            info = FileUtils.get_file_info(symlink)
            
            # En algunos sistemas, stat() sigue el symlink
            # Solo verificar que no lanza excepción
            assert info["name"] == "link.txt"
            
        except (OSError, NotImplementedError):
            # Algunos sistemas/setups no soportan symlinks
            pytest.skip("Symlinks no soportados en este sistema")
    
    def test_write_file_permission_denied(self, temp_directory):
        """Verifica manejo de error de permisos al escribir."""
        file_path = temp_directory / "readonly.txt"
        
        # Crear archivo y hacerlo de solo lectura
        file_path.write_text("Original")
        file_path.chmod(0o444)  # Solo lectura
        
        try:
            # Intentar sobrescribir debería fallar
            with pytest.raises(PermissionError):
                FileUtils.write_file(file_path, "New content")
        finally:
            # Restaurar permisos para limpieza
            file_path.chmod(0o644)
    
    def test_calculate_hash_consistent_across_calls(self, sample_text_file):
        """Verifica que el hash sea consistente en múltiples llamadas."""
        hash1 = FileUtils.calculate_hash(sample_text_file)
        hash2 = FileUtils.calculate_hash(sample_text_file)
        hash3 = FileUtils.calculate_hash(sample_text_file)
        
        assert hash1 == hash2 == hash3
    
    def test_file_info_timestamps(self, temp_directory):
        """Verifica que los timestamps sean razonables."""
        file_path = temp_directory / "timestamp_test.txt"
        
        # Crear archivo
        before_creation = datetime.now().timestamp()
        FileUtils.write_file(file_path, "Test")
        after_creation = datetime.now().timestamp()
        
        info = FileUtils.get_file_info(file_path)
        
        # Parsear timestamps ISO a datetime
        created = datetime.fromisoformat(info["created"].replace('Z', '+00:00'))
        modified = datetime.fromisoformat(info["modified"].replace('Z', '+00:00'))
        
        created_ts = created.timestamp()
        modified_ts = modified.timestamp()
        
        # Verificar que los timestamps están entre before y after
        assert before_creation <= created_ts <= after_creation
        assert before_creation <= modified_ts <= after_creation


# -------------------------------------------------------------------
# Tests de integración
# -------------------------------------------------------------------

class TestFileUtilsIntegration:
    """Tests de integración que combinan múltiples métodos."""
    
    def test_read_write_cycle(self, temp_directory):
        """Verifica ciclo completo de escritura y lectura."""
        file_path = temp_directory / "cycle.txt"
        original_content = "Original test content with special chars: café mañana"
        
        # Escribir
        FileUtils.write_file(file_path, original_content)
        
        # Leer y verificar
        read_content = FileUtils.read_file(file_path)
        assert read_content == original_content
        
        # Calcular hash
        hash_value = FileUtils.calculate_hash(file_path)
        assert len(hash_value) == 64
        
        # Obtener info
        info = FileUtils.get_file_info(file_path)
        assert info["size_bytes"] == len(original_content.encode('utf-8'))
        assert info["hash_sha256"] == hash_value
    
    def test_backup_and_restore(self, temp_directory):
        """Verifica funcionalidad de backup y restauración manual."""
        file_path = temp_directory / "data.txt"
        v1_content = "Version 1"
        v2_content = "Version 2"
        
        # Escribir versión 1
        FileUtils.write_file(file_path, v1_content)
        v1_hash = FileUtils.calculate_hash(file_path)
        
        # Sobrescribir con backup
        FileUtils.write_file(file_path, v2_content, backup=True)
        v2_hash = FileUtils.calculate_hash(file_path)
        
        # Encontrar backup
        backup_files = list(temp_directory.glob("data.txt.backup_*"))
        assert len(backup_files) == 1
        
        # Verificar que el backup tiene v1
        backup_content = FileUtils.read_file(backup_files[0])
        assert backup_content == v1_content
        
        # Verificar que el archivo actual tiene v2
        current_content = FileUtils.read_file(file_path)
        assert current_content == v2_content
    
    def test_directory_operations(self, nested_directory_structure):
        """Verifica operaciones complejas con directorios."""
        # Listar todos los archivos
        all_files = FileUtils.list_files(nested_directory_structure, pattern="*", recursive=True)
        assert len(all_files) > 0
        
        # Para cada archivo, obtener info y calcular hash
        for file_path in all_files:
            if file_path.is_file():
                info = FileUtils.get_file_info(file_path)
                assert info["is_file"] is True
                
                # Para archivos, el hash debe estar presente
                assert info["hash_sha256"] is not None
                assert len(info["hash_sha256"]) == 64
                
                # Verificar que el tamaño humano es una string válida
                assert isinstance(info["size_human"], str)
                assert any(unit in info["size_human"] for unit in ["B", "KB", "MB", "GB"])


# -------------------------------------------------------------------
# Tests de rendimiento y stress (opcionales)
# -------------------------------------------------------------------

class TestFileUtilsPerformance:
    """Tests de rendimiento para operaciones de archivos."""
    
    def test_read_large_file_performance(self, temp_directory):
        """Verifica que la lectura de archivos grandes funciona (sin timeout)."""
        # Crear archivo de 1MB para prueba de rendimiento
        file_path = temp_directory / "large.txt"
        large_content = "X" * (1024 * 1024)  # 1MB
        
        FileUtils.write_file(file_path, large_content)
        
        # Medir tiempo (aproximado)
        import time
        start = time.time()
        
        content = FileUtils.read_file(file_path)
        
        end = time.time()
        duration = end - start
        
        # Verificar que se leyó todo
        assert len(content) == len(large_content)
        
        # La lectura no debería tomar más de 2 segundos (ajustable)
        assert duration < 2.0, f"Lectura lenta: {duration:.2f} segundos"
    
    def test_hash_large_file_performance(self, temp_directory):
        """Verifica que el cálculo de hash para archivos grandes funciona."""
        # Crear archivo de 2MB
        file_path = temp_directory / "large_for_hash.bin"
        large_content = b"0" * (2 * 1024 * 1024)  # 2MB
        
        with open(file_path, 'wb') as f:
            f.write(large_content)
        
        import time
        start = time.time()
        
        hash_value = FileUtils.calculate_hash(file_path)
        
        end = time.time()
        duration = end - start
        
        assert len(hash_value) == 64
        # Cálculo de hash no debería tomar más de 5 segundos para 2MB
        assert duration < 5.0, f"Hash lento: {duration:.2f} segundos"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])