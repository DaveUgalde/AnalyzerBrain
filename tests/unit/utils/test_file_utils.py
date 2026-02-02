# tests/unit/test_file_utils.py
import pytest # type: ignore
from src.utils.file_utils import FileUtils

def test_read_write_file(tmp_path): # type: ignore
    test_file = tmp_path / "test.txt" # type: ignore
    content = "Hello, World!"
    
    FileUtils.write_file(test_file, content) # type: ignore
    assert FileUtils.read_file(test_file) == content # type: ignore

def test_list_files(tmp_path): # type: ignore
    # Crear estructura de directorios
    (tmp_path / "dir1").mkdir() # type: ignore
    (tmp_path / "dir2").mkdir() # type: ignore
    
    # Crear archivos
    (tmp_path / "file1.py").write_text("test") # type: ignore
    (tmp_path / "dir1" / "file2.py").write_text("test") # type: ignore
    (tmp_path / "file3.txt").write_text("test") # type: ignore
    
    files = FileUtils.list_files(tmp_path, pattern="*.py", recursive=True) # type: ignore
    assert len(files) == 2
    assert any("file1.py" in str(f) for f in files)
    assert any("file2.py" in str(f) for f in files)

def test_calculate_hash(tmp_path): # type: ignore
    test_file = tmp_path / "test.txt" # type: ignore
    content = "Hello, World!" # type: ignore
    test_file.write_text(content) # type: ignore
    
    hash_result = FileUtils.calculate_hash(test_file, "sha256") # type: ignore
    assert len(hash_result) == 64  # Longitud de hash SHA256 en hex
    assert hash_result == "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"

def test_get_file_info(tmp_path): # type: ignore
    test_file = tmp_path / "test.txt" # type: ignore
    test_file.write_text("content") # type: ignore
    
    info = FileUtils.get_file_info(test_file)  # type: ignore
    assert info["name"] == "test.txt"
    assert info["size_bytes"] > 0
    assert "created" in info
    assert "modified" in info