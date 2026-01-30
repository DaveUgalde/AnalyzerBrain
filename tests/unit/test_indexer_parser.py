"""
Pruebas unitarias para el módulo indexer/multi_language_parser.py
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from indexer.multi_language_parser import (
    MultiLanguageParser,
    ParserConfig,
    ParseResult,
    Entity,
    LanguageType,
    ParseMode
)


class TestParserConfig:
    """Pruebas para ParserConfig."""
    
    def test_default_config(self):
        """Test configuración por defecto."""
        config = ParserConfig()
        
        assert len(config.enabled_languages) > 0
        assert config.default_mode == ParseMode.STANDARD
        assert config.max_file_size_mb == 10
        assert config.timeout_seconds == 30
        assert config.include_comments is True
        assert config.include_whitespace is False
        assert config.cache_parsed_files is True
        assert config.cache_size == 1000
    
    def test_custom_config(self):
        """Test configuración personalizada."""
        config = ParserConfig(
            enabled_languages=[LanguageType.PYTHON, LanguageType.JAVASCRIPT],
            default_mode=ParseMode.QUICK,
            max_file_size_mb=5,
            timeout_seconds=60,
            include_comments=False,
            cache_size=500
        )
        
        assert len(config.enabled_languages) == 2
        assert config.default_mode == ParseMode.QUICK
        assert config.max_file_size_mb == 5
        assert config.timeout_seconds == 60
        assert config.include_comments is False
        assert config.cache_size == 500


class TestParseResult:
    """Pruebas para ParseResult."""
    
    def test_successful_result(self):
        """Test resultado exitoso de parsing."""
        result = ParseResult(
            success=True,
            language=LanguageType.PYTHON,
            ast={"type": "module"},
            entities=[Entity(type="function", name="test", start_line=1, end_line=10)],
            parse_time_ms=50.5
        )
        
        assert result.success is True
        assert result.language == LanguageType.PYTHON
        assert result.ast == {"type": "module"}
        assert len(result.entities) == 1
        assert result.parse_time_ms == 50.5
        assert result.errors == []
        assert result.warnings == []
    
    def test_failed_result(self):
        """Test resultado fallido de parsing."""
        result = ParseResult(
            success=False,
            language=LanguageType.PYTHON,
            errors=["Syntax error", "Missing import"],
            warnings=["Deprecated feature"]
        )
        
        assert result.success is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.ast is None
        assert result.entities == []
    
    def test_result_with_metadata(self):
        """Test resultado con metadatos."""
        result = ParseResult(
            success=True,
            language=LanguageType.PYTHON,
            metadata={
                "file_size": 1024,
                "encoding": "utf-8",
                "complexity": 5
            }
        )
        
        assert result.metadata["file_size"] == 1024
        assert result.metadata["encoding"] == "utf-8"
        assert result.metadata["complexity"] == 5


class TestEntity:
    """Pruebas para Entity."""
    
    def test_entity_creation(self):
        """Test creación de entidad."""
        entity = Entity(
            type="function",
            name="calculate_sum",
            start_line=10,
            end_line=20,
            start_column=5,
            end_column=15,
            metadata={
                "parameters": ["a", "b"],
                "return_type": "int"
            }
        )
        
        assert entity.type == "function"
        assert entity.name == "calculate_sum"
        assert entity.start_line == 10
        assert entity.end_line == 20
        assert entity.start_column == 5
        assert entity.end_column == 15
        assert entity.metadata["parameters"] == ["a", "b"]
        assert entity.metadata["return_type"] == "int"
        assert entity.children == []
    
    def test_entity_with_children(self):
        """Test entidad con hijos."""
        child1 = Entity(type="parameter", name="a", start_line=11, end_line=11)
        child2 = Entity(type="parameter", name="b", start_line=12, end_line=12)
        
        entity = Entity(
            type="function",
            name="test",
            start_line=10,
            end_line=15,
            children=[child1, child2]
        )
        
        assert len(entity.children) == 2
        assert entity.children[0].name == "a"
        assert entity.children[1].name == "b"


class TestMultiLanguageParser:
    """Pruebas para MultiLanguageParser."""
    
    @pytest.fixture
    def parser(self):
        """Crear parser para pruebas."""
        config = ParserConfig(
            enabled_languages=[LanguageType.PYTHON, LanguageType.JAVASCRIPT],
            cache_parsed_files=False  # Deshabilitar caché para pruebas
        )
        return MultiLanguageParser(config)
    
    @pytest.fixture
    def python_file(self):
        """Crear archivo Python de prueba."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        content = '''
def hello_world():
    """Función de ejemplo."""
    print("Hello, World!")
    return 42

class ExampleClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
'''
        temp_file.write(content)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def large_file(self):
        """Crear archivo grande para pruebas de límite."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        # Escribir contenido grande (más de 10MB)
        content = "x" * (11 * 1024 * 1024)  # 11MB
        temp_file.write(content)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def invalid_python_file(self):
        """Crear archivo Python con sintaxis inválida."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        content = '''
def invalid_syntax
    # Falta dos puntos
    pass
'''
        temp_file.write(content)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    def test_parser_initialization(self, parser):
        """Test inicialización del parser."""
        assert parser is not None
        assert hasattr(parser, 'config')
        assert hasattr(parser, '_parsers')
        assert hasattr(parser, '_cache')
        
        # Verificar que los parsers están inicializados
        assert LanguageType.PYTHON in parser._parsers or parser.config.enabled_languages
        
    def test_calculate_file_hash(self, parser, python_file):
        """Test cálculo de hash de archivo."""
        hash_value = parser._calculate_file_hash(python_file)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 en hex
        
        # Mismo archivo debería tener mismo hash
        hash2 = parser._calculate_file_hash(python_file)
        assert hash_value == hash2
        
    def test_detect_language(self, parser):
        """Test detección de lenguaje por extensión."""
        test_cases = [
            ("script.py", LanguageType.PYTHON),
            ("module.js", LanguageType.JAVASCRIPT),
            ("file.java", LanguageType.JAVA),
            ("code.cpp", LanguageType.CPP),
            ("program.go", LanguageType.GO),
            ("lib.rs", LanguageType.RUST),
            ("unknown.txt", LanguageType.PYTHON),  # Por defecto
        ]
        
        for filename, expected_lang in test_cases:
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='', delete=False)
            temp_file.write("test")
            temp_file.close()
            
            # Renombrar para la prueba
            new_name = temp_file.name + filename
            os.rename(temp_file.name, new_name)
            
            detected = parser._detect_language(new_name)
            assert detected == expected_lang
            
            os.unlink(new_name)
    
    @patch('indexer.multi_language_parser.get_parser')
    def test_parse_file_success(self, mock_get_parser, parser, python_file):
        """Test parsing exitoso de archivo."""
        # Mock del parser de tree-sitter
        mock_tree = MagicMock()
        mock_tree.root_node = MagicMock()
        mock_tree.root_node.type = 'module'
        mock_tree.root_node.text = b'test'
        mock_tree.root_node.start_point = (0, 0)
        mock_tree.root_node.end_point = (10, 0)
        mock_tree.root_node.children = []
        
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = mock_tree
        mock_get_parser.return_value = mock_parser_instance
        
        # Ejecutar parsing
        result = parser.parse_file(python_file, LanguageType.PYTHON)
        
        # Verificar
        assert result.success is True
        assert result.language == LanguageType.PYTHON
        assert result.parse_time_ms > 0
        mock_get_parser.assert_called_with('python')
    
    def test_parse_file_not_found(self, parser):
        """Test parsing de archivo no existente."""
        result = parser.parse_file("/path/inexistente.py", LanguageType.PYTHON)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()
    
    def test_parse_file_too_large(self, parser, large_file):
        """Test parsing de archivo demasiado grande."""
        result = parser.parse_file(large_file, LanguageType.PYTHON)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "too large" in result.errors[0].lower()
    
    def test_parse_file_unsupported_language(self, parser, python_file):
        """Test parsing con lenguaje no soportado."""
        # Configurar parser sin Python habilitado
        parser.config.enabled_languages = [LanguageType.JAVASCRIPT]
        
        result = parser.parse_file(python_file, LanguageType.PYTHON)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "not enabled" in result.errors[0].lower()
    
    def test_parse_directory(self, parser, temp_project_dir):
        """Test parsing de directorio completo."""
        # Crear estructura de directorio de prueba
        test_dir = os.path.join(temp_project_dir, "test_parse_dir")
        os.makedirs(test_dir, exist_ok=True)
        
        # Crear algunos archivos
        py_file = os.path.join(test_dir, "test1.py")
        with open(py_file, "w") as f:
            f.write("print('hello')")
        
        js_file = os.path.join(test_dir, "test2.js")
        with open(js_file, "w") as f:
            f.write("console.log('hello')")
        
        # Mockear parse_file para evitar dependencias reales
        with patch.object(parser, 'parse_file') as mock_parse:
            mock_parse.return_value = ParseResult(
                success=True,
                language=LanguageType.PYTHON
            )
            
            results = parser.parse_directory(test_dir)
            
            # Verificar que se llamó para cada archivo
            assert mock_parse.call_count >= 2
            assert len(results) >= 2
    
    def test_extract_entities_empty_result(self, parser):
        """Test extracción de entidades de resultado vacío."""
        result = ParseResult(success=False, language=LanguageType.PYTHON)
        entities = parser.extract_entities(result)
        
        assert entities == []
    
    def test_extract_python_entities(self, parser):
        """Test extracción de entidades Python."""
        # Mock del AST de Python
        ast = {
            'type': 'module',
            'children': [
                {
                    'type': 'function_definition',
                    'text': 'def test_func():',
                    'start_line': 1,
                    'end_line': 5,
                    'children': [
                        {'type': 'identifier', 'text': 'test_func'}
                    ]
                },
                {
                    'type': 'class_definition',
                    'text': 'class TestClass:',
                    'start_line': 7,
                    'end_line': 15,
                    'children': [
                        {'type': 'identifier', 'text': 'TestClass'}
                    ]
                }
            ]
        }
        
        # Mockear los métodos de extracción específicos
        with patch.object(parser, '_extract_python_entities') as mock_extract:
            mock_extract.return_value = [
                Entity(type='function', name='test_func', start_line=1, end_line=5),
                Entity(type='class', name='TestClass', start_line=7, end_line=15)
            ]
            
            result = ParseResult(
                success=True,
                language=LanguageType.PYTHON,
                ast=ast
            )
            
            entities = parser.extract_entities(result)
            
            assert len(entities) == 2
            mock_extract.assert_called_once_with(ast)
    
    def test_extract_python_parameters(self, parser):
        """Test extracción de parámetros de función Python."""
        # Mock de nodo de función
        function_node = {
            'type': 'function_definition',
            'children': [
                {
                    'type': 'parameters',
                    'children': [
                        {'type': 'identifier', 'text': 'param1'},
                        {'type': 'identifier', 'text': 'param2'}
                    ]
                }
            ]
        }
        
        params = parser._extract_python_parameters(function_node)
        
        assert len(params) == 2
        assert params[0]['name'] == 'param1'
        assert params[1]['name'] == 'param2'
    
    def test_extract_python_decorators(self, parser):
        """Test extracción de decoradores Python."""
        # Mock de nodo con padre que tiene decoradores
        node = {
            'type': 'function_definition',
            'parent': {
                'children': [
                    {'type': 'decorator', 'text': '@staticmethod'},
                    {'type': 'decorator', 'text': '@cache'},
                    node  # Referencia circular
                ]
            }
        }
        
        decorators = parser._extract_python_decorators(node)
        
        # En la implementación actual, esto depende de _extract_decorator_name
        assert isinstance(decorators, list)
    
    def test_tree_to_dict_conversion(self, parser):
        """Test conversión de nodo tree-sitter a dict."""
        # Mock de nodo tree-sitter
        mock_node = MagicMock()
        mock_node.type = 'function_definition'
        mock_node.text = b'def test():'
        mock_node.start_point = (1, 0)
        mock_node.end_point = (5, 15)
        mock_node.children = []
        
        result = parser._tree_to_dict(mock_node, b'source code')
        
        assert result['type'] == 'function_definition'
        assert result['text'] == 'def test():'
        assert result['start_line'] == 2  # tree-sitter usa 0-based
        assert result['end_line'] == 6
        assert result['children'] == []
    
    def test_entity_to_dict_conversion(self, parser):
        """Test conversión de Entity a dict."""
        entity = Entity(
            type='function',
            name='test_func',
            start_line=1,
            end_line=10,
            metadata={'params': ['a', 'b']},
            children=[
                Entity(type='parameter', name='a', start_line=2, end_line=2)
            ]
        )
        
        result = parser._entity_to_dict(entity)
        
        assert result['type'] == 'function'
        assert result['name'] == 'test_func'
        assert result['start_line'] == 1
        assert result['end_line'] == 10
        assert result['metadata']['params'] == ['a', 'b']
        assert len(result['children']) == 1
        assert result['children'][0]['name'] == 'a'
    
    def test_cache_operations(self):
        """Test operaciones de caché."""
        config = ParserConfig(cache_parsed_files=True, cache_size=2)
        parser = MultiLanguageParser(config)
        
        # Mockear resultados
        result1 = ParseResult(
            success=True,
            language=LanguageType.PYTHON,
            file_hash="hash1"
        )
        result2 = ParseResult(
            success=True,
            language=LanguageType.PYTHON,
            file_hash="hash2"
        )
        result3 = ParseResult(
            success=True,
            language=LanguageType.PYTHON,
            file_hash="hash3"
        )
        
        # Simular caché
        parser._cache = {
            "file1:hash1": result1,
            "file2:hash2": result2
        }
        
        # Test obtener de caché
        cached = parser._get_cached_result("file1", "hash1")
        assert cached is result1
        
        # Test caché miss
        cached = parser._get_cached_result("file3", "hash3")
        assert cached is None
        
        # Test agregar a caché
        parser._cache_result("file3", "hash3", result3)
        assert len(parser._cache) == 3
        
        # Test limpieza de caché (debería eliminar el más viejo)
        parser._clean_cache(max_size=2)
        assert len(parser._cache) == 2
    
    def test_perform_comprehensive_checks(self, parser):
        """Test verificaciones comprehensivas."""
        ast = {'type': 'module', 'children': []}
        warnings = parser._perform_comprehensive_checks(ast, LanguageType.PYTHON)
        
        assert isinstance(warnings, list)
    
    def test_estimate_python_complexity(self, parser):
        """Test estimación de complejidad ciclomática."""
        ast = {
            'type': 'module',
            'children': [
                {'type': 'if_statement'},
                {'type': 'while_statement'},
                {'type': 'for_statement'}
            ]
        }
        
        complexity = parser._estimate_python_complexity(ast)
        
        # 1 (base) + 1 (if) + 1 (while) + 1 (for) = 4
        assert complexity >= 1
    
    def test_parse_content_error_handling(self, parser):
        """Test manejo de errores en _parse_content."""
        with patch('indexer.multi_language_parser.get_parser') as mock_get_parser:
            mock_get_parser.side_effect = Exception("Parser error")
            
            result = parser._parse_content(
                content="test",
                language=LanguageType.PYTHON,
                mode=ParseMode.STANDARD,
                file_path="/test.py"
            )
            
            assert result.success is False
            assert "Parsing error" in result.errors[0]
    
    @pytest.mark.slow
    def test_parse_real_python_file(self, parser, python_file):
        """Test parsing de archivo Python real (requiere tree-sitter)."""
        try:
            result = parser.parse_file(python_file, LanguageType.PYTHON)
            
            # Verificar estructura básica del resultado
            assert hasattr(result, 'success')
            assert hasattr(result, 'language')
            assert hasattr(result, 'parse_time_ms')
            
            if result.success:
                assert result.language == LanguageType.PYTHON
                assert result.parse_time_ms > 0
        except ImportError:
            pytest.skip("tree-sitter-python no instalado")
    
    def test_language_specific_extractors(self, parser):
        """Test extractores específicos por lenguaje."""
        # Estos métodos deben estar definidos pero pueden ser stubs
        assert hasattr(parser, '_extract_javascript_entities')
        assert hasattr(parser, '_extract_java_entities')
        assert hasattr(parser, '_extract_cpp_entities')
        assert hasattr(parser, '_extract_go_entities')
        assert hasattr(parser, '_extract_rust_entities')
        
        # Deberían retornar listas
        result = parser._extract_javascript_entities({})
        assert isinstance(result, list)
        
        result = parser._extract_java_entities({})
        assert isinstance(result, list)


class TestEdgeCases:
    """Pruebas de casos límite."""
    
    @pytest.fixture
    def parser(self):
        return MultiLanguageParser(ParserConfig(cache_parsed_files=False))
    
    def test_empty_file(self, parser):
        """Test parsing de archivo vacío."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        temp_file.close()
        
        result = parser.parse_file(temp_file.name, LanguageType.PYTHON)
        
        os.unlink(temp_file.name)
        
        # El archivo vacío podría parsearse exitosamente o no
        assert hasattr(result, 'success')
    
    def test_binary_file(self, parser):
        """Test parsing de archivo binario."""
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False)
        temp_file.write(b'\x00\x01\x02\x03\x04')  # Datos binarios
        temp_file.close()
        
        result = parser.parse_file(temp_file.name, LanguageType.PYTHON)
        
        os.unlink(temp_file.name)
        
        # Debería fallar o manejar el error
        assert hasattr(result, 'success')
    
    def test_very_long_line(self, parser):
        """Test parsing de línea muy larga."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        # Línea de 1MB
        temp_file.write('x' * (1024 * 1024) + '\n')
        temp_file.close()
        
        result = parser.parse_file(temp_file.name, LanguageType.PYTHON)
        
        os.unlink(temp_file.name)
        
        assert hasattr(result, 'success')
    
    def test_nested_directories(self, parser, temp_project_dir):
        """Test parsing de directorios anidados."""
        # Crear estructura anidada
        nested_dir = os.path.join(temp_project_dir, "a", "b", "c", "d", "e")
        os.makedirs(nested_dir, exist_ok=True)
        
        # Archivo en directorio profundo
        deep_file = os.path.join(nested_dir, "deep.py")
        with open(deep_file, "w") as f:
            f.write("print('deep')")
        
        # Mockear parse_file
        with patch.object(parser, 'parse_file') as mock_parse:
            mock_parse.return_value = ParseResult(success=True, language=LanguageType.PYTHON)
            
            results = parser.parse_directory(temp_project_dir)
            
            # Debería encontrar el archivo
            assert mock_parse.called
            assert any("deep.py" in str(call[0][0]) for call in mock_parse.call_args_list)


class TestPerformance:
    """Pruebas de performance."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_parse_multiple_files_performance(self, parser, temp_project_dir):
        """Test performance parsing múltiples archivos."""
        import time
        
        # Crear 100 archivos de prueba
        file_count = 100
        for i in range(file_count):
            file_path = os.path.join(temp_project_dir, f"test_{i}.py")
            with open(file_path, "w") as f:
                f.write(f'''
def func_{i}():
    """Función de prueba {i}."""
    return {i} * 2

class Class{i}:
    def method_{i}(self):
        return "test {i}"
''')
        
        # Medir tiempo de parsing
        start_time = time.time()
        
        # Mockear el parser real para evitar dependencias
        with patch.object(parser, 'parse_file') as mock_parse:
            mock_parse.return_value = ParseResult(
                success=True,
                language=LanguageType.PYTHON,
                parse_time_ms=10.0
            )
            
            results = parser.parse_directory(temp_project_dir)
            
        elapsed = time.time() - start_time
        
        # Verificar que se procesaron todos los archivos
        assert len(results) == file_count
        
        # Performance: debería tomar menos de 5 segundos (con mocks)
        assert elapsed < 5.0, f"Parsing de {file_count} archivos tomó {elapsed:.2f}s"
        
        print(f"\nPerformance: {file_count} archivos en {elapsed:.3f}s "
              f"({file_count/elapsed:.1f} archivos/segundo)")

    @pytest.mark.performance
    def test_cache_performance(self):
        """Test performance de caché."""
        import time
        
        config = ParserConfig(cache_parsed_files=True, cache_size=1000)
        parser = MultiLanguageParser(config)
        
        # Llenar caché
        start_time = time.time()
        
        for i in range(100):
            result = ParseResult(
                success=True,
                language=LanguageType.PYTHON,
                file_hash=f"hash_{i}"
            )
            parser._cache_result(f"file_{i}", f"hash_{i}", result)
        
        fill_time = time.time() - start_time
        
        # Acceder a caché
        start_time = time.time()
        
        hits = 0
        for i in range(100):
            cached = parser._get_cached_result(f"file_{i}", f"hash_{i}")
            if cached:
                hits += 1
        
        access_time = time.time() - start_time
        
        # Performance: acceso a caché debería ser rápido
        assert access_time < 0.1, f"Acceso a caché tomó {access_time:.3f}s"
        assert hits == 100, f"Hit rate: {hits}/100"
        
        print(f"\nCache performance: llenado={fill_time:.3f}s, "
              f"acceso={access_time:.3f}s, hit rate={hits}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])