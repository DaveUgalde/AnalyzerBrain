"""
Módulo MultiLanguageParser - Parser para múltiples lenguajes de programación
"""

import ast
import re
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import javalang  # pip install javalang
import parso  # pip install parso
from lxml import etree

@dataclass
class ParseResult:
    """Resultado del parsing de un archivo"""
    success: bool
    language: str
    ast: Optional[Any] = None
    syntax_tree: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class CodeElement:
    """Elemento de código extraído"""
    type: str  # 'class', 'function', 'variable', 'import', etc.
    name: str
    line_start: int
    line_end: int
    parent: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

class MultiLanguageParser:
    """Parser multi-lenguaje para análisis de código fuente"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar parser multi-lenguaje"""
        self.config = config or {}
        self.supported_languages = {
            'python': self._parse_python,
            'java': self._parse_java,
            'javascript': self._parse_javascript,
            'typescript': self._parse_typescript,
            'cpp': self._parse_cpp,
            'c': self._parse_c,
            'html': self._parse_html,
            'css': self._parse_css,
            'json': self._parse_json,
            'xml': self._parse_xml,
            'yaml': self._parse_yaml,
            'markdown': self._parse_markdown,
        }
        
    def parse(self, content: str, language: str, file_path: Optional[str] = None) -> ParseResult:
        """
        Parsear contenido de código según el lenguaje
        
        Args:
            content: Contenido del archivo
            language: Lenguaje de programación
            file_path: Ruta del archivo (opcional)
            
        Returns:
            ParseResult: Resultado del parsing
        """
        if language not in self.supported_languages:
            return ParseResult(
                success=False,
                language=language,
                errors=[f"Lenguaje no soportado: {language}"]
            )
        
        try:
            parser_func = self.supported_languages[language]
            return parser_func(content, file_path)
        except Exception as e:
            return ParseResult(
                success=False,
                language=language,
                errors=[f"Error parsing {language}: {str(e)}"]
            )
    
    def _parse_python(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear código Python"""
        try:
            # Usar ast de Python estándar
            tree = ast.parse(content, filename=file_path or '<string>')
            
            # Convertir AST a diccionario
            syntax_tree = self._python_ast_to_dict(tree)
            
            # Extraer elementos
            elements = self._extract_python_elements(tree)
            
            return ParseResult(
                success=True,
                language='python',
                ast=tree,
                syntax_tree={
                    'type': 'module',
                    'body': syntax_tree,
                    'elements': elements
                }
            )
        except SyntaxError as e:
            return ParseResult(
                success=False,
                language='python',
                errors=[f"Syntax error at line {e.lineno}: {e.msg}"]
            )
    
    def _python_ast_to_dict(self, node) -> Dict[str, Any]:
        """Convertir AST de Python a diccionario"""
        if isinstance(node, ast.AST):
            result = {'type': type(node).__name__}
            
            for field in node._fields:
                value = getattr(node, field)
                
                if isinstance(value, list):
                    result[field] = [self._python_ast_to_dict(item) for item in value]
                elif isinstance(value, ast.AST):
                    result[field] = self._python_ast_to_dict(value)
                else:
                    result[field] = value
            
            # Agregar posición si está disponible
            if hasattr(node, 'lineno'):
                result['lineno'] = node.lineno
            if hasattr(node, 'col_offset'):
                result['col_offset'] = node.col_offset
            
            return result
        elif isinstance(node, list):
            return [self._python_ast_to_dict(item) for item in node]
        else:
            return node
    
    def _extract_python_elements(self, tree) -> List[Dict[str, Any]]:
        """Extraer elementos de código Python"""
        elements = []
        
        class ElementVisitor(ast.NodeVisitor):
            def __init__(self):
                self.elements = []
                self.current_class = None
            
            def visit_ClassDef(self, node):
                element = {
                    'type': 'class',
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno or node.lineno,
                    'bases': [self._get_name(base) for base in node.bases],
                    'methods': [],
                    'decorators': [self._get_name(decorator) for decorator in node.decorator_list]
                }
                self.current_class = node.name
                self.elements.append(element)
                self.generic_visit(node)
                self.current_class = None
            
            def visit_FunctionDef(self, node):
                element = {
                    'type': 'function',
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno or node.lineno,
                    'parent': self.current_class,
                    'args': [arg.arg for arg in node.args.args],
                    'decorators': [self._get_name(decorator) for decorator in node.decorator_list]
                }
                self.elements.append(element)
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    element = {
                        'type': 'import',
                        'name': alias.name,
                        'line_start': node.lineno,
                        'alias': alias.asname
                    }
                    self.elements.append(element)
            
            def visit_ImportFrom(self, node):
                for alias in node.names:
                    element = {
                        'type': 'import_from',
                        'module': node.module or '',
                        'name': alias.name,
                        'line_start': node.lineno,
                        'alias': alias.asname
                    }
                    self.elements.append(element)
            
            def _get_name(self, node):
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Attribute):
                    return f"{self._get_name(node.value)}.{node.attr}"
                return str(node)
        
        visitor = ElementVisitor()
        visitor.visit(tree)
        return visitor.elements
    
    def _parse_java(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear código Java"""
        try:
            tree = javalang.parse.parse(content)
            
            # Extraer elementos
            elements = self._extract_java_elements(tree)
            
            # Convertir a estructura de árbol
            syntax_tree = self._java_tree_to_dict(tree)
            
            return ParseResult(
                success=True,
                language='java',
                ast=tree,
                syntax_tree={
                    'type': 'compilation_unit',
                    'body': syntax_tree,
                    'elements': elements
                }
            )
        except javalang.parser.JavaSyntaxError as e:
            return ParseResult(
                success=False,
                language='java',
                errors=[f"Java syntax error at position {e.at.position}: {e.description}"]
            )
    
    def _java_tree_to_dict(self, node) -> Dict[str, Any]:
        """Convertir árbol Java a diccionario"""
        if hasattr(node, '_attrs'):
            result = {'type': type(node).__name__}
            
            for attr in node._attrs:
                value = getattr(node, attr)
                
                if isinstance(value, list):
                    result[attr] = [self._java_tree_to_dict(item) for item in value]
                elif hasattr(value, '_attrs'):
                    result[attr] = self._java_tree_to_dict(value)
                else:
                    result[attr] = value
            
            return result
        elif isinstance(node, list):
            return [self._java_tree_to_dict(item) for item in node]
        else:
            return node
    
    def _extract_java_elements(self, tree) -> List[Dict[str, Any]]:
        """Extraer elementos de código Java"""
        elements = []
        
        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                element = {
                    'type': 'class',
                    'name': node.name,
                    'line_start': node.position.line if node.position else 0,
                    'modifiers': node.modifiers,
                    'extends': node.extends.name if node.extends else None,
                    'implements': [impl.name for impl in node.implements] if node.implements else []
                }
                elements.append(element)
            
            elif isinstance(node, javalang.tree.MethodDeclaration):
                element = {
                    'type': 'method',
                    'name': node.name,
                    'line_start': node.position.line if node.position else 0,
                    'modifiers': node.modifiers,
                    'return_type': str(node.return_type) if node.return_type else 'void',
                    'parameters': [{
                        'type': str(param.type),
                        'name': param.name
                    } for param in node.parameters] if node.parameters else []
                }
                elements.append(element)
            
            elif isinstance(node, javalang.tree.Import):
                element = {
                    'type': 'import',
                    'name': node.path,
                    'static': node.static,
                    'wildcard': node.wildcard
                }
                elements.append(element)
        
        return elements
    
    def _parse_javascript(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear código JavaScript"""
        try:
            # Usar parso para JavaScript (aunque es principalmente para Python, puede manejar JS)
            import parso
            tree = parso.parse(content)
            
            elements = self._extract_javascript_elements(content)
            
            return ParseResult(
                success=True,
                language='javascript',
                ast=tree,
                syntax_tree={
                    'type': 'script',
                    'elements': elements
                }
            )
        except Exception as e:
            return ParseResult(
                success=False,
                language='javascript',
                errors=[f"JavaScript parsing error: {str(e)}"],
                warnings=["Parsing de JavaScript limitado - considere usar un parser dedicado"]
            )
    
    def _extract_javascript_elements(self, content: str) -> List[Dict[str, Any]]:
        """Extraer elementos de código JavaScript usando regex"""
        elements = []
        
        # Buscar funciones
        function_patterns = [
            r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)',
            r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
            r'let\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
            r'var\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
            r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*{',
        ]
        
        for i, line in enumerate(content.splitlines(), 1):
            line_stripped = line.strip()
            
            # Buscar import/require
            if line_stripped.startswith('import ') or 'require(' in line:
                elements.append({
                    'type': 'import',
                    'line': i,
                    'content': line_stripped[:100]
                })
            
            # Buscar funciones
            for pattern in function_patterns:
                match = re.search(pattern, line_stripped)
                if match:
                    func_name = match.group(1)
                    elements.append({
                        'type': 'function',
                        'name': func_name,
                        'line_start': i
                    })
        
        return elements
    
    def _parse_typescript(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear código TypeScript"""
        # Por ahora, usar el mismo parser que JavaScript
        result = self._parse_javascript(content, file_path)
        result.language = 'typescript'
        return result
    
    def _parse_cpp(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear código C++"""
        return self._parse_c_like(content, 'cpp', file_path)
    
    def _parse_c(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear código C"""
        return self._parse_c_like(content, 'c', file_path)
    
    def _parse_c_like(self, content: str, language: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear código C/C++"""
        elements = []
        
        # Patrones básicos para C/C++
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Incluir directivas
            if line_stripped.startswith('#'):
                elements.append({
                    'type': 'preprocessor',
                    'directive': line_stripped.split()[0] if line_stripped else '',
                    'line': i,
                    'content': line_stripped
                })
            
            # Buscar funciones
            elif re.match(r'^\w+\s+\w+\s*\([^)]*\)\s*(?:const)?\s*{', line_stripped):
                # Patrón simple para funciones
                match = re.match(r'^(\w+)\s+(\w+)\s*\([^)]*\)', line_stripped)
                if match:
                    elements.append({
                        'type': 'function',
                        'return_type': match.group(1),
                        'name': match.group(2),
                        'line_start': i
                    })
        
        return ParseResult(
            success=True,
            language=language,
            syntax_tree={
                'type': 'translation_unit',
                'elements': elements
            }
        )
    
    def _parse_html(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear HTML"""
        try:
            # Usar lxml para HTML
            parser = etree.HTMLParser()
            tree = etree.fromstring(content, parser)
            
            elements = self._extract_html_elements(tree)
            
            return ParseResult(
                success=True,
                language='html',
                ast=tree,
                syntax_tree={
                    'type': 'document',
                    'elements': elements
                }
            )
        except Exception as e:
            return ParseResult(
                success=False,
                language='html',
                errors=[f"HTML parsing error: {str(e)}"]
            )
    
    def _extract_html_elements(self, element, depth=0) -> List[Dict[str, Any]]:
        """Extraer elementos HTML recursivamente"""
        elements = []
        
        element_info = {
            'type': 'tag',
            'name': element.tag,
            'attributes': dict(element.attrib),
            'depth': depth
        }
        elements.append(element_info)
        
        for child in element:
            elements.extend(self._extract_html_elements(child, depth + 1))
        
        return elements
    
    def _parse_css(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear CSS"""
        elements = []
        
        # Patrones básicos para CSS
        lines = content.splitlines()
        in_rule = False
        current_rule = None
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Buscar reglas CSS
            if line_stripped.endswith('{'):
                selector = line_stripped[:-1].strip()
                current_rule = {
                    'type': 'rule',
                    'selector': selector,
                    'line_start': i,
                    'properties': []
                }
                elements.append(current_rule)
                in_rule = True
            
            elif line_stripped == '}':
                if current_rule:
                    current_rule['line_end'] = i
                in_rule = False
                current_rule = None
            
            elif in_rule and ':' in line_stripped and current_rule:
                parts = line_stripped.split(':', 1)
                if len(parts) == 2:
                    prop = parts[0].strip()
                    value = parts[1].rstrip(';').strip()
                    current_rule['properties'].append({
                        'property': prop,
                        'value': value,
                        'line': i
                    })
        
        return ParseResult(
            success=True,
            language='css',
            syntax_tree={
                'type': 'stylesheet',
                'elements': elements
            }
        )
    
    def _parse_json(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear JSON"""
        try:
            data = json.loads(content)
            
            elements = self._extract_json_elements(data, 'root')
            
            return ParseResult(
                success=True,
                language='json',
                ast=data,
                syntax_tree={
                    'type': 'document',
                    'elements': elements
                }
            )
        except json.JSONDecodeError as e:
            return ParseResult(
                success=False,
                language='json',
                errors=[f"JSON decoding error at line {e.lineno}: {e.msg}"]
            )
    
    def _extract_json_elements(self, data, parent_key: str) -> List[Dict[str, Any]]:
        """Extraer elementos JSON recursivamente"""
        elements = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                element = {
                    'type': 'object_property',
                    'key': key,
                    'value_type': type(value).__name__,
                    'parent': parent_key
                }
                
                if isinstance(value, (dict, list)):
                    element['children'] = self._extract_json_elements(value, key)
                else:
                    element['value'] = str(value)[:100]  # Limitar longitud
                
                elements.append(element)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                element = {
                    'type': 'array_item',
                    'index': i,
                    'value_type': type(item).__name__,
                    'parent': parent_key
                }
                
                if isinstance(item, (dict, list)):
                    element['children'] = self._extract_json_elements(item, f"{parent_key}[{i}]")
                else:
                    element['value'] = str(item)[:100]
                
                elements.append(element)
        
        return elements
    
    def _parse_xml(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear XML"""
        try:
            root = etree.fromstring(content.encode('utf-8'))
            
            elements = self._extract_xml_elements(root, 0)
            
            return ParseResult(
                success=True,
                language='xml',
                ast=root,
                syntax_tree={
                    'type': 'document',
                    'elements': elements
                }
            )
        except Exception as e:
            return ParseResult(
                success=False,
                language='xml',
                errors=[f"XML parsing error: {str(e)}"]
            )
    
    def _extract_xml_elements(self, element, depth: int) -> List[Dict[str, Any]]:
        """Extraer elementos XML recursivamente"""
        elements = []
        
        element_info = {
            'type': 'element',
            'name': element.tag,
            'depth': depth,
            'attributes': dict(element.attrib)
        }
        
        # Agregar texto si existe
        if element.text and element.text.strip():
            element_info['text'] = element.text.strip()[:200]
        
        elements.append(element_info)
        
        for child in element:
            elements.extend(self._extract_xml_elements(child, depth + 1))
        
        return elements
    
    def _parse_yaml(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear YAML"""
        try:
            import yaml
            data = yaml.safe_load(content)
            
            if data is None:
                data = {}
            
            elements = self._extract_yaml_elements(data, 'root')
            
            return ParseResult(
                success=True,
                language='yaml',
                ast=data,
                syntax_tree={
                    'type': 'document',
                    'elements': elements
                }
            )
        except Exception as e:
            return ParseResult(
                success=False,
                language='yaml',
                errors=[f"YAML parsing error: {str(e)}"]
            )
    
    def _extract_yaml_elements(self, data, parent_key: str) -> List[Dict[str, Any]]:
        """Extraer elementos YAML (similar a JSON)"""
        return self._extract_json_elements(data, parent_key)
    
    def _parse_markdown(self, content: str, file_path: Optional[str] = None) -> ParseResult:
        """Parsear Markdown"""
        elements = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Detectar headers
            if line_stripped.startswith('#'):
                level = 0
                while level < len(line_stripped) and line_stripped[level] == '#':
                    level += 1
                
                title = line_stripped[level:].strip()
                elements.append({
                    'type': 'header',
                    'level': level,
                    'title': title,
                    'line': i
                })
            
            # Detectar listas
            elif line_stripped.startswith(('-', '*', '+')):
                elements.append({
                    'type': 'list_item',
                    'content': line_stripped[1:].strip(),
                    'line': i
                })
            
            # Detectar código
            elif line_stripped.startswith('```') or line_stripped.startswith('~~~'):
                elements.append({
                    'type': 'code_block',
                    'line': i
                })
        
        return ParseResult(
            success=True,
            language='markdown',
            syntax_tree={
                'type': 'document',
                'elements': elements
            }
        )
    
    def get_code_elements(self, parse_result: ParseResult) -> List[CodeElement]:
        """
        Extraer elementos de código estandarizados del resultado del parsing
        
        Args:
            parse_result: Resultado del parsing
            
        Returns:
            List[CodeElement]: Elementos de código estandarizados
        """
        elements = []
        
        if not parse_result.success or not parse_result.syntax_tree:
            return elements
        
        syntax_tree = parse_result.syntax_tree
        
        if parse_result.language == 'python' and 'elements' in syntax_tree:
            for element_data in syntax_tree['elements']:
                element = CodeElement(
                    type=element_data.get('type', 'unknown'),
                    name=element_data.get('name', ''),
                    line_start=element_data.get('line_start', 0),
                    line_end=element_data.get('line_end', element_data.get('line_start', 0)),
                    parent=element_data.get('parent'),
                    attributes=element_data
                )
                elements.append(element)
        
        elif parse_result.language == 'java' and 'elements' in syntax_tree:
            for element_data in syntax_tree['elements']:
                element = CodeElement(
                    type=element_data.get('type', 'unknown'),
                    name=element_data.get('name', ''),
                    line_start=element_data.get('line_start', 0),
                    line_end=element_data.get('line_end', element_data.get('line_start', 0)),
                    attributes=element_data
                )
                elements.append(element)
        
        # Para otros lenguajes, adaptar según sea necesario
        
        return elements
    
    def detect_language(self, file_path: str, content: Optional[str] = None) -> str:
        """
        Detectar lenguaje de programación basado en extensión y contenido
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo (opcional)
            
        Returns:
            str: Lenguaje detectado
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        # Mapeo de extensiones a lenguajes
        extension_map = {
            '.py': 'python',
            '.java': 'java',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.rs': 'rust',
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.less': 'less',
            '.xml': 'xml',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.md': 'markdown',
            '.sql': 'sql',
            '.sh': 'bash',
            '.bat': 'batch',
            '.ps1': 'powershell',
        }
        
        # Primero por extensión
        if ext in extension_map:
            return extension_map[ext]
        
        # Si no se detecta por extensión, intentar por contenido
        if content:
            # Detectar shebang para scripts
            if content.startswith('#!'):
                if 'python' in content[:50]:
                    return 'python'
                elif 'bash' in content[:50] or 'sh' in content[:50]:
                    return 'bash'
                elif 'node' in content[:50]:
                    return 'javascript'
            
            # Detectar por patrones en el contenido
            if '<?php' in content[:100]:
                return 'php'
            elif '<!DOCTYPE html>' in content[:100] or '<html' in content[:100]:
                return 'html'
            elif 'package ' in content[:100] and ';' in content[:100]:
                return 'java'
        
        return 'unknown'