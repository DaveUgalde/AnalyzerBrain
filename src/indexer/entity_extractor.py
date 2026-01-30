"""
Módulo EntityExtractor - Extracción de entidades de código
"""

import re
import ast
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import javalang

@dataclass
class CodeEntity:
    """Entidad de código identificada"""
    type: str  # 'class', 'function', 'variable', 'constant', 'import', etc.
    name: str
    file_path: str
    line_start: int
    line_end: int
    scope: str = 'global'
    parent: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    parameters: List[Dict[str, str]] = field(default_factory=list)
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EntityRelationships:
    """Relaciones entre entidades"""
    inheritance: List[Tuple[str, str]] = field(default_factory=list)  # (child, parent)
    composition: List[Tuple[str, str]] = field(default_factory=list)  # (owner, component)
    aggregation: List[Tuple[str, str]] = field(default_factory=list)  # (owner, aggregated)
    dependency: List[Tuple[str, str]] = field(default_factory=list)   # (dependent, dependency)
    association: List[Tuple[str, str]] = field(default_factory=list)  # (source, target)

class EntityExtractor:
    """Extractor de entidades de código fuente"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar extractor de entidades"""
        self.config = config or {}
        self.language_handlers = {
            'python': self._extract_python_entities,
            'java': self._extract_java_entities,
            'javascript': self._extract_javascript_entities,
            'typescript': self._extract_typescript_entities,
            'cpp': self._extract_cpp_entities,
            'c': self._extract_c_entities,
        }
        
    def extract_entities(self, file_path: str, content: str, language: str) -> List[CodeEntity]:
        """
        Extraer entidades de un archivo de código
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            language: Lenguaje de programación
            
        Returns:
            List[CodeEntity]: Lista de entidades extraídas
        """
        if language not in self.language_handlers:
            # Intento de extracción genérica
            return self._extract_generic_entities(file_path, content, language)
        
        handler = self.language_handlers[language]
        return handler(file_path, content)
    
    def _extract_python_entities(self, file_path: str, content: str) -> List[CodeEntity]:
        """Extraer entidades de código Python"""
        entities = []
        
        try:
            tree = ast.parse(content)
            
            class EntityVisitor(ast.NodeVisitor):
                def __init__(self, file_path):
                    self.file_path = file_path
                    self.entities = []
                    self.current_class = None
                    self.current_function = None
                    self.imports = {}
                
                def visit_Module(self, node):
                    # Extraer docstring del módulo
                    docstring = ast.get_docstring(node)
                    if docstring:
                        entity = CodeEntity(
                            type='module',
                            name=Path(self.file_path).stem,
                            file_path=self.file_path,
                            line_start=1,
                            line_end=node.end_lineno or 1,
                            docstring=docstring
                        )
                        self.entities.append(entity)
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    # Extraer información de la clase
                    docstring = ast.get_docstring(node)
                    
                    # Obtener nombres de clases base
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append(self._get_attribute_name(base))
                    
                    entity = CodeEntity(
                        type='class',
                        name=node.name,
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        parent=self.current_class,
                        modifiers=self._get_decorators(node),
                        docstring=docstring,
                        metrics={
                            'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                            'attributes': len([n for n in node.body if isinstance(n, ast.Assign)]),
                            'bases': bases
                        }
                    )
                    self.entities.append(entity)
                    
                    # Guardar contexto actual
                    previous_class = self.current_class
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = previous_class
                
                def visit_FunctionDef(self, node):
                    # Extraer información de la función/método
                    docstring = ast.get_docstring(node)
                    
                    # Extraer parámetros
                    parameters = []
                    for arg in node.args.args:
                        param_name = arg.arg
                        param_type = None
                        
                        # Intentar obtener anotaciones de tipo
                        if arg.annotation:
                            param_type = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                        
                        parameters.append({
                            'name': param_name,
                            'type': param_type
                        })
                    
                    # Determinar tipo (función vs método)
                    entity_type = 'method' if self.current_class else 'function'
                    
                    # Determinar tipo de retorno
                    return_type = None
                    if node.returns:
                        return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
                    
                    entity = CodeEntity(
                        type=entity_type,
                        name=node.name,
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        scope='class' if self.current_class else 'module',
                        parent=self.current_class,
                        modifiers=self._get_decorators(node),
                        return_type=return_type,
                        parameters=parameters,
                        docstring=docstring,
                        metrics={
                            'parameters_count': len(parameters),
                            'has_return': any(isinstance(n, ast.Return) for n in ast.walk(node))
                        }
                    )
                    self.entities.append(entity)
                    
                    # Guardar contexto actual
                    previous_function = self.current_function
                    self.current_function = node.name
                    self.generic_visit(node)
                    self.current_function = previous_function
                
                def visit_AsyncFunctionDef(self, node):
                    # Manejar funciones asíncronas igual que las normales
                    self.visit_FunctionDef(node)
                
                def visit_Import(self, node):
                    for alias in node.names:
                        entity = CodeEntity(
                            type='import',
                            name=alias.name,
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=node.lineno,
                            scope='module'
                        )
                        self.entities.append(entity)
                        self.imports[alias.asname or alias.name] = alias.name
                
                def visit_ImportFrom(self, node):
                    for alias in node.names:
                        full_name = f"{node.module or ''}.{alias.name}"
                        entity = CodeEntity(
                            type='import',
                            name=full_name,
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=node.lineno,
                            scope='module'
                        )
                        self.entities.append(entity)
                        self.imports[alias.asname or alias.name] = full_name
                
                def visit_Assign(self, node):
                    # Extraer variables/constantes
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Determinar si es constante (MAYÚSCULAS)
                            is_constant = target.id.isupper()
                            
                            entity = CodeEntity(
                                type='constant' if is_constant else 'variable',
                                name=target.id,
                                file_path=self.file_path,
                                line_start=node.lineno,
                                line_end=node.lineno,
                                scope='class' if self.current_class else 'function' if self.current_function else 'module',
                                parent=self.current_class or self.current_function
                            )
                            self.entities.append(entity)
                
                def _get_decorators(self, node):
                    """Obtener nombres de decoradores"""
                    decorators = []
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            decorators.append(decorator.id)
                        elif isinstance(decorator, ast.Attribute):
                            decorators.append(self._get_attribute_name(decorator))
                        elif isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Name):
                                decorators.append(decorator.func.id)
                    return decorators
                
                def _get_attribute_name(self, node):
                    """Obtener nombre completo de atributo"""
                    parts = []
                    while isinstance(node, ast.Attribute):
                        parts.append(node.attr)
                        node = node.value
                    if isinstance(node, ast.Name):
                        parts.append(node.id)
                    return '.'.join(reversed(parts))
            
            visitor = EntityVisitor(file_path)
            visitor.visit(tree)
            entities = visitor.entities
            
        except SyntaxError as e:
            # Si hay error de sintaxis, usar extracción basada en regex
            entities = self._extract_python_entities_regex(file_path, content)
        
        return entities
    
    def _extract_python_entities_regex(self, file_path: str, content: str) -> List[CodeEntity]:
        """Extraer entidades Python usando regex (fallback)"""
        entities = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Buscar clases
            class_match = re.match(r'^class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_stripped)
            if class_match:
                entity = CodeEntity(
                    type='class',
                    name=class_match.group(1),
                    file_path=file_path,
                    line_start=i,
                    line_end=i
                )
                entities.append(entity)
            
            # Buscar funciones
            elif re.match(r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_stripped):
                func_match = re.match(r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_stripped)
                if func_match:
                    entity = CodeEntity(
                        type='function',
                        name=func_match.group(1),
                        file_path=file_path,
                        line_start=i,
                        line_end=i
                    )
                    entities.append(entity)
            
            # Buscar imports
            elif line_stripped.startswith('import ') or line_stripped.startswith('from '):
                entity = CodeEntity(
                    type='import',
                    name=line_stripped,
                    file_path=file_path,
                    line_start=i,
                    line_end=i
                )
                entities.append(entity)
        
        return entities
    
    def _extract_java_entities(self, file_path: str, content: str) -> List[CodeEntity]:
        """Extraer entidades de código Java"""
        entities = []
        
        try:
            tree = javalang.parse.parse(content)
            
            # Recorrer el árbol sintáctico
            for path, node in tree:
                if isinstance(node, javalang.tree.PackageDeclaration):
                    # Paquete
                    entity = CodeEntity(
                        type='package',
                        name=node.name,
                        file_path=file_path,
                        line_start=node.position.line if node.position else 1,
                        line_end=node.position.line if node.position else 1
                    )
                    entities.append(entity)
                
                elif isinstance(node, javalang.tree.Import):
                    # Importación
                    entity = CodeEntity(
                        type='import',
                        name=node.path,
                        file_path=file_path,
                        line_start=node.position.line if node.position else 1,
                        line_end=node.position.line if node.position else 1,
                        attributes={
                            'static': node.static,
                            'wildcard': node.wildcard
                        }
                    )
                    entities.append(entity)
                
                elif isinstance(node, javalang.tree.ClassDeclaration):
                    # Clase
                    entity = CodeEntity(
                        type='class',
                        name=node.name,
                        file_path=file_path,
                        line_start=node.position.line if node.position else 1,
                        line_end=self._find_class_end(content, node.position.line if node.position else 1),
                        modifiers=node.modifiers,
                        metrics={
                            'extends': node.extends.name if node.extends else None,
                            'implements': [impl.name for impl in node.implements] if node.implements else []
                        }
                    )
                    entities.append(entity)
                
                elif isinstance(node, javalang.tree.InterfaceDeclaration):
                    # Interfaz
                    entity = CodeEntity(
                        type='interface',
                        name=node.name,
                        file_path=file_path,
                        line_start=node.position.line if node.position else 1,
                        line_end=self._find_class_end(content, node.position.line if node.position else 1),
                        modifiers=node.modifiers
                    )
                    entities.append(entity)
                
                elif isinstance(node, javalang.tree.MethodDeclaration):
                    # Método
                    return_type = str(node.return_type) if node.return_type else 'void'
                    
                    parameters = []
                    if node.parameters:
                        for param in node.parameters:
                            parameters.append({
                                'name': param.name,
                                'type': str(param.type)
                            })
                    
                    entity = CodeEntity(
                        type='method',
                        name=node.name,
                        file_path=file_path,
                        line_start=node.position.line if node.position else 1,
                        line_end=self._find_method_end(content, node.position.line if node.position else 1),
                        modifiers=node.modifiers,
                        return_type=return_type,
                        parameters=parameters
                    )
                    entities.append(entity)
                
                elif isinstance(node, javalang.tree.FieldDeclaration):
                    # Campo (variable de clase)
                    for declarator in node.declarators:
                        entity = CodeEntity(
                            type='field',
                            name=declarator.name,
                            file_path=file_path,
                            line_start=node.position.line if node.position else 1,
                            line_end=node.position.line if node.position else 1,
                            modifiers=node.modifiers,
                            return_type=str(node.type) if node.type else None
                        )
                        entities.append(entity)
        
        except javalang.parser.JavaSyntaxError as e:
            # Fallback a extracción regex
            entities = self._extract_java_entities_regex(file_path, content)
        
        return entities
    
    def _find_class_end(self, content: str, start_line: int) -> int:
        """Encontrar fin de clase en Java"""
        lines = content.splitlines()
        brace_count = 0
        in_class = False
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            
            if i == start_line - 1:
                # En la línea de inicio, buscar primera {
                if '{' in line:
                    brace_count = 1
                    in_class = True
                continue
            
            if in_class:
                brace_count += line.count('{')
                brace_count -= line.count('}')
                
                if brace_count == 0:
                    return i + 1
        
        return start_line
    
    def _find_method_end(self, content: str, start_line: int) -> int:
        """Encontrar fin de método en Java"""
        lines = content.splitlines()
        brace_count = 0
        in_method = False
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            
            if i == start_line - 1:
                # En la línea de inicio, buscar primera {
                if '{' in line:
                    brace_count = 1
                    in_method = True
                continue
            
            if in_method:
                brace_count += line.count('{')
                brace_count -= line.count('}')
                
                if brace_count == 0:
                    return i + 1
        
        return start_line
    
    def _extract_java_entities_regex(self, file_path: str, content: str) -> List[CodeEntity]:
        """Extraer entidades Java usando regex"""
        entities = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Buscar paquete
            if line_stripped.startswith('package '):
                entity = CodeEntity(
                    type='package',
                    name=line_stripped[8:].rstrip(';'),
                    file_path=file_path,
                    line_start=i,
                    line_end=i
                )
                entities.append(entity)
            
            # Buscar imports
            elif line_stripped.startswith('import '):
                entity = CodeEntity(
                    type='import',
                    name=line_stripped[7:].rstrip(';'),
                    file_path=file_path,
                    line_start=i,
                    line_end=i
                )
                entities.append(entity)
            
            # Buscar clases
            elif re.match(r'^(public|private|protected|abstract|final|static)?\s*class\s+', line_stripped):
                match = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_stripped)
                if match:
                    entity = CodeEntity(
                        type='class',
                        name=match.group(1),
                        file_path=file_path,
                        line_start=i,
                        line_end=i
                    )
                    entities.append(entity)
            
            # Buscar métodos
            elif re.match(r'^\s*(public|private|protected|static|final|abstract|synchronized|native)\s+', line_stripped):
                # Patrón simple para métodos
                match = re.search(r'\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line_stripped)
                if match and 'class' not in line_stripped and 'interface' not in line_stripped:
                    entity = CodeEntity(
                        type='method',
                        name=match.group(1),
                        file_path=file_path,
                        line_start=i,
                        line_end=i
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_javascript_entities(self, file_path: str, content: str) -> List[CodeEntity]:
        """Extraer entidades de código JavaScript"""
        entities = []
        lines = content.splitlines()
        
        current_class = None
        current_function = None
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Buscar imports
            if line_stripped.startswith('import ') or line_stripped.startswith('export '):
                entity = CodeEntity(
                    type='import',
                    name=line_stripped,
                    file_path=file_path,
                    line_start=i,
                    line_end=i
                )
                entities.append(entity)
            
            # Buscar clases (ES6)
            elif re.match(r'^(export\s+)?(default\s+)?class\s+', line_stripped):
                match = re.search(r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', line_stripped)
                if match:
                    current_class = match.group(1)
                    entity = CodeEntity(
                        type='class',
                        name=current_class,
                        file_path=file_path,
                        line_start=i,
                        line_end=i
                    )
                    entities.append(entity)
            
            # Buscar funciones
            elif re.match(r'^(export\s+)?(default\s+)?(async\s+)?(function\s+|const\s+|let\s+|var\s+)?', line_stripped):
                # Detectar diferentes tipos de funciones
                patterns = [
                    r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(',
                    r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
                    r'let\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
                    r'var\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
                    r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*{',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, line_stripped)
                    if match:
                        func_name = match.group(1)
                        current_function = func_name
                        
                        entity_type = 'method' if current_class else 'function'
                        
                        entity = CodeEntity(
                            type=entity_type,
                            name=func_name,
                            file_path=file_path,
                            line_start=i,
                            line_end=i,
                            parent=current_class
                        )
                        entities.append(entity)
                        break
        
        return entities
    
    def _extract_typescript_entities(self, file_path: str, content: str) -> List[CodeEntity]:
        """Extraer entidades de código TypeScript"""
        # Por ahora, usar el mismo extractor que JavaScript
        entities = self._extract_javascript_entities(file_path, content)
        
        # Agregar detección específica de TypeScript
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Buscar interfaces TypeScript
            if re.match(r'^(export\s+)?interface\s+', line_stripped):
                match = re.search(r'interface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', line_stripped)
                if match:
                    entity = CodeEntity(
                        type='interface',
                        name=match.group(1),
                        file_path=file_path,
                        line_start=i,
                        line_end=i
                    )
                    entities.append(entity)
            
            # Buscar tipos TypeScript
            elif re.match(r'^(export\s+)?type\s+', line_stripped):
                match = re.search(r'type\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', line_stripped)
                if match:
                    entity = CodeEntity(
                        type='type',
                        name=match.group(1),
                        file_path=file_path,
                        line_start=i,
                        line_end=i
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_cpp_entities(self, file_path: str, content: str) -> List[CodeEntity]:
        """Extraer entidades de código C++"""
        entities = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('/*'):
                continue
            
            # Buscar includes
            if line_stripped.startswith('#include'):
                entity = CodeEntity(
                    type='include',
                    name=line_stripped,
                    file_path=file_path,
                    line_start=i,
                    line_end=i
                )
                entities.append(entity)
            
            # Buscar clases
            elif re.match(r'^class\s+', line_stripped) or re.match(r'^struct\s+', line_stripped):
                match = re.search(r'(class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_stripped)
                if match:
                    entity = CodeEntity(
                        type='class' if match.group(1) == 'class' else 'struct',
                        name=match.group(2),
                        file_path=file_path,
                        line_start=i,
                        line_end=i
                    )
                    entities.append(entity)
            
            # Buscar funciones
            elif re.match(r'^\w+\s+\w+\s*\([^)]*\)\s*(?:const)?\s*{?', line_stripped):
                # Patrón simple para funciones
                match = re.match(r'^(\w+)\s+(\w+)\s*\([^)]*\)', line_stripped)
                if match and match.group(2) not in ['if', 'while', 'for', 'switch']:
                    entity = CodeEntity(
                        type='function',
                        name=match.group(2),
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        return_type=match.group(1)
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_c_entities(self, file_path: str, content: str) -> List[CodeEntity]:
        """Extraer entidades de código C"""
        # Similar a C++ pero sin clases
        entities = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('/*'):
                continue
            
            # Buscar includes
            if line_stripped.startswith('#include'):
                entity = CodeEntity(
                    type='include',
                    name=line_stripped,
                    file_path=file_path,
                    line_start=i,
                    line_end=i
                )
                entities.append(entity)
            
            # Buscar structs
            elif re.match(r'^struct\s+', line_stripped):
                match = re.search(r'struct\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_stripped)
                if match:
                    entity = CodeEntity(
                        type='struct',
                        name=match.group(1),
                        file_path=file_path,
                        line_start=i,
                        line_end=i
                    )
                    entities.append(entity)
            
            # Buscar funciones
            elif re.match(r'^\w+\s+\w+\s*\([^)]*\)\s*{?', line_stripped):
                # Patrón simple para funciones
                match = re.match(r'^(\w+)\s+(\w+)\s*\([^)]*\)', line_stripped)
                if match and match.group(2) not in ['if', 'while', 'for', 'switch']:
                    entity = CodeEntity(
                        type='function',
                        name=match.group(2),
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        return_type=match.group(1)
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_generic_entities(self, file_path: str, content: str, language: str) -> List[CodeEntity]:
        """Extracción genérica de entidades para lenguajes no soportados"""
        entities = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Buscar patrones comunes
            # Funciones/métodos (patrón: palabra palabra(...))
            func_match = re.search(r'(\w+)\s+(\w+)\s*\(', line_stripped)
            if func_match and func_match.group(2) not in ['if', 'while', 'for', 'switch', 'catch']:
                entity = CodeEntity(
                    type='function',
                    name=func_match.group(2),
                    file_path=file_path,
                    line_start=i,
                    line_end=i
                )
                entities.append(entity)
            
            # Clases (patrón: class Nombre)
            class_match = re.search(r'class\s+(\w+)', line_stripped, re.IGNORECASE)
            if class_match:
                entity = CodeEntity(
                    type='class',
                    name=class_match.group(1),
                    file_path=file_path,
                    line_start=i,
                    line_end=i
                )
                entities.append(entity)
        
        return entities
    
    def extract_relationships(self, entities: List[CodeEntity]) -> EntityRelationships:
        """
        Extraer relaciones entre entidades
        
        Args:
            entities: Lista de entidades
            
        Returns:
            EntityRelationships: Relaciones extraídas
        """
        relationships = EntityRelationships()
        
        # Organizar entidades por tipo y nombre
        classes = {e.name: e for e in entities if e.type in ['class', 'struct', 'interface']}
        functions = {e.name: e for e in entities if e.type in ['function', 'method']}
        
        # Analizar relaciones
        for entity in entities:
            # Herencia (para clases)
            if entity.type == 'class' and 'metrics' in entity.metrics:
                bases = entity.metrics.get('bases', [])
                for base in bases:
                    if base in classes:
                        relationships.inheritance.append((entity.name, base))
            
            # Dependencias (imports/includes)
            if entity.type in ['import', 'include']:
                # Extraer nombre de la dependencia
                dep_name = entity.name
                # Encontrar qué entidades dependen de esto
                for other_entity in entities:
                    if other_entity != entity:
                        # Verificar si la entidad menciona la dependencia
                        if dep_name in other_entity.name or \
                           (other_entity.docstring and dep_name in other_entity.docstring):
                            relationships.dependency.append((other_entity.name, dep_name))
        
        return relationships
    
    def group_entities_by_type(self, entities: List[CodeEntity]) -> Dict[str, List[CodeEntity]]:
        """
        Agrupar entidades por tipo
        
        Args:
            entities: Lista de entidades
            
        Returns:
            Dict[str, List[CodeEntity]]: Entidades agrupadas por tipo
        """
        grouped = {}
        
        for entity in entities:
            if entity.type not in grouped:
                grouped[entity.type] = []
            grouped[entity.type].append(entity)
        
        return grouped
    
    def calculate_entity_metrics(self, entity: CodeEntity, content: str) -> Dict[str, Any]:
        """
        Calcular métricas para una entidad
        
        Args:
            entity: Entidad a analizar
            content: Contenido completo del archivo
            
        Returns:
            Dict[str, Any]: Métricas calculadas
        """
        metrics = {
            'lines_of_code': entity.line_end - entity.line_start + 1,
            'complexity': 1,  # Por defecto
            'parameters_count': len(entity.parameters) if entity.parameters else 0,
        }
        
        # Extraer el código de la entidad
        lines = content.splitlines()
        entity_lines = lines[entity.line_start-1:entity.line_end]
        entity_code = '\n'.join(entity_lines)
        
        # Calcular complejidad ciclomática aproximada
        if entity.type in ['function', 'method']:
            complexity = 1
            keywords = ['if', 'else', 'while', 'for', 'case', 'catch', '&&', '||', '?']
            
            for line in entity_lines:
                for keyword in keywords:
                    if keyword in line:
                        complexity += 1
            
            metrics['complexity'] = complexity
        
        # Contar llamadas a otras funciones
        if entity.type in ['function', 'method']:
            # Patrón simple para llamadas a funciones
            call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            calls = re.findall(call_pattern, entity_code)
            
            # Filtrar palabras clave y la función misma
            keywords = ['if', 'while', 'for', 'switch', 'catch', 'return']
            filtered_calls = [call for call in calls if call not in keywords and call != entity.name]
            
            metrics['function_calls'] = len(set(filtered_calls))
            metrics['unique_calls'] = list(set(filtered_calls))
        
        return metrics