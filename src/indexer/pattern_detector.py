"""
Módulo PatternDetector - Detección de patrones en código fuente
"""

import re
import ast
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import defaultdict
import json

class PatternType(Enum):
    """Tipos de patrones detectables"""
    DESIGN_PATTERN = "design_pattern"
    ANTI_PATTERN = "anti_pattern"
    ARCHITECTURAL_PATTERN = "architectural_pattern"
    CODE_SMELL = "code_smell"
    SECURITY_ISSUE = "security_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    IDIOM = "idiom"
    BEST_PRACTICE = "best_practice"

class PatternSeverity(Enum):
    """Niveles de severidad de patrones"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DetectedPattern:
    """Patrón detectado en el código"""
    pattern_id: str
    pattern_type: PatternType
    name: str
    description: str
    file_path: str
    line_start: int
    line_end: int
    severity: PatternSeverity
    confidence: float  # 0.0 a 1.0
    context: Optional[str] = None
    suggestion: Optional[str] = None
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternSummary:
    """Resumen de patrones detectados"""
    total_patterns: int = 0
    by_type: Dict[PatternType, int] = field(default_factory=dict)
    by_severity: Dict[PatternSeverity, int] = field(default_factory=dict)
    by_file: Dict[str, int] = field(default_factory=dict)
    most_common: List[Tuple[str, int]] = field(default_factory=list)

class PatternDetector:
    """Detector de patrones en código fuente"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar detector de patrones"""
        self.config = config or {}
        self.pattern_rules = self._load_pattern_rules()
        self.language_detectors = {
            'python': self._detect_python_patterns,
            'java': self._detect_java_patterns,
            'javascript': self._detect_javascript_patterns,
            'typescript': self._detect_typescript_patterns,
            'cpp': self._detect_cpp_patterns,
            'c': self._detect_c_patterns,
        }
        
    def _load_pattern_rules(self) -> Dict[str, Dict[str, Any]]:
        """Cargar reglas de detección de patrones"""
        # Patrones comunes para múltiples lenguajes
        return {
            # Patrones de diseño
            'singleton': {
                'type': PatternType.DESIGN_PATTERN,
                'description': 'Patrón Singleton - asegura una sola instancia',
                'severity': PatternSeverity.INFO,
                'detectors': {
                    'python': self._detect_python_singleton,
                    'java': self._detect_java_singleton,
                    'cpp': self._detect_cpp_singleton,
                }
            },
            'factory': {
                'type': PatternType.DESIGN_PATTERN,
                'description': 'Patrón Factory - creación de objetos',
                'severity': PatternSeverity.INFO,
                'detectors': {
                    'python': self._detect_python_factory,
                    'java': self._detect_java_factory,
                }
            },
            'observer': {
                'type': PatternType.DESIGN_PATTERN,
                'description': 'Patrón Observer - notificación de eventos',
                'severity': PatternSeverity.INFO,
                'detectors': {
                    'python': self._detect_python_observer,
                    'java': self._detect_java_observer,
                }
            },
            
            # Anti-patrones
            'god_class': {
                'type': PatternType.ANTI_PATTERN,
                'description': 'Clase Dios - demasiadas responsabilidades',
                'severity': PatternSeverity.HIGH,
                'detectors': {
                    'python': self._detect_god_class,
                    'java': self._detect_god_class,
                    'cpp': self._detect_god_class,
                }
            },
            'long_method': {
                'type': PatternType.ANTI_PATTERN,
                'description': 'Método/Función demasiado largo',
                'severity': PatternSeverity.MEDIUM,
                'detectors': {
                    'python': self._detect_long_method,
                    'java': self._detect_long_method,
                    'javascript': self._detect_long_method,
                }
            },
            'duplicate_code': {
                'type': PatternType.ANTI_PATTERN,
                'description': 'Código duplicado',
                'severity': PatternSeverity.MEDIUM,
                'detectors': {
                    'python': self._detect_duplicate_code,
                    'java': self._detect_duplicate_code,
                    'javascript': self._detect_duplicate_code,
                }
            },
            
            # Code smells
            'magic_number': {
                'type': PatternType.CODE_SMELL,
                'description': 'Número mágico - constante sin nombre',
                'severity': PatternSeverity.LOW,
                'detectors': {
                    'python': self._detect_magic_numbers,
                    'java': self._detect_magic_numbers,
                    'cpp': self._detect_magic_numbers,
                }
            },
            'deep_nesting': {
                'type': PatternType.CODE_SMELL,
                'description': 'Anidamiento profundo',
                'severity': PatternSeverity.MEDIUM,
                'detectors': {
                    'python': self._detect_deep_nesting,
                    'java': self._detect_deep_nesting,
                    'javascript': self._detect_deep_nesting,
                }
            },
            
            # Issues de seguridad
            'sql_injection': {
                'type': PatternType.SECURITY_ISSUE,
                'description': 'Posible inyección SQL',
                'severity': PatternSeverity.CRITICAL,
                'detectors': {
                    'python': self._detect_sql_injection,
                    'java': self._detect_sql_injection,
                    'javascript': self._detect_sql_injection,
                }
            },
            'hardcoded_password': {
                'type': PatternType.SECURITY_ISSUE,
                'description': 'Contraseña hardcodeada',
                'severity': PatternSeverity.HIGH,
                'detectors': {
                    'python': self._detect_hardcoded_password,
                    'java': self._detect_hardcoded_password,
                    'javascript': self._detect_hardcoded_password,
                }
            },
            
            # Issues de performance
            'nested_loop': {
                'type': PatternType.PERFORMANCE_ISSUE,
                'description': 'Bucles anidados con complejidad O(n²)',
                'severity': PatternSeverity.MEDIUM,
                'detectors': {
                    'python': self._detect_nested_loops,
                    'java': self._detect_nested_loops,
                    'cpp': self._detect_nested_loops,
                }
            },
        }
    
    def detect_patterns(self, file_path: str, content: str, 
                       language: str) -> List[DetectedPattern]:
        """
        Detectar patrones en un archivo de código
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            language: Lenguaje de programación
            
        Returns:
            List[DetectedPattern]: Patrones detectados
        """
        patterns = []
        
        if language not in self.language_detectors:
            # Lenguaje no soportado
            return patterns
        
        detector = self.language_detectors[language]
        try:
            patterns = detector(file_path, content)
        except Exception as e:
            # Si falla el detector específico, usar detección genérica
            patterns = self._detect_generic_patterns(file_path, content)
        
        return patterns
    
    def _detect_python_patterns(self, file_path: str, content: str) -> List[DetectedPattern]:
        """Detectar patrones en código Python"""
        patterns = []
        
        try:
            tree = ast.parse(content)
            
            # Detectar patrones específicos de Python
            patterns.extend(self._detect_python_specific_patterns(tree, file_path, content))
            
            # Detectar patrones comunes
            patterns.extend(self._detect_common_patterns(tree, file_path, content, 'python'))
            
        except SyntaxError:
            # Si hay error de sintaxis, usar detección basada en regex
            patterns = self._detect_python_patterns_regex(file_path, content)
        
        return patterns
    
    def _detect_python_specific_patterns(self, tree: ast.AST, file_path: str, 
                                        content: str) -> List[DetectedPattern]:
        """Detectar patrones específicos de Python"""
        patterns = []
        
        # Detectar uso de eval/exec
        patterns.extend(self._detect_python_eval_usage(tree, file_path))
        
        # Detectar bare except
        patterns.extend(self._detect_python_bare_except(tree, file_path))
        
        # Detectar mutable default arguments
        patterns.extend(self._detect_python_mutable_defaults(tree, file_path))
        
        # Detectar comprehensions anidados
        patterns.extend(self._detect_python_nested_comprehensions(tree, file_path))
        
        # Detectar funciones con muchos argumentos
        patterns.extend(self._detect_python_many_arguments(tree, file_path))
        
        return patterns
    
    def _detect_python_eval_usage(self, tree: ast.AST, file_path: str) -> List[DetectedPattern]:
        """Detectar uso de eval() o exec() en Python"""
        patterns = []
        
        class EvalVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.patterns = []
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec']:
                        pattern = DetectedPattern(
                            pattern_id='python_eval_usage',
                            pattern_type=PatternType.SECURITY_ISSUE,
                            name='Uso de eval/exec',
                            description='Uso de eval() o exec() - riesgo de seguridad',
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            severity=PatternSeverity.HIGH,
                            confidence=0.9,
                            suggestion='Evitar eval()/exec(), usar alternativas seguras',
                            references=['https://docs.python.org/3/library/functions.html#eval']
                        )
                        self.patterns.append(pattern)
                
                self.generic_visit(node)
        
        visitor = EvalVisitor(file_path)
        visitor.visit(tree)
        return visitor.patterns
    
    def _detect_python_bare_except(self, tree: ast.AST, file_path: str) -> List[DetectedPattern]:
        """Detectar except sin tipo específico en Python"""
        patterns = []
        
        class ExceptVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.patterns = []
            
            def visit_ExceptHandler(self, node):
                if node.type is None:  # except: sin tipo
                    pattern = DetectedPattern(
                        pattern_id='python_bare_except',
                        pattern_type=PatternType.CODE_SMELL,
                        name='Bare Except',
                        description='Except sin tipo específico - puede capturar excepciones no deseadas',
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        severity=PatternSeverity.MEDIUM,
                        confidence=0.8,
                        suggestion='Especificar el tipo de excepción a capturar',
                        references=['https://docs.python.org/3/tutorial/errors.html#handling-exceptions']
                    )
                    self.patterns.append(pattern)
                
                self.generic_visit(node)
        
        visitor = ExceptVisitor(file_path)
        visitor.visit(tree)
        return visitor.patterns
    
    def _detect_python_mutable_defaults(self, tree: ast.AST, file_path: str) -> List[DetectedPattern]:
        """Detectar argumentos por defecto mutables en Python"""
        patterns = []
        
        class DefaultVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.patterns = []
            
            def visit_FunctionDef(self, node):
                # Verificar argumentos por defecto
                defaults = node.args.defaults
                for i, default in enumerate(defaults):
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        pattern = DetectedPattern(
                            pattern_id='python_mutable_default',
                            pattern_type=PatternType.CODE_SMELL,
                            name='Argumento por defecto mutable',
                            description='Argumento por defecto mutable - puede causar comportamiento inesperado',
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            severity=PatternSeverity.MEDIUM,
                            confidence=0.9,
                            suggestion='Usar None como valor por defecto y crear mutable dentro de la función',
                            references=['https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments']
                        )
                        self.patterns.append(pattern)
                        break
                
                self.generic_visit(node)
        
        visitor = DefaultVisitor(file_path)
        visitor.visit(tree)
        return visitor.patterns
    
    def _detect_python_nested_comprehensions(self, tree: ast.AST, file_path: str) -> List[DetectedPattern]:
        """Detectar comprehensions anidados complejos en Python"""
        patterns = []
        
        class ComprehensionVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.patterns = []
                self.nesting_level = 0
            
            def visit_ListComp(self, node):
                self.nesting_level += 1
                if self.nesting_level > 2:
                    pattern = DetectedPattern(
                        pattern_id='python_nested_comprehension',
                        pattern_type=PatternType.PERFORMANCE_ISSUE,
                        name='Comprehension anidado',
                        description='Comprehension anidado múltiples veces - puede ser difícil de leer',
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        severity=PatternSeverity.LOW,
                        confidence=0.7,
                        suggestion='Considerar usar bucles for explícitos para claridad'
                    )
                    self.patterns.append(pattern)
                
                self.generic_visit(node)
                self.nesting_level -= 1
        
        visitor = ComprehensionVisitor(file_path)
        visitor.visit(tree)
        return visitor.patterns
    
    def _detect_python_many_arguments(self, tree: ast.AST, file_path: str) -> List[DetectedPattern]:
        """Detectar funciones con muchos argumentos en Python"""
        patterns = []
        
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.patterns = []
            
            def visit_FunctionDef(self, node):
                # Contar argumentos
                arg_count = len(node.args.args) + len(node.args.kwonlyargs)
                if node.args.vararg:
                    arg_count += 1
                if node.args.kwarg:
                    arg_count += 1
                
                if arg_count > 6:  # Umbral arbitrario
                    pattern = DetectedPattern(
                        pattern_id='python_many_arguments',
                        pattern_type=PatternType.CODE_SMELL,
                        name='Función con muchos argumentos',
                        description=f'Función con {arg_count} argumentos - viola principio de responsabilidad única',
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        severity=PatternSeverity.MEDIUM,
                        confidence=0.7,
                        suggestion='Considerar dividir la función o usar objetos parámetro'
                    )
                    self.patterns.append(pattern)
                
                self.generic_visit(node)
        
        visitor = FunctionVisitor(file_path)
        visitor.visit(tree)
        return visitor.patterns
    
    def _detect_python_singleton(self, tree: ast.AST, file_path: str) -> List[DetectedPattern]:
        """Detectar patrón Singleton en Python"""
        patterns = []
        
        class SingletonVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.patterns = []
            
            def visit_ClassDef(self, node):
                # Buscar atributos de clase que puedan indicar Singleton
                class_attrs = set()
                
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                class_attrs.add(target.id)
                    
                    # Buscar métodos de clase
                    if isinstance(item, ast.FunctionDef):
                        if item.name == '__new__':
                            class_attrs.add('__new__')
                
                # Patrones comunes de Singleton
                singleton_patterns = ['_instance', '_INSTANCE', 'instance', '__new__']
                if any(pattern in class_attrs for pattern in singleton_patterns):
                    pattern = DetectedPattern(
                        pattern_id='singleton_design_pattern',
                        pattern_type=PatternType.DESIGN_PATTERN,
                        name='Singleton Pattern',
                        description='Patrón Singleton detectado - una única instancia de clase',
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        severity=PatternSeverity.INFO,
                        confidence=0.6,
                        suggestion='Asegurar thread safety si es necesario'
                    )
                    self.patterns.append(pattern)
                
                self.generic_visit(node)
        
        visitor = SingletonVisitor(file_path)
        visitor.visit(tree)
        return visitor.patterns
    
    def _detect_python_factory(self, tree: ast.AST, file_path: str) -> List[DetectedPattern]:
        """Detectar patrón Factory en Python"""
        patterns = []
        
        class FactoryVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.patterns = []
            
            def visit_FunctionDef(self, node):
                # Buscar funciones que creen objetos
                function_name = node.name.lower()
                factory_keywords = ['create', 'make', 'build', 'factory', 'get_']
                
                if any(keyword in function_name for keyword in factory_keywords):
                    # Verificar si retorna instancias
                    has_return = False
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return):
                            has_return = True
                            break
                    
                    if has_return:
                        pattern = DetectedPattern(
                            pattern_id='factory_design_pattern',
                            pattern_type=PatternType.DESIGN_PATTERN,
                            name='Factory Pattern',
                            description='Patrón Factory detectado - creación de objetos',
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            severity=PatternSeverity.INFO,
                            confidence=0.5,
                            suggestion='Considerar Factory Method o Abstract Factory según complejidad'
                        )
                        self.patterns.append(pattern)
                
                self.generic_visit(node)
        
        visitor = FactoryVisitor(file_path)
        visitor.visit(tree)
        return visitor.patterns
    
    def _detect_python_observer(self, tree: ast.AST, file_path: str) -> List[DetectedPattern]:
        """Detectar patrón Observer en Python"""
        patterns = []
        
        class ObserverVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.patterns = []
            
            def visit_ClassDef(self, node):
                # Buscar atributos que sean listas
                list_attrs = []
                subscribe_methods = []
                notify_methods = []
                
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Attribute):
                                attr_name = target.attr.lower()
                                if 'list' in attr_name or 'observers' in attr_name:
                                    list_attrs.append(attr_name)
                    
                    if isinstance(item, ast.FunctionDef):
                        method_name = item.name.lower()
                        if any(keyword in method_name for keyword in ['add', 'register', 'subscribe']):
                            subscribe_methods.append(item.name)
                        elif any(keyword in method_name for keyword in ['notify', 'update', 'broadcast']):
                            notify_methods.append(item.name)
                
                if list_attrs and (subscribe_methods or notify_methods):
                    pattern = DetectedPattern(
                        pattern_id='observer_design_pattern',
                        pattern_type=PatternType.DESIGN_PATTERN,
                        name='Observer Pattern',
                        description='Patrón Observer detectado - notificación de eventos',
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        severity=PatternSeverity.INFO,
                        confidence=0.6,
                        suggestion='Considerar implementar interfaz clara para observers'
                    )
                    self.patterns.append(pattern)
                
                self.generic_visit(node)
        
        visitor = ObserverVisitor(file_path)
        visitor.visit(tree)
        return visitor.patterns
    
    def _detect_python_patterns_regex(self, file_path: str, content: str) -> List[DetectedPattern]:
        """Detectar patrones Python usando regex (fallback)"""
        patterns = []
        lines = content.splitlines()
        
        # Detectar eval/exec
        for i, line in enumerate(lines, 1):
            if re.search(r'\beval\s*\(', line) or re.search(r'\bexec\s*\(', line):
                patterns.append(DetectedPattern(
                    pattern_id='python_eval_usage_regex',
                    pattern_type=PatternType.SECURITY_ISSUE,
                    name='Uso de eval/exec',
                    description='Uso de eval() o exec() detectado',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=PatternSeverity.HIGH,
                    confidence=0.7,
                    suggestion='Evitar eval()/exec() por seguridad'
                ))
        
        return patterns
    
    def _detect_java_patterns(self, file_path: str, content: str) -> List[DetectedPattern]:
        """Detectar patrones en código Java"""
        patterns = []
        
        # Detectar patrones usando regex (simplificado)
        patterns.extend(self._detect_java_singleton(content, file_path))
        patterns.extend(self._detect_java_factory(content, file_path))
        patterns.extend(self._detect_common_patterns_regex(content, file_path, 'java'))
        
        return patterns
    
    def _detect_java_singleton(self, content: str, file_path: str) -> List[DetectedPattern]:
        """Detectar Singleton en Java usando regex"""
        patterns = []
        lines = content.splitlines()
        
        singleton_indicators = 0
        
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            if 'private static' in line_lower and 'instance' in line_lower:
                singleton_indicators += 1
            
            if 'getinstance()' in line_lower or 'getinstance (' in line_lower:
                singleton_indicators += 1
            
            if 'synchronized' in line_lower and 'instance' in line_lower:
                singleton_indicators += 1
            
            if singleton_indicators >= 2:
                pattern = DetectedPattern(
                    pattern_id='java_singleton',
                    pattern_type=PatternType.DESIGN_PATTERN,
                    name='Singleton Pattern',
                    description='Patrón Singleton detectado en Java',
                    file_path=file_path,
                    line_start=max(1, i - 5),
                    line_end=min(len(lines), i + 5),
                    severity=PatternSeverity.INFO,
                    confidence=0.7,
                    suggestion='Considerar enum para Singleton thread-safe'
                )
                patterns.append(pattern)
                singleton_indicators = 0
        
        return patterns
    
    def _detect_java_factory(self, content: str, file_path: str) -> List[DetectedPattern]:
        """Detectar Factory en Java"""
        patterns = []
        
        # Buscar clases Factory
        factory_classes = re.findall(r'class\s+(\w+Factory)\b', content)
        for class_name in factory_classes:
            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                if f'class {class_name}' in line:
                    pattern = DetectedPattern(
                        pattern_id='java_factory',
                        pattern_type=PatternType.DESIGN_PATTERN,
                        name='Factory Pattern',
                        description=f'Clase Factory detectada: {class_name}',
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        severity=PatternSeverity.INFO,
                        confidence=0.8,
                        suggestion='Considerar Factory Method o Abstract Factory según necesidades'
                    )
                    patterns.append(pattern)
                    break
        
        return patterns
    
    def _detect_javascript_patterns(self, file_path: str, content: str) -> List[DetectedPattern]:
        """Detectar patrones en código JavaScript"""
        patterns = []
        
        # Detectar patrones específicos de JavaScript
        patterns.extend(self._detect_javascript_callback_hell(content, file_path))
        patterns.extend(self._detect_javascript_global_vars(content, file_path))
        patterns.extend(self._detect_common_patterns_regex(content, file_path, 'javascript'))
        
        return patterns
    
    def _detect_javascript_callback_hell(self, content: str, file_path: str) -> List[DetectedPattern]:
        """Detectar Callback Hell en JavaScript"""
        patterns = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            # Contar niveles de anidamiento por llaves y paréntesis
            open_count = line.count('{') + line.count('(')
            close_count = line.count('}') + line.count(')')
            
            # Buscar patrones de callbacks anidados
            if line.count('})') > 2 or line.count('})();') > 1:
                pattern = DetectedPattern(
                    pattern_id='javascript_callback_hell',
                    pattern_type=PatternType.ANTI_PATTERN,
                    name='Callback Hell',
                    description='Múltiples callbacks anidados - difícil de leer y mantener',
                    file_path=file_path,
                    line_start=max(1, i - 10),
                    line_end=min(len(lines), i + 10),
                    severity=PatternSeverity.MEDIUM,
                    confidence=0.7,
                    suggestion='Usar Promises, async/await o bibliotecas como async.js',
                    references=['http://callbackhell.com/']
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_javascript_global_vars(self, content: str, file_path: str) -> List[DetectedPattern]:
        """Detectar variables globales en JavaScript"""
        patterns = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            # Buscar declaraciones de variables sin var/let/const
            if re.search(r'^\s*[a-zA-Z_$][a-zA-Z0-9_$]*\s*=', line) and \
               not re.search(r'\b(var|let|const)\s+', line):
                pattern = DetectedPattern(
                    pattern_id='javascript_global_variable',
                    pattern_type=PatternType.CODE_SMELL,
                    name='Variable Global',
                    description='Variable declarada sin var/let/const - se convierte en global',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=PatternSeverity.MEDIUM,
                    confidence=0.9,
                    suggestion='Usar var, let o const para declarar variables'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_typescript_patterns(self, file_path: str, content: str) -> List[DetectedPattern]:
        """Detectar patrones en código TypeScript"""
        patterns = []
        
        # Similar a JavaScript pero con chequeos de tipos
        patterns.extend(self._detect_typescript_any_type(content, file_path))
        patterns.extend(self._detect_javascript_patterns(file_path, content))
        
        return patterns
    
    def _detect_typescript_any_type(self, content: str, file_path: str) -> List[DetectedPattern]:
        """Detectar uso de tipo 'any' en TypeScript"""
        patterns = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            if re.search(r':\s*any\b', line) or re.search(r'\bany\[\]', line):
                pattern = DetectedPattern(
                    pattern_id='typescript_any_type',
                    pattern_type=PatternType.CODE_SMELL,
                    name='Uso de tipo any',
                    description='Uso del tipo any - pierde beneficios de TypeScript',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=PatternSeverity.LOW,
                    confidence=0.8,
                    suggestion='Usar tipos específicos o unknown en lugar de any'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_cpp_patterns(self, file_path: str, content: str) -> List[DetectedPattern]:
        """Detectar patrones en código C++"""
        patterns = []
        
        patterns.extend(self._detect_cpp_raw_pointers(content, file_path))
        patterns.extend(self._detect_cpp_memory_leaks(content, file_path))
        patterns.extend(self._detect_common_patterns_regex(content, file_path, 'cpp'))
        
        return patterns
    
    def _detect_cpp_raw_pointers(self, content: str, file_path: str) -> List[DetectedPattern]:
        """Detectar uso de raw pointers en C++ moderno"""
        patterns = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            if re.search(r'\bnew\s+', line) and not re.search(r'\bdelete\s+', line):
                pattern = DetectedPattern(
                    pattern_id='cpp_raw_pointer_new',
                    pattern_type=PatternType.CODE_SMELL,
                    name='Raw pointer con new',
                    description='Uso de new sin delete correspondiente - riesgo de memory leak',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=PatternSeverity.MEDIUM,
                    confidence=0.6,
                    suggestion='Usar smart pointers (unique_ptr, shared_ptr)'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_cpp_memory_leaks(self, content: str, file_path: str) -> List[DetectedPattern]:
        """Detectar posibles memory leaks en C++"""
        patterns = []
        
        # Contar news y deletes
        new_count = len(re.findall(r'\bnew\s+', content))
        delete_count = len(re.findall(r'\bdelete\s+', content))
        
        if new_count > delete_count:
            pattern = DetectedPattern(
                pattern_id='cpp_possible_memory_leak',
                pattern_type=PatternType.PERFORMANCE_ISSUE,
                name='Posible memory leak',
                description=f'Posible memory leak: {new_count} new vs {delete_count} delete',
                file_path=file_path,
                line_start=1,
                line_end=len(content.splitlines()),
                severity=PatternSeverity.HIGH,
                confidence=0.5,
                suggestion='Usar RAII y smart pointers para gestión automática de memoria'
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_c_patterns(self, file_path: str, content: str) -> List[DetectedPattern]:
        """Detectar patrones en código C"""
        patterns = []
        
        patterns.extend(self._detect_c_memory_issues(content, file_path))
        patterns.extend(self._detect_common_patterns_regex(content, file_path, 'c'))
        
        return patterns
    
    def _detect_c_memory_issues(self, content: str, file_path: str) -> List[DetectedPattern]:
        """Detectar issues de memoria en C"""
        patterns = []
        
        # Buscar malloc sin free
        malloc_count = len(re.findall(r'\bmalloc\s*\(', content))
        free_count = len(re.findall(r'\bfree\s*\(', content))
        
        if malloc_count > free_count:
            pattern = DetectedPattern(
                pattern_id='c_possible_memory_leak',
                pattern_type=PatternType.PERFORMANCE_ISSUE,
                name='Posible memory leak',
                description=f'Posible memory leak: {malloc_count} malloc vs {free_count} free',
                file_path=file_path,
                line_start=1,
                line_end=len(content.splitlines()),
                severity=PatternSeverity.HIGH,
                confidence=0.5,
                suggestion='Asegurar que cada malloc tenga su free correspondiente'
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_generic_patterns(self, file_path: str, content: str) -> List[DetectedPattern]:
        """Detección genérica de patrones para lenguajes no soportados"""
        patterns = []
        
        # Detectar líneas muy largas
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            if len(line) > 120:  # Umbral para línea larga
                pattern = DetectedPattern(
                    pattern_id='generic_long_line',
                    pattern_type=PatternType.CODE_SMELL,
                    name='Línea muy larga',
                    description=f'Línea con {len(line)} caracteres - difícil de leer',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=PatternSeverity.LOW,
                    confidence=0.9,
                    suggestion='Dividir la línea en múltiples líneas'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_common_patterns(self, tree: ast.AST, file_path: str, 
                               content: str, language: str) -> List[DetectedPattern]:
        """Detectar patrones comunes a todos los lenguajes"""
        patterns = []
        
        # Detectar God Class
        patterns.extend(self._detect_god_class(tree, file_path, language))
        
        # Detectar Long Method
        patterns.extend(self._detect_long_method(tree, file_path, language))
        
        # Detectar código duplicado (simplificado)
        patterns.extend(self._detect_duplicate_code(content, file_path, language))
        
        # Detectar números mágicos
        patterns.extend(self._detect_magic_numbers(content, file_path, language))
        
        # Detectar anidamiento profundo
        patterns.extend(self._detect_deep_nesting(tree, file_path, language))
        
        # Detectar SQL injection
        patterns.extend(self._detect_sql_injection(content, file_path, language))
        
        # Detectar contraseñas hardcodeadas
        patterns.extend(self._detect_hardcoded_password(content, file_path, language))
        
        # Detectar bucles anidados
        patterns.extend(self._detect_nested_loops(tree, file_path, language))
        
        return patterns
    
    def _detect_common_patterns_regex(self, content: str, file_path: str, 
                                    language: str) -> List[DetectedPattern]:
        """Detectar patrones comunes usando regex"""
        patterns = []
        
        # Detectar números mágicos
        patterns.extend(self._detect_magic_numbers(content, file_path, language))
        
        # Detectar SQL injection
        patterns.extend(self._detect_sql_injection(content, file_path, language))
        
        # Detectar contraseñas hardcodeadas
        patterns.extend(self._detect_hardcoded_password(content, file_path, language))
        
        return patterns
    
    def _detect_god_class(self, tree: ast.AST, file_path: str, 
                         language: str) -> List[DetectedPattern]:
        """Detectar God Class"""
        patterns = []
        
        if language == 'python':
            class GodClassVisitor(ast.NodeVisitor):
                def __init__(self, file_path):
                    self.file_path = file_path
                    self.patterns = []
                
                def visit_ClassDef(self, node):
                    # Contar métodos y atributos
                    method_count = 0
                    line_count = (node.end_lineno or node.lineno) - node.lineno + 1
                    
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            method_count += 1
                    
                    if method_count > 15 or line_count > 300:
                        pattern = DetectedPattern(
                            pattern_id='god_class',
                            pattern_type=PatternType.ANTI_PATTERN,
                            name='God Class',
                            description=f'Clase con {method_count} métodos y {line_count} líneas',
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            severity=PatternSeverity.HIGH,
                            confidence=min(0.9, method_count / 20.0),
                            suggestion='Dividir la clase en clases más pequeñas con responsabilidades únicas'
                        )
                        self.patterns.append(pattern)
                    
                    self.generic_visit(node)
            
            visitor = GodClassVisitor(file_path)
            visitor.visit(tree)
            patterns = visitor.patterns
        
        return patterns
    
    def _detect_long_method(self, tree: ast.AST, file_path: str, 
                           language: str) -> List[DetectedPattern]:
        """Detectar métodos/funciones largos"""
        patterns = []
        
        if language == 'python':
            class LongMethodVisitor(ast.NodeVisitor):
                def __init__(self, file_path):
                    self.file_path = file_path
                    self.patterns = []
                
                def visit_FunctionDef(self, node):
                    line_count = (node.end_lineno or node.lineno) - node.lineno + 1
                    
                    if line_count > 50:
                        pattern = DetectedPattern(
                            pattern_id='long_method',
                            pattern_type=PatternType.ANTI_PATTERN,
                            name='Método/Función largo',
                            description=f'Función con {line_count} líneas',
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            severity=PatternSeverity.MEDIUM,
                            confidence=min(0.9, line_count / 60.0),
                            suggestion='Dividir la función en funciones más pequeñas'
                        )
                        self.patterns.append(pattern)
                    
                    self.generic_visit(node)
            
            visitor = LongMethodVisitor(file_path)
            visitor.visit(tree)
            patterns = visitor.patterns
        
        return patterns
    
    def _detect_duplicate_code(self, content: str, file_path: str, 
                              language: str) -> List[DetectedPattern]:
        """Detectar código duplicado (simplificado)"""
        patterns = []
        
        # Buscar líneas duplicadas consecutivas (simplificación)
        lines = content.splitlines()
        
        for i in range(len(lines) - 2):
            if lines[i].strip() and lines[i] == lines[i + 1]:
                pattern = DetectedPattern(
                    pattern_id='duplicate_code_consecutive',
                    pattern_type=PatternType.ANTI_PATTERN,
                    name='Código duplicado',
                    description='Líneas idénticas consecutivas',
                    file_path=file_path,
                    line_start=i + 1,
                    line_end=i + 2,
                    severity=PatternSeverity.LOW,
                    confidence=0.8,
                    suggestion='Eliminar código duplicado'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_magic_numbers(self, content: str, file_path: str, 
                             language: str) -> List[DetectedPattern]:
        """Detectar números mágicos"""
        patterns = []
        lines = content.splitlines()
        
        # Números mágicos comunes (excluyendo 0, 1, -1 que suelen ser aceptables)
        magic_numbers = {
            '2': ['array index', 'buffer size'],
            '3': ['RGB components', 'tristate'],
            '4': ['IPv4 octets', 'bytes per int'],
            '7': ['days in week', 'colors in rainbow'],
            '8': ['bits in byte'],
            '10': ['decimal base'],
            '12': ['months in year'],
            '16': ['hex base', 'bits in short'],
            '24': ['hours in day'],
            '32': ['bits in int'],
            '64': ['bits in long'],
            '100': ['percentage'],
            '1024': ['bytes in KB'],
            '3600': ['seconds in hour'],
            '65535': ['max port number'],
        }
        
        for i, line in enumerate(lines, 1):
            # Buscar números en la línea
            numbers = re.findall(r'\b(\d+)\b', line)
            for num in numbers:
                if num in magic_numbers and int(num) not in [0, 1, -1]:
                    context = magic_numbers[num][0]
                    pattern = DetectedPattern(
                        pattern_id='magic_number',
                        pattern_type=PatternType.CODE_SMELL,
                        name='Número mágico',
                        description=f'Número mágico {num} ({context})',
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        severity=PatternSeverity.LOW,
                        confidence=0.7,
                        suggestion=f'Definir constante con nombre significativo para {num}'
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_deep_nesting(self, tree: ast.AST, file_path: str, 
                            language: str) -> List[DetectedPattern]:
        """Detectar anidamiento profundo"""
        patterns = []
        
        if language == 'python':
            class NestingVisitor(ast.NodeVisitor):
                def __init__(self, file_path):
                    self.file_path = file_path
                    self.patterns = []
                    self.max_depth = 0
                    self.current_depth = 0
                    self.current_function = None
                
                def visit_FunctionDef(self, node):
                    self.current_function = node.name
                    self.current_depth = 0
                    self.max_depth = 0
                    self.generic_visit(node)
                    
                    if self.max_depth > 4:
                        pattern = DetectedPattern(
                            pattern_id='deep_nesting',
                            pattern_type=PatternType.CODE_SMELL,
                            name='Anidamiento profundo',
                            description=f'Anidamiento de nivel {self.max_depth} en función {node.name}',
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            severity=PatternSeverity.MEDIUM,
                            confidence=min(0.9, self.max_depth / 6.0),
                            suggestion='Reducir nivel de anidamiento usando funciones auxiliares'
                        )
                        self.patterns.append(pattern)
                    
                    self.current_function = None
                
                def visit_If(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_For(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_While(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_Try(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
            
            visitor = NestingVisitor(file_path)
            visitor.visit(tree)
            patterns = visitor.patterns
        
        return patterns
    
    def _detect_sql_injection(self, content: str, file_path: str, 
                             language: str) -> List[DetectedPattern]:
        """Detectar posibles inyecciones SQL"""
        patterns = []
        lines = content.splitlines()
        
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE']
        
        for i, line in enumerate(lines, 1):
            line_upper = line.upper()
            
            # Buscar concatenación de strings con variables
            if any(keyword in line_upper for keyword in sql_keywords):
                # Patrones de concatenación peligrosa
                concat_patterns = {
                    'python': [r'\+.*%s', r'%\(.*\)s.*\+', r'format\(.*\)'],
                    'java': [r'\+'],
                    'javascript': [r'\+'],
                    'csharp': [r'\+'],
                }
                
                if language in concat_patterns:
                    for pattern in concat_patterns[language]:
                        if re.search(pattern, line):
                            pattern_obj = DetectedPattern(
                                pattern_id='sql_injection',
                                pattern_type=PatternType.SECURITY_ISSUE,
                                name='Posible inyección SQL',
                                description='Concatenación de strings en consulta SQL',
                                file_path=file_path,
                                line_start=i,
                                line_end=i,
                                severity=PatternSeverity.CRITICAL,
                                confidence=0.8,
                                suggestion='Usar parámetros preparados o consultas parametrizadas'
                            )
                            patterns.append(pattern_obj)
                            break
        
        return patterns
    
    def _detect_hardcoded_password(self, content: str, file_path: str, 
                                  language: str) -> List[DetectedPattern]:
        """Detectar contraseñas hardcodeadas"""
        patterns = []
        lines = content.splitlines()
        
        password_patterns = [
            r'password\s*[:=]\s*["\'][^"\']+["\']',
            r'passwd\s*[:=]\s*["\'][^"\']+["\']',
            r'pwd\s*[:=]\s*["\'][^"\']+["\']',
            r'secret\s*[:=]\s*["\'][^"\']+["\']',
            r'api[_-]?key\s*[:=]\s*["\'][^"\']+["\']',
            r'token\s*[:=]\s*["\'][^"\']+["\']',
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in password_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    pattern_obj = DetectedPattern(
                        pattern_id='hardcoded_password',
                        pattern_type=PatternType.SECURITY_ISSUE,
                        name='Contraseña/Token hardcodeado',
                        description='Credencial sensible hardcodeada en el código',
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        severity=PatternSeverity.HIGH,
                        confidence=0.9,
                        suggestion='Usar variables de entorno o sistemas de gestión de secretos'
                    )
                    patterns.append(pattern_obj)
                    break
        
        return patterns
    
    def _detect_nested_loops(self, tree: ast.AST, file_path: str, 
                            language: str) -> List[DetectedPattern]:
        """Detectar bucles anidados"""
        patterns = []
        
        if language == 'python':
            class NestedLoopVisitor(ast.NodeVisitor):
                def __init__(self, file_path):
                    self.file_path = file_path
                    self.patterns = []
                    self.loop_depth = 0
                    self.current_function = None
                
                def visit_FunctionDef(self, node):
                    self.current_function = node.name
                    self.loop_depth = 0
                    self.generic_visit(node)
                    self.current_function = None
                
                def visit_For(self, node):
                    self.loop_depth += 1
                    if self.loop_depth >= 3:
                        pattern = DetectedPattern(
                            pattern_id='nested_loops',
                            pattern_type=PatternType.PERFORMANCE_ISSUE,
                            name='Bucles anidados profundos',
                            description=f'{self.loop_depth} niveles de bucles anidados',
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            severity=PatternSeverity.MEDIUM,
                            confidence=min(0.9, self.loop_depth / 4.0),
                            suggestion='Considerar optimizar algoritmo para reducir complejidad'
                        )
                        self.patterns.append(pattern)
                    
                    self.generic_visit(node)
                    self.loop_depth -= 1
                
                def visit_While(self, node):
                    self.loop_depth += 1
                    if self.loop_depth >= 3:
                        pattern = DetectedPattern(
                            pattern_id='nested_loops',
                            pattern_type=PatternType.PERFORMANCE_ISSUE,
                            name='Bucles anidados profundos',
                            description=f'{self.loop_depth} niveles de bucles anidados',
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            severity=PatternSeverity.MEDIUM,
                            confidence=min(0.9, self.loop_depth / 4.0),
                            suggestion='Considerar optimizar algoritmo para reducir complejidad'
                        )
                        self.patterns.append(pattern)
                    
                    self.generic_visit(node)
                    self.loop_depth -= 1
            
            visitor = NestedLoopVisitor(file_path)
            visitor.visit(tree)
            patterns = visitor.patterns
        
        return patterns
    
    def analyze_project_patterns(self, project_patterns: Dict[str, List[DetectedPattern]]) -> PatternSummary:
        """
        Analizar patrones a nivel de proyecto
        
        Args:
            project_patterns: Diccionario de archivo -> patrones detectados
            
        Returns:
            PatternSummary: Resumen de patrones del proyecto
        """
        summary = PatternSummary()
        
        # Contar total de patrones
        all_patterns = []
        for file_path, patterns in project_patterns.items():
            summary.total_patterns += len(patterns)
            summary.by_file[file_path] = len(patterns)
            all_patterns.extend(patterns)
        
        # Contar por tipo y severidad
        for pattern in all_patterns:
            # Por tipo
            if pattern.pattern_type not in summary.by_type:
                summary.by_type[pattern.pattern_type] = 0
            summary.by_type[pattern.pattern_type] += 1
            
            # Por severidad
            if pattern.severity not in summary.by_severity:
                summary.by_severity[pattern.severity] = 0
            summary.by_severity[pattern.severity] += 1
        
        # Encontrar patrones más comunes
        pattern_counts = {}
        for pattern in all_patterns:
            if pattern.name not in pattern_counts:
                pattern_counts[pattern.name] = 0
            pattern_counts[pattern.name] += 1
        
        summary.most_common = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10
        
        return summary
    
    def generate_pattern_report(self, project_patterns: Dict[str, List[DetectedPattern]],
                               summary: PatternSummary,
                               format: str = 'markdown') -> str:
        """
        Generar reporte de patrones
        
        Args:
            project_patterns: Patrones detectados por archivo
            summary: Resumen de patrones
            format: Formato del reporte
            
        Returns:
            str: Reporte generado
        """
        if format == 'markdown':
            return self._generate_markdown_report(project_patterns, summary)
        elif format == 'html':
            return self._generate_html_report(project_patterns, summary)
        elif format == 'json':
            return self._generate_json_report(project_patterns, summary)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def _generate_markdown_report(self, project_patterns: Dict[str, List[DetectedPattern]],
                                 summary: PatternSummary) -> str:
        """Generar reporte en formato Markdown"""
        report = []
        
        report.append("# Reporte de Patrones de Código")
        report.append("")
        
        # Resumen
        report.append("## Resumen General")
        report.append(f"**Total de patrones detectados:** {summary.total_patterns}")
        report.append(f"**Archivos analizados:** {len(project_patterns)}")
        report.append("")
        
        # Distribución por tipo
        report.append("### Distribución por Tipo")
        for pattern_type, count in sorted(summary.by_type.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- **{pattern_type.value}:** {count}")
        report.append("")
        
        # Distribución por severidad
        report.append("### Distribución por Severidad")
        for severity, count in sorted(summary.by_severity.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- **{severity.value}:** {count}")
        report.append("")
        
        # Patrones más comunes
        report.append("### Patrones Más Comunes")
        for pattern_name, count in summary.most_common:
            report.append(f"- **{pattern_name}:** {count} ocurrencias")
        report.append("")
        
        # Detalles por archivo
        report.append("## Detalles por Archivo")
        report.append("")
        
        for file_path, patterns in sorted(project_patterns.items()):
            if patterns:
                report.append(f"### {file_path}")
                report.append(f"**Total patrones:** {len(patterns)}")
                report.append("")
                
                # Agrupar por severidad
                patterns_by_severity = {}
                for pattern in patterns:
                    if pattern.severity not in patterns_by_severity:
                        patterns_by_severity[pattern.severity] = []
                    patterns_by_severity[pattern.severity].append(pattern)
                
                for severity, pattern_list in sorted(patterns_by_severity.items()):
                    report.append(f"#### Severidad: {severity.value}")
                    for pattern in pattern_list:
                        report.append(f"- **{pattern.name}** (líneas {pattern.line_start}-{pattern.line_end})")
                        report.append(f"  - {pattern.description}")
                        if pattern.suggestion:
                            report.append(f"  - *Sugerencia:* {pattern.suggestion}")
                    report.append("")
        
        return '\n'.join(report)
    
    def _generate_html_report(self, project_patterns: Dict[str, List[DetectedPattern]],
                             summary: PatternSummary) -> str:
        """Generar reporte en formato HTML"""
        html = []
        
        html.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Patrones de Código</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
                .file-section { margin-top: 20px; border: 1px solid #ddd; padding: 15px; }
                .pattern-item { margin: 10px 0; padding: 10px; border-left: 4px solid; }
                .severity-critical { border-left-color: #d32f2f; background-color: #ffebee; }
                .severity-high { border-left-color: #f57c00; background-color: #fff3e0; }
                .severity-medium { border-left-color: #ffa000; background-color: #fff8e1; }
                .severity-low { border-left-color: #689f38; background-color: #f1f8e9; }
                .severity-info { border-left-color: #0288d1; background-color: #e1f5fe; }
                .pattern-type { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }
                .design-pattern { background: #c8e6c9; color: #2e7d32; }
                .anti-pattern { background: #ffcdd2; color: #c62828; }
                .code-smell { background: #fff9c4; color: #f57f17; }
                .security-issue { background: #f8bbd0; color: #ad1457; }
                .performance-issue { background: #e1bee7; color: #6a1b9a; }
            </style>
        </head>
        <body>
        """)
        
        html.append("<h1>Reporte de Patrones de Código</h1>")
        
        # Resumen
        html.append("<div class='summary'>")
        html.append(f"<h2>Resumen General</h2>")
        html.append(f"<p><strong>Total de patrones detectados:</strong> {summary.total_patterns}</p>")
        html.append(f"<p><strong>Archivos analizados:</strong> {len(project_patterns)}</p>")
        
        html.append("<h3>Distribución por Tipo</h3>")
        html.append("<ul>")
        for pattern_type, count in sorted(summary.by_type.items(), key=lambda x: x[1], reverse=True):
            html.append(f"<li><strong>{pattern_type.value}:</strong> {count}</li>")
        html.append("</ul>")
        
        html.append("<h3>Distribución por Severidad</h3>")
        html.append("<ul>")
        for severity, count in sorted(summary.by_severity.items(), key=lambda x: x[1], reverse=True):
            html.append(f"<li><strong>{severity.value}:</strong> {count}</li>")
        html.append("</ul>")
        html.append("</div>")
        
        # Detalles por archivo
        html.append("<h2>Detalles por Archivo</h2>")
        
        for file_path, patterns in sorted(project_patterns.items()):
            if patterns:
                html.append(f"<div class='file-section'>")
                html.append(f"<h3>{file_path}</h3>")
                html.append(f"<p><strong>Total patrones:</strong> {len(patterns)}</p>")
                
                for pattern in patterns:
                    severity_class = f"severity-{pattern.severity.value}"
                    type_class = pattern.pattern_type.value.replace('_', '-')
                    
                    html.append(f"<div class='pattern-item {severity_class}'>")
                    html.append(f"<span class='pattern-type {type_class}'>{pattern.pattern_type.value}</span>")
                    html.append(f"<h4>{pattern.name}</h4>")
                    html.append(f"<p><strong>Líneas:</strong> {pattern.line_start}-{pattern.line_end}</p>")
                    html.append(f"<p><strong>Confianza:</strong> {pattern.confidence:.2f}</p>")
                    html.append(f"<p>{pattern.description}</p>")
                    if pattern.suggestion:
                        html.append(f"<p><em>Sugerencia:</em> {pattern.suggestion}</p>")
                    html.append("</div>")
                
                html.append("</div>")
        
        html.append("</body></html>")
        
        return '\n'.join(html)
    
    def _generate_json_report(self, project_patterns: Dict[str, List[DetectedPattern]],
                             summary: PatternSummary) -> str:
        """Generar reporte en formato JSON"""
        report_data = {
            'summary': {
                'total_patterns': summary.total_patterns,
                'files_analyzed': len(project_patterns),
                'by_type': {pt.value: count for pt, count in summary.by_type.items()},
                'by_severity': {s.value: count for s, count in summary.by_severity.items()},
                'most_common': summary.most_common
            },
            'files': {}
        }
        
        for file_path, patterns in project_patterns.items():
            file_data = {
                'total_patterns': len(patterns),
                'patterns': []
            }
            
            for pattern in patterns:
                pattern_data = {
                    'id': pattern.pattern_id,
                    'name': pattern.name,
                    'type': pattern.pattern_type.value,
                    'description': pattern.description,
                    'line_start': pattern.line_start,
                    'line_end': pattern.line_end,
                    'severity': pattern.severity.value,
                    'confidence': pattern.confidence,
                    'suggestion': pattern.suggestion
                }
                file_data['patterns'].append(pattern_data)
            
            report_data['files'][file_path] = file_data
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def suggest_refactoring(self, patterns: List[DetectedPattern]) -> List[Dict[str, Any]]:
        """
        Sugerir refactorizaciones basadas en patrones detectados
        
        Args:
            patterns: Lista de patrones detectados
            
        Returns:
            List[Dict[str, Any]]: Sugerencias de refactorización
        """
        suggestions = []
        
        # Agrupar patrones por tipo
        patterns_by_type = {}
        for pattern in patterns:
            if pattern.pattern_type not in patterns_by_type:
                patterns_by_type[pattern.pattern_type] = []
            patterns_by_type[pattern.pattern_type].append(pattern)
        
        # Sugerencias basadas en anti-patrones
        if PatternType.ANTI_PATTERN in patterns_by_type:
            anti_patterns = patterns_by_type[PatternType.ANTI_PATTERN]
            
            # God Class
            god_classes = [p for p in anti_patterns if 'God Class' in p.name]
            if god_classes:
                suggestions.append({
                    'type': 'refactor_god_class',
                    'priority': 'high',
                    'description': 'Clases Dios detectadas - demasiadas responsabilidades',
                    'details': {
                        'count': len(god_classes),
                        'classes': [p.context for p in god_classes if p.context],
                        'suggestion': 'Dividir clases grandes en clases más pequeñas con responsabilidades únicas'
                    }
                })
            
            # Long Method
            long_methods = [p for p in anti_patterns if 'Long Method' in p.name]
            if long_methods:
                suggestions.append({
                    'type': 'refactor_long_methods',
                    'priority': 'medium',
                    'description': 'Métodos/Funciones demasiado largos',
                    'details': {
                        'count': len(long_methods),
                        'suggestion': 'Extraer porciones de código en funciones/métodos auxiliares'
                    }
                })
            
            # Duplicate Code
            duplicate_code = [p for p in anti_patterns if 'Duplicate Code' in p.name]
            if duplicate_code:
                suggestions.append({
                    'type': 'eliminate_duplicate_code',
                    'priority': 'medium',
                    'description': 'Código duplicado detectado',
                    'details': {
                        'count': len(duplicate_code),
                        'suggestion': 'Extraer código duplicado en funciones/métodos comunes'
                    }
                })
        
        # Sugerencias basadas en code smells
        if PatternType.CODE_SMELL in patterns_by_type:
            code_smells = patterns_by_type[PatternType.CODE_SMELL]
            
            # Magic Numbers
            magic_numbers = [p for p in code_smells if 'Magic Number' in p.name]
            if magic_numbers:
                suggestions.append({
                    'type': 'replace_magic_numbers',
                    'priority': 'low',
                    'description': 'Números mágicos detectados',
                    'details': {
                        'count': len(magic_numbers),
                        'suggestion': 'Definir constantes con nombres significativos'
                    }
                })
            
            # Deep Nesting
            deep_nesting = [p for p in code_smells if 'Deep Nesting' in p.name]
            if deep_nesting:
                suggestions.append({
                    'type': 'reduce_nesting',
                    'priority': 'medium',
                    'description': 'Anidamiento profundo detectado',
                    'details': {
                        'count': len(deep_nesting),
                        'suggestion': 'Usar early returns o extraer código en funciones auxiliares'
                    }
                })
        
        # Sugerencias basadas en issues de seguridad
        if PatternType.SECURITY_ISSUE in patterns_by_type:
            security_issues = patterns_by_type[PatternType.SECURITY_ISSUE]
            
            # SQL Injection
            sql_injections = [p for p in security_issues if 'SQL Injection' in p.name]
            if sql_injections:
                suggestions.append({
                    'type': 'fix_sql_injection',
                    'priority': 'critical',
                    'description': 'Posibles inyecciones SQL detectadas',
                    'details': {
                        'count': len(sql_injections),
                        'suggestion': 'Usar consultas parametrizadas o prepared statements'
                    }
                })
            
            # Hardcoded Passwords
            hardcoded_passwords = [p for p in security_issues if 'Hardcoded Password' in p.name]
            if hardcoded_passwords:
                suggestions.append({
                    'type': 'remove_hardcoded_passwords',
                    'priority': 'high',
                    'description': 'Contraseñas/Tokens hardcodeados detectados',
                    'details': {
                        'count': len(hardcoded_passwords),
                        'suggestion': 'Usar variables de entorno o sistemas de gestión de secretos'
                    }
                })
        
        # Sugerencias basadas en issues de performance
        if PatternType.PERFORMANCE_ISSUE in patterns_by_type:
            performance_issues = patterns_by_type[PatternType.PERFORMANCE_ISSUE]
            
            # Nested Loops
            nested_loops = [p for p in performance_issues if 'Nested Loops' in p.name]
            if nested_loops:
                suggestions.append({
                    'type': 'optimize_nested_loops',
                    'priority': 'medium',
                    'description': 'Bucles anidados detectados - posible O(n²)',
                    'details': {
                        'count': len(nested_loops),
                        'suggestion': 'Considerar optimizar algoritmo o usar estructuras de datos más eficientes'
                    }
                })
        
        return suggestions