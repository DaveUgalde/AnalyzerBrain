#!/usr/bin/env python3
"""
ANALIZADOR DE CÓDIGO DE PRIMERA CATEGORÍA - v2.1
Versión corregida con serialización JSON y manejo de errores AST.
"""

from __future__ import annotations

import ast
import argparse
import json
import hashlib
import yaml
import toml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Tuple, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
import statistics
import time
import sys
import re
from enum import Enum, auto
from datetime import datetime

# ============================================================
# CONFIGURACIÓN AVANZADA
# ============================================================

class AnalysisContext(Enum):
    """Contextos de análisis para aplicar reglas específicas."""
    CONFIGURATION = auto()
    API = auto()
    DATABASE = auto()
    BUSINESS_LOGIC = auto()
    UTILITY = auto()
    TEST = auto()
    DATA_MODEL = auto()
    SERVICE = auto()
    VALIDATOR = auto()

class CodeQualityTier(Enum):
    """Niveles de calidad de código."""
    EXCEPTIONAL = 95
    EXCELLENT = 85
    GOOD = 70
    ADEQUATE = 50
    NEEDS_IMPROVEMENT = 30
    PROBLEMATIC = 15
    CRITICAL = 0

# Umbrales contextuales
CONTEXTUAL_THRESHOLDS = {
    AnalysisContext.CONFIGURATION: {
        'max_lines': 800,
        'max_complexity': 15,
        'min_cohesion': 60,
        'max_coupling': 40
    },
    AnalysisContext.API: {
        'max_lines': 400,
        'max_complexity': 10,
        'min_cohesion': 70,
        'max_coupling': 30
    },
    AnalysisContext.BUSINESS_LOGIC: {
        'max_lines': 300,
        'max_complexity': 8,
        'min_cohesion': 80,
        'max_coupling': 20
    },
    AnalysisContext.UTILITY: {
        'max_lines': 200,
        'max_complexity': 6,
        'min_cohesion': 90,
        'max_coupling': 10
    }
}

# Librerías externas comunes
EXTERNAL_LIBRARIES = {
    'os', 'sys', 'json', 'time', 'datetime', 'math', 're',
    'typing', 'pathlib', 'hashlib', 'argparse', 'dataclasses',
    'collections', 'itertools', 'functools', 'threading',
    'multiprocessing', 'concurrent', 'asyncio', 'logging'
}

# ============================================================
# MODELOS DE DATOS AVANZADOS (MEJORADOS PARA SERIALIZACIÓN)
# ============================================================

@dataclass
class FunctionMetrics:
    """Métricas avanzadas de función."""
    name: str
    lineno: int
    end_lineno: int
    complexity: int
    lines: int
    parameters: int
    returns: int
    nesting_level: int
    has_docstring: bool
    calls: List[str]
    decorators: List[str]
    is_async: bool
    is_generator: bool
    cognitive_complexity: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return asdict(self)

@dataclass
class ClassMetrics:
    """Métricas avanzadas de clase."""
    name: str
    lineno: int
    methods: List[FunctionMetrics]
    attributes: List[str]
    inheritance: List[str]
    has_docstring: bool
    is_abstract: bool
    is_dataclass: bool
    is_exception: bool
    class_variables: int
    instance_variables: int
    property_count: int
    static_methods: int
    class_methods: int
    lcom4: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        result = asdict(self)
        result['methods'] = [m.to_dict() for m in self.methods]
        return result

@dataclass
class ModuleDependencies:
    """Dependencias de módulo."""
    imports: Set[str]
    import_from: Set[str]
    external_deps: Set[str]
    internal_deps: Set[str]
    relative_imports: Set[str]
    cyclic_dependencies: List[Tuple[str, str]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            'imports': list(self.imports),
            'import_from': list(self.import_from),
            'external_deps': list(self.external_deps),
            'internal_deps': list(self.internal_deps),
            'relative_imports': list(self.relative_imports),
            'cyclic_dependencies': self.cyclic_dependencies
        }

@dataclass
class FileMetrics:
    """Métricas completas de archivo."""
    path: str
    module: str
    lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    docstring_lines: int
    functions: List[FunctionMetrics]
    classes: List[ClassMetrics]
    dependencies: ModuleDependencies
    file_hash: str
    encoding: str
    line_endings: str
    
    # Métricas calculadas
    comment_ratio: float = 0.0
    function_density: float = 0.0
    class_density: float = 0.0
    maintainability_index: float = 0.0
    
    # Contexto
    contexts: List[str] = field(default_factory=list)
    design_patterns: List[str] = field(default_factory=list)
    solid_principles: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            'path': self.path,
            'module': self.module,
            'lines': self.lines,
            'code_lines': self.code_lines,
            'comment_lines': self.comment_lines,
            'blank_lines': self.blank_lines,
            'docstring_lines': self.docstring_lines,
            'functions': [f.to_dict() for f in self.functions],
            'classes': [c.to_dict() for c in self.classes],
            'dependencies': self.dependencies.to_dict(),
            'file_hash': self.file_hash,
            'encoding': self.encoding,
            'line_endings': self.line_endings,
            'comment_ratio': self.comment_ratio,
            'function_density': self.function_density,
            'class_density': self.class_density,
            'maintainability_index': self.maintainability_index,
            'contexts': self.contexts,
            'design_patterns': self.design_patterns,
            'solid_principles': self.solid_principles
        }

@dataclass
class QualityScore:
    """Puntaje de calidad multidimensional."""
    technical: float
    architectural: float
    performance: float
    security: float
    testability: float
    maintainability: float
    documentation: float
    composite: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'technical': self.technical,
            'architectural': self.architectural,
            'performance': self.performance,
            'security': self.security,
            'testability': self.testability,
            'maintainability': self.maintainability,
            'documentation': self.documentation,
            'composite': self.composite
        }

# ============================================================
# ANALIZADOR AST AVANZADO (CORREGIDO)
# ============================================================

class AdvancedASTAnalyzer(ast.NodeVisitor):
    """Analizador AST avanzado con métricas sofisticadas."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.functions: List[FunctionMetrics] = []
        self.classes: List[ClassMetrics] = []
        self.imports: Set[str] = set()
        self.import_from: Set[str] = set()
        self.relative_imports: Set[str] = set()
        self.current_class: Optional[str] = None
        self.nesting_level = 0
        
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self.imports.add(module_name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            module_name = node.module.split('.')[0]
            self.import_from.add(module_name)
            
            # Detectar imports relativos
            if node.level > 0:
                self.relative_imports.add(module_name)
        
        self.generic_visit(node)
    
    def _calculate_complexity(self, node: ast.AST) -> Tuple[int, int]:
        """Calcula complejidad ciclomática y cognitiva."""
        complexity = 1
        cognitive_complexity = 0
        
        # Tipos de nodos que incrementan complejidad
        complexity_nodes = (
            ast.If, ast.While, ast.For, ast.AsyncFor,
            ast.Try, ast.ExceptHandler, ast.With, ast.AsyncWith,
            ast.BoolOp
        )
        
        for child in ast.walk(node):
            if isinstance(child, complexity_nodes):
                complexity += 1
                # Incrementar complejidad cognitiva por anidamiento
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    cognitive_complexity += 1
        
        return complexity, cognitive_complexity
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Calcular métricas básicas
        params = len(node.args.args) + len(node.args.kwonlyargs)
        returns = sum(1 for n in ast.walk(node) if isinstance(n, ast.Return))
        has_docstring = ast.get_docstring(node) is not None
        
        # Calcular complejidades
        complexity, cognitive_complexity = self._calculate_complexity(node)
        
        # Extraer decoradores
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(decorator.attr)
        
        # Extraer llamadas
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        
        func_metrics = FunctionMetrics(
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            complexity=complexity,
            lines=(node.end_lineno or node.lineno) - node.lineno + 1,
            parameters=params,
            returns=returns,
            nesting_level=self.nesting_level,
            has_docstring=has_docstring,
            calls=calls,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_generator=any(isinstance(n, ast.Yield) or isinstance(n, ast.YieldFrom) 
                           for n in ast.walk(node)),
            cognitive_complexity=cognitive_complexity
        )
        
        self.functions.append(func_metrics)
        
        # Visitar hijos con contexto aumentado
        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
    
    def visit_ClassDef(self, node: ast.ClassDef):
        # Métricas de clase básicas
        inheritance = [base.id for base in node.bases if isinstance(base, ast.Name)]
        has_docstring = ast.get_docstring(node) is not None
        
        # Detectar tipo de clase
        is_abstract = any(isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod'
                         for decorator in node.decorator_list)
        is_dataclass = any(isinstance(decorator, ast.Name) and decorator.id == 'dataclass'
                          for decorator in node.decorator_list)
        is_exception = 'Exception' in inheritance or 'Error' in node.name
        
        # Extraer atributos
        attributes = []
        class_vars = 0
        instance_vars = 0
        
        for child in node.body:
            if isinstance(child, ast.AnnAssign):
                if isinstance(child.target, ast.Name):
                    attributes.append(child.target.id)
                    if child.value is None:
                        class_vars += 1
                    else:
                        instance_vars += 1
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
                        instance_vars += 1
        
        # Contar propiedades y métodos especiales
        property_count = 0
        static_methods = 0
        class_methods = 0
        
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                for decorator in child.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id == 'property':
                            property_count += 1
                        elif decorator.id == 'staticmethod':
                            static_methods += 1
                        elif decorator.id == 'classmethod':
                            class_methods += 1
        
        # Guardar contexto temporal
        prev_class = self.current_class
        self.current_class = node.name
        
        # Visitar métodos
        self.generic_visit(node)
        
        # Filtrar métodos de esta clase
        class_methods_list = []
        for func in self.functions:
            if func.lineno >= node.lineno and func.end_lineno <= (node.end_lineno or node.lineno):
                class_methods_list.append(func)
        
        # Calcular LCOM4 (Lack of Cohesion of Methods) simplificado
        lcom4 = self._calculate_lcom4_simple(class_methods_list, attributes)
        
        class_metrics = ClassMetrics(
            name=node.name,
            lineno=node.lineno,
            methods=class_methods_list,
            attributes=attributes,
            inheritance=inheritance,
            has_docstring=has_docstring,
            is_abstract=is_abstract,
            is_dataclass=is_dataclass,
            is_exception=is_exception,
            class_variables=class_vars,
            instance_variables=instance_vars,
            property_count=property_count,
            static_methods=static_methods,
            class_methods=class_methods,
            lcom4=lcom4
        )
        
        self.classes.append(class_metrics)
        self.current_class = prev_class
    
    def _calculate_lcom4_simple(self, methods: List[FunctionMetrics], 
                               attributes: List[str]) -> float:
        """Calcula LCOM4 simplificado."""
        if not methods or not attributes:
            return 0.0
        
        # Contar métodos que acceden a cada atributo
        attribute_access = {attr: 0 for attr in attributes}
        
        for method in methods:
            for call in method.calls:
                if call in attributes:
                    attribute_access[call] += 1
        
        # Métodos que no comparten atributos
        total_methods = len(methods)
        if total_methods <= 1:
            return 0.0
        
        # Calcular métrica simplificada
        shared_attributes = sum(1 for count in attribute_access.values() if count > 1)
        total_pairs = total_methods * (total_methods - 1) / 2
        
        if total_pairs == 0:
            return 0.0
        
        # LCOM4 simplificado: porcentaje de pares que no comparten atributos
        lcom4 = (1 - (shared_attributes / total_pairs)) * 100 if total_pairs > 0 else 100
        return min(100, max(0, lcom4))

# ============================================================
# SISTEMA DE PUNTAJES CONTEXTUAL (SIMPLIFICADO)
# ============================================================

class ContextualScoringSystem:
    """Sistema de puntuación que considera contexto."""
    
    @staticmethod
    def calculate_technical_score(file_metrics: FileMetrics) -> float:
        """Calcula puntaje técnico contextual."""
        base_score = 100.0
        
        if not file_metrics.functions:
            return 90.0  # Archivos sin funciones tienen buena puntuación por defecto
        
        # Penalizaciones por complejidad
        for func in file_metrics.functions:
            # Penalizar funciones muy complejas
            if func.complexity > 10:
                base_score -= (func.complexity - 10) * 2
            
            # Penalizar funciones muy largas
            if func.lines > 50:
                base_score -= (func.lines - 50) * 0.5
            
            # Penalizar anidamiento profundo
            if func.nesting_level > 3:
                base_score -= (func.nesting_level - 3) * 3
        
        # Bonus por documentación
        documented_funcs = sum(1 for f in file_metrics.functions if f.has_docstring)
        if file_metrics.functions:
            doc_ratio = documented_funcs / len(file_metrics.functions)
            if doc_ratio > 0.8:
                base_score += 10
            elif doc_ratio < 0.3:
                base_score -= 20
        
        # Ajustar por contexto
        if 'CONFIGURATION' in file_metrics.contexts:
            base_score += 5  # Archivos de configuración tienen más tolerancia
        
        return max(0, min(100, base_score))
    
    @staticmethod
    def calculate_architectural_score(file_metrics: FileMetrics) -> float:
        """Calcula puntaje arquitectónico."""
        base_score = 100.0
        
        # Cohesión (LCOM4)
        if file_metrics.classes:
            avg_lcom4 = statistics.mean(cls.lcom4 for cls in file_metrics.classes)
            if avg_lcom4 > 50:  # Baja cohesión
                base_score -= (avg_lcom4 - 50) * 0.5
        
        # Acoplamiento
        coupling = len(file_metrics.dependencies.internal_deps)
        if coupling > 10:
            base_score -= (coupling - 10) * 2
        
        # Herencia apropiada
        for cls in file_metrics.classes:
            if len(cls.inheritance) > 2:  # Herencia múltiple excesiva
                base_score -= 5
        
        # Bonus por principios SOLID
        if file_metrics.solid_principles:
            base_score += len(file_metrics.solid_principles) * 3
        
        return max(0, min(100, base_score))
    
    @staticmethod
    def calculate_security_score(file_metrics: FileMetrics) -> float:
        """Calcula puntaje de seguridad."""
        base_score = 100.0
        
        security_issues = 0
        
        for func in file_metrics.functions:
            func_calls = ' '.join(func.calls).lower()
            
            # Patrones inseguros
            insecure_patterns = ['eval(', 'exec(', 'pickle.loads', 'yaml.load']
            if any(pattern in func_calls for pattern in insecure_patterns):
                security_issues += 10
        
        base_score -= security_issues
        return max(0, min(100, base_score))

# ============================================================
# ANALIZADOR PRINCIPAL (SIMPLIFICADO Y CORREGIDO)
# ============================================================

class EliteCodeAnalyzer:
    """Analizador de código de primera categoría."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_cache: Dict[str, FileMetrics] = {}
        
        # Estadísticas
        self.stats = {
            'files_analyzed': 0,
            'errors': 0,
            'start_time': 0,
            'end_time': 0
        }
    
    def analyze_project(self, root_path: Path, 
                       max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Analiza un proyecto completo."""
        self.stats['start_time'] = time.time()
        
        print(f"🚀 Iniciando análisis de élite: {root_path}")
        
        # Descubrir archivos
        all_files = self._discover_files(root_path)
        print(f"📁 Encontrados {len(all_files)} archivos para analizar")
        
        if not all_files:
            print("⚠️  No se encontraron archivos para analizar")
            return {}
        
        # Analizar archivos en paralelo
        print("🔬 Analizando archivos...")
        file_metrics_list = self._analyze_files_parallel(all_files, max_workers)
        
        # Calcular métricas de proyecto
        print("📈 Calculando métricas globales...")
        project_metrics = self._calculate_project_metrics(file_metrics_list)
        
        # Evaluar calidad
        print("🏆 Evaluando calidad...")
        quality_assessment = self._assess_quality(file_metrics_list)
        
        # Generar reportes
        print("📊 Generando reportes...")
        reports = self._generate_reports(file_metrics_list, project_metrics, quality_assessment)
        
        self.stats['end_time'] = time.time()
        elapsed = self.stats['end_time'] - self.stats['start_time']
        
        print(f"✅ Análisis completado en {elapsed:.2f} segundos")
        print(f"📊 Archivos analizados: {self.stats['files_analyzed']}")
        print(f"⚠️  Errores: {self.stats['errors']}")
        
        return {
            'project_metrics': project_metrics,
            'quality_assessment': quality_assessment,
            'reports': reports,
            'statistics': self.stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _discover_files(self, root: Path) -> List[Path]:
        """Descubre todos los archivos relevantes."""
        include_patterns = ['*.py', '*.yaml', '*.yml', '*.json', '*.toml']
        
        exclude_dirs = {
            '.git', '__pycache__', '.pytest_cache',
            '.venv', 'venv', 'env', 'node_modules',
            'build', 'dist'
        }
        
        files = []
        for pattern in include_patterns:
            for file in root.rglob(pattern):
                # Excluir directorios no deseados
                if any(excluded in file.parts for excluded in exclude_dirs):
                    continue
                
                # Excluir archivos muy grandes
                try:
                    if file.stat().st_size > 5 * 1024 * 1024:
                        continue
                except OSError:
                    continue
                
                files.append(file)
        
        return files
    
    def _analyze_files_parallel(self, files: List[Path], 
                               max_workers: Optional[int]) -> List[FileMetrics]:
        """Analiza archivos en paralelo."""
        file_metrics_list = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._analyze_single_file, f): f for f in files}
            
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    if result:
                        file_metrics_list.append(result)
                        self.stats['files_analyzed'] += 1
                except Exception as e:
                    file_path = futures[future]
                    print(f"⚠️  Error analizando {file_path.name}: {str(e)[:100]}...")
                    self.stats['errors'] += 1
                
                if i % 5 == 0 or i == len(files):
                    print(f"  📊 Progreso: {i}/{len(files)} archivos analizados")
        
        return file_metrics_list
    
    def _analyze_single_file(self, file_path: Path) -> Optional[FileMetrics]:
        """Analiza un solo archivo."""
        try:
            # Leer contenido
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Determinar contexto basado en nombre y extensión
            contexts = self._determine_context(file_path, content)
            
            # Análisis basado en tipo de archivo
            if file_path.suffix == '.py':
                return self._analyze_python_file(file_path, content, contexts)
            else:
                return self._analyze_config_file(file_path, content, contexts)
                
        except Exception as e:
            print(f"⚠️  Error en análisis de {file_path.name}: {e}")
            return None
    
    def _determine_context(self, file_path: Path, content: str) -> List[str]:
        """Determina el contexto del archivo."""
        contexts = []
        file_name = file_path.name.lower()
        
        # Configuración
        if file_name.endswith(('.yaml', '.yml', '.json', '.toml', '.ini')):
            contexts.append('CONFIGURATION')
        
        # Python
        if file_path.suffix == '.py':
            # Buscar patrones en el contenido
            if 'class' in content and 'def' in content:
                if any(keyword in content.lower() for keyword in ['api', 'route', 'endpoint']):
                    contexts.append('API')
                elif any(keyword in content.lower() for keyword in ['model', 'schema', 'database']):
                    contexts.append('DATA_MODEL')
                else:
                    contexts.append('BUSINESS_LOGIC')
            elif 'def' in content:
                contexts.append('UTILITY')
        
        return contexts
    
    def _analyze_python_file(self, file_path: Path, content: str, 
                            contexts: List[str]) -> FileMetrics:
        """Analiza archivo Python."""
        # Métricas básicas de líneas
        lines = content.splitlines()
        total_lines = len(lines)
        
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        docstring_lines = 0
        
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                docstring_lines += 1
                continue
            
            if in_docstring:
                docstring_lines += 1
                continue
            
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            else:
                code_lines += 1
        
        # Análisis AST
        try:
            tree = ast.parse(content)
            analyzer = AdvancedASTAnalyzer(file_path)
            analyzer.visit(tree)
        except SyntaxError as e:
            print(f"⚠️  Error de sintaxis en {file_path.name}: {e}")
            analyzer = AdvancedASTAnalyzer(file_path)
        except Exception as e:
            print(f"⚠️  Error AST en {file_path.name}: {e}")
            analyzer = AdvancedASTAnalyzer(file_path)
        
        # Separar dependencias
        all_imports = analyzer.imports.union(analyzer.import_from)
        external_deps = {imp for imp in all_imports 
                        if imp.split('.')[0] in EXTERNAL_LIBRARIES}
        internal_deps = all_imports - external_deps
        
        # Calcular métricas adicionales
        comment_ratio = comment_lines / max(total_lines, 1)
        function_density = len(analyzer.functions) / max(code_lines, 1)
        class_density = len(analyzer.classes) / max(code_lines, 1)
        
        # Calcular índice de mantenibilidad (simplificado)
        avg_complexity = (statistics.mean([f.complexity for f in analyzer.functions]) 
                         if analyzer.functions else 0)
        maintainability_index = min(100, max(0, 171 - 5.2 * avg_complexity - 0.23 * code_lines / 1000))
        
        # Detectar patrones simples
        design_patterns = []
        for cls in analyzer.classes:
            if '_instance' in cls.attributes or 'get_instance' in [m.name for m in cls.methods]:
                design_patterns.append('SINGLETON')
            if 'Factory' in cls.name or any('create' in m.name.lower() for m in cls.methods):
                design_patterns.append('FACTORY')
        
        # Obtener módulo
        try:
            rel_path = file_path.relative_to(file_path.parents[1])
            module_name = str(rel_path.parent).replace('/', '.').replace('\\', '.')
        except:
            module_name = ''
        
        return FileMetrics(
            path=str(file_path),
            module=module_name,
            lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            docstring_lines=docstring_lines,
            functions=analyzer.functions,
            classes=analyzer.classes,
            dependencies=ModuleDependencies(
                imports=analyzer.imports,
                import_from=analyzer.import_from,
                external_deps=external_deps,
                internal_deps=internal_deps,
                relative_imports=analyzer.relative_imports,
                cyclic_dependencies=[]
            ),
            file_hash=hashlib.md5(content.encode()).hexdigest(),
            encoding='utf-8',
            line_endings='\n' if '\r\n' not in content else '\r\n',
            comment_ratio=comment_ratio,
            function_density=function_density,
            class_density=class_density,
            maintainability_index=maintainability_index,
            contexts=contexts,
            design_patterns=design_patterns,
            solid_principles=['SINGLE_RESPONSIBILITY']  # Simplificado
        )
    
    def _analyze_config_file(self, file_path: Path, content: str,
                            contexts: List[str]) -> FileMetrics:
        """Analiza archivo de configuración."""
        lines = content.splitlines()
        total_lines = len(lines)
        
        # Contar líneas significativas
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        blank_lines = sum(1 for line in lines if not line.strip())
        
        # Análisis básico de contenido
        has_env_vars = bool(re.search(r'\$\{.+?\}|\$[A-Z_][A-Z0-9_]*', content))
        
        return FileMetrics(
            path=str(file_path),
            module='',
            lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            docstring_lines=0,
            functions=[],
            classes=[],
            dependencies=ModuleDependencies(
                imports=set(),
                import_from=set(),
                external_deps=set(),
                internal_deps=set(),
                relative_imports=set(),
                cyclic_dependencies=[]
            ),
            file_hash=hashlib.md5(content.encode()).hexdigest(),
            encoding='utf-8',
            line_endings='\n' if '\r\n' not in content else '\r\n',
            comment_ratio=comment_lines / max(total_lines, 1),
            function_density=0,
            class_density=0,
            maintainability_index=85,  # Archivos de configuración suelen ser mantenibles
            contexts=contexts,
            design_patterns=[],
            solid_principles=[]
        )
    
    def _calculate_project_metrics(self, file_metrics_list: List[FileMetrics]) -> Dict[str, Any]:
        """Calcula métricas globales del proyecto."""
        if not file_metrics_list:
            return {}
        
        # Métricas agregadas
        total_files = len(file_metrics_list)
        total_lines = sum(f.lines for f in file_metrics_list)
        total_functions = sum(len(f.functions) for f in file_metrics_list)
        total_classes = sum(len(f.classes) for f in file_metrics_list)
        
        # Distribución por contexto
        context_distribution = defaultdict(int)
        for metrics in file_metrics_list:
            for context in metrics.contexts:
                context_distribution[context] += 1
        
        # Complejidad promedio
        all_functions = [func for metrics in file_metrics_list for func in metrics.functions]
        avg_complexity = (statistics.mean([f.complexity for f in all_functions]) 
                         if all_functions else 0)
        
        return {
            'total_files': total_files,
            'total_lines': total_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'context_distribution': dict(context_distribution),
            'average_complexity': round(avg_complexity, 2),
            'file_types': {
                'python': sum(1 for f in file_metrics_list if f.path.endswith('.py')),
                'yaml': sum(1 for f in file_metrics_list if f.path.endswith(('.yaml', '.yml'))),
                'json': sum(1 for f in file_metrics_list if f.path.endswith('.json')),
                'toml': sum(1 for f in file_metrics_list if f.path.endswith('.toml'))
            }
        }
    
    def _assess_quality(self, file_metrics_list: List[FileMetrics]) -> Dict[str, Any]:
        """Evalúa la calidad del proyecto."""
        if not file_metrics_list:
            return {}
        
        quality_scores = []
        
        for metrics in file_metrics_list:
            # Calcular puntajes individuales
            technical_score = ContextualScoringSystem.calculate_technical_score(metrics)
            architectural_score = ContextualScoringSystem.calculate_architectural_score(metrics)
            security_score = ContextualScoringSystem.calculate_security_score(metrics)
            
            # Puntajes adicionales
            testability_score = 80  # Estimado
            maintainability_score = metrics.maintainability_index
            # Calcular correctamente el ratio de documentación
            documentation_score = (metrics.comment_lines + metrics.docstring_lines) / max(metrics.lines, 1) * 100
            documentation_score = min(100, documentation_score)  # Cap at 100
            
            # Puntaje de rendimiento (simplificado)
            performance_score = 85  # Estimado base
            
            # Puntaje compuesto (ponderado)
            composite_score = (
                technical_score * 0.25 +
                architectural_score * 0.20 +
                performance_score * 0.15 +
                security_score * 0.15 +
                testability_score * 0.10 +
                maintainability_score * 0.10 +
                documentation_score * 0.05
            )
            
            quality_score = QualityScore(
                technical=round(technical_score, 2),
                architectural=round(architectural_score, 2),
                performance=round(performance_score, 2),
                security=round(security_score, 2),
                testability=round(testability_score, 2),
                maintainability=round(maintainability_score, 2),
                documentation=round(documentation_score, 2),
                composite=round(composite_score, 2)
            )
            
            quality_scores.append({
                'file': metrics.path,
                'scores': quality_score.to_dict(),
                'contexts': metrics.contexts,
                'design_patterns': metrics.design_patterns,
                'solid_principles': metrics.solid_principles,
                'issues': self._identify_issues(metrics)
            })
        
        # Calcular promedios del proyecto
        avg_scores = {}
        if quality_scores:
            for key in ['technical', 'architectural', 'performance', 'security', 
                       'testability', 'maintainability', 'documentation', 'composite']:
                values = [qs['scores'][key] for qs in quality_scores]
                avg_scores[key] = round(statistics.mean(values), 2)
        
        # Clasificar archivos por calidad
        exceptional_files = [qs for qs in quality_scores 
                           if qs['scores']['composite'] >= 95]
        problematic_files = [qs for qs in quality_scores 
                           if qs['scores']['composite'] < 50]
        
        return {
            'file_assessments': quality_scores,
            'project_averages': avg_scores,
            'exceptional_files': len(exceptional_files),
            'problematic_files': len(problematic_files),
            'quality_tier': self._determine_quality_tier(avg_scores.get('composite', 0))
        }
    
    def _identify_issues(self, metrics: FileMetrics) -> List[str]:
        """Identifica problemas en el archivo."""
        issues = []
        
        # Tamaño excesivo
        if metrics.lines > 500 and 'CONFIGURATION' not in metrics.contexts:
            issues.append(f"Archivo muy grande ({metrics.lines} líneas)")
        
        # Funciones complejas
        for func in metrics.functions:
            if func.complexity > 10:
                issues.append(f"Función '{func.name}' muy compleja ({func.complexity})")
            if func.lines > 50:
                issues.append(f"Función '{func.name}' muy larga ({func.lines} líneas)")
        
        # Documentación insuficiente
        if metrics.comment_ratio < 0.1:
            issues.append("Documentación insuficiente")
        
        return issues
    
    def _determine_quality_tier(self, composite_score: float) -> str:
        """Determina el nivel de calidad."""
        for tier in CodeQualityTier:
            if composite_score >= tier.value:
                return tier.name.replace('_', ' ').title()
        return "Unknown"
    
    def _generate_reports(self, file_metrics_list: List[FileMetrics],
                         project_metrics: Dict[str, Any],
                         quality_assessment: Dict[str, Any]) -> Dict[str, str]:
        """Genera reportes en diferentes formatos."""
        reports = {}
        
        # Reporte ejecutivo
        reports['executive'] = self._generate_executive_report(
            project_metrics, quality_assessment
        )
        
        # Reporte técnico
        reports['technical'] = self._generate_technical_report(
            file_metrics_list, quality_assessment
        )
        
        # JSON completo (ahora serializable)
        reports['json'] = json.dumps({
            'project_metrics': project_metrics,
            'quality_assessment': quality_assessment,
            'statistics': self.stats,
            'timestamp': datetime.now().isoformat()
        }, indent=2, default=str, ensure_ascii=False)
        
        return reports
    
    def _generate_executive_report(self, project_metrics: Dict[str, Any],
                                  quality_assessment: Dict[str, Any]) -> str:
        """Genera reporte ejecutivo."""
        lines = []
        
        lines.append("=" * 80)
        lines.append("📊 REPORTE EJECUTIVO - ANÁLISIS DE CÓDIGO")
        lines.append("=" * 80)
        lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Resumen del proyecto
        lines.append("📈 RESUMEN DEL PROYECTO")
        lines.append("-" * 40)
        lines.append(f"• Archivos analizados: {project_metrics.get('total_files', 0)}")
        lines.append(f"• Líneas totales: {project_metrics.get('total_lines', 0):,}")
        lines.append(f"• Funciones: {project_metrics.get('total_functions', 0)}")
        lines.append(f"• Clases: {project_metrics.get('total_classes', 0)}")
        lines.append("")
        
        # Puntajes promedio
        avg_scores = quality_assessment.get('project_averages', {})
        lines.append("🏆 PUNTAJES DE CALIDAD")
        lines.append("-" * 40)
        
        for dimension, score in avg_scores.items():
            tier = self._determine_quality_tier(score)
            lines.append(f"• {dimension.title()}: {score:.1f} ({tier})")
        
        lines.append("")
        
        # Hallazgos clave
        lines.append("🔍 HALLAZGOS CLAVE")
        lines.append("-" * 40)
        lines.append(f"• Archivos excepcionales: {quality_assessment.get('exceptional_files', 0)}")
        lines.append(f"• Archivos problemáticos: {quality_assessment.get('problematic_files', 0)}")
        lines.append(f"• Nivel de calidad: {quality_assessment.get('quality_tier', 'Unknown')}")
        lines.append("")
        
        # Recomendaciones
        lines.append("🚀 RECOMENDACIONES PRIORITARIAS")
        lines.append("-" * 40)
        lines.append("1. Revisar archivos problemáticos identificados")
        lines.append("2. Mejorar documentación en módulos críticos")
        lines.append("3. Optimizar funciones con alta complejidad")
        lines.append("4. Implementar pruebas unitarias")
        lines.append("5. Revisar arquitectura de módulos complejos")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_technical_report(self, file_metrics_list: List[FileMetrics],
                                  quality_assessment: Dict[str, Any]) -> str:
        """Genera reporte técnico detallado."""
        lines = []
        
        lines.append("=" * 80)
        lines.append("🔧 REPORTE TÉCNICO DETALLADO")
        lines.append("=" * 80)
        lines.append("")
        
        # Top 5 funciones más complejas
        all_functions = []
        for metrics in file_metrics_list:
            for func in metrics.functions:
                all_functions.append({
                    'file': Path(metrics.path).name,
                    'function': func.name,
                    'complexity': func.complexity,
                    'lines': func.lines
                })
        
        if all_functions:
            lines.append("🔴 TOP 5 FUNCIONES MÁS COMPLEJAS")
            lines.append("-" * 40)
            
            top_complex = sorted(all_functions, key=lambda x: x['complexity'], reverse=True)[:5]
            for i, func in enumerate(top_complex, 1):
                lines.append(f"{i}. {func['function']} ({func['file']})")
                lines.append(f"   Complejidad: {func['complexity']}, Líneas: {func['lines']}")
        
        lines.append("")
        
        # Archivos problemáticos
        problematic = quality_assessment.get('file_assessments', [])
        problematic = [p for p in problematic if p['scores']['composite'] < 70]
        
        if problematic:
            lines.append("⚠️  ARCHIVOS QUE NECESITAN ATENCIÓN")
            lines.append("-" * 40)
            
            for i, file_info in enumerate(problematic[:5], 1):
                file_name = Path(file_info['file']).name
                score = file_info['scores']['composite']
                lines.append(f"{i}. {file_name} - Puntaje: {score:.1f}")
                if file_info['issues']:
                    lines.append(f"   Problemas: {', '.join(file_info['issues'][:2])}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)

# ============================================================
# INTERFAZ DE LÍNEA DE COMANDOS
# ============================================================

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Analizador de Código de Élite - Sistema de análisis inteligente'
    )
    
    parser.add_argument(
        '--path',
        required=True,
        help='Ruta del proyecto a analizar'
    )
    
    parser.add_argument(
        '--output',
        default='./code_analysis',
        help='Directorio de salida para reportes'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Número de procesos paralelos (default: CPU count)'
    )
    
    parser.add_argument(
        '--format',
        choices=['all', 'executive', 'technical', 'json'],
        default='all',
        help='Formatos de salida'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    root_path = Path(args.path).resolve()
    if not root_path.exists():
        print(f"❌ Error: La ruta {args.path} no existe")
        return 1
    
    # Crear analizador
    analyzer = EliteCodeAnalyzer()
    
    try:
        # Ejecutar análisis
        results = analyzer.analyze_project(root_path, max_workers=args.workers)
        
        if not results:
            print("⚠️  No se generaron resultados del análisis")
            return 1
        
        # Guardar resultados
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar reportes según formato solicitado
        reports = results.get('reports', {})
        
        if args.format in ['all', 'executive'] and 'executive' in reports:
            exec_file = output_dir / f"executive_report_{timestamp}.txt"
            exec_file.write_text(reports['executive'], encoding='utf-8')
            print(f"📄 Reporte ejecutivo: {exec_file}")
        
        if args.format in ['all', 'technical'] and 'technical' in reports:
            tech_file = output_dir / f"technical_report_{timestamp}.txt"
            tech_file.write_text(reports['technical'], encoding='utf-8')
            print(f"📄 Reporte técnico: {tech_file}")
        
        if args.format in ['all', 'json'] and 'json' in reports:
            json_file = output_dir / f"full_analysis_{timestamp}.json"
            json_file.write_text(reports['json'], encoding='utf-8')
            print(f"📄 JSON completo: {json_file}")
        
        # Imprimir resumen ejecutivo en consola
        print("\n" + "=" * 80)
        if 'executive' in reports:
            print(reports['executive'])
        
        # Estadísticas finales
        stats = results.get('statistics', {})
        elapsed = stats.get('end_time', 0) - stats.get('start_time', 0)
        
        print(f"\n🎉 Análisis completado exitosamente!")
        print(f"⏱️  Tiempo total: {elapsed:.2f} segundos")
        print(f"📊 Archivos analizados: {stats.get('files_analyzed', 0)}")
        print(f"📂 Resultados guardados en: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Análisis interrumpido por el usuario")
        return 130
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    sys.exit(main())