"""
Módulo QualityAnalyzer - Análisis completo de calidad de código
"""

import ast
import re
import json
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from enum import Enum
import statistics

class QualityLevel(Enum):
    """Niveles de calidad de código"""
    EXCELLENT = "excelente"
    GOOD = "bueno"
    FAIR = "regular"
    POOR = "pobre"
    CRITICAL = "crítico"

class Severity(Enum):
    """Severidad de issues"""
    INFO = "info"
    LOW = "bajo"
    MEDIUM = "medio"
    HIGH = "alto"
    CRITICAL = "crítico"

@dataclass
class CodeIssue:
    """Issue de calidad de código"""
    id: str
    type: str
    description: str
    file_path: str
    line_start: int
    line_end: int
    severity: Severity
    confidence: float
    suggestion: str
    rule_id: Optional[str] = None
    category: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityMetric:
    """Métrica de calidad"""
    name: str
    value: float
    unit: Optional[str] = None
    threshold: Optional[float] = None
    weight: float = 1.0
    category: str = "general"
    description: Optional[str] = None
    status: str = "passed"  # passed, warning, failed

@dataclass
class FileQualityReport:
    """Reporte de calidad para un archivo"""
    file_path: str
    language: str
    overall_score: float
    quality_level: QualityLevel
    metrics: List[QualityMetric] = field(default_factory=list)
    issues: List[CodeIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "overall_score": self.overall_score,
            "quality_level": self.quality_level.value,
            "metrics": [{
                "name": m.name,
                "value": m.value,
                "unit": m.unit,
                "threshold": m.threshold,
                "status": m.status
            } for m in self.metrics],
            "issues_count": len(self.issues),
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class ProjectQualityReport:
    """Reporte de calidad para un proyecto completo"""
    project_path: str
    overall_score: float
    quality_level: QualityLevel
    file_reports: Dict[str, FileQualityReport] = field(default_factory=dict)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    issues_summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class QualityAnalyzer:
    """Analizador completo de calidad de código"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar analizador de calidad"""
        self.config = config or self._get_default_config()
        self.language_analyzers = {
            'python': self._analyze_python_file,
            'java': self._analyze_java_file,
            'javascript': self._analyze_javascript_file,
            'typescript': self._analyze_typescript_file,
            'cpp': self._analyze_cpp_file,
            'c': self._analyze_c_file,
            'go': self._analyze_go_file,
            'ruby': self._analyze_ruby_file,
            'php': self._analyze_php_file,
        }
        
        # Cargar reglas de calidad
        self.quality_rules = self._load_quality_rules()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Obtener configuración por defecto"""
        return {
            # Límites de complejidad
            'max_cyclomatic_complexity': 15,
            'max_function_lines': 50,
            'max_class_lines': 300,
            'max_file_lines': 1000,
            'max_parameters': 7,
            'max_nesting_depth': 4,
            
            # Límites de estilo
            'max_line_length': 120,
            'max_module_imports': 30,
            
            # Umbrales de métricas
            'min_comment_ratio': 10.0,  # Porcentaje mínimo de comentarios
            'max_duplication_percentage': 10.0,
            
            # Configuración de análisis
            'enable_security_analysis': True,
            'enable_performance_analysis': True,
            'enable_maintainability_analysis': True,
            
            # Pesos para cálculo de puntaje
            'weights': {
                'complexity': 0.25,
                'maintainability': 0.25,
                'security': 0.20,
                'performance': 0.15,
                'documentation': 0.15,
            }
        }
    
    def _load_quality_rules(self) -> Dict[str, Dict[str, Any]]:
        """Cargar reglas de calidad"""
        return {
            # Reglas de complejidad
            'complex_function': {
                'id': 'CMP-001',
                'category': 'complexity',
                'severity': Severity.MEDIUM,
                'description': 'Función demasiado compleja',
                'suggestion': 'Dividir en funciones más pequeñas',
                'threshold': 15  # complejidad ciclomática
            },
            'long_function': {
                'id': 'CMP-002',
                'category': 'complexity',
                'severity': Severity.MEDIUM,
                'description': 'Función demasiado larga',
                'suggestion': 'Extraer partes del código en funciones auxiliares',
                'threshold': 50  # líneas
            },
            'deep_nesting': {
                'id': 'CMP-003',
                'category': 'complexity',
                'severity': Severity.MEDIUM,
                'description': 'Anidamiento demasiado profundo',
                'suggestion': 'Reducir niveles de anidamiento',
                'threshold': 4  # niveles
            },
            
            # Reglas de mantenibilidad
            'duplicate_code': {
                'id': 'MNT-001',
                'category': 'maintainability',
                'severity': Severity.MEDIUM,
                'description': 'Código duplicado detectado',
                'suggestion': 'Extraer código común a funciones/métodos',
            },
            'magic_number': {
                'id': 'MNT-002',
                'category': 'maintainability',
                'severity': Severity.LOW,
                'description': 'Número mágico sin explicación',
                'suggestion': 'Definir constante con nombre significativo',
            },
            'long_parameter_list': {
                'id': 'MNT-003',
                'category': 'maintainability',
                'severity': Severity.MEDIUM,
                'description': 'Demasiados parámetros en función',
                'suggestion': 'Usar objeto parámetro o dividir función',
                'threshold': 7
            },
            
            # Reglas de seguridad
            'sql_injection': {
                'id': 'SEC-001',
                'category': 'security',
                'severity': Severity.CRITICAL,
                'description': 'Posible inyección SQL',
                'suggestion': 'Usar consultas parametrizadas',
            },
            'hardcoded_secret': {
                'id': 'SEC-002',
                'category': 'security',
                'severity': Severity.HIGH,
                'description': 'Secreto hardcodeado en código',
                'suggestion': 'Usar variables de entorno o sistema de secretos',
            },
            'eval_usage': {
                'id': 'SEC-003',
                'category': 'security',
                'severity': Severity.HIGH,
                'description': 'Uso de eval() - riesgo de seguridad',
                'suggestion': 'Evitar eval(), usar alternativas seguras',
            },
            
            # Reglas de rendimiento
            'nested_loops': {
                'id': 'PER-001',
                'category': 'performance',
                'severity': Severity.MEDIUM,
                'description': 'Bucles anidados - complejidad O(n²)',
                'suggestion': 'Optimizar algoritmo o usar estructuras más eficientes',
            },
            'inefficient_regex': {
                'id': 'PER-002',
                'category': 'performance',
                'severity': Severity.LOW,
                'description': 'Expresión regular ineficiente',
                'suggestion': 'Optimizar regex o usar métodos nativos',
            },
            'memory_leak_pattern': {
                'id': 'PER-003',
                'category': 'performance',
                'severity': Severity.HIGH,
                'description': 'Posible pérdida de memoria',
                'suggestion': 'Asegurar liberación de recursos',
            },
            
            # Reglas de documentación
            'missing_docstring': {
                'id': 'DOC-001',
                'category': 'documentation',
                'severity': Severity.LOW,
                'description': 'Falta documentación de función/clase',
                'suggestion': 'Agregar docstring/documentación',
            },
            'incomplete_docstring': {
                'id': 'DOC-002',
                'category': 'documentation',
                'severity': Severity.INFO,
                'description': 'Documentación incompleta',
                'suggestion': 'Completar parámetros y valor de retorno',
            },
        }
    
    def analyze_file(self, file_path: str, content: str, language: str) -> FileQualityReport:
        """
        Analizar calidad de un archivo de código
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            language: Lenguaje de programación
            
        Returns:
            FileQualityReport: Reporte de calidad del archivo
        """
        if language not in self.language_analyzers:
            return self._analyze_generic_file(file_path, content, language)
        
        analyzer = self.language_analyzers[language]
        return analyzer(file_path, content)
    
    def _analyze_python_file(self, file_path: str, content: str) -> FileQualityReport:
        """Analizar archivo Python"""
        report = FileQualityReport(
            file_path=file_path,
            language='python',
            overall_score=100.0,
            quality_level=QualityLevel.EXCELLENT
        )
        
        metrics = []
        issues = []
        
        try:
            tree = ast.parse(content)
            
            # Calcular métricas básicas
            basic_metrics = self._calculate_basic_metrics(content)
            metrics.extend(basic_metrics)
            
            # Análisis de complejidad
            complexity_metrics, complexity_issues = self._analyze_python_complexity(tree, file_path)
            metrics.extend(complexity_metrics)
            issues.extend(complexity_issues)
            
            # Análisis de mantenibilidad
            maintainability_metrics, maintainability_issues = self._analyze_python_maintainability(tree, content, file_path)
            metrics.extend(maintainability_metrics)
            issues.extend(maintainability_issues)
            
            # Análisis de seguridad
            if self.config['enable_security_analysis']:
                security_issues = self._analyze_python_security(content, file_path)
                issues.extend(security_issues)
            
            # Análisis de rendimiento
            if self.config['enable_performance_analysis']:
                performance_issues = self._analyze_python_performance(tree, content, file_path)
                issues.extend(performance_issues)
            
            # Análisis de documentación
            documentation_metrics, documentation_issues = self._analyze_python_documentation(tree, content, file_path)
            metrics.extend(documentation_metrics)
            issues.extend(documentation_issues)
            
            # Calcular puntaje final
            report.metrics = metrics
            report.issues = issues
            report.overall_score = self._calculate_file_score(metrics, issues)
            report.quality_level = self._determine_quality_level(report.overall_score)
            report.summary = self._generate_file_summary(metrics, issues)
            
        except SyntaxError as e:
            # Error de sintaxis - calidad crítica
            issues.append(CodeIssue(
                id='SYN-001',
                type='syntax_error',
                description=f"Error de sintaxis: {e.msg}",
                file_path=file_path,
                line_start=e.lineno or 1,
                line_end=e.lineno or 1,
                severity=Severity.CRITICAL,
                confidence=1.0,
                suggestion='Corregir error de sintaxis',
                category='syntax'
            ))
            
            report.metrics = metrics
            report.issues = issues
            report.overall_score = 0.0
            report.quality_level = QualityLevel.CRITICAL
            report.summary = {'syntax_errors': 1}
        
        return report
    
    def _calculate_basic_metrics(self, content: str) -> List[QualityMetric]:
        """Calcular métricas básicas del archivo"""
        metrics = []
        lines = content.splitlines()
        
        # Líneas totales
        total_lines = len(lines)
        metrics.append(QualityMetric(
            name='total_lines',
            value=total_lines,
            unit='líneas',
            category='size'
        ))
        
        # Líneas de código
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        metrics.append(QualityMetric(
            name='code_lines',
            value=code_lines,
            unit='líneas',
            category='size'
        ))
        
        # Líneas en blanco
        blank_lines = sum(1 for line in lines if not line.strip())
        metrics.append(QualityMetric(
            name='blank_lines',
            value=blank_lines,
            unit='líneas',
            category='size'
        ))
        
        # Líneas de comentario
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        metrics.append(QualityMetric(
            name='comment_lines',
            value=comment_lines,
            unit='líneas',
            category='documentation'
        ))
        
        # Ratio comentarios/código
        if code_lines > 0:
            comment_ratio = (comment_lines / code_lines) * 100
            metrics.append(QualityMetric(
                name='comment_ratio',
                value=comment_ratio,
                unit='%',
                threshold=self.config['min_comment_ratio'],
                category='documentation',
                status='warning' if comment_ratio < self.config['min_comment_ratio'] else 'passed'
            ))
        
        return metrics
    
    def _analyze_python_complexity(self, tree: ast.AST, file_path: str) -> Tuple[List[QualityMetric], List[CodeIssue]]:
        """Analizar complejidad de código Python"""
        metrics = []
        issues = []
        
        # Calcular complejidad ciclomática
        total_complexity = 0
        function_count = 0
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1
                self.function_complexities = []
                self.current_function = None
                
            def visit_FunctionDef(self, node):
                # Guardar complejidad anterior
                prev_complexity = self.complexity
                prev_function = self.current_function
                
                # Reiniciar para nueva función
                self.complexity = 1
                self.current_function = node.name
                
                self.generic_visit(node)
                
                # Guardar complejidad de la función
                self.function_complexities.append({
                    'name': node.name,
                    'complexity': self.complexity,
                    'lines': (node.end_lineno or node.lineno) - node.lineno + 1
                })
                
                # Restaurar estado anterior
                self.complexity = prev_complexity
                self.current_function = prev_function
                
            def visit_ClassDef(self, node):
                self.generic_visit(node)
                
            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_Try(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_BoolOp(self, node):
                # Cada operador lógico (and/or) añade complejidad
                self.complexity += len(node.values) - 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        # Calcular métricas de complejidad
        if visitor.function_complexities:
            complexities = [fc['complexity'] for fc in visitor.function_complexities]
            avg_complexity = sum(complexities) / len(complexities)
            max_complexity = max(complexities)
            
            metrics.append(QualityMetric(
                name='avg_cyclomatic_complexity',
                value=avg_complexity,
                threshold=self.config['max_cyclomatic_complexity'],
                category='complexity',
                status='warning' if avg_complexity > self.config['max_cyclomatic_complexity'] else 'passed'
            ))
            
            metrics.append(QualityMetric(
                name='max_cyclomatic_complexity',
                value=max_complexity,
                threshold=self.config['max_cyclomatic_complexity'],
                category='complexity',
                status='warning' if max_complexity > self.config['max_cyclomatic_complexity'] else 'passed'
            ))
            
            # Detectar funciones demasiado complejas
            for func_info in visitor.function_complexities:
                if func_info['complexity'] > self.config['max_cyclomatic_complexity']:
                    issues.append(CodeIssue(
                        id='CMP-001',
                        type='complex_function',
                        description=f"Función '{func_info['name']}' demasiado compleja (complejidad: {func_info['complexity']})",
                        file_path=file_path,
                        line_start=1,  # Sería mejor tener línea real
                        line_end=1,
                        severity=Severity.MEDIUM,
                        confidence=0.8,
                        suggestion='Dividir función en funciones más pequeñas',
                        rule_id='CMP-001',
                        category='complexity'
                    ))
                
                # Detectar funciones demasiado largas
                if func_info['lines'] > self.config['max_function_lines']:
                    issues.append(CodeIssue(
                        id='CMP-002',
                        type='long_function',
                        description=f"Función '{func_info['name']}' demasiado larga ({func_info['lines']} líneas)",
                        file_path=file_path,
                        line_start=1,
                        line_end=1,
                        severity=Severity.MEDIUM,
                        confidence=0.9,
                        suggestion='Extraer partes del código en funciones auxiliares',
                        rule_id='CMP-002',
                        category='complexity'
                    ))
        
        return metrics, issues
    
    def _analyze_python_maintainability(self, tree: ast.AST, content: str, file_path: str) -> Tuple[List[QualityMetric], List[CodeIssue]]:
        """Analizar mantenibilidad de código Python"""
        metrics = []
        issues = []
        
        # Detectar números mágicos
        magic_number_issues = self._detect_magic_numbers(content, file_path)
        issues.extend(magic_number_issues)
        
        # Detectar anidamiento profundo
        nesting_issues = self._detect_deep_nesting(tree, file_path)
        issues.extend(nesting_issues)
        
        # Detectar muchos parámetros
        parameter_issues = self._detect_long_parameter_lists(tree, file_path)
        issues.extend(parameter_issues)
        
        # Calcular métrica de duplicación (simplificada)
        duplication_score = self._calculate_duplication_score(content)
        metrics.append(QualityMetric(
            name='duplication_score',
            value=duplication_score,
            unit='%',
            threshold=self.config['max_duplication_percentage'],
            category='maintainability',
            status='warning' if duplication_score > self.config['max_duplication_percentage'] else 'passed'
        ))
        
        return metrics, issues
    
    def _detect_magic_numbers(self, content: str, file_path: str) -> List[CodeIssue]:
        """Detectar números mágicos"""
        issues = []
        lines = content.splitlines()
        
        # Excluir números comunes
        common_numbers = {'0', '1', '-1', '2', '10', '100', '1000'}
        
        for i, line in enumerate(lines, 1):
            # Buscar números en la línea
            numbers = re.findall(r'\b(\d+\.?\d*)\b', line)
            for num in numbers:
                if num not in common_numbers and float(num) not in [0, 1, -1]:
                    # Verificar si ya es una constante
                    if not re.search(rf'[A-Z_]+_{num}', line) and not re.search(rf'{num}\s*=', line):
                        issues.append(CodeIssue(
                            id='MNT-002',
                            type='magic_number',
                            description=f"Número mágico: {num}",
                            file_path=file_path,
                            line_start=i,
                            line_end=i,
                            severity=Severity.LOW,
                            confidence=0.7,
                            suggestion='Definir constante con nombre significativo',
                            rule_id='MNT-002',
                            category='maintainability'
                        ))
        
        return issues
    
    def _detect_deep_nesting(self, tree: ast.AST, file_path: str) -> List[CodeIssue]:
        """Detectar anidamiento profundo"""
        issues = []
        
        class NestingVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.issues = []
                self.nesting_level = 0
                self.current_function = None
                
            def visit_FunctionDef(self, node):
                self.current_function = node.name
                self.nesting_level = 0
                self.max_nesting = 0
                self.generic_visit(node)
                
                if self.max_nesting > 4:  # Umbral configurable
                    self.issues.append(CodeIssue(
                        id='CMP-003',
                        type='deep_nesting',
                        description=f"Función '{self.current_function}' tiene anidamiento profundo (nivel {self.max_nesting})",
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        severity=Severity.MEDIUM,
                        confidence=0.8,
                        suggestion='Reducir niveles de anidamiento usando funciones auxiliares',
                        rule_id='CMP-003',
                        category='complexity'
                    ))
                
                self.current_function = None
                
            def visit_If(self, node):
                self.nesting_level += 1
                self.max_nesting = max(self.max_nesting, self.nesting_level)
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_For(self, node):
                self.nesting_level += 1
                self.max_nesting = max(self.max_nesting, self.nesting_level)
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_While(self, node):
                self.nesting_level += 1
                self.max_nesting = max(self.max_nesting, self.nesting_level)
                self.generic_visit(node)
                self.nesting_level -= 1
                
            def visit_Try(self, node):
                self.nesting_level += 1
                self.max_nesting = max(self.max_nesting, self.nesting_level)
                self.generic_visit(node)
                self.nesting_level -= 1
        
        visitor = NestingVisitor(file_path)
        visitor.visit(tree)
        return visitor.issues
    
    def _detect_long_parameter_lists(self, tree: ast.AST, file_path: str) -> List[CodeIssue]:
        """Detectar listas de parámetros largas"""
        issues = []
        
        class ParameterVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.issues = []
                
            def visit_FunctionDef(self, node):
                # Contar parámetros
                param_count = len(node.args.args) + len(node.args.kwonlyargs)
                if node.args.vararg:
                    param_count += 1
                if node.args.kwarg:
                    param_count += 1
                
                if param_count > 7:  # Umbral configurable
                    self.issues.append(CodeIssue(
                        id='MNT-003',
                        type='long_parameter_list',
                        description=f"Función '{node.name}' tiene {param_count} parámetros",
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        severity=Severity.MEDIUM,
                        confidence=0.9,
                        suggestion='Usar objeto parámetro o dividir función',
                        rule_id='MNT-003',
                        category='maintainability'
                    ))
                
                self.generic_visit(node)
        
        visitor = ParameterVisitor(file_path)
        visitor.visit(tree)
        return visitor.issues
    
    def _calculate_duplication_score(self, content: str) -> float:
        """Calcular score de duplicación (simplificado)"""
        lines = content.splitlines()
        
        # Contar líneas duplicadas consecutivas
        duplicate_lines = 0
        for i in range(len(lines) - 1):
            if lines[i].strip() and lines[i] == lines[i + 1]:
                duplicate_lines += 1
        
        # Calcular porcentaje
        if len(lines) > 0:
            return (duplicate_lines / len(lines)) * 100
        return 0.0
    
    def _analyze_python_security(self, content: str, file_path: str) -> List[CodeIssue]:
        """Analizar problemas de seguridad en Python"""
        issues = []
        lines = content.splitlines()
        
        # Detectar eval/exec
        for i, line in enumerate(lines, 1):
            if re.search(r'\beval\s*\(', line) or re.search(r'\bexec\s*\(', line):
                issues.append(CodeIssue(
                    id='SEC-003',
                    type='eval_usage',
                    description='Uso de eval() o exec() detectado',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=Severity.HIGH,
                    confidence=0.9,
                    suggestion='Evitar eval()/exec(), usar alternativas seguras',
                    rule_id='SEC-003',
                    category='security'
                ))
        
        # Detectar inyección SQL
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP']
        for i, line in enumerate(lines, 1):
            if any(keyword in line.upper() for keyword in sql_keywords):
                # Buscar concatenación peligrosa
                if '+' in line or '%s' in line or 'format(' in line:
                    issues.append(CodeIssue(
                        id='SEC-001',
                        type='sql_injection',
                        description='Posible inyección SQL detectada',
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        severity=Severity.CRITICAL,
                        confidence=0.7,
                        suggestion='Usar consultas parametrizadas o ORM',
                        rule_id='SEC-001',
                        category='security'
                    ))
        
        # Detectar secretos hardcodeados
        secret_patterns = [
            r'password\s*[:=]\s*["\'][^"\']+["\']',
            r'secret\s*[:=]\s*["\'][^"\']+["\']',
            r'api[_-]?key\s*[:=]\s*["\'][^"\']+["\']',
            r'token\s*[:=]\s*["\'][^"\']+["\']',
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        id='SEC-002',
                        type='hardcoded_secret',
                        description='Secreto hardcodeado detectado',
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        severity=Severity.HIGH,
                        confidence=0.8,
                        suggestion='Usar variables de entorno o sistema de secretos',
                        rule_id='SEC-002',
                        category='security'
                    ))
                    break
        
        return issues
    
    def _analyze_python_performance(self, tree: ast.AST, content: str, file_path: str) -> List[CodeIssue]:
        """Analizar problemas de rendimiento en Python"""
        issues = []
        
        # Detectar bucles anidados
        class NestedLoopVisitor(ast.NodeVisitor):
            def __init__(self, file_path):
                self.file_path = file_path
                self.issues = []
                self.loop_depth = 0
                self.current_function = None
                
            def visit_FunctionDef(self, node):
                self.current_function = node.name
                self.loop_depth = 0
                self.generic_visit(node)
                self.current_function = None
                
            def visit_For(self, node):
                self.loop_depth += 1
                if self.loop_depth >= 2:
                    self.issues.append(CodeIssue(
                        id='PER-001',
                        type='nested_loops',
                        description=f"Bucles anidados en función '{self.current_function}'",
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        severity=Severity.MEDIUM,
                        confidence=0.6,
                        suggestion='Optimizar algoritmo o usar estructuras de datos más eficientes',
                        rule_id='PER-001',
                        category='performance'
                    ))
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_While(self, node):
                self.loop_depth += 1
                if self.loop_depth >= 2:
                    self.issues.append(CodeIssue(
                        id='PER-001',
                        type='nested_loops',
                        description=f"Bucles anidados en función '{self.current_function}'",
                        file_path=self.file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        severity=Severity.MEDIUM,
                        confidence=0.6,
                        suggestion='Optimizar algoritmo o usar estructuras de datos más eficientes',
                        rule_id='PER-001',
                        category='performance'
                    ))
                self.generic_visit(node)
                self.loop_depth -= 1
        
        visitor = NestedLoopVisitor(file_path)
        visitor.visit(tree)
        issues.extend(visitor.issues)
        
        # Detectar regex ineficientes
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            if 're.compile' in line:
                # Buscar patrones problemáticos
                if '.*.*' in line or '.+.*' in line or '.*.+' in line:
                    issues.append(CodeIssue(
                        id='PER-002',
                        type='inefficient_regex',
                        description='Expresión regular potencialmente ineficiente',
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        severity=Severity.LOW,
                        confidence=0.5,
                        suggestion='Optimizar patrón regex',
                        rule_id='PER-002',
                        category='performance'
                    ))
        
        return issues
    
    def _analyze_python_documentation(self, tree: ast.AST, content: str, file_path: str) -> Tuple[List[QualityMetric], List[CodeIssue]]:
        """Analizar documentación en Python"""
        metrics = []
        issues = []
        
        # Contar docstrings
        class DocstringVisitor(ast.NodeVisitor):
            def __init__(self):
                self.functions_with_docstring = 0
                self.total_functions = 0
                self.classes_with_docstring = 0
                self.total_classes = 0
                self.issues = []
                
            def visit_FunctionDef(self, node):
                self.total_functions += 1
                docstring = ast.get_docstring(node)
                if docstring:
                    self.functions_with_docstring += 1
                    # Verificar si la docstring es completa
                    if len(docstring.strip().split('\n')) < 3:
                        self.issues.append({
                            'type': 'incomplete_docstring',
                            'name': node.name,
                            'line': node.lineno
                        })
                else:
                    self.issues.append({
                        'type': 'missing_docstring',
                        'name': node.name,
                        'line': node.lineno
                    })
                self.generic_visit(node)
                
            def visit_ClassDef(self, node):
                self.total_classes += 1
                docstring = ast.get_docstring(node)
                if docstring:
                    self.classes_with_docstring += 1
                    if len(docstring.strip().split('\n')) < 3:
                        self.issues.append({
                            'type': 'incomplete_docstring',
                            'name': node.name,
                            'line': node.lineno
                        })
                else:
                    self.issues.append({
                        'type': 'missing_docstring',
                        'name': node.name,
                        'line': node.lineno
                    })
                self.generic_visit(node)
        
        visitor = DocstringVisitor()
        visitor.visit(tree)
        
        # Calcular métricas de documentación
        if visitor.total_functions > 0:
            func_doc_ratio = (visitor.functions_with_docstring / visitor.total_functions) * 100
            metrics.append(QualityMetric(
                name='function_doc_ratio',
                value=func_doc_ratio,
                unit='%',
                category='documentation'
            ))
        
        if visitor.total_classes > 0:
            class_doc_ratio = (visitor.classes_with_docstring / visitor.total_classes) * 100
            metrics.append(QualityMetric(
                name='class_doc_ratio',
                value=class_doc_ratio,
                unit='%',
                category='documentation'
            ))
        
        # Crear issues de documentación
        for issue_info in visitor.issues:
            if issue_info['type'] == 'missing_docstring':
                issues.append(CodeIssue(
                    id='DOC-001',
                    type='missing_docstring',
                    description=f"Falta docstring en {'clase' if 'class' in issue_info['name'] else 'función'} '{issue_info['name']}'",
                    file_path=file_path,
                    line_start=issue_info['line'],
                    line_end=issue_info['line'],
                    severity=Severity.LOW,
                    confidence=1.0,
                    suggestion='Agregar docstring/documentación',
                    rule_id='DOC-001',
                    category='documentation'
                ))
            else:  # incomplete_docstring
                issues.append(CodeIssue(
                    id='DOC-002',
                    type='incomplete_docstring',
                    description=f"Docstring incompleto en {'clase' if 'class' in issue_info['name'] else 'función'} '{issue_info['name']}'",
                    file_path=file_path,
                    line_start=issue_info['line'],
                    line_end=issue_info['line'],
                    severity=Severity.INFO,
                    confidence=0.7,
                    suggestion='Completar parámetros y valor de retorno en docstring',
                    rule_id='DOC-002',
                    category='documentation'
                ))
        
        return metrics, issues
    
    def _calculate_file_score(self, metrics: List[QualityMetric], issues: List[CodeIssue]) -> float:
        """Calcular puntaje de calidad para un archivo"""
        base_score = 100.0
        
        # Penalizar por métricas fuera de umbral
        for metric in metrics:
            if metric.threshold is not None:
                if metric.status == 'warning':
                    # Penalización del 5% por métrica en warning
                    base_score -= 5.0 * metric.weight
                elif metric.status == 'failed':
                    # Penalización del 10% por métrica fallida
                    base_score -= 10.0 * metric.weight
        
        # Penalizar por issues
        severity_weights = {
            Severity.INFO: 0.5,
            Severity.LOW: 1.0,
            Severity.MEDIUM: 3.0,
            Severity.HIGH: 7.0,
            Severity.CRITICAL: 15.0
        }
        
        for issue in issues:
            weight = severity_weights.get(issue.severity, 1.0)
            # Reducir penalización basada en confianza
            penalty = weight * (1.0 - issue.confidence * 0.5)
            base_score -= penalty
        
        # Asegurar puntaje dentro de rango [0, 100]
        return max(0.0, min(100.0, base_score))
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determinar nivel de calidad basado en puntaje"""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 75:
            return QualityLevel.GOOD
        elif score >= 60:
            return QualityLevel.FAIR
        elif score >= 40:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _generate_file_summary(self, metrics: List[QualityMetric], issues: List[CodeIssue]) -> Dict[str, Any]:
        """Generar resumen del análisis del archivo"""
        summary = {
            'total_metrics': len(metrics),
            'total_issues': len(issues),
            'issues_by_severity': {},
            'issues_by_category': {},
            'metrics_status': {
                'passed': 0,
                'warning': 0,
                'failed': 0
            }
        }
        
        # Contar issues por severidad
        for issue in issues:
            severity = issue.severity.value
            if severity not in summary['issues_by_severity']:
                summary['issues_by_severity'][severity] = 0
            summary['issues_by_severity'][severity] += 1
            
            # Contar por categoría
            category = issue.category or 'unknown'
            if category not in summary['issues_by_category']:
                summary['issues_by_category'][category] = 0
            summary['issues_by_category'][category] += 1
        
        # Contar métricas por estado
        for metric in metrics:
            if metric.status in summary['metrics_status']:
                summary['metrics_status'][metric.status] += 1
        
        return summary
    
    def _analyze_java_file(self, file_path: str, content: str) -> FileQualityReport:
        """Analizar archivo Java"""
        # Implementación simplificada para Java
        report = FileQualityReport(
            file_path=file_path,
            language='java',
            overall_score=100.0,
            quality_level=QualityLevel.EXCELLENT
        )
        
        metrics = []
        issues = []
        
        # Métricas básicas
        basic_metrics = self._calculate_basic_metrics(content)
        metrics.extend(basic_metrics)
        
        # Issues básicos detectados por regex
        java_issues = self._detect_java_issues(content, file_path)
        issues.extend(java_issues)
        
        # Calcular puntaje
        report.metrics = metrics
        report.issues = issues
        report.overall_score = self._calculate_file_score(metrics, issues)
        report.quality_level = self._determine_quality_level(report.overall_score)
        report.summary = self._generate_file_summary(metrics, issues)
        
        return report
    
    def _detect_java_issues(self, content: str, file_path: str) -> List[CodeIssue]:
        """Detectar issues comunes en Java"""
        issues = []
        lines = content.splitlines()
        
        # Detectar System.out.println en código de producción
        for i, line in enumerate(lines, 1):
            if 'System.out.println' in line or 'System.err.println' in line:
                issues.append(CodeIssue(
                    id='LOG-001',
                    type='system_out_usage',
                    description='Uso de System.out para logging',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=Severity.LOW,
                    confidence=0.9,
                    suggestion='Usar framework de logging apropiado',
                    category='maintainability'
                ))
        
        # Detectar catch genérico
        for i, line in enumerate(lines, 1):
            if 'catch (Exception' in line or 'catch(Exception' in line:
                issues.append(CodeIssue(
                    id='ERR-001',
                    type='generic_catch',
                    description='Captura genérica de Exception',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=Severity.MEDIUM,
                    confidence=0.8,
                    suggestion='Capturar excepciones específicas',
                    category='maintainability'
                ))
        
        return issues
    
    def _analyze_javascript_file(self, file_path: str, content: str) -> FileQualityReport:
        """Analizar archivo JavaScript"""
        report = FileQualityReport(
            file_path=file_path,
            language='javascript',
            overall_score=100.0,
            quality_level=QualityLevel.EXCELLENT
        )
        
        metrics = []
        issues = []
        
        # Métricas básicas
        basic_metrics = self._calculate_basic_metrics(content)
        metrics.extend(basic_metrics)
        
        # Issues específicos de JavaScript
        js_issues = self._detect_javascript_issues(content, file_path)
        issues.extend(js_issues)
        
        # Calcular puntaje
        report.metrics = metrics
        report.issues = issues
        report.overall_score = self._calculate_file_score(metrics, issues)
        report.quality_level = self._determine_quality_level(report.overall_score)
        report.summary = self._generate_file_summary(metrics, issues)
        
        return report
    
    def _detect_javascript_issues(self, content: str, file_path: str) -> List[CodeIssue]:
        """Detectar issues comunes en JavaScript"""
        issues = []
        lines = content.splitlines()
        
        # Detectar eval()
        for i, line in enumerate(lines, 1):
            if 'eval(' in line:
                issues.append(CodeIssue(
                    id='SEC-003',
                    type='eval_usage',
                    description='Uso de eval() detectado',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=Severity.HIGH,
                    confidence=0.9,
                    suggestion='Evitar eval(), usar alternativas seguras',
                    category='security'
                ))
        
        # Detectar console.log
        for i, line in enumerate(lines, 1):
            if 'console.log' in line:
                issues.append(CodeIssue(
                    id='LOG-002',
                    type='console_log',
                    description='console.log en código de producción',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=Severity.LOW,
                    confidence=0.9,
                    suggestion='Usar framework de logging o eliminar en producción',
                    category='maintainability'
                ))
        
        # Detectar == en lugar de ===
        for i, line in enumerate(lines, 1):
            if ' == ' in line and ' === ' not in line:
                # Verificar que no sea comparación con null/undefined
                if 'null' not in line and 'undefined' not in line:
                    issues.append(CodeIssue(
                        id='TYP-001',
                        type='loose_equality',
                        description='Uso de == en lugar de ===',
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        severity=Severity.LOW,
                        confidence=0.7,
                        suggestion='Usar === para comparación estricta',
                        category='maintainability'
                    ))
        
        return issues
    
    def _analyze_typescript_file(self, file_path: str, content: str) -> FileQualityReport:
        """Analizar archivo TypeScript"""
        # Similar a JavaScript pero con chequeos de tipos adicionales
        report = self._analyze_javascript_file(file_path, content)
        report.language = 'typescript'
        
        # Issues específicos de TypeScript
        ts_issues = self._detect_typescript_issues(content, file_path)
        report.issues.extend(ts_issues)
        
        # Recalcular puntaje
        report.overall_score = self._calculate_file_score(report.metrics, report.issues)
        report.quality_level = self._determine_quality_level(report.overall_score)
        report.summary = self._generate_file_summary(report.metrics, report.issues)
        
        return report
    
    def _detect_typescript_issues(self, content: str, file_path: str) -> List[CodeIssue]:
        """Detectar issues específicos de TypeScript"""
        issues = []
        lines = content.splitlines()
        
        # Detectar uso de any
        for i, line in enumerate(lines, 1):
            if ': any' in line or 'any[]' in line:
                issues.append(CodeIssue(
                    id='TYP-002',
                    type='any_type',
                    description='Uso del tipo any',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=Severity.LOW,
                    confidence=0.8,
                    suggestion='Usar tipos específicos en lugar de any',
                    category='maintainability'
                ))
        
        return issues
    
    def _analyze_cpp_file(self, file_path: str, content: str) -> FileQualityReport:
        """Analizar archivo C++"""
        report = FileQualityReport(
            file_path=file_path,
            language='cpp',
            overall_score=100.0,
            quality_level=QualityLevel.EXCELLENT
        )
        
        metrics = []
        issues = []
        
        # Métricas básicas
        basic_metrics = self._calculate_basic_metrics(content)
        metrics.extend(basic_metrics)
        
        # Issues específicos de C++
        cpp_issues = self._detect_cpp_issues(content, file_path)
        issues.extend(cpp_issues)
        
        # Calcular puntaje
        report.metrics = metrics
        report.issues = issues
        report.overall_score = self._calculate_file_score(metrics, issues)
        report.quality_level = self._determine_quality_level(report.overall_score)
        report.summary = self._generate_file_summary(metrics, issues)
        
        return report
    
    def _detect_cpp_issues(self, content: str, file_path: str) -> List[CodeIssue]:
        """Detectar issues comunes en C++"""
        issues = []
        lines = content.splitlines()
        
        # Detectar new sin delete
        for i, line in enumerate(lines, 1):
            if 'new ' in line and 'delete ' not in content[i:i+20]:
                issues.append(CodeIssue(
                    id='MEM-001',
                    type='possible_memory_leak',
                    description='new sin delete correspondiente',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=Severity.HIGH,
                    confidence=0.6,
                    suggestion='Usar smart pointers o asegurar delete',
                    category='performance'
                ))
        
        # Detectar goto
        for i, line in enumerate(lines, 1):
            if ' goto ' in line:
                issues.append(CodeIssue(
                    id='CTL-001',
                    type='goto_usage',
                    description='Uso de goto',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=Severity.MEDIUM,
                    confidence=0.9,
                    suggestion='Evitar goto, usar estructuras de control modernas',
                    category='maintainability'
                ))
        
        return issues
    
    def _analyze_c_file(self, file_path: str, content: str) -> FileQualityReport:
        """Analizar archivo C"""
        report = FileQualityReport(
            file_path=file_path,
            language='c',
            overall_score=100.0,
            quality_level=QualityLevel.EXCELLENT
        )
        
        metrics = []
        issues = []
        
        # Métricas básicas
        basic_metrics = self._calculate_basic_metrics(content)
        metrics.extend(basic_metrics)
        
        # Issues específicos de C
        c_issues = self._detect_c_issues(content, file_path)
        issues.extend(c_issues)
        
        # Calcular puntaje
        report.metrics = metrics
        report.issues = issues
        report.overall_score = self._calculate_file_score(metrics, issues)
        report.quality_level = self._determine_quality_level(report.overall_score)
        report.summary = self._generate_file_summary(metrics, issues)
        
        return report
    
    def _detect_c_issues(self, content: str, file_path: str) -> List[CodeIssue]:
        """Detectar issues comunes en C"""
        issues = []
        lines = content.splitlines()
        
        # Detectar malloc sin free
        malloc_count = content.count('malloc(')
        free_count = content.count('free(')
        
        if malloc_count > free_count:
            issues.append(CodeIssue(
                id='MEM-002',
                type='memory_management_issue',
                description=f'Posible pérdida de memoria: {malloc_count} malloc vs {free_count} free',
                file_path=file_path,
                line_start=1,
                line_end=len(lines),
                severity=Severity.HIGH,
                confidence=0.5,
                suggestion='Asegurar que cada malloc tenga su free correspondiente',
                category='performance'
            ))
        
        return issues
    
    def _analyze_go_file(self, file_path: str, content: str) -> FileQualityReport:
        """Analizar archivo Go"""
        return self._analyze_generic_file(file_path, content, 'go')
    
    def _analyze_ruby_file(self, file_path: str, content: str) -> FileQualityReport:
        """Analizar archivo Ruby"""
        return self._analyze_generic_file(file_path, content, 'ruby')
    
    def _analyze_php_file(self, file_path: str, content: str) -> FileQualityReport:
        """Analizar archivo PHP"""
        return self._analyze_generic_file(file_path, content, 'php')
    
    def _analyze_generic_file(self, file_path: str, content: str, language: str) -> FileQualityReport:
        """Análisis genérico para lenguajes no soportados específicamente"""
        report = FileQualityReport(
            file_path=file_path,
            language=language,
            overall_score=100.0,
            quality_level=QualityLevel.EXCELLENT
        )
        
        metrics = []
        issues = []
        
        # Métricas básicas
        basic_metrics = self._calculate_basic_metrics(content)
        metrics.extend(basic_metrics)
        
        # Issues genéricos
        generic_issues = self._detect_generic_issues(content, file_path, language)
        issues.extend(generic_issues)
        
        # Calcular puntaje
        report.metrics = metrics
        report.issues = issues
        report.overall_score = self._calculate_file_score(metrics, issues)
        report.quality_level = self._determine_quality_level(report.overall_score)
        report.summary = self._generate_file_summary(metrics, issues)
        
        return report
    
    def _detect_generic_issues(self, content: str, file_path: str, language: str) -> List[CodeIssue]:
        """Detectar issues genéricos"""
        issues = []
        lines = content.splitlines()
        
        # Detectar líneas muy largas
        for i, line in enumerate(lines, 1):
            if len(line) > self.config['max_line_length']:
                issues.append(CodeIssue(
                    id='STY-001',
                    type='long_line',
                    description=f'Línea demasiado larga ({len(line)} caracteres)',
                    file_path=file_path,
                    line_start=i,
                    line_end=i,
                    severity=Severity.LOW,
                    confidence=1.0,
                    suggestion=f'Dividir línea (máximo {self.config["max_line_length"]} caracteres)',
                    category='style'
                ))
        
        return issues
    
    def analyze_project(self, project_path: str, file_reports: Dict[str, FileQualityReport]) -> ProjectQualityReport:
        """
        Analizar calidad de un proyecto completo
        
        Args:
            project_path: Ruta del proyecto
            file_reports: Reportes de archivos individuales
            
        Returns:
            ProjectQualityReport: Reporte de calidad del proyecto
        """
        if not file_reports:
            return ProjectQualityReport(
                project_path=project_path,
                overall_score=0.0,
                quality_level=QualityLevel.CRITICAL,
                recommendations=['No se encontraron archivos para analizar']
            )
        
        # Calcular métricas agregadas
        scores = [report.overall_score for report in file_reports.values()]
        avg_score = statistics.mean(scores) if scores else 0.0
        
        # Distribución de calidad
        quality_distribution = {
            QualityLevel.EXCELLENT: 0,
            QualityLevel.GOOD: 0,
            QualityLevel.FAIR: 0,
            QualityLevel.POOR: 0,
            QualityLevel.CRITICAL: 0
        }
        
        for report in file_reports.values():
            quality_distribution[report.quality_level] += 1
        
        # Issues agregados
        all_issues = []
        for report in file_reports.values():
            all_issues.extend(report.issues)
        
        issues_by_severity = {}
        issues_by_category = {}
        
        for issue in all_issues:
            # Por severidad
            severity = issue.severity.value
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
            
            # Por categoría
            category = issue.category or 'unknown'
            issues_by_category[category] = issues_by_category.get(category, 0) + 1
        
        # Métricas agregadas
        metrics_summary = self._calculate_project_metrics(file_reports)
        
        # Generar recomendaciones
        recommendations = self._generate_project_recommendations(file_reports, all_issues)
        
        return ProjectQualityReport(
            project_path=project_path,
            overall_score=avg_score,
            quality_level=self._determine_quality_level(avg_score),
            file_reports=file_reports,
            metrics_summary=metrics_summary,
            issues_summary={
                'total': len(all_issues),
                'by_severity': issues_by_severity,
                'by_category': issues_by_category
            },
            recommendations=recommendations
        )
    
    def _calculate_project_metrics(self, file_reports: Dict[str, FileQualityReport]) -> Dict[str, Any]:
        """Calcular métricas agregadas del proyecto"""
        metrics_summary = {
            'total_files': len(file_reports),
            'total_lines': 0,
            'total_issues': 0,
            'avg_score': 0.0,
            'language_distribution': {},
            'quality_distribution': {}
        }
        
        scores = []
        for file_path, report in file_reports.items():
            # Contar líneas
            for metric in report.metrics:
                if metric.name == 'total_lines':
                    metrics_summary['total_lines'] += int(metric.value)
            
            # Contar issues
            metrics_summary['total_issues'] += len(report.issues)
            
            # Acumular puntaje
            scores.append(report.overall_score)
            
            # Distribución por lenguaje
            lang = report.language
            metrics_summary['language_distribution'][lang] = metrics_summary['language_distribution'].get(lang, 0) + 1
            
            # Distribución por calidad
            quality = report.quality_level.value
            metrics_summary['quality_distribution'][quality] = metrics_summary['quality_distribution'].get(quality, 0) + 1
        
        # Calcular promedio de puntajes
        if scores:
            metrics_summary['avg_score'] = statistics.mean(scores)
        
        return metrics_summary
    
    def _generate_project_recommendations(self, file_reports: Dict[str, FileQualityReport], 
                                         all_issues: List[CodeIssue]) -> List[str]:
        """Generar recomendaciones para el proyecto"""
        recommendations = []
        
        # Analizar issues críticos
        critical_issues = [i for i in all_issues if i.severity == Severity.CRITICAL]
        if critical_issues:
            recommendations.append(f"Corregir {len(critical_issues)} issues críticos (seguridad, errores graves)")
        
        # Analizar issues de alta severidad
        high_issues = [i for i in all_issues if i.severity == Severity.HIGH]
        if high_issues:
            recommendations.append(f"Atender {len(high_issues)} issues de alta prioridad")
        
        # Analizar archivos con peor calidad
        sorted_reports = sorted(
            file_reports.items(),
            key=lambda x: x[1].overall_score
        )
        
        worst_files = sorted_reports[:5]
        if worst_files:
            recommendations.append("Mejorar calidad de archivos con puntaje más bajo:")
            for file_path, report in worst_files:
                recommendations.append(f"  - {file_path}: {report.overall_score:.1f}/100")
        
        # Recomendaciones generales
        total_issues = len(all_issues)
        if total_issues > 100:
            recommendations.append("Priorizar reducción de issues técnicos (más de 100 detectados)")
        elif total_issues > 50:
            recommendations.append("Planificar sesiones de refactorización para reducir issues técnicos")
        
        # Recomendación basada en distribución de calidad
        excellent_count = sum(1 for r in file_reports.values() if r.quality_level == QualityLevel.EXCELLENT)
        if excellent_count / len(file_reports) < 0.2:  # Menos del 20% excelente
            recommendations.append("Implementar mejores prácticas de codificación y revisiones de código")
        
        return recommendations[:10]  # Limitar a 10 recomendaciones
    
    def generate_report(self, project_report: ProjectQualityReport, 
                       format: str = 'markdown') -> str:
        """
        Generar reporte de calidad
        
        Args:
            project_report: Reporte del proyecto
            format: Formato del reporte ('markdown', 'html', 'json')
            
        Returns:
            str: Reporte generado
        """
        if format == 'markdown':
            return self._generate_markdown_report(project_report)
        elif format == 'html':
            return self._generate_html_report(project_report)
        elif format == 'json':
            return self._generate_json_report(project_report)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def _generate_markdown_report(self, report: ProjectQualityReport) -> str:
        """Generar reporte en formato Markdown"""
        output = []
        
        # Encabezado
        output.append(f"# Reporte de Calidad de Código")
        output.append(f"**Proyecto:** {report.project_path}")
        output.append(f"**Fecha:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        
        # Resumen general
        output.append(f"## Resumen General")
        output.append(f"**Puntaje general:** {report.overall_score:.1f}/100")
        output.append(f"**Nivel de calidad:** {report.quality_level.value}")
        output.append(f"**Archivos analizados:** {len(report.file_reports)}")
        output.append(f"**Issues totales:** {report.issues_summary.get('total', 0)}")
        output.append("")
        
        # Distribución de calidad
        output.append(f"### Distribución de Calidad")
        quality_dist = report.metrics_summary.get('quality_distribution', {})
        for level, count in sorted(quality_dist.items()):
            percentage = (count / len(report.file_reports)) * 100
            output.append(f"- **{level}:** {count} archivos ({percentage:.1f}%)")
        output.append("")
        
        # Distribución por lenguaje
        output.append(f"### Distribución por Lenguaje")
        lang_dist = report.metrics_summary.get('language_distribution', {})
        for lang, count in sorted(lang_dist.items()):
            percentage = (count / len(report.file_reports)) * 100
            output.append(f"- **{lang}:** {count} archivos ({percentage:.1f}%)")
        output.append("")
        
        # Issues por severidad
        output.append(f"### Issues por Severidad")
        severity_dist = report.issues_summary.get('by_severity', {})
        for severity, count in sorted(severity_dist.items()):
            output.append(f"- **{severity}:** {count} issues")
        output.append("")
        
        # Issues por categoría
        output.append(f"### Issues por Categoría")
        category_dist = report.issues_summary.get('by_category', {})
        for category, count in sorted(category_dist.items()):
            output.append(f"- **{category}:** {count} issues")
        output.append("")
        
        # Recomendaciones
        output.append(f"### Recomendaciones")
        if report.recommendations:
            for i, rec in enumerate(report.recommendations, 1):
                output.append(f"{i}. {rec}")
        else:
            output.append("No hay recomendaciones específicas.")
        output.append("")
        
        # Archivos con peor calidad
        output.append(f"### Archivos que Necesitan Atención")
        sorted_files = sorted(
            report.file_reports.items(),
            key=lambda x: x[1].overall_score
        )
        
        for file_path, file_report in sorted_files[:10]:
            output.append(f"- **{file_path}**: {file_report.overall_score:.1f}/100 ({file_report.quality_level.value})")
            if file_report.issues:
                output.append(f"  - {len(file_report.issues)} issues")
        output.append("")
        
        # Detalles por archivo (solo si hay pocos)
        if len(report.file_reports) <= 20:
            output.append(f"### Detalles por Archivo")
            output.append("")
            
            for file_path, file_report in sorted(report.file_reports.items()):
                output.append(f"#### {file_path}")
                output.append(f"- **Puntaje:** {file_report.overall_score:.1f}/100")
                output.append(f"- **Lenguaje:** {file_report.language}")
                output.append(f"- **Issues:** {len(file_report.issues)}")
                
                # Issues por severidad
                issues_by_sev = {}
                for issue in file_report.issues:
                    sev = issue.severity.value
                    issues_by_sev[sev] = issues_by_sev.get(sev, 0) + 1
                
                if issues_by_sev:
                    output.append(f"- **Issues por severidad:**")
                    for sev, count in sorted(issues_by_sev.items()):
                        output.append(f"  - {sev}: {count}")
                
                output.append("")
        
        return '\n'.join(output)
    
    def _generate_html_report(self, report: ProjectQualityReport) -> str:
        """Generar reporte en formato HTML"""
        html = []
        
        html.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Calidad de Código</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
                .metric { margin: 5px 0; }
                .quality-badge {
                    display: inline-block;
                    padding: 3px 8px;
                    border-radius: 3px;
                    font-weight: bold;
                    color: white;
                }
                .excellent { background: #4CAF50; }
                .good { background: #8BC34A; }
                .fair { background: #FFC107; }
                .poor { background: #FF9800; }
                .critical { background: #F44336; }
                .file-list { margin-top: 20px; }
                .file-item { 
                    padding: 10px; 
                    border-bottom: 1px solid #ddd;
                    display: flex;
                    justify-content: space-between;
                }
                .issues-list { margin-left: 20px; }
                .issue-item { margin: 5px 0; }
                .severity-critical { color: #F44336; font-weight: bold; }
                .severity-high { color: #FF9800; }
                .severity-medium { color: #FFC107; }
                .severity-low { color: #8BC34A; }
                .severity-info { color: #2196F3; }
                .recommendations { 
                    background: #E8F5E8; 
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 20px;
                }
                .distribution-bar {
                    height: 20px;
                    background: #eee;
                    border-radius: 3px;
                    margin: 5px 0;
                    overflow: hidden;
                }
                .distribution-segment {
                    height: 100%;
                    float: left;
                    text-align: center;
                    color: white;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
        """)
        
        # Encabezado
        html.append(f"<h1>Reporte de Calidad de Código</h1>")
        html.append(f"<div class='summary'>")
        html.append(f"<p><strong>Proyecto:</strong> {report.project_path}</p>")
        html.append(f"<p><strong>Fecha:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"<p><strong>Puntaje general:</strong> {report.overall_score:.1f}/100</p>")
        
        # Badge de calidad
        quality_class = report.quality_level.value
        html.append(f"<p><strong>Nivel de calidad:</strong> ")
        html.append(f"<span class='quality-badge {quality_class}'>{report.quality_level.value.upper()}</span></p>")
        
        html.append(f"<p><strong>Archivos analizados:</strong> {len(report.file_reports)}</p>")
        html.append(f"<p><strong>Issues totales:</strong> {report.issues_summary.get('total', 0)}</p>")
        html.append("</div>")
        
        # Distribución de calidad
        html.append("<h2>Distribución de Calidad</h2>")
        quality_dist = report.metrics_summary.get('quality_distribution', {})
        
        html.append("<div class='distribution-bar'>")
        colors = {
            'excelente': '#4CAF50',
            'bueno': '#8BC34A',
            'regular': '#FFC107',
            'pobre': '#FF9800',
            'crítico': '#F44336'
        }
        
        for level, count in sorted(quality_dist.items()):
            percentage = (count / len(report.file_reports)) * 100
            color = colors.get(level, '#999')
            html.append(f"""
                <div class='distribution-segment' 
                     style='width: {percentage}%; background: {color};'
                     title='{level}: {count} archivos ({percentage:.1f}%)'>
                </div>
            """)
        html.append("</div>")
        
        # Leyenda
        html.append("<div style='margin-top: 10px;'>")
        for level, count in sorted(quality_dist.items()):
            percentage = (count / len(report.file_reports)) * 100
            color = colors.get(level, '#999')
            html.append(f"""
                <span style='margin-right: 15px;'>
                    <span style='display: inline-block; width: 12px; height: 12px; 
                          background: {color}; margin-right: 5px;'></span>
                    {level}: {count} ({percentage:.1f}%)
                </span>
            """)
        html.append("</div>")
        
        # Issues por severidad
        html.append("<h2>Issues por Severidad</h2>")
        severity_dist = report.issues_summary.get('by_severity', {})
        
        html.append("<ul>")
        severity_colors = {
            'crítico': 'severity-critical',
            'alto': 'severity-high',
            'medio': 'severity-medium',
            'bajo': 'severity-low',
            'info': 'severity-info'
        }
        
        for severity, count in sorted(severity_dist.items()):
            severity_class = severity_colors.get(severity, '')
            html.append(f"<li class='{severity_class}'><strong>{severity}:</strong> {count} issues</li>")
        html.append("</ul>")
        
        # Recomendaciones
        html.append("<div class='recommendations'>")
        html.append("<h2>Recomendaciones</h2>")
        
        if report.recommendations:
            html.append("<ol>")
            for rec in report.recommendations:
                html.append(f"<li>{rec}</li>")
            html.append("</ol>")
        else:
            html.append("<p>No hay recomendaciones específicas.</p>")
        html.append("</div>")
        
        # Archivos con peor calidad
        html.append("<h2>Archivos que Necesitan Atención</h2>")
        html.append("<div class='file-list'>")
        
        sorted_files = sorted(
            report.file_reports.items(),
            key=lambda x: x[1].overall_score
        )
        
        for file_path, file_report in sorted_files[:15]:
            quality_class = file_report.quality_level.value
            html.append("<div class='file-item'>")
            html.append(f"<div><strong>{file_path}</strong></div>")
            html.append(f"<div>")
            html.append(f"<span class='quality-badge {quality_class}'>{file_report.overall_score:.1f}/100</span>")
            if file_report.issues:
                html.append(f" <span>({len(file_report.issues)} issues)</span>")
            html.append("</div>")
            html.append("</div>")
        
        html.append("</div>")
        
        # Pie de página
        html.append(f"<hr>")
        html.append(f"<p style='color: #666; font-size: 0.9em;'>")
        html.append(f"Reporte generado automáticamente por QualityAnalyzer")
        html.append(f"</p>")
        
        html.append("</body></html>")
        
        return '\n'.join(html)
    
    def _generate_json_report(self, report: ProjectQualityReport) -> str:
        """Generar reporte en formato JSON"""
        report_data = {
            'project_path': report.project_path,
            'overall_score': report.overall_score,
            'quality_level': report.quality_level.value,
            'timestamp': report.timestamp.isoformat(),
            'metrics_summary': report.metrics_summary,
            'issues_summary': report.issues_summary,
            'recommendations': report.recommendations,
            'files': {}
        }
        
        for file_path, file_report in report.file_reports.items():
            report_data['files'][file_path] = file_report.to_dict()
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def export_metrics(self, project_report: ProjectQualityReport, 
                      format: str = 'csv') -> str:
        """
        Exportar métricas en diferentes formatos
        
        Args:
            project_report: Reporte del proyecto
            format: Formato de exportación ('csv', 'json', 'excel')
            
        Returns:
            str: Métricas exportadas
        """
        if format == 'csv':
            return self._export_metrics_csv(project_report)
        elif format == 'json':
            return self._generate_json_report(project_report)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def _export_metrics_csv(self, report: ProjectQualityReport) -> str:
        """Exportar métricas en formato CSV"""
        lines = []
        
        # Encabezado
        lines.append('file_path,language,overall_score,quality_level,total_issues,lines_of_code')
        
        # Datos de cada archivo
        for file_path, file_report in report.file_reports.items():
            # Encontrar líneas de código
            loc = 0
            for metric in file_report.metrics:
                if metric.name == 'code_lines':
                    loc = int(metric.value)
                    break
            
            lines.append(
                f'{file_path},'
                f'{file_report.language},'
                f'{file_report.overall_score:.2f},'
                f'{file_report.quality_level.value},'
                f'{len(file_report.issues)},'
                f'{loc}'
            )
        
        return '\n'.join(lines)
    
    def get_quality_trends(self, historical_reports: List[ProjectQualityReport]) -> Dict[str, Any]:
        """
        Analizar tendencias de calidad a lo largo del tiempo
        
        Args:
            historical_reports: Reportes históricos ordenados por fecha
            
        Returns:
            Dict[str, Any]: Tendencias detectadas
        """
        if len(historical_reports) < 2:
            return {'error': 'Se necesitan al menos 2 reportes para analizar tendencias'}
        
        trends = {
            'score_trend': [],
            'issues_trend': [],
            'quality_level_trend': [],
            'improving': False,
            'deteriorating': False,
            'stable': False
        }
        
        # Extraer datos históricos
        scores = []
        total_issues = []
        dates = []
        
        for report in historical_reports:
            scores.append(report.overall_score)
            total_issues.append(report.issues_summary.get('total', 0))
            dates.append(report.timestamp.strftime('%Y-%m-%d'))
            
            trends['score_trend'].append({
                'date': report.timestamp.strftime('%Y-%m-%d'),
                'score': report.overall_score
            })
            
            trends['issues_trend'].append({
                'date': report.timestamp.strftime('%Y-%m-%d'),
                'issues': report.issues_summary.get('total', 0)
            })
        
        # Determinar tendencia
        if len(scores) >= 2:
            first_score = scores[0]
            last_score = scores[-1]
            
            if last_score > first_score + 5:  # Mejora significativa
                trends['improving'] = True
                trends['improvement_percentage'] = ((last_score - first_score) / first_score) * 100
            elif last_score < first_score - 5:  # Deterioro significativo
                trends['deteriorating'] = True
                trends['deterioration_percentage'] = ((first_score - last_score) / first_score) * 100
            else:
                trends['stable'] = True
        
        # Calcular promedios móviles (si hay suficientes datos)
        if len(scores) >= 3:
            window = min(3, len(scores))
            moving_averages = []
            
            for i in range(len(scores) - window + 1):
                avg = sum(scores[i:i+window]) / window
                moving_averages.append({
                    'date': dates[i + window - 1],
                    'moving_average': avg
                })
            
            trends['moving_averages'] = moving_averages
        
        return trends