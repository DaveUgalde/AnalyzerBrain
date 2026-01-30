"""
exhaustive_project_analyzer_v2.py
Analiza exhaustivamente un proyecto de software basado en la arquitectura Project Brain.
"""

import os
import ast
import json
import yaml
import inspect
import importlib.util
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics

class AnalysisCategory(Enum):
    STRUCTURE = "structure"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    CODE_QUALITY = "code_quality"
    FUNCTIONALITY = "functionality"
    TESTS = "tests"
    USABILITY = "usability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"

@dataclass
class AnalysisResult:
    category: AnalysisCategory
    score: float  # 0-100
    issues: List[str]
    recommendations: List[str]
    details: Dict[str, Any] = field(default_factory=dict)

class ExhaustiveProjectAnalyzer:
    """Analizador exhaustivo basado en arquitectura Project Brain"""
    
    def __init__(self, root_dir=".", architecture_docs=None):
        self.root_dir = Path(root_dir).resolve()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Cargar documentos de arquitectura si se proporcionan
        self.architecture = self._load_architecture_docs(architecture_docs)
        
        self.results: Dict[AnalysisCategory, AnalysisResult] = {}
        self.files_cache = {}
        self.import_graph = {}
        self.function_usages = {}
        
    def _load_architecture_docs(self, docs_path):
        """Carga documentos de arquitectura para validación"""
        architecture = {
            "expected_modules": [
                "src/core",
                "src/indexer", 
                "src/embeddings",
                "src/graph",
                "src/agents",
                "src/memory",
                "src/api",
                "src/learning",
                "src/utils"
            ],
            "expected_files": {
                "core": ["orchestrator.py", "system_state.py", "workflow_manager.py"],
                "indexer": ["multi_language_parser.py", "project_scanner.py"],
                "embeddings": ["embedding_generator.py", "semantic_search.py"],
                "agents": ["base_agent.py", "agent_factory.py"]
            },
            "performance_targets": {
                "parsing": {"python_file_1000_lines": "< 500ms"},
                "embeddings": {"text_512_tokens": "< 100ms"},
                "queries": {"simple_question": "< 2s p95"}
            }
        }
        return architecture
    
    def _safe_read_file(self, file_path: Path, default_encoding: str = 'utf-8') -> Tuple[Optional[str], Optional[str]]:
        """
        Lee un archivo de manera segura intentando diferentes codificaciones.
        
        Returns:
            Tuple[content, encoding] o (None, None) si no se pudo leer
        """
        if not file_path.exists():
            return None, None
        
        # Lista de codificaciones a probar
        encodings = [
            default_encoding,
            'latin-1',
            'cp1252',
            'iso-8859-1',
            'utf-16',
            'utf-16-le',
            'utf-16-be',
            'cp850',
            'cp437'
        ]
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content, encoding
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        # Si ninguna codificación funciona, intentar leer como binario y decodificar ignorando errores
        try:
            with open(file_path, 'rb') as f:
                binary_content = f.read()
            # Intentar decodificar con utf-8 ignorando errores
            content = binary_content.decode('utf-8', errors='ignore')
            return content, 'utf-8 (forced)'
        except Exception:
            return None, None
    
    def analyze_complete_project(self) -> Dict:
        """Ejecuta análisis completo del proyecto"""
        print("🚀 INICIANDO ANÁLISIS EXHAUSTIVO DEL PROYECTO")
        print("=" * 100)
        
        # 1. Análisis de estructura y organización
        print("\n📁 1. ANALIZANDO ESTRUCTURA DEL PROYECTO...")
        self.results[AnalysisCategory.STRUCTURE] = self._analyze_structure()
        
        # 2. Análisis de completitud
        print("\n✅ 2. ANALIZANDO COMPLETITUD...")
        self.results[AnalysisCategory.COMPLETENESS] = self._analyze_completeness()
        
        # 3. Análisis de coherencia y flujos
        print("\n🔄 3. ANALIZANDO COHERENCIA...")
        self.results[AnalysisCategory.COHERENCE] = self._analyze_coherence()
        
        # 4. Análisis de calidad de código
        print("\n🔧 4. ANALIZANDO CALIDAD DE CÓDIGO...")
        self.results[AnalysisCategory.CODE_QUALITY] = self._analyze_code_quality()
        
        # 5. Análisis de funcionalidad
        print("\n⚙️  5. ANALIZANDO FUNCIONALIDAD...")
        self.results[AnalysisCategory.FUNCTIONALITY] = self._analyze_functionality()
        
        # 6. Análisis de tests
        print("\n🧪 6. ANALIZANDO TESTS...")
        self.results[AnalysisCategory.TESTS] = self._analyze_tests()
        
        # 7. Análisis de usabilidad
        print("\n🎯 7. ANALIZANDO USABILIDAD...")
        self.results[AnalysisCategory.USABILITY] = self._analyze_usability()
        
        # 8. Análisis de documentación
        print("\n📚 8. ANALIZANDO DOCUMENTACIÓN...")
        self.results[AnalysisCategory.DOCUMENTATION] = self._analyze_documentation()
        
        # 9. Análisis de rendimiento
        print("\n⚡ 9. ANALIZANDO RENDIMIENTO...")
        self.results[AnalysisCategory.PERFORMANCE] = self._analyze_performance()
        
        # 10. Análisis de seguridad
        print("\n🔒 10. ANALIZANDO SEGURIDAD...")
        self.results[AnalysisCategory.SECURITY] = self._analyze_security()
        
        # Generar reporte final
        report = self._generate_comprehensive_report()
        
        print(f"\n{'=' * 100}")
        print("✅ ANÁLISIS COMPLETADO")
        print(f"{'=' * 100}")
        
        return report
    
    def _analyze_structure(self) -> AnalysisResult:
        """Analiza estructura del proyecto"""
        issues = []
        recommendations = []
        details = {}
        
        # Mapear estructura completa
        structure_map = {}
        for root, dirs, files in os.walk(self.root_dir):
            rel_path = Path(root).relative_to(self.root_dir)
            
            # Excluir directorios no deseados
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      d not in ['__pycache__', 'venv', 'env', 'node_modules']]
            
            python_files = [f for f in files if f.endswith('.py')]
            other_files = [f for f in files if not f.endswith('.py')]
            
            structure_map[str(rel_path)] = {
                "python_files": python_files,
                "other_files": other_files,
                "total_files": len(files),
                "subdirectories": dirs.copy()
            }
        
        # Analizar organización modular
        modules_found = []
        expected_modules = self.architecture["expected_modules"]
        
        for module in expected_modules:
            module_path = self.root_dir / module
            if module_path.exists():
                files_count = len(list(module_path.rglob("*.py")))
                modules_found.append({
                    "name": module,
                    "exists": True,
                    "files": files_count,
                    "path": str(module_path.relative_to(self.root_dir))
                })
            else:
                modules_found.append({
                    "name": module,
                    "exists": False,
                    "files": 0,
                    "path": module
                })
                issues.append(f"Módulo faltante: {module}")
                recommendations.append(f"Crear estructura para: {module}")
        
        # Calcular métricas de estructura
        total_py_files = sum(len(v["python_files"]) for v in structure_map.values())
        avg_files_per_dir = total_py_files / max(len(structure_map), 1)
        
        # Verificar profundidad de directorios
        max_depth = max(len(Path(k).parts) for k in structure_map.keys())
        if max_depth > 6:
            issues.append(f"Profundidad excesiva de directorios: {max_depth} niveles")
            recommendations.append("Simplificar estructura de directorios")
        
        # Calcular puntuación
        score = self._calculate_structure_score(modules_found, total_py_files)
        
        details = {
            "structure_map": structure_map,
            "modules_found": modules_found,
            "total_python_files": total_py_files,
            "avg_files_per_dir": avg_files_per_dir,
            "max_depth": max_depth
        }
        
        return AnalysisResult(
            category=AnalysisCategory.STRUCTURE,
            score=score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _calculate_structure_score(self, modules_found, total_py_files):
        """Calcula puntuación de estructura"""
        base_score = 0
        
        # Puntos por módulos encontrados
        found_modules = sum(1 for m in modules_found if m["exists"])
        total_modules = len(modules_found)
        module_score = (found_modules / total_modules) * 40
        
        # Puntos por distribución de archivos
        if total_py_files > 50:
            file_score = 30
        elif total_py_files > 20:
            file_score = 20
        elif total_py_files > 10:
            file_score = 10
        else:
            file_score = 5
        
        # Puntos por organización
        org_score = 20  # Base
        # Podría añadir más criterios
        
        return min(module_score + file_score + org_score, 100)
    
    def _analyze_completeness(self) -> AnalysisResult:
        """Analiza completitud del proyecto"""
        issues = []
        recommendations = []
        
        # Obtener todos los archivos Python
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        # Analizar completitud por módulo
        completeness_by_module = {}
        
        for module in self.architecture["expected_modules"]:
            module_path = self.root_dir / module
            if not module_path.exists():
                continue
            
            # Archivos esperados en este módulo
            expected_files = self.architecture["expected_files"].get(
                Path(module).name, []
            )
            
            # Archivos encontrados
            found_files = []
            missing_files = []
            
            for expected_file in expected_files:
                file_path = module_path / expected_file
                if file_path.exists():
                    found_files.append(expected_file)
                    
                    # Analizar completitud del archivo
                    file_completeness = self._analyze_file_completeness(file_path)
                    if not file_completeness["complete"]:
                        issues.append(f"Archivo incompleto: {file_path}")
                        recommendations.append(f"Completar {file_path}: {file_completeness['missing']}")
                else:
                    missing_files.append(expected_file)
                    issues.append(f"Archivo faltante: {file_path}")
                    recommendations.append(f"Crear {file_path}")
            
            # Calcular porcentaje de completitud
            total_expected = len(expected_files)
            if total_expected > 0:
                completeness_pct = (len(found_files) / total_expected) * 100
            else:
                completeness_pct = 100
            
            completeness_by_module[module] = {
                "expected_files": expected_files,
                "found_files": found_files,
                "missing_files": missing_files,
                "completeness": completeness_pct
            }
        
        # Verificar dependencias entre módulos
        dependencies_complete = self._analyze_dependencies_completeness()
        
        # Calcular puntuación general
        module_scores = [v["completeness"] for v in completeness_by_module.values()]
        avg_completeness = statistics.mean(module_scores) if module_scores else 0
        
        score = avg_completeness * 0.7 + (100 if dependencies_complete else 50) * 0.3
        
        details = {
            "completeness_by_module": completeness_by_module,
            "dependencies_complete": dependencies_complete,
            "total_python_files": len(all_py_files)
        }
        
        return AnalysisResult(
            category=AnalysisCategory.COMPLETENESS,
            score=score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _analyze_file_completeness(self, file_path: Path) -> Dict:
        """Analiza completitud de un archivo específico"""
        if not file_path.exists():
            return {"complete": False, "missing": "Archivo no existe"}
        
        # Leer archivo de manera segura
        content, encoding = self._safe_read_file(file_path)
        
        if content is None:
            return {"complete": False, "missing": "No se pudo leer el archivo"}
        
        try:
            # Verificar que no sea solo un stub
            lines = content.strip().split('\n')
            non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            
            if len(non_empty_lines) < 5:
                return {"complete": False, "missing": "Contenido insuficiente"}
            
            # Verificar que tenga funciones/clases definidas
            try:
                tree = ast.parse(content)
                has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
                has_classes = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
                
                if not has_functions and not has_classes:
                    return {"complete": False, "missing": "No tiene funciones/clases definidas"}
            
            except SyntaxError:
                return {"complete": False, "missing": "Error de sintaxis"}
            
            return {"complete": True, "missing": "", "encoding": encoding}
            
        except Exception as e:
            return {"complete": False, "missing": f"Error al analizar: {str(e)}"}
    
    def _analyze_dependencies_completeness(self) -> bool:
        """Analiza completitud de dependencias"""
        # Buscar archivos de dependencias
        dep_files = [
            "requirements.txt", "pyproject.toml", "setup.py", 
            "requirements/base.txt", "requirements/prod.txt"
        ]
        
        found_deps = []
        for dep_file in dep_files:
            path = self.root_dir / dep_file
            if path.exists():
                found_deps.append(dep_file)
        
        return len(found_deps) > 0
    
    def _analyze_coherence(self) -> AnalysisResult:
        """Analiza coherencia del proyecto"""
        issues = []
        recommendations = []
        
        print("   🔍 Analizando imports y dependencias...")
        import_analysis = self._analyze_imports_and_dependencies()
        issues.extend(import_analysis["issues"])
        
        print("   🔍 Analizando flujos de trabajo...")
        workflow_analysis = self._analyze_workflows()
        issues.extend(workflow_analysis["issues"])
        
        print("   🔍 Analizando consistencia de interfaces...")
        interface_analysis = self._analyze_interfaces()
        issues.extend(interface_analysis["issues"])
        
        # Calcular métricas de coherencia
        coherence_score = self._calculate_coherence_score(
            import_analysis, workflow_analysis, interface_analysis
        )
        
        details = {
            "import_analysis": import_analysis,
            "workflow_analysis": workflow_analysis,
            "interface_analysis": interface_analysis
        }
        
        return AnalysisResult(
            category=AnalysisCategory.COHERENCE,
            score=coherence_score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _analyze_imports_and_dependencies(self) -> Dict:
        """Analiza imports y dependencias para detectar incoherencias"""
        issues = []
        details = {}
        
        # Construir grafo de imports
        import_graph = {}
        circular_deps = []
        unused_imports = []
        encoding_info = {}
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files:
            if '__pycache__' in str(py_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    issues.append(f"No se pudo leer el archivo: {py_file}")
                    continue
                
                encoding_info[str(py_file.relative_to(self.root_dir))] = encoding
                
                tree = ast.parse(content)
                
                # Extraer imports
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}" if module else alias.name)
                
                import_graph[str(py_file.relative_to(self.root_dir))] = imports
                
                # Verificar imports no utilizados (análisis básico)
                # Esto es complejo y requeriría análisis más profundo
                
            except SyntaxError as e:
                issues.append(f"Error de sintaxis en {py_file}: {e}")
            except Exception as e:
                issues.append(f"Error al procesar {py_file}: {e}")
        
        # Detectar dependencias circulares (simplificado)
        visited = set()
        path = set()
        
        def dfs(node):
            if node in path:
                return [list(path) + [node]]
            if node in visited:
                return []
            
            visited.add(node)
            path.add(node)
            
            cycles = []
            for neighbor in import_graph.get(node, []):
                cycles.extend(dfs(neighbor))
            
            path.remove(node)
            return cycles
        
        for node in import_graph:
            cycles = dfs(node)
            if cycles:
                circular_deps.extend(cycles)
        
        if circular_deps:
            issues.append(f"Dependencias circulares detectadas: {len(circular_deps)}")
        
        details = {
            "import_graph": import_graph,
            "circular_dependencies": circular_deps,
            "unused_imports": unused_imports,
            "encoding_info": encoding_info
        }
        
        return {"issues": issues, "details": details}
    
    def _analyze_workflows(self) -> Dict:
        """Analiza coherencia de flujos de trabajo"""
        issues = []
        details = {}
        
        # Buscar scripts de flujo de trabajo
        workflow_files = []
        for pattern in ["run_*.py", "pipeline_*.py", "workflow_*.py", "main.py"]:
            workflow_files.extend(self.root_dir.rglob(pattern))
        
        workflows = []
        for wf_file in workflow_files:
            if '__pycache__' in str(wf_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(wf_file)
                
                if content is None:
                    issues.append(f"No se pudo leer el workflow: {wf_file}")
                    continue
                
                # Analizar flujo básico
                lines = content.split('\n')
                workflow_info = {
                    "file": str(wf_file.relative_to(self.root_dir)),
                    "lines": len(lines),
                    "has_main": 'if __name__ == "__main__"' in content,
                    "function_calls": self._extract_function_calls(content),
                    "imports": self._extract_imports(content),
                    "encoding": encoding
                }
                workflows.append(workflow_info)
                
                # Verificar flujo completo
                if not workflow_info["has_main"] and "def main" not in content:
                    issues.append(f"Workflow sin punto de entrada claro: {wf_file}")
                
            except Exception as e:
                issues.append(f"Error al analizar workflow {wf_file}: {e}")
        
        # Verificar conectividad entre workflows
        if len(workflows) > 1:
            # Podríamos verificar si los workflows se llaman entre sí
            pass
        
        details = {"workflows": workflows}
        
        return {"issues": issues, "details": details}
    
    def _extract_function_calls(self, content: str) -> List[str]:
        """Extrae llamadas a funciones del contenido"""
        # Implementación simplificada
        calls = []
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            if '(' in stripped and ')' in stripped and '=' not in stripped.split('(')[0]:
                # Extraer nombre de función
                func_part = stripped.split('(')[0]
                if '.' in func_part:
                    calls.append(func_part)
                elif ' ' not in func_part:
                    calls.append(func_part)
        return calls[:10]  # Limitar para no hacer muy grande
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extrae imports del contenido"""
        imports = []
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                imports.append(stripped)
        return imports[:10]  # Limitar
    
    def _analyze_interfaces(self) -> Dict:
        """Analiza consistencia de interfaces"""
        issues = []
        details = {}
        
        # Buscar archivos con interfaces definidas
        interface_files = []
        for pattern in ["*_api.py", "*_interface.py", "base_*.py", "abstract_*.py"]:
            interface_files.extend(self.root_dir.rglob(pattern))
        
        interfaces = []
        for int_file in interface_files:
            if '__pycache__' in str(int_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(int_file)
                
                if content is None:
                    issues.append(f"No se pudo leer la interfaz: {int_file}")
                    continue
                
                tree = ast.parse(content)
                
                # Extraer definiciones de interfaces
                interface_defs = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Verificar si es clase base/abstracta
                        is_abstract = any(
                            isinstance(n, ast.FunctionDef) and any(
                                hasattr(decorator, 'id') and decorator.id == 'abstractmethod' 
                                for decorator in n.decorator_list 
                            )
                            for n in node.body
                        )
                        
                        if is_abstract or 'Base' in node.name or 'Abstract' in node.name:
                            methods = []
                            for n in node.body:
                                if isinstance(n, ast.FunctionDef):
                                    methods.append({
                                        "name": n.name,
                                        "args": [arg.arg for arg in n.args.args],
                                        "is_abstract": any(
                                            hasattr(decorator, 'id') and decorator.id == 'abstractmethod'
                                            for decorator in n.decorator_list
                                        )
                                    })
                            
                            interface_defs.append({
                                "name": node.name,
                                "methods": methods,
                                "is_abstract": is_abstract
                            })
                
                if interface_defs:
                    interfaces.append({
                        "file": str(int_file.relative_to(self.root_dir)),
                        "interfaces": interface_defs,
                        "encoding": encoding
                    })
                
            except Exception as e:
                issues.append(f"Error al analizar interfaz {int_file}: {e}")
        
        # Verificar consistencia
        for interface in interfaces:
            for int_def in interface["interfaces"]:
                # Verificar que las interfaces abstractas tengan implementaciones
                # Esto requeriría análisis más profundo
                pass
        
        details = {"interfaces": interfaces}
        
        return {"issues": issues, "details": details}
    
    def _calculate_coherence_score(self, import_analysis, workflow_analysis, interface_analysis):
        """Calcula puntuación de coherencia"""
        score = 100
        
        # Penalizar por issues
        total_issues = (
            len(import_analysis["issues"]) + 
            len(workflow_analysis["issues"]) + 
            len(interface_analysis["issues"])
        )
        
        score -= total_issues * 5
        
        # Penalizar por dependencias circulares
        circular_deps = len(import_analysis["details"].get("circular_dependencies", []))
        score -= circular_deps * 10
        
        # Recompensar por workflows bien definidos
        workflows = workflow_analysis["details"].get("workflows", [])
        valid_workflows = sum(1 for wf in workflows if wf.get("has_main", False))
        if workflows:
            workflow_score = (valid_workflows / len(workflows)) * 20
            score += workflow_score
        
        return max(0, min(score, 100))
    
    def _analyze_code_quality(self) -> AnalysisResult:
        """Analiza calidad de código"""
        issues = []
        recommendations = []
        
        print("   🔍 Analizando sintaxis...")
        syntax_analysis = self._analyze_syntax()
        issues.extend(syntax_analysis["issues"])
        
        print("   🔍 Analizando complejidad...")
        complexity_analysis = self._analyze_complexity()
        issues.extend(complexity_analysis["issues"])
        
        print("   🔍 Analizando convenciones...")
        convention_analysis = self._analyze_conventions()
        issues.extend(convention_analysis["issues"])
        
        print("   🔍 Analizando completitud de funciones...")
        function_analysis = self._analyze_function_completeness()
        issues.extend(function_analysis["issues"])
        recommendations.extend(function_analysis["recommendations"])
        
        # Calcular puntuación
        quality_score = self._calculate_quality_score(
            syntax_analysis, complexity_analysis, 
            convention_analysis, function_analysis
        )
        
        details = {
            "syntax_analysis": syntax_analysis,
            "complexity_analysis": complexity_analysis,
            "convention_analysis": convention_analysis,
            "function_analysis": function_analysis
        }
        
        return AnalysisResult(
            category=AnalysisCategory.CODE_QUALITY,
            score=quality_score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _analyze_syntax(self) -> Dict:
        """Analiza errores de sintaxis"""
        issues = []
        details = {"files_with_errors": [], "total_errors": 0}
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files:
            if '__pycache__' in str(py_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    details["files_with_errors"].append({
                        "file": str(py_file.relative_to(self.root_dir)),
                        "error": "No se pudo leer el archivo",
                        "line": "desconocido",
                        "encoding": "desconocido"
                    })
                    details["total_errors"] += 1
                    continue
                
                # Intentar parsear
                ast.parse(content)
                
            except SyntaxError as e:
                issues.append(f"Error de sintaxis en {py_file}: {e}")
                details["files_with_errors"].append({
                    "file": str(py_file.relative_to(self.root_dir)),
                    "error": str(e),
                    "line": e.lineno if hasattr(e, 'lineno') else "desconocido",
                    "encoding": encoding
                })
                details["total_errors"] += 1
        
        return {"issues": issues, "details": details}
    
    def _analyze_complexity(self) -> Dict:
        """Analiza complejidad del código"""
        issues = []
        details = {"file_complexities": [], "high_complexity_files": []}
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files:
            if '__pycache__' in str(py_file) or 'test' in str(py_file).lower():
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                tree = ast.parse(content)
                
                # Calcular complejidad ciclomática aproximada
                complexity = 1  # Base
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                        complexity += 1
                    elif isinstance(node, ast.ExceptHandler):
                        complexity += 1
                    elif isinstance(node, ast.BoolOp):
                        complexity += len(node.values) - 1
                
                # Contar funciones y líneas
                functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                lines = len(content.split('\n'))
                
                file_info = {
                    "file": str(py_file.relative_to(self.root_dir)),
                    "complexity": complexity,
                    "functions": len(functions),
                    "lines": lines,
                    "complexity_per_function": complexity / max(len(functions), 1),
                    "encoding": encoding
                }
                
                details["file_complexities"].append(file_info)
                
                # Marcar archivos muy complejos
                if complexity > 30:
                    issues.append(f"Alta complejidad en {py_file}: {complexity}")
                    details["high_complexity_files"].append(file_info)
                
            except Exception as e:
                pass
        
        return {"issues": issues, "details": details}
    
    def _analyze_conventions(self) -> Dict:
        """Analiza convenciones de código"""
        issues = []
        details = {"convention_violations": []}
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files:
            if '__pycache__' in str(py_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                lines = content.split('\n')
                violations = []
                
                # Verificar convenciones básicas
                for i, line in enumerate(lines, 1):
                    line_stripped = line.rstrip()
                    
                    # Longitud de línea
                    if len(line) > 120:
                        violations.append(f"Línea {i}: Excede 120 caracteres")
                    
                    # Espacios en blanco al final
                    if line_stripped and line_stripped[-1] == ' ':
                        violations.append(f"Línea {i}: Espacios al final")
                
                if violations:
                    details["convention_violations"].append({
                        "file": str(py_file.relative_to(self.root_dir)),
                        "violations": violations,
                        "encoding": encoding
                    })
                    issues.append(f"Convenciones violadas en {py_file}: {len(violations)} issues")
                
            except Exception as e:
                pass
        
        return {"issues": issues, "details": details}
    
    def _analyze_function_completeness(self) -> Dict:
        """Analiza completitud de funciones"""
        issues = []
        recommendations = []
        details = {"incomplete_functions": []}
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files:
            if '__pycache__' in str(py_file) or 'test' in str(py_file).lower():
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Verificar si la función tiene cuerpo
                        if len(node.body) == 0:
                            issues.append(f"Función vacía: {node.name} en {py_file}")
                            details["incomplete_functions"].append({
                                "file": str(py_file.relative_to(self.root_dir)),
                                "function": node.name,
                                "issue": "Función vacía",
                                "encoding": encoding
                            })
                            recommendations.append(f"Implementar función {node.name} en {py_file}")
                        
                        # Verificar si solo tiene pass
                        elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                            issues.append(f"Función solo con pass: {node.name} en {py_file}")
                            details["incomplete_functions"].append({
                                "file": str(py_file.relative_to(self.root_dir)),
                                "function": node.name,
                                "issue": "Solo tiene pass",
                                "encoding": encoding
                            })
                            recommendations.append(f"Implementar función {node.name} en {py_file}")
                        
                        # Verificar si tiene docstring pero no implementación
                        elif len(node.body) == 1 and isinstance(node.body[0], ast.Expr):
                            if isinstance(node.body[0].value, ast.Constant):
                                issues.append(f"Función solo con docstring: {node.name} en {py_file}")
                                details["incomplete_functions"].append({
                                    "file": str(py_file.relative_to(self.root_dir)),
                                    "function": node.name,
                                    "issue": "Solo tiene docstring",
                                    "encoding": encoding
                                })
                                recommendations.append(f"Implementar función {node.name} en {py_file}")
                
            except Exception as e:
                pass
        
        return {
            "issues": issues, 
            "recommendations": recommendations,
            "details": details
        }
    
    def _calculate_quality_score(self, syntax_analysis, complexity_analysis, 
                               convention_analysis, function_analysis):
        """Calcula puntuación de calidad"""
        score = 100
        
        # Penalizar por errores de sintaxis
        syntax_errors = syntax_analysis["details"]["total_errors"]
        score -= syntax_errors * 15
        
        # Penalizar por alta complejidad
        high_complexity = len(complexity_analysis["details"]["high_complexity_files"])
        score -= high_complexity * 10
        
        # Penalizar por violaciones de convenciones
        convention_violations = len(convention_analysis["details"]["convention_violations"])
        score -= convention_violations * 5
        
        # Penalizar por funciones incompletas
        incomplete_functions = len(function_analysis["details"]["incomplete_functions"])
        score -= incomplete_functions * 20
        
        return max(0, min(score, 100))
    
    def _analyze_functionality(self) -> AnalysisResult:
        """Analiza funcionalidad del proyecto"""
        issues = []
        recommendations = []
        
        print("   🔍 Analizando uso de funcionalidades...")
        usage_analysis = self._analyze_function_usage()
        issues.extend(usage_analysis["issues"])
        
        print("   🔍 Analizando implementaciones...")
        implementation_analysis = self._analyze_implementations()
        issues.extend(implementation_analysis["issues"])
        recommendations.extend(implementation_analysis["recommendations"])
        
        print("   🔍 Analizando integración...")
        integration_analysis = self._analyze_integration()
        issues.extend(integration_analysis["issues"])
        
        # Calcular puntuación
        functionality_score = self._calculate_functionality_score(
            usage_analysis, implementation_analysis, integration_analysis
        )
        
        details = {
            "usage_analysis": usage_analysis,
            "implementation_analysis": implementation_analysis,
            "integration_analysis": integration_analysis
        }
        
        return AnalysisResult(
            category=AnalysisCategory.FUNCTIONALITY,
            score=functionality_score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _analyze_function_usage(self) -> Dict:
        """Analiza uso de funciones"""
        issues = []
        details = {"unused_functions": [], "function_call_counts": {}}
        
        # Este análisis es complejo y simplificado
        # En una implementación real, se usaría herramientas como vulture o pylint
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        # Contar definiciones de funciones
        function_defs = {}
        for py_file in all_py_files:
            if '__pycache__' in str(py_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        if func_name.startswith('_') and not func_name.startswith('__'):
                            continue  # Ignorar métodos privados
                        
                        key = f"{py_file}::{func_name}"
                        function_defs[key] = {
                            "file": str(py_file.relative_to(self.root_dir)),
                            "function": func_name,
                            "line": node.lineno,
                            "encoding": encoding
                        }
                
            except Exception as e:
                pass
        
        # Buscar usos de funciones (simplificado)
        # En realidad, necesitaríamos análisis de flujo de datos
        
        details["total_functions_defined"] = len(function_defs)
        details["function_definitions"] = list(function_defs.values())[:20]  # Mostrar solo algunas
        
        return {"issues": issues, "details": details}
    
    def _analyze_implementations(self) -> Dict:
        """Analiza implementaciones de funciones"""
        issues = []
        recommendations = []
        details = {"implementation_issues": []}
        
        # Buscar funciones que deberían estar implementadas
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files:
            if '__pycache__' in str(py_file) or 'test' in str(py_file).lower():
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name.lower()
                        
                        # Verificar funciones críticas según arquitectura
                        critical_patterns = [
                            "process", "analyze", "generate", "train",
                            "predict", "evaluate", "validate", "execute"
                        ]
                        
                        if any(pattern in func_name for pattern in critical_patterns):
                            # Verificar si está bien implementada
                            if len(node.body) < 3:  # Muy pequeña
                                issues.append(f"Función crítica '{node.name}' parece subimplementada en {py_file}")
                                details["implementation_issues"].append({
                                    "file": str(py_file.relative_to(self.root_dir)),
                                    "function": node.name,
                                    "issue": "Posible subimplementación",
                                    "body_size": len(node.body),
                                    "encoding": encoding
                                })
                                recommendations.append(f"Revisar implementación de {node.name} en {py_file}")
                
            except Exception as e:
                pass
        
        return {
            "issues": issues,
            "recommendations": recommendations,
            "details": details
        }
    
    def _analyze_integration(self) -> Dict:
        """Analiza integración entre componentes"""
        issues = []
        details = {"integration_points": [], "integration_issues": []}
        
        # Buscar puntos de integración
        integration_patterns = [
            "import.*api", "from.*api", "api_client", "client.*connect",
            "requests.get", "requests.post", "subprocess.run", "websocket"
        ]
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files:
            if '__pycache__' in str(py_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                # Buscar patrones de integración
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    
                    for pattern in integration_patterns:
                        if pattern in line_lower:
                            details["integration_points"].append({
                                "file": str(py_file.relative_to(self.root_dir)),
                                "line": i,
                                "code": line.strip()[:100],
                                "pattern": pattern,
                                "encoding": encoding
                            })
                            
                            # Verificar manejo de errores
                            if "try:" not in content and "except" not in content:
                                if "requests" in line_lower or "http" in line_lower:
                                    issues.append(f"Llamada HTTP sin manejo de errores en {py_file}:{i}")
                                    details["integration_issues"].append({
                                        "file": str(py_file.relative_to(self.root_dir)),
                                        "line": i,
                                        "issue": "Falta manejo de errores en llamada HTTP",
                                        "encoding": encoding
                                    })
                
            except Exception as e:
                pass
        
        return {"issues": issues, "details": details}
    
    def _calculate_functionality_score(self, usage_analysis, implementation_analysis, 
                                     integration_analysis):
        """Calcula puntuación de funcionalidad"""
        score = 80  # Base
        
        # Penalizar por issues de implementación
        implementation_issues = len(implementation_analysis["issues"])
        score -= implementation_issues * 10
        
        # Penalizar por issues de integración
        integration_issues = len(integration_analysis["issues"])
        score -= integration_issues * 15
        
        # Recompensar por funciones definidas
        total_functions = usage_analysis["details"].get("total_functions_defined", 0)
        if total_functions > 50:
            score += 10
        elif total_functions > 20:
            score += 5
        
        # Recompensar por puntos de integración
        integration_points = len(integration_analysis["details"].get("integration_points", []))
        if integration_points > 10:
            score += 5
        
        return max(0, min(score, 100))
    
    def _analyze_tests(self) -> AnalysisResult:
        """Analiza tests del proyecto"""
        issues = []
        recommendations = []
        
        print("   🔍 Buscando archivos de test...")
        test_files = self._find_test_files()
        
        print("   🔍 Analizando cobertura de tests...")
        coverage_analysis = self._analyze_test_coverage(test_files)
        issues.extend(coverage_analysis["issues"])
        recommendations.extend(coverage_analysis["recommendations"])
        
        print("   🔍 Analizando calidad de tests...")
        quality_analysis = self._analyze_test_quality(test_files)
        issues.extend(quality_analysis["issues"])
        
        print("   🔍 Analizando tests faltantes...")
        missing_analysis = self._analyze_missing_tests()
        issues.extend(missing_analysis["issues"])
        recommendations.extend(missing_analysis["recommendations"])
        
        # Calcular puntuación
        test_score = self._calculate_test_score(
            test_files, coverage_analysis, quality_analysis, missing_analysis
        )
        
        details = {
            "test_files": test_files,
            "coverage_analysis": coverage_analysis,
            "quality_analysis": quality_analysis,
            "missing_analysis": missing_analysis
        }
        
        return AnalysisResult(
            category=AnalysisCategory.TESTS,
            score=test_score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _find_test_files(self) -> List[Dict]:
        """Encuentra archivos de test"""
        test_files = []
        
        patterns = ["test_*.py", "*_test.py", "*test*.py", "*spec*.py"]
        
        for pattern in patterns:
            for test_file in self.root_dir.rglob(pattern):
                if '__pycache__' in str(test_file):
                    continue
                
                # Excluir si está en directorios no deseados
                rel_path = str(test_file.relative_to(self.root_dir))
                if any(part.startswith('.') for part in Path(rel_path).parts):
                    continue
                
                try:
                    # Leer archivo de manera segura
                    content, encoding = self._safe_read_file(test_file)
                    
                    if content is None:
                        test_files.append({
                            "file": rel_path,
                            "error": "No se pudo leer el archivo",
                            "encoding": "desconocido"
                        })
                        continue
                    
                    # Contar casos de test
                    test_cases = 0
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name.startswith('test'):
                            test_cases += 1
                        elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                            for subnode in node.body:
                                if isinstance(subnode, ast.FunctionDef) and subnode.name.startswith('test'):
                                    test_cases += 1
                    
                    test_files.append({
                        "file": rel_path,
                        "test_cases": test_cases,
                        "lines": len(content.split('\n')),
                        "size_kb": test_file.stat().st_size / 1024,
                        "encoding": encoding
                    })
                    
                except Exception as e:
                    test_files.append({
                        "file": rel_path,
                        "error": str(e),
                        "encoding": "desconocido"
                    })
        
        return test_files
    
    def _analyze_test_coverage(self, test_files: List[Dict]) -> Dict:
        """Analiza cobertura de tests"""
        issues = []
        recommendations = []
        details = {"coverage_by_module": {}}
        
        # Módulos que deberían tener tests
        modules_to_test = self.architecture["expected_modules"]
        
        for module in modules_to_test:
            module_path = self.root_dir / module
            if not module_path.exists():
                continue
            
            # Contar archivos en el módulo
            module_files = list(module_path.rglob("*.py"))
            module_files = [f for f in module_files if '__pycache__' not in str(f)]
            
            # Buscar tests para este módulo
            module_tests = []
            for test_file in test_files:
                if "error" not in test_file and module in test_file["file"]:
                    module_tests.append(test_file)
            
            coverage_pct = 0
            if module_files:
                # Estimación simple: si hay tests, asumimos algo de cobertura
                coverage_pct = min(len(module_tests) / len(module_files) * 100, 100)
            
            details["coverage_by_module"][module] = {
                "files": len(module_files),
                "tests": len(module_tests),
                "coverage_percentage": coverage_pct
            }
            
            if coverage_pct < 50:
                issues.append(f"Cobertura baja para módulo {module}: {coverage_pct:.1f}%")
                recommendations.append(f"Añadir tests para módulo {module}")
        
        return {
            "issues": issues,
            "recommendations": recommendations,
            "details": details
        }
    
    def _analyze_test_quality(self, test_files: List[Dict]) -> Dict:
        """Analiza calidad de los tests"""
        issues = []
        details = {"test_quality_issues": []}
        
        for test_file in test_files:
            if "error" in test_file:
                continue
            
            file_path = self.root_dir / test_file["file"]
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(file_path)
                
                if content is None:
                    continue
                
                # Verificar aspectos de calidad
                quality_issues = []
                
                # Tiene imports necesarios
                if "import unittest" not in content and "import pytest" not in content:
                    quality_issues.append("No importa framework de testing")
                
                # Tiene asserts
                if "assert" not in content and "self.assertEqual" not in content:
                    quality_issues.append("No tiene asserts")
                
                # Tiene setup/teardown
                has_setup = "def setUp" in content or "def setup" in content
                has_teardown = "def tearDown" in content or "def teardown" in content
                
                if not has_setup and test_file["test_cases"] > 3:
                    quality_issues.append("Podría beneficiarse de setup")
                
                if quality_issues:
                    details["test_quality_issues"].append({
                        "file": test_file["file"],
                        "issues": quality_issues,
                        "encoding": encoding
                    })
                    issues.append(f"Problemas de calidad en tests: {test_file['file']}")
                
            except Exception as e:
                pass
        
        return {"issues": issues, "details": details}
    
    def _analyze_missing_tests(self) -> Dict:
        """Identifica tests faltantes"""
        issues = []
        recommendations = []
        details = {"missing_tests": []}
        
        # Buscar módulos críticos sin tests
        critical_modules = ["src/core", "src/indexer", "src/embeddings"]
        
        for module in critical_modules:
            module_path = self.root_dir / module
            if not module_path.exists():
                continue
            
            # Buscar archivos Python en el módulo
            module_files = list(module_path.rglob("*.py"))
            module_files = [f for f in module_files if '__pycache__' not in str(f)]
            
            for module_file in module_files[:10]:  # Limitar para no sobrecargar
                rel_path = str(module_file.relative_to(self.root_dir))
                file_name = module_file.name
                
                # Verificar si hay test correspondiente
                test_file_name = f"test_{file_name}"
                test_file_paths = [
                    self.root_dir / "tests" / test_file_name,
                    self.root_dir / "tests" / module_path.name / test_file_name,
                    module_path.parent / "tests" / test_file_name
                ]
                
                test_exists = any(p.exists() for p in test_file_paths)
                
                if not test_exists:
                    details["missing_tests"].append({
                        "module_file": rel_path,
                        "expected_test": test_file_name
                    })
        
        if details["missing_tests"]:
            issues.append(f"Tests faltantes: {len(details['missing_tests'])} archivos sin tests")
            recommendations.append("Crear tests para los archivos críticos")
        
        return {
            "issues": issues,
            "recommendations": recommendations,
            "details": details
        }
    
    def _calculate_test_score(self, test_files, coverage_analysis, 
                            quality_analysis, missing_analysis):
        """Calcula puntuación de tests"""
        if not test_files:
            return 0
        
        score = 0
        
        # Puntos por existencia de tests
        total_test_cases = sum(tf.get("test_cases", 0) for tf in test_files)
        if total_test_cases > 100:
            score += 40
        elif total_test_cases > 50:
            score += 30
        elif total_test_cases > 20:
            score += 20
        elif total_test_cases > 10:
            score += 10
        
        # Puntos por cobertura
        coverage_data = coverage_analysis["details"]["coverage_by_module"]
        if coverage_data:
            avg_coverage = statistics.mean(
                v["coverage_percentage"] for v in coverage_data.values()
            )
            score += min(avg_coverage * 0.4, 40)  # Hasta 40 puntos por cobertura
        
        # Penalizar por issues de calidad
        quality_issues = len(quality_analysis["details"]["test_quality_issues"])
        score -= quality_issues * 5
        
        # Penalizar por tests faltantes
        missing_tests = len(missing_analysis["details"]["missing_tests"])
        score -= missing_tests * 3
        
        return max(0, min(score, 100))
    
    def _analyze_usability(self) -> AnalysisResult:
        """Analiza usabilidad del proyecto"""
        issues = []
        recommendations = []
        details = {}
        
        # Verificar puntos de entrada
        entry_points = self._find_entry_points()
        details["entry_points"] = entry_points
        
        # Verificar documentación de uso
        usage_docs = self._check_usage_documentation()
        details["usage_docs"] = usage_docs
        
        # Verificar configuración
        config_files = self._check_configuration()
        details["config_files"] = config_files
        
        # Verificar dependencias
        dependencies = self._check_dependencies()
        details["dependencies"] = dependencies
        
        # Calcular puntuación
        usability_score = self._calculate_usability_score(
            entry_points, usage_docs, config_files, dependencies
        )
        
        # Generar issues y recomendaciones
        if not entry_points:
            issues.append("No se encontraron puntos de entrada claros (main, run_*.py, etc.)")
            recommendations.append("Crear script principal o puntos de entrada claros")
        
        if not usage_docs.get("has_readme", False):
            issues.append("Falta README o documentación básica")
            recommendations.append("Crear README.md con instrucciones básicas")
        
        if not config_files:
            issues.append("No se encontraron archivos de configuración")
            recommendations.append("Crear archivos de configuración (config.yaml, .env, etc.)")
        
        return AnalysisResult(
            category=AnalysisCategory.USABILITY,
            score=usability_score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _find_entry_points(self) -> List[Dict]:
        """Encuentra puntos de entrada del proyecto"""
        entry_points = []
        
        # Buscar scripts principales
        patterns = ["main.py", "run_*.py", "app.py", "cli.py", "__main__.py"]
        
        for pattern in patterns:
            for entry_file in self.root_dir.rglob(pattern):
                if '__pycache__' in str(entry_file):
                    continue
                
                try:
                    # Leer archivo de manera segura
                    content, encoding = self._safe_read_file(entry_file)
                    
                    if content is None:
                        entry_points.append({
                            "file": str(entry_file.relative_to(self.root_dir)),
                            "error": "No se pudo leer el archivo",
                            "encoding": "desconocido"
                        })
                        continue
                    
                    # Verificar si tiene punto de entrada
                    has_main = 'if __name__ == "__main__"' in content
                    has_main_function = 'def main' in content
                    
                    entry_points.append({
                        "file": str(entry_file.relative_to(self.root_dir)),
                        "has_main_block": has_main,
                        "has_main_function": has_main_function,
                        "lines": len(content.split('\n')),
                        "encoding": encoding
                    })
                    
                except Exception as e:
                    entry_points.append({
                        "file": str(entry_file.relative_to(self.root_dir)),
                        "error": str(e),
                        "encoding": "desconocido"
                    })
        
        return entry_points
    
    def _check_usage_documentation(self) -> Dict:
        """Verifica documentación de uso"""
        docs_info = {
            "has_readme": False,
            "has_install_instructions": False,
            "has_usage_examples": False,
            "doc_files": []
        }
        
        # Buscar README
        readme_files = list(self.root_dir.glob("README*"))
        if readme_files:
            docs_info["has_readme"] = True
            
            # Analizar contenido del README
            try:
                content, encoding = self._safe_read_file(readme_files[0])
                if content:
                    content_lower = content.lower()
                    
                    if "install" in content_lower or "setup" in content_lower:
                        docs_info["has_install_instructions"] = True
                    
                    if "example" in content_lower or "usage" in content_lower:
                        docs_info["has_usage_examples"] = True
            
            except:
                pass
        
        # Buscar otros archivos de documentación
        doc_patterns = ["*.md", "docs/*", "*.rst"]
        for pattern in doc_patterns:
            for doc_file in self.root_dir.rglob(pattern):
                if '__pycache__' in str(doc_file):
                    continue
                
                rel_path = str(doc_file.relative_to(self.root_dir))
                if "README" not in rel_path.upper():
                    docs_info["doc_files"].append(rel_path)
        
        return docs_info
    
    def _check_configuration(self) -> List[Dict]:
        """Verifica archivos de configuración"""
        config_files = []
        
        patterns = ["*.yaml", "*.yml", "*.json", "*.ini", "*.cfg", ".env*", "config/*"]
        
        for pattern in patterns:
            for config_file in self.root_dir.rglob(pattern):
                if '__pycache__' in str(config_file):
                    continue
                
                # Excluir algunos archivos
                rel_path = str(config_file.relative_to(self.root_dir))
                if any(part.startswith('.') and part != '.env' for part in Path(rel_path).parts):
                    continue
                
                config_files.append({
                    "file": rel_path,
                    "size_kb": config_file.stat().st_size / 1024
                })
        
        return config_files
    
    def _check_dependencies(self) -> Dict:
        """Verifica gestión de dependencias"""
        deps_info = {
            "has_requirements": False,
            "has_setup_py": False,
            "has_pyproject": False,
            "dependency_files": []
        }
        
        # Buscar archivos de dependencias
        dep_files = [
            "requirements.txt", "requirements/*.txt", 
            "setup.py", "pyproject.toml", "Pipfile", "environment.yml"
        ]
        
        for dep_pattern in dep_files:
            for dep_file in self.root_dir.rglob(dep_pattern):
                if '__pycache__' in str(dep_file):
                    continue
                
                rel_path = str(dep_file.relative_to(self.root_dir))
                deps_info["dependency_files"].append(rel_path)
                
                if "requirements" in rel_path:
                    deps_info["has_requirements"] = True
                elif "setup.py" in rel_path:
                    deps_info["has_setup_py"] = True
                elif "pyproject.toml" in rel_path:
                    deps_info["has_pyproject"] = True
        
        return deps_info
    
    def _calculate_usability_score(self, entry_points, usage_docs, 
                                 config_files, dependencies):
        """Calcula puntuación de usabilidad"""
        score = 0
        
        # Puntos por puntos de entrada
        if entry_points:
            score += 30
            valid_entry_points = sum(1 for ep in entry_points 
                                   if ep.get("has_main_block", False) or 
                                   ep.get("has_main_function", False))
            if valid_entry_points > 0:
                score += 10
        
        # Puntos por documentación
        if usage_docs["has_readme"]:
            score += 20
            if usage_docs["has_install_instructions"]:
                score += 10
            if usage_docs["has_usage_examples"]:
                score += 10
        
        # Puntos por configuración
        if config_files:
            score += 10
        
        # Puntos por dependencias
        if dependencies["has_requirements"] or dependencies["has_setup_py"]:
            score += 20
        
        return min(score, 100)
    
    def _analyze_documentation(self) -> AnalysisResult:
        """Analiza documentación del proyecto"""
        issues = []
        recommendations = []
        details = {}
        
        # Buscar documentación
        doc_files = list(self.root_dir.rglob("*.md")) + list(self.root_dir.rglob("*.rst"))
        doc_files = [f for f in doc_files if '__pycache__' not in str(f)]
        
        # Analizar documentación en código
        code_docs = self._analyze_code_documentation()
        
        # Calcular métricas
        total_doc_files = len(doc_files)
        avg_doc_lines = 0
        if doc_files:
            total_lines = 0
            for doc_file in doc_files[:10]:  # Muestreo
                try:
                    content, encoding = self._safe_read_file(doc_file)
                    if content:
                        total_lines += len(content.split('\n'))
                except:
                    pass
            avg_doc_lines = total_lines / len(doc_files[:10]) if doc_files[:10] else 0
        
        # Calcular puntuación
        doc_score = self._calculate_documentation_score(
            total_doc_files, avg_doc_lines, code_docs
        )
        
        # Generar issues y recomendaciones
        if total_doc_files < 3:
            issues.append("Documentación insuficiente")
            recommendations.append("Crear documentación básica: README, INSTALL, USAGE")
        
        if code_docs["docstring_percentage"] < 50:
            issues.append(f"Documentación en código baja: {code_docs['docstring_percentage']:.1f}%")
            recommendations.append("Añadir docstrings a funciones y clases importantes")
        
        details = {
            "total_doc_files": total_doc_files,
            "avg_doc_lines": avg_doc_lines,
            "code_documentation": code_docs
        }
        
        return AnalysisResult(
            category=AnalysisCategory.DOCUMENTATION,
            score=doc_score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _analyze_code_documentation(self) -> Dict:
        """Analiza documentación en el código"""
        details = {
            "files_with_docstrings": 0,
            "functions_with_docstrings": 0,
            "classes_with_docstrings": 0,
            "total_files_analyzed": 0,
            "total_functions": 0,
            "total_classes": 0,
            "docstring_percentage": 0
        }
        
        # Analizar muestra de archivos
        all_py_files = list(self.root_dir.rglob("*.py"))
        sample_files = all_py_files[:50]  # Muestra
        
        for py_file in sample_files:
            if '__pycache__' in str(py_file) or 'test' in str(py_file).lower():
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                tree = ast.parse(content)
                
                details["total_files_analyzed"] += 1
                
                # Verificar docstring del módulo
                if ast.get_docstring(tree):
                    details["files_with_docstrings"] += 1
                
                # Contar funciones y docstrings
                functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                details["total_functions"] += len(functions)
                
                for func in functions:
                    if ast.get_docstring(func):
                        details["functions_with_docstrings"] += 1
                
                # Contar clases y docstrings
                classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                details["total_classes"] += len(classes)
                
                for cls in classes:
                    if ast.get_docstring(cls):
                        details["classes_with_docstrings"] += 1
                
            except Exception as e:
                pass
        
        # Calcular porcentajes
        if details["total_functions"] > 0:
            func_percentage = (details["functions_with_docstrings"] / details["total_functions"]) * 100
        else:
            func_percentage = 0
        
        if details["total_classes"] > 0:
            class_percentage = (details["classes_with_docstrings"] / details["total_classes"]) * 100
        else:
            class_percentage = 0
        
        # Promedio general
        if details["total_functions"] + details["total_classes"] > 0:
            total_documented = details["functions_with_docstrings"] + details["classes_with_docstrings"]
            total_elements = details["total_functions"] + details["total_classes"]
            details["docstring_percentage"] = (total_documented / total_elements) * 100
        
        return details
    
    def _calculate_documentation_score(self, total_doc_files, avg_doc_lines, code_docs):
        """Calcula puntuación de documentación"""
        score = 0
        
        # Puntos por archivos de documentación
        if total_doc_files >= 5:
            score += 40
        elif total_doc_files >= 3:
            score += 25
        elif total_doc_files >= 1:
            score += 10
        
        # Puntos por documentación en código
        docstring_percentage = code_docs.get("docstring_percentage", 0)
        score += min(docstring_percentage * 0.6, 60)  # Hasta 60 puntos
        
        return min(score, 100)
    
    def _analyze_performance(self) -> AnalysisResult:
        """Analiza aspectos de rendimiento"""
        issues = []
        recommendations = []
        details = {}
        
        # Analizar patrones de rendimiento
        performance_patterns = self._analyze_performance_patterns()
        
        # Analizar posibles bottlenecks
        bottlenecks = self._identify_bottlenecks()
        
        # Calcular puntuación
        perf_score = self._calculate_performance_score(performance_patterns, bottlenecks)
        
        # Generar issues y recomendaciones
        if bottlenecks:
            issues.append(f"Posibles bottlenecks identificados: {len(bottlenecks)}")
            for bottleneck in bottlenecks[:3]:
                # CORRECCIÓN: usar 'issues' en lugar de 'issue'
                if bottleneck.get('issues') and len(bottleneck['issues']) > 0:
                    first_issue = bottleneck['issues'][0]
                    recommendations.append(f"Optimizar: {bottleneck['file']} - {first_issue}")
                else:
                    recommendations.append(f"Optimizar: {bottleneck['file']}")
        
        details = {
            "performance_patterns": performance_patterns,
            "bottlenecks": bottlenecks
        }
        
        return AnalysisResult(
            category=AnalysisCategory.PERFORMANCE,
            score=perf_score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _analyze_performance_patterns(self) -> List[Dict]:
        """Analiza patrones de rendimiento en el código"""
        patterns = []
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files[:30]:  # Muestra
            if '__pycache__' in str(py_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                file_patterns = []
                
                # Buscar patrones de rendimiento
                if "for.*for" in content or "while.*while" in content:
                    file_patterns.append("Nested loops")
                
                if "time.sleep" in content:
                    file_patterns.append("Sleep calls")
                
                if "requests.get" in content and "timeout" not in content:
                    file_patterns.append("HTTP calls without timeout")
                
                if "open(" in content and "with open" not in content:
                    file_patterns.append("File open without context manager")
                
                if file_patterns:
                    patterns.append({
                        "file": str(py_file.relative_to(self.root_dir)),
                        "patterns": file_patterns,
                        "encoding": encoding
                    })
                
            except Exception as e:
                pass
        
        return patterns
    
    def _identify_bottlenecks(self) -> List[Dict]:
        """Identifica posibles bottlenecks"""
        bottlenecks = []
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files[:30]:  # Muestra
            if '__pycache__' in str(py_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                lines = content.split('\n')
                issues = []
                
                for i, line in enumerate(lines, 1):
                    line_stripped = line.strip()
                    
                    # Buscar posibles bottlenecks
                    if "for " in line_stripped and "range(" in line_stripped:
                        # Verificar si es loop grande
                        if "range(1000" in line_stripped or "range(10000" in line_stripped:
                            issues.append(f"Línea {i}: Loop grande potencial")
                    
                    if "sorted(" in line_stripped or ".sort()" in line_stripped:
                        issues.append(f"Línea {i}: Operación de sorting")
                    
                    if "json.loads" in line_stripped or "json.dumps" in line_stripped:
                        issues.append(f"Línea {i}: Serialización JSON")
                
                if issues:
                    bottlenecks.append({
                        "file": str(py_file.relative_to(self.root_dir)),
                        "issues": issues,
                        "encoding": encoding
                    })
                
            except Exception as e:
                pass
        
        return bottlenecks
    
    def _calculate_performance_score(self, performance_patterns, bottlenecks):
        """Calcula puntuación de rendimiento"""
        score = 80  # Base
        
        # Penalizar por patrones de rendimiento
        total_patterns = sum(len(p["patterns"]) for p in performance_patterns)
        score -= total_patterns * 3
        
        # Penalizar por bottlenecks
        total_bottlenecks = len(bottlenecks)
        score -= total_bottlenecks * 5
        
        return max(0, min(score, 100))
    
    def _analyze_security(self) -> AnalysisResult:
        """Analiza aspectos de seguridad"""
        issues = []
        recommendations = []
        details = {}
        
        # Analizar vulnerabilidades comunes
        vulnerabilities = self._analyze_vulnerabilities()
        
        # Analizar prácticas de seguridad
        security_practices = self._analyze_security_practices()
        
        # Calcular puntuación
        security_score = self._calculate_security_score(vulnerabilities, security_practices)
        
        # Generar issues y recomendaciones
        if vulnerabilities:
            issues.append(f"Vulnerabilidades potenciales identificadas: {len(vulnerabilities)}")
            for vuln in vulnerabilities[:3]:
                if vuln.get('vulnerabilities') and len(vuln['vulnerabilities']) > 0:
                    first_vuln = vuln['vulnerabilities'][0]
                    recommendations.append(f"Corregir vulnerabilidad: {vuln['file']} - {first_vuln}")
        
        details = {
            "vulnerabilities": vulnerabilities,
            "security_practices": security_practices
        }
        
        return AnalysisResult(
            category=AnalysisCategory.SECURITY,
            score=security_score,
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _analyze_vulnerabilities(self) -> List[Dict]:
        """Analiza vulnerabilidades comunes"""
        vulnerabilities = []
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files[:30]:  # Muestra
            if '__pycache__' in str(py_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                lines = content.split('\n')
                vulns = []
                
                for i, line in enumerate(lines, 1):
                    line_stripped = line.strip()
                    
                    # Buscar vulnerabilidades comunes
                    if "eval(" in line_stripped:
                        vulns.append(f"Línea {i}: Uso de eval()")
                    
                    if "exec(" in line_stripped:
                        vulns.append(f"Línea {i}: Uso de exec()")
                    
                    if "pickle.loads" in line_stripped:
                        vulns.append(f"Línea {i}: Deserialización pickle sin validación")
                    
                    if "subprocess.call" in line_stripped and "shell=True" in line_stripped:
                        vulns.append(f"Línea {i}: subprocess con shell=True")
                    
                    if "password" in line_stripped.lower() and "=" in line_stripped:
                        if not any(secure in line_stripped for secure in ["getenv", "input", "config"]):
                            vulns.append(f"Línea {i}: Contraseña en código")
                
                if vulns:
                    vulnerabilities.append({
                        "file": str(py_file.relative_to(self.root_dir)),
                        "vulnerabilities": vulns,
                        "encoding": encoding
                    })
                
            except Exception as e:
                pass
        
        return vulnerabilities
    
    def _analyze_security_practices(self) -> Dict:
        """Analiza prácticas de seguridad"""
        practices = {
            "has_input_validation": False,
            "has_error_handling": False,
            "has_authentication": False,
            "has_encryption": False
        }
        
        all_py_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in all_py_files[:20]:  # Muestra
            if '__pycache__' in str(py_file):
                continue
            
            try:
                # Leer archivo de manera segura
                content, encoding = self._safe_read_file(py_file)
                
                if content is None:
                    continue
                
                # Buscar prácticas de seguridad
                if any(keyword in content for keyword in ["try:", "except", "try-except"]):
                    practices["has_error_handling"] = True
                
                if any(keyword in content for keyword in ["if.*input", "validate", "sanitize"]):
                    practices["has_input_validation"] = True
                
                if any(keyword in content for keyword in ["authenticate", "login", "auth"]):
                    practices["has_authentication"] = True
                
                if any(keyword in content for keyword in ["encrypt", "decrypt", "hash"]):
                    practices["has_encryption"] = True
                
            except Exception as e:
                pass
        
        return practices
    
    def _calculate_security_score(self, vulnerabilities, security_practices):
        """Calcula puntuación de seguridad"""
        score = 50  # Base
        
        # Penalizar por vulnerabilidades
        total_vulns = sum(len(v["vulnerabilities"]) for v in vulnerabilities)
        score -= total_vulns * 10
        
        # Recompensar por prácticas de seguridad
        practices_count = sum(1 for v in security_practices.values() if v)
        score += practices_count * 10
        
        return max(0, min(score, 100))
    
    def _generate_comprehensive_report(self) -> Dict:
        """Genera reporte comprehensivo"""
        report = {
            "metadata": {
                "project": str(self.root_dir.name),
                "analysis_date": datetime.now().isoformat(),
                "analyzer_version": "2.0"
            },
            "summary": {},
            "detailed_results": {},
            "overall_assessment": {},
            "action_plan": {}
        }
        
        # Calcular resumen
        scores = [r.score for r in self.results.values() if r.score is not None]
        overall_score = statistics.mean(scores) if scores else 0
        
        report["summary"] = {
            "overall_score": overall_score,
            "category_scores": {
                cat.value: result.score 
                for cat, result in self.results.items()
            },
            "total_issues": sum(len(r.issues) for r in self.results.values()),
            "total_recommendations": sum(len(r.recommendations) for r in self.results.values())
        }
        
        # Resultados detallados
        for cat, result in self.results.items():
            report["detailed_results"][cat.value] = {
                "score": result.score,
                "issues": result.issues[:10],  # Limitar para no hacer muy largo
                "recommendations": result.recommendations[:10],
                "details_summary": self._summarize_details(result.details)
            }
        
        # Evaluación general
        report["overall_assessment"] = self._generate_overall_assessment(overall_score)
        
        # Plan de acción
        report["action_plan"] = self._generate_action_plan()
        
        return report
    
    def _summarize_details(self, details: Dict) -> Dict:
        """Resume detalles para el reporte"""
        summary = {}
        
        for key, value in details.items():
            if isinstance(value, (int, float, str, bool)):
                summary[key] = value
            elif isinstance(value, list):
                summary[f"{key}_count"] = len(value)
                if value and isinstance(value[0], dict):
                    # Tomar algunas claves del primer elemento
                    first_item = value[0]
                    sample_keys = list(first_item.keys())[:3]
                    summary[f"{key}_sample"] = [
                        {k: first_item.get(k, "") for k in sample_keys}
                    ]
            elif isinstance(value, dict):
                summary[f"{key}_keys"] = list(value.keys())[:5]
        
        return summary
    
    def _generate_overall_assessment(self, overall_score: float) -> Dict:
        """Genera evaluación general"""
        if overall_score >= 80:
            status = "EXCELENTE"
            description = "El proyecto está bien estructurado y es funcional."
            usability = "Listo para uso productivo"
        elif overall_score >= 60:
            status = "BUENO"
            description = "El proyecto tiene buena base pero necesita mejoras."
            usability = "Utilizable con algunos ajustes"
        elif overall_score >= 40:
            status = "REGULAR"
            description = "El proyecto necesita trabajo significativo."
            usability = "Requiere ajustes importantes"
        else:
            status = "CRÍTICO"
            description = "El proyecto tiene problemas fundamentales."
            usability = "No recomendado para uso sin reestructuración"
        
        # Fortalezas y debilidades
        strengths = []
        weaknesses = []
        
        for cat, result in self.results.items():
            if result.score >= 70:
                strengths.append(cat.value)
            elif result.score < 40:
                weaknesses.append(cat.value)
        
        return {
            "status": status,
            "score": overall_score,
            "description": description,
            "usability": usability,
            "strengths": strengths,
            "weaknesses": weaknesses
        }
    
    def _generate_action_plan(self) -> Dict:
        """Genera plan de acción priorizado"""
        action_plan = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        # Recopilar todas las recomendaciones
        all_recommendations = []
        for cat, result in self.results.items():
            for rec in result.recommendations:
                all_recommendations.append({
                    "category": cat.value,
                    "recommendation": rec,
                    "priority": self._determine_priority(cat, result.score)
                })
        
        # Organizar por prioridad
        for rec in all_recommendations:
            action_plan[rec["priority"]].append(rec)
        
        # Limitar a 5 por prioridad
        for priority in action_plan:
            action_plan[priority] = action_plan[priority][:5]
        
        return action_plan
    
    def _determine_priority(self, category: AnalysisCategory, score: float) -> str:
        """Determina prioridad basada en categoría y puntuación"""
        if score < 30:
            return "critical"
        elif score < 50:
            return "high"
        elif score < 70:
            return "medium"
        else:
            return "low"
    
    def save_report(self, filename: str = None):
        """Guarda el reporte en un archivo"""
        if filename is None:
            filename = f"exhaustive_analysis_{self.timestamp}.json"
        
        report = self._generate_comprehensive_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Reporte JSON guardado en: {filename}")
        
        # También guardar versión legible
        txt_filename = f"exhaustive_analysis_{self.timestamp}.txt"
        self._save_readable_report(txt_filename, report)
        
        return filename, txt_filename
    
    def _save_readable_report(self, filename: str, report: Dict):
        """Guarda versión legible del reporte"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self._format_readable_report(report))
            print(f"📋 Reporte legible guardado en: {filename}")
        except Exception as e:
            print(f"⚠️  Error al guardar reporte legible: {e}")
    
    def _format_readable_report(self, report: Dict) -> str:
        """Formatea reporte para lectura humana"""
        lines = []
        
        lines.append("=" * 100)
        lines.append("ANÁLISIS EXHAUSTIVO DEL PROYECTO")
        lines.append("=" * 100)
        lines.append(f"Proyecto: {report['metadata']['project']}")
        lines.append(f"Fecha: {report['metadata']['analysis_date']}")
        lines.append("=" * 100)
        
        # Resumen
        lines.append("\n📊 RESUMEN EJECUTIVO")
        lines.append("-" * 80)
        
        summary = report["summary"]
        assessment = report["overall_assessment"]
        
        lines.append(f"Puntuación General: {assessment['score']:.1f}/100")
        lines.append(f"Estado: {assessment['status']}")
        lines.append(f"Usabilidad: {assessment['usability']}")
        lines.append(f"Descripción: {assessment['description']}")
        lines.append(f"Total Issues: {summary['total_issues']}")
        lines.append(f"Total Recomendaciones: {summary['total_recommendations']}")
        
        # Puntuaciones por categoría
        lines.append("\n📈 PUNTUACIONES POR CATEGORÍA")
        lines.append("-" * 80)
        
        for cat, score in summary["category_scores"].items():
            bar_length = int(score / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"{cat.upper():20} {score:5.1f}/100 {bar}")
        
        # Fortalezas y Debilidades
        lines.append("\n✅ FORTALEZAS")
        lines.append("-" * 80)
        for strength in assessment.get("strengths", []):
            lines.append(f"• {strength}")
        
        lines.append("\n⚠️  DEBILIDADES")
        lines.append("-" * 80)
        for weakness in assessment.get("weaknesses", []):
            lines.append(f"• {weakness}")
        
        # Resultados detallados
        lines.append("\n🔍 RESULTADOS DETALLADOS")
        lines.append("=" * 100)
        
        for cat_name, cat_result in report["detailed_results"].items():
            lines.append(f"\n{cat_name.upper()}: {cat_result['score']:.1f}/100")
            lines.append("-" * 60)
            
            if cat_result["issues"]:
                lines.append("ISSUES:")
                for issue in cat_result["issues"]:
                    lines.append(f"  • {issue}")
            
            if cat_result["recommendations"]:
                lines.append("\nRECOMENDACIONES:")
                for rec in cat_result["recommendations"]:
                    lines.append(f"  • {rec}")
        
        # Plan de acción
        lines.append("\n📋 PLAN DE ACCIÓN PRIORIZADO")
        lines.append("=" * 100)
        
        action_plan = report["action_plan"]
        
        lines.append("\n🚨 CRÍTICO (Hacer inmediatamente):")
        lines.append("-" * 80)
        for action in action_plan["critical"]:
            lines.append(f"• [{action['category']}] {action['recommendation']}")
        
        lines.append("\n🔴 ALTA (Hacer pronto):")
        lines.append("-" * 80)
        for action in action_plan["high"]:
            lines.append(f"• [{action['category']}] {action['recommendation']}")
        
        lines.append("\n🟡 MEDIA (Planificar):")
        lines.append("-" * 80)
        for action in action_plan["medium"]:
            lines.append(f"• [{action['category']}] {action['recommendation']}")
        
        lines.append("\n🟢 BAJA (Mejora continua):")
        lines.append("-" * 80)
        for action in action_plan["low"]:
            lines.append(f"• [{action['category']}] {action['recommendation']}")
        
        # Conclusión
        lines.append("\n" + "=" * 100)
        lines.append("CONCLUSIÓN")
        lines.append("=" * 100)
        
        if assessment["score"] >= 70:
            lines.append("✅ EL PROYECTO ES VIABLE Y FUNCIONAL")
            lines.append("Puede utilizarse o desplegarse con confianza.")
            lines.append("Se recomienda abordar las recomendaciones para mejorar.")
        elif assessment["score"] >= 50:
            lines.append("⚠️ EL PROYECTO REQUIERE MEJORAS")
            lines.append("Es funcional pero necesita trabajo en áreas críticas.")
            lines.append("Abordar las acciones críticas y altas prioridad.")
        else:
            lines.append("❌ EL PROYECTO NECESITA REESTRUCTURACIÓN")
            lines.append("No se recomienda usar en producción sin cambios significativos.")
            lines.append("Comenzar por las acciones críticas y reconsiderar arquitectura.")
        
        lines.append("\n" + "=" * 100)
        lines.append("FIN DEL ANÁLISIS")
        lines.append("=" * 100)
        
        return "\n".join(lines)

def main():
    """Función principal"""
    print("🚀 ANALIZADOR EXHAUSTIVO DE PROYECTOS")
    print("Basado en arquitectura Project Brain")
    print("=" * 100)
    
    # Pedir directorio del proyecto
    project_path = input("Ingrese la ruta del proyecto (dejar vacío para usar directorio actual): ").strip()
    
    if not project_path:
        project_path = "."
    
    # Crear analizador
    analyzer = ExhaustiveProjectAnalyzer(project_path)
    
    # Ejecutar análisis
    try:
        report = analyzer.analyze_complete_project()
        
        # Guardar reportes
        json_file, txt_file = analyzer.save_report()
        
        print(f"\n✅ ANÁLISIS COMPLETADO")
        print(f"📄 Reporte JSON: {json_file}")
        print(f"📋 Reporte legible: {txt_file}")
        
        # Mostrar resumen
        overall_score = report["summary"]["overall_score"]
        assessment = report["overall_assessment"]
        
        print(f"\n📊 Puntuación General: {overall_score:.1f}/100")
        print(f"🏷️  Estado: {assessment['status']}")
        print(f"🎯 Usabilidad: {assessment['usability']}")
        
        # Mostrar acciones críticas
        critical_actions = report["action_plan"]["critical"]
        if critical_actions:
            print(f"\n🚨 ACCIONES CRÍTICAS ({len(critical_actions)}):")
            for action in critical_actions:
                print(f"  • {action['recommendation']}")
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()