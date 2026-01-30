"""
Módulo DependencyMapper - Mapeo de dependencias entre archivos y módulos
"""

import re
import ast
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import networkx as nx
from collections import defaultdict

@dataclass
class Dependency:
    """Representación de una dependencia"""
    source: str
    target: str
    type: str  # 'import', 'include', 'call', 'inherit', 'implement', 'use'
    line: int = 0
    context: Optional[str] = None
    weight: float = 1.0

@dataclass
class ModuleGraph:
    """Grafo de módulos y sus dependencias"""
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Tuple[str, str, Dict[str, Any]]] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)
    clusters: Dict[str, List[str]] = field(default_factory=dict)

class DependencyMapper:
    """Mapeador de dependencias entre archivos y módulos"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar mapeador de dependencias"""
        self.config = config or {}
        self.graph = nx.DiGraph()
        self.dependencies: List[Dependency] = []
        
        # Configuración de detección por lenguaje
        self.language_patterns = {
            'python': self._analyze_python_dependencies,
            'java': self._analyze_java_dependencies,
            'javascript': self._analyze_javascript_dependencies,
            'typescript': self._analyze_typescript_dependencies,
            'cpp': self._analyze_cpp_dependencies,
            'c': self._analyze_c_dependencies,
        }
    
    def analyze_file(self, file_path: str, content: str, language: str) -> List[Dependency]:
        """
        Analizar dependencias en un archivo
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            language: Lenguaje de programación
            
        Returns:
            List[Dependency]: Lista de dependencias encontradas
        """
        if language not in self.language_patterns:
            return self._analyze_generic_dependencies(file_path, content, language)
        
        analyzer = self.language_patterns[language]
        return analyzer(file_path, content)
    
    def _analyze_python_dependencies(self, file_path: str, content: str) -> List[Dependency]:
        """Analizar dependencias Python"""
        dependencies = []
        
        try:
            tree = ast.parse(content)
            
            class DependencyVisitor(ast.NodeVisitor):
                def __init__(self, file_path):
                    self.file_path = file_path
                    self.dependencies = []
                    self.current_context = None
                
                def visit_Import(self, node):
                    for alias in node.names:
                        dep = Dependency(
                            source=self.file_path,
                            target=alias.name,
                            type='import',
                            line=node.lineno,
                            context=self.current_context
                        )
                        self.dependencies.append(dep)
                
                def visit_ImportFrom(self, node):
                    module = node.module or ''
                    for alias in node.names:
                        target = f"{module}.{alias.name}" if module else alias.name
                        dep = Dependency(
                            source=self.file_path,
                            target=target,
                            type='import_from',
                            line=node.lineno,
                            context=self.current_context
                        )
                        self.dependencies.append(dep)
                
                def visit_ClassDef(self, node):
                    # Guardar contexto actual
                    previous_context = self.current_context
                    self.current_context = f"class:{node.name}"
                    
                    # Analizar herencia
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            dep = Dependency(
                                source=self.file_path,
                                target=base.id,
                                type='inherit',
                                line=node.lineno,
                                context=self.current_context
                            )
                            self.dependencies.append(dep)
                    
                    self.generic_visit(node)
                    self.current_context = previous_context
                
                def visit_FunctionDef(self, node):
                    # Guardar contexto actual
                    previous_context = self.current_context
                    if self.current_context:
                        self.current_context = f"{self.current_context}.{node.name}"
                    else:
                        self.current_context = f"function:{node.name}"
                    
                    self.generic_visit(node)
                    self.current_context = previous_context
                
                def visit_Call(self, node):
                    # Analizar llamadas a funciones
                    if isinstance(node.func, ast.Name):
                        dep = Dependency(
                            source=self.file_path,
                            target=node.func.id,
                            type='call',
                            line=node.lineno,
                            context=self.current_context
                        )
                        self.dependencies.append(dep)
                    elif isinstance(node.func, ast.Attribute):
                        # Llamadas a métodos de objetos/modulos
                        module_name = self._get_attribute_module(node.func)
                        if module_name:
                            dep = Dependency(
                                source=self.file_path,
                                target=module_name,
                                type='use',
                                line=node.lineno,
                                context=self.current_context
                            )
                            self.dependencies.append(dep)
                    
                    self.generic_visit(node)
                
                def visit_Attribute(self, node):
                    # Uso de atributos de módulos
                    module_name = self._get_attribute_module(node)
                    if module_name:
                        dep = Dependency(
                            source=self.file_path,
                            target=module_name,
                            type='use',
                            line=getattr(node, 'lineno', 0),
                            context=self.current_context
                        )
                        self.dependencies.append(dep)
                    
                    self.generic_visit(node)
                
                def _get_attribute_module(self, node):
                    """Obtener nombre del módulo de un atributo"""
                    parts = []
                    current = node
                    
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                        return '.'.join(reversed(parts))
                    
                    return None
            
            visitor = DependencyVisitor(file_path)
            visitor.visit(tree)
            dependencies = visitor.dependencies
            
        except SyntaxError:
            # Fallback a análisis basado en regex
            dependencies = self._analyze_python_dependencies_regex(file_path, content)
        
        return dependencies
    
    def _analyze_python_dependencies_regex(self, file_path: str, content: str) -> List[Dependency]:
        """Analizar dependencias Python usando regex"""
        dependencies = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Importaciones
            if line_stripped.startswith('import '):
                # Extraer nombres después de import
                import_part = line_stripped[7:].strip()
                imports = [imp.strip() for imp in import_part.split(',')]
                
                for imp in imports:
                    # Separar alias
                    if ' as ' in imp:
                        module = imp.split(' as ')[0].strip()
                    else:
                        module = imp
                    
                    dep = Dependency(
                        source=file_path,
                        target=module,
                        type='import',
                        line=i
                    )
                    dependencies.append(dep)
            
            # Importaciones from
            elif line_stripped.startswith('from '):
                # Extraer módulo y nombres
                match = re.match(r'from\s+([^\s]+)\s+import\s+(.+)', line_stripped)
                if match:
                    module = match.group(1)
                    imports_part = match.group(2)
                    
                    # Separar múltiples imports
                    imports = [imp.strip() for imp in imports_part.split(',')]
                    
                    for imp in imports:
                        # Separar alias
                        if ' as ' in imp:
                            name = imp.split(' as ')[0].strip()
                        else:
                            name = imp
                        
                        target = f"{module}.{name}"
                        dep = Dependency(
                            source=file_path,
                            target=target,
                            type='import_from',
                            line=i
                        )
                        dependencies.append(dep)
        
        return dependencies
    
    def _analyze_java_dependencies(self, file_path: str, content: str) -> List[Dependency]:
        """Analizar dependencias Java"""
        dependencies = []
        lines = content.splitlines()
        
        current_class = None
        current_package = None
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Paquete
            if line_stripped.startswith('package '):
                current_package = line_stripped[8:].rstrip(';')
            
            # Imports
            elif line_stripped.startswith('import '):
                import_stmt = line_stripped[7:].rstrip(';')
                
                # Separar import estático
                is_static = import_stmt.startswith('static ')
                if is_static:
                    import_stmt = import_stmt[7:]
                
                # Separar wildcard
                has_wildcard = import_stmt.endswith('.*')
                if has_wildcard:
                    import_stmt = import_stmt[:-2]
                
                dep = Dependency(
                    source=file_path,
                    target=import_stmt,
                    type='import',
                    line=i,
                    context='static' if is_static else None
                )
                dependencies.append(dep)
            
            # Clases
            elif re.match(r'^(public|private|protected|abstract|final|static)?\s*class\s+', line_stripped):
                match = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line_stripped)
                if match:
                    current_class = match.group(1)
                    
                    # Herencia (extends)
                    if 'extends' in line_stripped:
                        extends_match = re.search(r'extends\s+([a-zA-Z_][a-zA-Z0-9_.]*)', line_stripped)
                        if extends_match:
                            dep = Dependency(
                                source=file_path,
                                target=extends_match.group(1),
                                type='inherit',
                                line=i,
                                context=f"class:{current_class}"
                            )
                            dependencies.append(dep)
                    
                    # Implementación (implements)
                    if 'implements' in line_stripped:
                        implements_match = re.search(r'implements\s+([a-zA-Z_][a-zA-Z0-9_.,\s]*)', line_stripped)
                        if implements_match:
                            interfaces = [i.strip() for i in implements_match.group(1).split(',')]
                            for interface in interfaces:
                                dep = Dependency(
                                    source=file_path,
                                    target=interface,
                                    type='implement',
                                    line=i,
                                    context=f"class:{current_class}"
                                )
                                dependencies.append(dep)
            
            # Uso de clases (new, cast, etc.)
            elif current_class and re.search(r'\bnew\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(', line_stripped):
                match = re.search(r'\bnew\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(', line_stripped)
                if match:
                    dep = Dependency(
                        source=file_path,
                        target=match.group(1),
                        type='instantiate',
                        line=i,
                        context=f"class:{current_class}"
                    )
                    dependencies.append(dep)
        
        return dependencies
    
    def _analyze_javascript_dependencies(self, file_path: str, content: str) -> List[Dependency]:
        """Analizar dependencias JavaScript"""
        dependencies = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # ES6 imports
            if line_stripped.startswith('import '):
                # Extraer la parte después de import
                import_part = line_stripped[7:].strip()
                
                # Diferentes formatos de import
                if ' from ' in import_part:
                    # import X from 'module'
                    # import {X, Y} from 'module'
                    parts = import_part.split(' from ', 1)
                    module = parts[1].strip().strip('"\';')
                    
                    dep = Dependency(
                        source=file_path,
                        target=module,
                        type='import',
                        line=i
                    )
                    dependencies.append(dep)
                else:
                    # import 'module'
                    module = import_part.strip().strip('"\';')
                    if module:
                        dep = Dependency(
                            source=file_path,
                            target=module,
                            type='import',
                            line=i
                        )
                        dependencies.append(dep)
            
            # CommonJS require
            elif 'require(' in line_stripped:
                match = re.search(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', line_stripped)
                if match:
                    dep = Dependency(
                        source=file_path,
                        target=match.group(1),
                        type='require',
                        line=i
                    )
                    dependencies.append(dep)
            
            # Exports (pueden indicar dependencias)
            elif line_stripped.startswith('export '):
                if ' from ' in line_stripped:
                    parts = line_stripped.split(' from ', 1)
                    module = parts[1].strip().strip('"\';')
                    
                    dep = Dependency(
                        source=file_path,
                        target=module,
                        type='export_from',
                        line=i
                    )
                    dependencies.append(dep)
        
        return dependencies
    
    def _analyze_typescript_dependencies(self, file_path: str, content: str) -> List[Dependency]:
        """Analizar dependencias TypeScript"""
        # Incluir todas las dependencias JavaScript
        dependencies = self._analyze_javascript_dependencies(file_path, content)
        
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # TypeScript specific: import type
            if line_stripped.startswith('import type '):
                import_part = line_stripped[11:].strip()
                if ' from ' in import_part:
                    parts = import_part.split(' from ', 1)
                    module = parts[1].strip().strip('"\';')
                    
                    dep = Dependency(
                        source=file_path,
                        target=module,
                        type='import_type',
                        line=i
                    )
                    dependencies.append(dep)
            
            # Herencia en interfaces/clases TypeScript
            elif re.match(r'^(export\s+)?(class|interface)\s+', line_stripped):
                if 'extends' in line_stripped:
                    match = re.search(r'extends\s+([a-zA-Z_$][a-zA-Z0-9_$.]*)', line_stripped)
                    if match:
                        dep = Dependency(
                            source=file_path,
                            target=match.group(1),
                            type='inherit',
                            line=i
                        )
                        dependencies.append(dep)
                
                if 'implements' in line_stripped:
                    match = re.search(r'implements\s+([a-zA-Z_$][a-zA-Z0-9_$.,\s]*)', line_stripped)
                    if match:
                        interfaces = [i.strip() for i in match.group(1).split(',')]
                        for interface in interfaces:
                            dep = Dependency(
                                source=file_path,
                                target=interface,
                                type='implement',
                                line=i
                            )
                            dependencies.append(dep)
        
        return dependencies
    
    def _analyze_cpp_dependencies(self, file_path: str, content: str) -> List[Dependency]:
        """Analizar dependencias C++"""
        dependencies = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped or line_stripped.startswith('//'):
                continue
            
            # Includes
            if line_stripped.startswith('#include'):
                # Extraer el archivo incluido
                if '<' in line_stripped and '>' in line_stripped:
                    # Include del sistema: #include <archivo>
                    match = re.search(r'<([^>]+)>', line_stripped)
                    target_type = 'system_include'
                elif '"' in line_stripped:
                    # Include local: #include "archivo"
                    match = re.search(r'"([^"]+)"', line_stripped)
                    target_type = 'local_include'
                else:
                    continue
                
                if match:
                    dep = Dependency(
                        source=file_path,
                        target=match.group(1),
                        type=target_type,
                        line=i
                    )
                    dependencies.append(dep)
            
            # Herencia en clases
            elif re.match(r'^class\s+', line_stripped):
                if ':' in line_stripped:
                    # Extraer parte después de :
                    inheritance_part = line_stripped.split(':', 1)[1].strip()
                    # Separar por coma para múltiple herencia
                    bases = [b.strip() for b in inheritance_part.split(',')]
                    
                    for base in bases:
                        # Limpiar modificadores (public, private, protected)
                        base_clean = re.sub(r'^(public|private|protected)\s+', '', base)
                        if base_clean:
                            dep = Dependency(
                                source=file_path,
                                target=base_clean,
                                type='inherit',
                                line=i
                            )
                            dependencies.append(dep)
        
        return dependencies
    
    def _analyze_c_dependencies(self, file_path: str, content: str) -> List[Dependency]:
        """Analizar dependencias C"""
        # Similar a C++ pero sin herencia
        dependencies = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped or line_stripped.startswith('//'):
                continue
            
            # Includes
            if line_stripped.startswith('#include'):
                if '<' in line_stripped and '>' in line_stripped:
                    match = re.search(r'<([^>]+)>', line_stripped)
                    target_type = 'system_include'
                elif '"' in line_stripped:
                    match = re.search(r'"([^"]+)"', line_stripped)
                    target_type = 'local_include'
                else:
                    continue
                
                if match:
                    dep = Dependency(
                        source=file_path,
                        target=match.group(1),
                        type=target_type,
                        line=i
                    )
                    dependencies.append(dep)
        
        return dependencies
    
    def _analyze_generic_dependencies(self, file_path: str, content: str, language: str) -> List[Dependency]:
        """Análisis genérico de dependencias"""
        dependencies = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Patrones comunes
            # Includes/Imports
            include_patterns = [
                r'^#include\s+[<"]([^>"]+)[>"]',
                r'^import\s+["\']?([^"\'\s;]+)["\']?',
                r'^require\s*\(\s*["\']([^"\']+)["\']\s*\)',
            ]
            
            for pattern in include_patterns:
                match = re.search(pattern, line_stripped)
                if match:
                    dep = Dependency(
                        source=file_path,
                        target=match.group(1),
                        type='include',
                        line=i
                    )
                    dependencies.append(dep)
                    break
        
        return dependencies
    
    def build_dependency_graph(self, files_dependencies: Dict[str, List[Dependency]]) -> ModuleGraph:
        """
        Construir grafo de dependencias a partir de análisis de archivos
        
        Args:
            files_dependencies: Diccionario de archivo -> lista de dependencias
            
        Returns:
            ModuleGraph: Grafo de dependencias construido
        """
        graph = ModuleGraph()
        
        # Agregar nodos (archivos)
        for file_path, deps in files_dependencies.items():
            node_id = self._normalize_path(file_path)
            
            if node_id not in graph.nodes:
                graph.nodes[node_id] = {
                    'path': file_path,
                    'dependency_count': len(deps),
                    'types': set()
                }
            
            # Agregar aristas (dependencias)
            for dep in deps:
                target_id = self._normalize_dependency(dep.target)
                
                if target_id and target_id != node_id:
                    edge_data = {
                        'type': dep.type,
                        'line': dep.line,
                        'context': dep.context,
                        'weight': dep.weight
                    }
                    
                    graph.edges.append((node_id, target_id, edge_data))
                    
                    # Registrar tipos de dependencia
                    graph.nodes[node_id]['types'].add(dep.type)
        
        # Detectar ciclos
        graph.cycles = self._find_cycles(graph)
        
        # Encontrar clusters (componentes fuertemente conexos)
        graph.clusters = self._find_clusters(graph)
        
        return graph
    
    def _normalize_path(self, path: str) -> str:
        """Normalizar ruta para usar como ID en el grafo"""
        # Convertir a ruta absoluta y normalizar
        try:
            abs_path = Path(path).resolve()
            return str(abs_path).replace('\\', '/')
        except:
            return path.replace('\\', '/')
    
    def _normalize_dependency(self, dependency: str) -> Optional[str]:
        """
        Normalizar nombre de dependencia
        
        Args:
            dependency: Nombre de dependencia crudo
            
        Returns:
            Optional[str]: Nombre normalizado o None si no es una dependencia de archivo
        """
        # Para dependencias de Python
        if '.' in dependency and not dependency.startswith('.'):
            # Podría ser un módulo Python
            # Convertir a ruta de archivo potencial
            parts = dependency.split('.')
            
            # Si termina con algo que parece un archivo
            if len(parts) > 1:
                # Intentar reconstruir como ruta
                return '/'.join(parts) + '.py'
        
        # Para includes de C/C++
        if any(dependency.endswith(ext) for ext in ['.h', '.hpp', '.c', '.cpp']):
            return dependency
        
        # Para imports de JavaScript/TypeScript
        if dependency.startswith('./') or dependency.startswith('../'):
            return dependency
        
        # Para otros casos, devolver el nombre original
        return dependency
    
    def _find_cycles(self, graph: ModuleGraph) -> List[List[str]]:
        """Encontrar ciclos en el grafo de dependencias"""
        cycles = []
        
        # Crear grafo NetworkX para análisis
        nx_graph = nx.DiGraph()
        
        # Agregar nodos
        for node_id in graph.nodes:
            nx_graph.add_node(node_id)
        
        # Agregar aristas
        for source, target, data in graph.edges:
            nx_graph.add_edge(source, target, **data)
        
        # Encontrar ciclos simples
        try:
            simple_cycles = list(nx.simple_cycles(nx_graph))
            cycles = simple_cycles
        except:
            # NetworkX puede lanzar excepción para grafos muy grandes
            pass
        
        return cycles
    
    def _find_clusters(self, graph: ModuleGraph) -> Dict[str, List[str]]:
        """Encontrar clusters (componentes fuertemente conexos)"""
        clusters = {}
        
        # Crear grafo NetworkX
        nx_graph = nx.DiGraph()
        
        # Agregar nodos y aristas
        for node_id in graph.nodes:
            nx_graph.add_node(node_id)
        
        for source, target, _ in graph.edges:
            nx_graph.add_edge(source, target)
        
        # Encontrar componentes fuertemente conexos
        scc = list(nx.strongly_connected_components(nx_graph))
        
        for i, component in enumerate(scc):
            if len(component) > 1:  # Solo clusters con múltiples nodos
                cluster_id = f"cluster_{i}"
                clusters[cluster_id] = list(component)
        
        return clusters
    
    def calculate_metrics(self, graph: ModuleGraph) -> Dict[str, Any]:
        """
        Calcular métricas del grafo de dependencias
        
        Args:
            graph: Grafo de dependencias
            
        Returns:
            Dict[str, Any]: Métricas calculadas
        """
        metrics = {
            'total_nodes': len(graph.nodes),
            'total_edges': len(graph.edges),
            'cycles_count': len(graph.cycles),
            'clusters_count': len(graph.clusters),
            'node_metrics': {},
            'graph_metrics': {}
        }
        
        # Crear grafo NetworkX para análisis
        nx_graph = nx.DiGraph()
        
        for node_id in graph.nodes:
            nx_graph.add_node(node_id)
        
        for source, target, data in graph.edges:
            nx_graph.add_edge(source, target, weight=data.get('weight', 1.0))
        
        # Métricas por nodo
        for node_id in nx_graph.nodes():
            in_degree = nx_graph.in_degree(node_id)
            out_degree = nx_graph.out_degree(node_id)
            
            metrics['node_metrics'][node_id] = {
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_dependencies': in_degree + out_degree
            }
        
        # Métricas del grafo completo
        if nx_graph.number_of_nodes() > 0:
            # Densidad del grafo
            metrics['graph_metrics']['density'] = nx.density(nx_graph)
            
            # Coeficiente de clustering promedio
            try:
                metrics['graph_metrics']['average_clustering'] = nx.average_clustering(nx_graph.to_undirected())
            except:
                metrics['graph_metrics']['average_clustering'] = 0.0
            
            # Diámetro (solo si el grafo es conexo)
            try:
                if nx.is_weakly_connected(nx_graph):
                    metrics['graph_metrics']['diameter'] = nx.diameter(nx_graph.to_undirected())
                else:
                    metrics['graph_metrics']['diameter'] = None
            except:
                metrics['graph_metrics']['diameter'] = None
        
        return metrics
    
    def find_circular_dependencies(self, graph: ModuleGraph) -> List[Dict[str, Any]]:
        """
        Encontrar y analizar dependencias circulares
        
        Args:
            graph: Grafo de dependencias
            
        Returns:
            List[Dict[str, Any]]: Dependencias circulares detectadas
        """
        circular_deps = []
        
        for cycle in graph.cycles:
            if len(cycle) > 1:  # Ciclos reales (no auto-referencias)
                cycle_info = {
                    'nodes': cycle,
                    'length': len(cycle),
                    'edges': []
                }
                
                # Encontrar aristas que forman el ciclo
                for i in range(len(cycle)):
                    source = cycle[i]
                    target = cycle[(i + 1) % len(cycle)]
                    
                    # Buscar la arista correspondiente
                    for edge in graph.edges:
                        if edge[0] == source and edge[1] == target:
                            cycle_info['edges'].append({
                                'source': source,
                                'target': target,
                                'type': edge[2].get('type'),
                                'line': edge[2].get('line')
                            })
                            break
                
                circular_deps.append(cycle_info)
        
        return circular_deps
    
    def suggest_refactoring(self, graph: ModuleGraph, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Sugerir refactorizaciones basadas en análisis de dependencias
        
        Args:
            graph: Grafo de dependencias
            metrics: Métricas calculadas
            
        Returns:
            List[Dict[str, Any]]: Sugerencias de refactorización
        """
        suggestions = []
        
        # Sugerir eliminar dependencias circulares
        for cycle_info in self.find_circular_dependencies(graph):
            suggestion = {
                'type': 'break_cycle',
                'priority': 'high',
                'description': f"Dependencia circular detectada entre {len(cycle_info['nodes'])} módulos",
                'details': {
                    'cycle': cycle_info['nodes'],
                    'suggestion': 'Considerar introducir interfaces o usar inyección de dependencias'
                }
            }
            suggestions.append(suggestion)
        
        # Sugerir módulos con alta acoplamiento
        for node_id, node_metrics in metrics.get('node_metrics', {}).items():
            total_deps = node_metrics.get('total_dependencies', 0)
            
            if total_deps > 10:  # Umbral para alta acoplamiento
                suggestion = {
                    'type': 'reduce_coupling',
                    'priority': 'medium',
                    'description': f"Módulo con alto acoplamiento: {node_id}",
                    'details': {
                        'in_degree': node_metrics.get('in_degree'),
                        'out_degree': node_metrics.get('out_degree'),
                        'suggestion': 'Considerar dividir el módulo o extraer funcionalidad común'
                    }
                }
                suggestions.append(suggestion)
        
        # Sugerir módulos con muchas dependencias salientes
        for node_id, node_metrics in metrics.get('node_metrics', {}).items():
            out_degree = node_metrics.get('out_degree', 0)
            
            if out_degree > 15:  # Umbral para muchas dependencias salientes
                suggestion = {
                    'type': 'reduce_outgoing_deps',
                    'priority': 'medium',
                    'description': f"Módulo con muchas dependencias salientes: {node_id}",
                    'details': {
                        'out_degree': out_degree,
                        'suggestion': 'El módulo podría estar violando el principio de responsabilidad única'
                    }
                }
                suggestions.append(suggestion)
        
        return suggestions
    
    def export_graph(self, graph: ModuleGraph, format: str = 'json') -> str:
        """
        Exportar grafo de dependencias en diferentes formatos
        
        Args:
            graph: Grafo de dependencias
            format: Formato de exportación ('json', 'dot', 'csv')
            
        Returns:
            str: Grafo exportado en el formato especificado
        """
        if format == 'json':
            export_data = {
                'nodes': [
                    {
                        'id': node_id,
                        **data
                    }
                    for node_id, data in graph.nodes.items()
                ],
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        **data
                    }
                    for source, target, data in graph.edges
                ],
                'cycles': graph.cycles,
                'clusters': graph.clusters
            }
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        
        elif format == 'dot':
            # Formato DOT para GraphViz
            dot_lines = ['digraph Dependencies {']
            dot_lines.append('  rankdir=LR;')
            dot_lines.append('  node [shape=box, style=filled, fillcolor=lightblue];')
            
            # Agregar nodos
            for node_id in graph.nodes:
                dot_lines.append(f'  "{node_id}" [label="{Path(node_id).name}"];')
            
            # Agregar aristas
            for source, target, data in graph.edges:
                edge_type = data.get('type', 'dependency')
                color = self._get_edge_color(edge_type)
                dot_lines.append(f'  "{source}" -> "{target}" [color="{color}", label="{edge_type}"];')
            
            # Resaltar ciclos
            for i, cycle in enumerate(graph.cycles):
                if len(cycle) > 1:
                    for j in range(len(cycle)):
                        source = cycle[j]
                        target = cycle[(j + 1) % len(cycle)]
                        dot_lines.append(f'  "{source}" -> "{target}" [color="red", penwidth=2.0];')
            
            dot_lines.append('}')
            return '\n'.join(dot_lines)
        
        elif format == 'csv':
            # CSV de aristas
            csv_lines = ['source,target,type,line,context,weight']
            for source, target, data in graph.edges:
                csv_lines.append(
                    f'{source},{target},{data.get("type","")},'
                    f'{data.get("line",0)},{data.get("context","")},'
                    f'{data.get("weight",1.0)}'
                )
            return '\n'.join(csv_lines)
        
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def _get_edge_color(self, edge_type: str) -> str:
        """Obtener color para tipo de arista en gráficos DOT"""
        color_map = {
            'import': 'blue',
            'include': 'green',
            'call': 'purple',
            'inherit': 'orange',
            'implement': 'brown',
            'use': 'gray',
            'require': 'cyan'
        }
        return color_map.get(edge_type, 'black')