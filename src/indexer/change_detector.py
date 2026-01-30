"""
Módulo ChangeDetector - Detección y análisis de cambios en código
"""

import os
import hashlib
import difflib
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict

@dataclass
class FileChange:
    """Representación de un cambio en un archivo"""
    file_path: str
    change_type: str  # 'added', 'modified', 'deleted', 'renamed'
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    diff: Optional[str] = None
    lines_added: int = 0
    lines_removed: int = 0
    lines_changed: int = 0
    author: Optional[str] = None
    commit_hash: Optional[str] = None

@dataclass
class ChangeSet:
    """Conjunto de cambios relacionados"""
    changes: List[FileChange] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    total_files: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0
    description: Optional[str] = None

class ChangeDetector:
    """Detector de cambios en archivos de código"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar detector de cambios"""
        self.config = config or {}
        self.history_file = self.config.get('history_file', '.codeindex_history.json')
        self.max_history_size = self.config.get('max_history_size', 1000)
        self.snapshot_cache: Dict[str, Dict[str, Any]] = {}
        
    def take_snapshot(self, project_path: str) -> Dict[str, Any]:
        """
        Tomar snapshot del estado actual del proyecto
        
        Args:
            project_path: Ruta del proyecto
            
        Returns:
            Dict[str, Any]: Snapshot del proyecto
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'project_path': project_path,
            'files': {},
            'metadata': {}
        }
        
        project_root = Path(project_path).resolve()
        
        # Escanear todos los archivos de código
        code_extensions = {'.py', '.java', '.js', '.jsx', '.ts', '.tsx', 
                          '.cpp', '.c', '.h', '.hpp', '.cs', '.go', 
                          '.rb', '.php', '.html', '.css', '.json', '.xml'}
        
        for file_path in project_root.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in code_extensions:
                rel_path = str(file_path.relative_to(project_root))
                
                try:
                    file_hash = self._calculate_file_hash(file_path)
                    file_stats = file_path.stat()
                    
                    snapshot['files'][rel_path] = {
                        'hash': file_hash,
                        'size': file_stats.st_size,
                        'modified': file_stats.st_mtime,
                        'created': file_stats.st_ctime
                    }
                except (IOError, OSError):
                    continue
        
        # Calcular métricas del snapshot
        snapshot['metadata'] = {
            'total_files': len(snapshot['files']),
            'total_size': sum(f['size'] for f in snapshot['files'].values()),
            'file_types': self._count_file_types(snapshot['files'])
        }
        
        # Guardar en caché
        snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.snapshot_cache[snapshot_id] = snapshot
        
        return snapshot
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcular hash MD5 de un archivo"""
        hasher = hashlib.md5()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except (IOError, OSError):
            return ""
    
    def _count_file_types(self, files: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """Contar tipos de archivo en el snapshot"""
        type_count = defaultdict(int)
        
        for file_path in files.keys():
            ext = Path(file_path).suffix.lower()
            type_count[ext] += 1
        
        return dict(type_count)
    
    def compare_snapshots(self, old_snapshot: Dict[str, Any], 
                         new_snapshot: Dict[str, Any]) -> ChangeSet:
        """
        Comparar dos snapshots y detectar cambios
        
        Args:
            old_snapshot: Snapshot anterior
            new_snapshot: Snapshot nuevo
            
        Returns:
            ChangeSet: Conjunto de cambios detectados
        """
        change_set = ChangeSet()
        
        old_files = old_snapshot.get('files', {})
        new_files = new_snapshot.get('files', {})
        
        # Encontrar archivos añadidos
        added_files = set(new_files.keys()) - set(old_files.keys())
        for file_path in added_files:
            change = FileChange(
                file_path=file_path,
                change_type='added',
                new_hash=new_files[file_path]['hash']
            )
            change_set.changes.append(change)
            change_set.total_lines_added += self._estimate_lines_added(file_path, new_snapshot)
        
        # Encontrar archivos eliminados
        deleted_files = set(old_files.keys()) - set(new_files.keys())
        for file_path in deleted_files:
            change = FileChange(
                file_path=file_path,
                change_type='deleted',
                old_hash=old_files[file_path]['hash']
            )
            change_set.changes.append(change)
            change_set.total_lines_removed += self._estimate_lines_removed(file_path, old_snapshot)
        
        # Encontrar archivos modificados
        common_files = set(old_files.keys()) & set(new_files.keys())
        for file_path in common_files:
            old_hash = old_files[file_path]['hash']
            new_hash = new_files[file_path]['hash']
            
            if old_hash != new_hash:
                # Archivo modificado
                diff_result = self._calculate_diff(file_path, old_snapshot, new_snapshot)
                
                change = FileChange(
                    file_path=file_path,
                    change_type='modified',
                    old_hash=old_hash,
                    new_hash=new_hash,
                    diff=diff_result.get('diff'),
                    lines_added=diff_result.get('lines_added', 0),
                    lines_removed=diff_result.get('lines_removed', 0),
                    lines_changed=diff_result.get('lines_changed', 0)
                )
                change_set.changes.append(change)
                change_set.total_lines_added += diff_result.get('lines_added', 0)
                change_set.total_lines_removed += diff_result.get('lines_removed', 0)
        
        # Detectar renombres (hash igual pero nombre diferente)
        renamed_files = self._detect_renamed_files(old_files, new_files)
        for old_name, new_name in renamed_files:
            change = FileChange(
                file_path=new_name,
                change_type='renamed',
                old_hash=old_files[old_name]['hash'],
                new_hash=new_files[new_name]['hash']
            )
            change_set.changes.append(change)
        
        change_set.total_files = len(change_set.changes)
        
        return change_set
    
    def _estimate_lines_added(self, file_path: str, snapshot: Dict[str, Any]) -> int:
        """Estimar líneas añadidas para un archivo nuevo"""
        # Estimación simple basada en tamaño
        file_info = snapshot['files'].get(file_path, {})
        size = file_info.get('size', 0)
        
        # Estimación: 50 bytes por línea (promedio)
        return max(1, size // 50)
    
    def _estimate_lines_removed(self, file_path: str, snapshot: Dict[str, Any]) -> int:
        """Estimar líneas eliminadas para un archivo borrado"""
        # Similar a la estimación de añadidas
        file_info = snapshot['files'].get(file_path, {})
        size = file_info.get('size', 0)
        return max(1, size // 50)
    
    def _calculate_diff(self, file_path: str, 
                       old_snapshot: Dict[str, Any], 
                       new_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcular diff entre dos versiones de un archivo
        
        Args:
            file_path: Ruta del archivo
            old_snapshot: Snapshot anterior
            new_snapshot: Snapshot nuevo
            
        Returns:
            Dict[str, Any]: Resultado del diff
        """
        result = {
            'lines_added': 0,
            'lines_removed': 0,
            'lines_changed': 0,
            'diff': None
        }
        
        # Obtener rutas completas
        project_path = old_snapshot.get('project_path', '')
        old_file_path = Path(project_path) / file_path
        new_file_path = Path(new_snapshot.get('project_path', '')) / file_path
        
        try:
            # Leer contenidos
            with open(old_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                old_content = f.read().splitlines()
            
            with open(new_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                new_content = f.read().splitlines()
            
            # Calcular diff
            diff = difflib.unified_diff(
                old_content,
                new_content,
                fromfile=f'a/{file_path}',
                tofile=f'b/{file_path}',
                lineterm=''
            )
            
            diff_text = '\n'.join(diff)
            result['diff'] = diff_text
            
            # Contar cambios
            for line in diff_text.splitlines():
                if line.startswith('+') and not line.startswith('+++'):
                    result['lines_added'] += 1
                elif line.startswith('-') and not line.startswith('---'):
                    result['lines_removed'] += 1
                elif line.startswith('?'):
                    result['lines_changed'] += 1
            
        except (IOError, OSError, UnicodeDecodeError):
            pass
        
        return result
    
    def _detect_renamed_files(self, old_files: Dict[str, Any], 
                             new_files: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Detectar archivos renombrados por hash"""
        renamed = []
        
        # Crear mapeo hash -> nombre para archivos antiguos
        old_hash_to_name = {}
        for name, info in old_files.items():
            file_hash = info.get('hash')
            if file_hash:
                old_hash_to_name[file_hash] = name
        
        # Buscar en archivos nuevos
        for new_name, new_info in new_files.items():
            new_hash = new_info.get('hash')
            if new_hash and new_hash in old_hash_to_name:
                old_name = old_hash_to_name[new_hash]
                if old_name != new_name:
                    renamed.append((old_name, new_name))
        
        return renamed
    
    def detect_changes_since(self, project_path: str, 
                           since_timestamp: Optional[datetime] = None) -> ChangeSet:
        """
        Detectar cambios desde un timestamp específico
        
        Args:
            project_path: Ruta del proyecto
            since_timestamp: Timestamp de referencia
            
        Returns:
            ChangeSet: Cambios detectados
        """
        # Cargar historial
        history = self._load_history()
        
        # Encontrar snapshot más cercano
        reference_snapshot = None
        if since_timestamp:
            for snapshot in history.get('snapshots', []):
                snapshot_time = datetime.fromisoformat(snapshot['timestamp'])
                if snapshot_time <= since_timestamp:
                    reference_snapshot = snapshot
        
        # Si no hay snapshot de referencia, crear uno vacío
        if not reference_snapshot:
            reference_snapshot = {
                'timestamp': since_timestamp.isoformat() if since_timestamp else datetime.now().isoformat(),
                'project_path': project_path,
                'files': {},
                'metadata': {'total_files': 0}
            }
        
        # Tomar snapshot actual
        current_snapshot = self.take_snapshot(project_path)
        
        # Comparar
        change_set = self.compare_snapshots(reference_snapshot, current_snapshot)
        
        # Agregar timestamp al change set
        change_set.timestamp = datetime.fromisoformat(current_snapshot['timestamp'])
        
        return change_set
    
    def analyze_change_impact(self, change_set: ChangeSet, 
                            dependency_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizar impacto de los cambios en el sistema
        
        Args:
            change_set: Conjunto de cambios
            dependency_graph: Grafo de dependencias
            
        Returns:
            Dict[str, Any]: Análisis de impacto
        """
        impact_analysis = {
            'affected_files': [],
            'affected_modules': [],
            'potential_breakage': [],
            'test_impact': [],
            'risk_level': 'low'
        }
        
        # Extraer archivos cambiados
        changed_files = {change.file_path for change in change_set.changes 
                        if change.change_type in ['modified', 'deleted']}
        
        if not changed_files:
            return impact_analysis
        
        # Buscar archivos que dependen de los cambiados
        affected_files = set()
        
        # Suponiendo que dependency_graph tiene estructura de grafo
        # Esta es una implementación simplificada
        for change_file in changed_files:
            # Buscar dependientes directos
            for edge in dependency_graph.get('edges', []):
                if edge['target'] == change_file:
                    affected_files.add(edge['source'])
        
        impact_analysis['affected_files'] = list(affected_files)
        
        # Calcular nivel de riesgo
        total_affected = len(affected_files)
        
        if total_affected == 0:
            impact_analysis['risk_level'] = 'low'
        elif total_affected <= 3:
            impact_analysis['risk_level'] = 'medium'
        elif total_affected <= 10:
            impact_analysis['risk_level'] = 'high'
        else:
            impact_analysis['risk_level'] = 'critical'
        
        # Identificar potenciales rupturas
        for change in change_set.changes:
            if change.change_type == 'deleted':
                impact_analysis['potential_breakage'].append({
                    'file': change.file_path,
                    'reason': 'Archivo eliminado',
                    'severity': 'high'
                })
            elif change.lines_removed > 50:
                impact_analysis['potential_breakage'].append({
                    'file': change.file_path,
                    'reason': f'Cambios extensos ({change.lines_removed} líneas eliminadas)',
                    'severity': 'medium'
                })
        
        return impact_analysis
    
    def track_changes_over_time(self, project_path: str, 
                              days: int = 30) -> Dict[str, Any]:
        """
        Rastrear cambios a lo largo del tiempo
        
        Args:
            project_path: Ruta del proyecto
            days: Número de días a analizar
            
        Returns:
            Dict[str, Any]: Tendencias de cambios
        """
        # Cargar historial
        history = self._load_history()
        snapshots = history.get('snapshots', [])
        
        if not snapshots:
            return {'error': 'No hay historial disponible'}
        
        # Filtrar snapshots de los últimos N días
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_snapshots = [
            s for s in snapshots 
            if datetime.fromisoformat(s['timestamp']).timestamp() > cutoff_date
        ]
        
        if len(recent_snapshots) < 2:
            return {'error': 'No hay suficientes snapshots recientes'}
        
        # Analizar tendencias
        trends = {
            'daily_changes': [],
            'file_growth': [],
            'change_frequency': defaultdict(int),
            'most_changed_files': [],
            'change_types': defaultdict(int)
        }
        
        # Ordenar snapshots por fecha
        recent_snapshots.sort(key=lambda x: x['timestamp'])
        
        # Analizar cambios entre snapshots consecutivos
        for i in range(len(recent_snapshots) - 1):
            old_snapshot = recent_snapshots[i]
            new_snapshot = recent_snapshots[i + 1]
            
            change_set = self.compare_snapshots(old_snapshot, new_snapshot)
            
            date = datetime.fromisoformat(new_snapshot['timestamp']).strftime('%Y-%m-%d')
            
            trends['daily_changes'].append({
                'date': date,
                'files_changed': change_set.total_files,
                'lines_added': change_set.total_lines_added,
                'lines_removed': change_set.total_lines_removed
            })
            
            # Contar tipos de cambio
            for change in change_set.changes:
                trends['change_types'][change.change_type] += 1
        
        # Identificar archivos más cambiados
        file_change_count = defaultdict(int)
        for snapshot in recent_snapshots:
            for file_path in snapshot.get('files', {}).keys():
                file_change_count[file_path] += 1
        
        most_changed = sorted(file_change_count.items(), key=lambda x: x[1], reverse=True)[:10]
        trends['most_changed_files'] = [
            {'file': file, 'changes': count} 
            for file, count in most_changed
        ]
        
        return trends
    
    def _load_history(self) -> Dict[str, Any]:
        """Cargar historial de cambios desde archivo"""
        if not os.path.exists(self.history_file):
            return {'snapshots': [], 'last_updated': datetime.now().isoformat()}
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {'snapshots': [], 'last_updated': datetime.now().isoformat()}
    
    def _save_history(self, history: Dict[str, Any]):
        """Guardar historial de cambios en archivo"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except IOError:
            pass
    
    def save_snapshot(self, snapshot: Dict[str, Any]):
        """
        Guardar snapshot en el historial
        
        Args:
            snapshot: Snapshot a guardar
        """
        history = self._load_history()
        
        # Agregar snapshot
        history['snapshots'].append(snapshot)
        history['last_updated'] = datetime.now().isoformat()
        
        # Limitar tamaño del historial
        if len(history['snapshots']) > self.max_history_size:
            history['snapshots'] = history['snapshots'][-self.max_history_size:]
        
        self._save_history(history)
    
    def generate_change_report(self, change_set: ChangeSet, 
                             format: str = 'markdown') -> str:
        """
        Generar reporte de cambios
        
        Args:
            change_set: Conjunto de cambios
            format: Formato del reporte ('markdown', 'html', 'json')
            
        Returns:
            str: Reporte generado
        """
        if format == 'markdown':
            return self._generate_markdown_report(change_set)
        elif format == 'json':
            return self._generate_json_report(change_set)
        elif format == 'html':
            return self._generate_html_report(change_set)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def _generate_markdown_report(self, change_set: ChangeSet) -> str:
        """Generar reporte en formato Markdown"""
        report = []
        
        report.append(f"# Reporte de Cambios")
        report.append(f"Fecha: {change_set.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total archivos cambiados: {change_set.total_files}")
        report.append(f"Líneas añadidas: {change_set.total_lines_added}")
        report.append(f"Líneas eliminadas: {change_set.total_lines_removed}")
        report.append("")
        
        # Agrupar por tipo de cambio
        changes_by_type = {}
        for change in change_set.changes:
            if change.change_type not in changes_by_type:
                changes_by_type[change.change_type] = []
            changes_by_type[change.change_type].append(change)
        
        for change_type, changes in changes_by_type.items():
            report.append(f"## {change_type.capitalize()} ({len(changes)})")
            
            for change in changes:
                report.append(f"* **{change.file_path}**")
                if change.lines_added > 0 or change.lines_removed > 0:
                    report.append(f"  - +{change.lines_added}/-{change.lines_removed} líneas")
                if change.author:
                    report.append(f"  - Autor: {change.author}")
                if change.commit_hash:
                    report.append(f"  - Commit: {change.commit_hash[:8]}")
            
            report.append("")
        
        return '\n'.join(report)
    
    def _generate_json_report(self, change_set: ChangeSet) -> str:
        """Generar reporte en formato JSON"""
        report_data = {
            'timestamp': change_set.timestamp.isoformat(),
            'total_files': change_set.total_files,
            'total_lines_added': change_set.total_lines_added,
            'total_lines_removed': change_set.total_lines_removed,
            'changes': []
        }
        
        for change in change_set.changes:
            change_data = {
                'file_path': change.file_path,
                'change_type': change.change_type,
                'lines_added': change.lines_added,
                'lines_removed': change.lines_removed,
                'lines_changed': change.lines_changed,
                'author': change.author,
                'commit_hash': change.commit_hash
            }
            report_data['changes'].append(change_data)
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def _generate_html_report(self, change_set: ChangeSet) -> str:
        """Generar reporte en formato HTML"""
        html = []
        
        html.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Cambios</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .summary { background: #f5f5f5; padding: 10px; border-radius: 5px; }
                .changes { margin-top: 20px; }
                .change-type { margin-top: 15px; padding: 10px; background: #e9e9e9; }
                .change-item { margin: 5px 0; padding: 5px; border-left: 3px solid #007acc; }
                .added { border-left-color: #4caf50; }
                .deleted { border-left-color: #f44336; }
                .modified { border-left-color: #ff9800; }
                .renamed { border-left-color: #9c27b0; }
            </style>
        </head>
        <body>
        """)
        
        html.append(f"<h1>Reporte de Cambios</h1>")
        html.append(f"<div class='summary'>")
        html.append(f"<p><strong>Fecha:</strong> {change_set.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"<p><strong>Total archivos cambiados:</strong> {change_set.total_files}</p>")
        html.append(f"<p><strong>Líneas añadidas:</strong> {change_set.total_lines_added}</p>")
        html.append(f"<p><strong>Líneas eliminadas:</strong> {change_set.total_lines_removed}</p>")
        html.append("</div>")
        
        # Agrupar por tipo de cambio
        changes_by_type = {}
        for change in change_set.changes:
            if change.change_type not in changes_by_type:
                changes_by_type[change.change_type] = []
            changes_by_type[change.change_type].append(change)
        
        html.append("<div class='changes'>")
        for change_type, changes in changes_by_type.items():
            html.append(f"<div class='change-type'>")
            html.append(f"<h3>{change_type.capitalize()} ({len(changes)})</h3>")
            
            for change in changes:
                css_class = change.change_type
                html.append(f"<div class='change-item {css_class}'>")
                html.append(f"<strong>{change.file_path}</strong>")
                
                if change.lines_added > 0 or change.lines_removed > 0:
                    html.append(f"<br>+{change.lines_added}/-{change.lines_removed} líneas")
                
                if change.author:
                    html.append(f"<br>Autor: {change.author}")
                
                if change.commit_hash:
                    html.append(f"<br>Commit: {change.commit_hash[:8]}")
                
                html.append("</div>")
            
            html.append("</div>")
        
        html.append("</div>")
        html.append("</body></html>")
        
        return '\n'.join(html)
    
    def detect_code_movements(self, old_snapshot: Dict[str, Any], 
                            new_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detectar movimientos de código entre archivos
        
        Args:
            old_snapshot: Snapshot anterior
            new_snapshot: Snapshot nuevo
            
        Returns:
            List[Dict[str, Any]]: Movimientos de código detectados
        """
        movements = []
        
        # Extraer archivos modificados
        old_files = old_snapshot.get('files', {})
        new_files = new_snapshot.get('files', {})
        
        modified_files = []
        for file_path in set(old_files.keys()) & set(new_files.keys()):
            if old_files[file_path]['hash'] != new_files[file_path]['hash']:
                modified_files.append(file_path)
        
        if len(modified_files) < 2:
            return movements
        
        # Comparar contenido de archivos modificados
        # Esta es una implementación simplificada
        project_path = old_snapshot.get('project_path', '')
        
        for i, file1 in enumerate(modified_files):
            for file2 in modified_files[i+1:]:
                similarity = self._calculate_file_similarity(
                    Path(project_path) / file1,
                    Path(project_path) / file2,
                    old_snapshot, new_snapshot
                )
                
                if similarity > 0.7:  # Umbral de similitud
                    movements.append({
                        'source_file': file1,
                        'target_file': file2,
                        'similarity': similarity,
                        'type': 'code_movement'
                    })
        
        return movements
    
    def _calculate_file_similarity(self, file1_path: Path, file2_path: Path,
                                 old_snapshot: Dict[str, Any], 
                                 new_snapshot: Dict[str, Any]) -> float:
        """Calcular similitud entre dos archivos"""
        try:
            # Leer contenidos
            with open(file1_path, 'r', encoding='utf-8', errors='ignore') as f:
                content1 = f.read()
            
            with open(file2_path, 'r', encoding='utf-8', errors='ignore') as f:
                content2 = f.read()
            
            # Usar SequenceMatcher para calcular similitud
            matcher = difflib.SequenceMatcher(None, content1, content2)
            return matcher.ratio()
            
        except (IOError, OSError, UnicodeDecodeError):
            return 0.0