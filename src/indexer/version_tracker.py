"""
Módulo VersionTracker - Seguimiento de versiones y control de cambios
"""

import os
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from enum import Enum
import subprocess
import sqlite3

class VersioningSystem(Enum):
    """Sistemas de control de versiones soportados"""
    GIT = 'git'
    SVN = 'svn'
    MERCURIAL = 'hg'
    NONE = 'none'

@dataclass
class VersionInfo:
    """Información de versión de un archivo"""
    file_path: str
    current_hash: str
    previous_hash: Optional[str] = None
    version: str = "1.0.0"
    author: Optional[str] = None
    last_modified: datetime = field(default_factory=datetime.now)
    commit_hash: Optional[str] = None
    change_description: Optional[str] = None

@dataclass
class RepositoryInfo:
    """Información del repositorio"""
    system: VersioningSystem
    root_path: str
    current_branch: Optional[str] = None
    latest_commit: Optional[str] = None
    commit_count: int = 0
    branch_count: int = 0
    tag_count: int = 0

class VersionTracker:
    """Seguimiento de versiones y control de cambios"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar tracker de versiones"""
        self.config = config or {}
        self.db_path = self.config.get('database_path', '.codeindex_versions.db')
        self._init_database()
        
    def _init_database(self):
        """Inicializar base de datos para seguimiento de versiones"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabla de versiones de archivos
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    version_hash TEXT NOT NULL,
                    previous_hash TEXT,
                    version_string TEXT,
                    author TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    commit_hash TEXT,
                    change_description TEXT,
                    metadata TEXT
                )
            ''')
            
            # Tabla de tags/etiquetas
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_name TEXT NOT NULL,
                    version_string TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    file_hashes TEXT
                )
            ''')
            
            # Índices para búsqueda eficiente
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON file_versions(file_path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON file_versions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_version_hash ON file_versions(version_hash)')
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            print(f"Error inicializando base de datos: {e}")
    
    def detect_versioning_system(self, project_path: str) -> RepositoryInfo:
        """
        Detectar sistema de control de versiones en uso
        
        Args:
            project_path: Ruta del proyecto
            
        Returns:
            RepositoryInfo: Información del repositorio detectado
        """
        project_root = Path(project_path).resolve()
        
        # Verificar Git
        git_dir = project_root / '.git'
        if git_dir.exists() and git_dir.is_dir():
            return self._get_git_info(project_root)
        
        # Verificar SVN
        svn_dir = project_root / '.svn'
        if svn_dir.exists() and svn_dir.is_dir():
            return self._get_svn_info(project_root)
        
        # Verificar Mercurial
        hg_dir = project_root / '.hg'
        if hg_dir.exists() and hg_dir.is_dir():
            return self._get_hg_info(project_root)
        
        # Sin sistema de control de versiones
        return RepositoryInfo(
            system=VersioningSystem.NONE,
            root_path=str(project_root)
        )
    
    def _get_git_info(self, project_root: Path) -> RepositoryInfo:
        """Obtener información de repositorio Git"""
        info = RepositoryInfo(
            system=VersioningSystem.GIT,
            root_path=str(project_root)
        )
        
        try:
            # Obtener rama actual
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                info.current_branch = result.stdout.strip()
            
            # Obtener último commit
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                info.latest_commit = result.stdout.strip()[:8]
            
            # Contar commits
            result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                info.commit_count = int(result.stdout.strip())
            
            # Contar ramas
            result = subprocess.run(
                ['git', 'branch', '-a'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                branches = result.stdout.strip().split('\n')
                info.branch_count = len([b for b in branches if b.strip()])
            
            # Contar tags
            result = subprocess.run(
                ['git', 'tag'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                tags = result.stdout.strip().split('\n')
                info.tag_count = len([t for t in tags if t.strip()])
                
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return info
    
    def _get_svn_info(self, project_root: Path) -> RepositoryInfo:
        """Obtener información de repositorio SVN"""
        info = RepositoryInfo(
            system=VersioningSystem.SVN,
            root_path=str(project_root)
        )
        
        try:
            # Obtener información SVN
            result = subprocess.run(
                ['svn', 'info'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('Revision:'):
                        info.latest_commit = line.split(':')[1].strip()
                        info.commit_count = int(info.latest_commit)
                    elif line.startswith('URL:'):
                        # Intentar extraer rama de la URL
                        url = line.split(':')[1].strip()
                        if '/branches/' in url:
                            info.current_branch = url.split('/branches/')[1].split('/')[0]
                        elif '/trunk' in url:
                            info.current_branch = 'trunk'
                        
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return info
    
    def _get_hg_info(self, project_root: Path) -> RepositoryInfo:
        """Obtener información de repositorio Mercurial"""
        info = RepositoryInfo(
            system=VersioningSystem.MERCURIAL,
            root_path=str(project_root)
        )
        
        try:
            # Obtener rama actual
            result = subprocess.run(
                ['hg', 'branch'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                info.current_branch = result.stdout.strip()
            
            # Obtener último commit
            result = subprocess.run(
                ['hg', 'log', '-l', '1', '--template', '{node}'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                info.latest_commit = result.stdout.strip()[:8]
            
            # Contar commits
            result = subprocess.run(
                ['hg', 'log', '--template', 'x'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                info.commit_count = len(result.stdout.strip())
                
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return info
    
    def track_file_version(self, file_path: str, content: str, 
                          author: Optional[str] = None,
                          commit_hash: Optional[str] = None,
                          change_description: Optional[str] = None) -> VersionInfo:
        """
        Rastrear versión de un archivo
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            author: Autor del cambio (opcional)
            commit_hash: Hash del commit (opcional)
            change_description: Descripción del cambio (opcional)
            
        Returns:
            VersionInfo: Información de versión
        """
        # Calcular hash del contenido actual
        current_hash = self._calculate_content_hash(content)
        
        # Obtener versión anterior
        previous_version = self._get_previous_version(file_path)
        previous_hash = previous_version.current_hash if previous_version else None
        
        # Determinar nueva versión
        new_version = self._determine_next_version(
            file_path, current_hash, previous_hash
        )
        
        # Crear objeto VersionInfo
        version_info = VersionInfo(
            file_path=file_path,
            current_hash=current_hash,
            previous_hash=previous_hash,
            version=new_version,
            author=author,
            commit_hash=commit_hash,
            change_description=change_description
        )
        
        # Guardar en base de datos
        self._save_version_to_db(version_info)
        
        return version_info
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calcular hash del contenido"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_previous_version(self, file_path: str) -> Optional[VersionInfo]:
        """Obtener versión anterior de un archivo"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_path, version_hash, previous_hash, version_string,
                       author, timestamp, commit_hash, change_description
                FROM file_versions
                WHERE file_path = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (file_path,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return VersionInfo(
                    file_path=row[0],
                    current_hash=row[1],
                    previous_hash=row[2],
                    version=row[3],
                    author=row[4],
                    last_modified=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                    commit_hash=row[6],
                    change_description=row[7]
                )
                
        except sqlite3.Error:
            pass
        
        return None
    
    def _determine_next_version(self, file_path: str, current_hash: str, 
                               previous_hash: Optional[str]) -> str:
        """Determinar siguiente número de versión"""
        if previous_hash is None:
            # Primera versión
            return "1.0.0"
        
        if current_hash == previous_hash:
            # Sin cambios
            return self._get_current_version(file_path) or "1.0.0"
        
        # Obtener versión actual
        current_version = self._get_current_version(file_path)
        if not current_version:
            return "1.0.1"
        
        # Incrementar versión (semántica simple)
        parts = current_version.split('.')
        if len(parts) == 3:
            try:
                major, minor, patch = map(int, parts)
                patch += 1
                return f"{major}.{minor}.{patch}"
            except ValueError:
                pass
        
        return "1.0.1"
    
    def _get_current_version(self, file_path: str) -> Optional[str]:
        """Obtener versión actual de un archivo"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT version_string
                FROM file_versions
                WHERE file_path = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (file_path,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return row[0]
                
        except sqlite3.Error:
            pass
        
        return None
    
    def _save_version_to_db(self, version_info: VersionInfo):
        """Guardar información de versión en base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO file_versions 
                (file_path, version_hash, previous_hash, version_string,
                 author, timestamp, commit_hash, change_description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version_info.file_path,
                version_info.current_hash,
                version_info.previous_hash,
                version_info.version,
                version_info.author,
                version_info.last_modified.isoformat(),
                version_info.commit_hash,
                version_info.change_description
            ))
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            print(f"Error guardando versión en BD: {e}")
    
    def get_version_history(self, file_path: str, limit: int = 10) -> List[VersionInfo]:
        """
        Obtener historial de versiones de un archivo
        
        Args:
            file_path: Ruta del archivo
            limit: Límite de versiones a retornar
            
        Returns:
            List[VersionInfo]: Historial de versiones
        """
        history = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_path, version_hash, previous_hash, version_string,
                       author, timestamp, commit_hash, change_description
                FROM file_versions
                WHERE file_path = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (file_path, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                version_info = VersionInfo(
                    file_path=row[0],
                    current_hash=row[1],
                    previous_hash=row[2],
                    version=row[3],
                    author=row[4],
                    last_modified=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                    commit_hash=row[6],
                    change_description=row[7]
                )
                history.append(version_info)
                
        except sqlite3.Error:
            pass
        
        return history
    
    def create_tag(self, tag_name: str, version_string: Optional[str] = None,
                  description: Optional[str] = None,
                  file_hashes: Optional[List[str]] = None) -> bool:
        """
        Crear una etiqueta/tag para el estado actual
        
        Args:
            tag_name: Nombre de la etiqueta
            version_string: String de versión (opcional)
            description: Descripción de la etiqueta (opcional)
            file_hashes: Lista de hashes de archivos a etiquetar (opcional)
            
        Returns:
            bool: True si se creó exitosamente
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convertir lista de hashes a JSON
            hashes_json = json.dumps(file_hashes) if file_hashes else None
            
            cursor.execute('''
                INSERT INTO version_tags 
                (tag_name, version_string, description, file_hashes)
                VALUES (?, ?, ?, ?)
            ''', (tag_name, version_string, description, hashes_json))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            print(f"Error creando tag: {e}")
            return False
    
    def get_tags(self) -> List[Dict[str, Any]]:
        """
        Obtener todas las etiquetas/tags
        
        Returns:
            List[Dict[str, Any]]: Lista de etiquetas
        """
        tags = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, tag_name, version_string, timestamp, description, file_hashes
                FROM version_tags
                ORDER BY timestamp DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                tag_info = {
                    'id': row[0],
                    'name': row[1],
                    'version': row[2],
                    'timestamp': datetime.fromisoformat(row[3]) if row[3] else None,
                    'description': row[4],
                    'file_hashes': json.loads(row[5]) if row[5] else []
                }
                tags.append(tag_info)
                
        except (sqlite3.Error, json.JSONDecodeError):
            pass
        
        return tags
    
    def restore_version(self, file_path: str, target_hash: str) -> Optional[str]:
        """
        Restaurar archivo a una versión específica
        
        Args:
            file_path: Ruta del archivo
            target_hash: Hash de la versión a restaurar
            
        Returns:
            Optional[str]: Contenido restaurado o None si no se encontró
        """
        # Esta implementación asume que el contenido se almacena externamente
        # En una implementación real, necesitarías almacenar el contenido
        
        # Por ahora, solo verificamos si la versión existe
        history = self.get_version_history(file_path, limit=100)
        
        for version in history:
            if version.current_hash == target_hash:
                # En una implementación real, aquí cargaríamos el contenido
                return f"<!-- Restored version {target_hash} of {file_path} -->"
        
        return None
    
    def compare_versions(self, file_path: str, hash1: str, hash2: str) -> Dict[str, Any]:
        """
        Comparar dos versiones de un archivo
        
        Args:
            file_path: Ruta del archivo
            hash1: Hash de la primera versión
            hash2: Hash de la segunda versión
            
        Returns:
            Dict[str, Any]: Resultado de la comparación
        """
        result = {
            'file': file_path,
            'version1': hash1,
            'version2': hash2,
            'are_equal': hash1 == hash2,
            'changes_detected': hash1 != hash2
        }
        
        if hash1 != hash2:
            # Obtener información de ambas versiones
            history = self.get_version_history(file_path, limit=100)
            
            version1_info = None
            version2_info = None
            
            for version in history:
                if version.current_hash == hash1:
                    version1_info = version
                if version.current_hash == hash2:
                    version2_info = version
                
                if version1_info and version2_info:
                    break
            
            if version1_info and version2_info:
                result['version1_info'] = {
                    'version': version1_info.version,
                    'author': version1_info.author,
                    'timestamp': version1_info.last_modified.isoformat(),
                    'commit': version1_info.commit_hash,
                    'description': version1_info.change_description
                }
                
                result['version2_info'] = {
                    'version': version2_info.version,
                    'author': version2_info.author,
                    'timestamp': version2_info.last_modified.isoformat(),
                    'commit': version2_info.commit_hash,
                    'description': version2_info.change_description
                }
        
        return result
    
    def get_file_statistics(self, file_path: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de versiones de un archivo
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Dict[str, Any]: Estadísticas del archivo
        """
        stats = {
            'file_path': file_path,
            'total_versions': 0,
            'first_version': None,
            'last_version': None,
            'authors': set(),
            'version_frequency': {},
            'change_rate': 0
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Contar versiones totales
            cursor.execute('''
                SELECT COUNT(*) FROM file_versions WHERE file_path = ?
            ''', (file_path,))
            stats['total_versions'] = cursor.fetchone()[0]
            
            # Obtener primera y última versión
            cursor.execute('''
                SELECT version_string, timestamp, author
                FROM file_versions
                WHERE file_path = ?
                ORDER BY timestamp
                LIMIT 1
            ''', (file_path,))
            
            first_row = cursor.fetchone()
            if first_row:
                stats['first_version'] = {
                    'version': first_row[0],
                    'timestamp': first_row[1],
                    'author': first_row[2]
                }
            
            cursor.execute('''
                SELECT version_string, timestamp, author
                FROM file_versions
                WHERE file_path = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (file_path,))
            
            last_row = cursor.fetchone()
            if last_row:
                stats['last_version'] = {
                    'version': last_row[0],
                    'timestamp': last_row[1],
                    'author': last_row[2]
                }
            
            # Obtener lista de autores
            cursor.execute('''
                SELECT DISTINCT author FROM file_versions 
                WHERE file_path = ? AND author IS NOT NULL
            ''', (file_path,))
            
            authors = cursor.fetchall()
            stats['authors'] = {author[0] for author in authors if author[0]}
            
            # Calcular frecuencia de cambios por mes
            cursor.execute('''
                SELECT strftime('%Y-%m', timestamp) as month, COUNT(*)
                FROM file_versions
                WHERE file_path = ?
                GROUP BY month
                ORDER BY month
            ''', (file_path,))
            
            monthly_data = cursor.fetchall()
            stats['version_frequency'] = {
                month: count for month, count in monthly_data
            }
            
            # Calcular tasa de cambio (versiones por día en promedio)
            if stats['total_versions'] > 1 and first_row and last_row:
                first_date = datetime.fromisoformat(first_row[1])
                last_date = datetime.fromisoformat(last_row[1])
                
                days_diff = (last_date - first_date).days
                if days_diff > 0:
                    stats['change_rate'] = stats['total_versions'] / days_diff
            
            conn.close()
            
        except sqlite3.Error:
            pass
        
        return stats
    
    def integrate_with_vcs(self, project_path: str) -> Dict[str, Any]:
        """
        Integrar con sistema de control de versiones externo
        
        Args:
            project_path: Ruta del proyecto
            
        Returns:
            Dict[str, Any]: Resultado de la integración
        """
        repo_info = self.detect_versioning_system(project_path)
        
        result = {
            'vcs_system': repo_info.system.value,
            'integration_successful': False,
            'commits_imported': 0,
            'details': {}
        }
        
        if repo_info.system == VersioningSystem.GIT:
            result.update(self._integrate_with_git(project_path, repo_info))
        elif repo_info.system == VersioningSystem.SVN:
            result.update(self._integrate_with_svn(project_path, repo_info))
        elif repo_info.system == VersioningSystem.MERCURIAL:
            result.update(self._integrate_with_hg(project_path, repo_info))
        else:
            result['details'] = {'message': 'No se detectó sistema de control de versiones'}
        
        return result
    
    def _integrate_with_git(self, project_path: str, repo_info: RepositoryInfo) -> Dict[str, Any]:
        """Integrar con repositorio Git"""
        result = {
            'integration_successful': True,
            'details': {
                'branch': repo_info.current_branch,
                'latest_commit': repo_info.latest_commit,
                'total_commits': repo_info.commit_count
            }
        }
        
        try:
            # Obtener historial de commits recientes
            git_cmd = ['git', 'log', '--oneline', '--no-abbrev-commit', '-10']
            process = subprocess.run(
                git_cmd,
                cwd=project_path,
                capture_output=True,
                text=True
            )
            
            if process.returncode == 0:
                commits = process.stdout.strip().split('\n')
                result['details']['recent_commits'] = commits
                result['commits_imported'] = min(10, len(commits))
        
        except (subprocess.SubprocessError, FileNotFoundError):
            result['integration_successful'] = False
            result['details']['error'] = 'Error ejecutando comandos Git'
        
        return result
    
    def _integrate_with_svn(self, project_path: str, repo_info: RepositoryInfo) -> Dict[str, Any]:
        """Integrar con repositorio SVN"""
        result = {
            'integration_successful': True,
            'details': {
                'revision': repo_info.latest_commit,
                'total_revisions': repo_info.commit_count
            }
        }
        
        try:
            # Obtener historial reciente
            svn_cmd = ['svn', 'log', '-l', '10']
            process = subprocess.run(
                svn_cmd,
                cwd=project_path,
                capture_output=True,
                text=True
            )
            
            if process.returncode == 0:
                result['details']['recent_log'] = process.stdout.strip()
                result['commits_imported'] = 10
        
        except (subprocess.SubprocessError, FileNotFoundError):
            result['integration_successful'] = False
            result['details']['error'] = 'Error ejecutando comandos SVN'
        
        return result
    
    def _integrate_with_hg(self, project_path: str, repo_info: RepositoryInfo) -> Dict[str, Any]:
        """Integrar con repositorio Mercurial"""
        result = {
            'integration_successful': True,
            'details': {
                'branch': repo_info.current_branch,
                'latest_changeset': repo_info.latest_commit
            }
        }
        
        try:
            # Obtener historial reciente
            hg_cmd = ['hg', 'log', '-l', '10', '--template', '{node|short}: {desc}\\n']
            process = subprocess.run(
                hg_cmd,
                cwd=project_path,
                capture_output=True,
                text=True
            )
            
            if process.returncode == 0:
                changesets = process.stdout.strip().split('\n')
                result['details']['recent_changesets'] = changesets
                result['commits_imported'] = min(10, len(changesets))
        
        except (subprocess.SubprocessError, FileNotFoundError):
            result['integration_successful'] = False
            result['details']['error'] = 'Error ejecutando comandos Mercurial'
        
        return result
    
    def generate_version_report(self, project_path: str, 
                              format: str = 'markdown') -> str:
        """
        Generar reporte de versiones del proyecto
        
        Args:
            project_path: Ruta del proyecto
            format: Formato del reporte ('markdown', 'html', 'json')
            
        Returns:
            str: Reporte generado
        """
        # Obtener información del repositorio
        repo_info = self.detect_versioning_system(project_path)
        
        # Obtener archivos más versionados
        most_versioned_files = self._get_most_versioned_files(limit=10)
        
        if format == 'markdown':
            return self._generate_version_markdown_report(repo_info, most_versioned_files)
        elif format == 'json':
            return self._generate_version_json_report(repo_info, most_versioned_files)
        elif format == 'html':
            return self._generate_version_html_report(repo_info, most_versioned_files)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def _get_most_versioned_files(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener archivos con más versiones"""
        files = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_path, COUNT(*) as version_count,
                       MAX(timestamp) as last_modified,
                       GROUP_CONCAT(DISTINCT author) as authors
                FROM file_versions
                GROUP BY file_path
                ORDER BY version_count DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                file_info = {
                    'file_path': row[0],
                    'version_count': row[1],
                    'last_modified': row[2],
                    'authors': row[3].split(',') if row[3] else []
                }
                files.append(file_info)
                
        except sqlite3.Error:
            pass
        
        return files
    
    def _generate_version_markdown_report(self, repo_info: RepositoryInfo,
                                        most_versioned_files: List[Dict[str, Any]]) -> str:
        """Generar reporte de versiones en Markdown"""
        report = []
        
        report.append("# Reporte de Control de Versiones")
        report.append("")
        report.append(f"**Sistema detectado:** {repo_info.system.value.upper()}")
        
        if repo_info.system != VersioningSystem.NONE:
            report.append(f"**Rama actual:** {repo_info.current_branch or 'N/A'}")
            report.append(f"**Último commit:** {repo_info.latest_commit or 'N/A'}")
            report.append(f"**Total commits:** {repo_info.commit_count}")
            report.append(f"**Total ramas:** {repo_info.branch_count}")
            report.append(f"**Total tags:** {repo_info.tag_count}")
        
        report.append("")
        report.append("## Archivos más versionados")
        report.append("")
        
        if most_versioned_files:
            report.append("| Archivo | Versiones | Última modificación | Autores |")
            report.append("|---------|-----------|---------------------|---------|")
            
            for file_info in most_versioned_files:
                authors = ', '.join(file_info['authors'][:3])
                if len(file_info['authors']) > 3:
                    authors += f" (+{len(file_info['authors']) - 3})"
                
                report.append(f"| {file_info['file_path']} | {file_info['version_count']} | "
                            f"{file_info['last_modified'][:10]} | {authors} |")
        else:
            report.append("No hay datos de versiones disponibles.")
        
        return '\n'.join(report)
    
    def _generate_version_json_report(self, repo_info: RepositoryInfo,
                                    most_versioned_files: List[Dict[str, Any]]) -> str:
        """Generar reporte de versiones en JSON"""
        report_data = {
            'vcs_system': repo_info.system.value,
            'repository_info': {
                'root_path': repo_info.root_path,
                'current_branch': repo_info.current_branch,
                'latest_commit': repo_info.latest_commit,
                'commit_count': repo_info.commit_count,
                'branch_count': repo_info.branch_count,
                'tag_count': repo_info.tag_count
            },
            'most_versioned_files': most_versioned_files
        }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def _generate_version_html_report(self, repo_info: RepositoryInfo,
                                    most_versioned_files: List[Dict[str, Any]]) -> str:
        """Generar reporte de versiones en HTML"""
        html = []
        
        html.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Control de Versiones</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .vcs-info { background: #f0f8ff; padding: 15px; border-radius: 5px; }
                .files-table { margin-top: 20px; width: 100%; border-collapse: collapse; }
                .files-table th, .files-table td { 
                    border: 1px solid #ddd; padding: 8px; text-align: left; 
                }
                .files-table th { background-color: #4CAF50; color: white; }
                .files-table tr:nth-child(even) { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
        """)
        
        html.append("<h1>Reporte de Control de Versiones</h1>")
        html.append("<div class='vcs-info'>")
        html.append(f"<p><strong>Sistema detectado:</strong> {repo_info.system.value.upper()}</p>")
        
        if repo_info.system != VersioningSystem.NONE:
            html.append(f"<p><strong>Rama actual:</strong> {repo_info.current_branch or 'N/A'}</p>")
            html.append(f"<p><strong>Último commit:</strong> {repo_info.latest_commit or 'N/A'}</p>")
            html.append(f"<p><strong>Total commits:</strong> {repo_info.commit_count}</p>")
            html.append(f"<p><strong>Total ramas:</strong> {repo_info.branch_count}</p>")
            html.append(f"<p><strong>Total tags:</strong> {repo_info.tag_count}</p>")
        
        html.append("</div>")
        
        html.append("<h2>Archivos más versionados</h2>")
        
        if most_versioned_files:
            html.append("<table class='files-table'>")
            html.append("<tr><th>Archivo</th><th>Versiones</th><th>Última modificación</th><th>Autores</th></tr>")
            
            for file_info in most_versioned_files:
                authors = ', '.join(file_info['authors'][:3])
                if len(file_info['authors']) > 3:
                    authors += f" (+{len(file_info['authors']) - 3})"
                
                html.append("<tr>")
                html.append(f"<td>{file_info['file_path']}</td>")
                html.append(f"<td>{file_info['version_count']}</td>")
                html.append(f"<td>{file_info['last_modified'][:10]}</td>")
                html.append(f"<td>{authors}</td>")
                html.append("</tr>")
            
            html.append("</table>")
        else:
            html.append("<p>No hay datos de versiones disponibles.</p>")
        
        html.append("</body></html>")
        
        return '\n'.join(html)