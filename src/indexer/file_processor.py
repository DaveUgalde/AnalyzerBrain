"""
Módulo FileProcessor - Procesamiento y análisis de archivos individuales
"""

import os
import hashlib
import chardet
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
import mimetypes
import magic  # python-magic

@dataclass
class FileMetadata:
    """Metadatos de un archivo procesado"""
    path: Path
    size: int
    created: datetime
    modified: datetime
    accessed: datetime
    mime_type: str
    encoding: str = "unknown"
    language: str = "unknown"
    line_count: int = 0
    word_count: int = 0
    char_count: int = 0
    hash_md5: str = ""
    hash_sha1: str = ""
    hash_sha256: str = ""

@dataclass
class FileContent:
    """Contenido procesado de un archivo"""
    raw_content: bytes
    text_content: str
    lines: List[str]
    tokens: List[str] = field(default_factory=list)
    ast: Optional[Any] = None
    syntax_tree: Optional[Dict[str, Any]] = None

class FileProcessor:
    """Procesador de archivos para extracción de contenido y metadatos"""
    
    # Mapeo de extensiones a lenguajes
    EXTENSION_TO_LANGUAGE = {
        '.py': 'python',
        '.java': 'java',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c-header',
        '.hpp': 'cpp-header',
        '.cs': 'csharp',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.rs': 'rust',
        '.html': 'html',
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
        '.ps1': 'powershell'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar procesador de archivos"""
        self.config = config or {}
        self.max_file_size = self.config.get('max_file_size', 10 * 1024 * 1024)  # 10MB por defecto
        self.supported_encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'ascii']
        
    def process_file(self, file_path: str) -> Tuple[FileMetadata, FileContent]:
        """
        Procesar un archivo y extraer metadatos y contenido
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Tuple[FileMetadata, FileContent]: Metadatos y contenido del archivo
        """
        path = Path(file_path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        # Obtener metadatos básicos
        stat_info = path.stat()
        metadata = self._extract_metadata(path, stat_info)
        
        # Leer y procesar contenido
        content = self._read_and_process_content(path, metadata)
        
        return metadata, content
    
    def _extract_metadata(self, path: Path, stat_info: os.stat_result) -> FileMetadata:
        """Extraer metadatos del archivo"""
        # Detectar MIME type
        mime_type = self._detect_mime_type(path)
        
        # Detectar lenguaje basado en extensión
        language = self.EXTENSION_TO_LANGUAGE.get(path.suffix.lower(), 'unknown')
        
        metadata = FileMetadata(
            path=path,
            size=stat_info.st_size,
            created=datetime.fromtimestamp(stat_info.st_ctime),
            modified=datetime.fromtimestamp(stat_info.st_mtime),
            accessed=datetime.fromtimestamp(stat_info.st_atime),
            mime_type=mime_type,
            language=language
        )
        
        # Calcular hashes si el archivo no es muy grande
        if metadata.size <= self.max_file_size:
            metadata.hash_md5 = self._calculate_hash(path, 'md5')
            metadata.hash_sha1 = self._calculate_hash(path, 'sha1')
            metadata.hash_sha256 = self._calculate_hash(path, 'sha256')
        
        return metadata
    
    def _detect_mime_type(self, path: Path) -> str:
        """Detectar tipo MIME del archivo"""
        try:
            # Usar python-magic si está disponible
            import magic
            mime_type = magic.from_file(str(path), mime=True)
            return mime_type
        except (ImportError, AttributeError):
            # Fallback a mimetypes
            mime_type, _ = mimetypes.guess_type(str(path))
            return mime_type or 'application/octet-stream'
    
    def _calculate_hash(self, path: Path, algorithm: str) -> str:
        """Calcular hash del archivo"""
        hash_func = getattr(hashlib, algorithm)()
        
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except (IOError, OSError):
            return ""
    
    def _read_and_process_content(self, path: Path, metadata: FileMetadata) -> FileContent:
        """Leer y procesar contenido del archivo"""
        # Verificar tamaño máximo
        if metadata.size > self.max_file_size:
            return FileContent(
                raw_content=b'',
                text_content='',
                lines=[],
                tokens=[]
            )
        
        # Leer contenido binario
        try:
            with open(path, 'rb') as f:
                raw_content = f.read()
        except (IOError, OSError) as e:
            raise IOError(f"No se pudo leer el archivo {path}: {str(e)}")
        
        # Detectar encoding
        encoding = self._detect_encoding(raw_content)
        metadata.encoding = encoding
        
        # Decodificar a texto
        try:
            text_content = raw_content.decode(encoding, errors='replace')
        except UnicodeDecodeError:
            # Fallback a latin-1
            text_content = raw_content.decode('latin-1', errors='replace')
        
        # Procesar líneas
        lines = text_content.splitlines()
        metadata.line_count = len(lines)
        
        # Contar palabras y caracteres
        words = text_content.split()
        metadata.word_count = len(words)
        metadata.char_count = len(text_content)
        
        # Tokenizar (implementación básica)
        tokens = self._tokenize_text(text_content, metadata.language)
        
        return FileContent(
            raw_content=raw_content,
            text_content=text_content,
            lines=lines,
            tokens=tokens
        )
    
    def _detect_encoding(self, raw_content: bytes) -> str:
        """Detectar encoding del contenido"""
        if not raw_content:
            return 'utf-8'
        
        # Usar chardet para detección
        try:
            result = chardet.detect(raw_content)
            if result['confidence'] > 0.7:
                return result['encoding'].lower()
        except:
            pass
        
        # Verificar encodings comunes
        for encoding in self.supported_encodings:
            try:
                raw_content.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        
        # Fallback
        return 'utf-8'
    
    def _tokenize_text(self, text: str, language: str) -> List[str]:
        """Tokenizar texto basado en lenguaje"""
        if language == 'python':
            # Tokenización básica para Python
            import re
            tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[+\-*/=<>!&|^~%]+|[(){}\[\],.:;@]', text)
            return tokens
        else:
            # Tokenización genérica
            import re
            tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
            return tokens
    
    def analyze_complexity(self, content: FileContent, language: str) -> Dict[str, Any]:
        """
        Analizar complejidad del archivo
        
        Args:
            content: Contenido del archivo
            language: Lenguaje del archivo
            
        Returns:
            Dict: Métricas de complejidad
        """
        metrics = {
            'lines_total': len(content.lines),
            'lines_code': 0,
            'lines_comment': 0,
            'lines_blank': 0,
            'cyclomatic_complexity': 0,
            'halstead_volume': 0
        }
        
        # Contar líneas de código, comentarios y blancas
        in_multiline_comment = False
        
        for line in content.lines:
            stripped = line.strip()
            
            if not stripped:
                metrics['lines_blank'] += 1
                continue
            
            # Detectar comentarios basados en lenguaje
            if language == 'python':
                if stripped.startswith('#'):
                    metrics['lines_comment'] += 1
                elif stripped.startswith('"""') or stripped.startswith("'''"):
                    metrics['lines_comment'] += 1
                    if stripped.count('"""') < 2 and stripped.count("'''") < 2:
                        in_multiline_comment = not in_multiline_comment
                elif in_multiline_comment:
                    metrics['lines_comment'] += 1
                else:
                    metrics['lines_code'] += 1
            
            elif language in ['javascript', 'typescript', 'java', 'cpp', 'c']:
                if stripped.startswith('//'):
                    metrics['lines_comment'] += 1
                elif stripped.startswith('/*'):
                    metrics['lines_comment'] += 1
                    if '*/' not in stripped:
                        in_multiline_comment = True
                elif stripped.endswith('*/'):
                    metrics['lines_comment'] += 1
                    in_multiline_comment = False
                elif in_multiline_comment:
                    metrics['lines_comment'] += 1
                else:
                    metrics['lines_code'] += 1
            
            else:
                # Para otros lenguajes, línea no vacía = código
                metrics['lines_code'] += 1
        
        # Calcular complejidad ciclomática aproximada
        metrics['cyclomatic_complexity'] = self._estimate_cyclomatic_complexity(
            content.text_content, language
        )
        
        # Calcular métricas Halstead aproximadas
        metrics['halstead_volume'] = self._estimate_halstead_volume(content.tokens)
        
        return metrics
    
    def _estimate_cyclomatic_complexity(self, text: str, language: str) -> int:
        """Estimar complejidad ciclomática"""
        complexity = 1  # Base
        
        # Patrones que incrementan complejidad
        patterns = [
            r'\bif\b', r'\belse\b', r'\bwhile\b', r'\bfor\b',
            r'\bcase\b', r'\bcatch\b', r'\bthrow\b', r'\breturn\b',
            r'\b&&\b', r'\b\|\|\b', r'\?\s*:',  # operadores ternarios
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            complexity += len(matches)
        
        return complexity
    
    def _estimate_halstead_volume(self, tokens: List[str]) -> float:
        """Estimar volumen Halstead"""
        if not tokens:
            return 0.0
        
        # Contar operadores y operandos únicos
        operators = set()
        operands = set()
        
        # Lista simple de operadores (puede ser extendida)
        operator_patterns = [
            '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
            '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '+=', '-=', '*=',
            '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '++', '--'
        ]
        
        for token in tokens:
            if token in operator_patterns or any(op in token for op in ['=', '<', '>', '!']):
                operators.add(token)
            else:
                operands.add(token)
        
        n1 = len(operators)  # Operadores únicos
        n2 = len(operands)   # Operandos únicos
        N1 = len([t for t in tokens if t in operators])  # Total operadores
        N2 = len([t for t in tokens if t not in operators])  # Total operandos
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Volumen Halstead: V = N * log2(n)
        N = N1 + N2
        n = n1 + n2
        volume = N * (n ** 0.5)  # Aproximación simplificada
        
        return round(volume, 2)
    
    def extract_imports(self, content: FileContent, language: str) -> List[str]:
        """
        Extraer importaciones/requires del archivo
        
        Args:
            content: Contenido del archivo
            language: Lenguaje del archivo
            
        Returns:
            List: Lista de importaciones
        """
        imports = []
        
        if language == 'python':
            import re
            patterns = [
                r'^\s*import\s+([a-zA-Z0-9_.]+)',
                r'^\s*from\s+([a-zA-Z0-9_.]+)\s+import',
            ]
            for line in content.lines:
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        imports.append(match.group(1))
        
        elif language in ['javascript', 'typescript']:
            import re
            patterns = [
                r'^\s*import\s+.*from\s+[\'"]([^\'"]+)[\'"]',
                r'^\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
                r'^\s*export\s+.*from\s+[\'"]([^\'"]+)[\'"]',
            ]
            for line in content.lines:
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        imports.append(match.group(1))
        
        elif language == 'java':
            import re
            pattern = r'^\s*import\s+([a-zA-Z0-9_.*]+)'
            for line in content.lines:
                match = re.search(pattern, line)
                if match:
                    imports.append(match.group(1))
        
        return list(set(imports))  # Eliminar duplicados
    
    def validate_syntax(self, content: FileContent, language: str) -> Dict[str, Any]:
        """
        Validar sintaxis del archivo
        
        Args:
            content: Contenido del archivo
            language: Lenguaje del archivo
            
        Returns:
            Dict: Resultados de validación
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validaciones básicas por lenguaje
        if language == 'python':
            try:
                import ast
                ast.parse(content.text_content)
            except SyntaxError as e:
                result['valid'] = False
                result['errors'].append({
                    'line': e.lineno or 0,
                    'column': e.offset or 0,
                    'message': str(e)
                })
        
        elif language == 'json':
            try:
                import json
                json.loads(content.text_content)
            except json.JSONDecodeError as e:
                result['valid'] = False
                result['errors'].append({
                    'line': e.lineno,
                    'column': e.colno,
                    'message': e.msg
                })
        
        # Validaciones comunes
        self._perform_common_validations(content, result)
        
        return result
    
    def _perform_common_validations(self, content: FileContent, result: Dict[str, Any]):
        """Realizar validaciones comunes"""
        # Verificar líneas muy largas
        max_line_length = self.config.get('max_line_length', 120)
        
        for i, line in enumerate(content.lines, 1):
            if len(line) > max_line_length:
                result['warnings'].append({
                    'line': i,
                    'column': max_line_length + 1,
                    'message': f'Línea muy larga ({len(line)} > {max_line_length} caracteres)'
                })
        
        # Verificar encoding
        if '�' in content.text_content:
            result['warnings'].append({
                'line': 0,
                'column': 0,
                'message': 'Caracteres de reemplazo Unicode detectados (posible problema de encoding)'
            })