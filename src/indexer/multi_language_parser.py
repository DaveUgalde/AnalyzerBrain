"""
Parser para múltiples lenguajes de programación.
Inicialmente solo soporta Python usando el módulo ast.
"""

import ast
from pathlib import Path
from typing import List
from loguru import logger

from ..core.exceptions import IndexerError
from ..utils.file_utils import safe_read_file
from .models import CodeEntity


class MultiLanguageParser:
    """Parser para múltiples lenguajes (comenzando con Python)."""

    def __init__(self):
        self.supported_extensions = {'.py'}

    def can_parse(self, file_path: Path) -> bool:
        """Determina si el parser puede manejar un archivo."""
        return file_path.suffix in self.supported_extensions

    def parse(self, file_path: Path) -> List[CodeEntity]:
        """Parsea un archivo y extrae entidades de código."""
        if not self.can_parse(file_path):
            return []

        try:
            content = safe_read_file(file_path)
            if file_path.suffix == '.py':
                return self._parse_python(content, file_path)
            else:
                return []
        except Exception as e:
            logger.error(f"Error parseando {file_path}: {e}")
            raise IndexerError(f"Error parseando {file_path}: {e}")

    def _parse_python(self, content: str, file_path: Path) -> List[CodeEntity]:
        """Parsea código Python usando ast."""
        entities = []
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                entity = CodeEntity(
                    type='class',
                    name=node.name,
                    location=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    properties={
                        'bases': [ast.unparse(base) for base in node.bases],
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    },
                )
                entities.append(entity)
            elif isinstance(node, ast.FunctionDef):
                entity = CodeEntity(
                    type='function',
                    name=node.name,
                    location=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    properties={
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [ast.unparse(d) for d in node.decorator_list],
                    },
                )
                entities.append(entity)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                # Para imports, podemos extraer los nombres
                for alias in node.names:
                    entity = CodeEntity(
                        type='import',
                        name=alias.name,
                        location=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                        properties={
                            'module': node.module if isinstance(node, ast.ImportFrom) else None,
                            'alias': alias.asname,
                        },
                    )
                    entities.append(entity)

        return entities
