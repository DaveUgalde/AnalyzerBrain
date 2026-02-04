from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class ProjectStructure:
    """Estructura de un proyecto."""

    root: Path
    files: List[Path]
    directories: List[Path]
    metadata: Dict[str, Any]


@dataclass
class CodeEntity:
    """Entidad de código extraída."""

    type: str  # 'class', 'function', 'variable', 'import', etc.
    name: str
    location: Path
    line_start: int
    line_end: int
    properties: Dict[str, Any]
    parent: Optional['CodeEntity'] = None
