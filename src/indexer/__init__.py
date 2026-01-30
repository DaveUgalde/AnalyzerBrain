"""
Módulo Indexer - Sistema de indexación y análisis de código fuente
"""

from .project_scanner import ProjectScanner
from .file_processor import FileProcessor
from .multi_language_parser import MultiLanguageParser
from .entity_extractor import EntityExtractor
from .dependency_mapper import DependencyMapper
from .change_detector import ChangeDetector
from .version_tracker import VersionTracker
from .quality_analyzer import QualityAnalyzer
from .pattern_detector import PatternDetector

__version__ = "1.0.0"
__all__ = [
    'ProjectScanner',
    'FileProcessor',
    'MultiLanguageParser',
    'EntityExtractor',
    'DependencyMapper',
    'ChangeDetector',
    'VersionTracker',
    'QualityAnalyzer',
    'PatternDetector'
]