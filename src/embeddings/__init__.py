"""
Módulo de embeddings - Representación vectorial del código y documentación.
Proporciona generación, almacenamiento y búsqueda de embeddings vectoriales
para análisis semántico y recuperación de conocimiento.
"""

from .embedding_models import EmbeddingModels
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .semantic_search import SemanticSearch
from .embedding_cache import EmbeddingCache
from .similarity_calculator import SimilarityCalculator
from .dimensionality_reducer import DimensionalityReducer

__version__ = "1.0.0"
__all__ = [
    "EmbeddingModels",
    "EmbeddingGenerator",
    "VectorStore",
    "SemanticSearch",
    "EmbeddingCache",
    "SimilarityCalculator",
    "DimensionalityReducer"
]