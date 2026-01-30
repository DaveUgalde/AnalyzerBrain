"""
memory_retriever.py - Sistema de recuperación de memorias.

Responsable de recuperar memorias del sistema utilizando múltiples estrategias:
- Recuperación por clave
- Recuperación por similitud semántica
- Recuperación por contexto temporal
- Recuperación por contexto semántico
- Combinación inteligente de recuperaciones
- Ranking y validación de resultados
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json
from uuid import uuid4
from pydantic import BaseModel, Field, validator
import numpy as np

from ..core.exceptions import MemoryException, ValidationError
from .memory_hierarchy import MemoryHierarchy
from ..embeddings.semantic_search import SemanticSearch
from ..embeddings.similarity_calculator import SimilarityCalculator


class RetrievalMethod(Enum):
    """Métodos de recuperación disponibles."""
    KEY = "key"
    SIMILARITY = "similarity"
    CONTEXT = "context"
    TIME = "time"
    HYBRID = "hybrid"


class RetrievalPriority(Enum):
    """Prioridades de recuperación."""
    RECENCY = "recency"
    RELEVANCE = "relevance"
    FREQUENCY = "frequency"
    IMPORTANCE = "importance"


@dataclass
class RetrievalConfig:
    """Configuración para recuperación de memorias."""
    method: RetrievalMethod = RetrievalMethod.HYBRID
    priority: RetrievalPriority = RetrievalPriority.RELEVANCE
    max_results: int = 10
    similarity_threshold: float = 0.7
    time_window_hours: Optional[int] = None
    context_weight: float = 0.6
    similarity_weight: float = 0.3
    recency_weight: float = 0.1
    combine_method: str = "weighted_sum"  # weighted_sum, rank_fusion, ensemble
    enable_caching: bool = True
    cache_ttl_seconds: int = 300


class MemoryQuery(BaseModel):
    """Consulta para recuperación de memoria."""
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    query_text: Optional[str] = None
    query_vector: Optional[List[float]] = None
    key: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    filters: Optional[Dict[str, Any]] = None
    config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('query_vector')
    def validate_query_vector(cls, v):
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Query vector must be a list")
            if len(v) == 0:
                raise ValueError("Query vector cannot be empty")
        return v


class RetrievalResult(BaseModel):
    """Resultado de recuperación de memoria."""
    query_id: str
    memories: List[Dict[str, Any]]
    scores: List[float]
    method_used: RetrievalMethod
    processing_time_ms: float
    total_memories_searched: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class MemoryRetriever:
    """
    Sistema avanzado de recuperación de memorias.
    
    Características:
    1. Múltiples estrategias de recuperación (key, similarity, context, time)
    2. Combinación inteligente de resultados
    3. Ranking por múltiples criterios
    4. Caché de consultas frecuentes
    5. Validación de calidad de recuperación
    """
    
    def __init__(
        self,
        memory_hierarchy: MemoryHierarchy,
        semantic_search: Optional[SemanticSearch] = None,
        similarity_calculator: Optional[SimilarityCalculator] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa el recuperador de memorias.
        
        Args:
            memory_hierarchy: Jerarquía de memoria para acceso
            semantic_search: Sistema de búsqueda semántica (opcional)
            similarity_calculator: Calculadora de similitudes (opcional)
            config: Configuración adicional
        """
        self.memory_hierarchy = memory_hierarchy
        self.semantic_search = semantic_search
        self.similarity_calculator = similarity_calculator or SimilarityCalculator()
        
        # Configuración por defecto
        self.config = config or {
            "default_max_results": 10,
            "similarity_threshold": 0.7,
            "cache_enabled": True,
            "cache_size": 1000,
            "combination_strategy": "weighted_sum"
        }
        
        # Caché de consultas
        self._query_cache: Dict[str, RetrievalResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Métricas
        self._metrics = {
            "retrievals_performed": 0,
            "average_recall": 0.0,
            "average_precision": 0.0,
            "cache_hit_rate": 0.0,
            "average_processing_time_ms": 0.0
        }
    
    def retrieve_by_key(self, key: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Recupera memorias por clave específica.
        
        Args:
            key: Clave única de la memoria
            **kwargs: Argumentos adicionales (filters, etc.)
            
        Returns:
            Lista de memorias que coinciden con la clave
            
        Raises:
            MemoryException: Si hay error en la recuperación
        """
        try:
            start_time = datetime.now()
            
            # Buscar en jerarquía de memoria
            memory = self.memory_hierarchy.retrieve_from_memory(key)
            
            if memory:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_metrics(success=True, processing_time=processing_time)
                return [memory]
            else:
                # Buscar en niveles más profundos si es necesario
                all_memories = self._search_all_memory_levels(key)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_metrics(success=len(all_memories) > 0, processing_time=processing_time)
                return all_memories
                
        except Exception as e:
            raise MemoryException(f"Failed to retrieve by key '{key}': {e}")
    
    def retrieve_by_similarity(
        self,
        query_vector: List[float],
        top_k: int = 10,
        threshold: float = 0.7,
        **kwargs
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Recupera memorias similares usando embeddings vectoriales.
        
        Args:
            query_vector: Vector de embeddings de consulta
            top_k: Número máximo de resultados
            threshold: Umbral de similitud mínimo
            **kwargs: Argumentos adicionales
            
        Returns:
            Lista de tuplas (memoria, score_similitud)
            
        Raises:
            MemoryException: Si hay error en la recuperación
        """
        try:
            start_time = datetime.now()
            
            results = []
            
            # Usar búsqueda semántica si está disponible
            if self.semantic_search:
                search_results = self.semantic_search.semantic_search(
                    query_vector=query_vector,
                    top_k=top_k * 2,  # Buscar más para filtrar después
                    threshold=threshold
                )
                
                for result in search_results:
                    # Obtener memoria completa por ID
                    memory_id = result.get("memory_id")
                    if memory_id:
                        memory = self.memory_hierarchy.retrieve_from_memory(memory_id)
                        if memory:
                            results.append((memory, result.get("score", 0.0)))
            
            # Si no hay suficiente con búsqueda semántica, buscar por similitud directa
            if len(results) < top_k:
                additional_results = self._search_by_similarity_direct(
                    query_vector, 
                    top_k - len(results),
                    threshold
                )
                results.extend(additional_results)
            
            # Ordenar por score y limitar
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(success=len(results) > 0, processing_time=processing_time)
            
            return results
            
        except Exception as e:
            raise MemoryException(f"Failed to retrieve by similarity: {e}")
    
    def retrieve_by_context(
        self,
        context: Dict[str, Any],
        max_results: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Recupera memorias relevantes para un contexto dado.
        
        Args:
            context: Diccionario con contexto (project_id, entity_type, tags, etc.)
            max_results: Máximo número de resultados
            **kwargs: Argumentos adicionales
            
        Returns:
            Lista de memorias relevantes al contexto
            
        Raises:
            MemoryException: Si hay error en la recuperación
        """
        try:
            start_time = datetime.now()
            
            # Extraer información del contexto
            project_id = context.get("project_id")
            entity_type = context.get("entity_type")
            tags = context.get("tags", [])
            time_range = context.get("time_range")
            
            # Construir filtros
            filters = {}
            if project_id:
                filters["project_id"] = project_id
            if entity_type:
                filters["type"] = entity_type
            if tags:
                filters["tags"] = {"$contains": tags}
            
            # Buscar memorias con filtros
            memories = self._search_memories_with_filters(filters, max_results)
            
            # Si hay rango de tiempo, filtrar por tiempo
            if time_range and memories:
                start_time_range, end_time_range = time_range
                memories = [
                    m for m in memories
                    if start_time_range <= m.get("timestamp", datetime.min) <= end_time_range
                ]
            
            # Ordenar por relevancia contextual
            memories = self._rank_by_context_relevance(memories, context)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(success=len(memories) > 0, processing_time=processing_time)
            
            return memories[:max_results]
            
        except Exception as e:
            raise MemoryException(f"Failed to retrieve by context: {e}")
    
    def retrieve_by_time(
        self,
        start_time: datetime,
        end_time: datetime,
        time_priority: str = "recent",
        max_results: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Recupera memorias dentro de un rango de tiempo.
        
        Args:
            start_time: Tiempo de inicio
            end_time: Tiempo de fin
            time_priority: Prioridad ('recent', 'oldest', 'frequent')
            max_results: Máximo número de resultados
            **kwargs: Argumentos adicionales
            
        Returns:
            Lista de memorias en el rango temporal
            
        Raises:
            MemoryException: Si hay error en la recuperación
        """
        try:
            start = datetime.now()
            
            # Construir filtro de tiempo
            time_filter = {
                "timestamp": {
                    "$gte": start_time.isoformat(),
                    "$lte": end_time.isoformat()
                }
            }
            
            # Buscar memorias en el rango de tiempo
            memories = self._search_memories_with_filters(time_filter, max_results * 2)
            
            # Ordenar según prioridad temporal
            if time_priority == "recent":
                memories.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
            elif time_priority == "oldest":
                memories.sort(key=lambda x: x.get("timestamp", datetime.min))
            elif time_priority == "frequent":
                # Ordenar por frecuencia de acceso
                memories.sort(key=lambda x: x.get("access_count", 0), reverse=True)
            
            processing_time = (datetime.now() - start).total_seconds() * 1000
            self._update_metrics(success=len(memories) > 0, processing_time=processing_time)
            
            return memories[:max_results]
            
        except Exception as e:
            raise MemoryException(f"Failed to retrieve by time: {e}")
    
    def combine_retrievals(
        self,
        retrieval_sets: List[List[Dict[str, Any]]],
        method: str = "weighted_sum",
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Combina múltiples conjuntos de recuperación.
        
        Args:
            retrieval_sets: Lista de listas de memorias recuperadas
            method: Método de combinación ('weighted_sum', 'rank_fusion', 'ensemble')
            weights: Pesos para cada conjunto (si es weighted_sum)
            **kwargs: Argumentos adicionales
            
        Returns:
            Lista combinada y ordenada de memorias
            
        Raises:
            ValidationError: Si los parámetros son inválidos
        """
        if not retrieval_sets:
            return []
        
        if method == "weighted_sum":
            return self._combine_weighted_sum(retrieval_sets, weights)
        elif method == "rank_fusion":
            return self._combine_rank_fusion(retrieval_sets)
        elif method == "ensemble":
            return self._combine_ensemble(retrieval_sets, **kwargs)
        else:
            raise ValidationError(f"Unknown combination method: {method}")
    
    def rank_retrieved_memories(
        self,
        memories: List[Dict[str, Any]],
        ranking_criteria: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Ordena memorias recuperadas por criterios múltiples.
        
        Args:
            memories: Lista de memorias a ordenar
            ranking_criteria: Diccionario de criterios y pesos
            **kwargs: Argumentos adicionales
            
        Returns:
            Lista de tuplas (memoria, score_total) ordenadas
            
        Raises:
            ValidationError: Si los criterios son inválidos
        """
        if not memories:
            return []
        
        # Criterios por defecto
        default_criteria = {
            "relevance": 0.4,
            "recency": 0.3,
            "frequency": 0.2,
            "importance": 0.1
        }
        
        criteria = ranking_criteria or default_criteria
        
        # Validar pesos
        total_weight = sum(criteria.values())
        if abs(total_weight - 1.0) > 0.01:  # Tolerancia pequeña
            raise ValidationError(f"Criteria weights must sum to 1.0, got {total_weight}")
        
        # Calcular scores para cada memoria
        scored_memories = []
        for memory in memories:
            total_score = 0.0
            
            # Score de relevancia (si está disponible)
            if "relevance" in criteria and "relevance_score" in memory:
                total_score += memory["relevance_score"] * criteria["relevance"]
            
            # Score de recencia
            if "recency" in criteria:
                recency_score = self._calculate_recency_score(memory)
                total_score += recency_score * criteria["recency"]
            
            # Score de frecuencia
            if "frequency" in criteria:
                frequency_score = self._calculate_frequency_score(memory)
                total_score += frequency_score * criteria["frequency"]
            
            # Score de importancia
            if "importance" in criteria:
                importance_score = self._calculate_importance_score(memory)
                total_score += importance_score * criteria["importance"]
            
            scored_memories.append((memory, total_score))
        
        # Ordenar por score descendente
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return scored_memories
    
    def validate_retrieval(
        self,
        retrieved_memories: List[Dict[str, Any]],
        expected_count: Optional[int] = None,
        min_similarity: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Valida la calidad de una recuperación de memorias.
        
        Args:
            retrieved_memories: Memorias recuperadas a validar
            expected_count: Número esperado de resultados (opcional)
            min_similarity: Similitud mínima esperada (opcional)
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con métricas de validación
        """
        validation_result = {
            "is_valid": True,
            "total_retrieved": len(retrieved_memories),
            "validation_metrics": {},
            "warnings": [],
            "errors": []
        }
        
        # Validar número de resultados
        if expected_count is not None:
            actual_count = len(retrieved_memories)
            validation_result["validation_metrics"]["count_match"] = {
                "expected": expected_count,
                "actual": actual_count,
                "match": actual_count >= expected_count * 0.8  # 80% de recall
            }
            
            if actual_count < expected_count * 0.5:
                validation_result["warnings"].append(
                    f"Low recall: retrieved {actual_count} of expected {expected_count}"
                )
        
        # Validar similitud (si hay scores)
        if min_similarity is not None:
            similarity_scores = [m.get("similarity_score", 0.0) for m in retrieved_memories]
            if similarity_scores:
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                min_actual_similarity = min(similarity_scores)
                
                validation_result["validation_metrics"]["similarity"] = {
                    "min_expected": min_similarity,
                    "min_actual": min_actual_similarity,
                    "average": avg_similarity,
                    "meets_threshold": min_actual_similarity >= min_similarity
                }
                
                if min_actual_similarity < min_similarity:
                    validation_result["errors"].append(
                        f"Low similarity: min={min_actual_similarity:.3f}, expected={min_similarity:.3f}"
                    )
        
        # Validar duplicados
        memory_ids = [m.get("id") for m in retrieved_memories if m.get("id")]
        unique_ids = set(memory_ids)
        duplicate_count = len(memory_ids) - len(unique_ids)
        
        validation_result["validation_metrics"]["duplicates"] = {
            "total": len(memory_ids),
            "unique": len(unique_ids),
            "duplicates": duplicate_count
        }
        
        if duplicate_count > 0:
            validation_result["warnings"].append(f"Found {duplicate_count} duplicate memories")
        
        # Validar calidad de datos
        quality_issues = self._validate_memory_quality(retrieved_memories)
        if quality_issues:
            validation_result["validation_metrics"]["quality_issues"] = quality_issues
            validation_result["warnings"].extend(
                [f"Quality issue: {issue}" for issue in quality_issues]
            )
        
        # Determinar si es válido
        has_errors = len(validation_result["errors"]) > 0
        validation_result["is_valid"] = not has_errors
        
        return validation_result
    
    def retrieve(
        self,
        query: MemoryQuery,
        use_cache: bool = True
    ) -> RetrievalResult:
        """
        Recuperación principal que selecciona automáticamente el mejor método.
        
        Args:
            query: Objeto de consulta con parámetros
            use_cache: Si se debe usar caché de consultas
            
        Returns:
            Resultado de la recuperación
            
        Raises:
            MemoryException: Si hay error en la recuperación
        """
        try:
            start_time = datetime.now()
            
            # Generar clave de caché
            cache_key = None
            if use_cache and self.config.get("cache_enabled", True):
                cache_key = self._generate_cache_key(query)
                if cache_key in self._query_cache:
                    self._cache_hits += 1
                    result = self._query_cache[cache_key]
                    # Actualizar timestamp para LRU
                    self._query_cache[cache_key] = result
                    return result
            
            self._cache_misses += 1
            
            # Seleccionar método de recuperación
            method = query.config.method
            
            if method == RetrievalMethod.KEY:
                memories = self.retrieve_by_key(
                    key=query.key or "",
                    **(query.filters or {})
                )
                scores = [1.0] * len(memories)  # Máxima confianza para key exacta
                
            elif method == RetrievalMethod.SIMILARITY:
                if not query.query_vector:
                    raise ValidationError("Query vector required for similarity retrieval")
                
                results = self.retrieve_by_similarity(
                    query_vector=query.query_vector,
                    top_k=query.config.max_results,
                    threshold=query.config.similarity_threshold,
                    **(query.filters or {})
                )
                memories = [r[0] for r in results]
                scores = [r[1] for r in results]
                
            elif method == RetrievalMethod.CONTEXT:
                if not query.context:
                    raise ValidationError("Context required for context retrieval")
                
                memories = self.retrieve_by_context(
                    context=query.context,
                    max_results=query.config.max_results,
                    **(query.filters or {})
                )
                scores = self._calculate_context_scores(memories, query.context)
                
            elif method == RetrievalMethod.TIME:
                if not query.time_range:
                    # Usar ventana por defecto
                    end_time = datetime.now()
                    start_time_range = end_time - timedelta(hours=query.config.time_window_hours or 24)
                    query.time_range = (start_time_range, end_time)
                
                memories = self.retrieve_by_time(
                    start_time=query.time_range[0],
                    end_time=query.time_range[1],
                    time_priority=query.config.priority.value,
                    max_results=query.config.max_results,
                    **(query.filters or {})
                )
                scores = self._calculate_time_scores(memories, query.time_range)
                
            elif method == RetrievalMethod.HYBRID:
                # Recuperación híbrida combinando múltiples métodos
                memories, scores = self._hybrid_retrieval(query)
                
            else:
                raise ValidationError(f"Unknown retrieval method: {method}")
            
            # Combinar con otras recuperaciones si se especifica
            if query.config.combine_method != "none" and len(memories) > 1:
                # En un caso real, aquí se combinarían múltiples recuperaciones
                pass
            
            # Ordenar por scores
            if scores and len(scores) == len(memories):
                sorted_pairs = sorted(zip(memories, scores), key=lambda x: x[1], reverse=True)
                memories = [pair[0] for pair in sorted_pairs]
                scores = [pair[1] for pair in sorted_pairs]
            
            # Limitar resultados
            memories = memories[:query.config.max_results]
            scores = scores[:query.config.max_results]
            
            # Crear resultado
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result = RetrievalResult(
                query_id=query.query_id,
                memories=memories,
                scores=scores,
                method_used=method,
                processing_time_ms=processing_time,
                total_memories_searched=self._estimate_total_searched(method),
                metadata={
                    "cache_hit": cache_key in self._query_cache if cache_key else False,
                    "config_used": query.config.__dict__
                }
            )
            
            # Guardar en caché
            if cache_key and self.config.get("cache_enabled", True):
                self._query_cache[cache_key] = result
                # Mantener tamaño de caché
                if len(self._query_cache) > self.config.get("cache_size", 1000):
                    oldest_key = next(iter(self._query_cache))
                    del self._query_cache[oldest_key]
            
            # Actualizar métricas
            self._update_metrics(success=True, processing_time=processing_time)
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(success=False, processing_time=processing_time)
            raise MemoryException(f"Retrieval failed: {e}")
    
    # ==================== MÉTODOS PRIVADOS ====================
    
    def _search_all_memory_levels(self, key: str) -> List[Dict[str, Any]]:
        """Busca en todos los niveles de memoria."""
        memories = []
        
        # Buscar en diferentes niveles (simulado)
        # En implementación real, esto buscaría en L1, L2, L3, etc.
        try:
            # Intento 1: Memoria principal
            memory = self.memory_hierarchy.retrieve_from_memory(key)
            if memory:
                memories.append(memory)
        except:
            pass
        
        # Podríamos buscar en otros almacenes aquí
        # Por ejemplo: base de datos, sistema de archivos, etc.
        
        return memories
    
    def _search_by_similarity_direct(
        self,
        query_vector: List[float],
        top_k: int,
        threshold: float
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Búsqueda directa por similitud (fallback)."""
        results = []
        
        # Obtener memorias candidatas (simulado)
        candidate_memories = self._get_candidate_memories(limit=top_k * 5)
        
        for memory in candidate_memories:
            # Extraer vector de la memoria
            memory_vector = memory.get("embedding")
            if memory_vector and isinstance(memory_vector, list):
                # Calcular similitud
                similarity = self.similarity_calculator.calculate_cosine_similarity(
                    query_vector, memory_vector
                )
                
                if similarity >= threshold:
                    results.append((memory, similarity))
        
        # Ordenar por similitud
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _search_memories_with_filters(
        self,
        filters: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Busca memorias con filtros."""
        # En implementación real, esto consultaría la base de datos
        # Por ahora, simulamos una búsqueda
        
        # Obtener todas las memorias (simulado)
        all_memories = self._get_candidate_memories(limit=limit * 10)
        
        # Aplicar filtros
        filtered_memories = []
        for memory in all_memories:
            if self._memory_matches_filters(memory, filters):
                filtered_memories.append(memory)
                
                if len(filtered_memories) >= limit:
                    break
        
        return filtered_memories
    
    def _memory_matches_filters(self, memory: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Verifica si una memoria coincide con los filtros."""
        for key, value in filters.items():
            if key not in memory:
                return False
            
            memory_value = memory[key]
            
            # Manejar diferentes tipos de filtros
            if isinstance(value, dict) and "$contains" in value:
                # Filtro de contención (para listas)
                if not isinstance(memory_value, list):
                    return False
                
                required_items = value["$contains"]
                if not all(item in memory_value for item in required_items):
                    return False
                    
            elif isinstance(value, dict) and "$gte" in value and "$lte" in value:
                # Filtro de rango
                if not (value["$gte"] <= memory_value <= value["$lte"]):
                    return False
                    
            elif memory_value != value:
                return False
        
        return True
    
    def _rank_by_context_relevance(
        self,
        memories: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Ordena memorias por relevancia contextual."""
        if not memories:
            return memories
        
        # Calcular score de relevancia para cada memoria
        scored_memories = []
        for memory in memories:
            score = 0.0
            
            # Coincidencia de project_id
            if context.get("project_id") == memory.get("project_id"):
                score += 0.4
            
            # Coincidencia de entity_type
            if context.get("entity_type") == memory.get("type"):
                score += 0.3
            
            # Coincidencia de tags
            context_tags = set(context.get("tags", []))
            memory_tags = set(memory.get("tags", []))
            if context_tags and memory_tags:
                tag_overlap = len(context_tags.intersection(memory_tags)) / len(context_tags)
                score += tag_overlap * 0.3
            
            scored_memories.append((memory, score))
        
        # Ordenar por score descendente
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, _ in scored_memories]
    
    def _combine_weighted_sum(
        self,
        retrieval_sets: List[List[Dict[str, Any]]],
        weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Combina conjuntos usando suma ponderada."""
        if not retrieval_sets:
            return []
        
        # Pesos por defecto si no se especifican
        if weights is None:
            weights = [1.0 / len(retrieval_sets)] * len(retrieval_sets)
        elif len(weights) != len(retrieval_sets):
            raise ValidationError("Number of weights must match number of retrieval sets")
        
        # Crear diccionario de memoria -> score acumulado
        memory_scores = {}
        
        for i, memory_list in enumerate(retrieval_sets):
            weight = weights[i]
            
            for j, memory in enumerate(memory_list):
                memory_id = memory.get("id", str(j))
                
                # Score basado en posición (inverso)
                position_score = 1.0 / (j + 1)
                
                if memory_id in memory_scores:
                    memory_scores[memory_id]["score"] += position_score * weight
                else:
                    memory_scores[memory_id] = {
                        "memory": memory,
                        "score": position_score * weight
                    }
        
        # Convertir a lista y ordenar
        combined = list(memory_scores.values())
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        return [item["memory"] for item in combined]
    
    def _combine_rank_fusion(
        self,
        retrieval_sets: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Combina conjuntos usando fusión de rankings."""
        # Implementación simplificada de Reciprocal Rank Fusion
        k = 60  # Parámetro de suavizado
        
        memory_scores = {}
        
        for memory_list in retrieval_sets:
            for rank, memory in enumerate(memory_list):
                memory_id = memory.get("id", str(rank))
                
                # Calcular score RRF
                rrf_score = 1.0 / (k + rank + 1)
                
                if memory_id in memory_scores:
                    memory_scores[memory_id]["score"] += rrf_score
                else:
                    memory_scores[memory_id] = {
                        "memory": memory,
                        "score": rrf_score
                    }
        
        # Ordenar por score
        combined = list(memory_scores.values())
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        return [item["memory"] for item in combined]
    
    def _combine_ensemble(
        self,
        retrieval_sets: List[List[Dict[str, Any]]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Combina conjuntos usando método ensemble."""
        # Primero, combinar con weighted sum
        weighted_result = self._combine_weighted_sum(retrieval_sets)
        
        # Luego, combinar con rank fusion
        rank_result = self._combine_rank_fusion(retrieval_sets)
        
        # Finalmente, combinar ambos resultados
        return self._combine_weighted_sum([weighted_result, rank_result], [0.5, 0.5])
    
    def _calculate_recency_score(self, memory: Dict[str, Any]) -> float:
        """Calcula score de recencia para una memoria."""
        timestamp = memory.get("timestamp")
        if not timestamp:
            return 0.5
        
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return 0.5
        
        # Score basado en antigüedad (más reciente = score más alto)
        age_days = (datetime.now() - timestamp).total_seconds() / (24 * 3600)
        
        # Decaimiento exponencial con vida media de 30 días
        half_life_days = 30.0
        score = 0.5 ** (age_days / half_life_days)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_frequency_score(self, memory: Dict[str, Any]) -> float:
        """Calcula score de frecuencia para una memoria."""
        access_count = memory.get("access_count", 0)
        
        # Normalizar usando función logarítmica
        if access_count <= 0:
            return 0.0
        
        score = np.log10(access_count + 1) / 3.0  # log10(1000) = 3, así que normalizamos
        
        return max(0.0, min(1.0, score))
    
    def _calculate_importance_score(self, memory: Dict[str, Any]) -> float:
        """Calcula score de importancia para una memoria."""
        # Basado en metadatos de importancia
        importance = memory.get("importance", 0.5)
        
        # Ajustar por tipo de memoria
        memory_type = memory.get("type", "")
        type_multiplier = {
            "function": 1.0,
            "class": 1.2,
            "api": 1.1,
            "config": 0.8,
            "test": 0.7,
            "documentation": 0.6
        }.get(memory_type, 1.0)
        
        return max(0.0, min(1.0, importance * type_multiplier))
    
    def _validate_memory_quality(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Valida la calidad de las memorias."""
        issues = []
        
        for i, memory in enumerate(memories):
            # Verificar campos requeridos
            if "id" not in memory:
                issues.append(f"Memory {i}: missing 'id' field")
            
            if "content" not in memory:
                issues.append(f"Memory {i}: missing 'content' field")
            
            # Verificar tipo de contenido
            content = memory.get("content", "")
            if not content or not isinstance(content, (str, dict, list)):
                issues.append(f"Memory {i}: invalid content type")
            
            # Verificar timestamp
            timestamp = memory.get("timestamp")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    issues.append(f"Memory {i}: invalid timestamp format")
        
        return issues
    
    def _hybrid_retrieval(self, query: MemoryQuery) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Realiza recuperación híbrida combinando múltiples métodos."""
        all_memories = []
        all_scores = []
        
        # Determinar qué métodos usar basado en la consulta
        methods_to_try = []
        
        if query.key:
            methods_to_try.append(("key", RetrievalMethod.KEY))
        
        if query.query_vector:
            methods_to_try.append(("similarity", RetrievalMethod.SIMILARITY))
        
        if query.context:
            methods_to_try.append(("context", RetrievalMethod.CONTEXT))
        
        if query.time_range:
            methods_to_try.append(("time", RetrievalMethod.TIME))
        
        # Si no hay métodos específicos, usar todos
        if not methods_to_try:
            methods_to_try = [
                ("similarity", RetrievalMethod.SIMILARITY),
                ("context", RetrievalMethod.CONTEXT),
                ("time", RetrievalMethod.TIME)
            ]
        
        # Ejecutar cada método
        method_results = []
        
        for method_name, method_type in methods_to_try:
            try:
                if method_type == RetrievalMethod.KEY:
                    memories = self.retrieve_by_key(query.key or "")
                    scores = [1.0] * len(memories)
                    
                elif method_type == RetrievalMethod.SIMILARITY:
                    results = self.retrieve_by_similarity(
                        query_vector=query.query_vector or [],
                        top_k=query.config.max_results,
                        threshold=query.config.similarity_threshold
                    )
                    memories = [r[0] for r in results]
                    scores = [r[1] for r in results]
                    
                elif method_type == RetrievalMethod.CONTEXT:
                    memories = self.retrieve_by_context(
                        context=query.context or {},
                        max_results=query.config.max_results
                    )
                    scores = self._calculate_context_scores(memories, query.context or {})
                    
                elif method_type == RetrievalMethod.TIME:
                    time_range = query.time_range
                    if not time_range:
                        end_time = datetime.now()
                        start_time = end_time - timedelta(hours=24)
                        time_range = (start_time, end_time)
                    
                    memories = self.retrieve_by_time(
                        start_time=time_range[0],
                        end_time=time_range[1],
                        time_priority=query.config.priority.value,
                        max_results=query.config.max_results
                    )
                    scores = self._calculate_time_scores(memories, time_range)
                
                method_results.append((memories, scores, method_name))
                
            except Exception as e:
                # Continuar con otros métodos si uno falla
                continue
        
        # Combinar resultados
        if method_results:
            # Usar pesos de la configuración
            context_weight = query.config.context_weight
            similarity_weight = query.config.similarity_weight
            recency_weight = query.config.recency_weight
            
            memory_dict = {}  # memory_id -> (memory, scores_by_method)
            
            for memories, scores, method_name in method_results:
                method_weight = {
                    "context": context_weight,
                    "similarity": similarity_weight,
                    "time": recency_weight,
                    "key": 1.0  # Máxima prioridad para key
                }.get(method_name, 0.5)
                
                for memory, score in zip(memories, scores):
                    memory_id = memory.get("id", str(uuid4()))
                    
                    if memory_id not in memory_dict:
                        memory_dict[memory_id] = {
                            "memory": memory,
                            "scores": {}
                        }
                    
                    memory_dict[memory_id]["scores"][method_name] = score * method_weight
            
            # Calcular score combinado para cada memoria
            for memory_id, data in memory_dict.items():
                memory = data["memory"]
                scores = data["scores"]
                
                # Promedio ponderado de scores
                if scores:
                    total_score = sum(scores.values())
                    avg_score = total_score / len(scores)
                else:
                    avg_score = 0.0
                
                all_memories.append(memory)
                all_scores.append(avg_score)
        
        return all_memories, all_scores
    
    def _calculate_context_scores(
        self,
        memories: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[float]:
        """Calcula scores de relevancia contextual."""
        scores = []
        
        for memory in memories:
            score = 0.0
            
            # Coincidencia exacta de project_id
            if context.get("project_id") == memory.get("project_id"):
                score += 0.3
            
            # Coincidencia de tipo de entidad
            if context.get("entity_type") == memory.get("type"):
                score += 0.2
            
            # Coincidencia de tags
            context_tags = set(context.get("tags", []))
            memory_tags = set(memory.get("tags", []))
            if context_tags and memory_tags:
                overlap = len(context_tags.intersection(memory_tags))
                score += (overlap / max(len(context_tags), 1)) * 0.3
            
            # Coincidencia de palabras clave en contenido
            if "keywords" in context and "content" in memory:
                keywords = context["keywords"]
                content = str(memory["content"]).lower()
                
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
                score += (keyword_matches / max(len(keywords), 1)) * 0.2
            
            scores.append(min(1.0, score))
        
        return scores
    
    def _calculate_time_scores(
        self,
        memories: List[Dict[str, Any]],
        time_range: Tuple[datetime, datetime]
    ) -> List[float]:
        """Calcula scores basados en tiempo."""
        if not time_range:
            return [0.5] * len(memories)
        
        start_time, end_time = time_range
        range_duration = (end_time - start_time).total_seconds()
        
        if range_duration <= 0:
            return [0.5] * len(memories)
        
        scores = []
        
        for memory in memories:
            timestamp = memory.get("timestamp")
            if not timestamp:
                scores.append(0.5)
                continue
            
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    scores.append(0.5)
                    continue
            
            # Score basado en proximidad al centro del rango
            time_center = start_time + (end_time - start_time) / 2
            time_diff = abs((timestamp - time_center).total_seconds())
            
            # Normalizar: score máximo en el centro, decae hacia los extremos
            normalized_diff = time_diff / (range_duration / 2)
            score = max(0.0, 1.0 - normalized_diff)
            
            scores.append(score)
        
        return scores
    
    def _generate_cache_key(self, query: MemoryQuery) -> str:
        """Genera una clave única para caché basada en la consulta."""
        # Crear string representativa de la consulta
        query_data = {
            "method": query.config.method.value,
            "priority": query.config.priority.value,
            "max_results": query.config.max_results,
            "key": query.key,
            "query_text": query.query_text,
            "context": json.dumps(query.context, sort_keys=True) if query.context else "",
            "filters": json.dumps(query.filters, sort_keys=True) if query.filters else ""
        }
        
        # Para vector, usar hash del vector
        if query.query_vector:
            vector_str = json.dumps(query.query_vector, sort_keys=True)
            query_data["query_vector_hash"] = hashlib.md5(vector_str.encode()).hexdigest()
        
        # Para rango de tiempo
        if query.time_range:
            query_data["time_range"] = (
                query.time_range[0].isoformat(),
                query.time_range[1].isoformat()
            )
        
        # Convertir a string y hash
        query_str = json.dumps(query_data, sort_keys=True)
        return f"retrieval_cache:{hashlib.md5(query_str.encode()).hexdigest()}"
    
    def _get_candidate_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene memorias candidatas para búsqueda (simulado)."""
        # En implementación real, esto consultaría la base de datos
        # Por ahora, retornamos una lista vacía
        return []
    
    def _estimate_total_searched(self, method: RetrievalMethod) -> int:
        """Estima el total de memorias buscadas para un método."""
        # En implementación real, esto vendría de métricas reales
        estimates = {
            RetrievalMethod.KEY: 1,
            RetrievalMethod.SIMILARITY: 1000,
            RetrievalMethod.CONTEXT: 500,
            RetrievalMethod.TIME: 200,
            RetrievalMethod.HYBRID: 2000
        }
        return estimates.get(method, 100)
    
    def _update_metrics(self, success: bool, processing_time: float) -> None:
        """Actualiza métricas del recuperador."""
        self._metrics["retrievals_performed"] += 1
        
        # Actualizar tiempo promedio
        current_avg = self._metrics["average_processing_time_ms"]
        total_retrievals = self._metrics["retrievals_performed"]
        
        self._metrics["average_processing_time_ms"] = (
            (current_avg * (total_retrievals - 1) + processing_time) / total_retrievals
        )
        
        # Actualizar hit rate de caché
        total_cache_accesses = self._cache_hits + self._cache_misses
        if total_cache_accesses > 0:
            self._metrics["cache_hit_rate"] = self._cache_hits / total_cache_accesses
    
    # ==================== MÉTODOS PÚBLICOS ADICIONALES ====================
    
    def get_retrieval_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del recuperador.
        
        Returns:
            Diccionario con métricas
        """
        return {
            **self._metrics,
            "cache_stats": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "size": len(self._query_cache)
            },
            "config": self.config
        }
    
    def clear_cache(self) -> None:
        """Limpia la caché de consultas."""
        self._query_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def optimize_retrieval(self) -> Dict[str, Any]:
        """
        Optimiza el recuperador basado en métricas.
        
        Returns:
            Diccionario con cambios aplicados
        """
        optimizations = {}
        
        # Optimizar tamaño de caché basado en hit rate
        hit_rate = self._metrics.get("cache_hit_rate", 0.0)
        current_cache_size = self.config.get("cache_size", 1000)
        
        if hit_rate < 0.3 and current_cache_size > 100:
            # Reducir caché si hit rate es bajo
            new_size = max(100, current_cache_size // 2)
            self.config["cache_size"] = new_size
            optimizations["cache_size_reduced"] = f"{current_cache_size} -> {new_size}"
            
        elif hit_rate > 0.7 and current_cache_size < 5000:
            # Aumentar caché si hit rate es alto
            new_size = min(5000, current_cache_size * 2)
            self.config["cache_size"] = new_size
            optimizations["cache_size_increased"] = f"{current_cache_size} -> {new_size}"
        
        # Limpiar caché si está muy llena
        if len(self._query_cache) > self.config.get("cache_size", 1000) * 1.2:
            items_to_remove = len(self._query_cache) - self.config.get("cache_size", 1000)
            keys_to_remove = list(self._query_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self._query_cache[key]
            
            optimizations["cache_cleaned"] = f"Removed {items_to_remove} items"
        
        return optimizations


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancias de dependencias (simuladas)
    class MockMemoryHierarchy:
        def retrieve_from_memory(self, key):
            return {"id": key, "content": f"Memory for {key}", "timestamp": datetime.now()}
    
    class MockSimilarityCalculator:
        def calculate_cosine_similarity(self, v1, v2):
            return 0.8 if v1 and v2 else 0.0
    
    # Inicializar recuperador
    memory_hierarchy = MockMemoryHierarchy()
    similarity_calculator = MockSimilarityCalculator()
    
    retriever = MemoryRetriever(
        memory_hierarchy=memory_hierarchy,
        similarity_calculator=similarity_calculator
    )
    
    # Ejemplo: Recuperación por clave
    print("=== Recuperación por clave ===")
    memories = retriever.retrieve_by_key("test_key")
    for memory in memories:
        print(f"  - {memory.get('id')}: {memory.get('content')[:50]}...")
    
    # Ejemplo: Recuperación por similitud
    print("\n=== Recuperación por similitud ===")
    query_vector = [0.1] * 384  # Vector de ejemplo
    results = retriever.retrieve_by_similarity(query_vector, top_k=3)
    for memory, score in results:
        print(f"  - Score {score:.3f}: {memory.get('id', 'unknown')}")
    
    # Ejemplo: Recuperación usando la interfaz principal
    print("\n=== Recuperación usando interfaz principal ===")
    query = MemoryQuery(
        query_text="Find recent API functions",
        context={"entity_type": "function", "tags": ["api"]},
        config=RetrievalConfig(
            method=RetrievalMethod.HYBRID,
            max_results=5
        )
    )
    
    result = retriever.retrieve(query)
    print(f"Recuperadas {len(result.memories)} memorias en {result.processing_time_ms:.1f}ms")
    print(f"Método usado: {result.method_used.value}")
    
    # Obtener métricas
    print("\n=== Métricas del recuperador ===")
    metrics = retriever.get_retrieval_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")