"""
SemanticSearch - Búsqueda semántica en embeddings.
Realiza búsquedas por similitud semántica usando embeddings vectoriales.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, validator
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore, SearchResult

class SearchType(Enum):
    """Tipos de búsqueda."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class FilterOperator(Enum):
    """Operadores para filtros."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"

class SearchFilter(BaseModel):
    """Filtro para búsquedas."""
    field: str
    operator: FilterOperator
    value: Any
    
    class Config:
        arbitrary_types_allowed = True

class SearchRequest(BaseModel):
    """Solicitud de búsqueda."""
    query: str
    search_type: SearchType = SearchType.HYBRID
    top_k: int = 10
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    filters: List[SearchFilter] = Field(default_factory=list)
    expand_query: bool = True
    use_cache: bool = True
    
    @validator('top_k')
    def validate_top_k(cls, v):
        if v <= 0 or v > 1000:
            raise ValueError("top_k must be between 1 and 1000")
        return v

class SearchResponse(BaseModel):
    """Respuesta de búsqueda."""
    query: str
    search_type: SearchType
    results: List[SearchResult] = Field(default_factory=list)
    total_results: int = 0
    processing_time_ms: float = 0.0
    query_expansion: Optional[List[str]] = None
    filters_applied: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

@dataclass
class SearchConfig:
    """Configuración de búsqueda semántica."""
    default_search_type: SearchType = SearchType.HYBRID
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    expansion_enabled: bool = True
    max_expansion_terms: int = 3
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_searches: int = 10
    timeout_seconds: int = 30
    
    # Modelos para diferentes tipos de contenido
    models: Dict[str, str] = field(default_factory=lambda: {
        "text": "all-MiniLM-L6-v2",
        "code": "microsoft/codebert-base",
        "document": "all-mpnet-base-v2"
    })

class SemanticSearch:
    """
    Motor de búsqueda semántica.
    
    Características:
    1. Búsqueda semántica usando embeddings
    2. Búsqueda por keywords tradicional
    3. Búsqueda híbrida (semántica + keywords)
    4. Expansión de consultas
    5. Filtrado avanzado
    6. Caché de resultados
    """
    
    def __init__(self, 
                 embedding_generator: EmbeddingGenerator,
                 vector_store: VectorStore,
                 config: Optional[SearchConfig] = None):
        """
        Inicializa el motor de búsqueda semántica.
        
        Args:
            embedding_generator: Generador de embeddings
            vector_store: Almacén vectorial
            config: Configuración de búsqueda (opcional)
        """
        self.generator = embedding_generator
        self.store = vector_store
        self.config = config or SearchConfig()
        self._cache: Dict[str, SearchResponse] = {}
        self._keyword_index: Dict[str, List[str]] = {}
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_searches)
        self._stats = {
            "total_searches": 0,
            "semantic_searches": 0,
            "keyword_searches": 0,
            "hybrid_searches": 0,
            "cache_hits": 0,
            "avg_processing_time_ms": 0.0
        }
        
        # Construir índice de keywords si es necesario
        self._build_keyword_index()
    
    async def semantic_search(self,
                            query: str,
                            top_k: int = 10,
                            threshold: Optional[float] = None,
                            filters: Optional[List[Dict[str, Any]]] = None) -> SearchResponse:
        """
        Realiza búsqueda semántica pura.
        
        Args:
            query: Consulta de búsqueda
            top_k: Número máximo de resultados
            threshold: Umbral mínimo de similitud
            filters: Filtros por metadatos
            
        Returns:
            SearchResponse con resultados
        """
        start_time = datetime.now()
        self._stats["total_searches"] += 1
        self._stats["semantic_searches"] += 1
        
        # Verificar caché
        cache_key = self._create_cache_key(query, SearchType.SEMANTIC, top_k, threshold, filters)
        if self.config.cache_enabled and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            response = self._cache[cache_key]
            response.metadata["cached"] = True
            return response
        
        try:
            # Expandir consulta si está habilitado
            expanded_queries = []
            if self.config.expansion_enabled:
                expanded_queries = await self._expand_query(query)
            
            # Generar embedding para la consulta
            query_embedding = await self.generator.generate_text_embedding(
                query,
                model_name=self.config.models["text"]
            )
            
            # Convertir filtros a formato del almacén vectorial
            store_filters = self._convert_filters(filters) if filters else None
            
            # Buscar en el almacén vectorial
            search_results = self.store.search_similar(
                query_embedding,
                top_k=top_k * 2,  # Buscar más para aplicar filtros
                threshold=threshold or 0.0,
                filters=store_filters
            )
            
            # Aplicar umbral si se especifica
            if threshold is not None:
                search_results = [r for r in search_results if r.score >= threshold]
            
            # Limitar a top_k
            search_results = search_results[:top_k]
            
            # Crear respuesta
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            response = SearchResponse(
                query=query,
                search_type=SearchType.SEMANTIC,
                results=search_results,
                total_results=len(search_results),
                processing_time_ms=processing_time,
                query_expansion=expanded_queries if expanded_queries else None,
                filters_applied=filters if filters else [],
                metadata={
                    "embedding_dimensions": len(query_embedding),
                    "search_method": "semantic_only"
                }
            )
            
            # Actualizar caché
            if self.config.cache_enabled:
                self._cache[cache_key] = response
                self._clean_cache()
            
            # Actualizar estadísticas
            self._update_stats(processing_time)
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return SearchResponse(
                query=query,
                search_type=SearchType.SEMANTIC,
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )
    
    async def keyword_search(self,
                           query: str,
                           top_k: int = 10,
                           filters: Optional[List[Dict[str, Any]]] = None) -> SearchResponse:
        """
        Realiza búsqueda por keywords tradicional.
        
        Args:
            query: Consulta de búsqueda
            top_k: Número máximo de resultados
            filters: Filtros por metadatos
            
        Returns:
            SearchResponse con resultados
        """
        start_time = datetime.now()
        self._stats["total_searches"] += 1
        self._stats["keyword_searches"] += 1
        
        # Verificar caché
        cache_key = self._create_cache_key(query, SearchType.KEYWORD, top_k, None, filters)
        if self.config.cache_enabled and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            response = self._cache[cache_key]
            response.metadata["cached"] = True
            return response
        
        try:
            # Tokenizar consulta
            keywords = self._extract_keywords(query)
            
            # Buscar en índice de keywords
            results = []
            for keyword in keywords:
                if keyword in self._keyword_index:
                    for doc_id in self._keyword_index[keyword]:
                        # Verificar que no esté ya en resultados
                        if not any(r.id == doc_id for r in results):
                            # Obtener metadata del documento
                            metadata = self._get_document_metadata(doc_id)
                            if metadata and self._apply_filters(metadata, filters):
                                score = self._calculate_keyword_score(keyword, doc_id, keywords)
                                results.append(SearchResult(
                                    id=doc_id,
                                    score=score,
                                    metadata=metadata
                                ))
            
            # Ordenar por score
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Limitar a top_k
            results = results[:top_k]
            
            # Crear respuesta
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            response = SearchResponse(
                query=query,
                search_type=SearchType.KEYWORD,
                results=results,
                total_results=len(results),
                processing_time_ms=processing_time,
                filters_applied=filters if filters else [],
                metadata={
                    "keywords_found": keywords,
                    "search_method": "keyword_only"
                }
            )
            
            # Actualizar caché
            if self.config.cache_enabled:
                self._cache[cache_key] = response
                self._clean_cache()
            
            # Actualizar estadísticas
            self._update_stats(processing_time)
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return SearchResponse(
                query=query,
                search_type=SearchType.KEYWORD,
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )
    
    async def hybrid_search(self,
                          query: str,
                          top_k: int = 10,
                          threshold: Optional[float] = None,
                          filters: Optional[List[Dict[str, Any]]] = None) -> SearchResponse:
        """
        Realiza búsqueda híbrida (semántica + keywords).
        
        Args:
            query: Consulta de búsqueda
            top_k: Número máximo de resultados
            threshold: Umbral mínimo de similitud
            filters: Filtros por metadatos
            
        Returns:
            SearchResponse con resultados
        """
        start_time = datetime.now()
        self._stats["total_searches"] += 1
        self._stats["hybrid_searches"] += 1
        
        # Verificar caché
        cache_key = self._create_cache_key(query, SearchType.HYBRID, top_k, threshold, filters)
        if self.config.cache_enabled and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            response = self._cache[cache_key]
            response.metadata["cached"] = True
            return response
        
        try:
            # Ejecutar búsquedas en paralelo
            semantic_future = self.semantic_search(query, top_k * 2, threshold, filters)
            keyword_future = self.keyword_search(query, top_k * 2, filters)
            
            semantic_response, keyword_response = await asyncio.gather(
                semantic_future, keyword_future
            )
            
            # Combinar resultados
            combined_results = self._combine_results(
                semantic_response.results,
                keyword_response.results,
                self.config.semantic_weight,
                self.config.keyword_weight
            )
            
            # Ordenar por score combinado
            combined_results.sort(key=lambda x: x.score, reverse=True)
            
            # Limitar a top_k
            final_results = combined_results[:top_k]
            
            # Crear respuesta
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            response = SearchResponse(
                query=query,
                search_type=SearchType.HYBRID,
                results=final_results,
                total_results=len(final_results),
                processing_time_ms=processing_time,
                query_expansion=semantic_response.query_expansion,
                filters_applied=filters if filters else [],
                metadata={
                    "semantic_results": len(semantic_response.results),
                    "keyword_results": len(keyword_response.results),
                    "semantic_weight": self.config.semantic_weight,
                    "keyword_weight": self.config.keyword_weight,
                    "search_method": "hybrid"
                }
            )
            
            # Actualizar caché
            if self.config.cache_enabled:
                self._cache[cache_key] = response
                self._clean_cache()
            
            # Actualizar estadísticas
            self._update_stats(processing_time)
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return SearchResponse(
                query=query,
                search_type=SearchType.HYBRID,
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Realiza búsqueda según los parámetros de la solicitud.
        
        Args:
            request: Solicitud de búsqueda
            
        Returns:
            SearchResponse con resultados
        """
        if request.search_type == SearchType.SEMANTIC:
            return await self.semantic_search(
                request.query,
                request.top_k,
                request.threshold,
                request.filters
            )
        elif request.search_type == SearchType.KEYWORD:
            return await self.keyword_search(
                request.query,
                request.top_k,
                request.filters
            )
        elif request.search_type == SearchType.HYBRID:
            return await self.hybrid_search(
                request.query,
                request.top_k,
                request.threshold,
                request.filters
            )
        else:
            raise ValueError(f"Unknown search type: {request.search_type}")
    
    def filter_search_results(self,
                            results: List[SearchResult],
                            filters: List[Dict[str, Any]]) -> List[SearchResult]:
        """
        Filtra resultados de búsqueda.
        
        Args:
            results: Resultados a filtrar
            filters: Filtros a aplicar
            
        Returns:
            Lista de resultados filtrados
        """
        if not filters:
            return results
        
        filtered_results = []
        for result in results:
            if self._apply_filters(result.metadata, filters):
                filtered_results.append(result)
        
        return filtered_results
    
    def rank_results(self,
                    results: List[SearchResult],
                    ranking_strategy: str = "score") -> List[SearchResult]:
        """
        Ranquea resultados usando diferentes estrategias.
        
        Args:
            results: Resultados a ranquear
            ranking_strategy: Estrategia de ranking
            
        Returns:
            Resultados ranqueados
        """
        if not results:
            return results
        
        if ranking_strategy == "score":
            # Ordenar por score descendente
            return sorted(results, key=lambda x: x.score, reverse=True)
        
        elif ranking_strategy == "recency":
            # Ordenar por fecha de creación (más reciente primero)
            return sorted(results, key=lambda x: 
                         x.metadata.get("created_at", ""), reverse=True)
        
        elif ranking_strategy == "relevance":
            # Combinación de score y otros factores
            def relevance_score(result):
                base_score = result.score
                # Añadir bonus por recency
                if "created_at" in result.metadata:
                    # Simular bonus por documentos recientes
                    # (implementación simplificada)
                    pass
                return base_score
            
            return sorted(results, key=relevance_score, reverse=True)
        
        else:
            warnings.warn(f"Unknown ranking strategy: {ranking_strategy}, using score")
            return sorted(results, key=lambda x: x.score, reverse=True)
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Expande una consulta con términos relacionados.
        
        Args:
            query: Consulta original
            
        Returns:
            Lista de consultas expandidas
        """
        if not self.config.expansion_enabled:
            return []
        
        try:
            # Generar embedding para la consulta
            query_embedding = await self.generator.generate_text_embedding(
                query,
                model_name=self.config.models["text"]
            )
            
            # Buscar términos similares en el índice
            similar_terms = []
            
            # Esta es una implementación simplificada
            # En producción, se usaría un índice de términos
            for term, doc_ids in self._keyword_index.items():
                if term.lower() != query.lower():
                    # Generar embedding para el término
                    term_embedding = await self.generator.generate_text_embedding(
                        term,
                        model_name=self.config.models["text"]
                    )
                    
                    # Calcular similitud
                    similarity = self._cosine_similarity(query_embedding, term_embedding)
                    if similarity > 0.7:  # Umbral para términos relacionados
                        similar_terms.append((term, similarity))
            
            # Ordenar por similitud
            similar_terms.sort(key=lambda x: x[1], reverse=True)
            
            # Tomar los mejores términos
            expansion_terms = [term for term, _ in similar_terms[:self.config.max_expansion_terms]]
            
            # Crear consultas expandidas
            expanded_queries = []
            for term in expansion_terms:
                expanded_queries.append(f"{query} {term}")
            
            return expanded_queries
            
        except Exception as e:
            warnings.warn(f"Query expansion failed: {str(e)}")
            return []
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de búsqueda.
        
        Returns:
            Dict con métricas
        """
        total_searches = self._stats["total_searches"]
        
        return {
            "total_searches": total_searches,
            "semantic_searches": self._stats["semantic_searches"],
            "keyword_searches": self._stats["keyword_searches"],
            "hybrid_searches": self._stats["hybrid_searches"],
            "cache_hits": self._stats["cache_hits"],
            "cache_hit_rate": (
                self._stats["cache_hits"] / total_searches 
                if total_searches > 0 else 0.0
            ),
            "avg_processing_time_ms": self._stats["avg_processing_time_ms"],
            "keyword_index_size": len(self._keyword_index),
            "cache_size": len(self._cache)
        }
    
    def optimize_search_index(self) -> Dict[str, Any]:
        """
        Optimiza el índice de búsqueda.
        
        Returns:
            Dict con resultados de optimización
        """
        results = {
            "keywords_before": len(self._keyword_index),
            "keywords_after": 0,
            "stopwords_removed": 0,
            "duplicates_removed": 0
        }
        
        # Remover stopwords
        stopwords = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for"}
        to_remove = []
        
        for keyword in self._keyword_index.keys():
            if keyword.lower() in stopwords or len(keyword) < 2:
                to_remove.append(keyword)
                results["stopwords_removed"] += 1
        
        for keyword in to_remove:
            del self._keyword_index[keyword]
        
        # Eliminar duplicados (case-insensitive)
        lower_keywords = {}
        to_remove = []
        
        for keyword in self._keyword_index.keys():
            lower = keyword.lower()
            if lower in lower_keywords:
                # Combinar listas de documentos
                lower_keywords[lower].extend(self._keyword_index[keyword])
                to_remove.append(keyword)
                results["duplicates_removed"] += 1
            else:
                lower_keywords[lower] = self._keyword_index[keyword]
        
        for keyword in to_remove:
            del self._keyword_index[keyword]
        
        # Actualizar índice con keywords en minúsculas
        self._keyword_index = lower_keywords
        
        results["keywords_after"] = len(self._keyword_index)
        
        return results
    
    # Métodos privados
    
    def _build_keyword_index(self) -> None:
        """Construye índice de keywords desde el almacén vectorial."""
        try:
            # Obtener todos los documentos del almacén
            store_stats = self.store.get_stats()
            
            # En un sistema real, iteraríamos sobre todos los documentos
            # Para esta implementación, usamos un enfoque simplificado
            # que asume que los metadatos ya contienen keywords
            
            # Aquí se cargarían keywords de una base de datos o archivo
            # Por ahora, inicializamos vacío
            self._keyword_index = {}
            
        except Exception as e:
            warnings.warn(f"Failed to build keyword index: {str(e)}")
            self._keyword_index = {}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrae keywords de un texto."""
        # Tokenización simple
        tokens = text.lower().split()
        
        # Filtrar tokens muy cortos
        keywords = [token for token in tokens if len(token) > 2]
        
        # Remover duplicados
        return list(set(keywords))
    
    def _get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene metadatos de un documento."""
        # En un sistema real, esto consultaría la base de datos
        # Por ahora, devolvemos un diccionario vacío
        return {"id": doc_id, "title": f"Document {doc_id}"}
    
    def _calculate_keyword_score(self, 
                                keyword: str, 
                                doc_id: str, 
                                query_keywords: List[str]) -> float:
        """Calcula score de keyword para un documento."""
        # Score simple basado en frecuencia
        if keyword in self._keyword_index and doc_id in self._keyword_index[keyword]:
            # Documento contiene el keyword
            base_score = 1.0
        else:
            base_score = 0.0
        
        # Bonus si el keyword está en la consulta
        if keyword in query_keywords:
            base_score += 0.5
        
        return min(base_score, 1.0)
    
    def _apply_filters(self, 
                      metadata: Dict[str, Any], 
                      filters: Optional[List[Dict[str, Any]]]) -> bool:
        """Aplica filtros a metadatos."""
        if not filters:
            return True
        
        for filter_item in filters:
            field = filter_item.get("field")
            operator = filter_item.get("operator")
            value = filter_item.get("value")
            
            if field not in metadata:
                return False
            
            field_value = metadata[field]
            
            if operator == "equals":
                if field_value != value:
                    return False
            elif operator == "not_equals":
                if field_value == value:
                    return False
            elif operator == "greater_than":
                if not (field_value > value):
                    return False
            elif operator == "less_than":
                if not (field_value < value):
                    return False
            elif operator == "in":
                if field_value not in value:
                    return False
            elif operator == "not_in":
                if field_value in value:
                    return False
            elif operator == "contains":
                if value not in str(field_value):
                    return False
            elif operator == "starts_with":
                if not str(field_value).startswith(value):
                    return False
            elif operator == "ends_with":
                if not str(field_value).endswith(value):
                    return False
        
        return True
    
    def _convert_filters(self, filters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convierte filtros al formato del almacén vectorial."""
        if not filters:
            return {}
        
        # ChromaDB usa un formato específico para filtros
        # Esta es una conversión simplificada
        where_filter = {}
        
        for filter_item in filters:
            field = filter_item.get("field")
            operator = filter_item.get("operator")
            value = filter_item.get("value")
            
            if operator == "equals":
                where_filter[field] = value
            elif operator == "in":
                where_filter[field] = {"$in": value}
            elif operator == "not_equals":
                where_filter[field] = {"$ne": value}
            # ... otros operadores
        
        return where_filter
    
    def _combine_results(self,
                        semantic_results: List[SearchResult],
                        keyword_results: List[SearchResult],
                        semantic_weight: float,
                        keyword_weight: float) -> List[SearchResult]:
        """Combina resultados de búsqueda semántica y por keywords."""
        # Mapear resultados por ID
        all_results = {}
        
        # Procesar resultados semánticos
        for result in semantic_results:
            all_results[result.id] = {
                "result": result,
                "semantic_score": result.score,
                "keyword_score": 0.0
            }
        
        # Procesar resultados por keywords
        for result in keyword_results:
            if result.id in all_results:
                # Ya existe, actualizar score de keyword
                all_results[result.id]["keyword_score"] = result.score
            else:
                # Nuevo resultado
                all_results[result.id] = {
                    "result": result,
                    "semantic_score": 0.0,
                    "keyword_score": result.score
                }
        
        # Calcular scores combinados
        combined_results = []
        for data in all_results.values():
            combined_score = (
                data["semantic_score"] * semantic_weight +
                data["keyword_score"] * keyword_weight
            )
            
            result = data["result"]
            result.score = combined_score
            combined_results.append(result)
        
        return combined_results
    
    def _create_cache_key(self,
                         query: str,
                         search_type: SearchType,
                         top_k: int,
                         threshold: Optional[float],
                         filters: Optional[List[Dict[str, Any]]]) -> str:
        """Crea clave única para caché."""
        import hashlib
        import json
        
        cache_data = {
            "query": query,
            "search_type": search_type.value,
            "top_k": top_k,
            "threshold": threshold,
            "filters": filters
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _clean_cache(self) -> None:
        """Limpia caché expirada."""
        current_time = datetime.now().timestamp()
        to_remove = []
        
        for key, response in self._cache.items():
            # Verificar TTL
            created_time = response.metadata.get("created_timestamp", 0)
            if current_time - created_time > self.config.cache_ttl_seconds:
                to_remove.append(key)
        
        for key in to_remove:
            del self._cache[key]
    
    def _update_stats(self, processing_time: float) -> None:
        """Actualiza estadísticas."""
        total_searches = self._stats["total_searches"]
        
        # Actualizar tiempo promedio de procesamiento
        if total_searches > 0:
            current_avg = self._stats["avg_processing_time_ms"]
            self._stats["avg_processing_time_ms"] = (
                (current_avg * (total_searches - 1) + processing_time) / total_searches
            )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud coseno."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = np.sqrt(sum(a * a for a in vec1))
        norm2 = np.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)

# Ejemplo de uso
if __name__ == "__main__":
    async def main():
        # Nota: Este ejemplo requiere dependencias externas
        print("SemanticSearch module loaded successfully")
        print("This module requires EmbeddingGenerator and VectorStore to be initialized")
    
    asyncio.run(main())