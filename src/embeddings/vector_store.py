"""
VectorStore - Almacenamiento y recuperación de vectores.
Gestiona bases de datos vectoriales para embeddings con búsqueda eficiente.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, validator
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pickle
import warnings
from datetime import datetime
import json

class VectorStoreType(Enum):
    """Tipos de almacenes vectoriales."""
    CHROMA = "chroma"
    FAISS = "faiss"
    ANN = "ann"  # Approximate Nearest Neighbor
    SIMPLE = "simple"  # Para desarrollo/testing

class DistanceMetric(Enum):
    """Métricas de distancia para búsqueda vectorial."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"

class VectorMetadata(BaseModel):
    """Metadatos para vectores almacenados."""
    entity_id: str
    entity_type: str
    content_hash: str
    model_name: str
    embedding_type: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class SearchResult(BaseModel):
    """Resultado de búsqueda vectorial."""
    id: str
    score: float
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

@dataclass
class VectorStoreConfig:
    """Configuración del almacén vectorial."""
    store_type: VectorStoreType = VectorStoreType.CHROMA
    persist_directory: str = "./data/vector_store"
    collection_name: str = "project_embeddings"
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    normalize_embeddings: bool = True
    auto_persist: bool = True
    persist_interval: int = 300  # segundos
    
    # Configuración específica de ChromaDB
    chroma_settings: Dict[str, Any] = field(default_factory=lambda: {
        "allow_reset": True,
        "anonymized_telemetry": False,
        "is_persistent": True
    })
    
    # Configuración de índice
    index_config: Dict[str, Any] = field(default_factory=lambda: {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,
        "hnsw:search_ef": 50,
        "hnsw:M": 16
    })

class VectorStore:
    """
    Almacén vectorial para embeddings.
    
    Responsabilidades:
    1. Almacenar vectores con metadatos
    2. Realizar búsquedas por similitud
    3. Gestionar colecciones de vectores
    4. Proporcionar persistencia y recuperación
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Inicializa el almacén vectorial.
        
        Args:
            config: Configuración del almacén (opcional)
        """
        self.config = config or VectorStoreConfig()
        self._client = None
        self._collection = None
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, VectorMetadata] = {}
        self._last_persist = datetime.now()
        
        # Crear directorio de persistencia
        os.makedirs(self.config.persist_directory, exist_ok=True)
        
        # Inicializar según tipo
        self._initialize_store()
    
    def add_vectors(self,
                   vectors: List[List[float]],
                   ids: List[str],
                   metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Añade vectores al almacén.
        
        Args:
            vectors: Lista de vectores
            ids: Lista de IDs únicos
            metadatas: Lista de metadatos
            
        Returns:
            Dict con resultados de la operación
            
        Raises:
            ValueError: Si las listas tienen longitudes diferentes
        """
        if len(vectors) != len(ids) or len(vectors) != len(metadatas):
            raise ValueError("All input lists must have the same length")
        
        results = {
            "added": 0,
            "updated": 0,
            "errors": []
        }
        
        for vector, id_, metadata in zip(vectors, ids, metadatas):
            try:
                # Normalizar si está configurado
                if self.config.normalize_embeddings:
                    vector = self._normalize_vector(vector)
                
                # Crear metadata completa
                full_metadata = VectorMetadata(
                    entity_id=id_,
                    entity_type=metadata.get("entity_type", "unknown"),
                    content_hash=metadata.get("content_hash", ""),
                    model_name=metadata.get("model_name", "unknown"),
                    embedding_type=metadata.get("embedding_type", "text"),
                    metadata=metadata
                )
                
                # Añadir según tipo de almacén
                if self.config.store_type == VectorStoreType.CHROMA:
                    self._add_to_chroma(vector, id_, full_metadata)
                else:
                    self._add_to_simple(vector, id_, full_metadata)
                
                results["added"] += 1
                
            except Exception as e:
                results["errors"].append(f"{id_}: {str(e)}")
        
        # Persistir si es necesario
        if self.config.auto_persist:
            self._auto_persist()
        
        return results
    
    def search_similar(self,
                      query_vector: List[float],
                      top_k: int = 10,
                      threshold: Optional[float] = None,
                      filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Busca vectores similares al vector de consulta.
        
        Args:
            query_vector: Vector de consulta
            top_k: Número máximo de resultados
            threshold: Umbral mínimo de similitud
            filters: Filtros por metadatos
            
        Returns:
            Lista de resultados ordenados por similitud
        """
        if not query_vector:
            return []
        
        # Normalizar vector de consulta
        if self.config.normalize_embeddings:
            query_vector = self._normalize_vector(query_vector)
        
        # Realizar búsqueda según tipo de almacén
        if self.config.store_type == VectorStoreType.CHROMA:
            results = self._search_chroma(query_vector, top_k, filters)
        else:
            results = self._search_simple(query_vector, top_k)
        
        # Aplicar umbral si se especifica
        if threshold is not None:
            results = [r for r in results if r.score >= threshold]
        
        return results[:top_k]
    
    def update_vector(self,
                     vector_id: str,
                     new_vector: Optional[List[float]] = None,
                     new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Actualiza un vector existente.
        
        Args:
            vector_id: ID del vector a actualizar
            new_vector: Nuevo vector (opcional)
            new_metadata: Nuevos metadatos (opcional)
            
        Returns:
            bool: True si la actualización fue exitosa
        """
        # Verificar que existe
        if self.config.store_type == VectorStoreType.CHROMA:
            exists = self._exists_in_chroma(vector_id)
        else:
            exists = vector_id in self._vectors
        
        if not exists:
            return False
        
        try:
            # Actualizar vector si se proporciona
            if new_vector is not None:
                if self.config.normalize_embeddings:
                    new_vector = self._normalize_vector(new_vector)
                
                if self.config.store_type == VectorStoreType.CHROMA:
                    self._update_chroma(vector_id, new_vector, new_metadata)
                else:
                    self._vectors[vector_id] = new_vector
            
            # Actualizar metadatos si se proporcionan
            if new_metadata is not None and vector_id in self._metadata:
                current = self._metadata[vector_id]
                current.metadata.update(new_metadata)
                current.updated_at = datetime.now()
            
            # Persistir si es necesario
            if self.config.auto_persist:
                self._auto_persist()
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to update vector {vector_id}: {str(e)}")
            return False
    
    def delete_vector(self, vector_id: str) -> bool:
        """
        Elimina un vector del almacén.
        
        Args:
            vector_id: ID del vector a eliminar
            
        Returns:
            bool: True si se eliminó exitosamente
        """
        try:
            if self.config.store_type == VectorStoreType.CHROMA:
                if self._collection:
                    self._collection.delete(ids=[vector_id])
            else:
                self._vectors.pop(vector_id, None)
                self._metadata.pop(vector_id, None)
            
            # Persistir si es necesario
            if self.config.auto_persist:
                self._auto_persist()
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to delete vector {vector_id}: {str(e)}")
            return False
    
    def create_index(self, index_type: str = "hnsw", **kwargs) -> bool:
        """
        Crea un índice para búsqueda eficiente.
        
        Args:
            index_type: Tipo de índice (hnsw, ivf, etc.)
            **kwargs: Parámetros específicos del índice
            
        Returns:
            bool: True si el índice se creó exitosamente
        """
        try:
            if self.config.store_type == VectorStoreType.CHROMA:
                # ChromaDB crea índices automáticamente
                return True
            
            elif self.config.store_type == VectorStoreType.SIMPLE:
                # Para almacén simple, no se necesita índice
                return True
            
            # Aquí se podría añadir soporte para otros almacenes como FAISS
            else:
                warnings.warn(f"Index creation not supported for {self.config.store_type}")
                return False
                
        except Exception as e:
            warnings.warn(f"Failed to create index: {str(e)}")
            return False
    
    def optimize_store(self) -> Dict[str, Any]:
        """
        Optimiza el almacén para mejor performance.
        
        Returns:
            Dict con resultados de optimización
        """
        results = {
            "vectors_before": len(self._vectors),
            "vectors_after": 0,
            "duplicates_removed": 0,
            "metadata_cleaned": 0,
            "time_taken_ms": 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Eliminar duplicados
            if self.config.store_type == VectorStoreType.SIMPLE:
                results.update(self._optimize_simple_store())
            
            # Reconstruir índice si es necesario
            if self.config.store_type == VectorStoreType.CHROMA:
                results.update(self._optimize_chroma_store())
            
            # Persistir cambios
            if self.config.auto_persist:
                self._persist()
            
            results["vectors_after"] = len(self._vectors)
            results["time_taken_ms"] = (datetime.now() - start_time).total_seconds() * 1000
            
            return results
            
        except Exception as e:
            warnings.warn(f"Failed to optimize store: {str(e)}")
            results["error"] = str(e)
            return results
    
    def export_vectors(self, 
                      format: str = "json",
                      file_path: Optional[str] = None) -> Union[Dict, str]:
        """
        Exporta vectores a diferentes formatos.
        
        Args:
            format: Formato de exportación (json, pickle, csv)
            file_path: Ruta del archivo de salida (opcional)
            
        Returns:
            Datos exportados o ruta del archivo
        """
        export_data = {
            "vectors": self._vectors,
            "metadata": {
                id_: metadata.dict() 
                for id_, metadata in self._metadata.items()
            },
            "config": self.config.__dict__,
            "exported_at": datetime.now().isoformat()
        }
        
        if format == "json":
            data = json.dumps(export_data, default=str, indent=2)
        elif format == "pickle":
            data = pickle.dumps(export_data)
        elif format == "csv":
            data = self._export_to_csv(export_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Guardar en archivo si se especifica
        if file_path:
            mode = "w" if format == "json" or format == "csv" else "wb"
            with open(file_path, mode) as f:
                f.write(data if isinstance(data, str) else data.decode())
            return file_path
        
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del almacén.
        
        Returns:
            Dict con estadísticas
        """
        if self.config.store_type == VectorStoreType.CHROMA and self._collection:
            collection_stats = self._collection.count()
        else:
            collection_stats = len(self._vectors)
        
        return {
            "store_type": self.config.store_type.value,
            "total_vectors": collection_stats,
            "collection_name": self.config.collection_name,
            "distance_metric": self.config.distance_metric.value,
            "normalize_embeddings": self.config.normalize_embeddings,
            "persist_directory": self.config.persist_directory,
            "last_persist": self._last_persist.isoformat() if self._last_persist else None
        }
    
    def clear_store(self) -> bool:
        """
        Limpia todos los vectores del almacén.
        
        Returns:
            bool: True si se limpió exitosamente
        """
        try:
            if self.config.store_type == VectorStoreType.CHROMA and self._collection:
                self._collection.delete(where={})
            else:
                self._vectors.clear()
                self._metadata.clear()
            
            # Persistir cambios
            self._persist()
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to clear store: {str(e)}")
            return False
    
    # Métodos privados
    
    def _initialize_store(self) -> None:
        """Inicializa el almacén según el tipo."""
        try:
            if self.config.store_type == VectorStoreType.CHROMA:
                self._initialize_chroma()
            elif self.config.store_type == VectorStoreType.SIMPLE:
                self._initialize_simple()
            else:
                raise ValueError(f"Unsupported store type: {self.config.store_type}")
                
        except Exception as e:
            warnings.warn(f"Failed to initialize store: {str(e)}")
            # Fallback a almacén simple
            self.config.store_type = VectorStoreType.SIMPLE
            self._initialize_simple()
    
    def _initialize_chroma(self) -> None:
        """Inicializa ChromaDB."""
        try:
            # Crear cliente
            self._client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.config.persist_directory,
                **self.config.chroma_settings
            ))
            
            # Crear o obtener colección
            try:
                self._collection = self._client.get_collection(
                    name=self.config.collection_name
                )
            except:
                # Crear nueva colección
                self._collection = self._client.create_collection(
                    name=self.config.collection_name,
                    metadata={
                        "hnsw:space": self.config.distance_metric.value,
                        **self.config.index_config
                    }
                )
            
            # Cargar vectores y metadatos en memoria para acceso rápido
            self._load_chroma_data()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {str(e)}")
    
    def _initialize_simple(self) -> None:
        """Inicializa almacén simple en memoria."""
        # Cargar datos persistentes si existen
        data_file = Path(self.config.persist_directory) / "simple_store.pkl"
        if data_file.exists():
            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    self._vectors = data.get("vectors", {})
                    self._metadata = data.get("metadata", {})
            except:
                self._vectors = {}
                self._metadata = {}
        else:
            self._vectors = {}
            self._metadata = {}
    
    def _add_to_chroma(self, 
                      vector: List[float], 
                      id_: str, 
                      metadata: VectorMetadata) -> None:
        """Añade vector a ChromaDB."""
        if not self._collection:
            raise RuntimeError("ChromaDB collection not initialized")
        
        # Convertir metadatos para Chroma
        chroma_metadata = {
            "entity_id": metadata.entity_id,
            "entity_type": metadata.entity_type,
            "content_hash": metadata.content_hash,
            "model_name": metadata.model_name,
            "embedding_type": metadata.embedding_type,
            "created_at": metadata.created_at.isoformat(),
            **metadata.metadata
        }
        
        # Añadir o actualizar
        self._collection.upsert(
            embeddings=[vector],
            ids=[id_],
            metadatas=[chroma_metadata]
        )
        
        # Actualizar caché en memoria
        self._vectors[id_] = vector
        self._metadata[id_] = metadata
    
    def _add_to_simple(self, 
                      vector: List[float], 
                      id_: str, 
                      metadata: VectorMetadata) -> None:
        """Añade vector a almacén simple."""
        self._vectors[id_] = vector
        self._metadata[id_] = metadata
    
    def _search_chroma(self, 
                      query_vector: List[float], 
                      top_k: int,
                      filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Busca en ChromaDB."""
        if not self._collection:
            return []
        
        try:
            # Construir filtro where
            where_filter = None
            if filters:
                where_filter = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_filter[key] = {"$in": value}
                    else:
                        where_filter[key] = value
            
            # Realizar búsqueda
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_filter,
                include=["embeddings", "metadatas", "distances"]
            )
            
            # Convertir a formato estándar
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, id_ in enumerate(results["ids"][0]):
                    score = 1.0 - results["distances"][0][i]  # Convertir distancia a similitud
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    embedding = results["embeddings"][0][i] if results["embeddings"] else None
                    
                    search_results.append(SearchResult(
                        id=id_,
                        score=score,
                        embedding=embedding,
                        metadata=metadata
                    ))
            
            return search_results
            
        except Exception as e:
            warnings.warn(f"ChromaDB search failed: {str(e)}")
            return []
    
    def _search_simple(self, 
                      query_vector: List[float], 
                      top_k: int) -> List[SearchResult]:
        """Busca en almacén simple."""
        if not self._vectors:
            return []
        
        # Calcular similitud con todos los vectores
        similarities = []
        for id_, vector in self._vectors.items():
            if vector:
                similarity = self._calculate_similarity(query_vector, vector)
                metadata = self._metadata.get(id_)
                similarities.append((id_, similarity, vector, metadata))
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Convertir a resultados
        search_results = []
        for id_, similarity, vector, metadata in similarities[:top_k]:
            meta_dict = metadata.dict() if metadata else {}
            search_results.append(SearchResult(
                id=id_,
                score=similarity,
                embedding=vector,
                metadata=meta_dict
            ))
        
        return search_results
    
    def _update_chroma(self, 
                      vector_id: str, 
                      new_vector: List[float],
                      new_metadata: Optional[Dict[str, Any]]) -> None:
        """Actualiza vector en ChromaDB."""
        if not self._collection:
            return
        
        metadata = None
        if new_metadata:
            # Obtener metadatos existentes
            try:
                existing = self._collection.get(ids=[vector_id], include=["metadatas"])
                if existing["metadatas"]:
                    metadata = existing["metadatas"][0]
                    metadata.update(new_metadata)
            except:
                metadata = new_metadata
        
        self._collection.update(
            embeddings=[new_vector],
            ids=[vector_id],
            metadatas=[metadata] if metadata else None
        )
    
    def _exists_in_chroma(self, vector_id: str) -> bool:
        """Verifica si un vector existe en ChromaDB."""
        if not self._collection:
            return False
        
        try:
            result = self._collection.get(ids=[vector_id])
            return bool(result["ids"])
        except:
            return False
    
    def _load_chroma_data(self) -> None:
        """Carga datos de ChromaDB en memoria para acceso rápido."""
        if not self._collection:
            return
        
        try:
            # Obtener todos los vectores
            results = self._collection.get(include=["embeddings", "metadatas"])
            
            for i, id_ in enumerate(results["ids"]):
                # Vector
                if results["embeddings"] and i < len(results["embeddings"]):
                    self._vectors[id_] = results["embeddings"][i]
                
                # Metadatos
                if results["metadatas"] and i < len(results["metadatas"]):
                    metadata_dict = results["metadatas"][i]
                    
                    # Convertir string de fecha a datetime
                    created_at = datetime.now()
                    if "created_at" in metadata_dict:
                        try:
                            created_at = datetime.fromisoformat(metadata_dict["created_at"])
                        except:
                            pass
                    
                    metadata = VectorMetadata(
                        entity_id=metadata_dict.get("entity_id", id_),
                        entity_type=metadata_dict.get("entity_type", "unknown"),
                        content_hash=metadata_dict.get("content_hash", ""),
                        model_name=metadata_dict.get("model_name", "unknown"),
                        embedding_type=metadata_dict.get("embedding_type", "text"),
                        created_at=created_at,
                        metadata=metadata_dict
                    )
                    
                    self._metadata[id_] = metadata
                    
        except Exception as e:
            warnings.warn(f"Failed to load ChromaDB data: {str(e)}")
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """Normaliza un vector a longitud 1."""
        if not vector:
            return vector
        
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        
        return (np.array(vector) / norm).tolist()
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud según métrica configurada."""
        if self.config.distance_metric == DistanceMetric.COSINE:
            return self._cosine_similarity(vec1, vec2)
        elif self.config.distance_metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_similarity(vec1, vec2)
        elif self.config.distance_metric == DistanceMetric.DOT:
            return self._dot_product(vec1, vec2)
        else:
            return self._cosine_similarity(vec1, vec2)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud coseno."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = np.sqrt(sum(a * a for a in vec1))
        norm2 = np.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def _euclidean_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud euclidiana (inversa de distancia)."""
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
        return 1.0 / (1.0 + distance)
    
    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula producto punto."""
        return sum(a * b for a, b in zip(vec1, vec2))
    
    def _optimize_simple_store(self) -> Dict[str, Any]:
        """Optimiza almacén simple."""
        results = {
            "duplicates_removed": 0,
            "metadata_cleaned": 0
        }
        
        # Eliminar vectores duplicados (mismo hash de contenido)
        content_hashes = {}
        to_remove = []
        
        for id_, metadata in self._metadata.items():
            if id_ in self._vectors:
                content_hash = metadata.content_hash
                if content_hash and content_hash in content_hashes:
                    # Vector duplicado
                    to_remove.append(id_)
                    results["duplicates_removed"] += 1
                else:
                    content_hashes[content_hash] = id_
        
        # Eliminar duplicados
        for id_ in to_remove:
            self._vectors.pop(id_, None)
            self._metadata.pop(id_, None)
        
        return results
    
    def _optimize_chroma_store(self) -> Dict[str, Any]:
        """Optimiza almacén ChromaDB."""
        results = {
            "duplicates_removed": 0,
            "metadata_cleaned": 0
        }
        
        # ChromaDB maneja la optimización internamente
        # Podríamos añadir limpieza específica aquí
        return results
    
    def _export_to_csv(self, data: Dict) -> str:
        """Exporta datos a formato CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Escribir encabezados
        writer.writerow(["id", "vector", "entity_type", "model_name", "created_at"])
        
        # Escribir datos
        for id_, vector in data["vectors"].items():
            metadata = data["metadata"].get(id_, {})
            vector_str = ",".join(str(v) for v in vector)
            writer.writerow([
                id_,
                vector_str,
                metadata.get("entity_type", ""),
                metadata.get("model_name", ""),
                metadata.get("created_at", "")
            ])
        
        return output.getvalue()
    
    def _auto_persist(self) -> None:
        """Persiste automáticamente si ha pasado suficiente tiempo."""
        current_time = datetime.now()
        time_diff = (current_time - self._last_persist).total_seconds()
        
        if time_diff >= self.config.persist_interval:
            self._persist()
            self._last_persist = current_time
    
    def _persist(self) -> None:
        """Persiste datos según tipo de almacén."""
        try:
            if self.config.store_type == VectorStoreType.CHROMA:
                # ChromaDB persiste automáticamente
                pass
            elif self.config.store_type == VectorStoreType.SIMPLE:
                # Persistir en archivo pickle
                data = {
                    "vectors": self._vectors,
                    "metadata": self._metadata
                }
                
                data_file = Path(self.config.persist_directory) / "simple_store.pkl"
                with open(data_file, 'wb') as f:
                    pickle.dump(data, f)
            
            self._last_persist = datetime.now()
            
        except Exception as e:
            warnings.warn(f"Failed to persist store: {str(e)}")

# Ejemplo de uso
if __name__ == "__main__":
    # Crear configuración
    config = VectorStoreConfig(
        store_type=VectorStoreType.SIMPLE,
        persist_directory="./test_data/vector_store"
    )
    
    # Crear almacén
    store = VectorStore(config)
    
    # Añadir algunos vectores
    vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.1, 0.2, 0.3]
    ]
    
    ids = ["vec1", "vec2", "vec3"]
    metadatas = [
        {"entity_type": "text", "content": "Hello world"},
        {"entity_type": "code", "language": "python"},
        {"entity_type": "document", "title": "Test doc"}
    ]
    
    result = store.add_vectors(vectors, ids, metadatas)
    print(f"Added vectors: {result}")
    
    # Buscar similares
    query = [0.1, 0.2, 0.3, 0.4]
    similar = store.search_similar(query, top_k=2)
    print(f"Similar vectors: {len(similar)} found")
    
    for r in similar:
        print(f"  - ID: {r.id}, Score: {r.score:.4f}")
    
    # Obtener estadísticas
    stats = store.get_stats()
    print(f"Store stats: {stats}")
    
    # Exportar
    export_data = store.export_vectors(format="json")
    print(f"Exported data length: {len(export_data) if isinstance(export_data, str) else 'binary'}")