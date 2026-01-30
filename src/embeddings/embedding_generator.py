"""
EmbeddingGenerator - Generación de embeddings vectoriales.
Genera embeddings para texto, código y documentos usando modelos pre-entrenados.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from pydantic import BaseModel, Field, validator
import torch
from tqdm import tqdm
from datetime import datetime
import hashlib
import warnings

from .embedding_models import EmbeddingModels, ModelType

class EmbeddingType(Enum):
    """Tipos de embeddings."""
    TEXT = "text"
    CODE = "code"
    DOCUMENT = "document"
    HYBRID = "hybrid"

class EmbeddingRequest(BaseModel):
    """Solicitud de generación de embeddings."""
    content: Union[str, List[str]]
    content_type: EmbeddingType = EmbeddingType.TEXT
    language: Optional[str] = None
    model_name: Optional[str] = None
    normalize: bool = True
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None
    context: Optional[str] = None
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Chunk size must be positive")
        return v
    
    @validator('overlap')
    def validate_overlap(cls, v, values):
        if v is not None and 'chunk_size' in values:
            if values['chunk_size'] is not None and v >= values['chunk_size']:
                raise ValueError("Overlap must be smaller than chunk size")
        return v

class EmbeddingResult(BaseModel):
    """Resultado de generación de embeddings."""
    success: bool
    embeddings: Optional[List[List[float]]] = None
    model_used: Optional[str] = None
    dimensions: Optional[int] = None
    num_embeddings: int = 0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

@dataclass
class GenerationConfig:
    """Configuración del generador de embeddings."""
    default_batch_size: int = 32
    max_batch_size: int = 256
    device: str = "cpu"  # cpu, cuda, auto
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    cache_size: int = 10000
    max_text_length: int = 8192
    num_workers: int = 4
    use_multiprocessing: bool = False
    
    # Modelos por defecto
    default_models: Dict[EmbeddingType, str] = field(
        default_factory=lambda: {
            EmbeddingType.TEXT: "all-MiniLM-L6-v2",
            EmbeddingType.CODE: "microsoft/codebert-base",
            EmbeddingType.DOCUMENT: "all-mpnet-base-v2",
            EmbeddingType.HYBRID: "intfloat/multilingual-e5-large"
        }
    )

class EmbeddingGenerator:
    """
    Generador de embeddings vectoriales.
    
    Características:
    1. Generación de embeddings para texto, código y documentos
    2. Procesamiento por lotes (batch processing)
    3. Normalización y padding automático
    4. Caché inteligente de embeddings
    5. Soporte para múltiples modelos
    """
    
    def __init__(self, 
                 embedding_models: EmbeddingModels,
                 config: Optional[GenerationConfig] = None):
        """
        Inicializa el generador de embeddings.
        
        Args:
            embedding_models: Instancia de EmbeddingModels
            config: Configuración del generador (opcional)
        """
        self.models = embedding_models
        self.config = config or GenerationConfig()
        self._cache: Dict[str, List[float]] = {}
        self._batch_cache: Dict[str, List[List[float]]] = {}
        self._executor = None
        self._stats = {
            "total_embeddings_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time_ms": 0.0,
            "total_processing_time_ms": 0.0
        }
        
    async def generate_text_embedding(self, 
                                     text: str, 
                                     model_name: Optional[str] = None,
                                     normalize: bool = True) -> List[float]:
        """
        Genera embedding para texto.
        
        Args:
            text: Texto a convertir en embedding
            model_name: Nombre del modelo (opcional)
            normalize: Si True, normaliza el embedding
            
        Returns:
            List[float]: Embedding vectorial
            
        Raises:
            ValueError: Si el texto está vacío o es muy largo
            RuntimeError: Si la generación falla
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
            warnings.warn(f"Text truncated to {self.config.max_text_length} characters")
        
        # Verificar caché
        cache_key = self._create_cache_key(text, model_name, "text")
        if self.config.cache_embeddings and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]
        
        self._stats["cache_misses"] += 1
        start_time = datetime.now()
        
        try:
            # Determinar modelo
            model = model_name or self.config.default_models[EmbeddingType.TEXT]
            
            # Cargar modelo si no está cargado
            if not self.models.load_model(model):
                raise RuntimeError(f"Failed to load model {model}")
            
            # Preprocesar texto
            processed_text = self._preprocess_text(text)
            
            # Generar embedding
            embedding = await self._generate_embedding_single(
                processed_text, model, "text"
            )
            
            # Normalizar si es necesario
            if normalize:
                embedding = self._normalize_embedding(embedding)
            
            # Almacenar en caché
            if self.config.cache_embeddings:
                self._cache[cache_key] = embedding
                self._clean_cache()
            
            # Actualizar estadísticas
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(processing_time, 1)
            
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate text embedding: {str(e)}")
    
    async def generate_code_embedding(self,
                                     code: str,
                                     language: str,
                                     context: Optional[str] = None,
                                     model_name: Optional[str] = None) -> List[float]:
        """
        Genera embedding para código.
        
        Args:
            code: Código fuente
            language: Lenguaje de programación
            context: Contexto adicional (opcional)
            model_name: Nombre del modelo (opcional)
            
        Returns:
            List[float]: Embedding vectorial
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")
        
        # Crear contenido combinado
        content = f"Language: {language}\n"
        if context:
            content += f"Context: {context}\n"
        content += f"Code:\n{code}"
        
        # Verificar caché
        cache_key = self._create_cache_key(content, model_name, "code")
        if self.config.cache_embeddings and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]
        
        self._stats["cache_misses"] += 1
        start_time = datetime.now()
        
        try:
            # Determinar modelo
            model = model_name or self.config.default_models[EmbeddingType.CODE]
            
            # Cargar modelo si no está cargado
            if not self.models.load_model(model):
                raise RuntimeError(f"Failed to load model {model}")
            
            # Preprocesar código
            processed_code = self._preprocess_code(code, language)
            
            # Generar embedding
            embedding = await self._generate_embedding_single(
                processed_code, model, "code"
            )
            
            # Normalizar
            embedding = self._normalize_embedding(embedding)
            
            # Almacenar en caché
            if self.config.cache_embeddings:
                self._cache[cache_key] = embedding
                self._clean_cache()
            
            # Actualizar estadísticas
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(processing_time, 1)
            
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate code embedding: {str(e)}")
    
    async def generate_document_embedding(self,
                                         document_path: str,
                                         chunk_size: int = 512,
                                         overlap: int = 50) -> List[List[float]]:
        """
        Genera embeddings para un documento dividiéndolo en chunks.
        
        Args:
            document_path: Ruta al documento
            chunk_size: Tamaño de cada chunk en tokens
            overlap: Solapamiento entre chunks
            
        Returns:
            List[List[float]]: Lista de embeddings por chunk
        """
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Verificar caché
        doc_hash = self._calculate_file_hash(document_path)
        cache_key = f"doc:{doc_hash}:{chunk_size}:{overlap}"
        
        if self.config.cache_embeddings and cache_key in self._batch_cache:
            self._stats["cache_hits"] += 1
            return self._batch_cache[cache_key]
        
        self._stats["cache_misses"] += 1
        start_time = datetime.now()
        
        try:
            # Leer documento
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Dividir en chunks
            chunks = self._chunk_text(content, chunk_size, overlap)
            
            if not chunks:
                raise ValueError("Document produced no chunks")
            
            # Generar embeddings por lotes
            embeddings = await self.batch_generate(
                chunks,
                model_name=self.config.default_models[EmbeddingType.DOCUMENT]
            )
            
            # Almacenar en caché
            if self.config.cache_embeddings:
                self._batch_cache[cache_key] = embeddings
                self._clean_batch_cache()
            
            # Actualizar estadísticas
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(processing_time, len(embeddings))
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate document embeddings: {str(e)}")
    
    async def batch_generate(self,
                            texts: List[str],
                            model_name: Optional[str] = None,
                            batch_size: Optional[int] = None,
                            show_progress: bool = True) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos en lote.
        
        Args:
            texts: Lista de textos
            model_name: Nombre del modelo (opcional)
            batch_size: Tamaño del lote (opcional)
            show_progress: Mostrar barra de progreso
            
        Returns:
            List[List[float]]: Lista de embeddings
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.config.default_batch_size
        batch_size = min(batch_size, self.config.max_batch_size)
        
        # Verificar caché
        cache_key = self._create_batch_cache_key(texts, model_name)
        if self.config.cache_embeddings and cache_key in self._batch_cache:
            self._stats["cache_hits"] += 1
            return self._batch_cache[cache_key]
        
        self._stats["cache_misses"] += 1
        start_time = datetime.now()
        
        try:
            # Determinar modelo
            model = model_name or self.config.default_models[EmbeddingType.TEXT]
            
            # Cargar modelo si no está cargado
            if not self.models.load_model(model):
                raise RuntimeError(f"Failed to load model {model}")
            
            # Preprocesar textos
            processed_texts = [self._preprocess_text(t) for t in texts]
            
            # Generar embeddings en lotes
            all_embeddings = []
            
            progress = range(0, len(processed_texts), batch_size)
            if show_progress:
                progress = tqdm(progress, desc="Generating embeddings")
            
            for i in progress:
                batch = processed_texts[i:i + batch_size]
                
                # Verificar caché individual para cada texto
                batch_embeddings = []
                uncached_indices = []
                uncached_texts = []
                
                for j, text in enumerate(batch):
                    text_cache_key = self._create_cache_key(text, model, "text")
                    if self.config.cache_embeddings and text_cache_key in self._cache:
                        batch_embeddings.append(self._cache[text_cache_key])
                    else:
                        uncached_indices.append(j)
                        uncached_texts.append(text)
                
                # Generar embeddings para textos no cacheados
                if uncached_texts:
                    uncached_embeddings = await self._generate_embeddings_batch(
                        uncached_texts, model, "text"
                    )
                    
                    # Normalizar
                    if self.config.normalize_embeddings:
                        uncached_embeddings = [
                            self._normalize_embedding(e) for e in uncached_embeddings
                        ]
                    
                    # Combinar resultados y actualizar caché
                    result_idx = 0
                    for j in range(len(batch)):
                        if j in uncached_indices:
                            embedding = uncached_embeddings[result_idx]
                            batch_embeddings.insert(j, embedding)
                            
                            # Actualizar caché individual
                            if self.config.cache_embeddings:
                                text_cache_key = self._create_cache_key(
                                    batch[j], model, "text"
                                )
                                self._cache[text_cache_key] = embedding
                            
                            result_idx += 1
                
                all_embeddings.extend(batch_embeddings)
            
            # Almacenar en caché de lotes
            if self.config.cache_embeddings:
                self._batch_cache[cache_key] = all_embeddings
                self._clean_batch_cache()
            
            # Actualizar estadísticas
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(processing_time, len(all_embeddings))
            
            return all_embeddings
            
        except Exception as e:
            raise RuntimeError(f"Batch generation failed: {str(e)}")
    
    def normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Normaliza una lista de embeddings.
        
        Args:
            embeddings: Lista de embeddings
            
        Returns:
            List[List[float]]: Embeddings normalizados
        """
        if not embeddings:
            return []
        
        normalized = []
        for emb in embeddings:
            if emb:  # Verificar que no esté vacío
                normalized.append(self._normalize_embedding(emb))
        
        return normalized
    
    def pad_embeddings(self, 
                      embeddings: List[List[float]], 
                      target_length: int) -> List[List[float]]:
        """
        Aplica padding a embeddings para que tengan la misma longitud.
        
        Args:
            embeddings: Lista de embeddings
            target_length: Longitud objetivo
            
        Returns:
            List[List[float]]: Embeddings con padding
        """
        if not embeddings:
            return []
        
        padded = []
        for emb in embeddings:
            if len(emb) < target_length:
                # Añadir ceros al final
                padded_emb = emb + [0.0] * (target_length - len(emb))
            elif len(emb) > target_length:
                # Truncar
                padded_emb = emb[:target_length]
            else:
                padded_emb = emb
            
            padded.append(padded_emb)
        
        return padded
    
    def truncate_embeddings(self, 
                           embeddings: List[List[float]], 
                           target_length: int) -> List[List[float]]:
        """
        Trunca embeddings a una longitud específica.
        
        Args:
            embeddings: Lista de embeddings
            target_length: Longitud objetivo
            
        Returns:
            List[List[float]]: Embeddings truncados
        """
        if not embeddings:
            return []
        
        return [emb[:target_length] for emb in embeddings]
    
    def average_embeddings(self, 
                          embeddings: List[List[float]], 
                          weights: Optional[List[float]] = None) -> List[float]:
        """
        Calcula el promedio ponderado de embeddings.
        
        Args:
            embeddings: Lista de embeddings
            weights: Pesos para cada embedding (opcional)
            
        Returns:
            List[float]: Embedding promedio
        """
        if not embeddings:
            return []
        
        if weights is None:
            weights = [1.0] * len(embeddings)
        
        if len(weights) != len(embeddings):
            raise ValueError("Weights must have same length as embeddings")
        
        # Calcular promedio ponderado
        weighted_sum = np.zeros(len(embeddings[0]), dtype=np.float32)
        total_weight = 0.0
        
        for emb, weight in zip(embeddings, weights):
            weighted_sum += np.array(emb) * weight
            total_weight += weight
        
        if total_weight == 0:
            return weighted_sum.tolist()
        
        average = (weighted_sum / total_weight).tolist()
        
        # Normalizar
        if self.config.normalize_embeddings:
            average = self._normalize_embedding(average)
        
        return average
    
    def cache_embedding(self, 
                       key: str, 
                       embedding: List[float], 
                       ttl_seconds: int = 3600) -> bool:
        """
        Almacena un embedding en caché.
        
        Args:
            key: Clave única para el embedding
            embedding: Embedding a almacenar
            ttl_seconds: Tiempo de vida en segundos
            
        Returns:
            bool: True si se almacenó exitosamente
        """
        try:
            # Añadir metadata de TTL
            metadata = {
                "embedding": embedding,
                "timestamp": datetime.now().timestamp(),
                "ttl": ttl_seconds
            }
            
            self._cache[key] = metadata
            self._clean_cache()
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to cache embedding: {str(e)}")
            return False
    
    def get_cached_embedding(self, key: str) -> Optional[List[float]]:
        """
        Obtiene un embedding desde caché.
        
        Args:
            key: Clave del embedding
            
        Returns:
            Optional[List[float]]: Embedding si existe y no ha expirado
        """
        if key not in self._cache:
            return None
        
        metadata = self._cache[key]
        
        # Verificar TTL
        current_time = datetime.now().timestamp()
        if current_time - metadata["timestamp"] > metadata["ttl"]:
            # Eliminar expirado
            del self._cache[key]
            return None
        
        return metadata["embedding"]
    
    def invalidate_cache(self, keys: Optional[List[str]] = None) -> int:
        """
        Invalida entradas de caché.
        
        Args:
            keys: Lista de claves a invalidar (None = todas)
            
        Returns:
            int: Número de entradas invalidadas
        """
        if keys is None:
            count = len(self._cache) + len(self._batch_cache)
            self._cache.clear()
            self._batch_cache.clear()
            return count
        
        count = 0
        for key in keys:
            if key in self._cache:
                del self._cache[key]
                count += 1
            if key in self._batch_cache:
                del self._batch_cache[key]
                count += 1
        
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de caché.
        
        Returns:
            Dict con estadísticas
        """
        return {
            "single_embeddings": len(self._cache),
            "batch_embeddings": len(self._batch_cache),
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "hit_rate": (
                self._stats["cache_hits"] / 
                (self._stats["cache_hits"] + self._stats["cache_misses"])
                if (self._stats["cache_hits"] + self._stats["cache_misses"]) > 0
                else 0.0
            ),
            "total_embeddings_generated": self._stats["total_embeddings_generated"],
            "avg_processing_time_ms": self._stats["avg_processing_time_ms"]
        }
    
    def validate_embedding(self, 
                          embedding: List[float], 
                          expected_dimensions: int) -> bool:
        """
        Valida un embedding.
        
        Args:
            embedding: Embedding a validar
            expected_dimensions: Dimensiones esperadas
            
        Returns:
            bool: True si el embedding es válido
        """
        if not embedding:
            return False
        
        if len(embedding) != expected_dimensions:
            return False
        
        # Verificar que no sea todo ceros o NaN
        if all(v == 0 for v in embedding):
            return False
        
        if any(np.isnan(v) for v in embedding):
            return False
        
        return True
    
    def compare_embeddings(self, 
                          embedding1: List[float], 
                          embedding2: List[float], 
                          metric: str = "cosine") -> float:
        """
        Compara dos embeddings usando una métrica específica.
        
        Args:
            embedding1: Primer embedding
            embedding2: Segundo embedding
            metric: Métrica de similitud
            
        Returns:
            float: Score de similitud
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        if len(embedding1) != len(embedding2):
            # Intentar normalizar longitudes
            min_len = min(len(embedding1), len(embedding2))
            embedding1 = embedding1[:min_len]
            embedding2 = embedding2[:min_len]
        
        if metric == "cosine":
            return self._cosine_similarity(embedding1, embedding2)
        elif metric == "euclidean":
            return self._euclidean_similarity(embedding1, embedding2)
        elif metric == "dot":
            return self._dot_product(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_similar_embeddings(self,
                               query: List[float],
                               candidates: List[List[float]],
                               top_k: int = 10,
                               threshold: float = 0.7) -> List[Tuple[int, float]]:
        """
        Encuentra embeddings similares a una consulta.
        
        Args:
            query: Embedding de consulta
            candidates: Lista de embeddings candidatos
            top_k: Número máximo de resultados
            threshold: Umbral mínimo de similitud
            
        Returns:
            List[Tuple[int, float]]: Índices y scores de similitud
        """
        if not query or not candidates:
            return []
        
        similarities = []
        for i, candidate in enumerate(candidates):
            if candidate:  # Ignorar vacíos
                similarity = self.compare_embeddings(query, candidate, "cosine")
                if similarity >= threshold:
                    similarities.append((i, similarity))
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    # Métodos privados
    
    async def _generate_embedding_single(self, 
                                        text: str, 
                                        model_name: str,
                                        content_type: str) -> List[float]:
        """Genera embedding para un solo texto."""
        model = self.models.get_model(model_name)
        tokenizer = self.models.get_tokenizer(model_name)
        
        # Tokenizar
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(model.device)
        
        # Generar embedding
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Usar el último estado oculto y promediar sobre tokens
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Aplicar máscara de atención
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
        
        return embedding.tolist()
    
    async def _generate_embeddings_batch(self,
                                        texts: List[str],
                                        model_name: str,
                                        content_type: str) -> List[List[float]]:
        """Genera embeddings para un lote de textos."""
        model = self.models.get_model(model_name)
        tokenizer = self.models.get_tokenizer(model_name)
        
        embeddings = []
        
        # Procesar en sub-lotes para evitar OOM
        sub_batch_size = min(8, max(1, len(texts) // 4))
        
        for i in range(0, len(texts), sub_batch_size):
            batch_texts = texts[i:i + sub_batch_size]
            
            # Tokenizar lote
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(model.device)
            
            # Generar embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Aplicar máscara de atención y promediar
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                
                embeddings.extend(batch_embeddings.tolist())
        
        return embeddings
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normaliza un embedding a longitud 1."""
        if not embedding:
            return embedding
        
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        
        normalized = (np.array(embedding) / norm).tolist()
        return normalized
    
    def _preprocess_text(self, text: str, language: Optional[str] = None) -> str:
        """Preprocesa texto para generación de embeddings."""
        # Limpiar espacios y caracteres especiales
        text = ' '.join(text.split())
        
        # Manejar casos específicos por lenguaje
        if language == "python":
            # Preservar indentación para código Python
            text = text.strip()
        elif language:
            # Para otros lenguajes, limpiar más agresivamente
            text = text.replace('\t', ' ').replace('\r', '')
        
        return text
    
    def _preprocess_code(self, code: str, language: str) -> str:
        """Preprocesa código para generación de embeddings."""
        # Limpiar código preservando estructura
        lines = code.split('\n')
        processed_lines = []
        
        for line in lines:
            # Eliminar comentarios (simplificado)
            if '#' in line and language == "python":
                line = line.split('#')[0]
            
            # Eliminar espacios en blanco al final
            line = line.rstrip()
            
            if line.strip():  # Mantener líneas no vacías
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Divide texto en chunks con solapamiento."""
        if not text:
            return []
        
        # Tokenizar simple (por palabras para simplificar)
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunk = ' '.join(chunk_words)
                chunks.append(chunk)
            
            # Si llegamos al final
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def _calculate_embedding_dimension(self, model_name: str) -> int:
        """Calcula la dimensión de embedding de un modelo."""
        info = self.models.get_model_info(model_name)
        return info.get("dimensions", 384)
    
    def _validate_batch_size(self, batch_size: int, available_memory: int) -> int:
        """Valida el tamaño de lote según memoria disponible."""
        # Estimación simple: 4 bytes por dimensión por elemento
        avg_dimensions = 384
        memory_per_item = avg_dimensions * 4  # bytes
        
        max_items = available_memory // memory_per_item
        return min(batch_size, max(1, max_items))
    
    def _create_cache_key(self, content: str, model_name: str, content_type: str) -> str:
        """Crea clave única para caché."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{content_type}:{model_name}:{content_hash}"
    
    def _create_batch_cache_key(self, texts: List[str], model_name: str) -> str:
        """Crea clave única para caché de lotes."""
        combined = ':'.join(texts)
        batch_hash = hashlib.md5(combined.encode()).hexdigest()
        return f"batch:{model_name}:{batch_hash}"
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcula hash de un archivo."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _clean_cache(self) -> None:
        """Limpia caché excediendo tamaño máximo."""
        if len(self._cache) > self.config.cache_size:
            # Eliminar las más antiguas
            items = list(self._cache.items())
            items.sort(key=lambda x: x[1].get("timestamp", 0))
            
            # Mantener solo las más recientes
            to_remove = len(items) - self.config.cache_size
            for key, _ in items[:to_remove]:
                del self._cache[key]
    
    def _clean_batch_cache(self) -> None:
        """Limpia caché de lotes excediendo tamaño máximo."""
        max_batch_cache = self.config.cache_size // 10
        if len(self._batch_cache) > max_batch_cache:
            items = list(self._batch_cache.items())
            
            # Eliminar aleatoriamente (estrategia simple)
            import random
            to_remove = len(items) - max_batch_cache
            for _ in range(to_remove):
                key = random.choice(list(self._batch_cache.keys()))
                del self._batch_cache[key]
    
    def _update_stats(self, processing_time: float, num_embeddings: int) -> None:
        """Actualiza estadísticas."""
        self._stats["total_embeddings_generated"] += num_embeddings
        self._stats["total_processing_time_ms"] += processing_time
        
        if self._stats["total_embeddings_generated"] > 0:
            self._stats["avg_processing_time_ms"] = (
                self._stats["total_processing_time_ms"] / 
                self._stats["total_embeddings_generated"]
            )
    
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
        # Convertir distancia a similitud (0-1)
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula producto punto."""
        return sum(a * b for a, b in zip(vec1, vec2))

# Ejemplo de uso
if __name__ == "__main__":
    async def main():
        # Crear gestor de modelos
        models = EmbeddingModels()
        
        # Crear generador
        config = GenerationConfig(device="cpu", cache_embeddings=True)
        generator = EmbeddingGenerator(models, config)
        
        # Generar embedding de texto
        text = "This is an example text for embedding generation."
        embedding = await generator.generate_text_embedding(text)
        print(f"Text embedding generated: {len(embedding)} dimensions")
        
        # Generar embeddings por lote
        texts = [
            "First example text",
            "Second example text",
            "Third example text"
        ]
        embeddings = await generator.batch_generate(texts)
        print(f"Batch embeddings generated: {len(embeddings)}")
        
        # Obtener estadísticas
        stats = generator.get_cache_stats()
        print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        
        # Comparar embeddings
        if len(embeddings) >= 2:
            similarity = generator.compare_embeddings(embeddings[0], embeddings[1])
            print(f"Similarity between first two embeddings: {similarity:.4f}")
    
    asyncio.run(main())