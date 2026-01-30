"""
EpisodicMemory - Memoria episódica para eventos y experiencias específicas.
Maneja el almacenamiento y recuperación de eventos con contexto temporal.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import json
from ..core.exceptions import MemoryException, ValidationError

class EpisodeType(str, Enum):
    """Tipos de episodios en memoria episódica."""
    ANALYSIS = "analysis"
    QUERY = "query"
    LEARNING = "learning"
    ERROR = "error"
    FEEDBACK = "feedback"
    COLLABORATION = "collaboration"

class EpisodeRelevance(str, Enum):
    """Niveles de relevancia de episodios."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Episode:
    """Representación de un episodio en memoria."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EpisodeType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    relevance: EpisodeRelevance = EpisodeRelevance.MEDIUM
    duration_seconds: float = 0.0
    linked_episodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el episodio a diccionario."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "relevance": self.relevance.value,
            "duration_seconds": self.duration_seconds,
            "linked_episodes": self.linked_episodes,
            "metadata": self.metadata
        }

class EpisodicMemoryConfig(BaseModel):
    """Configuración de memoria episódica."""
    max_episodes: int = 10000
    retention_days: int = 90
    relevance_decay_rate: float = 0.1  # Tasa de decaimiento por día
    compression_enabled: bool = True
    compression_threshold: int = 1000  # Comprimir cuando se exceda este número
    indexing_enabled: bool = True
    index_fields: List[str] = Field(default_factory=lambda: ["type", "relevance", "timestamp"])

class EpisodicMemory:
    """
    Sistema de memoria episódica para almacenar eventos y experiencias específicas.
    
    Características:
    1. Almacenamiento temporal de eventos con contexto completo
    2. Vinculación entre episodios relacionados
    3. Consolidación de episodios antiguos
    4. Búsqueda por contexto temporal y semántico
    5. Decaimiento de relevancia automático
    """
    
    def __init__(self, config: Optional[EpisodicMemoryConfig] = None):
        """
        Inicializa la memoria episódica.
        
        Args:
            config: Configuración de la memoria (opcional)
        """
        self.config = config or EpisodicMemoryConfig()
        self.episodes: Dict[str, Episode] = {}
        self.temporal_index: Dict[str, List[str]] = {}
        self.type_index: Dict[EpisodeType, List[str]] = {ep_type: [] for ep_type in EpisodeType}
        self.relevance_index: Dict[EpisodeRelevance, List[str]] = {rel: [] for rel in EpisodeRelevance}
        self.compressed_episodes: Dict[str, Dict] = {}
        
        # Métricas
        self.metrics = {
            "total_episodes_recorded": 0,
            "active_episodes": 0,
            "compressed_episodes": 0,
            "average_episode_size_bytes": 0,
            "recall_hits": 0,
            "recall_misses": 0
        }
        
    def record_episode(self, episode_type: EpisodeType, content: Dict[str, Any], 
                      context: Optional[Dict[str, Any]] = None,
                      relevance: EpisodeRelevance = EpisodeRelevance.MEDIUM,
                      duration_seconds: float = 0.0,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Registra un nuevo episodio en la memoria.
        
        Args:
            episode_type: Tipo de episodio
            content: Contenido del episodio
            context: Contexto adicional (opcional)
            relevance: Relevancia del episodio
            duration_seconds: Duración en segundos (opcional)
            metadata: Metadatos adicionales (opcional)
            
        Returns:
            str: ID del episodio registrado
            
        Raises:
            ValidationError: Si el contenido está vacío o es inválido
        """
        try:
            # Validar entrada
            if not content:
                raise ValidationError("Episode content cannot be empty")
            
            # Crear episodio
            episode = Episode(
                type=episode_type,
                content=content,
                context=context or {},
                relevance=relevance,
                duration_seconds=duration_seconds,
                metadata=metadata or {}
            )
            
            # Almacenar episodio
            self.episodes[episode.id] = episode
            
            # Actualizar índices
            self._update_indices(episode)
            
            # Actualizar métricas
            self.metrics["total_episodes_recorded"] += 1
            self.metrics["active_episodes"] += 1
            
            # Comprimir si es necesario
            if self.config.compression_enabled and len(self.episodes) > self.config.compression_threshold:
                self._compress_old_episodes()
            
            return episode.id
            
        except Exception as e:
            raise MemoryException(f"Failed to record episode: {e}")
    
    def recall_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        Recupera un episodio específico por su ID.
        
        Args:
            episode_id: ID del episodio a recuperar
            
        Returns:
            Dict con el episodio o None si no se encuentra
            
        Raises:
            MemoryException: Si hay error en la recuperación
        """
        try:
            # Buscar en episodios activos
            if episode_id in self.episodes:
                self.metrics["recall_hits"] += 1
                return self.episodes[episode_id].to_dict()
            
            # Buscar en episodios comprimidos
            if episode_id in self.compressed_episodes:
                self.metrics["recall_hits"] += 1
                return self.compressed_episodes[episode_id]
            
            self.metrics["recall_misses"] += 1
            return None
            
        except Exception as e:
            raise MemoryException(f"Failed to recall episode {episode_id}: {e}")
    
    def link_episodes(self, source_episode_id: str, target_episode_id: str, 
                     link_type: str = "related") -> bool:
        """
        Establece un vínculo entre dos episodios.
        
        Args:
            source_episode_id: ID del episodio origen
            target_episode_id: ID del episodio destino
            link_type: Tipo de vínculo (opcional)
            
        Returns:
            bool: True si el vínculo se estableció exitosamente
            
        Raises:
            ValidationError: Si algún episodio no existe
        """
        try:
            # Verificar que ambos episodios existan
            source_episode = self.episodes.get(source_episode_id)
            target_episode = self.episodes.get(target_episode_id)
            
            if not source_episode:
                raise ValidationError(f"Source episode not found: {source_episode_id}")
            if not target_episode:
                raise ValidationError(f"Target episode not found: {target_episode_id}")
            
            # Establecer vínculo bidireccional
            if target_episode_id not in source_episode.linked_episodes:
                source_episode.linked_episodes.append(target_episode_id)
            
            if source_episode_id not in target_episode.linked_episodes:
                target_episode.linked_episodes.append(source_episode_id)
            
            # Actualizar metadatos con tipo de vínculo
            source_episode.metadata.setdefault("links", {})[target_episode_id] = {
                "type": link_type,
                "timestamp": datetime.now().isoformat()
            }
            
            target_episode.metadata.setdefault("links", {})[source_episode_id] = {
                "type": link_type,
                "timestamp": datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            raise MemoryException(f"Failed to link episodes: {e}")
    
    def consolidate_episodes(self, age_threshold_days: int = 7) -> Dict[str, Any]:
        """
        Consolida episodios antiguos agrupándolos por similitud.
        
        Args:
            age_threshold_days: Edad mínima para consolidación
            
        Returns:
            Dict con estadísticas de consolidación
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=age_threshold_days)
            old_episodes = [
                ep for ep in self.episodes.values() 
                if ep.timestamp < cutoff_date
            ]
            
            if not old_episodes:
                return {"consolidated": 0, "skipped": 0, "compressed": 0}
            
            # Agrupar por tipo y contexto similar
            grouped = self._group_similar_episodes(old_episodes)
            
            consolidated_count = 0
            for group_key, episodes in grouped.items():
                if len(episodes) > 1:
                    # Consolidar episodios similares
                    consolidated = self._create_consolidated_episode(episodes)
                    
                    # Eliminar episodios originales
                    for episode in episodes:
                        del self.episodes[episode.id]
                        self._remove_from_indices(episode.id)
                    
                    # Almacenar episodio consolidado
                    self.compressed_episodes[consolidated["id"]] = consolidated
                    consolidated_count += 1
            
            # Actualizar métricas
            self.metrics["active_episodes"] = len(self.episodes)
            self.metrics["compressed_episodes"] = len(self.compressed_episodes)
            
            return {
                "consolidated": consolidated_count,
                "remaining_active": len(self.episodes),
                "total_compressed": len(self.compressed_episodes),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise MemoryException(f"Failed to consolidate episodes: {e}")
    
    def forget_episode(self, episode_id: str, permanent: bool = False) -> bool:
        """
        Olvida (elimina) un episodio específico.
        
        Args:
            episode_id: ID del episodio a olvidar
            permanent: Si True, elimina permanentemente (no archiva)
            
        Returns:
            bool: True si se olvidó exitosamente
            
        Raises:
            ValidationError: Si el episodio no existe
        """
        try:
            # Verificar que el episodio existe
            if episode_id not in self.episodes and episode_id not in self.compressed_episodes:
                raise ValidationError(f"Episode not found: {episode_id}")
            
            # Eliminar de memoria activa
            if episode_id in self.episodes:
                episode = self.episodes[episode_id]
                del self.episodes[episode_id]
                self._remove_from_indices(episode_id)
                
                # Archivar si no es permanente
                if not permanent:
                    self.compressed_episodes[episode_id] = episode.to_dict()
            
            # Eliminar de memoria comprimida
            elif episode_id in self.compressed_episodes:
                if permanent:
                    del self.compressed_episodes[episode_id]
            
            # Actualizar métricas
            self.metrics["active_episodes"] = len(self.episodes)
            self.metrics["compressed_episodes"] = len(self.compressed_episodes)
            
            return True
            
        except Exception as e:
            raise MemoryException(f"Failed to forget episode {episode_id}: {e}")
    
    def search_episodes(self, query: Optional[Dict[str, Any]] = None,
                       episode_type: Optional[EpisodeType] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       relevance: Optional[EpisodeRelevance] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Busca episodios que coincidan con los criterios.
        
        Args:
            query: Consulta de contenido (búsqueda semántica básica)
            episode_type: Filtrar por tipo de episodio
            start_date: Fecha de inicio para filtro temporal
            end_date: Fecha de fin para filtro temporal
            relevance: Filtrar por relevancia
            limit: Límite de resultados
            
        Returns:
            Lista de episodios que coinciden
        """
        try:
            results = []
            
            # Buscar en episodios activos
            for episode in self.episodes.values():
                if self._matches_search_criteria(episode, query, episode_type, 
                                               start_date, end_date, relevance):
                    results.append(episode.to_dict())
            
            # Buscar en episodios comprimidos
            for episode_data in self.compressed_episodes.values():
                episode = Episode(**episode_data)
                if self._matches_search_criteria(episode, query, episode_type,
                                               start_date, end_date, relevance):
                    results.append(episode_data)
            
            # Ordenar por timestamp (más recientes primero)
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            raise MemoryException(f"Failed to search episodes: {e}")
    
    def get_episodic_timeline(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Obtiene una línea temporal de episodios recientes.
        
        Args:
            hours_back: Horas hacia atrás para incluir
            
        Returns:
            Dict con línea temporal estructurada
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            timeline = {
                "start_time": cutoff_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "episodes_by_type": {},
                "episodes_by_hour": {},
                "statistics": {
                    "total_episodes": 0,
                    "episodes_per_type": {},
                    "average_duration": 0,
                    "busiest_hour": None
                }
            }
            
            # Filtrar episodios recientes
            recent_episodes = [
                ep for ep in self.episodes.values() 
                if ep.timestamp >= cutoff_time
            ]
            
            total_duration = 0
            hour_counts = {}
            
            for episode in recent_episodes:
                # Agrupar por tipo
                ep_type = episode.type.value
                timeline["episodes_by_type"].setdefault(ep_type, []).append(
                    episode.to_dict()
                )
                
                # Agrupar por hora
                episode_hour = episode.timestamp.strftime("%Y-%m-%d %H:00")
                timeline["episodes_by_hour"].setdefault(episode_hour, []).append(
                    episode.id
                )
                
                # Contar por hora
                hour_counts[episode_hour] = hour_counts.get(episode_hour, 0) + 1
                
                # Acumular duración
                total_duration += episode.duration_seconds
            
            # Calcular estadísticas
            timeline["statistics"]["total_episodes"] = len(recent_episodes)
            
            for ep_type, episodes in timeline["episodes_by_type"].items():
                timeline["statistics"]["episodes_per_type"][ep_type] = len(episodes)
            
            if recent_episodes:
                timeline["statistics"]["average_duration"] = (
                    total_duration / len(recent_episodes)
                )
            
            if hour_counts:
                busiest_hour = max(hour_counts.items(), key=lambda x: x[1])
                timeline["statistics"]["busiest_hour"] = {
                    "hour": busiest_hour[0],
                    "episode_count": busiest_hour[1]
                }
            
            return timeline
            
        except Exception as e:
            raise MemoryException(f"Failed to get episodic timeline: {e}")
    
    # Métodos privados de implementación
    
    def _update_indices(self, episode: Episode) -> None:
        """Actualiza todos los índices con un nuevo episodio."""
        # Índice temporal (por día)
        date_key = episode.timestamp.strftime("%Y-%m-%d")
        self.temporal_index.setdefault(date_key, []).append(episode.id)
        
        # Índice por tipo
        self.type_index[episode.type].append(episode.id)
        
        # Índice por relevancia
        self.relevance_index[episode.relevance].append(episode.id)
    
    def _remove_from_indices(self, episode_id: str) -> None:
        """Elimina un episodio de todos los índices."""
        # Buscar en índices temporales
        for date_key, episodes in self.temporal_index.items():
            if episode_id in episodes:
                episodes.remove(episode_id)
                if not episodes:
                    del self.temporal_index[date_key]
        
        # Buscar en índices por tipo
        for ep_type, episodes in self.type_index.items():
            if episode_id in episodes:
                episodes.remove(episode_id)
        
        # Buscar en índices por relevancia
        for relevance, episodes in self.relevance_index.items():
            if episode_id in episodes:
                episodes.remove(episode_id)
    
    def _compress_old_episodes(self) -> None:
        """Comprime episodios antiguos para ahorrar memoria."""
        try:
            # Encontrar episodios más antiguos que excedan el límite
            if len(self.episodes) <= self.config.max_episodes:
                return
            
            episodes_to_compress = len(self.episodes) - self.config.max_episodes
            
            # Ordenar episodios por timestamp (más antiguos primero)
            sorted_episodes = sorted(
                self.episodes.values(),
                key=lambda x: x.timestamp
            )
            
            # Comprimir los más antiguos
            for i in range(min(episodes_to_compress, len(sorted_episodes))):
                episode = sorted_episodes[i]
                
                # Mover a comprimidos
                self.compressed_episodes[episode.id] = episode.to_dict()
                
                # Eliminar de activos
                del self.episodes[episode.id]
                self._remove_from_indices(episode.id)
            
            # Actualizar métricas
            self.metrics["active_episodes"] = len(self.episodes)
            self.metrics["compressed_episodes"] = len(self.compressed_episodes)
            
        except Exception as e:
            raise MemoryException(f"Failed to compress old episodes: {e}")
    
    def _group_similar_episodes(self, episodes: List[Episode]) -> Dict[str, List[Episode]]:
        """Agrupa episodios similares para consolidación."""
        groups = {}
        
        for episode in episodes:
            # Crear clave de agrupación basada en tipo y contexto principal
            context_key = json.dumps(episode.context.get("main_context", {}), sort_keys=True)
            group_key = f"{episode.type.value}_{context_key}"
            
            groups.setdefault(group_key, []).append(episode)
        
        return groups
    
    def _create_consolidated_episode(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Crea un episodio consolidado a partir de múltiples episodios similares."""
        # Usar el episodio más reciente como base
        base_episode = max(episodes, key=lambda x: x.timestamp)
        
        # Consolidar contenido
        consolidated_content = {
            "original_count": len(episodes),
            "time_range": {
                "start": min(ep.timestamp for ep in episodes).isoformat(),
                "end": max(ep.timestamp for ep in episodes).isoformat()
            },
            "summary": f"Consolidated {len(episodes)} episodes of type {base_episode.type.value}",
            "original_episodes": [ep.id for ep in episodes],
            "combined_content": [ep.content for ep in episodes]
        }
        
        # Crear episodio consolidado
        consolidated_episode = Episode(
            type=base_episode.type,
            content=consolidated_content,
            context=base_episode.context,
            relevance=max(ep.relevance for ep in episodes),
            duration_seconds=sum(ep.duration_seconds for ep in episodes),
            metadata={
                "consolidated": True,
                "original_episode_count": len(episodes),
                "consolidation_timestamp": datetime.now().isoformat()
            }
        )
        
        return consolidated_episode.to_dict()
    
    def _matches_search_criteria(self, episode: Episode, query: Optional[Dict],
                               episode_type: Optional[EpisodeType],
                               start_date: Optional[datetime],
                               end_date: Optional[datetime],
                               relevance: Optional[EpisodeRelevance]) -> bool:
        """Verifica si un episodio coincide con los criterios de búsqueda."""
        # Filtrar por tipo
        if episode_type and episode.type != episode_type:
            return False
        
        # Filtrar por fecha
        if start_date and episode.timestamp < start_date:
            return False
        if end_date and episode.timestamp > end_date:
            return False
        
        # Filtrar por relevancia
        if relevance and episode.relevance != relevance:
            return False
        
        # Filtrar por consulta de contenido (búsqueda simple)
        if query:
            for key, value in query.items():
                if key in episode.content:
                    if isinstance(value, str) and isinstance(episode.content[key], str):
                        if value.lower() not in episode.content[key].lower():
                            return False
                    elif episode.content[key] != value:
                        return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de la memoria episódica."""
        return {
            **self.metrics,
            "config": self.config.dict(),
            "timestamp": datetime.now().isoformat()
        }