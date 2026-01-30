"""
MemoryHierarchy - Sistema de jerarquía de memoria multi-nivel.
Maneja diferentes niveles de almacenamiento con diferentes características.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import json
import threading
import time
from ..core.exceptions import MemoryException, ValidationError

class MemoryLevel(str, Enum):
    """Niveles en la jerarquía de memoria."""
    L1 = "L1"          # Memoria ultrarrápida (CPU cache, RAM rápida)
    L2 = "L2"          # Memoria rápida (RAM principal)
    L3 = "L3"          # Memoria estándar (RAM, SSD rápido)
    L4 = "L4"          # Memoria de alta capacidad (SSD, NVMe)
    L5 = "L5"          # Almacenamiento masivo (HDD, almacenamiento en red)
    ARCHIVE = "archive" # Archivo frío (backups, almacenamiento a largo plazo)

class AccessPattern(str, Enum):
    """Patrones de acceso a memoria."""
    RANDOM = "random"
    SEQUENTIAL = "sequential"
    TEMPORAL_LOCALITY = "temporal_locality"
    SPATIAL_LOCALITY = "spatial_locality"
    FREQUENT = "frequent"
    INFREQUENT = "infrequent"

class EvictionPolicy(str, Enum):
    """Políticas de desalojo para caché."""
    LRU = "LRU"        # Least Recently Used
    LFU = "LFU"        # Least Frequently Used
    FIFO = "FIFO"      # First In First Out
    LIFO = "LIFO"      # Last In First Out
    RANDOM = "RANDOM"  # Selección aleatoria
    ARC = "ARC"        # Adaptive Replacement Cache

@dataclass
class MemoryItemInfo:
    """Información de un ítem en la jerarquía de memoria."""
    id: str
    level: MemoryLevel
    size_bytes: int
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    promoted_at: Optional[datetime] = None
    demoted_at: Optional[datetime] = None
    access_pattern: AccessPattern = AccessPattern.RANDOM
    metadata: Dict[str, Any] = field(default_factory=dict)

class MemoryLevelConfig(BaseModel):
    """Configuración de un nivel de memoria."""
    level: MemoryLevel
    max_size_bytes: int
    current_size_bytes: int = 0
    access_time_ns: int  # Tiempo de acceso en nanosegundos
    cost_per_gb_per_month: float = 0.0
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    auto_promote: bool = True
    auto_demote: bool = True
    promotion_threshold: int = 10  # Número de accesos para promover
    demotion_threshold_hours: int = 168  # 1 semana sin acceso para degradar

class MemoryHierarchyConfig(BaseModel):
    """Configuración completa de la jerarquía de memoria."""
    levels: List[MemoryLevelConfig]
    monitoring_enabled: bool = True
    auto_optimize: bool = True
    optimize_interval_seconds: int = 300  # Cada 5 minutos
    max_promotion_rate: float = 0.1  # Máximo 10% de items promovidos por ciclo
    enable_compression: bool = True
    compression_level: int = 6

class MemoryHierarchy:
    """
    Sistema de jerarquía de memoria multi-nivel.
    
    Características:
    1. Gestión automática de niveles de almacenamiento
    2. Promoción y degradación basada en patrones de uso
    3. Políticas de desalojo configurables por nivel
    4. Monitoreo de métricas de performance
    5. Optimización automática de ubicación de datos
    6. Balance entre velocidad y capacidad
    """
    
    def __init__(self, config: MemoryHierarchyConfig):
        """
        Inicializa la jerarquía de memoria.
        
        Args:
            config: Configuración de la jerarquía
            
        Raises:
            ValidationError: Si la configuración es inválida
        """
        self.config = config
        self.levels: Dict[MemoryLevel, MemoryLevelConfig] = {
            level_config.level: level_config 
            for level_config in config.levels
        }
        
        # Almacenamiento por nivel
        self.storage: Dict[MemoryLevel, Dict[str, Any]] = {
            level: {} for level in self.levels.keys()
        }
        
        # Índices y metadatos
        self.item_info: Dict[str, MemoryItemInfo] = {}
        self.access_history: Dict[str, List[datetime]] = {}
        self.lock = threading.RLock()
        
        # Métricas
        self.metrics = {
            "total_stores": 0,
            "total_retrieves": 0,
            "promotions": 0,
            "demotions": 0,
            "evictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_access_time_ns": 0,
            "total_data_moved_bytes": 0,
            "optimization_cycles": 0
        }
        
        # Iniciar optimizador si está habilitado
        if config.auto_optimize:
            self._start_optimizer()
    
    def store_in_memory(self, item_id: str, data: Any, 
                       initial_level: Optional[MemoryLevel] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Almacena un ítem en la jerarquía de memoria.
        
        Args:
            item_id: Identificador único del ítem
            data: Datos a almacenar
            initial_level: Nivel inicial (opcional, se auto-selecciona)
            metadata: Metadatos adicionales (opcional)
            
        Returns:
            bool: True si se almacenó exitosamente
            
        Raises:
            ValidationError: Si el ítem ya existe o los datos son inválidos
            MemoryException: Si no hay espacio disponible
        """
        with self.lock:
            try:
                # Validar entrada
                if item_id in self.item_info:
                    raise ValidationError(f"Item already exists: {item_id}")
                
                if data is None:
                    raise ValidationError("Data cannot be None")
                
                # Calcular tamaño
                size_bytes = self._calculate_size(data)
                
                # Determinar nivel inicial
                if initial_level is None:
                    initial_level = self._determine_initial_level(size_bytes, metadata or {})
                
                if initial_level not in self.levels:
                    raise ValidationError(f"Invalid memory level: {initial_level}")
                
                # Verificar capacidad
                level_config = self.levels[initial_level]
                if level_config.current_size_bytes + size_bytes > level_config.max_size_bytes:
                    # Intentar liberar espacio
                    if not self._free_space(initial_level, size_bytes):
                        raise MemoryException(f"Insufficient space in level {initial_level}")
                
                # Comprimir si está habilitado
                if self.config.enable_compression and size_bytes > 1024:
                    data = self._compress_data(data)
                    size_bytes = self._calculate_size(data)
                
                # Almacenar datos
                self.storage[initial_level][item_id] = data
                
                # Actualizar configuración de nivel
                level_config.current_size_bytes += size_bytes
                
                # Crear metadatos
                item_info = MemoryItemInfo(
                    id=item_id,
                    level=initial_level,
                    size_bytes=size_bytes,
                    metadata=metadata or {},
                    access_pattern=self._detect_access_pattern(item_id, data, metadata)
                )
                
                self.item_info[item_id] = item_info
                self.access_history[item_id] = [datetime.now()]
                
                # Actualizar métricas
                self.metrics["total_stores"] += 1
                
                return True
                
            except Exception as e:
                raise MemoryException(f"Failed to store item {item_id}: {e}")
    
    def retrieve_from_memory(self, item_id: str, 
                           update_access: bool = True) -> Optional[Any]:
        """
        Recupera un ítem de la jerarquía de memoria.
        
        Args:
            item_id: Identificador del ítem a recuperar
            update_access: Si True, actualiza contadores de acceso
            
        Returns:
            Los datos recuperados o None si no se encuentra
            
        Raises:
            MemoryException: Si hay error en la recuperación
        """
        with self.lock:
            try:
                # Buscar ítem en todos los niveles
                current_level, data = self._find_item(item_id)
                
                if data is None:
                    self.metrics["cache_misses"] += 1
                    return None
                
                # Actualizar información de acceso
                if update_access:
                    self._update_access_info(item_id, current_level)
                    self.metrics["cache_hits"] += 1
                
                # Descomprimir si es necesario
                if self.config.enable_compression and isinstance(data, dict) and data.get("compressed"):
                    data = self._decompress_data(data)
                
                # Actualizar métricas de tiempo de acceso
                access_time = self.levels[current_level].access_time_ns
                self._update_access_time_metrics(access_time)
                
                self.metrics["total_retrieves"] += 1
                
                return data
                
            except Exception as e:
                raise MemoryException(f"Failed to retrieve item {item_id}: {e}")
    
    def promote_memory(self, item_id: str, target_level: Optional[MemoryLevel] = None) -> bool:
        """
        Promueve un ítem a un nivel superior en la jerarquía.
        
        Args:
            item_id: Identificador del ítem a promover
            target_level: Nivel objetivo (opcional, se auto-selecciona)
            
        Returns:
            bool: True si se promovió exitosamente
            
        Raises:
            ValidationError: Si el ítem no existe o ya está en el nivel más alto
            MemoryException: Si no hay espacio en el nivel objetivo
        """
        with self.lock:
            try:
                # Verificar que el ítem existe
                if item_id not in self.item_info:
                    raise ValidationError(f"Item not found: {item_id}")
                
                item_info = self.item_info[item_id]
                current_level = item_info.level
                
                # Determinar nivel objetivo
                if target_level is None:
                    target_level = self._get_next_higher_level(current_level)
                
                if target_level is None:
                    raise ValidationError(f"Cannot promote from {current_level}: already at highest level")
                
                # Verificar que no sea el mismo nivel
                if target_level == current_level:
                    return True  # Ya está en el nivel objetivo
                
                # Verificar capacidad en nivel objetivo
                target_config = self.levels[target_level]
                if target_config.current_size_bytes + item_info.size_bytes > target_config.max_size_bytes:
                    # Intentar liberar espacio
                    if not self._free_space(target_level, item_info.size_bytes):
                        raise MemoryException(f"Insufficient space in target level {target_level}")
                
                # Mover datos
                data = self.storage[current_level].pop(item_id)
                self.storage[target_level][item_id] = data
                
                # Actualizar tamaños
                current_config = self.levels[current_level]
                current_config.current_size_bytes -= item_info.size_bytes
                target_config.current_size_bytes += item_info.size_bytes
                
                # Actualizar metadatos del ítem
                item_info.level = target_level
                item_info.promoted_at = datetime.now()
                item_info.access_count += 1
                
                # Actualizar métricas
                self.metrics["promotions"] += 1
                self.metrics["total_data_moved_bytes"] += item_info.size_bytes
                
                return True
                
            except Exception as e:
                raise MemoryException(f"Failed to promote item {item_id}: {e}")
    
    def demote_memory(self, item_id: str, target_level: Optional[MemoryLevel] = None) -> bool:
        """
        Degrada un ítem a un nivel inferior en la jerarquía.
        
        Args:
            item_id: Identificador del ítem a degradar
            target_level: Nivel objetivo (opcional, se auto-selecciona)
            
        Returns:
            bool: True si se degradó exitosamente
            
        Raises:
            ValidationError: Si el ítem no existe o ya está en el nivel más bajo
            MemoryException: Si no hay espacio en el nivel objetivo
        """
        with self.lock:
            try:
                # Verificar que el ítem existe
                if item_id not in self.item_info:
                    raise ValidationError(f"Item not found: {item_id}")
                
                item_info = self.item_info[item_id]
                current_level = item_info.level
                
                # Determinar nivel objetivo
                if target_level is None:
                    target_level = self._get_next_lower_level(current_level)
                
                if target_level is None:
                    raise ValidationError(f"Cannot demote from {current_level}: already at lowest level")
                
                # Verificar que no sea el mismo nivel
                if target_level == current_level:
                    return True  # Ya está en el nivel objetivo
                
                # Verificar capacidad en nivel objetivo
                target_config = self.levels[target_level]
                if target_config.current_size_bytes + item_info.size_bytes > target_config.max_size_bytes:
                    # Intentar liberar espacio
                    if not self._free_space(target_level, item_info.size_bytes):
                        # Si no hay espacio, considerar eliminar el ítem
                        if self._should_evict_instead(item_info):
                            self.evict_from_memory(item_id)
                            return True
                        raise MemoryException(f"Insufficient space in target level {target_level}")
                
                # Mover datos
                data = self.storage[current_level].pop(item_id)
                self.storage[target_level][item_id] = data
                
                # Actualizar tamaños
                current_config = self.levels[current_level]
                current_config.current_size_bytes -= item_info.size_bytes
                target_config.current_size_bytes += item_info.size_bytes
                
                # Actualizar metadatos del ítem
                item_info.level = target_level
                item_info.demoted_at = datetime.now()
                
                # Comprimir si se mueve a nivel más lento
                if self._is_slower_level(target_level, current_level) and self.config.enable_compression:
                    compressed_data = self._compress_data(data)
                    self.storage[target_level][item_id] = compressed_data
                    item_info.size_bytes = self._calculate_size(compressed_data)
                
                # Actualizar métricas
                self.metrics["demotions"] += 1
                self.metrics["total_data_moved_bytes"] += item_info.size_bytes
                
                return True
                
            except Exception as e:
                raise MemoryException(f"Failed to demote item {item_id}: {e}")
    
    def evict_from_memory(self, item_id: str, permanent: bool = False) -> bool:
        """
        Expulsa un ítem de la memoria.
        
        Args:
            item_id: Identificador del ítem a expulsar
            permanent: Si True, elimina permanentemente en lugar de archivar
            
        Returns:
            bool: True si se expulsó exitosamente
            
        Raises:
            ValidationError: Si el ítem no existe
        """
        with self.lock:
            try:
                # Verificar que el ítem existe
                if item_id not in self.item_info:
                    raise ValidationError(f"Item not found: {item_id}")
                
                item_info = self.item_info[item_id]
                current_level = item_info.level
                
                # Eliminar de almacenamiento
                if item_id in self.storage[current_level]:
                    del self.storage[current_level][item_id]
                
                # Actualizar tamaño del nivel
                level_config = self.levels[current_level]
                level_config.current_size_bytes -= item_info.size_bytes
                
                # Archivar o eliminar permanentemente
                if not permanent and current_level != MemoryLevel.ARCHIVE:
                    # Mover a archivo
                    archive_data = {
                        "original_data": item_info.metadata.get("summary", "No summary"),
                        "original_size": item_info.size_bytes,
                        "evicted_at": datetime.now().isoformat(),
                        "original_level": current_level.value
                    }
                    
                    if MemoryLevel.ARCHIVE in self.storage:
                        self.storage[MemoryLevel.ARCHIVE][item_id] = archive_data
                        if MemoryLevel.ARCHIVE in self.levels:
                            self.levels[MemoryLevel.ARCHIVE].current_size_bytes += len(str(archive_data))
                
                # Eliminar metadatos
                del self.item_info[item_id]
                if item_id in self.access_history:
                    del self.access_history[item_id]
                
                # Actualizar métricas
                self.metrics["evictions"] += 1
                
                return True
                
            except Exception as e:
                raise MemoryException(f"Failed to evict item {item_id}: {e}")
    
    def optimize_hierarchy(self) -> Dict[str, Any]:
        """
        Optimiza automáticamente la jerarquía de memoria.
        
        Returns:
            Dict con resultados de la optimización
        """
        with self.lock:
            try:
                optimization_results = {
                    "timestamp": datetime.now().isoformat(),
                    "promotions": 0,
                    "demotions": 0,
                    "evictions": 0,
                    "items_analyzed": 0,
                    "performance_improvement": 0.0
                }
                
                start_time = time.time()
                
                # Analizar todos los ítems para optimización
                items_to_optimize = list(self.item_info.items())
                optimization_results["items_analyzed"] = len(items_to_optimize)
                
                # Limitar tasa de promoción
                max_promotions = int(len(items_to_optimize) * self.config.max_promotion_rate)
                promotions_made = 0
                
                for item_id, item_info in items_to_optimize:
                    # Evaluar si debe promoverse
                    if (item_info.access_count >= self.levels[item_info.level].promotion_threshold and
                        promotions_made < max_promotions):
                        
                        if self.promote_memory(item_id):
                            optimization_results["promotions"] += 1
                            promotions_made += 1
                    
                    # Evaluar si debe degradarse
                    elif (item_info.last_accessed < 
                          datetime.now() - timedelta(hours=self.levels[item_info.level].demotion_threshold_hours)):
                        
                        if self.demote_memory(item_id):
                            optimization_results["demotions"] += 1
                    
                    # Evaluar si debe expulsarse
                    elif (item_info.level == MemoryLevel.L5 and 
                          item_info.last_accessed < datetime.now() - timedelta(days=30)):
                        
                        if self.evict_from_memory(item_id):
                            optimization_results["evictions"] += 1
                
                # Calcular mejora de performance
                end_time = time.time()
                optimization_time = end_time - start_time
                
                # Estimación de mejora basada en movimientos
                estimated_improvement = (
                    optimization_results["promotions"] * 0.1 +  # 10% mejora por promoción
                    optimization_results["demotions"] * -0.05   # 5% penalización por degradación
                )
                optimization_results["performance_improvement"] = estimated_improvement
                
                # Actualizar métricas
                self.metrics["optimization_cycles"] += 1
                
                return optimization_results
                
            except Exception as e:
                raise MemoryException(f"Failed to optimize hierarchy: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas de la jerarquía de memoria.
        
        Returns:
            Dict con estadísticas detalladas
        """
        with self.lock:
            try:
                stats = {
                    "timestamp": datetime.now().isoformat(),
                    "levels": {},
                    "overall": {
                        "total_items": len(self.item_info),
                        "total_size_bytes": 0,
                        "total_capacity_bytes": 0,
                        "utilization_percent": 0.0,
                        "average_access_time_ns": self.metrics["average_access_time_ns"],
                        "cache_hit_rate": 0.0
                    },
                    "metrics": self.metrics.copy(),
                    "access_patterns": {},
                    "recommendations": []
                }
                
                # Estadísticas por nivel
                for level_name, level_config in self.levels.items():
                    level_storage = self.storage.get(level_name, {})
                    level_items = [item_id for item_id in self.item_info.values() 
                                  if item_id in level_storage]
                    
                    stats["levels"][level_name.value] = {
                        "item_count": len(level_storage),
                        "current_size_bytes": level_config.current_size_bytes,
                        "max_size_bytes": level_config.max_size_bytes,
                        "utilization_percent": (
                            (level_config.current_size_bytes / level_config.max_size_bytes * 100)
                            if level_config.max_size_bytes > 0 else 0
                        ),
                        "access_time_ns": level_config.access_time_ns,
                        "cost_per_gb_per_month": level_config.cost_per_gb_per_month,
                        "eviction_policy": level_config.eviction_policy.value,
                        "items": [
                            {
                                "id": item_id,
                                "size_bytes": self.item_info[item_id].size_bytes,
                                "access_count": self.item_info[item_id].access_count,
                                "last_accessed": self.item_info[item_id].last_accessed.isoformat(),
                                "access_pattern": self.item_info[item_id].access_pattern.value
                            }
                            for item_id in level_items[:10]  # Primeros 10 ítems
                        ]
                    }
                    
                    # Acumular totales
                    stats["overall"]["total_size_bytes"] += level_config.current_size_bytes
                    stats["overall"]["total_capacity_bytes"] += level_config.max_size_bytes
                
                # Calcular métricas generales
                if stats["overall"]["total_capacity_bytes"] > 0:
                    stats["overall"]["utilization_percent"] = (
                        stats["overall"]["total_size_bytes"] / 
                        stats["overall"]["total_capacity_bytes"] * 100
                    )
                
                if self.metrics["total_retrieves"] > 0:
                    stats["overall"]["cache_hit_rate"] = (
                        self.metrics["cache_hits"] / self.metrics["total_retrieves"] * 100
                    )
                
                # Patrones de acceso
                for item_info in self.item_info.values():
                    pattern = item_info.access_pattern.value
                    stats["access_patterns"][pattern] = stats["access_patterns"].get(pattern, 0) + 1
                
                # Generar recomendaciones
                stats["recommendations"] = self._generate_statistics_recommendations(stats)
                
                return stats
                
            except Exception as e:
                raise MemoryException(f"Failed to get memory stats: {e}")
    
    # Métodos privados de implementación
    
    def _calculate_size(self, data: Any) -> int:
        """Calcula el tamaño en bytes de los datos."""
        if isinstance(data, (str, bytes)):
            return len(data)
        elif isinstance(data, dict):
            return len(json.dumps(data))
        else:
            return len(str(data))
    
    def _determine_initial_level(self, size_bytes: int, 
                                metadata: Dict[str, Any]) -> MemoryLevel:
        """Determina el nivel inicial para almacenar un ítem."""
        # Por defecto, usar L3 (memoria estándar)
        default_level = MemoryLevel.L3
        
        # Ajustar basado en tamaño
        if size_bytes < 1024:  # < 1KB
            return MemoryLevel.L1
        elif size_bytes < 10240:  # < 10KB
            return MemoryLevel.L2
        elif size_bytes < 1048576:  # < 1MB
            return default_level
        elif size_bytes < 10485760:  # < 10MB
            return MemoryLevel.L4
        else:  # >= 10MB
            return MemoryLevel.L5
        
        # Nota: El archivo se usa solo para expulsión, no para almacenamiento inicial
    
    def _detect_access_pattern(self, item_id: str, data: Any, 
                              metadata: Dict[str, Any]) -> AccessPattern:
        """Detecta el patrón de acceso probable para un ítem."""
        # Por defecto, acceso aleatorio
        pattern = AccessPattern.RANDOM
        
        # Basado en metadatos
        if metadata.get("frequent_access", False):
            pattern = AccessPattern.FREQUENT
        elif metadata.get("sequential", False):
            pattern = AccessPattern.SEQUENTIAL
        
        # Basado en tipo de datos
        if isinstance(data, dict):
            if "temporal" in str(data).lower():
                pattern = AccessPattern.TEMPORAL_LOCALITY
            elif "spatial" in str(data).lower():
                pattern = AccessPattern.SPATIAL_LOCALITY
        
        return pattern
    
    def _find_item(self, item_id: str) -> Tuple[Optional[MemoryLevel], Optional[Any]]:
        """Encuentra un ítem en todos los niveles de almacenamiento."""
        for level, storage in self.storage.items():
            if item_id in storage:
                return level, storage[item_id]
        
        return None, None
    
    def _update_access_info(self, item_id: str, level: MemoryLevel) -> None:
        """Actualiza la información de acceso de un ítem."""
        if item_id in self.item_info:
            item_info = self.item_info[item_id]
            item_info.access_count += 1
            item_info.last_accessed = datetime.now()
            
            # Registrar en historial
            self.access_history.setdefault(item_id, []).append(datetime.now())
            
            # Auto-promoción si está habilitada
            if (self.levels[level].auto_promote and 
                item_info.access_count >= self.levels[level].promotion_threshold):
                
                # Promover de forma asíncrona
                threading.Thread(
                    target=self.promote_memory,
                    args=(item_id,),
                    daemon=True
                ).start()
    
    def _update_access_time_metrics(self, access_time_ns: int) -> None:
        """Actualiza métricas de tiempo de acceso."""
        total_retrieves = self.metrics["total_retrieves"]
        current_avg = self.metrics["average_access_time_ns"]
        
        if total_retrieves > 0:
            self.metrics["average_access_time_ns"] = (
                (current_avg * (total_retrieves - 1) + access_time_ns) / total_retrieves
            )
        else:
            self.metrics["average_access_time_ns"] = access_time_ns
    
    def _get_next_higher_level(self, current_level: MemoryLevel) -> Optional[MemoryLevel]:
        """Obtiene el siguiente nivel más alto en la jerarquía."""
        level_order = [MemoryLevel.L5, MemoryLevel.L4, MemoryLevel.L3, 
                      MemoryLevel.L2, MemoryLevel.L1]
        
        try:
            current_index = level_order.index(current_level)
            if current_index > 0:
                return level_order[current_index - 1]
        except ValueError:
            pass
        
        return None
    
    def _get_next_lower_level(self, current_level: MemoryLevel) -> Optional[MemoryLevel]:
        """Obtiene el siguiente nivel más bajo en la jerarquía."""
        level_order = [MemoryLevel.L5, MemoryLevel.L4, MemoryLevel.L3, 
                      MemoryLevel.L2, MemoryLevel.L1]
        
        try:
            current_index = level_order.index(current_level)
            if current_index < len(level_order) - 1:
                return level_order[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _free_space(self, level: MemoryLevel, required_bytes: int) -> bool:
        """Libera espacio en un nivel usando la política de desalojo."""
        level_config = self.levels[level]
        storage = self.storage[level]
        
        if not storage:
            return False
        
        # Seleccionar víctimas basado en política de desalojo
        victims = self._select_victims(level, storage, required_bytes)
        
        if not victims:
            return False
        
        # Expulsar víctimas
        freed_bytes = 0
        for victim_id in victims:
            if victim_id in self.item_info:
                victim_size = self.item_info[victim_id].size_bytes
                
                # Degradar a nivel inferior o expulsar
                lower_level = self._get_next_lower_level(level)
                if lower_level and self._has_space(lower_level, victim_size):
                    self.demote_memory(victim_id, lower_level)
                else:
                    self.evict_from_memory(victim_id)
                
                freed_bytes += victim_size
                
                if freed_bytes >= required_bytes:
                    break
        
        return freed_bytes >= required_bytes
    
    def _select_victims(self, level: MemoryLevel, storage: Dict[str, Any], 
                       required_bytes: int) -> List[str]:
        """Selecciona víctimas para desalojo basado en la política."""
        policy = self.levels[level].eviction_policy
        items = list(storage.keys())
        
        if not items:
            return []
        
        if policy == EvictionPolicy.LRU:
            # Least Recently Used
            sorted_items = sorted(
                items,
                key=lambda x: self.item_info.get(x, MemoryItemInfo(id=x, level=level, size_bytes=0)).last_accessed
            )
            return sorted_items[:self._calculate_victim_count(items, required_bytes)]
        
        elif policy == EvictionPolicy.LFU:
            # Least Frequently Used
            sorted_items = sorted(
                items,
                key=lambda x: self.item_info.get(x, MemoryItemInfo(id=x, level=level, size_bytes=0)).access_count
            )
            return sorted_items[:self._calculate_victim_count(items, required_bytes)]
        
        elif policy == EvictionPolicy.FIFO:
            # First In First Out
            sorted_items = sorted(
                items,
                key=lambda x: self.item_info.get(x, MemoryItemInfo(id=x, level=level, size_bytes=0)).created_at
            )
            return sorted_items[:self._calculate_victim_count(items, required_bytes)]
        
        elif policy == EvictionPolicy.LIFO:
            # Last In First Out
            sorted_items = sorted(
                items,
                key=lambda x: self.item_info.get(x, MemoryItemInfo(id=x, level=level, size_bytes=0)).created_at,
                reverse=True
            )
            return sorted_items[:self._calculate_victim_count(items, required_bytes)]
        
        else:  # RANDOM o por defecto
            import random
            victim_count = self._calculate_victim_count(items, required_bytes)
            return random.sample(items, min(victim_count, len(items)))
    
    def _calculate_victim_count(self, items: List[str], required_bytes: int) -> int:
        """Calcula cuántas víctimas se necesitan para liberar espacio."""
        total_size = 0
        count = 0
        
        for item_id in items:
            if item_id in self.item_info:
                total_size += self.item_info[item_id].size_bytes
                count += 1
                
                if total_size >= required_bytes:
                    break
        
        return count
    
    def _has_space(self, level: MemoryLevel, size_bytes: int) -> bool:
        """Verifica si un nivel tiene espacio para un ítem."""
        level_config = self.levels.get(level)
        if not level_config:
            return False
        
        return level_config.current_size_bytes + size_bytes <= level_config.max_size_bytes
    
    def _should_evict_instead(self, item_info: MemoryItemInfo) -> bool:
        """Determina si es mejor expulsar un ítem en lugar de degradarlo."""
        # Expulsar si es muy grande y poco accedido
        if (item_info.size_bytes > 10485760 and  # > 10MB
            item_info.access_count < 3 and
            item_info.last_accessed < datetime.now() - timedelta(days=7)):
            return True
        
        # Expulsar si está en nivel bajo y muy viejo
        if (item_info.level in [MemoryLevel.L4, MemoryLevel.L5] and
            item_info.last_accessed < datetime.now() - timedelta(days=30)):
            return True
        
        return False
    
    def _is_slower_level(self, level1: MemoryLevel, level2: MemoryLevel) -> bool:
        """Determina si level1 es más lento que level2."""
        level_order = [MemoryLevel.L1, MemoryLevel.L2, MemoryLevel.L3, 
                      MemoryLevel.L4, MemoryLevel.L5, MemoryLevel.ARCHIVE]
        
        try:
            idx1 = level_order.index(level1)
            idx2 = level_order.index(level2)
            return idx1 > idx2  # Índice mayor = más lento
        except ValueError:
            return False
    
    def _compress_data(self, data: Any) -> Dict[str, Any]:
        """Comprime datos usando el algoritmo configurado."""
        import zlib
        
        try:
            serialized = json.dumps(data).encode('utf-8')
            compressed = zlib.compress(serialized, level=self.config.compression_level)
            
            return {
                "compressed": True,
                "algorithm": "zlib",
                "level": self.config.compression_level,
                "original_size": len(serialized),
                "data": compressed
            }
        except:
            # Si falla la compresión, devolver datos originales
            return data
    
    def _decompress_data(self, compressed_data: Dict[str, Any]) -> Any:
        """Descomprime datos previamente comprimidos."""
        import zlib
        
        try:
            if (isinstance(compressed_data, dict) and 
                compressed_data.get("compressed") and 
                compressed_data.get("algorithm") == "zlib"):
                
                decompressed = zlib.decompress(compressed_data["data"])
                return json.loads(decompressed.decode('utf-8'))
        except:
            pass
        
        # Si falla la descompresión, devolver datos originales
        return compressed_data
    
    def _generate_statistics_recommendations(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera recomendaciones basadas en estadísticas."""
        recommendations = []
        
        # Recomendación de utilización
        utilization = stats["overall"]["utilization_percent"]
        if utilization > 90:
            recommendations.append({
                "type": "capacity",
                "priority": "high",
                "message": f"High memory utilization: {utilization:.1f}%",
                "suggestion": "Consider increasing capacity or implementing more aggressive eviction policies"
            })
        elif utilization < 30:
            recommendations.append({
                "type": "efficiency",
                "priority": "low",
                "message": f"Low memory utilization: {utilization:.1f}%",
                "suggestion": "Consider reducing memory allocation to save costs"
            })
        
        # Recomendación de hit rate
        hit_rate = stats["overall"]["cache_hit_rate"]
        if hit_rate < 70:
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "message": f"Low cache hit rate: {hit_rate:.1f}%",
                "suggestion": "Review promotion thresholds and consider caching more frequently accessed items"
            })
        
        # Recomendación de distribución
        level_stats = stats["levels"]
        l1_utilization = level_stats.get("L1", {}).get("utilization_percent", 0)
        l5_utilization = level_stats.get("L5", {}).get("utilization_percent", 0)
        
        if l1_utilization < 50 and l5_utilization > 80:
            recommendations.append({
                "type": "distribution",
                "priority": "medium",
                "message": "Poor memory distribution: L1 underutilized while L5 is full",
                "suggestion": "Adjust promotion/demotion thresholds to move more items to faster memory"
            })
        
        return recommendations
    
    def _start_optimizer(self) -> None:
        """Inicia el optimizador automático en segundo plano."""
        def optimizer_loop():
            while True:
                time.sleep(self.config.optimize_interval_seconds)
                try:
                    self.optimize_hierarchy()
                except Exception as e:
                    # Log error but don't crash
                    print(f"Optimizer error: {e}")
        
        optimizer_thread = threading.Thread(target=optimizer_loop, daemon=True)
        optimizer_thread.start()