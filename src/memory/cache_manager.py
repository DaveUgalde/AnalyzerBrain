"""
CacheManager - Gestión de caché multi-nivel con políticas inteligentes.
Sistema de caché para almacenamiento temporal de resultados costosos.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from datetime import datetime, timedelta
import json
from collections import OrderedDict, defaultdict
import uuid
import hashlib
import pickle
from pathlib import Path

from ..core.exceptions import MemoryException, ValidationError

class CachePolicy(Enum):
    """Políticas de caché."""
    LRU = "LRU"      # Least Recently Used
    LFU = "LFU"      # Least Frequently Used
    FIFO = "FIFO"    # First In First Out
    LIFO = "LIFO"    # Last In First Out
    MRU = "MRU"      # Most Recently Used
    ARC = "ARC"      # Adaptive Replacement Cache
    TTL = "TTL"      # Time To Live

class CacheLevel(Enum):
    """Niveles de caché."""
    MEMORY = "memory"    # Caché en memoria RAM
    DISK = "disk"        # Caché en disco
    DISTRIBUTED = "distributed"  # Caché distribuido

@dataclass
class CacheItem:
    """Elemento en caché."""
    key: str
    value: Any
    level: CacheLevel = CacheLevel.MEMORY
    policy: CachePolicy = CachePolicy.LRU
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None
    expiration_time: Optional[datetime] = None
    
    # Para políticas específicas
    frequency: int = 1  # Para LFU
    priority: int = 1   # Para prioridades personalizadas
    
    def __post_init__(self):
        """Inicialización posterior para calcular tiempo de expiración."""
        if self.ttl_seconds is not None:
            self.expiration_time = self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def access(self) -> None:
        """Registra un acceso al item."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        self.frequency += 1
    
    def is_expired(self) -> bool:
        """Verifica si el item ha expirado."""
        if self.expiration_time is None:
            return False
        return datetime.now() > self.expiration_time
    
    def calculate_score(self, policy: CachePolicy) -> float:
        """
        Calcula score para políticas de reemplazo.
        Score más bajo = más probable de ser expulsado.
        """
        current_time = datetime.now()
        
        if policy == CachePolicy.LRU:
            # Menos recientemente usado = score más bajo
            time_since_access = (current_time - self.last_accessed).total_seconds()
            return -time_since_access  # Negativo para que menor = más antiguo
            
        elif policy == CachePolicy.LFU:
            # Menos frecuentemente usado = score más bajo
            return -self.frequency
            
        elif policy == CachePolicy.FIFO:
            # Más antiguo = score más bajo
            time_since_creation = (current_time - self.created_at).total_seconds()
            return -time_since_creation
            
        elif policy == CachePolicy.LIFO:
            # Más reciente = score más bajo (se expulsará primero)
            time_since_creation = (current_time - self.created_at).total_seconds()
            return time_since_creation  # Positivo, menor = más reciente
            
        elif policy == CachePolicy.MRU:
            # Más recientemente usado = score más bajo
            time_since_access = (current_time - self.last_accessed).total_seconds()
            return time_since_access  # Positivo, menor = más reciente
            
        elif policy == CachePolicy.TTL:
            # Cercano a expirar = score más bajo
            if self.expiration_time:
                time_to_expire = (self.expiration_time - current_time).total_seconds()
                return -time_to_expire  # Negativo, menor = más cercano a expirar
            else:
                return float('inf')  # Nunca expira = score infinito
        
        else:
            # Por defecto: LRU
            time_since_access = (current_time - self.last_accessed).total_seconds()
            return -time_since_access
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            "key": self.key,
            "value": pickle.dumps(self.value) if self.value is not None else None,
            "level": self.level.value,
            "policy": self.policy.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "ttl_seconds": self.ttl_seconds,
            "expiration_time": self.expiration_time.isoformat() if self.expiration_time else None,
            "frequency": self.frequency,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheItem':
        """Crea desde diccionario."""
        item = cls(
            key=data["key"],
            value=pickle.loads(data["value"]) if data["value"] else None,
            level=CacheLevel(data["level"]),
            policy=CachePolicy(data["policy"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            size_bytes=data["size_bytes"],
            ttl_seconds=data["ttl_seconds"],
            frequency=data["frequency"],
            priority=data["priority"]
        )
        
        if data["expiration_time"]:
            item.expiration_time = datetime.fromisoformat(data["expiration_time"])
        
        return item

class CacheManager:
    """
    Gestor de caché multi-nivel con políticas inteligentes.
    
    Características:
    1. Soporte para múltiples niveles de caché (memoria, disco, distribuido)
    2. Múltiples políticas de expulsión (LRU, LFU, FIFO, LIFO, MRU, ARC, TTL)
    3. Time-to-live (TTL) y expiración automática
    4. Pre-caching y pre-carga predictiva
    5. Estadísticas y monitoreo en tiempo real
    6. Invalidation por patrones y dependencias
    7. Compresión y optimización de espacio
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el gestor de caché.
        
        Args:
            config: Configuración del caché
        """
        self.config = config or self._default_config()
        
        # Cachés por nivel
        self.caches: Dict[CacheLevel, Dict[str, CacheItem]] = {
            CacheLevel.MEMORY: OrderedDict(),
            CacheLevel.DISK: OrderedDict(),
            CacheLevel.DISTRIBUTED: OrderedDict()
        }
        
        # Límites por nivel
        self.level_limits = {
            CacheLevel.MEMORY: self.config["level_limits"]["memory"],
            CacheLevel.DISK: self.config["level_limits"]["disk"],
            CacheLevel.DISTRIBUTED: self.config["level_limits"]["distributed"]
        }
        
        # Política actual
        self.policy = CachePolicy(self.config["default_policy"])
        
        # Para política ARC
        if self.policy == CachePolicy.ARC:
            self._init_arc()
        
        # Índices para búsqueda rápida
        self.indices = {
            "by_pattern": defaultdict(list),
            "by_dependency": defaultdict(list),
            "expiration_queue": []  # Heap para expiraciones
        }
        
        # Estadísticas
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "expirations": 0,
            "total_size_bytes": 0,
            "hit_rate": 0.0,
            "avg_access_time_ms": 0.0
        }
        
        # Bloqueo para concurrencia
        self._lock = asyncio.Lock()
        
        # Iniciar tareas de mantenimiento
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._stats_task = asyncio.create_task(self._periodic_stats_update())
        
        # Cargar caché de disco si existe
        self._load_disk_cache()
    
    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto del caché."""
        return {
            "level_limits": {
                "memory": 1000,     # Máximo 1000 items en memoria
                "disk": 10000,      # Máximo 10000 items en disco
                "distributed": 0    # Ilimitado por defecto
            },
            "default_policy": "LRU",
            "default_ttl": 3600,    # 1 hora por defecto
            "compression_enabled": True,
            "compression_threshold": 1024,  # Comprimir items > 1KB
            "prefetch_enabled": True,
            "prefetch_threshold": 0.7,      # Pre-cache si probabilidad > 70%
            "invalidation_enabled": True,
            "cleanup_interval": 60,         # Limpieza cada 60 segundos
            "stats_interval": 300,          # Actualizar stats cada 5 minutos
            "disk_cache_path": "./data/cache",
            "enable_monitoring": True
        }
    
    def _init_arc(self) -> None:
        """Inicializa estructuras para política ARC."""
        self.t1 = OrderedDict()  # T1: Items recientemente accedidos una vez
        self.t2 = OrderedDict()  # T2: Items frecuentemente accedidos
        self.b1 = OrderedDict()  # B1: Historial de expulsiones de T1
        self.b2 = OrderedDict()  # B2: Historial de expulsiones de T2
        self.p = 0  # Parámetro de adaptación (0 <= p <= c)
    
    async def get_from_cache(
        self, 
        key: str,
        level: Optional[CacheLevel] = None,
        update_access: bool = True
    ) -> Optional[Any]:
        """
        Obtiene un valor del caché.
        
        Args:
            key: Clave a buscar
            level: Nivel específico (None = buscar en todos)
            update_access: Si True, actualiza contadores de acceso
            
        Returns:
            Valor almacenado o None si no existe
        """
        start_time = time.time()
        
        async with self._lock:
            # Determinar niveles a buscar
            if level is not None:
                levels_to_search = [level]
            else:
                # Buscar de más rápido a más lento
                levels_to_search = [
                    CacheLevel.MEMORY,
                    CacheLevel.DISK,
                    CacheLevel.DISTRIBUTED
                ]
            
            for current_level in levels_to_search:
                cache = self.caches[current_level]
                
                if key in cache:
                    item = cache[key]
                    
                    # Verificar expiración
                    if item.is_expired():
                        await self._remove_item(key, current_level)
                        self.stats["expirations"] += 1
                        self.stats["misses"] += 1
                        continue
                    
                    # Actualizar acceso
                    if update_access:
                        item.access()
                        
                        # Mover al frente (para LRU en OrderedDict)
                        if current_level == CacheLevel.MEMORY:
                            cache.move_to_end(key)
                        
                        # Actualizar política ARC si está activa
                        if self.policy == CachePolicy.ARC:
                            self._update_arc_on_access(key, current_level)
                    
                    # Promover a nivel superior si es frecuentemente accedido
                    if (item.access_count > 10 and 
                        current_level != CacheLevel.MEMORY and
                        self._can_promote(current_level)):
                        await self._promote_item(key, current_level)
                    
                    # Actualizar estadísticas
                    access_time = (time.time() - start_time) * 1000
                    self._update_access_stats(hit=True, access_time=access_time)
                    
                    return item.value
            
            # No encontrado
            access_time = (time.time() - start_time) * 1000
            self._update_access_stats(hit=False, access_time=access_time)
            
            return None
    
    async def set_in_cache(
        self,
        key: str,
        value: Any,
        level: Optional[CacheLevel] = None,
        ttl_seconds: Optional[float] = None,
        policy: Optional[CachePolicy] = None,
        dependencies: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        priority: int = 1
    ) -> bool:
        """
        Almacena un valor en el caché.
        
        Args:
            key: Clave para almacenar
            value: Valor a almacenar
            level: Nivel de caché (None = auto-determinado)
            ttl_seconds: Tiempo de vida en segundos
            policy: Política de caché para este item
            dependencies: Dependencias para invalidación
            patterns: Patrones para invalidación
            priority: Prioridad (1-10, más alto = más importante)
            
        Returns:
            bool: True si se almacenó exitosamente
        """
        async with self._lock:
            try:
                # Determinar nivel si no se especifica
                if level is None:
                    level = self._determine_cache_level(value, priority, ttl_seconds)
                
                # Usar TTL por defecto si no se especifica
                if ttl_seconds is None:
                    ttl_seconds = self.config["default_ttl"]
                
                # Usar política por defecto si no se especifica
                if policy is None:
                    policy = self.policy
                
                # Calcular tamaño
                try:
                    serialized = pickle.dumps(value)
                    size_bytes = len(serialized)
                except:
                    size_bytes = 1024  # Tamaño por defecto
                
                # Crear item de caché
                item = CacheItem(
                    key=key,
                    value=value,
                    level=level,
                    policy=policy,
                    size_bytes=size_bytes,
                    ttl_seconds=ttl_seconds,
                    priority=priority
                )
                
                # Aplicar compresión si está habilitado
                if (self.config["compression_enabled"] and 
                    size_bytes > self.config["compression_threshold"]):
                    item = self._compress_item(item)
                
                # Verificar capacidad
                cache = self.caches[level]
                limit = self.level_limits[level]
                
                if len(cache) >= limit and limit > 0:
                    # Necesitamos liberar espacio
                    if not await self._free_cache_space(level, size_bytes):
                        # Si no podemos liberar espacio, intentar en otro nivel
                        if level != CacheLevel.DISTRIBUTED:
                            next_level = self._get_next_cache_level(level)
                            return await self.set_in_cache(
                                key, value, next_level, ttl_seconds, 
                                policy, dependencies, patterns, priority
                            )
                        return False
                
                # Almacenar en caché
                cache[key] = item
                
                # Para LRU en memoria, mover al final (más reciente)
                if level == CacheLevel.MEMORY:
                    cache.move_to_end(key)
                
                # Actualizar política ARC
                if self.policy == CachePolicy.ARC:
                    self._update_arc_on_insert(key, level)
                
                # Actualizar índices
                if dependencies:
                    for dep in dependencies:
                        self.indices["by_dependency"][dep].append((key, level))
                
                if patterns:
                    for pattern in patterns:
                        self.indices["by_pattern"][pattern].append((key, level))
                
                # Añadir a cola de expiración
                if item.expiration_time:
                    heapq.heappush(
                        self.indices["expiration_queue"],
                        (item.expiration_time.timestamp(), key, level)
                    )
                
                # Actualizar estadísticas
                self.stats["sets"] += 1
                self.stats["total_size_bytes"] += item.size_bytes
                
                # Guardar en disco si es nivel disco
                if level == CacheLevel.DISK:
                    self._save_item_to_disk(item)
                
                # Pre-cache dependencias si está habilitado
                if self.config["prefetch_enabled"] and dependencies:
                    asyncio.create_task(self._prefetch_dependencies(dependencies))
                
                return True
                
            except Exception as e:
                print(f"Error setting cache item: {e}")
                return False
    
    async def invalidate_cache_entry(
        self,
        key: str,
        level: Optional[CacheLevel] = None
    ) -> bool:
        """
        Invalida una entrada específica del caché.
        
        Args:
            key: Clave a invalidar
            level: Nivel específico (None = todos los niveles)
            
        Returns:
            bool: True si se invalidó exitosamente
        """
        async with self._lock:
            invalidated = False
            
            # Determinar niveles
            if level is not None:
                levels = [level]
            else:
                levels = list(CacheLevel)
            
            for current_level in levels:
                if key in self.caches[current_level]:
                    if await self._remove_item(key, current_level):
                        invalidated = True
            
            return invalidated
    
    async def invalidate_by_pattern(
        self,
        pattern: str,
        level: Optional[CacheLevel] = None
    ) -> int:
        """
        Invalida entradas del caché que coincidan con un patrón.
        
        Args:
            pattern: Patrón a buscar (puede contener wildcards)
            level: Nivel específico (None = todos los niveles)
            
        Returns:
            int: Número de entradas invalidadas
        """
        if not self.config["invalidation_enabled"]:
            return 0
        
        async with self._lock:
            invalidated = 0
            
            # Buscar en índice de patrones
            if pattern in self.indices["by_pattern"]:
                entries = self.indices["by_pattern"][pattern]
                
                for key, entry_level in entries[:]:  # Copia para modificar durante iteración
                    # Filtrar por nivel si se especifica
                    if level is not None and entry_level != level:
                        continue
                    
                    if await self.invalidate_cache_entry(key, entry_level):
                        invalidated += 1
                        # Remover del índice
                        entries.remove((key, entry_level))
            
            return invalidated
    
    async def invalidate_by_dependency(
        self,
        dependency: str,
        level: Optional[CacheLevel] = None
    ) -> int:
        """
        Invalida entradas del caché que dependen de un recurso.
        
        Args:
            dependency: Dependencia a invalidar
            level: Nivel específico (None = todos los niveles)
            
        Returns:
            int: Número de entradas invalidadas
        """
        if not self.config["invalidation_enabled"]:
            return 0
        
        async with self._lock:
            invalidated = 0
            
            # Buscar en índice de dependencias
            if dependency in self.indices["by_dependency"]:
                entries = self.indices["by_dependency"][dependency]
                
                for key, entry_level in entries[:]:
                    # Filtrar por nivel si se especifica
                    if level is not None and entry_level != level:
                        continue
                    
                    if await self.invalidate_cache_entry(key, entry_level):
                        invalidated += 1
                        # Remover del índice
                        entries.remove((key, entry_level))
            
            return invalidated
    
    async def clear_cache(
        self,
        level: Optional[CacheLevel] = None,
        pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Limpia el caché según criterios.
        
        Args:
            level: Nivel específico (None = todos los niveles)
            pattern: Patrón para filtrar (None = todos)
            
        Returns:
            Dict con resultados de la limpieza
        """
        async with self._lock:
            results = {
                "items_removed": 0,
                "space_freed_bytes": 0,
                "levels_cleared": []
            }
            
            # Determinar niveles
            if level is not None:
                levels = [level]
            else:
                levels = list(CacheLevel)
            
            for current_level in levels:
                cache = self.caches[current_level]
                items_to_remove = []
                
                # Filtrar por patrón si se especifica
                if pattern:
                    for key, item in cache.items():
                        if self._matches_pattern(key, pattern):
                            items_to_remove.append((key, item))
                else:
                    # Todos los items
                    items_to_remove = [(key, item) for key, item in cache.items()]
                
                # Eliminar items
                for key, item in items_to_remove:
                    if await self._remove_item(key, current_level):
                        results["items_removed"] += 1
                        results["space_freed_bytes"] += item.size_bytes
                
                if items_to_remove:
                    results["levels_cleared"].append(current_level.value)
            
            return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas del caché.
        
        Returns:
            Dict con estadísticas completas
        """
        stats = self.stats.copy()
        
        # Agregar estadísticas por nivel
        for level in CacheLevel:
            cache = self.caches[level]
            level_stats = {
                "item_count": len(cache),
                "total_size_bytes": sum(item.size_bytes for item in cache.values()),
                "avg_item_size_bytes": 0,
                "hit_rate": 0.0,
                "oldest_item": None,
                "newest_item": None
            }
            
            if cache:
                # Calcular tamaño promedio
                total_size = level_stats["total_size_bytes"]
                level_stats["avg_item_size_bytes"] = total_size / len(cache)
                
                # Encontrar items más viejo y más nuevo
                items = list(cache.values())
                oldest = min(items, key=lambda x: x.created_at)
                newest = max(items, key=lambda x: x.created_at)
                
                level_stats["oldest_item"] = {
                    "key": oldest.key,
                    "age_seconds": (datetime.now() - oldest.created_at).total_seconds()
                }
                level_stats["newest_item"] = {
                    "key": newest.key,
                    "age_seconds": (datetime.now() - newest.created_at).total_seconds()
                }
            
            stats[f"level_{level.value}_stats"] = level_stats
        
        # Calcular tasa de aciertos
        total_accesses = stats["hits"] + stats["misses"]
        if total_accesses > 0:
            stats["hit_rate"] = stats["hits"] / total_accesses
        
        # Información de capacidad
        stats["capacity_info"] = {}
        for level in CacheLevel:
            limit = self.level_limits[level]
            current = len(self.caches[level])
            stats["capacity_info"][level.value] = {
                "current": current,
                "limit": limit if limit > 0 else "unlimited",
                "percentage": (current / limit * 100) if limit > 0 else 0
            }
        
        # Política actual
        stats["current_policy"] = self.policy.value
        
        # Tiempo de funcionamiento
        if hasattr(self, 'start_time'):
            stats["uptime_seconds"] = time.time() - self.start_time
        
        return stats
    
    async def optimize_cache_policy(self) -> Dict[str, Any]:
        """
        Optimiza la política de caché basándose en patrones de acceso.
        
        Returns:
            Dict con resultados de la optimización
        """
        async with self._lock:
            results = {
                "previous_policy": self.policy.value,
                "new_policy": self.policy.value,
                "reason": "No optimization needed",
                "predicted_improvement": "0%"
            }
            
            # Analizar patrones de acceso
            total_accesses = self.stats["hits"] + self.stats["misses"]
            
            if total_accesses < 100:
                # No hay suficiente data para optimizar
                return results
            
            # Calcular métricas para diferentes políticas
            policy_metrics = {}
            
            # Simular LRU
            lru_score = self._simulate_policy_score(CachePolicy.LRU)
            policy_metrics["LRU"] = lru_score
            
            # Simular LFU
            lfu_score = self._simulate_policy_score(CachePolicy.LFU)
            policy_metrics["LFU"] = lfu_score
            
            # Simular ARC si hay suficiente data
            arc_score = self._simulate_policy_score(CachePolicy.ARC)
            policy_metrics["ARC"] = arc_score
            
            # Encontrar mejor política
            best_policy = max(policy_metrics.items(), key=lambda x: x[1])
            
            if best_policy[0] != self.policy.value and best_policy[1] > 1.1:
                # Cambiar política si hay mejora > 10%
                self.policy = CachePolicy(best_policy[0])
                results["new_policy"] = self.policy.value
                results["reason"] = f"Policy change predicted {best_policy[1]:.1f}x improvement"
                results["predicted_improvement"] = f"{(best_policy[1] - 1) * 100:.1f}%"
                
                # Reinicializar estructuras para nueva política
                if self.policy == CachePolicy.ARC:
                    self._init_arc()
            
            return results
    
    async def prefetch_to_cache(
        self,
        keys: List[str],
        fetch_function: Callable[[str], Any],
        level: CacheLevel = CacheLevel.MEMORY,
        ttl_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Pre-carga items en el caché de manera predictiva.
        
        Args:
            keys: Lista de claves a pre-cargar
            fetch_function: Función para obtener valores
            level: Nivel de caché
            ttl_seconds: Tiempo de vida en segundos
            
        Returns:
            Dict con resultados del pre-caching
        """
        if not self.config["prefetch_enabled"]:
            return {"prefetched": 0, "skipped": len(keys), "reason": "prefetch disabled"}
        
        results = {
            "prefetched": 0,
            "skipped": 0,
            "errors": 0,
            "details": []
        }
        
        for key in keys:
            try:
                # Verificar si ya está en caché
                existing = await self.get_from_cache(key, level, update_access=False)
                
                if existing is not None:
                    results["skipped"] += 1
                    results["details"].append({
                        "key": key,
                        "status": "already_cached",
                        "level": level.value
                    })
                    continue
                
                # Calcular probabilidad de acceso
                access_probability = self._calculate_access_probability(key)
                
                if access_probability < self.config["prefetch_threshold"]:
                    results["skipped"] += 1
                    results["details"].append({
                        "key": key,
                        "status": "low_probability",
                        "probability": access_probability
                    })
                    continue
                
                # Obtener valor
                value = await asyncio.get_event_loop().run_in_executor(
                    None, fetch_function, key
                )
                
                # Almacenar en caché
                success = await self.set_in_cache(
                    key, value, level, ttl_seconds
                )
                
                if success:
                    results["prefetched"] += 1
                    results["details"].append({
                        "key": key,
                        "status": "prefetched",
                        "level": level.value,
                        "probability": access_probability
                    })
                else:
                    results["errors"] += 1
                    results["details"].append({
                        "key": key,
                        "status": "failed",
                        "level": level.value
                    })
                    
            except Exception as e:
                results["errors"] += 1
                results["details"].append({
                    "key": key,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    # Métodos auxiliares protegidos
    
    def _determine_cache_level(
        self, 
        value: Any, 
        priority: int,
        ttl_seconds: Optional[float]
    ) -> CacheLevel:
        """Determina el nivel de caché apropiado para un item."""
        # Items de alta prioridad y TTL corto van a memoria
        if (priority >= 8 and 
            ttl_seconds and ttl_seconds < 300):  # < 5 minutos
            return CacheLevel.MEMORY
        
        # Items muy grandes van a disco
        try:
            size = len(pickle.dumps(value))
            if size > 1048576:  # > 1MB
                return CacheLevel.DISK
        except:
            pass
        
        # Por defecto: memoria para equilibrio
        return CacheLevel.MEMORY
    
    def _get_next_cache_level(self, current_level: CacheLevel) -> CacheLevel:
        """Obtiene el siguiente nivel de caché (más lento)."""
        level_order = [CacheLevel.MEMORY, CacheLevel.DISK, CacheLevel.DISTRIBUTED]
        index = level_order.index(current_level)
        if index < len(level_order) - 1:
            return level_order[index + 1]
        return current_level  # Ya está en el nivel más lento
    
    async def _free_cache_space(
        self, 
        level: CacheLevel, 
        required_bytes: int
    ) -> bool:
        """Libera espacio en un nivel de caché."""
        cache = self.caches[level]
        
        # Si no hay límite, siempre hay espacio
        limit = self.level_limits[level]
        if limit == 0:
            return True
        
        # Calcular cuántos items necesitamos expulsar
        current_count = len(cache)
        if current_count < limit:
            return True
        
        items_to_evict = []
        current_time = datetime.now()
        
        # Seleccionar candidatos según política
        if self.policy == CachePolicy.LRU:
            # LRU: expulsar los menos recientemente usados
            items = list(cache.items())
            items_to_evict = items[:max(1, len(items) // 10)]  # 10% más antiguos
            
        elif self.policy == CachePolicy.LFU:
            # LFU: expulsar los menos frecuentemente usados
            items = sorted(
                cache.items(),
                key=lambda x: x[1].frequency
            )
            items_to_evict = items[:max(1, len(items) // 10)]
            
        elif self.policy == CachePolicy.FIFO:
            # FIFO: expulsar los más antiguos
            items = list(cache.items())
            items_to_evict = items[:max(1, len(items) // 10)]
            
        elif self.policy == CachePolicy.TTL:
            # TTL: expulsar los que expiran pronto
            items = sorted(
                cache.items(),
                key=lambda x: x[1].expiration_time or datetime.max
            )
            items_to_evict = items[:max(1, len(items) // 10)]
            
        elif self.policy == CachePolicy.ARC:
            # ARC: usar algoritmos adaptativos
            return await self._arc_free_space(level, required_bytes)
        
        # Expulsar items seleccionados
        space_freed = 0
        for key, item in items_to_evict:
            if space_freed >= required_bytes:
                break
            
            if await self._remove_item(key, level):
                space_freed += item.size_bytes
        
        return space_freed >= required_bytes
    
    async def _arc_free_space(
        self, 
        level: CacheLevel, 
        required_bytes: int
    ) -> bool:
        """Libera espacio usando política ARC."""
        space_freed = 0
        
        # Expulsar de T1 (menos frecuentes)
        while self.t1 and space_freed < required_bytes:
            key, _ = self.t1.popitem(last=False)
            if key in self.caches[level]:
                item = self.caches[level][key]
                if await self._remove_item(key, level):
                    space_freed += item.size_bytes
        
        # Si aún no hay suficiente espacio, expulsar de T2
        while self.t2 and space_freed < required_bytes:
            key, _ = self.t2.popitem(last=False)
            if key in self.caches[level]:
                item = self.caches[level][key]
                if await self._remove_item(key, level):
                    space_freed += item.size_bytes
        
        return space_freed >= required_bytes
    
    async def _remove_item(self, key: str, level: CacheLevel) -> bool:
        """Elimina un item del caché."""
        if key not in self.caches[level]:
            return False
        
        item = self.caches[level][key]
        
        # Eliminar de caché
        del self.caches[level][key]
        
        # Eliminar de política ARC
        if self.policy == CachePolicy.ARC:
            self.t1.pop(key, None)
            self.t2.pop(key, None)
            self.b1.pop(key, None)
            self.b2.pop(key, None)
        
        # Eliminar de índices
        self._remove_from_indices(key, level, item)
        
        # Eliminar de cola de expiración
        self._remove_from_expiration_queue(key, level)
        
        # Eliminar de disco si es necesario
        if level == CacheLevel.DISK:
            self._remove_item_from_disk(key)
        
        # Actualizar estadísticas
        self.stats["deletes"] += 1
        self.stats["evictions"] += 1
        self.stats["total_size_bytes"] -= item.size_bytes
        
        return True
    
    async def _promote_item(self, key: str, from_level: CacheLevel) -> bool:
        """Promueve un item a un nivel superior de caché."""
        # Determinar nivel destino
        if from_level == CacheLevel.DISTRIBUTED:
            to_level = CacheLevel.DISK
        elif from_level == CacheLevel.DISK:
            to_level = CacheLevel.MEMORY
        else:
            return False  # Ya está en el nivel más alto
        
        # Obtener item
        if key not in self.caches[from_level]:
            return False
        
        item = self.caches[from_level][key]
        
        # Verificar capacidad en nivel destino
        cache = self.caches[to_level]
        limit = self.level_limits[to_level]
        
        if len(cache) >= limit and limit > 0:
            # Necesitamos liberar espacio
            if not await self._free_cache_space(to_level, item.size_bytes):
                return False
        
        # Mover item
        del self.caches[from_level][key]
        cache[key] = item
        item.level = to_level
        
        # Para LRU en memoria, mover al final
        if to_level == CacheLevel.MEMORY:
            cache.move_to_end(key)
        
        # Actualizar índices
        self._update_indices_on_move(key, from_level, to_level, item)
        
        # Guardar/eliminar de disco según niveles
        if from_level == CacheLevel.DISK:
            self._remove_item_from_disk(key)
        if to_level == CacheLevel.DISK:
            self._save_item_to_disk(item)
        
        return True
    
    def _update_arc_on_insert(self, key: str, level: CacheLevel) -> None:
        """Actualiza estructuras ARC al insertar un item."""
        if level != CacheLevel.MEMORY:
            return
        
        # Nuevos items van a T1
        self.t1[key] = datetime.now()
    
    def _update_arc_on_access(self, key: str, level: CacheLevel) -> None:
        """Actualiza estructuras ARC al acceder a un item."""
        if level != CacheLevel.MEMORY:
            return
        
        # Mover de T1 a T2 si estaba en T1
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = datetime.now()
        elif key in self.t2:
            # Re-ordenar en T2 (mover al final)
            del self.t2[key]
            self.t2[key] = datetime.now()
    
    def _update_indices_on_move(
        self, 
        key: str, 
        from_level: CacheLevel,
        to_level: CacheLevel,
        item: CacheItem
    ) -> None:
        """Actualiza índices al mover un item entre niveles."""
        # Actualizar índices de patrones
        for pattern, entries in self.indices["by_pattern"].items():
            for i, (entry_key, entry_level) in enumerate(entries):
                if entry_key == key and entry_level == from_level:
                    entries[i] = (key, to_level)
        
        # Actualizar índices de dependencias
        for dep, entries in self.indices["by_dependency"].items():
            for i, (entry_key, entry_level) in enumerate(entries):
                if entry_key == key and entry_level == from_level:
                    entries[i] = (key, to_level)
        
        # Actualizar cola de expiración
        self._remove_from_expiration_queue(key, from_level)
        if item.expiration_time:
            heapq.heappush(
                self.indices["expiration_queue"],
                (item.expiration_time.timestamp(), key, to_level)
            )
    
    def _remove_from_indices(
        self, 
        key: str, 
        level: CacheLevel,
        item: CacheItem
    ) -> None:
        """Elimina un item de todos los índices."""
        # Eliminar de índices de patrones
        for pattern, entries in self.indices["by_pattern"].items():
            entries[:] = [(k, l) for k, l in entries if not (k == key and l == level)]
        
        # Eliminar de índices de dependencias
        for dep, entries in self.indices["by_dependency"].items():
            entries[:] = [(k, l) for k, l in entries if not (k == key and l == level)]
    
    def _remove_from_expiration_queue(self, key: str, level: CacheLevel) -> None:
        """Elimina un item de la cola de expiración."""
        new_queue = []
        for timestamp, queue_key, queue_level in self.indices["expiration_queue"]:
            if not (queue_key == key and queue_level == level):
                heapq.heappush(new_queue, (timestamp, queue_key, queue_level))
        
        self.indices["expiration_queue"] = new_queue
    
    def _compress_item(self, item: CacheItem) -> CacheItem:
        """Comprime un item de caché."""
        try:
            import gzip
            import io
            
            serialized = pickle.dumps(item.value)
            if len(serialized) > self.config["compression_threshold"]:
                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
                    f.write(serialized)
                compressed = buffer.getvalue()
                
                if len(compressed) < len(serialized):
                    # Crear nuevo item con valor comprimido
                    compressed_item = CacheItem(
                        key=item.key,
                        value=compressed,
                        level=item.level,
                        policy=item.policy,
                        size_bytes=len(compressed),
                        ttl_seconds=item.ttl_seconds,
                        priority=item.priority
                    )
                    
                    # Copiar otros campos
                    compressed_item.created_at = item.created_at
                    compressed_item.last_accessed = item.last_accessed
                    compressed_item.access_count = item.access_count
                    compressed_item.frequency = item.frequency
                    
                    # Marcar como comprimido
                    if not hasattr(compressed_item, 'metadata'):
                        compressed_item.metadata = {}
                    compressed_item.metadata["compressed"] = True
                    compressed_item.metadata["original_size"] = len(serialized)
                    
                    return compressed_item
        except Exception:
            pass
        
        return item
    
    def _decompress_item(self, item: CacheItem) -> CacheItem:
        """Descomprime un item de caché."""
        if not hasattr(item, 'metadata') or not item.metadata.get("compressed", False):
            return item
        
        try:
            import gzip
            import io
            
            compressed = item.value
            buffer = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
                decompressed = pickle.load(f)
            
            # Crear nuevo item con valor descomprimido
            decompressed_item = CacheItem(
                key=item.key,
                value=decompressed,
                level=item.level,
                policy=item.policy,
                size_bytes=item.metadata.get("original_size", item.size_bytes),
                ttl_seconds=item.ttl_seconds,
                priority=item.priority
            )
            
            # Copiar otros campos
            decompressed_item.created_at = item.created_at
            decompressed_item.last_accessed = item.last_accessed
            decompressed_item.access_count = item.access_count
            decompressed_item.frequency = item.frequency
            
            return decompressed_item
        except Exception:
            return item
    
    def _calculate_access_probability(self, key: str) -> float:
        """Calcula probabilidad de acceso para una clave."""
        # Implementación simplificada
        # En una implementación real, usaríamos historial de acceso
        
        # Verificar si es una dependencia común
        if key in self.indices["by_dependency"]:
            dependency_count = len(self.indices["by_dependency"][key])
            return min(1.0, dependency_count / 10.0)  # Normalizar
        
        # Verificar si coincide con patrones comunes
        pattern_match = False
        for pattern in self.indices["by_pattern"]:
            if self._matches_pattern(key, pattern):
                pattern_match = True
                break
        
        if pattern_match:
            return 0.5
        
        # Probabilidad base baja
        return 0.1
    
    async def _prefetch_dependencies(self, dependencies: List[str]) -> None:
        """Pre-carga dependencias relacionadas."""
        # Implementación simplificada
        # En una implementación real, buscaríamos items relacionados
        
        for dep in dependencies:
            # Verificar si esta dependencia es común
            if dep in self.indices["by_dependency"]:
                entries = self.indices["by_dependency"][dep]
                if len(entries) > 3:  # Dependencia común
                    # Aquí podríamos pre-cargar items relacionados
                    pass
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Verifica si una clave coincide con un patrón."""
        # Soporte básico para wildcards
        if '*' in pattern:
            # Convertir patrón a regex
            import re
            regex_pattern = pattern.replace('*', '.*')
            return bool(re.match(regex_pattern, key))
        else:
            return key == pattern
    
    def _simulate_policy_score(self, policy: CachePolicy) -> float:
        """Simula el score de una política de caché."""
        # Implementación simplificada
        # En una implementación real, simularíamos acceso
        
        current_hit_rate = self.stats.get("hit_rate", 0.5)
        
        # Scores basados en características del workload
        if policy == CachePolicy.LRU:
            # LRU funciona bien para access patterns temporales locales
            return current_hit_rate * 1.0
            
        elif policy == CachePolicy.LFU:
            # LFU funciona bien para access patterns con repetición
            return current_hit_rate * 1.1
            
        elif policy == CachePolicy.ARC:
            # ARC se adapta a diferentes patterns
            return current_hit_rate * 1.2
            
        else:
            return current_hit_rate
    
    def _can_promote(self, from_level: CacheLevel) -> bool:
        """Verifica si un item puede ser promovido a nivel superior."""
        to_level = None
        if from_level == CacheLevel.DISK:
            to_level = CacheLevel.MEMORY
        elif from_level == CacheLevel.DISTRIBUTED:
            to_level = CacheLevel.DISK
        
        if to_level is None:
            return False
        
        # Verificar límites
        limit = self.level_limits[to_level]
        if limit > 0 and len(self.caches[to_level]) >= limit:
            return False
        
        return True
    
    def _update_access_stats(self, hit: bool, access_time: float) -> None:
        """Actualiza estadísticas de acceso."""
        if hit:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1
        
        # Actualizar tiempo promedio de acceso
        total_accesses = self.stats["hits"] + self.stats["misses"]
        current_avg = self.stats["avg_access_time_ms"]
        
        self.stats["avg_access_time_ms"] = (
            (current_avg * (total_accesses - 1) + access_time) / total_accesses
        )
    
    def _save_item_to_disk(self, item: CacheItem) -> None:
        """Guarda un item en disco."""
        try:
            cache_path = Path(self.config["disk_cache_path"])
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Crear nombre de archivo seguro
            safe_key = hashlib.md5(item.key.encode()).hexdigest()
            file_path = cache_path / f"{safe_key}.cache"
            
            with open(file_path, 'wb') as f:
                pickle.dump(item.to_dict(), f)
                
        except Exception as e:
            print(f"Failed to save cache item to disk: {e}")
    
    def _remove_item_from_disk(self, key: str) -> None:
        """Elimina un item del disco."""
        try:
            cache_path = Path(self.config["disk_cache_path"])
            safe_key = hashlib.md5(key.encode()).hexdigest()
            file_path = cache_path / f"{safe_key}.cache"
            
            if file_path.exists():
                file_path.unlink()
                
        except Exception as e:
            print(f"Failed to remove cache item from disk: {e}")
    
    def _load_disk_cache(self) -> None:
        """Carga el caché de disco."""
        try:
            cache_path = Path(self.config["disk_cache_path"])
            if not cache_path.exists():
                return
            
            # Cargar todos los archivos .cache
            for file_path in cache_path.glob("*.cache"):
                try:
                    with open(file_path, 'rb') as f:
                        item_data = pickle.load(f)
                    
                    item = CacheItem.from_dict(item_data)
                    
                    # Verificar expiración
                    if item.is_expired():
                        file_path.unlink()
                        continue
                    
                    # Almacenar en caché de disco
                    self.caches[CacheLevel.DISK][item.key] = item
                    
                    # Actualizar estadísticas
                    self.stats["total_size_bytes"] += item.size_bytes
                    
                except Exception as e:
                    print(f"Failed to load cache file {file_path}: {e}")
                    file_path.unlink()
                    
        except Exception as e:
            print(f"Failed to load disk cache: {e}")
    
    async def _periodic_cleanup(self) -> None:
        """Limpieza periódica del caché."""
        try:
            while True:
                await asyncio.sleep(self.config["cleanup_interval"])
                
                async with self._lock:
                    # Limpiar items expirados
                    expired_count = 0
                    current_time = time.time()
                    
                    # Procesar cola de expiración
                    while (self.indices["expiration_queue"] and 
                           self.indices["expiration_queue"][0][0] < current_time):
                        
                        timestamp, key, level = heapq.heappop(
                            self.indices["expiration_queue"]
                        )
                        
                        if (level in self.caches and 
                            key in self.caches[level]):
                            
                            item = self.caches[level][key]
                            if item.is_expired():
                                await self._remove_item(key, level)
                                expired_count += 1
                    
                    # Limpiar índices de dependencias/patrones huérfanos
                    self._clean_orphaned_indices()
                    
                    # Registrar limpieza
                    if expired_count > 0:
                        print(f"Cache cleanup: removed {expired_count} expired items")
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in cache cleanup task: {e}")
    
    async def _periodic_stats_update(self) -> None:
        """Actualización periódica de estadísticas."""
        try:
            while True:
                await asyncio.sleep(self.config["stats_interval"])
                
                # Guardar estadísticas actuales
                stats = self.get_cache_stats()
                
                # Podríamos guardar históricos aquí
                # o enviar a sistema de monitoreo
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in stats update task: {e}")
    
    def _clean_orphaned_indices(self) -> None:
        """Limpia índices huérfanos."""
        # Limpiar índices de dependencias
        for dep in list(self.indices["by_dependency"].keys()):
            entries = self.indices["by_dependency"][dep]
            valid_entries = []
            
            for key, level in entries:
                if level in self.caches and key in self.caches[level]:
                    valid_entries.append((key, level))
            
            if valid_entries:
                self.indices["by_dependency"][dep] = valid_entries
            else:
                del self.indices["by_dependency"][dep]
        
        # Limpiar índices de patrones
        for pattern in list(self.indices["by_pattern"].keys()):
            entries = self.indices["by_pattern"][pattern]
            valid_entries = []
            
            for key, level in entries:
                if level in self.caches and key in self.caches[level]:
                    valid_entries.append((key, level))
            
            if valid_entries:
                self.indices["by_pattern"][pattern] = valid_entries
            else:
                del self.indices["by_pattern"][pattern]
    
    async def shutdown(self) -> None:
        """Apaga el gestor de caché de manera controlada."""
        # Cancelar tareas de mantenimiento
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self, '_stats_task'):
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass
        
        # Guardar caché en disco
        self._save_cache_to_disk()
        
        # Guardar estadísticas
        self._save_stats()
    
    def _save_cache_to_disk(self) -> None:
        """Guarda todo el caché en disco."""
        try:
            cache_path = Path(self.config["disk_cache_path"])
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Guardar items de memoria en disco
            memory_cache = self.caches[CacheLevel.MEMORY]
            for key, item in memory_cache.items():
                if not item.is_expired():
                    # Guardar en disco
                    self._save_item_to_disk(item)
            
        except Exception as e:
            print(f"Failed to save cache to disk: {e}")
    
    def _save_stats(self) -> None:
        """Guarda estadísticas en disco."""
        try:
            stats_path = Path("./data/cache_stats")
            stats_path.mkdir(parents=True, exist_ok=True)
            
            stats_file = stats_path / "cache_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save cache stats: {e}")