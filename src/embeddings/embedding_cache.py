"""
EmbeddingCache - Sistema de caché para embeddings.
Gestiona caché multi-nivel para embeddings vectoriales.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import hashlib
import pickle
import json
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import warnings
from collections import OrderedDict

class CacheLevel(Enum):
    """Niveles de caché."""
    L1 = "L1"  # Memoria (más rápido)
    L2 = "L2"  # Disco (más capacidad)
    L3 = "L3"  # Red/Remoto (más capacidad aún)

class CachePolicy(Enum):
    """Políticas de reemplazo de caché."""
    LRU = "LRU"  # Least Recently Used
    LFU = "LFU"  # Least Frequently Used
    FIFO = "FIFO"  # First In First Out
    TTL = "TTL"  # Time To Live

class CacheEntry(BaseModel):
    """Entrada de caché."""
    key: str
    value: Any
    created_at: datetime = Field(default_factory=datetime.now)
    accessed_at: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    def is_expired(self) -> bool:
        """Verifica si la entrada ha expirado."""
        if self.ttl_seconds is None:
            return False
        
        current_time = datetime.now()
        age = current_time - self.created_at
        return age.total_seconds() > self.ttl_seconds
    
    def update_access(self) -> None:
        """Actualiza tiempo de acceso."""
        self.accessed_at = datetime.now()
        self.access_count += 1

@dataclass
class CacheLevelConfig:
    """Configuración de un nivel de caché."""
    level: CacheLevel
    max_size_bytes: int
    max_entries: int
    policy: CachePolicy
    ttl_seconds: Optional[int] = None
    enabled: bool = True

class CacheStats(BaseModel):
    """Estadísticas de caché."""
    level: CacheLevel
    hits: int = 0
    misses: int = 0
    current_size_bytes: int = 0
    current_entries: int = 0
    evictions: int = 0
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0
    
    class Config:
        arbitrary_types_allowed = True

class EmbeddingCache:
    """
    Sistema de caché multi-nivel para embeddings.
    
    Características:
    1. Caché multi-nivel (L1: memoria, L2: disco, L3: remoto)
    2. Políticas de reemplazo configurables
    3. Expiración automática (TTL)
    4. Prefetch y precarga
    5. Estadísticas detalladas
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Inicializa el sistema de caché.
        
        Args:
            cache_dir: Directorio para caché en disco
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración por niveles
        self.levels = {
            CacheLevel.L1: CacheLevelConfig(
                level=CacheLevel.L1,
                max_size_bytes=100 * 1024 * 1024,  # 100 MB
                max_entries=1000,
                policy=CachePolicy.LRU,
                ttl_seconds=300  # 5 minutos
            ),
            CacheLevel.L2: CacheLevelConfig(
                level=CacheLevel.L2,
                max_size_bytes=1024 * 1024 * 1024,  # 1 GB
                max_entries=10000,
                policy=CachePolicy.LRU,
                ttl_seconds=3600  # 1 hora
            ),
            CacheLevel.L3: CacheLevelConfig(
                level=CacheLevel.L3,
                max_size_bytes=10 * 1024 * 1024 * 1024,  # 10 GB
                max_entries=100000,
                policy=CachePolicy.TTL,
                ttl_seconds=86400,  # 24 horas
                enabled=False  # Por defecto deshabilitado
            )
        }
        
        # Almacenamiento por niveles
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l2_files: Dict[str, Path] = {}
        self._l3_remote: Dict[str, Any] = {}  # Simulado
        
        # Estadísticas
        self._stats = {
            CacheLevel.L1: CacheStats(level=CacheLevel.L1),
            CacheLevel.L2: CacheStats(level=CacheLevel.L2),
            CacheLevel.L3: CacheStats(level=CacheLevel.L3)
        }
        
        # Cargar caché L2 desde disco
        self._load_l2_cache()
    
    def get_cached_embedding(self, 
                            key: str,
                            check_lower_levels: bool = True) -> Optional[Any]:
        """
        Obtiene un embedding desde caché.
        
        Args:
            key: Clave del embedding
            check_lower_levels: Si True, busca en niveles inferiores
            
        Returns:
            Embedding si está en caché, None en caso contrario
        """
        start_time = datetime.now()
        
        # Buscar en L1 (memoria)
        if self.levels[CacheLevel.L1].enabled and key in self._l1_cache:
            entry = self._l1_cache[key]
            
            # Verificar expiración
            if entry.is_expired():
                self._remove_from_l1(key)
                self._stats[CacheLevel.L1].misses += 1
            else:
                # Actualizar acceso
                entry.update_access()
                self._l1_cache.move_to_end(key)  # LRU
                
                # Actualizar estadísticas
                self._stats[CacheLevel.L1].hits += 1
                self._update_access_time(CacheLevel.L1, start_time)
                
                return entry.value
        
        # Buscar en L2 (disco) si está habilitado
        if (check_lower_levels and 
            self.levels[CacheLevel.L2].enabled and 
            key in self._l2_files):
            
            try:
                entry = self._load_from_l2(key)
                
                if entry and not entry.is_expired():
                    # Promover a L1 si hay espacio
                    if self._has_space_in_l1(entry.size_bytes):
                        self._add_to_l1(entry)
                    
                    # Actualizar estadísticas
                    self._stats[CacheLevel.L2].hits += 1
                    self._update_access_time(CacheLevel.L2, start_time)
                    
                    return entry.value
                else:
                    # Eliminar entrada expirada
                    self._remove_from_l2(key)
                    self._stats[CacheLevel.L2].misses += 1
                    
            except Exception as e:
                warnings.warn(f"Failed to load from L2 cache: {str(e)}")
                self._stats[CacheLevel.L2].misses += 1
        
        # Buscar en L3 (remoto) si está habilitado
        if (check_lower_levels and 
            self.levels[CacheLevel.L3].enabled and 
            key in self._l3_remote):
            
            # Simulación de caché remoto
            entry = self._l3_remote.get(key)
            if entry and not entry.is_expired():
                # Promover a niveles superiores
                if self._has_space_in_l2(entry.size_bytes):
                    self._add_to_l2(entry)
                if self._has_space_in_l1(entry.size_bytes):
                    self._add_to_l1(entry)
                
                # Actualizar estadísticas
                self._stats[CacheLevel.L3].hits += 1
                self._update_access_time(CacheLevel.L3, start_time)
                
                return entry.value
            else:
                self._stats[CacheLevel.L3].misses += 1
        
        # No encontrado en ningún nivel
        for level in [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]:
            if self.levels[level].enabled:
                self._stats[level].misses += 1
        
        return None
    
    def cache_embedding(self, 
                       key: str, 
                       value: Any,
                       ttl_seconds: Optional[int] = None,
                       level: CacheLevel = CacheLevel.L1,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Almacena un embedding en caché.
        
        Args:
            key: Clave única
            value: Valor a almacenar
            ttl_seconds: Tiempo de vida en segundos
            level: Nivel de caché
            metadata: Metadatos adicionales
            
        Returns:
            bool: True si se almacenó exitosamente
        """
        if not self.levels[level].enabled:
            warnings.warn(f"Cache level {level} is not enabled")
            return False
        
        try:
            # Calcular tamaño
            size_bytes = self._calculate_size(value)
            
            # Crear entrada
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.levels[level].ttl_seconds,
                metadata=metadata or {}
            )
            
            # Almacenar en nivel especificado
            success = False
            
            if level == CacheLevel.L1:
                success = self._add_to_l1(entry)
            elif level == CacheLevel.L2:
                success = self._add_to_l2(entry)
            elif level == CacheLevel.L3:
                success = self._add_to_l3(entry)
            
            return success
            
        except Exception as e:
            warnings.warn(f"Failed to cache embedding: {str(e)}")
            return False
    
    def invalidate_cache(self, 
                        keys: Optional[List[str]] = None,
                        level: Optional[CacheLevel] = None) -> int:
        """
        Invalida entradas de caché.
        
        Args:
            keys: Lista de claves a invalidar (None = todas)
            level: Nivel específico (None = todos los niveles)
            
        Returns:
            int: Número de entradas invalidadas
        """
        levels_to_check = [level] if level else list(CacheLevel)
        total_invalidated = 0
        
        for cache_level in levels_to_check:
            if not self.levels[cache_level].enabled:
                continue
            
            if cache_level == CacheLevel.L1:
                total_invalidated += self._invalidate_l1(keys)
            elif cache_level == CacheLevel.L2:
                total_invalidated += self._invalidate_l2(keys)
            elif cache_level == CacheLevel.L3:
                total_invalidated += self._invalidate_l3(keys)
        
        return total_invalidated
    
    def clear_cache(self, level: Optional[CacheLevel] = None) -> Dict[CacheLevel, int]:
        """
        Limpia completamente la caché.
        
        Args:
            level: Nivel específico (None = todos los niveles)
            
        Returns:
            Dict con número de entradas eliminadas por nivel
        """
        results = {}
        levels_to_clear = [level] if level else list(CacheLevel)
        
        for cache_level in levels_to_clear:
            if not self.levels[cache_level].enabled:
                continue
            
            if cache_level == CacheLevel.L1:
                count = len(self._l1_cache)
                self._l1_cache.clear()
                self._stats[cache_level].current_entries = 0
                self._stats[cache_level].current_size_bytes = 0
                results[cache_level] = count
            
            elif cache_level == CacheLevel.L2:
                count = self._clear_l2_cache()
                results[cache_level] = count
            
            elif cache_level == CacheLevel.L3:
                count = len(self._l3_remote)
                self._l3_remote.clear()
                self._stats[cache_level].current_entries = 0
                self._stats[cache_level].current_size_bytes = 0
                results[cache_level] = count
        
        return results
    
    def get_cache_stats(self, level: Optional[CacheLevel] = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de caché.
        
        Args:
            level: Nivel específico (None = todos los niveles)
            
        Returns:
            Dict con estadísticas
        """
        if level:
            return self._get_level_stats(level)
        else:
            stats = {}
            for cache_level in CacheLevel:
                if self.levels[cache_level].enabled:
                    stats[cache_level.value] = self._get_level_stats(cache_level)
            
            # Estadísticas globales
            stats["global"] = self._get_global_stats()
            return stats
    
    def optimize_cache(self) -> Dict[CacheLevel, Dict[str, Any]]:
        """
        Optimiza la caché aplicando políticas de limpieza.
        
        Returns:
            Dict con resultados de optimización por nivel
        """
        results = {}
        
        for cache_level in CacheLevel:
            if not self.levels[cache_level].enabled:
                continue
            
            level_results = {}
            
            if cache_level == CacheLevel.L1:
                level_results = self._optimize_l1()
            elif cache_level == CacheLevel.L2:
                level_results = self._optimize_l2()
            elif cache_level == CacheLevel.L3:
                level_results = self._optimize_l3()
            
            results[cache_level] = level_results
        
        return results
    
    def preload_cache(self, 
                     key_pattern: Optional[str] = None,
                     level: CacheLevel = CacheLevel.L2) -> int:
        """
        Precarga caché con entradas frecuentes.
        
        Args:
            key_pattern: Patrón de claves a precargar
            level: Nivel de caché
            
        Returns:
            int: Número de entradas precargadas
        """
        if not self.levels[level].enabled:
            return 0
        
        # En un sistema real, esto cargaría desde una base de datos
        # o archivo de configuración las entradas frecuentes
        
        # Por ahora, implementación simulada
        if level == CacheLevel.L2:
            # Cargar archivos existentes
            return self._load_l2_cache()
        
        return 0
    
    def prefetch_to_cache(self, 
                         keys: List[str],
                         fetch_function: callable,
                         level: CacheLevel = CacheLevel.L1) -> Dict[str, bool]:
        """
        Prefetch de entradas a caché.
        
        Args:
            keys: Lista de claves a precargar
            fetch_function: Función para obtener valores
            level: Nivel de caché
            
        Returns:
            Dict con éxito por clave
        """
        results = {}
        
        for key in keys:
            try:
                # Verificar si ya está en caché
                if self.get_cached_embedding(key, check_lower_levels=False):
                    results[key] = True
                    continue
                
                # Obtener valor
                value = fetch_function(key)
                if value:
                    # Almacenar en caché
                    success = self.cache_embedding(key, value, level=level)
                    results[key] = success
                else:
                    results[key] = False
                    
            except Exception as e:
                warnings.warn(f"Prefetch failed for key {key}: {str(e)}")
                results[key] = False
        
        return results
    
    # Métodos privados - L1 Cache (Memoria)
    
    def _add_to_l1(self, entry: CacheEntry) -> bool:
        """Añade entrada a caché L1."""
        # Verificar espacio
        if not self._has_space_in_l1(entry.size_bytes):
            if not self._make_space_in_l1(entry.size_bytes):
                return False
        
        # Añadir entrada
        self._l1_cache[entry.key] = entry
        
        # Actualizar estadísticas
        self._stats[CacheLevel.L1].current_entries += 1
        self._stats[CacheLevel.L1].current_size_bytes += entry.size_bytes
        
        return True
    
    def _remove_from_l1(self, key: str) -> bool:
        """Elimina entrada de caché L1."""
        if key in self._l1_cache:
            entry = self._l1_cache[key]
            del self._l1_cache[key]
            
            # Actualizar estadísticas
            self._stats[CacheLevel.L1].current_entries -= 1
            self._stats[CacheLevel.L1].current_size_bytes -= entry.size_bytes
            
            return True
        
        return False
    
    def _has_space_in_l1(self, required_bytes: int) -> bool:
        """Verifica si hay espacio en L1."""
        config = self.levels[CacheLevel.L1]
        current_size = self._stats[CacheLevel.L1].current_size_bytes
        
        return (current_size + required_bytes) <= config.max_size_bytes
    
    def _make_space_in_l1(self, required_bytes: int) -> bool:
        """Libera espacio en L1 según política."""
        config = self.levels[CacheLevel.L1]
        
        # Verificar si es posible hacer espacio
        if required_bytes > config.max_size_bytes:
            return False
        
        # Aplicar política de reemplazo
        if config.policy == CachePolicy.LRU:
            return self._make_space_lru(CacheLevel.L1, required_bytes)
        elif config.policy == CachePolicy.LFU:
            return self._make_space_lfu(CacheLevel.L1, required_bytes)
        elif config.policy == CachePolicy.FIFO:
            return self._make_space_fifo(CacheLevel.L1, required_bytes)
        elif config.policy == CachePolicy.TTL:
            return self._make_space_ttl(CacheLevel.L1, required_bytes)
        else:
            return self._make_space_lru(CacheLevel.L1, required_bytes)
    
    def _invalidate_l1(self, keys: Optional[List[str]]) -> int:
        """Invalida entradas en L1."""
        if keys is None:
            count = len(self._l1_cache)
            self._l1_cache.clear()
            self._stats[CacheLevel.L1].current_entries = 0
            self._stats[CacheLevel.L1].current_size_bytes = 0
            return count
        
        count = 0
        for key in keys:
            if self._remove_from_l1(key):
                count += 1
        
        return count
    
    def _optimize_l1(self) -> Dict[str, Any]:
        """Optimiza caché L1."""
        results = {
            "entries_before": len(self._l1_cache),
            "entries_after": 0,
            "expired_removed": 0,
            "size_freed_bytes": 0
        }
        
        # Eliminar expirados
        to_remove = []
        freed_bytes = 0
        
        for key, entry in self._l1_cache.items():
            if entry.is_expired():
                to_remove.append(key)
                freed_bytes += entry.size_bytes
        
        for key in to_remove:
            self._remove_from_l1(key)
        
        results["expired_removed"] = len(to_remove)
        results["size_freed_bytes"] = freed_bytes
        results["entries_after"] = len(self._l1_cache)
        
        return results
    
    # Métodos privados - L2 Cache (Disco)
    
    def _add_to_l2(self, entry: CacheEntry) -> bool:
        """Añade entrada a caché L2."""
        # Verificar espacio
        if not self._has_space_in_l2(entry.size_bytes):
            if not self._make_space_in_l2(entry.size_bytes):
                return False
        
        try:
            # Guardar en disco
            file_path = self.cache_dir / f"l2_{entry.key}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
            
            # Actualizar índice
            self._l2_files[entry.key] = file_path
            
            # Actualizar estadísticas
            self._stats[CacheLevel.L2].current_entries += 1
            self._stats[CacheLevel.L2].current_size_bytes += entry.size_bytes
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to add to L2 cache: {str(e)}")
            return False
    
    def _load_from_l2(self, key: str) -> Optional[CacheEntry]:
        """Carga entrada desde caché L2."""
        if key not in self._l2_files:
            return None
        
        try:
            file_path = self._l2_files[key]
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
            
            return entry
            
        except Exception as e:
            warnings.warn(f"Failed to load from L2 cache: {str(e)}")
            return None
    
    def _remove_from_l2(self, key: str) -> bool:
        """Elimina entrada de caché L2."""
        if key in self._l2_files:
            try:
                file_path = self._l2_files[key]
                
                # Obtener tamaño antes de eliminar
                size_bytes = 0
                if file_path.exists():
                    size_bytes = file_path.stat().st_size
                
                # Eliminar archivo
                file_path.unlink(missing_ok=True)
                
                # Eliminar del índice
                del self._l2_files[key]
                
                # Actualizar estadísticas
                self._stats[CacheLevel.L2].current_entries -= 1
                self._stats[CacheLevel.L2].current_size_bytes -= size_bytes
                
                return True
                
            except Exception as e:
                warnings.warn(f"Failed to remove from L2 cache: {str(e)}")
                return False
        
        return False
    
    def _has_space_in_l2(self, required_bytes: int) -> bool:
        """Verifica si hay espacio en L2."""
        config = self.levels[CacheLevel.L2]
        current_size = self._stats[CacheLevel.L2].current_size_bytes
        
        return (current_size + required_bytes) <= config.max_size_bytes
    
    def _make_space_in_l2(self, required_bytes: int) -> bool:
        """Libera espacio en L2."""
        config = self.levels[CacheLevel.L2]
        
        # Verificar si es posible hacer espacio
        if required_bytes > config.max_size_bytes:
            return False
        
        # Primero eliminar expirados
        self._clean_expired_l2()
        
        # Si todavía no hay espacio, aplicar política
        if not self._has_space_in_l2(required_bytes):
            if config.policy == CachePolicy.LRU:
                return self._make_space_lru_disk(required_bytes)
            elif config.policy == CachePolicy.LFU:
                return self._make_space_lfu_disk(required_bytes)
            else:
                return self._make_space_lru_disk(required_bytes)
        
        return True
    
    def _clean_expired_l2(self) -> None:
        """Limpia entradas expiradas en L2."""
        expired_keys = []
        
        for key in list(self._l2_files.keys()):
            entry = self._load_from_l2(key)
            if entry and entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_from_l2(key)
    
    def _make_space_lru_disk(self, required_bytes: int) -> bool:
        """Libera espacio en L2 usando LRU."""
        # Para implementación simplificada, eliminamos archivos más antiguos
        try:
            # Obtener archivos ordenados por fecha de modificación
            files_with_mtime = []
            for key, file_path in self._l2_files.items():
                if file_path.exists():
                    mtime = file_path.stat().st_mtime
                    files_with_mtime.append((key, file_path, mtime))
            
            # Ordenar por fecha de modificación (más antiguos primero)
            files_with_mtime.sort(key=lambda x: x[2])
            
            # Eliminar hasta liberar espacio suficiente
            freed_bytes = 0
            for key, file_path, _ in files_with_mtime:
                if file_path.exists():
                    size = file_path.stat().st_size
                    self._remove_from_l2(key)
                    freed_bytes += size
                    
                    if freed_bytes >= required_bytes:
                        break
            
            return freed_bytes >= required_bytes
            
        except Exception as e:
            warnings.warn(f"Failed to make space in L2: {str(e)}")
            return False
    
    def _make_space_lfu_disk(self, required_bytes: int) -> bool:
        """Libera espacio en L2 usando LFU."""
        # Implementación simplificada
        return self._make_space_lru_disk(required_bytes)
    
    def _invalidate_l2(self, keys: Optional[List[str]]) -> int:
        """Invalida entradas en L2."""
        if keys is None:
            count = len(self._l2_files)
            for key in list(self._l2_files.keys()):
                self._remove_from_l2(key)
            return count
        
        count = 0
        for key in keys:
            if self._remove_from_l2(key):
                count += 1
        
        return count
    
    def _clear_l2_cache(self) -> int:
        """Limpia completamente la caché L2."""
        count = len(self._l2_files)
        
        # Eliminar todos los archivos
        for file_path in self._l2_files.values():
            try:
                file_path.unlink(missing_ok=True)
            except:
                pass
        
        # Limpiar índice y estadísticas
        self._l2_files.clear()
        self._stats[CacheLevel.L2].current_entries = 0
        self._stats[CacheLevel.L2].current_size_bytes = 0
        
        return count
    
    def _load_l2_cache(self) -> int:
        """Carga caché L2 desde disco."""
        try:
            count = 0
            total_size = 0
            
            for file_path in self.cache_dir.glob("l2_*.pkl"):
                try:
                    with open(file_path, 'rb') as f:
                        entry = pickle.load(f)
                    
                    # Verificar expiración
                    if entry.is_expired():
                        file_path.unlink()
                        continue
                    
                    # Añadir al índice
                    key = entry.key
                    self._l2_files[key] = file_path
                    
                    # Actualizar contadores
                    count += 1
                    total_size += file_path.stat().st_size
                    
                except:
                    # Archivo corrupto, eliminarlo
                    file_path.unlink(missing_ok=True)
            
            # Actualizar estadísticas
            self._stats[CacheLevel.L2].current_entries = count
            self._stats[CacheLevel.L2].current_size_bytes = total_size
            
            return count
            
        except Exception as e:
            warnings.warn(f"Failed to load L2 cache: {str(e)}")
            return 0
    
    def _optimize_l2(self) -> Dict[str, Any]:
        """Optimiza caché L2."""
        results = {
            "entries_before": len(self._l2_files),
            "entries_after": 0,
            "expired_removed": 0,
            "size_freed_bytes": 0
        }
        
        # Eliminar expirados
        self._clean_expired_l2()
        
        # Eliminar archivos corruptos
        corrupted_keys = []
        for key, file_path in list(self._l2_files.items()):
            if not file_path.exists():
                corrupted_keys.append(key)
            else:
                try:
                    with open(file_path, 'rb') as f:
                        pickle.load(f)
                except:
                    corrupted_keys.append(key)
        
        for key in corrupted_keys:
            self._remove_from_l2(key)
        
        results["entries_after"] = len(self._l2_files)
        
        return results
    
    # Métodos privados - L3 Cache (Remoto - Simulado)
    
    def _add_to_l3(self, entry: CacheEntry) -> bool:
        """Añade entrada a caché L3 (simulado)."""
        # Simulación de caché remoto
        self._l3_remote[entry.key] = entry
        
        # Actualizar estadísticas
        self._stats[CacheLevel.L3].current_entries += 1
        self._stats[CacheLevel.L3].current_size_bytes += entry.size_bytes
        
        return True
    
    def _invalidate_l3(self, keys: Optional[List[str]]) -> int:
        """Invalida entradas en L3."""
        if keys is None:
            count = len(self._l3_remote)
            self._l3_remote.clear()
            self._stats[CacheLevel.L3].current_entries = 0
            self._stats[CacheLevel.L3].current_size_bytes = 0
            return count
        
        count = 0
        for key in keys:
            if key in self._l3_remote:
                entry = self._l3_remote[key]
                del self._l3_remote[key]
                
                # Actualizar estadísticas
                self._stats[CacheLevel.L3].current_entries -= 1
                self._stats[CacheLevel.L3].current_size_bytes -= entry.size_bytes
                
                count += 1
        
        return count
    
    def _optimize_l3(self) -> Dict[str, Any]:
        """Optimiza caché L3."""
        results = {
            "entries_before": len(self._l3_remote),
            "entries_after": 0,
            "expired_removed": 0
        }
        
        # Eliminar expirados
        expired_keys = []
        for key, entry in self._l3_remote.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._l3_remote[key]
        
        results["expired_removed"] = len(expired_keys)
        results["entries_after"] = len(self._l3_remote)
        
        return results
    
    # Métodos privados - Comunes
    
    def _calculate_size(self, value: Any) -> int:
        """Calcula tamaño aproximado en bytes de un valor."""
        try:
            if isinstance(value, (list, tuple)):
                # Estimación para listas de floats
                return len(value) * 8  # 8 bytes por float
            elif isinstance(value, dict):
                return len(str(value).encode('utf-8'))
            else:
                return len(pickle.dumps(value))
        except:
            return 1024  # Tamaño por defecto
    
    def _make_space_lru(self, level: CacheLevel, required_bytes: int) -> bool:
        """Libera espacio usando política LRU."""
        if level == CacheLevel.L1:
            cache = self._l1_cache
        else:
            return False
        
        freed_bytes = 0
        while cache and freed_bytes < required_bytes:
            # LRU: eliminar el primero (menos recientemente usado)
            key, entry = next(iter(cache.items()))
            self._remove_from_l1(key)
            freed_bytes += entry.size_bytes
        
        return freed_bytes >= required_bytes
    
    def _make_space_lfu(self, level: CacheLevel, required_bytes: int) -> bool:
        """Libera espacio usando política LFU."""
        if level == CacheLevel.L1:
            # Ordenar por conteo de accesos
            sorted_entries = sorted(
                self._l1_cache.items(),
                key=lambda x: x[1].access_count
            )
            
            freed_bytes = 0
            for key, entry in sorted_entries:
                if freed_bytes >= required_bytes:
                    break
                
                self._remove_from_l1(key)
                freed_bytes += entry.size_bytes
            
            return freed_bytes >= required_bytes
        
        return False
    
    def _make_space_fifo(self, level: CacheLevel, required_bytes: int) -> bool:
        """Libera espacio usando política FIFO."""
        # Similar a LRU pero sin mover al final al acceder
        return self._make_space_lru(level, required_bytes)
    
    def _make_space_ttl(self, level: CacheLevel, required_bytes: int) -> bool:
        """Libera espacio usando política TTL."""
        if level == CacheLevel.L1:
            # Eliminar expirados primero
            expired_keys = []
            for key, entry in self._l1_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            freed_bytes = 0
            for key in expired_keys:
                entry = self._l1_cache[key]
                self._remove_from_l1(key)
                freed_bytes += entry.size_bytes
            
            # Si todavía no hay espacio, usar LRU
            if freed_bytes < required_bytes:
                return self._make_space_lru(level, required_bytes - freed_bytes)
            
            return True
        
        return False
    
    def _get_level_stats(self, level: CacheLevel) -> Dict[str, Any]:
        """Obtiene estadísticas de un nivel específico."""
        stats = self._stats[level]
        config = self.levels[level]
        
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "current_entries": stats.current_entries,
            "max_entries": config.max_entries,
            "current_size_bytes": stats.current_size_bytes,
            "max_size_bytes": config.max_size_bytes,
            "usage_percentage": (
                (stats.current_size_bytes / config.max_size_bytes * 100)
                if config.max_size_bytes > 0 else 0
            ),
            "evictions": stats.evictions,
            "avg_access_time_ms": stats.avg_access_time_ms,
            "enabled": config.enabled,
            "policy": config.policy.value,
            "ttl_seconds": config.ttl_seconds
        }
    
    def _get_global_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas globales."""
        total_hits = sum(s.hits for s in self._stats.values())
        total_misses = sum(s.misses for s in self._stats.values())
        total_requests = total_hits + total_misses
        
        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_requests": total_requests,
            "global_hit_rate": total_hits / total_requests if total_requests > 0 else 0,
            "total_entries": sum(s.current_entries for s in self._stats.values()),
            "total_size_bytes": sum(s.current_size_bytes for s in self._stats.values())
        }
    
    def _update_access_time(self, level: CacheLevel, start_time: datetime) -> None:
        """Actualiza tiempo promedio de acceso."""
        access_time = (datetime.now() - start_time).total_seconds() * 1000
        stats = self._stats[level]
        
        total_accesses = stats.hits + stats.misses
        if total_accesses > 0:
            stats.avg_access_time_ms = (
                (stats.avg_access_time_ms * (total_accesses - 1) + access_time) / total_accesses
            )

# Ejemplo de uso
if __name__ == "__main__":
    # Crear caché
    cache = EmbeddingCache("./test_data/cache")
    
    # Almacenar algunos embeddings
    embeddings = {
        "doc1": [0.1, 0.2, 0.3, 0.4],
        "doc2": [0.5, 0.6, 0.7, 0.8],
        "doc3": [0.9, 0.1, 0.2, 0.3]
    }
    
    for key, value in embeddings.items():
        cache.cache_embedding(key, value, ttl_seconds=60, level=CacheLevel.L1)
    
    # Obtener de caché
    cached = cache.get_cached_embedding("doc1")
    print(f"Retrieved from cache: {cached is not None}")
    
    # Obtener estadísticas
    stats = cache.get_cache_stats()
    print(f"L1 hit rate: {stats['L1']['hit_rate']:.2%}")
    
    # Optimizar caché
    optimization = cache.optimize_cache()
    print(f"Optimization results: {optimization}")
    
    # Limpiar caché
    cleared = cache.clear_cache(CacheLevel.L1)
    print(f"Cleared {cleared[CacheLevel.L1]} entries from L1")