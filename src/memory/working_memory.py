"""
WorkingMemory - Memoria de trabajo para almacenamiento temporal y gestión de atención.
Sistema para mantener información activa durante el procesamiento actual.
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Deque
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from datetime import datetime, timedelta
from collections import deque
import json
import uuid
import heapq

from ..core.exceptions import MemoryException, ValidationError

class AttentionLevel(Enum):
    """Niveles de atención para elementos en memoria de trabajo."""
    FOCUSED = "focused"      # Máxima atención, procesamiento activo
    ACTIVE = "active"        # Atención activa, disponible rápidamente
    BACKGROUND = "background" # En segundo plano, puede ser recuperado
    PERIPHERAL = "peripheral" # Periférico, pronto será descartado

class WorkingMemoryItem:
    """Elemento en memoria de trabajo."""
    
    def __init__(
        self,
        content: Any,
        item_type: str = "generic",
        attention_level: AttentionLevel = AttentionLevel.ACTIVE,
        ttl_seconds: Optional[float] = None,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.item_type = item_type
        self.attention_level = attention_level
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.ttl_seconds = ttl_seconds
        self.importance = max(0.0, min(1.0, importance))
        self.metadata = metadata or {}
        
        # Para gestión de atención
        self.attention_score = self._calculate_attention_score()
    
    def _calculate_attention_score(self) -> float:
        """Calcula score de atención basado en múltiples factores."""
        score = 0.0
        
        # Factor de nivel de atención
        attention_factors = {
            AttentionLevel.FOCUSED: 1.0,
            AttentionLevel.ACTIVE: 0.7,
            AttentionLevel.BACKGROUND: 0.4,
            AttentionLevel.PERIPHERAL: 0.1
        }
        score += attention_factors.get(self.attention_level, 0.5)
        
        # Factor de importancia
        score += 0.3 * self.importance
        
        # Factor de frescura (más reciente = más alto)
        age_seconds = (datetime.now() - self.created_at).total_seconds()
        freshness = max(0.0, 1.0 - (age_seconds / 3600))  # Decae en 1 hora
        score += 0.2 * freshness
        
        return min(1.0, score)
    
    def access(self) -> None:
        """Registra un acceso al elemento."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        self.attention_score = self._calculate_attention_score()
    
    def update_attention(self, new_level: AttentionLevel) -> None:
        """Actualiza el nivel de atención del elemento."""
        self.attention_level = new_level
        self.attention_score = self._calculate_attention_score()
    
    def is_expired(self) -> bool:
        """Verifica si el elemento ha expirado por TTL."""
        if self.ttl_seconds is None:
            return False
        age_seconds = (datetime.now() - self.created_at).total_seconds()
        return age_seconds > self.ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            "id": self.id,
            "content": self.content,
            "item_type": self.item_type,
            "attention_level": self.attention_level.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "ttl_seconds": self.ttl_seconds,
            "importance": self.importance,
            "metadata": self.metadata,
            "attention_score": self.attention_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkingMemoryItem':
        """Crea desde diccionario."""
        item = cls(
            content=data["content"],
            item_type=data["item_type"],
            attention_level=AttentionLevel(data["attention_level"]),
            ttl_seconds=data["ttl_seconds"],
            importance=data["importance"],
            metadata=data["metadata"]
        )
        
        # Restaurar campos específicos
        item.id = data["id"]
        item.created_at = datetime.fromisoformat(data["created_at"])
        item.last_accessed = datetime.fromisoformat(data["last_accessed"])
        item.access_count = data["access_count"]
        item.attention_score = data["attention_score"]
        
        return item

class WorkingMemory:
    """
    Sistema de memoria de trabajo para gestión de información activa.
    
    Características:
    1. Capacidad limitada con gestión inteligente de espacio
    2. Múltiples niveles de atención (focused, active, background, peripheral)
    3. Descarte automático basado en TTL y relevancia
    4. Foco de atención y cambio de contexto
    5. Búsqueda rápida por tipo y contenido
    6. Integración con otras memorias (promoción a largo plazo)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa la memoria de trabajo.
        
        Args:
            config: Configuración de la memoria
        """
        self.config = config or self._default_config()
        
        # Almacenamiento de elementos
        self.items: Dict[str, WorkingMemoryItem] = {}
        
        # Índices para búsqueda rápida
        self.indices = {
            "by_type": defaultdict(list),
            "by_attention_level": defaultdict(list),
            "recently_accessed": deque(maxlen=100)  # Cola de IDs recientes
        }
        
        # Capacidad y límites
        self.capacity = self.config["capacity"]
        self.current_load = 0
        
        # Elemento enfocado actualmente
        self.focused_item: Optional[str] = None
        
        # Contexto actual
        self.context: Dict[str, Any] = {
            "session_id": str(uuid.uuid4()),
            "start_time": datetime.now(),
            "current_task": None,
            "current_goal": None,
            "recent_actions": deque(maxlen=10)
        }
        
        # Estadísticas
        self.stats = {
            "total_items_added": 0,
            "current_item_count": 0,
            "items_discarded": 0,
            "avg_attention_score": 0.0,
            "focus_changes": 0,
            "context_switches": 0
        }
        
        # Bloqueo para concurrencia
        self._lock = asyncio.Lock()
        
        # Iniciar limpieza periódica
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto de memoria de trabajo."""
        return {
            "capacity": 100,  # Máximo número de items
            "attention_levels": {
                "focused": {"max_items": 1, "ttl_seconds": 300},
                "active": {"max_items": 10, "ttl_seconds": 1800},
                "background": {"max_items": 30, "ttl_seconds": 3600},
                "peripheral": {"max_items": 59, "ttl_seconds": 7200}
            },
            "auto_cleanup_interval": 60,  # Limpieza cada 60 segundos
            "promotion_threshold": 0.8,   # Umbral para promover a memoria a largo plazo
            "default_ttl": 3600,  # TTL por defecto: 1 hora
            "enable_context_tracking": True,
            "max_context_history": 10
        }
    
    async def add_to_working_memory(
        self,
        content: Any,
        item_type: str = "generic",
        attention_level: Optional[AttentionLevel] = None,
        ttl_seconds: Optional[float] = None,
        importance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        auto_focus: bool = False
    ) -> str:
        """
        Añade un elemento a la memoria de trabajo.
        
        Args:
            content: Contenido a almacenar
            item_type: Tipo del elemento
            attention_level: Nivel de atención (None = auto-determinado)
            ttl_seconds: Tiempo de vida en segundos
            importance: Importancia (0.0-1.0)
            metadata: Metadatos adicionales
            auto_focus: Si True, enfoca automáticamente este elemento
            
        Returns:
            str: ID del elemento creado
        """
        async with self._lock:
            # Determinar nivel de atención si no se especifica
            if attention_level is None:
                attention_level = self._determine_attention_level(
                    item_type, importance, metadata
                )
            
            # Usar TTL por defecto si no se especifica
            if ttl_seconds is None:
                ttl_seconds = self.config["default_ttl"]
            
            # Crear elemento
            item = WorkingMemoryItem(
                content=content,
                item_type=item_type,
                attention_level=attention_level,
                ttl_seconds=ttl_seconds,
                importance=importance,
                metadata=metadata or {}
            )
            
            # Verificar capacidad
            if len(self.items) >= self.capacity:
                await self._make_space_for_new_item()
            
            # Almacenar elemento
            self.items[item.id] = item
            
            # Actualizar índices
            self._update_indices(item)
            
            # Actualizar estadísticas
            self.stats["total_items_added"] += 1
            self.stats["current_item_count"] = len(self.items)
            
            # Enfocar automáticamente si se solicita
            if auto_focus:
                await self.focus_attention(item.id)
            
            # Recalcular atención promedio
            self._update_attention_stats()
            
            return item.id
    
    async def remove_from_working_memory(self, item_id: str) -> bool:
        """
        Elimina un elemento específico de la memoria de trabajo.
        
        Args:
            item_id: ID del elemento a eliminar
            
        Returns:
            bool: True si se eliminó exitosamente
        """
        async with self._lock:
            if item_id not in self.items:
                return False
            
            item = self.items[item_id]
            
            # Eliminar de índices
            self._remove_from_indices(item)
            
            # Si era el elemento enfocado, limpiar foco
            if self.focused_item == item_id:
                self.focused_item = None
            
            # Eliminar elemento
            del self.items[item_id]
            
            # Actualizar estadísticas
            self.stats["current_item_count"] = len(self.items)
            self.stats["items_discarded"] += 1
            
            # Recalcular atención promedio
            self._update_attention_stats()
            
            return True
    
    def get_working_memory_contents(
        self,
        attention_level: Optional[AttentionLevel] = None,
        item_type: Optional[str] = None,
        min_importance: float = 0.0,
        max_age_seconds: Optional[float] = None,
        limit: int = 100,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Obtiene el contenido actual de la memoria de trabajo.
        
        Args:
            attention_level: Filtrar por nivel de atención
            item_type: Filtrar por tipo de elemento
            min_importance: Importancia mínima
            max_age_seconds: Edad máxima en segundos
            limit: Máximo de resultados
            include_expired: Incluir elementos expirados
            
        Returns:
            Lista de elementos con sus metadatos
        """
        # No necesita lock ya que es solo lectura y usamos copias
        results = []
        current_time = datetime.now()
        
        for item in self.items.values():
            # Filtrar por nivel de atención
            if attention_level and item.attention_level != attention_level:
                continue
            
            # Filtrar por tipo
            if item_type and item.item_type != item_type:
                continue
            
            # Filtrar por importancia
            if item.importance < min_importance:
                continue
            
            # Filtrar por edad
            if max_age_seconds:
                age_seconds = (current_time - item.created_at).total_seconds()
                if age_seconds > max_age_seconds:
                    continue
            
            # Filtrar expirados
            if not include_expired and item.is_expired():
                continue
            
            # Añadir a resultados
            results.append({
                "item": item.to_dict(),
                "context": self._get_item_context(item)
            })
            
            if len(results) >= limit:
                break
        
        # Ordenar por score de atención (descendente)
        results.sort(key=lambda x: x["item"]["attention_score"], reverse=True)
        
        return results
    
    async def clear_working_memory(
        self,
        attention_level: Optional[AttentionLevel] = None,
        item_type: Optional[str] = None,
        max_age_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Limpia la memoria de trabajo según criterios.
        
        Args:
            attention_level: Limpiar solo este nivel de atención
            item_type: Limpiar solo este tipo de elementos
            max_age_seconds: Limpiar elementos más viejos que esto
            
        Returns:
            Dict con resultados de la limpieza
        """
        async with self._lock:
            items_to_remove = []
            current_time = datetime.now()
            
            for item_id, item in self.items.items():
                # Filtrar por nivel de atención
                if attention_level and item.attention_level != attention_level:
                    continue
                
                # Filtrar por tipo
                if item_type and item.item_type != item_type:
                    continue
                
                # Filtrar por edad
                if max_age_seconds:
                    age_seconds = (current_time - item.created_at).total_seconds()
                    if age_seconds <= max_age_seconds:
                        continue
                
                items_to_remove.append(item_id)
            
            # Eliminar elementos
            removed_count = 0
            for item_id in items_to_remove:
                if await self.remove_from_working_memory(item_id):
                    removed_count += 1
            
            # Limpiar foco si el elemento enfocado fue eliminado
            if (self.focused_item and 
                self.focused_item not in self.items):
                self.focused_item = None
            
            return {
                "removed_count": removed_count,
                "remaining_count": len(self.items),
                "filters_applied": {
                    "attention_level": attention_level.value if attention_level else None,
                    "item_type": item_type,
                    "max_age_seconds": max_age_seconds
                }
            }
    
    async def focus_attention(self, item_id: str) -> bool:
        """
        Enfoca la atención en un elemento específico.
        
        Args:
            item_id: ID del elemento a enfocar
            
        Returns:
            bool: True si se cambió el foco exitosamente
        """
        async with self._lock:
            if item_id not in self.items:
                return False
            
            item = self.items[item_id]
            
            # Si ya está enfocado, solo actualizar acceso
            if self.focused_item == item_id:
                item.access()
                return True
            
            # Actualizar nivel de atención del elemento anterior
            if self.focused_item and self.focused_item in self.items:
                old_item = self.items[self.focused_item]
                old_item.update_attention(AttentionLevel.ACTIVE)
                
                # Actualizar índice
                self._update_attention_index(old_item)
            
            # Enfocar nuevo elemento
            item.update_attention(AttentionLevel.FOCUSED)
            item.access()
            self.focused_item = item_id
            
            # Actualizar índice
            self._update_attention_index(item)
            
            # Actualizar estadísticas
            self.stats["focus_changes"] += 1
            
            # Actualizar contexto
            self._update_context_with_focus_change(item)
            
            return True
    
    async def update_working_memory(
        self,
        item_id: str,
        content: Optional[Any] = None,
        attention_level: Optional[AttentionLevel] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        merge_metadata: bool = True
    ) -> bool:
        """
        Actualiza un elemento existente en la memoria de trabajo.
        
        Args:
            item_id: ID del elemento
            content: Nuevo contenido (opcional)
            attention_level: Nuevo nivel de atención (opcional)
            importance: Nueva importancia (opcional)
            metadata: Nuevos metadatos (opcional)
            merge_metadata: Si True, fusiona con metadatos existentes
            
        Returns:
            bool: True si se actualizó exitosamente
        """
        async with self._lock:
            if item_id not in self.items:
                return False
            
            item = self.items[item_id]
            item.access()
            
            # Actualizar contenido si se proporciona
            if content is not None:
                item.content = content
            
            # Actualizar nivel de atención si se proporciona
            if attention_level is not None:
                old_level = item.attention_level
                item.update_attention(attention_level)
                
                # Actualizar índice si cambió el nivel
                if old_level != attention_level:
                    self._update_attention_index(item, old_level=old_level)
            
            # Actualizar importancia si se proporciona
            if importance is not None:
                item.importance = max(0.0, min(1.0, importance))
            
            # Actualizar metadatos
            if metadata is not None:
                if merge_metadata:
                    item.metadata.update(metadata)
                else:
                    item.metadata = metadata
            
            # Recalcular score de atención
            item.attention_score = item._calculate_attention_score()
            
            return True
    
    def check_working_memory_limit(self) -> Dict[str, Any]:
        """
        Verifica los límites de capacidad de la memoria de trabajo.
        
        Returns:
            Dict con estado de capacidad y recomendaciones
        """
        total_items = len(self.items)
        capacity = self.capacity
        
        # Calcular uso por nivel de atención
        usage_by_level = {}
        for level in AttentionLevel:
            count = len(self.indices["by_attention_level"].get(level, []))
            max_items = self.config["attention_levels"].get(
                level.value, {}
            ).get("max_items", float('inf'))
            
            usage_by_level[level.value] = {
                "count": count,
                "max": max_items if max_items != float('inf') else "unlimited",
                "percentage": (count / max_items * 100) if max_items != float('inf') else 0
            }
        
        # Calcar carga actual
        current_load = total_items / capacity * 100 if capacity > 0 else 0
        
        # Determinar estado
        if current_load >= 90:
            status = "critical"
            recommendation = "Clear memory immediately"
        elif current_load >= 70:
            status = "warning"
            recommendation = "Consider clearing peripheral items"
        elif current_load >= 50:
            status = "moderate"
            recommendation = "Monitor memory usage"
        else:
            status = "healthy"
            recommendation = "Memory usage is optimal"
        
        return {
            "status": status,
            "current_load_percentage": current_load,
            "total_items": total_items,
            "capacity": capacity,
            "usage_by_attention_level": usage_by_level,
            "recommendation": recommendation,
            "focused_item": self.focused_item,
            "context": self.context.copy()
        }
    
    # Métodos auxiliares protegidos
    
    def _determine_attention_level(
        self, 
        item_type: str, 
        importance: float,
        metadata: Optional[Dict[str, Any]]
    ) -> AttentionLevel:
        """Determina el nivel de atención inicial para un elemento."""
        # Reglas basadas en tipo
        type_rules = {
            "error": AttentionLevel.FOCUSED,
            "critical": AttentionLevel.FOCUSED,
            "query": AttentionLevel.ACTIVE,
            "result": AttentionLevel.ACTIVE,
            "intermediate": AttentionLevel.BACKGROUND,
            "cache": AttentionLevel.BACKGROUND,
            "log": AttentionLevel.PERIPHERAL,
            "debug": AttentionLevel.PERIPHERAL
        }
        
        # Verificar reglas de tipo
        if item_type in type_rules:
            return type_rules[item_type]
        
        # Basado en importancia
        if importance > 0.8:
            return AttentionLevel.FOCUSED
        elif importance > 0.5:
            return AttentionLevel.ACTIVE
        elif importance > 0.2:
            return AttentionLevel.BACKGROUND
        else:
            return AttentionLevel.PERIPHERAL
    
    async def _make_space_for_new_item(self) -> None:
        """Libera espacio para un nuevo elemento."""
        # Priorizar eliminación de elementos periféricos expirados
        items_to_remove = []
        
        for item_id, item in self.items.items():
            if (item.attention_level == AttentionLevel.PERIPHERAL and 
                item.is_expired()):
                items_to_remove.append(item_id)
        
        # Si no hay suficientes elementos periféricos expirados,
        # eliminar los periféricos con menor score de atención
        if len(items_to_remove) < 5:  # Queremos al menos 5 espacios
            peripheral_items = [
                (item_id, item) 
                for item_id, item in self.items.items()
                if item.attention_level == AttentionLevel.PERIPHERAL
            ]
            
            # Ordenar por score de atención (ascendente)
            peripheral_items.sort(key=lambda x: x[1].attention_score)
            
            # Tomar los elementos con menor score
            additional_needed = 5 - len(items_to_remove)
            additional_items = peripheral_items[:additional_needed]
            items_to_remove.extend([item_id for item_id, _ in additional_items])
        
        # Eliminar elementos seleccionados
        for item_id in items_to_remove[:5]:  # Máximo 5 a la vez
            await self.remove_from_working_memory(item_id)
    
    def _update_indices(self, item: WorkingMemoryItem) -> None:
        """Actualiza todos los índices para un elemento."""
        # Índice por tipo
        self.indices["by_type"][item.item_type].append(item.id)
        
        # Índice por nivel de atención
        self.indices["by_attention_level"][item.attention_level].append(item.id)
        
        # Índice de recientemente accedidos
        if item.id in self.indices["recently_accessed"]:
            self.indices["recently_accessed"].remove(item.id)
        self.indices["recently_accessed"].append(item.id)
    
    def _remove_from_indices(self, item: WorkingMemoryItem) -> None:
        """Elimina un elemento de todos los índices."""
        # Índice por tipo
        if item.id in self.indices["by_type"][item.item_type]:
            self.indices["by_type"][item.item_type].remove(item.id)
        
        # Índice por nivel de atención
        if item.id in self.indices["by_attention_level"][item.attention_level]:
            self.indices["by_attention_level"][item.attention_level].remove(item.id)
        
        # Índice de recientemente accedidos
        if item.id in self.indices["recently_accessed"]:
            self.indices["recently_accessed"].remove(item.id)
    
    def _update_attention_index(
        self, 
        item: WorkingMemoryItem, 
        old_level: Optional[AttentionLevel] = None
    ) -> None:
        """Actualiza el índice de nivel de atención para un elemento."""
        # Remover del nivel anterior si se especifica
        if old_level is not None:
            if item.id in self.indices["by_attention_level"][old_level]:
                self.indices["by_attention_level"][old_level].remove(item.id)
        
        # Añadir al nuevo nivel
        self.indices["by_attention_level"][item.attention_level].append(item.id)
    
    def _update_attention_stats(self) -> None:
        """Actualiza estadísticas de atención."""
        if not self.items:
            self.stats["avg_attention_score"] = 0.0
            return
        
        total_score = sum(item.attention_score for item in self.items.values())
        self.stats["avg_attention_score"] = total_score / len(self.items)
    
    def _update_context_with_focus_change(self, item: WorkingMemoryItem) -> None:
        """Actualiza el contexto con un cambio de foco."""
        if not self.config["enable_context_tracking"]:
            return
        
        # Actualizar contexto actual
        self.context["current_focus"] = {
            "item_id": item.id,
            "item_type": item.item_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Añadir a historial de acciones
        self.context["recent_actions"].append({
            "action": "focus_change",
            "item_id": item.id,
            "item_type": item.item_type,
            "timestamp": datetime.now().isoformat()
        })
    
    def _get_item_context(self, item: WorkingMemoryItem) -> Dict[str, Any]:
        """Obtiene contexto relevante para un elemento."""
        context = {
            "is_focused": self.focused_item == item.id,
            "age_seconds": (datetime.now() - item.created_at).total_seconds(),
            "time_since_last_access": (datetime.now() - item.last_accessed).total_seconds(),
            "attention_rank": None
        }
        
        # Calcular rango de atención entre elementos del mismo nivel
        same_level_items = [
            (iid, self.items[iid]) 
            for iid in self.indices["by_attention_level"].get(item.attention_level, [])
            if iid in self.items
        ]
        
        if same_level_items:
            # Ordenar por score de atención
            same_level_items.sort(key=lambda x: x[1].attention_score, reverse=True)
            
            # Encontrar posición de este elemento
            for rank, (iid, _) in enumerate(same_level_items, 1):
                if iid == item.id:
                    context["attention_rank"] = rank
                    context["total_in_level"] = len(same_level_items)
                    break
        
        return context
    
    async def _periodic_cleanup(self) -> None:
        """Tarea periódica para limpiar memoria de trabajo."""
        try:
            while True:
                await asyncio.sleep(self.config["auto_cleanup_interval"])
                
                async with self._lock:
                    # Limpiar elementos expirados
                    expired_count = 0
                    items_to_remove = []
                    
                    for item_id, item in self.items.items():
                        if item.is_expired():
                            items_to_remove.append(item_id)
                    
                    for item_id in items_to_remove:
                        if await self.remove_from_working_memory(item_id):
                            expired_count += 1
                    
                    # Degradar elementos inactivos
                    degraded_count = 0
                    current_time = datetime.now()
                    
                    for item_id, item in self.items.items():
                        # Elementos activos sin acceso reciente -> background
                        if (item.attention_level == AttentionLevel.ACTIVE and
                            (current_time - item.last_accessed).total_seconds() > 300):  # 5 minutos
                            item.update_attention(AttentionLevel.BACKGROUND)
                            self._update_attention_index(item, old_level=AttentionLevel.ACTIVE)
                            degraded_count += 1
                        
                        # Elementos background sin acceso reciente -> peripheral
                        elif (item.attention_level == AttentionLevel.BACKGROUND and
                              (current_time - item.last_accessed).total_seconds() > 900):  # 15 minutos
                            item.update_attention(AttentionLevel.PERIPHERAL)
                            self._update_attention_index(item, old_level=AttentionLevel.BACKGROUND)
                            degraded_count += 1
                    
                    # Promover elementos importantes a memoria a largo plazo
                    promoted_count = 0
                    if self.config["promotion_threshold"] > 0:
                        for item_id, item in list(self.items.items()):
                            if (item.attention_score >= self.config["promotion_threshold"] and
                                item.attention_level in [AttentionLevel.FOCUSED, AttentionLevel.ACTIVE]):
                                
                                # Aquí se integraría con otros módulos de memoria
                                # Por ahora solo registramos la intención
                                if item.metadata.get("promotable", True):
                                    promoted_count += 1
                    
                    # Actualizar estadísticas de limpieza
                    if expired_count > 0 or degraded_count > 0 or promoted_count > 0:
                        self.stats["last_cleanup"] = {
                            "timestamp": datetime.now().isoformat(),
                            "expired_removed": expired_count,
                            "items_degraded": degraded_count,
                            "items_promoted": promoted_count,
                            "remaining_items": len(self.items)
                        }
                        
        except asyncio.CancelledError:
            # Tarea cancelada, salir limpiamente
            pass
        except Exception as e:
            print(f"Error in working memory cleanup task: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de la memoria de trabajo."""
        stats = self.stats.copy()
        
        # Agregar detalles adicionales
        stats["current_state"] = {
            "total_items": len(self.items),
            "focused_item": self.focused_item,
            "has_focus": self.focused_item is not None,
            "context_session_id": self.context.get("session_id"),
            "session_duration_seconds": (
                datetime.now() - self.context.get("start_time", datetime.now())
            ).total_seconds()
        }
        
        # Distribución por nivel de atención
        attention_distribution = {}
        for level in AttentionLevel:
            count = len(self.indices["by_attention_level"].get(level, []))
            attention_distribution[level.value] = {
                "count": count,
                "percentage": (count / len(self.items) * 100) if self.items else 0
            }
        stats["attention_distribution"] = attention_distribution
        
        # Distribución por tipo
        type_distribution = {}
        for item_type, item_ids in self.indices["by_type"].items():
            valid_ids = [iid for iid in item_ids if iid in self.items]
            if valid_ids:
                type_distribution[item_type] = len(valid_ids)
        stats["type_distribution"] = type_distribution
        
        # Elementos recientemente accedidos
        recent_items = []
        for item_id in list(self.indices["recently_accessed"])[-10:]:  # Últimos 10
            if item_id in self.items:
                item = self.items[item_id]
                recent_items.append({
                    "id": item.id,
                    "type": item.item_type,
                    "attention_level": item.attention_level.value,
                    "last_accessed": item.last_accessed.isoformat(),
                    "attention_score": item.attention_score
                })
        stats["recently_accessed_items"] = recent_items
        
        # Capacidad y límites
        stats["capacity_info"] = self.check_working_memory_limit()
        
        # Contexto actual
        if self.config["enable_context_tracking"]:
            stats["current_context"] = {
                "task": self.context.get("current_task"),
                "goal": self.context.get("current_goal"),
                "recent_action_count": len(self.context.get("recent_actions", [])),
                "session_start": self.context.get("start_time").isoformat() 
                    if self.context.get("start_time") else None
            }
        
        return stats
    
    async def shutdown(self) -> None:
        """Apaga la memoria de trabajo de manera controlada."""
        # Cancelar tarea de limpieza
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Limpiar memoria
        await self.clear_working_memory()