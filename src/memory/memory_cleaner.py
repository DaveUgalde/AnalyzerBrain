"""
MemoryCleaner - Sistema de limpieza y mantenimiento de memoria.
Responsable de limpiar, comprimir y archivar recuerdos antiguos o irrelevantes.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import json
import zlib
import pickle
from ..core.exceptions import MemoryException, ValidationError

class CleanupStrategy(str, Enum):
    """Estrategias de limpieza de memoria."""
    AGE_BASED = "age_based"           # Basado en antigüedad
    RELEVANCE_BASED = "relevance_based" # Basado en relevancia
    FREQUENCY_BASED = "frequency_based" # Basado en frecuencia de uso
    SIZE_BASED = "size_based"         # Basado en tamaño
    HYBRID = "hybrid"                 # Combinación de estrategias

class CleanupAction(str, Enum):
    """Acciones de limpieza disponibles."""
    DELETE = "delete"                 # Eliminar permanentemente
    ARCHIVE = "archive"              # Archivar en almacenamiento frío
    COMPRESS = "compress"            # Comprimir para ahorrar espacio
    SUMMARIZE = "summarize"          # Crear resumen y eliminar detalles
    MOVE_TO_SLOWER_STORAGE = "move_to_slower_storage"  # Mover a almacenamiento más lento

@dataclass
class MemoryItem:
    """Elemento de memoria con metadatos de limpieza."""
    id: str
    content: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    relevance_score: float = 1.0  # 0.0 - 1.0
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    cleanup_metadata: Dict[str, Any] = field(default_factory=dict)

class MemoryCleanerConfig(BaseModel):
    """Configuración del limpiador de memoria."""
    enabled: bool = True
    cleanup_interval_seconds: int = 3600  # Cada hora
    strategies: List[CleanupStrategy] = Field(
        default_factory=lambda: [CleanupStrategy.AGE_BASED, CleanupStrategy.RELEVANCE_BASED]
    )
    
    # Umbrales por estrategia
    age_threshold_days: int = 90
    relevance_threshold: float = 0.3
    min_access_count: int = 1
    max_total_size_mb: int = 1024
    
    # Acciones por defecto
    default_action: CleanupAction = CleanupAction.COMPRESS
    aggressive_cleanup: bool = False
    
    # Configuración de compresión
    compression_level: int = 6  # 0-9, donde 9 es máxima compresión
    compression_algorithm: str = "zlib"  # zlib, lz4, gzip
    
    # Retención
    retention_policy: Dict[str, Any] = Field(default_factory=dict)

class MemoryCleaner:
    """
    Sistema de limpieza y mantenimiento de memoria.
    
    Responsabilidades:
    1. Identificar recuerdos candidatos para limpieza
    2. Aplicar estrategias de limpieza según configuración
    3. Comprimir recuerdos para ahorrar espacio
    4. Archivar recuerdos antiguos
    5. Validar integridad de memoria
    6. Reparar recuerdos corruptos
    7. Generar reportes de limpieza
    """
    
    def __init__(self, config: Optional[MemoryCleanerConfig] = None):
        """
        Inicializa el limpiador de memoria.
        
        Args:
            config: Configuración del limpiador (opcional)
        """
        self.config = config or MemoryCleanerConfig()
        self.cleanup_history: List[Dict[str, Any]] = []
        self.memory_items: Dict[str, MemoryItem] = {}
        
        # Métricas
        self.metrics = {
            "cleanup_cycles": 0,
            "items_cleaned": 0,
            "bytes_reclaimed": 0,
            "compression_ratio": 1.0,
            "errors": 0,
            "last_cleanup": None
        }
        
        # Estado
        self.is_cleaning = False
        self.last_cleanup_time = None
        
    def clean_expired_memories(self, memory_store: Dict[str, Any]) -> Dict[str, Any]:
        """
        Limpia recuerdos expirados basados en antigüedad.
        
        Args:
            memory_store: Diccionario de recuerdos a limpiar
            
        Returns:
            Dict con resultados de la limpieza
        """
        try:
            self.is_cleaning = True
            
            cutoff_date = datetime.now() - timedelta(days=self.config.age_threshold_days)
            expired_items = []
            cleaned_items = []
            
            for item_id, item_data in memory_store.items():
                # Convertir a MemoryItem si es necesario
                memory_item = self._create_memory_item(item_id, item_data)
                
                if memory_item.created_at < cutoff_date:
                    expired_items.append(memory_item)
                    
                    # Aplicar acción de limpieza
                    result = self._apply_cleanup_action(memory_item, CleanupAction.ARCHIVE)
                    
                    if result["success"]:
                        cleaned_items.append({
                            "id": item_id,
                            "action": "archived",
                            "size_bytes": memory_item.size_bytes,
                            "age_days": (datetime.now() - memory_item.created_at).days
                        })
                        
                        # Eliminar del almacenamiento original si se archivó
                        if self.config.aggressive_cleanup:
                            del memory_store[item_id]
            
            # Actualizar métricas
            self.metrics["cleanup_cycles"] += 1
            self.metrics["items_cleaned"] += len(cleaned_items)
            self.metrics["last_cleanup"] = datetime.now().isoformat()
            
            return {
                "status": "completed",
                "strategy": "age_based",
                "expired_items_found": len(expired_items),
                "cleaned_items": cleaned_items,
                "total_bytes_reclaimed": sum(item["size_bytes"] for item in cleaned_items),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.metrics["errors"] += 1
            raise MemoryException(f"Failed to clean expired memories: {e}")
        finally:
            self.is_cleaning = False
    
    def clean_low_relevance_memories(self, memory_store: Dict[str, Any], 
                                   relevance_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Limpia recuerdos con baja relevancia.
        
        Args:
            memory_store: Diccionario de recuerdos a limpiar
            relevance_scores: Diccionario con puntajes de relevancia
            
        Returns:
            Dict con resultados de la limpieza
        """
        try:
            self.is_cleaning = True
            
            low_relevance_items = []
            cleaned_items = []
            
            for item_id, relevance in relevance_scores.items():
                if relevance < self.config.relevance_threshold and item_id in memory_store:
                    item_data = memory_store[item_id]
                    memory_item = self._create_memory_item(item_id, item_data)
                    
                    low_relevance_items.append((memory_item, relevance))
                    
                    # Aplicar acción de limpieza
                    action = CleanupAction.COMPRESS if relevance > 0.1 else CleanupAction.DELETE
                    result = self._apply_cleanup_action(memory_item, action)
                    
                    if result["success"]:
                        cleaned_items.append({
                            "id": item_id,
                            "action": action.value,
                            "relevance_score": relevance,
                            "size_bytes": memory_item.size_bytes
                        })
                        
                        # Eliminar si la acción fue DELETE
                        if action == CleanupAction.DELETE and self.config.aggressive_cleanup:
                            del memory_store[item_id]
            
            # Actualizar métricas
            self.metrics["cleanup_cycles"] += 1
            self.metrics["items_cleaned"] += len(cleaned_items)
            
            return {
                "status": "completed",
                "strategy": "relevance_based",
                "low_relevance_items_found": len(low_relevance_items),
                "cleaned_items": cleaned_items,
                "average_relevance_cleaned": (
                    sum(item["relevance_score"] for item in cleaned_items) / len(cleaned_items)
                    if cleaned_items else 0
                ),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.metrics["errors"] += 1
            raise MemoryException(f"Failed to clean low relevance memories: {e}")
        finally:
            self.is_cleaning = False
    
    def compress_memories(self, memory_store: Dict[str, Any], 
                         target_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Comprime recuerdos para ahorrar espacio.
        
        Args:
            memory_store: Diccionario de recuerdos a comprimir
            target_ratio: Ratio de compresión objetivo
            
        Returns:
            Dict con resultados de la compresión
        """
        try:
            self.is_cleaning = True
            
            original_size = 0
            compressed_size = 0
            compressed_items = []
            
            for item_id, item_data in memory_store.items():
                memory_item = self._create_memory_item(item_id, item_data)
                original_size += memory_item.size_bytes
                
                # Comprimir si el tamaño es significativo
                if memory_item.size_bytes > 1024:  # 1KB mínimo
                    compressed = self._compress_data(item_data)
                    compression_ratio = len(compressed) / memory_item.size_bytes
                    
                    if compression_ratio < target_ratio:
                        # Actualizar con datos comprimidos
                        memory_store[item_id] = {
                            "compressed": True,
                            "data": compressed,
                            "original_size": memory_item.size_bytes,
                            "compression_ratio": compression_ratio,
                            "compressed_at": datetime.now().isoformat()
                        }
                        
                        compressed_size += len(compressed)
                        compressed_items.append({
                            "id": item_id,
                            "original_size": memory_item.size_bytes,
                            "compressed_size": len(compressed),
                            "compression_ratio": compression_ratio
                        })
            
            # Calcular métricas de compresión
            if original_size > 0:
                overall_ratio = compressed_size / original_size
                self.metrics["compression_ratio"] = overall_ratio
            
            return {
                "status": "completed",
                "original_size_bytes": original_size,
                "compressed_size_bytes": compressed_size,
                "compression_ratio": overall_ratio if original_size > 0 else 1.0,
                "items_compressed": len(compressed_items),
                "bytes_saved": original_size - compressed_size,
                "compressed_items": compressed_items,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.metrics["errors"] += 1
            raise MemoryException(f"Failed to compress memories: {e}")
        finally:
            self.is_cleaning = False
    
    def archive_old_memories(self, memory_store: Dict[str, Any], 
                           archive_path: str) -> Dict[str, Any]:
        """
        Archiva recuerdos antiguos en almacenamiento secundario.
        
        Args:
            memory_store: Diccionario de recuerdos a archivar
            archive_path: Ruta para almacenar archivos archivados
            
        Returns:
            Dict con resultados del archivado
        """
        try:
            self.is_cleaning = True
            
            import os
            import json
            
            # Crear directorio de archivo si no existe
            os.makedirs(archive_path, exist_ok=True)
            
            archived_items = []
            archive_file = os.path.join(archive_path, f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            # Identificar recuerdos para archivar (antiguos y poco accedidos)
            cutoff_date = datetime.now() - timedelta(days=180)  # 6 meses
            items_to_archive = []
            
            for item_id, item_data in memory_store.items():
                memory_item = self._create_memory_item(item_id, item_data)
                
                if (memory_item.created_at < cutoff_date and 
                    memory_item.access_count < 3):
                    items_to_archive.append({
                        "id": item_id,
                        "data": item_data,
                        "metadata": {
                            "created_at": memory_item.created_at.isoformat(),
                            "last_accessed": memory_item.last_accessed.isoformat(),
                            "access_count": memory_item.access_count,
                            "size_bytes": memory_item.size_bytes
                        }
                    })
            
            # Guardar en archivo
            if items_to_archive:
                with open(archive_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "archived_at": datetime.now().isoformat(),
                        "item_count": len(items_to_archive),
                        "items": items_to_archive
                    }, f, indent=2)
                
                # Eliminar de memoria principal
                for item in items_to_archive:
                    if item["id"] in memory_store:
                        archived_items.append({
                            "id": item["id"],
                            "archive_file": archive_file,
                            "size_bytes": item["metadata"]["size_bytes"]
                        })
                        del memory_store[item["id"]]
            
            return {
                "status": "completed",
                "archived_items_count": len(archived_items),
                "archive_file": archive_file,
                "archive_size_bytes": os.path.getsize(archive_file) if os.path.exists(archive_file) else 0,
                "archived_items": archived_items,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.metrics["errors"] += 1
            raise MemoryException(f"Failed to archive old memories: {e}")
        finally:
            self.is_cleaning = False
    
    def validate_memory_integrity(self, memory_store: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida la integridad de los recuerdos almacenados.
        
        Args:
            memory_store: Diccionario de recuerdos a validar
            
        Returns:
            Dict con resultados de validación
        """
        try:
            validation_results = {
                "total_items": len(memory_store),
                "valid_items": 0,
                "invalid_items": 0,
                "corrupted_items": [],
                "validation_errors": [],
                "integrity_score": 1.0
            }
            
            for item_id, item_data in memory_store.items():
                try:
                    # Validar estructura básica
                    if not isinstance(item_data, dict):
                        raise ValidationError(f"Item {item_id} is not a dictionary")
                    
                    # Validar campos requeridos
                    required_fields = ["content", "metadata"]
                    for field in required_fields:
                        if field not in item_data:
                            raise ValidationError(f"Item {item_id} missing required field: {field}")
                    
                    # Validar serialización/deserialización
                    test_serialized = json.dumps(item_data)
                    test_deserialized = json.loads(test_serialized)
                    
                    # Validar tamaño consistente
                    size_estimate = len(test_serialized)
                    
                    validation_results["valid_items"] += 1
                    
                except Exception as e:
                    validation_results["invalid_items"] += 1
                    validation_results["corrupted_items"].append({
                        "id": item_id,
                        "error": str(e),
                        "action": "needs_repair"
                    })
                    validation_results["validation_errors"].append(str(e))
            
            # Calcular puntaje de integridad
            if validation_results["total_items"] > 0:
                validation_results["integrity_score"] = (
                    validation_results["valid_items"] / validation_results["total_items"]
                )
            
            return validation_results
            
        except Exception as e:
            raise MemoryException(f"Failed to validate memory integrity: {e}")
    
    def repair_corrupted_memory(self, memory_store: Dict[str, Any], 
                              corrupted_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Intenta reparar recuerdos corruptos.
        
        Args:
            memory_store: Diccionario de recuerdos
            corrupted_items: Lista de items corruptos a reparar
            
        Returns:
            Dict con resultados de reparación
        """
        try:
            repair_results = {
                "attempted_repairs": len(corrupted_items),
                "successful_repairs": 0,
                "failed_repairs": 0,
                "repaired_items": [],
                "repair_errors": []
            }
            
            for corrupted_item in corrupted_items:
                item_id = corrupted_item["id"]
                
                try:
                    if item_id not in memory_store:
                        raise MemoryException(f"Item {item_id} not found in memory store")
                    
                    # Intentar diferentes estrategias de reparación
                    original_item = memory_store[item_id]
                    
                    # Estrategia 1: Reconstruir desde backups si existen
                    repaired = self._attempt_repair_from_backup(item_id, original_item)
                    
                    # Estrategia 2: Extraer partes válidas
                    if not repaired:
                        repaired = self._extract_valid_parts(original_item)
                    
                    # Estrategia 3: Reemplazar con marcador de posición
                    if not repaired:
                        repaired = {
                            "content": f"REPAIRED_PLACEHOLDER_FOR_{item_id}",
                            "metadata": {
                                "original_id": item_id,
                                "repair_status": "placeholder_created",
                                "repair_timestamp": datetime.now().isoformat(),
                                "original_error": corrupted_item.get("error", "unknown")
                            },
                            "repaired": True
                        }
                    
                    # Actualizar en almacenamiento
                    memory_store[item_id] = repaired
                    
                    repair_results["successful_repairs"] += 1
                    repair_results["repaired_items"].append({
                        "id": item_id,
                        "repair_strategy": repaired.get("metadata", {}).get("repair_status", "unknown"),
                        "repair_timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    repair_results["failed_repairs"] += 1
                    repair_results["repair_errors"].append({
                        "item_id": item_id,
                        "error": str(e)
                    })
            
            return repair_results
            
        except Exception as e:
            raise MemoryException(f"Failed to repair corrupted memory: {e}")
    
    def generate_cleanup_report(self) -> Dict[str, Any]:
        """
        Genera un reporte completo de actividades de limpieza.
        
        Returns:
            Dict con reporte detallado
        """
        try:
            # Calcular estadísticas
            total_cycles = self.metrics["cleanup_cycles"]
            total_items_cleaned = self.metrics["items_cleaned"]
            total_bytes_reclaimed = self.metrics["bytes_reclaimed"]
            
            # Analizar historial de limpieza
            recent_cleanups = self.cleanup_history[-10:]  # Últimas 10 limpiezas
            cleanup_by_strategy = {}
            
            for cleanup in recent_cleanups:
                strategy = cleanup.get("strategy", "unknown")
                cleanup_by_strategy[strategy] = cleanup_by_strategy.get(strategy, 0) + 1
            
            # Generar recomendaciones
            recommendations = self._generate_cleanup_recommendations()
            
            report = {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.now().isoformat(),
                "time_period": {
                    "start": (
                        self.cleanup_history[0]["timestamp"] 
                        if self.cleanup_history else datetime.now().isoformat()
                    ),
                    "end": datetime.now().isoformat()
                },
                "summary_metrics": {
                    "total_cleanup_cycles": total_cycles,
                    "total_items_cleaned": total_items_cleaned,
                    "total_bytes_reclaimed": total_bytes_reclaimed,
                    "average_compression_ratio": self.metrics["compression_ratio"],
                    "cleanup_errors": self.metrics["errors"],
                    "current_memory_items": len(self.memory_items)
                },
                "recent_activity": {
                    "last_10_cleanups": recent_cleanups,
                    "cleanup_distribution_by_strategy": cleanup_by_strategy,
                    "most_active_cleanup_time": self._find_most_active_time()
                },
                "recommendations": recommendations,
                "config_summary": self.config.dict(),
                "next_scheduled_cleanup": (
                    (self.last_cleanup_time + timedelta(seconds=self.config.cleanup_interval_seconds)).isoformat()
                    if self.last_cleanup_time else "not_scheduled"
                )
            }
            
            return report
            
        except Exception as e:
            raise MemoryException(f"Failed to generate cleanup report: {e}")
    
    # Métodos privados de implementación
    
    def _create_memory_item(self, item_id: str, item_data: Any) -> MemoryItem:
        """Crea un MemoryItem a partir de datos crudos."""
        if isinstance(item_data, dict):
            created_at = (
                datetime.fromisoformat(item_data.get("created_at"))
                if "created_at" in item_data
                else datetime.now()
            )
            last_accessed = (
                datetime.fromisoformat(item_data.get("last_accessed"))
                if "last_accessed" in item_data
                else datetime.now()
            )
            size_bytes = len(json.dumps(item_data))
        else:
            created_at = datetime.now()
            last_accessed = datetime.now()
            size_bytes = len(str(item_data))
        
        return MemoryItem(
            id=item_id,
            content=item_data,
            created_at=created_at,
            last_accessed=last_accessed,
            size_bytes=size_bytes
        )
    
    def _apply_cleanup_action(self, memory_item: MemoryItem, 
                            action: CleanupAction) -> Dict[str, Any]:
        """Aplica una acción de limpieza a un MemoryItem."""
        try:
            result = {
                "success": False,
                "action": action.value,
                "item_id": memory_item.id,
                "original_size": memory_item.size_bytes,
                "timestamp": datetime.now().isoformat()
            }
            
            if action == CleanupAction.COMPRESS:
                # Comprimir contenido
                compressed = self._compress_data(memory_item.content)
                compression_ratio = len(compressed) / memory_item.size_bytes
                
                result.update({
                    "success": True,
                    "compressed_size": len(compressed),
                    "compression_ratio": compression_ratio,
                    "new_size_bytes": len(compressed)
                })
                
            elif action == CleanupAction.ARCHIVE:
                # Marcar para archivado
                result.update({
                    "success": True,
                    "archive_status": "marked_for_archival",
                    "new_size_bytes": memory_item.size_bytes
                })
                
            elif action == CleanupAction.SUMMARIZE:
                # Crear resumen
                summary = self._create_summary(memory_item.content)
                result.update({
                    "success": True,
                    "summary_size": len(summary),
                    "new_size_bytes": len(summary)
                })
                
            elif action == CleanupAction.DELETE:
                # Marcar para eliminación
                result.update({
                    "success": True,
                    "delete_status": "marked_for_deletion",
                    "new_size_bytes": 0
                })
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action.value,
                "timestamp": datetime.now().isoformat()
            }
    
    def _compress_data(self, data: Any) -> bytes:
        """Comprime datos usando el algoritmo configurado."""
        if self.config.compression_algorithm == "zlib":
            return zlib.compress(
                json.dumps(data).encode('utf-8'),
                level=self.config.compression_level
            )
        elif self.config.compression_algorithm == "lz4":
            import lz4.frame
            return lz4.frame.compress(
                json.dumps(data).encode('utf-8')
            )
        else:  # gzip por defecto
            import gzip
            return gzip.compress(
                json.dumps(data).encode('utf-8'),
                compresslevel=self.config.compression_level
            )
    
    def _create_summary(self, content: Any) -> str:
        """Crea un resumen de contenido para compresión."""
        if isinstance(content, dict):
            # Extraer campos clave
            summary_fields = {}
            for key in ["id", "type", "name", "summary", "timestamp"]:
                if key in content:
                    summary_fields[key] = content[key]
            
            # Añadir conteo si es una lista o dict
            if "items" in content and isinstance(content["items"], list):
                summary_fields["item_count"] = len(content["items"])
            
            return json.dumps({"summary": summary_fields})
        
        elif isinstance(content, list):
            return json.dumps({
                "summary": f"List with {len(content)} items",
                "first_3_items": content[:3] if len(content) > 3 else content
            })
        
        else:
            # Truncar contenido largo
            content_str = str(content)
            if len(content_str) > 500:
                return content_str[:500] + "... [truncated]"
            return content_str
    
    def _attempt_repair_from_backup(self, item_id: str, original_item: Any) -> Optional[Dict]:
        """Intenta reparar desde copias de seguridad."""
        # Esta es una implementación básica
        # En un sistema real, buscaría en un sistema de versionado o backups
        return None
    
    def _extract_valid_parts(self, corrupted_item: Any) -> Optional[Dict]:
        """Extrae partes válidas de un item corrupto."""
        if isinstance(corrupted_item, dict):
            valid_parts = {}
            
            for key, value in corrupted_item.items():
                try:
                    # Intentar serializar/deserializar para validar
                    json.dumps(value)
                    valid_parts[key] = value
                except:
                    # Saltar valores no serializables
                    continue
            
            if valid_parts:
                return {
                    "content": valid_parts,
                    "metadata": {
                        "repair_status": "partial_extraction",
                        "original_keys": len(corrupted_item),
                        "valid_keys": len(valid_parts),
                        "repair_timestamp": datetime.now().isoformat()
                    }
                }
        
        return None
    
    def _generate_cleanup_recommendations(self) -> List[Dict[str, Any]]:
        """Genera recomendaciones basadas en métricas de limpieza."""
        recommendations = []
        
        # Recomendación basada en ratio de compresión
        if self.metrics["compression_ratio"] > 0.8:
            recommendations.append({
                "type": "compression",
                "priority": "low",
                "message": "Compression efficiency is good. Consider increasing compression level.",
                "suggestion": "Increase compression_level to 7 or higher"
            })
        
        # Recomendación basada en frecuencia de limpieza
        if self.metrics["cleanup_cycles"] < 10:
            recommendations.append({
                "type": "schedule",
                "priority": "medium",
                "message": "Limited cleanup history. Consider more frequent cleanups.",
                "suggestion": "Reduce cleanup_interval_seconds to 1800 (30 minutes)"
            })
        
        # Recomendación basada en errores
        if self.metrics["errors"] > 5:
            recommendations.append({
                "type": "reliability",
                "priority": "high",
                "message": f"High error rate in cleanups: {self.metrics['errors']} errors",
                "suggestion": "Review cleanup strategies and error handling"
            })
        
        return recommendations
    
    def _find_most_active_time(self) -> Optional[Dict[str, Any]]:
        """Encuentra el período más activo para limpieza."""
        if not self.cleanup_history:
            return None
        
        # Agrupar por hora del día
        hour_counts = {}
        for cleanup in self.cleanup_history:
            timestamp = datetime.fromisoformat(cleanup["timestamp"])
            hour = timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        if hour_counts:
            most_active_hour = max(hour_counts.items(), key=lambda x: x[1])
            return {
                "hour": most_active_hour[0],
                "cleanup_count": most_active_hour[1],
                "hour_range": f"{most_active_hour[0]}:00 - {most_active_hour[0] + 1}:00"
            }
        
        return None