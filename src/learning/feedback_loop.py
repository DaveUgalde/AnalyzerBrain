"""
FeedbackLoop - Sistema de procesamiento y integración de feedback.
Captura feedback de usuarios, lo procesa y lo integra en el conocimiento del sistema.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from datetime import datetime
from collections import defaultdict
from pydantic import BaseModel, Field, validator
import numpy as np
import logging

from ..core.exceptions import BrainException, ValidationError

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Tipos de feedback soportados."""
    CORRECTION = "correction"          # Corrección de respuesta
    CONFIRMATION = "confirmation"      # Confirmación de que está bien
    REINFORCEMENT = "reinforcement"    # Refuerzo positivo
    RELEVANCE = "relevance"            # Relevancia de la respuesta
    USEFULNESS = "usefulness"          # Utilidad de la respuesta
    DETAIL_LEVEL = "detail_level"      # Nivel de detalle
    TECHNICAL_ACCURACY = "technical_accuracy"  # Precisión técnica

class FeedbackSource(Enum):
    """Fuentes de feedback."""
    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"
    AUTOMATED = "automated"
    EXTERNAL = "external"

class FeedbackPriority(Enum):
    """Prioridad del feedback."""
    CRITICAL = 0    # Corrección crítica que afecta precisión
    HIGH = 1        # Feedback importante que mejora significativamente
    MEDIUM = 2      # Feedback que mejora calidad general
    LOW = 3         # Mejoras menores o ajustes

@dataclass
class FeedbackItem:
    """Elemento individual de feedback."""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feedback_type: FeedbackType = FeedbackType.CONFIRMATION
    source: FeedbackSource = FeedbackSource.USER
    priority: FeedbackPriority = FeedbackPriority.MEDIUM
    
    # Contexto
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    agent_id: Optional[str] = None
    entity_id: Optional[str] = None
    
    # Contenido
    content: Dict[str, Any] = field(default_factory=dict)
    original_input: Optional[Dict[str, Any]] = None
    original_output: Optional[Dict[str, Any]] = None
    expected_output: Optional[Dict[str, Any]] = None
    
    # Metadata
    confidence: float = 1.0
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    integrated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "source": self.source.value,
            "priority": self.priority.value,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "entity_id": self.entity_id,
            "content": self.content,
            "original_input": self.original_input,
            "original_output": self.original_output,
            "expected_output": self.expected_output,
            "confidence": self.confidence,
            "weight": self.weight,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "integrated_at": self.integrated_at.isoformat() if self.integrated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
        """Crea desde diccionario."""
        feedback = cls()
        for key, value in data.items():
            if hasattr(feedback, key):
                if key in ['feedback_type', 'source', 'priority']:
                    enum_class = {
                        'feedback_type': FeedbackType,
                        'source': FeedbackSource,
                        'priority': FeedbackPriority
                    }[key]
                    value = enum_class(value) if value else None
                elif key in ['created_at', 'processed_at', 'integrated_at'] and value:
                    value = datetime.fromisoformat(value)
                setattr(feedback, key, value)
        return feedback

class FeedbackMetrics(BaseModel):
    """Métricas del feedback."""
    total_feedback_received: int = 0
    feedback_by_type: Dict[str, int] = Field(default_factory=dict)
    feedback_by_source: Dict[str, int] = Field(default_factory=dict)
    average_confidence: float = 0.0
    integration_success_rate: float = 0.0
    average_processing_time_ms: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def update(self, feedback_type: str, source: str, success: bool, 
               processing_time_ms: float, confidence: float = 1.0):
        """Actualiza métricas."""
        self.total_feedback_received += 1
        self.feedback_by_type[feedback_type] = self.feedback_by_type.get(feedback_type, 0) + 1
        self.feedback_by_source[source] = self.feedback_by_source.get(source, 0) + 1
        
        # Actualizar promedio de confianza
        if self.total_feedback_received == 1:
            self.average_confidence = confidence
        else:
            self.average_confidence = (
                (self.average_confidence * (self.total_feedback_received - 1) + confidence) 
                / self.total_feedback_received
            )
        
        # Actualizar tasa de éxito
        total_processed = sum(self.feedback_by_type.values())
        if success:
            successful = sum(1 for v in self.feedback_by_type.values())
            self.integration_success_rate = successful / total_processed if total_processed > 0 else 0
        
        # Actualizar tiempo promedio de procesamiento
        if self.total_feedback_received == 1:
            self.average_processing_time_ms = processing_time_ms
        else:
            self.average_processing_time_ms = (
                (self.average_processing_time_ms * (self.total_feedback_received - 1) + processing_time_ms)
                / self.total_feedback_received
            )
        
        self.last_updated = datetime.now()

class FeedbackLoop:
    """
    Sistema de bucle de feedback para capturar, procesar e integrar feedback.
    
    Características:
    1. Captura feedback de múltiples fuentes (usuarios, sistema, agentes)
    2. Valida y clasifica feedback por tipo y prioridad
    3. Procesa feedback en lotes para eficiencia
    4. Integra feedback en el conocimiento del sistema
    5. Mide impacto del feedback en la mejora del sistema
    6. Proporciona mecanismos para feedback negativo (correcciones)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el bucle de feedback.
        
        Args:
            config: Configuración del feedback loop
        """
        self.config = config or {}
        self.feedback_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.processing_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.processed_feedback: Dict[str, FeedbackItem] = {}
        self.integration_handlers: Dict[str, Callable] = {}
        self.validation_rules: Dict[str, Callable] = {}
        self.metrics = FeedbackMetrics()
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        self.integration_task: Optional[asyncio.Task] = None
        
        # Configurar handlers por defecto
        self._setup_default_handlers()
        self._setup_default_validation_rules()
        
        logger.info("FeedbackLoop inicializado")
    
    async def start(self) -> bool:
        """
        Inicia el procesamiento de feedback.
        
        Returns:
            bool: True si se inició correctamente
        """
        if self.is_running:
            logger.warning("FeedbackLoop ya está ejecutándose")
            return False
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_feedback_loop())
        self.integration_task = asyncio.create_task(self._integrate_feedback_loop())
        
        logger.info("FeedbackLoop iniciado")
        return True
    
    async def stop(self) -> bool:
        """
        Detiene el procesamiento de feedback.
        
        Returns:
            bool: True si se detuvo correctamente
        """
        if not self.is_running:
            return False
        
        self.is_running = False
        
        # Cancelar tareas
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        if self.integration_task:
            self.integration_task.cancel()
            try:
                await self.integration_task
            except asyncio.CancelledError:
                pass
        
        logger.info("FeedbackLoop detenido")
        return True
    
    # ========== FUNCIONES PÚBLICAS PRINCIPALES ==========
    
    async def collect_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """
        Recoge feedback del sistema.
        
        Args:
            feedback_data: Datos del feedback en formato estructurado
            
        Returns:
            str: ID del feedback recibido
            
        Raises:
            ValidationError: Si el feedback no es válido
            QueueFullError: Si la cola está llena
        """
        start_time = datetime.now()
        
        try:
            # Validar feedback
            self._validate_feedback_structure(feedback_data)
            
            # Crear ítem de feedback
            feedback_item = self._create_feedback_item(feedback_data)
            
            # Validar contenido específico
            await self._validate_feedback_content(feedback_item)
            
            # Añadir a la cola
            await self.feedback_queue.put(feedback_item)
            
            # Registrar métrica
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.update(
                feedback_type=feedback_item.feedback_type.value,
                source=feedback_item.source.value,
                success=True,
                processing_time_ms=processing_time,
                confidence=feedback_item.confidence
            )
            
            logger.info(f"Feedback recibido: {feedback_item.feedback_id} "
                       f"({feedback_item.feedback_type.value})")
            
            return feedback_item.feedback_id
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error recogiendo feedback: {e}")
            
            self.metrics.update(
                feedback_type=feedback_data.get('feedback_type', 'unknown'),
                source=feedback_data.get('source', 'unknown'),
                success=False,
                processing_time_ms=processing_time
            )
            
            raise ValidationError(f"Error collecting feedback: {e}")
    
    async def process_feedback(self, feedback_id: Optional[str] = None, 
                             batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Procesa feedback pendiente.
        
        Args:
            feedback_id: ID específico a procesar (opcional)
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            List[Dict]: Resultados del procesamiento
        """
        results = []
        
        if feedback_id:
            # Procesar feedback específico
            if feedback_id in self.processed_feedback:
                feedback_item = self.processed_feedback[feedback_id]
                result = await self._process_single_feedback(feedback_item)
                results.append(result)
        else:
            # Procesar lote de feedback
            batch = []
            try:
                for _ in range(batch_size):
                    if self.feedback_queue.empty():
                        break
                    feedback_item = await asyncio.wait_for(
                        self.feedback_queue.get(), 
                        timeout=1.0
                    )
                    batch.append(feedback_item)
            except asyncio.TimeoutError:
                pass
            
            if batch:
                batch_results = await self._process_feedback_batch(batch)
                results.extend(batch_results)
        
        return results
    
    async def integrate_feedback(self, feedback_id: str, 
                               integration_options: Optional[Dict] = None) -> bool:
        """
        Integra feedback procesado en el sistema.
        
        Args:
            feedback_id: ID del feedback a integrar
            integration_options: Opciones de integración
            
        Returns:
            bool: True si la integración fue exitosa
        """
        if feedback_id not in self.processed_feedback:
            logger.warning(f"Feedback no encontrado para integración: {feedback_id}")
            return False
        
        feedback_item = self.processed_feedback[feedback_id]
        start_time = datetime.now()
        
        try:
            # Obtener handler apropiado
            handler_key = f"{feedback_item.feedback_type.value}_{feedback_item.source.value}"
            handler = self.integration_handlers.get(handler_key)
            
            if not handler:
                logger.warning(f"No hay handler para: {handler_key}")
                return False
            
            # Ejecutar integración
            success = await handler(feedback_item, integration_options or {})
            
            if success:
                feedback_item.integrated_at = datetime.now()
                logger.info(f"Feedback integrado: {feedback_id}")
            else:
                logger.warning(f"Fallo integración feedback: {feedback_id}")
            
            # Actualizar métricas
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.update(
                feedback_type=feedback_item.feedback_type.value,
                source=feedback_item.source.value,
                success=success,
                processing_time_ms=processing_time
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error integrando feedback {feedback_id}: {e}")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.update(
                feedback_type=feedback_item.feedback_type.value,
                source=feedback_item.source.value,
                success=False,
                processing_time_ms=processing_time
            )
            
            return False
    
    async def measure_feedback_impact(self, timeframe_days: int = 30, 
                                    feedback_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Mide el impacto del feedback en el sistema.
        
        Args:
            timeframe_days: Período de tiempo a considerar
            feedback_types: Tipos de feedback a incluir (None = todos)
            
        Returns:
            Dict: Métricas de impacto
        """
        # Filtrar feedback por tiempo y tipo
        cutoff_date = datetime.now().replace(
            day=datetime.now().day - timeframe_days
        )
        
        relevant_feedback = []
        for feedback_id, item in self.processed_feedback.items():
            if item.created_at >= cutoff_date:
                if feedback_types and item.feedback_type.value not in feedback_types:
                    continue
                relevant_feedback.append(item)
        
        # Calcular métricas de impacto
        impact_metrics = {
            "timeframe_days": timeframe_days,
            "total_feedback": len(relevant_feedback),
            "feedback_by_type": defaultdict(int),
            "integration_rate": 0,
            "average_confidence": 0.0,
            "priority_distribution": defaultdict(int),
            "impact_scores": defaultdict(float)
        }
        
        if not relevant_feedback:
            return impact_metrics
        
        integrated_count = 0
        total_confidence = 0.0
        
        for item in relevant_feedback:
            # Conteo por tipo
            impact_metrics["feedback_by_type"][item.feedback_type.value] += 1
            
            # Distribución por prioridad
            impact_metrics["priority_distribution"][item.priority.value] += 1
            
            # Estadísticas de integración
            if item.integrated_at:
                integrated_count += 1
            
            # Confianza
            total_confidence += item.confidence
            
            # Calcular score de impacto basado en prioridad y confianza
            priority_weight = {
                FeedbackPriority.CRITICAL: 5.0,
                FeedbackPriority.HIGH: 3.0,
                FeedbackPriority.MEDIUM: 1.0,
                FeedbackPriority.LOW: 0.5
            }.get(item.priority, 1.0)
            
            impact_score = priority_weight * item.confidence * item.weight
            impact_metrics["impact_scores"][item.feedback_type.value] += impact_score
        
        # Calcular promedios
        impact_metrics["integration_rate"] = integrated_count / len(relevant_feedback)
        impact_metrics["average_confidence"] = total_confidence / len(relevant_feedback)
        
        # Normalizar scores de impacto
        for feedback_type in impact_metrics["impact_scores"]:
            count = impact_metrics["feedback_by_type"][feedback_type]
            if count > 0:
                impact_metrics["impact_scores"][feedback_type] /= count
        
        return impact_metrics
    
    async def optimize_feedback_collection(self, 
                                         target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimiza la recolección de feedback basado en métricas objetivo.
        
        Args:
            target_metrics: Métricas objetivo (tasa de integración, confianza, etc.)
            
        Returns:
            Dict: Recomendaciones de optimización
        """
        recommendations = []
        
        # Analizar métricas actuales
        current_metrics = {
            "integration_success_rate": self.metrics.integration_success_rate,
            "average_confidence": self.metrics.average_confidence,
            "average_processing_time_ms": self.metrics.average_processing_time_ms
        }
        
        # Comparar con objetivos y generar recomendaciones
        if (target_metrics.get("integration_success_rate", 0.8) > 
            current_metrics["integration_success_rate"]):
            recommendations.append({
                "area": "integration_success_rate",
                "current": current_metrics["integration_success_rate"],
                "target": target_metrics.get("integration_success_rate", 0.8),
                "recommendation": "Mejorar validación de feedback antes de integración",
                "actions": [
                    "Añadir validaciones adicionales para feedback de baja confianza",
                    "Implementar revisión por pares para feedback crítico",
                    "Aumentar umbral de confianza para integración automática"
                ]
            })
        
        if (target_metrics.get("average_confidence", 0.7) > 
            current_metrics["average_confidence"]):
            recommendations.append({
                "area": "average_confidence",
                "current": current_metrics["average_confidence"],
                "target": target_metrics.get("average_confidence", 0.7),
                "recommendation": "Mejorar calidad del feedback recibido",
                "actions": [
                    "Proporcionar guías más claras para dar feedback",
                    "Implementar sistema de reputación para fuentes de feedback",
                    "Solicitar confirmación para feedback de baja confianza"
                ]
            })
        
        if (target_metrics.get("average_processing_time_ms", 1000) < 
            current_metrics["average_processing_time_ms"]):
            recommendations.append({
                "area": "processing_time",
                "current": current_metrics["average_processing_time_ms"],
                "target": target_metrics.get("average_processing_time_ms", 1000),
                "recommendation": "Reducir tiempo de procesamiento de feedback",
                "actions": [
                    "Optimizar procesamiento por lotes",
                    "Implementar caché para validaciones frecuentes",
                    "Paralelizar procesamiento de feedback no relacionado"
                ]
            })
        
        return {
            "current_metrics": current_metrics,
            "target_metrics": target_metrics,
            "recommendations": recommendations,
            "priority": "HIGH" if recommendations else "LOW"
        }
    
    async def validate_feedback_quality(self, feedback_id: str) -> Dict[str, Any]:
        """
        Valida la calidad de un ítem de feedback específico.
        
        Args:
            feedback_id: ID del feedback a validar
            
        Returns:
            Dict: Resultados de validación de calidad
        """
        if feedback_id not in self.processed_feedback:
            return {
                "valid": False,
                "error": "Feedback not found",
                "scores": {},
                "recommendations": []
            }
        
        feedback_item = self.processed_feedback[feedback_id]
        quality_scores = {}
        
        # 1. Completitud
        completeness_score = self._calculate_completeness_score(feedback_item)
        quality_scores["completeness"] = completeness_score
        
        # 2. Consistencia
        consistency_score = self._calculate_consistency_score(feedback_item)
        quality_scores["consistency"] = consistency_score
        
        # 3. Relevancia
        relevance_score = self._calculate_relevance_score(feedback_item)
        quality_scores["relevance"] = relevance_score
        
        # 4. Confiabilidad de la fuente
        source_reliability_score = self._calculate_source_reliability_score(feedback_item)
        quality_scores["source_reliability"] = source_reliability_score
        
        # 5. Especificidad
        specificity_score = self._calculate_specificity_score(feedback_item)
        quality_scores["specificity"] = specificity_score
        
        # Calcular score total ponderado
        weights = {
            "completeness": 0.2,
            "consistency": 0.2,
            "relevance": 0.25,
            "source_reliability": 0.2,
            "specificity": 0.15
        }
        
        total_score = sum(
            quality_scores[dim] * weight 
            for dim, weight in weights.items()
        )
        
        # Generar recomendaciones
        recommendations = []
        if completeness_score < 0.7:
            recommendations.append("El feedback carece de información esencial")
        if consistency_score < 0.6:
            recommendations.append("Hay inconsistencias internas en el feedback")
        if relevance_score < 0.5:
            recommendations.append("El feedback puede no ser relevante para el contexto")
        
        return {
            "valid": total_score >= 0.6,
            "total_score": total_score,
            "scores": quality_scores,
            "weights": weights,
            "recommendations": recommendations,
            "priority": "HIGH" if total_score < 0.6 else "MEDIUM" if total_score < 0.8 else "LOW"
        }
    
    async def generate_feedback_report(self, 
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Genera un reporte de feedback para un período de tiempo.
        
        Args:
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            
        Returns:
            Dict: Reporte de feedback
        """
        if not start_date:
            start_date = datetime.now().replace(day=1)  # Inicio del mes
        if not end_date:
            end_date = datetime.now()
        
        # Filtrar feedback por período
        feedback_in_period = []
        for feedback_id, item in self.processed_feedback.items():
            if start_date <= item.created_at <= end_date:
                feedback_in_period.append(item)
        
        # Agrupar por tipo
        feedback_by_type = defaultdict(list)
        for item in feedback_in_period:
            feedback_by_type[item.feedback_type.value].append(item)
        
        # Calcular estadísticas
        total_feedback = len(feedback_in_period)
        integrated_count = sum(1 for item in feedback_in_period if item.integrated_at)
        
        # Distribución por prioridad
        priority_dist = defaultdict(int)
        for item in feedback_in_period:
            priority_dist[item.priority.value] += 1
        
        # Top fuentes
        source_dist = defaultdict(int)
        for item in feedback_in_period:
            source_dist[item.source.value] += 1
        
        # Calcular tendencias (si hay suficiente data histórica)
        trends = self._calculate_feedback_trends(start_date, end_date)
        
        # Generar insights
        insights = []
        if integrated_count / total_feedback < 0.5 and total_feedback > 10:
            insights.append("Baja tasa de integración - revisar proceso de integración")
        
        if FeedbackPriority.CRITICAL.value in priority_dist and priority_dist[FeedbackPriority.CRITICAL.value] > 5:
            insights.append("Alto volumen de feedback crítico - posible problema sistemático")
        
        # Recomendaciones
        recommendations = []
        if total_feedback < 20:
            recommendations.append("Incentivar más feedback para mejor análisis")
        
        if trends.get("growth_rate", 0) < 0.1 and total_feedback > 30:
            recommendations.append("Feedback estancado - considerar nuevos mecanismos de recolección")
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days
            },
            "summary": {
                "total_feedback": total_feedback,
                "integrated_feedback": integrated_count,
                "integration_rate": integrated_count / total_feedback if total_feedback > 0 else 0,
                "average_confidence": self.metrics.average_confidence
            },
            "distribution": {
                "by_type": {k: len(v) for k, v in feedback_by_type.items()},
                "by_priority": dict(priority_dist),
                "by_source": dict(source_dist)
            },
            "trends": trends,
            "insights": insights,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
    
    # ========== FUNCIONES PRIVADAS DE IMPLEMENTACIÓN ==========
    
    def _setup_default_handlers(self) -> None:
        """Configura handlers de integración por defecto."""
        # Handler para correcciones de usuarios
        async def user_correction_handler(feedback: FeedbackItem, options: Dict) -> bool:
            """Integra correcciones de usuarios."""
            try:
                # Obtener agente o componente afectado
                agent_id = feedback.agent_id
                entity_id = feedback.entity_id
                
                # Actualizar conocimiento con corrección
                # Esta es una implementación simplificada
                # En implementación real se integraría con el agente específico
                
                logger.info(f"Integrando corrección de usuario para agente {agent_id}")
                return True
            except Exception as e:
                logger.error(f"Error en handler de corrección: {e}")
                return False
        
        # Handler para confirmaciones
        async def confirmation_handler(feedback: FeedbackItem, options: Dict) -> bool:
            """Integra confirmaciones positivas."""
            try:
                # Reforzar conocimiento existente
                # Aumentar confianza de entidades relacionadas
                
                logger.info(f"Integrando confirmación para {feedback.entity_id}")
                return True
            except Exception as e:
                logger.error(f"Error en handler de confirmación: {e}")
                return False
        
        # Registrar handlers
        self.register_integration_handler(
            FeedbackType.CORRECTION, FeedbackSource.USER, user_correction_handler
        )
        self.register_integration_handler(
            FeedbackType.CONFIRMATION, FeedbackSource.USER, confirmation_handler
        )
    
    def _setup_default_validation_rules(self) -> None:
        """Configura reglas de validación por defecto."""
        # Regla para feedback de corrección
        def validate_correction(feedback: FeedbackItem) -> Tuple[bool, str]:
            """Valida feedback de corrección."""
            if not feedback.original_output:
                return False, "Corrección requiere output original"
            if not feedback.expected_output:
                return False, "Corrección requiere output esperado"
            if feedback.confidence < 0.5:
                return False, "Confianza muy baja para corrección"
            return True, "Válido"
        
        # Regla para feedback de confirmación
        def validate_confirmation(feedback: FeedbackItem) -> Tuple[bool, str]:
            """Valida feedback de confirmación."""
            if not feedback.original_output:
                return False, "Confirmación requiere output original"
            return True, "Válido"
        
        # Registrar reglas
        self.register_validation_rule(FeedbackType.CORRECTION, validate_correction)
        self.register_validation_rule(FeedbackType.CONFIRMATION, validate_confirmation)
    
    def _validate_feedback_structure(self, feedback_data: Dict[str, Any]) -> None:
        """Valida estructura básica del feedback."""
        required_fields = ['feedback_type', 'source', 'content']
        for field in required_fields:
            if field not in feedback_data:
                raise ValidationError(f"Campo requerido faltante: {field}")
        
        # Validar tipos de enum
        try:
            FeedbackType(feedback_data['feedback_type'])
        except ValueError:
            raise ValidationError(f"Tipo de feedback inválido: {feedback_data['feedback_type']}")
        
        try:
            FeedbackSource(feedback_data['source'])
        except ValueError:
            raise ValidationError(f"Fuente de feedback inválida: {feedback_data['source']}")
    
    def _create_feedback_item(self, feedback_data: Dict[str, Any]) -> FeedbackItem:
        """Crea un ítem de feedback desde datos crudos."""
        return FeedbackItem(
            feedback_type=FeedbackType(feedback_data['feedback_type']),
            source=FeedbackSource(feedback_data['source']),
            priority=FeedbackPriority(feedback_data.get('priority', 'medium')),
            request_id=feedback_data.get('request_id'),
            session_id=feedback_data.get('session_id'),
            project_id=feedback_data.get('project_id'),
            agent_id=feedback_data.get('agent_id'),
            entity_id=feedback_data.get('entity_id'),
            content=feedback_data['content'],
            original_input=feedback_data.get('original_input'),
            original_output=feedback_data.get('original_output'),
            expected_output=feedback_data.get('expected_output'),
            confidence=feedback_data.get('confidence', 1.0),
            weight=feedback_data.get('weight', 1.0),
            tags=feedback_data.get('tags', [])
        )
    
    async def _validate_feedback_content(self, feedback: FeedbackItem) -> None:
        """Valida contenido específico del feedback."""
        # Obtener regla de validación
        validator = self.validation_rules.get(feedback.feedback_type.value)
        if validator:
            is_valid, message = validator(feedback)
            if not is_valid:
                raise ValidationError(f"Feedback inválido: {message}")
    
    async def _process_feedback_loop(self) -> None:
        """Loop principal de procesamiento de feedback."""
        logger.info("Iniciando loop de procesamiento de feedback")
        
        while self.is_running:
            try:
                # Procesar feedback en lotes
                await self.process_feedback(batch_size=10)
                
                # Esperar antes del próximo lote
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                logger.info("Loop de procesamiento cancelado")
                break
            except Exception as e:
                logger.error(f"Error en loop de procesamiento: {e}")
                await asyncio.sleep(10)  # Esperar antes de reintentar
    
    async def _integrate_feedback_loop(self) -> None:
        """Loop principal de integración de feedback."""
        logger.info("Iniciando loop de integración de feedback")
        
        while self.is_running:
            try:
                # Buscar feedback procesado no integrado
                to_integrate = []
                for feedback_id, item in self.processed_feedback.items():
                    if not item.integrated_at and item.processed_at:
                        to_integrate.append(feedback_id)
                
                # Integrar en lotes
                for feedback_id in to_integrate[:5]:  # Máximo 5 por ciclo
                    await self.integrate_feedback(feedback_id)
                
                # Esperar antes del próximo ciclo
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                logger.info("Loop de integración cancelado")
                break
            except Exception as e:
                logger.error(f"Error en loop de integración: {e}")
                await asyncio.sleep(30)  # Esperar antes de reintentar
    
    async def _process_single_feedback(self, feedback_item: FeedbackItem) -> Dict[str, Any]:
        """Procesa un único ítem de feedback."""
        start_time = datetime.now()
        
        try:
            # 1. Análisis inicial
            analysis_result = await self._analyze_feedback(feedback_item)
            
            # 2. Clasificación
            classification = await self._classify_feedback(feedback_item)
            
            # 3. Enriquecimiento
            enriched_feedback = await self._enrich_feedback(feedback_item, analysis_result)
            
            # 4. Validación final
            validation_result = await self.validate_feedback_quality(feedback_item.feedback_id)
            
            # Marcar como procesado
            feedback_item.processed_at = datetime.now()
            feedback_item.weight = analysis_result.get('weight', 1.0)
            self.processed_feedback[feedback_item.feedback_id] = feedback_item
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                "feedback_id": feedback_item.feedback_id,
                "status": "processed",
                "analysis": analysis_result,
                "classification": classification,
                "validation": validation_result,
                "processing_time_ms": processing_time,
                "processed_at": feedback_item.processed_at.isoformat()
            }
            
            logger.info(f"Feedback procesado: {feedback_item.feedback_id}")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error procesando feedback {feedback_item.feedback_id}: {e}")
            
            return {
                "feedback_id": feedback_item.feedback_id,
                "status": "failed",
                "error": str(e),
                "processing_time_ms": processing_time
            }
    
    async def _process_feedback_batch(self, batch: List[FeedbackItem]) -> List[Dict[str, Any]]:
        """Procesa un lote de feedback."""
        results = []
        
        # Procesar en paralelo
        tasks = [self._process_single_feedback(item) for item in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Error en procesamiento por lotes: {result}")
            else:
                results.append(result)
        
        return results
    
    async def _analyze_feedback(self, feedback: FeedbackItem) -> Dict[str, Any]:
        """Analiza feedback para extraer información útil."""
        analysis = {
            "weight": feedback.weight,
            "confidence": feedback.confidence,
            "priority": feedback.priority.value,
            "complexity": "low"
        }
        
        # Calcular peso basado en tipo y prioridad
        type_weights = {
            FeedbackType.CORRECTION: 3.0,
            FeedbackType.CONFIRMATION: 1.0,
            FeedbackType.REINFORCEMENT: 2.0,
            FeedbackType.RELEVANCE: 1.5,
            FeedbackType.USEFULNESS: 1.5,
            FeedbackType.DETAIL_LEVEL: 1.0,
            FeedbackType.TECHNICAL_ACCURACY: 2.5
        }
        
        priority_multipliers = {
            FeedbackPriority.CRITICAL: 5.0,
            FeedbackPriority.HIGH: 3.0,
            FeedbackPriority.MEDIUM: 1.0,
            FeedbackPriority.LOW: 0.5
        }
        
        base_weight = type_weights.get(feedback.feedback_type, 1.0)
        priority_multiplier = priority_multipliers.get(feedback.priority, 1.0)
        
        analysis["weight"] = base_weight * priority_multiplier * feedback.confidence
        
        # Calcular complejidad
        content_length = len(str(feedback.content))
        has_original = bool(feedback.original_input or feedback.original_output)
        has_expected = bool(feedback.expected_output)
        
        complexity_score = 0
        if content_length > 500:
            complexity_score += 1
        if has_original and has_expected:
            complexity_score += 2
        elif has_original or has_expected:
            complexity_score += 1
        
        if complexity_score >= 2:
            analysis["complexity"] = "high"
        elif complexity_score == 1:
            analysis["complexity"] = "medium"
        
        return analysis
    
    async def _classify_feedback(self, feedback: FeedbackItem) -> Dict[str, Any]:
        """Clasifica feedback en categorías."""
        classification = {
            "primary_category": feedback.feedback_type.value,
            "secondary_categories": [],
            "urgency": "normal",
            "scope": "specific"
        }
        
        # Determinar urgencia basada en prioridad
        if feedback.priority == FeedbackPriority.CRITICAL:
            classification["urgency"] = "immediate"
        elif feedback.priority == FeedbackPriority.HIGH:
            classification["urgency"] = "high"
        
        # Determinar alcance basado en contenido
        content_str = str(feedback.content).lower()
        if any(word in content_str for word in ["all", "every", "always", "never", "general"]):
            classification["scope"] = "general"
        
        # Identificar categorías secundarias basadas en contenido
        tech_terms = ["bug", "error", "fix", "correct", "wrong", "incorrect"]
        if any(term in content_str for term in tech_terms):
            classification["secondary_categories"].append("technical_correction")
        
        clarity_terms = ["clear", "understand", "confusing", "explain", "clarify"]
        if any(term in content_str for term in clarity_terms):
            classification["secondary_categories"].append("clarity")
        
        return classification
    
    async def _enrich_feedback(self, feedback: FeedbackItem, 
                             analysis_result: Dict[str, Any]) -> FeedbackItem:
        """Enriquece feedback con información adicional."""
        # Ajustar peso basado en análisis
        feedback.weight = analysis_result.get('weight', feedback.weight)
        
        # Añadir tags basados en clasificación
        if analysis_result.get('complexity') == 'high':
            feedback.tags.append('complex')
        
        if feedback.priority == FeedbackPriority.CRITICAL:
            feedback.tags.append('critical')
        
        # Añadir timestamp de procesamiento
        feedback.processed_at = datetime.now()
        
        return feedback
    
    def _calculate_completeness_score(self, feedback: FeedbackItem) -> float:
        """Calcula score de completitud del feedback."""
        score = 0.0
        max_score = 5.0
        
        # Contenido no vacío
        if feedback.content:
            score += 1.0
        
        # Tipo y fuente especificados
        if feedback.feedback_type and feedback.source:
            score += 1.0
        
        # Contexto proporcionado
        if feedback.request_id or feedback.session_id or feedback.project_id:
            score += 1.0
        
        # Elementos específicos según tipo
        if feedback.feedback_type == FeedbackType.CORRECTION:
            if feedback.original_output and feedback.expected_output:
                score += 2.0
        elif feedback.feedback_type == FeedbackType.CONFIRMATION:
            if feedback.original_output:
                score += 2.0
        
        return score / max_score
    
    def _calculate_consistency_score(self, feedback: FeedbackItem) -> float:
        """Calcula score de consistencia interna del feedback."""
        # Verificar consistencia entre tipo y contenido
        content_str = str(feedback.content).lower()
        
        # Mapeo de tipos a palabras clave esperadas
        type_keywords = {
            FeedbackType.CORRECTION: ["wrong", "incorrect", "error", "fix", "correct"],
            FeedbackType.CONFIRMATION: ["correct", "right", "good", "accurate", "yes"],
            FeedbackType.REINFORCEMENT: ["helpful", "useful", "good", "excellent", "thanks"],
            FeedbackType.RELEVANCE: ["relevant", "related", "on topic", "pertinent"],
            FeedbackType.USEFULNESS: ["useful", "helpful", "practical", "applicable"],
            FeedbackType.DETAIL_LEVEL: ["detailed", "brief", "depth", "superficial"],
            FeedbackType.TECHNICAL_ACCURACY: ["accurate", "precise", "technical", "specific"]
        }
        
        expected_keywords = type_keywords.get(feedback.feedback_type, [])
        
        # Verificar presencia de palabras clave relevantes
        matching_keywords = sum(1 for keyword in expected_keywords 
                              if keyword in content_str)
        
        # Score basado en proporción de palabras clave encontradas
        if expected_keywords:
            return matching_keywords / len(expected_keywords)
        
        return 0.5  # Score por defecto si no hay palabras clave específicas
    
    def _calculate_relevance_score(self, feedback: FeedbackItem) -> float:
        """Calcula score de relevancia del feedback."""
        # Relevancia basada en contexto
        has_context = bool(feedback.request_id or feedback.session_id or 
                          feedback.project_id or feedback.agent_id)
        
        # Relevancia basada en especificidad
        content_str = str(feedback.content)
        specificity_score = min(len(content_str.split()) / 100, 1.0)
        
        # Score combinado
        base_score = 0.5
        if has_context:
            base_score += 0.3
        base_score += specificity_score * 0.2
        
        return min(base_score, 1.0)
    
    def _calculate_source_reliability_score(self, feedback: FeedbackItem) -> float:
        """Calcula score de confiabilidad de la fuente."""
        source_scores = {
            FeedbackSource.SYSTEM: 1.0,    # Sistema es muy confiable
            FeedbackSource.AGENT: 0.9,      # Agentes son confiables
            FeedbackSource.AUTOMATED: 0.8,  # Automatizado es bastante confiable
            FeedbackSource.USER: 0.7,       # Usuario es moderadamente confiable
            FeedbackSource.EXTERNAL: 0.6    # Externo es menos confiable
        }
        
        base_score = source_scores.get(feedback.source, 0.5)
        
        # Ajustar por confianza reportada
        adjusted_score = base_score * feedback.confidence
        
        return adjusted_score
    
    def _calculate_specificity_score(self, feedback: FeedbackItem) -> float:
        """Calcula score de especificidad del feedback."""
        content_str = str(feedback.content)
        
        # Medir longitud
        word_count = len(content_str.split())
        length_score = min(word_count / 50, 1.0)  # Máximo 50 palabras
        
        # Medir uso de términos específicos
        specific_terms = ["line", "function", "class", "file", "variable", 
                         "method", "parameter", "return", "import"]
        
        specific_count = sum(1 for term in specific_terms 
                           if term in content_str.lower())
        specificity_score = min(specific_count / 5, 1.0)  # Máximo 5 términos
        
        # Combinar scores
        combined_score = (length_score * 0.4 + specificity_score * 0.6)
        
        return combined_score
    
    def _calculate_feedback_trends(self, start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """Calcula tendencias en el feedback."""
        # Agrupar por día
        daily_counts = defaultdict(int)
        for item in self.processed_feedback.values():
            if start_date <= item.created_at <= end_date:
                day_key = item.created_at.strftime("%Y-%m-%d")
                daily_counts[day_key] += 1
        
        if not daily_counts:
            return {"growth_rate": 0, "volatility": 0, "trend": "stable"}
        
        # Calcular crecimiento
        days = sorted(daily_counts.keys())
        if len(days) >= 2:
            first_count = daily_counts[days[0]]
            last_count = daily_counts[days[-1]]
            
            if first_count > 0:
                growth_rate = (last_count - first_count) / first_count
            else:
                growth_rate = last_count if last_count > 0 else 0
        else:
            growth_rate = 0
        
        # Calcular volatilidad
        counts = list(daily_counts.values())
        if len(counts) >= 2:
            mean_count = sum(counts) / len(counts)
            variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
            volatility = variance ** 0.5 / mean_count if mean_count > 0 else 0
        else:
            volatility = 0
        
        # Determinar tendencia
        if growth_rate > 0.1:
            trend = "increasing"
        elif growth_rate < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "growth_rate": growth_rate,
            "volatility": volatility,
            "trend": trend,
            "average_daily": sum(counts) / len(counts) if counts else 0,
            "peak_daily": max(counts) if counts else 0
        }
    
    # ========== FUNCIONES AUXILIARES ==========
    
    def register_integration_handler(self, feedback_type: FeedbackType, 
                                   source: FeedbackSource, 
                                   handler: Callable) -> None:
        """
        Registra un handler para integrar feedback específico.
        
        Args:
            feedback_type: Tipo de feedback
            source: Fuente del feedback
            handler: Función que maneja la integración
        """
        key = f"{feedback_type.value}_{source.value}"
        self.integration_handlers[key] = handler
        logger.info(f"Handler registrado para: {key}")
    
    def register_validation_rule(self, feedback_type: FeedbackType, 
                               validator: Callable) -> None:
        """
        Registra una regla de validación para un tipo de feedback.
        
        Args:
            feedback_type: Tipo de feedback
            validator: Función de validación
        """
        self.validation_rules[feedback_type.value] = validator
        logger.info(f"Regla de validación registrada para: {feedback_type.value}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas actuales del feedback loop."""
        return {
            "feedback_loop_metrics": self.metrics.dict(),
            "queue_sizes": {
                "feedback_queue": self.feedback_queue.qsize(),
                "processing_queue": self.processing_queue.qsize()
            },
            "processed_count": len(self.processed_feedback),
            "integrated_count": sum(1 for item in self.processed_feedback.values() 
                                  if item.integrated_at),
            "is_running": self.is_running
        }
    
    def clear_processed_feedback(self, older_than_days: int = 30) -> int:
        """
        Limpia feedback procesado antiguo.
        
        Args:
            older_than_days: Eliminar feedback más antiguo que estos días
            
        Returns:
            int: Número de ítems eliminados
        """
        cutoff_date = datetime.now().replace(
            day=datetime.now().day - older_than_days
        )
        
        to_delete = []
        for feedback_id, item in self.processed_feedback.items():
            if item.created_at < cutoff_date:
                to_delete.append(feedback_id)
        
        for feedback_id in to_delete:
            del self.processed_feedback[feedback_id]
        
        logger.info(f"Eliminados {len(to_delete)} ítems de feedback antiguos")
        return len(to_delete)