"""
analyst_agent.py - Agente analista especializado en análisis de métricas, detección de anomalías y generación de insights.

Este agente implementa capacidades avanzadas de análisis de datos, predicción de tendencias,
detección de anomalías y generación de recomendaciones optimizadas para sistemas de software.
"""

import asyncio
import logging
import json
import pickle
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from collections import defaultdict, deque
from statistics import mean, median, stdev, variance
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings

from ..core.exceptions import BrainException, ValidationError
from ..core.event_bus import EventBus
from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentState, AgentConfig, AgentMemoryType

logger = logging.getLogger(__name__)

# ============================================================================
# TIPOS Y ESTRUCTURAS DE DATOS
# ============================================================================

class AnalysisType(Enum):
    """Tipos de análisis que puede realizar el agente."""
    SYSTEM_METRICS = "system_metrics"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    SECURITY = "security"
    USAGE = "usage"
    PREDICTIVE = "predictive"
    COMPARATIVE = "comparative"
    TREND = "trend"

class MetricType(Enum):
    """Tipos de métricas que se pueden analizar."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    DISTRIBUTION = "distribution"

class AnomalySeverity(Enum):
    """Niveles de severidad de anomalías."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MetricDataPoint:
    """Punto de datos de métrica."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TimeSeriesData:
    """Serie temporal de datos."""
    metric_name: str
    metric_type: MetricType
    data_points: List[MetricDataPoint]
    unit: str = ""
    description: str = ""

@dataclass
class AnomalyDetectionResult:
    """Resultado de detección de anomalías."""
    anomaly_id: str
    metric_name: str
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float
    severity: AnomalySeverity
    confidence: float
    pattern: str = ""
    root_cause_hypotheses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendAnalysisResult:
    """Resultado de análisis de tendencias."""
    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    trend_strength: float  # 0-1
    slope: float
    intercept: float
    r_squared: float
    forecast_values: List[float] = field(default_factory=list)
    forecast_timestamps: List[datetime] = field(default_factory=list)
    seasonality_detected: bool = False
    seasonality_period: Optional[int] = None
    changepoints: List[datetime] = field(default_factory=list)

@dataclass
class Insight:
    """Insight generado por el analista."""
    insight_id: str
    title: str
    description: str
    category: str
    impact_score: float  # 0-1
    confidence: float  # 0-1
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    expiration: Optional[datetime] = None

@dataclass
class OptimizationRecommendation:
    """Recomendación de optimización."""
    recommendation_id: str
    title: str
    description: str
    area: str  # "performance", "cost", "reliability", "security", "maintainability"
    estimated_impact: float  # 0-1
    estimated_effort: float  # 0-1
    priority: float  # impact/effort ratio
    prerequisites: List[str] = field(default_factory=list)
    implementation_steps: List[str] = field(default_factory=list)
    validation_metrics: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================

class AnalystAgentConfig(AgentConfig):
    """Configuración específica del AnalystAgent."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Configuración de análisis
        self.analysis_window_hours: int = kwargs.get('analysis_window_hours', 24)
        self.min_data_points: int = kwargs.get('min_data_points', 10)
        self.anomaly_detection_threshold: float = kwargs.get('anomaly_detection_threshold', 0.95)
        self.trend_analysis_periods: int = kwargs.get('trend_analysis_periods', 7)
        self.forecast_horizon: int = kwargs.get('forecast_horizon', 24)
        
        # Configuración de algoritmos
        self.isolation_forest_contamination: float = kwargs.get('isolation_forest_contamination', 0.1)
        self.dbscan_eps: float = kwargs.get('dbscan_eps', 0.5)
        self.dbscan_min_samples: int = kwargs.get('dbscan_min_samples', 5)
        self.z_score_threshold: float = kwargs.get('z_score_threshold', 3.0)
        
        # Configuración de reportes
        self.report_generation_interval: int = kwargs.get('report_generation_interval', 3600)  # segundos
        self.insight_expiration_hours: int = kwargs.get('insight_expiration_hours', 24)
        self.max_insights_per_category: int = kwargs.get('max_insights_per_category', 10)
        
        # Métricas a monitorear
        self.monitored_metrics: List[str] = kwargs.get('monitored_metrics', [
            'system.cpu.usage',
            'system.memory.usage',
            'system.disk.io',
            'system.network.bandwidth',
            'application.response.time',
            'application.error.rate',
            'database.query.time',
            'cache.hit.rate',
            'queue.length',
            'throughput',
        ])
        
        # Umbrales de alerta
        self.alert_thresholds: Dict[str, Dict[str, float]] = kwargs.get('alert_thresholds', {
            'system.cpu.usage': {'warning': 80.0, 'critical': 95.0},
            'system.memory.usage': {'warning': 85.0, 'critical': 95.0},
            'application.error.rate': {'warning': 1.0, 'critical': 5.0},
            'application.response.time': {'warning': 1000.0, 'critical': 5000.0},  # ms
        })

class AnalysisPattern:
    """Patrones de análisis reconocidos."""
    
    PATTERNS = {
        'spike': {
            'description': 'Aumento repentino en el valor de la métrica',
            'detection_method': 'derivative_threshold',
            'severity': 'MEDIUM',
        },
        'dip': {
            'description': 'Caída repentina en el valor de la métrica',
            'detection_method': 'derivative_threshold',
            'severity': 'MEDIUM',
        },
        'gradual_increase': {
            'description': 'Aumento gradual sostenido',
            'detection_method': 'trend_analysis',
            'severity': 'LOW',
        },
        'gradual_decrease': {
            'description': 'Disminución gradual sostenida',
            'detection_method': 'trend_analysis',
            'severity': 'LOW',
        },
        'seasonality': {
            'description': 'Patrón cíclico repetitivo',
            'detection_method': 'frequency_analysis',
            'severity': 'INFO',
        },
        'regime_change': {
            'description': 'Cambio en el comportamiento base',
            'detection_method': 'changepoint_detection',
            'severity': 'HIGH',
        },
        'outlier_cluster': {
            'description': 'Grupo de valores atípicos',
            'detection_method': 'clustering',
            'severity': 'HIGH',
        },
    }

# ============================================================================
# CLASE PRINCIPAL - ANALYST AGENT
# ============================================================================

class AnalystAgent(BaseAgent):
    """
    Agente analista especializado en análisis de métricas y generación de insights.
    
    Capacidades principales:
    1. Análisis de métricas del sistema en tiempo real
    2. Detección de anomalías usando múltiples algoritmos
    3. Predicción de tendencias y forecasting
    4. Generación de insights y recomendaciones
    5. Validación de recolección de métricas
    6. Generación de reportes analíticos
    """
    
    # Constantes específicas del agente
    AGENT_TYPE = "analyst"
    DESCRIPTION = "Agente analista especializado en análisis de métricas, detección de anomalías y generación de insights"
    
    def __init__(self, agent_id: str, capabilities: List[str], config: Optional[Dict[str, Any]] = None,
                 orchestrator: Any = None):
        """
        Inicializa el AnalystAgent.
        
        Args:
            agent_id: Identificador único del agente
            capabilities: Lista de capacidades del agente
            config: Configuración específica del agente
            orchestrator: Referencia al orquestador (opcional)
        """
        # Configuración específica
        agent_config = AnalystAgentConfig(
            agent_type=self.AGENT_TYPE,
            agent_id=agent_id,
            capabilities=capabilities,
            **({} if config is None else config)
        )
        
        super().__init__(agent_id=agent_id, config=agent_config)
        
        # Referencia al orquestador
        self.orchestrator = orchestrator
        
        # Estado específico del analista
        self.analysis_state: Dict[str, Any] = {
            'last_analysis_time': None,
            'analysis_count': 0,
            'anomalies_detected': 0,
            'insights_generated': 0,
            'metrics_processed': 0,
        }
        
        # Almacenamiento de datos
        self.metric_store: Dict[str, TimeSeriesData] = {}
        self.anomalies: Dict[str, List[AnomalyDetectionResult]] = defaultdict(list)
        self.insights: Dict[str, List[Insight]] = defaultdict(list)
        self.trends: Dict[str, TrendAnalysisResult] = {}
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Modelos de ML
        self.isolation_forest_models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Configuración de análisis
        self.analysis_window = timedelta(hours=self.config.analysis_window_hours)
        self.min_data_points = self.config.min_data_points
        
        # Subscripciones a eventos
        self.event_subscriptions: Set[str] = set()
        
        # Cache de análisis
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=30)
        
        logger.info(f"AnalystAgent {agent_id} inicializado con capacidades: {capabilities}")
    
    # ============================================================================
    # MÉTODOS PÚBLICOS PRINCIPALES (Interfaz BaseAgent)
    # ============================================================================
    
    async def _initialize_internal(self) -> bool:
        """
        Inicialización interna específica del AnalystAgent.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            logger.info(f"Inicializando AnalystAgent {self.agent_id}")
            
            # 1. Cargar estado previo si existe
            await self._load_persistent_state()
            
            # 2. Inicializar modelos de ML
            await self._initialize_ml_models()
            
            # 3. Suscribirse a eventos del sistema
            await self._subscribe_to_events()
            
            # 4. Programar análisis periódicos
            await self._schedule_periodic_analyses()
            
            # 5. Iniciar recolección de métricas
            await self._start_metrics_collection()
            
            self.state['status'] = 'initialized'
            self.state['last_initialized'] = datetime.now()
            
            logger.info(f"AnalystAgent {self.agent_id} inicializado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando AnalystAgent {self.agent_id}: {e}")
            self.state['status'] = 'error'
            self.state['error'] = str(e)
            return False
    
    async def _process_internal(self, input_data: AgentInput) -> AgentOutput:
        """
        Procesa una solicitud de análisis.
        
        Args:
            input_data: Datos de entrada para el análisis
            
        Returns:
            AgentOutput: Resultado del análisis
        """
        start_time = datetime.now()
        result = None
        
        try:
            # Validar entrada
            self._validate_input_specific(input_data)
            
            # Determinar tipo de análisis solicitado
            analysis_type = input_data.parameters.get('analysis_type', 'system_metrics')
            
            # Ejecutar análisis según tipo
            if analysis_type == 'system_metrics':
                result = await self.analyze_system_metrics(input_data.parameters)
            elif analysis_type == 'detect_anomalies':
                result = await self.detect_anomalies(input_data.parameters)
            elif analysis_type == 'predict_trends':
                result = await self.predict_trends(input_data.parameters)
            elif analysis_type == 'generate_insights':
                result = await self.generate_insights(input_data.parameters)
            elif analysis_type == 'recommend_optimizations':
                result = await self.recommend_optimizations(input_data.parameters)
            elif analysis_type == 'validate_metric_collection':
                result = await self.validate_metric_collection(input_data.parameters)
            elif analysis_type == 'generate_analytics_report':
                result = await self.generate_analytics_report(input_data.parameters)
            else:
                raise ValidationError(f"Tipo de análisis no soportado: {analysis_type}")
            
            # Actualizar métricas
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(result, processing_time)
            
            # Guardar estado
            await self._save_state()
            
            return AgentOutput(
                success=True,
                data=result,
                metadata={
                    'processing_time': processing_time,
                    'agent_id': self.agent_id,
                    'analysis_type': analysis_type,
                    'timestamp': datetime.now().isoformat(),
                }
            )
            
        except Exception as e:
            logger.error(f"Error procesando solicitud en AnalystAgent {self.agent_id}: {e}")
            return AgentOutput(
                success=False,
                error=str(e),
                metadata={
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                }
            )
    
    async def _learn_internal(self, feedback: Dict[str, Any]) -> bool:
        """
        Aprende de la retroalimentación recibida.
        
        Args:
            feedback: Diccionario con retroalimentación
            
        Returns:
            bool: True si el aprendizaje fue exitoso
        """
        try:
            logger.info(f"AnalystAgent {self.agent_id} recibiendo feedback: {feedback.get('type')}")
            
            feedback_type = feedback.get('type', 'general')
            
            if feedback_type == 'anomaly_feedback':
                await self._learn_from_anomaly_feedback(feedback)
            elif feedback_type == 'insight_feedback':
                await self._learn_from_insight_feedback(feedback)
            elif feedback_type == 'recommendation_feedback':
                await self._learn_from_recommendation_feedback(feedback)
            elif feedback_type == 'model_correction':
                await self._learn_from_model_correction(feedback)
            else:
                await self._learn_general_feedback(feedback)
            
            # Ajustar umbrales si es necesario
            if 'adjust_thresholds' in feedback:
                await self._adjust_detection_thresholds(feedback['adjust_thresholds'])
            
            # Guardar estado después del aprendizaje
            await self._save_state()
            
            logger.info(f"Feedback procesado exitosamente por AnalystAgent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error procesando feedback en AnalystAgent {self.agent_id}: {e}")
            return False
    
    def _validate_input_specific(self, input_data: AgentInput) -> None:
        """
        Valida la entrada específica para el AnalystAgent.
        
        Args:
            input_data: Datos de entrada a validar
            
        Raises:
            ValidationError: Si la entrada no es válida
        """
        if not input_data.parameters:
            raise ValidationError("Los parámetros de entrada no pueden estar vacíos")
        
        # Validar análisis_type si está presente
        if 'analysis_type' in input_data.parameters:
            valid_types = [
                'system_metrics', 'detect_anomalies', 'predict_trends',
                'generate_insights', 'recommend_optimizations',
                'validate_metric_collection', 'generate_analytics_report'
            ]
            
            if input_data.parameters['analysis_type'] not in valid_types:
                raise ValidationError(
                    f"analysis_type debe ser uno de: {valid_types}"
                )
        
        # Validar métricas si están presentes
        if 'metrics' in input_data.parameters:
            metrics = input_data.parameters['metrics']
            if not isinstance(metrics, list):
                raise ValidationError("El parámetro 'metrics' debe ser una lista")
            
            for metric in metrics:
                if not isinstance(metric, str):
                    raise ValidationError("Cada métrica debe ser un string")
    
    async def _save_state(self) -> None:
        """Guarda el estado del agente en almacenamiento persistente."""
        try:
            state_data = {
                'agent_state': self.state,
                'analysis_state': self.analysis_state,
                'metric_store_summary': {
                    metric: len(data.data_points) 
                    for metric, data in self.metric_store.items()
                },
                'anomalies_count': {
                    metric: len(anomalies) 
                    for metric, anomalies in self.anomalies.items()
                },
                'insights_count': {
                    category: len(insights) 
                    for category, insights in self.insights.items()
                },
                'last_saved': datetime.now().isoformat(),
            }
            
            # Guardar en memoria persistente
            memory_key = f"analyst_agent_state_{self.agent_id}"
            await self.store_memory(
                AgentMemoryType.PERSISTENT,
                {
                    'type': 'agent_state',
                    'data': state_data,
                    'timestamp': datetime.now().isoformat(),
                }
            )
            
            logger.debug(f"Estado de AnalystAgent {self.agent_id} guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de AnalystAgent {self.agent_id}: {e}")
    
    # ============================================================================
    # CAPACIDADES ESPECÍFICAS DEL ANALISTA
    # ============================================================================
    
    async def analyze_system_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza métricas del sistema.
        
        Args:
            params: Parámetros del análisis
            
        Returns:
            Dict con resultados del análisis
        """
        logger.info(f"AnalystAgent {self.agent_id} analizando métricas del sistema")
        
        try:
            # Extraer parámetros
            metric_names = params.get('metrics', self.config.monitored_metrics)
            time_range = params.get('time_range', '1h')
            aggregation = params.get('aggregation', 'avg')
            
            # Obtener métricas
            metrics_data = await self._collect_metrics(metric_names, time_range)
            
            if not metrics_data:
                return {
                    'status': 'no_data',
                    'message': 'No hay datos de métricas disponibles',
                    'timestamp': datetime.now().isoformat(),
                }
            
            # Realizar análisis
            analysis_results = {}
            for metric_name, time_series in metrics_data.items():
                analysis = await self._analyze_single_metric(time_series, aggregation)
                analysis_results[metric_name] = analysis
            
            # Análisis comparativo entre métricas
            comparative_analysis = await self._perform_comparative_analysis(metrics_data)
            
            # Calcular estadísticas generales
            overall_stats = self._calculate_overall_statistics(analysis_results)
            
            # Generar resumen ejecutivo
            executive_summary = self._generate_executive_summary(analysis_results, overall_stats)
            
            # Actualizar estado
            self.analysis_state['last_analysis_time'] = datetime.now()
            self.analysis_state['analysis_count'] += 1
            self.analysis_state['metrics_processed'] += len(metrics_data)
            
            result = {
                'status': 'success',
                'executive_summary': executive_summary,
                'detailed_analysis': analysis_results,
                'comparative_analysis': comparative_analysis,
                'overall_statistics': overall_stats,
                'timestamp': datetime.now().isoformat(),
                'metrics_analyzed': len(metrics_data),
                'analysis_duration': self.analysis_state.get('last_analysis_duration', 0),
            }
            
            # Guardar en cache
            cache_key = self._generate_cache_key('system_metrics', params)
            self.analysis_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now(),
            }
            
            logger.info(f"Análisis de métricas completado para {len(metrics_data)} métricas")
            return result
            
        except Exception as e:
            logger.error(f"Error analizando métricas del sistema: {e}")
            raise
    
    async def detect_anomalies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta anomalías en las métricas del sistema.
        
        Args:
            params: Parámetros para detección de anomalías
            
        Returns:
            Dict con anomalías detectadas
        """
        logger.info(f"AnalystAgent {self.agent_id} detectando anomalías")
        
        try:
            # Extraer parámetros
            metric_names = params.get('metrics', self.config.monitored_metrics)
            detection_methods = params.get('detection_methods', ['z_score', 'isolation_forest', 'dbscan'])
            severity_threshold = params.get('severity_threshold', 0.7)
            
            # Obtener métricas
            metrics_data = await self._collect_metrics(metric_names, '24h')
            
            if not metrics_data:
                return {
                    'status': 'no_data',
                    'message': 'No hay datos de métricas disponibles',
                    'timestamp': datetime.now().isoformat(),
                }
            
            # Detectar anomalías para cada métrica
            all_anomalies = []
            metrics_with_anomalies = []
            
            for metric_name, time_series in metrics_data.items():
                if len(time_series.data_points) < self.min_data_points:
                    continue
                
                # Aplicar múltiples métodos de detección
                anomalies = await self._apply_multiple_detection_methods(
                    time_series, detection_methods
                )
                
                # Filtrar por severidad
                filtered_anomalies = [
                    anomaly for anomaly in anomalies
                    if self._calculate_anomaly_severity_score(anomaly) >= severity_threshold
                ]
                
                if filtered_anomalies:
                    all_anomalies.extend(filtered_anomalies)
                    metrics_with_anomalies.append(metric_name)
                    
                    # Almacenar anomalías
                    self.anomalies[metric_name].extend(filtered_anomalies)
                    
                    # Limitar historial de anomalías por métrica
                    max_anomalies_per_metric = 100
                    if len(self.anomalies[metric_name]) > max_anomalies_per_metric:
                        self.anomalies[metric_name] = self.anomalies[metric_name][-max_anomalies_per_metric:]
            
            # Agrupar anomalías por patrones
            grouped_anomalies = self._group_anomalies_by_pattern(all_anomalies)
            
            # Calcular impacto agregado
            total_impact = self._calculate_anomalies_impact(all_anomalies)
            
            # Generar recomendaciones para anomalías críticas
            critical_anomalies = [
                anomaly for anomaly in all_anomalies
                if anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
            ]
            
            recommendations = []
            if critical_anomalies:
                recommendations = await self._generate_anomaly_recommendations(critical_anomalies)
            
            # Actualizar estado
            self.analysis_state['anomalies_detected'] += len(all_anomalies)
            
            result = {
                'status': 'success',
                'total_anomalies_detected': len(all_anomalies),
                'metrics_with_anomalies': metrics_with_anomalies,
                'anomalies_by_severity': self._count_anomalies_by_severity(all_anomalies),
                'grouped_anomalies': grouped_anomalies,
                'critical_anomalies': [asdict(a) for a in critical_anomalies[:10]],  # Limitar salida
                'recommendations': recommendations,
                'total_impact_score': total_impact,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Publicar eventos para anomalías críticas
            if critical_anomalies:
                await self._publish_anomaly_events(critical_anomalies)
            
            logger.info(f"Detección de anomalías completada: {len(all_anomalies)} anomalías encontradas")
            return result
            
        except Exception as e:
            logger.error(f"Error detectando anomalías: {e}")
            raise
    
    async def predict_trends(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predice tendencias futuras basadas en datos históricos.
        
        Args:
            params: Parámetros para predicción de tendencias
            
        Returns:
            Dict con predicciones y análisis de tendencias
        """
        logger.info(f"AnalystAgent {self.agent_id} prediciendo tendencias")
        
        try:
            # Extraer parámetros
            metric_names = params.get('metrics', self.config.monitored_metrics)
            forecast_horizon = params.get('forecast_horizon', self.config.forecast_horizon)
            confidence_level = params.get('confidence_level', 0.95)
            
            # Obtener datos históricos
            historical_data = {}
            for metric_name in metric_names:
                if metric_name in self.metric_store:
                    time_series = self.metric_store[metric_name]
                    if len(time_series.data_points) >= self.min_data_points:
                        historical_data[metric_name] = time_series
            
            if not historical_data:
                return {
                    'status': 'insufficient_data',
                    'message': 'Datos históricos insuficientes para predicción',
                    'timestamp': datetime.now().isoformat(),
                }
            
            # Realizar predicciones para cada métrica
            predictions = {}
            trend_analyses = {}
            
            for metric_name, time_series in historical_data.items():
                # Analizar tendencia actual
                trend_analysis = await self._analyze_trend(time_series)
                trend_analyses[metric_name] = trend_analysis
                
                # Predecir valores futuros
                forecast = await self._forecast_values(
                    time_series, 
                    forecast_horizon, 
                    confidence_level
                )
                
                predictions[metric_name] = forecast
            
            # Identificar tendencias cruzadas
            cross_trends = await self._analyze_cross_metric_trends(trend_analyses)
            
            # Generar insights de tendencias
            trend_insights = await self._generate_trend_insights(trend_analyses, predictions)
            
            # Calcular indicadores de riesgo
            risk_indicators = self._calculate_trend_risk_indicators(predictions)
            
            result = {
                'status': 'success',
                'trend_analyses': {
                    metric: asdict(analysis) 
                    for metric, analysis in trend_analyses.items()
                },
                'predictions': predictions,
                'cross_trends': cross_trends,
                'trend_insights': trend_insights,
                'risk_indicators': risk_indicators,
                'forecast_horizon': forecast_horizon,
                'confidence_level': confidence_level,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Actualizar almacenamiento de tendencias
            for metric_name, analysis in trend_analyses.items():
                self.trends[metric_name] = analysis
            
            logger.info(f"Predicción de tendencias completada para {len(historical_data)} métricas")
            return result
            
        except Exception as e:
            logger.error(f"Error prediciendo tendencias: {e}")
            raise
    
    async def generate_insights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera insights basados en análisis de métricas.
        
        Args:
            params: Parámetros para generación de insights
            
        Returns:
            Dict con insights generados
        """
        logger.info(f"AnalystAgent {self.agent_id} generando insights")
        
        try:
            # Extraer parámetros
            insight_categories = params.get('categories', ['performance', 'cost', 'reliability', 'security'])
            min_confidence = params.get('min_confidence', 0.7)
            limit = params.get('limit', 20)
            
            # Recolectar datos para análisis
            analysis_data = await self._collect_insight_data(insight_categories)
            
            if not analysis_data:
                return {
                    'status': 'no_data',
                    'message': 'Datos insuficientes para generar insights',
                    'timestamp': datetime.now().isoformat(),
                }
            
            # Generar insights por categoría
            generated_insights = []
            
            for category in insight_categories:
                category_data = analysis_data.get(category, {})
                if not category_data:
                    continue
                
                # Generar insights específicos de la categoría
                category_insights = await self._generate_category_insights(
                    category, category_data, min_confidence
                )
                
                # Filtrar y ordenar por impacto
                filtered_insights = [
                    insight for insight in category_insights
                    if insight.confidence >= min_confidence
                ]
                
                filtered_insights.sort(key=lambda x: x.impact_score, reverse=True)
                
                # Limitar por categoría
                max_per_category = self.config.max_insights_per_category
                category_insights_limited = filtered_insights[:max_per_category]
                
                generated_insights.extend(category_insights_limited)
                
                # Almacenar insights
                self.insights[category].extend(category_insights_limited)
                
                # Limitar historial de insights por categoría
                if len(self.insights[category]) > max_per_category * 2:
                    self.insights[category] = self.insights[category][-max_per_category * 2:]
            
            # Ordenar todos los insights por impacto
            generated_insights.sort(key=lambda x: (x.impact_score * x.confidence), reverse=True)
            
            # Limitar total de insights
            final_insights = generated_insights[:limit]
            
            # Agrupar insights por prioridad
            prioritized_insights = self._prioritize_insights(final_insights)
            
            # Calcular impacto agregado
            total_impact = sum(insight.impact_score for insight in final_insights)
            avg_confidence = mean(insight.confidence for insight in final_insights) if final_insights else 0
            
            # Actualizar estado
            self.analysis_state['insights_generated'] += len(final_insights)
            
            result = {
                'status': 'success',
                'total_insights_generated': len(final_insights),
                'insights_by_category': {
                    category: len([i for i in final_insights if i.category == category])
                    for category in insight_categories
                },
                'prioritized_insights': prioritized_insights,
                'top_insights': [asdict(insight) for insight in final_insights[:10]],
                'aggregate_impact': total_impact,
                'average_confidence': avg_confidence,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Publicar insights importantes
            high_impact_insights = [
                insight for insight in final_insights 
                if insight.impact_score >= 0.8 and insight.confidence >= 0.8
            ]
            
            if high_impact_insights:
                await self._publish_insight_events(high_impact_insights)
            
            logger.info(f"Generación de insights completada: {len(final_insights)} insights generados")
            return result
            
        except Exception as e:
            logger.error(f"Error generando insights: {e}")
            raise
    
    async def recommend_optimizations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recomienda optimizaciones basadas en análisis de métricas.
        
        Args:
            params: Parámetros para recomendaciones de optimización
            
        Returns:
            Dict con recomendaciones de optimización
        """
        logger.info(f"AnalystAgent {self.agent_id} generando recomendaciones de optimización")
        
        try:
            # Extraer parámetros
            optimization_areas = params.get('areas', ['performance', 'cost', 'reliability', 'security'])
            max_recommendations = params.get('max_recommendations', 10)
            min_impact = params.get('min_impact', 0.3)
            
            # Analizar áreas para optimización
            area_analyses = {}
            for area in optimization_areas:
                analysis = await self._analyze_optimization_area(area)
                if analysis['optimization_potential'] > min_impact:
                    area_analyses[area] = analysis
            
            if not area_analyses:
                return {
                    'status': 'no_optimizations',
                    'message': 'No se encontraron áreas con suficiente potencial de optimización',
                    'timestamp': datetime.now().isoformat(),
                }
            
            # Generar recomendaciones para cada área
            all_recommendations = []
            
            for area, analysis in area_analyses.items():
                area_recommendations = await self._generate_area_recommendations(area, analysis)
                
                # Filtrar por impacto mínimo
                filtered_recommendations = [
                    rec for rec in area_recommendations
                    if rec.estimated_impact >= min_impact
                ]
                
                all_recommendations.extend(filtered_recommendations)
            
            # Ordenar por prioridad (impacto/effort)
            all_recommendations.sort(key=lambda x: x.priority, reverse=True)
            
            # Limitar número de recomendaciones
            final_recommendations = all_recommendations[:max_recommendations]
            
            # Agrupar por área y prioridad
            grouped_recommendations = self._group_recommendations(final_recommendations)
            
            # Calcular impacto total estimado
            total_estimated_impact = sum(rec.estimated_impact for rec in final_recommendations)
            total_estimated_effort = sum(rec.estimated_effort for rec in final_recommendations)
            
            # Generar plan de implementación
            implementation_plan = await self._generate_implementation_plan(final_recommendations)
            
            # Actualizar almacenamiento
            self.recommendations.extend(final_recommendations)
            
            # Limitar historial de recomendaciones
            max_recommendations_history = 50
            if len(self.recommendations) > max_recommendations_history:
                self.recommendations = self.recommendations[-max_recommendations_history:]
            
            result = {
                'status': 'success',
                'total_recommendations': len(final_recommendations),
                'recommendations_by_area': {
                    area: len([r for r in final_recommendations if r.area == area])
                    for area in optimization_areas
                },
                'grouped_recommendations': grouped_recommendations,
                'top_recommendations': [asdict(rec) for rec in final_recommendations[:5]],
                'implementation_plan': implementation_plan,
                'total_estimated_impact': total_estimated_impact,
                'total_estimated_effort': total_estimated_effort,
                'overall_priority_score': total_estimated_impact / total_estimated_effort if total_estimated_effort > 0 else 0,
                'timestamp': datetime.now().isoformat(),
            }
            
            logger.info(f"Generación de recomendaciones completada: {len(final_recommendations)} recomendaciones")
            return result
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones de optimización: {e}")
            raise
    
    async def validate_metric_collection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida la recolección y calidad de las métricas.
        
        Args:
            params: Parámetros para validación de métricas
            
        Returns:
            Dict con resultados de validación
        """
        logger.info(f"AnalystAgent {self.agent_id} validando recolección de métricas")
        
        try:
            # Extraer parámetros
            metric_names = params.get('metrics', self.config.monitored_metrics)
            validation_criteria = params.get('criteria', ['completeness', 'consistency', 'timeliness', 'accuracy'])
            
            # Validar cada métrica
            validation_results = {}
            issues_found = []
            
            for metric_name in metric_names:
                if metric_name not in self.metric_store:
                    issues_found.append({
                        'metric': metric_name,
                        'issue': 'metric_not_collected',
                        'severity': 'HIGH',
                        'description': f'La métrica {metric_name} no está siendo recolectada',
                    })
                    continue
                
                time_series = self.metric_store[metric_name]
                
                # Aplicar criterios de validación
                metric_validation = {}
                for criterion in validation_criteria:
                    if criterion == 'completeness':
                        result = self._validate_completeness(time_series)
                    elif criterion == 'consistency':
                        result = self._validate_consistency(time_series)
                    elif criterion == 'timeliness':
                        result = self._validate_timeliness(time_series)
                    elif criterion == 'accuracy':
                        result = self._validate_accuracy(time_series)
                    else:
                        result = {'valid': True, 'score': 1.0, 'issues': []}
                    
                    metric_validation[criterion] = result
                    
                    # Recolectar issues
                    if result.get('issues'):
                        for issue in result['issues']:
                            issues_found.append({
                                'metric': metric_name,
                                'criterion': criterion,
                                **issue,
                            })
                
                validation_results[metric_name] = metric_validation
            
            # Calcular puntajes agregados
            overall_scores = self._calculate_validation_scores(validation_results)
            
            # Generar recomendaciones de mejora
            improvement_recommendations = []
            if issues_found:
                improvement_recommendations = await self._generate_metric_improvement_recommendations(issues_found)
            
            # Clasificar métricas por calidad
            metric_quality_grades = self._grade_metric_quality(validation_results)
            
            result = {
                'status': 'success',
                'metrics_validated': len(validation_results),
                'validation_results': validation_results,
                'overall_scores': overall_scores,
                'issues_found': len(issues_found),
                'critical_issues': [
                    issue for issue in issues_found 
                    if issue.get('severity') in ['HIGH', 'CRITICAL']
                ][:20],  # Limitar salida
                'improvement_recommendations': improvement_recommendations,
                'metric_quality_grades': metric_quality_grades,
                'timestamp': datetime.now().isoformat(),
            }
            
            # Publicar alertas para issues críticos
            critical_issues = [
                issue for issue in issues_found 
                if issue.get('severity') in ['HIGH', 'CRITICAL']
            ]
            
            if critical_issues:
                await self._publish_validation_events(critical_issues)
            
            logger.info(f"Validación de métricas completada: {len(validation_results)} métricas validadas")
            return result
            
        except Exception as e:
            logger.error(f"Error validando recolección de métricas: {e}")
            raise
    
    async def generate_analytics_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera un reporte analítico completo.
        
        Args:
            params: Parámetros para generación del reporte
            
        Returns:
            Dict con el reporte analítico
        """
        logger.info(f"AnalystAgent {self.agent_id} generando reporte analítico")
        
        try:
            # Extraer parámetros
            report_type = params.get('report_type', 'comprehensive')
            time_range = params.get('time_range', '24h')
            include_sections = params.get('sections', [
                'executive_summary', 'metrics_analysis', 'anomalies', 
                'trends', 'insights', 'recommendations'
            ])
            
            report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Recolectar datos para el reporte
            report_data = {
                'metadata': {
                    'report_id': report_id,
                    'generated_by': self.agent_id,
                    'generated_at': datetime.now().isoformat(),
                    'time_range': time_range,
                    'report_type': report_type,
                }
            }
            
            # Generar secciones del reporte
            if 'executive_summary' in include_sections:
                report_data['executive_summary'] = await self._generate_executive_report_summary(time_range)
            
            if 'metrics_analysis' in include_sections:
                report_data['metrics_analysis'] = await self.analyze_system_metrics({
                    'time_range': time_range,
                })
            
            if 'anomalies' in include_sections:
                report_data['anomalies'] = await self.detect_anomalies({
                    'time_range': time_range,
                })
            
            if 'trends' in include_sections:
                report_data['trends'] = await self.predict_trends({
                    'forecast_horizon': 12,  # Pronóstico de 12 períodos
                })
            
            if 'insights' in include_sections:
                report_data['insights'] = await self.generate_insights({
                    'limit': 15,
                })
            
            if 'recommendations' in include_sections:
                report_data['recommendations'] = await self.recommend_optimizations({
                    'max_recommendations': 10,
                })
            
            # Calcular KPIs del reporte
            report_kpis = await self._calculate_report_kpis(report_data)
            
            # Generar conclusiones
            conclusions = await self._generate_report_conclusions(report_data)
            
            # Agregar KPIs y conclusiones
            report_data['kpis'] = report_kpis
            report_data['conclusions'] = conclusions
            
            # Formatear reporte según tipo
            formatted_report = await self._format_report(report_data, report_type)
            
            # Almacenar reporte
            await self._store_report(report_id, report_data)
            
            result = {
                'status': 'success',
                'report_id': report_id,
                'report_type': report_type,
                'sections_included': include_sections,
                'report_summary': {
                    'total_metrics_analyzed': report_kpis.get('total_metrics', 0),
                    'anomalies_detected': report_kpis.get('total_anomalies', 0),
                    'insights_generated': report_kpis.get('total_insights', 0),
                    'recommendations_provided': report_kpis.get('total_recommendations', 0),
                    'overall_health_score': report_kpis.get('overall_health_score', 0),
                },
                'formatted_report': formatted_report if report_type == 'summary' else report_data,
                'timestamp': datetime.now().isoformat(),
            }
            
            logger.info(f"Reporte analítico generado: {report_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generando reporte analítico: {e}")
            raise
    
    # ============================================================================
    # MÉTODOS AUXILIARES PRIVADOS
    # ============================================================================
    
    async def _initialize_ml_models(self):
        """Inicializa modelos de machine learning para detección de anomalías."""
        logger.info(f"Inicializando modelos ML para AnalystAgent {self.agent_id}")
        
        for metric_name in self.config.monitored_metrics:
            self.isolation_forest_models[metric_name] = IsolationForest(
                contamination=self.config.isolation_forest_contamination,
                random_state=42,
                n_estimators=100,
            )
            
            self.scalers[metric_name] = StandardScaler()
    
    async def _subscribe_to_events(self):
        """Se suscribe a eventos del sistema."""
        if self.orchestrator and hasattr(self.orchestrator, 'event_bus'):
            event_bus: EventBus = self.orchestrator.event_bus
            
            # Suscribirse a eventos de métricas
            await event_bus.subscribe('metrics.updated', self._handle_metrics_update)
            await event_bus.subscribe('system.alert', self._handle_system_alert)
            await event_bus.subscribe('analysis.requested', self._handle_analysis_request)
            
            self.event_subscriptions.update([
                'metrics.updated',
                'system.alert', 
                'analysis.requested',
            ])
            
            logger.info(f"AnalystAgent {self.agent_id} suscrito a eventos del sistema")
    
    async def _schedule_periodic_analyses(self):
        """Programa análisis periódicos."""
        # En una implementación real, se usaría asyncio.create_task
        # para programar análisis periódicos
        
        logger.info(f"AnalystAgent {self.agent_id} programando análisis periódicos")
        
        # Análisis de métricas cada 5 minutos
        self.periodic_tasks = [
            asyncio.create_task(self._periodic_metrics_analysis()),
            asyncio.create_task(self._periodic_anomaly_detection()),
            asyncio.create_task(self._periodic_report_generation()),
        ]
    
    async def _start_metrics_collection(self):
        """Inicia la recolección de métricas."""
        logger.info(f"AnalystAgent {self.agent_id} iniciando recolección de métricas")
        
        # En una implementación real, se conectaría a un sistema de métricas
        # como Prometheus, Datadog, o un sistema personalizado
        
        # Por ahora, inicializamos con datos de ejemplo
        await self._initialize_sample_metrics()
    
    async def _collect_metrics(self, metric_names: List[str], time_range: str) -> Dict[str, TimeSeriesData]:
        """
        Recolecta métricas del sistema.
        
        Args:
            metric_names: Lista de nombres de métricas
            time_range: Rango de tiempo (ej: '1h', '24h', '7d')
            
        Returns:
            Dict con series temporales de métricas
        """
        collected_metrics = {}
        
        for metric_name in metric_names:
            if metric_name in self.metric_store:
                # Filtrar por rango de tiempo
                time_series = self.metric_store[metric_name]
                filtered_points = self._filter_by_time_range(time_series.data_points, time_range)
                
                if filtered_points:
                    collected_metrics[metric_name] = TimeSeriesData(
                        metric_name=metric_name,
                        metric_type=time_series.metric_type,
                        data_points=filtered_points,
                        unit=time_series.unit,
                        description=time_series.description,
                    )
        
        return collected_metrics
    
    async def _analyze_single_metric(self, time_series: TimeSeriesData, aggregation: str) -> Dict[str, Any]:
        """
        Analiza una única métrica.
        
        Args:
            time_series: Serie temporal a analizar
            aggregation: Tipo de agregación ('avg', 'sum', 'max', 'min', 'p95', etc.)
            
        Returns:
            Dict con análisis de la métrica
        """
        values = [point.value for point in time_series.data_points]
        timestamps = [point.timestamp for point in time_series.data_points]
        
        if not values:
            return {'error': 'No data points available'}
        
        # Estadísticas básicas
        stats_result = {
            'count': len(values),
            'mean': mean(values) if len(values) > 1 else values[0],
            'median': median(values) if len(values) > 1 else values[0],
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values),
        }
        
        # Estadísticas adicionales si hay suficientes datos
        if len(values) > 2:
            try:
                stats_result.update({
                    'std_dev': stdev(values),
                    'variance': variance(values),
                    'coefficient_of_variation': stdev(values) / mean(values) if mean(values) != 0 else 0,
                })
            except:
                pass
        
        # Valor agregado según tipo de agregación
        if aggregation == 'avg':
            aggregated_value = stats_result['mean']
        elif aggregation == 'sum':
            aggregated_value = sum(values)
        elif aggregation == 'max':
            aggregated_value = stats_result['max']
        elif aggregation == 'min':
            aggregated_value = stats_result['min']
        elif aggregation == 'p95':
            aggregated_value = np.percentile(values, 95)
        elif aggregation == 'p99':
            aggregated_value = np.percentile(values, 99)
        else:
            aggregated_value = stats_result['mean']
        
        # Detectar valores atípicos simples
        outliers = self._detect_simple_outliers(values)
        
        # Calcular tasa de cambio
        change_rate = self._calculate_change_rate(values, timestamps)
        
        return {
            'statistics': stats_result,
            'aggregated_value': aggregated_value,
            'aggregation_type': aggregation,
            'outliers_detected': len(outliers),
            'outlier_indices': outliers,
            'change_rate': change_rate,
            'time_range': {
                'start': timestamps[0].isoformat() if timestamps else None,
                'end': timestamps[-1].isoformat() if timestamps else None,
                'duration_seconds': (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0,
            },
        }
    
    async def _apply_multiple_detection_methods(self, time_series: TimeSeriesData, 
                                              methods: List[str]) -> List[AnomalyDetectionResult]:
        """
        Aplica múltiples métodos de detección de anomalías.
        
        Args:
            time_series: Serie temporal a analizar
            methods: Lista de métodos a aplicar
            
        Returns:
            Lista de anomalías detectadas
        """
        all_anomalies = []
        values = [point.value for point in time_series.data_points]
        timestamps = [point.timestamp for point in time_series.data_points]
        
        if len(values) < self.min_data_points:
            return []
        
        for method in methods:
            try:
                if method == 'z_score':
                    anomalies = self._detect_anomalies_z_score(values, timestamps, time_series.metric_name)
                elif method == 'isolation_forest':
                    anomalies = await self._detect_anomalies_isolation_forest(values, timestamps, time_series.metric_name)
                elif method == 'dbscan':
                    anomalies = self._detect_anomalies_dbscan(values, timestamps, time_series.metric_name)
                elif method == 'threshold':
                    anomalies = self._detect_anomalies_threshold(values, timestamps, time_series.metric_name)
                elif method == 'moving_average':
                    anomalies = self._detect_anomalies_moving_average(values, timestamps, time_series.metric_name)
                else:
                    continue
                
                all_anomalies.extend(anomalies)
                
            except Exception as e:
                logger.warning(f"Error aplicando método {method} para {time_series.metric_name}: {e}")
        
        # Eliminar duplicados (mismo timestamp y métrica)
        unique_anomalies = []
        seen = set()
        
        for anomaly in all_anomalies:
            key = (anomaly.metric_name, anomaly.timestamp)
            if key not in seen:
                seen.add(key)
                unique_anomalies.append(anomaly)
        
        return unique_anomalies
    
    def _detect_anomalies_z_score(self, values: List[float], timestamps: List[datetime], 
                                metric_name: str) -> List[AnomalyDetectionResult]:
        """Detecta anomalías usando el método Z-Score."""
        anomalies = []
        
        if len(values) < 2:
            return anomalies
        
        mean_val = mean(values)
        std_val = stdev(values) if len(values) > 1 else 0
        
        if std_val == 0:
            return anomalies
        
        threshold = self.config.z_score_threshold
        
        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            z_score = abs((value - mean_val) / std_val)
            
            if z_score > threshold:
                # Determinar severidad basada en el z-score
                if z_score > threshold * 3:
                    severity = AnomalySeverity.CRITICAL
                elif z_score > threshold * 2:
                    severity = AnomalySeverity.HIGH
                elif z_score > threshold * 1.5:
                    severity = AnomalySeverity.MEDIUM
                else:
                    severity = AnomalySeverity.LOW
                
                anomaly = AnomalyDetectionResult(
                    anomaly_id=f"zscore_{metric_name}_{timestamp.timestamp()}",
                    metric_name=metric_name,
                    timestamp=timestamp,
                    value=value,
                    expected_value=mean_val,
                    deviation=z_score,
                    severity=severity,
                    confidence=min(z_score / (threshold * 4), 1.0),
                    pattern="statistical_outlier",
                    root_cause_hypotheses=[
                        f"Valor estadísticamente atípico (z-score: {z_score:.2f})",
                        "Posible error de medición o evento inusual",
                    ],
                    metadata={
                        'detection_method': 'z_score',
                        'z_score': z_score,
                        'threshold': threshold,
                        'mean': mean_val,
                        'std_dev': std_val,
                        'index': i,
                    },
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_anomalies_isolation_forest(self, values: List[float], timestamps: List[datetime],
                                               metric_name: str) -> List[AnomalyDetectionResult]:
        """Detecta anomalías usando Isolation Forest."""
        anomalies = []
        
        if len(values) < 10:  # Isolation Forest necesita más datos
            return anomalies
        
        # Preparar datos
        X = np.array(values).reshape(-1, 1)
        
        # Obtener o entrenar modelo
        if metric_name in self.isolation_forest_models:
            model = self.isolation_forest_models[metric_name]
            scaler = self.scalers[metric_name]
            
            # Escalar datos
            X_scaled = scaler.transform(X)
            
            # Predecir anomalías
            predictions = model.fit_predict(X_scaled)
            anomaly_scores = model.score_samples(X_scaled)
            
            # Identificar anomalías (predictions == -1)
            for i, (pred, score, value, timestamp) in enumerate(zip(predictions, anomaly_scores, values, timestamps)):
                if pred == -1:
                    # Convertir score de anomalía a severidad
                    # Scores más negativos = más anómalo
                    normalized_score = abs(score)
                    
                    if normalized_score > 0.7:
                        severity = AnomalySeverity.CRITICAL
                    elif normalized_score > 0.5:
                        severity = AnomalySeverity.HIGH
                    elif normalized_score > 0.3:
                        severity = AnomalySeverity.MEDIUM
                    else:
                        severity = AnomalySeverity.LOW
                    
                    anomaly = AnomalyDetectionResult(
                        anomaly_id=f"iforest_{metric_name}_{timestamp.timestamp()}",
                        metric_name=metric_name,
                        timestamp=timestamp,
                        value=value,
                        expected_value=np.median(values),
                        deviation=normalized_score,
                        severity=severity,
                        confidence=normalized_score,
                        pattern="isolation_forest_outlier",
                        root_cause_hypotheses=[
                            f"Patrón inusual detectado por Isolation Forest (score: {normalized_score:.2f})",
                            "Comportamiento diferente del patrón histórico",
                        ],
                        metadata={
                            'detection_method': 'isolation_forest',
                            'anomaly_score': score,
                            'normalized_score': normalized_score,
                            'index': i,
                        },
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_anomalies_dbscan(self, values: List[float], timestamps: List[datetime],
                                metric_name: str) -> List[AnomalyDetectionResult]:
        """Detecta anomalías usando DBSCAN."""
        anomalies = []
        
        if len(values) < self.config.dbscan_min_samples * 2:
            return anomalies
        
        # Preparar datos con índice temporal
        X = np.array(list(zip(range(len(values)), values)))
        
        # Aplicar DBSCAN
        dbscan = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
        )
        
        clusters = dbscan.fit_predict(X)
        
        # Los puntos con cluster = -1 son anomalías
        for i, (cluster, value, timestamp) in enumerate(zip(clusters, values, timestamps)):
            if cluster == -1:
                # Calcular distancia al cluster más cercano
                distances = []
                for j, other_cluster in enumerate(clusters):
                    if other_cluster != -1 and i != j:
                        dist = abs(value - values[j])
                        distances.append(dist)
                
                avg_distance = mean(distances) if distances else 0
                
                # Determinar severidad
                if avg_distance > np.percentile(values, 90) - np.percentile(values, 10):
                    severity = AnomalySeverity.HIGH
                else:
                    severity = AnomalySeverity.MEDIUM
                
                anomaly = AnomalyDetectionResult(
                    anomaly_id=f"dbscan_{metric_name}_{timestamp.timestamp()}",
                    metric_name=metric_name,
                    timestamp=timestamp,
                    value=value,
                    expected_value=np.median([v for v, c in zip(values, clusters) if c != -1]),
                    deviation=avg_distance,
                    severity=severity,
                    confidence=min(avg_distance / (max(values) - min(values)), 1.0) if max(values) != min(values) else 0.5,
                    pattern="density_based_outlier",
                    root_cause_hypotheses=[
                        "Punto de baja densidad en el espacio de características",
                        "Comportamiento aislado del resto de datos",
                    ],
                    metadata={
                        'detection_method': 'dbscan',
                        'cluster_id': -1,
                        'avg_distance_to_cluster': avg_distance,
                        'index': i,
                    },
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_anomaly_severity_score(self, anomaly: AnomalyDetectionResult) -> float:
        """Calcula un score de severidad para una anomalía."""
        base_score = 0.0
        
        # Base en el nivel de severidad
        severity_weights = {
            AnomalySeverity.INFO: 0.1,
            AnomalySeverity.LOW: 0.3,
            AnomalySeverity.MEDIUM: 0.6,
            AnomalySeverity.HIGH: 0.8,
            AnomalySeverity.CRITICAL: 1.0,
        }
        
        base_score = severity_weights.get(anomaly.severity, 0.5)
        
        # Ajustar por confianza
        adjusted_score = base_score * anomaly.confidence
        
        # Ajustar por desviación (si es muy grande)
        if anomaly.deviation > 10:  # Desviación muy grande
            adjusted_score = min(adjusted_score * 1.2, 1.0)
        
        return adjusted_score
    
    async def _analyze_trend(self, time_series: TimeSeriesData) -> TrendAnalysisResult:
        """Analiza la tendencia de una serie temporal."""
        values = [point.value for point in time_series.data_points]
        timestamps = [point.timestamp for point in time_series.data_points]
        
        if len(values) < 2:
            return TrendAnalysisResult(
                metric_name=time_series.metric_name,
                trend_direction="insufficient_data",
                trend_strength=0.0,
                slope=0.0,
                intercept=0.0,
                r_squared=0.0,
            )
        
        # Convertir timestamps a valores numéricos
        time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Regresión lineal simple
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
        r_squared = r_value ** 2
        
        # Determinar dirección de tendencia
        if abs(slope) < 0.001:
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "increasing"
            strength = min(abs(slope) * 100, 1.0)  # Normalizado
        else:
            direction = "decreasing"
            strength = min(abs(slope) * 100, 1.0)
        
        # Detectar estacionalidad (simplificado)
        seasonal, period = self._detect_seasonality(values)
        
        # Detectar puntos de cambio
        changepoints = self._detect_changepoints(values, timestamps)
        
        return TrendAnalysisResult(
            metric_name=time_series.metric_name,
            trend_direction=direction,
            trend_strength=strength,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            seasonality_detected=seasonal,
            seasonality_period=period,
            changepoints=changepoints,
        )
    
    async def _forecast_values(self, time_series: TimeSeriesData, horizon: int, 
                             confidence_level: float) -> Dict[str, Any]:
        """Pronostica valores futuros de una serie temporal."""
        values = [point.value for point in time_series.data_points]
        
        if len(values) < self.min_data_points:
            return {
                'forecast': [],
                'confidence_intervals': [],
                'method': 'insufficient_data',
            }
        
        # Método simple: promedio móvil
        window_size = min(5, len(values) // 2)
        last_values = values[-window_size:]
        
        # Pronóstico simple
        forecast = []
        for _ in range(horizon):
            # Promedio ponderado de valores recientes
            forecast_value = np.average(last_values, weights=range(1, len(last_values) + 1))
            forecast.append(forecast_value)
            
            # Actualizar para siguiente paso (simulando)
            last_values.append(forecast_value)
            last_values = last_values[1:]
        
        # Calcular intervalos de confianza
        std_dev = np.std(values) if len(values) > 1 else 0
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * std_dev / np.sqrt(len(values))
        
        confidence_intervals = [
            {
                'lower': max(0, f - margin_of_error),  # No valores negativos para métricas
                'upper': f + margin_of_error,
                'confidence': confidence_level,
            }
            for f in forecast
        ]
        
        # Generar timestamps futuros
        last_timestamp = time_series.data_points[-1].timestamp
        forecast_timestamps = [
            last_timestamp + timedelta(hours=i+1)  # Asumir intervalos de 1 hora
            for i in range(horizon)
        ]
        
        return {
            'forecast': forecast,
            'forecast_timestamps': [ts.isoformat() for ts in forecast_timestamps],
            'confidence_intervals': confidence_intervals,
            'method': 'weighted_moving_average',
            'window_size': window_size,
            'std_dev': std_dev,
            'margin_of_error': margin_of_error,
        }
    
    # ============================================================================
    # MÉTODOS DE UTILIDAD
    # ============================================================================
    
    def _filter_by_time_range(self, data_points: List[MetricDataPoint], 
                            time_range: str) -> List[MetricDataPoint]:
        """Filtra puntos de datos por rango de tiempo."""
        now = datetime.now()
        
        if time_range.endswith('h'):
            hours = int(time_range[:-1])
            cutoff = now - timedelta(hours=hours)
        elif time_range.endswith('d'):
            days = int(time_range[:-1])
            cutoff = now - timedelta(days=days)
        elif time_range.endswith('m'):
            minutes = int(time_range[:-1])
            cutoff = now - timedelta(minutes=minutes)
        else:
            # Valor por defecto: 1 hora
            cutoff = now - timedelta(hours=1)
        
        return [point for point in data_points if point.timestamp >= cutoff]
    
    def _detect_simple_outliers(self, values: List[float]) -> List[int]:
        """Detecta valores atípicos simples usando el rango intercuartílico."""
        if len(values) < 4:
            return []
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return []
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)
        
        return outliers
    
    def _calculate_change_rate(self, values: List[float], timestamps: List[datetime]) -> float:
        """Calcula la tasa de cambio de los valores."""
        if len(values) < 2:
            return 0.0
        
        # Calcular diferencias porcentuales
        changes = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                change = (values[i] - values[i-1]) / abs(values[i-1])
                changes.append(abs(change))
        
        if not changes:
            return 0.0
        
        return mean(changes)
    
    def _detect_seasonality(self, values: List[float]) -> Tuple[bool, Optional[int]]:
        """Detecta estacionalidad en una serie temporal (simplificado)."""
        if len(values) < 20:
            return False, None
        
        # Usar autocorrelación simple
        max_lag = min(10, len(values) // 2)
        
        # Calcular autocorrelación
        autocorr = []
        for lag in range(1, max_lag + 1):
            if lag < len(values):
                corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                autocorr.append(abs(corr))
        
        # Encontrar picos en autocorrelación
        if autocorr:
            max_corr = max(autocorr)
            if max_corr > 0.7:  # Umbral alto para estacionalidad clara
                period = autocorr.index(max_corr) + 1
                return True, period
        
        return False, None
    
    def _detect_changepoints(self, values: List[float], timestamps: List[datetime]) -> List[datetime]:
        """Detecta puntos de cambio en una serie temporal (simplificado)."""
        if len(values) < 10:
            return []
        
        changepoints = []
        
        # Método simple: buscar cambios significativos en la media móvil
        window_size = min(5, len(values) // 3)
        
        for i in range(window_size, len(values) - window_size):
            prev_mean = mean(values[i-window_size:i])
            next_mean = mean(values[i:i+window_size])
            
            # Calcular cambio porcentual
            if prev_mean != 0:
                change = abs(next_mean - prev_mean) / abs(prev_mean)
                
                # Si el cambio es mayor al 50%
                if change > 0.5:
                    changepoints.append(timestamps[i])
        
        return changepoints
    
    def _generate_cache_key(self, analysis_type: str, params: Dict[str, Any]) -> str:
        """Genera una clave de caché única para un análisis."""
        params_str = json.dumps(params, sort_keys=True, default=str)
        hash_input = f"{analysis_type}_{params_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _initialize_sample_metrics(self):
        """Inicializa métricas de ejemplo para desarrollo."""
        logger.info(f"Inicializando métricas de ejemplo para AnalystAgent {self.agent_id}")
        
        now = datetime.now()
        
        # Métricas de ejemplo
        sample_metrics = {
            'system.cpu.usage': {
                'type': MetricType.GAUGE,
                'unit': 'percent',
                'description': 'Uso de CPU del sistema',
                'values': np.random.normal(40, 10, 100).tolist(),  # Normal around 40%
            },
            'system.memory.usage': {
                'type': MetricType.GAUGE,
                'unit': 'percent',
                'description': 'Uso de memoria del sistema',
                'values': np.random.normal(60, 15, 100).tolist(),  # Normal around 60%
            },
            'application.response.time': {
                'type': MetricType.HISTOGRAM,
                'unit': 'milliseconds',
                'description': 'Tiempo de respuesta de la aplicación',
                'values': np.random.exponential(200, 100).tolist(),  # Exponential, mean 200ms
            },
            'application.error.rate': {
                'type': MetricType.RATE,
                'unit': 'errors per second',
                'description': 'Tasa de errores de la aplicación',
                'values': np.random.poisson(2, 100).tolist(),  # Poisson, mean 2
            },
        }
        
        # Crear series temporales de ejemplo
        for metric_name, metric_info in sample_metrics.items():
            data_points = []
            
            for i, value in enumerate(metric_info['values']):
                timestamp = now - timedelta(minutes=len(metric_info['values']) - i)
                
                # Añadir algo de variación temporal
                if i == 50:  # Un pico artificial
                    value = value * 3
                
                data_points.append(MetricDataPoint(
                    timestamp=timestamp,
                    value=max(0, value),  # No valores negativos
                    labels={'source': 'sample'},
                    metadata={'sample': True},
                ))
            
            self.metric_store[metric_name] = TimeSeriesData(
                metric_name=metric_name,
                metric_type=metric_info['type'],
                data_points=data_points,
                unit=metric_info['unit'],
                description=metric_info['description'],
            )
        
        logger.info(f"Métricas de ejemplo inicializadas: {len(self.metric_store)} series")
    
    # ============================================================================
    # MÉTODOS DE EVENT HANDLING
    # ============================================================================
    
    async def _handle_metrics_update(self, event_data: Dict[str, Any]):
        """Maneja actualizaciones de métricas."""
        try:
            metric_name = event_data.get('metric_name')
            value = event_data.get('value')
            timestamp = event_data.get('timestamp', datetime.now())
            labels = event_data.get('labels', {})
            metadata = event_data.get('metadata', {})
            
            if metric_name and value is not None:
                # Crear o actualizar serie temporal
                if metric_name not in self.metric_store:
                    self.metric_store[metric_name] = TimeSeriesData(
                        metric_name=metric_name,
                        metric_type=MetricType.GAUGE,  # Por defecto
                        data_points=[],
                        unit=event_data.get('unit', ''),
                        description=event_data.get('description', ''),
                    )
                
                # Añadir punto de datos
                data_point = MetricDataPoint(
                    timestamp=timestamp,
                    value=float(value),
                    labels=labels,
                    metadata=metadata,
                )
                
                self.metric_store[metric_name].data_points.append(data_point)
                
                # Limitar tamaño de serie temporal
                max_points = 1000
                if len(self.metric_store[metric_name].data_points) > max_points:
                    self.metric_store[metric_name].data_points = self.metric_store[metric_name].data_points[-max_points:]
                
                # Actualizar modelos de ML si es necesario
                if metric_name in self.isolation_forest_models:
                    await self._update_ml_model(metric_name)
                
                logger.debug(f"Métrica actualizada: {metric_name} = {value}")
                
        except Exception as e:
            logger.error(f"Error manejando actualización de métrica: {e}")
    
    async def _handle_system_alert(self, event_data: Dict[str, Any]):
        """Maneja alertas del sistema."""
        try:
            alert_type = event_data.get('type')
            severity = event_data.get('severity', 'medium')
            message = event_data.get('message', '')
            source = event_data.get('source', 'unknown')
            
            logger.warning(f"Alerta del sistema recibida: {alert_type} - {severity} - {message}")
            
            # Registrar alerta para análisis
            await self.store_memory(
                AgentMemoryType.EPISODIC,
                {
                    'type': 'system_alert',
                    'alert_type': alert_type,
                    'severity': severity,
                    'message': message,
                    'source': source,
                    'timestamp': datetime.now().isoformat(),
                }
            )
            
            # Si es una alerta crítica, analizar métricas inmediatamente
            if severity in ['high', 'critical']:
                await self.trigger_immediate_analysis(source)
                
        except Exception as e:
            logger.error(f"Error manejando alerta del sistema: {e}")
    
    async def _handle_analysis_request(self, event_data: Dict[str, Any]):
        """Maneja solicitudes de análisis."""
        try:
            request_type = event_data.get('request_type')
            parameters = event_data.get('parameters', {})
            request_id = event_data.get('request_id', str(uuid.uuid4()))
            
            logger.info(f"Solicitud de análisis recibida: {request_type} - {request_id}")
            
            # Procesar solicitud
            result = await self.process(AgentInput(
                agent_id=self.agent_id,
                request_id=request_id,
                parameters={**parameters, 'analysis_type': request_type},
                context=event_data.get('context', {}),
            ))
            
            # Publicar resultado si es exitoso
            if result.success and self.orchestrator:
                await self.orchestrator.event_bus.publish_async(
                    'analysis.completed',
                    {
                        'request_id': request_id,
                        'result': result.data,
                        'agent_id': self.agent_id,
                        'timestamp': datetime.now().isoformat(),
                    }
                )
                
        except Exception as e:
            logger.error(f"Error manejando solicitud de análisis: {e}")
    
    async def trigger_immediate_analysis(self, trigger_source: str):
        """Dispara un análisis inmediato en respuesta a un evento."""
        logger.info(f"AnalystAgent {self.agent_id} realizando análisis inmediato por {trigger_source}")
        
        # Análisis rápido de métricas críticas
        try:
            await self.analyze_system_metrics({
                'metrics': ['system.cpu.usage', 'system.memory.usage', 'application.error.rate'],
                'time_range': '5m',
                'aggregation': 'max',
            })
            
            # Detección rápida de anomalías
            await self.detect_anomalies({
                'metrics': ['system.cpu.usage', 'system.memory.usage'],
                'detection_methods': ['z_score', 'threshold'],
                'severity_threshold': 0.5,
            })
            
        except Exception as e:
            logger.error(f"Error en análisis inmediato: {e}")
    
    # ============================================================================
    # MÉTODOS PERIÓDICOS
    # ============================================================================
    
    async def _periodic_metrics_analysis(self):
        """Análisis periódico de métricas."""
        while self.state.get('status') == 'initialized':
            try:
                await asyncio.sleep(300)  # Cada 5 minutos
                
                logger.debug(f"AnalystAgent {self.agent_id} ejecutando análisis periódico de métricas")
                
                await self.analyze_system_metrics({
                    'time_range': '1h',
                    'aggregation': 'avg',
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en análisis periódico de métricas: {e}")
                await asyncio.sleep(60)  # Esperar antes de reintentar
    
    async def _periodic_anomaly_detection(self):
        """Detección periódica de anomalías."""
        while self.state.get('status') == 'initialized':
            try:
                await asyncio.sleep(600)  # Cada 10 minutos
                
                logger.debug(f"AnalystAgent {self.agent_id} ejecutando detección periódica de anomalías")
                
                await self.detect_anomalies({
                    'detection_methods': ['z_score', 'isolation_forest'],
                    'severity_threshold': 0.7,
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en detección periódica de anomalías: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_report_generation(self):
        """Generación periódica de reportes."""
        while self.state.get('status') == 'initialized':
            try:
                await asyncio.sleep(self.config.report_generation_interval)
                
                logger.info(f"AnalystAgent {self.agent_id} generando reporte periódico")
                
                await self.generate_analytics_report({
                    'report_type': 'summary',
                    'time_range': '24h',
                    'sections': ['executive_summary', 'metrics_analysis', 'anomalies', 'insights'],
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generando reporte periódico: {e}")
                await asyncio.sleep(300)  # Esperar 5 minutos antes de reintentar
    
    # ============================================================================
    # MÉTODOS DE APRENDIZAJE
    # ============================================================================
    
    async def _learn_from_anomaly_feedback(self, feedback: Dict[str, Any]):
        """Aprende de la retroalimentación sobre anomalías."""
        anomaly_id = feedback.get('anomaly_id')
        feedback_type = feedback.get('feedback_type')  # 'confirmed', 'false_positive', 'severity_correction'
        corrected_severity = feedback.get('corrected_severity')
        
        # Buscar anomalía en el historial
        found_anomaly = None
        for metric_anomalies in self.anomalies.values():
            for anomaly in metric_anomalies:
                if anomaly.anomaly_id == anomaly_id:
                    found_anomaly = anomaly
                    break
            if found_anomaly:
                break
        
        if not found_anomaly:
            logger.warning(f"Anomalía no encontrada para feedback: {anomaly_id}")
            return
        
        if feedback_type == 'confirmed':
            # Anomalía confirmada, aumentar confianza del método de detección
            detection_method = found_anomaly.metadata.get('detection_method')
            if detection_method:
                await self._reinforce_detection_method(detection_method, True)
                
        elif feedback_type == 'false_positive':
            # Falso positivo, ajustar umbrales o modelos
            detection_method = found_anomaly.metadata.get('detection_method')
            if detection_method:
                await self._reinforce_detection_method(detection_method, False)
                
                # Ajustar umbral si es z-score
                if detection_method == 'z_score':
                    await self._adjust_z_score_threshold(found_anomaly.metric_name, False)
                    
        elif feedback_type == 'severity_correction' and corrected_severity:
            # Corrección de severidad, aprender de la corrección
            try:
                corrected_sev_enum = AnomalySeverity(corrected_severity)
                await self._learn_severity_calibration(
                    found_anomaly.metric_name,
                    found_anomaly.severity,
                    corrected_sev_enum,
                    found_anomaly.deviation,
                )
            except ValueError:
                pass
        
        logger.info(f"Feedback de anomalía procesado: {anomaly_id} - {feedback_type}")
    
    async def _learn_from_insight_feedback(self, feedback: Dict[str, Any]):
        """Aprende de la retroalimentación sobre insights."""
        insight_id = feedback.get('insight_id')
        feedback_type = feedback.get('feedback_type')  # 'useful', 'not_useful', 'impact_correction'
        corrected_impact = feedback.get('corrected_impact')
        
        # Buscar insight en el historial
        found_insight = None
        for category_insights in self.insights.values():
            for insight in category_insights:
                if insight.insight_id == insight_id:
                    found_insight = insight
                    break
            if found_insight:
                break
        
        if not found_insight:
            logger.warning(f"Insight no encontrado para feedback: {insight_id}")
            return
        
        if feedback_type == 'useful':
            # Insight útil, reforzar patrones similares
            await self._reinforce_insight_pattern(found_insight.category, True)
            
        elif feedback_type == 'not_useful':
            # Insight no útil, evitar patrones similares
            await self._reinforce_insight_pattern(found_insight.category, False)
            
        elif feedback_type == 'impact_correction' and corrected_impact is not None:
            # Corrección de impacto, ajustar cálculo de impacto
            await self._adjust_impact_calibration(
                found_insight.category,
                found_insight.impact_score,
                corrected_impact,
            )
        
        logger.info(f"Feedback de insight procesado: {insight_id} - {feedback_type}")
    
    async def _learn_from_recommendation_feedback(self, feedback: Dict[str, Any]):
        """Aprende de la retroalimentación sobre recomendaciones."""
        recommendation_id = feedback.get('recommendation_id')
        feedback_type = feedback.get('feedback_type')  # 'implemented', 'rejected', 'impact_feedback'
        actual_impact = feedback.get('actual_impact')
        actual_effort = feedback.get('actual_effort')
        
        # Buscar recomendación en el historial
        found_recommendation = None
        for recommendation in self.recommendations:
            if recommendation.recommendation_id == recommendation_id:
                found_recommendation = recommendation
                break
        
        if not found_recommendation:
            logger.warning(f"Recomendación no encontrada para feedback: {recommendation_id}")
            return
        
        if feedback_type == 'implemented':
            # Recomendación implementada, aprender de resultados reales
            if actual_impact is not None:
                await self._learn_from_implementation_result(
                    found_recommendation.area,
                    found_recommendation.estimated_impact,
                    actual_impact,
                    found_recommendation.estimated_effort,
                    actual_effort,
                )
                
        elif feedback_type == 'rejected':
            # Recomendación rechazada, entender por qué
            rejection_reason = feedback.get('rejection_reason', 'unknown')
            await self._learn_from_rejection(
                found_recommendation.area,
                rejection_reason,
                found_recommendation.estimated_impact,
                found_recommendation.estimated_effort,
            )
        
        logger.info(f"Feedback de recomendación procesado: {recommendation_id} - {feedback_type}")
    
    async def _adjust_detection_thresholds(self, adjustments: Dict[str, Any]):
        """Ajusta los umbrales de detección basado en feedback."""
        for metric_name, adjustment in adjustments.items():
            if 'z_score_threshold' in adjustment:
                new_threshold = adjustment['z_score_threshold']
                if metric_name in self.config.alert_thresholds:
                    # Ajustar umbral para esta métrica
                    pass  # Implementar ajuste específico
        
        logger.debug(f"Umbrales de detección ajustados: {len(adjustments)} métricas")
    
    # ============================================================================
    # MÉTODOS DE INICIALIZACIÓN Y LIMPIEZA
    # ============================================================================
    
    async def _load_persistent_state(self):
        """Carga el estado persistente del agente."""
        try:
            # Intentar cargar estado desde memoria persistente
            memories = await self.retrieve_memory(
                AgentMemoryType.PERSISTENT,
                {'type': 'agent_state'},
                limit=1,
            )
            
            if memories:
                state_data = memories[0].get('data', {})
                
                # Cargar estado del agente
                if 'agent_state' in state_data:
                    self.state.update(state_data['agent_state'])
                
                # Cargar estado de análisis
                if 'analysis_state' in state_data:
                    self.analysis_state.update(state_data['analysis_state'])
                
                logger.info(f"Estado persistente cargado para AnalystAgent {self.agent_id}")
                
        except Exception as e:
            logger.warning(f"No se pudo cargar estado persistente: {e}")
    
    async def shutdown(self) -> bool:
        """
        Apaga el agente de manera controlada.
        
        Returns:
            bool: True si el apagado fue exitoso
        """
        try:
            logger.info(f"Apagando AnalystAgent {self.agent_id}")
            
            # Cancelar tareas periódicas
            for task in getattr(self, 'periodic_tasks', []):
                task.cancel()
            
            # Guardar estado final
            await self._save_state()
            
            # Limpiar suscripciones a eventos
            if self.orchestrator and hasattr(self.orchestrator, 'event_bus'):
                event_bus: EventBus = self.orchestrator.event_bus
                for event in self.event_subscriptions:
                    await event_bus.unsubscribe(event, self)
            
            # Limpiar almacenamiento en memoria
            self.metric_store.clear()
            self.anomalies.clear()
            self.insights.clear()
            self.trends.clear()
            self.recommendations.clear()
            
            # Actualizar estado
            self.state['status'] = 'shutdown'
            self.state['last_shutdown'] = datetime.now().isoformat()
            
            logger.info(f"AnalystAgent {self.agent_id} apagado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error apagando AnalystAgent {self.agent_id}: {e}")
            return False
    
    # ============================================================================
    # MÉTODOS DE DIAGNÓSTICO Y MONITOREO
    # ============================================================================
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado detallado del agente.
        
        Returns:
            Dict con estado detallado
        """
        status = {
            'agent_id': self.agent_id,
            'agent_type': self.AGENT_TYPE,
            'status': self.state.get('status', 'unknown'),
            'capabilities': self.capabilities,
            'analysis_state': self.analysis_state,
            'storage_stats': {
                'metrics_stored': len(self.metric_store),
                'total_data_points': sum(len(ts.data_points) for ts in self.metric_store.values()),
                'anomalies_stored': sum(len(anomalies) for anomalies in self.anomalies.values()),
                'insights_stored': sum(len(insights) for insights in self.insights.values()),
                'recommendations_stored': len(self.recommendations),
                'trends_analyzed': len(self.trends),
            },
            'performance_metrics': {
                'analysis_count': self.analysis_state.get('analysis_count', 0),
                'avg_processing_time': self.state.get('avg_processing_time', 0),
                'success_rate': self.state.get('success_rate', 0),
            },
            'config_summary': {
                'monitored_metrics_count': len(self.config.monitored_metrics),
                'analysis_window_hours': self.config.analysis_window_hours,
                'anomaly_detection_threshold': self.config.anomaly_detection_threshold,
                'forecast_horizon': self.config.forecast_horizon,
            },
            'event_subscriptions': list(self.event_subscriptions),
            'periodic_tasks_active': len(getattr(self, 'periodic_tasks', [])),
            'last_updated': datetime.now().isoformat(),
        }
        
        return status
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado de salud del agente.
        
        Returns:
            Dict con estado de salud
        """
        now = datetime.now()
        last_analysis = self.analysis_state.get('last_analysis_time')
        
        # Verificar última actividad
        if last_analysis:
            time_since_last_analysis = (now - last_analysis).total_seconds()
            analysis_stale = time_since_last_analysis > 3600  # 1 hora
        else:
            analysis_stale = True
        
        # Verificar almacenamiento
        metric_count = len(self.metric_store)
        has_metrics = metric_count > 0
        
        # Verificar modelos ML
        ml_models_initialized = len(self.isolation_forest_models) > 0
        
        health_status = {
            'agent_id': self.agent_id,
            'status': 'healthy',
            'checks': {
                'agent_initialized': self.state.get('status') == 'initialized',
                'metrics_available': has_metrics,
                'ml_models_ready': ml_models_initialized,
                'analysis_active': not analysis_stale,
                'memory_usage_ok': True,  # Simplificado
                'event_subscriptions_active': len(self.event_subscriptions) > 0,
            },
            'details': {
                'metric_count': metric_count,
                'ml_model_count': len(self.isolation_forest_models),
                'time_since_last_analysis': time_since_last_analysis if last_analysis else None,
                'analysis_stale': analysis_stale,
            },
            'timestamp': now.isoformat(),
        }
        
        # Determinar estado general
        failed_checks = sum(1 for check in health_status['checks'].values() if not check)
        if failed_checks == 0:
            health_status['status'] = 'healthy'
        elif failed_checks <= 2:
            health_status['status'] = 'degraded'
        else:
            health_status['status'] = 'unhealthy'
        
        return health_status


# ============================================================================
# FUNCIÓN DE FÁBRICA PARA CREACIÓN DEL AGENTE
# ============================================================================

def create_analyst_agent(agent_id: str, capabilities: Optional[List[str]] = None,
                       config: Optional[Dict[str, Any]] = None, 
                       orchestrator: Any = None) -> AnalystAgent:
    """
    Función de fábrica para crear instancias de AnalystAgent.
    
    Args:
        agent_id: Identificador único del agente
        capabilities: Lista de capacidades (opcional, usa valores por defecto si no se proporciona)
        config: Configuración adicional (opcional)
        orchestrator: Referencia al orquestador (opcional)
    
    Returns:
        AnalystAgent: Instancia del agente creada
    """
    if capabilities is None:
        capabilities = [
            'analyze_system_metrics',
            'detect_anomalies', 
            'predict_trends',
            'generate_insights',
            'recommend_optimizations',
            'validate_metric_collection',
            'generate_analytics_report',
        ]
    
    return AnalystAgent(
        agent_id=agent_id,
        capabilities=capabilities,
        config=config or {},
        orchestrator=orchestrator,
    )


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

async def example_usage():
    """Ejemplo de uso del AnalystAgent."""
    # Crear agente
    agent = create_analyst_agent(
        agent_id="analyst_1",
        capabilities=["analyze_system_metrics", "detect_anomalies"],
        config={
            "monitored_metrics": ["system.cpu.usage", "system.memory.usage"],
            "analysis_window_hours": 24,
        }
    )
    
    # Inicializar agente
    await agent.initialize()
    
    # Ejecutar análisis
    result = await agent.analyze_system_metrics({
        "time_range": "1h",
        "aggregation": "avg",
    })
    
    print(f"Análisis completado: {result.get('status')}")
    print(f"Métricas analizadas: {result.get('metrics_analyzed', 0)}")
    
    # Detectar anomalías
    anomalies = await agent.detect_anomalies({
        "detection_methods": ["z_score"],
        "severity_threshold": 0.8,
    })
    
    print(f"Anomalías detectadas: {anomalies.get('total_anomalies_detected', 0)}")
    
    # Obtener estado del agente
    status = agent.get_detailed_status()
    print(f"Estado del agente: {status.get('status')}")
    
    # Apagar agente
    await agent.shutdown()


if __name__ == "__main__":
    # Ejecutar ejemplo si se ejecuta directamente
    asyncio.run(example_usage())