"""
MetricsCollector - Sistema de recolección, agregación y análisis de métricas para Project Brain.
Proporciona monitoreo en tiempo real, alertas y optimización de la colección de métricas.
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import statistics
import json
import logging
from pathlib import Path
from pydantic import BaseModel, Field, validator

from ..core.exceptions import BrainException

class MetricType(str, Enum):
    """Tipos de métricas soportadas."""
    COUNTER = "counter"          # Solo incrementa (ej: requests, errors)
    GAUGE = "gauge"              # Sube y baja (ej: memoria, conexiones)
    HISTOGRAM = "histogram"      # Distribución de valores (ej: latencia)
    SUMMARY = "summary"          # Resumen estadístico pre-calculado
    RATE = "rate"                # Tasa por unidad de tiempo (ej: requests/seg)

class AggregationMethod(str, Enum):
    """Métodos de agregación de métricas."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"
    STDDEV = "stddev"
    RATE = "rate"

class AlertSeverity(str, Enum):
    """Niveles de severidad para alertas."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricDefinition:
    """Definición de una métrica a recolectar."""
    name: str
    type: MetricType
    description: str = ""
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    aggregation_window_seconds: int = 60  # Ventana por defecto: 1 minuto
    retention_days: int = 7  # Retención por defecto: 7 días
    sampling_rate: float = 1.0  # Tasa de muestreo (1.0 = 100%)

@dataclass
class AlertRule:
    """Regla para generar alertas basadas en métricas."""
    name: str
    metric_name: str
    condition: str  # Ej: "value > 100", "rate > 10", "p95 > 500"
    severity: AlertSeverity
    duration_seconds: int = 60  # Cuánto tiempo debe cumplirse la condición
    cooldown_seconds: int = 300  # Tiempo mínimo entre alertas similares
    message: str = ""
    actions: List[str] = field(default_factory=list)  # Acciones a ejecutar

@dataclass
class MetricSample:
    """Muestra individual de una métrica."""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregatedMetric:
    """Métrica agregada en una ventana de tiempo."""
    name: str
    type: MetricType
    values: Dict[str, float]  # Diferentes agregaciones (sum, avg, p95, etc.)
    window_start: datetime
    window_end: datetime
    sample_count: int
    labels: Dict[str, str] = field(default_factory=dict)

class MetricTrend(BaseModel):
    """Tendencia calculada para una métrica."""
    metric_name: str
    trend: str  # "increasing", "decreasing", "stable"
    slope: float
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    current_value: float
    predicted_value_1h: Optional[float] = None
    change_percentage_1h: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True

class Alert(BaseModel):
    """Alerta generada por una condición de métrica."""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold: Optional[float] = None
    condition: str
    timestamp: datetime = Field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    
    class Config:
        arbitrary_types_allowed = True

class MetricsCollector:
    """
    Sistema centralizado de recolección y análisis de métricas.
    
    Características:
    1. Recolección de métricas de múltiples fuentes
    2. Agregación en tiempo real con diferentes ventanas
    3. Cálculo de estadísticas avanzadas
    4. Detección de anomalías y tendencias
    5. Sistema de alertas configurable
    6. Exportación a múltiples formatos
    7. Optimización automática de recolección
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el recolector de métricas.
        
        Args:
            config: Configuración del recolector (opcional)
        """
        self.config = config or {}
        
        # Almacenamiento de métricas
        self._metrics_definitions: Dict[str, MetricDefinition] = {}
        self._metric_samples: Dict[str, List[MetricSample]] = defaultdict(list)
        self._aggregated_metrics: Dict[str, List[AggregatedMetric]] = defaultdict(list)
        
        # Alertas y reglas
        self._alert_rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        
        # Tendencias y anomalías
        self._metric_trends: Dict[str, MetricTrend] = {}
        self._anomaly_detectors: Dict[str, Any] = {}
        
        # Configuración
        self._retention_period = timedelta(days=self.config.get("retention_days", 7))
        self._aggregation_windows = [60, 300, 3600, 86400]  # 1m, 5m, 1h, 1d
        self._max_samples_per_metric = self.config.get("max_samples", 10000)
        
        # Estado y bloqueos
        self._lock = threading.RLock()
        self._running = False
        self._aggregation_thread: Optional[threading.Thread] = None
        self._alert_check_thread: Optional[threading.Thread] = None
        
        # Métricas internas
        self._internal_metrics = {
            "metrics_collected": 0,
            "aggregations_performed": 0,
            "alerts_triggered": 0,
            "errors_encountered": 0,
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def collect_metric(
        self,
        name: str,
        value: float,
        metric_type: Union[MetricType, str] = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Recolecta una métrica individual.
        
        Args:
            name: Nombre de la métrica
            value: Valor de la métrica
            metric_type: Tipo de métrica
            labels: Etiquetas para dimensiones adicionales
            timestamp: Timestamp (por defecto ahora)
            metadata: Metadatos adicionales
            
        Returns:
            bool: True si se recolectó exitosamente
            
        Example:
            collect_metric("api_request_duration", 0.125, MetricType.HISTOGRAM, 
                          {"endpoint": "/analyze", "method": "POST"})
        """
        try:
            with self._lock:
                # Crear definición si no existe
                if name not in self._metrics_definitions:
                    metric_type_enum = MetricType(metric_type) if isinstance(metric_type, str) else metric_type
                    self._metrics_definitions[name] = MetricDefinition(
                        name=name,
                        type=metric_type_enum,
                        description=f"Automatically created metric: {name}"
                    )
                
                # Crear muestra
                sample = MetricSample(
                    value=value,
                    timestamp=timestamp or datetime.now(),
                    labels=labels or {},
                    metadata=metadata or {}
                )
                
                # Almacenar muestra
                self._metric_samples[name].append(sample)
                
                # Limitar cantidad de muestras
                if len(self._metric_samples[name]) > self._max_samples_per_metric:
                    self._metric_samples[name] = self._metric_samples[name][-self._max_samples_per_metric:]
                
                # Actualizar métricas internas
                self._internal_metrics["metrics_collected"] += 1
                
                # Verificar alertas (en background)
                self._check_alerts_async(name, value, labels or {})
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error collecting metric {name}: {e}")
            self._internal_metrics["errors_encountered"] += 1
            return False
    
    def aggregate_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        window_seconds: int = 60,
        aggregation_methods: Optional[List[AggregationMethod]] = None,
        labels_filter: Optional[Dict[str, str]] = None,
        force: bool = False
    ) -> Dict[str, AggregatedMetric]:
        """
        Agrega métricas en una ventana de tiempo específica.
        
        Args:
            metric_names: Lista de métricas a agregar (None = todas)
            window_seconds: Ventana de agregación en segundos
            aggregation_methods: Métodos de agregación a aplicar
            labels_filter: Filtro por etiquetas
            force: Forzar re-agregación incluso si ya existe
            
        Returns:
            Dict con métricas agregadas
            
        Example:
            aggregate_metrics(["api_requests", "api_errors"], 
                             window_seconds=300,
                             aggregation_methods=[AggregationMethod.SUM, AggregationMethod.AVERAGE])
        """
        try:
            with self._lock:
                metrics_to_aggregate = metric_names or list(self._metrics_definitions.keys())
                aggregation_methods = aggregation_methods or [
                    AggregationMethod.SUM, 
                    AggregationMethod.AVERAGE,
                    AggregationMethod.P95
                ]
                
                aggregated = {}
                now = datetime.now()
                window_start = now - timedelta(seconds=window_seconds)
                
                for metric_name in metrics_to_aggregate:
                    if metric_name not in self._metric_samples:
                        continue
                    
                    # Obtener muestras dentro de la ventana
                    samples = [
                        s for s in self._metric_samples[metric_name]
                        if s.timestamp >= window_start
                    ]
                    
                    # Filtrar por etiquetas si se especifica
                    if labels_filter:
                        samples = [
                            s for s in samples
                            if all(s.labels.get(k) == v for k, v in labels_filter.items())
                        ]
                    
                    if not samples:
                        continue
                    
                    # Calcular agregaciones
                    values = {}
                    values_list = [s.value for s in samples]
                    
                    for method in aggregation_methods:
                        if method == AggregationMethod.SUM:
                            values["sum"] = sum(values_list)
                        elif method == AggregationMethod.AVERAGE:
                            values["average"] = statistics.mean(values_list)
                        elif method == AggregationMethod.MIN:
                            values["min"] = min(values_list)
                        elif method == AggregationMethod.MAX:
                            values["max"] = max(values_list)
                        elif method == AggregationMethod.COUNT:
                            values["count"] = len(values_list)
                        elif method == AggregationMethod.P50:
                            values["p50"] = statistics.quantiles(values_list, n=100)[49]
                        elif method == AggregationMethod.P95:
                            values["p95"] = statistics.quantiles(values_list, n=100)[94]
                        elif method == AggregationMethod.P99:
                            values["p99"] = statistics.quantiles(values_list, n=100)[98]
                        elif method == AggregationMethod.STDDEV:
                            values["stddev"] = statistics.stdev(values_list) if len(values_list) > 1 else 0
                        elif method == AggregationMethod.RATE:
                            # Calcular tasa por segundo
                            if len(samples) >= 2:
                                time_span = (samples[-1].timestamp - samples[0].timestamp).total_seconds()
                                if time_span > 0:
                                    values["rate"] = len(samples) / time_span
                    
                    # Crear métrica agregada
                    metric_def = self._metrics_definitions.get(metric_name)
                    aggregated[metric_name] = AggregatedMetric(
                        name=metric_name,
                        type=metric_def.type if metric_def else MetricType.GAUGE,
                        values=values,
                        window_start=window_start,
                        window_end=now,
                        sample_count=len(samples),
                        labels=labels_filter or {}
                    )
                    
                    # Almacenar para referencia futura
                    if force or metric_name not in self._aggregated_metrics:
                        self._aggregated_metrics[metric_name].append(aggregated[metric_name])
                        
                        # Limitar cantidad de agregaciones almacenadas
                        if len(self._aggregated_metrics[metric_name]) > 100:
                            self._aggregated_metrics[metric_name].pop(0)
                
                self._internal_metrics["aggregations_performed"] += 1
                return aggregated
                
        except Exception as e:
            self.logger.error(f"Error aggregating metrics: {e}")
            self._internal_metrics["errors_encountered"] += 1
            return {}
    
    def calculate_statistics(
        self,
        metric_name: str,
        window_seconds: int = 3600,
        include_labels: bool = False,
        percentiles: List[float] = None
    ) -> Dict[str, Any]:
        """
        Calcula estadísticas detalladas para una métrica.
        
        Args:
            metric_name: Nombre de la métrica
            window_seconds: Ventana de tiempo para cálculo
            include_labels: Incluir estadísticas por etiqueta
            percentiles: Percentiles a calcular
            
        Returns:
            Dict con estadísticas calculadas
            
        Example:
            calculate_statistics("api_response_time", 
                               window_seconds=300,
                               percentiles=[50, 90, 95, 99])
        """
        try:
            with self._lock:
                if metric_name not in self._metric_samples:
                    return {}
                
                percentiles = percentiles or [50, 95, 99]
                now = datetime.now()
                window_start = now - timedelta(seconds=window_seconds)
                
                # Obtener muestras dentro de la ventana
                samples = [
                    s for s in self._metric_samples[metric_name]
                    if s.timestamp >= window_start
                ]
                
                if not samples:
                    return {"error": "No samples in window"}
                
                values = [s.value for s in samples]
                timestamps = [s.timestamp for s in samples]
                
                # Estadísticas básicas
                stats = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "time_range": {
                        "start": min(timestamps).isoformat(),
                        "end": max(timestamps).isoformat(),
                        "seconds": (max(timestamps) - min(timestamps)).total_seconds()
                    }
                }
                
                # Percentiles
                if len(values) >= 2:
                    try:
                        for p in percentiles:
                            if 0 < p < 100:
                                stats[f"p{p}"] = statistics.quantiles(values, n=100)[int(p) - 1]
                    except (ValueError, IndexError):
                        pass
                
                # Tasa (si aplica)
                if len(samples) >= 2:
                    time_span = (samples[-1].timestamp - samples[0].timestamp).total_seconds()
                    if time_span > 0:
                        stats["rate_per_second"] = len(samples) / time_span
                
                # Estadísticas por etiqueta si se solicita
                if include_labels:
                    label_stats = {}
                    label_groups = defaultdict(list)
                    
                    for sample in samples:
                        # Crear clave única para combinación de etiquetas
                        label_key = json.dumps(sample.labels, sort_keys=True)
                        label_groups[label_key].append(sample.value)
                    
                    for label_key, label_values in label_groups.items():
                        if len(label_values) >= 2:
                            label_stats[label_key] = {
                                "count": len(label_values),
                                "mean": statistics.mean(label_values),
                                "min": min(label_values),
                                "max": max(label_values)
                            }
                    
                    if label_stats:
                        stats["by_labels"] = label_stats
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error calculating statistics for {metric_name}: {e}")
            return {"error": str(e)}
    
    def export_metrics(
        self,
        format: str = "json",
        metric_names: Optional[List[str]] = None,
        window_seconds: int = 300,
        include_samples: bool = False,
        compression: bool = False
    ) -> Union[str, Dict, bytes]:
        """
        Exporta métricas en diferentes formatos.
        
        Args:
            format: Formato de exportación (json, prometheus, csv, influxdb)
            metric_names: Métricas a exportar (None = todas)
            window_seconds: Ventana de tiempo para exportar
            include_samples: Incluir muestras individuales
            compression: Comprimir salida si es posible
            
        Returns:
            Métricas en el formato solicitado
            
        Raises:
            ValueError: Si el formato no es soportado
        """
        try:
            with self._lock:
                now = datetime.now()
                window_start = now - timedelta(seconds=window_seconds)
                
                metrics_to_export = metric_names or list(self._metrics_definitions.keys())
                
                if format == "json":
                    return self._export_json(metrics_to_export, window_start, include_samples)
                elif format == "prometheus":
                    return self._export_prometheus(metrics_to_export, window_start)
                elif format == "csv":
                    return self._export_csv(metrics_to_export, window_start)
                elif format == "influxdb":
                    return self._export_influxdb(metrics_to_export, window_start)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                    
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            raise BrainException(f"Failed to export metrics: {e}")
    
    def _export_json(
        self,
        metric_names: List[str],
        window_start: datetime,
        include_samples: bool
    ) -> Dict[str, Any]:
        """Exporta métricas en formato JSON."""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "window_start": window_start.isoformat(),
            "metrics": {}
        }
        
        for metric_name in metric_names:
            if metric_name not in self._metric_samples:
                continue
            
            # Obtener muestras en ventana
            samples = [
                s for s in self._metric_samples[metric_name]
                if s.timestamp >= window_start
            ]
            
            if not samples:
                continue
            
            metric_def = self._metrics_definitions.get(metric_name)
            metric_data = {
                "name": metric_name,
                "type": metric_def.type.value if metric_def else "unknown",
                "description": metric_def.description if metric_def else "",
                "unit": metric_def.unit if metric_def else "",
                "sample_count": len(samples),
                "aggregated": self._aggregate_single_metric(samples)
            }
            
            if include_samples:
                metric_data["samples"] = [
                    {
                        "value": s.value,
                        "timestamp": s.timestamp.isoformat(),
                        "labels": s.labels,
                        "metadata": s.metadata
                    }
                    for s in samples[-100:]  # Limitar a 100 muestras
                ]
            
            export_data["metrics"][metric_name] = metric_data
        
        return export_data
    
    def _export_prometheus(self, metric_names: List[str], window_start: datetime) -> str:
        """Exporta métricas en formato Prometheus."""
        lines = []
        
        for metric_name in metric_names:
            if metric_name not in self._metric_samples:
                continue
            
            # Obtener muestras en ventana
            samples = [
                s for s in self._metric_samples[metric_name]
                if s.timestamp >= window_start
            ]
            
            if not samples:
                continue
            
            # Agrupar por etiquetas
            label_groups = defaultdict(list)
            for sample in samples:
                label_key = json.dumps(sample.labels, sort_keys=True)
                label_groups[label_key].append(sample.value)
            
            # Crear líneas Prometheus
            metric_def = self._metrics_definitions.get(metric_name)
            metric_type = metric_def.type if metric_def else MetricType.GAUGE
            
            for label_key, values in label_groups.items():
                labels = json.loads(label_key) if label_key != "{}" else {}
                
                # Formatear etiquetas para Prometheus
                label_str = ""
                if labels:
                    label_parts = [f'{k}="{v}"' for k, v in labels.items()]
                    label_str = "{" + ",".join(label_parts) + "}"
                
                # Añadir línea HELP
                if metric_def and metric_def.description:
                    lines.append(f'# HELP {metric_name} {metric_def.description}')
                
                # Añadir línea TYPE
                type_map = {
                    MetricType.COUNTER: "counter",
                    MetricType.GAUGE: "gauge",
                    MetricType.HISTOGRAM: "histogram",
                    MetricType.SUMMARY: "summary"
                }
                lines.append(f'# TYPE {metric_name} {type_map.get(metric_type, "gauge")}')
                
                # Añadir valor actual (último valor)
                if values:
                    lines.append(f'{metric_name}{label_str} {values[-1]}')
        
        return "\n".join(lines)
    
    def _export_csv(self, metric_names: List[str], window_start: datetime) -> str:
        """Exporta métricas en formato CSV."""
        lines = ["timestamp,metric_name,value,labels"]
        
        for metric_name in metric_names:
            if metric_name not in self._metric_samples:
                continue
            
            samples = [
                s for s in self._metric_samples[metric_name]
                if s.timestamp >= window_start
            ]
            
            for sample in samples:
                label_str = json.dumps(sample.labels) if sample.labels else ""
                lines.append(
                    f'{sample.timestamp.isoformat()},'
                    f'{metric_name},'
                    f'{sample.value},'
                    f'"{label_str}"'
                )
        
        return "\n".join(lines)
    
    def _export_influxdb(self, metric_names: List[str], window_start: datetime) -> str:
        """Exporta métricas en formato InfluxDB Line Protocol."""
        lines = []
        
        for metric_name in metric_names:
            if metric_name not in self._metric_samples:
                continue
            
            samples = [
                s for s in self._metric_samples[metric_name]
                if s.timestamp >= window_start
            ]
            
            for sample in samples:
                # Escapar caracteres especiales
                escaped_name = metric_name.replace(" ", "\\ ").replace(",", "\\,")
                
                # Formatear etiquetas
                tag_parts = []
                for k, v in sample.labels.items():
                    escaped_k = str(k).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
                    escaped_v = str(v).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
                    tag_parts.append(f'{escaped_k}={escaped_v}')
                
                tags = ",".join(tag_parts)
                if tags:
                    measurement = f'{escaped_name},{tags}'
                else:
                    measurement = escaped_name
                
                # Timestamp en nanosegundos
                timestamp_ns = int(sample.timestamp.timestamp() * 1e9)
                
                lines.append(f'{measurement} value={sample.value} {timestamp_ns}')
        
        return "\n".join(lines)
    
    def _aggregate_single_metric(self, samples: List[MetricSample]) -> Dict[str, float]:
        """Agrega muestras individuales en estadísticas."""
        values = [s.value for s in samples]
        
        if not values:
            return {}
        
        stats = {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values)
        }
        
        if len(values) >= 2:
            stats["std_dev"] = statistics.stdev(values)
            try:
                stats["p95"] = statistics.quantiles(values, n=100)[94]
                stats["p99"] = statistics.quantiles(values, n=100)[98]
            except (ValueError, IndexError):
                pass
        
        return stats
    
    def monitor_metric_trends(
        self,
        metric_name: str,
        window_hours: int = 24,
        method: str = "linear"
    ) -> Optional[MetricTrend]:
        """
        Monitorea y calcula tendencias para una métrica.
        
        Args:
            metric_name: Nombre de la métrica a analizar
            window_hours: Ventana de análisis en horas
            method: Método de análisis (linear, exponential, seasonal)
            
        Returns:
            MetricTrend con análisis de tendencia o None si no hay datos
        """
        try:
            with self._lock:
                if metric_name not in self._metric_samples:
                    return None
                
                now = datetime.now()
                window_start = now - timedelta(hours=window_hours)
                
                # Obtener muestras en ventana
                samples = [
                    s for s in self._metric_samples[metric_name]
                    if s.timestamp >= window_start
                ]
                
                if len(samples) < 10:  # Mínimo de muestras para análisis
                    return None
                
                # Preparar datos para análisis
                timestamps = [s.timestamp.timestamp() for s in samples]
                values = [s.value for s in samples]
                
                # Análisis de tendencia lineal simple
                if method == "linear":
                    return self._analyze_linear_trend(metric_name, timestamps, values, samples[-1].value)
                
                # Análisis de tendencia exponencial
                elif method == "exponential":
                    return self._analyze_exponential_trend(metric_name, timestamps, values, samples[-1].value)
                
                else:
                    self.logger.warning(f"Unsupported trend analysis method: {method}")
                    return self._analyze_linear_trend(metric_name, timestamps, values, samples[-1].value)
                    
        except Exception as e:
            self.logger.error(f"Error monitoring trends for {metric_name}: {e}")
            return None
    
    def _analyze_linear_trend(
        self,
        metric_name: str,
        timestamps: List[float],
        values: List[float],
        current_value: float
    ) -> MetricTrend:
        """Analiza tendencia lineal usando regresión simple."""
        try:
            # Regresión lineal simple
            n = len(timestamps)
            sum_x = sum(timestamps)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(timestamps, values))
            sum_x2 = sum(x * x for x in timestamps)
            
            # Pendiente
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                slope = 0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # Calcular confianza (R² simplificado)
            if len(values) >= 3:
                y_mean = sum_y / n
                ss_total = sum((y - y_mean) ** 2 for y in values)
                ss_residual = sum((y - (slope * x + (sum_y/n - slope * sum_x/n))) ** 2 
                                for x, y in zip(timestamps, values))
                
                if ss_total > 0:
                    confidence = 1 - (ss_residual / ss_total)
                    confidence = max(0.0, min(1.0, confidence))
                else:
                    confidence = 0.0
            else:
                confidence = 0.0
            
            # Determinar dirección de tendencia
            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Predecir valor en 1 hora
            if slope != 0:
                last_timestamp = timestamps[-1]
                predicted_value_1h = current_value + slope * 3600
                change_percentage = ((predicted_value_1h - current_value) / abs(current_value)) * 100 if current_value != 0 else 0
            else:
                predicted_value_1h = current_value
                change_percentage = 0
            
            trend_data = MetricTrend(
                metric_name=metric_name,
                trend=trend,
                slope=slope,
                confidence=confidence,
                current_value=current_value,
                predicted_value_1h=predicted_value_1h if abs(slope) > 0.0001 else None,
                change_percentage_1h=change_percentage if abs(change_percentage) > 0.1 else None
            )
            
            self._metric_trends[metric_name] = trend_data
            return trend_data
            
        except Exception as e:
            self.logger.error(f"Error in linear trend analysis: {e}")
            return MetricTrend(
                metric_name=metric_name,
                trend="unknown",
                slope=0,
                confidence=0,
                current_value=current_value
            )
    
    def _analyze_exponential_trend(
        self,
        metric_name: str,
        timestamps: List[float],
        values: List[float],
        current_value: float
    ) -> MetricTrend:
        """Analiza tendencia exponencial (para métricas que crecen/decae exponencialmente)."""
        # Para simplificar, usamos logaritmo y regresión lineal
        try:
            # Filtrar valores no positivos para logaritmo
            positive_values = [(t, v) for t, v in zip(timestamps, values) if v > 0]
            
            if len(positive_values) < 5:
                return self._analyze_linear_trend(metric_name, timestamps, values, current_value)
            
            pos_timestamps, pos_values = zip(*positive_values)
            log_values = [math.log(v) for v in pos_values]
            
            # Regresión lineal en valores logarítmicos
            n = len(pos_timestamps)
            sum_x = sum(pos_timestamps)
            sum_y = sum(log_values)
            sum_xy = sum(x * y for x, y in zip(pos_timestamps, log_values))
            sum_x2 = sum(x * x for x in pos_timestamps)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                slope = 0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # La pendiente en escala logarítmica representa tasa de crecimiento exponencial
            growth_rate = slope
            
            if growth_rate > 0.001:
                trend = "increasing"
            elif growth_rate < -0.001:
                trend = "decreasing"
            else:
                trend = "stable"
            
            return MetricTrend(
                metric_name=metric_name,
                trend=trend,
                slope=growth_rate,
                confidence=0.7,  # Confianza aproximada
                current_value=current_value
            )
            
        except Exception as e:
            self.logger.error(f"Error in exponential trend analysis: {e}")
            return self._analyze_linear_trend(metric_name, timestamps, values, current_value)
    
    def alert_on_anomalies(
        self,
        metric_name: str,
        strategy: str = "threshold",
        threshold: Optional[float] = None,
        window_seconds: int = 300,
        sensitivity: float = 2.0
    ) -> List[Alert]:
        """
        Detecta anomalías en una métrica y genera alertas.
        
        Args:
            metric_name: Nombre de la métrica
            strategy: Estrategia de detección (threshold, stddev, iqr, rate_change)
            threshold: Umbral para estrategia de threshold
            window_seconds: Ventana para análisis
            sensitivity: Sensibilidad de detección (1.0 = normal)
            
        Returns:
            Lista de alertas generadas
        """
        alerts = []
        
        try:
            with self._lock:
                if metric_name not in self._metric_samples:
                    return alerts
                
                now = datetime.now()
                window_start = now - timedelta(seconds=window_seconds)
                
                # Obtener muestras en ventana
                samples = [
                    s for s in self._metric_samples[metric_name]
                    if s.timestamp >= window_start
                ]
                
                if len(samples) < 5:  # Mínimo para análisis
                    return alerts
                
                values = [s.value for s in samples]
                current_value = values[-1]
                
                # Aplicar estrategia de detección
                if strategy == "threshold" and threshold is not None:
                    if current_value > threshold:
                        alerts.append(self._create_threshold_alert(
                            metric_name, current_value, threshold, "above"
                        ))
                    elif current_value < -threshold:
                        alerts.append(self._create_threshold_alert(
                            metric_name, current_value, -threshold, "below"
                        ))
                
                elif strategy == "stddev":
                    alerts.extend(self._detect_stddev_anomalies(
                        metric_name, values, samples, sensitivity
                    ))
                
                elif strategy == "iqr":
                    alerts.extend(self._detect_iqr_anomalies(
                        metric_name, values, samples, sensitivity
                    ))
                
                elif strategy == "rate_change":
                    alerts.extend(self._detect_rate_change_anomalies(
                        metric_name, samples, sensitivity
                    ))
                
                # Procesar alertas generadas
                for alert in alerts:
                    self._process_alert(alert)
                
                return alerts
                
        except Exception as e:
            self.logger.error(f"Error detecting anomalies for {metric_name}: {e}")
            return []
    
    def _create_threshold_alert(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        direction: str
    ) -> Alert:
        """Crea alerta por umbral."""
        alert_id = f"threshold_{metric_name}_{int(time.time())}"
        
        return Alert(
            id=alert_id,
            rule_name=f"{metric_name}_threshold",
            severity=AlertSeverity.WARNING if direction == "above" else AlertSeverity.INFO,
            message=f"Metric {metric_name} {direction} threshold: {value:.2f} {'>' if direction == 'above' else '<'} {threshold:.2f}",
            metric_name=metric_name,
            metric_value=value,
            threshold=threshold,
            condition=f"value {'>' if direction == 'above' else '<'} {threshold}"
        )
    
    def _detect_stddev_anomalies(
        self,
        metric_name: str,
        values: List[float],
        samples: List[MetricSample],
        sensitivity: float
    ) -> List[Alert]:
        """Detecta anomalías usando desviación estándar."""
        alerts = []
        
        if len(values) < 10:
            return alerts
        
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        if std_dev == 0:
            return alerts
        
        current_value = values[-1]
        z_score = abs(current_value - mean) / std_dev
        
        if z_score > sensitivity * 2:  # 2 sigma
            severity = AlertSeverity.CRITICAL if z_score > sensitivity * 3 else AlertSeverity.WARNING
            
            alerts.append(Alert(
                id=f"stddev_{metric_name}_{int(time.time())}",
                rule_name=f"{metric_name}_stddev",
                severity=severity,
                message=f"Metric {metric_name} anomaly detected: z-score {z_score:.2f}, value {current_value:.2f}",
                metric_name=metric_name,
                metric_value=current_value,
                condition=f"z-score > {sensitivity * 2}"
            ))
        
        return alerts
    
    def _detect_iqr_anomalies(
        self,
        metric_name: str,
        values: List[float],
        samples: List[MetricSample],
        sensitivity: float
    ) -> List[Alert]:
        """Detecta anomalías usando rango intercuartil (IQR)."""
        alerts = []
        
        if len(values) < 10:
            return alerts
        
        try:
            q1 = statistics.quantiles(values, n=4)[0]  # 25%
            q3 = statistics.quantiles(values, n=4)[2]  # 75%
            iqr = q3 - q1
            
            if iqr == 0:
                return alerts
            
            lower_bound = q1 - sensitivity * 1.5 * iqr
            upper_bound = q3 + sensitivity * 1.5 * iqr
            
            current_value = values[-1]
            
            if current_value < lower_bound or current_value > upper_bound:
                direction = "below" if current_value < lower_bound else "above"
                bound = lower_bound if direction == "below" else upper_bound
                
                alerts.append(Alert(
                    id=f"iqr_{metric_name}_{int(time.time())}",
                    rule_name=f"{metric_name}_iqr",
                    severity=AlertSeverity.WARNING,
                    message=f"Metric {metric_name} {direction} IQR bounds: {current_value:.2f} {'<' if direction == 'below' else '>'} {bound:.2f}",
                    metric_name=metric_name,
                    metric_value=current_value,
                    threshold=bound,
                    condition=f"value {direction} IQR bound"
                ))
        
        except (ValueError, IndexError):
            pass
        
        return alerts
    
    def _detect_rate_change_anomalies(
        self,
        metric_name: str,
        samples: List[MetricSample],
        sensitivity: float
    ) -> List[Alert]:
        """Detecta anomalías en la tasa de cambio."""
        alerts = []
        
        if len(samples) < 20:
            return alerts
        
        # Calcular tasas de cambio entre puntos consecutivos
        rates = []
        for i in range(1, len(samples)):
            time_diff = (samples[i].timestamp - samples[i-1].timestamp).total_seconds()
            if time_diff > 0:
                rate = (samples[i].value - samples[i-1].value) / time_diff
                rates.append(rate)
        
        if len(rates) < 10:
            return alerts
        
        # Analizar distribución de tasas
        recent_rates = rates[-5:]  # Últimas 5 tasas
        historical_rates = rates[:-5]  # Tasas históricas
        
        if not historical_rates:
            return alerts
        
        hist_mean = statistics.mean(historical_rates)
        hist_std = statistics.stdev(historical_rates) if len(historical_rates) > 1 else 0
        
        if hist_std == 0:
            return alerts
        
        # Verificar si las tasas recientes son anómalas
        for i, rate in enumerate(recent_rates):
            z_score = abs(rate - hist_mean) / hist_std
            
            if z_score > sensitivity * 3:
                alerts.append(Alert(
                    id=f"rate_{metric_name}_{int(time.time())}_{i}",
                    rule_name=f"{metric_name}_rate_change",
                    severity=AlertSeverity.WARNING,
                    message=f"Rate change anomaly for {metric_name}: z-score {z_score:.2f}, rate {rate:.4f}/s",
                    metric_name=metric_name,
                    metric_value=rate,
                    condition=f"rate z-score > {sensitivity * 3}"
                ))
        
        return alerts
    
    def _process_alert(self, alert: Alert) -> None:
        """Procesa una alerta generada."""
        # Verificar si ya hay una alerta similar activa
        alert_key = f"{alert.rule_name}_{alert.metric_name}"
        
        if alert_key in self._active_alerts:
            # Actualizar alerta existente
            existing = self._active_alerts[alert_key]
            existing.timestamp = alert.timestamp
            existing.metric_value = alert.metric_value
        else:
            # Añadir nueva alerta
            self._active_alerts[alert_key] = alert
            self._alert_history.append(alert)
            
            # Limitar historial
            if len(self._alert_history) > 1000:
                self._alert_history = self._alert_history[-1000:]
            
            # Ejecutar acciones (si las hay)
            self._execute_alert_actions(alert)
            
            # Actualizar métricas internas
            self._internal_metrics["alerts_triggered"] += 1
            
            # Loggear alerta
            self.logger.warning(
                f"Alert triggered: {alert.severity.value} - {alert.message}"
            )
    
    def _execute_alert_actions(self, alert: Alert) -> None:
        """Ejecuta acciones asociadas a una alerta."""
        # En una implementación real, esto podría enviar notificaciones,
        # crear tickets, ejecutar scripts, etc.
        
        # Por ahora solo loggear
        self.logger.info(f"Executing actions for alert: {alert.id}")
        
        # Ejemplo: Notificar por log según severidad
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.ERROR:
            self.logger.error(f"ERROR ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            self.logger.warning(f"WARNING ALERT: {alert.message}")
    
    def optimize_metric_collection(
        self,
        metric_name: Optional[str] = None,
        strategy: str = "sampling",
        target_samples: int = 1000,
        compression_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        Optimiza la recolección de métricas para reducir uso de recursos.
        
        Args:
            metric_name: Nombre de la métrica a optimizar (None = todas)
            strategy: Estrategia de optimización (sampling, aggregation, compression)
            target_samples: Número objetivo de muestras a mantener
            compression_ratio: Ratio de compresión (0.1 = mantener 10%)
            
        Returns:
            Dict con resultados de la optimización
        """
        optimization_results = {
            "metrics_optimized": 0,
            "samples_removed": 0,
            "memory_saved_bytes": 0,
            "processing_time_saved": 0
        }
        
        try:
            with self._lock:
                metrics_to_optimize = [metric_name] if metric_name else list(self._metric_samples.keys())
                
                for name in metrics_to_optimize:
                    if name not in self._metric_samples:
                        continue
                    
                    samples = self._metric_samples[name]
                    initial_count = len(samples)
                    
                    if initial_count <= target_samples:
                        continue  # No necesita optimización
                    
                    if strategy == "sampling":
                        # Muestreo aleatorio manteniendo distribución temporal
                        self._apply_sampling_optimization(name, samples, target_samples)
                    
                    elif strategy == "aggregation":
                        # Reemplazar muestras antiguas con valores agregados
                        self._apply_aggregation_optimization(name, samples, compression_ratio)
                    
                    elif strategy == "compression":
                        # Comprimir datos antiguos manteniendo resúmenes estadísticos
                        self._apply_compression_optimization(name, samples, compression_ratio)
                    
                    final_count = len(self._metric_samples[name])
                    optimization_results["samples_removed"] += (initial_count - final_count)
                    optimization_results["metrics_optimized"] += 1
                    
                    # Estimación de memoria ahorrada (aproximada)
                    optimization_results["memory_saved_bytes"] += (initial_count - final_count) * 100
                
                return optimization_results
                
        except Exception as e:
            self.logger.error(f"Error optimizing metric collection: {e}")
            return optimization_results
    
    def _apply_sampling_optimization(
        self,
        metric_name: str,
        samples: List[MetricSample],
        target_samples: int
    ) -> None:
        """Aplica optimización por muestreo."""
        if len(samples) <= target_samples:
            return
        
        # Mantener muestras más recientes y muestrear aleatoriamente las antiguas
        recent_count = min(target_samples // 2, len(samples))
        old_count = target_samples - recent_count
        
        # Separar muestras recientes y antiguas
        recent_samples = samples[-recent_count:]
        old_samples = samples[:-recent_count]
        
        # Muestrear aleatoriamente las antiguas
        if len(old_samples) > old_count:
            import random
            old_samples = random.sample(old_samples, old_count)
        
        # Combinar y actualizar
        self._metric_samples[metric_name] = old_samples + recent_samples
    
    def _apply_aggregation_optimization(
        self,
        metric_name: str,
        samples: List[MetricSample],
        compression_ratio: float
    ) -> None:
        """Aplica optimización por agregación."""
        if len(samples) < 100:  # Solo optimizar si hay suficientes muestras
            return
        
        # Agrupar muestras antiguas en ventanas y reemplazar con valores agregados
        old_samples = samples[:-100]  # Mantener 100 muestras recientes
        recent_samples = samples[-100:]
        
        if len(old_samples) < 10:
            return
        
        # Agrupar por intervalos de tiempo
        time_groups = defaultdict(list)
        for sample in old_samples:
            # Agrupar por hora
            hour_key = sample.timestamp.replace(minute=0, second=0, microsecond=0)
            time_groups[hour_key].append(sample)
        
        # Crear muestras agregadas por grupo
        aggregated_samples = []
        for hour, group_samples in time_groups.items():
            if group_samples:
                values = [s.value for s in group_samples]
                
                # Crear muestra agregada representativa
                aggregated_sample = MetricSample(
                    value=statistics.mean(values),
                    timestamp=hour,
                    labels=group_samples[0].labels if group_samples else {},
                    metadata={
                        "aggregated": True,
                        "original_count": len(group_samples),
                        "min": min(values),
                        "max": max(values),
                        "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                    }
                )
                aggregated_samples.append(aggregated_sample)
        
        # Combinar muestras agregadas con muestras recientes
        self._metric_samples[metric_name] = aggregated_samples + recent_samples
    
    def _apply_compression_optimization(
        self,
        metric_name: str,
        samples: List[MetricSample],
        compression_ratio: float
    ) -> None:
        """Aplica optimización por compresión (manteniendo resúmenes estadísticos)."""
        if len(samples) < 50:
            return
        
        # Comprimir datos antiguos manteniendo solo resúmenes estadísticos
        keep_count = max(50, int(len(samples) * compression_ratio))
        recent_samples = samples[-keep_count:]
        old_samples = samples[:-keep_count]
        
        if not old_samples:
            return
        
        # Calcular resúmenes estadísticos para muestras antiguas
        old_values = [s.value for s in old_samples]
        
        # Crear muestra de resumen
        summary_sample = MetricSample(
            value=statistics.mean(old_values) if old_values else 0,
            timestamp=old_samples[0].timestamp,
            labels=old_samples[0].labels if old_samples else {},
            metadata={
                "compressed_summary": True,
                "original_count": len(old_samples),
                "time_range": {
                    "start": old_samples[0].timestamp.isoformat(),
                    "end": old_samples[-1].timestamp.isoformat()
                },
                "statistics": {
                    "mean": statistics.mean(old_values) if old_values else 0,
                    "min": min(old_values) if old_values else 0,
                    "max": max(old_values) if old_values else 0,
                    "std_dev": statistics.stdev(old_values) if len(old_values) > 1 else 0
                }
            }
        )
        
        # Reemplazar muestras antiguas con resumen
        self._metric_samples[metric_name] = [summary_sample] + recent_samples
    
    def _check_alerts_async(self, metric_name: str, value: float, labels: Dict[str, str]) -> None:
        """Verifica reglas de alerta para una métrica (ejecución asíncrona)."""
        # En una implementación real, esto ejecutaría en un thread pool
        # Por simplicidad, verificamos solo las reglas de umbral simple
        
        # Verificar reglas específicas para esta métrica
        for rule_name, rule in self._alert_rules.items():
            if rule.metric_name == metric_name:
                # Verificar condición (implementación simple)
                try:
                    # Evaluar condición simple (ej: "value > 100")
                    if ">" in rule.condition:
                        parts = rule.condition.split(">")
                        if len(parts) == 2:
                            threshold = float(parts[1].strip())
                            if value > threshold:
                                alert = Alert(
                                    id=f"{rule_name}_{int(time.time())}",
                                    rule_name=rule_name,
                                    severity=rule.severity,
                                    message=rule.message or f"{metric_name} exceeded threshold: {value} > {threshold}",
                                    metric_name=metric_name,
                                    metric_value=value,
                                    threshold=threshold,
                                    condition=rule.condition
                                )
                                self._process_alert(alert)
                    
                    elif "<" in rule.condition:
                        parts = rule.condition.split("<")
                        if len(parts) == 2:
                            threshold = float(parts[1].strip())
                            if value < threshold:
                                alert = Alert(
                                    id=f"{rule_name}_{int(time.time())}",
                                    rule_name=rule_name,
                                    severity=rule.severity,
                                    message=rule.message or f"{metric_name} below threshold: {value} < {threshold}",
                                    metric_name=metric_name,
                                    metric_value=value,
                                    threshold=threshold,
                                    condition=rule.condition
                                )
                                self._process_alert(alert)
                
                except (ValueError, IndexError):
                    pass
    
    def get_metrics_status(self) -> Dict[str, Any]:
        """Obtiene estado y estadísticas del recolector de métricas."""
        with self._lock:
            return {
                "metrics_defined": len(self._metrics_definitions),
                "metrics_with_samples": len(self._metric_samples),
                "total_samples": sum(len(samples) for samples in self._metric_samples.values()),
                "active_alerts": len(self._active_alerts),
                "alert_history_count": len(self._alert_history),
                "internal_metrics": self._internal_metrics,
                "aggregation_windows": self._aggregation_windows,
                "retention_period_days": self._retention_period.days
            }

# Funciones de utilidad para el módulo

def setup_metrics_collector(config: Dict[str, Any]) -> MetricsCollector:
    """
    Configura e inicializa un recolector de métricas.
    
    Args:
        config: Configuración del recolector
        
    Returns:
        MetricsCollector inicializado
    """
    collector = MetricsCollector(config)
    
    # Definir métricas del sistema por defecto
    default_metrics = [
        MetricDefinition(
            name="system_cpu_usage",
            type=MetricType.GAUGE,
            description="CPU usage percentage",
            unit="percent"
        ),
        MetricDefinition(
            name="system_memory_usage",
            type=MetricType.GAUGE,
            description="Memory usage percentage",
            unit="percent"
        ),
        MetricDefinition(
            name="api_request_count",
            type=MetricType.COUNTER,
            description="Total API requests processed",
            unit="requests"
        ),
        MetricDefinition(
            name="api_response_time",
            type=MetricType.HISTOGRAM,
            description="API response time in milliseconds",
            unit="milliseconds"
        ),
        MetricDefinition(
            name="errors_total",
            type=MetricType.COUNTER,
            description="Total errors encountered",
            unit="errors"
        ),
    ]
    
    for metric_def in default_metrics:
        collector._metrics_definitions[metric_def.name] = metric_def
    
    return collector

# Ejemplo de uso
if __name__ == "__main__":
    import random
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear recolector
    collector = setup_metrics_collector({
        "retention_days": 7,
        "max_samples": 5000
    })
    
    # Simular recolección de métricas
    for i in range(100):
        collector.collect_metric("api_response_time", random.uniform(50, 200))
        collector.collect_metric("api_request_count", 1)
        collector.collect_metric("system_cpu_usage", random.uniform(10, 80))
        time.sleep(0.1)
    
    # Agregar métricas
    aggregated = collector.aggregate_metrics(
        ["api_response_time", "api_request_count"],
        window_seconds=30
    )
    
    print("Aggregated metrics:", json.dumps({
        k: {ak: round(av, 2) for ak, av in v.values.items()}
        for k, v in aggregated.items()
    }, indent=2))
    
    # Calcular estadísticas
    stats = collector.calculate_statistics(
        "api_response_time",
        window_seconds=60,
        percentiles=[50, 95, 99]
    )
    
    print("\nStatistics:", json.dumps({
        k: round(v, 2) if isinstance(v, float) else v
        for k, v in stats.items()
    }, indent=2))
    
    # Exportar métricas
    json_export = collector.export_metrics(format="json", window_seconds=60)
    print("\nJSON Export (first 500 chars):", json.dumps(json_export)[:500] + "...")
    
    # Monitorear tendencias
    trend = collector.monitor_metric_trends("api_response_time", window_hours=1)
    if trend:
        print(f"\nTrend: {trend.trend} (slope: {trend.slope:.4f}, confidence: {trend.confidence:.2f})")
    
    # Detectar anomalías
    alerts = collector.alert_on_anomalies(
        "api_response_time",
        strategy="stddev",
        sensitivity=2.0
    )
    
    if alerts:
        print(f"\nAlerts generated: {len(alerts)}")
        for alert in alerts:
            print(f"  - {alert.severity.value}: {alert.message}")
    
    # Optimizar recolección
    optimization = collector.optimize_metric_collection(
        strategy="sampling",
        target_samples=500
    )
    
    print(f"\nOptimization results: {optimization}")
    
    # Estado del recolector
    status = collector.get_metrics_status()
    print(f"\nCollector status: {status['total_samples']} samples, {status['active_alerts']} active alerts")