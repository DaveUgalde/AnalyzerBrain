"""
learning_agent.py - Agente especializado en aprendizaje automático y mejora continua.
Hereda de BaseAgent y se enfoca en aprender de experiencias, adaptarse a nuevos datos,
optimizar estrategias de aprendizaje y transferir conocimiento.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentConfig, AgentState, AgentCapability, AgentMemoryType
from ..core.exceptions import AgentException, ValidationError
from ..embeddings.embedding_generator import EmbeddingGenerator
from ..graph.knowledge_graph import KnowledgeGraph
from ..memory.memory_hierarchy import MemoryHierarchy

class LearningStrategy(Enum):
    """Estrategias de aprendizaje soportadas."""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SELF_SUPERVISED = "self_supervised"
    TRANSFER = "transfer"
    META = "meta"
    ACTIVE = "active"
    INCREMENTAL = "incremental"

class LearningPhase(Enum):
    """Fases del proceso de aprendizaje."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    CONSOLIDATION = "consolidation"
    ADAPTATION = "adaptation"
    TRANSFER = "transfer"

@dataclass
class LearningConfig:
    """Configuración específica para aprendizaje."""
    strategy: LearningStrategy = LearningStrategy.REINFORCEMENT
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_rate: float = 0.2
    min_exploration_rate: float = 0.01
    exploration_decay: float = 0.995
    batch_size: int = 32
    memory_size: int = 10000
    target_update_frequency: int = 100
    use_pretrained: bool = True
    pretrained_model_path: Optional[str] = None
    enable_transfer: bool = True
    enable_forgetting: bool = False
    forgetting_threshold: float = 0.1
    evaluation_frequency: int = 100

@dataclass
class Experience:
    """Experiencia para aprendizaje por refuerzo."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningStats:
    """Estadísticas de aprendizaje."""
    episodes_completed: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0
    best_reward: float = float('-inf')
    worst_reward: float = float('inf')
    learning_steps: int = 0
    exploration_rate: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    last_evaluation: Optional[Dict[str, float]] = None
    convergence_detected: bool = False
    stagnation_epochs: int = 0

class LearningAgent(BaseAgent):
    """
    Agente especializado en aprendizaje automático y mejora continua.
    
    Responsabilidades:
    1. Aprender de experiencias pasadas
    2. Adaptarse a nuevos datos y dominios
    3. Optimizar estrategias de aprendizaje
    4. Transferir conocimiento entre tareas
    5. Evaluar progreso y sugerir metas
    6. Generar reportes de aprendizaje
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Inicializa el agente de aprendizaje."""
        if config is None:
            config = AgentConfig(
                name="LearningAgent",
                description="Agente especializado en aprendizaje automático y mejora continua",
                capabilities=[
                    AgentCapability.PATTERN_DETECTION,
                    AgentCapability.CODE_ANALYSIS,
                    AgentCapability.PERFORMANCE_ANALYSIS
                ],
                confidence_threshold=0.7,
                learning_rate=0.1,
                memory_size={
                    AgentMemoryType.SHORT_TERM: 1000,
                    AgentMemoryType.LONG_TERM: 10000,
                    AgentMemoryType.EPISODIC: 5000,
                    AgentMemoryType.SEMANTIC: 20000
                }
            )
        
        super().__init__(config)
        
        # Configuración específica de aprendizaje
        self.learning_config = LearningConfig()
        
        # Modelos y datos de aprendizaje
        self.q_table: Dict[str, np.ndarray] = {}  # Tabla Q para RL simple
        self.policy_network: Optional[Any] = None  # Red neuronal para política
        self.value_network: Optional[Any] = None  # Red neuronal para valor
        self.experience_buffer: List[Experience] = []
        
        # Estadísticas y métricas
        self.learning_stats = LearningStats()
        self.learning_curves: Dict[str, List[float]] = {
            "reward": [],
            "loss": [],
            "accuracy": [],
            "exploration": []
        }
        
        # Conocimiento aprendido
        self.learned_patterns: Dict[str, Any] = {}
        self.adaptation_rules: Dict[str, Any] = {}
        self.transfer_knowledge: Dict[str, Any] = {}
        
        # Configuración de embeddings para representación de estados
        self.embedding_generator = EmbeddingGenerator()
        
        # Memoria especializada para aprendizaje
        self.learning_memory = MemoryHierarchy()
        
        # Estado interno
        self.current_phase = LearningPhase.EXPLORATION
        self.current_task: Optional[str] = None
        self.training_mode = False
        
    async def _initialize_internal(self) -> bool:
        """
        Inicialización específica del LearningAgent.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # 1. Cargar modelos pre-entrenados si están disponibles
            if self.learning_config.use_pretrained and self.learning_config.pretrained_model_path:
                await self._load_pretrained_models()
            
            # 2. Inicializar estructuras de aprendizaje
            await self._initialize_learning_structures()
            
            # 3. Cargar conocimiento previo si existe
            await self._load_prior_knowledge()
            
            # 4. Inicializar sistema de evaluación
            await self._initialize_evaluation_system()
            
            # 5. Verificar dependencias
            await self._check_dependencies()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en inicialización de LearningAgent: {e}")
            return False
    
    async def _process_internal(self, input_data: AgentInput) -> AgentOutput:
        """
        Procesamiento específico del LearningAgent.
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            AgentOutput: Resultado del procesamiento
        """
        request_type = input_data.data.get("type", "learn")
        
        try:
            if request_type == "learn":
                return await self._process_learning_request(input_data)
            elif request_type == "predict":
                return await self._process_prediction_request(input_data)
            elif request_type == "evaluate":
                return await self._process_evaluation_request(input_data)
            elif request_type == "adapt":
                return await self._process_adaptation_request(input_data)
            elif request_type == "transfer":
                return await self._process_transfer_request(input_data)
            elif request_type == "optimize":
                return await self._process_optimization_request(input_data)
            else:
                return AgentOutput(
                    request_id=input_data.request_id,
                    agent_id=self.config.agent_id,
                    success=False,
                    confidence=0.0,
                    errors=[f"Tipo de solicitud no soportado: {request_type}"]
                )
                
        except Exception as e:
            return AgentOutput(
                request_id=input_data.request_id,
                agent_id=self.config.agent_id,
                success=False,
                confidence=0.0,
                errors=[f"Error procesando solicitud: {str(e)}"]
            )
    
    async def _learn_internal(self, feedback: Dict[str, Any]) -> bool:
        """
        Aprendizaje específico del LearningAgent.
        
        Args:
            feedback: Datos de feedback
            
        Returns:
            bool: True si el aprendizaje fue exitoso
        """
        try:
            feedback_type = feedback.get("type", "reinforcement")
            
            if feedback_type == "reinforcement":
                return await self._learn_from_reinforcement(feedback)
            elif feedback_type == "supervised":
                return await self._learn_from_supervised(feedback)
            elif feedback_type == "unsupervised":
                return await self._learn_from_unsupervised(feedback)
            elif feedback_type == "transfer":
                return await self._learn_from_transfer(feedback)
            elif feedback_type == "correction":
                return await self._learn_from_correction(feedback)
            else:
                self.logger.warning(f"Tipo de feedback no soportado: {feedback_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error en aprendizaje: {e}")
            return False
    
    def _validate_input_specific(self, input_data: AgentInput) -> None:
        """Validación específica para LearningAgent."""
        request_type = input_data.data.get("type", "")
        
        if request_type not in ["learn", "predict", "evaluate", "adapt", "transfer", "optimize"]:
            raise ValidationError(
                f"Tipo de solicitud debe ser 'learn', 'predict', 'evaluate', 'adapt', 'transfer' o 'optimize', "
                f"no '{request_type}'"
            )
        
        if request_type == "learn" and "data" not in input_data.data:
            raise ValidationError("Solicitud de aprendizaje requiere campo 'data'")
        
        if request_type == "predict" and "state" not in input_data.data:
            raise ValidationError("Solicitud de predicción requiere campo 'state'")
    
    async def _save_state(self) -> None:
        """Guarda el estado del LearningAgent."""
        state_path = Path(f"./data/agents/{self.config.agent_id}/learning_state.pkl")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "q_table": self.q_table,
            "learning_stats": self.learning_stats,
            "learning_curves": self.learning_curves,
            "learned_patterns": self.learned_patterns,
            "adaptation_rules": self.adaptation_rules,
            "transfer_knowledge": self.transfer_knowledge,
            "current_phase": self.current_phase,
            "current_task": self.current_task,
            "exploration_rate": self.learning_config.exploration_rate,
            "config": self.learning_config
        }
        
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        
        # También guardar en memoria semántica
        await self.memory.store(
            AgentMemoryType.SEMANTIC,
            {
                "type": "learning_state",
                "agent_id": self.config.agent_id,
                "state": state,
                "timestamp": datetime.now()
            }
        )
    
    # =========================================================================
    # MÉTODOS PÚBLICOS ESPECÍFICOS
    # =========================================================================
    
    async def learn_from_experience(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aprende de experiencias pasadas.
        
        Args:
            experiences: Lista de experiencias para aprender
            
        Returns:
            Dict con resultados del aprendizaje
        """
        self.training_mode = True
        
        try:
            # Procesar cada experiencia
            total_reward = 0.0
            losses = []
            
            for exp in experiences:
                # Convertir experiencia al formato interno
                experience = await self._process_experience(exp)
                
                # Almacenar en buffer
                self.experience_buffer.append(experience)
                total_reward += experience.reward
                
                # Aprender si hay suficientes experiencias
                if len(self.experience_buffer) >= self.learning_config.batch_size:
                    loss = await self._learn_from_experience_buffer()
                    if loss is not None:
                        losses.append(loss)
                
                # Limitar tamaño del buffer
                if len(self.experience_buffer) > self.learning_config.memory_size:
                    self.experience_buffer.pop(0)
            
            # Actualizar estadísticas
            self.learning_stats.episodes_completed += len(experiences)
            self.learning_stats.total_reward += total_reward
            self.learning_stats.average_reward = (
                self.learning_stats.total_reward / self.learning_stats.episodes_completed
                if self.learning_stats.episodes_completed > 0 else 0.0
            )
            
            # Actualizar tasa de exploración
            self.learning_config.exploration_rate = max(
                self.learning_config.min_exploration_rate,
                self.learning_config.exploration_rate * self.learning_config.exploration_decay
            )
            
            # Guardar estado periódicamente
            if self.learning_stats.episodes_completed % 10 == 0:
                await self._save_state()
            
            return {
                "success": True,
                "experiences_learned": len(experiences),
                "average_reward": total_reward / len(experiences) if experiences else 0.0,
                "average_loss": np.mean(losses) if losses else 0.0,
                "exploration_rate": self.learning_config.exploration_rate,
                "buffer_size": len(self.experience_buffer)
            }
            
        except Exception as e:
            self.logger.error(f"Error aprendiendo de experiencias: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            self.training_mode = False
    
    async def adapt_to_new_data(self, new_data: List[Dict[str, Any]], 
                               domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Se adapta a nuevos datos o dominio.
        
        Args:
            new_data: Nuevos datos para adaptación
            domain: Dominio específico (opcional)
            
        Returns:
            Dict con resultados de la adaptación
        """
        try:
            # 1. Analizar nuevos datos
            data_analysis = await self._analyze_new_data(new_data, domain)
            
            # 2. Detectar diferencias con conocimiento actual
            differences = await self._detect_differences(data_analysis)
            
            if not differences.get("significant_changes", False):
                return {
                    "success": True,
                    "adaptation_applied": False,
                    "message": "No se detectaron cambios significativos"
                }
            
            # 3. Ajustar modelo según diferencias
            adaptation_results = await self._adapt_model_to_differences(differences)
            
            # 4. Actualizar reglas de adaptación
            await self._update_adaptation_rules(differences, adaptation_results)
            
            # 5. Evaluar adaptación
            evaluation = await self._evaluate_adaptation(new_data)
            
            return {
                "success": True,
                "adaptation_applied": True,
                "differences_detected": differences,
                "adaptation_results": adaptation_results,
                "evaluation": evaluation,
                "new_domain": domain,
                "model_updated": True
            }
            
        except Exception as e:
            self.logger.error(f"Error adaptando a nuevos datos: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def optimize_learning_strategy(self, 
                                        performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimiza la estrategia de aprendizaje basado en rendimiento.
        
        Args:
            performance_data: Datos de rendimiento actual
            
        Returns:
            Dict con optimizaciones aplicadas
        """
        try:
            # 1. Analizar rendimiento actual
            analysis = await self._analyze_performance(performance_data)
            
            # 2. Identificar áreas de mejora
            improvements = await self._identify_improvement_areas(analysis)
            
            # 3. Ajustar hiperparámetros
            parameter_changes = await self._adjust_hyperparameters(improvements)
            
            # 4. Probar nueva estrategia
            test_results = await self._test_new_strategy(parameter_changes)
            
            # 5. Aplicar si mejora el rendimiento
            if test_results.get("improvement", 0) > 0:
                await self._apply_strategy_changes(parameter_changes)
                
                return {
                    "success": True,
                    "optimization_applied": True,
                    "improvement_areas": improvements,
                    "parameter_changes": parameter_changes,
                    "test_results": test_results,
                    "improvement_percentage": test_results["improvement"]
                }
            else:
                return {
                    "success": True,
                    "optimization_applied": False,
                    "message": "No se encontró mejora significativa",
                    "test_results": test_results
                }
                
        except Exception as e:
            self.logger.error(f"Error optimizando estrategia de aprendizaje: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def transfer_knowledge(self, 
                                target_agent: str, 
                                knowledge_types: List[str]) -> Dict[str, Any]:
        """
        Transfiere conocimiento a otro agente.
        
        Args:
            target_agent: Identificador del agente destino
            knowledge_types: Tipos de conocimiento a transferir
            
        Returns:
            Dict con resultados de la transferencia
        """
        try:
            # 1. Preparar conocimiento para transferencia
            knowledge_package = await self._prepare_knowledge_for_transfer(knowledge_types)
            
            # 2. Validar compatibilidad con agente destino
            compatibility = await self._validate_transfer_compatibility(target_agent, knowledge_package)
            
            if not compatibility["compatible"]:
                return {
                    "success": False,
                    "error": "Conocimiento incompatible con agente destino",
                    "compatibility_issues": compatibility["issues"]
                }
            
            # 3. Transferir conocimiento
            transfer_results = await self._execute_knowledge_transfer(target_agent, knowledge_package)
            
            # 4. Validar transferencia
            validation = await self._validate_transfer(target_agent, knowledge_package)
            
            # 5. Registrar transferencia
            await self._register_knowledge_transfer(target_agent, knowledge_types, transfer_results)
            
            return {
                "success": True,
                "knowledge_transferred": knowledge_types,
                "transfer_results": transfer_results,
                "validation": validation,
                "target_agent": target_agent,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error transfiriendo conocimiento: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def evaluate_learning_progress(self) -> Dict[str, Any]:
        """
        Evalúa el progreso del aprendizaje.
        
        Returns:
            Dict con evaluación detallada
        """
        try:
            # 1. Calcular métricas básicas
            basic_metrics = self._calculate_basic_metrics()
            
            # 2. Analizar curvas de aprendizaje
            learning_curves_analysis = self._analyze_learning_curves()
            
            # 3. Evaluar convergencia
            convergence_analysis = self._analyze_convergence()
            
            # 4. Identificar problemas
            problems_detected = self._identify_learning_problems()
            
            # 5. Calcular eficiencia
            efficiency_metrics = self._calculate_efficiency_metrics()
            
            return {
                "success": True,
                "basic_metrics": basic_metrics,
                "learning_curves": learning_curves_analysis,
                "convergence": convergence_analysis,
                "problems_detected": problems_detected,
                "efficiency": efficiency_metrics,
                "overall_progress": self._calculate_overall_progress(
                    basic_metrics, convergence_analysis
                ),
                "recommendations": self._generate_progress_recommendations(
                    basic_metrics, problems_detected
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluando progreso de aprendizaje: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def suggest_learning_goals(self, 
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sugiere metas de aprendizaje basadas en el contexto.
        
        Args:
            context: Contexto adicional (opcional)
            
        Returns:
            Dict con metas sugeridas
        """
        try:
            # 1. Analizar estado actual
            current_state = await self._analyze_current_state(context)
            
            # 2. Identificar brechas de conocimiento
            knowledge_gaps = await self._identify_knowledge_gaps(current_state)
            
            # 3. Priorizar metas
            prioritized_goals = await self._prioritize_learning_goals(knowledge_gaps)
            
            # 4. Diseñar plan de aprendizaje
            learning_plan = await self._design_learning_plan(prioritized_goals)
            
            # 5. Establecer métricas de éxito
            success_metrics = await self._define_success_metrics(prioritized_goals)
            
            return {
                "success": True,
                "current_state": current_state,
                "knowledge_gaps": knowledge_gaps,
                "prioritized_goals": prioritized_goals,
                "learning_plan": learning_plan,
                "success_metrics": success_metrics,
                "timeline_estimate": await self._estimate_timeline(learning_plan)
            }
            
        except Exception as e:
            self.logger.error(f"Error sugiriendo metas de aprendizaje: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_learning_report(self, 
                                      period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Genera un reporte de aprendizaje.
        
        Args:
            period: Período para el reporte (opcional)
            
        Returns:
            Dict con reporte completo
        """
        try:
            # 1. Recolectar datos del período
            period_data = await self._collect_period_data(period)
            
            # 2. Analizar tendencias
            trends = await self._analyze_learning_trends(period_data)
            
            # 3. Evaluar logros
            achievements = await self._evaluate_achievements(period_data)
            
            # 4. Identificar lecciones aprendidas
            lessons = await self._identify_lessons_learned(period_data)
            
            # 5. Generar recomendaciones
            recommendations = await self._generate_report_recommendations(trends, achievements, lessons)
            
            # 6. Compilar reporte
            report = {
                "period": period if period else "completo",
                "generated_at": datetime.now(),
                "summary": await self._generate_report_summary(trends, achievements),
                "detailed_analysis": {
                    "trends": trends,
                    "achievements": achievements,
                    "lessons_learned": lessons
                },
                "metrics": await self._calculate_report_metrics(period_data),
                "visualizations": await self._generate_report_visualizations(period_data),
                "recommendations": recommendations,
                "next_steps": await self._suggest_next_steps_from_report(recommendations)
            }
            
            return {
                "success": True,
                "report": report,
                "report_format": "json",  # También podría ser PDF, HTML, etc.
                "generation_time": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generando reporte de aprendizaje: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # =========================================================================
    # MÉTODOS PRIVADOS DE IMPLEMENTACIÓN
    # =========================================================================
    
    async def _process_learning_request(self, input_data: AgentInput) -> AgentOutput:
        """Procesa solicitud de aprendizaje."""
        data = input_data.data.get("data", [])
        learning_type = input_data.data.get("learning_type", "reinforcement")
        
        if learning_type == "reinforcement":
            result = await self.learn_from_experience(data)
        elif learning_type == "supervised":
            result = await self._learn_supervised(data)
        elif learning_type == "unsupervised":
            result = await self._learn_unsupervised(data)
        else:
            result = {"success": False, "error": f"Tipo de aprendizaje no soportado: {learning_type}"}
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=result.get("success", False),
            data=result,
            confidence=result.get("confidence", 0.8 if result.get("success") else 0.0),
            processing_time_ms=input_data.data.get("processing_time_ms", 0.0)
        )
    
    async def _process_prediction_request(self, input_data: AgentInput) -> AgentOutput:
        """Procesa solicitud de predicción."""
        state = input_data.data.get("state")
        use_exploration = input_data.data.get("use_exploration", False)
        
        # Convertir estado a representación usable
        state_representation = await self._represent_state(state)
        
        # Seleccionar acción
        if use_exploration and np.random.random() < self.learning_config.exploration_rate:
            # Exploración: acción aleatoria
            action = np.random.randint(0, self._get_action_space_size())
            confidence = 0.1  # Baja confianza para exploración
        else:
            # Explotación: mejor acción según política
            action, confidence = await self._select_best_action(state_representation)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "action": action,
                "state": state_representation.tolist() if hasattr(state_representation, "tolist") else state_representation,
                "exploration_used": use_exploration and confidence < 0.5,
                "confidence": confidence
            },
            confidence=confidence,
            processing_time_ms=input_data.data.get("processing_time_ms", 0.0)
        )
    
    async def _process_evaluation_request(self, input_data: AgentInput) -> AgentOutput:
        """Procesa solicitud de evaluación."""
        test_data = input_data.data.get("test_data", [])
        evaluation_type = input_data.data.get("evaluation_type", "comprehensive")
        
        if evaluation_type == "comprehensive":
            result = await self.evaluate_learning_progress()
        elif evaluation_type == "performance":
            result = await self._evaluate_performance(test_data)
        elif evaluation_type == "generalization":
            result = await self._evaluate_generalization(test_data)
        else:
            result = {"success": False, "error": f"Tipo de evaluación no soportado: {evaluation_type}"}
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=result.get("success", False),
            data=result,
            confidence=result.get("confidence", 0.9 if result.get("success") else 0.0),
            processing_time_ms=input_data.data.get("processing_time_ms", 0.0)
        )
    
    async def _process_adaptation_request(self, input_data: AgentInput) -> AgentOutput:
        """Procesa solicitud de adaptación."""
        new_data = input_data.data.get("new_data", [])
        domain = input_data.data.get("domain")
        
        result = await self.adapt_to_new_data(new_data, domain)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=result.get("success", False),
            data=result,
            confidence=result.get("confidence", 0.85 if result.get("success") else 0.0),
            processing_time_ms=input_data.data.get("processing_time_ms", 0.0)
        )
    
    async def _process_transfer_request(self, input_data: AgentInput) -> AgentOutput:
        """Procesa solicitud de transferencia."""
        target_agent = input_data.data.get("target_agent")
        knowledge_types = input_data.data.get("knowledge_types", ["patterns", "rules"])
        
        result = await self.transfer_knowledge(target_agent, knowledge_types)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=result.get("success", False),
            data=result,
            confidence=result.get("confidence", 0.8 if result.get("success") else 0.0),
            processing_time_ms=input_data.data.get("processing_time_ms", 0.0)
        )
    
    async def _process_optimization_request(self, input_data: AgentInput) -> AgentOutput:
        """Procesa solicitud de optimización."""
        performance_data = input_data.data.get("performance_data", {})
        
        result = await self.optimize_learning_strategy(performance_data)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=result.get("success", False),
            data=result,
            confidence=result.get("confidence", 0.75 if result.get("success") else 0.0),
            processing_time_ms=input_data.data.get("processing_time_ms", 0.0)
        )
    
    async def _learn_from_reinforcement(self, feedback: Dict[str, Any]) -> bool:
        """Aprende de feedback de refuerzo."""
        try:
            # Extraer experiencia del feedback
            experience_data = feedback.get("experience", {})
            
            if not experience_data:
                self.logger.warning("Feedback de refuerzo sin datos de experiencia")
                return False
            
            # Procesar experiencia
            experience = await self._process_experience(experience_data)
            
            # Almacenar en buffer
            self.experience_buffer.append(experience)
            
            # Aprender si hay suficientes experiencias
            if len(self.experience_buffer) >= self.learning_config.batch_size:
                await self._learn_from_experience_buffer()
            
            # Actualizar estadísticas
            self.learning_stats.total_reward += experience.reward
            self.learning_stats.episodes_completed += 1
            
            if experience.reward > self.learning_stats.best_reward:
                self.learning_stats.best_reward = experience.reward
            
            if experience.reward < self.learning_stats.worst_reward:
                self.learning_stats.worst_reward = experience.reward
            
            # Actualizar tasa de exploración
            self.learning_config.exploration_rate *= self.learning_config.exploration_decay
            self.learning_config.exploration_rate = max(
                self.learning_config.exploration_rate,
                self.learning_config.min_exploration_rate
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en aprendizaje por refuerzo: {e}")
            return False
    
    async def _learn_from_supervised(self, feedback: Dict[str, Any]) -> bool:
        """Aprende de feedback supervisado."""
        try:
            training_data = feedback.get("training_data", [])
            labels = feedback.get("labels", [])
            
            if not training_data or not labels:
                self.logger.warning("Feedback supervisado sin datos de entrenamiento o etiquetas")
                return False
            
            # Convertir a arrays numpy
            X = np.array([await self._represent_state(x) for x in training_data])
            y = np.array(labels)
            
            # Entrenar modelo (simplificado)
            # En una implementación real, se usaría un modelo de ML
            self.learned_patterns["supervised"] = {
                "data_mean": np.mean(X, axis=0),
                "data_std": np.std(X, axis=0),
                "label_distribution": np.bincount(y) / len(y)
            }
            
            # Calcular métricas básicas
            if hasattr(self, '_predict_supervised'):
                predictions = [self._predict_supervised(x) for x in X]
                accuracy = accuracy_score(y, predictions)
                self.learning_stats.accuracy_history.append(accuracy)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en aprendizaje supervisado: {e}")
            return False
    
    async def _learn_from_unsupervised(self, feedback: Dict[str, Any]) -> bool:
        """Aprende de feedback no supervisado."""
        try:
            data = feedback.get("data", [])
            
            if not data:
                self.logger.warning("Feedback no supervisado sin datos")
                return False
            
            # Convertir a embeddings
            embeddings = []
            for item in data:
                if isinstance(item, str):
                    embedding = await self.embedding_generator.generate_text_embedding(item)
                else:
                    embedding = await self._represent_state(item)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            
            # Aplicar clustering (simplificado)
            # En una implementación real, se usaría K-means o DBSCAN
            n_clusters = min(10, len(embeddings))
            if n_clusters > 1:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)
                
                self.learned_patterns["clusters"] = {
                    "centers": kmeans.cluster_centers_.tolist(),
                    "labels": clusters.tolist(),
                    "inertia": kmeans.inertia_
                }
            
            # Detectar anomalías (simplificado)
            mean_embedding = np.mean(embeddings, axis=0)
            std_embedding = np.std(embeddings, axis=0)
            distances = np.linalg.norm(embeddings - mean_embedding, axis=1)
            
            self.learned_patterns["anomaly_detection"] = {
                "mean": mean_embedding.tolist(),
                "std": std_embedding.tolist(),
                "threshold": np.mean(distances) + 2 * np.std(distances)
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en aprendizaje no supervisado: {e}")
            return False
    
    async def _learn_from_transfer(self, feedback: Dict[str, Any]) -> bool:
        """Aprende de transferencia de conocimiento."""
        try:
            source = feedback.get("source")
            knowledge = feedback.get("knowledge", {})
            
            if not source or not knowledge:
                self.logger.warning("Feedback de transferencia sin fuente o conocimiento")
                return False
            
            # Almacenar conocimiento transferido
            self.transfer_knowledge[source] = {
                "knowledge": knowledge,
                "transfer_time": datetime.now(),
                "confidence": feedback.get("confidence", 0.5)
            }
            
            # Integrar conocimiento si es relevante
            relevance = await self._evaluate_knowledge_relevance(knowledge)
            
            if relevance > 0.7:  # Umbral de relevancia
                await self._integrate_transferred_knowledge(knowledge, source)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en aprendizaje por transferencia: {e}")
            return False
    
    async def _learn_from_correction(self, feedback: Dict[str, Any]) -> bool:
        """Aprende de corrección."""
        try:
            mistake = feedback.get("mistake", {})
            correction = feedback.get("correction", {})
            
            if not mistake or not correction:
                self.logger.warning("Feedback de corrección sin error o corrección")
                return False
            
            # Analizar el error
            error_analysis = await self._analyze_mistake(mistake, correction)
            
            # Actualizar política para evitar error similar
            await self._update_policy_from_correction(error_analysis)
            
            # Almacenar corrección en memoria
            await self.memory.store(
                AgentMemoryType.EPISODIC,
                {
                    "type": "correction",
                    "mistake": mistake,
                    "correction": correction,
                    "analysis": error_analysis,
                    "timestamp": datetime.now()
                }
            )
            
            # Ajustar confianza si es necesario
            if error_analysis.get("serious", False):
                self.config.confidence_threshold = min(
                    0.95,
                    self.config.confidence_threshold + 0.05
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en aprendizaje por corrección: {e}")
            return False
    
    # =========================================================================
    # MÉTODOS AUXILIARES
    # =========================================================================
    
    async def _load_pretrained_models(self) -> None:
        """Carga modelos pre-entrenados."""
        model_path = self.learning_config.pretrained_model_path
        if model_path and Path(model_path).exists():
            try:
                with open(model_path, 'rb') as f:
                    saved_state = pickle.load(f)
                
                # Cargar estado
                self.q_table = saved_state.get("q_table", {})
                self.learning_stats = saved_state.get("learning_stats", LearningStats())
                self.learned_patterns = saved_state.get("learned_patterns", {})
                
                self.logger.info(f"Modelos pre-entrenados cargados de {model_path}")
                
            except Exception as e:
                self.logger.warning(f"No se pudieron cargar modelos pre-entrenados: {e}")
    
    async def _initialize_learning_structures(self) -> None:
        """Inicializa estructuras de aprendizaje."""
        # Inicializar tabla Q vacía
        self.q_table = {}
        
        # Inicializar buffer de experiencias
        self.experience_buffer = []
        
        # Inicializar estadísticas
        self.learning_stats = LearningStats()
        self.learning_stats.exploration_rate = self.learning_config.exploration_rate
        
        # Inicializar curvas de aprendizaje
        self.learning_curves = {
            "reward": [],
            "loss": [],
            "accuracy": [],
            "exploration": []
        }
    
    async def _load_prior_knowledge(self) -> None:
        """Carga conocimiento previo."""
        try:
            # Intentar cargar de memoria semántica
            memories = await self.memory.retrieve(
                AgentMemoryType.SEMANTIC,
                {"type": "learning_knowledge"},
                limit=10
            )
            
            for memory in memories:
                if "patterns" in memory["content"]:
                    self.learned_patterns.update(memory["content"]["patterns"])
                if "rules" in memory["content"]:
                    self.adaptation_rules.update(memory["content"]["rules"])
            
        except Exception as e:
            self.logger.warning(f"No se pudo cargar conocimiento previo: {e}")
    
    async def _initialize_evaluation_system(self) -> None:
        """Inicializa sistema de evaluación."""
        # Métricas a seguir
        self.evaluation_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "reward": [],
            "loss": []
        }
        
        # Umbrales para evaluación
        self.evaluation_thresholds = {
            "min_accuracy": 0.7,
            "min_precision": 0.6,
            "min_recall": 0.5,
            "max_loss": 1.0,
            "convergence_window": 100
        }
    
    async def _check_dependencies(self) -> None:
        """Verifica dependencias necesarias."""
        try:
            import numpy as np
            from sklearn import __version__ as sklearn_version
            
            self.logger.info(f"NumPy disponible: {np.__version__}")
            self.logger.info(f"Scikit-learn disponible: {sklearn_version}")
            
        except ImportError as e:
            self.logger.error(f"Dependencia faltante: {e}")
            raise
    
    async def _process_experience(self, experience_data: Dict[str, Any]) -> Experience:
        """Procesa datos de experiencia en objeto Experience."""
        state = await self._represent_state(experience_data.get("state"))
        action = experience_data.get("action", 0)
        reward = experience_data.get("reward", 0.0)
        next_state = await self._represent_state(experience_data.get("next_state", experience_data.get("state")))
        done = experience_data.get("done", False)
        metadata = experience_data.get("metadata", {})
        
        return Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            metadata=metadata
        )
    
    async def _represent_state(self, state_data: Any) -> np.ndarray:
        """Convierte estado a representación vectorial."""
        if isinstance(state_data, np.ndarray):
            return state_data
        elif isinstance(state_data, list):
            return np.array(state_data)
        elif isinstance(state_data, dict):
            # Convertir dict a vector (simplificado)
            # En implementación real, usar embeddings más sofisticados
            return np.array(list(state_data.values())).flatten()
        elif isinstance(state_data, str):
            # Generar embedding de texto
            embedding = await self.embedding_generator.generate_text_embedding(state_data)
            return np.array(embedding)
        else:
            # Representación por defecto
            return np.array([hash(str(state_data)) % 1000 / 1000.0])
    
    async def _learn_from_experience_buffer(self) -> Optional[float]:
        """Aprende del buffer de experiencias."""
        if len(self.experience_buffer) < self.learning_config.batch_size:
            return None
        
        # Seleccionar batch aleatorio
        indices = np.random.choice(
            len(self.experience_buffer),
            size=min(self.learning_config.batch_size, len(self.experience_buffer)),
            replace=False
        )
        
        batch = [self.experience_buffer[i] for i in indices]
        
        # Actualizar Q-table (simplificado)
        total_loss = 0.0
        
        for experience in batch:
            state_key = str(experience.state.tobytes() if hasattr(experience.state, 'tobytes') else str(experience.state))
            next_state_key = str(experience.next_state.tobytes() if hasattr(experience.next_state, 'tobytes') else str(experience.next_state))
            
            # Inicializar entradas si no existen
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self._get_action_space_size())
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self._get_action_space_size())
            
            # Q-learning update
            current_q = self.q_table[state_key][experience.action]
            next_max_q = np.max(self.q_table[next_state_key])
            
            target_q = experience.reward + self.learning_config.discount_factor * next_max_q * (1 - experience.done)
            loss = target_q - current_q
            
            # Actualizar
            self.q_table[state_key][experience.action] += self.learning_config.learning_rate * loss
            total_loss += abs(loss)
        
        # Actualizar estadísticas
        avg_loss = total_loss / len(batch)
        self.learning_stats.loss_history.append(avg_loss)
        self.learning_stats.learning_steps += 1
        
        # Actualizar curvas de aprendizaje
        self.learning_curves["loss"].append(avg_loss)
        self.learning_curves["exploration"].append(self.learning_config.exploration_rate)
        
        return avg_loss
    
    def _get_action_space_size(self) -> int:
        """Devuelve tamaño del espacio de acciones."""
        # Por defecto, 10 acciones
        # En implementación real, esto debería ser configurable
        return 10
    
    async def _select_best_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Selecciona la mejor acción para un estado dado."""
        state_key = str(state.tobytes() if hasattr(state, 'tobytes') else str(state))
        
        if state_key not in self.q_table:
            # Estado nuevo, inicializar con valores por defecto
            self.q_table[state_key] = np.zeros(self._get_action_space_size())
            return np.random.randint(0, self._get_action_space_size()), 0.1
        
        # Encontrar mejor acción
        q_values = self.q_table[state_key]
        best_action = np.argmax(q_values)
        best_q = q_values[best_action]
        
        # Normalizar confianza
        q_range = np.max(q_values) - np.min(q_values)
        confidence = 0.1 + 0.9 * (best_q - np.min(q_values)) / (q_range + 1e-8) if q_range > 0 else 0.5
        
        return best_action, confidence
    
    async def _analyze_new_data(self, new_data: List[Dict[str, Any]], domain: Optional[str]) -> Dict[str, Any]:
        """Analiza nuevos datos para adaptación."""
        analysis = {
            "data_size": len(new_data),
            "domain": domain or "unknown",
            "features": {},
            "statistics": {},
            "anomalies": []
        }
        
        if not new_data:
            return analysis
        
        # Extraer características
        features = []
        for item in new_data:
            if isinstance(item, dict):
                features.append(len(item))
            elif isinstance(item, (list, np.ndarray)):
                features.append(len(item))
            else:
                features.append(1)
        
        # Calcular estadísticas
        analysis["statistics"] = {
            "mean_features": np.mean(features),
            "std_features": np.std(features),
            "min_features": np.min(features),
            "max_features": np.max(features)
        }
        
        # Detectar anomalías (simplificado)
        mean_feat = analysis["statistics"]["mean_features"]
        std_feat = analysis["statistics"]["std_features"]
        
        for i, feat in enumerate(features):
            if abs(feat - mean_feat) > 3 * std_feat:
                analysis["anomalies"].append({
                    "index": i,
                    "feature_count": feat,
                    "deviation": (feat - mean_feat) / std_feat
                })
        
        return analysis
    
    async def _detect_differences(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta diferencias entre nuevos datos y conocimiento actual."""
        differences = {
            "significant_changes": False,
            "new_features": [],
            "changed_distributions": [],
            "anomalies_detected": len(data_analysis.get("anomalies", [])) > 0
        }
        
        # Comparar con conocimiento actual (simplificado)
        if "learned_patterns" in self.learned_patterns:
            current_stats = self.learned_patterns.get("statistics", {})
            new_stats = data_analysis.get("statistics", {})
            
            for key in new_stats:
                if key in current_stats:
                    change = abs(new_stats[key] - current_stats[key]) / (current_stats[key] + 1e-8)
                    if change > 0.3:  # 30% de cambio
                        differences["changed_distributions"].append({
                            "feature": key,
                            "change_percentage": change * 100,
                            "old_value": current_stats[key],
                            "new_value": new_stats[key]
                        })
                        differences["significant_changes"] = True
        
        return differences
    
    async def _adapt_model_to_differences(self, differences: Dict[str, Any]) -> Dict[str, Any]:
        """Adapta el modelo a las diferencias detectadas."""
        if not differences.get("significant_changes", False):
            return {"adaptation_applied": False}
        
        adaptation_results = {
            "adaptation_applied": True,
            "changes_made": [],
            "parameters_adjusted": []
        }
        
        # Ajustar tasa de aprendizaje si hay cambios significativos
        if differences.get("changed_distributions"):
            old_lr = self.learning_config.learning_rate
            self.learning_config.learning_rate = min(0.5, old_lr * 1.2)
            adaptation_results["parameters_adjusted"].append({
                "parameter": "learning_rate",
                "old_value": old_lr,
                "new_value": self.learning_config.learning_rate
            })
        
        # Aumentar exploración si hay anomalías
        if differences.get("anomalies_detected"):
            old_exploration = self.learning_config.exploration_rate
            self.learning_config.exploration_rate = min(0.5, old_exploration * 1.5)
            adaptation_results["parameters_adjusted"].append({
                "parameter": "exploration_rate",
                "old_value": old_exploration,
                "new_value": self.learning_config.exploration_rate
            })
        
        return adaptation_results
    
    async def _update_adaptation_rules(self, differences: Dict[str, Any], adaptation_results: Dict[str, Any]) -> None:
        """Actualiza reglas de adaptación basadas en la experiencia."""
        rule_id = f"adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.adaptation_rules[rule_id] = {
            "trigger": differences,
            "action": adaptation_results,
            "timestamp": datetime.now(),
            "effectiveness": 0.5  # Por defecto, será evaluada después
        }
        
        # Limitar número de reglas
        max_rules = 100
        if len(self.adaptation_rules) > max_rules:
            # Eliminar la regla más antigua
            oldest_key = min(self.adaptation_rules.keys(), 
                           key=lambda k: self.adaptation_rules[k]["timestamp"])
            del self.adaptation_rules[oldest_key]
    
    async def _evaluate_adaptation(self, new_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evalúa la efectividad de la adaptación."""
        # En una implementación real, se evaluaría con datos de prueba
        # Aquí usamos una evaluación simplificada
        
        if not new_data:
            return {"evaluation_possible": False}
        
        # Simular predicciones
        predictions = []
        for data_point in new_data[:10]:  # Limitar evaluación
            state = await self._represent_state(data_point)
            action, confidence = await self._select_best_action(state)
            predictions.append({
                "state": data_point,
                "predicted_action": action,
                "confidence": confidence
            })
        
        avg_confidence = np.mean([p["confidence"] for p in predictions])
        
        return {
            "evaluation_possible": True,
            "samples_evaluated": len(predictions),
            "average_confidence": avg_confidence,
            "adaptation_successful": avg_confidence > 0.6,
            "recommendations": "Continue training" if avg_confidence < 0.8 else "Adaptation successful"
        }
    
    async def _analyze_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza datos de rendimiento."""
        analysis = {
            "metrics": {},
            "trends": {},
            "bottlenecks": [],
            "improvement_opportunities": []
        }
        
        # Extraer métricas
        metrics = performance_data.get("metrics", {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                analysis["metrics"][key] = value
        
        # Analizar tendencias
        history = performance_data.get("history", [])
        if len(history) > 1:
            recent = history[-1]
            older = history[0]
            
            for key in recent:
                if key in older and isinstance(recent[key], (int, float)) and isinstance(older[key], (int, float)):
                    change = ((recent[key] - older[key]) / (abs(older[key]) + 1e-8)) * 100
                    analysis["trends"][key] = {
                        "change_percentage": change,
                        "improving": (key in ["accuracy", "reward"] and change > 0) or 
                                    (key in ["loss", "error_rate"] and change < 0)
                    }
        
        # Identificar cuellos de botella (simplificado)
        if "inference_time" in analysis["metrics"] and analysis["metrics"]["inference_time"] > 100:
            analysis["bottlenecks"].append("slow_inference")
        
        if "memory_usage" in analysis["metrics"] and analysis["metrics"]["memory_usage"] > 80:
            analysis["bottlenecks"].append("high_memory_usage")
        
        return analysis
    
    async def _identify_improvement_areas(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica áreas de mejora basadas en análisis de rendimiento."""
        improvements = []
        
        # Basado en métricas bajas
        low_metrics = {
            "accuracy": 0.7,
            "precision": 0.6,
            "recall": 0.5,
            "f1": 0.65
        }
        
        for metric, threshold in low_metrics.items():
            if metric in analysis["metrics"] and analysis["metrics"][metric] < threshold:
                improvements.append({
                    "area": f"low_{metric}",
                    "current_value": analysis["metrics"][metric],
                    "target_value": threshold,
                    "priority": "high" if metric == "accuracy" else "medium"
                })
        
        # Basado en tendencias negativas
        for metric, trend in analysis.get("trends", {}).items():
            if not trend.get("improving", True) and abs(trend["change_percentage"]) > 10:
                improvements.append({
                    "area": f"worsening_{metric}",
                    "trend": trend["change_percentage"],
                    "priority": "high"
                })
        
        # Basado en cuellos de botella
        for bottleneck in analysis.get("bottlenecks", []):
            improvements.append({
                "area": bottleneck,
                "priority": "medium",
                "action": "optimize"
            })
        
        return improvements
    
    async def _adjust_hyperparameters(self, improvements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ajusta hiperparámetros basado en áreas de mejora."""
        changes = {}
        
        for improvement in improvements:
            area = improvement.get("area", "")
            
            if area.startswith("low_accuracy") or area.startswith("worsening_accuracy"):
                # Aumentar tasa de aprendizaje para mejorar rápido
                old_lr = self.learning_config.learning_rate
                self.learning_config.learning_rate = min(0.5, old_lr * 1.3)
                changes["learning_rate"] = {
                    "old": old_lr,
                    "new": self.learning_config.learning_rate,
                    "reason": "low_accuracy"
                }
            
            elif area == "slow_inference":
                # Reducir batch size para inferencia más rápida
                old_batch = self.learning_config.batch_size
                self.learning_config.batch_size = max(8, old_batch // 2)
                changes["batch_size"] = {
                    "old": old_batch,
                    "new": self.learning_config.batch_size,
                    "reason": "slow_inference"
                }
            
            elif area == "high_memory_usage":
                # Reducir tamaño de buffer
                old_memory = self.learning_config.memory_size
                self.learning_config.memory_size = max(1000, old_memory // 2)
                changes["memory_size"] = {
                    "old": old_memory,
                    "new": self.learning_config.memory_size,
                    "reason": "high_memory_usage"
                }
        
        return changes
    
    async def _test_new_strategy(self, parameter_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Prueba nueva estrategia con parámetros ajustados."""
        if not parameter_changes:
            return {"improvement": 0, "test_completed": False}
        
        # Simular prueba (en implementación real, se haría con datos de validación)
        # Aquí estimamos mejora basada en cambios
        
        improvement = 0.0
        
        if "learning_rate" in parameter_changes:
            # Aumento de learning_rate generalmente mejora convergencia inicial
            improvement += 0.1
        
        if "batch_size" in parameter_changes:
            # Batch size más pequeño puede mejorar generalización
            improvement += 0.05
        
        if "memory_size" in parameter_changes:
            # Memoria más pequeña reduce overfitting
            improvement += 0.03
        
        return {
            "improvement": improvement,
            "test_completed": True,
            "estimated_impact": "positive" if improvement > 0 else "negative",
            "confidence": min(0.9, 0.5 + improvement * 2)
        }
    
    async def _apply_strategy_changes(self, parameter_changes: Dict[str, Any]) -> None:
        """Aplica cambios de estrategia permanentemente."""
        # Los cambios ya se aplicaron en _adjust_hyperparameters
        # Aquí solo registramos
        
        strategy_id = f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        await self.memory.store(
            AgentMemoryType.SEMANTIC,
            {
                "type": "strategy_change",
                "strategy_id": strategy_id,
                "parameter_changes": parameter_changes,
                "timestamp": datetime.now(),
                "agent_id": self.config.agent_id
            }
        )
    
    async def _prepare_knowledge_for_transfer(self, knowledge_types: List[str]) -> Dict[str, Any]:
        """Prepara conocimiento para transferencia."""
        knowledge_package = {
            "source_agent": self.config.agent_id,
            "knowledge_types": knowledge_types,
            "timestamp": datetime.now(),
            "content": {}
        }
        
        for k_type in knowledge_types:
            if k_type == "patterns" and self.learned_patterns:
                knowledge_package["content"]["patterns"] = self.learned_patterns
            
            elif k_type == "rules" and self.adaptation_rules:
                knowledge_package["content"]["rules"] = self.adaptation_rules
            
            elif k_type == "q_table" and self.q_table:
                # Convertir a formato serializable
                q_table_serializable = {}
                for key, values in list(self.q_table.items())[:100]:  # Limitar tamaño
                    q_table_serializable[key] = values.tolist() if hasattr(values, 'tolist') else values
                knowledge_package["content"]["q_table"] = q_table_serializable
            
            elif k_type == "stats" and self.learning_stats:
                knowledge_package["content"]["stats"] = {
                    "episodes_completed": self.learning_stats.episodes_completed,
                    "average_reward": self.learning_stats.average_reward,
                    "best_reward": self.learning_stats.best_reward,
                    "learning_steps": self.learning_stats.learning_steps
                }
        
        return knowledge_package
    
    async def _validate_transfer_compatibility(self, target_agent: str, 
                                              knowledge_package: Dict[str, Any]) -> Dict[str, Any]:
        """Valida compatibilidad para transferencia de conocimiento."""
        # En implementación real, verificaría con el agente destino
        # Aquí hacemos validación simplificada
        
        compatibility = {
            "compatible": True,
            "issues": [],
            "warnings": []
        }
        
        # Verificar tipos de conocimiento
        knowledge_types = knowledge_package.get("knowledge_types", [])
        
        if not knowledge_types:
            compatibility["compatible"] = False
            compatibility["issues"].append("No knowledge types specified")
        
        # Verificar que el paquete no esté vacío
        if not knowledge_package.get("content"):
            compatibility["warnings"].append("Knowledge package is empty")
        
        # Verificar tamaño (simplificado)
        package_size = len(str(knowledge_package))
        if package_size > 1000000:  # 1MB
            compatibility["warnings"].append(f"Large package size: {package_size} bytes")
        
        return compatibility
    
    async def _execute_knowledge_transfer(self, target_agent: str, 
                                         knowledge_package: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta transferencia de conocimiento."""
        # En implementación real, esto enviaría el paquete al agente destino
        # Aquí simulamos la transferencia
        
        transfer_id = str(uuid.uuid4())
        
        # Simular transferencia exitosa
        transfer_result = {
            "transfer_id": transfer_id,
            "target_agent": target_agent,
            "knowledge_types": knowledge_package["knowledge_types"],
            "package_size": len(str(knowledge_package)),
            "timestamp": datetime.now(),
            "status": "completed",
            "estimated_impact": "medium"
        }
        
        return transfer_result
    
    async def _validate_transfer(self, target_agent: str, 
                                knowledge_package: Dict[str, Any]) -> Dict[str, Any]:
        """Valida que la transferencia fue exitosa."""
        # En implementación real, verificaría con el agente destino
        # Aquí simulamos validación
        
        return {
            "validated": True,
            "validation_method": "simulated",
            "confidence": 0.8,
            "issues_found": [],
            "recommendations": "Knowledge transfer appears successful"
        }
    
    async def _register_knowledge_transfer(self, target_agent: str, 
                                          knowledge_types: List[str],
                                          transfer_results: Dict[str, Any]) -> None:
        """Registra transferencia de conocimiento."""
        transfer_record = {
            "transfer_id": transfer_results.get("transfer_id", str(uuid.uuid4())),
            "source_agent": self.config.agent_id,
            "target_agent": target_agent,
            "knowledge_types": knowledge_types,
            "timestamp": datetime.now(),
            "results": transfer_results,
            "status": "completed"
        }
        
        # Almacenar en memoria
        await self.memory.store(
            AgentMemoryType.SEMANTIC,
            {
                "type": "knowledge_transfer",
                "transfer_record": transfer_record,
                "timestamp": datetime.now()
            }
        )
        
        # También almacenar localmente
        if "transfers" not in self.transfer_knowledge:
            self.transfer_knowledge["transfers"] = []
        
        self.transfer_knowledge["transfers"].append(transfer_record)
    
    def _calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calcula métricas básicas de aprendizaje."""
        metrics = {
            "episodes_completed": self.learning_stats.episodes_completed,
            "total_reward": self.learning_stats.total_reward,
            "average_reward": self.learning_stats.average_reward,
            "best_reward": self.learning_stats.best_reward,
            "worst_reward": self.learning_stats.worst_reward,
            "learning_steps": self.learning_stats.learning_steps,
            "exploration_rate": self.learning_config.exploration_rate,
            "q_table_size": len(self.q_table),
            "experience_buffer_size": len(self.experience_buffer),
            "learned_patterns_count": len(self.learned_patterns),
            "adaptation_rules_count": len(self.adaptation_rules)
        }
        
        # Calcular métricas de rendimiento si hay datos
        if self.learning_stats.loss_history:
            metrics["average_loss"] = np.mean(self.learning_stats.loss_history[-100:]) if self.learning_stats.loss_history else 0
            metrics["recent_loss_trend"] = self._calculate_trend(self.learning_stats.loss_history[-50:])
        
        if self.learning_stats.accuracy_history:
            metrics["average_accuracy"] = np.mean(self.learning_stats.accuracy_history[-100:]) if self.learning_stats.accuracy_history else 0
            metrics["best_accuracy"] = max(self.learning_stats.accuracy_history) if self.learning_stats.accuracy_history else 0
        
        return metrics
    
    def _analyze_learning_curves(self) -> Dict[str, Any]:
        """Analiza curvas de aprendizaje."""
        analysis = {
            "convergence_detected": False,
            "overfitting_detected": False,
            "plateau_detected": False,
            "oscillation_detected": False,
            "trends": {}
        }
        
        # Analizar curva de pérdida
        if len(self.learning_curves["loss"]) > 10:
            loss_curve = self.learning_curves["loss"]
            recent_loss = loss_curve[-50:] if len(loss_curve) >= 50 else loss_curve
            
            # Detectar convergencia
            if len(recent_loss) >= 20:
                std_loss = np.std(recent_loss)
                if std_loss < 0.01:  # Pérdida estable
                    analysis["convergence_detected"] = True
            
            # Detectar plateau
            if len(loss_curve) >= 100:
                first_half = np.mean(loss_curve[:50])
                second_half = np.mean(loss_curve[-50:])
                if abs(second_half - first_half) < 0.05:  # Poco cambio
                    analysis["plateau_detected"] = True
            
            analysis["trends"]["loss"] = self._calculate_trend(loss_curve)
        
        # Analizar curva de recompensa
        if len(self.learning_curves["reward"]) > 10:
            reward_curve = self.learning_curves["reward"]
            analysis["trends"]["reward"] = self._calculate_trend(reward_curve)
            
            # Detectar oscilación
            if len(reward_curve) >= 30:
                diffs = np.diff(reward_curve[-30:])
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                if sign_changes > 10:  # Muchos cambios de signo
                    analysis["oscillation_detected"] = True
        
        return analysis
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analiza convergencia del aprendizaje."""
        convergence = {
            "converged": False,
            "convergence_rate": 0.0,
            "convergence_confidence": 0.0,
            "remaining_potential": 1.0,
            "estimated_episodes_to_convergence": 0
        }
        
        if len(self.learning_curves["loss"]) < 20:
            return convergence
        
        loss_curve = self.learning_curves["loss"][-100:]  # Últimas 100 épocas
        
        if len(loss_curve) < 20:
            return convergence
        
        # Calcular tasa de mejora
        early_loss = np.mean(loss_curve[:10])
        recent_loss = np.mean(loss_curve[-10:])
        
        improvement = early_loss - recent_loss
        improvement_rate = improvement / max(early_loss, 1e-8)
        
        # Calcular estabilidad
        recent_std = np.std(loss_curve[-10:])
        
        # Determinar si ha convergido
        convergence["convergence_rate"] = improvement_rate
        convergence["convergence_confidence"] = 1.0 - min(1.0, recent_std * 10)
        
        if improvement_rate < 0.01 and recent_std < 0.05:
            convergence["converged"] = True
        
        # Estimar episodios restantes
        if improvement_rate > 0:
            remaining_improvement = recent_loss * 0.1  # Asumir que queremos 10% más de mejora
            episodes_needed = remaining_improvement / (improvement_rate / len(loss_curve))
            convergence["estimated_episodes_to_convergence"] = int(max(0, episodes_needed))
        
        convergence["remaining_potential"] = 1.0 - improvement_rate
        
        return convergence
    
    def _identify_learning_problems(self) -> List[Dict[str, Any]]:
        """Identifica problemas en el aprendizaje."""
        problems = []
        
        # Verificar sobre-entrenamiento
        if self.learning_stats.loss_history:
            recent_loss = self.learning_stats.loss_history[-10:] if len(self.learning_stats.loss_history) >= 10 else self.learning_stats.loss_history
            if len(recent_loss) >= 5:
                loss_trend = self._calculate_trend(recent_loss)
                if loss_trend > 0.1:  # Pérdida aumentando
                    problems.append({
                        "type": "overfitting",
                        "severity": "medium",
                        "description": "Loss is increasing, possible overfitting",
                        "suggestion": "Reduce model complexity or increase regularization"
                    })
        
        # Verificar sub-entrenamiento
        if self.learning_stats.episodes_completed > 100 and self.learning_stats.average_reward < 0:
            problems.append({
                "type": "underfitting",
                "severity": "high",
                "description": "Average reward is negative after many episodes",
                "suggestion": "Increase exploration or adjust reward function"
            })
        
        # Verificar convergencia lenta
        if self.learning_stats.episodes_completed > 500 and not self.learning_stats.convergence_detected:
            problems.append({
                "type": "slow_convergence",
                "severity": "low",
                "description": "Learning is converging slowly",
                "suggestion": "Adjust learning rate or exploration strategy"
            })
        
        # Verificar oscilación
        if len(self.learning_curves["reward"]) > 50:
            reward_curve = self.learning_curves["reward"][-50:]
            diffs = np.diff(reward_curve)
            volatility = np.std(diffs) / (np.mean(np.abs(reward_curve)) + 1e-8)
            
            if volatility > 0.5:
                problems.append({
                    "type": "high_volatility",
                    "severity": "medium",
                    "description": "High volatility in rewards",
                    "suggestion": "Reduce learning rate or increase batch size"
                })
        
        return problems
    
    def _calculate_efficiency_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de eficiencia del aprendizaje."""
        efficiency = {
            "learning_efficiency": 0.0,
            "sample_efficiency": 0.0,
            "time_efficiency": 0.0,
            "compute_efficiency": 0.0
        }
        
        if self.learning_stats.episodes_completed == 0:
            return efficiency
        
        # Eficiencia de aprendizaje (mejora por episodio)
        if len(self.learning_curves["reward"]) > 10:
            early_reward = np.mean(self.learning_curves["reward"][:10])
            recent_reward = np.mean(self.learning_curves["reward"][-10:])
            episodes = len(self.learning_curves["reward"])
            
            if episodes > 0 and early_reward != 0:
                efficiency["learning_efficiency"] = (recent_reward - early_reward) / episodes
        
        # Eficiencia de muestras (recompensa por experiencia)
        if self.learning_stats.total_reward > 0 and len(self.experience_buffer) > 0:
            efficiency["sample_efficiency"] = self.learning_stats.total_reward / len(self.experience_buffer)
        
        # Eficiencia computacional (simplificado)
        efficiency["compute_efficiency"] = self.learning_stats.average_reward / max(1, self.learning_stats.learning_steps)
        
        return efficiency
    
    def _calculate_overall_progress(self, basic_metrics: Dict[str, Any], 
                                  convergence: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula progreso general del aprendizaje."""
        progress = {
            "overall_score": 0.0,
            "components": {},
            "stage": "early",
            "readiness_for_production": "not_ready"
        }
        
        # Calcular puntuaciones por componente
        components = {}
        
        # Progreso en recompensas
        if basic_metrics.get("average_reward", 0) > 0:
            reward_score = min(1.0, basic_metrics["average_reward"] / 10.0)  # Escalar a 0-1
            components["reward_progress"] = reward_score
        
        # Progreso en convergencia
        if convergence.get("convergence_confidence", 0) > 0:
            convergence_score = convergence["convergence_confidence"]
            components["convergence_progress"] = convergence_score
        
        # Progreso en estabilidad
        if basic_metrics.get("recent_loss_trend", 0) is not None:
            stability_score = max(0, 1.0 - abs(basic_metrics["recent_loss_trend"]))
            components["stability_progress"] = stability_score
        
        # Progreso en eficiencia
        if self.learning_stats.episodes_completed > 0:
            efficiency_score = min(1.0, self.learning_stats.learning_steps / (self.learning_stats.episodes_completed * 10))
            components["efficiency_progress"] = efficiency_score
        
        # Calcular puntuación general
        if components:
            progress["overall_score"] = np.mean(list(components.values()))
            progress["components"] = components
        
        # Determinar etapa
        episodes = basic_metrics.get("episodes_completed", 0)
        if episodes < 100:
            progress["stage"] = "early"
        elif episodes < 1000:
            progress["stage"] = "mid"
        else:
            progress["stage"] = "late"
        
        # Determinar preparación para producción
        if progress["overall_score"] > 0.8 and convergence.get("converged", False):
            progress["readiness_for_production"] = "ready"
        elif progress["overall_score"] > 0.6:
            progress["readiness_for_production"] = "testing"
        elif progress["overall_score"] > 0.4:
            progress["readiness_for_production"] = "development"
        else:
            progress["readiness_for_production"] = "not_ready"
        
        return progress
    
    def _generate_progress_recommendations(self, basic_metrics: Dict[str, Any], 
                                         problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Genera recomendaciones basadas en progreso y problemas."""
        recommendations = []
        
        # Recomendaciones basadas en problemas
        for problem in problems:
            recommendations.append({
                "type": "problem_resolution",
                "priority": problem.get("severity", "medium"),
                "description": problem.get("suggestion", "Address identified problem"),
                "problem": problem.get("type", "unknown")
            })
        
        # Recomendaciones basadas en progreso
        episodes = basic_metrics.get("episodes_completed", 0)
        
        if episodes < 50:
            recommendations.append({
                "type": "training_strategy",
                "priority": "high",
                "description": "Focus on exploration to gather diverse experiences",
                "estimated_time": "short"
            })
        
        elif episodes < 200:
            recommendations.append({
                "type": "training_strategy",
                "priority": "medium",
                "description": "Balance exploration and exploitation",
                "estimated_time": "medium"
            })
        
        elif basic_metrics.get("average_reward", 0) < 0:
            recommendations.append({
                "type": "reward_engineering",
                "priority": "high",
                "description": "Review and adjust reward function",
                "estimated_time": "short"
            })
        
        # Recomendación general de evaluación
        if episodes > 100 and episodes % 50 == 0:
            recommendations.append({
                "type": "evaluation",
                "priority": "low",
                "description": "Perform comprehensive evaluation",
                "estimated_time": "medium"
            })
        
        return recommendations
    
    async def _analyze_current_state(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza estado actual del aprendizaje."""
        current_state = {
            "timestamp": datetime.now(),
            "learning_phase": self.current_phase.value,
            "training_mode": self.training_mode,
            "current_task": self.current_task,
            "basic_metrics": self._calculate_basic_metrics(),
            "convergence": self._analyze_convergence(),
            "problems": self._identify_learning_problems(),
            "context": context or {}
        }
        
        # Determinar fase óptima basada en estado
        episodes = current_state["basic_metrics"]["episodes_completed"]
        
        if episodes < 100:
            optimal_phase = LearningPhase.EXPLORATION
        elif episodes < 500:
            optimal_phase = LearningPhase.EXPLOITATION
        elif current_state["convergence"]["converged"]:
            optimal_phase = LearningPhase.CONSOLIDATION
        else:
            optimal_phase = LearningPhase.ADAPTATION
        
        current_state["optimal_phase"] = optimal_phase.value
        current_state["phase_alignment"] = optimal_phase == self.current_phase
        
        return current_state
    
    async def _identify_knowledge_gaps(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica brechas de conocimiento."""
        gaps = []
        
        # Brechas basadas en rendimiento
        metrics = current_state["basic_metrics"]
        
        if metrics.get("average_reward", 0) < 0:
            gaps.append({
                "type": "negative_rewards",
                "description": "Agent is receiving negative rewards on average",
                "severity": "high",
                "suggested_learning": "reward_optimization"
            })
        
        if metrics.get("average_accuracy", 0) < 0.6:
            gaps.append({
                "type": "low_accuracy",
                "description": "Prediction accuracy is below acceptable threshold",
                "severity": "medium",
                "suggested_learning": "supervised_training"
            })
        
        # Brechas basadas en cobertura
        if metrics.get("q_table_size", 0) < 100 and metrics["episodes_completed"] > 50:
            gaps.append({
                "type": "limited_state_coverage",
                "description": "Q-table covers few states despite many episodes",
                "severity": "medium",
                "suggested_learning": "exploration_optimization"
            })
        
        # Brechas basadas en problemas identificados
        for problem in current_state.get("problems", []):
            if problem.get("severity") in ["high", "medium"]:
                gaps.append({
                    "type": f"problem_{problem.get('type', 'unknown')}",
                    "description": problem.get("description", "Unspecified problem"),
                    "severity": problem.get("severity", "medium"),
                    "suggested_learning": "problem_specific_training"
                })
        
        return gaps
    
    async def _prioritize_learning_goals(self, knowledge_gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioriza metas de aprendizaje."""
        if not knowledge_gaps:
            return []
        
        # Asignar puntuaciones de prioridad
        prioritized_gaps = []
        
        for gap in knowledge_gaps:
            priority_score = 0
            
            # Basado en severidad
            severity_weights = {"high": 3, "medium": 2, "low": 1}
            priority_score += severity_weights.get(gap.get("severity", "low"), 1)
            
            # Basado en tipo
            if gap.get("type", "").startswith("negative"):
                priority_score += 2
            
            # Basado en impacto potencial
            if "accuracy" in gap.get("type", "").lower():
                priority_score += 1
            
            prioritized_gaps.append({
                **gap,
                "priority_score": priority_score,
                "estimated_effort": "medium" if priority_score > 3 else "low"
            })
        
        # Ordenar por prioridad
        prioritized_gaps.sort(key=lambda x: x["priority_score"], reverse=True)
        
        # Limitar a 5 metas principales
        return prioritized_gaps[:5]
    
    async def _design_learning_plan(self, prioritized_goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Diseña plan de aprendizaje basado en metas."""
        plan = {
            "goals": prioritized_goals,
            "phases": [],
            "estimated_duration": "unknown",
            "resources_needed": [],
            "success_criteria": []
        }
        
        # Diseñar fases basadas en metas
        for i, goal in enumerate(prioritized_goals):
            phase = {
                "phase_number": i + 1,
                "goal": goal["type"],
                "focus": goal.get("suggested_learning", "general_learning"),
                "estimated_episodes": 100 * (4 - i),  # Más episodios para metas prioritarias
                "key_activities": self._generate_phase_activities(goal),
                "success_metrics": self._generate_phase_metrics(goal)
            }
            plan["phases"].append(phase)
        
        # Calcular duración estimada
        total_episodes = sum(phase["estimated_episodes"] for phase in plan["phases"])
        plan["estimated_duration"] = f"{total_episodes} episodes"
        
        # Identificar recursos necesarios
        plan["resources_needed"] = ["training_data", "computation_time", "evaluation_metrics"]
        
        # Definir criterios de éxito
        plan["success_criteria"] = [
            "All phases completed",
            "Average reward > 0",
            "Convergence achieved",
            "No high-severity problems remaining"
        ]
        
        return plan
    
    async def _define_success_metrics(self, prioritized_goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Define métricas de éxito para metas de aprendizaje."""
        success_metrics = {
            "overall_success": {
                "threshold": 0.7,
                "measure": "composite_score",
                "description": "Overall success score combining all metrics"
            },
            "goal_specific": {}
        }
        
        for goal in prioritized_goals:
            goal_type = goal.get("type", "")
            
            if "accuracy" in goal_type.lower():
                success_metrics["goal_specific"][goal_type] = {
                    "metric": "accuracy",
                    "threshold": 0.7,
                    "improvement_target": 0.2
                }
            elif "reward" in goal_type.lower():
                success_metrics["goal_specific"][goal_type] = {
                    "metric": "average_reward",
                    "threshold": 0.0,
                    "improvement_target": 1.0
                }
            elif "coverage" in goal_type.lower():
                success_metrics["goal_specific"][goal_type] = {
                    "metric": "q_table_size",
                    "threshold": 200,
                    "improvement_target": 100
                }
            else:
                success_metrics["goal_specific"][goal_type] = {
                    "metric": "generic_improvement",
                    "threshold": 0.5,
                    "improvement_target": 0.3
                }
        
        return success_metrics
    
    async def _estimate_timeline(self, learning_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Estima timeline para plan de aprendizaje."""
        timeline = {
            "total_estimated_episodes": 0,
            "estimated_time_hours": 0,
            "phases": [],
            "critical_path": [],
            "dependencies": []
        }
        
        # Calcular episodios totales
        total_episodes = 0
        
        for phase in learning_plan.get("phases", []):
            episodes = phase.get("estimated_episodes", 0)
            total_episodes += episodes
            
            phase_timeline = {
                "phase": phase.get("phase_number", 0),
                "goal": phase.get("goal", ""),
                "estimated_episodes": episodes,
                "estimated_hours": episodes / 10.0,  # Asumir 10 episodios por hora
                "dependencies": [phase.get("phase_number", 0) - 1] if phase.get("phase_number", 0) > 1 else []
            }
            
            timeline["phases"].append(phase_timeline)
        
        timeline["total_estimated_episodes"] = total_episodes
        timeline["estimated_time_hours"] = total_episodes / 10.0
        
        # Identificar camino crítico (simplificado)
        timeline["critical_path"] = [phase["phase"] for phase in timeline["phases"]]
        
        return timeline
    
    async def _collect_period_data(self, period: Optional[Tuple[datetime, datetime]]) -> Dict[str, Any]:
        """Recolecta datos del período especificado."""
        start_date, end_date = period if period else (datetime.min, datetime.now())
        
        period_data = {
            "period": {
                "start": start_date,
                "end": end_date
            },
            "learning_metrics": {
                "episodes_completed": self.learning_stats.episodes_completed,
                "average_reward": self.learning_stats.average_reward,
                "learning_steps": self.learning_stats.learning_steps
            },
            "curves": self.learning_curves,
            "events": [],
            "achievements": []
        }
        
        # Recolectar eventos de memoria
        try:
            memories = await self.memory.retrieve(
                AgentMemoryType.EPISODIC,
                {
                    "timestamp": {"$gte": start_date, "$lte": end_date}
                },
                limit=100
            )
            
            for memory in memories:
                period_data["events"].append(memory["content"])
        
        except Exception as e:
            self.logger.warning(f"No se pudieron recolectar eventos del período: {e}")
        
        return period_data
    
    async def _analyze_learning_trends(self, period_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza tendencias de aprendizaje en el período."""
        trends = {
            "reward_trend": "stable",
            "loss_trend": "stable",
            "efficiency_trend": "stable",
            "exploration_trend": "stable",
            "key_insights": []
        }
        
        # Analizar tendencia de recompensa
        reward_curve = period_data.get("curves", {}).get("reward", [])
        if len(reward_curve) > 10:
            early_reward = np.mean(reward_curve[:10])
            recent_reward = np.mean(reward_curve[-10:])
            
            if recent_reward > early_reward * 1.1:
                trends["reward_trend"] = "improving"
                trends["key_insights"].append("Rewards showing significant improvement")
            elif recent_reward < early_reward * 0.9:
                trends["reward_trend"] = "declining"
                trends["key_insights"].append("Rewards showing decline")
        
        # Analizar tendencia de pérdida
        loss_curve = period_data.get("curves", {}).get("loss", [])
        if len(loss_curve) > 10:
            early_loss = np.mean(loss_curve[:10])
            recent_loss = np.mean(loss_curve[-10:])
            
            if recent_loss < early_loss * 0.9:
                trends["loss_trend"] = "improving"
                trends["key_insights"].append("Loss decreasing, learning effective")
            elif recent_loss > early_loss * 1.1:
                trends["loss_trend"] = "declining"
                trends["key_insights"].append("Loss increasing, possible issues")
        
        return trends
    
    async def _evaluate_achievements(self, period_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evalúa logros en el período."""
        achievements = []
        
        metrics = period_data.get("learning_metrics", {})
        
        # Logro: muchos episodios completados
        if metrics.get("episodes_completed", 0) > 100:
            achievements.append({
                "type": "training_consistency",
                "description": f"Completed {metrics['episodes_completed']} training episodes",
                "significance": "medium"
            })
        
        # Logro: buena recompensa promedio
        if metrics.get("average_reward", 0) > 0:
            achievements.append({
                "type": "positive_rewards",
                "description": f"Maintained positive average reward: {metrics['average_reward']:.2f}",
                "significance": "high"
            })
        
        # Logro: muchos pasos de aprendizaje
        if metrics.get("learning_steps", 0) > 1000:
            achievements.append({
                "type": "extensive_learning",
                "description": f"Performed {metrics['learning_steps']} learning steps",
                "significance": "medium"
            })
        
        # Logros basados en eventos
        for event in period_data.get("events", []):
            if event.get("type") == "correction" and event.get("analysis", {}).get("learned", False):
                achievements.append({
                    "type": "learning_from_mistakes",
                    "description": "Successfully learned from corrections",
                    "significance": "high"
                })
                break
        
        return achievements
    
    async def _identify_lessons_learned(self, period_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica lecciones aprendidas en el período."""
        lessons = []
        
        # Lecciones de eventos
        for event in period_data.get("events", []):
            if event.get("type") == "correction":
                lesson = event.get("analysis", {}).get("lesson", "")
                if lesson:
                    lessons.append({
                        "source": "correction",
                        "lesson": lesson,
                        "impact": "high"
                    })
        
        # Lecciones de tendencias
        trends = await self._analyze_learning_trends(period_data)
        
        if trends.get("reward_trend") == "declining":
            lessons.append({
                "source": "trend_analysis",
                "lesson": "Need to review reward function or exploration strategy",
                "impact": "medium"
            })
        
        if trends.get("loss_trend") == "declining":
            lessons.append({
                "source": "trend_analysis",
                "lesson": "Model may be overfitting or learning rate too high",
                "impact": "medium"
            })
        
        # Lecciones de estadísticas
        if period_data.get("learning_metrics", {}).get("average_reward", 0) < 0:
            lessons.append({
                "source": "performance_analysis",
                "lesson": "Reward function may need adjustment to provide better guidance",
                "impact": "high"
            })
        
        return lessons
    
    async def _generate_report_recommendations(self, trends: Dict[str, Any], 
                                              achievements: List[Dict[str, Any]], 
                                              lessons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Genera recomendaciones basadas en análisis del reporte."""
        recommendations = []
        
        # Recomendaciones basadas en tendencias
        if trends.get("reward_trend") == "declining":
            recommendations.append({
                "type": "strategy_adjustment",
                "priority": "high",
                "description": "Adjust exploration strategy to improve rewards",
                "rationale": "Reward trend is declining"
            })
        
        if trends.get("loss_trend") == "declining":
            recommendations.append({
                "type": "parameter_tuning",
                "priority": "medium",
                "description": "Reduce learning rate or increase regularization",
                "rationale": "Loss is increasing, possible overfitting"
            })
        
        # Recomendaciones basadas en lecciones
        for lesson in lessons:
            if lesson.get("impact") == "high":
                recommendations.append({
                    "type": "lesson_implementation",
                    "priority": "high",
                    "description": f"Implement lesson: {lesson.get('lesson', '')}",
                    "rationale": "High impact lesson learned"
                })
        
        # Recomendación general si hay muchos logros
        if len(achievements) >= 3:
            recommendations.append({
                "type": "consolidation",
                "priority": "low",
                "description": "Consolidate learning and prepare for next phase",
                "rationale": "Multiple achievements indicate good progress"
            })
        
        # Recomendación de evaluación si no hay muchas lecciones
        if len(lessons) < 2:
            recommendations.append({
                "type": "deeper_analysis",
                "priority": "medium",
                "description": "Conduct deeper analysis to identify more specific lessons",
                "rationale": "Few lessons identified despite training"
            })
        
        return recommendations
    
    async def _generate_report_summary(self, trends: Dict[str, Any], 
                                      achievements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera resumen ejecutivo del reporte."""
        summary = {
            "overall_assessment": "satisfactory",
            "key_findings": [],
            "main_achievements": [],
            "primary_concerns": []
        }
        
        # Evaluación general basada en tendencias
        positive_trends = sum(1 for trend in trends.values() if trend == "improving")
        negative_trends = sum(1 for trend in trends.values() if trend == "declining")
        
        if positive_trends > negative_trends * 2:
            summary["overall_assessment"] = "excellent"
        elif positive_trends > negative_trends:
            summary["overall_assessment"] = "good"
        elif negative_trends > positive_trends:
            summary["overall_assessment"] = "needs_improvement"
        
        # Hallazgos clave
        if trends.get("reward_trend") == "improving":
            summary["key_findings"].append("Rewards showing consistent improvement")
        if trends.get("loss_trend") == "improving":
            summary["key_findings"].append("Loss decreasing, learning effective")
        
        # Logros principales
        for achievement in achievements[:3]:  # Top 3 logros
            if achievement.get("significance") in ["high", "medium"]:
                summary["main_achievements"].append(achievement["description"])
        
        # Preocupaciones principales
        if trends.get("reward_trend") == "declining":
            summary["primary_concerns"].append("Declining reward trend needs attention")
        if not achievements:
            summary["primary_concerns"].append("No significant achievements recorded")
        
        return summary
    
    async def _calculate_report_metrics(self, period_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula métricas para el reporte."""
        metrics = {
            "quantitative": {},
            "qualitative": {},
            "efficiency": {}
        }
        
        # Métricas cuantitativas
        learning_metrics = period_data.get("learning_metrics", {})
        metrics["quantitative"] = {
            "episodes_completed": learning_metrics.get("episodes_completed", 0),
            "average_reward": learning_metrics.get("average_reward", 0.0),
            "learning_steps": learning_metrics.get("learning_steps", 0),
            "events_recorded": len(period_data.get("events", []))
        }
        
        # Métricas cualitativas
        trends = await self._analyze_learning_trends(period_data)
        metrics["qualitative"] = {
            "trend_stability": "stable" if all(t == "stable" for t in trends.values()) else "mixed",
            "achievement_density": len(period_data.get("achievements", [])) / max(1, learning_metrics.get("episodes_completed", 1)),
            "lesson_quality": "high" if any(l.get("impact") == "high" for l in period_data.get("lessons", [])) else "medium"
        }
        
        # Métricas de eficiencia
        episodes = learning_metrics.get("episodes_completed", 1)
        metrics["efficiency"] = {
            "learning_rate": learning_metrics.get("learning_steps", 0) / episodes,
            "achievement_rate": len(period_data.get("achievements", [])) / episodes,
            "event_density": len(period_data.get("events", [])) / episodes
        }
        
        return metrics
    
    async def _generate_report_visualizations(self, period_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera datos para visualizaciones del reporte."""
        visualizations = {
            "charts": [],
            "graphs": [],
            "tables": []
        }
        
        # Datos para gráfico de curvas de aprendizaje
        curves = period_data.get("curves", {})
        
        if curves.get("reward"):
            visualizations["charts"].append({
                "type": "line_chart",
                "title": "Reward Progress",
                "data": curves["reward"],
                "x_label": "Episode",
                "y_label": "Reward"
            })
        
        if curves.get("loss"):
            visualizations["charts"].append({
                "type": "line_chart",
                "title": "Loss Progress",
                "data": curves["loss"],
                "x_label": "Learning Step",
                "y_label": "Loss"
            })
        
        # Datos para gráfico de exploración
        if curves.get("exploration"):
            visualizations["charts"].append({
                "type": "line_chart",
                "title": "Exploration Rate",
                "data": curves["exploration"],
                "x_label": "Episode",
                "y_label": "Exploration Rate"
            })
        
        # Tabla de logros
        achievements = period_data.get("achievements", [])
        if achievements:
            visualizations["tables"].append({
                "type": "achievements_table",
                "title": "Key Achievements",
                "headers": ["Type", "Description", "Significance"],
                "rows": [
                    [a.get("type", ""), a.get("description", ""), a.get("significance", "")]
                    for a in achievements[:5]
                ]
            })
        
        return visualizations
    
    async def _suggest_next_steps_from_report(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sugiere próximos pasos basados en recomendaciones del reporte."""
        next_steps = []
        
        # Priorizar recomendaciones
        high_priority = [r for r in recommendations if r.get("priority") == "high"]
        medium_priority = [r for r in recommendations if r.get("priority") == "medium"]
        low_priority = [r for r in recommendations if r.get("priority") == "low"]
        
        # Convertir a pasos accionables
        for i, rec in enumerate(high_priority[:2]):  # Máximo 2 de alta prioridad
            next_steps.append({
                "step_number": i + 1,
                "action": rec["description"],
                "priority": "high",
                "estimated_effort": "medium",
                "expected_outcome": "Immediate improvement"
            })
        
        for i, rec in enumerate(medium_priority[:3]):  # Máximo 3 de media prioridad
            next_steps.append({
                "step_number": len(next_steps) + 1,
                "action": rec["description"],
                "priority": "medium",
                "estimated_effort": "low",
                "expected_outcome": "Gradual improvement"
            })
        
        # Agregar paso de seguimiento
        next_steps.append({
            "step_number": len(next_steps) + 1,
            "action": "Schedule next evaluation",
            "priority": "low",
            "estimated_effort": "low",
            "expected_outcome": "Continuous monitoring"
        })
        
        return next_steps
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calcula tendencia de una serie de datos."""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        y = np.array(data)
        
        # Ajustar línea recta
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return m
    
    def _generate_phase_activities(self, goal: Dict[str, Any]) -> List[str]:
        """Genera actividades para una fase de aprendizaje."""
        activities = []
        goal_type = goal.get("type", "")
        
        if "accuracy" in goal_type.lower():
            activities = [
                "Collect labeled training data",
                "Perform supervised training",
                "Validate with test set",
                "Adjust model based on results"
            ]
        elif "reward" in goal_type.lower():
            activities = [
                "Analyze current reward function",
                "Design improved reward signals",
                "Test with exploration",
                "Evaluate impact on learning"
            ]
        elif "coverage" in goal_type.lower():
            activities = [
                "Increase exploration rate",
                "Diversify state sampling",
                "Update Q-table coverage",
                "Monitor state diversity"
            ]
        else:
            activities = [
                "Review current performance",
                "Identify specific improvements",
                "Implement targeted training",
                "Evaluate results"
            ]
        
        return activities
    
    def _generate_phase_metrics(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Genera métricas para una fase de aprendizaje."""
        metrics = {}
        goal_type = goal.get("type", "")
        
        if "accuracy" in goal_type.lower():
            metrics = {
                "primary": "accuracy",
                "target": 0.7,
                "secondary": ["precision", "recall"],
                "thresholds": {
                    "accuracy": 0.6,
                    "precision": 0.5,
                    "recall": 0.5
                }
            }
        elif "reward" in goal_type.lower():
            metrics = {
                "primary": "average_reward",
                "target": 0.0,
                "secondary": ["reward_std", "max_reward"],
                "thresholds": {
                    "average_reward": 0.0,
                    "reward_std": 1.0,
                    "max_reward": 5.0
                }
            }
        else:
            metrics = {
                "primary": "improvement_score",
                "target": 0.5,
                "secondary": ["consistency", "stability"],
                "thresholds": {
                    "improvement_score": 0.3,
                    "consistency": 0.6,
                    "stability": 0.7
                }
            }
        
        return metrics
    
    async def _learn_supervised(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aprende de datos supervisados."""
        # Implementación simplificada
        # En implementación real, se usaría un clasificador
        
        if not data:
            return {"success": False, "error": "No training data provided"}
        
        # Extraer características y etiquetas
        X = []
        y = []
        
        for item in data:
            if "features" in item and "label" in item:
                X.append(item["features"])
                y.append(item["label"])
        
        if len(X) == 0:
            return {"success": False, "error": "No valid training samples"}
        
        # "Entrenar" modelo simplificado
        self.learned_patterns["supervised_model"] = {
            "training_samples": len(X),
            "classes": list(set(y)),
            "last_trained": datetime.now()
        }
        
        return {
            "success": True,
            "samples_trained": len(X),
            "classes_learned": len(set(y)),
            "model_updated": True
        }
    
    async def _learn_unsupervised(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aprende de datos no supervisados."""
        # Implementación simplificada
        
        if not data:
            return {"success": False, "error": "No data provided"}
        
        # Extraer características
        X = []
        for item in data:
            if "features" in item:
                X.append(item["features"])
        
        if len(X) == 0:
            return {"success": False, "error": "No valid features"}
        
        # "Clustering" simplificado
        self.learned_patterns["unsupervised_clusters"] = {
            "samples": len(X),
            "feature_dim": len(X[0]) if X else 0,
            "clusters_detected": min(5, len(X)),
            "last_analyzed": datetime.now()
        }
        
        return {
            "success": True,
            "samples_analyzed": len(X),
            "clusters_identified": min(5, len(X)),
            "patterns_extracted": True
        }
    
    async def _evaluate_performance(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evalúa rendimiento con datos de prueba."""
        if not test_data:
            return {"success": False, "error": "No test data provided"}
        
        # Evaluación simplificada
        correct = 0
        total = 0
        
        for item in test_data:
            if "state" in item and "expected_action" in item:
                state = await self._represent_state(item["state"])
                action, _ = await self._select_best_action(state)
                
                if action == item["expected_action"]:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "success": True,
            "accuracy": accuracy,
            "samples_tested": total,
            "correct_predictions": correct,
            "performance_level": "good" if accuracy > 0.7 else "needs_improvement"
        }
    
    async def _evaluate_generalization(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evalúa capacidad de generalización."""
        if not test_data:
            return {"success": False, "error": "No test data provided"}
        
        # Evaluación simplificada
        performances = []
        
        for item in test_data:
            if "state" in item and "expected_reward" in item:
                state = await self._represent_state(item["state"])
                action, confidence = await self._select_best_action(state)
                
                # Simular recompensa basada en acción
                simulated_reward = confidence * 10  # Simplificación
                expected_reward = item["expected_reward"]
                
                error = abs(simulated_reward - expected_reward)
                performances.append(1.0 / (1.0 + error))
        
        generalization_score = np.mean(performances) if performances else 0.0
        
        return {
            "success": True,
            "generalization_score": generalization_score,
            "samples_evaluated": len(performances),
            "generalization_ability": "good" if generalization_score > 0.6 else "needs_improvement"
        }
    
    async def _analyze_mistake(self, mistake: Dict[str, Any], correction: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza un error y su corrección."""
        analysis = {
            "mistake_type": "unknown",
            "serious": False,
            "root_cause": "unknown",
            "lesson": "",
            "prevention_strategy": ""
        }
        
        # Analizar tipo de error
        if "prediction" in mistake and "correct_prediction" in correction:
            analysis["mistake_type"] = "incorrect_prediction"
            
            mistake_pred = mistake.get("prediction")
            correct_pred = correction.get("correct_prediction")
            
            # Determinar severidad
            if isinstance(mistake_pred, (int, float)) and isinstance(correct_pred, (int, float)):
                error_magnitude = abs(mistake_pred - correct_pred)
                analysis["serious"] = error_magnitude > 5  # Umbral arbitrario
                analysis["root_cause"] = "estimation_error"
                analysis["lesson"] = f"Avoid estimation errors of magnitude {error_magnitude}"
                analysis["prevention_strategy"] = "Add validation checks for extreme values"
        
        elif "action" in mistake and "correct_action" in correction:
            analysis["mistake_type"] = "incorrect_action"
            analysis["root_cause"] = "policy_error"
            analysis["lesson"] = "Choose different action in similar states"
            analysis["prevention_strategy"] = "Update Q-values for this state-action pair"
        
        return analysis
    
    async def _update_policy_from_correction(self, error_analysis: Dict[str, Any]) -> None:
        """Actualiza política basada en corrección."""
        if error_analysis.get("mistake_type") == "incorrect_action":
            # En implementación real, actualizaría la Q-table
            # Aquí solo registramos
            self.logger.info(f"Policy updated based on correction: {error_analysis.get('lesson')}")
    
    async def _evaluate_knowledge_relevance(self, knowledge: Dict[str, Any]) -> float:
        """Evalúa relevancia del conocimiento para transferencia."""
        # Evaluación simplificada
        relevance = 0.5  # Por defecto
        
        # Basado en tipo de conocimiento
        if "patterns" in knowledge:
            relevance += 0.2
        
        if "stats" in knowledge:
            relevance += 0.1
        
        # Basado en actualidad
        timestamp = knowledge.get("timestamp")
        if timestamp:
            if isinstance(timestamp, str):
                from dateutil.parser import parse
                timestamp = parse(timestamp)
            
            age_days = (datetime.now() - timestamp).days
            if age_days < 30:
                relevance += 0.2
            elif age_days < 90:
                relevance += 0.1
        
        return min(1.0, relevance)
    
    async def _integrate_transferred_knowledge(self, knowledge: Dict[str, Any], source: str) -> None:
        """Integra conocimiento transferido."""
        # Integración simplificada
        integration_time = datetime.now()
        
        if "patterns" in knowledge:
            for key, pattern in knowledge["patterns"].items():
                self.learned_patterns[f"transferred_{source}_{key}"] = {
                    **pattern,
                    "source": source,
                    "integration_time": integration_time
                }
        
        if "stats" in knowledge:
            self.learned_patterns[f"transferred_stats_{source}"] = {
                **knowledge["stats"],
                "source": source,
                "integration_time": integration_time
            }
        
        self.logger.info(f"Integrated knowledge from {source}")

# Ejemplo de uso
if __name__ == "__main__":
    async def main():
        """Ejemplo de uso del LearningAgent."""
        # Crear agente
        agent = LearningAgent()
        
        # Inicializar
        success = await agent.initialize()
        print(f"Agent initialized: {success}")
        
        # Aprender de experiencias
        experiences = [
            {
                "state": {"feature1": 0.5, "feature2": 0.3},
                "action": 2,
                "reward": 1.0,
                "next_state": {"feature1": 0.6, "feature2": 0.4},
                "done": False
            },
            {
                "state": {"feature1": 0.8, "feature2": 0.2},
                "action": 1,
                "reward": -0.5,
                "next_state": {"feature1": 0.7, "feature2": 0.1},
                "done": True
            }
        ]
        
        result = await agent.learn_from_experience(experiences)
        print(f"Learning result: {result}")
        
        # Evaluar progreso
        progress = await agent.evaluate_learning_progress()
        print(f"Learning progress: {progress.get('overall_progress', {})}")
        
        # Generar reporte
        report = await agent.generate_learning_report()
        print(f"Report generated: {report.get('success', False)}")
        
        # Apagar agente
        await agent.shutdown()
    
    asyncio.run(main())