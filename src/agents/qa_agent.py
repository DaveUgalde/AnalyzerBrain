"""
QuestionAnsweringAgent - Agente especializado en responder preguntas sobre código.
Implementa procesamiento de lenguaje natural, recuperación de contexto y generación de respuestas.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from datetime import datetime
import re

from ..core.exceptions import AgentException, ValidationError
from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentCapability, AgentConfig
from ..embeddings.semantic_search import SemanticSearch
from ..graph.graph_query_engine import GraphQueryEngine
from ..memory.memory_retriever import MemoryRetriever

class QuestionType(Enum):
    """Tipos de preguntas que puede procesar el agente."""
    CODE_LOCATION = "code_location"         # Dónde está definido X
    CODE_EXPLANATION = "code_explanation"   # Qué hace X
    CODE_RELATIONSHIP = "code_relationship" # Cómo se relaciona X con Y
    CODE_EXAMPLE = "code_example"          # Ejemplo de uso de X
    CODE_ISSUE = "code_issue"              # Problema con X
    CODE_REFACTOR = "code_refactor"        # Cómo mejorar X
    SYSTEM_QUERY = "system_query"          # Preguntas sobre el sistema
    GENERAL = "general"                    # Preguntas generales

@dataclass
class QAAgentConfig(AgentConfig):
    """Configuración específica del QuestionAnsweringAgent."""
    max_context_tokens: int = 4000
    min_confidence_threshold: float = 0.3
    enable_code_snippets: bool = True
    enable_explanations: bool = True
    enable_followups: bool = True
    max_followup_depth: int = 3
    supported_question_types: List[QuestionType] = field(default_factory=lambda: list(QuestionType))
    fallback_to_general: bool = True
    
class QuestionAnalysis(BaseModel):
    """Análisis de una pregunta."""
    question: str
    question_type: QuestionType
    confidence: float
    entities: List[Dict[str, Any]] = []
    intent: str
    sub_questions: List[str] = []
    requires_code_context: bool = False
    
class AnswerContext(BaseModel):
    """Contexto para generar una respuesta."""
    relevant_code: List[Dict[str, Any]] = []
    related_functions: List[Dict[str, Any]] = []
    documentation: List[Dict[str, Any]] = []
    previous_answers: List[Dict[str, Any]] = []
    project_context: Dict[str, Any] = {}
    
class QuestionAnsweringAgent(BaseAgent):
    """
    Agente especializado en responder preguntas sobre código y proyectos.
    
    Características:
    1. Clasificación de tipos de preguntas
    2. Recuperación de contexto relevante
    3. Generación de respuestas con explicaciones
    4. Sugerencias de preguntas de seguimiento
    5. Aprendizaje de respuestas efectivas
    """
    
    def __init__(self, config: Optional[QAAgentConfig] = None):
        """Inicializa el QuestionAnsweringAgent."""
        if config is None:
            config = QAAgentConfig(
                name="QuestionAnsweringAgent",
                description="Agente especializado en responder preguntas sobre código",
                capabilities=[
                    AgentCapability.QUESTION_ANSWERING,
                    AgentCapability.CODE_ANALYSIS
                ],
                confidence_threshold=0.7,
                max_processing_time=30
            )
        
        super().__init__(config)
        self.semantic_search: Optional[SemanticSearch] = None
        self.graph_query: Optional[GraphQueryEngine] = None
        self.memory_retriever: Optional[MemoryRetriever] = None
        self.question_history: List[Dict[str, Any]] = []
        
    async def _initialize_internal(self) -> bool:
        """Inicialización específica del QuestionAnsweringAgent."""
        try:
            # Obtener dependencias
            self.semantic_search = self.dependencies.get('semantic_search')
            self.graph_query = self.dependencies.get('graph_query_engine')
            self.memory_retriever = self.dependencies.get('memory_retriever')
            
            # Verificar dependencias críticas
            if not self.semantic_search:
                self.logger.warning("SemanticSearch not available, some features will be limited")
            
            # Cargar modelos de NLP si es necesario
            await self._load_nlp_models()
            
            # Cargar historial de preguntas frecuentes
            await self._load_frequent_questions()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize QAAgent: {e}")
            return False
    
    async def _process_internal(self, input_data: AgentInput) -> AgentOutput:
        """Procesa una pregunta y genera una respuesta."""
        try:
            # Extraer pregunta
            question = input_data.data.get("question", "")
            if not question:
                raise ValidationError("Question is required")
            
            # Analizar la pregunta
            question_analysis = await self._analyze_question(question)
            
            # Recuperar contexto relevante
            context = await self._retrieve_relevant_context(question_analysis, input_data.context)
            
            # Generar respuesta
            answer = await self._generate_answer(question_analysis, context)
            
            # Sugerir preguntas de seguimiento
            followups = await self._suggest_followup_questions(question_analysis, answer)
            
            # Validar la respuesta
            confidence = await self._validate_answer(question_analysis, answer)
            
            # Almacenar en historial
            await self._store_question_answer(question, answer, confidence, input_data.context)
            
            return AgentOutput(
                request_id=input_data.request_id,
                agent_id=self.config.agent_id,
                success=True,
                data={
                    "answer": answer,
                    "context": context.dict() if hasattr(context, 'dict') else context,
                    "question_analysis": question_analysis.dict() if hasattr(question_analysis, 'dict') else question_analysis,
                    "followup_suggestions": followups,
                    "confidence_breakdown": {
                        "question_understanding": question_analysis.confidence,
                        "context_relevance": context.get("relevance_score", 0.7),
                        "answer_quality": confidence
                    }
                },
                confidence=confidence,
                reasoning=[
                    f"Question classified as: {question_analysis.question_type.value}",
                    f"Found {len(context.get('relevant_code', []))} relevant code snippets",
                    f"Generated answer with {confidence:.2f} confidence"
                ]
            )
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return self._create_error_output(input_data.request_id, str(e))
    
    async def _learn_internal(self, feedback: Dict[str, Any]) -> bool:
        """Aprende del feedback de respuestas."""
        try:
            feedback_type = feedback.get("type", "general")
            
            if feedback_type == "answer_correction":
                return await self._learn_from_correction(feedback)
            elif feedback_type == "answer_rating":
                return await self._learn_from_rating(feedback)
            elif feedback_type == "conversation_feedback":
                return await self._learn_from_conversation(feedback)
            else:
                self.logger.warning(f"Unknown feedback type: {feedback_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in learning: {e}")
            return False
    
    def _validate_input_specific(self, input_data: AgentInput) -> None:
        """Validación específica para QuestionAnsweringAgent."""
        if "question" not in input_data.data:
            raise ValidationError("Input must contain 'question' field")
        
        question = input_data.data.get("question", "")
        if len(question.strip()) < 3:
            raise ValidationError("Question must be at least 3 characters")
        
        if len(question) > 1000:
            raise ValidationError("Question too long (max 1000 characters)")
    
    async def _save_state(self) -> None:
        """Guarda el estado del agente."""
        try:
            # Guardar historial de preguntas
            state_data = {
                "question_history": self.question_history[-100:],  # Últimas 100 preguntas
                "frequent_patterns": await self._extract_frequent_patterns(),
                "learning_updates": self.metrics.get("learning_events", [])
            }
            
            # En implementación real, guardaríamos a un almacenamiento persistente
            self.logger.info(f"Saving QAAgent state with {len(self.question_history)} questions")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    # ===== FUNCIONES PÚBLICAS ESPECÍFICAS =====
    
    async def answer_question(self, question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Responde una pregunta sobre el código.
        
        Args:
            question: Pregunta en lenguaje natural
            context: Contexto adicional (proyecto, archivo actual, etc.)
            
        Returns:
            Dict con respuesta estructurada
        """
        input_data = AgentInput(
            data={"question": question},
            context=context or {},
            priority=1
        )
        
        output = await self.process(input_data)
        
        if output.success:
            return output.data
        else:
            return {"error": output.errors, "answer": "No pude procesar tu pregunta."}
    
    async def clarify_question(self, question: str, 
                             clarification_options: List[str]) -> Dict[str, Any]:
        """
        Solicita aclaración cuando una pregunta es ambigua.
        
        Args:
            question: Pregunta ambigua
            clarification_options: Opciones para aclarar
            
        Returns:
            Dict con solicitud de aclaración
        """
        analysis = await self._analyze_question(question)
        
        if analysis.confidence < self.config.min_confidence_threshold:
            return {
                "needs_clarification": True,
                "original_question": question,
                "possible_interpretations": clarification_options,
                "suggested_clarification": await self._suggest_clarification(analysis, clarification_options)
            }
        
        return {"needs_clarification": False}
    
    async def provide_explanations(self, code_element: str, 
                                 explanation_type: str = "detailed") -> Dict[str, Any]:
        """
        Proporciona explicaciones sobre un elemento de código.
        
        Args:
            code_element: Nombre del elemento (función, clase, variable)
            explanation_type: Tipo de explicación (brief, detailed, examples)
            
        Returns:
            Dict con explicaciones
        """
        # Buscar el elemento en el conocimiento
        search_results = await self._search_code_element(code_element)
        
        if not search_results:
            return {"found": False, "element": code_element}
        
        explanations = []
        
        for result in search_results[:3]:  # Top 3 resultados
            explanation = await self._generate_explanation(result, explanation_type)
            explanations.append(explanation)
        
        return {
            "found": True,
            "element": code_element,
            "explanations": explanations,
            "summary": await self._summarize_explanations(explanations)
        }
    
    async def suggest_related_questions(self, question: str, 
                                      limit: int = 5) -> List[str]:
        """
        Sugiere preguntas relacionadas.
        
        Args:
            question: Pregunta original
            limit: Número máximo de sugerencias
            
        Returns:
            Lista de preguntas relacionadas
        """
        analysis = await self._analyze_question(question)
        
        related = []
        
        # 1. Basado en entidades extraídas
        for entity in analysis.entities:
            entity_questions = await self._generate_entity_questions(entity)
            related.extend(entity_questions)
        
        # 2. Basado en tipo de pregunta
        type_questions = await self._get_question_type_suggestions(analysis.question_type)
        related.extend(type_questions)
        
        # 3. Basado en historial
        history_questions = await self._get_historical_suggestions(question)
        related.extend(history_questions)
        
        # Eliminar duplicados y la pregunta original
        unique_questions = []
        seen = set([question.lower()])
        
        for q in related:
            q_lower = q.lower()
            if q_lower not in seen and len(unique_questions) < limit:
                unique_questions.append(q)
                seen.add(q_lower)
        
        return unique_questions
    
    async def validate_answer(self, question: str, 
                            answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida una respuesta generada.
        
        Args:
            question: Pregunta original
            answer: Respuesta a validar
            
        Returns:
            Dict con resultados de validación
        """
        validation_results = {
            "completeness": await self._validate_completeness(question, answer),
            "accuracy": await self._validate_accuracy(question, answer),
            "relevance": await self._validate_relevance(question, answer),
            "clarity": await self._validate_clarity(answer),
            "actionability": await self._validate_actionability(answer)
        }
        
        # Calcular puntuación general
        scores = [v.get("score", 0) for v in validation_results.values() 
                 if isinstance(v, dict) and "score" in v]
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "overall_score": overall_score,
            "validation_results": validation_results,
            "suggestions": await self._generate_validation_suggestions(validation_results),
            "is_valid": overall_score >= 0.7  # Umbral de validez
        }
    
    async def learn_from_feedback(self, question: str, 
                                answer: Dict[str, Any],
                                feedback: Dict[str, Any]) -> bool:
        """
        Aprende del feedback sobre una respuesta.
        
        Args:
            question: Pregunta original
            answer: Respuesta dada
            feedback: Feedback del usuario
            
        Returns:
            bool: True si el aprendizaje fue exitoso
        """
        learning_event = {
            "timestamp": datetime.now(),
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "agent_version": self.config.version
        }
        
        # Almacenar en memoria
        self.memory.store(
            AgentMemoryType.EPISODIC,
            learning_event
        )
        
        # Actualizar patrones de aprendizaje
        await self._update_learning_patterns(question, answer, feedback)
        
        # Ajustar confianza si es necesario
        if "confidence_impact" in feedback:
            self.config.confidence_threshold = max(
                0.1,
                min(0.95, self.config.confidence_threshold + feedback["confidence_impact"])
            )
        
        return True
    
    async def improve_answer_quality(self) -> Dict[str, Any]:
        """
        Analiza y mejora la calidad general de las respuestas.
        
        Returns:
            Dict con análisis de calidad y mejoras
        """
        quality_metrics = await self._calculate_quality_metrics()
        
        improvements = []
        
        # Identificar áreas de mejora
        if quality_metrics.get("avg_confidence", 0) < 0.7:
            improvements.append({
                "area": "confidence",
                "suggestion": "Aumentar umbral de confianza mínima",
                "action": await self._improve_confidence_strategy()
            })
        
        if quality_metrics.get("clarity_score", 0) < 0.8:
            improvements.append({
                "area": "clarity",
                "suggestion": "Mejorar claridad de explicaciones",
                "action": await self._improve_clarity_strategy()
            })
        
        if quality_metrics.get("completeness_score", 0) < 0.8:
            improvements.append({
                "area": "completeness",
                "suggestion": "Proporcionar respuestas más completas",
                "action": await self._improve_completeness_strategy()
            })
        
        return {
            "quality_metrics": quality_metrics,
            "improvements_needed": improvements,
            "current_strategies": await self._get_current_strategies(),
            "recommendations": await self._generate_quality_recommendations(quality_metrics)
        }
    
    # ===== FUNCIONES PRIVADAS DE IMPLEMENTACIÓN =====
    
    async def _analyze_question(self, question: str) -> QuestionAnalysis:
        """Analiza una pregunta para determinar tipo y entidades."""
        # Implementación simplificada - en producción usaríamos modelos de NLP
        
        question_lower = question.lower()
        
        # Detectar tipo de pregunta
        question_type = QuestionType.GENERAL
        confidence = 0.5
        
        # Patrones para detección de tipo
        patterns = {
            QuestionType.CODE_LOCATION: [
                r"dónde está.*definid[ao]",
                r"en qué archivo.*",
                r"ubicación de.*",
                r"where is.*defined",
                r"location of.*"
            ],
            QuestionType.CODE_EXPLANATION: [
                r"qué hace.*",
                r"cómo funciona.*",
                r"explica.*",
                r"what does.*do",
                r"how does.*work"
            ],
            QuestionType.CODE_EXAMPLE: [
                r"ejemplo de.*",
                r"muestra.*código",
                r"cómo usar.*",
                r"example of.*",
                r"how to use.*"
            ]
        }
        
        for qtype, regex_patterns in patterns.items():
            for pattern in regex_patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    question_type = qtype
                    confidence = 0.8
                    break
        
        # Extraer entidades (nombres de funciones, clases, variables)
        entities = await self._extract_entities_from_question(question)
        
        # Determinar intención
        intent = await self._determine_intent(question, question_type)
        
        return QuestionAnalysis(
            question=question,
            question_type=question_type,
            confidence=confidence,
            entities=entities,
            intent=intent,
            requires_code_context=question_type != QuestionType.GENERAL
        )
    
    async def _retrieve_relevant_context(self, 
                                        analysis: QuestionAnalysis,
                                        user_context: Optional[Dict]) -> AnswerContext:
        """Recupera contexto relevante para responder la pregunta."""
        context = AnswerContext()
        
        if not analysis.requires_code_context:
            return context
        
        # 1. Buscar código relevante
        for entity in analysis.entities:
            if entity.get("type") in ["function", "class", "variable"]:
                code_results = await self._search_code_by_entity(entity)
                context.relevant_code.extend(code_results)
        
        # 2. Buscar en grafo de conocimiento
        if self.graph_query and analysis.entities:
            graph_results = await self._query_knowledge_graph(analysis.entities)
            context.related_functions.extend(graph_results)
        
        # 3. Buscar documentación
        doc_results = await self._search_documentation(analysis.question)
        context.documentation.extend(doc_results)
        
        # 4. Buscar respuestas anteriores similares
        if self.memory_retriever:
            memory_results = await self.memory_retriever.retrieve_by_similarity(
                query={"question": analysis.question},
                limit=5
            )
            context.previous_answers.extend(memory_results)
        
        # 5. Añadir contexto del proyecto
        if user_context and "project_id" in user_context:
            project_info = await self._get_project_context(user_context["project_id"])
            context.project_context = project_info
        
        return context
    
    async def _generate_answer(self, 
                             analysis: QuestionAnalysis,
                             context: AnswerContext) -> Dict[str, Any]:
        """Genera una respuesta basada en el análisis y contexto."""
        
        # Plantillas de respuesta por tipo de pregunta
        templates = {
            QuestionType.CODE_LOCATION: self._generate_location_answer,
            QuestionType.CODE_EXPLANATION: self._generate_explanation_answer,
            QuestionType.CODE_EXAMPLE: self._generate_example_answer,
            QuestionType.CODE_ISSUE: self._generate_issue_answer,
            QuestionType.CODE_REFACTOR: self._generate_refactor_answer
        }
        
        generator = templates.get(analysis.question_type, self._generate_general_answer)
        
        return await generator(analysis, context)
    
    async def _suggest_followup_questions(self,
                                        analysis: QuestionAnalysis,
                                        answer: Dict[str, Any]) -> List[str]:
        """Sugiere preguntas de seguimiento relacionadas."""
        followups = []
        
        # Basado en tipo de pregunta
        if analysis.question_type == QuestionType.CODE_EXPLANATION:
            followups.extend([
                f"¿Cómo se usa {analysis.entities[0]['name'] if analysis.entities else 'esto'}?",
                f"¿Qué parámetros acepta {analysis.entities[0]['name'] if analysis.entities else 'esta función'}?",
                f"¿Hay alternativas a {analysis.entities[0]['name'] if analysis.entities else 'este enfoque'}?"
            ])
        
        # Basado en entidades mencionadas
        for entity in analysis.entities:
            if entity.get("type") == "function":
                followups.append(f"¿Qué otras funciones llaman a {entity['name']}?")
                followups.append(f"¿Qué funciones llama {entity['name']}?")
        
        return followups[:self.config.max_followup_depth]
    
    async def _validate_answer(self,
                             analysis: QuestionAnalysis,
                             answer: Dict[str, Any]) -> float:
        """Valida la calidad de una respuesta generada."""
        
        scores = []
        
        # 1. Relevancia
        relevance = await self._calculate_relevance(analysis, answer)
        scores.append(relevance)
        
        # 2. Completitud
        completeness = await self._calculate_completeness(analysis, answer)
        scores.append(completeness)
        
        # 3. Precisión (si tenemos contexto para verificar)
        if analysis.entities and answer.get("code_examples"):
            accuracy = await self._calculate_accuracy(answer)
            scores.append(accuracy)
        
        # 4. Claridad
        clarity = await self._calculate_clarity(answer)
        scores.append(clarity)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    async def _learn_from_correction(self, feedback: Dict) -> bool:
        """Aprende de correcciones a respuestas."""
        correction = feedback.get("correction", {})
        original_question = correction.get("original_question")
        corrected_answer = correction.get("corrected_answer")
        
        if not original_question or not corrected_answer:
            return False
        
        # Actualizar historial
        self.question_history.append({
            "question": original_question,
            "answer": corrected_answer,
            "type": "corrected",
            "timestamp": datetime.now()
        })
        
        # Ajustar patrones de aprendizaje
        await self._adjust_patterns_from_correction(original_question, corrected_answer)
        
        return True
    
    async def _load_nlp_models(self) -> None:
        """Carga modelos de procesamiento de lenguaje natural."""
        # En implementación real, cargaríamos modelos como spaCy, NLTK, etc.
        self.logger.info("Loading NLP models...")
        
        # Modelos simplificados para ejemplo
        self.nlp_models = {
            "entity_extractor": self._simple_entity_extractor,
            "intent_classifier": self._simple_intent_classifier,
            "sentiment_analyzer": self._simple_sentiment_analyzer
        }
    
    async def _load_frequent_questions(self) -> None:
        """Carga preguntas frecuentes de almacenamiento persistente."""
        # En implementación real, cargaríamos de base de datos
        self.frequent_questions = [
            "¿Dónde está definida la función main?",
            "¿Qué hace este código?",
            "¿Cómo puedo usar esta API?",
            "¿Hay algún ejemplo de esto?",
            "¿Qué errores puede tener este código?"
        ]
    
    async def _extract_entities_from_question(self, question: str) -> List[Dict[str, Any]]:
        """Extrae entidades (nombres de código) de una pregunta."""
        entities = []
        
        # Patrones para extraer nombres de funciones, clases, variables
        patterns = [
            r'\b([A-Z][a-zA-Z0-9_]*)\b',  # Clases (CamelCase)
            r'\b([a-z][a-zA-Z0-9_]*)\b',  # Funciones/variables (snake_case)
            r'`([^`]+)`',                 # Código entre backticks
            r'"([^"]+)"',                 # Cadenas entre comillas
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, question)
            for match in matches:
                entity_name = match.group(1)
                
                # Determinar tipo basado en convenciones de nomenclatura
                if entity_name[0].isupper():
                    entity_type = "class"
                elif '_' in entity_name:
                    entity_type = "function" if '()' in question else "variable"
                else:
                    entity_type = "unknown"
                
                entities.append({
                    "name": entity_name,
                    "type": entity_type,
                    "position": match.start(),
                    "confidence": 0.7
                })
        
        return entities
    
    async def _determine_intent(self, question: str, question_type: QuestionType) -> str:
        """Determina la intención detrás de una pregunta."""
        question_lower = question.lower()
        
        intents = {
            "understand": ["qué", "cómo", "por qué", "explica", "comprender"],
            "locate": ["dónde", "ubicación", "encontrar", "buscar"],
            "use": ["usar", "utilizar", "ejemplo", "implementar"],
            "debug": ["error", "problema", "bug", "falla", "no funciona"],
            "improve": ["mejorar", "optimizar", "refactorizar", "limpiar"]
        }
        
        for intent, keywords in intents.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return intent
        
        return "general"
    
    async def _search_code_by_entity(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Busca código relacionado con una entidad."""
        if not self.semantic_search:
            return []
        
        try:
            results = await self.semantic_search.search(
                query=entity["name"],
                filters={"type": entity["type"]},
                limit=5
            )
            return results
        except Exception as e:
            self.logger.error(f"Error searching code: {e}")
            return []
    
    async def _generate_location_answer(self, 
                                      analysis: QuestionAnalysis,
                                      context: AnswerContext) -> Dict[str, Any]:
        """Genera respuesta para preguntas de ubicación."""
        
        if not context.relevant_code:
            return {
                "found": False,
                "message": f"No encontré información sobre {analysis.entities[0]['name'] if analysis.entities else 'eso'}",
                "suggestions": ["Verifica el nombre", "Busca en otros archivos"]
            }
        
        first_result = context.relevant_code[0]
        
        return {
            "found": True,
            "element": first_result.get("name", ""),
            "location": {
                "file": first_result.get("file_path", ""),
                "line": first_result.get("line_number", 0),
                "line_range": [first_result.get("start_line", 0), first_result.get("end_line", 0)]
            },
            "type": first_result.get("type", "unknown"),
            "signature": first_result.get("signature", ""),
            "code_snippet": first_result.get("code_snippet", ""),
            "additional_locations": context.relevant_code[1:] if len(context.relevant_code) > 1 else []
        }
    
    async def _generate_explanation_answer(self,
                                         analysis: QuestionAnalysis,
                                         context: AnswerContext) -> Dict[str, Any]:
        """Genera respuesta para preguntas de explicación."""
        
        if not context.relevant_code:
            return {
                "explanation": f"No tengo información específica sobre {analysis.entities[0]['name'] if analysis.entities else 'eso'}",
                "confidence": 0.3
            }
        
        element = context.relevant_code[0]
        
        explanation = f"La {'función' if element.get('type') == 'function' else 'clase' if element.get('type') == 'class' else 'elemento'} "
        explanation += f"`{element.get('name', '')}` "
        
        if element.get("docstring"):
            explanation += f"tiene la siguiente documentación:\n\n{element['docstring']}"
        elif element.get("signature"):
            explanation += f"tiene la firma: `{element['signature']}`"
            
            if element.get("parameters"):
                explanation += "\n\nParámetros:\n"
                for param in element["parameters"]:
                    explanation += f"- `{param.get('name', '')}`"
                    if param.get("type"):
                        explanation += f": {param['type']}"
                    if param.get("default"):
                        explanation += f" (default: {param['default']})"
                    explanation += "\n"
        
        if context.documentation:
            explanation += "\n\nDocumentación adicional:\n"
            for doc in context.documentation[:2]:
                explanation += f"- {doc.get('content', '')[:200]}...\n"
        
        return {
            "explanation": explanation,
            "element_info": element,
            "related_elements": context.related_functions[:3],
            "confidence": 0.8
        }
    
    async def _generate_general_answer(self,
                                     analysis: QuestionAnalysis,
                                     context: AnswerContext) -> Dict[str, Any]:
        """Genera respuesta para preguntas generales."""
        return {
            "answer": "Esta es una respuesta general. Para preguntas más específicas sobre código, intenta mencionar nombres de funciones, clases o archivos específicos.",
            "suggestions": [
                "Menciona el nombre específico de lo que quieres saber",
                "Pregunta sobre ubicación, funcionalidad o ejemplos",
                "Proporciona contexto sobre el proyecto"
            ],
            "confidence": 0.5
        }
    
    async def _create_error_output(self, request_id: str, error: str) -> AgentOutput:
        """Crea una salida de error estandarizada."""
        return AgentOutput(
            request_id=request_id,
            agent_id=self.config.agent_id,
            success=False,
            confidence=0.0,
            errors=[error],
            warnings=["Error procesando la pregunta"],
            data={
                "fallback_answer": "Lo siento, hubo un error procesando tu pregunta. Por favor, intenta reformularla.",
                "error_details": error
            }
        )
    
    # Métodos auxiliares simplificados
    async def _simple_entity_extractor(self, text: str) -> List[Dict[str, Any]]:
        return []
    
    async def _simple_intent_classifier(self, text: str) -> str:
        return "general"
    
    async def _simple_sentiment_analyzer(self, text: str) -> float:
        return 0.5
    
    async def _store_question_answer(self, question: str, answer: Dict, 
                                   confidence: float, context: Dict) -> None:
        """Almacena pregunta y respuesta en historial."""
        self.question_history.append({
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "context": context,
            "timestamp": datetime.now(),
            "agent_version": self.config.version
        })
        
        # Mantener tamaño del historial
        if len(self.question_history) > 1000:
            self.question_history = self.question_history[-1000:]

# Ejemplo de uso
if __name__ == "__main__":
    async def main():
        agent = QuestionAnsweringAgent()
        
        # Inicializar
        await agent.initialize()
        
        # Ejemplo de pregunta
        result = await agent.answer_question(
            "¿Dónde está definida la función process_data?",
            context={"project_id": "proj_123"}
        )
        
        print(f"Respuesta: {result.get('answer', {}).get('explanation', 'No answer')}")
        
        # Sugerir preguntas relacionadas
        related = await agent.suggest_related_questions(
            "¿Qué hace la función calculate_total?"
        )
        print(f"Preguntas relacionadas: {related}")
        
        await agent.shutdown()
    
    asyncio.run(main())