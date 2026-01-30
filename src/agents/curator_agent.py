"""
CuratorAgent - Agente especializado en curación y organización del conocimiento.
Responsable de validar, organizar, enriquecer y mantener la calidad del conocimiento.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import hashlib

from ..core.exceptions import AgentException, ValidationError
from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentCapability, AgentConfig
from ..graph.knowledge_graph import KnowledgeGraph
from ..embeddings.embedding_generator import EmbeddingGenerator

class KnowledgeQualityLevel(Enum):
    """Niveles de calidad del conocimiento."""
    LOW = "low"          # No verificado, baja confianza
    MEDIUM = "medium"    # Parcialmente verificado
    HIGH = "high"        # Verificado y validado
    EXPERT = "expert"    # Validado por expertos

class KnowledgeType(Enum):
    """Tipos de conocimiento que puede curar el agente."""
    CODE_KNOWLEDGE = "code_knowledge"      # Conocimiento de código
    ARCHITECTURE = "architecture"          # Conocimiento arquitectónico
    DOMAIN = "domain"                      # Conocimiento del dominio
    BEST_PRACTICES = "best_practices"      # Mejores prácticas
    PATTERNS = "patterns"                  # Patrones de diseño
    ISSUES = "issues"                      # Problemas conocidos
    SOLUTIONS = "solutions"                # Soluciones validadas

@dataclass
class CuratorConfig(AgentConfig):
    """Configuración específica del CuratorAgent."""
    quality_thresholds: Dict[KnowledgeQualityLevel, float] = field(
        default_factory=lambda: {
            KnowledgeQualityLevel.LOW: 0.3,
            KnowledgeQualityLevel.MEDIUM: 0.6,
            KnowledgeQualityLevel.HIGH: 0.8,
            KnowledgeQualityLevel.EXPERT: 0.95
        }
    )
    max_knowledge_items: int = 10000
    pruning_strategy: str = "relevance_based"  # relevance_based, time_based, hybrid
    min_relevance_score: float = 0.3
    auto_curation: bool = True
    curation_intervals: Dict[str, int] = field(
        default_factory=lambda: {
            "quick": 300,      # 5 minutos
            "standard": 3600,  # 1 hora
            "deep": 86400      # 1 día
        }
    )
    knowledge_types: List[KnowledgeType] = field(
        default_factory=lambda: list(KnowledgeType)
    )

class KnowledgeItem(BaseModel):
    """Elemento de conocimiento curatable."""
    id: str
    type: KnowledgeType
    content: Dict[str, Any]
    source: str
    quality_level: KnowledgeQualityLevel = KnowledgeQualityLevel.LOW
    confidence: float = 0.5
    relevance_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
class CurationResult(BaseModel):
    """Resultado de una operación de curación."""
    operation: str
    knowledge_item: KnowledgeItem
    changes: List[Dict[str, Any]]
    quality_delta: float = 0.0
    confidence_delta: float = 0.0

class CuratorAgent(BaseAgent):
    """
    Agente especializado en curación y organización del conocimiento.
    
    Responsabilidades:
    1. Validar la calidad del conocimiento
    2. Organizar y categorizar conocimiento
    3. Enriquecer conocimiento con metadatos
    4. Podar conocimiento irrelevante u obsoleto
    5. Mantener consistencia del conocimiento
    6. Generar reportes de calidad
    """
    
    def __init__(self, config: Optional[CuratorConfig] = None):
        """Inicializa el CuratorAgent."""
        if config is None:
            config = CuratorConfig(
                name="CuratorAgent",
                description="Agente especializado en curación y organización del conocimiento",
                capabilities=[
                    AgentCapability.CODE_ANALYSIS,
                    AgentCapability.PATTERN_DETECTION
                ],
                confidence_threshold=0.8,
                max_processing_time=60
            )
        
        super().__init__(config)
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        self.curation_history: List[Dict[str, Any]] = []
        
    async def _initialize_internal(self) -> bool:
        """Inicialización específica del CuratorAgent."""
        try:
            # Obtener dependencias
            self.knowledge_graph = self.dependencies.get('knowledge_graph')
            self.embedding_generator = self.dependencies.get('embedding_generator')
            
            if not self.knowledge_graph:
                raise AgentException("KnowledgeGraph dependency is required")
            
            # Cargar conocimiento existente
            await self._load_existing_knowledge()
            
            # Iniciar tareas de curación automática
            if self.config.auto_curation:
                asyncio.create_task(self._auto_curation_task())
            
            # Inicializar índices de búsqueda
            await self._initialize_search_indices()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CuratorAgent: {e}")
            return False
    
    async def _process_internal(self, input_data: AgentInput) -> AgentOutput:
        """Procesa una operación de curación."""
        try:
            operation = input_data.data.get("operation", "curate")
            
            if operation == "curate":
                result = await self.curate_knowledge(input_data.data.get("knowledge_item"))
            elif operation == "validate":
                result = await self.validate_knowledge(input_data.data.get("knowledge_item"))
            elif operation == "organize":
                result = await self.organize_knowledge(input_data.data.get("category"))
            elif operation == "prune":
                result = await self.prune_knowledge(input_data.data.get("criteria"))
            elif operation == "enrich":
                result = await self.enrich_knowledge(input_data.data.get("knowledge_item"))
            elif operation == "link":
                result = await self.link_related_knowledge(input_data.data.get("items"))
            elif operation == "report":
                result = await self.generate_knowledge_report(input_data.data.get("report_type"))
            else:
                raise ValidationError(f"Unknown operation: {operation}")
            
            return AgentOutput(
                request_id=input_data.request_id,
                agent_id=self.config.agent_id,
                success=True,
                data={"result": result.dict() if hasattr(result, 'dict') else result},
                confidence=0.9,
                reasoning=[f"Completed {operation} operation successfully"]
            )
            
        except Exception as e:
            self.logger.error(f"Error in curation operation: {e}")
            return AgentOutput(
                request_id=input_data.request_id,
                agent_id=self.config.agent_id,
                success=False,
                errors=[str(e)],
                confidence=0.0
            )
    
    async def _learn_internal(self, feedback: Dict[str, Any]) -> bool:
        """Aprende del feedback de operaciones de curación."""
        try:
            feedback_type = feedback.get("type", "curation_feedback")
            
            if feedback_type == "quality_feedback":
                return await self._learn_from_quality_feedback(feedback)
            elif feedback_type == "organization_feedback":
                return await self._learn_from_organization_feedback(feedback)
            elif feedback_type == "pruning_feedback":
                return await self._learn_from_pruning_feedback(feedback)
            else:
                self.logger.warning(f"Unknown feedback type: {feedback_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in learning: {e}")
            return False
    
    def _validate_input_specific(self, input_data: AgentInput) -> None:
        """Validación específica para CuratorAgent."""
        operation = input_data.data.get("operation", "")
        
        valid_operations = [
            "curate", "validate", "organize", "prune", 
            "enrich", "link", "report"
        ]
        
        if operation and operation not in valid_operations:
            raise ValidationError(f"Operation must be one of {valid_operations}")
    
    async def _save_state(self) -> None:
        """Guarda el estado del agente."""
        try:
            # Guardar conocimiento curado
            state_data = {
                "knowledge_base_size": len(self.knowledge_base),
                "curation_history": self.curation_history[-100:],
                "quality_metrics": await self._calculate_quality_metrics(),
                "last_curation": datetime.now()
            }
            
            # En implementación real, guardaríamos a almacenamiento persistente
            self.logger.info(f"Saving CuratorAgent state with {len(self.knowledge_base)} items")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    # ===== FUNCIONES PÚBLICAS ESPECÍFICAS =====
    
    async def curate_knowledge(self, knowledge_data: Dict[str, Any]) -> CurationResult:
        """
        Realiza curación completa de un elemento de conocimiento.
        
        Args:
            knowledge_data: Datos del conocimiento a curar
            
        Returns:
            Resultado de la curación
        """
        # Crear o actualizar elemento de conocimiento
        knowledge_item = await self._create_or_update_knowledge_item(knowledge_data)
        
        # Realizar pasos de curación
        curation_steps = [
            self._validate_knowledge_item,
            self._categorize_knowledge,
            self._enrich_with_metadata,
            self._assess_quality,
            self._link_to_related
        ]
        
        changes = []
        quality_delta = 0.0
        confidence_delta = 0.0
        
        original_quality = knowledge_item.quality_level
        original_confidence = knowledge_item.confidence
        
        for step in curation_steps:
            try:
                step_result = await step(knowledge_item)
                if step_result:
                    changes.append(step_result)
            except Exception as e:
                self.logger.warning(f"Error in curation step {step.__name__}: {e}")
        
        # Calcular deltas
        quality_delta = self._calculate_quality_delta(original_quality, knowledge_item.quality_level)
        confidence_delta = knowledge_item.confidence - original_confidence
        
        # Almacenar en conocimiento base
        self.knowledge_base[knowledge_item.id] = knowledge_item
        
        # Actualizar grafo de conocimiento
        if self.knowledge_graph:
            await self._update_knowledge_graph(knowledge_item)
        
        # Registrar en historial
        self.curation_history.append({
            "timestamp": datetime.now(),
            "item_id": knowledge_item.id,
            "operation": "curate",
            "changes": changes,
            "quality_delta": quality_delta,
            "confidence_delta": confidence_delta
        })
        
        return CurationResult(
            operation="curate",
            knowledge_item=knowledge_item,
            changes=changes,
            quality_delta=quality_delta,
            confidence_delta=confidence_delta
        )
    
    async def validate_knowledge(self, knowledge_item: KnowledgeItem) -> Dict[str, Any]:
        """
        Valida la calidad y veracidad del conocimiento.
        
        Args:
            knowledge_item: Elemento de conocimiento a validar
            
        Returns:
            Dict con resultados de validación
        """
        validation_results = {
            "completeness": await self._validate_completeness(knowledge_item),
            "accuracy": await self._validate_accuracy(knowledge_item),
            "relevance": await self._validate_relevance(knowledge_item),
            "consistency": await self._validate_consistency(knowledge_item),
            "timeliness": await self._validate_timeliness(knowledge_item)
        }
        
        # Calcular puntuación general
        scores = [result.get("score", 0) for result in validation_results.values()]
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # Determinar nivel de calidad
        quality_level = self._determine_quality_level(overall_score)
        
        # Actualizar elemento
        knowledge_item.quality_level = quality_level
        knowledge_item.confidence = overall_score
        knowledge_item.updated_at = datetime.now()
        
        return {
            "validation_results": validation_results,
            "overall_score": overall_score,
            "quality_level": quality_level.value,
            "recommendations": await self._generate_validation_recommendations(validation_results)
        }
    
    async def organize_knowledge(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Organiza el conocimiento por categorías y relaciones.
        
        Args:
            category: Categoría específica para organizar (opcional)
            
        Returns:
            Dict con resultados de organización
        """
        organization_results = {}
        
        if category:
            # Organizar categoría específica
            items_in_category = [
                item for item in self.knowledge_base.values() 
                if item.metadata.get("category") == category
            ]
            
            organization_results[category] = await self._organize_category(
                items_in_category, category
            )
        else:
            # Organizar todo el conocimiento
            categories = await self._extract_categories()
            
            for category_name in categories:
                items_in_category = [
                    item for item in self.knowledge_base.values() 
                    if item.metadata.get("category") == category_name
                ]
                
                if items_in_category:
                    organization_results[category_name] = await self._organize_category(
                        items_in_category, category_name
                    )
        
        # Crear índices de búsqueda
        await self._rebuild_search_indices()
        
        return {
            "organized_categories": len(organization_results),
            "total_items_organized": sum(len(v.get("items", [])) for v in organization_results.values()),
            "organization_results": organization_results,
            "new_relationships": await self._count_new_relationships()
        }
    
    async def prune_knowledge(self, criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Elimina conocimiento irrelevante u obsoleto.
        
        Args:
            criteria: Criterios para la poda
            
        Returns:
            Dict con resultados de la poda
        """
        if not criteria:
            criteria = self._get_default_pruning_criteria()
        
        items_to_prune = []
        reasons = {}
        
        # Evaluar cada elemento contra los criterios
        for item_id, item in list(self.knowledge_base.items()):
            should_prune, reason = await self._should_prune_item(item, criteria)
            
            if should_prune:
                items_to_prune.append(item_id)
                reasons[item_id] = reason
        
        # Realizar la poda
        pruned_items = []
        for item_id in items_to_prune:
            pruned_item = self.knowledge_base.pop(item_id, None)
            if pruned_item:
                pruned_items.append(pruned_item)
                
                # Eliminar del grafo de conocimiento
                if self.knowledge_graph:
                    await self.knowledge_graph.remove_node(item_id)
        
        # Compactar índices
        await self._compact_knowledge_base()
        
        return {
            "pruned_count": len(pruned_items),
            "remaining_count": len(self.knowledge_base),
            "pruned_items": [item.id for item in pruned_items],
            "pruning_reasons": reasons,
            "space_saved": await self._calculate_space_saved(pruned_items)
        }
    
    async def enrich_knowledge(self, knowledge_item: KnowledgeItem) -> KnowledgeItem:
        """
        Enriquece un elemento de conocimiento con metadatos y contexto.
        
        Args:
            knowledge_item: Elemento a enriquecer
            
        Returns:
            Elemento enriquecido
        """
        enrichment_steps = [
            self._add_semantic_metadata,
            self._extract_key_concepts,
            self._generate_summary,
            self._add_cross_references,
            self._calculate_relevance_score
        ]
        
        for step in enrichment_steps:
            try:
                await step(knowledge_item)
            except Exception as e:
                self.logger.warning(f"Error in enrichment step {step.__name__}: {e}")
        
        # Actualizar timestamp
        knowledge_item.updated_at = datetime.now()
        
        return knowledge_item
    
    async def link_related_knowledge(self, items: List[KnowledgeItem]) -> Dict[str, Any]:
        """
        Encuentra y establece relaciones entre elementos de conocimiento.
        
        Args:
            items: Elementos para relacionar
            
        Returns:
            Dict con relaciones establecidas
        """
        relationships = []
        
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                # Calcular similitud
                similarity = await self._calculate_similarity(item1, item2)
                
                if similarity > 0.7:  # Umbral de similitud
                    relationship_type = await self._determine_relationship_type(item1, item2, similarity)
                    
                    relationship = {
                        "source": item1.id,
                        "target": item2.id,
                        "type": relationship_type,
                        "similarity": similarity,
                        "confidence": min(item1.confidence, item2.confidence)
                    }
                    
                    relationships.append(relationship)
                    
                    # Agregar al grafo de conocimiento
                    if self.knowledge_graph:
                        await self.knowledge_graph.add_edge(
                            source_id=item1.id,
                            target_id=item2.id,
                            relationship_type=relationship_type,
                            weight=similarity
                        )
        
        return {
            "total_relationships": len(relationships),
            "relationships": relationships,
            "average_similarity": sum(r["similarity"] for r in relationships) / len(relationships) if relationships else 0,
            "graph_updated": self.knowledge_graph is not None
        }
    
    async def generate_knowledge_report(self, report_type: str = "summary") -> Dict[str, Any]:
        """
        Genera reportes sobre el estado del conocimiento.
        
        Args:
            report_type: Tipo de reporte
            
        Returns:
            Dict con reporte generado
        """
        report_generators = {
            "summary": self._generate_summary_report,
            "quality": self._generate_quality_report,
            "coverage": self._generate_coverage_report,
            "usage": self._generate_usage_report,
            "trends": self._generate_trends_report
        }
        
        generator = report_generators.get(report_type)
        if not generator:
            raise ValidationError(f"Unknown report type: {report_type}")
        
        return await generator()
    
    # ===== FUNCIONES PRIVADAS DE IMPLEMENTACIÓN =====
    
    async def _load_existing_knowledge(self) -> None:
        """Carga conocimiento existente del almacenamiento."""
        try:
            if self.knowledge_graph:
                # Cargar desde grafo de conocimiento
                nodes = await self.knowledge_graph.get_all_nodes()
                
                for node in nodes:
                    knowledge_item = KnowledgeItem(
                        id=node["id"],
                        type=KnowledgeType(node.get("type", "code_knowledge")),
                        content=node.get("properties", {}),
                        source=node.get("source", "unknown"),
                        quality_level=KnowledgeQualityLevel(node.get("quality_level", "low")),
                        confidence=node.get("confidence", 0.5),
                        metadata=node.get("metadata", {}),
                        created_at=datetime.fromisoformat(node.get("created_at", datetime.now().isoformat())),
                        updated_at=datetime.fromisoformat(node.get("updated_at", datetime.now().isoformat())),
                        last_accessed=datetime.fromisoformat(node.get("last_accessed", datetime.now().isoformat())),
                        access_count=node.get("access_count", 0)
                    )
                    
                    self.knowledge_base[knowledge_item.id] = knowledge_item
                
                self.logger.info(f"Loaded {len(self.knowledge_base)} knowledge items from graph")
        except Exception as e:
            self.logger.error(f"Error loading existing knowledge: {e}")
    
    async def _auto_curation_task(self) -> None:
        """Tarea de curación automática en intervalos regulares."""
        while self._initialized:
            try:
                # Curación rápida cada 5 minutos
                await asyncio.sleep(self.config.curation_intervals["quick"])
                await self._quick_curation_cycle()
                
                # Curación estándar cada hora
                if len(self.curation_history) % 12 == 0:  # Cada 12 ciclos rápidos
                    await self._standard_curation_cycle()
                
                # Curación profunda cada día
                if len(self.curation_history) % 288 == 0:  # Cada 288 ciclos rápidos (24 horas)
                    await self._deep_curation_cycle()
                    
            except Exception as e:
                self.logger.error(f"Error in auto curation task: {e}")
                await asyncio.sleep(60)  # Esperar antes de reintentar
    
    async def _quick_curation_cycle(self) -> None:
        """Ciclo de curación rápida (alta prioridad)."""
        # Validar elementos de baja calidad
        low_quality_items = [
            item for item in self.knowledge_base.values()
            if item.quality_level == KnowledgeQualityLevel.LOW
        ][:10]  # Limitar a 10 items por ciclo
        
        for item in low_quality_items:
            await self.curate_knowledge(item.dict())
    
    async def _standard_curation_cycle(self) -> None:
        """Ciclo de curación estándar (mantenimiento)."""
        # Organizar conocimiento
        await self.organize_knowledge()
        
        # Enriquecer elementos importantes
        important_items = [
            item for item in self.knowledge_base.values()
            if item.relevance_score > 0.8
        ][:5]
        
        for item in important_items:
            await self.enrich_knowledge(item)
    
    async def _deep_curation_cycle(self) -> None:
        """Ciclo de curación profunda (optimización completa)."""
        # Generar reportes
        await self.generate_knowledge_report("summary")
        await self.generate_knowledge_report("quality")
        
        # Poda de conocimiento
        await self.prune_knowledge()
        
        # Reconstruir índices
        await self._rebuild_search_indices()
    
    async def _create_or_update_knowledge_item(self, data: Dict[str, Any]) -> KnowledgeItem:
        """Crea o actualiza un elemento de conocimiento."""
        # Generar ID único basado en contenido
        content_hash = hashlib.sha256(
            str(data.get("content", {})).encode()
        ).hexdigest()[:16]
        
        item_id = f"knowledge_{content_hash}"
        
        if item_id in self.knowledge_base:
            # Actualizar existente
            existing_item = self.knowledge_base[item_id]
            existing_item.content.update(data.get("content", {}))
            existing_item.updated_at = datetime.now()
            return existing_item
        else:
            # Crear nuevo
            return KnowledgeItem(
                id=item_id,
                type=KnowledgeType(data.get("type", "code_knowledge")),
                content=data.get("content", {}),
                source=data.get("source", "system"),
                metadata=data.get("metadata", {})
            )
    
    async def _validate_knowledge_item(self, item: KnowledgeItem) -> Optional[Dict[str, Any]]:
        """Valida un elemento de conocimiento."""
        validation = await self.validate_knowledge(item)
        
        if validation["overall_score"] > item.confidence:
            old_confidence = item.confidence
            item.confidence = validation["overall_score"]
            item.quality_level = validation["quality_level"]
            
            return {
                "operation": "validation",
                "field": "confidence",
                "old_value": old_confidence,
                "new_value": item.confidence,
                "quality_level": item.quality_level.value
            }
        
        return None
    
    async def _categorize_knowledge(self, item: KnowledgeItem) -> Optional[Dict[str, Any]]:
        """Categoriza un elemento de conocimiento."""
        if "category" in item.metadata:
            return None
        
        category = await self._determine_category(item)
        
        if category:
            item.metadata["category"] = category
            
            return {
                "operation": "categorization",
                "field": "category",
                "new_value": category
            }
        
        return None
    
    async def _enrich_with_metadata(self, item: KnowledgeItem) -> Optional[Dict[str, Any]]:
        """Enriquece con metadatos semánticos."""
        metadata_added = []
        
        # Extraer conceptos clave
        key_concepts = await self._extract_key_concepts_from_content(item.content)
        if key_concepts and "key_concepts" not in item.metadata:
            item.metadata["key_concepts"] = key_concepts
            metadata_added.append("key_concepts")
        
        # Generar resumen
        summary = await self._generate_content_summary(item.content)
        if summary and "summary" not in item.metadata:
            item.metadata["summary"] = summary
            metadata_added.append("summary")
        
        # Calcular relevancia
        relevance = await self._calculate_content_relevance(item.content)
        if relevance != item.relevance_score:
            old_relevance = item.relevance_score
            item.relevance_score = relevance
            metadata_added.append(f"relevance_score ({old_relevance} -> {relevance})")
        
        if metadata_added:
            return {
                "operation": "enrichment",
                "metadata_added": metadata_added
            }
        
        return None
    
    async def _assess_quality(self, item: KnowledgeItem) -> Optional[Dict[str, Any]]:
        """Evalúa y actualiza la calidad del conocimiento."""
        quality_score = await self._calculate_quality_score(item)
        
        old_quality = item.quality_level
        new_quality = self._determine_quality_level(quality_score)
        
        if new_quality != old_quality:
            item.quality_level = new_quality
            
            return {
                "operation": "quality_assessment",
                "field": "quality_level",
                "old_value": old_quality.value,
                "new_value": new_quality.value,
                "score": quality_score
            }
        
        return None
    
    async def _link_to_related(self, item: KnowledgeItem) -> Optional[Dict[str, Any]]:
        """Encuentra y enlaza conocimiento relacionado."""
        related_items = await self._find_related_knowledge(item)
        
        if related_items:
            # Actualizar metadatos
            if "related_items" not in item.metadata:
                item.metadata["related_items"] = []
            
            new_relations = []
            for related in related_items:
                if related["id"] not in item.metadata["related_items"]:
                    item.metadata["related_items"].append(related["id"])
                    new_relations.append(related["id"])
            
            if new_relations:
                return {
                    "operation": "linking",
                    "new_relations": new_relations,
                    "total_relations": len(item.metadata["related_items"])
                }
        
        return None
    
    async def _update_knowledge_graph(self, item: KnowledgeItem) -> None:
        """Actualiza el grafo de conocimiento con el elemento."""
        # Agregar o actualizar nodo
        await self.knowledge_graph.add_node(
            node_id=item.id,
            node_type=item.type.value,
            properties={
                **item.content,
                "quality_level": item.quality_level.value,
                "confidence": item.confidence,
                "metadata": item.metadata,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            }
        )
    
    async def _determine_category(self, item: KnowledgeItem) -> Optional[str]:
        """Determina la categoría de un elemento de conocimiento."""
        # Basado en tipo de conocimiento
        type_categories = {
            KnowledgeType.CODE_KNOWLEDGE: ["functions", "classes", "modules"],
            KnowledgeType.ARCHITECTURE: ["patterns", "components", "interfaces"],
            KnowledgeType.DOMAIN: ["concepts", "business_rules", "entities"],
            KnowledgeType.BEST_PRACTICES: ["guidelines", "standards", "conventions"],
            KnowledgeType.PATTERNS: ["design_patterns", "anti_patterns", "idioms"],
            KnowledgeType.ISSUES: ["bugs", "vulnerabilities", "code_smells"],
            KnowledgeType.SOLUTIONS: ["fixes", "workarounds", "optimizations"]
        }
        
        categories = type_categories.get(item.type, [])
        
        # Usar contenido para determinar sub-categoría
        if categories and item.content:
            content_text = str(item.content).lower()
            
            for category in categories:
                if category in content_text:
                    return category
        
        return categories[0] if categories else "general"
    
    async def _calculate_quality_score(self, item: KnowledgeItem) -> float:
        """Calcula un puntaje de calidad para el conocimiento."""
        factors = {
            "completeness": 0.2,
            "accuracy": 0.3,
            "relevance": 0.2,
            "timeliness": 0.15,
            "source_reliability": 0.15
        }
        
        scores = {}
        
        # Completitud
        completeness = await self._assess_completeness(item)
        scores["completeness"] = completeness
        
        # Precisión (basado en confianza y verificaciones)
        accuracy = item.confidence * 0.7 + (1 if item.quality_level == KnowledgeQualityLevel.EXPERT else 0.5)
        scores["accuracy"] = accuracy
        
        # Relevancia
        scores["relevance"] = item.relevance_score
        
        # Actualidad
        age_days = (datetime.now() - item.updated_at).days
        timeliness = max(0, 1 - (age_days / 365))  # Decae en un año
        scores["timeliness"] = timeliness
        
        # Confiabilidad de la fuente
        source_score = self._assess_source_reliability(item.source)
        scores["source_reliability"] = source_score
        
        # Calcular puntaje ponderado
        total_score = sum(scores[factor] * weight for factor, weight in factors.items())
        
        return min(1.0, total_score)
    
    def _determine_quality_level(self, score: float) -> KnowledgeQualityLevel:
        """Determina el nivel de calidad basado en el puntaje."""
        if score >= self.config.quality_thresholds[KnowledgeQualityLevel.EXPERT]:
            return KnowledgeQualityLevel.EXPERT
        elif score >= self.config.quality_thresholds[KnowledgeQualityLevel.HIGH]:
            return KnowledgeQualityLevel.HIGH
        elif score >= self.config.quality_thresholds[KnowledgeQualityLevel.MEDIUM]:
            return KnowledgeQualityLevel.MEDIUM
        else:
            return KnowledgeQualityLevel.LOW
    
    async def _should_prune_item(self, item: KnowledgeItem, criteria: Dict[str, Any]) -> Tuple[bool, str]:
        """Determina si un elemento debe ser podado."""
        reasons = []
        
        # Relevancia baja
        if item.relevance_score < self.config.min_relevance_score:
            reasons.append(f"Low relevance ({item.relevance_score:.2f})")
        
        # Calidad muy baja
        if item.quality_level == KnowledgeQualityLevel.LOW and item.confidence < 0.3:
            reasons.append(f"Very low quality (confidence: {item.confidence:.2f})")
        
        # Muy antiguo y no accedido
        age_days = (datetime.now() - item.last_accessed).days
        if age_days > 365 and item.access_count < 5:  # 1 año sin uso frecuente
            reasons.append(f"Old and unused ({age_days} days, {item.access_count} accesses)")
        
        # Duplicados (simplificado)
        if await self._is_duplicate(item):
            reasons.append("Duplicate content")
        
        return len(reasons) > 0, "; ".join(reasons)
    
    async def _generate_summary_report(self) -> Dict[str, Any]:
        """Genera reporte resumen del conocimiento."""
        return {
            "report_type": "summary",
            "timestamp": datetime.now(),
            "total_items": len(self.knowledge_base),
            "by_quality": {
                level.value: len([i for i in self.knowledge_base.values() if i.quality_level == level])
                for level in KnowledgeQualityLevel
            },
            "by_type": {
                ktype.value: len([i for i in self.knowledge_base.values() if i.type == ktype])
                for ktype in KnowledgeType
            },
            "avg_confidence": sum(i.confidence for i in self.knowledge_base.values()) / len(self.knowledge_base) if self.knowledge_base else 0,
            "avg_relevance": sum(i.relevance_score for i in self.knowledge_base.values()) / len(self.knowledge_base) if self.knowledge_base else 0,
            "recent_curations": len([h for h in self.curation_history if (datetime.now() - h["timestamp"]).days < 7])
        }
    
    # Métodos auxiliares simplificados
    async def _initialize_search_indices(self) -> None:
        """Inicializa índices para búsqueda eficiente."""
        self.search_indices = {
            "by_category": {},
            "by_quality": {},
            "by_type": {}
        }
    
    async def _rebuild_search_indices(self) -> None:
        """Reconstruye índices de búsqueda."""
        self.search_indices = {"by_category": {}, "by_quality": {}, "by_type": {}}
        
        for item in self.knowledge_base.values():
            # Índice por categoría
            category = item.metadata.get("category", "uncategorized")
            if category not in self.search_indices["by_category"]:
                self.search_indices["by_category"][category] = []
            self.search_indices["by_category"][category].append(item.id)
            
            # Índice por calidad
            quality = item.quality_level.value
            if quality not in self.search_indices["by_quality"]:
                self.search_indices["by_quality"][quality] = []
            self.search_indices["by_quality"][quality].append(item.id)
            
            # Índice por tipo
            ktype = item.type.value
            if ktype not in self.search_indices["by_type"]:
                self.search_indices["by_type"][ktype] = []
            self.search_indices["by_type"][ktype].append(item.id)
    
    def _get_default_pruning_criteria(self) -> Dict[str, Any]:
        """Obtiene criterios de poda por defecto."""
        return {
            "min_relevance": self.config.min_relevance_score,
            "min_confidence": 0.3,
            "max_age_days": 365,
            "min_access_count": 5,
            "allow_duplicates": False
        }
    
    def _assess_source_reliability(self, source: str) -> float:
        """Evalúa la confiabilidad de una fuente."""
        reliable_sources = ["expert_review", "validated_fix", "official_docs"]
        medium_sources = ["code_analysis", "user_feedback", "community"]
        low_sources = ["auto_generated", "unverified", "unknown"]
        
        if source in reliable_sources:
            return 0.9
        elif source in medium_sources:
            return 0.6
        elif source in low_sources:
            return 0.3
        else:
            return 0.5
    
    async def _is_duplicate(self, item: KnowledgeItem) -> bool:
        """Verifica si el conocimiento está duplicado."""
        # Simplificado: comparar hash del contenido
        content_str = str(item.content)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        
        for other_item in self.knowledge_base.values():
            if other_item.id == item.id:
                continue
                
            other_content_str = str(other_item.content)
            other_hash = hashlib.sha256(other_content_str.encode()).hexdigest()
            
            if content_hash == other_hash:
                return True
        
        return False
    
    def _calculate_quality_delta(self, old: KnowledgeQualityLevel, new: KnowledgeQualityLevel) -> float:
        """Calcula el cambio en calidad."""
        quality_values = {
            KnowledgeQualityLevel.LOW: 0.25,
            KnowledgeQualityLevel.MEDIUM: 0.5,
            KnowledgeQualityLevel.HIGH: 0.75,
            KnowledgeQualityLevel.EXPERT: 1.0
        }
        
        return quality_values[new] - quality_values[old]
    
    async def _calculate_space_saved(self, pruned_items: List[KnowledgeItem]) -> int:
        """Calcula el espacio ahorrado por la poda."""
        # Estimación simplificada: 1KB por item
        return len(pruned_items) * 1024
    
    async def _compact_knowledge_base(self) -> None:
        """Compacta el conocimiento base después de la poda."""
        # Reconstruir índices
        await self._rebuild_search_indices()
        
        # Limpiar metadatos temporales
        for item in self.knowledge_base.values():
            if "temp_" in item.metadata:
                del item.metadata["temp_"]

# Ejemplo de uso
if __name__ == "__main__":
    async def main():
        agent = CuratorAgent()
        
        # Inicializar
        await agent.initialize()
        
        # Ejemplo de conocimiento para curar
        knowledge_data = {
            "type": "code_knowledge",
            "content": {
                "function_name": "calculate_total",
                "description": "Calculates total price with tax",
                "parameters": ["items", "tax_rate"],
                "return_type": "float"
            },
            "source": "code_analysis",
            "metadata": {"language": "python"}
        }
        
        # Curar conocimiento
        result = await agent.curate_knowledge(knowledge_data)
        print(f"Knowledge curated: {result.knowledge_item.id}")
        print(f"Quality level: {result.knowledge_item.quality_level.value}")
        
        # Generar reporte
        report = await agent.generate_knowledge_report("summary")
        print(f"Total knowledge items: {report['total_items']}")
        
        await agent.shutdown()
    
    asyncio.run(main())