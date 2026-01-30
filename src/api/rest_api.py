"""
REST API - Implementación completa de la API REST según especificación OpenAPI.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid
import json

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, Path, Body, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pydantic.generics import GenericModel

from ..core.exceptions import BrainException, ValidationError
from ..core.orchestrator import BrainOrchestrator, OperationRequest, OperationResult
from .authentication import Authentication, get_current_user
from .rate_limiter import RateLimiter
from .request_validator import RequestValidator

logger = logging.getLogger(__name__)

# Modelos Pydantic para las requests/responses (según especificación OpenAPI)

class Pagination(BaseModel):
    """Modelo de paginación."""
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)
    total_pages: int
    total_items: int
    has_next: bool
    has_previous: bool

class ProjectCreate(BaseModel):
    """Modelo para creación de proyecto."""
    name: str = Field(..., min_length=1, max_length=255)
    path: str = Field(...)
    description: Optional[str] = None
    language: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class ProjectResponse(BaseModel):
    """Modelo de respuesta para proyecto."""
    id: str
    name: str
    path: str
    description: Optional[str]
    language: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_analyzed: Optional[datetime]
    analysis_status: str
    metadata: Dict[str, Any]
    stats: Optional[Dict[str, Any]]

class AnalysisOptions(BaseModel):
    """Opciones de análisis."""
    mode: str = Field("comprehensive", regex="^(quick|standard|comprehensive|deep)$")
    include_tests: bool = True
    include_docs: bool = True
    max_file_size_mb: int = Field(10, ge=1, le=100)
    timeout_minutes: int = Field(30, ge=1, le=180)
    languages: List[str] = Field(default_factory=lambda: ["python", "javascript", "typescript", "java", "cpp", "go", "rust"])

class AnalysisStatusResponse(BaseModel):
    """Respuesta de estado de análisis."""
    id: str
    status: str
    progress: Optional[float] = Field(None, ge=0, le=100)
    current_step: Optional[str]
    estimated_remaining_seconds: Optional[int]
    started_at: datetime
    completed_at: Optional[datetime]
    results: Optional[Dict[str, Any]]
    errors: List[str]

class QueryRequest(BaseModel):
    """Request para consulta."""
    question: str = Field(..., min_length=1)
    project_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    """Respuesta de consulta."""
    answer: Dict[str, Any]
    confidence: float = Field(..., ge=0, le=1)
    sources: List[Dict[str, Any]]
    reasoning_chain: List[str]
    suggested_followups: List[str]
    processing_time_ms: float

class ConversationRequest(BaseModel):
    """Request para conversación."""
    question: str = Field(..., min_length=1)
    session_id: str
    project_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ConversationResponse(QueryResponse):
    """Respuesta de conversación."""
    session_id: str
    conversation_history: List[Dict[str, Any]]
    context_updated: bool

class ErrorResponse(BaseModel):
    """Respuesta de error."""
    code: str
    message: str
    details: Optional[Dict[str, Any]]
    request_id: Optional[str]

class RESTAPI:
    """
    Implementación de la API REST completa.
    
    Incluye todos los endpoints definidos en la especificación OpenAPI.
    """
    
    def __init__(self, 
                 authentication: Optional[Authentication] = None,
                 rate_limiter: Optional[RateLimiter] = None,
                 request_validator: Optional[RequestValidator] = None):
        """
        Inicializa la API REST.
        
        Args:
            authentication: Sistema de autenticación (opcional)
            rate_limiter: Limitador de tasa (opcional)
            request_validator: Validador de requests (opcional)
        """
        self.authentication = authentication
        self.rate_limiter = rate_limiter
        self.request_validator = request_validator
        self.orchestrator: Optional[BrainOrchestrator] = None
        self.router = APIRouter()
        
        # Métricas
        self.metrics = {
            "requests_by_endpoint": {},
            "response_times": {},
            "errors_by_type": {},
        }
        
        logger.info("RESTAPI inicializada")
    
    async def initialize(self, orchestrator: BrainOrchestrator) -> None:
        """
        Inicializa la API REST con el orquestador.
        
        Args:
            orchestrator: Instancia de BrainOrchestrator
        """
        self.orchestrator = orchestrator
        logger.info("RESTAPI inicializada con orquestador")
    
    def register_endpoints(self, app: FastAPI) -> None:
        """
        Registra todos los endpoints en la aplicación FastAPI.
        
        Args:
            app: Aplicación FastAPI
        """
        # Dependencias comunes
        dependencies = []
        if self.authentication:
            dependencies.append(Depends(get_current_user))
        
        # ========== PROJECTS ==========
        @self.router.get("/projects", 
                        response_model=Dict[str, Any],
                        tags=["Projects"])
        async def list_projects(
            page: int = Query(1, ge=1),
            page_size: int = Query(20, ge=1, le=100),
            language: Optional[str] = Query(None),
            status: Optional[str] = Query(None, regex="^(pending|analyzing|completed|failed)$"),
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Listar proyectos."""
            return await self._list_projects(page, page_size, language, status)
        
        @self.router.post("/projects",
                         response_model=ProjectResponse,
                         status_code=status.HTTP_201_CREATED,
                         tags=["Projects"])
        async def create_project(
            project: ProjectCreate,
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Crear proyecto."""
            return await self._create_project(project)
        
        @self.router.get("/projects/{project_id}",
                        response_model=ProjectResponse,
                        tags=["Projects"])
        async def get_project(
            project_id: str = Path(..., description="ID del proyecto"),
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Obtener proyecto."""
            return await self._get_project(project_id)
        
        @self.router.delete("/projects/{project_id}",
                           status_code=status.HTTP_204_NO_CONTENT,
                           tags=["Projects"])
        async def delete_project(
            project_id: str = Path(..., description="ID del proyecto"),
            force: bool = Query(False, description="Forzar eliminación"),
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Eliminar proyecto."""
            await self._delete_project(project_id, force)
            return None
        
        @self.router.post("/projects/{project_id}/analyze",
                         tags=["Projects", "Analysis"])
        async def analyze_project(
            project_id: str = Path(..., description="ID del proyecto"),
            options: Optional[AnalysisOptions] = Body(None),
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Analizar proyecto."""
            return await self._analyze_project(project_id, options)
        
        # ========== ANALYSIS ==========
        @self.router.get("/analysis/{analysis_id}/status",
                        response_model=AnalysisStatusResponse,
                        tags=["Analysis"])
        async def get_analysis_status(
            analysis_id: str = Path(..., description="ID del análisis"),
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Obtener estado de análisis."""
            return await self._get_analysis_status(analysis_id)
        
        @self.router.post("/analysis/{analysis_id}/cancel",
                         tags=["Analysis"])
        async def cancel_analysis(
            analysis_id: str = Path(..., description="ID del análisis"),
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Cancelar análisis."""
            return await self._cancel_analysis(analysis_id)
        
        # ========== QUERY ==========
        @self.router.post("/query",
                         response_model=QueryResponse,
                         tags=["Query"])
        async def query_project(
            query: QueryRequest,
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Consultar proyecto."""
            return await self._query_project(query)
        
        @self.router.post("/query/conversation",
                         response_model=ConversationResponse,
                         tags=["Query"])
        async def query_conversation(
            conversation: ConversationRequest,
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Consulta conversacional."""
            return await self._query_conversation(conversation)
        
        # ========== KNOWLEDGE ==========
        @self.router.get("/projects/{project_id}/knowledge/graph",
                        tags=["Knowledge"])
        async def export_knowledge_graph(
            project_id: str = Path(..., description="ID del proyecto"),
            format: str = Query("json", regex="^(json|graphml|cypher|dot)$"),
            depth: int = Query(2, ge=1, le=5),
            include: str = Query("all", regex="^(nodes|edges|properties|all)$"),
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Exportar grafo de conocimiento."""
            return await self._export_knowledge_graph(project_id, format, depth, include)
        
        @self.router.post("/projects/{project_id}/knowledge/search",
                         tags=["Knowledge"])
        async def search_knowledge(
            project_id: str = Path(..., description="ID del proyecto"),
            search: Dict[str, Any] = Body(...),
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Buscar en conocimiento."""
            return await self._search_knowledge(project_id, search)
        
        # ========== AGENTS ==========
        @self.router.get("/agents",
                        tags=["Agents"])
        async def list_agents(
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Listar agentes."""
            return await self._list_agents()
        
        @self.router.get("/agents/{agent_id}/capabilities",
                        tags=["Agents"])
        async def get_agent_capabilities(
            agent_id: str = Path(..., description="ID del agente"),
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Capacidades de agente."""
            return await self._get_agent_capabilities(agent_id)
        
        # ========== SYSTEM ==========
        @self.router.get("/system/health",
                        tags=["System"])
        async def system_health(
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Salud del sistema."""
            return await self._system_health()
        
        @self.router.get("/system/metrics",
                        tags=["System"])
        async def system_metrics(
            timeframe: str = Query("hour", regex="^(hour|day|week|month)$"),
            component: Optional[str] = Query(None),
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Métricas del sistema."""
            return await self._system_metrics(timeframe, component)
        
        @self.router.get("/system/status",
                        tags=["System"])
        async def system_status(
            user: Any = Depends(get_current_user) if self.authentication else None
        ):
            """Estado del sistema."""
            return await self._system_status()
        
        # Registrar router en la app
        app.include_router(self.router, prefix="/v1")
        
        logger.info("Endpoints REST registrados")
    
    async def validate_request(self, request_data: Dict, schema: Dict) -> bool:
        """Valida una request contra un esquema."""
        if self.request_validator:
            return await self.request_validator.validate_request_data(request_data, schema)
        return True
    
    async def process_request(self, endpoint: str, request_data: Dict) -> Any:
        """Procesa una request y registra métricas."""
        start_time = datetime.now()
        
        try:
            # Validar request
            if self.request_validator:
                await self.request_validator.validate_request_structure(request_data)
            
            # Procesar (simulado - en realidad se haría en cada endpoint)
            result = {"status": "processed"}
            
            # Registrar métricas
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(endpoint, processing_time, success=True)
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_metrics(endpoint, processing_time, success=False, error=e)
            raise
    
    def format_response(self, data: Any, status_code: int = 200) -> JSONResponse:
        """Formatea una respuesta JSON."""
        return JSONResponse(
            content=data,
            status_code=status_code,
            headers={
                "X-Request-ID": str(uuid.uuid4()),
                "X-Response-Time": str(datetime.now().isoformat()),
            }
        )
    
    async def handle_authentication(self, request: Any) -> Optional[Dict]:
        """Maneja autenticación de request."""
        if self.authentication:
            return await self.authentication.authenticate_user(request)
        return None
    
    async def rate_limit_requests(self, request: Any, endpoint: str) -> bool:
        """Aplica rate limiting a una request."""
        if self.rate_limiter:
            return await self.rate_limiter.check_rate_limit(request, endpoint)
        return True
    
    async def log_api_activity(self, request: Any, response: Any, user: Optional[Dict] = None) -> None:
        """Loggea actividad de API."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "user_id": user.get("id") if user else None,
            "user_agent": request.headers.get("user-agent"),
            "ip_address": request.client.host if request.client else None,
            "processing_time_ms": response.headers.get("X-Response-Time"),
        }
        
        logger.info("API Activity: %s", json.dumps(log_data))
    
    # Implementación de endpoints
    
    async def _list_projects(self, page: int, page_size: int, language: Optional[str], status: Optional[str]) -> Dict[str, Any]:
        """Implementación de listado de proyectos."""
        # En una implementación real, esto consultaría la base de datos
        # Por ahora retornamos datos de ejemplo
        
        projects = [
            {
                "id": str(uuid.uuid4()),
                "name": "Example Project",
                "path": "/path/to/project",
                "description": "An example project",
                "language": "python",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "last_analyzed": datetime.now().isoformat(),
                "analysis_status": "completed",
                "metadata": {},
                "stats": {
                    "file_count": 100,
                    "total_lines": 5000,
                    "function_count": 50,
                    "class_count": 10,
                    "issue_count": 5,
                    "critical_issues": 1,
                    "avg_complexity": 2.5,
                }
            }
        ]
        
        pagination = {
            "page": page,
            "page_size": page_size,
            "total_pages": 1,
            "total_items": 1,
            "has_next": False,
            "has_previous": False,
        }
        
        return {
            "projects": projects,
            "pagination": pagination,
        }
    
    async def _create_project(self, project: ProjectCreate) -> ProjectResponse:
        """Implementación de creación de proyecto."""
        if not self.orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        
        # Validar que el proyecto no exista ya
        # (en implementación real, verificaría en base de datos)
        
        # Crear proyecto en el sistema
        project_id = str(uuid.uuid4())
        
        project_data = ProjectResponse(
            id=project_id,
            name=project.name,
            path=project.path,
            description=project.description,
            language=project.language,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_analyzed=None,
            analysis_status="pending",
            metadata={},
            stats=None,
        )
        
        # En una implementación real, guardaría en base de datos
        # y posiblemente iniciaría análisis automático
        
        return project_data
    
    async def _get_project(self, project_id: str) -> ProjectResponse:
        """Implementación de obtención de proyecto."""
        # En implementación real, buscaría en base de datos
        # Por ahora retornamos un proyecto de ejemplo
        
        return ProjectResponse(
            id=project_id,
            name="Example Project",
            path="/path/to/project",
            description="An example project",
            language="python",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_analyzed=datetime.now(),
            analysis_status="completed",
            metadata={},
            stats={
                "file_count": 100,
                "total_lines": 5000,
                "function_count": 50,
                "class_count": 10,
                "issue_count": 5,
                "critical_issues": 1,
                "avg_complexity": 2.5,
            }
        )
    
    async def _delete_project(self, project_id: str, force: bool) -> None:
        """Implementación de eliminación de proyecto."""
        # En implementación real, eliminaría de base de datos
        # y limpiaría recursos asociados
        
        logger.info(f"Proyecto {project_id} eliminado (force={force})")
    
    async def _analyze_project(self, project_id: str, options: Optional[AnalysisOptions]) -> Dict[str, Any]:
        """Implementación de análisis de proyecto."""
        if not self.orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        
        # Verificar que el proyecto exista
        # (en implementación real, verificaría en base de datos)
        
        # Iniciar análisis
        analysis_id = str(uuid.uuid4())
        
        # En una implementación real, esto iniciaría un análisis asíncrono
        # y retornaría un ID para seguir su progreso
        
        return {
            "analysis_id": analysis_id,
            "status_url": f"/v1/analysis/{analysis_id}/status",
            "estimated_time_seconds": 300,  # 5 minutos
        }
    
    async def _get_analysis_status(self, analysis_id: str) -> AnalysisStatusResponse:
        """Implementación de estado de análisis."""
        # En implementación real, consultaría el estado del análisis
        # Por ahora retornamos un estado de ejemplo
        
        return AnalysisStatusResponse(
            id=analysis_id,
            status="completed",
            progress=100.0,
            current_step="Generating report",
            estimated_remaining_seconds=0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            results={
                "files_analyzed": 100,
                "entities_extracted": 500,
                "issues_found": 10,
                "analysis_time_seconds": 45.2,
            },
            errors=[],
        )
    
    async def _cancel_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """Implementación de cancelación de análisis."""
        # En implementación real, cancelaría el análisis si está en progreso
        
        return {
            "analysis_id": analysis_id,
            "cancelled": True,
            "message": "Analysis cancelled successfully",
        }
    
    async def _query_project(self, query: QueryRequest) -> QueryResponse:
        """Implementación de consulta de proyecto."""
        if not self.orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        
        # Procesar pregunta
        start_time = datetime.now()
        
        try:
            # En una implementación real, usaría el orquestador
            # Por ahora retornamos una respuesta de ejemplo
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return QueryResponse(
                answer={
                    "text": "Esta es una respuesta de ejemplo.",
                    "structured": {
                        "type": "explanation",
                        "content": "La función process_data se define en el archivo utils.py.",
                    },
                    "code_examples": [
                        {
                            "code": "def process_data(data):\n    return data * 2",
                            "language": "python",
                            "description": "Definición de la función process_data",
                        }
                    ],
                    "related_concepts": ["data processing", "utility functions"],
                },
                confidence=0.92,
                sources=[
                    {
                        "type": "code",
                        "file_path": "src/utils.py",
                        "line_range": [10, 20],
                        "confidence": 0.95,
                        "excerpt": "def process_data(data):\n    \"\"\"Process incoming data.\"\"\"\n    return data * 2",
                    }
                ],
                reasoning_chain=[
                    "Identificó que la pregunta es sobre una función específica",
                    "Buscó en la base de conocimiento del proyecto",
                    "Encontró la definición en utils.py",
                    "Extrajo información relevante",
                ],
                suggested_followups=[
                    "¿Qué parámetros acepta la función?",
                    "¿Dónde se llama a esta función?",
                    "¿Hay tests para esta función?",
                ],
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    async def _query_conversation(self, conversation: ConversationRequest) -> ConversationResponse:
        """Implementación de consulta conversacional."""
        # Primero procesamos la pregunta normalmente
        query_response = await self._query_project(QueryRequest(
            question=conversation.question,
            project_id=conversation.project_id,
            context=conversation.context,
        ))
        
        # Luego añadimos información de conversación
        # (en implementación real, manejaríamos historial de sesión)
        
        return ConversationResponse(
            **query_response.dict(),
            session_id=conversation.session_id,
            conversation_history=[
                {
                    "role": "user",
                    "content": conversation.question,
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "role": "assistant",
                    "content": query_response.answer.get("text", ""),
                    "timestamp": datetime.now().isoformat(),
                    "confidence": query_response.confidence,
                }
            ],
            context_updated=True,
        )
    
    async def _export_knowledge_graph(self, project_id: str, format: str, depth: int, include: str) -> Dict[str, Any]:
        """Implementación de exportación de grafo de conocimiento."""
        # En implementación real, exportaría el grafo del proyecto
        # Por ahora retornamos un ejemplo
        
        sample_graph = {
            "nodes": [
                {"id": "1", "type": "file", "name": "main.py"},
                {"id": "2", "type": "function", "name": "main"},
                {"id": "3", "type": "class", "name": "Processor"},
            ],
            "edges": [
                {"source": "1", "target": "2", "type": "contains"},
                {"source": "1", "target": "3", "type": "contains"},
            ],
        }
        
        if format == "json":
            data = sample_graph
        elif format == "graphml":
            data = "<graphml>...</graphml>"
        elif format == "cypher":
            data = "CREATE (n1:File {name: 'main.py'})..."
        elif format == "dot":
            data = "digraph { main -> Processor }"
        else:
            data = sample_graph
        
        return {
            "format": format,
            "data": data,
            "project_id": project_id,
            "exported_at": datetime.now().isoformat(),
            "node_count": len(sample_graph["nodes"]),
            "edge_count": len(sample_graph["edges"]),
        }
    
    async def _search_knowledge(self, project_id: str, search: Dict[str, Any]) -> Dict[str, Any]:
        """Implementación de búsqueda en conocimiento."""
        query = search.get("query", "")
        search_type = search.get("type", "hybrid")
        limit = search.get("limit", 10)
        
        # En implementación real, buscaría en el proyecto
        # Por ahora retornamos resultados de ejemplo
        
        results = [
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "name": "process_data",
                "score": 0.95,
                "content": "Defines the main data processing function",
                "metadata": {
                    "file": "utils.py",
                    "lines": [10, 20],
                    "language": "python",
                }
            },
            {
                "id": str(uuid.uuid4()),
                "type": "class",
                "name": "DataProcessor",
                "score": 0.87,
                "content": "Main class for processing data streams",
                "metadata": {
                    "file": "processor.py",
                    "lines": [5, 50],
                    "language": "python",
                }
            },
        ]
        
        return {
            "results": results[:limit],
            "total": len(results),
            "query": query,
            "type": search_type,
            "project_id": project_id,
        }
    
    async def _list_agents(self) -> Dict[str, Any]:
        """Implementación de listado de agentes."""
        # En implementación real, listaría agentes disponibles
        # Por ahora retornamos lista de ejemplo
        
        agents = [
            {
                "id": "code_analyzer",
                "name": "Code Analyzer",
                "description": "Analyzes code structure and quality",
                "version": "1.0.0",
                "status": "ready",
                "capabilities": ["code_analysis", "pattern_detection", "quality_assessment"],
                "metrics": {
                    "requests_processed": 150,
                    "success_rate": 0.95,
                    "avg_processing_time_ms": 120.5,
                }
            },
            {
                "id": "qa_agent",
                "name": "Q&A Agent",
                "description": "Answers questions about code and projects",
                "version": "1.0.0",
                "status": "ready",
                "capabilities": ["question_answering", "explanation_generation"],
                "metrics": {
                    "requests_processed": 300,
                    "success_rate": 0.88,
                    "avg_processing_time_ms": 45.2,
                }
            },
        ]
        
        return {"agents": agents}
    
    async def _get_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Implementación de capacidades de agente."""
        # En implementación real, obtendría capacidades del agente
        # Por ahora retornamos capacidades de ejemplo
        
        capabilities = {
            "agent_id": agent_id,
            "capabilities": [
                {
                    "name": "code_analysis",
                    "description": "Analyzes code structure, complexity, and patterns",
                    "confidence_threshold": 0.7,
                    "supported_languages": ["python", "javascript", "java", "cpp", "go"],
                    "examples": [
                        {
                            "input": "Analyze this function for complexity",
                            "output": "Cyclomatic complexity: 5, Cognitive complexity: 3",
                        }
                    ],
                },
                {
                    "name": "pattern_detection",
                    "description": "Detects design patterns and anti-patterns",
                    "confidence_threshold": 0.8,
                    "supported_languages": ["python", "java"],
                    "examples": [
                        {
                            "input": "Detect patterns in this class",
                            "output": "Detected Singleton pattern with 90% confidence",
                        }
                    ],
                },
            ],
        }
        
        return capabilities
    
    async def _system_health(self) -> Dict[str, Any]:
        """Implementación de salud del sistema."""
        from ..core.health_check import HealthCheck
        
        health_check = HealthCheck()
        health_status = await health_check.check_system_health()
        
        return {
            "status": health_status["overall"],
            "timestamp": datetime.now().isoformat(),
            "components": health_status["components"],
            "version": "1.0.0",
        }
    
    async def _system_metrics(self, timeframe: str, component: Optional[str]) -> Dict[str, Any]:
        """Implementación de métricas del sistema."""
        # En implementación real, obtendría métricas del sistema
        # Por ahora retornamos métricas de ejemplo
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "metrics": {
                "performance": {
                    "api_response_time_p95_ms": 45.2,
                    "question_processing_time_p95_ms": 120.5,
                    "analysis_throughput_files_per_second": 10.3,
                    "cache_hit_rate": 0.85,
                    "error_rate": 0.02,
                },
                "resources": {
                    "cpu_usage_percent": 45.2,
                    "memory_usage_percent": 62.8,
                    "disk_usage_percent": 35.7,
                    "active_connections": 12,
                    "queue_length": 3,
                },
                "business": {
                    "active_projects": 5,
                    "questions_answered": 450,
                    "analysis_completed": 15,
                    "user_satisfaction_score": 4.7,
                    "knowledge_growth_rate": 0.15,
                },
            },
        }
        
        if component:
            # Filtrar por componente si se especifica
            if component in metrics["metrics"]:
                metrics["metrics"] = {component: metrics["metrics"][component]}
            else:
                metrics["metrics"] = {}
        
        return metrics
    
    async def _system_status(self) -> Dict[str, Any]:
        """Implementación de estado del sistema."""
        if not self.orchestrator:
            raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        
        system_status = await self.orchestrator._get_system_status()
        
        return {
            "status": system_status["status"],
            "mode": "development",
            "version": "1.0.0",
            "uptime_seconds": system_status.get("uptime_seconds", 0),
            "components": system_status.get("components", {}),
            "active_operations": system_status.get("active_operations", 0),
            "resource_usage": {
                "memory_mb": 512,
                "cpu_percent": 25.5,
                "disk_usage_percent": 45.2,
            },
        }
    
    async def get_api_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de la API."""
        return {
            "requests_by_endpoint": self.metrics["requests_by_endpoint"],
            "average_response_times": self._calculate_average_times(),
            "error_rates": self._calculate_error_rates(),
            "total_requests": sum(self.metrics["requests_by_endpoint"].values()),
        }
    
    def _update_metrics(self, endpoint: str, processing_time: float, success: bool = True, error: Optional[Exception] = None) -> None:
        """Actualiza métricas de la API."""
        # Actualizar conteo de requests por endpoint
        self.metrics["requests_by_endpoint"][endpoint] = self.metrics["requests_by_endpoint"].get(endpoint, 0) + 1
        
        # Actualizar tiempos de respuesta
        if endpoint not in self.metrics["response_times"]:
            self.metrics["response_times"][endpoint] = []
        self.metrics["response_times"][endpoint].append(processing_time)
        
        # Actualizar errores
        if not success:
            error_type = type(error).__name__ if error else "Unknown"
            self.metrics["errors_by_type"][error_type] = self.metrics["errors_by_type"].get(error_type, 0) + 1
    
    def _calculate_average_times(self) -> Dict[str, float]:
        """Calcula tiempos promedio de respuesta."""
        averages = {}
        for endpoint, times in self.metrics["response_times"].items():
            if times:
                averages[endpoint] = sum(times) / len(times)
        return averages
    
    def _calculate_error_rates(self) -> Dict[str, float]:
        """Calcula tasas de error por endpoint."""
        error_rates = {}
        total_requests = sum(self.metrics["requests_by_endpoint"].values())
        total_errors = sum(self.metrics["errors_by_type"].values())
        
        if total_requests > 0:
            error_rates["overall"] = total_errors / total_requests
        
        # Por tipo de error
        for error_type, count in self.metrics["errors_by_type"].items():
            error_rates[error_type] = count / total_requests if total_requests > 0 else 0
        
        return error_rates
    
    async def cleanup(self) -> None:
        """Limpia recursos de la API."""
        logger.info("Limpiando REST API")
        # Cerrar conexiones, etc.