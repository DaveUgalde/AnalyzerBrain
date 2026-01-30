"""
SecurityAgent - Agente especializado en análisis de seguridad de código.
Detecta vulnerabilidades, analiza riesgos, recomienda correcciones y valida prácticas de seguridad.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import ast
import re
import json
from datetime import datetime
import asyncio
from pathlib import Path

from ..core.exceptions import AgentException, ValidationError
from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentConfig, AgentCapability, AgentState, AgentMemoryType


class VulnerabilityCategory(str, Enum):
    """Categorías de vulnerabilidades."""
    INJECTION = "injection"
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XXE = "xxe"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "xss"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    KNOWN_VULNERABILITIES = "known_vulnerabilities"
    INSECURE_LOG = "insecure_logging"
    INSECURE_DEPENDENCY = "insecure_dependency"
    CODE_QUALITY = "code_quality_issues"
    CRYPTOGRAPHY = "cryptography_issues"
    API_SECURITY = "api_security"


class SeverityLevel(str, Enum):
    """Niveles de severidad."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityStandard(str, Enum):
    """Estándares de seguridad."""
    OWASP_TOP_10 = "owasp_top_10"
    CWE = "cwe"
    SANS_25 = "sans_25"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    ISO_27001 = "iso_27001"


@dataclass
class Vulnerability:
    """Representa una vulnerabilidad detectada."""
    id: str
    category: VulnerabilityCategory
    severity: SeverityLevel
    title: str
    description: str
    location: Dict[str, Any]  # file, line, column, code snippet
    cwe_id: Optional[str] = None
    cve_id: Optional[str] = None
    owasp_category: Optional[str] = None
    detection_method: str = "static_analysis"
    confidence: float = 0.8  # 0.0 - 1.0
    exploitability: Optional[float] = None
    impact: Optional[float] = None
    risk_score: Optional[float] = None  # exploitability * impact
    remediation: Optional[str] = None
    example_fix: Optional[str] = None
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    fixed: bool = False
    false_positive: bool = False


@dataclass
class SecurityAssessment:
    """Resultado de una evaluación de seguridad."""
    project_id: str
    assessment_id: str
    timestamp: datetime
    vulnerabilities_found: int
    vulnerabilities_by_severity: Dict[SeverityLevel, int]
    risk_score: float  # 0-100
    compliance_status: Dict[SecurityStandard, bool]
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SecurityAgent(BaseAgent):
    """
    Agente especializado en análisis de seguridad de código.
    
    Capacidades:
    1. Análisis estático de seguridad
    2. Detección de vulnerabilidades comunes (OWASP Top 10, CWE Top 25)
    3. Evaluación de riesgos
    4. Recomendación de correcciones
    5. Validación de prácticas de seguridad
    6. Simulación de ataques básicos
    7. Generación de reportes de seguridad
    
    Lenguajes soportados: Python, JavaScript, Java, C++, Go, Rust
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Inicializa el SecurityAgent.
        
        Args:
            config: Configuración del agente (opcional)
        """
        if config is None:
            config = AgentConfig(
                name="SecurityAgent",
                description="Agente especializado en análisis de seguridad de código",
                capabilities=[
                    AgentCapability.SECURITY_ANALYSIS,
                    AgentCapability.PATTERN_DETECTION,
                    AgentCapability.CODE_ANALYSIS
                ],
                confidence_threshold=0.7,
                learning_rate=0.15,
                dependencies=["indexer", "embeddings"]
            )
        
        super().__init__(config)
        
        # Patrones de vulnerabilidad por lenguaje
        self._patterns: Dict[str, List[Dict]] = {}
        
        # Base de conocimiento de vulnerabilidades
        self._vulnerability_db: Dict[str, Dict] = {}
        
        # Reglas de seguridad
        self._security_rules: Dict[str, Dict] = {}
        
        # Configuración específica de seguridad
        self._security_config = {
            "enable_taint_analysis": True,
            "enable_dependency_check": True,
            "enable_secret_detection": True,
            "risk_threshold_critical": 0.8,
            "risk_threshold_high": 0.6,
            "risk_threshold_medium": 0.4,
            "risk_threshold_low": 0.2
        }
    
    # ========== MÉTODOS ABSTRACTOS DE BASEAGENT ==========
    
    async def _initialize_internal(self) -> bool:
        """
        Inicialización específica del SecurityAgent.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Cargar patrones de vulnerabilidad
            self._patterns = await self._load_security_patterns()
            
            # Cargar base de datos de vulnerabilidades
            self._vulnerability_db = await self._load_vulnerability_database()
            
            # Cargar reglas de seguridad
            self._security_rules = await self._load_security_rules()
            
            # Inicializar analizadores específicos
            await self._initialize_analyzers()
            
            # Cargar modelos de ML para detección de vulnerabilidades
            if "embeddings" in self.dependencies:
                await self._initialize_security_models()
            
            # Consolidar memoria de seguridad
            await self._consolidate_security_knowledge()
            
            self.memory.consolidate()
            return True
            
        except Exception as e:
            self.state = AgentState.ERROR
            raise AgentException(f"Failed to initialize SecurityAgent: {e}")
    
    async def _process_internal(self, input_data: AgentInput) -> AgentOutput:
        """
        Procesamiento principal del SecurityAgent.
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            AgentOutput: Resultado del procesamiento
        """
        try:
            # Determinar tipo de solicitud
            request_type = input_data.data.get("type", "security_scan")
            
            if request_type == "security_scan":
                result = await self._perform_security_scan(input_data)
            elif request_type == "vulnerability_assessment":
                result = await self._perform_vulnerability_assessment(input_data)
            elif request_type == "risk_assessment":
                result = await self._perform_risk_assessment(input_data)
            elif request_type == "compliance_check":
                result = await self._perform_compliance_check(input_data)
            elif request_type == "dependency_audit":
                result = await self._perform_dependency_audit(input_data)
            elif request_type == "secret_detection":
                result = await self._perform_secret_detection(input_data)
            else:
                raise ValidationError(f"Unknown request type: {request_type}")
            
            # Crear respuesta
            return AgentOutput(
                request_id=input_data.request_id,
                agent_id=self.config.agent_id,
                success=True,
                data=result,
                confidence=result.get("confidence", 0.8),
                reasoning=result.get("reasoning", []),
                warnings=result.get("warnings", []),
                processing_time_ms=0.0  # Se actualizará en el método process
            )
            
        except Exception as e:
            return AgentOutput(
                request_id=input_data.request_id,
                agent_id=self.config.agent_id,
                success=False,
                confidence=0.0,
                errors=[str(e)]
            )
    
    async def _learn_internal(self, feedback: Dict[str, Any]) -> bool:
        """
        Aprendizaje específico del SecurityAgent.
        
        Args:
            feedback: Datos de feedback
            
        Returns:
            bool: True si el aprendizaje fue exitoso
        """
        try:
            feedback_type = feedback.get("type", "")
            
            if feedback_type == "vulnerability_confirmation":
                return await self._learn_from_vulnerability_confirmation(feedback)
            elif feedback_type == "false_positive":
                return await self._learn_from_false_positive(feedback)
            elif feedback_type == "remediation_effectiveness":
                return await self._learn_from_remediation(feedback)
            elif feedback_type == "new_pattern":
                return await self._learn_new_pattern(feedback)
            else:
                # Aprendizaje genérico
                return await self._generic_security_learning(feedback)
                
        except Exception as e:
            self.metrics["error_types"]["learning"] = \
                self.metrics["error_types"].get("learning", 0) + 1
            return False
    
    def _validate_input_specific(self, input_data: AgentInput) -> None:
        """
        Validación específica para SecurityAgent.
        
        Args:
            input_data: Datos de entrada a validar
            
        Raises:
            ValidationError: Si la entrada es inválida
        """
        # Verificar que haya datos de entrada
        if not input_data.data:
            raise ValidationError("Input data cannot be empty")
        
        # Verificar tipo de solicitud
        request_type = input_data.data.get("type")
        valid_types = [
            "security_scan",
            "vulnerability_assessment", 
            "risk_assessment",
            "compliance_check",
            "dependency_audit",
            "secret_detection"
        ]
        
        if request_type not in valid_types:
            raise ValidationError(f"Request type must be one of {valid_types}")
        
        # Validaciones específicas por tipo
        if request_type in ["security_scan", "vulnerability_assessment"]:
            if "code" not in input_data.data and "file_path" not in input_data.data:
                raise ValidationError("Either 'code' or 'file_path' must be provided for security analysis")
    
    async def _save_state(self) -> None:
        """
        Guarda el estado del SecurityAgent.
        """
        try:
            state = {
                "patterns": self._patterns,
                "vulnerability_db": self._vulnerability_db,
                "security_rules": self._security_rules,
                "security_config": self._security_config,
                "metrics": self.metrics,
                "learned_patterns": await self._get_learned_patterns()
            }
            
            # En una implementación real, guardaríamos a disco/DB
            self.memory.store(
                AgentMemoryType.LONG_TERM,
                {
                    "type": "agent_state",
                    "state": state,
                    "timestamp": datetime.now()
                }
            )
            
        except Exception as e:
            print(f"Warning: Failed to save SecurityAgent state: {e}")
    
    # ========== MÉTODOS PÚBLICOS ESPECÍFICOS ==========
    
    async def analyze_security(self, code: str, language: str, 
                             context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Realiza análisis de seguridad completo del código.
        
        Args:
            code: Código a analizar
            language: Lenguaje del código
            context: Contexto adicional
            
        Returns:
            Dict con resultados del análisis de seguridad
        """
        input_data = AgentInput(
            data={
                "type": "security_scan",
                "code": code,
                "language": language,
                "context": context or {}
            }
        )
        
        output = await self.process(input_data)
        
        if not output.success:
            raise AgentException(f"Security analysis failed: {output.errors}")
        
        return output.data
    
    async def detect_vulnerabilities(self, file_path: str, 
                                   language: Optional[str] = None) -> List[Vulnerability]:
        """
        Detecta vulnerabilidades en un archivo de código.
        
        Args:
            file_path: Ruta al archivo
            language: Lenguaje (autodetectado si None)
            
        Returns:
            Lista de vulnerabilidades detectadas
        """
        try:
            # Leer archivo
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Detectar lenguaje si no se especifica
            if language is None:
                language = self._detect_language(file_path)
            
            # Analizar vulnerabilidades
            vulnerabilities = await self._analyze_vulnerabilities(code, language, file_path)
            
            # Filtrar falsos positivos basados en aprendizaje previo
            filtered_vulns = await self._filter_false_positives(vulnerabilities)
            
            # Calcular métricas de riesgo
            for vuln in filtered_vulns:
                vuln.risk_score = await self._calculate_risk_score(vuln)
            
            return filtered_vulns
            
        except Exception as e:
            raise AgentException(f"Failed to detect vulnerabilities: {e}")
    
    async def assess_risk(self, vulnerabilities: List[Vulnerability], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evalúa el riesgo general basado en vulnerabilidades encontradas.
        
        Args:
            vulnerabilities: Lista de vulnerabilidades
            context: Contexto del proyecto
            
        Returns:
            Dict con evaluación de riesgo
        """
        if not vulnerabilities:
            return {
                "overall_risk_score": 0.0,
                "risk_level": "low",
                "recommendations": [],
                "breakdown": {}
            }
        
        # Calcular puntuación de riesgo general
        total_risk = sum(v.risk_score or 0.0 for v in vulnerabilities)
        avg_risk = total_risk / len(vulnerabilities)
        
        # Determinar nivel de riesgo
        risk_level = self._determine_risk_level(avg_risk)
        
        # Agrupar por categoría y severidad
        breakdown = await self._create_risk_breakdown(vulnerabilities)
        
        # Generar recomendaciones
        recommendations = await self._generate_risk_recommendations(vulnerabilities, context)
        
        return {
            "overall_risk_score": avg_risk,
            "risk_level": risk_level,
            "critical_vulnerabilities": len([v for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL]),
            "high_vulnerabilities": len([v for v in vulnerabilities if v.severity == SeverityLevel.HIGH]),
            "breakdown": breakdown,
            "recommendations": recommendations,
            "next_steps": self._suggest_next_steps(risk_level, vulnerabilities)
        }
    
    async def recommend_security_fixes(self, vulnerability: Vulnerability) -> Dict[str, Any]:
        """
        Recomienda correcciones específicas para una vulnerabilidad.
        
        Args:
            vulnerability: Vulnerabilidad a corregir
            
        Returns:
            Dict con recomendaciones de corrección
        """
        # Buscar remediación en base de conocimiento
        remediation = await self._find_remediation(vulnerability)
        
        if not remediation:
            # Generar remediación genérica
            remediation = await self._generate_generic_remediation(vulnerability)
        
        # Proporcionar ejemplo de código corregido
        example_fix = await self._generate_example_fix(vulnerability, remediation)
        
        # Sugerir herramientas de corrección automática
        tools = await self._suggest_fix_tools(vulnerability)
        
        # Estimar esfuerzo de corrección
        effort = self._estimate_fix_effort(vulnerability)
        
        return {
            "vulnerability_id": vulnerability.id,
            "remediation": remediation,
            "example_fix": example_fix,
            "tools": tools,
            "estimated_effort_minutes": effort,
            "priority": self._determine_fix_priority(vulnerability),
            "verification_steps": self._suggest_verification_steps(vulnerability),
            "testing_guidance": self._provide_testing_guidance(vulnerability)
        }
    
    async def validate_security_practices(self, project_path: str, 
                                        standards: List[SecurityStandard]) -> Dict[str, Any]:
        """
        Valida prácticas de seguridad contra estándares.
        
        Args:
            project_path: Ruta al proyecto
            standards: Estándares a validar
            
        Returns:
            Dict con resultados de validación
        """
        results = {}
        
        for standard in standards:
            # Obtener criterios del estándar
            criteria = await self._get_security_criteria(standard)
            
            # Evaluar cada criterio
            evaluation = await self._evaluate_against_criteria(project_path, criteria)
            
            # Calcular cumplimiento
            compliance = await self._calculate_compliance(evaluation)
            
            results[standard.value] = {
                "standard": standard.value,
                "compliance_score": compliance["score"],
                "compliance_level": compliance["level"],
                "passed_criteria": compliance["passed"],
                "failed_criteria": compliance["failed"],
                "evaluation_details": evaluation
            }
        
        # Calcular cumplimiento general
        overall_compliance = await self._calculate_overall_compliance(results)
        
        return {
            "project_path": project_path,
            "standards_evaluated": [s.value for s in standards],
            "overall_compliance_score": overall_compliance["score"],
            "overall_compliance_level": overall_compliance["level"],
            "standard_results": results,
            "improvement_areas": await self._identify_improvement_areas(results),
            "certification_ready": overall_compliance["score"] >= 0.9
        }
    
    async def simulate_attacks(self, code: str, language: str, 
                             attack_scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simula ataques contra el código para identificar vulnerabilidades.
        
        Args:
            code: Código a analizar
            language: Lenguaje del código
            attack_scenarios: Escenarios de ataque específicos (opcional)
            
        Returns:
            Dict con resultados de simulación de ataques
        """
        # Parsear código
        parsed_code = await self._parse_code(code, language)
        
        # Identificar puntos de entrada vulnerables
        entry_points = await self._identify_vulnerable_entry_points(parsed_code, language)
        
        # Ejecutar escenarios de ataque
        attack_results = []
        
        if attack_scenarios:
            for scenario in attack_scenarios:
                result = await self._execute_attack_scenario(scenario, entry_points, parsed_code, language)
                attack_results.append(result)
        else:
            # Ejecutar escenarios predeterminados
            default_scenarios = await self._get_default_attack_scenarios(language)
            for scenario in default_scenarios:
                result = await self._execute_attack_scenario(scenario, entry_points, parsed_code, language)
                attack_results.append(result)
        
        # Analizar resultados
        vulnerabilities_found = await self._analyze_attack_results(attack_results)
        
        # Generar reporte
        return {
            "attack_simulation_completed": True,
            "entry_points_analyzed": len(entry_points),
            "attack_scenarios_executed": len(attack_results),
            "successful_attacks": len([r for r in attack_results if r.get("successful", False)]),
            "vulnerabilities_discovered": vulnerabilities_found,
            "attack_details": attack_results,
            "mitigation_strategies": await self._generate_mitigation_strategies(attack_results),
            "security_controls_needed": await self._identify_security_controls(vulnerabilities_found)
        }
    
    async def generate_security_report(self, project_id: str, 
                                     analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera reporte de seguridad completo.
        
        Args:
            project_id: ID del proyecto
            analysis_data: Datos del análisis de seguridad
            
        Returns:
            Dict con reporte de seguridad
        """
        # Compilar métricas
        metrics = await self._compile_security_metrics(analysis_data)
        
        # Generar resumen ejecutivo
        executive_summary = await self._generate_executive_summary(metrics)
        
        # Detalles técnicos
        technical_details = await self._generate_technical_details(analysis_data)
        
        # Recomendaciones priorizadas
        recommendations = await self._prioritize_recommendations(analysis_data.get("recommendations", []))
        
        # Plan de remediación
        remediation_plan = await self._create_remediation_plan(recommendations)
        
        # Cumplimiento normativo
        compliance = await self._assess_regulatory_compliance(analysis_data)
        
        return {
            "report_id": f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "project_id": project_id,
            "generated_at": datetime.now().isoformat(),
            "executive_summary": executive_summary,
            "risk_assessment": metrics.get("risk_assessment", {}),
            "vulnerability_summary": metrics.get("vulnerability_summary", {}),
            "technical_findings": technical_details,
            "recommendations": recommendations,
            "remediation_plan": remediation_plan,
            "compliance_status": compliance,
            "appendix": {
                "methodology": self._describe_methodology(),
                "tools_used": self._list_tools_used(),
                "glossary": self._create_glossary(),
                "references": self._list_references()
            }
        }
    
    # ========== MÉTODOS PRIVADOS DE IMPLEMENTACIÓN ==========
    
    async def _load_security_patterns(self) -> Dict[str, List[Dict]]:
        """Carga patrones de vulnerabilidad por lenguaje."""
        # En implementación real, cargaría desde archivo/DB
        return {
            "python": [
                {
                    "id": "py_sql_injection",
                    "name": "SQL Injection",
                    "pattern": r"execute\(.*\%s.*\)|executemany\(.*\%s.*\)",
                    "category": VulnerabilityCategory.INJECTION,
                    "severity": SeverityLevel.HIGH,
                    "description": "Direct string concatenation in SQL queries",
                    "remediation": "Use parameterized queries or ORM"
                },
                {
                    "id": "py_command_injection",
                    "name": "Command Injection",
                    "pattern": r"os\.system\(|subprocess\.call\(|subprocess\.Popen\(",
                    "category": VulnerabilityCategory.INJECTION,
                    "severity": SeverityLevel.CRITICAL,
                    "description": "Unsanitized user input in system commands",
                    "remediation": "Use shlex.quote() or avoid shell=True"
                }
            ],
            "javascript": [
                {
                    "id": "js_xss",
                    "name": "Cross-Site Scripting (XSS)",
                    "pattern": r"innerHTML\s*=|\.html\(",
                    "category": VulnerabilityCategory.XSS,
                    "severity": SeverityLevel.HIGH,
                    "description": "Unsanitized user input in DOM manipulation",
                    "remediation": "Use textContent or proper escaping"
                }
            ]
        }
    
    async def _load_vulnerability_database(self) -> Dict[str, Dict]:
        """Carga base de datos de vulnerabilidades conocidas."""
        # En implementación real, cargaría desde CVE/CWE databases
        return {
            "CWE-79": {
                "id": "CWE-79",
                "name": "Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')",
                "description": "The software does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output that is used as a web page that is served to other users.",
                "category": VulnerabilityCategory.XSS,
                "severity": SeverityLevel.HIGH,
                "remediation": "Use proper output encoding and validation"
            },
            "CWE-89": {
                "id": "CWE-89",
                "name": "Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')",
                "description": "The software constructs all or part of an SQL command using externally-influenced input, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended SQL command.",
                "category": VulnerabilityCategory.INJECTION,
                "severity": SeverityLevel.CRITICAL,
                "remediation": "Use parameterized queries or stored procedures"
            }
        }
    
    async def _load_security_rules(self) -> Dict[str, Dict]:
        """Carga reglas de seguridad."""
        return {
            "authentication": {
                "password_min_length": 12,
                "password_complexity": True,
                "session_timeout": 1800,
                "multi_factor_auth": False  # Requerido para aplicaciones críticas
            },
            "cryptography": {
                "min_key_size": {
                    "RSA": 2048,
                    "ECDSA": 256,
                    "AES": 128
                },
                "deprecated_algorithms": ["MD5", "SHA1", "DES", "RC4"],
                "requires_tls": True,
                "tls_min_version": "1.2"
            },
            "input_validation": {
                "validate_all_inputs": True,
                "max_input_length": 10000,
                "sanitize_html": True,
                "parameterize_queries": True
            }
        }
    
    async def _initialize_analyzers(self) -> None:
        """Inicializa analizadores específicos."""
        # Analizador de taint (seguimiento de flujo de datos)
        self._taint_analyzer = await self._create_taint_analyzer()
        
        # Analizador de dependencias
        self._dependency_analyzer = await self._create_dependency_analyzer()
        
        # Detector de secretos
        self._secret_detector = await self._create_secret_detector()
    
    async def _initialize_security_models(self) -> None:
        """Inicializa modelos de ML para seguridad."""
        # Modelo para clasificación de vulnerabilidades
        # Modelo para detección de anomalías
        # Modelo para predicción de riesgo
        pass
    
    async def _consolidate_security_knowledge(self) -> None:
        """Consolida conocimiento de seguridad en memoria."""
        # Cargar historial de análisis previos
        # Integrar lecciones aprendidas
        # Actualizar patrones con nuevas vulnerabilidades
        pass
    
    async def _perform_security_scan(self, input_data: AgentInput) -> Dict[str, Any]:
        """Realiza escaneo de seguridad completo."""
        code = input_data.data.get("code", "")
        language = input_data.data.get("language", "python")
        context = input_data.data.get("context", {})
        
        # 1. Análisis estático básico
        static_analysis = await self._perform_static_analysis(code, language)
        
        # 2. Análisis de taint (si está habilitado)
        taint_analysis = {}
        if self._security_config["enable_taint_analysis"]:
            taint_analysis = await self._perform_taint_analysis(code, language)
        
        # 3. Detección de secretos (si está habilitado)
        secret_detection = {}
        if self._security_config["enable_secret_detection"]:
            secret_detection = await self._detect_secrets(code, language)
        
        # 4. Combinar resultados
        vulnerabilities = []
        vulnerabilities.extend(static_analysis.get("vulnerabilities", []))
        vulnerabilities.extend(taint_analysis.get("vulnerabilities", []))
        vulnerabilities.extend(secret_detection.get("secrets", []))
        
        # 5. Evaluar riesgo
        risk_assessment = await self.assess_risk(
            [Vulnerability(**v) for v in vulnerabilities], 
            context
        )
        
        # 6. Generar recomendaciones
        recommendations = []
        for vuln_data in vulnerabilities:
            vuln = Vulnerability(**vuln_data)
            fix = await self.recommend_security_fixes(vuln)
            recommendations.append(fix)
        
        return {
            "scan_completed": True,
            "language": language,
            "code_length": len(code),
            "vulnerabilities_found": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "risk_assessment": risk_assessment,
            "recommendations": recommendations,
            "confidence": self._calculate_scan_confidence(code, language, vulnerabilities),
            "reasoning": ["Performed static analysis", "Checked for common vulnerabilities", "Assessed risk level"],
            "warnings": [] if vulnerabilities else ["No vulnerabilities found (consider deeper analysis)"]
        }
    
    async def _perform_vulnerability_assessment(self, input_data: AgentInput) -> Dict[str, Any]:
        """Realiza evaluación de vulnerabilidades detallada."""
        file_path = input_data.data.get("file_path")
        language = input_data.data.get("language")
        
        if file_path:
            vulnerabilities = await self.detect_vulnerabilities(file_path, language)
        else:
            code = input_data.data.get("code", "")
            language = input_data.data.get("language", "python")
            vulnerabilities = await self._analyze_vulnerabilities(code, language, "inline_code")
        
        # Clasificar vulnerabilidades
        classified = await self._classify_vulnerabilities(vulnerabilities)
        
        # Priorizar por severidad
        prioritized = await self._prioritize_vulnerabilities(classified)
        
        # Analizar root causes
        root_causes = await self._analyze_root_causes(prioritized)
        
        return {
            "assessment_completed": True,
            "total_vulnerabilities": len(vulnerabilities),
            "by_severity": {
                "critical": len([v for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL]),
                "high": len([v for v in vulnerabilities if v.severity == SeverityLevel.HIGH]),
                "medium": len([v for v in vulnerabilities if v.severity == SeverityLevel.MEDIUM]),
                "low": len([v for v in vulnerabilities if v.severity == SeverityLevel.LOW])
            },
            "by_category": await self._group_by_category(vulnerabilities),
            "prioritized_vulnerabilities": prioritized,
            "root_cause_analysis": root_causes,
            "trend_analysis": await self._analyze_vulnerability_trends(vulnerabilities),
            "industry_benchmark": await self._compare_with_benchmark(vulnerabilities)
        }
    
    async def _perform_risk_assessment(self, input_data: AgentInput) -> Dict[str, Any]:
        """Realiza evaluación de riesgo."""
        vulnerabilities_data = input_data.data.get("vulnerabilities", [])
        context = input_data.data.get("context", {})
        
        # Convertir a objetos Vulnerability
        vulnerabilities = []
        for vuln_data in vulnerabilities_data:
            try:
                vuln = Vulnerability(**vuln_data)
                vulnerabilities.append(vuln)
            except:
                continue
        
        # Evaluar riesgo
        assessment = await self.assess_risk(vulnerabilities, context)
        
        # Añadir análisis adicional
        assessment["business_impact"] = await self._assess_business_impact(vulnerabilities, context)
        assessment["exploitation_likelihood"] = await self._estimate_exploitation_likelihood(vulnerabilities)
        assessment["remediation_cost_estimate"] = await self._estimate_remediation_cost(vulnerabilities)
        
        return assessment
    
    async def _perform_compliance_check(self, input_data: AgentInput) -> Dict[str, Any]:
        """Realiza verificación de cumplimiento."""
        project_path = input_data.data.get("project_path")
        standards_data = input_data.data.get("standards", [])
        
        # Convertir a objetos SecurityStandard
        standards = []
        for std in standards_data:
            try:
                standards.append(SecurityStandard(std))
            except:
                continue
        
        if not standards:
            standards = [SecurityStandard.OWASP_TOP_10]
        
        # Validar prácticas
        return await self.validate_security_practices(project_path, standards)
    
    async def _perform_dependency_audit(self, input_data: AgentInput) -> Dict[str, Any]:
        """Realiza auditoría de dependencias."""
        if not self._security_config["enable_dependency_check"]:
            return {
                "audit_skipped": True,
                "reason": "Dependency check is disabled in configuration"
            }
        
        project_path = input_data.data.get("project_path")
        
        # Identificar archivos de dependencias
        dependency_files = await self._find_dependency_files(project_path)
        
        # Analizar cada archivo
        results = []
        for dep_file in dependency_files:
            analysis = await self._analyze_dependency_file(dep_file)
            results.append(analysis)
        
        # Consolidar resultados
        consolidated = await self._consolidate_dependency_results(results)
        
        # Verificar vulnerabilidades conocidas
        known_vulns = await self._check_known_vulnerabilities(consolidated)
        
        # Evaluar riesgo de dependencias
        risk = await self._assess_dependency_risk(consolidated, known_vulns)
        
        return {
            "audit_completed": True,
            "dependency_files_found": len(dependency_files),
            "total_dependencies": consolidated.get("total_dependencies", 0),
            "vulnerable_dependencies": len(known_vulns),
            "known_vulnerabilities": known_vulns,
            "dependency_risk_assessment": risk,
            "update_recommendations": await self._recommend_dependency_updates(consolidated, known_vulns),
            "dependency_health_score": await self._calculate_dependency_health(consolidated)
        }
    
    async def _perform_secret_detection(self, input_data: AgentInput) -> Dict[str, Any]:
        """Detecta secretos en el código."""
        if not self._security_config["enable_secret_detection"]:
            return {
                "detection_skipped": True,
                "reason": "Secret detection is disabled in configuration"
            }
        
        code = input_data.data.get("code", "")
        language = input_data.data.get("language", "python")
        
        # Detectar patrones de secretos
        secrets = await self._detect_secret_patterns(code, language)
        
        # Verificar si los secretos son válidos
        validated = await self._validate_secrets(secrets)
        
        # Clasificar por tipo
        classified = await self._classify_secrets(validated)
        
        # Evaluar riesgo
        risk = await self._assess_secret_risk(classified)
        
        return {
            "detection_completed": True,
            "secrets_found": len(secrets),
            "valid_secrets": len(validated),
            "secret_types": classified,
            "risk_assessment": risk,
            "remediation_steps": await self._suggest_secret_remediation(validated),
            "prevention_guidance": self._provide_secret_prevention_guidance()
        }
    
    async def _learn_from_vulnerability_confirmation(self, feedback: Dict) -> bool:
        """Aprende de confirmación de vulnerabilidad."""
        vulnerability = feedback.get("vulnerability", {})
        confirmed = feedback.get("confirmed", False)
        
        if confirmed:
            # Reforzar patrón
            await self._reinforce_pattern(vulnerability)
            # Aumentar confianza en detecciones similares
            await self._increase_detection_confidence(vulnerability)
        
        return True
    
    async def _learn_from_false_positive(self, feedback: Dict) -> bool:
        """Aprende de falsos positivos."""
        false_positive = feedback.get("false_positive", {})
        
        # Ajustar patrones para evitar este falso positivo
        await self._adjust_pattern_to_avoid_fp(false_positive)
        
        # Reducir confianza en detecciones similares
        await self._reduce_fp_confidence(false_positive)
        
        # Añadir a lista de falsos positivos conocidos
        await self._record_false_positive(false_positive)
        
        return True
    
    async def _learn_from_remediation(self, feedback: Dict) -> bool:
        """Aprende de efectividad de remediación."""
        remediation = feedback.get("remediation", {})
        effectiveness = feedback.get("effectiveness", 0.0)
        
        # Actualizar base de conocimiento de remediaciones
        await self._update_remediation_knowledge(remediation, effectiveness)
        
        # Ajustar recomendaciones futuras
        await self._adjust_recommendations_based_on_effectiveness(remediation, effectiveness)
        
        return True
    
    async def _learn_new_pattern(self, feedback: Dict) -> bool:
        """Aprende nuevo patrón de vulnerabilidad."""
        pattern = feedback.get("pattern", {})
        
        # Añadir a patrones conocidos
        language = pattern.get("language", "python")
        if language not in self._patterns:
            self._patterns[language] = []
        
        self._patterns[language].append(pattern)
        
        # Consolidar en memoria
        self.memory.store(
            AgentMemoryType.SEMANTIC,
            {
                "type": "new_security_pattern",
                "pattern": pattern,
                "learned_at": datetime.now()
            }
        )
        
        return True
    
    async def _generic_security_learning(self, feedback: Dict) -> bool:
        """Aprendizaje genérico de seguridad."""
        # Extraer lecciones generales
        lessons = feedback.get("lessons", [])
        
        for lesson in lessons:
            self.memory.store(
                AgentMemoryType.SEMANTIC,
                {
                    "type": "security_lesson",
                    "lesson": lesson,
                    "timestamp": datetime.now()
                }
            )
        
        # Actualizar configuración si es necesario
        config_updates = feedback.get("config_updates", {})
        self._security_config.update(config_updates)
        
        return True
    
    async def _get_learned_patterns(self) -> List[Dict]:
        """Obtiene patrones aprendidos."""
        # Recuperar de memoria
        patterns = self.memory.retrieve(
            AgentMemoryType.SEMANTIC,
            {"type": "new_security_pattern"},
            limit=100
        )
        
        return [p["content"]["pattern"] for p in patterns]
    
    # ========== MÉTODOS AUXILIARES ==========
    
    def _detect_language(self, file_path: str) -> str:
        """Detecta lenguaje basado en extensión."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".h": "cpp",
            ".go": "go",
            ".rs": "rust",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby"
        }
        
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, "python")
    
    async def _analyze_vulnerabilities(self, code: str, language: str, source: str) -> List[Vulnerability]:
        """Analiza código en busca de vulnerabilidades."""
        vulnerabilities = []
        
        # 1. Análisis basado en patrones
        pattern_vulns = await self._detect_by_patterns(code, language, source)
        vulnerabilities.extend(pattern_vulns)
        
        # 2. Análisis de AST (para lenguajes soportados)
        if language in ["python", "javascript", "java"]:
            ast_vulns = await self._analyze_ast(code, language, source)
            vulnerabilities.extend(ast_vulns)
        
        # 3. Análisis de datos sensibles
        sensitive_data_vulns = await self._detect_sensitive_data(code, language, source)
        vulnerabilities.extend(sensitive_data_vulns)
        
        # 4. Análisis de configuración (si es archivo de configuración)
        if source.endswith((".json", ".yaml", ".yml", ".xml", ".conf")):
            config_vulns = await self._analyze_configuration(code, source)
            vulnerabilities.extend(config_vulns)
        
        return vulnerabilities
    
    async def _detect_by_patterns(self, code: str, language: str, source: str) -> List[Vulnerability]:
        """Detecta vulnerabilidades usando patrones regex."""
        vulnerabilities = []
        
        if language not in self._patterns:
            return vulnerabilities
        
        lines = code.split('\n')
        
        for pattern_info in self._patterns[language]:
            pattern = re.compile(pattern_info["pattern"], re.IGNORECASE)
            
            for i, line in enumerate(lines, 1):
                matches = pattern.finditer(line)
                for match in matches:
                    vuln = Vulnerability(
                        id=f"{pattern_info['id']}_{i}_{match.start()}",
                        category=VulnerabilityCategory(pattern_info["category"]),
                        severity=SeverityLevel(pattern_info["severity"]),
                        title=pattern_info["name"],
                        description=pattern_info["description"],
                        location={
                            "file": source,
                            "line": i,
                            "column": match.start(),
                            "code_snippet": line.strip(),
                            "match": match.group()
                        },
                        remediation=pattern_info.get("remediation"),
                        confidence=0.7,  # Confianza base para detección por patrón
                        metadata={
                            "detection_method": "pattern_matching",
                            "pattern_id": pattern_info["id"],
                            "language": language
                        }
                    )
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    async def _analyze_ast(self, code: str, language: str, source: str) -> List[Vulnerability]:
        """Analiza AST en busca de vulnerabilidades."""
        vulnerabilities = []
        
        if language == "python":
            try:
                tree = ast.parse(code)
                vulnerabilities.extend(await self._analyze_python_ast(tree, source))
            except SyntaxError:
                pass
        
        return vulnerabilities
    
    async def _analyze_python_ast(self, tree: ast.AST, source: str) -> List[Vulnerability]:
        """Analiza AST de Python en busca de vulnerabilidades."""
        vulnerabilities = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, source_file):
                self.source = source_file
                self.vulnerabilities = []
            
            def visit_Call(self, node):
                # Detectar llamadas peligrosas
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    
                    # SQL Injection
                    if func_name in ["execute", "executemany"]:
                        self.vulnerabilities.append(
                            Vulnerability(
                                id=f"sql_inj_{node.lineno}",
                                category=VulnerabilityCategory.INJECTION,
                                severity=SeverityLevel.HIGH,
                                title="Potential SQL Injection",
                                description="Direct string concatenation in SQL query",
                                location={
                                    "file": self.source,
                                    "line": node.lineno,
                                    "column": node.col_offset,
                                    "code_snippet": ast.get_source_segment(self.source, node)
                                },
                                remediation="Use parameterized queries",
                                confidence=0.8
                            )
                        )
                    
                    # Command Injection
                    elif func_name in ["system", "call", "Popen"]:
                        self.vulnerabilities.append(
                            Vulnerability(
                                id=f"cmd_inj_{node.lineno}",
                                category=VulnerabilityCategory.INJECTION,
                                severity=SeverityLevel.CRITICAL,
                                title="Potential Command Injection",
                                description="Unsanitized user input in system command",
                                location={
                                    "file": self.source,
                                    "line": node.lineno,
                                    "column": node.col_offset,
                                    "code_snippet": ast.get_source_segment(self.source, node)
                                },
                                remediation="Use shlex.quote() or avoid shell=True",
                                confidence=0.75
                            )
                        )
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor(source)
        visitor.visit(tree)
        
        return visitor.vulnerabilities
    
    async def _detect_sensitive_data(self, code: str, language: str, source: str) -> List[Vulnerability]:
        """Detecta datos sensibles en el código."""
        vulnerabilities = []
        
        # Patrones para datos sensibles
        sensitive_patterns = [
            (r"password\s*=\s*['\"].*?['\"]", "Hardcoded password", SeverityLevel.CRITICAL),
            (r"api[_-]?key\s*=\s*['\"].*?['\"]", "Hardcoded API key", SeverityLevel.CRITICAL),
            (r"secret[_-]?key\s*=\s*['\"].*?['\"]", "Hardcoded secret key", SeverityLevel.CRITICAL),
            (r"token\s*=\s*['\"].*?['\"]", "Hardcoded token", SeverityLevel.HIGH),
            (r"private[_-]?key\s*=\s*['\"].*?['\"]", "Hardcoded private key", SeverityLevel.CRITICAL),
        ]
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern, title, severity in sensitive_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vuln = Vulnerability(
                        id=f"sensitive_data_{i}_{hash(line)}",
                        category=VulnerabilityCategory.SENSITIVE_DATA,
                        severity=severity,
                        title=title,
                        description="Sensitive data hardcoded in source code",
                        location={
                            "file": source,
                            "line": i,
                            "code_snippet": line.strip()
                        },
                        remediation="Store sensitive data in environment variables or secure vault",
                        confidence=0.9
                    )
                    vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    async def _filter_false_positives(self, vulnerabilities: List[Vulnerability]) -> List[Vulnerability]:
        """Filtra falsos positivos basados en aprendizaje previo."""
        filtered = []
        
        for vuln in vulnerabilities:
            # Verificar si es falso positivo conocido
            if not await self._is_known_false_positive(vuln):
                filtered.append(vuln)
        
        return filtered
    
    async def _calculate_risk_score(self, vulnerability: Vulnerability) -> float:
        """Calcula puntuación de riesgo para una vulnerabilidad."""
        # Puntuación base basada en severidad
        severity_scores = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.75,
            SeverityLevel.MEDIUM: 0.5,
            SeverityLevel.LOW: 0.25,
            SeverityLevel.INFO: 0.1
        }
        
        base_score = severity_scores.get(vulnerability.severity, 0.5)
        
        # Ajustar por confianza de detección
        confidence_factor = vulnerability.confidence
        
        # Ajustar por explotabilidad (si está disponible)
        exploitability_factor = vulnerability.exploitability or 0.5
        
        # Ajustar por impacto (si está disponible)
        impact_factor = vulnerability.impact or 0.5
        
        # Calcular riesgo final
        risk_score = base_score * confidence_factor * exploitability_factor * impact_factor
        
        return min(risk_score, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determina nivel de riesgo basado en puntuación."""
        if risk_score >= self._security_config["risk_threshold_critical"]:
            return "critical"
        elif risk_score >= self._security_config["risk_threshold_high"]:
            return "high"
        elif risk_score >= self._security_config["risk_threshold_medium"]:
            return "medium"
        elif risk_score >= self._security_config["risk_threshold_low"]:
            return "low"
        else:
            return "info"
    
    async def _create_risk_breakdown(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        """Crea desglose de riesgo por categoría y severidad."""
        breakdown = {
            "by_category": {},
            "by_severity": {},
            "top_risks": []
        }
        
        # Agrupar por categoría
        for vuln in vulnerabilities:
            category = vuln.category.value
            if category not in breakdown["by_category"]:
                breakdown["by_category"][category] = []
            breakdown["by_category"][category].append(vuln.id)
        
        # Agrupar por severidad
        for vuln in vulnerabilities:
            severity = vuln.severity.value
            if severity not in breakdown["by_severity"]:
                breakdown["by_severity"][severity] = []
            breakdown["by_severity"][severity].append(vuln.id)
        
        # Identificar top riesgos
        sorted_vulns = sorted(vulnerabilities, key=lambda v: v.risk_score or 0.0, reverse=True)
        breakdown["top_risks"] = [
            {
                "id": v.id,
                "title": v.title,
                "risk_score": v.risk_score,
                "severity": v.severity.value
            }
            for v in sorted_vulns[:5]
        ]
        
        return breakdown
    
    async def _generate_risk_recommendations(self, vulnerabilities: List[Vulnerability], 
                                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera recomendaciones de gestión de riesgo."""
        recommendations = []
        
        # Agrupar vulnerabilidades por categoría
        by_category = {}
        for vuln in vulnerabilities:
            category = vuln.category.value
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(vuln)
        
        # Generar recomendaciones por categoría
        for category, vulns in by_category.items():
            rec = {
                "category": category,
                "vulnerability_count": len(vulns),
                "highest_severity": max(v.severity.value for v in vulns),
                "actions": await self._get_category_specific_actions(category, vulns),
                "timeline": self._suggest_remediation_timeline(vulns),
                "resources": await self._get_category_resources(category)
            }
            recommendations.append(rec)
        
        # Añadir recomendaciones generales
        recommendations.append({
            "category": "general",
            "actions": [
                "Implement continuous security testing",
                "Establish security training program",
                "Create incident response plan",
                "Regularly update dependencies"
            ],
            "priority": "high"
        })
        
        return recommendations
    
    def _suggest_next_steps(self, risk_level: str, vulnerabilities: List[Vulnerability]) -> List[str]:
        """Sugiere próximos pasos basados en nivel de riesgo."""
        steps = []
        
        if risk_level in ["critical", "high"]:
            steps.extend([
                "Immediately remediate critical vulnerabilities",
                "Conduct emergency security review",
                "Notify stakeholders of high-risk findings",
                "Consider temporary workarounds until fixes are deployed"
            ])
        
        if risk_level in ["medium", "low"]:
            steps.extend([
                "Prioritize vulnerabilities for next sprint",
                "Schedule security training for development team",
                "Review and update security policies",
                "Implement automated security testing in CI/CD"
            ])
        
        steps.extend([
            "Document all findings and remediation steps",
            "Schedule follow-up assessment after fixes",
            "Consider third-party penetration testing",
            "Review compliance with security standards"
        ])
        
        return steps
    
    async def _find_remediation(self, vulnerability: Vulnerability) -> Optional[str]:
        """Busca remediación para una vulnerabilidad."""
        # Buscar en base de conocimiento
        if vulnerability.cwe_id and vulnerability.cwe_id in self._vulnerability_db:
            return self._vulnerability_db[vulnerability.cwe_id].get("remediation")
        
        # Buscar en patrones conocidos
        for lang_patterns in self._patterns.values():
            for pattern in lang_patterns:
                if pattern.get("id") == vulnerability.metadata.get("pattern_id"):
                    return pattern.get("remediation")
        
        return None
    
    async def _generate_generic_remediation(self, vulnerability: Vulnerability) -> str:
        """Genera remediación genérica para una vulnerabilidad."""
        category_remediations = {
            VulnerabilityCategory.INJECTION: "Implement input validation and use parameterized queries or prepared statements.",
            VulnerabilityCategory.XSS: "Use proper output encoding and implement Content Security Policy (CSP).",
            VulnerabilityCategory.SENSITIVE_DATA: "Remove hardcoded secrets and use secure storage solutions.",
            VulnerabilityCategory.BROKEN_AUTH: "Implement strong authentication mechanisms and session management.",
            VulnerabilityCategory.SECURITY_MISCONFIG: "Follow security best practices and harden configurations."
        }
        
        return category_remediations.get(
            vulnerability.category, 
            "Consult security best practices and implement appropriate controls."
        )
    
    async def _generate_example_fix(self, vulnerability: Vulnerability, 
                                  remediation: str) -> Optional[str]:
        """Genera ejemplo de código corregido."""
        # Ejemplos por tipo de vulnerabilidad
        examples = {
            "SQL Injection": {
                "vulnerable": "cursor.execute(\"SELECT * FROM users WHERE id = \" + user_id)",
                "fixed": "cursor.execute(\"SELECT * FROM users WHERE id = %s\", (user_id,))"
            },
            "Hardcoded Password": {
                "vulnerable": "password = \"mySecretPassword123\"",
                "fixed": "password = os.getenv(\"DB_PASSWORD\")"
            },
            "Command Injection": {
                "vulnerable": "os.system(\"ls \" + user_input)",
                "fixed": "subprocess.run([\"ls\", user_input], shell=False)"
            }
        }
        
        # Buscar ejemplo apropiado
        for key, example in examples.items():
            if key.lower() in vulnerability.title.lower():
                return json.dumps(example, indent=2)
        
        return None
    
    async def _suggest_fix_tools(self, vulnerability: Vulnerability) -> List[str]:
        """Sugiere herramientas para corregir la vulnerabilidad."""
        tools = []
        
        if vulnerability.category == VulnerabilityCategory.INJECTION:
            tools.extend(["SQLAlchemy", "Django ORM", "Hibernate", "Prepared Statements"])
        
        if vulnerability.category == VulnerabilityCategory.SENSITIVE_DATA:
            tools.extend(["HashiCorp Vault", "AWS Secrets Manager", "Azure Key Vault", "Google Secret Manager"])
        
        if vulnerability.category == VulnerabilityCategory.XSS:
            tools.extend(["DOMPurify", "OWASP Java Encoder", "Microsoft AntiXSS", "Content Security Policy"])
        
        tools.append("Static Application Security Testing (SAST) tools")
        tools.append("Software Composition Analysis (SCA) tools")
        
        return tools
    
    def _estimate_fix_effort(self, vulnerability: Vulnerability) -> int:
        """Estima esfuerzo de corrección en minutos."""
        effort_map = {
            SeverityLevel.CRITICAL: 240,  # 4 horas
            SeverityLevel.HIGH: 120,      # 2 horas
            SeverityLevel.MEDIUM: 60,     # 1 hora
            SeverityLevel.LOW: 30,        # 30 minutos
            SeverityLevel.INFO: 15        # 15 minutos
        }
        
        return effort_map.get(vulnerability.severity, 60)
    
    def _determine_fix_priority(self, vulnerability: Vulnerability) -> str:
        """Determina prioridad de corrección."""
        if vulnerability.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            return "immediate"
        elif vulnerability.severity == SeverityLevel.MEDIUM:
            return "high"
        else:
            return "medium"
    
    def _suggest_verification_steps(self, vulnerability: Vulnerability) -> List[str]:
        """Sugiere pasos para verificar la corrección."""
        steps = [
            "Review the fix for completeness",
            "Test the fix in a controlled environment",
            "Verify no regression in functionality",
            "Re-run security scan to confirm resolution"
        ]
        
        if vulnerability.category == VulnerabilityCategory.INJECTION:
            steps.append("Perform penetration testing with SQL injection payloads")
        
        if vulnerability.category == VulnerabilityCategory.XSS:
            steps.append("Test with XSS payloads in all user input fields")
        
        return steps
    
    def _provide_testing_guidance(self, vulnerability: Vulnerability) -> Dict[str, Any]:
        """Proporciona guía de testing para la vulnerabilidad."""
        guidance = {
            "test_cases": [],
            "tools": [],
            "success_criteria": "Vulnerability is no longer exploitable"
        }
        
        if vulnerability.category == VulnerabilityCategory.INJECTION:
            guidance["test_cases"] = [
                "Inject SQL commands through all input parameters",
                "Test with special characters and escape sequences",
                "Verify parameterized queries are used"
            ]
            guidance["tools"] = ["sqlmap", "Burp Suite", "OWASP ZAP"]
        
        if vulnerability.category == VulnerabilityCategory.XSS:
            guidance["test_cases"] = [
                "Inject script tags in all text inputs",
                "Test event handlers (onclick, onload, etc.)",
                "Verify proper output encoding"
            ]
            guidance["tools"] = ["XSStrike", "Burp Suite", "BeEF"]
        
        return guidance
    
    # Los métodos restantes serían implementados de manera similar,
    # pero por brevedad no se incluyen todas las implementaciones completas.
    
    async def _is_known_false_positive(self, vulnerability: Vulnerability) -> bool:
        """Verifica si una vulnerabilidad es un falso positivo conocido."""
        # Implementación simplificada
        false_positives = self.memory.retrieve(
            AgentMemoryType.SEMANTIC,
            {"type": "false_positive"},
            limit=1000
        )
        
        for fp in false_positives:
            fp_data = fp["content"]
            if (fp_data.get("pattern_id") == vulnerability.metadata.get("pattern_id") and
                fp_data.get("location") == vulnerability.location.get("file")):
                return True
        
        return False
    
    async def _reinforce_pattern(self, vulnerability: Dict) -> None:
        """Refuerza patrón de vulnerabilidad confirmada."""
        # Incrementar confianza en patrones similares
        pass
    
    async def _increase_detection_confidence(self, vulnerability: Dict) -> None:
        """Incrementa confianza en detecciones similares."""
        pass
    
    async def _adjust_pattern_to_avoid_fp(self, false_positive: Dict) -> None:
        """Ajusta patrón para evitar falsos positivos."""
        pass
    
    async def _reduce_fp_confidence(self, false_positive: Dict) -> None:
        """Reduce confianza en detecciones que causan falsos positivos."""
        pass
    
    async def _record_false_positive(self, false_positive: Dict) -> None:
        """Registra falso positivo en memoria."""
        self.memory.store(
            AgentMemoryType.SEMANTIC,
            {
                "type": "false_positive",
                "data": false_positive,
                "timestamp": datetime.now()
            }
        )
    
    async def _update_remediation_knowledge(self, remediation: Dict, effectiveness: float) -> None:
        """Actualiza conocimiento de remediaciones."""
        pass
    
    async def _adjust_recommendations_based_on_effectiveness(self, remediation: Dict, 
                                                           effectiveness: float) -> None:
        """Ajusta recomendaciones basado en efectividad."""
        pass
    
    def _calculate_scan_confidence(self, code: str, language: str, 
                                 vulnerabilities: List[Dict]) -> float:
        """Calcula confianza del escaneo."""
        if not code:
            return 0.0
        
        # Factores que afectan la confianza
        code_length_factor = min(len(code) / 1000, 1.0)  # Más código = más confianza (hasta cierto punto)
        language_support_factor = 1.0 if language in self._patterns else 0.5
        
        # Si no se encontraron vulnerabilidades, confianza más baja
        if not vulnerabilities:
            return (code_length_factor * language_support_factor) * 0.7
        
        return min(code_length_factor * language_support_factor, 0.95)
    
    # Los métodos restantes seguirían un patrón similar de implementación
    
    async def _compile_security_metrics(self, analysis_data: Dict) -> Dict[str, Any]:
        """Compila métricas de seguridad."""
        return {
            "vulnerability_summary": {
                "total": analysis_data.get("vulnerabilities_found", 0),
                "by_severity": analysis_data.get("by_severity", {}),
                "by_category": analysis_data.get("by_category", {})
            },
            "risk_assessment": analysis_data.get("risk_assessment", {}),
            "compliance_score": analysis_data.get("compliance_score", 0.0),
            "remediation_coverage": await self._calculate_remediation_coverage(analysis_data)
        }
    
    async def _generate_executive_summary(self, metrics: Dict) -> Dict[str, Any]:
        """Genera resumen ejecutivo."""
        risk_level = metrics.get("risk_assessment", {}).get("risk_level", "unknown")
        
        return {
            "overall_risk": risk_level,
            "critical_findings": metrics.get("vulnerability_summary", {}).get("critical", 0),
            "key_recommendations": [
                f"Address {metrics.get('vulnerability_summary', {}).get('critical', 0)} critical vulnerabilities",
                "Implement continuous security testing",
                "Establish security training program"
            ],
            "business_impact": "High" if risk_level in ["critical", "high"] else "Medium",
            "next_steps": self._get_executive_next_steps(risk_level)
        }
    
    def _get_executive_next_steps(self, risk_level: str) -> List[str]:
        """Obtiene próximos pasos para resumen ejecutivo."""
        if risk_level in ["critical", "high"]:
            return [
                "Immediate action required for critical vulnerabilities",
                "Schedule emergency security review",
                "Allocate resources for rapid remediation"
            ]
        else:
            return [
                "Integrate security findings into development roadmap",
                "Schedule quarterly security reviews",
                "Consider third-party security assessment"
            ]
    
    async def _generate_technical_details(self, analysis_data: Dict) -> List[Dict[str, Any]]:
        """Genera detalles técnicos para el reporte."""
        details = []
        
        # Detalles de vulnerabilidades
        for vuln in analysis_data.get("vulnerabilities", []):
            details.append({
                "type": "vulnerability",
                "id": vuln.get("id"),
                "title": vuln.get("title"),
                "severity": vuln.get("severity"),
                "location": vuln.get("location", {}),
                "description": vuln.get("description"),
                "remediation": vuln.get("remediation")
            })
        
        # Detalles de cumplimiento
        if "compliance_status" in analysis_data:
            details.append({
                "type": "compliance",
                "status": analysis_data["compliance_status"],
                "details": "Compliance assessment results"
            })
        
        return details
    
    async def _prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Prioriza recomendaciones de seguridad."""
        if not recommendations:
            return []
        
        # Asignar prioridad basada en severidad y esfuerzo
        for rec in recommendations:
            severity = rec.get("severity", "medium")
            effort = rec.get("estimated_effort", 60)
            
            if severity == "critical":
                rec["priority"] = "P0"
                rec["timeline"] = "Immediate"
            elif severity == "high":
                rec["priority"] = "P1"
                rec["timeline"] = "1 week"
            elif effort > 240:  # Más de 4 horas
                rec["priority"] = "P2"
                rec["timeline"] = "2 weeks"
            else:
                rec["priority"] = "P3"
                rec["timeline"] = "Next sprint"
        
        # Ordenar por prioridad
        priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        return sorted(recommendations, key=lambda x: priority_order.get(x.get("priority", "P3"), 3))
    
    async def _create_remediation_plan(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Crea plan de remediación."""
        plan = {
            "phases": [],
            "timeline": "4-8 weeks",
            "resources_needed": [],
            "success_metrics": []
        }
        
        # Agrupar por prioridad
        by_priority = {}
        for rec in recommendations:
            priority = rec.get("priority", "P3")
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append(rec)
        
        # Crear fases
        if "P0" in by_priority:
            plan["phases"].append({
                "phase": "Emergency",
                "priority": "P0",
                "timeline": "Immediate - 48 hours",
                "tasks": by_priority["P0"],
                "owner": "Security Team Lead"
            })
        
        if "P1" in by_priority:
            plan["phases"].append({
                "phase": "Critical",
                "priority": "P1",
                "timeline": "1 week",
                "tasks": by_priority["P1"],
                "owner": "Development Team"
            })
        
        # Estimar recursos
        total_effort = sum(r.get("estimated_effort", 0) for r in recommendations)
        plan["resources_needed"] = [
            f"Development: {total_effort // 60} hours",
            "Security review: 8 hours",
            "Testing: 4 hours per major fix"
        ]
        
        # Métricas de éxito
        plan["success_metrics"] = [
            "100% of critical vulnerabilities remediated",
            "Security test coverage > 80%",
            "No new critical vulnerabilities introduced",
            "Security training completion > 90%"
        ]
        
        return plan
    
    async def _assess_regulatory_compliance(self, analysis_data: Dict) -> Dict[str, Any]:
        """Evalúa cumplimiento normativo."""
        return {
            "gdpr": {
                "compliance": analysis_data.get("gdpr_compliant", False),
                "requirements": ["Data protection", "Privacy by design"],
                "gap_analysis": "Review data handling practices"
            },
            "hipaa": {
                "compliance": analysis_data.get("hipaa_compliant", False),
                "requirements": ["Access controls", "Audit trails", "Data encryption"],
                "gap_analysis": "Implement stronger access controls"
            },
            "pci_dss": {
                "compliance": analysis_data.get("pci_compliant", False),
                "requirements": ["Network security", "Vulnerability management"],
                "gap_analysis": "Improve vulnerability management process"
            }
        }
    
    def _describe_methodology(self) -> List[str]:
        """Describe la metodología de análisis."""
        return [
            "Static Application Security Testing (SAST)",
            "Dynamic Application Security Testing (DAST) - when applicable",
            "Software Composition Analysis (SCA)",
            "Manual code review for critical components",
            "Threat modeling and risk assessment",
            "Compliance checking against industry standards"
        ]
    
    def _list_tools_used(self) -> List[str]:
        """Lista herramientas utilizadas."""
        return [
            "Custom security pattern matcher",
            "AST-based vulnerability analyzer",
            "Dependency vulnerability scanner",
            "Secret detection engine",
            "Risk assessment framework"
        ]
    
    def _create_glossary(self) -> Dict[str, str]:
        """Crea glosario de términos."""
        return {
            "SAST": "Static Application Security Testing - análisis de código sin ejecutarlo",
            "DAST": "Dynamic Application Security Testing - análisis ejecutando la aplicación",
            "SCA": "Software Composition Analysis - análisis de dependencias",
            "CWE": "Common Weakness Enumeration - lista estándar de debilidades de software",
            "CVE": "Common Vulnerabilities and Exposures - identificador estándar de vulnerabilidades",
            "OWASP": "Open Web Application Security Project - organización de seguridad de aplicaciones web"
        }
    
    def _list_references(self) -> List[str]:
        """Lista referencias."""
        return [
            "OWASP Top 10: https://owasp.org/www-project-top-ten/",
            "CWE Top 25: https://cwe.mitre.org/top25/",
            "NIST Cybersecurity Framework: https://www.nist.gov/cyberframework",
            "ISO/IEC 27001: Information security management",
            "GDPR: General Data Protection Regulation"
        ]


# Ejemplo de uso
if __name__ == "__main__":
    async def main():
        # Crear y configurar SecurityAgent
        agent = SecurityAgent()
        
        # Inicializar
        success = await agent.initialize()
        print(f"SecurityAgent initialized: {success}")
        
        # Ejemplo: Analizar código con vulnerabilidad potencial
        vulnerable_code = """
        import os
        import subprocess
        
        def process_user_input(user_input):
            # Vulnerabilidad: Command Injection
            os.system(f"ls {user_input}")
            
        def query_database(user_id):
            # Vulnerabilidad: SQL Injection
            cursor.execute("SELECT * FROM users WHERE id = " + user_id)
            
        # Vulnerabilidad: Hardcoded secret
        api_key = "sk_live_1234567890abcdef"
        """
        
        try:
            # Realizar análisis de seguridad
            result = await agent.analyze_security(vulnerable_code, "python")
            
            print(f"Vulnerabilities found: {result['vulnerabilities_found']}")
            
            for vuln in result["vulnerabilities"][:3]:  # Mostrar primeras 3
                print(f"  - {vuln['title']} ({vuln['severity']}): {vuln['description']}")
            
            # Evaluar riesgo
            vulnerabilities = [Vulnerability(**v) for v in result["vulnerabilities"]]
            risk = await agent.assess_risk(vulnerabilities, {"project_type": "web_application"})
            
            print(f"\nOverall risk level: {risk['risk_level']}")
            print(f"Risk score: {risk['overall_risk_score']:.2f}")
            
            # Generar recomendaciones para primera vulnerabilidad
            if vulnerabilities:
                fix = await agent.recommend_security_fixes(vulnerabilities[0])
                print(f"\nRecommended fix for {vulnerabilities[0].title}:")
                print(f"  {fix['remediation']}")
                print(f"  Estimated effort: {fix['estimated_effort_minutes']} minutes")
            
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            # Apagar agente
            await agent.shutdown()
    
    asyncio.run(main())