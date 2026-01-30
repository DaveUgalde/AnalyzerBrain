"""
ArchitectAgent - Specialized agent for architecture analysis and design.
"""

from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
from .base_agent import BaseAgent, AgentConfig, AgentInput, AgentOutput, AgentCapability, AgentMemoryType
from ..core.exceptions import ValidationError


class ArchitectAgent(BaseAgent):
    """
    Agent specialized in architecture analysis and design.
    
    Capabilities:
    1. Analyze system architecture
    2. Detect architectural smells
    3. Suggest architectural improvements
    4. Validate architectural decisions
    5. Design system components
    6. Evaluate architectural patterns
    7. Generate architecture documentation
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="ArchitectAgent",
                description="Analyzes and designs system architecture",
                capabilities=[
                    AgentCapability.ARCHITECTURE_REVIEW,
                    AgentCapability.PATTERN_DETECTION
                ],
                confidence_threshold=0.75,
                learning_rate=0.15,
                dependencies=["indexer", "graph"]
            )
        
        super().__init__(config)
        self.architectural_patterns: Dict[str, Any] = {}
        self.design_principles: Dict[str, Any] = {}
        self.quality_attributes: Dict[str, Any] = {}
        
    async def _initialize_internal(self) -> bool:
        """Initialize ArchitectAgent with architectural knowledge."""
        try:
            # Load architectural patterns
            self.architectural_patterns = await self._load_architectural_patterns()
            
            # Load design principles
            self.design_principles = await self._load_design_principles()
            
            # Load quality attributes
            self.quality_attributes = await self._load_quality_attributes()
            
            # Initialize analysis tools
            await self._initialize_architecture_tools()
            
            return True
        except Exception as e:
            print(f"Failed to initialize ArchitectAgent: {e}")
            return False
    
    async def _process_internal(self, input_data: AgentInput) -> AgentOutput:
        """Process architecture analysis request."""
        analysis_type = input_data.data.get("analysis_type", "architecture_review")
        
        try:
            if analysis_type == "architecture_analysis":
                result = await self.analyze_architecture(input_data)
            elif analysis_type == "detect_smells":
                result = await self.detect_arch_smells(input_data)
            elif analysis_type == "suggest_improvements":
                result = await self.suggest_arch_improvements(input_data)
            elif analysis_type == "validate_decisions":
                result = await self.validate_arch_decisions(input_data)
            elif analysis_type == "design_components":
                result = await self.design_system_components(input_data)
            elif analysis_type == "evaluate_patterns":
                result = await self.evaluate_arch_patterns(input_data)
            elif analysis_type == "generate_documentation":
                result = await self.generate_arch_documentation(input_data)
            else:
                # General architecture review
                result = await self._perform_general_architecture_review(input_data)
            
            return result
        except Exception as e:
            return self._handle_error(e, {
                "request_id": input_data.request_id,
                "analysis_type": analysis_type
            })
    
    async def _learn_internal(self, feedback: Dict[str, Any]) -> bool:
        """Learn from architecture feedback."""
        feedback_type = feedback.get("type")
        
        if feedback_type == "architecture_correction":
            return await self._learn_from_architecture_correction(feedback)
        elif feedback_type == "pattern_validation":
            return await self._learn_from_pattern_validation(feedback)
        elif feedback_type == "design_principle":
            return await self._update_design_principle(feedback)
        else:
            # General learning
            return await self._general_architecture_learning(feedback)
    
    def _validate_input_specific(self, input_data: AgentInput) -> None:
        """Validate architecture analysis specific input."""
        if "project_structure" not in input_data.data and "components" not in input_data.data:
            raise ValidationError("Architecture analysis requires 'project_structure' or 'components' in input data")
        
        analysis_type = input_data.data.get("analysis_type", "architecture_review")
        valid_types = [
            "architecture_review", "architecture_analysis", "detect_smells",
            "suggest_improvements", "validate_decisions", "design_components",
            "evaluate_patterns", "generate_documentation"
        ]
        
        if analysis_type not in valid_types:
            raise ValidationError(f"analysis_type must be one of {valid_types}")
    
    async def _save_state(self) -> None:
        """Save ArchitectAgent state."""
        state_data = {
            "architectural_patterns": self.architectural_patterns,
            "design_principles": self.design_principles,
            "quality_attributes": self.quality_attributes,
            "timestamp": datetime.now()
        }
        
        # Store in semantic memory
        self.store_memory(
            AgentMemoryType.SEMANTIC,
            {
                "type": "agent_state",
                "agent": "ArchitectAgent",
                "state": state_data
            }
        )
    
    # Public architecture methods
    
    async def analyze_architecture(self, input_data: AgentInput) -> AgentOutput:
        """Analyze system architecture."""
        project_structure = input_data.data.get("project_structure", {})
        components = input_data.data.get("components", [])
        
        # Analyze architecture
        architecture_analysis = await self._perform_architecture_analysis(project_structure, components)
        quality_assessment = await self._assess_architecture_quality(project_structure, components)
        
        confidence = self._calculate_architecture_confidence(architecture_analysis, quality_assessment)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "architecture_analysis": architecture_analysis,
                "quality_assessment": quality_assessment,
                "strengths": self._identify_architecture_strengths(architecture_analysis, quality_assessment),
                "weaknesses": self._identify_architecture_weaknesses(architecture_analysis, quality_assessment)
            },
            confidence=confidence,
            reasoning=["Analyzed component structure", "Evaluated architectural quality", "Assessed design principles"],
            warnings=[] if confidence > 0.7 else ["Limited architecture information available"]
        )
    
    async def detect_arch_smells(self, input_data: AgentInput) -> AgentOutput:
        """Detect architectural smells."""
        project_structure = input_data.data.get("project_structure", {})
        components = input_data.data.get("components", [])
        
        # Detect architectural smells
        smells = await self._detect_architectural_smells(project_structure, components)
        anti_patterns = await self._identify_anti_patterns(project_structure, components)
        
        confidence = self._calculate_smell_detection_confidence(smells, anti_patterns)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "architectural_smells": smells,
                "anti_patterns": anti_patterns,
                "severity_analysis": self._analyze_smell_severity(smells + anti_patterns),
                "impact_assessment": self._assess_smell_impact(smells + anti_patterns, project_structure)
            },
            confidence=confidence,
            reasoning=["Scanned for architectural smells", "Identified anti-patterns", "Assessed severity and impact"],
            warnings=["Critical architectural issues detected"] if self._has_critical_smells(smells) else []
        )
    
    async def suggest_arch_improvements(self, input_data: AgentInput) -> AgentOutput:
        """Suggest architectural improvements."""
        project_structure = input_data.data.get("project_structure", {})
        components = input_data.data.get("components", [])
        current_issues = input_data.data.get("current_issues", [])
        
        # Generate improvement suggestions
        improvements = await self._generate_architectural_improvements(project_structure, components, current_issues)
        refactoring_strategies = await self._suggest_refactoring_strategies(project_structure, components, current_issues)
        
        confidence = 0.8
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "architectural_improvements": improvements,
                "refactoring_strategies": refactoring_strategies,
                "priority_recommendations": self._prioritize_arch_improvements(improvements + refactoring_strategies),
                "migration_path": self._plan_architecture_migration(improvements, project_structure)
            },
            confidence=confidence,
            reasoning=["Analyzed current architecture", "Generated improvement strategies", "Planned migration path"],
            warnings=["Major architectural changes recommended"] if self._requires_major_changes(improvements) else []
        )
    
    async def validate_arch_decisions(self, input_data: AgentInput) -> AgentOutput:
        """Validate architectural decisions."""
        decisions = input_data.data.get("decisions", [])
        context = input_data.data.get("context", {})
        
        # Validate decisions
        validation_results = await self._validate_architecture_decisions(decisions, context)
        alternative_suggestions = await self._suggest_alternatives(decisions, context)
        
        confidence = 0.85
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "validation_results": validation_results,
                "alternative_suggestions": alternative_suggestions,
                "decision_quality": self._assess_decision_quality(validation_results),
                "risk_analysis": self._analyze_decision_risks(decisions, validation_results)
            },
            confidence=confidence,
            reasoning=["Evaluated decision rationale", "Checked against best practices", "Assessed risks and alternatives"],
            warnings=["High-risk decisions identified"] if self._has_high_risk_decisions(validation_results) else []
        )
    
    async def design_system_components(self, input_data: AgentInput) -> AgentOutput:
        """Design system components."""
        requirements = input_data.data.get("requirements", {})
        constraints = input_data.data.get("constraints", {})
        
        # Design components
        component_designs = await self._design_architecture_components(requirements, constraints)
        interfaces = await self._define_component_interfaces(component_designs, requirements)
        
        confidence = 0.75
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "component_designs": component_designs,
                "interfaces": interfaces,
                "design_rationale": self._explain_design_rationale(component_designs, requirements),
                "implementation_guidance": self._provide_implementation_guidance(component_designs, interfaces)
            },
            confidence=confidence,
            reasoning=["Analyzed requirements and constraints", "Designed system components", "Defined interfaces"],
            warnings=["Complex component interactions identified"] if self._has_complex_interactions(interfaces) else []
        )
    
    async def evaluate_arch_patterns(self, input_data: AgentInput) -> AgentOutput:
        """Evaluate architectural patterns."""
        current_architecture = input_data.data.get("current_architecture", {})
        requirements = input_data.data.get("requirements", {})
        
        # Evaluate patterns
        pattern_evaluations = await self._evaluate_architectural_patterns(current_architecture, requirements)
        pattern_recommendations = await self._recommend_architectural_patterns(current_architecture, requirements)
        
        confidence = 0.8
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "pattern_evaluations": pattern_evaluations,
                "pattern_recommendations": pattern_recommendations,
                "suitability_analysis": self._analyze_pattern_suitability(pattern_evaluations, requirements),
                "adoption_plan": self._plan_pattern_adoption(pattern_recommendations, current_architecture)
            },
            confidence=confidence,
            reasoning=["Analyzed current architecture", "Evaluated pattern suitability", "Recommended appropriate patterns"],
            warnings=["Pattern mismatch detected"] if self._has_pattern_mismatches(pattern_evaluations) else []
        )
    
    async def generate_arch_documentation(self, input_data: AgentInput) -> AgentOutput:
        """Generate architecture documentation."""
        architecture = input_data.data.get("architecture", {})
        audience = input_data.data.get("audience", "developers")
        
        # Generate documentation
        documentation = await self._generate_architecture_documentation(architecture, audience)
        diagrams = await self._generate_architecture_diagrams(architecture, audience)
        
        confidence = 0.9
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "documentation": documentation,
                "diagrams": diagrams,
                "documentation_quality": self._assess_documentation_quality(documentation, architecture),
                "audience_suitability": self._check_audience_suitability(documentation, audience)
            },
            confidence=confidence,
            reasoning=["Analyzed architecture structure", "Generated comprehensive documentation", "Created visual diagrams"],
            warnings=["Incomplete architecture information"] if self._has_missing_architecture_info(architecture) else []
        )
    
    # Helper methods
    
    async def _load_architectural_patterns(self) -> Dict[str, Any]:
        """Load architectural patterns."""
        return {
            "layered": {
                "description": "Separates concerns into layers",
                "use_cases": ["Enterprise applications", "Web applications"],
                "benefits": ["Separation of concerns", "Maintainability", "Testability"],
                "drawbacks": ["Performance overhead", "Tight coupling between layers"]
            },
            "microservices": {
                "description": "Decomposes application into small, independent services",
                "use_cases": ["Large-scale applications", "Cloud-native applications"],
                "benefits": ["Scalability", "Independent deployment", "Technology diversity"],
                "drawbacks": ["Distributed system complexity", "Operational overhead"]
            },
            "event_driven": {
                "description": "Components communicate through events",
                "use_cases": ["Real-time systems", "Asynchronous processing"],
                "benefits": ["Loose coupling", "Scalability", "Responsiveness"],
                "drawbacks": ["Complex error handling", "Debugging difficulty"]
            },
            "hexagonal": {
                "description": "Separates core logic from external concerns",
                "use_cases": ["Domain-driven design", "Test-driven development"],
                "benefits": ["Testability", "Independence from external systems", "Flexibility"],
                "drawbacks": ["Complexity", "Over-engineering for simple systems"]
            }
        }
    
    async def _load_design_principles(self) -> Dict[str, Any]:
        """Load design principles."""
        return {
            "solid": {
                "single_responsibility": "A class should have only one reason to change",
                "open_closed": "Software entities should be open for extension but closed for modification",
                "liskov_substitution": "Objects should be replaceable with instances of their subtypes",
                "interface_segregation": "Many client-specific interfaces are better than one general-purpose interface",
                "dependency_inversion": "Depend upon abstractions, not concretions"
            },
            "other_principles": {
                "dry": "Don't Repeat Yourself",
                "kiss": "Keep It Simple, Stupid",
                "yagni": "You Aren't Gonna Need It",
                "separation_of_concerns": "Separate different concerns into distinct sections"
            }
        }
    
    async def _load_quality_attributes(self) -> Dict[str, Any]:
        """Load quality attributes."""
        return {
            "performance": ["response_time", "throughput", "resource_utilization"],
            "scalability": ["horizontal_scaling", "vertical_scaling", "load_distribution"],
            "reliability": ["fault_tolerance", "availability", "recoverability"],
            "security": ["authentication", "authorization", "data_protection"],
            "maintainability": ["modifiability", "testability", "analyzability"],
            "usability": ["learnability", "efficiency", "satisfaction"]
        }
    
    async def _initialize_architecture_tools(self) -> None:
        """Initialize architecture analysis tools."""
        # Would initialize architecture analysis libraries
        pass
    
    async def _perform_general_architecture_review(self, input_data: AgentInput) -> AgentOutput:
        """Perform general architecture review."""
        # Combine multiple analysis types
        analysis_result = await self.analyze_architecture(input_data)
        smells_result = await self.detect_arch_smells(input_data)
        
        confidence = (analysis_result.confidence + smells_result.confidence) / 2
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "architecture_analysis": analysis_result.data,
                "architectural_smells": smells_result.data,
                "overall_assessment": self._assess_overall_architecture(
                    analysis_result.data, 
                    smells_result.data
                ),
                "key_findings": self._extract_key_findings(analysis_result.data, smells_result.data)
            },
            confidence=confidence,
            reasoning=["Performed comprehensive architecture review", "Combined analysis techniques"],
            warnings=[]
        )
    
    async def _perform_architecture_analysis(self, project_structure: Dict, components: List) -> Dict[str, Any]:
        """Perform detailed architecture analysis."""
        analysis = {
            "component_count": len(components),
            "dependency_analysis": self._analyze_dependencies(components),
            "cohesion_coupling": self._assess_cohesion_coupling(components),
            "architectural_style": self._identify_architectural_style(project_structure, components),
            "communication_patterns": self._identify_communication_patterns(components)
        }
        
        return analysis
    
    async def _assess_architecture_quality(self, project_structure: Dict, components: List) -> Dict[str, Any]:
        """Assess architecture quality."""
        quality_metrics = {
            "modularity": self._assess_modularity(components),
            "scalability": self._assess_scalability(project_structure, components),
            "maintainability": self._assess_maintainability(components),
            "testability": self._assess_testability(components),
            "security": self._assess_security(project_structure, components)
        }
        
        return quality_metrics
    
    def _calculate_architecture_confidence(self, analysis: Dict, quality: Dict) -> float:
        """Calculate confidence in architecture analysis."""
        base_confidence = 0.7
        
        # Adjust based on analysis depth
        if analysis.get("component_count", 0) > 0:
            base_confidence += 0.1
        
        if quality.get("modularity", {}).get("score", 0) > 0:
            base_confidence += 0.05
        
        # Adjust based on data completeness
        if len(analysis) >= 3 and len(quality) >= 3:
            base_confidence += 0.1
        
        return min(max(base_confidence, 0.3), 0.95)
    
    async def _detect_architectural_smells(self, project_structure: Dict, components: List) -> List[Dict[str, Any]]:
        """Detect architectural smells."""
        smells = []
        
        # Common architectural smells
        smell_patterns = [
            {
                "name": "Cyclic Dependency",
                "detection": self._detect_cyclic_dependencies(components),
                "severity": "high"
            },
            {
                "name": "God Component",
                "detection": self._detect_god_components(components),
                "severity": "medium"
            },
            {
                "name": "Shotgun Surgery",
                "detection": self._detect_shotgun_surgery(components),
                "severity": "medium"
            },
            {
                "name": "Feature Envy",
                "detection": self._detect_feature_envy(components),
                "severity": "low"
            }
        ]
        
        for pattern in smell_patterns:
            detection_result = pattern["detection"]
            if detection_result["found"]:
                smells.append({
                    "name": pattern["name"],
                    "severity": pattern["severity"],
                    "description": detection_result["description"],
                    "components": detection_result.get("components", []),
                    "impact": self._assess_smell_impact_pattern(pattern["name"], detection_result)
                })
        
        return smells
    
    async def _identify_anti_patterns(self, project_structure: Dict, components: List) -> List[Dict[str, Any]]:
        """Identify architectural anti-patterns."""
        anti_patterns = []
        
        # Common anti-patterns
        patterns_to_check = [
            "Big Ball of Mud",
            "Stovepipe System",
            "Vendor Lock-in",
            "Tight Coupling",
            "Database as Integration Point"
        ]
        
        for pattern_name in patterns_to_check:
            detection = self._check_anti_pattern(pattern_name, project_structure, components)
            if detection["found"]:
                anti_patterns.append({
                    "name": pattern_name,
                    "description": detection["description"],
                    "evidence": detection.get("evidence", []),
                    "recommendation": detection.get("recommendation", "")
                })
        
        return anti_patterns
    
    def _calculate_smell_detection_confidence(self, smells: List, anti_patterns: List) -> float:
        """Calculate confidence in smell detection."""
        base_confidence = 0.75
        
        # Higher confidence if we found issues
        if smells or anti_patterns:
            base_confidence += 0.1
        
        # Adjust based on detection specificity
        specific_detections = sum(1 for s in smells if len(s.get("components", [])) > 0)
        if specific_detections > 0:
            base_confidence += 0.05
        
        return min(max(base_confidence, 0.4), 0.95)
    
    async def _generate_architectural_improvements(self, project_structure: Dict, 
                                                  components: List, current_issues: List) -> List[Dict[str, Any]]:
        """Generate architectural improvement suggestions."""
        improvements = []
        
        # Generate improvements based on issues
        for issue in current_issues:
            improvement = self._suggest_improvement_for_issue(issue, project_structure, components)
            if improvement:
                improvements.append(improvement)
        
        # General improvements
        general_improvements = [
            {
                "type": "modularization",
                "description": "Improve module boundaries for better separation of concerns",
                "benefit": "Enhanced maintainability and testability",
                "effort": "medium"
            },
            {
                "type": "dependency_management",
                "description": "Reduce coupling between components",
                "benefit": "Increased flexibility and reusability",
                "effort": "high"
            }
        ]
        
        improvements.extend(general_improvements)
        
        return improvements
    
    async def _validate_architecture_decisions(self, decisions: List, context: Dict) -> List[Dict[str, Any]]:
        """Validate architectural decisions."""
        validation_results = []
        
        for decision in decisions:
            validation = self._validate_single_decision(decision, context)
            validation_results.append({
                "decision": decision.get("description", "Unknown"),
                "validation": validation,
                "rationale_quality": self._assess_decision_rationale(decision),
                "alternatives_considered": self._check_alternatives_considered(decision)
            })
        
        return validation_results
    
    async def _design_architecture_components(self, requirements: Dict, constraints: Dict) -> List[Dict[str, Any]]:
        """Design architecture components."""
        components = []
        
        # Extract functional requirements
        functional_reqs = requirements.get("functional", [])
        
        for req in functional_reqs[:5]:  # Limit to top 5 for simplicity
            component = self._design_component_for_requirement(req, constraints)
            if component:
                components.append(component)
        
        # Add cross-cutting components
        cross_cutting = [
            {
                "name": "Authentication Service",
                "responsibility": "Handle user authentication and authorization",
                "interfaces": ["login", "logout", "validate_token"],
                "dependencies": ["User Database", "Token Service"]
            },
            {
                "name": "Logging Service",
                "responsibility": "Centralized logging and monitoring",
                "interfaces": ["log_event", "get_logs", "set_log_level"],
                "dependencies": []
            }
        ]
        
        components.extend(cross_cutting)
        
        return components
    
    async def _evaluate_architectural_patterns(self, current_architecture: Dict, 
                                              requirements: Dict) -> List[Dict[str, Any]]:
        """Evaluate architectural patterns."""
        evaluations = []
        
        for pattern_name, pattern_info in self.architectural_patterns.items():
            evaluation = self._evaluate_pattern_for_architecture(
                pattern_name, 
                pattern_info, 
                current_architecture, 
                requirements
            )
            evaluations.append({
                "pattern": pattern_name,
                "evaluation": evaluation,
                "suitability_score": self._calculate_pattern_suitability(evaluation, requirements),
                "adoption_complexity": self._estimate_adoption_complexity(pattern_name, current_architecture)
            })
        
        return evaluations
    
    async def _generate_architecture_documentation(self, architecture: Dict, audience: str) -> Dict[str, Any]:
        """Generate architecture documentation."""
        documentation = {
            "overview": self._generate_architecture_overview(architecture),
            "components": self._document_architecture_components(architecture),
            "interactions": self._document_component_interactions(architecture),
            "decisions": self._document_architecture_decisions(architecture),
            "quality_attributes": self._document_quality_attributes(architecture)
        }
        
        # Tailor for audience
        if audience == "developers":
            documentation["technical_details"] = self._add_technical_details(architecture)
        elif audience == "managers":
            documentation["summary"] = self._create_management_summary(architecture)
        
        return documentation
    
    # Architecture analysis helper methods
    
    def _analyze_dependencies(self, components: List) -> Dict[str, Any]:
        """Analyze component dependencies."""
        if not components:
            return {"total_dependencies": 0, "cyclic": False, "dependency_graph": {}}
        
        # Simplified dependency analysis
        total_deps = 0
        dependency_graph = {}
        
        for component in components:
            deps = component.get("dependencies", [])
            total_deps += len(deps)
            dependency_graph[component.get("name", "unknown")] = deps
        
        return {
            "total_dependencies": total_deps,
            "average_dependencies": total_deps / max(len(components), 1),
            "cyclic": self._check_cyclic_dependencies_simple(dependency_graph),
            "dependency_graph": dependency_graph
        }
    
    def _assess_cohesion_coupling(self, components: List) -> Dict[str, Any]:
        """Assess cohesion and coupling."""
        if not components:
            return {"cohesion_score": 0, "coupling_score": 0, "balance": "unknown"}
        
        # Simplified assessment
        cohesion_score = 0
        coupling_score = 0
        
        for component in components:
            # Assume components with clear single responsibility have high cohesion
            responsibilities = component.get("responsibilities", [])
            if len(responsibilities) == 1:
                cohesion_score += 1
            elif len(responsibilities) <= 3:
                cohesion_score += 0.5
            
            # Coupling based on dependencies
            deps = component.get("dependencies", [])
            if len(deps) == 0:
                coupling_score += 0  # No coupling
            elif len(deps) <= 2:
                coupling_score += 0.3  # Low coupling
            elif len(deps) <= 5:
                coupling_score += 0.7  # Medium coupling
            else:
                coupling_score += 1  # High coupling
        
        avg_cohesion = cohesion_score / max(len(components), 1)
        avg_coupling = coupling_score / max(len(components), 1)
        
        return {
            "cohesion_score": avg_cohesion,
            "coupling_score": avg_coupling,
            "balance": "good" if avg_cohesion > 0.7 and avg_coupling < 0.4 else 
                      "fair" if avg_cohesion > 0.5 and avg_coupling < 0.6 else 
                      "poor"
        }
    
    def _identify_architectural_style(self, project_structure: Dict, components: List) -> str:
        """Identify architectural style."""
        if not components:
            return "unknown"
        
        # Check for microservices characteristics
        if self._has_microservices_characteristics(components):
            return "microservices"
        
        # Check for layered architecture
        if self._has_layered_characteristics(project_structure):
            return "layered"
        
        # Check for event-driven
        if self._has_event_driven_characteristics(components):
            return "event_driven"
        
        return "monolithic"
    
    def _identify_communication_patterns(self, components: List) -> List[str]:
        """Identify communication patterns."""
        patterns = []
        
        # Check for synchronous communication
        if any("api_call" in str(comp).lower() or "http" in str(comp).lower() for comp in components):
            patterns.append("synchronous_rpc")
        
        # Check for asynchronous communication
        if any("queue" in str(comp).lower() or "event" in str(comp).lower() for comp in components):
            patterns.append("asynchronous_messaging")
        
        # Check for database sharing
        if any("shared_database" in str(comp).lower() for comp in components):
            patterns.append("shared_database")
        
        return patterns if patterns else ["direct_calls"]
    
    def _assess_modularity(self, components: List) -> Dict[str, Any]:
        """Assess modularity."""
        if not components:
            return {"score": 0, "assessment": "insufficient_data"}
        
        modularity_score = 0
        factors = []
        
        # Check for clear interfaces
        components_with_interfaces = sum(1 for comp in components if comp.get("interfaces"))
        interface_clarity = components_with_interfaces / len(components)
        
        if interface_clarity > 0.8:
            modularity_score += 1
            factors.append("clear_interfaces")
        elif interface_clarity > 0.5:
            modularity_score += 0.5
            factors.append("partial_interfaces")
        
        # Check for low coupling
        coupling_data = self._assess_cohesion_coupling(components)
        if coupling_data["coupling_score"] < 0.3:
            modularity_score += 1
            factors.append("low_coupling")
        elif coupling_data["coupling_score"] < 0.6:
            modularity_score += 0.5
            factors.append("moderate_coupling")
        
        # Check for high cohesion
        if coupling_data["cohesion_score"] > 0.7:
            modularity_score += 1
            factors.append("high_cohesion")
        elif coupling_data["cohesion_score"] > 0.5:
            modularity_score += 0.5
            factors.append("moderate_cohesion")
        
        normalized_score = modularity_score / 3  # Max possible score
        
        return {
            "score": normalized_score,
            "factors": factors,
            "assessment": "excellent" if normalized_score > 0.8 else 
                         "good" if normalized_score > 0.6 else 
                         "fair" if normalized_score > 0.4 else 
                         "poor"
        }
    
    def _assess_scalability(self, project_structure: Dict, components: List) -> Dict[str, Any]:
        """Assess scalability."""
        scalability_score = 0
        factors = []
        
        # Check for stateless components
        stateless_components = sum(1 for comp in components if comp.get("stateful") is False)
        if stateless_components > 0:
            scalability_score += 0.5
            factors.append("stateless_components")
        
        # Check for horizontal scaling capability
        if self._supports_horizontal_scaling(components):
            scalability_score += 1
            factors.append("horizontal_scaling")
        
        # Check for load balancing
        if any("load_balancer" in str(comp).lower() for comp in components):
            scalability_score += 0.5
            factors.append("load_balancing")
        
        # Check for caching
        if any("cache" in str(comp).lower() for comp in components):
            scalability_score += 0.5
            factors.append("caching")
        
        normalized_score = scalability_score / 2.5  # Max possible score
        
        return {
            "score": normalized_score,
            "factors": factors,
            "assessment": "highly_scalable" if normalized_score > 0.8 else 
                         "scalable" if normalized_score > 0.6 else 
                         "moderately_scalable" if normalized_score > 0.4 else 
                         "limited_scalability"
        }
    
    def _assess_maintainability(self, components: List) -> Dict[str, Any]:
        """Assess maintainability."""
        maintainability_score = 0
        factors = []
        
        # Check for documentation
        documented_components = sum(1 for comp in components if comp.get("documentation"))
        if documented_components / max(len(components), 1) > 0.5:
            maintainability_score += 0.5
            factors.append("good_documentation")
        
        # Check for test coverage
        if any("test_coverage" in str(comp).lower() for comp in components):
            maintainability_score += 0.5
            factors.append("test_coverage")
        
        # Check for modularity
        modularity = self._assess_modularity(components)
        if modularity["score"] > 0.6:
            maintainability_score += 1
            factors.append("good_modularity")
        elif modularity["score"] > 0.4:
            maintainability_score += 0.5
            factors.append("moderate_modularity")
        
        normalized_score = maintainability_score / 2  # Max possible score
        
        return {
            "score": normalized_score,
            "factors": factors,
            "assessment": "highly_maintainable" if normalized_score > 0.8 else 
                         "maintainable" if normalized_score > 0.6 else 
                         "moderately_maintainable" if normalized_score > 0.4 else 
                         "difficult_to_maintain"
        }
    
    def _assess_testability(self, components: List) -> Dict[str, Any]:
        """Assess testability."""
        testability_score = 0
        factors = []
        
        # Check for dependency injection
        if any("dependency_injection" in str(comp).lower() for comp in components):
            testability_score += 1
            factors.append("dependency_injection")
        
        # Check for interfaces
        components_with_interfaces = sum(1 for comp in components if comp.get("interfaces"))
        if components_with_interfaces > 0:
            testability_score += 0.5
            factors.append("defined_interfaces")
        
        # Check for mocking capability
        if self._supports_mocking(components):
            testability_score += 0.5
            factors.append("mockable_components")
        
        normalized_score = testability_score / 2  # Max possible score
        
        return {
            "score": normalized_score,
            "factors": factors,
            "assessment": "highly_testable" if normalized_score > 0.8 else 
                         "testable" if normalized_score > 0.6 else 
                         "moderately_testable" if normalized_score > 0.4 else 
                         "difficult_to_test"
        }
    
    def _assess_security(self, project_structure: Dict, components: List) -> Dict[str, Any]:
        """Assess security."""
        security_score = 0
        factors = []
        
        # Check for authentication
        if any("authentication" in str(comp).lower() or "auth" in str(comp).lower() for comp in components):
            security_score += 0.5
            factors.append("authentication")
        
        # Check for authorization
        if any("authorization" in str(comp).lower() or "permission" in str(comp).lower() for comp in components):
            security_score += 0.5
            factors.append("authorization")
        
        # Check for encryption
        if any("encryption" in str(comp).lower() or "ssl" in str(comp).lower() for comp in components):
            security_score += 0.5
            factors.append("encryption")
        
        # Check for input validation
        if any("validation" in str(comp).lower() for comp in components):
            security_score += 0.5
            factors.append("input_validation")
        
        normalized_score = security_score / 2  # Max possible score
        
        return {
            "score": normalized_score,
            "factors": factors,
            "assessment": "highly_secure" if normalized_score > 0.8 else 
                         "secure" if normalized_score > 0.6 else 
                         "moderately_secure" if normalized_score > 0.4 else 
                         "security_concerns"
        }
    
    def _identify_architecture_strengths(self, analysis: Dict, quality: Dict) -> List[str]:
        """Identify architecture strengths."""
        strengths = []
        
        # Check cohesion-coupling balance
        if analysis.get("cohesion_coupling", {}).get("balance") == "good":
            strengths.append("Good balance between cohesion and coupling")
        
        # Check modularity
        if quality.get("modularity", {}).get("assessment") in ["excellent", "good"]:
            strengths.append("Strong modular design")
        
        # Check scalability
        if quality.get("scalability", {}).get("assessment") in ["highly_scalable", "scalable"]:
            strengths.append("Good scalability characteristics")
        
        # Check for clear architectural style
        style = analysis.get("architectural_style", "unknown")
        if style != "unknown":
            strengths.append(f"Clear {style} architectural style")
        
        return strengths if strengths else ["No major strengths identified"]
    
    def _identify_architecture_weaknesses(self, analysis: Dict, quality: Dict) -> List[str]:
        """Identify architecture weaknesses."""
        weaknesses = []
        
        # Check for cyclic dependencies
        if analysis.get("dependency_analysis", {}).get("cyclic"):
            weaknesses.append("Cyclic dependencies detected")
        
        # Check for poor modularity
        if quality.get("modularity", {}).get("assessment") in ["poor"]:
            weaknesses.append("Poor modularity - high coupling and/or low cohesion")
        
        # Check for limited scalability
        if quality.get("scalability", {}).get("assessment") in ["limited_scalability"]:
            weaknesses.append("Limited scalability potential")
        
        # Check for security concerns
        if quality.get("security", {}).get("assessment") in ["security_concerns"]:
            weaknesses.append("Security architecture needs improvement")
        
        return weaknesses if weaknesses else ["No major weaknesses identified"]
    
    # Smell detection helper methods
    
    def _detect_cyclic_dependencies(self, components: List) -> Dict[str, Any]:
        """Detect cyclic dependencies."""
        # Simplified detection
        dependency_graph = {}
        for component in components:
            name = component.get("name", "unknown")
            deps = component.get("dependencies", [])
            dependency_graph[name] = deps
        
        has_cycle = self._check_cyclic_dependencies_simple(dependency_graph)
        
        return {
            "found": has_cycle,
            "description": "Components have circular dependencies that can cause maintenance issues",
            "components": list(dependency_graph.keys()) if has_cycle else []
        }
    
    def _detect_god_components(self, components: List) -> Dict[str, Any]:
        """Detect god components (components with too many responsibilities)."""
        god_components = []
        
        for component in components:
            responsibilities = component.get("responsibilities", [])
            dependencies = component.get("dependencies", [])
            
            # Simple heuristic: component with many responsibilities and many dependencies
            if len(responsibilities) > 5 and len(dependencies) > 10:
                god_components.append(component.get("name", "unknown"))
        
        return {
            "found": len(god_components) > 0,
            "description": f"Found {len(god_components)} components with too many responsibilities",
            "components": god_components
        }
    
    def _detect_shotgun_surgery(self, components: List) -> Dict[str, Any]:
        """Detect shotgun surgery (changes require modifications in many places)."""
        # Simplified detection based on shared dependencies
        if len(components) < 3:
            return {"found": False, "description": "Insufficient components for analysis"}
        
        # Look for components that share many of the same dependencies
        dependency_counts = {}
        for component in components:
            deps = tuple(sorted(component.get("dependencies", [])))
            dependency_counts[deps] = dependency_counts.get(deps, 0) + 1
        
        # If many components share exact same dependencies, might indicate shotgun surgery
        shared_dep_sets = [(deps, count) for deps, count in dependency_counts.items() if count > 2]
        
        return {
            "found": len(shared_dep_sets) > 0,
            "description": f"Found {len(shared_dep_sets)} sets of components with identical dependencies",
            "components": [f"{count} components share dependencies" for deps, count in shared_dep_sets]
        }
    
    def _detect_feature_envy(self, components: List) -> Dict[str, Any]:
        """Detect feature envy (component uses more features of another component than its own)."""
        # Simplified detection
        feature_envy_components = []
        
        for component in components:
            name = component.get("name", "")
            # Look for components that primarily interact with one other component
            dependencies = component.get("dependencies", [])
            if len(dependencies) == 1:
                feature_envy_components.append(name)
        
        return {
            "found": len(feature_envy_components) > 0,
            "description": f"Found {len(feature_envy_components)} components that seem overly dependent on a single other component",
            "components": feature_envy_components
        }
    
    def _check_cyclic_dependencies_simple(self, graph: Dict[str, List[str]]) -> bool:
        """Simple cyclic dependency check."""
        # Very simplified - in real implementation would use proper graph algorithms
        for node, deps in graph.items():
            for dep in deps:
                if dep in graph and node in graph.get(dep, []):
                    return True
        return False
    
    def _check_anti_pattern(self, pattern_name: str, project_structure: Dict, 
                           components: List) -> Dict[str, Any]:
        """Check for specific anti-pattern."""
        check_methods = {
            "Big Ball of Mud": self._check_big_ball_of_mud,
            "Stovepipe System": self._check_stovepipe_system,
            "Vendor Lock-in": self._check_vendor_lock_in,
            "Tight Coupling": self._check_tight_coupling,
            "Database as Integration Point": self._check_database_integration_point
        }
        
        if pattern_name in check_methods:
            return check_methods[pattern_name](project_structure, components)
        
        return {"found": False, "description": f"Unknown anti-pattern: {pattern_name}"}
    
    def _check_big_ball_of_mud(self, project_structure: Dict, components: List) -> Dict[str, Any]:
        """Check for Big Ball of Mud anti-pattern."""
        # Heuristic: many components, unclear structure, high coupling
        if len(components) > 20:
            coupling = self._assess_cohesion_coupling(components)
            if coupling["coupling_score"] > 0.7:
                return {
                    "found": True,
                    "description": "System shows characteristics of Big Ball of Mud: many components with high coupling",
                    "evidence": [
                        f"High coupling score: {coupling['coupling_score']:.2f}",
                        f"Large number of components: {len(components)}"
                    ],
                    "recommendation": "Refactor into clearer modules with better separation of concerns"
                }
        
        return {"found": False, "description": "No clear evidence of Big Ball of Mud pattern"}
    
    def _check_stovepipe_system(self, project_structure: Dict, components: List) -> Dict[str, Any]:
        """Check for Stovepipe System anti-pattern."""
        # Heuristic: components don't communicate, duplicated functionality
        communication_patterns = self._identify_communication_patterns(components)
        
        if not communication_patterns or "direct_calls" in communication_patterns:
            # Look for duplicated responsibilities
            responsibilities = []
            for component in components:
                responsibilities.extend(component.get("responsibilities", []))
            
            duplicate_responsibilities = []
            from collections import Counter
            responsibility_counts = Counter(responsibilities)
            for resp, count in responsibility_counts.items():
                if count > 1:
                    duplicate_responsibilities.append(f"{resp}: {count} times")
            
            if duplicate_responsibilities:
                return {
                    "found": True,
                    "description": "System shows characteristics of Stovepipe: limited communication with duplicated functionality",
                    "evidence": [
                        f"Duplicated responsibilities: {', '.join(duplicate_responsibilities[:3])}",
                        "Limited communication patterns"
                    ],
                    "recommendation": "Establish clear interfaces and shared services"
                }
        
        return {"found": False, "description": "No clear evidence of Stovepipe System pattern"}
    
    def _check_vendor_lock_in(self, project_structure: Dict, components: List) -> Dict[str, Any]:
        """Check for Vendor Lock-in anti-pattern."""
        # Look for vendor-specific technologies
        vendor_technologies = [
            "aws", "azure", "gcp", "oracle", "salesforce", "sap",
            "windows", "proprietary", "vendor_specific"
        ]
        
        vendor_components = []
        for component in components:
            tech_stack = str(component.get("technology", "")).lower()
            for vendor in vendor_technologies:
                if vendor in tech_stack:
                    vendor_components.append(component.get("name", "unknown"))
                    break
        
        if vendor_components:
            return {
                "found": True,
                "description": f"Found {len(vendor_components)} components using vendor-specific technologies",
                "evidence": [
                    f"Vendor-locked components: {', '.join(vendor_components[:3])}"
                ],
                "recommendation": "Introduce abstraction layers to reduce vendor dependency"
            }
        
        return {"found": False, "description": "No clear evidence of Vendor Lock-in"}
    
    def _check_tight_coupling(self, project_structure: Dict, components: List) -> Dict[str, Any]:
        """Check for Tight Coupling anti-pattern."""
        coupling = self._assess_cohesion_coupling(components)
        
        if coupling["coupling_score"] > 0.8:
            return {
                "found": True,
                "description": "System shows very high coupling between components",
                "evidence": [
                    f"Coupling score: {coupling['coupling_score']:.2f}",
                    f"Balance assessment: {coupling['balance']}"
                ],
                "recommendation": "Introduce interfaces and dependency injection to reduce coupling"
            }
        
        return {"found": False, "description": "No extreme tight coupling detected"}
    
    def _check_database_integration_point(self, project_structure: Dict, components: List) -> Dict[str, Any]:
        """Check for Database as Integration Point anti-pattern."""
        # Look for components that share databases
        database_components = []
        for component in components:
            if "database" in str(component.get("dependencies", [])).lower():
                database_components.append(component.get("name", "unknown"))
        
        if len(database_components) > 3:
            return {
                "found": True,
                "description": "Multiple components share database as integration point",
                "evidence": [
                    f"{len(database_components)} components depend on shared database"
                ],
                "recommendation": "Introduce service layers or APIs for inter-component communication"
            }
        
        return {"found": False, "description": "No clear evidence of Database as Integration Point"}
    
    def _analyze_smell_severity(self, smells: List) -> Dict[str, int]:
        """Analyze smell severity distribution."""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for smell in smells:
            severity = smell.get("severity", "medium").lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return severity_counts
    
    def _assess_smell_impact(self, smells: List, project_structure: Dict) -> Dict[str, Any]:
        """Assess overall impact of smells."""
        if not smells:
            return {"overall_impact": "low", "areas_affected": []}
        
        # Calculate impact score
        severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        impact_score = 0
        
        affected_areas = set()
        for smell in smells:
            severity = smell.get("severity", "medium").lower()
            impact_score += severity_weights.get(severity, 1)
            
            # Track affected areas
            components = smell.get("components", [])
            affected_areas.update(components[:3])  # Limit to top 3
        
        # Normalize score
        max_possible = len(smells) * 4
        normalized_score = impact_score / max_possible if max_possible > 0 else 0
        
        impact_level = "critical" if normalized_score > 0.75 else \
                       "high" if normalized_score > 0.5 else \
                       "medium" if normalized_score > 0.25 else \
                       "low"
        
        return {
            "overall_impact": impact_level,
            "impact_score": normalized_score,
            "areas_affected": list(affected_areas)[:5],  # Limit to top 5
            "smell_count": len(smells)
        }
    
    def _assess_smell_impact_pattern(self, smell_name: str, detection_result: Dict) -> str:
        """Assess impact of specific smell pattern."""
        impact_map = {
            "Cyclic Dependency": "High - Can cause maintenance nightmares and deployment issues",
            "God Component": "Medium - Reduces flexibility and increases testing complexity",
            "Shotgun Surgery": "Medium - Makes changes difficult and error-prone",
            "Feature Envy": "Low - Mostly a design issue affecting readability"
        }
        
        return impact_map.get(smell_name, "Unknown impact level")
    
    def _has_critical_smells(self, smells: List) -> bool:
        """Check if there are critical smells."""
        return any(s.get("severity") == "critical" for s in smells)
    
    # Improvement suggestion helper methods
    
    def _suggest_improvement_for_issue(self, issue: Dict, project_structure: Dict, 
                                      components: List) -> Optional[Dict[str, Any]]:
        """Suggest improvement for specific issue."""
        issue_type = issue.get("type", "")
        
        improvement_templates = {
            "cyclic_dependency": {
                "type": "refactoring",
                "description": "Break cyclic dependencies by introducing interfaces or event-based communication",
                "approach": ["Identify dependency cycle", "Introduce abstraction layer", "Use dependency inversion"],
                "benefit": "Eliminates deployment and testing issues caused by cycles",
                "effort": "high"
            },
            "high_coupling": {
                "type": "refactoring",
                "description": "Reduce coupling through interface segregation and dependency injection",
                "approach": ["Define clear interfaces", "Implement dependency injection", "Use service locator pattern"],
                "benefit": "Increases flexibility and testability",
                "effort": "medium"
            },
            "low_cohesion": {
                "type": "modularization",
                "description": "Improve cohesion by separating concerns into dedicated components",
                "approach": ["Identify distinct responsibilities", "Extract into separate components", "Define clear boundaries"],
                "benefit": "Improves maintainability and understandability",
                "effort": "medium"
            },
            "scalability_issue": {
                "type": "architecture",
                "description": "Improve scalability through stateless design and horizontal scaling",
                "approach": ["Make components stateless", "Introduce caching", "Implement load balancing"],
                "benefit": "Enables handling increased load",
                "effort": "high"
            }
        }
        
        if issue_type in improvement_templates:
            return improvement_templates[issue_type]
        
        return None
    
    async def _suggest_refactoring_strategies(self, project_structure: Dict, 
                                             components: List, current_issues: List) -> List[Dict[str, Any]]:
        """Suggest refactoring strategies."""
        strategies = []
        
        # Based on architecture style
        style = self._identify_architectural_style(project_structure, components)
        
        if style == "monolithic":
            strategies.append({
                "name": "Modular Monolith",
                "description": "Refactor monolithic application into well-defined modules",
                "steps": [
                    "Identify bounded contexts",
                    "Define module boundaries",
                    "Establish clear interfaces",
                    "Gradually extract modules"
                ],
                "applicability": "Large monolithic applications",
                "risk": "medium"
            })
        
        # Based on coupling issues
        coupling = self._assess_cohesion_coupling(components)
        if coupling["coupling_score"] > 0.6:
            strategies.append({
                "name": "Dependency Reduction",
                "description": "Reduce coupling between components",
                "steps": [
                    "Analyze dependency graph",
                    "Identify unnecessary dependencies",
                    "Introduce abstraction layers",
                    "Implement dependency injection"
                ],
                "applicability": "Highly coupled systems",
                "risk": "low"
            })
        
        return strategies
    
    def _prioritize_arch_improvements(self, improvements: List) -> List[Dict[str, Any]]:
        """Prioritize architectural improvements."""
        if not improvements:
            return []
        
        # Add priority based on type and effort
        prioritized = []
        for improvement in improvements:
            priority_score = 0
            
            # Type-based priority
            if improvement.get("type") in ["refactoring", "architecture"]:
                priority_score += 2
            elif improvement.get("type") == "modularization":
                priority_score += 1
            
            # Effort-based priority (lower effort gets higher priority)
            effort = improvement.get("effort", "medium")
            if effort == "low":
                priority_score += 2
            elif effort == "medium":
                priority_score += 1
            
            # Benefit-based priority
            if "scalability" in str(improvement.get("benefit", "")).lower():
                priority_score += 1
            
            prioritized.append({
                **improvement,
                "priority_score": priority_score,
                "priority": "high" if priority_score >= 3 else "medium" if priority_score >= 2 else "low"
            })
        
        # Sort by priority
        return sorted(prioritized, key=lambda x: x["priority_score"], reverse=True)
    
    def _plan_architecture_migration(self, improvements: List, current_architecture: Dict) -> Dict[str, Any]:
        """Plan architecture migration."""
        if not improvements:
            return {"strategy": "no_changes_needed", "timeline": "N/A"}
        
        # Categorize improvements by effort
        high_effort = [i for i in improvements if i.get("effort") == "high"]
        medium_effort = [i for i in improvements if i.get("effort") == "medium"]
        low_effort = [i for i in improvements if i.get("effort") == "low"]
        
        # Create migration plan
        phases = []
        
        # Phase 1: Quick wins (low effort)
        if low_effort:
            phases.append({
                "phase": 1,
                "duration": "2-4 weeks",
                "focus": "Quick wins and low-hanging fruit",
                "improvements": low_effort[:3],
                "risk": "low"
            })
        
        # Phase 2: Core improvements (medium effort)
        if medium_effort:
            phases.append({
                "phase": 2,
                "duration": "1-3 months",
                "focus": "Core architectural improvements",
                "improvements": medium_effort[:5],
                "risk": "medium"
            })
        
        # Phase 3: Major refactoring (high effort)
        if high_effort:
            phases.append({
                "phase": 3,
                "duration": "3-6 months",
                "focus": "Major architectural changes",
                "improvements": high_effort[:3],
                "risk": "high"
            })
        
        return {
            "strategy": "phased_migration",
            "phases": phases,
            "total_estimated_duration": f"{len(phases) * 2}-{len(phases) * 6} months",
            "recommendation": "Start with phase 1 to build momentum and demonstrate value"
        }
    
    def _requires_major_changes(self, improvements: List) -> bool:
        """Check if improvements require major changes."""
        return any(i.get("effort") == "high" for i in improvements)
    
    # Decision validation helper methods
    
    def _validate_single_decision(self, decision: Dict, context: Dict) -> Dict[str, Any]:
        """Validate a single architectural decision."""
        validation = {
            "completeness": self._check_decision_completeness(decision),
            "consistency": self._check_decision_consistency(decision, context),
            "alignment": self._check_decision_alignment(decision, context),
            "tradeoffs_considered": self._check_decision_tradeoffs(decision)
        }
        
        # Overall validation
        passing_checks = sum(1 for check in validation.values() if check.get("passed", False))
        total_checks = len(validation)
        
        validation["overall"] = {
            "passed": passing_checks == total_checks,
            "score": passing_checks / total_checks,
            "assessment": "strong" if passing_checks == total_checks else 
                         "adequate" if passing_checks >= total_checks * 0.7 else 
                         "weak"
        }
        
        return validation
    
    def _check_decision_completeness(self, decision: Dict) -> Dict[str, Any]:
        """Check decision completeness."""
        required_fields = ["description", "rationale", "alternatives_considered"]
        missing_fields = [field for field in required_fields if field not in decision]
        
        return {
            "passed": len(missing_fields) == 0,
            "missing_fields": missing_fields,
            "message": "Complete" if len(missing_fields) == 0 else f"Missing: {', '.join(missing_fields)}"
        }
    
    def _check_decision_consistency(self, decision: Dict, context: Dict) -> Dict[str, Any]:
        """Check decision consistency with context."""
        # Check if decision aligns with architectural principles in context
        principles = context.get("principles", [])
        decision_desc = str(decision.get("description", "")).lower()
        
        inconsistencies = []
        for principle in principles[:3]:  # Check against top 3 principles
            principle_lower = principle.lower()
            # Simple keyword check
            if "modular" in principle_lower and "monolithic" in decision_desc:
                inconsistencies.append(f"Decision seems monolithic but principle emphasizes modularity")
            if "scalable" in principle_lower and "single_server" in decision_desc:
                inconsistencies.append(f"Decision limits scalability but principle emphasizes scalability")
        
        return {
            "passed": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies,
            "message": "Consistent" if len(inconsistencies) == 0 else f"Inconsistent: {', '.join(inconsistencies[:2])}"
        }
    
    def _check_decision_alignment(self, decision: Dict, context: Dict) -> Dict[str, Any]:
        """Check decision alignment with requirements."""
        requirements = context.get("requirements", [])
        decision_desc = str(decision.get("description", "")).lower()
        
        alignment_issues = []
        for req in requirements[:3]:  # Check against top 3 requirements
            req_lower = str(req).lower()
            # Check if decision addresses requirement
            if "security" in req_lower and "security" not in decision_desc:
                alignment_issues.append("Security requirement not addressed")
            if "performance" in req_lower and "performance" not in decision_desc:
                alignment_issues.append("Performance requirement not addressed")
        
        return {
            "passed": len(alignment_issues) == 0,
            "alignment_issues": alignment_issues,
            "message": "Aligned" if len(alignment_issues) == 0 else f"Alignment issues: {', '.join(alignment_issues[:2])}"
        }
    
    def _check_decision_tradeoffs(self, decision: Dict) -> Dict[str, Any]:
        """Check if tradeoffs were considered."""
        rationale = str(decision.get("rationale", "")).lower()
        alternatives = decision.get("alternatives_considered", [])
        
        tradeoff_indicators = [
            "tradeoff", "trade-off", "pros and cons", "advantages and disadvantages",
            "benefits and drawbacks", "strengths and weaknesses"
        ]
        
        has_tradeoff_discussion = any(indicator in rationale for indicator in tradeoff_indicators)
        has_alternatives = len(alternatives) > 0
        
        return {
            "passed": has_tradeoff_discussion or has_alternatives,
            "has_tradeoff_discussion": has_tradeoff_discussion,
            "has_alternatives": has_alternatives,
            "message": "Tradeoffs considered" if has_tradeoff_discussion or has_alternatives else "No tradeoff analysis"
        }
    
    def _assess_decision_quality(self, validation_results: List) -> Dict[str, Any]:
        """Assess overall decision quality."""
        if not validation_results:
            return {"score": 0, "assessment": "no_decisions"}
        
        scores = []
        for result in validation_results:
            overall = result.get("validation", {}).get("overall", {})
            scores.append(overall.get("score", 0))
        
        avg_score = sum(scores) / len(scores)
        
        return {
            "average_score": avg_score,
            "assessment": "excellent" if avg_score > 0.9 else 
                         "good" if avg_score > 0.7 else 
                         "fair" if avg_score > 0.5 else 
                         "poor",
            "recommendation": "No action needed" if avg_score > 0.7 else 
                             "Review decisions" if avg_score > 0.5 else 
                             "Major review required"
        }
    
    def _analyze_decision_risks(self, decisions: List, validation_results: List) -> Dict[str, Any]:
        """Analyze decision risks."""
        if not decisions:
            return {"risk_level": "low", "high_risk_decisions": []}
        
        high_risk_decisions = []
        for i, (decision, validation) in enumerate(zip(decisions, validation_results)):
            overall = validation.get("validation", {}).get("overall", {})
            if overall.get("score", 1) < 0.5:
                high_risk_decisions.append({
                    "decision": decision.get("description", f"Decision {i+1}"),
                    "risk_reasons": self._identify_risk_reasons(validation)
                })
        
        risk_level = "high" if len(high_risk_decisions) > 2 else \
                     "medium" if len(high_risk_decisions) > 0 else \
                     "low"
        
        return {
            "risk_level": risk_level,
            "high_risk_decisions": high_risk_decisions[:3],  # Limit to top 3
            "total_decisions": len(decisions),
            "high_risk_count": len(high_risk_decisions)
        }
    
    def _identify_risk_reasons(self, validation: Dict) -> List[str]:
        """Identify reasons for decision risk."""
        reasons = []
        validation_data = validation.get("validation", {})
        
        if not validation_data.get("completeness", {}).get("passed", True):
            reasons.append("Incomplete decision documentation")
        
        if not validation_data.get("consistency", {}).get("passed", True):
            reasons.append("Inconsistent with architectural principles")
        
        if not validation_data.get("alignment", {}).get("passed", True):
            reasons.append("Not aligned with requirements")
        
        if not validation_data.get("tradeoffs_considered", {}).get("passed", True):
            reasons.append("No tradeoff analysis")
        
        return reasons if reasons else ["Unspecified risk factors"]
    
    def _assess_decision_rationale(self, decision: Dict) -> str:
        """Assess decision rationale quality."""
        rationale = decision.get("rationale", "")
        
        if not rationale:
            return "missing"
        
        rationale_lower = rationale.lower()
        
        # Check for key elements
        elements = ["because", "therefore", "since", "due to", "as a result"]
        has_connectors = any(element in rationale_lower for element in elements)
        
        # Check length
        word_count = len(rationale.split())
        
        if word_count < 10:
            return "too_brief"
        elif word_count > 100:
            return "too_verbose"
        elif has_connectors:
            return "well_reasoned"
        else:
            return "adequate"
    
    def _check_alternatives_considered(self, decision: Dict) -> bool:
        """Check if alternatives were considered."""
        alternatives = decision.get("alternatives_considered", [])
        return len(alternatives) > 0
    
    def _has_high_risk_decisions(self, validation_results: List) -> bool:
        """Check if there are high-risk decisions."""
        for result in validation_results:
            overall = result.get("validation", {}).get("overall", {})
            if overall.get("score", 1) < 0.5:
                return True
        return False
    
    # Component design helper methods
    
    def _design_component_for_requirement(self, requirement: Dict, constraints: Dict) -> Optional[Dict[str, Any]]:
        """Design component for a requirement."""
        if not requirement:
            return None
        
        req_type = requirement.get("type", "functional")
        req_desc = requirement.get("description", "")
        
        # Simple component design based on requirement type
        if "user" in req_desc.lower() and "manage" in req_desc.lower():
            return {
                "name": "User Management Service",
                "responsibility": "Handle user lifecycle and authentication",
                "interfaces": ["create_user", "update_user", "delete_user", "authenticate"],
                "dependencies": ["Database", "Email Service"],
                "constraints": constraints.get("user_management", {})
            }
        elif "data" in req_desc.lower() and "process" in req_desc.lower():
            return {
                "name": "Data Processing Service",
                "responsibility": "Process and transform data",
                "interfaces": ["ingest_data", "process_data", "export_data"],
                "dependencies": ["Message Queue", "Storage Service"],
                "constraints": constraints.get("data_processing", {})
            }
        
        # Generic component
        return {
            "name": f"Service for {req_desc[:30]}",
            "responsibility": f"Handle {req_desc}",
            "interfaces": ["execute", "validate", "report"],
            "dependencies": [],
            "constraints": {}
        }
    
    async def _define_component_interfaces(self, component_designs: List, requirements: Dict) -> List[Dict[str, Any]]:
        """Define component interfaces."""
        interfaces = []
        
        for component in component_designs:
            component_name = component.get("name", "")
            component_interfaces = component.get("interfaces", [])
            
            for interface_name in component_interfaces:
                interfaces.append({
                    "component": component_name,
                    "interface": interface_name,
                    "protocol": self._determine_interface_protocol(interface_name, requirements),
                    "data_format": self._determine_data_format(interface_name, requirements),
                    "authentication_required": self._needs_authentication(interface_name)
                })
        
        return interfaces
    
    def _determine_interface_protocol(self, interface_name: str, requirements: Dict) -> str:
        """Determine interface protocol."""
        if any(word in interface_name.lower() for word in ["api", "rest", "http"]):
            return "REST"
        elif any(word in interface_name.lower() for word in ["event", "message", "queue"]):
            return "Message Queue"
        elif any(word in interface_name.lower() for word in ["stream", "real-time"]):
            return "WebSocket"
        else:
            return "RPC"
    
    def _determine_data_format(self, interface_name: str, requirements: Dict) -> str:
        """Determine data format."""
        if any(word in interface_name.lower() for word in ["json", "api"]):
            return "JSON"
        elif any(word in interface_name.lower() for word in ["xml", "soap"]):
            return "XML"
        elif any(word in interface_name.lower() for word in ["binary", "protobuf"]):
            return "Protobuf"
        else:
            return "JSON"
    
    def _needs_authentication(self, interface_name: str) -> bool:
        """Determine if interface needs authentication."""
        sensitive_operations = ["create", "update", "delete", "authenticate", "authorize"]
        return any(op in interface_name.lower() for op in sensitive_operations)
    
    def _explain_design_rationale(self, component_designs: List, requirements: Dict) -> Dict[str, Any]:
        """Explain design rationale."""
        rationale = {
            "separation_of_concerns": "Components divided based on distinct responsibilities",
            "reusability": "Designed interfaces to allow component reuse across the system",
            "maintainability": "Clear boundaries and minimal dependencies for easier maintenance"
        }
        
        # Add specific rationales
        if requirements.get("scalability"):
            rationale["scalability"] = "Stateless design and independent deployment for horizontal scaling"
        
        if requirements.get("security"):
            rationale["security"] = "Authentication on sensitive interfaces, principle of least privilege"
        
        return rationale
    
    def _provide_implementation_guidance(self, component_designs: List, interfaces: List) -> Dict[str, Any]:
        """Provide implementation guidance."""
        guidance = {
            "technology_selection": {
                "backend": "Choose based on team expertise and performance requirements",
                "database": "Consider SQL for transactions, NoSQL for scalability",
                "messaging": "Use message queues for async communication between components"
            },
            "testing_strategy": {
                "unit_tests": "Test each component in isolation",
                "integration_tests": "Test interfaces between components",
                "contract_tests": "Ensure interface compatibility"
            },
            "deployment": {
                "containers": "Package components in containers for consistent deployment",
                "orchestration": "Use Kubernetes or similar for container orchestration",
                "monitoring": "Implement comprehensive logging and monitoring"
            }
        }
        
        return guidance
    
    def _has_complex_interactions(self, interfaces: List) -> bool:
        """Check if there are complex component interactions."""
        if not interfaces:
            return False
        
        # Count interfaces per component
        from collections import Counter
        component_counts = Counter(i["component"] for i in interfaces)
        
        # Consider complex if any component has many interfaces
        return any(count > 5 for count in component_counts.values())
    
    # Pattern evaluation helper methods
    
    def _evaluate_pattern_for_architecture(self, pattern_name: str, pattern_info: Dict, 
                                          current_architecture: Dict, requirements: Dict) -> Dict[str, Any]:
        """Evaluate pattern for current architecture."""
        evaluation = {
            "applicability": self._assess_pattern_applicability(pattern_name, current_architecture, requirements),
            "benefits": pattern_info.get("benefits", []),
            "drawbacks": pattern_info.get("drawbacks", []),
            "fit_score": self._calculate_pattern_fit_score(pattern_name, current_architecture, requirements)
        }
        
        return evaluation
    
    def _assess_pattern_applicability(self, pattern_name: str, current_architecture: Dict, 
                                     requirements: Dict) -> str:
        """Assess pattern applicability."""
        current_style = current_architecture.get("style", "unknown")
        
        applicability_map = {
            "microservices": {
                "high": ["monolithic", "layered"],
                "medium": ["event_driven"],
                "low": ["microservices"]
            },
            "layered": {
                "high": ["monolithic"],
                "medium": ["microservices", "event_driven"],
                "low": ["layered"]
            },
            "event_driven": {
                "high": ["monolithic", "layered"],
                "medium": ["microservices"],
                "low": ["event_driven"]
            },
            "hexagonal": {
                "high": ["monolithic", "layered"],
                "medium": ["microservices"],
                "low": ["hexagonal", "event_driven"]
            }
        }
        
        if pattern_name in applicability_map:
            pattern_applicability = applicability_map[pattern_name]
            for applicability, styles in pattern_applicability.items():
                if current_style in styles:
                    return applicability
        
        return "unknown"
    
    def _calculate_pattern_fit_score(self, pattern_name: str, current_architecture: Dict, 
                                    requirements: Dict) -> float:
        """Calculate pattern fit score."""
        # Base score based on applicability
        applicability = self._assess_pattern_applicability(pattern_name, current_architecture, requirements)
        applicability_scores = {"high": 0.8, "medium": 0.5, "low": 0.2, "unknown": 0.1}
        base_score = applicability_scores.get(applicability, 0.1)
        
        # Adjust based on requirements
        if requirements.get("scalability") and pattern_name == "microservices":
            base_score += 0.15
        
        if requirements.get("maintainability") and pattern_name == "layered":
            base_score += 0.1
        
        if requirements.get("testability") and pattern_name == "hexagonal":
            base_score += 0.1
        
        return min(max(base_score, 0.0), 1.0)
    
    async def _recommend_architectural_patterns(self, current_architecture: Dict, 
                                               requirements: Dict) -> List[Dict[str, Any]]:
        """Recommend architectural patterns."""
        recommendations = []
        
        for pattern_name, pattern_info in self.architectural_patterns.items():
            evaluation = self._evaluate_pattern_for_architecture(
                pattern_name, pattern_info, current_architecture, requirements
            )
            
            if evaluation["fit_score"] > 0.6:
                recommendations.append({
                    "pattern": pattern_name,
                    "description": pattern_info.get("description", ""),
                    "fit_score": evaluation["fit_score"],
                    "recommendation_strength": self._determine_recommendation_strength(evaluation["fit_score"]),
                    "key_benefits": pattern_info.get("benefits", [])[:3]
                })
        
        # Sort by fit score
        return sorted(recommendations, key=lambda x: x["fit_score"], reverse=True)
    
    def _determine_recommendation_strength(self, fit_score: float) -> str:
        """Determine recommendation strength."""
        if fit_score > 0.8:
            return "strongly_recommended"
        elif fit_score > 0.6:
            return "recommended"
        elif fit_score > 0.4:
            return "consider"
        else:
            return "not_recommended"
    
    def _analyze_pattern_suitability(self, pattern_evaluations: List, requirements: Dict) -> Dict[str, Any]:
        """Analyze pattern suitability."""
        if not pattern_evaluations:
            return {"best_pattern": "unknown", "suitability_score": 0}
        
        # Find best pattern
        best_pattern = max(pattern_evaluations, key=lambda x: x.get("fit_score", 0))
        
        return {
            "best_pattern": best_pattern.get("pattern", "unknown"),
            "best_fit_score": best_pattern.get("fit_score", 0),
            "suitability_distribution": self._get_suitability_distribution(pattern_evaluations),
            "recommendation": f"Consider {best_pattern.get('pattern')} with fit score {best_pattern.get('fit_score', 0):.2f}"
        }
    
    def _get_suitability_distribution(self, pattern_evaluations: List) -> Dict[str, int]:
        """Get suitability distribution."""
        distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
        for evaluation in pattern_evaluations:
            fit_score = evaluation.get("fit_score", 0)
            if fit_score > 0.8:
                distribution["excellent"] += 1
            elif fit_score > 0.6:
                distribution["good"] += 1
            elif fit_score > 0.4:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    def _plan_pattern_adoption(self, pattern_recommendations: List, current_architecture: Dict) -> Dict[str, Any]:
        """Plan pattern adoption."""
        if not pattern_recommendations:
            return {"adoption_strategy": "no_changes", "timeline": "N/A"}
        
        top_recommendation = pattern_recommendations[0]
        pattern_name = top_recommendation.get("pattern", "")
        
        adoption_plans = {
            "microservices": {
                "strategy": "strangler_fig",
                "approach": "Gradually extract services from monolith",
                "phases": ["Identify bounded contexts", "Extract first service", "Establish communication", "Continue extraction"],
                "timeline": "6-18 months"
            },
            "layered": {
                "strategy": "refactoring",
                "approach": "Refactor into clear layers",
                "phases": ["Define layer boundaries", "Extract presentation layer", "Extract business logic", "Extract data access"],
                "timeline": "3-9 months"
            },
            "event_driven": {
                "strategy": "event_enrichment",
                "approach": "Introduce events alongside existing communication",
                "phases": ["Identify key events", "Introduce event bus", "Migrate components to events", "Decommission old communication"],
                "timeline": "4-12 months"
            }
        }
        
        plan = adoption_plans.get(pattern_name, {
            "strategy": "gradual_migration",
            "approach": "Plan based on specific pattern characteristics",
            "phases": ["Assessment", "Planning", "Pilot", "Full migration"],
            "timeline": "3-12 months"
        })
        
        return {
            "recommended_pattern": pattern_name,
            "adoption_plan": plan,
            "risk_assessment": self._assess_adoption_risk(pattern_name, current_architecture)
        }
    
    def _assess_adoption_risk(self, pattern_name: str, current_architecture: Dict) -> str:
        """Assess adoption risk."""
        risk_levels = {
            "microservices": "high",  # Distributed system complexity
            "event_driven": "medium",  # Async programming complexity
            "layered": "low",  # Well-understood pattern
            "hexagonal": "medium"  # Architectural complexity
        }
        
        return risk_levels.get(pattern_name, "medium")
    
    def _has_pattern_mismatches(self, pattern_evaluations: List) -> bool:
        """Check for pattern mismatches."""
        # Look for patterns with very low fit scores
        return any(e.get("fit_score", 1) < 0.3 for e in pattern_evaluations)
    
    # Documentation helper methods
    
    def _generate_architecture_overview(self, architecture: Dict) -> Dict[str, Any]:
        """Generate architecture overview."""
        return {
            "system_purpose": architecture.get("purpose", "Not specified"),
            "key_drivers": architecture.get("drivers", ["Not specified"]),
            "architectural_style": architecture.get("style", "unknown"),
            "key_components": list(architecture.get("components", {}).keys())[:5],
            "quality_attributes": architecture.get("quality_attributes", ["Not specified"])
        }
    
    def _document_architecture_components(self, architecture: Dict) -> List[Dict[str, Any]]:
        """Document architecture components."""
        components = architecture.get("components", {})
        
        documented_components = []
        for name, component in list(components.items())[:10]:  # Limit to 10 components
            documented_components.append({
                "name": name,
                "responsibility": component.get("responsibility", "Not specified"),
                "interfaces": component.get("interfaces", []),
                "dependencies": component.get("dependencies", []),
                "technology": component.get("technology", "Not specified")
            })
        
        return documented_components
    
    def _document_component_interactions(self, architecture: Dict) -> List[Dict[str, Any]]:
        """Document component interactions."""
        interactions = []
        components = architecture.get("components", {})
        
        # Create simplified interaction diagram
        for name, component in list(components.items())[:5]:  # Limit to 5 components
            deps = component.get("dependencies", [])
            for dep in deps[:3]:  # Limit to 3 dependencies per component
                interactions.append({
                    "from": name,
                    "to": dep,
                    "protocol": self._infer_interaction_protocol(component, dep),
                    "purpose": f"{name} depends on {dep}"
                })
        
        return interactions
    
    def _infer_interaction_protocol(self, component: Dict, dependency: str) -> str:
        """Infer interaction protocol."""
        # Simple inference based on component characteristics
        if any(word in str(component).lower() for word in ["api", "rest"]):
            return "HTTP/REST"
        elif any(word in str(component).lower() for word in ["queue", "message"]):
            return "Message Queue"
        elif any(word in str(component).lower() for word in ["database", "db"]):
            return "Database"
        else:
            return "Direct Call"
    
    def _document_architecture_decisions(self, architecture: Dict) -> List[Dict[str, Any]]:
        """Document architecture decisions."""
        decisions = architecture.get("decisions", [])
        
        documented_decisions = []
        for i, decision in enumerate(decisions[:5]):  # Limit to 5 decisions
            documented_decisions.append({
                "id": f"AD-{i+1:03d}",
                "description": decision.get("description", f"Decision {i+1}"),
                "status": decision.get("status", "proposed"),
                "rationale": decision.get("rationale", "Not documented"),
                "consequences": decision.get("consequences", ["Unknown"])
            })
        
        return documented_decisions
    
    def _document_quality_attributes(self, architecture: Dict) -> Dict[str, Any]:
        """Document quality attributes."""
        quality_attrs = architecture.get("quality_attributes", {})
        
        documented = {}
        for attr, strategies in quality_attrs.items():
            documented[attr] = {
                "importance": strategies.get("importance", "medium"),
                "strategies": strategies.get("strategies", ["Not specified"]),
                "metrics": strategies.get("metrics", ["Not specified"])
            }
        
        return documented
    
    def _add_technical_details(self, architecture: Dict) -> Dict[str, Any]:
        """Add technical details for developer audience."""
        return {
            "deployment_architecture": architecture.get("deployment", "Not specified"),
            "technology_stack": architecture.get("technologies", ["Not specified"]),
            "development_guidelines": architecture.get("guidelines", ["Not specified"]),
            "testing_strategy": architecture.get("testing", "Not specified"),
            "monitoring_approach": architecture.get("monitoring", "Not specified")
        }
    
    def _create_management_summary(self, architecture: Dict) -> Dict[str, Any]:
        """Create management summary."""
        return {
            "executive_summary": architecture.get("purpose", "System architecture overview"),
            "key_benefits": architecture.get("benefits", ["Improved scalability", "Enhanced maintainability"]),
            "cost_considerations": architecture.get("costs", ["Initial development", "Ongoing maintenance"]),
            "timeline": architecture.get("timeline", "Not specified"),
            "success_metrics": architecture.get("metrics", ["System availability", "Performance", "User satisfaction"])
        }
    
    def _assess_documentation_quality(self, documentation: Dict, architecture: Dict) -> Dict[str, Any]:
        """Assess documentation quality."""
        completeness_score = 0
        total_sections = 0
        
        # Check for key sections
        key_sections = ["overview", "components", "interactions", "decisions"]
        for section in key_sections:
            if section in documentation and documentation[section]:
                completeness_score += 1
            total_sections += 1
        
        # Check detail level
        detail_level = "basic"
        component_count = len(documentation.get("components", []))
        if component_count > 5:
            detail_level = "detailed"
        if component_count > 10:
            detail_level = "comprehensive"
        
        completeness = completeness_score / total_sections if total_sections > 0 else 0
        
        return {
            "completeness": completeness,
            "detail_level": detail_level,
            "assessment": "excellent" if completeness > 0.9 else 
                         "good" if completeness > 0.7 else 
                         "fair" if completeness > 0.5 else 
                         "poor"
        }
    
    def _check_audience_suitability(self, documentation: Dict, audience: str) -> Dict[str, Any]:
        """Check audience suitability."""
        suitability = {
            "developers": self._check_developer_suitability(documentation),
            "managers": self._check_manager_suitability(documentation),
            "architects": self._check_architect_suitability(documentation)
        }
        
        return {
            "target_audience": audience,
            "suitable_for_target": suitability.get(audience, False),
            "recommendations": self._generate_audience_recommendations(documentation, audience)
        }
    
    def _check_developer_suitability(self, documentation: Dict) -> bool:
        """Check if documentation is suitable for developers."""
        required = ["components", "interactions", "technical_details"]
        return all(section in documentation for section in required)
    
    def _check_manager_suitability(self, documentation: Dict) -> bool:
        """Check if documentation is suitable for managers."""
        return "summary" in documentation or "overview" in documentation
    
    def _check_architect_suitability(self, documentation: Dict) -> bool:
        """Check if documentation is suitable for architects."""
        required = ["decisions", "quality_attributes", "components"]
        return all(section in documentation for section in required)
    
    def _generate_audience_recommendations(self, documentation: Dict, audience: str) -> List[str]:
        """Generate audience-specific recommendations."""
        recommendations = []
        
        if audience == "developers" and "technical_details" not in documentation:
            recommendations.append("Add technical details section with deployment and development guidelines")
        
        if audience == "managers" and "summary" not in documentation:
            recommendations.append("Add executive summary with key benefits and costs")
        
        if audience == "architects" and "decisions" not in documentation:
            recommendations.append("Add architecture decisions with rationale and alternatives")
        
        return recommendations
    
    def _has_missing_architecture_info(self, architecture: Dict) -> bool:
        """Check if architecture information is missing."""
        required_fields = ["components", "style"]
        return any(field not in architecture for field in required_fields)
    
    # Architecture characteristic detection
    
    def _has_microservices_characteristics(self, components: List) -> bool:
        """Check for microservices characteristics."""
        if len(components) < 3:
            return False
        
        # Look for independent deployment indicators
        independent_count = 0
        for component in components:
            if component.get("independently_deployable", False):
                independent_count += 1
        
        return independent_count >= len(components) * 0.5  # At least half are independent
    
    def _has_layered_characteristics(self, project_structure: Dict) -> bool:
        """Check for layered architecture characteristics."""
        layers = ["presentation", "business", "data", "persistence"]
        structure_str = str(project_structure).lower()
        
        return any(layer in structure_str for layer in layers)
    
    def _has_event_driven_characteristics(self, components: List) -> bool:
        """Check for event-driven characteristics."""
        event_indicators = ["event", "message", "queue", "publish", "subscribe", "stream"]
        
        for component in components:
            component_str = str(component).lower()
            if any(indicator in component_str for indicator in event_indicators):
                return True
        
        return False
    
    def _supports_horizontal_scaling(self, components: List) -> bool:
        """Check if architecture supports horizontal scaling."""
        stateless_components = sum(1 for comp in components if not comp.get("stateful", True))
        return stateless_components >= len(components) * 0.7  # At least 70% stateless
    
    def _supports_mocking(self, components: List) -> bool:
        """Check if components support mocking for testing."""
        components_with_interfaces = sum(1 for comp in components if comp.get("interfaces"))
        return components_with_interfaces >= len(components) * 0.5  # At least half have defined interfaces
    
    # General assessment methods
    
    def _assess_overall_architecture(self, analysis_data: Dict, smells_data: Dict) -> Dict[str, Any]:
        """Assess overall architecture."""
        quality_score = analysis_data.get("quality_assessment", {}).get("overall_score", 0.5)
        smell_impact = smells_data.get("impact_assessment", {}).get("impact_score", 0)
        
        # Combine scores
        overall_score = (quality_score * 0.7) + ((1 - smell_impact) * 0.3)
        
        return {
            "overall_score": overall_score,
            "assessment": "excellent" if overall_score > 0.8 else 
                         "good" if overall_score > 0.6 else 
                         "fair" if overall_score > 0.4 else 
                         "needs_improvement",
            "strengths": analysis_data.get("strengths", []),
            "weaknesses": analysis_data.get("weaknesses", []),
            "recommendation": self._generate_overall_recommendation(overall_score, smells_data)
        }
    
    def _extract_key_findings(self, analysis_data: Dict, smells_data: Dict) -> List[str]:
        """Extract key findings from analysis."""
        findings = []
        
        # From architecture analysis
        style = analysis_data.get("architecture_analysis", {}).get("architectural_style", "unknown")
        findings.append(f"Architectural style: {style}")
        
        # From quality assessment
        quality = analysis_data.get("quality_assessment", {})
        for category, assessment in quality.items():
            if isinstance(assessment, dict) and "assessment" in assessment:
                findings.append(f"{category}: {assessment['assessment']}")
        
        # From smells
        smell_count = len(smells_data.get("architectural_smells", []))
        if smell_count > 0:
            findings.append(f"Found {smell_count} architectural smells")
        
        return findings[:5]  # Limit to top 5
    
    def _generate_overall_recommendation(self, overall_score: float, smells_data: Dict) -> str:
        """Generate overall recommendation."""
        if overall_score > 0.8:
            return "Architecture is in good shape. Continue current practices."
        elif overall_score > 0.6:
            return "Architecture is acceptable. Consider addressing minor issues."
        elif overall_score > 0.4:
            smell_count = len(smells_data.get("architectural_smells", []))
            return f"Architecture needs improvement. Address {smell_count} identified smells."
        else:
            return "Significant architectural improvements needed. Consider major refactoring."
    
    # Learning methods
    
    async def _learn_from_architecture_correction(self, feedback: Dict) -> bool:
        """Learn from architecture correction feedback."""
        correction = feedback.get("correction", {})
        
        # Update patterns based on correction
        if "false_positive" in correction:
            pattern_name = correction.get("pattern_name")
            if pattern_name and pattern_name in self.architectural_patterns:
                # Note the correction for future reference
                self._log_architecture_correction(pattern_name, correction)
        
        return True
    
    async def _learn_from_pattern_validation(self, feedback: Dict) -> bool:
        """Learn from pattern validation feedback."""
        validation = feedback.get("validation", {})
        pattern_name = validation.get("pattern_name")
        was_correct = validation.get("was_correct", False)
        
        if pattern_name and pattern_name in self.architectural_patterns:
            # Adjust confidence in this pattern
            adjustment = 0.1 if was_correct else -0.1
            self._adjust_pattern_confidence(pattern_name, adjustment)
        
        return True
    
    async def _update_design_principle(self, feedback: Dict) -> bool:
        """Update design principle based on feedback."""
        principle_update = feedback.get("principle_update", {})
        
        if "principle" in principle_update and "application" in principle_update:
            principle = principle_update["principle"]
            application = principle_update["application"]
            
            # Store in semantic memory
            self.store_memory(
                AgentMemoryType.SEMANTIC,
                {
                    "type": "design_principle_application",
                    "principle": principle,
                    "application": application,
                    "timestamp": datetime.now()
                }
            )
            
            return True
        
        return False
    
    async def _general_architecture_learning(self, feedback: Dict) -> bool:
        """General learning from architecture feedback."""
        # Update confidence based on feedback accuracy
        if "accuracy_rating" in feedback:
            rating = feedback["accuracy_rating"]
            if rating < 0.5:
                self.config.confidence_threshold = max(0.5, self.config.confidence_threshold - 0.05)
            elif rating > 0.8:
                self.config.confidence_threshold = min(0.95, self.config.confidence_threshold + 0.03)
        
        return True
    
    def _log_architecture_correction(self, pattern_name: str, correction: Dict) -> None:
        """Log architecture correction."""
        # Store correction in memory
        self.store_memory(
            AgentMemoryType.SEMANTIC,
            {
                "type": "architecture_correction",
                "pattern": pattern_name,
                "correction": correction,
                "timestamp": datetime.now()
            }
        )
    
    def _adjust_pattern_confidence(self, pattern_name: str, adjustment: float) -> None:
        """Adjust confidence in an architectural pattern."""
        # Would implement pattern confidence tracking
        pass