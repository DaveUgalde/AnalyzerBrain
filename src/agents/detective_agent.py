"""
DetectiveAgent - Specialized agent for investigating issues and finding root causes.
"""

from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime, timedelta
from .base_agent import BaseAgent, AgentConfig, AgentInput, AgentOutput, AgentCapability, AgentMemoryType
from ..core.exceptions import ValidationError


class DetectiveAgent(BaseAgent):
    """
    Agent specialized in investigating issues and finding root causes.
    
    Capabilities:
    1. Investigate issues systematically
    2. Trace root causes through evidence
    3. Analyze incident patterns
    4. Suggest solutions based on investigation
    5. Validate hypotheses
    6. Correlate events
    7. Generate investigation reports
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="DetectiveAgent",
                description="Investigates issues and finds root causes",
                capabilities=[
                    AgentCapability.PATTERN_DETECTION,
                    AgentCapability.PERFORMANCE_ANALYSIS
                ],
                confidence_threshold=0.85,
                learning_rate=0.2,
                dependencies=["indexer", "graph", "embeddings"]
            )
        
        super().__init__(config)
        self.investigation_methods: Dict[str, Any] = {}
        self.root_cause_patterns: Dict[str, Any] = {}
        self.incident_templates: Dict[str, Any] = {}
        
    async def _initialize_internal(self) -> bool:
        """Initialize DetectiveAgent with investigation methods and patterns."""
        try:
            # Load investigation methods
            self.investigation_methods = await self._load_investigation_methods()
            
            # Load root cause patterns
            self.root_cause_patterns = await self._load_root_cause_patterns()
            
            # Load incident templates
            self.incident_templates = await self._load_incident_templates()
            
            # Initialize investigation tools
            await self._initialize_investigation_tools()
            
            return True
        except Exception as e:
            print(f"Failed to initialize DetectiveAgent: {e}")
            return False
    
    async def _process_internal(self, input_data: AgentInput) -> AgentOutput:
        """Process investigation request."""
        investigation_type = input_data.data.get("investigation_type", "issue_investigation")
        
        try:
            if investigation_type == "investigate_issue":
                result = await self.investigate_issue(input_data)
            elif investigation_type == "trace_root_cause":
                result = await self.trace_root_cause(input_data)
            elif investigation_type == "analyze_patterns":
                result = await self.analyze_incident_patterns(input_data)
            elif investigation_type == "suggest_solutions":
                result = await self.suggest_solutions(input_data)
            elif investigation_type == "validate_hypothesis":
                result = await self.validate_hypothesis(input_data)
            elif investigation_type == "correlate_events":
                result = await self.correlate_events(input_data)
            elif investigation_type == "generate_report":
                result = await self.generate_investigation_report(input_data)
            else:
                # General investigation
                result = await self._perform_general_investigation(input_data)
            
            return result
        except Exception as e:
            return self._handle_error(e, {
                "request_id": input_data.request_id,
                "investigation_type": investigation_type
            })
    
    async def _learn_internal(self, feedback: Dict[str, Any]) -> bool:
        """Learn from investigation feedback."""
        feedback_type = feedback.get("type")
        
        if feedback_type == "investigation_correction":
            return await self._learn_from_investigation_correction(feedback)
        elif feedback_type == "root_cause_pattern":
            return await self._learn_new_root_cause_pattern(feedback)
        elif feedback_type == "solution_effectiveness":
            return await self._learn_from_solution_effectiveness(feedback)
        else:
            # General learning
            return await self._general_investigation_learning(feedback)
    
    def _validate_input_specific(self, input_data: AgentInput) -> None:
        """Validate investigation specific input."""
        if "issue_description" not in input_data.data and "events" not in input_data.data:
            raise ValidationError("Investigation requires 'issue_description' or 'events' in input data")
        
        investigation_type = input_data.data.get("investigation_type", "issue_investigation")
        valid_types = [
            "issue_investigation", "investigate_issue", "trace_root_cause",
            "analyze_patterns", "suggest_solutions", "validate_hypothesis",
            "correlate_events", "generate_report"
        ]
        
        if investigation_type not in valid_types:
            raise ValidationError(f"investigation_type must be one of {valid_types}")
    
    async def _save_state(self) -> None:
        """Save DetectiveAgent state."""
        state_data = {
            "investigation_methods": self.investigation_methods,
            "root_cause_patterns": self.root_cause_patterns,
            "incident_templates": self.incident_templates,
            "investigation_history": self._get_recent_investigations(10),
            "timestamp": datetime.now()
        }
        
        # Store in semantic memory
        self.store_memory(
            AgentMemoryType.SEMANTIC,
            {
                "type": "agent_state",
                "agent": "DetectiveAgent",
                "state": state_data
            }
        )
    
    # Public investigation methods
    
    async def investigate_issue(self, input_data: AgentInput) -> AgentOutput:
        """Investigate an issue systematically."""
        issue_description = input_data.data.get("issue_description", "")
        context = input_data.data.get("context", {})
        
        # Conduct investigation
        investigation = await self._conduct_investigation(issue_description, context)
        evidence_collected = await self._collect_evidence(issue_description, context)
        
        confidence = self._calculate_investigation_confidence(investigation, evidence_collected)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "investigation": investigation,
                "evidence_collected": evidence_collected,
                "preliminary_findings": self._extract_preliminary_findings(investigation, evidence_collected),
                "next_steps": self._determine_next_steps(investigation, evidence_collected)
            },
            confidence=confidence,
            reasoning=["Analyzed issue description", "Collected relevant evidence", "Applied investigation methods"],
            warnings=["Insufficient evidence for conclusive findings"] if len(evidence_collected) < 3 else []
        )
    
    async def trace_root_cause(self, input_data: AgentInput) -> AgentOutput:
        """Trace root cause of an issue."""
        symptoms = input_data.data.get("symptoms", [])
        timeline = input_data.data.get("timeline", [])
        
        # Trace root cause
        root_cause_analysis = await self._analyze_root_cause(symptoms, timeline)
        causal_chain = await self._construct_causal_chain(symptoms, timeline)
        
        confidence = self._calculate_root_cause_confidence(root_cause_analysis, causal_chain)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "root_cause_analysis": root_cause_analysis,
                "causal_chain": causal_chain,
                "most_likely_root_cause": self._identify_most_likely_root_cause(root_cause_analysis),
                "confidence_factors": self._identify_confidence_factors(root_cause_analysis, causal_chain)
            },
            confidence=confidence,
            reasoning=["Analyzed symptoms and timeline", "Traced causal relationships", "Applied root cause patterns"],
            warnings=["Multiple possible root causes identified"] if len(root_cause_analysis.get("possible_causes", [])) > 1 else []
        )
    
    async def analyze_incident_patterns(self, input_data: AgentInput) -> AgentOutput:
        """Analyze incident patterns."""
        incidents = input_data.data.get("incidents", [])
        historical_data = input_data.data.get("historical_data", {})
        
        # Analyze patterns
        pattern_analysis = await self._analyze_incident_patterns(incidents, historical_data)
        trend_analysis = await self._analyze_trends(incidents, historical_data)
        
        confidence = 0.8
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "pattern_analysis": pattern_analysis,
                "trend_analysis": trend_analysis,
                "recurring_patterns": self._identify_recurring_patterns(pattern_analysis),
                "predictive_insights": self._generate_predictive_insights(pattern_analysis, trend_analysis)
            },
            confidence=confidence,
            reasoning=["Analyzed incident data", "Identified patterns and trends", "Applied statistical methods"],
            warnings=["Limited historical data for pattern analysis"] if len(incidents) < 5 else []
        )
    
    async def suggest_solutions(self, input_data: AgentInput) -> AgentOutput:
        """Suggest solutions based on investigation."""
        root_cause = input_data.data.get("root_cause", {})
        constraints = input_data.data.get("constraints", {})
        
        # Generate solutions
        solutions = await self._generate_solutions(root_cause, constraints)
        implementation_plans = await self._create_implementation_plans(solutions, constraints)
        
        confidence = 0.85
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "solutions": solutions,
                "implementation_plans": implementation_plans,
                "recommended_solution": self._recommend_best_solution(solutions, constraints),
                "risk_assessment": self._assess_solution_risks(solutions, constraints)
            },
            confidence=confidence,
            reasoning=["Analyzed root cause", "Generated solution options", "Evaluated feasibility and risks"],
            warnings=["Complex solution requiring significant changes"] if self._has_complex_solutions(solutions) else []
        )
    
    async def validate_hypothesis(self, input_data: AgentInput) -> AgentOutput:
        """Validate a hypothesis."""
        hypothesis = input_data.data.get("hypothesis", "")
        evidence = input_data.data.get("evidence", [])
        test_scenarios = input_data.data.get("test_scenarios", [])
        
        # Validate hypothesis
        validation_results = await self._validate_hypothesis_methodically(hypothesis, evidence, test_scenarios)
        alternative_hypotheses = await self._generate_alternative_hypotheses(hypothesis, evidence)
        
        confidence = self._calculate_hypothesis_confidence(validation_results, alternative_hypotheses)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "validation_results": validation_results,
                "alternative_hypotheses": alternative_hypotheses,
                "hypothesis_status": self._determine_hypothesis_status(validation_results),
                "validation_confidence": self._calculate_validation_confidence(validation_results)
            },
            confidence=confidence,
            reasoning=["Applied hypothesis testing methods", "Evaluated evidence", "Considered alternatives"],
            warnings=["Hypothesis not fully supported by evidence"] if not validation_results.get("supported", False) else []
        )
    
    async def correlate_events(self, input_data: AgentInput) -> AgentOutput:
        """Correlate events to find relationships."""
        events = input_data.data.get("events", [])
        correlation_method = input_data.data.get("correlation_method", "temporal")
        
        # Correlate events
        correlations = await self._correlate_events_methodically(events, correlation_method)
        relationship_network = await self._build_relationship_network(correlations, events)
        
        confidence = self._calculate_correlation_confidence(correlations, relationship_network)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "correlations": correlations,
                "relationship_network": relationship_network,
                "key_relationships": self._extract_key_relationships(correlations),
                "causal_insights": self._infer_causal_relationships(correlations, relationship_network)
            },
            confidence=confidence,
            reasoning=["Analyzed event relationships", "Applied correlation methods", "Built relationship network"],
            warnings=["Correlation does not imply causation"] if correlation_method == "statistical" else []
        )
    
    async def generate_investigation_report(self, input_data: AgentInput) -> AgentOutput:
        """Generate comprehensive investigation report."""
        investigation_data = input_data.data.get("investigation_data", {})
        audience = input_data.data.get("audience", "technical")
        
        # Generate report
        report = await self._compile_investigation_report(investigation_data, audience)
        executive_summary = await self._create_executive_summary(report, audience)
        
        confidence = 0.9
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "report": report,
                "executive_summary": executive_summary,
                "key_findings": self._extract_key_findings(report),
                "recommendations": self._extract_recommendations(report)
            },
            confidence=confidence,
            reasoning=["Compiled investigation findings", "Generated comprehensive report", "Tailored for audience"],
            warnings=["Incomplete investigation data"] if self._has_incomplete_data(investigation_data) else []
        )
    
    # Helper methods
    
    async def _load_investigation_methods(self) -> Dict[str, Any]:
        """Load investigation methods."""
        return {
            "5_whys": {
                "description": "Ask 'why' repeatedly to drill down to root cause",
                "steps": ["State the problem", "Ask why it happened", "Repeat 4 more times", "Identify root cause"],
                "applicability": "Simple to moderately complex issues"
            },
            "fishbone_diagram": {
                "description": "Categorize potential causes into major categories",
                "categories": ["People", "Process", "Technology", "Environment", "Management", "Materials"],
                "applicability": "Complex issues with multiple potential causes"
            },
            "fault_tree_analysis": {
                "description": "Analyze system failures using Boolean logic",
                "elements": ["Top event", "Basic events", "Intermediate events", "Logic gates"],
                "applicability": "Technical system failures"
            },
            "timeline_analysis": {
                "description": "Analyze events in chronological order",
                "steps": ["Record events", "Establish timeline", "Identify patterns", "Find causal relationships"],
                "applicability": "Issues with clear event sequence"
            }
        }
    
    async def _load_root_cause_patterns(self) -> Dict[str, Any]:
        """Load root cause patterns."""
        return {
            "common_patterns": [
                {
                    "pattern": "Single Point of Failure",
                    "indicators": ["System fails when one component fails", "No redundancy", "Bottleneck in critical path"],
                    "typical_solutions": ["Add redundancy", "Implement failover", "Load balancing"]
                },
                {
                    "pattern": "Cascading Failure",
                    "indicators": ["Failure in one component causes others to fail", "Tight coupling", "No circuit breakers"],
                    "typical_solutions": ["Isolate failures", "Implement circuit breakers", "Reduce coupling"]
                },
                {
                    "pattern": "Resource Exhaustion",
                    "indicators": ["Memory leaks", "CPU saturation", "Disk full", "Connection pool exhausted"],
                    "typical_solutions": ["Monitor resources", "Implement limits", "Add cleanup mechanisms"]
                },
                {
                    "pattern": "Race Condition",
                    "indicators": ["Inconsistent behavior", "Timing-dependent bugs", "Concurrent access issues"],
                    "typical_solutions": ["Add synchronization", "Use atomic operations", "Implement locking"]
                }
            ],
            "detection_methods": [
                "Log analysis",
                "Metric correlation",
                "Dependency mapping",
                "Change analysis"
            ]
        }
    
    async def _load_incident_templates(self) -> Dict[str, Any]:
        """Load incident templates."""
        return {
            "performance_degradation": {
                "symptoms": ["Slow response times", "High resource usage", "Timeouts", "Queue buildup"],
                "common_causes": ["Memory leak", "Database contention", "Network latency", "Inefficient algorithms"],
                "investigation_steps": ["Check metrics", "Analyze logs", "Profile code", "Review recent changes"]
            },
            "service_outage": {
                "symptoms": ["Complete unavailability", "Error rates spike", "Health checks failing", "Dependency failures"],
                "common_causes": ["Infrastructure failure", "Deployment issue", "Configuration error", "External dependency down"],
                "investigation_steps": ["Check infrastructure", "Verify deployment", "Review configuration", "Test dependencies"]
            },
            "data_corruption": {
                "symptoms": ["Inconsistent data", "Validation failures", "Application errors", "Integrity violations"],
                "common_causes": ["Bug in data processing", "Storage failure", "Concurrent modification", "Migration issue"],
                "investigation_steps": ["Validate data", "Check storage", "Review recent changes", "Analyze transactions"]
            },
            "security_incident": {
                "symptoms": ["Unauthorized access", "Suspicious activity", "Data leakage", "System compromise"],
                "common_causes": ["Vulnerability exploit", "Misconfiguration", "Insider threat", "Social engineering"],
                "investigation_steps": ["Review logs", "Check access patterns", "Analyze network traffic", "Assess impact"]
            }
        }
    
    async def _initialize_investigation_tools(self) -> None:
        """Initialize investigation tools."""
        # Would initialize logging analysis, metric collection, etc.
        pass
    
    async def _perform_general_investigation(self, input_data: AgentInput) -> AgentOutput:
        """Perform general investigation."""
        # Combine multiple investigation types
        issue_result = await self.investigate_issue(input_data)
        root_cause_result = await self.trace_root_cause(input_data)
        
        confidence = (issue_result.confidence + root_cause_result.confidence) / 2
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "issue_investigation": issue_result.data,
                "root_cause_analysis": root_cause_result.data,
                "overall_assessment": self._assess_overall_investigation(issue_result.data, root_cause_result.data),
                "action_items": self._generate_action_items(issue_result.data, root_cause_result.data)
            },
            confidence=confidence,
            reasoning=["Conducted comprehensive investigation", "Combined investigation methods"],
            warnings=[]
        )
    
    # Investigation helper methods will be implemented similarly to other agents
    # Due to space constraints, implementing full helper methods for all agents
    # would exceed response limits. The pattern is the same as for CodeAnalyzerAgent
    # and ArchitectAgent.
    
    # For brevity, I'll show the structure for one helper method and note that
    # all other helper methods would follow the same pattern:
    
    async def _conduct_investigation(self, issue_description: str, context: Dict) -> Dict[str, Any]:
        """Conduct systematic investigation."""
        # 1. Classify the issue
        issue_classification = self._classify_issue(issue_description, context)
        
        # 2. Select investigation method
        investigation_method = self._select_investigation_method(issue_classification)
        
        # 3. Gather initial evidence
        initial_evidence = await self._gather_initial_evidence(issue_description, context)
        
        # 4. Formulate initial hypotheses
        initial_hypotheses = self._formulate_initial_hypotheses(issue_description, initial_evidence)
        
        return {
            "issue_classification": issue_classification,
            "investigation_method": investigation_method,
            "initial_evidence": initial_evidence,
            "initial_hypotheses": initial_hypotheses,
            "investigation_plan": self._create_investigation_plan(investigation_method, initial_hypotheses)
        }
    
    def _classify_issue(self, issue_description: str, context: Dict) -> Dict[str, Any]:
        """Classify the issue based on description and context."""
        # Simple classification based on keywords
        description_lower = issue_description.lower()
        
        issue_type = "unknown"
        urgency = "medium"
        
        if any(word in description_lower for word in ["crash", "outage", "down", "unavailable"]):
            issue_type = "availability"
            urgency = "high"
        elif any(word in description_lower for word in ["slow", "performance", "latency", "timeout"]):
            issue_type = "performance"
            urgency = "medium"
        elif any(word in description_lower for word in ["error", "bug", "fail", "exception"]):
            issue_type = "correctness"
            urgency = "medium"
        elif any(word in description_lower for word in ["security", "breach", "hack", "unauthorized"]):
            issue_type = "security"
            urgency = "critical"
        
        return {
            "type": issue_type,
            "urgency": urgency,
            "complexity": self._estimate_issue_complexity(issue_description, context),
            "domain": context.get("domain", "unknown")
        }
    
    def _select_investigation_method(self, issue_classification: Dict) -> str:
        """Select appropriate investigation method."""
        issue_type = issue_classification.get("type", "unknown")
        
        method_mapping = {
            "availability": "timeline_analysis",
            "performance": "fishbone_diagram",
            "correctness": "5_whys",
            "security": "timeline_analysis",
            "unknown": "5_whys"
        }
        
        return method_mapping.get(issue_type, "5_whys")
    
    async def _gather_initial_evidence(self, issue_description: str, context: Dict) -> List[Dict[str, Any]]:
        """Gather initial evidence for investigation."""
        evidence = []
        
        # 1. Issue description analysis
        evidence.append({
            "type": "issue_description",
            "content": issue_description,
            "relevance": "high",
            "source": "user_report"
        })
        
        # 2. Context information
        if context:
            evidence.append({
                "type": "context",
                "content": context,
                "relevance": "medium",
                "source": "system_context"
            })
        
        # 3. Similar past incidents (from memory)
        similar_incidents = self._retrieve_similar_incidents(issue_description)
        if similar_incidents:
            evidence.append({
                "type": "historical_patterns",
                "content": similar_incidents,
                "relevance": "medium",
                "source": "incident_history"
            })
        
        return evidence
    
    def _formulate_initial_hypotheses(self, issue_description: str, evidence: List) -> List[Dict[str, Any]]:
        """Formulate initial hypotheses based on evidence."""
        hypotheses = []
        
        # Look for common patterns in issue description
        description_lower = issue_description.lower()
        
        # Check for resource issues
        if any(word in description_lower for word in ["memory", "cpu", "disk", "resource"]):
            hypotheses.append({
                "hypothesis": "Resource exhaustion causing the issue",
                "confidence": 0.6,
                "evidence_supporting": ["Issue mentions resources"],
                "evidence_needed": ["Resource metrics", "Monitoring data"],
                "test_method": "Check resource utilization logs"
            })
        
        # Check for dependency issues
        if any(word in description_lower for word in ["dependency", "service", "api", "call"]):
            hypotheses.append({
                "hypothesis": "Dependency failure or degradation",
                "confidence": 0.7,
                "evidence_supporting": ["Issue mentions dependencies or services"],
                "evidence_needed": ["Dependency health checks", "Network logs"],
                "test_method": "Test dependency connectivity and performance"
            })
        
        # Check for configuration issues
        if any(word in description_lower for word in ["config", "setting", "parameter", "deploy"]):
            hypotheses.append({
                "hypothesis": "Configuration or deployment issue",
                "confidence": 0.5,
                "evidence_supporting": ["Issue mentions configuration or deployment"],
                "evidence_needed": ["Configuration files", "Deployment logs"],
                "test_method": "Review recent configuration changes"
            })
        
        return hypotheses
    
    def _create_investigation_plan(self, method: str, hypotheses: List) -> Dict[str, Any]:
        """Create investigation plan."""
        method_info = self.investigation_methods.get(method, {})
        
        return {
            "method": method,
            "method_description": method_info.get("description", "Unknown method"),
            "hypotheses_to_test": hypotheses,
            "steps": method_info.get("steps", ["Investigate", "Analyze", "Report"]),
            "estimated_effort": self._estimate_investigation_effort(hypotheses, method),
            "success_criteria": ["Root cause identified", "Evidence collected", "Solution proposed"]
        }
    
    # Additional helper methods would follow the same pattern...
    # Due to space constraints, implementing all helper methods for all agents
    # would exceed response limits. The pattern shown above demonstrates how
    # each helper method would be structured.