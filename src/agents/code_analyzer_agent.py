"""
CodeAnalyzerAgent - Specialized agent for code analysis.
"""

from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
from .base_agent import BaseAgent, AgentConfig, AgentInput, AgentOutput, AgentCapability, AgentMemoryType
from ..core.exceptions import ValidationError


class CodeAnalyzerAgent(BaseAgent):
    """
    Agent specialized in code analysis.
    
    Capabilities:
    1. Analyze code quality and complexity
    2. Detect bugs and issues
    3. Suggest improvements
    4. Review code style
    5. Calculate metrics
    6. Compare code versions
    7. Generate analysis reports
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="CodeAnalyzerAgent",
                description="Analyzes code quality, complexity, and patterns",
                capabilities=[
                    AgentCapability.CODE_ANALYSIS,
                    AgentCapability.PATTERN_DETECTION,
                    AgentCapability.REFACTORING_SUGGESTION
                ],
                confidence_threshold=0.8,
                learning_rate=0.2,
                dependencies=["indexer", "embeddings"]
            )
        
        super().__init__(config)
        self.code_patterns: Dict[str, Any] = {}
        self.quality_rules: Dict[str, Any] = {}
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        
    async def _initialize_internal(self) -> bool:
        """Initialize CodeAnalyzerAgent with code analysis patterns and rules."""
        try:
            # Load code analysis patterns
            self.code_patterns = await self._load_code_patterns()
            
            # Load quality rules
            self.quality_rules = await self._load_quality_rules()
            
            # Initialize analysis tools
            await self._initialize_analysis_tools()
            
            return True
        except Exception as e:
            print(f"Failed to initialize CodeAnalyzerAgent: {e}")
            return False
    
    async def _process_internal(self, input_data: AgentInput) -> AgentOutput:
        """Process code analysis request."""
        analysis_type = input_data.data.get("analysis_type", "general")
        
        try:
            if analysis_type == "quality":
                result = await self.analyze_code_quality(input_data)
            elif analysis_type == "bugs":
                result = await self.detect_bugs(input_data)
            elif analysis_type == "improvements":
                result = await self.suggest_improvements(input_data)
            elif analysis_type == "style":
                result = await self.review_code_style(input_data)
            elif analysis_type == "metrics":
                result = await self.calculate_metrics(input_data)
            elif analysis_type == "compare":
                result = await self.compare_code_versions(input_data)
            elif analysis_type == "report":
                result = await self.generate_analysis_report(input_data)
            else:
                # General analysis
                result = await self._perform_general_analysis(input_data)
            
            return result
        except Exception as e:
            return self._handle_error(e, {
                "request_id": input_data.request_id,
                "analysis_type": analysis_type
            })
    
    async def _learn_internal(self, feedback: Dict[str, Any]) -> bool:
        """Learn from code analysis feedback."""
        feedback_type = feedback.get("type")
        
        if feedback_type == "correction":
            return await self._learn_from_correction(feedback)
        elif feedback_type == "pattern":
            return await self._learn_new_pattern(feedback)
        elif feedback_type == "rule":
            return await self._update_quality_rule(feedback)
        else:
            # General learning
            return await self._general_learning(feedback)
    
    def _validate_input_specific(self, input_data: AgentInput) -> None:
        """Validate code analysis specific input."""
        if "code" not in input_data.data and "file_path" not in input_data.data:
            raise ValidationError("Code analysis requires 'code' or 'file_path' in input data")
        
        analysis_type = input_data.data.get("analysis_type", "general")
        valid_types = ["general", "quality", "bugs", "improvements", "style", "metrics", "compare", "report"]
        
        if analysis_type not in valid_types:
            raise ValidationError(f"analysis_type must be one of {valid_types}")
    
    async def _save_state(self) -> None:
        """Save CodeAnalyzerAgent state."""
        state_data = {
            "code_patterns": self.code_patterns,
            "quality_rules": self.quality_rules,
            "analysis_cache_keys": list(self._analysis_cache.keys()),
            "timestamp": datetime.now()
        }
        
        # Store in semantic memory
        self.store_memory(
            AgentMemoryType.SEMANTIC,
            {
                "type": "agent_state",
                "agent": "CodeAnalyzerAgent",
                "state": state_data
            }
        )
    
    # Public analysis methods
    
    async def analyze_code_quality(self, input_data: AgentInput) -> AgentOutput:
        """Analyze code quality."""
        code = input_data.data.get("code", "")
        language = input_data.data.get("language", "python")
        
        # Perform quality analysis
        quality_metrics = await self._calculate_quality_metrics(code, language)
        issues = await self._detect_quality_issues(code, language)
        
        confidence = self._calculate_quality_confidence(quality_metrics, issues)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "quality_metrics": quality_metrics,
                "issues": issues,
                "overall_score": self._calculate_overall_score(quality_metrics),
                "language": language
            },
            confidence=confidence,
            reasoning=["Analyzed code structure", "Checked quality rules", "Calculated metrics"],
            warnings=[] if confidence > 0.7 else ["Low confidence in analysis"]
        )
    
    async def detect_bugs(self, input_data: AgentInput) -> AgentOutput:
        """Detect bugs in code."""
        code = input_data.data.get("code", "")
        language = input_data.data.get("language", "python")
        
        # Detect bugs
        bugs = await self._scan_for_bugs(code, language)
        potential_bugs = await self._identify_potential_bugs(code, language)
        
        confidence = self._calculate_bug_detection_confidence(bugs, potential_bugs)
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "confirmed_bugs": bugs,
                "potential_bugs": potential_bugs,
                "bug_count": len(bugs) + len(potential_bugs),
                "severity_distribution": self._calculate_severity_distribution(bugs + potential_bugs)
            },
            confidence=confidence,
            reasoning=["Scanned for common bug patterns", "Analyzed control flow", "Checked error handling"],
            warnings=["Potential false positives in bug detection"] if len(potential_bugs) > 5 else []
        )
    
    async def suggest_improvements(self, input_data: AgentInput) -> AgentOutput:
        """Suggest code improvements."""
        code = input_data.data.get("code", "")
        language = input_data.data.get("language", "python")
        
        # Generate improvement suggestions
        improvements = await self._generate_improvement_suggestions(code, language)
        refactoring_ops = await self._suggest_refactoring_operations(code, language)
        
        confidence = 0.85  # High confidence for improvement suggestions
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "improvements": improvements,
                "refactoring_operations": refactoring_ops,
                "priority_suggestions": self._prioritize_suggestions(improvements + refactoring_ops),
                "estimated_effort": self._estimate_improvement_effort(improvements + refactoring_ops)
            },
            confidence=confidence,
            reasoning=["Analyzed code structure", "Identified improvement opportunities", "Prioritized suggestions"],
            warnings=[]
        )
    
    async def review_code_style(self, input_data: AgentInput) -> AgentOutput:
        """Review code style."""
        code = input_data.data.get("code", "")
        language = input_data.data.get("language", "python")
        style_guide = input_data.data.get("style_guide", "pep8")
        
        # Review code style
        style_issues = await self._check_code_style(code, language, style_guide)
        formatting_suggestions = await self._suggest_formatting_improvements(code, language, style_guide)
        
        confidence = 0.9  # High confidence for style checking
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "style_issues": style_issues,
                "formatting_suggestions": formatting_suggestions,
                "compliance_score": self._calculate_style_compliance(style_issues, code),
                "style_guide": style_guide
            },
            confidence=confidence,
            reasoning=["Checked against style guide", "Analyzed formatting", "Identified deviations"],
            warnings=[]
        )
    
    async def calculate_metrics(self, input_data: AgentInput) -> AgentOutput:
        """Calculate code metrics."""
        code = input_data.data.get("code", "")
        language = input_data.data.get("language", "python")
        
        # Calculate various metrics
        complexity_metrics = await self._calculate_complexity_metrics(code, language)
        maintainability_metrics = await self._calculate_maintainability_metrics(code, language)
        testability_metrics = await self._calculate_testability_metrics(code, language)
        
        confidence = 0.95  # Very high confidence for metric calculation
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "complexity_metrics": complexity_metrics,
                "maintainability_metrics": maintainability_metrics,
                "testability_metrics": testability_metrics,
                "overall_assessment": self._assess_overall_quality(
                    complexity_metrics, 
                    maintainability_metrics, 
                    testability_metrics
                )
            },
            confidence=confidence,
            reasoning=["Calculated cyclomatic complexity", "Assessed maintainability", "Evaluated testability"],
            warnings=[]
        )
    
    async def compare_code_versions(self, input_data: AgentInput) -> AgentOutput:
        """Compare different versions of code."""
        code_v1 = input_data.data.get("code_v1", "")
        code_v2 = input_data.data.get("code_v2", "")
        language = input_data.data.get("language", "python")
        
        # Compare versions
        differences = await self._find_code_differences(code_v1, code_v2, language)
        quality_changes = await self._analyze_quality_changes(code_v1, code_v2, language)
        impact_analysis = await self._analyze_change_impact(code_v1, code_v2, language)
        
        confidence = 0.8
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "differences": differences,
                "quality_changes": quality_changes,
                "impact_analysis": impact_analysis,
                "improvement_direction": self._determine_improvement_direction(quality_changes),
                "change_summary": self._generate_change_summary(differences, quality_changes)
            },
            confidence=confidence,
            reasoning=["Compared code structures", "Analyzed quality changes", "Assessed impact"],
            warnings=["Significant quality degradation detected"] if self._has_quality_degradation(quality_changes) else []
        )
    
    async def generate_analysis_report(self, input_data: AgentInput) -> AgentOutput:
        """Generate comprehensive analysis report."""
        code = input_data.data.get("code", "")
        language = input_data.data.get("language", "python")
        
        # Generate comprehensive report
        quality_analysis = await self.analyze_code_quality(input_data)
        bug_analysis = await self.detect_bugs(input_data)
        improvement_analysis = await self.suggest_improvements(input_data)
        
        # Compile report
        report = await self._compile_analysis_report(
            quality_analysis.data,
            bug_analysis.data,
            improvement_analysis.data,
            language
        )
        
        confidence = min(
            quality_analysis.confidence,
            bug_analysis.confidence,
            improvement_analysis.confidence
        )
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data=report,
            confidence=confidence,
            reasoning=["Performed comprehensive analysis", "Compiled detailed report", "Generated recommendations"],
            warnings=["Consider manual review for critical sections"] if report.get("critical_issues", 0) > 0 else []
        )
    
    # Helper methods
    
    async def _load_code_patterns(self) -> Dict[str, Any]:
        """Load code analysis patterns."""
        # In real implementation, would load from file or database
        return {
            "anti_patterns": [
                {"name": "god_object", "description": "Class with too many responsibilities"},
                {"name": "spaghetti_code", "description": "Complex and tangled control flow"},
                {"name": "magic_numbers", "description": "Unnamed numerical constants"},
                {"name": "duplicate_code", "description": "Identical or similar code in multiple places"}
            ],
            "design_patterns": [
                {"name": "singleton", "description": "Ensures a class has only one instance"},
                {"name": "factory", "description": "Creates objects without specifying exact class"},
                {"name": "observer", "description": "Publish-subscribe pattern for events"},
                {"name": "strategy", "description": "Defines a family of algorithms"}
            ],
            "code_smells": [
                {"name": "long_method", "threshold": 20},
                {"name": "large_class", "threshold": 200},
                {"name": "too_many_parameters", "threshold": 5},
                {"name": "feature_envy", "description": "Method uses more features of another class"}
            ]
        }
    
    async def _load_quality_rules(self) -> Dict[str, Any]:
        """Load code quality rules."""
        return {
            "complexity": {
                "cyclomatic_complexity": {"good": 10, "acceptable": 20, "bad": 30},
                "cognitive_complexity": {"good": 15, "acceptable": 30, "bad": 50}
            },
            "maintainability": {
                "lines_of_code": {"good": 100, "acceptable": 500, "bad": 1000},
                "comment_density": {"good": 20, "acceptable": 10, "bad": 5}
            },
            "testability": {
                "test_coverage": {"good": 80, "acceptable": 60, "bad": 40},
                "assertion_density": {"good": 0.3, "acceptable": 0.1, "bad": 0.05}
            }
        }
    
    async def _initialize_analysis_tools(self) -> None:
        """Initialize code analysis tools."""
        # Would initialize linters, parsers, etc.
        pass
    
    async def _perform_general_analysis(self, input_data: AgentInput) -> AgentOutput:
        """Perform general code analysis."""
        # Combine multiple analysis types
        quality_result = await self.analyze_code_quality(input_data)
        bug_result = await self.detect_bugs(input_data)
        
        confidence = (quality_result.confidence + bug_result.confidence) / 2
        
        return AgentOutput(
            request_id=input_data.request_id,
            agent_id=self.config.agent_id,
            success=True,
            data={
                "quality_analysis": quality_result.data,
                "bug_analysis": bug_result.data,
                "summary": {
                    "quality_score": quality_result.data.get("overall_score", 0),
                    "bug_count": bug_result.data.get("bug_count", 0),
                    "critical_issues": self._count_critical_issues(bug_result.data)
                }
            },
            confidence=confidence,
            reasoning=["Performed comprehensive code analysis", "Combined multiple analysis techniques"],
            warnings=[]
        )
    
    async def _calculate_quality_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate code quality metrics."""
        # Simplified implementation
        lines = code.count('\n') + 1
        functions = code.count('def ') if language == 'python' else code.count('function ')
        classes = code.count('class ') if language == 'python' else code.count('class ')
        
        return {
            "lines_of_code": lines,
            "function_count": functions,
            "class_count": classes,
            "average_function_length": lines / max(functions, 1),
            "comment_ratio": self._calculate_comment_ratio(code, language),
            "complexity_estimate": self._estimate_complexity(code, language)
        }
    
    async def _detect_quality_issues(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Detect quality issues in code."""
        issues = []
        
        # Check for common issues
        if len(code) > 1000:
            issues.append({
                "type": "size",
                "severity": "warning",
                "message": "File is very large, consider splitting",
                "location": "global"
            })
        
        # Check for long functions
        if language == 'python':
            lines = code.split('\n')
            in_function = False
            function_start = 0
            function_lines = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    if in_function and function_lines > 20:
                        issues.append({
                            "type": "long_function",
                            "severity": "warning",
                            "message": f"Function is {function_lines} lines long",
                            "location": f"line {function_start + 1}"
                        })
                    in_function = True
                    function_start = i
                    function_lines = 0
                elif in_function:
                    function_lines += 1
                    if line.strip() and not line.strip().startswith('#'):
                        function_lines += 1
        
        return issues
    
    def _calculate_quality_confidence(self, metrics: Dict[str, Any], issues: List[Dict]) -> float:
        """Calculate confidence in quality analysis."""
        base_confidence = 0.8
        
        # Adjust based on metrics availability
        if metrics.get("lines_of_code", 0) > 0:
            base_confidence += 0.1
        
        # Adjust based on issues found
        if issues:
            base_confidence -= 0.05 * min(len(issues), 4)
        
        return min(max(base_confidence, 0.3), 0.95)
    
    async def _scan_for_bugs(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Scan for bugs in code."""
        bugs = []
        
        # Common bug patterns
        bug_patterns = [
            ("None comparison with ==", "if variable == None:", "warning"),
            ("Mutable default argument", "def func(arg=[]):", "error"),
            ("Unused import", "import ", "info"),
            ("Infinite loop possibility", "while True:", "warning"),
            ("Division by zero risk", "/ 0", "error")
        ]
        
        for pattern, example, severity in bug_patterns:
            if example in code:
                bugs.append({
                    "pattern": pattern,
                    "severity": severity,
                    "message": f"Possible {pattern.lower()}",
                    "example": example,
                    "location": self._find_pattern_location(code, example)
                })
        
        return bugs
    
    async def _generate_improvement_suggestions(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Generate improvement suggestions."""
        suggestions = []
        
        # Analyze for improvement opportunities
        if language == 'python':
            # Check for list comprehensions
            if 'for ' in code and 'range(' in code and 'append(' in code:
                suggestions.append({
                    "type": "optimization",
                    "description": "Consider using list comprehension",
                    "example_before": "result = []\nfor i in range(10):\n    result.append(i*2)",
                    "example_after": "result = [i*2 for i in range(10)]",
                    "benefit": "More concise and often faster"
                })
            
            # Check for string concatenation
            if code.count('+') > 3 and '"' in code:
                suggestions.append({
                    "type": "style",
                    "description": "Consider using f-strings or .format()",
                    "example_before": '"Hello " + name + "! You are " + str(age) + " years old."',
                    "example_after": 'f"Hello {name}! You are {age} years old."',
                    "benefit": "More readable and efficient"
                })
        
        return suggestions
    
    # Learning methods
    
    async def _learn_from_correction(self, feedback: Dict) -> bool:
        """Learn from correction feedback."""
        correction = feedback.get("correction", {})
        
        # Update patterns based on correction
        if "false_positive" in correction:
            pattern = correction.get("pattern")
            if pattern and pattern in self.code_patterns.get("anti_patterns", []):
                # Reduce confidence in this pattern
                self._adjust_pattern_confidence(pattern, -0.1)
        
        return True
    
    async def _learn_new_pattern(self, feedback: Dict) -> bool:
        """Learn new code pattern from feedback."""
        pattern = feedback.get("pattern", {})
        
        if pattern and "name" in pattern:
            pattern_type = pattern.get("type", "anti_pattern")
            
            if pattern_type == "anti_pattern":
                self.code_patterns.setdefault("anti_patterns", []).append(pattern)
            elif pattern_type == "design_pattern":
                self.code_patterns.setdefault("design_patterns", []).append(pattern)
            
            return True
        
        return False
    
    async def _update_quality_rule(self, feedback: Dict) -> bool:
        """Update quality rule based on feedback."""
        rule_update = feedback.get("rule_update", {})
        
        if "metric" in rule_update and "thresholds" in rule_update:
            metric = rule_update["metric"]
            thresholds = rule_update["thresholds"]
            
            self.quality_rules.setdefault("custom", {})[metric] = thresholds
            return True
        
        return False
    
    async def _general_learning(self, feedback: Dict) -> bool:
        """General learning from feedback."""
        # Update confidence based on feedback
        if "accuracy_rating" in feedback:
            rating = feedback["accuracy_rating"]
            if rating < 0.5:
                self.config.confidence_threshold = max(0.5, self.config.confidence_threshold - 0.05)
            elif rating > 0.8:
                self.config.confidence_threshold = min(0.95, self.config.confidence_threshold + 0.03)
        
        return True
    
    def _adjust_pattern_confidence(self, pattern_name: str, adjustment: float) -> None:
        """Adjust confidence in a pattern."""
        # Would implement pattern confidence tracking
        pass
    
    # Utility methods
    
    def _calculate_comment_ratio(self, code: str, language: str) -> float:
        """Calculate comment to code ratio."""
        lines = code.split('\n')
        code_lines = 0
        comment_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped:
                if language == 'python' and stripped.startswith('#'):
                    comment_lines += 1
                elif language == 'javascript' and (stripped.startswith('//') or stripped.startswith('/*')):
                    comment_lines += 1
                else:
                    code_lines += 1
        
        return comment_lines / max(code_lines, 1)
    
    def _estimate_complexity(self, code: str, language: str) -> int:
        """Estimate code complexity."""
        complexity = 1
        
        # Count decision points
        decision_keywords = ['if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except ', 'case ']
        
        for keyword in decision_keywords:
            complexity += code.count(keyword)
        
        return complexity
    
    def _find_pattern_location(self, code: str, pattern: str) -> str:
        """Find location of pattern in code."""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if pattern in line:
                return f"line {i + 1}"
        return "unknown"
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall code quality score."""
        # Simplified scoring
        score = 100
        
        # Penalize for size
        loc = metrics.get("lines_of_code", 0)
        if loc > 500:
            score -= 20
        elif loc > 1000:
            score -= 40
        
        # Penalize for low comment ratio
        comment_ratio = metrics.get("comment_ratio", 0)
        if comment_ratio < 0.1:
            score -= 10
        elif comment_ratio < 0.05:
            score -= 20
        
        # Penalize for high complexity
        complexity = metrics.get("complexity_estimate", 1)
        if complexity > 50:
            score -= 30
        elif complexity > 20:
            score -= 15
        
        return max(0, min(100, score)) / 100.0
    
    def _calculate_bug_detection_confidence(self, bugs: List, potential_bugs: List) -> float:
        """Calculate confidence in bug detection."""
        base_confidence = 0.7
        
        # Higher confidence if we found confirmed bugs
        if bugs:
            base_confidence += 0.1
        
        # Lower confidence if many potential bugs
        if len(potential_bugs) > 10:
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.3), 0.95)
    
    def _calculate_severity_distribution(self, bugs: List) -> Dict[str, int]:
        """Calculate severity distribution of bugs."""
        distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        
        for bug in bugs:
            severity = bug.get("severity", "medium").lower()
            distribution[severity] = distribution.get(severity, 0) + 1
        
        return distribution
    
    def _prioritize_suggestions(self, suggestions: List) -> List:
        """Prioritize improvement suggestions."""
        # Simple prioritization by type
        priority_order = {"error": 0, "warning": 1, "optimization": 2, "style": 3}
        
        return sorted(
            suggestions,
            key=lambda x: priority_order.get(x.get("type", "style"), 4)
        )
    
    def _estimate_improvement_effort(self, suggestions: List) -> Dict[str, str]:
        """Estimate effort required for improvements."""
        effort_levels = {"low": 0, "medium": 0, "high": 0}
        
        for suggestion in suggestions:
            suggestion_type = suggestion.get("type", "style")
            
            if suggestion_type in ["error", "warning"]:
                effort_levels["high"] += 1
            elif suggestion_type == "optimization":
                effort_levels["medium"] += 1
            else:
                effort_levels["low"] += 1
        
        return effort_levels
    
    async def _check_code_style(self, code: str, language: str, style_guide: str) -> List[Dict]:
        """Check code against style guide."""
        # Simplified style checking
        issues = []
        
        # Check line length
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 79 and style_guide == "pep8":
                issues.append({
                    "type": "line_length",
                    "severity": "warning",
                    "message": f"Line {i+1} exceeds 79 characters",
                    "line": i + 1,
                    "length": len(line)
                })
        
        return issues
    
    def _calculate_style_compliance(self, issues: List, code: str) -> float:
        """Calculate style compliance score."""
        lines = len(code.split('\n'))
        if lines == 0:
            return 1.0
        
        issue_count = len([i for i in issues if i.get("severity") in ["error", "warning"]])
        
        compliance = 1.0 - (issue_count / max(lines, 1))
        return max(0.0, min(1.0, compliance))
    
    async def _calculate_complexity_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate complexity metrics."""
        return {
            "cyclomatic_complexity": self._estimate_complexity(code, language),
            "cognitive_complexity": self._estimate_complexity(code, language) * 1.5,
            "halstead_volume": self._estimate_halstead_volume(code, language)
        }
    
    def _estimate_halstead_volume(self, code: str, language: str) -> float:
        """Estimate Halstead volume."""
        # Simplified estimation
        operators = ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not']
        
        operator_count = 0
        for op in operators:
            operator_count += code.count(op)
        
        return operator_count * 2  # Very simplified
    
    async def _calculate_maintainability_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate maintainability metrics."""
        metrics = self._calculate_quality_metrics(code, language)
        
        return {
            "maintainability_index": 100 - (metrics.get("complexity_estimate", 0) * 0.5),
            "technical_debt": self._estimate_technical_debt(code, language),
            "duplication_rate": self._estimate_duplication_rate(code, language)
        }
    
    def _estimate_technical_debt(self, code: str, language: str) -> float:
        """Estimate technical debt."""
        issues = self._detect_quality_issues(code, language)
        return len(issues) * 0.5  # Simplified
    
    def _estimate_duplication_rate(self, code: str, language: str) -> float:
        """Estimate code duplication rate."""
        # Simplified estimation
        lines = code.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        
        if len(lines) == 0:
            return 0.0
        
        return 1.0 - (len(unique_lines) / len(lines))
    
    async def _calculate_testability_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate testability metrics."""
        return {
            "test_coverage_estimate": self._estimate_test_coverage(code, language),
            "assertion_density": self._calculate_assertion_density(code, language),
            "dependency_complexity": self._estimate_dependency_complexity(code, language)
        }
    
    def _estimate_test_coverage(self, code: str, language: str) -> float:
        """Estimate test coverage."""
        # Look for test-related code
        test_keywords = ['test_', 'Test', 'assert', 'expect', 'should', 'given_when_then']
        
        test_lines = 0
        total_lines = 0
        
        lines = code.split('\n')
        for line in lines:
            if line.strip():
                total_lines += 1
                if any(keyword in line for keyword in test_keywords):
                    test_lines += 1
        
        return test_lines / max(total_lines, 1)
    
    def _calculate_assertion_density(self, code: str, language: str) -> float:
        """Calculate assertion density."""
        assertion_keywords = ['assert', 'expect', 'verify', 'check']
        
        assertion_count = 0
        lines = code.split('\n')
        
        for line in lines:
            if any(keyword in line for keyword in assertion_keywords):
                assertion_count += 1
        
        return assertion_count / max(len(lines), 1)
    
    def _estimate_dependency_complexity(self, code: str, language: str) -> int:
        """Estimate dependency complexity."""
        dependency_keywords = ['import ', 'require', 'include', 'from ', 'using ']
        
        dependency_count = 0
        for keyword in dependency_keywords:
            dependency_count += code.count(keyword)
        
        return dependency_count
    
    def _assess_overall_quality(self, complexity: Dict, maintainability: Dict, testability: Dict) -> str:
        """Assess overall code quality."""
        scores = []
        
        # Complexity score (lower is better)
        cc = complexity.get("cyclomatic_complexity", 0)
        if cc < 10:
            scores.append(1.0)
        elif cc < 20:
            scores.append(0.7)
        elif cc < 30:
            scores.append(0.4)
        else:
            scores.append(0.1)
        
        # Maintainability score (higher is better)
        mi = maintainability.get("maintainability_index", 0)
        if mi > 85:
            scores.append(1.0)
        elif mi > 70:
            scores.append(0.7)
        elif mi > 50:
            scores.append(0.4)
        else:
            scores.append(0.1)
        
        # Testability score (higher is better)
        tc = testability.get("test_coverage_estimate", 0)
        if tc > 0.7:
            scores.append(1.0)
        elif tc > 0.5:
            scores.append(0.7)
        elif tc > 0.3:
            scores.append(0.4)
        else:
            scores.append(0.1)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score > 0.7:
            return "excellent"
        elif avg_score > 0.5:
            return "good"
        elif avg_score > 0.3:
            return "fair"
        else:
            return "poor"
    
    async def _find_code_differences(self, code_v1: str, code_v2: str, language: str) -> List[Dict]:
        """Find differences between code versions."""
        # Simplified diff
        lines_v1 = code_v1.split('\n')
        lines_v2 = code_v2.split('\n')
        
        differences = []
        max_lines = max(len(lines_v1), len(lines_v2))
        
        for i in range(max_lines):
            line_v1 = lines_v1[i] if i < len(lines_v1) else ""
            line_v2 = lines_v2[i] if i < len(lines_v2) else ""
            
            if line_v1 != line_v2:
                differences.append({
                    "line": i + 1,
                    "type": "modified" if line_v1 and line_v2 else ("added" if line_v2 else "removed"),
                    "old": line_v1,
                    "new": line_v2
                })
        
        return differences
    
    async def _analyze_quality_changes(self, code_v1: str, code_v2: str, language: str) -> Dict[str, Any]:
        """Analyze quality changes between versions."""
        metrics_v1 = await self._calculate_quality_metrics(code_v1, language)
        metrics_v2 = await self._calculate_quality_metrics(code_v2, language)
        
        changes = {}
        for key in metrics_v1:
            if key in metrics_v2:
                old_val = metrics_v1[key]
                new_val = metrics_v2[key]
                
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    change = new_val - old_val
                    percent_change = (change / old_val * 100) if old_val != 0 else 0
                    
                    changes[key] = {
                        "old": old_val,
                        "new": new_val,
                        "change": change,
                        "percent_change": percent_change,
                        "direction": "improvement" if (key == "comment_ratio" and change > 0) or 
                                                  (key != "comment_ratio" and change < 0) else "degradation"
                    }
        
        return changes
    
    async def _analyze_change_impact(self, code_v1: str, code_v2: str, language: str) -> Dict[str, Any]:
        """Analyze impact of code changes."""
        # Simplified impact analysis
        lines_v1 = code_v1.split('\n')
        lines_v2 = code_v2.split('\n')
        
        added_lines = len([l for l in lines_v2 if l.strip() and l not in lines_v1])
        removed_lines = len([l for l in lines_v1 if l.strip() and l not in lines_v2])
        modified_lines = len([i for i in range(min(len(lines_v1), len(lines_v2))) 
                              if lines_v1[i].strip() and lines_v2[i].strip() and lines_v1[i] != lines_v2[i]])
        
        return {
            "added_lines": added_lines,
            "removed_lines": removed_lines,
            "modified_lines": modified_lines,
            "total_changes": added_lines + removed_lines + modified_lines,
            "change_density": (added_lines + removed_lines + modified_lines) / max(len(lines_v1), 1)
        }
    
    def _determine_improvement_direction(self, quality_changes: Dict) -> str:
        """Determine if changes represent improvement or degradation."""
        improvements = 0
        degradations = 0
        
        for change_data in quality_changes.values():
            if change_data.get("direction") == "improvement":
                improvements += 1
            elif change_data.get("direction") == "degradation":
                degradations += 1
        
        if improvements > degradations * 2:
            return "significant improvement"
        elif improvements > degradations:
            return "improvement"
        elif degradations > improvements * 2:
            return "significant degradation"
        elif degradations > improvements:
            return "degradation"
        else:
            return "neutral"
    
    def _generate_change_summary(self, differences: List, quality_changes: Dict) -> Dict[str, Any]:
        """Generate summary of changes."""
        return {
            "total_differences": len(differences),
            "quality_metrics_changed": len(quality_changes),
            "improvements": sum(1 for c in quality_changes.values() if c.get("direction") == "improvement"),
            "degradations": sum(1 for c in quality_changes.values() if c.get("direction") == "degradation"),
            "most_significant_change": self._find_most_significant_change(quality_changes)
        }
    
    def _find_most_significant_change(self, quality_changes: Dict) -> Optional[Dict]:
        """Find the most significant quality change."""
        if not quality_changes:
            return None
        
        most_significant = None
        max_percent_change = 0
        
        for metric, change_data in quality_changes.items():
            percent_change = abs(change_data.get("percent_change", 0))
            
            if percent_change > max_percent_change:
                max_percent_change = percent_change
                most_significant = {
                    "metric": metric,
                    "percent_change": change_data.get("percent_change", 0),
                    "direction": change_data.get("direction", "neutral"),
                    "impact": "high" if percent_change > 50 else ("medium" if percent_change > 20 else "low")
                }
        
        return most_significant
    
    def _has_quality_degradation(self, quality_changes: Dict) -> bool:
        """Check if there is significant quality degradation."""
        degradations = sum(1 for c in quality_changes.values() if c.get("direction") == "degradation")
        improvements = sum(1 for c in quality_changes.values() if c.get("direction") == "improvement")
        
        return degradations > improvements * 1.5
    
    async def _compile_analysis_report(self, quality_data: Dict, bug_data: Dict, 
                                      improvement_data: Dict, language: str) -> Dict[str, Any]:
        """Compile comprehensive analysis report."""
        return {
            "summary": {
                "language": language,
                "analysis_timestamp": datetime.now().isoformat(),
                "overall_quality": quality_data.get("overall_score", 0),
                "critical_issues": self._count_critical_issues(bug_data),
                "improvement_opportunities": len(improvement_data.get("improvements", [])) + 
                                           len(improvement_data.get("refactoring_operations", []))
            },
            "quality_analysis": quality_data,
            "bug_analysis": bug_data,
            "improvement_analysis": improvement_data,
            "recommendations": {
                "immediate": self._extract_immediate_recommendations(bug_data, improvement_data),
                "short_term": self._extract_short_term_recommendations(quality_data, improvement_data),
                "long_term": self._extract_long_term_recommendations(quality_data)
            },
            "action_items": self._generate_action_items(quality_data, bug_data, improvement_data)
        }
    
    def _count_critical_issues(self, bug_data: Dict) -> int:
        """Count critical issues in bug data."""
        critical_count = 0
        
        for bugs in [bug_data.get("confirmed_bugs", []), bug_data.get("potential_bugs", [])]:
            for bug in bugs:
                if bug.get("severity") in ["critical", "error"]:
                    critical_count += 1
        
        return critical_count
    
    def _extract_immediate_recommendations(self, bug_data: Dict, improvement_data: Dict) -> List[str]:
        """Extract immediate recommendations."""
        recommendations = []
        
        # Critical bugs
        for bug in bug_data.get("confirmed_bugs", []):
            if bug.get("severity") in ["critical", "error"]:
                recommendations.append(f"Fix critical bug: {bug.get('message', 'Unknown')}")
        
        # High priority improvements
        for improvement in improvement_data.get("improvements", []):
            if improvement.get("type") in ["error", "warning"]:
                recommendations.append(f"Address: {improvement.get('description', 'Unknown')}")
        
        return recommendations[:5]  # Limit to top 5
    
    def _extract_short_term_recommendations(self, quality_data: Dict, improvement_data: Dict) -> List[str]:
        """Extract short-term recommendations."""
        recommendations = []
        
        # Quality issues
        for issue in quality_data.get("issues", []):
            if issue.get("severity") == "warning":
                recommendations.append(f"Improve: {issue.get('message', 'Unknown')}")
        
        # Optimization improvements
        for improvement in improvement_data.get("improvements", []):
            if improvement.get("type") == "optimization":
                recommendations.append(f"Optimize: {improvement.get('description', 'Unknown')}")
        
        return recommendations[:5]
    
    def _extract_long_term_recommendations(self, quality_data: Dict) -> List[str]:
        """Extract long-term recommendations."""
        recommendations = []
        
        # Based on quality metrics
        metrics = quality_data.get("quality_metrics", {})
        
        if metrics.get("lines_of_code", 0) > 500:
            recommendations.append("Consider refactoring into smaller modules")
        
        if metrics.get("complexity_estimate", 0) > 30:
            recommendations.append("Reduce code complexity through refactoring")
        
        if metrics.get("comment_ratio", 0) < 0.1:
            recommendations.append("Improve code documentation and comments")
        
        return recommendations
    
    def _generate_action_items(self, quality_data: Dict, bug_data: Dict, improvement_data: Dict) -> List[Dict[str, Any]]:
        """Generate actionable items from analysis."""
        action_items = []
        
        # Bug fixes
        for bug in bug_data.get("confirmed_bugs", []):
            action_items.append({
                "type": "bug_fix",
                "priority": "high" if bug.get("severity") in ["critical", "error"] else "medium",
                "description": bug.get("message", "Fix bug"),
                "estimated_effort": "1-2 hours"
            })
        
        # Quality improvements
        for issue in quality_data.get("issues", []):
            action_items.append({
                "type": "quality_improvement",
                "priority": "medium" if issue.get("severity") == "warning" else "low",
                "description": issue.get("message", "Improve code quality"),
                "estimated_effort": "2-4 hours"
            })
        
        # Optimizations
        for improvement in improvement_data.get("improvements", []):
            if improvement.get("type") == "optimization":
                action_items.append({
                    "type": "optimization",
                    "priority": "medium",
                    "description": improvement.get("description", "Optimize code"),
                    "estimated_effort": "3-5 hours"
                })
        
        return action_items[:10]  # Limit to top 10