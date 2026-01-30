-- Exportación de Grafo de Conocimiento - Project Brain
-- Proyecto: {{PROJECT_NAME}} (ID: {{PROJECT_ID}})
-- Fecha: {{TIMESTAMP}}
-- Versión: 1.0.1

-- ============================================
-- CONFIGURACIÓN INICIAL
-- ============================================

-- ⚠️ OPCIONAL: limpiar base
-- MATCH (n) DETACH DELETE n;

-- Constraints
CREATE CONSTRAINT file_id_unique IF NOT EXISTS
FOR (f:File) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT function_id_unique IF NOT EXISTS
FOR (f:Function) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT class_id_unique IF NOT EXISTS
FOR (c:Class) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT variable_id_unique IF NOT EXISTS
FOR (v:Variable) REQUIRE v.id IS UNIQUE;

-- Índices
CREATE INDEX file_name_index IF NOT EXISTS FOR (f:File) ON (f.name);
CREATE INDEX function_name_index IF NOT EXISTS FOR (f:Function) ON (f.name);
CREATE INDEX class_name_index IF NOT EXISTS FOR (c:Class) ON (c.name);

-- ============================================
-- NODOS: ARCHIVOS
-- ============================================

{{#FILES}}
MERGE (file_{{ID}}:File {id: "{{ID}}"})
ON CREATE SET
  file_{{ID}}.created_at = datetime("{{CREATED_AT}}")
SET
  file_{{ID}}.name = "{{NAME}}",
  file_{{ID}}.path = "{{PATH}}",
  file_{{ID}}.extension = "{{EXTENSION}}",
  file_{{ID}}.language = "{{LANGUAGE}}",
  file_{{ID}}.size_bytes = {{SIZE_BYTES}},
  file_{{ID}}.line_count = {{LINE_COUNT}},
  file_{{ID}}.content_hash = "{{CONTENT_HASH}}",
  file_{{ID}}.encoding = "{{ENCODING}}",
  file_{{ID}}.complexity_level = "{{COMPLEXITY_LEVEL}}",
  file_{{ID}}.maintainability_index = {{MAINTAINABILITY_INDEX}},
  file_{{ID}}.test_coverage = {{TEST_COVERAGE}},
  file_{{ID}}.issue_count = {{ISSUE_COUNT}},
  file_{{ID}}.updated_at = datetime("{{UPDATED_AT}}");
{{/FILES}}

-- ============================================
-- NODOS: FUNCIONES
-- ============================================

{{#FUNCTIONS}}
MERGE (func_{{ID}}:Function {id: "{{ID}}"})
ON CREATE SET
  func_{{ID}}.created_at = datetime("{{CREATED_AT}}")
SET
  func_{{ID}}.name = "{{NAME}}",
  func_{{ID}}.signature = "{{SIGNATURE}}",
  func_{{ID}}.return_type = "{{RETURN_TYPE}}",
  func_{{ID}}.start_line = {{START_LINE}},
  func_{{ID}}.end_line = {{END_LINE}},
  func_{{ID}}.start_column = {{START_COLUMN}},
  func_{{ID}}.end_column = {{END_COLUMN}},
  func_{{ID}}.is_async = {{IS_ASYNC}},
  func_{{ID}}.is_generator = {{IS_GENERATOR}},
  func_{{ID}}.is_coroutine = {{IS_COROUTINE}},
  func_{{ID}}.decorators = {{DECORATORS}},
  func_{{ID}}.docstring = "{{DOCSTRING}}",
  func_{{ID}}.cyclomatic_complexity = {{CYCLOMATIC_COMPLEXITY}},
  func_{{ID}}.cognitive_complexity = {{COGNITIVE_COMPLEXITY}},
  func_{{ID}}.lines_of_code = {{LINES_OF_CODE}},
  func_{{ID}}.parameter_count = {{PARAMETER_COUNT}},
  func_{{ID}}.updated_at = datetime("{{UPDATED_AT}}");
{{/FUNCTIONS}}

-- ============================================
-- NODOS: CLASES
-- ============================================

{{#CLASSES}}
MERGE (class_{{ID}}:Class {id: "{{ID}}"})
ON CREATE SET
  class_{{ID}}.created_at = datetime("{{CREATED_AT}}")
SET
  class_{{ID}}.name = "{{NAME}}",
  class_{{ID}}.full_name = "{{FULL_NAME}}",
  class_{{ID}}.bases = {{BASES}},
  class_{{ID}}.decorators = {{DECORATORS}},
  class_{{ID}}.docstring = "{{DOCSTRING}}",
  class_{{ID}}.is_abstract = {{IS_ABSTRACT}},
  class_{{ID}}.is_final = {{IS_FINAL}},
  class_{{ID}}.is_dataclass = {{IS_DATACLASS}},
  class_{{ID}}.access_modifier = "{{ACCESS_MODIFIER}}",
  class_{{ID}}.method_count = {{METHOD_COUNT}},
  class_{{ID}}.attribute_count = {{ATTRIBUTE_COUNT}},
  class_{{ID}}.inheritance_depth = {{INHERITANCE_DEPTH}},
  class_{{ID}}.cohesion_score = {{COHESION_SCORE}},
  class_{{ID}}.updated_at = datetime("{{UPDATED_AT}}");
{{/CLASSES}}

-- ============================================
-- NODOS: VARIABLES
-- ============================================

{{#VARIABLES}}
MERGE (var_{{ID}}:Variable {id: "{{ID}}"})
ON CREATE SET
  var_{{ID}}.created_at = datetime("{{CREATED_AT}}")
SET
  var_{{ID}}.name = "{{NAME}}",
  var_{{ID}}.var_type = "{{VAR_TYPE}}",
  var_{{ID}}.value = "{{VALUE}}",
  var_{{ID}}.scope = "{{SCOPE}}",
  var_{{ID}}.is_constant = {{IS_CONSTANT}},
  var_{{ID}}.is_mutable = {{IS_MUTABLE}},
  var_{{ID}}.line_declared = {{LINE_DECLARED}},
  var_{{ID}}.type_inference_confidence = {{TYPE_INFERENCE_CONFIDENCE}},
  var_{{ID}}.updated_at = datetime("{{UPDATED_AT}}");
{{/VARIABLES}}

-- ============================================
-- NODOS: ISSUES
-- ============================================

{{#ISSUES}}
MERGE (issue_{{ID}}:Issue {id: "{{ID}}"})
ON CREATE SET
  issue_{{ID}}.created_at = datetime("{{CREATED_AT}}")
SET
  issue_{{ID}}.issue_type = "{{ISSUE_TYPE}}",
  issue_{{ID}}.severity = "{{SEVERITY}}",
  issue_{{ID}}.message = "{{MESSAGE}}",
  issue_{{ID}}.file_path = "{{FILE_PATH}}",
  issue_{{ID}}.start_line = {{START_LINE}},
  issue_{{ID}}.end_line = {{END_LINE}},
  issue_{{ID}}.start_column = {{START_COLUMN}},
  issue_{{ID}}.end_column = {{END_COLUMN}},
  issue_{{ID}}.rule_id = "{{RULE_ID}}",
  issue_{{ID}}.rule_category = "{{RULE_CATEGORY}}",
  issue_{{ID}}.effort_minutes = {{EFFORT_MINUTES}},
  issue_{{ID}}.tags = {{TAGS}},
  issue_{{ID}}.confidence = {{CONFIDENCE}},
  issue_{{ID}}.is_false_positive = {{IS_FALSE_POSITIVE}},
  issue_{{ID}}.is_fixed = {{IS_FIXED}},
  issue_{{ID}}.suggestion = "{{SUGGESTION}}",
  issue_{{ID}}.example_fix = "{{EXAMPLE_FIX}}",
  issue_{{ID}}.fixed_at =
    CASE
      WHEN "{{FIXED_AT}}" = "" OR "{{FIXED_AT}}" IS NULL
      THEN NULL
      ELSE datetime("{{FIXED_AT}}")
    END,
  issue_{{ID}}.updated_at = datetime("{{UPDATED_AT}}");
{{/ISSUES}}

-- ============================================
-- METADATA DE EXPORTACIÓN
-- ============================================

MERGE (export:ExportMetadata {export_id: "{{EXPORT_ID}}"})
SET
  export.project_id = "{{PROJECT_ID}}",
  export.export_type = "cypher",
  export.timestamp = datetime("{{TIMESTAMP}}"),
  export.node_count = {{NODE_COUNT}},
  export.relationship_count = {{RELATIONSHIP_COUNT}},
  export.version = "1.0.1",
  export.system_version = "{{SYSTEM_VERSION}}";
