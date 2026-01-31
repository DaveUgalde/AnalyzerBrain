-- scripts/init-db.sql
-- ============================================
-- SCRIPT DE INICIALIZACIÓN DE BASE DE DATOS
-- Project Brain - PostgreSQL Schema
-- ============================================

-- ============================================
-- CONFIGURACIÓN INICIAL
-- ============================================

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

-- ============================================
-- EXTENSIONES
-- ============================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
-- CREATE EXTENSION IF NOT EXISTS "vector"; -- opcional (pgvector)

-- ============================================
-- TABLAS PRINCIPALES
-- ============================================

CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    path TEXT NOT NULL UNIQUE,
    description TEXT,
    language VARCHAR(50),

    file_count INTEGER NOT NULL DEFAULT 0 CHECK (file_count >= 0),
    total_lines BIGINT NOT NULL DEFAULT 0 CHECK (total_lines >= 0),
    function_count INTEGER NOT NULL DEFAULT 0 CHECK (function_count >= 0),
    class_count INTEGER NOT NULL DEFAULT 0 CHECK (class_count >= 0),
    issue_count INTEGER NOT NULL DEFAULT 0 CHECK (issue_count >= 0),

    analysis_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    last_analyzed TIMESTAMPTZ,
    analysis_duration_seconds INTEGER CHECK (analysis_duration_seconds >= 0),

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    tags TEXT[] NOT NULL DEFAULT '{}',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    path TEXT NOT NULL,
    name VARCHAR(255) NOT NULL,
    extension VARCHAR(20),
    language VARCHAR(50),

    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    line_count INTEGER NOT NULL CHECK (line_count >= 0),
    content_hash CHAR(64) NOT NULL,
    encoding VARCHAR(50) NOT NULL DEFAULT 'utf-8',

    complexity_level VARCHAR(20),
    maintainability_index NUMERIC(5,2) CHECK (maintainability_index BETWEEN 0 AND 100),
    test_coverage NUMERIC(5,2) CHECK (test_coverage BETWEEN 0 AND 100),
    duplication_rate NUMERIC(5,2) CHECK (duplication_rate BETWEEN 0 AND 100),

    parsed_ast JSONB,
    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    issues JSONB NOT NULL DEFAULT '[]'::jsonb,
    dependencies JSONB NOT NULL DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_parsed TIMESTAMPTZ,

    UNIQUE (project_id, path)
);

-- ============================================
-- FUNCIONES UTILITARIAS
-- ============================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- NOTA:
-- Esta función asume que las tablas functions, classes e issues existen.
-- check_function_bodies = false evita errores al crearla antes.

CREATE OR REPLACE FUNCTION update_project_stats(project_uuid UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE projects p
    SET
        file_count = (
            SELECT COUNT(*) FROM files WHERE project_id = p.id
        ),
        total_lines = COALESCE((
            SELECT SUM(line_count) FROM files WHERE project_id = p.id
        ), 0),
        function_count = (
            SELECT COUNT(*) FROM functions WHERE project_id = p.id
        ),
        class_count = (
            SELECT COUNT(*) FROM classes WHERE project_id = p.id
        ),
        issue_count = (
            SELECT COUNT(*) FROM issues
            WHERE project_id = p.id AND is_fixed = FALSE
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE p.id = project_uuid;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- LIMPIEZA DE DATOS ANTIGUOS
-- ============================================

CREATE OR REPLACE FUNCTION cleanup_old_data(retention_days INTEGER DEFAULT 90)
RETURNS TABLE(deleted_count INTEGER, table_name TEXT) AS $$
DECLARE
    deleted_rows INTEGER;
BEGIN
    DELETE FROM operation_logs
    WHERE started_at < CURRENT_TIMESTAMP - make_interval(days => retention_days);
    GET DIAGNOSTICS deleted_rows = ROW_COUNT;
    RETURN QUERY SELECT deleted_rows, 'operation_logs';

    DELETE FROM user_interactions
    WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => retention_days)
      AND feedback_score IS NULL;
    GET DIAGNOSTICS deleted_rows = ROW_COUNT;
    RETURN QUERY SELECT deleted_rows, 'user_interactions';

    DELETE FROM system_metrics
    WHERE timestamp < CURRENT_TIMESTAMP - make_interval(days => retention_days)
      AND metric_name NOT LIKE 'daily_%';
    GET DIAGNOSTICS deleted_rows = ROW_COUNT;
    RETURN QUERY SELECT deleted_rows, 'system_metrics';
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- SEARCH DE ENTIDADES
-- ============================================

CREATE OR REPLACE FUNCTION search_entities(
    search_query TEXT,
    project_uuid UUID DEFAULT NULL,
    limit_results INTEGER DEFAULT 100
)
RETURNS TABLE(
    entity_id UUID,
    entity_type TEXT,
    entity_name TEXT,
    project_id UUID,
    file_path TEXT,
    line_number INTEGER,
    match_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT *
    FROM (
        SELECT
            fn.id AS entity_id,
            'function'::TEXT AS entity_type,
            fn.name AS entity_name,
            fn.project_id,
            f.path AS file_path,
            fn.start_line AS line_number,
            similarity(fn.name, search_query) AS match_score
        FROM functions fn
        JOIN files f ON fn.file_id = f.id

        UNION ALL

        SELECT
            c.id,
            'class',
            c.name,
            c.project_id,
            f.path,
            1,
            similarity(c.name, search_query)
        FROM classes c
        JOIN files f ON c.file_id = f.id

        UNION ALL

        SELECT
            f.id,
            'file',
            f.name,
            f.project_id,
            f.path,
            1,
            similarity(f.name, search_query)
        FROM files f
    ) s
    WHERE (project_uuid IS NULL OR s.project_id = project_uuid)
    ORDER BY match_score DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- TRIGGERS
-- ============================================

DROP TRIGGER IF EXISTS update_projects_updated_at ON projects;
CREATE TRIGGER update_projects_updated_at
BEFORE UPDATE ON projects
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Repetir este patrón para:
-- files, functions, classes, issues, embeddings

-- ============================================
-- FIN
-- ============================================
