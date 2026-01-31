#!/usr/bin/env bash
# Script de inicialización de bases de datos para Docker - Project Brain

set -Eeuo pipefail

echo "==========================================="
echo "INICIALIZACIÓN DE BASES DE DATOS - PROJECT BRAIN"
echo "Fecha: $(date -Iseconds)"
echo "==========================================="

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

fail() {
    log "❌ ERROR: $*"
    exit 1
}

trap 'fail "Fallo inesperado en la línea $LINENO"' ERR

# =========================
# Variables de entorno
# =========================
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-project_brain}"
POSTGRES_USER="${POSTGRES_USER:-brain_user}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-brain_password}"

NEO4J_HOST="${NEO4J_HOST:-neo4j}"
NEO4J_PORT="${NEO4J_PORT:-7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-brain_password}"

REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-brain_password}"

DATA_DIR="/app/data"
LOG_DIR="/app/logs"
CONFIG_DIR="/app/config"

mkdir -p "$DATA_DIR/embeddings"

# =========================
# Esperar servicios (con límite)
# =========================
MAX_RETRIES=60

log "Esperando PostgreSQL..."
for ((i=1; i<=MAX_RETRIES; i++)); do
    if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" > /dev/null 2>&1; then
        log "✅ PostgreSQL disponible"
        break
    fi
    sleep 2
    [[ "$i" -eq "$MAX_RETRIES" ]] && fail "PostgreSQL no respondió"
done

log "Esperando Neo4j..."
for ((i=1; i<=MAX_RETRIES; i++)); do
    if command -v nc >/dev/null 2>&1; then
        nc -z "$NEO4J_HOST" "$NEO4J_PORT" && break
    else
        (echo > /dev/tcp/"$NEO4J_HOST"/"$NEO4J_PORT") >/dev/null 2>&1 && break
    fi
    sleep 2
    [[ "$i" -eq "$MAX_RETRIES" ]] && fail "Neo4j no respondió"
done
log "✅ Neo4j disponible"

log "Esperando Redis..."
for ((i=1; i<=MAX_RETRIES; i++)); do
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" ping 2>/dev/null | grep -q PONG; then
        log "✅ Redis disponible"
        break
    fi
    sleep 2
    [[ "$i" -eq "$MAX_RETRIES" ]] && fail "Redis no respondió"
done

# =========================
# PostgreSQL
# =========================
log "Configurando PostgreSQL..."
PGPASSWORD="$POSTGRES_PASSWORD" psql \
    -h "$POSTGRES_HOST" \
    -p "$POSTGRES_PORT" \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" <<'EOSQL'
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
EOSQL
log "✅ PostgreSQL configurado"

# =========================
# Neo4j
# =========================
log "Configurando Neo4j..."
cypher-shell \
    -u "$NEO4J_USER" \
    -p "$NEO4J_PASSWORD" \
    -a "bolt://${NEO4J_HOST}:${NEO4J_PORT}" <<'CYPHER' || true
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (fn:Function) REQUIRE fn.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (i:Import) REQUIRE i.id IS UNIQUE;

CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.name);
CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.path);
CREATE INDEX IF NOT EXISTS FOR (fn:Function) ON (fn.name);
CREATE INDEX IF NOT EXISTS FOR (c:Class) ON (c.name);
CYPHER
log "✅ Neo4j configurado"

# =========================
# Redis
# =========================
log "Configurando Redis..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" <<EOF || true
CONFIG SET maxmemory 1gb
CONFIG SET maxmemory-policy allkeys-lru
SET system:initialized true
SET system:version "1.0.0"
SET system:start_time "$(date -Iseconds)"
HSET cache:stats hits 0 misses 0 size 0
EOF
log "✅ Redis configurado"

# =========================
# ChromaDB
# =========================
log "Inicializando ChromaDB..."
python3 <<'PYTHON'
import chromadb
from chromadb.config import Settings

settings = Settings(
    persist_directory="/app/data/embeddings",
    anonymized_telemetry=False
)

client = chromadb.Client(settings)

collection = client.get_or_create_collection(
    name="project_knowledge",
    metadata={
        "description": "Embeddings de conocimiento de proyectos",
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "similarity_metric": "cosine"
    }
)

print(f"Colección lista: {collection.name}")
print(f"Embeddings actuales: {collection.count()}")
PYTHON
log "✅ ChromaDB inicializado"

# =========================
# Directorios
# =========================
log "Creando estructura de directorios..."
mkdir -p \
    "$DATA_DIR"/{projects,embeddings,graph_exports,cache,state,backups} \
    "$LOG_DIR"/{api,analysis,agents,system} \
    "$CONFIG_DIR"

# =========================
# Permisos
# =========================
if id appuser &>/dev/null; then
    chown -R appuser:appuser "$DATA_DIR" "$LOG_DIR"
fi
chmod -R 755 "$DATA_DIR" "$LOG_DIR"

# =========================
# Final
# =========================
echo "==========================================="
echo "INICIALIZACIÓN COMPLETADA EXITOSAMENTE"
echo "==========================================="
echo "PostgreSQL : ✓"
echo "Neo4j      : ✓"
echo "Redis      : ✓"
echo "ChromaDB   : ✓"
echo "Directorios: ✓"
echo "==========================================="
