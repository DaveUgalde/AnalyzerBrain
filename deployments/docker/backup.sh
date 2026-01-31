#!/usr/bin/env bash
# Script de backup automatizado para Project Brain

set -Eeuo pipefail

echo "==========================================="
echo "BACKUP AUTOMATIZADO - PROJECT BRAIN"
echo "Fecha: $(date -Iseconds)"
echo "==========================================="

# ======================
# Configuración general
# ======================
BACKUP_DIR="/app/data/backups"
DATA_DIR="/app/data"
CONFIG_DIR="/app/config"
LOG_DIR="/app/logs"

TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
BACKUP_NAME="project_brain_backup_${TIMESTAMP}"
WORK_DIR="${BACKUP_DIR}/${BACKUP_NAME}"

mkdir -p "$WORK_DIR"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

fail() {
    log "❌ ERROR: $*"
    exit 1
}

trap 'fail "Fallo inesperado en la línea $LINENO"' ERR

# ======================
# Variables de entorno
# ======================
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

# ======================
# 1. Backup PostgreSQL
# ======================
log "Iniciando backup de PostgreSQL..."
PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
    -h "$POSTGRES_HOST" \
    -p "$POSTGRES_PORT" \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    -Fc \
    --no-owner \
    --no-acl \
    -f "$WORK_DIR/postgres.dump"

log "✅ Backup PostgreSQL completado"

# ======================
# 2. Backup Neo4j
# ======================
log "Iniciando backup de Neo4j..."
cypher-shell \
    -u "$NEO4J_USER" \
    -p "$NEO4J_PASSWORD" \
    -a "bolt://${NEO4J_HOST}:${NEO4J_PORT}" \
    "CALL apoc.export.cypher.all(null, {format:'cypher-shell'})" \
    > "$WORK_DIR/neo4j.cypher"

if [[ ! -s "$WORK_DIR/neo4j.cypher" ]]; then
    log "⚠️  Backup Neo4j generado pero vacío"
else
    log "✅ Backup Neo4j completado"
fi

# ======================
# 3. Backup Redis
# ======================
log "Iniciando backup de Redis..."
redis-cli \
    -h "$REDIS_HOST" \
    -p "$REDIS_PORT" \
    -a "$REDIS_PASSWORD" \
    --rdb "$WORK_DIR/redis.rdb"

log "✅ Backup Redis completado"

# ======================
# 4. Backup ChromaDB
# ======================
if [[ -d "${DATA_DIR}/embeddings" ]]; then
    log "Iniciando backup de ChromaDB..."
    tar -czf "$WORK_DIR/chromadb.tar.gz" -C "$DATA_DIR" embeddings
    log "✅ Backup ChromaDB completado"
else
    log "⚠️  Directorio embeddings no encontrado, omitido"
fi

# ======================
# 5. Backup configuración
# ======================
if [[ -d "$CONFIG_DIR" ]]; then
    log "Iniciando backup de configuración..."
    tar -czf "$WORK_DIR/config.tar.gz" -C /app config
    log "✅ Backup configuración completado"
fi

# ======================
# 6. Backup logs recientes
# ======================
if [[ -d "$LOG_DIR" ]]; then
    log "Iniciando backup de logs (últimos 7 días)..."
    mapfile -d '' LOG_FILES < <(find "$LOG_DIR" -name "*.log" -mtime -7 -print0)

    if (( ${#LOG_FILES[@]} > 0 )); then
        printf '%s\0' "${LOG_FILES[@]}" \
            | tar --null -czf "$WORK_DIR/logs.tar.gz" --files-from=-
        log "✅ Backup logs completado"
    else
        log "⚠️  No hay logs recientes para respaldar"
    fi
fi

# ======================
# 7. Metadatos
# ======================
log "Generando metadata..."
cat > "$WORK_DIR/metadata.json" <<EOF
{
  "backup_name": "${BACKUP_NAME}",
  "timestamp": "$(date -Iseconds)",
  "version": "1.0.0",
  "components": {
    "postgres": true,
    "neo4j": true,
    "redis": true,
    "chromadb": true,
    "config": true,
    "logs": true
  },
  "sizes": {
    "postgres": $(stat -c%s "$WORK_DIR/postgres.dump" 2>/dev/null || echo 0),
    "neo4j": $(stat -c%s "$WORK_DIR/neo4j.cypher" 2>/dev/null || echo 0),
    "redis": $(stat -c%s "$WORK_DIR/redis.rdb" 2>/dev/null || echo 0)
  }
}
EOF

log "✅ Metadata creada"

# ======================
# 8. Empaquetado final
# ======================
log "Comprimiendo backup final..."
cd "$BACKUP_DIR"
tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"

ARCHIVE="${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
SIZE_BYTES=$(stat -c%s "$ARCHIVE")

log "✅ Backup creado: ${ARCHIVE}"

# ======================
# 9. Limpieza
# ======================
rm -rf "$WORK_DIR"

# ======================
# 10. Rotación (30 días)
# ======================
log "Aplicando rotación de backups..."
find "$BACKUP_DIR" -name "project_brain_backup_*.tar.gz" -mtime +30 -delete

# ======================
# 11. Verificación
# ======================
log "Verificando integridad del backup..."
tar -tzf "$ARCHIVE" > /dev/null
log "✅ Integridad verificada"

# ======================
# 12. Registro
# ======================
echo "$(date -Iseconds) | ${BACKUP_NAME}.tar.gz | ${SIZE_BYTES} bytes" \
    >> "$BACKUP_DIR/backup_history.log"

echo "${BACKUP_NAME}.tar.gz" > "$BACKUP_DIR/latest_backup.txt"
echo "$SIZE_BYTES" > "$BACKUP_DIR/latest_backup_size.txt"

# ======================
# Resumen
# ======================
echo ""
echo "==========================================="
echo "BACKUP COMPLETADO EXITOSAMENTE"
echo "==========================================="
echo "Archivo: ${BACKUP_NAME}.tar.gz"
echo "Tamaño: $(numfmt --to=iec-i "$SIZE_BYTES")"
echo "Ubicación: $BACKUP_DIR"
echo "==========================================="

exit 0
