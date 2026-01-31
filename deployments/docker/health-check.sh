#!/usr/bin/env bash
# Script de health check para contenedores Docker - Project Brain

set -Eeuo pipefail

HEALTH_STATUS=0
ERROR_MESSAGES=()

# Valores seguros para el resumen
DISK_USAGE="N/A"
MEM_FREE_GB="N/A"

echo "==========================================="
echo "HEALTH CHECK - PROJECT BRAIN"
echo "Fecha: $(date -Iseconds)"
echo "==========================================="

log_ok()   { echo "✓ $1"; }
log_fail() { echo "✗ $1"; }

# =========================
# Función genérica de check
# =========================
check_service() {
    local service_name="$1"
    shift
    local cmd=("$@")

    echo -n "Verificando ${service_name}... "
    if ( "${cmd[@]}" ) > /dev/null 2>&1; then
        log_ok ""
    else
        log_fail ""
        ERROR_MESSAGES+=("${service_name} no está disponible")
        HEALTH_STATUS=1
    fi
}

# =========================
# Variables de entorno
# =========================
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-brain_user}"

NEO4J_HOST="${NEO4J_HOST:-neo4j}"
NEO4J_PORT="${NEO4J_PORT:-7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-brain_password}"

REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-brain_password}"

# =========================
# 1. PostgreSQL
# =========================
check_service "PostgreSQL" \
    pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER"

# =========================
# 2. Neo4j
# =========================
check_service "Neo4j" \
    cypher-shell \
        -u "$NEO4J_USER" \
        -p "$NEO4J_PASSWORD" \
        -a "bolt://${NEO4J_HOST}:${NEO4J_PORT}" \
        "RETURN 1"

# =========================
# 3. Redis
# =========================
echo -n "Verificando Redis... "
if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" ping 2>/dev/null | grep -q PONG; then
    log_ok ""
else
    log_fail ""
    ERROR_MESSAGES+=("Redis no responde correctamente")
    HEALTH_STATUS=1
fi

# =========================
# 4. Espacio en disco
# =========================
echo -n "Verificando espacio en disco... "
DISK_USAGE=$(df /app | awk 'NR==2 {gsub("%","",$5); print $5}')
if [[ "$DISK_USAGE" -lt 90 ]]; then
    log_ok "(${DISK_USAGE}%)"
else
    log_fail "(${DISK_USAGE}% - CRÍTICO)"
    ERROR_MESSAGES+=("Uso de disco alto: ${DISK_USAGE}%")
    HEALTH_STATUS=1
fi

# =========================
# 5. Memoria
# =========================
echo -n "Verificando memoria... "
MEM_FREE_MB=$(free -m | awk 'NR==2 {print ($7 ? $7 : $4)}')
MEM_FREE_GB=$(awk "BEGIN {printf \"%.2f\", ${MEM_FREE_MB}/1024}")

if [[ "$MEM_FREE_MB" -gt 512 ]]; then
    log_ok "(${MEM_FREE_GB}GB libre)"
else
    log_fail "(${MEM_FREE_GB}GB libre - BAJO)"
    ERROR_MESSAGES+=("Memoria libre baja: ${MEM_FREE_GB}GB")
    HEALTH_STATUS=1
fi

# =========================
# 6. API interna
# =========================
echo -n "Verificando API interna... "
if curl -fs http://localhost:8000/health > /dev/null; then
    log_ok ""
else
    log_fail ""
    ERROR_MESSAGES+=("API interna no responde")
    HEALTH_STATUS=1
fi

# =========================
# 7. Archivos esenciales
# =========================
ESSENTIAL_FILES=(
    "/app/config/system.yaml"
    "/app/data/embeddings"
)

for file in "${ESSENTIAL_FILES[@]}"; do
    echo -n "Verificando ${file}... "
    if [[ -e "$file" ]]; then
        log_ok ""
    else
        log_fail ""
        ERROR_MESSAGES+=("Archivo o directorio esencial faltante: ${file}")
        HEALTH_STATUS=1
    fi
done

# =========================
# 8. Procesos críticos
# =========================
CRITICAL_PROCESSES=("uvicorn" "python")

for process in "${CRITICAL_PROCESSES[@]}"; do
    echo -n "Verificando proceso ${process}... "
    if pgrep -f "$process" > /dev/null; then
        log_ok ""
    else
        log_fail ""
        ERROR_MESSAGES+=("Proceso crítico no ejecutándose: ${process}")
        HEALTH_STATUS=1
    fi
done

# =========================
# 9. Puertos críticos
# =========================
echo -n "Verificando puertos críticos... "
if command -v nc >/dev/null 2>&1 && nc -z localhost 8000 && nc -z localhost 8001; then
    log_ok ""
else
    log_fail ""
    ERROR_MESSAGES+=("Puertos esenciales no están escuchando")
    HEALTH_STATUS=1
fi

# =========================
# 10. Logs recientes
# =========================
echo -n "Verificando errores recientes en logs... "
if [[ -d /app/logs ]]; then
    if find /app/logs -name "*.log" -type f -mtime -1 -print0 \
        | xargs -0 grep -qiE "ERROR|CRITICAL|Exception"; then
        log_fail ""
        ERROR_MESSAGES+=("Errores recientes encontrados en logs")
        HEALTH_STATUS=1
    else
        log_ok ""
    fi
else
    log_ok "(logs no presentes)"
fi

# =========================
# Resumen
# =========================
echo ""
echo "==========================================="
echo "RESUMEN DE HEALTH CHECK"
echo "==========================================="

if [[ "$HEALTH_STATUS" -eq 0 ]]; then
    echo "✅ SISTEMA SALUDABLE"
else
    echo "❌ SISTEMA NO SALUDABLE"
    for error in "${ERROR_MESSAGES[@]}"; do
        echo "  • $error"
    done
fi

echo ""
echo "Métricas:"
echo "  • Uso de disco: ${DISK_USAGE}%"
echo "  • Memoria libre: ${MEM_FREE_GB}GB"
echo "  • Hora: $(date)"
echo "  • Uptime: $(uptime -p)"

# =========================
# Debug info (no rompe health)
# =========================
echo ""
echo "==========================================="
echo "DEBUG INFO"
echo "==========================================="
python3 --version 2>/dev/null || true
env | grep -E "POSTGRES|NEO4J|REDIS|BRAIN" | sort || true
ps aux | grep -E "uvicorn|python" | grep -v grep || true
tail -10 /app/logs/api/access.log 2>/dev/null || true

exit "$HEALTH_STATUS"
