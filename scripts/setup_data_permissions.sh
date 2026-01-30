#!/bin/bash
# scripts/setup_data_permissions.sh
# Script para configurar permisos de datos de forma segura

set -euo pipefail

echo "=== CONFIGURANDO PERMISOS DE DATOS ==="

DATA_DIR="${DATA_DIR:-./data}"

# Verificar que el directorio existe
if [ ! -d "$DATA_DIR" ]; then
    echo "📁 Creando directorio de datos: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

echo "🔐 Configurando permisos..."

# Directorios: 755 (rwxr-xr-x)
if find "$DATA_DIR" -type d | grep -q .; then
    find "$DATA_DIR" -type d -exec chmod 755 {} +
fi

# Archivos: 644 (rw-r--r--)
if find "$DATA_DIR" -type f | grep -q .; then
    find "$DATA_DIR" -type f -exec chmod 644 {} +
fi

# Archivos ejecutables específicos (si existen)
EXEC_SCRIPT="$DATA_DIR/scripts/init.sh"
if [ -f "$EXEC_SCRIPT" ]; then
    chmod 755 "$EXEC_SCRIPT"
fi

# Configurar ownership solo si se ejecuta como root
if [ "$(id -u)" -eq 0 ]; then
    DATA_USER="${DATA_USER:-${SUDO_USER:-}}"

    if [ -n "$DATA_USER" ]; then
        echo "👤 Configurando ownership para usuario: $DATA_USER"
        chown -R "$DATA_USER:$DATA_USER" "$DATA_DIR"
    else
        echo "⚠️  No se pudo determinar usuario para ownership (DATA_USER/SUDO_USER vacío)"
    fi
fi

# Configurar ACLs si el sistema lo soporta
if command -v setfacl >/dev/null 2>&1; then
    echo "🧩 Configurando ACLs (si aplican)..."

    # Acceso de solo lectura/ejecución para servicios web comunes
    setfacl -R -m u:www-data:rx "$DATA_DIR" 2>/dev/null || true
    setfacl -R -m u:nginx:rx "$DATA_DIR" 2>/dev/null || true
fi

echo "✅ Permisos configurados correctamente"

echo ""
echo "=== RESUMEN DE PERMISOS ==="
ls -la "$DATA_DIR"
