#!/usr/bin/env bash
# ==========================================================
# Script: setup.sh
#
# Descripción:
#   Script de instalación rápida para ANALYZERBRAIN.
#   Prepara un entorno de desarrollo local con Python,
#   dependencias base, estructura de directorios y
#   archivos de configuración iniciales.
#
# Uso:
#   ./setup.sh
#
# Requisitos:
#   - bash >= 4.x
#   - python >= 3.9
#   - pip
#   - virtualenv (incluido en Python)
#
# Qué hace:
#   - Verifica el entorno de ejecución
#   - Crea y activa un entorno virtual
#   - Instala dependencias base
#   - Inicializa estructura de directorios
#   - Genera archivos de configuración por defecto
#
# Qué NO hace:
#   - No inicia el sistema
#   - No ejecuta migraciones
#   - No sobrescribe configuraciones existentes
#
# ==========================================================


# Detener ejecución ante cualquier error
set -e

echo "🚀 Configurando ANALYZERBRAIN..."


# ----------------------------------------------------------
# Verificaciones iniciales
# ----------------------------------------------------------

# Verificar que estamos en la raíz del proyecto
if [ ! -f "pyproject.toml" ] && [ ! -f "setup.py" ]; then
    echo "❌ No se encuentra pyproject.toml ni setup.py."
    echo "   Ejecuta este script desde la raíz del proyecto."
    exit 1
fi

# Verificar disponibilidad de Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no encontrado. Instala Python 3.9 o superior."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✅ Python $PYTHON_VERSION detectado"


# ----------------------------------------------------------
# Entorno virtual
# ----------------------------------------------------------

echo "🔧 Creando entorno virtual..."
python3 -m venv venv

# Activar entorno virtual
# shellcheck disable=SC1091
source venv/bin/activate


# ----------------------------------------------------------
# Instalación de dependencias
# ----------------------------------------------------------

echo "📦 Actualizando pip y herramientas base..."
pip install --upgrade pip setuptools wheel

echo "📥 Instalando dependencias base..."
pip install -r requirements/base.txt

echo "🔨 Instalando ANALYZERBRAIN en modo desarrollo..."
pip install -e .


# ----------------------------------------------------------
# Estructura de directorios
# ----------------------------------------------------------

echo "📁 Creando estructura de directorios..."
mkdir -p data/{backups,cache,embeddings,graph_exports,projects,state}
mkdir -p logs
mkdir -p config


# ----------------------------------------------------------
# Archivos de configuración
# ----------------------------------------------------------

# Crear archivo .env si no existe
if [ ! -f .env ]; then
    echo "📄 Creando archivo .env de ejemplo..."
    cp .env.example .env
    echo "⚠️  Edita el archivo .env con tus configuraciones"
fi

# Crear configuración YAML inicial si no existe
if [ ! -f config/system_config.yaml ]; then
    echo "⚙️  Creando configuración YAML inicial..."
    cat > config/system_config.yaml << 'EOF'
# Configuración inicial del sistema ANALYZERBRAIN
system:
  name: "ANALYZERBRAIN"
  version: "0.1.0"
  max_workers: 4
  timeout_seconds: 300
EOF
fi


# ----------------------------------------------------------
# Mensaje final
# ----------------------------------------------------------

echo ""
echo "🎉 ¡ANALYZERBRAIN configurado exitosamente!"
echo ""
echo "📋 Próximos pasos:"
echo "1. Edita el archivo .env con tus configuraciones"
echo "2. Inicia el sistema: python -m src.main init"
echo "3. Analiza un proyecto: python -m src.main analyze /ruta/proyecto"
echo "4. Usa la shell interactiva: python -m src.main shell"
echo ""

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "🔧 Para activar el entorno virtual manualmente:"
    echo "   source venv/bin/activate"
else
    echo "✅ Entorno virtual activo: $VIRTUAL_ENV"
fi

echo ""
echo "💡 Para más información, consulta README.md"
