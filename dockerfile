FROM python:3.10.13-slim

# Evita prompts interactivos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

ENV PYTHONPATH=/app/src

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    gcc \
    g++ \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip a la versión más reciente
RUN python -m pip install --upgrade pip

# Copiar requirements
COPY requirements /app/requirements

# ESTRATEGIA SIMPLIFICADA: Instalar todo en orden normal
# 1. Instalar las dependencias base primero usando comillas para los operadores < y >
RUN pip install --no-cache-dir \
    "pydantic>=2.5.0,<2.11.0" \
    "pydantic-settings>=2.0.0,<2.5.0" \
    "typing-extensions>=4.8.0,<5.0.0" \
    "python-dotenv>=1.0.0,<2.0.0" \
    "anyio>=3.7.0,<4.0.0" \
    "aiofiles>=23.0.0,<24.0.0" \
    "aiohttp>=3.9.0,<4.0.0" \
    "httpx>=0.26.0,<0.28.0" \
    "numpy>=1.24.0,<1.27.0" \
    "pandas>=2.1.0,<2.3.0" \
    "scipy>=1.11.0,<1.13.0" \
    "networkx>=3.0.0,<3.3.0" \
    "platformdirs>=4.0.0,<5.0.0" \
    "filelock>=3.12.0,<4.0.0" \
    "psutil>=5.9.0,<6.0.0" \
    "structlog>=23.0.0,<24.0.0" \
    "prometheus-client>=0.19.0,<0.20.0" \
    "cryptography>=42.0.0,<43.0.0" \
    "tqdm>=4.65.0,<5.0.0"

# 2. Instalar API y databases
RUN pip install --no-cache-dir -r /app/requirements/api.txt
RUN pip install --no-cache-dir -r /app/requirements/databases.txt

# 3. Instalar agents.txt (con la versión actualizada de OpenAI)
RUN pip install --no-cache-dir -r /app/requirements/agents.txt

# Instalar spacy y modelo - CORREGIDO
RUN pip install --no-cache-dir spacy==3.7.2 && \
    pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Copiamos el resto del proyecto
COPY . .

# Comando por defecto
CMD ["python", "scripts/init_project.py"]