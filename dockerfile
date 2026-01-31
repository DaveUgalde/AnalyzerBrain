FROM python:3.11-slim

# Evita prompts interactivos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Dependencias del sistema (psycopg2, neo4j, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiamos requirements primero (mejor cache)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copiamos el resto del proyecto
COPY . .

# Comando por defecto
CMD ["python", "scripts/init_project.py"]
