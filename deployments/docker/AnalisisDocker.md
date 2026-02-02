
Workflow Test + Docker:

Tests en CI/CD SEPARADO de la construcción (RECOMENDADO)

1. Desarrollo local → Tests locales
2. Push a repo → CI pipeline:
   a. Construir imagen base SIN tests
   b. Ejecutar tests en contenedor temporal
   c. Si tests pasan → Construir imagen de producción
3. Despliegue de imagen verificada



Estructura Docker recomendada:

text
ANALYZERBRAIN/
├── Dockerfile              # Para producción
├── Dockerfile.dev         # Para desarrollo
├── docker-compose.yml     # Servicios dependientes
├── docker-compose.test.yml # Para testing
└── .dockerignore
Dockerfile para desarrollo:

dockerfile
# Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (caché de Docker)
COPY requirements/base.txt /app/requirements/base.txt
COPY requirements/dev.txt /app/requirements/dev.txt

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/dev.txt

# Copiar código fuente
COPY . /app

# Crear directorios necesarios
RUN mkdir -p /app/data /app/logs /app/config

# Variables de entorno por defecto
ENV PYTHONPATH=/app
ENV ENVIRONMENT=development
ENV DATA_DIR=/app/data
ENV LOG_DIR=/app/logs

# Puerto para API (futuro)
EXPOSE 8000

# Comando por defecto (desarrollo)
CMD ["python", "-m", "src.main", "init"]
docker-compose.yml para servicios dependientes:

yaml
version: '3.8'

services:
  analyzerbrain:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ./:/app
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=development
    depends_on:
      - postgres
      - redis
      - neo4j
    command: ["python", "-m", "src.main", "init"]

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: analyzerbrain
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/password
    volumes:
      - neo4j_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"

volumes:
  postgres_data:
  redis_data:
  neo4j_data: