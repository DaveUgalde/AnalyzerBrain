# Directorio de Embeddings Vectoriales

Este directorio almacena la base de datos vectorial ChromaDB para embeddings de código y texto.

## Estructura de ChromaDB

ChromaDB crea automáticamente la siguiente estructura:
ARCHIVO: data/embeddings/README.md

markdown
# Directorio de Embeddings Vectoriales

Este directorio almacena la base de datos vectorial ChromaDB para embeddings de código y texto.

## Estructura de ChromaDB

ChromaDB crea automáticamente la siguiente estructura:
embeddings/
├── chroma.sqlite3 # Base de datos SQLite principal
├── chroma_settings/ # Configuración del sistema
├── index/ # Índices para búsqueda rápida
└── collections/ # Colecciones de embeddings
├── code_embeddings/ # Embeddings de código
│ ├── embeddings.h5 # Vectores
│ ├── metadata.pkl # Metadatos
│ └── ...
├── text_embeddings/ # Embeddings de texto
├── document_embeddings/# Embeddings de documentos
└── ...

text

## Colecciones Predeterminadas

1. **code_embeddings**: Embeddings de funciones, clases, métodos
   - Dimensión: 384 (all-MiniLM-L6-v2)
   - Métrica: cosine similarity
   - Normalizado: true

2. **text_embeddings**: Embeddings de documentación, comentarios
   - Dimensión: 768 (all-mpnet-base-v2)
   - Métrica: dot product
   - Normalizado: true

3. **document_embeddings**: Embeddings de documentos completos
   - Dimensión: 384
   - Métrica: cosine similarity
   - Normalizado: true

## Configuración

La configuración se encuentra en `chromadb_config.yaml` y se carga al inicializar el sistema.

## Operaciones Comunes

### Buscar embeddings similares
```python
from src.embeddings.vector_store import VectorStore

store = VectorStore(collection_name="code_embeddings")
results = store.search_similar(
    query_embedding=embedding,
    top_k=10,
    threshold=0.7
)
Añadir nuevos embeddings

python
store.add_vectors(
    ids=["func_123", "func_456"],
    embeddings=[emb1, emb2],
    metadatas=[
        {"type": "function", "name": "process_data"},
        {"type": "class", "name": "DataProcessor"}
    ]
)
Mantenimiento

Optimización: Ejecutar chroma optimize periódicamente
Backup: Incluido en backups automáticos
Migrations: ChromaDB maneja migraciones automáticamente