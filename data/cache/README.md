# Directorio de Caché Distribuida

Este directorio implementa la caché de nivel 3 (disco) según la jerarquía definida en la arquitectura.

## Jerarquía de Caché
8. ARCHIVO: data/cache/README.md

markdown
# Directorio de Caché Distribuida

Este directorio implementa la caché de nivel 3 (disco) según la jerarquía definida en la arquitectura.

## Jerarquía de Caché
Nivel 1: Memoria (LRU, 1000 items, TTL 5min)
Nivel 2: Redis (10000 items, TTL 1h)
Nivel 3: Disco (100000 items, TTL 24h) <- Este directorio

text

## Estructura de Archivos
cache/
├── l3_cache_config.json # Configuración de caché de disco
├── shard_0/ # Fragmento 0 para distribución
│ ├── embeddings/ # Caché de embeddings
│ │ ├── {hash1}.pkl
│ │ ├── {hash2}.pkl
│ │ └── ...
│ ├── ast/ # Caché de ASTs parseados
│ │ ├── {file_hash1}.json
│ │ └── ...
│ ├── analysis/ # Caché de análisis
│ │ ├── {project_id1}.json
│ │ └── ...
│ └── queries/ # Caché de consultas frecuentes
│ ├── {query_hash1}.json
│ └── ...
├── shard_1/ # Fragmento 1
├── shard_2/ # Fragmento 2
└── metadata/ # Metadatos y estadísticas
├── cache_stats.json
├── access_patterns.json
└── eviction_log.json

text

## Formato de Archivos de Caché

### Archivos .pkl (embeddings)
```python
{
  "key": "hash_o_clave",
  "value": embedding_vector,  # Lista/array de floats
  "metadata": {
    "created_at": "2024-01-01T12:00:00Z",
    "accessed_at": "2024-01-01T12:00:00Z",
    "access_count": 1,
    "size_bytes": 1536,
    "type": "embedding",
    "model": "all-MiniLM-L6-v2"
  }
}
Archivos .json (AST y análisis)

json
{
  "key": "file_hash_o_clave",
  "value": {
    "ast": {...},
    "entities": [...],
    "metadata": {...}
  },
  "metadata": {
    "created_at": "2024-01-01T12:00:00Z",
    "accessed_at": "2024-01-01T12:00:00Z",
    "access_count": 5,
    "size_bytes": 2048,
    "type": "ast",
    "language": "python"
  }
}
Políticas de Caché

Inserción

Los items entran por L1 (memoria)
Si L1 está lleno, items se mueven a L2
Si L2 está lleno, items se mueven a L3 (disco)
Items en L3 se comprimen automáticamente
Recuperación

Buscar en L1 (cache hit más rápido)
Si no está en L1, buscar en L2
Si no está en L2, buscar en L3
Si se encuentra en L3, promover a L1/L2
Evicción

L1: LRU (Least Recently Used)
L2: TTL + LFU (Least Frequently Used)
L3: TTL + tamaño + antigüedad
Configuración

Ver l3_cache_config.json para ajustes específicos.

Mantenimiento

Limpieza Automática

bash
# Ejecutar limpieza de caché expirada
python -m src.memory.cache_manager cleanup --level=l3

# Optimizar fragmentos
python -m src.memory.cache_manager optimize

# Generar reporte de estadísticas
python -m src.memory.cache_manager stats
Monitorización

Tasa de hit/miss por nivel
Tiempo promedio de acceso
Distribución de tamaños
Patrones de acceso
Rendimiento Esperado

Nivel	Tiempo de Acceso	Tamaño Máximo	Hit Rate Esperado
L1	< 1ms	1000 items	40%
L2	1-5ms	10000 items	30%
L3	10-50ms	100000 items	20%
Total	-	-	90%+