 ARCHIVO: data/backups/README.md

markdown
# Directorio de Backups Automáticos

Este directorio almacena los backups automáticos del sistema Project Brain según la configuración definida en `config/system.yaml`.

## Estructura de Backups
backups/
├── full/ # Backups completos del sistema
│ ├── full_20240101_030000.tar.gz
│ ├── full_20240102_030000.tar.gz
│ └── ...
├── incremental/ # Backups incrementales
│ ├── incremental_20240101_120000.tar.gz
│ ├── incremental_20240101_180000.tar.gz
│ └── ...
├── components/ # Backups por componente
│ ├── database/
│ │ ├── postgres_20240101_030000.dump
│ │ └── ...
│ ├── embeddings/
│ │ ├── chromadb_20240101_030000.tar.gz
│ │ └── ...
│ ├── graph/
│ │ ├── neo4j_20240101_030000.dump
│ │ └── ...
│ └── state/
│ ├── system_state_20240101_030000.json.gz
│ └── ...
├── snapshots/ # Snapshots de estado
│ ├── snapshot_pre_update_20240101.tar.gz
│ ├── snapshot_post_update_20240101.tar.gz
│ └── ...
└── metadata/ # Metadatos de backups
├── backup_manifest.json
├── retention_policy.json
├── verification_log.json
└── ...

text

## Tipos de Backup

### 1. Backup Completo
- Incluye todo el sistema
- Programado: Diario a las 3 AM
- Retención: 30 días
- Tamaño: ~1-10 GB dependiendo del sistema

### 2. Backup Incremental
- Solo cambios desde último backup completo
- Programado: Cada 6 horas
- Retención: 7 días
- Tamaño: ~100-500 MB

### 3. Backup por Componente
- Componentes individuales (DB, embeddings, etc.)
- Programado: Según configuración
- Retención: Variable
- Tamaño: Depende del componente

### 4. Snapshots
- Estado en punto específico del tiempo
- Manual o antes de cambios importantes
- Retención: Según política
- Tamaño: ~1-5 GB

## Política de Retención
Backup Completo:

Últimos 7 días: Mantener todos
Últimos 30 días: Mantener diarios
Últimos 365 días: Mantener semanales
Más de 1 año: Mantener mensuales
Backup Incremental:

Últimas 24 horas: Mantener todos
Últimos 7 días: Mantener cada 6 horas
Más de 7 días: Eliminar
Snapshots:

Mantener por 90 días
Snapshots importantes: Mantener indefinidamente
text

## Verificación de Backups

### Verificación Automática
```bash
# Verificar integridad de backup
python scripts/verify_backup.py --backup=full_20240101_030000.tar.gz

# Verificar todos los backups
python scripts/verify_backup.py --all

# Generar reporte de verificación
python scripts/verify_backup.py --report
Métricas de Verificación

Checksum MD5/SHA256
Integridad de archivos
Tamaño esperado vs real
Tiempo de restauración estimado
Consistencia de datos
Restauración

Restauración Completa

bash
# Listar backups disponibles
python scripts/backup_restore.py list

# Restaurar backup completo
python scripts/backup_restore.py restore \
  --backup=full_20240101_030000.tar.gz \
  --target=/

# Restaurar a punto específico en el tiempo
python scripts/backup_restore.py restore-point \
  --datetime="2024-01-01 12:00:00"
Restauración Parcial

bash
# Restaurar solo base de datos
python scripts/backup_restore.py restore-component \
  --component=database \
  --backup=postgres_20240101_030000.dump

# Restaurar solo embeddings
python scripts/backup_restore.py restore-component \
  --component=embeddings \
  --backup=chromadb_20240101_030000.tar.gz

# Restaurar solo estado del sistema
python scripts/backup_restore.py restore-component \
  --component=state \
  --backup=system_state_20240101_030000.json.gz
Monitorización

Métricas Clave

Tamaño total de backups
Último backup exitoso
Tiempo de backup/restauración
Tasa de compresión
Espacio disponible
Alertas

Backup fallido
Espacio insuficiente
Backup no realizado en 24h
Verificación fallida
Tiempo de restauración excesivo
Configuración

La configuración de backup se encuentra en:

config/system.yaml → sección backup
data/backups/backup_manifest.json → metadatos
data/backups/retention_policy.json → política de retención
Scripts Relacionados

scripts/backup_restore.py: Backup y restauración
scripts/verify_backup.py: Verificación de backups
scripts/manage_backups.py: Gestión de políticas
scripts/monitor_backups.py: Monitorización
text
