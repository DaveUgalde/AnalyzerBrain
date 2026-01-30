# docs/deployment/README.md
# Guía de Despliegue - Project Brain

## Despliegue en Diferentes Entornos
Esta guía cubre el despliegue de Project Brain en diversos entornos, desde desarrollo local hasta producción a escala.

## Índice de Documentación

### 1. Despliegue Local
- [Requisitos del Sistema](local/requirements.md)
- [Instalación Manual](local/manual_installation.md)
- [Docker Compose](local/docker_compose.md)
- [Verificación de Instalación](local/verification.md)

### 2. Despliegue con Docker
- [Imágenes Docker](docker/images.md)
- [Docker Compose para Producción](docker/compose_production.md)
- [Optimización de Imágenes](docker/image_optimization.md)
- [Seguridad en Contenedores](docker/container_security.md)

### 3. Despliegue con Kubernetes
- [Configuración de Kubernetes](kubernetes/configuration.md)
- [Helm Charts](kubernetes/helm_charts.md)
- [Auto-scaling](kubernetes/autoscaling.md)
- [High Availability](kubernetes/high_availability.md)

### 4. Configuración de Producción
- [Configuración del Sistema](production/system_configuration.md)
- [Bases de Datos](production/databases.md)
- [Almacenamiento y Backups](production/storage_backups.md)
- [Red y Seguridad](production/network_security.md)

### 5. Monitoreo y Observabilidad
- [Métricas del Sistema](monitoring/system_metrics.md)
- [Logging Centralizado](monitoring/centralized_logging.md)
- [Alertas y Notificaciones](monitoring/alerts_notifications.md)
- [Dashboards (Grafana)](monitoring/dashboards.md)

### 6. Mantenimiento y Operaciones
- [Actualizaciones](operations/updates.md)
- [Backup y Recuperación](operations/backup_recovery.md)
- [Escalabilidad](operations/scaling.md)
- [Solución de Problemas en Producción](operations/troubleshooting.md)

### 7. Seguridad
- [Hardening del Sistema](security/hardening.md)
- [Autenticación y Autorización](security/authentication.md)
- [Protección de Datos](security/data_protection.md)
- [Auditoría y Cumplimiento](security/audit_compliance.md)

### 8. Optimización de Rendimiento
- [Tuning de Bases de Datos](performance/database_tuning.md)
- [Optimización de Memoria](performance/memory_optimization.md)
- [Caché y CDN](performance/cache_cdn.md)
- [Load Balancing](performance/load_balancing.md)