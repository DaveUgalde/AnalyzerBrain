"""
ANALYZERBRAIN - Sistema Inteligente de Análisis de Código.

Este paquete proporciona análisis de código inteligente combinando
técnicas de IA, procesamiento de lenguaje natural y grafos de conocimiento.

Módulos principales:
    - core: Núcleo del sistema, configuración y orquestación
    - api: Interfaces de usuario (REST, CLI, Web)
    - agents: Agentes especializados para análisis
    - indexer: Indexación y parsing de código fuente
    - graph: Grafo de conocimiento
    - embeddings: Representación vectorial y búsqueda semántica
    - memory: Sistema jerárquico de memoria
    - learning: Aprendizaje automático y adaptación
    - utils: Utilidades compartidas

Autor: ANALYZERBRAIN Team
Fecha: 2024
Versión: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "ANALYZERBRAIN Team"
__email__ = "team@analyzerbrain.dev"

# Configuración inicial del logging
from .utils.logging_config import setup_default_logging

# Configura logging solo si no está ya configurado
import sys

if "loguru" not in sys.modules:
    setup_default_logging()

# Archivo: src/init.py
# Propósito: Inicialización del paquete y configuración del logging por defecto.
# Número de funciones: 0 (solo código de inicialización).
# Lista de funciones por nombre: No aplica.
# Número de líneas de código: 22 líneas.
# Dependencias: logging_config (del mismo paquete).
# Instalaciones necesarias: loguru>=0.7.0,<0.8.0.
