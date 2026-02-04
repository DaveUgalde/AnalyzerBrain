"""
Extractor de entidades: Procesa las entidades parseadas y las enriquece.
"""

from typing import List
from .models import CodeEntity


class EntityExtractor:
    """Extrae y enriquece entidades de código."""

    def __init__(self):
        pass

    def extract(self, entities: List[CodeEntity]) -> List[CodeEntity]:
        """Procesa una lista de entidades y las enriquece con información adicional."""
        # Por ahora, simplemente retornamos las entidades sin cambios.
        # En el futuro, aquí se podrían añadir más análisis.
        return entities

    def _enrich_entity(self, entity: CodeEntity) -> CodeEntity:
        """Enriquece una entidad con información adicional."""
        # Por ejemplo, calcular complejidad ciclomática, etc.
        return entity
