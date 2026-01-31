from pathlib import Path
from typing import Dict, Any, Optional

from .chroma import init_chroma
from .operations import store_embedding, get_embedding, batch_store
from .search import search_similar
from .cache import cleanup_cache
from .metadata import get_stats


class VectorStorage:
    """Sistema de almacenamiento vectorial."""

    def __init__(self, storage_path: Path, config: Optional[Dict[str, Any]] = None):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.config = config or {
            "persist_directory": str(storage_path / "chroma"),
            "anonymized_telemetry": False,
        }

        self.client, self.collection = init_chroma(self)

    # Operaciones básicas
    store_embedding = store_embedding
    get_embedding = get_embedding

    # Búsqueda
    search_similar = search_similar

    # Batch
    batch_store = batch_store

    # Stats & mantenimiento
    get_stats = get_stats
    cleanup_cache = cleanup_cache
