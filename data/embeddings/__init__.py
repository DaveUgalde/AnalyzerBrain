# data/embeddings/__init__.py
"""
Sistema de almacenamiento vectorial basado en ChromaDB.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from ...core.exceptions import EmbeddingException, ConfigurationError


class VectorStorage:
    """Sistema de almacenamiento vectorial."""

    def __init__(self, storage_path: Path, config: Optional[Dict[str, Any]] = None):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.config = config or {
            "persist_directory": str(storage_path / "chroma"),
            "anonymized_telemetry": False
        }

        self._init_directories()
        self._init_chroma()

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def _init_directories(self) -> None:
        directories = [
            self.storage_path / "chroma",
            self.storage_path / "cache",
            self.storage_path / "indices" / "hnsw",
            self.storage_path / "indices" / "ivf",
            self.storage_path / "models",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _init_chroma(self) -> None:
        if not CHROMA_AVAILABLE:
            raise ConfigurationError(
                "ChromaDB not installed. Install with 'pip install chromadb'"
            )

        try:
            self.client = chromadb.PersistentClient(
                path=self.config["persist_directory"],
                settings=Settings(
                    anonymized_telemetry=self.config.get("anonymized_telemetry", False)
                ),
            )

            self.collection = self.client.get_or_create_collection(
                name="project_embeddings",
                metadata={"description": "Main embeddings collection for Project Brain"},
            )

        except Exception as e:
            raise EmbeddingException(f"Failed to initialize ChromaDB: {e}")

    # ------------------------------------------------------------------
    # Operaciones básicas
    # ------------------------------------------------------------------

    def store_embedding(
        self,
        entity_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        model_name: str = "default",
    ) -> bool:
        try:
            self.collection.add(
                ids=[entity_id],
                embeddings=[embedding],
                metadatas=[{
                    **metadata,
                    "model": model_name,
                    "stored_at": datetime.now().isoformat(),
                    "dimensions": len(embedding),
                }],
            )

            self._cache_embedding(entity_id, embedding, metadata, model_name)
            self._update_metadata("stored", entity_id, model_name)
            return True

        except Exception as e:
            raise EmbeddingException(f"Failed to store embedding: {e}")

    def get_embedding(self, entity_id: str) -> Optional[Dict[str, Any]]:
        cached = self._get_cached_embedding(entity_id)
        if cached:
            return cached

        try:
            result = self.collection.get(
                ids=[entity_id],
                include=["embeddings", "metadatas"],
            )

            if result and result.get("ids"):
                return {
                    "embedding": result["embeddings"][0],
                    "metadata": result["metadatas"][0],
                    "source": "chromadb",
                }

        except Exception:
            return None

        return None

    # ------------------------------------------------------------------
    # Búsqueda
    # ------------------------------------------------------------------

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
                include=["embeddings", "metadatas", "distances"],
            )

            similar: List[Dict[str, Any]] = []

            if results.get("ids"):
                for i, entity_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance

                    if similarity >= threshold:
                        similar.append({
                            "entity_id": entity_id,
                            "embedding": results["embeddings"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "similarity": similarity,
                            "distance": distance,
                        })

            return similar

        except Exception as e:
            raise EmbeddingException(f"Search failed: {e}")

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def batch_store(
        self,
        embeddings: List[Tuple[str, List[float], Dict[str, Any]]],
        model_name: str = "default",
    ) -> Dict[str, Any]:
        stats = {
            "total": len(embeddings),
            "stored": 0,
            "failed": 0,
            "errors": [],
        }

        ids, vectors, metadatas = [], [], []

        for entity_id, embedding, metadata in embeddings:
            try:
                ids.append(entity_id)
                vectors.append(embedding)
                metadatas.append({
                    **metadata,
                    "model": model_name,
                    "stored_at": datetime.now().isoformat(),
                    "dimensions": len(embedding),
                    "batch_operation": True,
                })

                self._cache_embedding(entity_id, embedding, metadata, model_name)

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"{entity_id}: {e}")

        if ids:
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=vectors,
                    metadatas=metadatas,
                )
                stats["stored"] = len(ids)
            except Exception as e:
                raise EmbeddingException(f"Batch store failed: {e}")

        return stats

    # ------------------------------------------------------------------
    # Caché
    # ------------------------------------------------------------------

    def _cache_embedding(
        self,
        entity_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        model_name: str,
    ) -> None:
        cache_key = hashlib.sha256(
            f"{model_name}:{entity_id}".encode()
        ).hexdigest()[:16]

        model_dir = self.storage_path / "cache" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        cache_file = model_dir / f"{cache_key}.json"

        with open(cache_file, "w") as f:
            json.dump({
                "entity_id": entity_id,
                "embedding": embedding,
                "metadata": metadata,
                "model": model_name,
                "cached_at": datetime.now().isoformat(),
                "cache_key": cache_key,
                "source": "cache",
            }, f, indent=2)

    def _get_cached_embedding(self, entity_id: str) -> Optional[Dict[str, Any]]:
        cache_dir = self.storage_path / "cache"

        if not cache_dir.exists():
            return None

        for model_dir in cache_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for cache_file in model_dir.glob("*.json"):
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                        if data.get("entity_id") == entity_id:
                            return data
                except Exception:
                    continue

        return None

    # ------------------------------------------------------------------
    # Metadata & Stats
    # ------------------------------------------------------------------

    def _update_metadata(self, operation: str, entity_id: str, model_name: str) -> None:
        metadata_path = self.storage_path / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {
                "created_at": datetime.now().isoformat(),
                "total_embeddings": 0,
                "operations": [],
                "models": {},
            }

        metadata["total_embeddings"] = self.collection.count()
        metadata["last_updated"] = datetime.now().isoformat()

        metadata["operations"].append({
            "type": operation,
            "entity_id": entity_id,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
        })

        metadata["models"][model_name] = metadata["models"].get(model_name, 0) + 1

        metadata["operations"] = metadata["operations"][-1000:]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        try:
            return {
                "total_embeddings": self.collection.count(),
                "chroma_size_bytes": self._get_directory_size(self.storage_path / "chroma"),
                "cache_size_bytes": self._get_directory_size(self.storage_path / "cache"),
            }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _get_directory_size(self, path: Path) -> int:
        total = 0
        if not path.exists():
            return 0

        for dirpath, _, filenames in os.walk(path):
            for name in filenames:
                fp = os.path.join(dirpath, name)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total

    def cleanup_cache(self, max_age_days: int = 30) -> int:
        cache_dir = self.storage_path / "cache"
        cutoff = datetime.now().timestamp() - (max_age_days * 86400)
        removed = 0

        if not cache_dir.exists():
            return 0

        for model_dir in cache_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for cache_file in model_dir.glob("*.json"):
                if cache_file.stat().st_mtime < cutoff:
                    try:
                        cache_file.unlink()
                        removed += 1
                    except Exception:
                        pass

        return removed
