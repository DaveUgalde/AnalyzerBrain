from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from .cache import cache_embedding, get_cached_embedding
from .metadata import update_metadata
from ...core.exceptions import EmbeddingException


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

        cache_embedding(self, entity_id, embedding, metadata, model_name)
        update_metadata(self, "stored", entity_id, model_name)
        return True

    except Exception as e:
        raise EmbeddingException(f"Failed to store embedding: {e}")


def get_embedding(self, entity_id: str) -> Optional[Dict[str, Any]]:
    cached = get_cached_embedding(self, entity_id)
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

            cache_embedding(self, entity_id, embedding, metadata, model_name)

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
