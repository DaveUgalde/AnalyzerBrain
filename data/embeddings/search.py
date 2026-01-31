from typing import Dict, Any, List, Optional
from ...core.exceptions import EmbeddingException


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

        similar = []

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
