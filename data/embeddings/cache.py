import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


def cache_embedding(
    storage,
    entity_id: str,
    embedding: List[float],
    metadata: Dict[str, Any],
    model_name: str,
) -> None:
    cache_key = hashlib.sha256(
        f"{model_name}:{entity_id}".encode()
    ).hexdigest()[:16]

    model_dir = storage.storage_path / "cache" / model_name
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


def get_cached_embedding(storage, entity_id: str) -> Optional[Dict[str, Any]]:
    cache_dir = storage.storage_path / "cache"

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


def cleanup_cache(storage, max_age_days: int = 30) -> int:
    cache_dir = storage.storage_path / "cache"
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
