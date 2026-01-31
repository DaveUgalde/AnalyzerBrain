import json
from datetime import datetime
from typing import Dict, Any

from .utils import get_directory_size


def update_metadata(storage, operation: str, entity_id: str, model_name: str) -> None:
    metadata_path = storage.storage_path / "metadata.json"

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

    metadata["total_embeddings"] = storage.collection.count()
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


def get_stats(storage) -> Dict[str, Any]:
    try:
        return {
            "total_embeddings": storage.collection.count(),
            "chroma_size_bytes": get_directory_size(storage.storage_path / "chroma"),
            "cache_size_bytes": get_directory_size(storage.storage_path / "cache"),
        }
    except Exception as e:
        return {"error": str(e)}
