import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any

from .models import CacheEntry


class DiskCache:
    def __init__(self, path: Path, max_size: int):
        self.path = path
        self.max_size = max_size
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.path / "index.json"

    def _hash(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        index = self._load_index()
        meta = index.get(key)
        if not meta:
            return None

        file = self.path / f"{meta['hash']}.pkl"
        if not file.exists():
            index.pop(key, None)
            self._save_index(index)
            return None

        try:
            entry: CacheEntry = pickle.loads(file.read_bytes())
            if entry.expired:
                file.unlink(missing_ok=True)
                index.pop(key, None)
                self._save_index(index)
                return None
            return entry.value
        except Exception:
            return None

    def set(self, key: str, entry: CacheEntry) -> bool:
        try:
            index = self._load_index()
            h = self._hash(key)
            (self.path / f"{h}.pkl").write_bytes(pickle.dumps(entry))

            index[key] = {
                "hash": h,
                "created_at": time.time(),
            }

            if len(index) > self.max_size:
                for k in list(index.keys())[:100]:
                    (self.path / f"{index[k]['hash']}.pkl").unlink(missing_ok=True)
                    index.pop(k, None)

            self._save_index(index)
            return True
        except Exception:
            return False

    def _load_index(self) -> Dict[str, Any]:
        if self.index_file.exists():
            try:
                return json.loads(self.index_file.read_text())
            except Exception:
                pass
        return {}

    def _save_index(self, index: Dict[str, Any]) -> None:
        self.index_file.write_text(json.dumps(index, indent=2))
