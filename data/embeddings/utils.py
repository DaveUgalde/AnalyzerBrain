import os
from pathlib import Path


def get_directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0

    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total
