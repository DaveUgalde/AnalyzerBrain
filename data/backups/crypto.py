from pathlib import Path
from typing import Optional

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class CryptoManager:
    def __init__(self, base_path: Path, enabled: bool):
        self.cipher: Optional[Fernet] = None

        if enabled and CRYPTO_AVAILABLE:
            key_file = base_path / "encryption.key"
            if key_file.exists():
                key = key_file.read_bytes()
            else:
                key = Fernet.generate_key()
                key_file.write_bytes(key)

            self.cipher = Fernet(key)

    def encrypt(self, data: bytes) -> bytes:
        return self.cipher.encrypt(data) if self.cipher else data

    def decrypt(self, data: bytes) -> bytes:
        return self.cipher.decrypt(data) if self.cipher else data

    @property
    def enabled(self) -> bool:
        return self.cipher is not None
