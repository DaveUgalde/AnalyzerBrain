try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from ...core.exceptions import EmbeddingException, ConfigurationError


def init_chroma(storage):
    if not CHROMA_AVAILABLE:
        raise ConfigurationError(
            "ChromaDB not installed. Install with 'pip install chromadb'"
        )

    try:
        client = chromadb.PersistentClient(
            path=storage.config["persist_directory"],
            settings=Settings(
                anonymized_telemetry=storage.config.get("anonymized_telemetry", False)
            ),
        )

        collection = client.get_or_create_collection(
            name="project_embeddings",
            metadata={"description": "Main embeddings collection for Project Brain"},
        )

        return client, collection

    except Exception as e:
        raise EmbeddingException(f"Failed to initialize ChromaDB: {e}")
