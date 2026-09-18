from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


def test_chroma_persists_synthetic_vectors(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    settings = ChromaSettings(anonymized_telemetry=False)

    with chromadb.PersistentClient(path=str(persist_dir), settings=settings) as client:
        collection = client.get_or_create_collection(
            name="stage1_roundtrip",
            embedding_function=None,
        )
        collection.add(
            ids=["alpha", "beta"],
            documents=["first synthetic doc", "second synthetic doc"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        stored = collection.get(ids=["alpha"], include=["documents", "embeddings"])
        assert stored["documents"] == ["first synthetic doc"]

    with chromadb.PersistentClient(path=str(persist_dir), settings=settings) as client:
        collection = client.get_collection(
            name="stage1_roundtrip",
            embedding_function=None,
        )
        queried = collection.query(
            query_embeddings=[[1.0, 0.0, 0.0]],
            n_results=1,
            include=["documents"],
        )
        assert queried["ids"][0] == ["alpha"]
        assert queried["documents"][0] == ["first synthetic doc"]
