from pathlib import Path

import chromadb

from app.schemas import RetrievalResult

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "kb_docs"

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))


def retrieve(query: str, k: int = 3) -> RetrievalResult:
    collection = _client.get_collection(COLLECTION_NAME)
    results = collection.query(query_texts=[query], n_results=k)

    chunks = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    sources = [m["source"] for m in metadatas]
    confidence = max(0.0, min(1.0, 1 - distances[0])) if distances else 0.0

    return RetrievalResult(chunks=chunks, sources=sources, confidence=confidence)
