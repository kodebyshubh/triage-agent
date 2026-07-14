"""One-time script: chunk data/kb/*.md and embed into a local persistent ChromaDB collection.

Run with: python -m app.ingest
"""

import re
from pathlib import Path

import chromadb

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "kb_docs"


def chunk_markdown(text: str) -> list[str]:
    """Split on '## ' headers so each FAQ entry (header + body) is one chunk."""
    sections = re.split(r"\n(?=## )", text.strip())
    return [s.strip() for s in sections if s.strip() and not s.strip().startswith("# ")]


def ingest() -> int:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    ids, documents, metadatas = [], [], []
    for md_file in sorted(KB_DIR.glob("*.md")):
        chunks = chunk_markdown(md_file.read_text(encoding="utf-8"))
        for i, chunk in enumerate(chunks):
            ids.append(f"{md_file.stem}-{i}")
            documents.append(chunk)
            metadatas.append({"source": md_file.name})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


if __name__ == "__main__":
    n = ingest()
    print(f"Ingested {n} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")
