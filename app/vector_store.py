"""Persistent local vector store (Chroma).

Invariant: doc_id / filename / page metadata is attached to EVERY vector and
travels with it through retrieval. The collection uses cosine similarity.
"""
import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME
from .models import Chunk


class VectorStore:
    def __init__(self, path=CHROMA_DIR, collection=COLLECTION_NAME):
        self._client = chromadb.PersistentClient(path=str(path))
        try:
            self._collection = self._client.get_collection(collection)
            meta = getattr(self._collection, "metadata", None) or {}
            self._space = meta.get("hnsw:space", "cosine")
        except Exception:
            self._collection = self._client.create_collection(
                collection, metadata={"hnsw:space": "cosine"}
            )
            self._space = "cosine"

    @staticmethod
    def _to_similarity(distance: float, space: str) -> float:
        if space == "cosine":
            return 1.0 - distance
        return 1.0 / (1.0 + distance)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "page": c.page,
                    "index": c.index,
                }
                for c in chunks
            ],
        )

    def query(self, text: str, top_k: int, doc_ids: list[str] | None = None) -> list[dict]:
        total = self._collection.count()
        if total == 0:
            return []
        n = min(top_k, total)
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        res = self._collection.query(
            query_texts=[text],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "meta": meta,
                    "score": self._to_similarity(float(dist), self._space),
                }
            )
        return hits

    def delete_doc(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._collection.count()
