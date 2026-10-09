"""Retrieval: question -> scored evidence with source metadata attached."""
from .config import TOP_K
from .models import Evidence
from .vector_store import VectorStore

_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def retrieve(question: str, doc_ids: list[str] | None = None, top_k: int = TOP_K) -> list[Evidence]:
    hits = get_store().query(question, top_k=top_k, doc_ids=doc_ids)
    return [
        Evidence(
            id=f"E{i + 1}",
            chunk_id=h["chunk_id"],
            doc_id=h["meta"]["doc_id"],
            filename=h["meta"]["filename"],
            page=int(h["meta"]["page"]),
            text=h["text"],
            score=round(h["score"], 4),
        )
        for i, h in enumerate(hits)
    ]
