"""Legal Drafting — EVIDENCE FIRST → DRAFT SECOND.

Isolated workflow built on top of the existing grounded architecture:
reuses retrieval, the sufficiency threshold, the evidence packet builder,
the LLM bridge and the deterministic claim-grinding metrics from
app/evaluation. It is imported ONLY by the new /api/draft endpoint — the
RAG chat path (answer_generator) never imports it and is unaffected.
"""
from .drafter import DraftRequest, DraftResponse, draft_document
from .requirements import DRAFT_TYPES

__all__ = ["DraftRequest", "DraftResponse", "draft_document", "DRAFT_TYPES"]
