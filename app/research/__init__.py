"""Legal Research workflow (third feature) — evidence-first research over the
same uploaded documents and vector store that power RAG Chat and Legal
Drafting. See app/research/research_service.py for the pipeline."""
from .research_service import (STATUS_GROUNDED, STATUS_LIMITED,
                               STATUS_NO_PRECEDENT, STATUS_UNVERIFIED,
                               CaseConnection, JudgmentEntry, RelevantLaw,
                               ResearchRequest, ResearchResponse,
                               ResearchSource, research_document,
                               validate_research)

__all__ = [
    "ResearchRequest", "ResearchResponse", "ResearchSource", "RelevantLaw",
    "CaseConnection", "JudgmentEntry", "research_document", "validate_research",
    "STATUS_GROUNDED", "STATUS_LIMITED", "STATUS_NO_PRECEDENT",
    "STATUS_UNVERIFIED",
]
