"""Baseline RAG for comparison:  Retrieve -> Generate.

A deliberately normal/simple RAG pipeline:
    question -> existing retriever (top-k chunks) -> prompt with the raw
    chunks -> LLM -> free-text answer.

It does NOT include anything LegalLens-specific:
  * no retrieval-score threshold gate,
  * no evidence-sufficiency check,
  * no citation validation,
  * no refusal enforcement (the model may still *choose* to say the context
    lacks the answer — that is exactly what refusal-correctness measures).

It reuses the project's real retriever and real LLM bridge — no second vector
store, no mock outputs. Results are whatever this pipeline actually produces.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..llm import LLMError, run_llm
from ..models import Evidence
from ..retriever import retrieve
from .metrics import looks_like_refusal

BASELINE_PROMPT = """You are a helpful assistant. Answer the question using ONLY the
context excerpts below. Be concise and specific. If the context does not contain
the answer, say so briefly.

CONTEXT EXCERPTS:
{context}

QUESTION: {question}
Answer:"""


@dataclass
class BaselineResult:
    question: str
    answer: str
    evidence: list[Evidence]
    refused: bool
    error: str | None = None


def _context(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no context excerpts retrieved)"
    return "\n\n".join(
        f"[excerpt {i + 1}] ({ev.filename}, page {ev.page})\n{ev.text}"
        for i, ev in enumerate(evidence)
    )


def run_baseline(question: str, doc_ids: list[str] | None = None) -> BaselineResult:
    """Retrieve -> Generate. Raises nothing; LLM failures are reported as errors."""
    evidence = retrieve(question, doc_ids=doc_ids)
    prompt = BASELINE_PROMPT.format(context=_context(evidence), question=question)
    try:
        answer = run_llm(prompt).strip()
    except LLMError as exc:
        return BaselineResult(question=question, answer="", evidence=evidence,
                              refused=False, error=f"model error: {exc}")
    return BaselineResult(
        question=question,
        answer=answer,
        evidence=evidence,
        refused=looks_like_refusal(answer),
        error=None,
    )
