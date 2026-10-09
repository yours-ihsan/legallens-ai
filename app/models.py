"""Shared data models. Every chunk/evidence carries source metadata end to end."""
from typing import Optional

from pydantic import BaseModel


class PageText(BaseModel):
    page_number: int
    text: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    index: int
    text: str


class Evidence(BaseModel):
    id: str  # E1, E2, ... assigned AFTER retrieval; the only IDs the LLM may cite
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str
    score: float  # cosine similarity, higher = closer match


class Citation(BaseModel):
    evidence_id: str
    filename: str
    page: int
    quote: str


class Confidence(BaseModel):
    score: float
    band: str  # High / Medium / Low
    retrieval: float
    model: float


class PremiseCheck(BaseModel):
    """One assumption extracted from the question, marked from the chunks."""
    assumption: str
    verdict: str  # SUPPORTED / CONTRADICTED / NOT_FOUND (app/premise.py)
    evidence_id: Optional[str] = None
    note: Optional[str] = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    insufficient: bool
    citations: list[Citation]
    evidence: list[Evidence]
    confidence: Confidence
    reason: Optional[str] = None
    # Who caused a refusal: "retrieval" (below the low floor, LLM not
    # consulted), "premise" (every assumption NOT_FOUND, LLM not consulted),
    # "system" (LLM/parse failure), "model" (model declined), "verifier"
    # (validator rejected the model's claims). None when answered.
    refusal_stage: Optional[str] = None
    # THREE-WAY evidence decision, shown to the user:
    #   refusal_low_evidence       best score < LOW_FLOOR, LLM not consulted
    #   refusal_no_premise_support every assumption NOT_FOUND, LLM not consulted
    #   refusal_system / refusal_model / refusal_verifier  downstream failures
    #   premise_correction         answered; >= 1 contradicted assumption
    #   grounded_answer             answered normally
    decision: str = "grounded_answer"
    # The premise check itself (computed by Python from the chunks only).
    premises: list[PremiseCheck] = []


class ChatRequest(BaseModel):
    question: str
    doc_ids: Optional[list[str]] = None


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    uploaded_at: str
