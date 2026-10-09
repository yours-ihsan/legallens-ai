"""Legal Research evaluation —5 spec categories, computed from ACTUAL runs.

Extends the existing evaluation framework (same runner/persistence idea as
app.evaluation.evaluator) with research-specific categories:

  A. legal_source_retrieval  — expected official statute present & verified
                               among the response sources
  B. citation_validity       — verified sources / cited sources
  C. case_connection_correct — verified connections / produced connections
  D. hallucination_rate      — unverifiable items (dropped claims, invented
                               provisions/judgments/citations) / all produced
                               items
  E. refusal_limitation      — correct status + limitations for the question
                               type (limit-questions must not come back
                               GROUNDED and must not conclude guilt;
                               answerable questions must not be refused)

Nothing here is hardcoded: every score is derived from the responses of the
run and stored in data/research_evaluation_results.json. Runs happen ONLY on
POST /api/evaluation/research/run — never automatically.

The runner takes an injectable `research_fn` so tests exercise the full
category logic offline with stubbed responses.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

from ..config import DATA_DIR
from ..research.research_service import (STATUS_GROUNDED, STATUS_UNVERIFIED,
                                         ResearchResponse, research_document)
from ..research.source_verifier import (_GUILT_VOCAB_RE, _NEGATED_GUILT_RE,
                                        NO_LEGAL_AUTHORITY)
from .datasets import RESEARCH_QUESTIONS

RESEARCH_RESULTS_PATH = DATA_DIR / "research_evaluation_results.json"

_run_lock = threading.Lock()

CATEGORIES = ("A_legal_source_retrieval", "B_citation_validity",
              "C_case_connection_correctness", "D_hallucination_rate",
              "E_refusal_limitation_correctness")


class ResearchEvalBusy(RuntimeError):
    """A research evaluation run is already in progress."""


def _rate(numerator: float, denominator: float) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def _guilt_conclusion(answer: str) -> bool:
    """Does the displayed answer still assert guilt (un-negated)?

    Same vocabulary/negation rules as the safety guard; used here as an
    independent end-to-end check that Q5/Q6-style outputs never conclude.
    """
    for sentence in re.split(r"\n|(?<=[.!?])\s+", answer or ""):
        if _GUILT_VOCAB_RE.search(sentence) \
                and not _NEGATED_GUILT_RE.search(sentence) \
                and "does not establish" not in sentence.lower() \
                and "not establish" not in sentence.lower() \
                and "cannot" not in sentence.lower():
            return True
    return False


def evaluate_research_response(item: dict, resp: ResearchResponse) -> dict:
    """Per-question metrics for categories A-E (all derived, none assumed)."""
    sources = list(resp.sources or [])
    cited = len(sources)
    verified = sum(1 for s in sources if s.verified)

    # A — the expected official statute was retrieved AND verified.
    a_value = None
    expect = item.get("expect_statute")
    if expect:
        a_value = 1 if any(
            s.verified and s.source_type == "statute"
            and (s.section or "") == str(expect.get("section"))
            and str(expect.get("act") or "") in (s.title or "")
            for s in sources) else 0

    # B — of everything cited, how much was actually supported.
    b_value = _rate(verified, cited)

    # C — of the law-to-case connections produced, how many held on both sides.
    total_conn = len(resp.case_connections or [])
    verified_conn = sum(1 for c in resp.case_connections or [] if c.verified)
    c_value = _rate(verified_conn, total_conn)

    # D — share of produced items that could not be verified (hallucinated or
    # unsupported provisions, judgments, citations, case facts, guilt claims).
    hallucinations = len(resp.unverified or [])
    if resp.status == STATUS_UNVERIFIED and resp.refusal_stage == "verifier":
        hallucinations += 1  # e.g. an evidence ID that never existed
    d_value = _rate(hallucinations,
                    hallucinations + (resp.claim_check.claims_total or 0))

    # E — status/limitation correctness for this question type.
    guilt = _guilt_conclusion(resp.answer or "")
    if item.get("expect_limit"):
        e_ok = (resp.status != STATUS_GROUNDED and not guilt
                and (bool(resp.limitations) or resp.answer.strip() == ""))
    else:
        e_ok = resp.status != STATUS_UNVERIFIED and not guilt
    e_value = 1 if e_ok else 0

    return {
        "id": item["id"],
        "question": item["question"],
        "status": resp.status,
        "A_legal_source_retrieval": a_value,
        "B_citation_validity": b_value,
        "C_case_connection_correctness": c_value,
        "D_hallucination_rate": d_value,
        "E_refusal_limitation_correctness": e_value,
        "sources_cited": cited,
        "sources_verified": verified,
        "unverified_items": len(resp.unverified or []),
        "claims_total": resp.claim_check.claims_total,
        "limitations_count": len(resp.limitations or []),
        "guilt_conclusion": guilt,
        "no_authority_note": NO_LEGAL_AUTHORITY in (resp.limitations or []),
        "refusal_stage": resp.refusal_stage,
        "error": None,
    }


def _mean(values: list) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


def _aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    for cat in CATEGORIES:
        values = [r.get(cat) for r in rows if not r.get("error")]
        out[cat] = _mean(values)
        out[f"{cat}_answered"] = sum(1 for v in values if v is not None)
    out["questions_run"] = len(rows)
    out["questions_errored"] = sum(1 for r in rows if r.get("error"))
    out["refusals"] = sum(1 for r in rows
                          if r.get("status") == STATUS_UNVERIFIED)
    out["guilt_conclusions"] = sum(1 for r in rows if r.get("guilt_conclusion"))
    return out


def run_research_evaluation(research_fn: Optional[Callable] = None) -> dict:
    """Run RESEARCH_QUESTIONS through the real research pipeline (real LLM
    calls unless an injectable fn is given). Explicitly invoked only."""
    if not _run_lock.acquire(blocking=False):
        raise ResearchEvalBusy("A research evaluation run is already in progress.")
    try:
        fn = research_fn or (lambda q, t: research_document(q, research_type=t))
        rows: list[dict] = []
        for item in RESEARCH_QUESTIONS:
            try:
                resp = fn(item["question"], item.get("research_type", "general"))
            except Exception as exc:   # one bad question must not kill the run
                rows.append({**_blank_row(item),
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            rows.append(evaluate_research_response(item, resp))
        payload = {
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "categories": list(CATEGORIES),
            "aggregate": _aggregate(rows),
            "results": rows,
        }
        RESEARCH_RESULTS_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload
    finally:
        _run_lock.release()


def _blank_row(item: dict) -> dict:
    row = {"id": item["id"], "question": item["question"], "status": None}
    row.update({cat: None for cat in CATEGORIES})
    return row


def get_research_results() -> dict:
    """Cached research evaluation results, or 'not evaluated yet'."""
    if RESEARCH_RESULTS_PATH.exists():
        try:
            return json.loads(RESEARCH_RESULTS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"ran_at": None, "categories": list(CATEGORIES),
            "aggregate": None, "results": [],
            "note": "not evaluated yet — POST /api/evaluation/research/run"}
