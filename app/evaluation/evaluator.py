"""Evaluation orchestrator — runs both pipelines and computes real metrics.

Isolation guarantees:
  * imported only by app/main.py (behind /api/evaluation*); the normal
    PDF -> ... -> answer path never imports this module,
  * the LegalLens side calls the REAL `answer_question()` unchanged
    (threshold gate, validator, refusal behaviour all intact),
  * the baseline side calls `run_baseline()` (retrieve -> generate only),
  * every percentage in the output is calculated from these runs and
    persisted to data/evaluation_results.json — nothing is hardcoded,
  * `legallens_fn` / `baseline_fn` are injectable so the offline test suite
    can run the whole evaluation without any LLM call.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Optional

from ..answer_generator import REFUSAL, answer_question
from ..config import DATA_DIR
from ..retriever import get_store
from .baseline import BaselineResult, run_baseline
from .datasets import EVAL_QUESTIONS
from .metrics import aggregate, evaluate_answer_claims, find_expected_source

RESULTS_PATH = DATA_DIR / "evaluation_results.json"

DEFAULT_WORKERS = max(1, int(os.environ.get("LEGALLENS_EVAL_WORKERS", "4")))

_run_lock = threading.Lock()


class EvaluationBusy(RuntimeError):
    """A run is already in progress."""


# ------------------------------------------------------------------ per system
def _legallens_claims(resp) -> list[dict]:
    """Claims only exist for non-refused answers; a refusal asserts no facts."""
    if resp.insufficient or resp.answer.strip() == REFUSAL:
        return []
    return evaluate_answer_claims(resp.answer, resp.evidence, REFUSAL)


def _eval_legallens(question: str, fn: Callable) -> dict:
    try:
        resp = fn(question)
    except Exception as exc:  # a failing system must not kill the whole run
        return {"answer": "", "refused": False, "error": f"{type(exc).__name__}: {exc}",
                "claims": [], "_evidence": [],
                "retrieval": {"success": False, "evidence_id": None}}
    return {
        "answer": resp.answer,
        "refused": bool(resp.insufficient),
        "stage": resp.refusal_stage,
        "error": None,
        "claims": _legallens_claims(resp),
        "_evidence": resp.evidence,
        "retrieval": {"success": False, "evidence_id": None},
    }


def _eval_baseline(question: str, fn: Callable) -> dict:
    try:
        result = fn(question)
    except Exception as exc:
        return {"answer": "", "refused": False, "error": f"{type(exc).__name__}: {exc}",
                "claims": [], "retrieval": {"success": False, "evidence_id": None}}
    if not isinstance(result, BaselineResult):  # tolerate simple stubs in tests
        result = BaselineResult(question=question, answer=str(result), evidence=[],
                                refused=False)
    return {
        "answer": result.answer,
        "refused": bool(result.refused),
        "error": result.error,
        "claims": evaluate_answer_claims(result.answer, result.evidence, REFUSAL),
        "_evidence": result.evidence,
        "retrieval": {"success": False, "evidence_id": None},
    }


# ------------------------------------------------------------------- one item
def _evaluate_one(item: dict, legallens_fn: Callable, baseline_fn: Callable) -> dict:
    question = item["question"]
    expected = item.get("expected_evidence", [])

    ll = _eval_legallens(question, legallens_fn)
    bl = _eval_baseline(question, baseline_fn)

    # Retrieval success is evaluated per system against its own evidence set
    # (both use the same retriever, so the values are expected to agree).
    if item.get("answerable"):
        for block in (ll, bl):
            ev_list = block.pop("_evidence", [])
            found, ev = find_expected_source(ev_list, expected)
            block["retrieval"] = {"success": bool(found),
                                  "evidence_id": getattr(ev, "id", None),
                                  "filename": getattr(ev, "filename", None),
                                  "page": getattr(ev, "page", None)}
    else:
        ll.pop("_evidence", None)
        bl.pop("_evidence", None)
        ll["retrieval"] = {"success": None, "evidence_id": None}
        bl["retrieval"] = {"success": None, "evidence_id": None}

    return {
        "id": item["id"],
        "question": question,
        "answerable": bool(item.get("answerable")),
        "expected_facts": item.get("expected_facts", []),
        "baseline": bl,
        "legallens": ll,
    }


# ------------------------------------------------------------------ run / read
def get_results() -> dict:
    """Cached results, or an explicit 'not evaluated yet' state."""
    if not RESULTS_PATH.exists():
        return {"evaluated": False, "ran_at": None, "results": None}
    try:
        payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"evaluated": False, "ran_at": None, "results": None}
    return {
        "evaluated": True,
        "ran_at": payload.get("ran_at"),
        "results": payload.get("results"),
    }


def run_evaluation(questions: Optional[list[dict]] = None,
                   legallens_fn: Optional[Callable] = None,
                   baseline_fn: Optional[Callable] = None,
                   workers: Optional[int] = None) -> dict:
    """Execute the evaluation and persist the calculated results."""
    if not _run_lock.acquire(blocking=False):
        raise EvaluationBusy("an evaluation run is already in progress")
    try:
        items = questions if questions is not None else EVAL_QUESTIONS
        legallens_fn = legallens_fn or answer_question
        baseline_fn = baseline_fn or run_baseline
        workers = workers or DEFAULT_WORKERS

        get_store()  # warm the singleton before worker threads query it

        qresults: list[dict] = []
        if workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
                futures = [pool.submit(_evaluate_one, it, legallens_fn, baseline_fn)
                           for it in items]
                qresults = [f.result() for f in futures]
        else:
            qresults = [_evaluate_one(it, legallens_fn, baseline_fn) for it in items]

        results = aggregate(qresults)
        ran_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = {"ran_at": ran_at, "results": results}
        tmp = RESULTS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(RESULTS_PATH)
        return {"evaluated": True, "ran_at": ran_at, "results": results}
    finally:
        _run_lock.release()
