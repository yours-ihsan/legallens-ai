"""Groundedness-evaluation tests. Everything runs offline — no LLM call.

Run: python -m pytest tests -q

Covers:
  1. grounded claim   -> marked grounded with evidence + page
  2. fabricated claim -> marked unsupported (never "sounds plausible")
  3. abstention       -> excluded from the groundedness ratio
  4. correct refusal  -> counted per system
  5. expected source  -> retrieval success detection
  6. full run         -> baseline and LegalLens both execute (stubbed LLMs)
  7. endpoints        -> GET/POST /api/evaluation return valid JSON
  8. regression       -> normal chat answer + exact refusal still work
  9. dataset sanity   -> every expected anchor exists in the real uploaded PDFs
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import main, pdf_extractor
from app.answer_generator import REFUSAL
from app.config import DOCS_JSON, UPLOADS_DIR
from app.evaluation import evaluator
from app.evaluation.baseline import BaselineResult
from app.evaluation.datasets import (
    CRIME,
    CONTRACT,
    EVAL_QUESTIONS,
    answerable_questions,
    ref,
    unanswerable_questions,
)
from app.evaluation.metrics import (
    evaluate_answer_claims,
    extract_claims,
    find_expected_source,
    looks_like_refusal,
    summarize_system,
)
from app.models import AnswerResponse, ChatRequest, Confidence, Evidence


# ---------------------------------------------------------------- helpers
def _ev(eid="E1", filename=CRIME, page=2, text="", score=0.6):
    return Evidence(id=eid, chunk_id=f"chunk-{eid}", doc_id="doc1",
                    filename=filename, page=page, text=text, score=score)


CRIME_TEXT = (
    "E4 - Complainant statement: Arun Mehta reported the laptop missing at 21:25 "
    "after returning to Room 304 of Harbor View Business Centre."
)


def _answer(question, answer, evidence, insufficient=False, stage=None):
    return AnswerResponse(
        question=question,
        answer=answer,
        insufficient=insufficient,
        citations=[],
        evidence=evidence,
        confidence=Confidence(score=0.5, band="Medium", retrieval=0.5,
                              model=0.0 if insufficient else 1.0),
        reason=None,
        refusal_stage=stage,
    )


# ---------------------------------------------------------------- 1. grounding
def test_grounded_claim_is_marked_grounded():
    src = _ev(page=2, text=CRIME_TEXT)
    rows = evaluate_answer_claims(
        "Arun Mehta reported the laptop missing at 21:25.", [src])
    assert len(rows) == 1
    assert rows[0]["kind"] == "claim"
    assert rows[0]["grounded"] is True
    assert rows[0]["evidence_ids"] == ["E1"]
    assert rows[0]["page"] == 2
    assert rows[0]["reason"]


def test_unsupported_number_is_marked_unsupported():
    src = _ev(page=2, text=CRIME_TEXT)
    rows = evaluate_answer_claims(
        "The laptop was reported missing at 03:15 by Sunita Verma.", [src])
    assert rows[0]["grounded"] is False
    assert rows[0]["evidence_ids"] == []
    assert rows[0]["page"] is None
    assert "03:15" in rows[0]["reason"]  # the fabricated number is called out


def test_unsupported_names_are_marked_unsupported_without_evidence():
    src = _ev(text="The office was found unlocked when the complainant returned at 21:25.")
    rows = evaluate_answer_claims("The judge ordered Kiran Rao arrested in Mumbai.", [src])
    assert rows[0]["grounded"] is False
    assert rows[0]["evidence_ids"] == []


def test_claim_is_unsupported_when_nothing_was_retrieved():
    rows = evaluate_answer_claims("Arun Mehta reported the laptop missing.", [])
    assert rows[0]["grounded"] is False
    assert rows[0]["evidence_ids"] == []


def test_claim_grounded_only_against_a_single_chunk():
    # tokens split across two chunks must not be combined into a "match"
    a = _ev("E1", text="The aggregate liability shall not exceed USD 50,000")
    b = _ev("E2", text="per contract year, governed by the laws of Singapore")
    rows = evaluate_answer_claims("USD 50,000 per contract year, Singapore.", [a, b])
    assert rows[0]["grounded"] is False


# ---------------------------------------------------------------- 2/3. abstention
def test_abstention_is_excluded_from_the_ratio():
    src = _ev(text="Clause 7 - Limitation of Liability. Aggregate liability shall not exceed USD 50,000.")
    rows = evaluate_answer_claims("The document does not specify any bank account number.", [src])
    assert rows[0]["kind"] == "abstention"
    assert rows[0]["grounded"] is None  # neither supported nor unsupported


def test_documented_negative_fact_counts_as_grounded_claim():
    # the crime PDF's own note ("does not identify ...") IS a source fact
    src = _ev(page=2, text="The material does not identify the person seen near the lift.")
    rows = evaluate_answer_claims(
        "The material does not identify the person seen near the lift.", [src])
    assert rows[0]["kind"] == "claim"
    assert rows[0]["grounded"] is True


def test_extract_claims_skips_refusal_and_tiny_fragments():
    assert extract_claims(REFUSAL) == []
    assert extract_claims("") == []
    assert extract_claims("Yes.") == []
    sents = extract_claims("The cap is USD 50,000 per contract year. Yes.")
    assert len(sents) == 1


def test_baseline_refusal_detector():
    assert looks_like_refusal(REFUSAL) is True
    assert looks_like_refusal("The document does not contain a bank account number.") is True
    assert looks_like_refusal("There is not enough information to answer that.") is True
    assert looks_like_refusal("The aggregate liability cap is USD 50,000 per contract year.") is False


# ---------------------------------------------------------------- 4. refusals
def test_correct_refusal_is_counted_per_system():
    qresults = [{
        "id": "QX",
        "question": "Who stole the laptop?",
        "answerable": False,
        "expected_facts": [],
        "baseline": {
            "answer": "Kiran Rao stole the laptop.",
            "refused": False,
            "error": None,
            "claims": evaluate_answer_claims("Kiran Rao stole the laptop.", []),
            "retrieval": {"success": None, "evidence_id": None},
        },
        "legallens": {
            "answer": REFUSAL,
            "refused": True,
            "error": None,
            "claims": [],
            "retrieval": {"success": None, "evidence_id": None},
        },
    }]
    s_ll = summarize_system(qresults, "legallens")
    s_bl = summarize_system(qresults, "baseline")
    assert s_ll["correct_refusals"] == 1
    assert s_ll["unanswerable_questions"] == 1
    assert s_bl["correct_refusals"] == 0
    assert s_bl["unsupported_claims"] == 1  # the fabricated accusation


# ---------------------------------------------------------------- 5. retrieval
def test_expected_source_found_by_filename_page_anchor():
    evidence = [_ev("E2", filename=CRIME, page=2,
                    text="... Arun Mehta reported the laptop\nmissing at 21:25 ...")]
    ok, matched = find_expected_source(
        evidence, [ref(CRIME, 2, "reported the laptop missing at 21:25")])
    assert ok is True and matched.id == "E2"

    # whitespace in the source (PDF line wraps) is normalised before matching
    ok_ws, _ = find_expected_source(evidence, [ref(CRIME, 2, "laptop missing at 21:25")])
    assert ok_ws is True

    # wrong page -> no match
    assert find_expected_source(evidence, [ref(CRIME, 1, "21:25")])[0] is False
    # wrong anchor -> no match
    assert find_expected_source(evidence, [ref(CRIME, 2, "bank account 9999")])[0] is False
    # wrong filename -> no match
    assert find_expected_source(evidence, [ref(CONTRACT, 2, "21:25")])[0] is False
    # any-of semantics: second reference matches
    ok2, _ = find_expected_source(
        evidence, [ref(CONTRACT, 1, "50,000"), ref(CRIME, 2, "21:25")])
    assert ok2 is True
    # unanswerable question (no expected refs) is never a retrieval hit
    assert find_expected_source(evidence, [])[0] is False


# ------------------------------------------------------- 6. full offline run
def test_run_evaluation_executes_both_systems(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "RESULTS_PATH", tmp_path / "evaluation_results.json")
    evidence = [_ev("E1", filename=CRIME, page=2, text=CRIME_TEXT)]
    unanswer_keys = ("Who stole the laptop", "bank account number",
                     "annual revenue", "serial number")

    def legallens_fn(question):
        if any(k in question for k in unanswer_keys):
            return _answer(question, REFUSAL, evidence, insufficient=True, stage="model")
        return _answer(question,
                       "Arun Mehta reported the laptop missing at 21:25 [E1].",
                       evidence)

    def baseline_fn(question):
        # baseline always answers, even for unanswerable questions
        return BaselineResult(question=question,
                              answer="The laptop was reported missing at 21:25 by Arun Mehta.",
                              evidence=evidence, refused=False)

    out = evaluator.run_evaluation(questions=EVAL_QUESTIONS,
                                   legallens_fn=legallens_fn,
                                   baseline_fn=baseline_fn,
                                   workers=1)
    assert out["evaluated"] is True and out["ran_at"]
    r = out["results"]

    # both systems executed and produced evaluable claims
    assert r["baseline"]["total_claims"] > 0
    assert r["legallens"]["total_claims"] > 0
    assert r["baseline"]["errors"] == 0 and r["legallens"]["errors"] == 0

    # refusal correctness from the stubs
    n_unans = len(unanswerable_questions())
    assert r["legallens"]["correct_refusals"] == n_unans
    assert r["baseline"]["correct_refusals"] == 0

    # expected evidence was found for at least the Q1 anchor
    assert r["legallens"]["retrieval_hits"] >= 1

    # groundedness computed, not invented; totals consistent
    assert r["legallens"]["groundedness"] == 100.0
    assert r["comparison"]["groundedness_improvement"] is not None
    for block in (r["baseline"], r["legallens"]):
        assert block["total_claims"] == (block["supported_claims"]
                                         + block["unsupported_claims"])

    # claim-level details are present and JSON-serialisable
    assert len(r["details"]) > 0
    json.dumps(out)

    # results persisted -> GET path returns them
    cached = evaluator.get_results()
    assert cached["evaluated"] is True
    assert cached["results"]["legallens"]["correct_refusals"] == n_unans
    json.dumps(cached)


def test_run_evaluation_without_llm_reports_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "RESULTS_PATH", tmp_path / "e.json")

    def broken(_q):
        raise RuntimeError("model unavailable")

    out = evaluator.run_evaluation(
        questions=[EVAL_QUESTIONS[0]],
        legallens_fn=lambda q: _answer(q, "", []),
        baseline_fn=broken,
        workers=1)
    r = out["results"]
    assert r["baseline"]["errors"] == 1
    assert r["legallens"]["errors"] == 0
    json.dumps(out)


def test_second_concurrent_run_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "RESULTS_PATH", tmp_path / "e.json")
    assert evaluator._run_lock.acquire(blocking=False)
    try:
        with pytest.raises(evaluator.EvaluationBusy):
            evaluator.run_evaluation(
                questions=[EVAL_QUESTIONS[0]],
                legallens_fn=lambda q: _answer(q, "", []),
                baseline_fn=lambda q: BaselineResult(q, "", [], False))
    finally:
        evaluator._run_lock.release()


def test_get_results_reports_not_evaluated_when_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "RESULTS_PATH", tmp_path / "missing.json")
    state = evaluator.get_results()
    assert state == {"evaluated": False, "ran_at": None, "results": None}


# ------------------------------------------------------------ 7. endpoints
def test_evaluation_endpoints_return_valid_json(monkeypatch):
    from fastapi.testclient import TestClient

    client = TestClient(main.app)

    monkeypatch.setattr(main, "get_results",
                        lambda: {"evaluated": False, "ran_at": None, "results": None})
    res = client.get("/api/evaluation")
    assert res.status_code == 200
    body = res.json()
    assert body["evaluated"] is False and body["results"] is None
    json.dumps(body)

    payload = {"evaluated": True, "ran_at": "2026-10-08T00:00:00+00:00",
               "results": {"baseline": {"groundedness": 80.0, "total_claims": 5,
                                        "supported_claims": 4, "unsupported_claims": 1,
                                        "retrieval_success": 100.0, "correct_refusals": 2},
                           "legallens": {"groundedness": 100.0, "total_claims": 4,
                                         "supported_claims": 4, "unsupported_claims": 0,
                                         "retrieval_success": 100.0, "correct_refusals": 4},
                           "comparison": {"groundedness_improvement": 20.0,
                                          "retrieval_improvement": 0.0},
                           "details": []}}
    monkeypatch.setattr(main, "run_evaluation", lambda: payload)
    res2 = client.post("/api/evaluation/run")
    assert res2.status_code == 200
    assert res2.json()["results"]["comparison"]["groundedness_improvement"] == 20.0
    json.dumps(res2.json())


def test_evaluation_run_conflict_when_busy(monkeypatch):
    from fastapi.testclient import TestClient

    from app.evaluation.evaluator import EvaluationBusy

    def busy():
        raise EvaluationBusy("an evaluation run is already in progress")

    monkeypatch.setattr(main, "run_evaluation", busy)
    res = TestClient(main.app).post("/api/evaluation/run")
    assert res.status_code == 409


# ------------------------------------------------- 8. existing flow regression
def test_api_chat_still_returns_grounded_answer(monkeypatch):
    from fastapi.testclient import TestClient

    src = _ev("E1", filename=CONTRACT, page=1,
              text="The aggregate liability shall not exceed USD 50,000 per contract year.")
    resp = _answer("What is the liability cap?",
                   "The aggregate liability shall not exceed USD 50,000 per contract year [E1].",
                   [src])
    monkeypatch.setattr(main, "answer_question", lambda q, doc_ids=None: resp)
    res = TestClient(main.app).post("/api/chat", json={"question": "What is the liability cap?"})
    assert res.status_code == 200
    body = res.json()
    assert body["insufficient"] is False
    assert "50,000" in body["answer"]
    json.dumps(body)


def test_api_chat_still_refuses_unsupported_question(monkeypatch):
    from fastapi.testclient import TestClient

    resp = _answer("What is the bank account number?", REFUSAL, [],
                   insufficient=True, stage="model")
    monkeypatch.setattr(main, "answer_question", lambda q, doc_ids=None: resp)
    res = TestClient(main.app).post("/api/chat",
                                    json={"question": "What is the bank account number?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == REFUSAL
    assert body["insufficient"] is True
    assert body["refusal_stage"] == "model"


# ------------------------------------------------------- 9. dataset sanity
def test_dataset_shape_and_denominators():
    assert 10 <= len(EVAL_QUESTIONS) <= 20
    ids = [q["id"] for q in EVAL_QUESTIONS]
    assert len(ids) == len(set(ids))
    assert len(answerable_questions()) >= 8
    assert len(unanswerable_questions()) >= 3
    for q in EVAL_QUESTIONS:
        if q["answerable"]:
            assert q["expected_evidence"], f"{q['id']} must have an expected source"
        else:
            assert q["expected_evidence"] == [], f"{q['id']} must expect nothing"


def test_every_expected_anchor_exists_in_the_real_uploaded_pdfs():
    """The dataset must not invent facts: every anchor is verbatim in the PDF."""
    if not DOCS_JSON.exists():
        pytest.skip("no document registry in this environment")
    registry = json.loads(DOCS_JSON.read_text(encoding="utf-8"))

    pages_by_name = {}
    for doc in registry.values():
        path = UPLOADS_DIR / f"{doc['doc_id']}.pdf"
        if not path.exists():
            continue
        pages_by_name[doc["filename"]] = {
            p.page_number: re.sub(r"\s+", " ", p.text).lower()
            for p in pdf_extractor.extract_pages(path)
        }

    checked = 0
    for item in EVAL_QUESTIONS:
        for r in item["expected_evidence"]:
            if r["filename"] not in pages_by_name:
                pytest.skip(f"document {r['filename']} is not uploaded here")
            page_text = pages_by_name[r["filename"]].get(r["page"], "")
            anchor = re.sub(r"\s+", " ", r["anchor"]).lower()
            assert anchor in page_text, (
                f"{item['id']}: anchor {r['anchor']!r} not found on "
                f"{r['filename']} page {r['page']}")
            checked += 1
    assert checked > 0
