"""Premise check + the chat pipeline's three-way decision.

Offline only: retrieval and the LLM are stubbed, so no network is touched.
Covers app/premise.py (extraction + marking) and the REFUSE / CORRECT / ANSWER
branches in answer_generator.answer_question, including the invariant that the
verifier still checks every claim after the premise step.
"""
from app import main
from app.answer_generator import REFUSAL, answer_question, premise_block, validate
from app.models import Evidence
from app.premise import (CONTRADICTED, MAX_ASSUMPTIONS, NOT_FOUND, SUPPORTED,
                         check_premises, extract_assumptions, mark_assumption)
from fastapi.testclient import TestClient

# ------------------------------------------------------------------ fixtures
FACTS = (
    "LegalLens Fictional Crime Case - Case Summary. An incident was reported at "
    "21:40 for a theft of a laptop from a private office. Arun Mehta is the "
    "reporting party. The matter was entered in the station house diary."
)
DENIALS = (
    "The source does not state that anyone was arrested, detained, accused or "
    "charged. The material does not identify the person seen, does not establish "
    "that any named individual took the laptop, and does not record the outcome "
    "of any investigation."
)
CONTRACT = (
    "Clause 7 - Limitation of Liability. The aggregate liability of either party "
    "under this Agreement shall not exceed USD 50,000 for any contract year."
)
CONVICTION = "After trial, John committed the theft and was convicted of theft in 2021."


def _ev(eid, text, score=0.55, page=1, filename="case.pdf"):
    return Evidence(id=eid, chunk_id=f"chunk-{eid.lower()}", doc_id="doc1",
                    filename=filename, page=page, index=0, text=text, score=score)


CASE_EV = [_ev("E1", FACTS, score=0.43), _ev("E2", DENIALS, score=0.41)]
CONTRACT_EV = [_ev("E1", CONTRACT, score=0.55, filename="sample.pdf")]


# ---------------------------------------------------------------- extraction
def test_extracts_that_clause_presupposition():
    q = "Can we conclude that Rohan Das committed the theft?"
    assumptions = extract_assumptions(q)
    assert "Rohan Das committed the theft" in assumptions


def test_extracts_wh_event_presupposition():
    assert extract_assumptions("When was the accused arrested?") == ["the accused arrested"]


def test_extracts_definite_np_and_drops_meta_head_noun():
    q = "What law may be relevant to the alleged theft in this case?"
    assumptions = extract_assumptions(q)
    assert assumptions == ["the alleged theft"]      # "the case" is meta-reference


def test_no_assumptions_for_plain_question():
    assert extract_assumptions("What happened in the case?") == []
    assert extract_assumptions("Who is liable under clause 7?") == []


def test_drops_assumption_contained_in_a_longer_one():
    q = "Give me the legal sections that prove the accused is guilty."
    assumptions = extract_assumptions(q)
    assert "the accused is guilty" in assumptions
    assert "the accused" not in assumptions          # contained -> dropped
    assert len(assumptions) <= MAX_ASSUMPTIONS


# ------------------------------------------------------------------- marking
def test_supported_assumption_is_traced_to_a_chunk():
    p = mark_assumption("the maximum aggregate liability", CONTRACT_EV)
    assert p["verdict"] == SUPPORTED
    assert p["evidence_id"] == "E1"


def test_supported_hedged_offence_is_not_a_guilt_presumption():
    p = mark_assumption("the alleged theft", CASE_EV)
    assert p["verdict"] == SUPPORTED                 # hedged + the theft exists


def test_guilt_presumption_contradicted_by_identity_denial():
    p = mark_assumption("Rohan Das committed the theft", CASE_EV)
    assert p["verdict"] == CONTRADICTED
    assert p["evidence_id"] == "E2"                  # chunk with the denials


def test_arrest_presumption_contradicted_when_material_records_none():
    p = mark_assumption("the accused arrested", CASE_EV)
    assert p["verdict"] == CONTRADICTED
    assert p["evidence_id"] in {"E1", "E2"}


def test_identity_reference_contradicted_not_supported_from_absence_text():
    # Lexical overlap exists only inside abstention sentences -> not support.
    p = mark_assumption("the accused", CASE_EV)
    assert p["verdict"] == CONTRADICTED


def test_guilt_can_be_supported_when_a_source_asserts_it():
    p = mark_assumption("John committed the theft", [_ev("E1", CONVICTION)])
    assert p["verdict"] == SUPPORTED
    assert p["evidence_id"] == "E1"


def test_unrelated_assumption_is_not_found():
    p = mark_assumption("the parking fines", CONTRACT_EV)
    assert p["verdict"] == NOT_FOUND
    assert p["evidence_id"] is None


def test_marking_without_evidence_is_not_found_not_crash():
    assert mark_assumption("the theft", [])["verdict"] == NOT_FOUND
    assert check_premises("What happened?", []) == []


# ---------------------------------------------------- decision: REFUSE branch
def test_below_low_floor_refuses_without_llm(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("LLM must not be called below the low floor")
    monkeypatch.setattr("app.answer_generator.run_llm", _fail)
    monkeypatch.setattr("app.answer_generator.retrieve",
                        lambda q, doc_ids=None: [_ev("E1", CONTRACT_EV[0].text, score=0.05)])

    resp = answer_question("What are the parking fines on the planet Mars?")
    assert resp.insufficient is True
    assert resp.answer == REFUSAL
    assert resp.refusal_stage == "retrieval"
    assert resp.decision == "refusal_low_evidence"
    assert "low floor" in (resp.reason or "")
    assert resp.premises == []


def test_all_not_found_assumptions_refuse_at_premise_stage(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("LLM must not be called when all assumptions are NOT_FOUND")
    monkeypatch.setattr("app.answer_generator.run_llm", _fail)
    monkeypatch.setattr("app.answer_generator.retrieve",
                        lambda q, doc_ids=None: [_ev("E1", CONTRACT_EV[0].text, score=0.5)])

    resp = answer_question("What are the parking fines on the planet Mars?")
    assert resp.insufficient is True
    assert resp.answer == REFUSAL
    assert resp.refusal_stage == "premise"
    assert resp.decision == "refusal_no_premise_support"
    assert resp.premises and all(p.verdict == NOT_FOUND for p in resp.premises)
    assert "parking fines" in (resp.reason or "")


# -------------------------------------------------- decision: CORRECT branch
def test_contradicted_assumption_takes_the_correction_branch(monkeypatch):
    seen = {}

    def _llm(prompt):
        seen["prompt"] = prompt
        return ('{"answer": "The sources state that the material does not establish '
                'that any named individual took the laptop, not that Rohan Das '
                'committed the theft [E2].", "insufficient": false, "confidence": 0.8, '
                '"citations": ["E2"]}')

    monkeypatch.setattr("app.answer_generator.retrieve", lambda q, doc_ids=None: CASE_EV)
    monkeypatch.setattr("app.answer_generator.run_llm", _llm)

    resp = answer_question("Can we conclude that Rohan Das committed the theft?")
    assert resp.insufficient is False
    assert resp.decision == "premise_correction"
    assert any(p.verdict == CONTRADICTED and p.evidence_id == "E2" for p in resp.premises)
    # The prompt must carry the system-computed verdicts.
    assert "PREMISE CHECK" in seen["prompt"]
    assert "CONTRADICTED" in seen["prompt"]
    assert "authoritative" in seen["prompt"]
    assert "see [E2]" in seen["prompt"]
    assert resp.citations and resp.citations[0].evidence_id == "E2"


# ---------------------------------------------------- decision: ANSWER branch
def test_supported_assumption_answers_normally(monkeypatch):
    seen = {}

    def _llm(prompt):
        seen["prompt"] = prompt
        return ('{"answer": "The aggregate liability shall not exceed USD 50,000 for '
                'any contract year [E1].", "insufficient": false, "confidence": 0.9, '
                '"citations": ["E1"]}')

    monkeypatch.setattr("app.answer_generator.retrieve", lambda q, doc_ids=None: CONTRACT_EV)
    monkeypatch.setattr("app.answer_generator.run_llm", _llm)

    resp = answer_question("What is the maximum aggregate liability under this agreement?")
    assert resp.insufficient is False
    assert resp.decision == "grounded_answer"
    assert resp.premises and resp.premises[0].verdict == SUPPORTED
    assert resp.premises[0].evidence_id == "E1"
    assert "PREMISE CHECK" in seen["prompt"]
    assert "SUPPORTED" in seen["prompt"]


def test_score_between_floor_and_old_threshold_reaches_the_llm(monkeypatch):
    """0.10 <= score < 0.30 must be passed to the LLM (old gate refused)."""
    def _llm(prompt):
        return ('{"answer": "Clause 7 concerns limitation of liability [E1].", '
                '"insufficient": false, "confidence": 0.7, "citations": ["E1"]}')

    monkeypatch.setattr("app.answer_generator.retrieve",
                        lambda q, doc_ids=None: [_ev("E1", CONTRACT_EV[0].text, score=0.25)])
    monkeypatch.setattr("app.answer_generator.run_llm", _llm)

    resp = answer_question("Who is liable under clause 7?")
    assert resp.insufficient is False
    assert resp.decision == "grounded_answer"
    assert resp.confidence.retrieval == 0.25


# --------------------------------------------- invariant: verifier still runs
def test_verifier_still_rejects_hallucinated_ids_after_premise_step(monkeypatch):
    def _llm(prompt):
        return ('{"answer": "The liability is USD 50,000 [E9].", "insufficient": false, '
                '"confidence": 0.9, "citations": ["E9"]}')

    monkeypatch.setattr("app.answer_generator.retrieve", lambda q, doc_ids=None: CONTRACT_EV)
    monkeypatch.setattr("app.answer_generator.run_llm", _llm)

    resp = answer_question("What is the maximum aggregate liability under this agreement?")
    assert resp.insufficient is True
    assert resp.refusal_stage == "verifier"
    assert resp.decision == "refusal_verifier"
    # Premise verdicts survive onto the refusal response.
    assert resp.premises and resp.premises[0].verdict == SUPPORTED
    assert resp.answer == REFUSAL


def test_validate_without_premises_stays_backward_compatible():
    ok = validate("q", CONTRACT_EV, 0.55,
                  {"answer": "USD 50,000 [E1].", "insufficient": False,
                   "confidence": 0.9, "citations": ["E1"]})
    assert ok.decision == "grounded_answer"
    assert ok.premises == []

    not_dict = validate("q", CONTRACT_EV, 0.55, "not a JSON object")
    assert not_dict.decision == "refusal_verifier"

    no_answer = validate("q", CONTRACT_EV, 0.55, {"nope": True})
    assert no_answer.decision == "refusal_model"


# ---------------------------------------------------------------- prompt glue
def test_premise_block_is_empty_and_instructions_are_precise():
    assert premise_block([]) == ""
    block = premise_block([{"assumption": "Rohan Das committed the theft",
                            "verdict": CONTRADICTED, "evidence_id": "E2",
                            "note": "x"}])
    assert "PREMISE CHECK" in block
    assert "CONTRADICTED" in block
    assert "see [E2]" in block
    assert "The sources state X, not Y [E#]" in block
    assert "authoritative" in block


# ------------------------------------------------------------- API contract
def test_api_chat_exposes_decision_and_premises(monkeypatch):
    def _llm(prompt):
        return ('{"answer": "The aggregate liability shall not exceed USD 50,000 for '
                'any contract year [E1].", "insufficient": false, "confidence": 0.9, '
                '"citations": ["E1"]}')

    monkeypatch.setattr("app.answer_generator.retrieve", lambda q, doc_ids=None: CONTRACT_EV)
    monkeypatch.setattr("app.answer_generator.run_llm", _llm)

    res = TestClient(main.app).post(
        "/api/chat", json={"question": "What is the maximum aggregate liability under this agreement?"})
    assert res.status_code == 200
    body = res.json()
    assert body["decision"] == "grounded_answer"
    assert isinstance(body["premises"], list) and body["premises"]
    assert body["premises"][0]["verdict"] in (SUPPORTED, CONTRADICTED, NOT_FOUND)


def test_api_chat_refusal_reports_premise_case(monkeypatch):
    monkeypatch.setattr("app.answer_generator.retrieve",
                        lambda q, doc_ids=None: [_ev("E1", CONTRACT_EV[0].text, score=0.5)])
    monkeypatch.setattr("app.answer_generator.run_llm",
                        lambda prompt: (_ for _ in ()).throw(AssertionError("no LLM")))

    res = TestClient(main.app).post(
        "/api/chat", json={"question": "What are the parking fines on the planet Mars?"})
    assert res.status_code == 200
    body = res.json()
    assert body["insufficient"] is True
    assert body["answer"] == REFUSAL
    assert body["refusal_stage"] == "premise"
    assert body["decision"] == "refusal_no_premise_support"
    assert all(p["verdict"] == NOT_FOUND for p in body["premises"])
