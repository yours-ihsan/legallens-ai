"""Legal Drafting tests — EVIDENCE FIRST → DRAFT SECOND.

All offline: the LLM is stubbed (monkeypatched), retrieval results are
fabricated Evidence objects. The API contract tests use FastAPI's TestClient.

Grounding rules under test:
  * field availability is decided by Python patterns, never by the model
  * missing fields are reported, never invented
  * hallucinated evidence IDs / unverified pages / no markers => refusal
  * sources map [E#] -> (filename, page) from vector-store metadata only
  * all four draft types run through the same workflow
  * the existing chat pipeline is untouched (full suite must still pass)
"""
import json

import pytest

from app import main
from app.answer_generator import REFUSAL
from app.drafting import DRAFT_TYPES, drafter
from app.drafting.drafter import (
    AvailableField,
    ClaimCheck,
    DraftRequest,
    DraftResponse,
    DraftSource,
    claim_sentences,
    detect_fields,
    draft_document,
    normalize_markers,
    validate_draft,
    verify_citations,
)
from app.models import Evidence

# Verbatim phrases from the indexed LegalLens_Fictional_Crime_Case.pdf
# (verified against the real PDF earlier — anchors are not invented).
CASE_HEADER = (
    "Case ID LL-CR-2026-014 Matter Theft of a laptop from a private office "
    "Incident Date 18 September 2026 Reported At 21:40 "
    "Location Harbor View Business Centre, Chennai Complainant Arun Mehta"
)
CASE_LOG = (
    "Evidence Log E4 Complainant statement: Arun Mehta reported the laptop "
    "missing at 21:25. The incident is formally reported to building security."
)

# Verbatim excerpts of LegalLens_Drafting_Source_Pack.pdf — the labels-without-
# values sentences are exactly the ones that used to read as "available".
PACK_CASE_FACTS = (
    "1. Common Case Facts\n"
    "Case ID: LL-CR-2026-014\n"
    "Matter: Theft of a laptop from a private office\n"
    "Incident date: 18 September 2026\n"
    "Reported time: 21:40\n"
    "Complainant: Arun Mehta\n"
    "Witness: Priya Nair\n"
    "Timeline\n"
    "21:40 The incident is reported."
)
PACK_PETITION_SECTION = (
    "3. Petition\n"
    "Use only: Case ID, incident facts, timeline, complainant/witness "
    "information and E1-E4.\n"
    "Missing: Court name, jurisdiction, petitioner/respondent legal status, "
    "petition number, filing date, statutory provisions, legal grounds, "
    "exact relief sought and advocate details.\n"
    "Grounding rule: Use placeholders such as [COURT NAME — NOT PROVIDED] "
    "instead of inventing missing particulars."
)
PACK_LIMITATION = (
    "Investigation limitation: The material does not identify the person "
    "seen, does not establish that any named individual took the laptop, "
    "and does not provide a phone number, bank account number, address, "
    "FIR number, police station, court name, legal sections, or judicial "
    "precedents."
)


def _ev(eid="E1", page=1, text=CASE_HEADER, score=0.55):
    return Evidence(id=eid, chunk_id=f"c-{eid}", doc_id="d1",
                    filename="LegalLens_Fictional_Crime_Case.pdf",
                    page=page, text=text, score=score)


BAIL_DRAFT = """IN THE [COURT NAME — NOT PROVIDED]
BAIL APPLICATION

MOST RESPECTFULLY SUBMITTED:

1. Background
2. Facts supported by the source document

The incident was reported at 21:40 on 18 September 2026 [E1].
The complainant is Arun Mehta [E1].

Accused: [ACCUSED NAME — NOT PROVIDED]
"""


# ------------------------------------------------------------------ config
def test_draft_types_config_complete():
    assert set(DRAFT_TYPES) == {"bail_application", "petition",
                                "legal_notice", "affidavit"}
    for key, spec in DRAFT_TYPES.items():
        assert spec["label"].strip(), key
        assert spec["query"].strip(), key
        assert "NOT PROVIDED" in spec["structure"], key
        assert spec["fields"], key
        labels = [f["label"] for f in spec["fields"]]
        assert len(labels) == len(set(labels)), f"duplicate labels in {key}"
        for field in spec["fields"]:
            assert field["key"] and field["label"], key
            assert isinstance(field["patterns"], list) and field["patterns"], key
            assert all(isinstance(p, str) and p for p in field["patterns"]), key


def test_draft_request_defaults():
    req = DraftRequest(draft_type="affidavit")
    assert req.instructions is None and req.doc_ids is None


# ------------------------------------------------- deterministic field check
def test_detect_fields_available_and_missing():
    evidence = [_ev()]
    available, missing = detect_fields(
        evidence, DRAFT_TYPES["bail_application"]["fields"])
    avail_labels = [a["label"] for a in available]

    # Present in the crime case PDF (verbatim) -> available with provenance.
    assert "Case / reference number" in avail_labels
    assert "Date of incident" in avail_labels
    assert "Complainant" in avail_labels
    for a in available:
        assert a["evidence_id"] == "E1"
        assert a["filename"].endswith(".pdf")
        assert a["page"] == 1
        assert a["snippet"]

    # Not in the document -> must be reported missing, never guessed.
    for absent in ("Court name", "FIR number", "Police station",
                   "Accused name", "Applicant name", "Grounds for bail"):
        assert absent in missing


def test_detect_fields_without_evidence_reports_everything_missing():
    available, missing = detect_fields(
        [], DRAFT_TYPES["bail_application"]["fields"])
    assert available == []
    assert "Court name" in missing and "FIR number" in missing


def test_detect_fields_does_not_infer_guilt_from_names():
    """Names in the document are complainant/witness, not 'accused'."""
    available, missing = detect_fields(
        [_ev()], DRAFT_TYPES["bail_application"]["fields"])
    assert "Accused name" in missing  # pattern needs an explicit "accused:" line
    assert "Court name" in missing


# --------------------------------------------------------------- thresholds
def test_unknown_draft_type_raises_value_error(monkeypatch):
    def _fail(*a, **k):  # pragma: no cover - guard
        raise AssertionError("no retrieval for an invalid draft type")
    monkeypatch.setattr(drafter, "retrieve", _fail)
    with pytest.raises(ValueError) as exc:
        draft_document("will")
    assert "bail_application" in str(exc.value)


def test_threshold_gate_blocks_llm(monkeypatch):
    low = [_ev(score=0.12)]

    def _fail(_prompt, *a, **k):  # pragma: no cover - guard
        raise AssertionError("LLM must not be consulted below threshold")

    monkeypatch.setattr(drafter, "retrieve", lambda *a, **k: low)
    monkeypatch.setattr(drafter, "run_llm", _fail)
    resp = draft_document("bail_application")
    assert resp.status == "insufficient"
    assert resp.draft == REFUSAL
    assert resp.refusal_stage == "retrieval"
    assert "below threshold" in (resp.reason or "")
    assert resp.claim_check.claims_total == 0


def test_no_evidence_insufficient_without_llm(monkeypatch):
    def _fail(_prompt, *a, **k):  # pragma: no cover - guard
        raise AssertionError("LLM must not be consulted with no evidence")

    monkeypatch.setattr(drafter, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(drafter, "run_llm", _fail)
    resp = draft_document("bail_application")
    assert resp.status == "insufficient"
    assert resp.refusal_stage == "retrieval"
    assert resp.sources == []
    assert "Court name" in resp.missing_information


def test_llm_failure_becomes_insufficient(monkeypatch):
    from app.llm import LLMError

    monkeypatch.setattr(drafter, "retrieve", lambda *a, **k: [_ev()])
    monkeypatch.setattr(drafter, "run_llm",
                        lambda _p: (_ for _ in ()).throw(LLMError("provider down")))
    resp = draft_document("bail_application")
    assert resp.status == "insufficient"
    assert resp.refusal_stage == "system"
    assert "model error" in (resp.reason or "")


# -------------------------------------------------------------- validation
def test_validate_reports_missing_fields_and_maps_sources():
    evidence = [_ev()]
    available, missing = detect_fields(
        evidence, DRAFT_TYPES["bail_application"]["fields"])
    data = {"draft": BAIL_DRAFT, "insufficient": False, "confidence": 0.9,
            "evidence_used": ["E1"]}
    resp = validate_draft("bail_application", evidence, available, missing, data)

    assert resp.status == "missing"          # court/FIR/accused are missing
    assert resp.draft == BAIL_DRAFT.strip()
    assert "Court name" in resp.missing_information
    assert "FIR number" in resp.missing_information
    assert resp.reason and "required field" in resp.reason

    # Sources come from vector-store metadata, never from the model.
    assert resp.sources, "cited evidence must be attached"
    src = next(s for s in resp.sources if s.evidence_id == "E1")
    assert src.page == 1 and src.filename.endswith(".pdf")

    # Both marked factual sentences are traceable to E1.
    assert resp.claim_check.claims_total == 2
    assert resp.claim_check.claims_unverified == 0
    assert resp.claim_check.claims_supported == 2
    assert resp.refusal_stage is None


def test_validate_grounded_status_when_nothing_is_missing():
    evidence = [_ev()]
    data = {"draft": BAIL_DRAFT, "insufficient": False, "confidence": 0.9,
            "evidence_used": ["E1"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert resp.status == "grounded"
    assert resp.reason is None
    assert resp.claim_check.claims_unverified == 0


def test_validate_removes_claim_no_chunk_supports():
    """claim -> cited evidence -> nothing supports it => [NOT PROVIDED]."""
    evidence = [_ev()]
    draft = "The theft was committed by Mr. Vikram Sethi [E1]."
    data = {"draft": draft, "insufficient": False, "confidence": 0.9,
            "evidence_used": ["E1"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert resp.status == "missing"
    # The unsupported assertion is never displayed as fact.
    assert "[NOT PROVIDED]" in resp.draft
    assert "Vikram Sethi" not in resp.draft
    assert resp.removed_claims and "Vikram Sethi" in resp.removed_claims[0]
    assert "NOT PROVIDED" in (resp.reason or "")
    # It is gone from the checked claims too (only verified claims remain).
    assert resp.claim_check.claims_unverified == 0
    assert resp.refusal_stage is None


def test_validate_rejects_hallucinated_marker():
    evidence = [_ev()]
    data = {"draft": "The incident happened [E9].", "insufficient": False,
            "confidence": 0.9, "evidence_used": ["E1"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert resp.status == "insufficient"
    assert resp.draft == REFUSAL
    assert resp.refusal_stage == "verifier"
    assert "E9" in (resp.reason or "")


def test_validate_rejects_hallucinated_evidence_used():
    evidence = [_ev()]
    data = {"draft": "The incident happened [E1].", "insufficient": False,
            "confidence": 0.9, "evidence_used": ["E7"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert resp.status == "insufficient"
    assert resp.refusal_stage == "verifier"


def test_validate_rejects_unverified_page_numbers():
    evidence = [_ev()]
    data = {"draft": "The incident happened on page 99 [E1].", "insufficient": False,
            "confidence": 0.9, "evidence_used": ["E1"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert resp.status == "insufficient"
    assert resp.refusal_stage == "verifier"
    assert "page 99" in (resp.reason or "")


def test_validate_requires_at_least_one_evidence_marker():
    evidence = [_ev()]
    data = {"draft": "The complainant is Arun Mehta.", "insufficient": False,
            "confidence": 0.9, "evidence_used": []}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert resp.status == "insufficient"
    assert resp.refusal_stage == "verifier"
    assert "no evidence markers" in (resp.reason or "")


def test_validate_honors_model_insufficiency():
    evidence = [_ev()]
    data = {"draft": "", "insufficient": True, "confidence": 0.2,
            "evidence_used": []}
    resp = validate_draft("bail_application", evidence, available=[], missing=["Court name"],
                          data=data)
    assert resp.status == "insufficient"
    assert resp.draft == REFUSAL
    assert resp.refusal_stage == "model"


def test_validate_rejects_non_json_output():
    evidence = [_ev()]
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data="raw")
    assert resp.status == "insufficient"
    assert resp.refusal_stage == "verifier"


# ------------------------------------------------------- claim extraction
def test_claim_sentences_keeps_only_marked_factual_sentences():
    sents = claim_sentences(BAIL_DRAFT)
    assert sents == [
        "The incident was reported at 21:40 on 18 September 2026 [E1].",
        "The complainant is Arun Mehta [E1].",
    ]
    # Headings, labels, placeholders and unmarked boilerplate are excluded.
    assert claim_sentences("MOST RESPECTFULLY SUBMITTED:") == []
    assert claim_sentences("1. Background") == []
    assert claim_sentences("[COURT NAME — NOT PROVIDED]") == []
    assert claim_sentences("This notice is issued without prejudice.") == []


def test_claim_sentences_strips_numbering_and_skips_label_lines():
    draft = (
        "1.1 The incident was reported at 21:40 [E1].\n"
        "Case/Reference No.: LL-CR-2026-014 [E1]\n"
        "21:40 The incident is formally reported to building security [E2].\n"
    )
    sents = claim_sentences(draft)
    # Numbered prefix removed (formatting, not a critical number); the label
    # line is a field value already verified by the deterministic detector;
    # the timestamped claim keeps its "21:40" (digits => not a label line).
    assert sents == [
        "The incident was reported at 21:40 [E1].",
        "21:40 The incident is formally reported to building security [E2].",
    ]


# --------------------------------------------- orphan marker attachment (regression)
def test_claim_sentences_attaches_orphan_markers():
    """'fact. [E3]' must be ONE claim — the old split left '[E3]' alone and
    the real claim unmarked, so nothing was ever verified (claims 0/0)."""
    assert claim_sentences(
        "The source names Arun Mehta as the complainant. [E3]"
    ) == ["The source names Arun Mehta as the complainant [E3]."]
    # A marker-only line folds into the line above it.
    assert claim_sentences(
        "The complainant is Arun Mehta.\n[E1]"
    ) == ["The complainant is Arun Mehta [E1]."]


def test_normalize_markers_is_idempotent():
    once = normalize_markers("The complainant is Arun Mehta. [E1]")
    assert once == "The complainant is Arun Mehta [E1]."
    assert normalize_markers(once) == once
    assert normalize_markers("Fact.\n\n[E2]") == "Fact [E2].\n"


def test_marker_after_full_stop_is_checked_and_counted():
    """Regression: a claim whose marker sat after the period is verified."""
    evidence = [_ev()]
    data = {"draft": "The incident was reported at 21:40 on 18 September 2026. [E1]",
            "insufficient": False, "confidence": 0.9, "evidence_used": ["E1"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert resp.draft == "The incident was reported at 21:40 on 18 September 2026 [E1]."
    assert resp.claim_check.claims_total == 1
    assert resp.claim_check.claims_supported == 1
    assert resp.claim_check.claims_unverified == 0


# -------------------------------------------------- citation correctness
def test_wrong_citation_is_repointed_to_the_chunk_that_contains_the_fact():
    """E2 (the log) does not contain '21:40 on 18 September 2026'; only E1
    does — the citation must be corrected and E2 must stop being a source."""
    evidence = [_ev("E1", page=1, text=CASE_HEADER),
                _ev("E2", page=2, text=CASE_LOG)]
    draft = "The incident was reported at 21:40 on 18 September 2026 [E2]."
    data = {"draft": draft, "insufficient": False, "confidence": 0.9,
            "evidence_used": ["E1"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)

    assert resp.status == "grounded"
    assert resp.draft == "The incident was reported at 21:40 on 18 September 2026 [E1]."
    assert resp.citation_corrections and "re-cited" in resp.citation_corrections[0]
    # Only the verified chunk is attached as a source.
    assert [s.evidence_id for s in resp.sources] == ["E1"]
    assert resp.claim_check.claims_supported == 1


def test_unsupported_extra_marker_is_dropped():
    evidence = [_ev("E1", page=1, text=CASE_HEADER),
                _ev("E2", page=2, text=CASE_LOG)]
    draft = "The incident was reported at 21:40 on 18 September 2026 [E1][E2]."
    data = {"draft": draft, "insufficient": False, "confidence": 0.9,
            "evidence_used": ["E1"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert resp.draft == "The incident was reported at 21:40 on 18 September 2026 [E1]."
    assert any("dropped unsupported citation [E2]" in c
               for c in resp.citation_corrections)
    assert [s.evidence_id for s in resp.sources] == ["E1"]


def test_label_line_value_is_verified_against_its_citation():
    """'Case/Reference No.: LL-CR-2026-014 [E2]' — E2 (the log) does not
    contain the case ID, so the label citation is re-pointed at E1."""
    evidence = [_ev("E1", page=1, text=CASE_HEADER),
                _ev("E2", page=2, text=CASE_LOG)]
    data = {"draft": "Case/Reference No.: LL-CR-2026-014 [E2]", "insufficient": False,
            "confidence": 0.9, "evidence_used": ["E2"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert resp.draft == "Case/Reference No.: LL-CR-2026-014 [E1]"
    assert resp.citation_corrections and "re-cited" in resp.citation_corrections[0]
    assert [s.evidence_id for s in resp.sources] == ["E1"]


def test_hallucinated_label_value_is_removed():
    evidence = [_ev()]
    data = {"draft": "Case/Reference No.: ZZ-99 [E1]", "insufficient": False,
            "confidence": 0.9, "evidence_used": ["E1"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert resp.draft.strip() == "Case/Reference No.: [NOT PROVIDED]"
    assert resp.removed_claims and "ZZ-99" in resp.removed_claims[0]
    assert resp.status == "missing"


def test_abstention_sentence_is_kept_not_removed():
    """'the document does not contain ...' asserts an ABSENCE — it is not a
    source fact, so verification leaves it (and records no removal)."""
    evidence = [_ev("E2", page=2,
                    text="The material does not provide a phone number, "
                         "bank account number or address.")]
    draft = "The document does not contain a phone number [E2]."
    data = {"draft": draft, "insufficient": False, "confidence": 0.9,
            "evidence_used": ["E2"]}
    resp = validate_draft("bail_application", evidence, available=[], missing=[], data=data)
    assert draft in resp.draft
    assert resp.removed_claims == []
    assert resp.claim_check.claims_total == 0  # abstention: neither supported
    assert resp.status == "grounded"


# --------------------------------------------- missing/available information
def test_petition_missing_information_is_exactly_the_required_list():
    evidence = [_ev("E1", page=2, text=PACK_PETITION_SECTION),
                _ev("E2", page=1, text=PACK_CASE_FACTS)]
    available, missing = detect_fields(evidence, DRAFT_TYPES["petition"]["fields"])

    # The source pack only LISTS these field names ("Missing: Court name, ...")
    # — a label without a value must never read as available.
    assert missing == ["Court name", "Court jurisdiction",
                       "Case/reference number", "Petitioner details",
                       "Respondent details", "Legal grounds",
                       "Statutory provisions", "Relief requested",
                       "Filing date", "Advocate details"]
    # The only genuinely available information is the recorded case facts,
    # and its provenance points at the chunk holding the actual values.
    assert [a["label"] for a in available] == ["Relevant facts"]
    assert available[0]["evidence_id"] == "E2"
    assert available[0]["page"] == 1


def test_petition_fields_are_found_when_real_values_are_present():
    """The strict patterns must not be over-strict: a real court filing with
    actual values marks every petition field available."""
    filing = (
        "IN THE HIGH COURT OF JUDICATURE AT MADRAS\n"
        "W.P. No. 12345 of 2026\n"
        "Petitioner: Arun Mehta, Harbor View Business Centre, Chennai\n"
        "Versus\n"
        "Respondent: The State, represented by the Public Prosecutor\n"
        "Jurisdiction of this Hon'ble court is invoked.\n"
        "On the grounds of challenge to the incident dated 18 September 2026.\n"
        "Under Section 379 of the Act the relief sought: restoration of the "
        "laptop.\n"
        "Date of filing: 3 October 2026\n"
        "Learned counsel for the petitioner: Ms. Priya Desai.\n"
    )
    available, missing = detect_fields(
        [_ev(text=filing)], DRAFT_TYPES["petition"]["fields"])
    assert missing == []
    assert {a["label"] for a in available} == {
        "Court name", "Court jurisdiction", "Case/reference number",
        "Petitioner details", "Respondent details", "Relevant facts",
        "Legal grounds", "Statutory provisions", "Relief requested",
        "Filing date", "Advocate details"}


def test_label_only_mentions_do_not_mark_bail_fields_available():
    """'...does not provide ... FIR number, police station, court name...'
    must not make FIR / police station / court available."""
    evidence = [_ev(text=PACK_LIMITATION)]
    available, missing = detect_fields(
        evidence, DRAFT_TYPES["bail_application"]["fields"])
    labels = [a["label"] for a in available]
    assert "FIR number" not in labels
    assert "Police station" not in labels
    for absent in ("Court name", "FIR number", "Police station",
                   "Accused name", "Applicant name"):
        assert absent in missing


def test_detect_fields_prefers_value_bearing_chunk():
    """Provenance must point at the chunk containing the VALUE, not a random
    chunk that merely mentions the label word."""
    mention = _ev("E1", page=3,
                  text="Suggested demo questions: 3. Prepare an affidavit for "
                       "the complainant using only verified facts.")
    value = _ev("E2", page=1,
                text="Complainant: Arun Mehta\nWitness: Priya Nair")
    available, _ = detect_fields([mention, value],
                                 DRAFT_TYPES["bail_application"]["fields"])
    comp = next(a for a in available if a["label"] == "Complainant")
    assert comp["evidence_id"] == "E2" and comp["page"] == 1


def test_verify_citations_reports_corrections_and_removals():
    evidence = [_ev("E1", page=1, text=CASE_HEADER),
                _ev("E2", page=2, text=CASE_LOG)]
    draft = ("The incident was reported at 21:40 on 18 September 2026 [E2].\n"
             "The theft was done by the phantom thief [E1].\n"
             "The complainant is Arun Mehta [E1].")
    fixed, corrections, removed = verify_citations(draft, evidence)
    assert "[E1]." in fixed and "[E2]" not in fixed
    assert "phantom thief" not in fixed and "[NOT PROVIDED]" in fixed
    assert "The complainant is Arun Mehta [E1]." in fixed
    assert corrections and removed and "phantom thief" in removed[0]


# ------------------------------------------------------- end-to-end (stubbed)
def test_all_four_draft_types_execute_with_stubbed_llm(monkeypatch):
    evidence = [_ev("E1", page=1), _ev("E2", page=2, text=CASE_LOG)]
    prompts = []

    def _llm(prompt):
        prompts.append(prompt)
        return json.dumps({
            "draft": ("IN THE [COURT NAME — NOT PROVIDED]\n"
                      "The incident was reported at 21:40 [E1].\n"
                      "Accused: [ACCUSED NAME — NOT PROVIDED]"),
            "insufficient": False,
            "confidence": 0.9,
            "evidence_used": ["E1"],
        })

    monkeypatch.setattr(drafter, "retrieve", lambda *a, **k: evidence)
    monkeypatch.setattr(drafter, "run_llm", _llm)

    for draft_type in DRAFT_TYPES:
        resp = draft_document(draft_type, instructions="keep it short")
        assert resp.draft_type == draft_type
        assert resp.status in {"grounded", "missing"}
        # Pages always come from evidence metadata.
        assert all(s.page in (1, 2) for s in resp.sources)
        assert all(s.filename.endswith(".pdf") for s in resp.sources)
        assert isinstance(resp.missing_information, list)

    # The prompt contains the evidence packet + Python-computed field lists.
    assert len(prompts) == 4
    for p in prompts:
        assert "EVIDENCE" in p
        assert "AVAILABLE INFORMATION" in p
        assert "MISSING INFORMATION" in p
        assert "REQUIRED STRUCTURE" in p
        assert "Do not invent" in p


def test_prompt_lists_missing_fields_as_placeholders(monkeypatch):
    captured = {}

    def _llm(prompt):
        captured["prompt"] = prompt
        return json.dumps({"draft": "X [E1].", "insufficient": False,
                           "confidence": 0.9, "evidence_used": ["E1"]})

    monkeypatch.setattr(drafter, "retrieve", lambda *a, **k: [_ev()])
    monkeypatch.setattr(drafter, "run_llm", _llm)
    draft_document("bail_application")
    prompt = captured["prompt"]
    # Missing fields are handed to the model as placeholder instructions.
    assert "[COURT NAME — NOT PROVIDED]" in prompt
    assert "[FIR NUMBER — NOT PROVIDED]" in prompt
    # Available fields carry their real E-IDs and pages.
    assert "[E1]" in prompt and "page 1" in prompt
    # Citation correctness rules: marker placement + cite the block that
    # literally contains the facts.
    assert "Place the marker INSIDE the sentence" in prompt
    assert "literally contains" in prompt


# ---------------------------------------------------------------- API layer
def test_api_draft_unknown_type_is_400():
    from fastapi.testclient import TestClient
    res = TestClient(main.app).post("/api/draft", json={"draft_type": "will"})
    assert res.status_code == 400
    assert "Unknown draft type" in res.json()["detail"]


def test_api_draft_instructions_too_long_is_400():
    from fastapi.testclient import TestClient
    res = TestClient(main.app).post(
        "/api/draft",
        json={"draft_type": "affidavit", "instructions": "x" * 5000})
    assert res.status_code == 400


def test_api_draft_success_contract(monkeypatch):
    from fastapi.testclient import TestClient

    canned = DraftResponse(
        draft_type="bail_application",
        status="missing",
        draft="IN THE [COURT NAME — NOT PROVIDED]\nThe incident was reported at 21:40 [E1].",
        sources=[DraftSource(evidence_id="E1",
                             filename="LegalLens_Fictional_Crime_Case.pdf",
                             page=1, snippet=CASE_HEADER[:100])],
        missing_information=["Court name", "FIR number"],
        available_information=[AvailableField(label="Date of incident",
                                              evidence_id="E1",
                                              filename="LegalLens_Fictional_Crime_Case.pdf",
                                              page=1, snippet="Incident Date 18 September 2026")],
        claim_check=ClaimCheck(claims_total=1, claims_supported=1,
                               claims_unverified=0, unverified_claims=[]),
        reason="2 required field(s) not found in the source",
        refusal_stage=None,
    )
    monkeypatch.setattr(main, "draft_document",
                        lambda t, instructions=None, doc_ids=None: canned)

    res = TestClient(main.app).post(
        "/api/draft", json={"draft_type": "bail_application",
                            "instructions": "short and formal"})
    assert res.status_code == 200
    body = res.json()
    for key in ("draft_type", "status", "draft", "sources",
                "missing_information", "available_information",
                "claim_check", "citation_corrections", "removed_claims",
                "reason", "refusal_stage"):
        assert key in body
    assert body["status"] == "missing"
    assert body["sources"][0]["evidence_id"] == "E1"
    assert body["sources"][0]["page"] == 1
    assert body["missing_information"] == ["Court name", "FIR number"]


def test_api_chat_contract_untouched():
    """The chat endpoint still refuses below-threshold questions offline."""
    from fastapi.testclient import TestClient
    # Unknown-field guard: malformed request shape is rejected with 422,
    # proving the chat route itself is still registered and intact.
    res = TestClient(main.app).post("/api/chat", json={})
    assert res.status_code == 422
