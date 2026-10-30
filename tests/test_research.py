"""Legal Research tests — EVIDENCE FIRST → VERIFY → REASON → ANSWER.

All offline: the LLM is stubbed (monkeypatched), the live India Code search
is blocked, statute evidence comes from the real bundled OFFICIAL snapshot
(app/research/legal_sources.json, fetched verbatim from India Code), and case
evidence uses verbatim phrases from LegalLens_Drafting_Source_Pack.pdf.

Grounding rules under test:
  * legal issue is computed by Python (cautious wording), never the model
  * authority/section/url/tier come from the registry, never the model
  * law entries must cite retrieved LEGAL AUTHORITY that supports them
  * case connections must hold on BOTH sides (law + case fact)
  * judgments need a judgment source, else NO_VERIFIED_PRECEDENT_FOUND
  * guilt conclusions and uncited case statements are removed (Q5 / Q6)
  * exact refusal string + refusal_stage for retrieval/model/verifier failures
  * statuses are exact spec strings — never fabricated confidence numbers
  * API contract + evaluation categories A-E computed from actual (stubbed)
    runs
  * the chat/drafting pipelines are untouched (the full suite must pass)
"""
import json
import re

import pytest

from app import main
from app.answer_generator import REFUSAL
from app.config import REFUSAL_THRESHOLD
from app.evaluation import research_eval
from app.evaluation.datasets import RESEARCH_QUESTIONS
from app.llm import LLMError
from app.models import Evidence
from app.research import legal_source_retriever as lsr
from app.research import research_service as rs
from app.research.legal_source_retriever import (
    GatherResult, classify_evidence, content_kind, load_registry,
    registry_by_doc_id, search_live_sources, snapshot_doc_id,
    source_filename)
from app.research.research_prompt import (RESEARCH_SYSTEM_PROMPT,
                                          RESEARCH_TYPES, build_prompt,
                                          identify_legal_issue,
                                          legal_search_terms)
from app.research.source_verifier import (NO_LEGAL_AUTHORITY, NO_PRECEDENT,
                                          extract_ids,
                                          strip_unsafe_statements)

# Verbatim phrases from the indexed LegalLens_Drafting_Source_Pack.pdf.
PACK_FACTS = (
    "Case ID: LL-CR-2026-014 Matter: Theft of a laptop from a private office "
    "Reported time: 21:40 Complainant: Arun Mehta Security guard: Rohan Das "
    "21:12 Rohan Das reports seeing a person carrying a black backpack near "
    "the third-floor lift. 21:25 Arun Mehta notices that the laptop is "
    "missing. 21:40 The incident is reported."
)
PACK_NOTE = (
    "Investigation limitation: The material does not identify the person "
    "seen, does not establish that any named individual took the laptop, "
    "and does not provide a phone number, bank account number, address, "
    "FIR number, police station, court name, legal sections, or judicial "
    "precedents."
)


# ------------------------------------------------------------------ helpers
def _entry(section: str = "303") -> dict:
    return next(e for e in load_registry()
                if e["section"] == section and "Nyaya" in e["act"])


def _covers(entry: dict) -> str:
    """Verbatim subsection text pulled from the official snapshot itself."""
    match = re.search(r"\(1\).*?commit theft\.", entry["text"], re.S)
    if match:
        return re.sub(r"\s+", " ", match.group(0)).strip()
    return re.sub(r"\s+", " ", entry["text"][:220]).strip()


def _case_ev(eid: str, page: int, text: str, score: float = 0.6) -> Evidence:
    return Evidence(id=eid, chunk_id=f"pack:{eid}", doc_id="pack",
                    filename="LegalLens_Drafting_Source_Pack.pdf",
                    page=page, text=text, score=score)


def _statute_ev(eid: str, entry: dict, score: float = 0.55) -> Evidence:
    return Evidence(id=eid, chunk_id=f"{snapshot_doc_id(entry)}:0",
                    doc_id=snapshot_doc_id(entry),
                    filename=source_filename(entry["act"], entry["section"]),
                    page=int(entry["page"]), text=entry["text"][:900],
                    score=score)


def _bundle(section: str = "303"):
    """(evidence, classes, meta) — case E1/E2 + official statute E3."""
    entry = _entry(section)
    evidence = [_case_ev("E1", 1, PACK_FACTS),
                _case_ev("E2", 2, PACK_NOTE),
                _statute_ev("E3", entry)]
    meta = registry_by_doc_id()
    return evidence, classify_evidence(evidence, meta), meta


def _gather(evidence, meta, case_ids=("E1", "E2"), live_error=None) -> GatherResult:
    case = [e for e in evidence if e.id in case_ids]
    legal = [e for e in evidence if e.id not in case_ids]
    return GatherResult(case_evidence=case, legal_evidence=legal,
                        evidence=evidence, meta_by_doc_id=meta,
                        live_error=live_error)


def _fail(*_args, **_kwargs):
    raise AssertionError("the LLM must not be called")


def _llm(data):
    return lambda _prompt, **_kwargs: json.dumps(data)


def _good_data(entry: dict | None = None, **overrides) -> dict:
    entry = entry or _entry()
    provision = f"Section {entry['section']} of {entry['act']}"
    data = {
        "answer": (
            "RESEARCH ANSWER\n"
            "The incident was reported at 21:40 for a theft of a laptop from "
            "a private office [E1].\n\n"
            "CONNECTION TO THIS CASE\n"
            "The material records that Rohan Das reports seeing a person "
            "carrying a black backpack near the third-floor lift at "
            "approximately 21:12 [E1]."),
        "insufficient": False,
        "relevant_law": [{"provision": provision, "covers": _covers(entry),
                          "cites": ["E3"]}],
        "case_connections": [{
            "provision": provision, "what_it_covers": _covers(entry),
            "case_fact": ("Rohan Das reports seeing a person carrying a black "
                          "backpack near the third-floor lift at approximately 21:12"),
            "cites": ["E1", "E3"],
            "sufficiency": ("The available material does not establish that any "
                            "named individual took the laptop."),
            "missing": "missing: intent and identification of the person seen",
        }],
        "judgments": [],
        "limitations": [],
    }
    data.update(overrides)
    return data


# ------------------------------------------------------------------ config
def test_research_types_and_statuses_are_exact_spec_strings():
    assert RESEARCH_TYPES == ("general", "law", "judgment", "precedent")
    assert rs.STATUS_GROUNDED == "GROUNDED_RESEARCH"
    assert rs.STATUS_LIMITED == "RESEARCH_WITH_LIMITATIONS"
    assert rs.STATUS_NO_PRECEDENT == "NO_VERIFIED_PRECEDENT_FOUND"
    assert rs.STATUS_UNVERIFIED == "UNVERIFIED_RESEARCH"
    # No fabricated confidence anywhere in the response model.
    assert "confidence" not in rs.ResearchResponse.model_fields
    assert "score" not in rs.ResearchResponse.model_fields


def test_system_prompt_contains_required_grounding_rules():
    prompt = RESEARCH_SYSTEM_PROMPT
    assert "grounded legal research assistant" in prompt
    assert "Never invent a statute" in prompt
    assert "judgment, case citation, court, URL" in prompt
    assert REFUSAL in prompt
    assert NO_LEGAL_AUTHORITY in prompt
    assert "FALSE-PREMISE RULE" in prompt
    assert "NEVER state or imply that any person is guilty" in prompt
    for key in ("answer", "relevant_law", "case_connections", "judgments",
                "limitations", "insufficient"):
        assert f'"{key}"' in prompt


# --------------------------------------------------- legal issue (Python)
def test_legal_issue_is_deterministic_and_cautious():
    issue = identify_legal_issue(
        "What law may be relevant to the alleged theft in this case?")
    assert issue.startswith("Potentially relevant legal provisions concerning theft")
    # never converts an allegation into an established fact
    assert "guilty" not in issue.lower() and "established" not in issue.lower()

    fallback = identify_legal_issue("Explain this document to me.")
    assert fallback.startswith("The issue concerns")
    assert "Explain this document to me." in fallback

    issue6 = identify_legal_issue(
        "Give me the legal sections that prove the accused is guilty.")
    assert issue6.startswith("Potentially relevant legal provisions")


def test_legal_search_terms_derive_from_topic():
    assert "theft" in legal_search_terms(
        "What law applies to theft?")
    assert legal_search_terms("???")  # never empty
    # case_context participates in topic detection
    assert "bail" in legal_search_terms("What should we do next?",
                                        case_context="bail application")


def test_prompt_assembly_is_evidence_packet_only():
    evidence, classes, _ = _bundle()
    from app.answer_generator import build_packet
    prompt = build_prompt(build_packet(evidence), classes, evidence,
                          "Potentially relevant legal provisions concerning theft.",
                          "law", "What law applies to theft?")
    assert "SOURCE CLASSIFICATION (computed by the system — authoritative)" in prompt
    assert "LEGAL AUTHORITY" in prompt and "CASE MATERIAL" in prompt
    assert "tier 1" in prompt
    assert "RESEARCH TYPE: law" in prompt
    assert "QUESTION: What law applies to theft?" in prompt
    assert "Respond with the JSON object only." in prompt
    # the packet itself is present, built by answer_generator.build_packet
    assert "LegalLens_Drafting_Source_Pack.pdf" in prompt
    assert "[E1]" in prompt


# ------------------------------------------------------------ source layer
def test_bundled_registry_is_official_and_verbatim():
    """Anti-fabrication: every bundled source is real India Code metadata."""
    registry = load_registry()
    assert len(registry) >= 10
    for entry in registry:
        assert entry["tier"] == 1
        assert entry["url"].startswith("https://indiacode.gov.in/handle/")
        assert entry["source_type"] == "statute"
        assert entry["act"] and entry["section"]
        assert len(entry["text"]) > 80
        assert not entry["repealed"]


def test_content_kind_classification_is_conservative():
    assert content_kind("Section 378 of the Indian Penal Code, 1860. Whoever "
                        "commits theft shall be punished.") == "statute"
    assert content_kind("AIR 1973 SC 514 The appellant was held guilty.") == "judgment"
    assert content_kind("(2019) 5 SCC 123 held by the Supreme Court of India.") == "judgment"
    assert content_kind(PACK_FACTS) == "case_evidence"   # no Section N -> case
    assert content_kind(PACK_NOTE) == "case_evidence"


def test_live_search_disabled_env_never_touches_network(monkeypatch):
    monkeypatch.setenv("LEGALLENS_LIVE_SOURCES", "0")
    records, error = search_live_sources("theft")
    assert records == [] and error is None


def test_live_search_failure_degrades_to_snapshot(monkeypatch):
    def _raise(*_a, **_k):
        raise OSError("offline")
    monkeypatch.setenv("LEGALLENS_LIVE_SOURCES", "1")
    monkeypatch.setattr("urllib.request.urlopen", _raise)
    records, error = search_live_sources("theft", timeout=0.1)
    assert records == []
    assert error and "unavailable" in error


def test_gather_merges_case_and_legal_with_sequential_ids(monkeypatch):
    """Same retriever for case + same store class for law; E-IDs stay unique."""
    entry303, entry305 = _entry("303"), _entry("305")
    illus = _statute_ev("", entry303).model_copy(
        update={"id": "", "chunk_id": f"{snapshot_doc_id(entry303)}:6",
                "text": "(p) illustration text about theft"})
    opening = _statute_ev("", entry303)
    ev305 = _statute_ev("", entry305)
    live318 = {"doc_id": "live:abc", "title": "BNS — Section 318. Cheating.",
               "source_type": "statute", "authority": "Government of India",
               "act": "The Bharatiya Nyaya Sanhita, 2023", "section": "318",
               "tier": 1, "url": "https://indiacode.gov.in/handle/123456789/x",
               "page": 93, "repealed": False, "text": "Cheating text body.",
               "fetched_live": True}
    live303 = dict(live318, doc_id="live:dup", section="303", page=86)

    monkeypatch.setattr(lsr, "retrieve",
                        lambda *_a, **_k: [_case_ev("E1", 1, PACK_FACTS)])
    monkeypatch.setattr(lsr, "_snapshot_evidence",
                        lambda _q: [(illus, entry303), (opening, entry303),
                                    (ev305, entry305)])
    monkeypatch.setattr(lsr, "search_live_sources",
                        lambda _q, **_k: ([live318, live303], None))

    result = lsr.gather_evidence("theft")
    assert [e.id for e in result.case_evidence] == ["E1"]
    legal_ids = [e.id for e in result.legal_evidence]
    assert legal_ids[0] == "E2" and len(set(legal_ids)) == len(legal_ids)
    # one chunk per section, preferring the section-OPENING chunk over the
    # illustration; live dup of s303 deduped against the snapshot
    chunk_ids = [e.chunk_id for e in result.legal_evidence]
    assert f"{snapshot_doc_id(entry303)}:0" in chunk_ids
    assert not any(cid.endswith(":6") for cid in chunk_ids)
    sections = [str(result.meta_by_doc_id[e.doc_id].get("section"))
                for e in result.legal_evidence]
    assert sections.count("303") == 1 and "305" in sections and "318" in sections
    assert result.live_error is None

    classes = classify_evidence(result.evidence, result.meta_by_doc_id)
    assert classes["E1"]["kind"] == "case_evidence"
    assert all(classes[eid]["kind"] == "statute" for eid in legal_ids)
    assert classes["E2"]["url"].startswith("https://indiacode.gov.in/")


def test_gather_keeps_snapshot_when_live_search_fails(monkeypatch):
    entry303 = _entry("303")
    monkeypatch.setattr(lsr, "retrieve",
                        lambda *_a, **_k: [_case_ev("E1", 1, PACK_FACTS)])
    monkeypatch.setattr(lsr, "_snapshot_evidence",
                        lambda _q: [(_statute_ev("", entry303), entry303)])
    monkeypatch.setattr(lsr, "search_live_sources",
                        lambda _q, **_k: ([], "live India Code search unavailable (URLError)"))
    result = lsr.gather_evidence("theft")
    assert len(result.legal_evidence) == 1
    assert result.live_error and "unavailable" in result.live_error


# ---------------------------------------------------- workflow: refusals
def test_no_case_evidence_is_refusal_without_llm(monkeypatch):
    evidence, _, meta = _bundle()
    monkeypatch.setattr(rs, "gather_evidence",
                        lambda *_a, **_k: _gather(evidence, meta, case_ids=()))
    monkeypatch.setattr(rs, "run_llm", _fail)
    resp = rs.research_document("What law applies to theft?")
    assert resp.status == rs.STATUS_UNVERIFIED
    assert resp.answer == REFUSAL
    assert resp.refusal_stage == "retrieval"
    assert resp.sources == [] and resp.case_connections == []


def test_threshold_gate_blocks_llm(monkeypatch):
    evidence, _, meta = _bundle()
    low = [evidence[0].model_copy(
        update={"score": max(0.0, REFUSAL_THRESHOLD - 0.2)})]
    g = GatherResult(case_evidence=low, legal_evidence=[], evidence=low,
                     meta_by_doc_id=meta, live_error=None)
    monkeypatch.setattr(rs, "gather_evidence", lambda *_a, **_k: g)
    monkeypatch.setattr(rs, "run_llm", _fail)
    resp = rs.research_document("What law applies to theft?")
    assert resp.status == rs.STATUS_UNVERIFIED
    assert resp.answer == REFUSAL and resp.refusal_stage == "retrieval"
    assert "below threshold" in (resp.reason or "")


def test_llm_failure_is_refusal(monkeypatch):
    evidence, _, meta = _bundle()
    monkeypatch.setattr(rs, "gather_evidence",
                        lambda *_a, **_k: _gather(evidence, meta))

    def _boom(_prompt, **_kwargs):
        raise LLMError("provider down")
    monkeypatch.setattr(rs, "run_llm", _boom)
    resp = rs.research_document("What law applies to theft?")
    assert resp.status == rs.STATUS_UNVERIFIED
    assert resp.answer == REFUSAL and resp.refusal_stage == "system"
    assert "provider down" in (resp.reason or "")


def test_model_insufficient_is_refusal(monkeypatch):
    evidence, _, meta = _bundle()
    monkeypatch.setattr(rs, "gather_evidence",
                        lambda *_a, **_k: _gather(evidence, meta))
    monkeypatch.setattr(rs, "run_llm", _llm({
        "insufficient": True, "answer": REFUSAL, "relevant_law": [],
        "case_connections": [], "judgments": [], "limitations": []}))
    resp = rs.research_document("What law applies to theft?")
    assert resp.status == rs.STATUS_UNVERIFIED
    assert resp.answer == REFUSAL and resp.refusal_stage == "model"


def test_hallucinated_evidence_id_is_refusal(monkeypatch):
    evidence, _, meta = _bundle()
    monkeypatch.setattr(rs, "gather_evidence",
                        lambda *_a, **_k: _gather(evidence, meta))
    monkeypatch.setattr(rs, "run_llm", _llm({
        "insufficient": False,
        "answer": "Theft is defined in section 303 [E9].",
        "relevant_law": [], "case_connections": [], "judgments": [],
        "limitations": []}))
    resp = rs.research_document("What law applies to theft?")
    assert resp.status == rs.STATUS_UNVERIFIED
    assert resp.answer == REFUSAL and resp.refusal_stage == "verifier"
    assert "E9" in (resp.reason or "")


def test_unverified_page_number_is_refusal():
    evidence, classes, _ = _bundle()
    bad_page = max(ev.page for ev in evidence) + 7
    data = _good_data()
    data["answer"] = f"See page {bad_page} for details [E1]."
    resp = rs.validate_research("law", "issue", evidence, classes, data)
    assert resp.status == rs.STATUS_UNVERIFIED
    assert resp.refusal_stage == "verifier"
    assert f"page {bad_page}" in (resp.reason or "")


def test_unknown_research_type_raises_before_any_call(monkeypatch):
    monkeypatch.setattr(rs, "gather_evidence", _fail)
    with pytest.raises(ValueError):
        rs.research_document("q", research_type="will")


# --------------------------------------------------- workflow: happy path
def test_grounded_research_end_to_end(monkeypatch):
    evidence, _, meta = _bundle()
    monkeypatch.setattr(rs, "gather_evidence",
                        lambda *_a, **_k: _gather(evidence, meta))
    monkeypatch.setattr(rs, "run_llm", _llm(_good_data()))
    resp = rs.research_document(
        "What law may be relevant to the alleged theft in this case?",
        research_type="law")
    assert resp.status == rs.STATUS_GROUNDED
    assert resp.legal_issue.startswith("Potentially relevant legal provisions")
    assert resp.answer and REFUSAL not in resp.answer
    # sources: verified flags + registry provenance only
    by_id = {s.evidence_id: s for s in resp.sources}
    assert by_id["E1"].verified and by_id["E3"].verified
    assert by_id["E3"].source_type == "statute"
    assert by_id["E3"].url.startswith("https://indiacode.gov.in/handle/")
    assert by_id["E3"].section == "303"
    assert "Government of India" in by_id["E3"].authority
    assert by_id["E1"].source_type == "case_evidence" and by_id["E1"].url is None
    assert resp.relevant_law and resp.relevant_law[0].verified
    assert resp.case_connections and resp.case_connections[0].verified
    assert resp.judgments == [] and resp.unverified == []
    assert resp.claim_check.claims_total >= 1
    assert resp.claim_check.claims_unverified == 0
    assert resp.refusal_stage is None and resp.reason is None


def test_model_supplied_authority_and_url_are_ignored(monkeypatch):
    """SECURITY: the model cannot mint authority — registry metadata wins."""
    evidence, _, meta = _bundle()
    data = _good_data()
    data["relevant_law"][0]["authority"] = "Supreme Court of India"
    data["relevant_law"][0]["url"] = "https://evil.example/fake-law"
    data["relevant_law"][0]["tier"] = 1
    data["judgments"] = [{"case_name": "Made Up v. State",
                          "citation": "(2099) 1 SCC 1",
                          "court": "Supreme Court of India", "date": "2099",
                          "principle": "anything", "cites": ["E3"]}]
    monkeypatch.setattr(rs, "gather_evidence",
                        lambda *_a, **_k: _gather(evidence, meta))
    monkeypatch.setattr(rs, "run_llm", _llm(data))
    resp = rs.research_document("What law applies?", research_type="law")

    law = resp.relevant_law[0]
    assert law.authority == "Government of India — Ministry of Home Affairs (India Code)" \
        or "Government of India" in law.authority
    assert law.url.startswith("https://indiacode.gov.in/")
    assert "evil.example" not in json.dumps(resp.model_dump())
    # statute block cannot back a judgment: dropped as unverified
    assert resp.judgments == []
    assert any("judgment source" in u.reason for u in resp.unverified)


def test_research_with_limitations_when_no_authority_retrieved(monkeypatch):
    evidence, classes, meta = _bundle()
    data = _good_data(relevant_law=[], case_connections=[])
    data["answer"] = ("The incident was reported at 21:40 for a theft of a "
                      "laptop from a private office [E1].")
    resp = rs.validate_research("law", "issue", evidence, classes, data)
    assert resp.status == rs.STATUS_LIMITED
    assert NO_LEGAL_AUTHORITY in resp.limitations
    assert all(s.source_type == "case_evidence" for s in resp.sources)


# ------------------------------------------------- entry verification
def test_law_entry_must_cite_legal_authority():
    evidence, classes, _ = _bundle()
    data = _good_data()
    # cites the CASE chunk instead of the statute block
    data["relevant_law"][0]["cites"] = ["E1"]
    resp = rs.validate_research("law", "issue", evidence, classes, data)
    assert resp.relevant_law == []
    assert any("legal-authority" in u.reason for u in resp.unverified)
    assert resp.status in (rs.STATUS_LIMITED, rs.STATUS_GROUNDED)


def test_law_entry_with_unsupported_provision_is_dropped():
    evidence, classes, _ = _bundle()
    data = _good_data(relevant_law=[{
        "provision": "Section 420 of the Indian Penal Code, 1860",
        "covers": "punishes cheating with imprisonment of seven years",
        "cites": ["E3"]}])
    data["case_connections"] = []
    resp = rs.validate_research("law", "issue", evidence, classes, data)
    assert resp.relevant_law == []
    assert any("420" in u.claim or "no retrieved legal authority" in u.reason
               for u in resp.unverified)
    assert resp.status == rs.STATUS_LIMITED
    assert NO_LEGAL_AUTHORITY in resp.limitations


def test_case_connection_requires_both_sides():
    evidence, classes, _ = _bundle()
    data = _good_data(case_connections=[{
        "provision": f"Section {_entry()['section']} of {_entry()['act']}",
        "what_it_covers": _covers(_entry()),
        "case_fact": "Kiran Rao was arrested at 22:00 after a confession",
        "cites": ["E1", "E3"],
        "sufficiency": "n/a", "missing": "n/a"}])
    resp = rs.validate_research("law", "issue", evidence, classes, data)
    assert resp.case_connections == []
    assert any("no retrieved case material" in u.reason for u in resp.unverified)
    # law + case still verified elsewhere in the answer
    assert resp.relevant_law and resp.relevant_law[0].verified


def test_judgment_query_without_judgment_source_reports_no_precedent():
    evidence, classes, _ = _bundle()
    data = _good_data(judgments=[{
        "case_name": "Rohan Das v. State",
        "court": "Supreme Court of India",
        "date": "2019",
        "citation": "(2019) 3 SCC 456",
        "principle": "mere opportunity does not prove theft",
        "relationship": "analogous",
        "cites": ["E3"]}])
    resp = rs.validate_research("judgment", "issue", evidence, classes, data)
    assert resp.status == rs.STATUS_NO_PRECEDENT
    assert resp.judgments == []
    assert NO_PRECEDENT in resp.limitations
    assert any("judgment source" in u.reason for u in resp.unverified)


# ----------------------------------------------------- safety guards (Q5/Q6)
def test_q5_guilt_conclusion_is_never_concluded():
    """Spec Q5: 'Can we conclude that Rohan Das committed the theft?'"""
    evidence, classes, _ = _bundle()
    data = _good_data(answer=(
        "Rohan Das committed the theft [E1]. "
        "The material records that Rohan Das reports seeing a person carrying "
        "a black backpack near the third-floor lift at approximately 21:12 [E1]. "
        "The material does not establish that any named individual took the "
        "laptop [E2]."))
    data["relevant_law"] = []
    data["case_connections"] = []
    resp = rs.validate_research("general", "issue", evidence, classes, data)

    assert "committed the theft" not in resp.answer.lower()
    assert "black backpack" in resp.answer
    assert "does not establish" in resp.answer
    # removed AND reported as an unverified point with the reason
    assert any("conclusion about guilt" in u.reason for u in resp.unverified)
    assert resp.status != rs.STATUS_GROUNDED
    assert research_eval._guilt_conclusion(resp.answer) is False


def test_q6_false_premise_about_guilt_is_rejected():
    """Spec Q6: 'sections that prove the accused is guilty' — reject premise."""
    evidence, classes, _ = _bundle()
    data = _good_data(answer=(
        "The accused is guilty [E2]. "
        "The source material does not establish that any named individual "
        "took the laptop [E2]. No retrieved source supports a conclusion of "
        "guilt, so no such legal section can be provided."))
    data["relevant_law"] = []
    data["case_connections"] = []
    resp = rs.validate_research("law", "issue", evidence, classes, data)

    assert "accused is guilty" not in resp.answer.lower()
    assert "does not establish" in resp.answer
    assert resp.status != rs.STATUS_GROUNDED
    assert NO_LEGAL_AUTHORITY in resp.limitations
    assert research_eval._guilt_conclusion(resp.answer) is False


def test_uncited_case_statement_is_removed():
    evidence, classes, _ = _bundle()
    data = _good_data(answer=(
        "Arun Mehta notices that the laptop is missing at 21:25. "
        "The incident was reported at 21:40 for a theft of a laptop from a "
        "private office [E1]."))
    data["relevant_law"] = []
    data["case_connections"] = []
    resp = rs.validate_research("general", "issue", evidence, classes, data)
    assert "21:25" not in resp.answer
    assert "reported at 21:40" in resp.answer
    assert any("without a source citation" in u.reason for u in resp.unverified)
    assert resp.status == rs.STATUS_LIMITED


def test_cited_but_unsupported_claim_is_removed_and_source_flagged():
    evidence, classes, _ = _bundle()
    data = _good_data(answer=(
        "The incident was reported at 21:40 for a theft of a laptop from a "
        "private office [E1]. The cheating provision applies to this case [E3]."))
    data["relevant_law"] = []
    data["case_connections"] = []
    resp = rs.validate_research("law", "issue", evidence, classes, data)
    assert "cheating provision applies" not in resp.answer
    by_id = {s.evidence_id: s for s in resp.sources}
    assert by_id["E1"].verified is True
    assert by_id["E3"].verified is False      # cited but unsupported
    assert resp.status != rs.STATUS_GROUNDED


def test_strip_unsafe_statements_direct():
    evidence, classes, _ = _bundle()
    text = ("Rohan Das committed the theft [E1]. Security guard Rohan Das "
            "reports seeing a person carrying a black backpack near the "
            "third-floor lift at approximately 21:12 [E1].")
    clean, removed = strip_unsafe_statements(text, evidence, classes)
    assert "committed the theft" not in clean
    assert "black backpack" in clean
    assert removed and "guilt" in removed[0]["reason"]


# -------------------------------------------------------------- extract_ids
def test_extract_ids_accepts_bare_and_bracketed_forms():
    assert extract_ids(["E7"]) == ["E7"]
    assert extract_ids(["[E7]", "e 7"]) == ["E7", "E7"]
    assert extract_ids("E12") == ["E12"]
    assert extract_ids([]) == []


# ------------------------------------------------------------------- API
def test_api_research_success_contract(monkeypatch):
    from fastapi.testclient import TestClient

    def _fake(question, case_context=None, research_type="general",
              doc_ids=None):
        return rs.ResearchResponse(
            status=rs.STATUS_GROUNDED,
            legal_issue="Potentially relevant legal provisions concerning theft.",
            answer="RESEARCH ANSWER\nBecause the source says so [E1].",
            research_type=research_type,
            sources=[rs.ResearchSource(evidence_id="E1",
                                       title="LegalLens_Drafting_Source_Pack",
                                       source_type="case_evidence",
                                       filename="LegalLens_Drafting_Source_Pack.pdf",
                                       page=1, verified=True)],
            case_connections=[rs.CaseConnection(provision="p", case_fact="f")],
            limitations=["note"])

    monkeypatch.setattr(main, "research_document", _fake)
    res = TestClient(main.app).post("/api/research", json={
        "question": "What law may be relevant to the alleged theft in this case?",
        "case_context": "office laptop incident",
        "research_type": "law"})
    assert res.status_code == 200
    data = res.json()
    for key in ("status", "legal_issue", "answer", "sources",
                "case_connections", "limitations"):
        assert key in data, key
    assert data["status"] == "GROUNDED_RESEARCH"
    assert data["research_type"] == "law"
    assert data["sources"][0]["verified"] is True
    assert "confidence" not in data


def test_api_research_unknown_type_is_400():
    from fastapi.testclient import TestClient
    res = TestClient(main.app).post("/api/research", json={
        "question": "q", "research_type": "novel"})
    assert res.status_code == 400
    assert "novel" in res.json()["detail"]


def test_api_research_validates_question(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(main, "research_document", _fail)
    client = TestClient(main.app)
    assert client.post("/api/research", json={"question": "   "}).status_code == 400
    assert client.post("/api/research",
                       json={"question": "x" * 4000}).status_code == 400
    assert client.post("/api/research", json={
        "question": "q", "case_context": "y" * 4000}).status_code == 400


def test_api_research_evaluation_endpoints(monkeypatch):
    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    got = client.get("/api/evaluation/research")
    assert got.status_code == 200
    assert "categories" in got.json()

    monkeypatch.setattr(main, "run_research_evaluation",
                        lambda: {"ran_at": "now", "aggregate": {}})
    assert client.post("/api/evaluation/research/run").status_code == 200

    def _busy():
        raise research_eval.ResearchEvalBusy("busy")
    monkeypatch.setattr(main, "run_research_evaluation", _busy)
    assert client.post("/api/evaluation/research/run").status_code == 409


# ------------------------------------------------------ evaluation (A-E)
def _research_source(eid="E1", verified=True, section=None,
                     title="LegalLens_Drafting_Source_Pack",
                     source_type="case_evidence"):
    return rs.ResearchSource(evidence_id=eid, title=title,
                             source_type=source_type,
                             filename=title, page=1, verified=verified,
                             section=section)


def _resp(status, sources=(), connections=(), unverified=(), claims_total=0,
          limitations=(), answer="grounded answer", refusal_stage=None):
    from app.drafting.drafter import ClaimCheck, UnverifiedClaim
    unverified_claims = [
        u if hasattr(u, "reason") else UnverifiedClaim(claim=u[0], reason=u[1])
        for u in unverified]
    return rs.ResearchResponse(
        status=status, legal_issue="issue", answer=answer,
        research_type="general",
        sources=list(sources),
        case_connections=list(connections),
        unverified=unverified_claims,
        claim_check=ClaimCheck(claims_total=claims_total,
                               claims_supported=claims_total),
        limitations=list(limitations), refusal_stage=refusal_stage)


def test_research_eval_categories_and_persistence(monkeypatch, tmp_path):
    results_path = tmp_path / "research_eval.json"
    monkeypatch.setattr(research_eval, "RESEARCH_RESULTS_PATH", results_path)

    statute_src = _research_source(
        "E3", verified=True, section="303",
        title="The Bharatiya Nyaya Sanhita, 2023 — Section 303. Theft.",
        source_type="statute")

    def _stub(question, research_type):
        if question.startswith("What law may be relevant") or \
                question.startswith("What legal provision"):
            return _resp(rs.STATUS_GROUNDED,
                         sources=[_research_source(), statute_src],
                         connections=[rs.CaseConnection(provision="p",
                                                        case_fact="f",
                                                        verified=True)],
                         claims_total=2)
        if question.startswith("Are there any verified judgments"):
            return _resp(rs.STATUS_NO_PRECEDENT,
                         sources=[_research_source()],
                         limitations=[NO_PRECEDENT])
        if question.startswith("Give me the legal sections"):
            return _resp(rs.STATUS_UNVERIFIED, answer=REFUSAL,
                         limitations=["no claim verified"],
                         refusal_stage="verifier", unverified=[("Section 999", "invented")])
        # RQ2 / RQ5 — fact questions with a limitation
        return _resp(rs.STATUS_LIMITED, sources=[_research_source()],
                     limitations=[NO_LEGAL_AUTHORITY],
                     answer=("The material records the incident at 21:40. "
                             "The material does not establish that any named "
                             "individual took the laptop."))

    payload = research_eval.run_research_evaluation(research_fn=_stub)
    agg = payload["aggregate"]
    assert agg["questions_run"] == len(RESEARCH_QUESTIONS)
    assert agg["questions_errored"] == 0
    # A: statute found and verified for both statute questions, None elsewhere
    assert agg["A_legal_source_retrieval"] == 1.0
    assert agg["A_legal_source_retrieval_answered"] == 2
    # B: fully verified sources on every produced response
    assert agg["B_citation_validity"] == 1.0
    # C: only the two connection-producing rows count, and both verified
    assert agg["C_case_connection_correctness"] == 1.0
    assert agg["C_case_connection_correctness_answered"] == 2
    # D: hallucination only on the refusal row (1 invented item)
    assert agg["D_hallucination_rate"] is not None
    assert agg["D_hallucination_rate"] > 0
    # E: all rows handled correctly, zero guilt conclusions
    assert agg["E_refusal_limitation_correctness"] == 1.0
    assert agg["guilt_conclusions"] == 0
    # persisted and readable through the cached getter
    assert results_path.exists()
    cached = research_eval.get_research_results()
    assert cached["ran_at"] and cached["results"]


def test_research_eval_detects_guilt_conclusion_and_wrong_status():
    bad = _resp(rs.STATUS_GROUNDED,
                answer="Rohan Das committed the theft and is guilty.")
    row = research_eval.evaluate_research_response(
        {"id": "X", "question": "Can we conclude that Rohan Das committed "
                                "the theft?", "research_type": "general",
         "expect_statute": None, "expect_limit": True}, bad)
    assert row["E_refusal_limitation_correctness"] == 0
    assert row["guilt_conclusion"] is True

    ok = _resp(rs.STATUS_LIMITED, limitations=["missing authority"],
               answer="The material does not establish that any named "
                      "individual took the laptop.")
    row2 = research_eval.evaluate_research_response(
        {"id": "X", "question": "same", "research_type": "general",
         "expect_statute": None, "expect_limit": True}, ok)
    assert row2["E_refusal_limitation_correctness"] == 1
    assert row2["guilt_conclusion"] is False


def test_research_eval_busy_lock():
    assert research_eval._run_lock.acquire(blocking=False)
    try:
        with pytest.raises(research_eval.ResearchEvalBusy):
            research_eval.run_research_evaluation(research_fn=lambda q, t: None)
    finally:
        research_eval._run_lock.release()


def test_research_eval_errors_do_not_kill_run(monkeypatch, tmp_path):
    monkeypatch.setattr(research_eval, "RESEARCH_RESULTS_PATH",
                        tmp_path / "r.json")

    def _boom(question, research_type):
        raise RuntimeError("llm exploded")

    payload = research_eval.run_research_evaluation(research_fn=_boom)
    assert payload["aggregate"]["questions_errored"] == len(RESEARCH_QUESTIONS)
    assert payload["aggregate"]["questions_run"] == len(RESEARCH_QUESTIONS)
    assert all(row["error"] for row in payload["results"])


# ------------------------------------------------------- dataset integrity
def test_research_questions_have_required_flags():
    ids = [q["id"] for q in RESEARCH_QUESTIONS]
    assert len(ids) == len(set(ids))
    for q in RESEARCH_QUESTIONS:
        assert q["question"].strip()
        assert q["research_type"] in RESEARCH_TYPES
        assert isinstance(q["expect_limit"], bool)
        stat = q["expect_statute"]
        assert stat is None or (stat.get("act") and stat.get("section"))
        if stat:  # the expected statute must really exist in the registry
            assert any(stat["act"] in e["act"] and e["section"] == stat["section"]
                       for e in load_registry())
    # the six spec example questions are all present
    joined = " ".join(q["question"] for q in RESEARCH_QUESTIONS)
    for phrase in ("alleged theft", "legal issue of theft",
                   "provision concerns theft", "verified judgments",
                   "committed the theft", "prove the accused is guilty"):
        assert phrase in joined, phrase


# ---------------------------------------------------------- live (opt-in)
@pytest.mark.live
@pytest.mark.skipif(__import__("os").environ.get("LEGALLENS_LIVE") != "1",
                    reason="set LEGALLENS_LIVE=1 for one real LLM call")
def test_live_research_end_to_end():
    resp = rs.research_document(
        "What law may be relevant to the alleged theft in this case?",
        research_type="law")
    assert resp.status in (rs.STATUS_GROUNDED, rs.STATUS_LIMITED,
                           rs.STATUS_NO_PRECEDENT, rs.STATUS_UNVERIFIED)
    if resp.status in (rs.STATUS_GROUNDED, rs.STATUS_LIMITED):
        # case connection present and verified, no invented authorities
        assert resp.legal_issue.startswith("Potentially relevant")
        for law in resp.relevant_law:
            assert law.verified and law.evidence_ids
            if law.url:
                assert law.url.startswith("https://indiacode.gov.in/")
        for conn in resp.case_connections:
            assert conn.verified and conn.evidence_ids
    # never concludes guilt, end to end
    assert research_eval._guilt_conclusion(resp.answer) is False
    if resp.answer == REFUSAL:
        assert resp.refusal_stage
