"""Pipeline tests. Everything except the `live` test runs offline and fast.

Run:  python -m pytest tests -q
Live: LEGALLENS_LIVE=1 python -m pytest tests -q   (adds one real LLM call)
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import chunker, pdf_extractor, retriever
from app.answer_generator import REFUSAL, answer_question, build_packet, validate
from app.config import CHUNK_MAX_CHARS
from app.llm import LLMError, extract_json, parse_run_output
from app.models import Evidence
from app.vector_store import VectorStore
from generate_fixture import PAGE_TEXTS, make_sample_pdf


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    path = ROOT / "tests" / "fixtures" / "sample.pdf"
    if not path.exists():
        make_sample_pdf(path)
    return path


@pytest.fixture()
def store(sample_pdf, tmp_path, monkeypatch):
    """Fresh Chroma store with the sample PDF indexed; patches the app's store."""
    pages = pdf_extractor.extract_pages(sample_pdf)
    chunks = chunker.chunk_pages(pages, doc_id="doc1", filename="sample.pdf")
    s = VectorStore(path=tmp_path / "chroma", collection="test")
    s.add_chunks(chunks)
    monkeypatch.setattr(retriever, "_store", s)
    return s


def _ev(eid: str, page: int, score: float = 0.6, filename: str = "contract.pdf",
        text: str = "Liability shall not exceed USD 50,000 per contract year.") -> Evidence:
    return Evidence(
        id=eid, chunk_id=f"doc1:{eid}", doc_id="doc1", filename=filename,
        page=page, text=text, score=score,
    )


# ---------------------------------------------------------------- extraction
def test_extract_preserves_page_numbers(sample_pdf):
    pages = pdf_extractor.extract_pages(sample_pdf)
    assert [p.page_number for p in pages] == [1, 2]
    assert "Clause 7" in pages[0].text
    assert "Meridian Logistics" in pages[0].text
    # page-2-only content must not leak onto page 1
    assert "SIAC" in pages[1].text
    assert "SIAC" not in pages[0].text


def test_extract_rejects_textless_pdf(tmp_path):
    from pypdf import PdfWriter
    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(str(blank))
    with pytest.raises(ValueError, match="No extractable text"):
        pdf_extractor.extract_pages(blank)


# ---------------------------------------------------------------- chunking
def test_chunks_keep_page_metadata(sample_pdf):
    pages = pdf_extractor.extract_pages(sample_pdf)
    chunks = chunker.chunk_pages(pages, doc_id="d1", filename="x.pdf")
    assert chunks
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.page in (1, 2)
        assert c.doc_id == "d1" and c.filename == "x.pdf"
        assert len(c.text) <= CHUNK_MAX_CHARS
        # whitespace-normalized chunk text must be a literal substring of its page
        expected = " ".join(PAGE_TEXTS[c.page - 1].split())
        assert " ".join(c.text.split()) in expected
    # page 1 chunks come from page-1 text only
    assert all("SIAC" not in c.text for c in chunks if c.page == 1)


def test_long_text_is_split_with_overlap():
    text = " ".join(f"sentence number {i} of the clause." for i in range(1, 120))
    pieces = chunker._split_page(text, max_chars=300, overlap=60)
    assert len(pieces) > 3
    assert all(len(p) <= 300 for p in pieces)
    assert all(p.strip() for p in pieces)
    # overlap: the boundary sentence appears in two consecutive pieces
    assert any(word in pieces[1] for word in pieces[0].split()[-3:])


# ---------------------------------------------------------------- retrieval
def test_vector_store_roundtrip_and_metadata(store):
    hits = store.query("What is the limitation of liability?", top_k=3)
    assert hits
    top = hits[0]
    assert top["meta"]["doc_id"] == "doc1"
    assert top["meta"]["filename"] == "sample.pdf"
    assert top["meta"]["page"] in (1, 2)
    assert top["score"] > 0.2

    assert store.count() > 0
    store.delete_doc("doc1")
    assert store.count() == 0
    assert store.query("liability", top_k=3) == []


def test_low_score_gate_refuses_without_llm(store, monkeypatch):
    """Below the LOW floor, questions must never reach the LLM."""
    def _fail(*args, **kwargs):
        raise AssertionError("LLM must not be called when retrieval is weak")

    monkeypatch.setattr("app.answer_generator.run_llm", _fail)
    # Simulate retrieval too weak to clear the floor (env is read at import).
    monkeypatch.setattr("app.answer_generator.LOW_FLOOR", 0.99)
    resp = answer_question("What are the parking fines on the planet Mars?")
    assert resp.insufficient is True
    assert resp.answer == REFUSAL
    assert resp.refusal_stage == "retrieval"
    assert resp.decision == "refusal_low_evidence"
    assert "low floor" in (resp.reason or "")


def test_llm_failure_becomes_refusal(store, monkeypatch):
    def _raise(*args, **kwargs):
        raise LLMError("provider exploded")

    monkeypatch.setattr("app.answer_generator.run_llm", _raise)
    resp = answer_question("What is the maximum aggregate liability under this agreement?")
    assert resp.insufficient is True
    assert resp.answer == REFUSAL
    assert resp.refusal_stage == "system"
    assert "model error" in (resp.reason or "")


# ---------------------------------------------------------------- validation
GOOD = {
    "answer": "Liability is capped at USD 50,000 per contract year.",
    "insufficient": False,
    "confidence": 0.9,
    "citations": ["E1"],
}


def test_validate_maps_citations_to_filename_and_page():
    evidence = [_ev("E1", page=3), _ev("E2", page=7, score=0.4)]
    resp = validate("What is the cap?", evidence, top_score=0.62, data=dict(GOOD))
    assert resp.insufficient is False
    assert resp.refusal_stage is None
    assert len(resp.citations) == 1
    assert resp.citations[0].filename == "contract.pdf"
    assert resp.citations[0].page == 3  # page comes from OUR metadata, not the model
    assert resp.confidence.band == "High"
    assert resp.confidence.score == pytest.approx(0.5 * 0.62 + 0.5 * 0.9, abs=1e-6)


def test_validate_rejects_hallucinated_evidence_id():
    evidence = [_ev("E1", page=3)]
    data = dict(GOOD, citations=["E99"])
    resp = validate("q", evidence, 0.6, data)
    assert resp.insufficient is True
    assert resp.answer == REFUSAL
    assert resp.refusal_stage == "verifier"
    assert "non-existent" in (resp.reason or "")


def test_validate_requires_citations():
    evidence = [_ev("E1", page=3)]
    resp = validate("q", evidence, 0.6, dict(GOOD, citations=[]))
    assert resp.insufficient is True
    assert resp.refusal_stage == "verifier"
    assert "no citations" in (resp.reason or "")


def test_validate_honors_model_insufficiency():
    evidence = [_ev("E1", page=3)]
    data = {"answer": REFUSAL, "insufficient": True, "confidence": 0.1, "citations": []}
    resp = validate("q", evidence, 0.6, data)
    assert resp.insufficient is True
    assert resp.answer == REFUSAL
    assert resp.refusal_stage == "model"


def test_validate_rejects_unverified_page_numbers():
    evidence = [_ev("E1", page=3)]
    data = dict(GOOD, answer="The cap is USD 50,000 (see page 99).")
    resp = validate("q", evidence, 0.6, data)
    assert resp.insufficient is True
    assert resp.refusal_stage == "verifier"
    assert "unverified page 99" in (resp.reason or "")


def test_validate_accepts_page_numbers_that_exist_in_evidence():
    evidence = [_ev("E1", page=3)]
    data = dict(GOOD, answer="As stated on page 3, liability is capped.")
    resp = validate("q", evidence, 0.6, data)
    assert resp.insufficient is False


def test_false_premise_correction_is_validated_as_grounded():
    """A grounded correction ("The sources state X, not Y [E1]") must pass validation."""
    evidence = [_ev("E1", page=1)]
    data = {
        "answer": "The sources state USD 50,000 per contract year, not USD 1,000 [E1].",
        "insufficient": False,
        "confidence": 0.9,
        "citations": [],
    }
    resp = validate("Why is the cap USD 1,000?", evidence, 0.6, data)
    assert resp.insufficient is False
    assert resp.refusal_stage is None
    # [E1] marker in the answer text is merged into real citations
    assert len(resp.citations) == 1
    assert resp.citations[0].page == 1


def test_answer_text_marker_cannot_hallucinate_evidence():
    evidence = [_ev("E1", page=1)]
    data = dict(GOOD, answer="Liability is capped [E99].")
    resp = validate("q", evidence, 0.6, data)
    assert resp.insufficient is True
    assert resp.refusal_stage == "verifier"
    assert "non-existent" in (resp.reason or "")


def test_system_prompt_requires_corrections_for_false_premises():
    from app.answer_generator import SYSTEM_PROMPT
    assert "FALSE-PREMISE RULE" in SYSTEM_PROMPT
    assert "DO NOT refuse" in SYSTEM_PROMPT
    assert "insufficient\" to true ONLY when the evidence says nothing relevant" in SYSTEM_PROMPT


def test_build_packet_labels_evidence_with_source_and_page():
    evidence = [_ev("E1", page=3), _ev("E2", page=7, filename="deal.pdf")]
    packet = build_packet(evidence)
    assert "[E1] (contract.pdf, page 3" in packet
    assert "[E2] (deal.pdf, page 7" in packet
    assert evidence[0].text in packet


# ---------------------------------------------------------------- LLM bridge
def test_parse_run_output_concatenates_text_parts():
    lines = [
        '{"type":"step_start","part":{"type":"step-start"}}',
        '{"type":"text","part":{"type":"text","text":"Hello"}}',
        '{"type":"text","part":{"type":"text","text":" world!"}}',
        "not json at all",
    ]
    assert parse_run_output("\n".join(lines)) == "Hello\nworld!"


def test_parse_run_output_raises_on_provider_error():
    lines = [
        '{"type":"error","error":{"type":"provider.internal","message":"Endpoint is unavailable."}}',
        '{"type":"text","part":{"type":"text","text":"should not appear"}}',
    ]
    with pytest.raises(LLMError, match="Endpoint is unavailable"):
        parse_run_output("\n".join(lines), returncode=1)


def test_extract_json_tolerates_fences():
    raw = 'Sure! Here is the JSON:\n```json\n{"answer": "hi", "insufficient": false}\n```'
    assert extract_json(raw) == {"answer": "hi", "insufficient": False}
    with pytest.raises(LLMError):
        extract_json("no json here")


# ---------------------------------------------------------------- live (opt-in)
@pytest.mark.live
@pytest.mark.skipif(os.environ.get("LEGALLENS_LIVE") != "1", reason="set LEGALLENS_LIVE=1")
def test_live_end_to_end_grounded_answer(store):
    resp = answer_question("What is the maximum aggregate liability under this agreement?")
    assert resp.insufficient is False, resp.reason
    assert resp.citations, "answer must cite at least one evidence item"
    assert all(c.page in (1, 2) for c in resp.citations)
    assert resp.answer != REFUSAL
