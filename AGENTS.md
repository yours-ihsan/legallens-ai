# AGENTS.md

HACKNEX 2026 prototype "LegalLens AI": grounded RAG chat over uploaded legal PDFs.
Reliability and a working demo over features. No auth, payments, or deployment.

## Commands (Windows / PowerShell)

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt   # bare `pip` is NOT on PATH
uvicorn app.main:app --reload               # http://localhost:8000
python -m pytest tests -q                   # offline tests (~4 s)
$env:LEGALLENS_LIVE=1; python -m pytest tests -q   # adds one real LLM call
```

## Hard invariants — do not break

- The LLM must receive **only** the retrieved evidence packet
  (`answer_generator.build_packet`) plus the system-computed PREMISE CHECK
  verdicts (`premise_block`). Never give it tools, files, or other documents.
- Page numbers come from vector-store metadata only, never from the model.
  The model cites evidence IDs (`E1`…); Python maps them to `(filename, page)`.
- The refusal string is exact: `Insufficient evidence in the provided documents.`
  Chat uses a **three-way decision**: retrieval score < `LOW_FLOOR` 0.10
  (LLM not called, `stage=retrieval`), every assumption `NOT_FOUND` by the
  premise check (LLM not called, `stage=premise`), otherwise answer/cited
  correction — then `validate()` still rejects model `insufficient=true`,
  hallucinated/missing citations, page text outside the evidence, or any
  LLM/parse failure (`stage=model|verifier|system`).
- `app/premise.py` is pure and deterministic: assumption extraction and the
  SUPPORTED / CONTRADICTED / NOT_FOUND verdicts come from Python + chunks only,
  never from the model.
- Every chunk must carry `doc_id`, `filename`, `page` metadata through
  extraction → chunking → Chroma → retrieval. Chunks are per-page (never span
  pages) so citations stay unambiguous.
- `answer_generator.validate()` is pure and unit-tested without the LLM — keep it
  that way when adding checks.

## Environment quirks (verified, not guesses)

- **No API keys exist.** LLM calls go through the local OpenCode CLI:
  `opencode run --format json -m <model>` with the prompt on **stdin**
  (`app/llm.py`). stdout is NDJSON — concatenate `part.type=="text"` parts;
  `type=="error"` events + non-zero exit mean provider failure.
- Model is pinned (`LEGALLENS_MODEL`, default `opencode/mimo-v2.6-flash-free`).
  The OpenCode *default* model currently returns provider 503s — do not remove `-m`.
- Python 3.14 in `.venv`. Chroma downloads its ONNX embedding model once to
  `~/.cache/chroma` (first run needs network, ~80 MB).
- LLM calls take ~5–60 s; `/api/chat` is a **sync** endpoint on purpose (runs in
  FastAPI's thread pool so the subprocess never blocks the event loop).
- Thresholds/bands are env-tunable (`LEGALLENS_LOW_FLOOR`, `LEGALLENS_THRESHOLD`,
  `LEGALLENS_TOP_K`, `LEGALLENS_BAND_*`) — calibrate against the demo PDF before
  hard-coding. Chat gates at the 0.10 low floor; drafting and research gate at
  `REFUSAL_THRESHOLD` 0.30.

## Layout

```
app/         pdf_extractor → chunker → vector_store (Chroma, cosine) →
             retriever → premise (deterministic premise check) →
             answer_generator (trust boundary) → llm (opencode CLI)
             main.py = FastAPI routes + doc registry (data/docs.json)
static/      plain HTML/CSS/JS, no build step
tests/       offline pipeline tests + generate_fixture.py (rebuilds sample.pdf)
data/        uploads, vectors, registry — all gitignored
```
