# LegalLens AI — HACKNEX 2026 prototype

A grounded RAG system for legal PDFs. It answers questions, generates legal
drafts, performs evidence-first legal research, and evaluates its own
groundedness — always from retrieved sources only. When the retrieved evidence
is too weak, it refuses explicitly with the exact string
`Insufficient evidence in the provided documents.` instead of guessing.

Stack: **FastAPI + Chroma + pypdf** backend, **vanilla HTML/CSS/JS** frontend
(no build step, no external frontend libraries), LLM reached through the local
**OpenCode CLI** — **no API keys anywhere**.

---

## What it does

* **Ingests PDFs** — upload → text extraction with page numbers → per-page
  chunks → local vector index (every vector carries `doc_id`, `filename`,
  `page`).
* **Answers questions over them** — retrieves the top chunks, checks the
  question's assumptions against those chunks (SUPPORTED / CONTRADICTED /
  NOT_FOUND), prompts the LLM with *only* the retrieved evidence plus the
  system-computed premise verdicts, validates every citation, and returns a
  cited answer, a cited correction of a false premise, or the exact refusal.
* **Generates grounded legal drafts** — bail applications, petitions, legal
  notices, affidavits — reporting which required information is present in the
  source and which is **missing** (never invented).
* **Performs evidence-first legal research** — identifies the legal issue
  deterministically, retrieves case evidence plus official statutory sources
  (bundled India Code snapshot, live India Code search best-effort), verifies
  every law-to-case connection and every citation in Python, and answers with
  cautious language — an allegation is never converted into fact and guilt is
  never concluded.
* **Measures its own groundedness** — runs a deterministic
  claim-traceability comparison between a plain Retrieve→Generate baseline and
  the real LegalLens pipeline over a fixed 19-question dataset, plus a
  research-track evaluation over six research questions (categories A–E).

---

## Workflows — what is implemented

| Workflow | Endpoint(s) | UI | Status |
|---|---|---|---|
| Document upload / list / delete | `POST/GET/DELETE /api/documents` | sidebar (drag & drop) | ✅ implemented |
| Grounded RAG chat | `POST /api/chat` | tab **Grounded Chat** | ✅ implemented |
| Legal drafting | `POST /api/draft` | tab **Legal Drafting** | ✅ implemented |
| Legal research | `POST /api/research` | tab **Legal Research** | ✅ implemented |
| Groundedness evaluation vs baseline | `GET /api/evaluation`, `POST /api/evaluation/run` | tab **Evaluation** | ✅ implemented |
| Research evaluation (categories A–E) | `GET /api/evaluation/research`, `POST /api/evaluation/research/run` | API only | ✅ implemented |

**All four draft types are implemented** — they are pure configuration in
`app/drafting/requirements.py` (`DRAFT_TYPES`), each with its own professional
structure template, retrieval query, and required-field detectors:

| Key | Draft type | Required fields (detected by Python regex, e.g.) |
|---|---|---|
| `bail_application` | Bail Application | court name, FIR number, police station, accused/applicant name, case number, incident date, case details, complainant, grounds for bail |
| `petition` | Petition | court name, court jurisdiction, case/reference number, petitioner details, respondent details, relevant facts, legal grounds, statutory provisions, relief requested, filing date, advocate details |
| `legal_notice` | Legal Notice | date of notice, sender, recipient, issue/dispute, requested action, relevant facts |
| `affidavit` | Affidavit | deponent, statements to affirm, relevant facts, date/place of sworn statement |

The evaluation compares two *chat* pipelines (baseline vs LegalLens); a
drafting-vs-baseline evaluation track is **not** implemented — each draft
response instead carries its own deterministic `claim_check` (see below).

---

## Architecture & pipeline

### Ingestion

```
PDF (≤ 50 MB, must contain extractable text — scans/OCR are not supported)
  → pypdf extraction           page numbers = physical 1-based page order,
                                the ONLY source of page numbers in the system
  → per-page chunking           max 1000 chars, 150-char overlap,
                                chunks NEVER span pages → 1 chunk = 1 page
  → Chroma (persistent, cosine) every vector stores doc_id / filename / page
                                embeddings computed locally (Chroma ONNX model)
  → data/docs.json registry
```

### RAG chat (`app/answer_generator.py`)

```
question
  → retrieve()                  top-K = 6 scored chunks → Evidence E1…E6
  → LOW-FLOOR gate              best score < 0.10 → refusal, LLM NOT called
  → premise check (app/premise.py)  DETERMINISTIC — Python extracts the question's
                                assumptions and marks each SUPPORTED /
                                CONTRADICTED / NOT_FOUND from the chunks only;
                                all NOT_FOUND → refusal, LLM NOT called
  → evidence-only prompt        SYSTEM_PROMPT + PREMISE CHECK verdicts +
                                build_packet(evidence) + question
                                (the model sees nothing but the packet)
  → LLM (OpenCode CLI)          must return JSON:
                                {answer, insufficient, confidence, citations[]}
  → validate()                  every check below must pass — still runs on
                                every branch, corrections included
  → AnswerResponse              cited answer + evidence + confidence + decision
                                + premises, or the exact refusal string
```

**Three-way decision** (reported as `decision`, shown in the UI):
`grounded_answer` (evidence supports the question) ·
`premise_correction` (≥ 1 assumption CONTRADICTED — the answer opens with a
cited correction "The sources state X, not Y [E#]") ·
`refusal_low_evidence` / `refusal_no_premise_support` (nothing above the 0.10
floor, or every assumption NOT_FOUND — in both cases the LLM is not consulted)
plus `refusal_system` / `refusal_model` / `refusal_verifier` for downstream
failures.

Refusal stages (reported as `refusal_stage`): **retrieval** (nothing indexed /
below the 0.10 low floor), **premise** (every assumption NOT_FOUND in the
chunks), **system** (LLM or JSON-parse failure), **model** (the model
itself declined), **verifier** (validation rejected the output).

Confidence: `score = 0.5·retrieval + 0.5·model`; the band comes from the
retrieval score — **High ≥ 0.55**, **Medium ≥ 0.40**, else **Low**.

The system prompt also carries a **false-premise rule**: if a question embeds an
assumption the evidence contradicts, the model must answer with a grounded
correction ("The sources state X, not Y") plus citations — refusal is reserved
for questions the evidence says nothing about. The premise check makes that rule
a first-class, machine-checked step: its verdicts are injected into the prompt
and echoed in the response.

### Legal drafting (`app/drafting/`, imported only by `/api/draft`)

```
draft_type + instructions
  → retrieve() + threshold gate         (same retriever, REFUSAL_THRESHOLD 0.30)
  → deterministic field detection       Python regex over the evidence — the
                                        model never decides what exists
  → missing-information report          computed BEFORE any LLM call
  → evidence-only drafting prompt       build_packet + structure template +
                                        available/missing field lists
  → validate()                          same E-ID / page rules as chat, plus:
                                        a draft with no [E#] markers is refused
  → verify_citations()                  claim → CITED chunk → verify: wrong
                                        citations are re-pointed at the chunk
                                        that actually contains the claim,
                                        unsupported markers are dropped, and a
                                        claim no chunk supports is replaced
                                        with [NOT PROVIDED] (never displayed)
  → claim_check                         every remaining [E#]-marked sentence
                                        re-checked with app/evaluation/metrics
  → { draft, sources, missing_information, available_information,
      claim_check, citation_corrections, removed_claims, status }
```

Status: `grounded` (nothing missing, all marked claims traceable) ·
`missing` (draft produced but required fields, untraceable claims and/or
removed unsupported claims exist, all listed) · `insufficient`
(gate/LLM/verifier failure — no draft produced, exact refusal string returned).

### Legal research (`app/research/`, imported only by `/api/research`)

```
question + research_type(law|judgment|precedent|general)
  → identify_legal_issue()          DETERMINISTIC Python topic lexicon — the
                                    model never names the issue or the statute
                                    search terms
  → gather_evidence()               case evidence  = the EXISTING retriever
                                    (same chunks, same scores, E1..En)
                                    legal authority = bundled India Code
                                    snapshot (app/research/legal_sources.json,
                                    verbatim BNS/BNSS sections, official
                                    metadata) seeded into a SEPARATE Chroma
                                    collection `legallens_legal` in the same
                                    store, + live India Code API search
                                    (best-effort, LEGALLENS_LIVE_SOURCES=1;
                                    failure degrades to snapshot with a
                                    limitation note, never an error)
                                    legal blocks continue at E(n+1)...
  → threshold gate                  best CASE score < 0.30 → refusal, LLM NOT
                                    called (REFUSAL_THRESHOLD; chat's gate is
                                    the separate 0.10 low floor)
  → evidence-only research prompt   build_packet(evidence) + Python-computed
                                    SOURCE CLASSIFICATION block (tier,
                                    authority, section, URL come from the
                                    registry/metadata — never from the model)
  → LLM (OpenCode CLI)              "grounded legal research assistant" system
                                    role; must return JSON {answer, relevant_law,
                                    case_connections, judgments, limitations,
                                    insufficient}; FALSE-PREMISE RULE: a
                                    question asking to prove guilt is corrected,
                                    never answered
  → validate_research()             pure Python (no LLM, no network):
                                    marker/page checks (same rules as chat)
                                    → strip_unsafe_statements(): an unsupported
                                      guilt conclusion or an uncited
                                      case-specific/legal statement is REMOVED
                                      and reported — a guilt sentence survives
                                      only if a retrieved source literally
                                      asserts it AND the claim grounds
                                    → verify_citations() on the answer prose
                                    → verify_law_entries(): each relevant_law
                                      entry must cite a LEGAL-AUTHORITY block
                                      that literally supports the provision
                                      (checked against the full official text,
                                      not the chunk slice)
                                    → verify_case_connections(): must hold on
                                      BOTH sides — provision grounded in legal
                                      authority AND case fact grounded in case
                                      material — else dropped with a reason
                                    → verify_judgments(): a precedent needs a
                                      judgment-classed source; none →
                                      NO_VERIFIED_PRECEDENT_FOUND
                                    → status: GROUNDED (legal + case verified)
                                      / RESEARCH_WITH_LIMITATIONS (exactly one
                                      side verified, or any unverified item was
                                      excluded) / UNVERIFIED_RESEARCH (exact
                                      refusal + refusal_stage)
  → ResearchResponse                {status, legal_issue, answer, sources,
                                    relevant_law, case_connections, judgments,
                                    limitations, citation_corrections,
                                    removed_claims, unverified, claim_check,
                                    reason, refusal_stage} — NO confidence
                                    field, no score is invented
```

**Source tiers** are metadata, not model output: Tier 1 = official
repositories (India Code / Government of India), Tier 2 = established legal
databases, Tier 3 = general web. `verified` is only ever computed from actual
retrieval plus a passing claim check — the model cannot mark a source
verified. Statutes, sections, judgments, courts, URLs and dates that cannot be
traced to a retrieved source are never displayed; the constant
`Verified legal authority was not found in the available sources.` is returned
instead.

**Judgment honesty**: no live judgment source exists in this prototype, so
`judgment`/`precedent` research honestly returns
`NO_VERIFIED_PRECEDENT_FOUND` unless a judgment-classed chunk really is in the
corpus (the bundled snapshot contains statutes only).

### Groundedness evaluation (`app/evaluation/`, imported only by `/api/evaluation*`)

```
19 fixed questions ──┬── baseline:    run_baseline()  = Retrieve → Generate
                     │                (real retrieve() + real run_llm(),
                     │                 NO gate, NO citation validation,
                     │                 NO refusal enforcement)
                     └── legallens:   the real, unmodified answer_question()
                                       (gate + packet + validator + refusal)
                              ↓
        deterministic claim grounding (no LLM judge) → aggregate
                              ↓
        data/evaluation_results.json (atomic write, run-lock protected)
```

Both sides query the same Chroma collection through the same retriever, so
`retrieval_improvement` is honestly `0.0`; the measurable difference comes from
the verification/refusal layer.

### Project layout

```
app/
  main.py            FastAPI routes + document registry (data/docs.json)
  pdf_extractor.py   pypdf page-preserving extraction
  chunker.py         per-page chunking (page metadata attached)
  vector_store.py    Chroma wrapper (cosine, metadata on every vector)
  retriever.py       question → scored Evidence E1…E6
  answer_generator.py  SYSTEM_PROMPT, build_packet, validate(), refusal,
                       three-way decision (premise_block)
  premise.py         deterministic premise check: extract the question's
                     assumptions, mark each SUPPORTED / CONTRADICTED /
                     NOT_FOUND from the chunks only (pure, no LLM)
  llm.py             OpenCode CLI bridge (run_llm / extract_json)
  config.py          all tunables (env-overridable)
  drafting/          Legal Drafting workflow (requirements.py = the 4 types)
  research/          Legal Research workflow:
    legal_sources.json    bundled verbatim India Code sections (official
                          metadata: act, section, authority, tier, url, page)
    legal_source_retriever.py  snapshot seeding into the separate
                          `legallens_legal` collection + live India Code
                          search + evidence classification (case vs statute
                          vs judgment) + merged gather_evidence()
    research_prompt.py    deterministic legal-issue identification, statute
                          search terms, RESEARCH_SYSTEM_PROMPT, build_prompt
    source_verifier.py    pure verification: strip_unsafe_statements,
                          verify_law_entries / _case_connections / _judgments,
                          constants (exact refusal + NO_LEGAL_AUTHORITY)
    research_service.py   ResearchRequest/Response models, research_document(),
                          validate_research()
  evaluation/        datasets (incl. RESEARCH_QUESTIONS), metrics, baseline,
                     evaluator, research_eval (categories A–E)
static/              index.html, app.js, style.css (no build step)
tests/               145 offline tests + fixture generator
data/                uploads/, chroma/, docs.json, evaluation_results.json,
                     research_evaluation_results.json
```

### Configuration (all env-overridable, `app/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `LEGALLENS_MODEL` | `opencode/mimo-v2.6-flash-free` | model passed to `opencode run -m` |
| `LEGALLENS_LLM_TIMEOUT` | `180` | seconds per LLM call |
| `LEGALLENS_TOP_K` | `6` | chunks retrieved |
| `LEGALLENS_THRESHOLD` | `0.30` | refusal gate on best cosine score (drafting + research) |
| `LEGALLENS_LOW_FLOOR` | `0.10` | chat low floor: below it the LLM is not consulted (above it the premise check decides) |
| `LEGALLENS_BAND_HIGH` / `LEGALLENS_BAND_MEDIUM` | `0.55` / `0.40` | confidence band cutoffs |
| `LEGALLENS_EVAL_WORKERS` | `4` | parallel threads during an evaluation run |
| `LEGALLENS_LIVE` | unset | set to `1` to enable the live LLM test |
| `LEGALLENS_LIVE_SOURCES` | `1` | set to `0` to disable the live India Code search (research then uses only the bundled snapshot) |
| `LEGALLENS_SOURCE_TIMEOUT` | `8` | seconds per live India Code query (a failure degrades to the snapshot, never an error) |

---

## Install & run

Windows / PowerShell (the venv used in development is Python 3.14):

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt   # bare `pip` is NOT on PATH

uvicorn app.main:app --reload               # → http://localhost:8000
# LAN-accessible (for demoing from another device):
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Dependencies (`requirements.txt`): `fastapi`, `uvicorn`, `pypdf`, `chromadb`,
`python-multipart`, `pytest`, `httpx`.

**No API keys are needed.** LLM calls shell out to the local OpenCode CLI —
`opencode run --format json -m <model>` with the prompt on stdin; stdout is
NDJSON and is parsed by `app/llm.py` (provider errors and non-zero exits become
`LLMError`, which becomes a refusal, never a crash). Embeddings run locally via
Chroma's ONNX model (downloaded once to `~/.cache/chroma` on first use).

Notes:

* The first indexing run downloads the embedding model; LLM calls typically
  take a few seconds up to the 180 s timeout, and `/api/chat` + `/api/draft`
  are synchronous on purpose (they run in FastAPI's thread pool so the
  subprocess never blocks the event loop).
* Upload real contracts or the sample crime-case PDF from the UI — the demo
  does not depend on any pre-baked answers.

### Tests

```powershell
.venv\Scripts\python.exe -m pytest tests -q    # 142 passed, 3 skipped (~5 s)
$env:LEGALLENS_LIVE=1; python -m pytest tests -q   # un-skips 2 live LLM tests
```

145 test functions across `test_pipeline.py`, `test_premise.py`,
`test_evaluation.py`, `test_drafting.py` and `test_research.py`:
extraction/chunking/metadata, low-floor gate, hallucinated IDs and unverified
pages, refusal stages, **the three-way decision** (assumption extraction,
SUPPORTED/CONTRADICTED/NOT_FOUND marking, REFUSE/CORRECT/ANSWER branches with
the LLM stubbed, verifier still running after the premise step), evaluation
metrics/dataset anchors, drafting field detection, **citation
correctness** (wrong citations re-pointed, unsupported markers dropped,
unsupported claims replaced with `[NOT PROVIDED]`, orphan-marker regression),
**missing-information correctness** (the Petition's exact required-field
list; label-only mentions never mark a field available), **research
correctness** (deterministic legal issue, registry-derived authorities that
the model cannot override, both-sides connection verification, judgment
honesty, the guilt-conclusion guards for "can we conclude X committed the
theft" / "sections that prove guilt", evaluation categories A–E) and API
contracts. Three tests can skip: the two marked `live` in `pytest.ini`
(needs `LEGALLENS_LIVE=1` and make a real LLM call) and the dataset-anchor
test, which skips when its source PDF is not in the current corpus.

---

## Corpus & data

**Runtime corpus** — registered in `data/docs.json`, indexed in
`data/chroma/`, originals in `data/uploads/`. As of the last run
(`GET /api/health` → `{"documents": 2, "chunks": 9}`):

| Document | Pages | Chunks | Role |
|---|---|---|---|
| `LegalLens_Drafting_Source_Pack.pdf` | 3 | 5 | fictional drafting source pack: common case facts (Case ID LL-CR-2026-014, incident 18 September 2026, timeline, complainant/witness), evidence records E1–E4, and per-draft-type "use only / missing" sections |
| `LegalLens_Fictional_Crime_Case.pdf` | 2 | 4 | fictional case file: incident summary, timeline, evidence log, and the deliberate absence statements ("does not establish that any named individual took the laptop", "does not state that anyone was arrested…") that drive corrections and refusals |

Statutes live in a **separate** Chroma collection (`legallens_legal`, 37
chunks / 11 sections) used only by Legal Research — chat retrieval returns
case-file chunks only.

The corpus is whatever the user has uploaded — the tables above describe the
last verified state, and `GET /api/health` reports the live one.

**Evaluation dataset** — `app/evaluation/datasets.py`: **19 questions —
15 answerable, 4 intentionally unanswerable** — written *only* from
`LegalLens_Fictional_Crime_Case.pdf` and `sample.pdf`. Each item carries
`question`, `answerable`, `expected_facts` (documentation) and
`expected_evidence` as stable source references (`filename` + `page` + verbatim
`anchor`), **not** E-IDs (E1…En are re-assigned by retrieval rank on every
query). `tests/test_evaluation.py` verifies every anchor verbatim against the
uploaded PDFs, so the dataset cannot contain invented facts (the test skips
when a source PDF is not in the current corpus).

**Input limits**: PDF only, ≤ 50 MB, must contain extractable text (image-only
PDFs are rejected with a clear error), questions/instructions ≤ 2000 chars.

**Persisted state**: `data/evaluation_results.json` (cached evaluation run),
`data/last_eval_response.json` (raw last run response). Deleting
`data/` resets the whole system to empty.

---

## How citations are verified

Page numbers can only originate from the PDF, never from the model:

1. **Provenance is structural.** `pypdf` physical page order → chunk →
   Chroma metadata (`doc_id`, `filename`, `page`) → `Evidence` objects. Every
   chunk has exactly one page because chunks never span pages.
2. **The model only names evidence IDs.** The prompt forbids page numbers and
   filenames in the answer text; it may write `[E1]`-style markers and must
   list supporting IDs in `citations`. `build_packet()` shows each block as
   `[E1] (filename, page N, relevance S)`.
3. **`validate()` re-checks everything in Python** (`answer_generator.py`,
   pure and unit-tested without any LLM call). It rejects →
   `Insufficient evidence in the provided documents.` with a `refusal_stage`:
   * output that is not a JSON object (**verifier**), or the model set
     `insufficient: true` or returned an empty answer (**model**);
   * any cited ID or `[E#]` marker that does not exist in the retrieved
     evidence (**verifier**);
   * zero citations (**verifier**);
   * any page number in the answer text not present in the retrieved evidence
     — `page 99` cannot pass unless page 99 really is in the packet
     (**verifier**);
   * LLM/JSON failures (**system**), scores below the 0.10 low floor
     (**retrieval**), or a question whose every assumption is NOT_FOUND in
     the chunks (**premise**) — in the last two cases the LLM is never
     even called).
4. **Valid IDs are mapped back to metadata**: the response's `citations` are
   built from vector-store `(filename, page, quote)`, so a citation can only
   say what the index actually holds.
5. **Drafting verifies every `[E#]` against its own chunk.** `validate()`
   applies the same existence/page rules as chat (a draft with no markers at
   all is refused). Then `verify_citations()` checks each marked sentence
   against **the cited chunk specifically** — never the best chunk elsewhere
   in the packet:
   * if the cited chunk does not contain the claim, the citation is
     re-pointed at the retrieved chunk that does (source hierarchy: the
     chunk holding the fact, in retrieval order) — recorded in
     `citation_corrections`;
   * an extra marker on a sentence that chunk does not support is dropped;
   * a claim **no** retrieved chunk supports is removed from the draft and
     replaced with `[NOT PROVIDED]` — recorded in `removed_claims` — so an
     unsupported assertion is never displayed as fact.

   Sources are then built only from the surviving verified markers plus
   field-detection support. The remaining marked sentences are re-checked
   with the same deterministic claim metric (single chunk, all numbers
   verbatim, ≥ 75 % of content words present) and reported in `claim_check`.
   Field availability for drafts is decided by Python regex over the
   evidence before the model runs, and the patterns require **value syntax**
   (`<field>: <Value>`, `<field> is <Value>`, number-bearing docket forms) —
   so a document that merely *lists* field names (its own "Missing: Court
   name, …" note) never marks those fields available. Missing fields are
   reported as missing, never filled in.

---

## Running the groundedness evaluation

The evaluation lives in `app/evaluation/` and is imported only by the two
evaluation endpoints — the normal chat path never touches it.

### Start a run

```powershell
# API (synchronous — real LLM calls, can take several minutes)
curl.exe -X POST http://localhost:8000/api/evaluation/run
# or in the UI: http://localhost:8000 → "Evaluation" tab → "Run Groundedness Evaluation"
```

* Runs both systems over all 19 questions with `LEGALLENS_EVAL_WORKERS`
  threads (default 4).
* A second concurrent run returns **HTTP 409** (`EvaluationBusy`).
* Results are written atomically to `data/evaluation_results.json`.

### Read results

```powershell
curl.exe http://localhost:8000/api/evaluation     # cached results
```

`GET /api/evaluation` returns `{"evaluated": false, ...}` — and the UI shows
**"Not evaluated yet"** — until a real run exists. No percentage anywhere in
the UI or API is hardcoded; everything shown is computed from a run.

### What is measured (deterministic, no LLM judge)

* **Groundedness** = supported claims / total factual claims × 100. A claim is
  supported only if, within a **single** retrieved chunk, every number/time in
  it appears verbatim **and** ≥ 75 % of its content words appear
  (`GROUND_THRESHOLD = 0.75`).
* **Unsupported claims** = everything else that fails that test.
* **Abstentions** ("does not contain…", "not specified…") are excluded from
  both numerator and denominator; a negative fact that *is* verbatim in the
  document counts as grounded.
* **Retrieval success** = answerable questions whose expected source appears
  in the retrieved evidence.
* **Correct refusals** = unanswerable questions refused (LegalLens: exact
  refusal / `insufficient`; baseline: a documented keyword detector).
* **Comparison** = `groundedness_improvement`, `retrieval_improvement`
  (percentage points, LegalLens − baseline).

### Cached run (currently in `data/evaluation_results.json`)

Ran at `2026-10-08T13:58:40Z`, 19 questions, 0 errors:

| Metric | Baseline (Retrieve→Generate) | LegalLens | Δ |
|---|---|---|---|
| Groundedness | 50.0 % (12/24 claims) | 90.0 % (18/20 claims) | **+40.0 pp** |
| Unsupported claims | 12 | 2 | −10 |
| Correct refusals | 4/4 | 3/4 | −1 |
| Retrieval success | 100 % (15/15) | 100 % (15/15) | 0.0 pp (shared retriever) |

Re-running overwrites these numbers with whatever the next run computes.

The UI's Evaluation tab shows the same data: comparison bars, refusal and
unsupported-claim blocks, and an expandable per-question table where each
percentage traces down to individual claims with their evidence ID and page.

---

## Running the research evaluation (categories A–E)

`app/evaluation/research_eval.py` (imported only by the two
`/api/evaluation/research*` endpoints) runs the six fixed research questions
RQ1–RQ6 from `app/evaluation/datasets.py` through the real
`research_document()` pipeline and derives five categories **per response** —
no score is hardcoded and nothing is auto-run:

| Category | What it measures | How (deterministic) |
|---|---|---|
| A — legal source retrieval | did the expected statute really surface? | expected act+section (RQ1/RQ3 expect BNS 2023 s.303) present among `sources` with `verified` |
| B — citation validity | cited-and-verified / cited | `verified sources / cited sources` per response, averaged |
| C — case connection correctness | connections that held on BOTH sides | `verified connections / produced connections`, averaged |
| D — hallucination rate | unverified items the pipeline had to exclude | `unverified / (claims + unverified)` per response, averaged |
| E — refusal / limitation correctness | did the run refuse or caveat when it should? | question's `expect_limit` flag vs the returned status + limitations + guilt check |

```powershell
curl.exe -X POST http://localhost:8000/api/evaluation/research/run   # real LLM calls
curl.exe http://localhost:8000/api/evaluation/research               # cached results
```

Results are persisted to `data/research_evaluation_results.json`
(run-lock protected, 409 on a concurrent run, per-question errors isolated).

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | serves the UI |
| GET | `/api/health` | model, document count, chunk count |
| POST | `/api/documents` | multipart `file` PDF → extract → chunk → embed |
| GET | `/api/documents` | list indexed documents |
| DELETE | `/api/documents/{id}` | remove document + its vectors |
| POST | `/api/chat` | `{question, doc_ids?}` → `{answer, decision, premises[], citations, refusal_stage?}` — grounded answer, cited correction, or refusal |
| POST | `/api/draft` | `{draft_type, instructions?, doc_ids?}` → draft + sources + missing information |
| POST | `/api/research` | `{question, case_context?, research_type}` → legal issue + verified research answer + sources/limitations |
| GET | `/api/evaluation` | cached evaluation results, or `{"evaluated": false}` |
| POST | `/api/evaluation/run` | execute a groundedness run (real LLM calls) |
| GET | `/api/evaluation/research` | cached research-evaluation results, or `{"evaluated": false}` |
| POST | `/api/evaluation/research/run` | execute the research evaluation over RQ1–RQ6 (real LLM calls; 409 if a run is in progress) |

---

## Limitations

* **No OCR** — scanned/image-only PDFs are rejected; text must be extractable.
* **Grounding is deterministic lexical traceability** — transparent and
  auditable, but it under-counts heavily paraphrased claims and can rarely
  over-count a claim whose nouns match while its predicate is fabricated.
  Claim-level rows are exposed so every number can be audited.
* **Drafting field detection is regex-based** and deliberately errs toward
  "missing" (a placeholder is safe; a wrong "available" could smuggle an
  invented value into a legal document).
* **Draft citation verification is conservative** — a claim that really is in
  the source but heavily paraphrased can fail the ≥ 75 % check and be
  replaced with `[NOT PROVIDED]`; the original wording is listed in
  `removed_claims`, so nothing is dropped silently. The drafting prompt asks
  for source wording to keep this rare.
* **Retrieval success is near "is the source in the index at all"** on this
  small corpus with `TOP_K = 6`; it discriminates more as documents are added.
* **No drafting-vs-baseline evaluation track yet** — documented as a future
  extension; per-response `claim_check` uses the same metric today.
* **Research legal sources are a bundled snapshot plus best-effort live
  search** — the snapshot (`app/research/legal_sources.json`) holds verbatim
  BNS/BNSS sections fetched from India Code at a point in time; the live
  India Code query (Tier 1 only) can fail or be slow and then degrades to the
  snapshot with a limitation note. There is **no live judgment source**, so
  judgment/precedent research honestly reports
  `NO_VERIFIED_PRECEDENT_FOUND` rather than inventing case law.
* **Research grounding is conservative** — a provision or connection whose
  wording does not literally match the retrieved text is dropped and listed
  under `unverified` (and demotes a GROUNDED status to RESEARCH WITH
  LIMITATIONS), preferring a visible gap over an unsupported legal claim.
* No authentication, payments, or deployment config — this is a hackathon
  prototype, and the drafting assistant is not a substitute for a lawyer.
