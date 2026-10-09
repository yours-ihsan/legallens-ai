"""FastAPI entrypoint.

Upload:  PDF -> extract pages -> chunk (page metadata attached) -> embed -> Chroma
Ask:     question -> retrieve top-k -> (threshold gate) -> evidence-only LLM ->
         validate citations -> grounded answer or exact refusal string
"""
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import chunker, pdf_extractor
from .answer_generator import answer_question
from .config import DOCS_JSON, MAX_PDF_BYTES, MAX_QUESTION_CHARS, MODEL, STATIC_DIR, UPLOADS_DIR
from .drafting import DraftRequest, draft_document
from .evaluation.evaluator import EvaluationBusy, get_results, run_evaluation
from .evaluation.research_eval import (ResearchEvalBusy, get_research_results,
                                       run_research_evaluation)
from .models import ChatRequest, DocumentInfo
from .retriever import get_store
from .research import ResearchRequest, research_document

app = FastAPI(title="LegalLens AI", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_registry_lock = threading.Lock()


def _load_registry() -> dict:
    if DOCS_JSON.exists():
        try:
            return json.loads(DOCS_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_registry(registry: dict) -> None:
    DOCS_JSON.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    with _registry_lock:
        docs = len(_load_registry())
    return {"status": "ok", "model": MODEL, "documents": docs, "chunks": get_store().count()}


@app.post("/api/documents", response_model=DocumentInfo)
def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 50 MB limit.")

    doc_id = uuid.uuid4().hex[:12]
    path = UPLOADS_DIR / f"{doc_id}.pdf"
    path.write_bytes(content)

    try:
        pages = pdf_extractor.extract_pages(path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not extract text: {exc}") from exc

    chunks = chunker.chunk_pages(pages, doc_id=doc_id, filename=filename)
    get_store().add_chunks(chunks)

    info = DocumentInfo(
        doc_id=doc_id,
        filename=filename,
        pages=len(pages),
        chunks=len(chunks),
        uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    with _registry_lock:
        registry = _load_registry()
        registry[doc_id] = info.model_dump()
        _save_registry(registry)
    return info


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    with _registry_lock:
        registry = _load_registry()
    docs = [DocumentInfo(**entry) for entry in registry.values()]
    docs.sort(key=lambda d: d.uploaded_at, reverse=True)
    return docs


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    with _registry_lock:
        registry = _load_registry()
        if doc_id not in registry:
            raise HTTPException(status_code=404, detail="Document not found.")
        registry.pop(doc_id)
        _save_registry(registry)
    get_store().delete_doc(doc_id)
    (UPLOADS_DIR / f"{doc_id}.pdf").unlink(missing_ok=True)
    return {"deleted": doc_id}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Question is too long.")
    # Sync endpoint on purpose: runs in FastAPI's thread pool so the blocking
    # LLM subprocess (~15-60s) never stalls the event loop.
    answer = answer_question(question, doc_ids=request.doc_ids or None)
    return answer.model_dump()


# ------------------------------------------------------------------ evaluation
# Groundedness-vs-baseline evaluation. Separate module (app/evaluation), never
# on the normal answering path. GET = cached results (or "not evaluated yet");
# POST = execute a run (real LLM calls; can take several minutes).
@app.get("/api/evaluation")
def evaluation_status() -> dict:
    return get_results()


@app.post("/api/evaluation/run")
def evaluation_run() -> dict:
    try:
        return run_evaluation()
    except EvaluationBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# Research evaluation extension (categories A-E, all computed from the actual
# run). GET = cached results; POST = execute a run (real LLM calls). Never run
# automatically.
@app.get("/api/evaluation/research")
def research_evaluation_status() -> dict:
    return get_research_results()


@app.post("/api/evaluation/research/run")
def research_evaluation_run() -> dict:
    try:
        return run_research_evaluation()
    except ResearchEvalBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# ------------------------------------------------------------------- drafting
# Legal Drafting workflow (app/drafting): retrieve -> sufficiency gate ->
# deterministic missing-information check -> evidence-only draft -> validate.
# Separate from /api/chat; the chat pipeline is untouched.
@app.post("/api/draft")
def create_draft(request: DraftRequest) -> dict:
    instructions = (request.instructions or "").strip()
    if len(instructions) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Additional instructions are too long.")
    try:
        result = draft_document(request.draft_type, instructions=instructions,
                                doc_ids=request.doc_ids or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Sync endpoint on purpose (like /api/chat): runs in the thread pool so the
    # blocking LLM subprocess never stalls the event loop.
    return result.model_dump()


# ---------------------------------------------------------------- research
# Legal Research workflow (app/research): EVIDENCE FIRST -> VERIFY -> REASON
# -> ANSWER. Deterministic legal-issue identification, case evidence through
# the EXISTING retriever, legal authority through the same Chroma store
# (bundled official India Code snapshot + best-effort live official search),
# one evidence packet, one evidence-only LLM call, strict citation
# verification. Separate from /api/chat and /api/draft; those pipelines are
# untouched (they query only the case collection).
@app.post("/api/research")
def legal_research(request: ResearchRequest) -> dict:
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Question is too long.")
    case_context = (request.case_context or "").strip()
    if len(case_context) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Case context is too long.")
    try:
        result = research_document(question, case_context=case_context,
                                   research_type=request.research_type,
                                   doc_ids=request.doc_ids or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Sync endpoint on purpose (like /api/chat): runs in the thread pool so the
    # blocking LLM subprocess never stalls the event loop.
    return result.model_dump()
