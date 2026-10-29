"""FastAPI entrypoint.

Upload:  PDF -> extract pages -> chunk (page metadata attached) -> embed -> Chroma
Ask:     question -> retrieve top-k -> (threshold gate) -> evidence-only LLM ->
         validate citations -> grounded answer or exact refusal string
"""
import json
import logging
import os
import secrets
import threading
from typing import Optional
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import Cookie, Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from .document_classifier import classify_legal_case_document
auth_logger = logging.getLogger("legallens.auth")

from . import auth, chunker, config, pdf_extractor
from .auth import AuthResponse, AuthUser, LoginRequest, RegisterRequest, RegisterResponse
from .answer_generator import answer_question
from .config import (
    DOCS_JSON,
    MAX_PDF_BYTES,
    MAX_QUESTION_CHARS,
    MODEL,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    STATIC_DIR,
    UPLOADS_DIR,
)
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


def get_current_user_optional(
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
    authorization: Optional[str] = Header(None),
) -> Optional[AuthUser]:
    token = session_token
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        return None
    return auth.get_session_user(token)


@app.get("/")
def index(current_user: Optional[AuthUser] = Depends(get_current_user_optional)) -> Response:
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/login")
def login_page(current_user: Optional[AuthUser] = Depends(get_current_user_optional)) -> Response:
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    resp = FileResponse(STATIC_DIR / "login.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


# ------------------------------------------------------------------ auth routes
@app.post("/api/auth/register", response_model=RegisterResponse)
def register(req: RegisterRequest) -> RegisterResponse:
    try:
        user = auth.register_user(name=req.name, email=req.email, password=req.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Success message without auto-bypassing password authentication
    return RegisterResponse(
        status="registered",
        message="Account created successfully! Please sign in with your email and password.",
        user=user,
    )


@app.post("/api/auth/login", response_model=AuthResponse)
def login(req: LoginRequest, response: Response) -> AuthResponse:
    user = auth.authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = auth.create_session(user)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return AuthResponse(user=user, token=token)


@app.post("/api/auth/logout")
def logout(
    response: Response,
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
    authorization: Optional[str] = Header(None),
) -> dict:
    token = session_token
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if token:
        auth.invalidate_session(token)
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")
    return {"status": "logged_out"}


@app.get("/api/auth/me")
def current_user_info(current_user: Optional[AuthUser] = Depends(get_current_user_optional)) -> dict:
    if not current_user:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": current_user.model_dump()}


# ----------------------------------------------------------------- Google OAuth
def _get_google_oauth_config(request: Request) -> tuple[str, str, str]:
    """Dynamically resolves Google OAuth credentials and redirect URI from env or config."""
    client_id = (
        os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        or getattr(config, "GOOGLE_CLIENT_ID", "").strip()
    )
    client_secret = (
        os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
        or getattr(config, "GOOGLE_CLIENT_SECRET", "").strip()
    )

    # In live app (not unit tests), dynamically reload .env if credentials are not found in os.environ
    if (not client_id or not client_secret) and ("PYTEST_CURRENT_TEST" not in os.environ):
        config.load_environment(override=True)
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()

    default_redirect = f"{str(request.base_url).rstrip('/')}/api/auth/google/callback"
    redirect_uri = (
        os.environ.get("GOOGLE_REDIRECT_URI", "").strip()
        or getattr(config, "GOOGLE_REDIRECT_URI", "").strip()
        or default_redirect
    )

    auth_logger.info(
        "Google OAuth configuration: client_id_configured=%s, client_secret_configured=%s, redirect_uri=%s",
        bool(client_id),
        bool(client_secret),
        redirect_uri,
    )
    return client_id, client_secret, redirect_uri


@app.get("/api/auth/google/url")
def google_auth_url(request: Request, response: Response) -> dict:
    """Returns the Google OAuth 2.0 authorization URL or clear inline error message."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"

    try:
        client_id, client_secret, redirect_uri = _get_google_oauth_config(request)
        if not client_id or not client_secret:
            auth_logger.warning(
                "Google OAuth URL requested but credentials missing: client_id_configured=%s, client_secret_configured=%s",
                bool(client_id),
                bool(client_secret),
            )
            return {
                "configured": False,
                "auth_url": None,
                "client_id": None,
                "redirect_uri": redirect_uri,
                "message": "Google sign-in is not configured on this server. Please configure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env.",
            }

        state = secrets.token_urlsafe(16)
        params = {
            "client_id": client_id,
            "response_type": "code",
            "scope": "openid email profile",
            "redirect_uri": redirect_uri,
            "state": state,
            "prompt": "select_account",
            "access_type": "offline",
        }
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
        auth_logger.info("Generated Google OAuth URL successfully for redirect_uri=%s", redirect_uri)
        return {
            "configured": True,
            "auth_url": auth_url,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
        }
    except Exception as exc:
        auth_logger.error("Error generating Google OAuth URL: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error generating Google OAuth URL.") from exc


@app.get("/api/auth/google/callback")
def google_auth_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
) -> Response:
    """Handles Google OAuth authorization code callback, verifies identity, and establishes session."""
    auth_logger.info(
        "Google OAuth callback received: code_present=%s, state_present=%s, error=%s",
        bool(code),
        bool(state),
        error,
    )
    client_id, client_secret, redirect_uri = _get_google_oauth_config(request)
    if error:
        err_msg = urllib.parse.quote(f"Google authentication was cancelled or failed: {error}")
        return RedirectResponse(url=f"/login?error={err_msg}", status_code=status.HTTP_302_FOUND)

    if not code:
        err_msg = urllib.parse.quote("Missing authorization code from Google.")
        return RedirectResponse(url=f"/login?error={err_msg}", status_code=status.HTTP_302_FOUND)

    if not client_id or not client_secret:
        auth_logger.warning("Google callback invoked but credentials missing.")
        err_msg = urllib.parse.quote("Google OAuth credentials are not configured on this server.")
        return RedirectResponse(url=f"/login?error={err_msg}", status_code=status.HTTP_302_FOUND)

    # Exchange authorization code for tokens
    try:
        token_url = "https://oauth2.googleapis.com/token"
        payload = {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        with httpx.Client(timeout=15.0) as http_client:
            token_res = http_client.post(token_url, data=payload)
            if token_res.status_code != 200:
                auth_logger.warning("Google token exchange returned non-200: %s", token_res.status_code)
                err_msg = urllib.parse.quote("Failed to exchange authorization code with Google.")
                return RedirectResponse(url=f"/login?error={err_msg}", status_code=status.HTTP_302_FOUND)

            token_data = token_res.json()
            access_token = token_data.get("access_token")

            # Verify and retrieve userinfo from Google's OpenID Connect endpoint
            userinfo_res = http_client.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if userinfo_res.status_code != 200:
                auth_logger.warning("Google userinfo returned non-200: %s", userinfo_res.status_code)
                err_msg = urllib.parse.quote("Failed to fetch verified user profile from Google.")
                return RedirectResponse(url=f"/login?error={err_msg}", status_code=status.HTTP_302_FOUND)

            user_info = userinfo_res.json()
            email = user_info.get("email")
            google_sub = user_info.get("sub")
            name = user_info.get("name")
            email_verified = user_info.get("email_verified", False)

            if not email or not google_sub:
                err_msg = urllib.parse.quote("Incomplete profile data received from Google.")
                return RedirectResponse(url=f"/login?error={err_msg}", status_code=status.HTTP_302_FOUND)

            if not email_verified:
                err_msg = urllib.parse.quote("Your Google account email is not verified.")
                return RedirectResponse(url=f"/login?error={err_msg}", status_code=status.HTTP_302_FOUND)

    except Exception as exc:
        auth_logger.error("Exception during Google OAuth authentication: %s", exc, exc_info=True)
        err_msg = urllib.parse.quote(f"Network error during Google authentication: {str(exc)}")
        return RedirectResponse(url=f"/login?error={err_msg}", status_code=status.HTTP_302_FOUND)

    # STRICT REGISTRATION ENFORCEMENT:
    # A user who has NOT registered an account in LegalLens AI must NEVER be allowed
    # to log in through Continue with Google. Selecting a Google account is not the same as registering.
    try:
        user = auth.authenticate_google_user(email=email, google_id=google_sub, name=name)
    except Exception as exc:
        auth_logger.error("Exception during user lookup for Google OAuth: %s", exc, exc_info=True)
        err_msg = urllib.parse.quote(f"Authentication error: {str(exc)}")
        return RedirectResponse(url=f"/login?error={err_msg}", status_code=status.HTTP_302_FOUND)

    if not user:
        auth_logger.warning("Google sign-in rejected: %s is not registered in LegalLens AI", email)
        err_msg = urllib.parse.quote("Account not found. Please create an account first.")
        return RedirectResponse(url=f"/login?error={err_msg}", status_code=status.HTTP_302_FOUND)

    # Establish authenticated session for registered user ONLY
    token = auth.create_session(user)
    auth_logger.info("Google sign-in successful for registered user: %s (id=%s)", user.email, user.user_id)
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


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

    # Classify the extracted PDF text before creating chunks or indexing.
    classification = classify_legal_case_document(
        filename,
        "\\n".join(
            getattr(page, "text", str(page))
            for page in pages
        ),
    )

    if not classification.accepted:
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=classification.reason,
        )

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
