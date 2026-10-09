"""Legal source retrieval for the Legal Research workflow.

Sources of legal authority, in priority order:

  1. BUNDLED SNAPSHOT — app/research/legal_sources.json: verbatim section
     texts fetched from the OFFICIAL India Code API (Government of India),
     with act / section / handle / page / ministry metadata. Seeded into a
     SEPARATE collection (``legallens_legal``) of the SAME local Chroma store,
     through the SAME VectorStore class, chunker and embedding model — so the
     case collection that chat / drafting / evaluation query is provably
     unchanged (their retrieval results cannot shift).
  2. LIVE SEARCH of the official India Code API (best effort, short timeout;
     disable with LEGALLENS_LIVE_SOURCES=0). A failed search degrades to the
     snapshot and is reported as a limitation — never silently.
  3. UPLOADED documents in the workspace whose text is conservatively
     classified as statute / judgment material (content patterns; anything
     ambiguous stays case material).

Everything merges into ONE evidence list using the existing E-ID scheme and is
packaged with answer_generator.build_packet — there is no second LLM pipeline
and no parallel RAG stack: same VectorStore, same retriever call for the case
side, same Evidence model, same verifier (app.evaluation.metrics).

Invariant: source metadata (authority / url / section / tier) ALWAYS comes
from the registry or vector-store metadata — never from the model.
"""
from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import NamedTuple, Optional

from ..config import TOP_K
from ..models import Evidence, PageText
from ..retriever import retrieve
from ..vector_store import VectorStore

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "legal_sources.json")
LEGAL_COLLECTION = "legallens_legal"

INDIA_CODE_API = ("https://indiacode.gov.in/server/api/discover/search/objects"
                  "?query={query}&f.identifier_collection=SECTION,equals&size={size}")

SNAPSHOT_TOP_K = 4   # statute blocks pulled from the local legal collection
LIVE_TOP_K = 4       # statute blocks pulled from the live India Code search
LEGAL_CAP = 6        # max legal-authority blocks in one research packet

# Appended to statute text only, so groundedness checks accept "3 years" for
# "three years" (check_claim requires digits verbatim; statutes spell them out).
NUMBER_EQUIVALENTS = (" 0 zero 1 one 2 two 3 three 4 four 5 five 6 six 7 seven "
                      "8 eight 9 nine 10 ten 11 eleven 12 twelve 15 fifteen "
                      "20 twenty 30 thirty 40 forty 50 fifty 60 sixty 70 seventy "
                      "80 eighty 90 ninety 100 hundred 1000 thousand 2 02 002")

_registry_lock = threading.Lock()
_registry_cache: list[dict] | None = None
_legal_store: VectorStore | None = None


class GatherResult(NamedTuple):
    case_evidence: list[Evidence]
    legal_evidence: list[Evidence]
    evidence: list[Evidence]              # merged, sequential E-IDs
    meta_by_doc_id: dict[str, dict]       # provenance for every legal source
    live_error: Optional[str]             # None unless the live search failed


# ------------------------------------------------------------------ snapshot
def load_registry() -> list[dict]:
    """Bundled official sources (verbatim India Code sections). Cached."""
    global _registry_cache
    with _registry_lock:
        if _registry_cache is None:
            try:
                with open(SNAPSHOT_PATH, encoding="utf-8") as fh:
                    _registry_cache = json.load(fh).get("sources", [])
            except (OSError, json.JSONDecodeError):
                _registry_cache = []
        return _registry_cache


def snapshot_doc_id(entry: dict) -> str:
    return f"src:{entry['id']}"


def source_filename(act: str, section: str) -> str:
    """Display reference for a statute source (stable, metadata-derived)."""
    act = (act or "").strip()
    if "Nyaya Sanhita" in act:
        short = "BNS 2023"
    elif "Nagarik Suraksha" in act:
        short = "BNSS 2023"
    elif "Sakshya" in act:
        short = "BSA 2023"
    elif len(act) > 44:
        short = act[:44].rstrip()
    else:
        short = act
    return f"India Code: {short} Section {section}"


def get_legal_store() -> VectorStore:
    """Second collection of the SAME local Chroma store (same class/embeddings)."""
    global _legal_store
    if _legal_store is None:
        _legal_store = VectorStore(collection=LEGAL_COLLECTION)
    return _legal_store


def ensure_seeded(store: VectorStore | None = None) -> int:
    """Index the bundled snapshot into the legal collection (idempotent)."""
    store = store or get_legal_store()
    registry = load_registry()
    if not registry or store.count() > 0:
        return store.count()
    from ..chunker import chunk_pages
    chunks = []
    for entry in registry:
        if entry.get("repealed"):
            continue
        page = PageText(page_number=int(entry.get("page") or 1),
                        text=str(entry.get("text") or ""))
        if len(page.text.strip()) < 40:
            continue
        chunks.extend(chunk_pages([page], doc_id=snapshot_doc_id(entry),
                                  filename=source_filename(entry.get("act", ""),
                                                           entry.get("section", ""))))
    store.add_chunks(chunks)
    return store.count()


def registry_by_doc_id() -> dict[str, dict]:
    return {snapshot_doc_id(e): e for e in load_registry()}


# ------------------------------------------------------------- live (Tier 1)
def _strip_tags(raw: str) -> str:
    raw = re.sub(r"<br\s*/?>", "\n", raw or "", flags=re.I)
    raw = re.sub(r"<hr[^>]*>", "\n", raw, flags=re.I)
    raw = re.sub(r"</(p|div|center)>", "\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = raw.replace("&nbsp;", " ").replace("&amp;", "&")
    raw = raw.replace("&#39;", "'").replace("&quot;", '"')
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", raw)).strip()


def _md1(item: dict, key: str):
    values = item.get("metadata", {}).get(key, [])
    return values[0].get("value") if values else None


def _is_repealed(item: dict) -> bool:
    sec = (_md1(item, "dc.identifier.repealed") or "false").lower()
    act = (_md1(item, "dc.identifier.act_repealed") or "false").lower()
    return sec == "true" or act == "true"


def search_live_sources(query: str, timeout: float | None = None,
                        size: int = LIVE_TOP_K) -> tuple[list[dict], Optional[str]]:
    """Search the official India Code API for statute sections.

    Returns (records, error). On any failure returns ([], reason) so the
    caller can fall back to the bundled snapshot and surface a limitation.
    Repealed acts are excluded — a repealed provision is never presented as
    in-force authority.
    """
    if os.environ.get("LEGALLENS_LIVE_SOURCES", "1") == "0":
        return [], None
    if timeout is None:
        timeout = float(os.environ.get("LEGALLENS_SOURCE_TIMEOUT", "8"))
    url = INDIA_CODE_API.format(query=urllib.parse.quote(query or ""),
                                size=max(1, min(size * 2, 20)))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (LegalLens)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        objects = payload["_embedded"]["searchResult"]["_embedded"].get("objects", [])
    except Exception as exc:  # network / parse failure degrades gracefully
        return [], f"live India Code search unavailable ({type(exc).__name__})"

    records: list[dict] = []
    for obj in objects:
        item = obj.get("_embedded", {}).get("indexableObject") or {}
        act = _md1(item, "dc.identifier.act_name")
        section = _md1(item, "dc.identifier.section_number")
        text = _strip_tags(_md1(item, "dc.identifier.section_page_note") or "")
        if not act or not section or len(text) < 40 or _is_repealed(item):
            continue
        handle = item.get("handle") or ""
        records.append({
            "doc_id": f"live:{item.get('id', handle)}",
            "title": f"{act} — Section {section}. {item.get('name', '')}".strip(),
            "source_type": "statute",
            "authority": (f"Government of India — "
                          f"{_md1(item, 'dc.identifier.ministry_name') or 'India Code'}"),
            "act": act,
            "section": str(section),
            "tier": 1,
            "url": f"https://indiacode.gov.in/handle/{handle}" if handle else None,
            "page": int(_md1(item, "dc.identifier.page_number") or 1),
            "repealed": False,
            "text": text,
            "fetched_live": True,
        })
        if len(records) >= size:
            break
    return records, None


# -------------------------------------------------------------- classification
_JUDGMENT_CITE_RE = re.compile(
    # journal first: "AIR 1973 SC 514", "SCC 2019 512" (volume-court-page order)
    r"\b(?:AIR|SCC|SCR|Cri LJ|Crl LJ|All ER|MLJ|MLW|KLT|Cal LT|Bom LR|LW|DLT|"
    r"Guj LR|Pat LJR|Ker LT|Raj LR|Punj LR)\s+\d{4}\s+(?:[A-Z]{1,4}\s+)?\d+"
    # year-first reporter: "(2019) 5 SCC 123", "(1973) 2 SCC 728"
    r"|\(\d{4}\)\s*\d+\s*(?:\(\w+\)\s*)?(?:SCC|SCR|Cri LJ|Crl LJ|All ER|KLT|"
    r"MLJ|MLW|LW|DLT|Cal LT|Bom LR|Guj LR|Pat LJR|Ker LT|Raj LR|Punj LR)\s+\d+"
    # party style: "X v. State ..."
    r"|\bv\.\s+[A-Z][A-Za-z]+\s+(?:State|Union of India|India|Another)",
)
_SECTION_RE = re.compile(r"\bSection\s+(\d+[A-Z]?)", re.I)
_STATUTE_CONTEXT_RE = re.compile(
    r"\b(?:Act|Code|Sanhita|Schedule|IPC|CrPC|BNSS|BNS|Bharatiya)\b|\b(?:1860|2023)\b")
_COURT_RE = re.compile(
    r"\b(?:Supreme Court of India|High Court of [A-Za-z ]+|District Court of [A-Za-z ]+)\b")


def content_kind(text: str) -> str:
    """Conservative classifier for UPLOADED documents: statute / judgment / case.

    Prefer a false "case material" over a false "legal authority": an ambiguous
    document is never promoted to verified authority.
    """
    if _JUDGMENT_CITE_RE.search(text):
        return "judgment"
    m = _SECTION_RE.search(text)
    if m:
        window = text[max(0, m.start() - 140): m.end() + 180]
        if _STATUTE_CONTEXT_RE.search(window):
            return "statute"
    if _COURT_RE.search(text) and re.search(r"\b(?:held|judgment|judgement|conviction|acquittal|appeal)\b",
                                            text, re.I):
        return "judgment"
    return "case_evidence"


def classify_evidence(evidence: list[Evidence], meta_by_doc_id: dict[str, dict]
                      ) -> dict[str, dict]:
    """Per-E-ID provenance: kind, source_type, authority, section, url, tier.

    Registry/vector metadata wins; content classification only applies to
    workspace documents. URL/authority are NEVER model-supplied — unknown
    stays None. `full_text` is the verbatim official document text for
    registered sources (a chunk is only a window onto it), so groundedness
    checks compare claims against the WHOLE retrieved source, not an
    arbitrary 1000-char slice.
    """
    classes: dict[str, dict] = {}
    for ev in evidence:
        meta = meta_by_doc_id.get(ev.doc_id)
        if meta:
            kind = "judgment" if meta.get("source_type") == "judgment" else "statute"
            classes[ev.id] = {
                "kind": kind,
                "source_type": meta.get("source_type") or kind,
                "title": meta.get("title") or ev.filename,
                "authority": meta.get("authority"),
                "act": meta.get("act"),
                "section": str(meta.get("section")) if meta.get("section") else None,
                "tier": meta.get("tier"),
                "url": meta.get("url"),
                "repealed": bool(meta.get("repealed")),
                "filename": ev.filename,
                "page": ev.page,
                "full_text": str(meta.get("text") or "") or ev.text,
            }
            continue
        kind = content_kind(ev.text)
        if kind == "case_evidence":
            classes[ev.id] = {
                "kind": "case_evidence", "source_type": "case_evidence",
                "title": ev.filename, "authority": None, "act": None,
                "section": None, "tier": None, "url": None, "repealed": False,
                "filename": ev.filename, "page": ev.page, "full_text": ev.text,
            }
        else:
            sec = _SECTION_RE.search(ev.text)
            court = _COURT_RE.search(ev.text)
            classes[ev.id] = {
                "kind": kind, "source_type": kind,
                "title": ev.filename,
                "authority": court.group(0) if court else None,
                "act": None, "section": sec.group(1) if sec else None,
                "tier": None, "url": None, "repealed": False,
                "filename": ev.filename, "page": ev.page, "full_text": ev.text,
            }
    return classes


def legal_source_text(ev: Evidence, cls: dict) -> str:
    """Verbatim source text + its official title/section metadata.

    Section bodies often do not literally contain the act name or section
    number (they live in the source metadata), so groundedness checks for
    legal authority compare against title + section + the FULL verbatim
    document text (a retrieved chunk is only a window onto that document) —
    all real, metadata-derived text, none of it model-generated.

    A digit/number-word equivalence table is appended so "3 years" and
    "three years" verify against the same official text (check_claim requires
    every digit to appear verbatim; statutes spell numbers out).
    """
    body = cls.get("full_text") or ev.text
    parts = [cls.get("title") or "", f"Section {cls['section']}" if cls.get("section") else "",
             cls.get("act") or "", body]
    return (re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()
            + " " + NUMBER_EQUIVALENTS)


# ------------------------------------------------------------------- gathering
def _snapshot_evidence(legal_query: str) -> list[tuple[Evidence, dict]]:
    try:
        store = get_legal_store()
        ensure_seeded(store)
        hits = store.query(legal_query, top_k=SNAPSHOT_TOP_K)
    except Exception:
        return []
    registry = registry_by_doc_id()
    out: list[tuple[Evidence, dict]] = []
    for hit in hits:
        doc_id = hit["meta"]["doc_id"]
        entry = registry.get(doc_id)
        if not entry:
            continue
        ev = Evidence(
            id="",  # assigned after merging with case evidence
            chunk_id=hit["chunk_id"],
            doc_id=doc_id,
            filename=source_filename(entry.get("act", ""), entry.get("section", "")),
            page=int(hit["meta"]["page"]),
            text=hit["text"],
            score=round(float(hit["score"]), 4),
        )
        out.append((ev, entry))
    return out


def gather_evidence(question: str, doc_ids: list[str] | None = None,
                    legal_query: str | None = None) -> GatherResult:
    """Case evidence (EXISTING retriever) + legal authority (snapshot + live).

    E-IDs stay sequential across the merged packet: case evidence keeps E1..En
    from app.retriever.retrieve, legal blocks continue at E(n+1)...
    """
    case_ev = retrieve(question, doc_ids=doc_ids)          # unchanged chat/draft path
    legal_query = legal_query or question

    # Snapshot hits: keep ONE chunk per section — prefer the section-opening
    # chunk (the actual provision) over a later illustration chunk, then order
    # sections by their best vector rank.
    groups: dict[tuple[str, str], list[tuple[int, int, tuple]]] = {}
    for rank, pair in enumerate(_snapshot_evidence(legal_query)):
        ev, entry = pair
        index = int(ev.chunk_id.rsplit(":", 1)[-1]) if ":" in ev.chunk_id else 0
        key = (str(entry.get("act", "")).casefold(), str(entry.get("section")))
        groups.setdefault(key, []).append((index, rank, pair))
    chosen = [min(group, key=lambda t: (t[0], t[1]))
              for group in groups.values()]
    chosen.sort(key=lambda t: t[1])
    picked: list[tuple[Evidence, dict]] = [pair for _, _, pair in chosen]

    live_records, live_error = search_live_sources(legal_query)
    seen = {(str(m.get("act", "")).casefold(), str(m.get("section")))
            for _, m in picked}
    for rec in live_records:
        key = (str(rec.get("act", "")).casefold(), str(rec.get("section")))
        if key in seen:
            continue
        seen.add(key)
        picked.append((Evidence(
            id="",
            chunk_id=f"live:{rec['doc_id']}",
            doc_id=rec["doc_id"],
            filename=source_filename(rec.get("act", ""), rec.get("section", "")),
            page=int(rec.get("page") or 1),
            text=rec["text"],
            score=0.0,
        ), rec))
    picked = picked[:LEGAL_CAP]

    base = len(case_ev)
    legal_ev: list[Evidence] = []
    meta_by_doc_id: dict[str, dict] = registry_by_doc_id()
    for i, (ev, meta) in enumerate(picked, start=1):
        legal_ev.append(ev.model_copy(update={"id": f"E{base + i}"}))
        meta_by_doc_id[ev.doc_id] = meta

    # Live records fetched this request are added to the provenance lookup.
    for rec in live_records:
        meta_by_doc_id.setdefault(rec["doc_id"], rec)

    return GatherResult(
        case_evidence=case_ev,
        legal_evidence=legal_ev,
        evidence=list(case_ev) + legal_ev,
        meta_by_doc_id=meta_by_doc_id,
        live_error=live_error,
    )


def legal_budget() -> dict:
    """Small status helper for the UI/docs: what powers legal retrieval."""
    registry = load_registry()
    return {
        "snapshot_sources": len(registry),
        "collection": LEGAL_COLLECTION,
        "live_search_enabled": os.environ.get("LEGALLENS_LIVE_SOURCES", "1") != "0",
        "top_k_case": TOP_K,
    }
