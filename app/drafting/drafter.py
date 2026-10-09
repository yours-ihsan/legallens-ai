"""Legal Drafting workflow — EVIDENCE FIRST → DRAFT SECOND.

Pipeline (mirrors the chat architecture, reuses its parts):

    document -> retrieve (app.retriever)
             -> sufficiency gate (REFUSAL_THRESHOLD, same as chat)
             -> deterministic field detection (requirements.py patterns)
             -> evidence-only drafting LLM call (build_packet from answer_generator)
             -> pure validation (hallucinated IDs / unverified pages rejected)
             -> citation verification: for every marked claim, check that THE
                CITED chunk (not just some chunk) supports it; re-point wrong
                citations, drop unsupported markers, remove unsupported claims
                ([NOT PROVIDED]) — verify_citations()
             -> claim-level traceability check (app.evaluation.metrics)
             -> DRAFT + SOURCES + MISSING INFORMATION

Safety invariants (same as the chat trust boundary):
  * The LLM receives ONLY the evidence packet + Python-computed field lists.
  * It never decides what is available or missing — Python does.
  * Page numbers / filenames in the response come from vector-store metadata;
    the model only emits [E#] markers, which Python maps to (filename, page).
  * Hallucinated evidence IDs, unverified page numbers or LLM/parse failures
    collapse the result to status "insufficient" with the exact REFUSAL string.
  * Missing information is reported, never invented; the draft must carry
    "[LABEL — NOT PROVIDED]" placeholders for it.
  * CITATION CORRECTNESS: every [E#] on a factual claim is verified against
    that exact chunk. A citation pointing at a chunk that does not contain the
    claim is re-pointed at one that does; a claim no retrieved chunk supports
    is removed from the draft and recorded as "[NOT PROVIDED]" (never asserted
    without support).
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from ..answer_generator import REFUSAL, build_packet
from ..config import REFUSAL_THRESHOLD
from ..evaluation.metrics import check_claim, evaluate_answer_claims, is_abstention
from ..llm import LLMError, extract_json, run_llm
from ..models import Evidence
from ..retriever import retrieve
from .requirements import DRAFT_TYPES

# ---------------------------------------------------------------- models
class DraftRequest(BaseModel):
    draft_type: str
    instructions: Optional[str] = None
    doc_ids: Optional[list[str]] = None


class DraftSource(BaseModel):
    evidence_id: str
    filename: str
    page: int
    snippet: str


class AvailableField(BaseModel):
    label: str
    evidence_id: Optional[str] = None
    filename: Optional[str] = None
    page: Optional[int] = None
    snippet: Optional[str] = None


class UnverifiedClaim(BaseModel):
    claim: str
    reason: str


class ClaimCheck(BaseModel):
    claims_total: int = 0
    claims_supported: int = 0
    claims_unverified: int = 0
    unverified_claims: list[UnverifiedClaim] = []


class DraftResponse(BaseModel):
    draft_type: str
    status: str  # "grounded" | "missing" | "insufficient"
    draft: str
    sources: list[DraftSource] = []
    missing_information: list[str] = []
    available_information: list[AvailableField] = []
    claim_check: ClaimCheck = ClaimCheck()
    citation_corrections: list[str] = []
    removed_claims: list[str] = []
    reason: Optional[str] = None
    refusal_stage: Optional[str] = None


# ---------------------------------------------------------------- prompt
DRAFT_SYSTEM_PROMPT = f"""You are a grounded legal drafting assistant.

Generate the requested legal document using ONLY the verified facts and
evidence supplied in the EVIDENCE packet below. The evidence packet is the
entire universe of information available to you.

STRICT RULES:
1. Do not invent or infer factual details. Never invent: names, dates, times,
   addresses, case numbers, FIR numbers, police stations, witnesses, evidence,
   legal sections, statutes, judgments, citations, or procedural history.
2. Every factual statement in the draft must be supported by the evidence;
   end each such statement with an [E1]-style marker referencing the evidence
   that supports it. Place the marker INSIDE the sentence, before its final
   full stop ("The complainant is Arun Mehta [E1]."), never after the sentence
   ("...Arun Mehta. [E1]"). Before attaching a marker, verify that the cited
   block's own text literally contains that sentence's facts — do not cite a
   block merely because it contains surrounding text. Keep each sentence
   within ONE evidence block — never merge facts from different blocks into a
   single sentence — and use the source's own wording (quote or minimally
   adapt) for names, dates, times and events so every statement stays
   verifiably traceable.
3. Never write page numbers or filenames inside the draft text.
4. For every entry in MISSING INFORMATION insert a clearly labeled placeholder
   using exactly: [UPPERCASE LABEL — NOT PROVIDED]. Never fill a placeholder
   with a guessed value.
5. Legal authorities: cite a legal section, statute, judgment or precedent ONLY
   if it appears literally in the evidence. If legal authorities are required
   but not present, write exactly: "Verified legal authority is not available
   in the provided documents."
6. Do not infer guilt and do not present allegations as established facts.
   Attribute contested facts to their source ("the source states ...").
7. Follow the REQUIRED STRUCTURE, but add no jurisdiction-specific legal
   language, grounds or claims unless the evidence supports them.
8. Set "insufficient" to true ONLY when the evidence is too weak to produce
   anything faithful. In that case set "draft" to "" and "evidence_used" to [].
9. Respond with ONLY a JSON object and no other text, using exactly this schema:
{{"draft": string, "insufficient": boolean, "confidence": number, "evidence_used": ["E1", ...]}}"""


# ------------------------------------------------- deterministic field check
def _snippet(text: str, start: int, end: int) -> str:
    lo, hi = max(0, start - 60), min(len(text), end + 90)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def detect_fields(evidence: list[Evidence], fields: list[dict]) -> tuple[list[dict], list[str]]:
    """Which required fields does the source actually contain?

    Deterministic regex scan over the retrieved evidence — the model is never
    consulted. Returns (available, missing_labels); each available field keeps
    the supporting evidence ID / filename / page from vector-store metadata.

    When several chunks match, a VALUE-bearing match wins over a bare mention
    of the label word ("Complainant: Arun Mehta" beats a demo-question line
    that merely says "the complainant"), so the recorded provenance points at
    the chunk that actually contains the information. Ties keep retrieval
    order. Prefer a false "missing" over a false "available".
    """
    available: list[dict] = []
    missing: list[str] = []
    for field in fields:
        best = None  # (score, first-seen-order, match, evidence)
        order = 0
        for pattern in field["patterns"]:
            for ev in evidence:
                m = re.search(pattern, ev.text, flags=re.I)
                order += 1
                if not m:
                    continue
                # Does a value follow the match (": Value" / a number) or sit
                # inside it? A short word gap is allowed ("Incident date: 18").
                near = ev.text[m.end():m.end() + 40]
                score = 0
                if re.search(r"[:\-\u2013]\s*[A-Z0-9]", m.group(0)) \
                        or re.search(r"\d", m.group(0)) \
                        or re.match(r"(?:[,.\s]*[A-Za-z]{0,10}\s*[:\-\u2013]\s*[A-Z0-9]"
                                    r"|[,.\s]*\d)", near):
                    score = 1
                if best is None or score > best[0]:
                    best = (score, order, m, ev)
        if best:
            _, _, m, ev = best
            available.append({"label": field["label"],
                              "evidence_id": ev.id,
                              "filename": ev.filename,
                              "page": ev.page,
                              "snippet": _snippet(ev.text, m.start(), m.end())})
        else:
            missing.append(field["label"])
    return available, missing


# ---------------------------------------------------------------- validation
_MARKER_LINE_RE = re.compile(r"\[E\d+\](?:\s*\[E\d+\])*\s*$", re.I)
_ANY_MARKER_RE = re.compile(r"\[(E\d+)\]", re.I)
_NUMBERING_RE = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+")


def normalize_markers(draft: str) -> str:
    """Attach orphan evidence markers to the sentence they belong to.

    Models often write "The complainant is Arun Mehta. [E3]" — a marker after
    the full stop would otherwise split off as its own "sentence", leaving the
    real claim unmarked (and its citation unchecked). Two deterministic passes:
    fold marker-only lines into the line above, then pull trailing markers in
    front of the sentence-ending punctuation. Everything else is untouched.
    Idempotent.
    """
    folded: list[str] = []
    for raw in draft.splitlines():
        s = raw.strip()
        if _MARKER_LINE_RE.fullmatch(s):
            j = len(folded) - 1
            while j >= 0 and not folded[j].strip():
                j -= 1
            if j >= 0:
                folded[j] = folded[j].rstrip() + " " + s
                continue
        folded.append(raw)
    return "\n".join(
        re.sub(r"([.!?])\s+(\[E\d+\](?:\s*\[E\d+\])*)", r" \2\1", line)
        for line in folded
    )


def _classify_sentence(s: str) -> tuple[str, str]:
    """What kind of marked line/sentence is this?

    Returns ("claim", sentence-with-marker) | ("label", value-text)
          | ("skip", ""). Deterministic — no model involved.

    * Headings (ALL-CAPS), bare labels ("MOST RESPECTFULLY SUBMITTED:"),
      placeholder lines ("... NOT PROVIDED") and unmarked boilerplate assert
      nothing about the source -> "skip".
    * Form-field label lines ("Case/Reference No.: LL-CR-2026-014 [E1]") carry
      a VALUE, not a prose claim -> "label" (verified on its value).
      Labels containing digits (timestamps like "21:40 ...") are real claims.
    * Everything else marked with [E#] is a factual claim candidate.
    """
    s = s.strip()
    if not s or "NOT PROVIDED" in s.upper():
        return "skip", ""
    if s == s.upper() and any(c.isalpha() for c in s):
        return "skip", ""  # heading, e.g. "BAIL APPLICATION"
    if s.endswith(":"):
        return "skip", ""  # label line with no value
    if "[E" not in s.upper():
        return "skip", ""  # unmarked boilerplate
    label = s.partition(":")[0]
    if ":" in s and len(label) <= 40 and not any(ch.isdigit() for ch in label):
        # Numbering is formatting, not a factual number.
        value = _NUMBERING_RE.sub("", s.partition(":")[2]).strip()
        return "label", value
    return "claim", _NUMBERING_RE.sub("", s)


def claim_sentences(draft: str) -> list[str]:
    """Factual claims in a draft = sentences the model explicitly marked with
    an [E#] evidence marker (orphan markers attached first — see
    normalize_markers). Structure headings (ALL-CAPS / label lines / numbered
    section titles), placeholder lines ("... NOT PROVIDED") and boilerplate
    without markers are not factual assertions about the source, so they are
    not claim candidates. Deterministic — no model involved.
    """
    kept: list[str] = []
    for line in normalize_markers(draft).splitlines():
        for sent in re.split(r"(?<=[.!?])\s+", line):
            kind, payload = _classify_sentence(sent)
            if kind == "claim":
                kept.append(payload)
    return kept


def _marker_ids(text: str) -> list[str]:
    return [m.upper() for m in _ANY_MARKER_RE.findall(text)]


def verify_citations(draft: str, evidence: list[Evidence]
                     ) -> tuple[str, list[str], list[str]]:
    """VALIDATION STEP: claim -> cited evidence -> verify the evidence supports it.

    For every marked sentence, check each attached [E#] against THAT exact
    chunk (check_claim with the single cited chunk — never the best chunk
    across the packet):
      * if a cited chunk supports the claim, keep the sentence and drop any
        other marker on it that does not support it;
      * if no cited chunk supports it, re-point the markers at the first
        retrieved chunk that does (source hierarchy: a chunk that actually
        contains the fact, in retrieval order);
      * if NO retrieved chunk supports it, remove the claim from the draft and
        write [NOT PROVIDED] in its place — an unsupported assertion is never
        displayed. The claim is recorded for the response.

    Label lines ("Case/Reference No.: LL-CR-2026-014 [E3]") get the same check
    on their VALUE. Abstention sentences ("the source does not identify ...")
    assert the ABSENCE of a fact, so they are only re-pointed when a chunk
    contains them verbatim, otherwise left as-is. Deterministic — reuses
    app.evaluation.metrics; no model involved.

    Returns (draft, citation_corrections, removed_claims).
    """
    by_id = {ev.id: ev for ev in evidence}
    out: list[str] = []
    corrections: list[str] = []
    removed: list[str] = []

    def verdict(claim: str, ev: Evidence) -> dict:
        return check_claim(claim, [ev])

    def display(text: str) -> str:
        return re.sub(r"\s*\[E\d+\]", "", text, flags=re.I).strip()

    for raw in normalize_markers(draft).splitlines():
        if "[E" not in raw.upper() or "NOT PROVIDED" in raw.upper():
            out.append(raw)
            continue

        prefix_m = re.match(r"^(\s*\d+(?:\.\d+)*[.)]?\s+)(.*)$", raw)
        prefix, rest = (prefix_m.group(1), prefix_m.group(2)) if prefix_m else ("", raw)
        parts = re.split(r"(?<=[.!?])\s+", rest) if rest.strip() else [rest]
        new_parts: list[str] = []

        for part in parts:
            if "[E" not in part.upper():
                new_parts.append(part)
                continue
            kind, payload = _classify_sentence(part)
            if kind == "skip":
                new_parts.append(part)
                continue

            cited = _marker_ids(part)
            if not cited or any(c not in by_id for c in cited):
                new_parts.append(part)  # existence already enforced upstream
                continue
            target = payload  # what must the cited chunk support?
            applicable = verdict(target, by_id[cited[0]])["applicable"]
            if not applicable:
                new_parts.append(part)  # no factual content -> nothing to check
                continue
            grounded_ids = [ev.id for ev in evidence if verdict(target, ev)["grounded"]]

            if kind == "label" or not is_abstention(target):
                supporting = [c for c in cited if verdict(target, by_id[c])["grounded"]]
            else:
                # abstention: supported only if a chunk says it verbatim
                supporting = [c for c in cited if c in grounded_ids]

            if supporting:
                for c in cited:
                    if c not in supporting:
                        part = re.sub(r"\s*\[" + c + r"\]", "", part)
                        corrections.append(
                            f"dropped unsupported citation [{c}] — the chunk does "
                            f"not contain: {display(target)}")
                new_parts.append(part)
                continue

            found = next((eid for eid in grounded_ids
                          if verdict(target, by_id[eid])["grounded"]), None)
            if found:
                for c in dict.fromkeys(cited):
                    part = part.replace(f"[{c}]", f"[{found}]")
                corrections.append(
                    f"re-cited {'/'.join(f'[{c}]' for c in dict.fromkeys(cited))} "
                    f"-> [{found}] — only {found} contains: {display(target)}")
                new_parts.append(part)
                continue

            if kind == "label" or not is_abstention(target):
                # No retrieved chunk supports it -> never display it as fact.
                if kind == "label":
                    new_parts.append(re.sub(r":\s.*$", ": [NOT PROVIDED]", part))
                else:
                    kp = _NUMBERING_RE.match(part)
                    new_parts.append((kp.group(0) if kp else "") + "[NOT PROVIDED]")
                removed.append(display(target))
            else:
                new_parts.append(part)  # unverifiable abstention: keep as-is

        out.append(prefix + " ".join(new_parts))

    return "\n".join(out), corrections, removed


def _insufficient(draft_type: str, evidence: list[Evidence], available: list[dict],
                  missing: list[str], reason: str, stage: str) -> DraftResponse:
    return DraftResponse(
        draft_type=draft_type,
        status="insufficient",
        draft=REFUSAL,
        sources=[],
        missing_information=missing,
        available_information=[AvailableField(**a) for a in available],
        claim_check=ClaimCheck(),
        reason=reason,
        refusal_stage=stage,
    )


def validate_draft(draft_type: str, evidence: list[Evidence], available: list[dict],
                   missing: list[str], data: object) -> DraftResponse:
    """Pure validation of parsed model output — unit-tested without any LLM call."""
    if not isinstance(data, dict):
        return _insufficient(draft_type, evidence, available, missing,
                             "model output was not a JSON object", stage="verifier")

    if data.get("insufficient") is True:
        return _insufficient(draft_type, evidence, available, missing,
                             "model reported insufficient evidence", stage="model")

    draft = normalize_markers(str(data.get("draft") or "").strip())
    if not draft or draft == REFUSAL:
        return _insufficient(draft_type, evidence, available, missing,
                             "model returned no usable draft", stage="model")

    by_id = {ev.id: ev for ev in evidence}

    # [E#] markers inside the draft must all exist.
    text_ids = [m.upper() for m in _ANY_MARKER_RE.findall(draft)]
    for tid in text_ids:
        if tid not in by_id:
            return _insufficient(draft_type, evidence, available, missing,
                                 f"draft references non-existent evidence {tid}",
                                 stage="verifier")

    # evidence_used IDs must all exist.
    raw_used = data.get("evidence_used")
    if not isinstance(raw_used, list):
        raw_used = []
    used_ids: list[str] = []
    for cid in raw_used:
        cid = str(cid).strip().upper()
        if cid not in by_id:
            return _insufficient(draft_type, evidence, available, missing,
                                 f"model cited non-existent evidence {cid!r}",
                                 stage="verifier")
        used_ids.append(cid)

    # The draft must not reference page numbers outside the retrieved evidence.
    evidence_pages = {ev.page for ev in evidence}
    for match in re.finditer(r"\b(?:page|pp?\.?)\s*(\d+)", draft, re.I):
        if int(match.group(1)) not in evidence_pages:
            return _insufficient(draft_type, evidence, available, missing,
                                 f"draft references unverified page {match.group(1)}",
                                 stage="verifier")

    # CITATION CORRECTNESS: claim -> cited evidence -> verify. Wrong citations
    # are re-pointed at the chunk that actually contains the claim; claims no
    # retrieved chunk supports are removed ([NOT PROVIDED]), never displayed.
    draft, corrections, removed = verify_citations(draft, evidence)
    final_ids = [m.upper() for m in _ANY_MARKER_RE.findall(draft)]

    # No evidence markers anywhere => the model asserted facts it refused to
    # trace. Same spirit as chat's "model provided no citations" refusal.
    if not final_ids and not used_ids:
        return _insufficient(draft_type, evidence, available, missing,
                             "draft cites no evidence markers", stage="verifier")

    # Sources = VERIFIED cited markers ∪ field-detection support, mapped to
    # (filename, page) from vector-store metadata only. A chunk whose marker
    # was dropped as unsupported is not attached — sources must contain only
    # evidence that supports what is displayed.
    source_ids = list(dict.fromkeys(
        final_ids + [a["evidence_id"] for a in available
                     if a.get("evidence_id")]))
    sources = [DraftSource(evidence_id=eid,
                           filename=by_id[eid].filename,
                           page=by_id[eid].page,
                           snippet=by_id[eid].text[:240])
               for eid in source_ids if eid in by_id]

    # Claim-level traceability (reuses the evaluation metrics — deterministic).
    # Only sentences the model marked with [E#] are factual claim candidates;
    # headings/placeholders/boilerplate are filtered by claim_sentences().
    marked = " ".join(claim_sentences(draft))
    rows = [r for r in evaluate_answer_claims(marked, evidence, REFUSAL)
            if r["kind"] == "claim"]
    unverified = [UnverifiedClaim(claim=r["claim"], reason=r["reason"])
                  for r in rows if not r["grounded"]]
    supported = sum(1 for r in rows if r["grounded"])
    check = ClaimCheck(claims_total=len(rows), claims_supported=supported,
                       claims_unverified=len(unverified), unverified_claims=unverified)

    problems = []
    if missing:
        problems.append(f"{len(missing)} required field(s) not found in the source")
    if unverified:
        problems.append(f"{len(unverified)} draft claim(s) not traceable to the evidence")
    if removed:
        problems.append(
            f"{len(removed)} unsupported claim(s) replaced with [NOT PROVIDED]")

    return DraftResponse(
        draft_type=draft_type,
        status="missing" if problems else "grounded",
        draft=draft,
        sources=sources,
        missing_information=missing,
        available_information=[AvailableField(**a) for a in available],
        claim_check=check,
        citation_corrections=corrections,
        removed_claims=removed,
        reason="; ".join(problems) if problems else None,
        refusal_stage=None,
    )


# ---------------------------------------------------------------- workflow
def draft_document(draft_type: str, instructions: Optional[str] = None,
                   doc_ids: Optional[list[str]] = None) -> DraftResponse:
    """Full workflow. Raises ValueError for an unknown draft type (-> HTTP 400)."""
    if draft_type not in DRAFT_TYPES:
        allowed = ", ".join(sorted(DRAFT_TYPES))
        raise ValueError(f"Unknown draft type {draft_type!r}. Allowed: {allowed}")
    spec = DRAFT_TYPES[draft_type]
    instructions = (instructions or "").strip()

    query = " ".join(p for p in (spec["label"], spec["query"], instructions) if p).strip()
    evidence = retrieve(query, doc_ids=doc_ids)
    top_score = evidence[0].score if evidence else 0.0

    # Deterministic availability check happens BEFORE any model call.
    available, missing = detect_fields(evidence, spec["fields"])

    if not evidence:
        return _insufficient(draft_type, evidence, available, missing,
                             "no indexed documents or no matching chunks",
                             stage="retrieval")
    if top_score < REFUSAL_THRESHOLD:
        return _insufficient(
            draft_type, evidence, available, missing,
            f"best retrieval score {top_score:.2f} is below threshold "
            f"{REFUSAL_THRESHOLD:.2f}; the LLM was not consulted",
            stage="retrieval",
        )

    avail_lines = "\n".join(
        f"- {a['label']} — supported by [{a['evidence_id']}] ({a['filename']}, page {a['page']})"
        for a in available
    ) or "- (none)"
    missing_lines = "\n".join(
        f"- {m} → insert placeholder [{m.upper()} — NOT PROVIDED]" for m in missing
    ) or "- (none)"

    prompt = (
        f"{DRAFT_SYSTEM_PROMPT}\n\n{build_packet(evidence)}\n\n"
        f"DRAFT TYPE: {spec['label']}\n\n"
        f"REQUIRED STRUCTURE:\n{spec['structure']}\n\n"
        f"AVAILABLE INFORMATION (verified in the evidence):\n{avail_lines}\n\n"
        f"MISSING INFORMATION (must be placeholders in the draft):\n{missing_lines}\n\n"
        f"ADDITIONAL INSTRUCTIONS: {instructions or '(none)'}\n\n"
        f"Respond with the JSON object only."
    )
    try:
        raw = run_llm(prompt)
        data = extract_json(raw)
    except (LLMError, ValueError) as exc:
        return _insufficient(draft_type, evidence, available, missing,
                             f"model error: {exc}", stage="system")

    return validate_draft(draft_type, evidence, available, missing, data)
