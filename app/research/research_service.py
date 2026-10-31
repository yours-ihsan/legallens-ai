"""Legal Research workflow — EVIDENCE FIRST -> VERIFY -> REASON -> ANSWER.

One pipeline built on the EXISTING stack (no second RAG system):

  question + case context

    -> deterministic legal-issue identification (Python, research_prompt)

    -> case evidence via the existing app.retriever.retrieve()

       + legal authority via the same Chroma store (bundled India Code

         snapshot collection, best-effort live India Code search)   [gather]

    -> ONE evidence packet, E-IDs, answer_generator.build_packet()

    -> one LLM call (research_prompt.build_prompt, JSON schema)

    -> PURE verification (source_verifier): every claim must be traceable to

       the retrieved chunk that literally contains it; guilt conclusions and

       uncited case statements are removed; unverified law/judgments/entries

       are dropped and reported; source metadata comes from the registry.

    -> deterministic status (no fabricated confidence):

         GROUNDED_RESEARCH          verified legal authority AND verified

                                     case connection

         RESEARCH_WITH_LIMITATIONS  one side verified, the other missing

         NO_VERIFIED_PRECEDENT_FOUND  judgment/precedent query, no judgment

         UNVERIFIED_RESEARCH        nothing could be verified (-> REFUSAL,

                                     refusal_stage records where it stopped)

Trust boundary mirrors chat: the LLM sees only build_packet()'s evidence (plus

Python-computed classification/issue metadata), page numbers come from vector

store metadata only, and the exact refusal string is reused from

app.answer_generator.

"""

from __future__ import annotations

import re

from typing import Optional

from pydantic import BaseModel

from ..answer_generator import REFUSAL, build_packet

from ..config import REFUSAL_THRESHOLD

from ..drafting.drafter import (ClaimCheck, UnverifiedClaim, claim_sentences,

                                normalize_markers, verify_citations)

from ..evaluation.metrics import evaluate_answer_claims

from ..llm import LLMError, extract_json, run_llm

from ..models import Evidence

from .legal_source_retriever import classify_evidence, gather_evidence

from .research_prompt import (RESEARCH_TYPES, build_prompt,

                              identify_legal_issue, legal_search_terms)

from .source_verifier import (LEGAL_KINDS, NO_LEGAL_AUTHORITY, NO_PRECEDENT,

                              _markers, answer_grounded_ids, build_sources,

                              extract_ids, strip_unsafe_statements,

                              verify_case_connections, verify_judgments,

                              verify_law_entries)

# ---------------------------------------------------------------- statuses

STATUS_GROUNDED = "GROUNDED_RESEARCH"

STATUS_LIMITED = "RESEARCH_WITH_LIMITATIONS"

STATUS_NO_PRECEDENT = "NO_VERIFIED_PRECEDENT_FOUND"

STATUS_UNVERIFIED = "UNVERIFIED_RESEARCH"

LIMITATIONS_CAP = 14

# ------------------------------------------------------------------- models

class ResearchRequest(BaseModel):

    question: str

    case_context: Optional[str] = None

    research_type: str = "general"

    doc_ids: Optional[list[str]] = None

class ResearchSource(BaseModel):

    evidence_id: str

    title: str

    source_type: str                      # statute | judgment | case_evidence

    authority: Optional[str] = None       # registry/official metadata only

    section: Optional[str] = None

    url: Optional[str] = None             # real URL or None — never invented

    tier: Optional[int] = None            # 1 official / 2 established db / 3 web

    verified: bool = False                # a verified claim actually cited it

    relevance: str = ""

    filename: str

    page: int

    excerpt: str = ""

class RelevantLaw(BaseModel):

    provision: str

    covers: str = ""

    evidence_ids: list[str] = []

    verified: bool = True

    authority: Optional[str] = None

    url: Optional[str] = None

    tier: Optional[int] = None

    filename: Optional[str] = None

    page: Optional[int] = None

class CaseConnection(BaseModel):

    provision: str

    what_it_covers: str = ""

    case_fact: str

    evidence_ids: list[str] = []          # case-fact cites

    law_evidence_ids: list[str] = []      # provision cites

    sufficiency: str = ""

    missing: str = ""

    verified: bool = True

class JudgmentEntry(BaseModel):

    case_name: str

    court: str = ""

    date: str = ""

    citation: str = ""

    principle: str = ""

    relationship: str = ""

    evidence_ids: list[str] = []

    verified: bool = True

    url: Optional[str] = None

class ResearchResponse(BaseModel):

    status: str

    legal_issue: str

    answer: str

    research_type: str

    # Retrieved uploaded-document evidence, separate from cited legal sources.
    case_evidence: list[Evidence] = []

    sources: list[ResearchSource] = []

    relevant_law: list[RelevantLaw] = []

    case_connections: list[CaseConnection] = []

    judgments: list[JudgmentEntry] = []

    limitations: list[str] = []

    citation_corrections: list[str] = []

    removed_claims: list[str] = []

    unverified: list[UnverifiedClaim] = []

    claim_check: ClaimCheck = ClaimCheck()

    reason: Optional[str] = None

    refusal_stage: Optional[str] = None

def _entries(raw) -> list[dict]:

    if not isinstance(raw, list):

        return []

    return [e for e in raw if isinstance(e, dict)]

def _ordered_ids(*groups) -> list[str]:

    out: list[str] = []

    for group in groups:

        for eid in group:

            if eid not in out:

                out.append(eid)

    return out

def _unique_strings(items: list[str], cap: int) -> list[str]:

    out: list[str] = []

    for item in items:

        item = re.sub(r"\s+", " ", str(item)).strip()

        if item and item not in out:

            out.append(item)

    return out[:cap]

def _unverified(research_type: str, legal_issue: str, reason: str,

                stage: str,

                case_evidence: Optional[list[Evidence]] = None) -> ResearchResponse:

    """Insufficient / unverified response — exact refusal string, no sources."""

    return ResearchResponse(

        status=STATUS_UNVERIFIED,

        legal_issue=legal_issue,

        answer=REFUSAL,

        research_type=research_type,

        case_evidence=case_evidence or [],

        limitations=[reason],

        reason=reason,

        refusal_stage=stage,

    )

def _source_info(classes: dict, evidence_ids: list[str]) -> dict:

    """Registry-derived provenance for a structured entry (never model text)."""

    for eid in evidence_ids:

        cls = classes.get(eid)

        if cls:

            return {"authority": cls.get("authority"), "url": cls.get("url"),

                    "tier": cls.get("tier"), "filename": cls.get("filename"),

                    "page": cls.get("page")}

    return {"authority": None, "url": None, "tier": None,

            "filename": None, "page": None}

# ------------------------------------------------------------- validation

def validate_research(research_type: str, legal_issue: str,

                      evidence: list[Evidence], classes: dict[str, dict],

                      data: object, live_error: Optional[str] = None

                      ) -> ResearchResponse:

    """Pure validation of parsed model output — unit-tested with no LLM call."""

    case_evidence = [
        ev for ev in evidence
        if classes.get(ev.id, {}).get("kind") == "case_evidence"
    ]

    if not isinstance(data, dict):

        return _unverified(research_type, legal_issue,

                           "model output was not a JSON object", "verifier", case_evidence)

    if data.get("insufficient") is True:

        return _unverified(research_type, legal_issue,

                           "model reported insufficient evidence", "model", case_evidence)

    answer = normalize_markers(str(data.get("answer") or "").strip())

    if not answer or answer == REFUSAL:

        return _unverified(research_type, legal_issue,

                           "model returned no usable answer", "model", case_evidence)

    by_id = {ev.id: ev for ev in evidence}

    # 1) Every cited ID — answer markers AND entry cites — must exist.

    answer_ids = _markers(answer)

    entry_ids: list[str] = []

    for key in ("relevant_law", "case_connections", "judgments"):

        for entry in _entries(data.get(key)):

            entry_ids.extend(extract_ids(entry.get("cites") or []))

    for cid in _ordered_ids(answer_ids, entry_ids):

        if cid not in by_id:

            return _unverified(research_type, legal_issue,

                               f"answer references non-existent evidence {cid}",

                               "verifier", case_evidence)

    # 2) Page numbers come from vector-store metadata only, never the model.

    evidence_pages = {ev.page for ev in evidence}

    for match in re.finditer(r"\b(?:page|pp?\.?)\s*(\d+)", answer, re.I):

        if int(match.group(1)) not in evidence_pages:

            return _unverified(

                research_type, legal_issue,

                f"answer references unverified page {match.group(1)}", "verifier", case_evidence)

    # 3) SAFETY: no guilt conclusions, no uncited case-specific statements.

    answer, safety_removed = strip_unsafe_statements(answer, evidence, classes)

    # 4) CITATION CORRECTNESS: claim -> cited chunk -> supported?  Re-point or

    #    remove ([NOT PROVIDED]) — an unsupported assertion is never displayed.

    answer, corrections, removed = verify_citations(answer, evidence)

    removed_records = [

        {"claim": text,

         "reason": "claim was not supported by any retrieved chunk"}

        for text in removed]

    if not any(ch.isalpha() for ch in answer.replace("NOT PROVIDED", "")):

        return _unverified(

            research_type, legal_issue,

            "no supported claim remained after verification", "verifier", case_evidence)

    # 5) Structured entries — law/judgments need legal authority, connections

    #    must hold on BOTH the law side and the case-fact side.

    law_kept, law_dropped, c1 = verify_law_entries(

        data.get("relevant_law"), evidence, classes)

    conn_kept, conn_dropped, c2 = verify_case_connections(

        data.get("case_connections"), evidence, classes)

    judg_kept, judg_dropped, c3 = verify_judgments(

        data.get("judgments"), evidence, classes)

    corrections = corrections + c1 + c2 + c3

    # 6) Grounded IDs + why each source is cited.

    answer_grounded = answer_grounded_ids(answer, evidence, classes)

    grounded = set(answer_grounded)

    roles: dict[str, list[str]] = {}

    def _role(eids: list[str], text: str) -> None:

        for eid in eids:

            grounded.add(eid)

            roles.setdefault(eid, []).append(text)

    for eid in answer_grounded:

        roles.setdefault(eid, []).append("supports a verified statement in the answer")

    for entry in law_kept:

        _role(entry["evidence_ids"], "supports RELEVANT LAW")

    for entry in conn_kept:

        _role(entry["evidence_ids"], "supports CASE CONNECTION")

        _role(entry["law_evidence_ids"], "supports the connected legal provision")

    for entry in judg_kept:

        _role(entry["evidence_ids"], "supports a verified judgment")

    roles = {eid: list(dict.fromkeys(v)) for eid, v in roles.items()}

    # 7) Sources: every block the model actually cited, with registry

    #    provenance; `verified` only when a surviving claim checked out.

    cited = _ordered_ids(answer_ids, entry_ids)

    sources = [ResearchSource(**s)

               for s in build_sources(cited, grounded, evidence, classes, roles)]

    # 8) Deterministic notes the model cannot suppress.

    problems: list[str] = []

    model_limits = ([str(x) for x in data.get("limitations")]

                    if isinstance(data.get("limitations"), list) else [])

    limitations = list(model_limits)

    if live_error:

        limitations.append(

            "Live search of official sources was unavailable; statutory sources "

            "come from the bundled official snapshot.")

    if safety_removed:

        limitations.append(

            f"{len(safety_removed)} unsupported statement(s) about guilt or "

            "case facts were removed from the answer.")

        problems.append(f"{len(safety_removed)} unsupported statement(s) removed")

    if removed:

        limitations.append(

            f"{len(removed)} claim(s) in the answer were not supported by any "

            "retrieved chunk and were removed.")

        problems.append(f"{len(removed)} unsupported answer claim(s) removed")

    for record in (law_dropped + conn_dropped + judg_dropped)[:6]:

        limitations.append(

            "Unverified point excluded from the research output: "

            f"{record['claim'][:110]} ({record['reason']}).")

    if law_dropped:

        problems.append(f"{len(law_dropped)} law entr(y/ies) excluded as unverified")

    if conn_dropped:

        problems.append(f"{len(conn_dropped)} case connection(s) excluded as unverified")

    if judg_dropped:

        problems.append(f"{len(judg_dropped)} judgment entr(y/ies) excluded as unverified")

    # 9) Status — deterministic, never a fabricated score.

    answer_legal = {eid for eid in answer_grounded

                    if classes.get(eid, {}).get("kind") in LEGAL_KINDS}

    answer_case = {eid for eid in answer_grounded

                   if classes.get(eid, {}).get("kind") == "case_evidence"}

    has_law = bool(law_kept) or bool(judg_kept) or bool(answer_legal)

    has_case = bool(conn_kept) or bool(answer_case)

    if not has_law:

        limitations.append(NO_LEGAL_AUTHORITY)

    if research_type in ("judgment", "precedent") and not judg_kept:

        limitations.append(NO_PRECEDENT)

    if research_type in ("judgment", "precedent") and not judg_kept:

        status = STATUS_NO_PRECEDENT

    elif has_law and has_case:

        status = STATUS_GROUNDED

    elif has_law or has_case:

        status = STATUS_LIMITED

    else:

        status = STATUS_UNVERIFIED

    # 10) Claim-level traceability (reuses evaluation metrics — deterministic).

    marked = " ".join(claim_sentences(answer))

    rows = [r for r in evaluate_answer_claims(marked, evidence, REFUSAL)

            if r["kind"] == "claim"]

    unverified_rows = [UnverifiedClaim(claim=r["claim"], reason=r["reason"])

                       for r in rows if not r["grounded"]]

    supported = sum(1 for r in rows if r["grounded"])

    check = ClaimCheck(claims_total=len(rows), claims_supported=supported,

                       claims_unverified=len(unverified_rows),

                       unverified_claims=unverified_rows)

    # Nothing in the answer or the structured entries survived verification:

    # this is the insufficient/unverified response shape (exact refusal string).

    if status == STATUS_UNVERIFIED:

        reason = "no claim or structured entry could be verified against a " \
                 "retrieved source"

        return ResearchResponse(

            status=STATUS_UNVERIFIED,

            legal_issue=legal_issue,

            answer=REFUSAL,

            research_type=research_type,

            case_evidence=case_evidence,

            reason=reason,

            refusal_stage="verifier",

            limitations=_unique_strings(limitations + [reason], LIMITATIONS_CAP),

        )

    # Unverified answer claims can never keep the grounded badge.

    if status == STATUS_GROUNDED and check.claims_unverified:

        status = STATUS_LIMITED

        limitations.append(

            f"{check.claims_unverified} answer claim(s) could not be traced to "

            "their cited source.")

    unverified = [UnverifiedClaim(**r) for r in

                  (safety_removed + removed_records + law_dropped

                   + conn_dropped + judg_dropped)]

    # The grounded badge means EVERYTHING produced was verified — if any point

    # had to be excluded, the honest label is RESEARCH WITH LIMITATIONS.

    if status == STATUS_GROUNDED and unverified:

        status = STATUS_LIMITED

        limitations.append(

            f"{len(unverified)} unverified item(s) were excluded from the "

            "research output.")

    return ResearchResponse(

        status=status,

        legal_issue=legal_issue,

        answer=answer,

        research_type=research_type,

        case_evidence=case_evidence,

        sources=sources,

        relevant_law=[RelevantLaw(

            evidence_ids=e["evidence_ids"], verified=True,

            provision=e["provision"], covers=e["covers"],

            **_source_info(classes, e["evidence_ids"]))

            for e in law_kept],

        case_connections=[CaseConnection(

            evidence_ids=e["evidence_ids"],

            law_evidence_ids=e["law_evidence_ids"], verified=True,

            provision=e["provision"], what_it_covers=e["what_it_covers"],

            case_fact=e["case_fact"], sufficiency=e["sufficiency"],

            missing=e["missing"]) for e in conn_kept],

        judgments=[JudgmentEntry(

            evidence_ids=e["evidence_ids"], verified=True,

            case_name=e["case_name"], court=e["court"], date=e["date"],

            citation=e["citation"], principle=e["principle"],

            relationship=e["relationship"],

            url=classes.get(e["evidence_ids"][0], {}).get("url")

            if e["evidence_ids"] else None)

            for e in judg_kept],

        limitations=_unique_strings(limitations, LIMITATIONS_CAP),

        citation_corrections=_unique_strings(corrections, 30),

        removed_claims=_unique_strings(

            [r["claim"] for r in safety_removed + removed_records], 30),

        unverified=unverified,

        claim_check=check,

        reason="; ".join(problems) or None,

        refusal_stage=None,

    )

# --------------------------------------------------------------- workflow

def research_document(question: str, case_context: Optional[str] = None,

                      research_type: str = "general",

                      doc_ids: Optional[list[str]] = None) -> ResearchResponse:

    """Full workflow. Raises ValueError for an unknown research type (-> 400)."""

    if research_type not in RESEARCH_TYPES:

        raise ValueError(

            f"Unknown research type {research_type!r}. "

            f"Allowed: {', '.join(RESEARCH_TYPES)}")

    question = (question or "").strip()

    if not question:

        raise ValueError("question must not be empty")

    case_context = (case_context or "").strip()

    # EVIDENCE FIRST, and the legal issue is identified BEFORE retrieval —

    # deterministic Python, cautious wording, never a model guess.

    legal_issue = identify_legal_issue(question, case_context)

    search_terms = legal_search_terms(question, case_context)

    gather = gather_evidence(question, doc_ids=doc_ids, legal_query=search_terms)

    evidence = gather.evidence

    classes = classify_evidence(evidence, gather.meta_by_doc_id)

    top_score = gather.case_evidence[0].score if gather.case_evidence else 0.0

    # Sufficiency gate BEFORE any model call (same threshold as chat).

    if not gather.case_evidence:

        return _unverified(research_type, legal_issue,

                           "no indexed documents or no matching chunks",

                           "retrieval", gather.case_evidence)

    if top_score < REFUSAL_THRESHOLD:

        return _unverified(

            research_type, legal_issue,

            f"retrieval score {top_score:.2f} below threshold "

            f"{REFUSAL_THRESHOLD}", "retrieval", gather.case_evidence)

    packet = build_packet(evidence)          # the ONLY thing the LLM sees

    prompt = build_prompt(packet, classes, evidence, legal_issue,

                          research_type, question, case_context)

    try:

        data = extract_json(run_llm(prompt, json_mode=True))

    except (LLMError, ValueError) as exc:

        return _unverified(research_type, legal_issue,

                           f"model error: {exc}", "system", gather.case_evidence)

    return validate_research(research_type, legal_issue, evidence, classes,

                              data, live_error=gather.live_error)
