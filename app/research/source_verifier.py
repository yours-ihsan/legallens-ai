"""Source & citation verification for Legal Research — pure, no LLM, no network.

The chain demanded of every research output:

    CLAIM -> SOURCE -> DOES THE SOURCE ACTUALLY SUPPORT THE CLAIM?
        YES -> show the citation
        NO  -> re-point to the chunk that does, or remove the claim/report it

Pieces:
  * answer prose  — reuses drafting's verify_citations() (marker-level check
    against THE cited chunk, re-point, drop) — same citation-grounding
    principle the RAG evaluation uses (app.evaluation.metrics.check_claim);
  * relevant_law / judgments — must cite a block classified LEGAL AUTHORITY
    that literally supports the provision (statute text is checked against
    title + section + body, all metadata-derived);
  * case_connections — must hold on BOTH sides: provision grounded in legal
    authority AND case fact grounded in case material;
  * safety guards — an unsupported conclusion about guilt and any uncited
    case-specific / legal statement is REMOVED and reported; an allegation is
    never displayed as an established fact;
  * source records — authority / section / url / tier come from the registry
    or vector-store metadata only, never from the model.

Deterministic throughout: reuse of app.evaluation.metrics, zero model calls.
"""
from __future__ import annotations

import re

from ..drafting.drafter import verify_citations
from ..evaluation.metrics import check_claim, is_abstention
from ..models import Evidence
from .legal_source_retriever import legal_source_text

LEGAL_KINDS = ("statute", "judgment")

NO_LEGAL_AUTHORITY = "Verified legal authority was not found in the available sources."
NO_PRECEDENT = "No verified judgment was found."

_MARKER_RE = re.compile(r"\[(E\d+)\]", re.I)
_GUILT_VOCAB_RE = re.compile(
    r"\b(?:guilty|convicted|conviction|culpable|stole|stolen|stealing|"
    r"committed\s+(?:the\s+|an?\s+)?(?:theft|offence|offenses|crime|robbery|murder|cheating)|"
    r"is\s+the\s+(?:thief|culprit)|prove[sn]?\s+(?:guilt|that|who))\b", re.I)
_NEGATED_GUILT_RE = re.compile(
    r"\b(?:not|no|never|cannot|can\s*not|couldn'?t|does\s+not|do\s+not|did\s+not|"
    r"insufficient|unable|nothing|fails?\s+to|neither|lacks?)\b[^.]{0,90}"
    r"\b(?:guilty|guilt|commit|commits|committed|stole|steal|theft|convict)\b", re.I)
# The SOURCE ITSELF must assert the conclusion (not merely contain the nouns):
# this is what stops the generic 75% recall rule from letting "X committed the
# theft" through when the cited chunk only names X and mentions "theft".
_SOURCE_GUILT_RE = re.compile(
    r"\b(?:guilty|convicted|conviction|culpable)\b"
    r"|\b(?:stole|stolen|steal|stealing)\b"
    r"|\bcommit(?:ted|s|ing)?\s+(?:the\s+|an?\s+)?"
    r"(?:theft|offence|offense|crime|robbery|murder|cheating)\b"
    r"|\b(?:theft|robbery|offence)\s+(?:was|is)\s+committed\b"
    r"|\bis\s+the\s+(?:thief|culprit)\b", re.I)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_CASE_ID_RE = re.compile(r"\bLL-[A-Z]{2}-\d{4}-\d+\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{3,}(?:\s+[A-Z0-9][A-Za-z0-9]{1,})+\b")
_SECTION_MENTION_RE = re.compile(
    r"\bSection\s+\d+[A-Z]?|\b(?:Bharatiya Nyaya Sanhita|Bharatiya Nagarik Suraksha "
    r"Sanhita|Indian Penal Code|Code of Criminal Procedure)\b")
_COMMON_HEAD = {"The", "This", "That", "These", "Those", "There", "Missing", "Case",
                "Expected", "Every", "Where", "What", "When", "Who", "How", "Note",
                "New", "Per", "All", "Any", "For", "And", "But", "Her", "His"}

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


_BARE_ID_RE = re.compile(r"(?<![A-Za-z0-9])([Ee])\s*(\d+)(?!\d)")


# --------------------------------------------------------------- helpers
def _markers(text: str) -> list[str]:
    return [m.upper() for m in _MARKER_RE.findall(text)]


def extract_ids(values) -> list[str]:
    """Evidence IDs from a model `cites` list — accepts "E7", "[E7]", "e 7".

    The schema asks for bare ["E7"]; the model may still wrap it in brackets.
    Both normalise to the canonical E-ID form.
    """
    if not isinstance(values, list):
        values = [values]
    out: list[str] = []
    for value in values:
        out.extend(f"E{m.group(2)}" for m in _BARE_ID_RE.finditer(str(value)))
    return out


def _strip_markers(text: str) -> str:
    return re.sub(r"\s*\[E\d+\]", "", text, flags=re.I).strip()


def _sentence_list(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []
    return [s for s in _SENT_SPLIT_RE.split(line) if s.strip()]


def _case_entities(evidence: list[Evidence], classes: dict[str, dict]) -> set[str]:
    """Multi-word proper nouns appearing in CASE material (never legal blocks).

    Used to detect case-specific statements that were written WITHOUT a
    citation. Sentence-initial capitalisation cannot fool it because only
    multi-word capitalised sequences inside the retrieved case text count.
    """
    entities: set[str] = set()
    for ev in evidence:
        if classes.get(ev.id, {}).get("kind") != "case_evidence":
            continue
        for match in _ENTITY_RE.findall(ev.text):
            tokens = match.split()
            if len(tokens) < 2 or any(len(t) < 2 for t in tokens):
                continue
            if tokens[0] in _COMMON_HEAD:
                continue
            if any(t.lower() in {"and", "or", "of", "in", "on", "at", "to", "for"} for t in tokens):
                continue
            entities.add(match)
    return entities


def _grounded(claim: str, chunks: list[Evidence], classes: dict[str, dict],
              legal: bool = False) -> bool:
    if not chunks:
        return False
    proxied = []
    for ev in chunks:
        cls = classes.get(ev.id, {})
        if legal or cls.get("kind") in LEGAL_KINDS:
            ev = ev.model_copy(update={"text": legal_source_text(ev, cls)})
        proxied.append(ev)
    return bool(check_claim(claim, proxied)["grounded"])


def _entry_list(raw) -> list[dict]:
    if not isinstance(raw, list):
        return []
    return [e for e in raw if isinstance(e, dict)]


# --------------------------------------------------------- safety guards
def strip_unsafe_statements(answer: str, evidence: list[Evidence],
                            classes: dict[str, dict]
                            ) -> tuple[str, list[dict]]:
    """Remove unsupported guilt conclusions and uncited case/legal statements.

    Deterministic sentence-level scan (markers or no markers — the model
    cannot dodge verification by omitting a citation):
      * a sentence asserting guilt/offence is kept only if a retrieved chunk
        supports it verbatim, or it is an abstention/negated correction;
      * a case-specific sentence (time, case ID, named entity from case
        material) or a legal-provision sentence must carry an [E#] marker,
        otherwise it is removed as unsourced.
    Returns (answer, removed records).
    """
    by_id = {ev.id: ev for ev in evidence}
    entities = _case_entities(evidence, classes)
    removed: list[dict] = []

    def candidate(line: str) -> bool:
        return bool(_GUILT_VOCAB_RE.search(line) or _TIME_RE.search(line)
                    or _CASE_ID_RE.search(line) or _SECTION_MENTION_RE.search(line)
                    or any(e in line for e in entities))

    def reason_for(sentence: str) -> str | None:
        if len(sentence.split()) < 4:
            return None
        stripped = _strip_markers(sentence)
        if stripped and stripped.upper() == stripped and len(stripped) < 70:
            return None  # section heading, not a claim
        cited = [by_id[c] for c in _markers(sentence) if c in by_id]

        # (a) A conclusion about guilt survives ONLY if a retrieved source
        #     literally asserts it AND the claim traces to that source. The
        #     source-assertion half is what stops the generic 75% recall rule
        #     from waving "X committed the theft" through a chunk that merely
        #     names X and mentions "theft".
        if _GUILT_VOCAB_RE.search(sentence) and not is_abstention(sentence) \
                and not _NEGATED_GUILT_RE.search(sentence):
            chunks = cited or [ev for ev in evidence
                               if classes.get(ev.id, {}).get("kind") == "case_evidence"]
            source_asserts = any(
                _SOURCE_GUILT_RE.search(classes.get(ev.id, {}).get("full_text")
                                        or ev.text)
                for ev in chunks)
            if not (_grounded(stripped, chunks, classes) and source_asserts):
                return ("asserted a conclusion about guilt or an offence that no "
                        "retrieved source supports")

        if not cited and not is_abstention(sentence):
            if (_GUILT_VOCAB_RE.search(sentence)
                    and not _NEGATED_GUILT_RE.search(sentence)):
                return "conclusion about guilt written without a source citation"
            if _CASE_ID_RE.search(sentence) or _TIME_RE.search(sentence) \
                    or any(e in sentence for e in entities):
                return "case-specific statement written without a source citation"
            if _SECTION_MENTION_RE.search(sentence):
                return "legal-provision statement written without a source citation"
        return None

    out_lines: list[str] = []
    for line in answer.splitlines():
        if not candidate(line):
            out_lines.append(line)
            continue
        kept_parts: list[str] = []
        for sentence in _sentence_list(line):
            why = reason_for(sentence)
            if why:
                removed.append({"claim": _strip_markers(sentence), "reason": why})
            else:
                kept_parts.append(sentence)
        if kept_parts:
            out_lines.append(" ".join(kept_parts))
        elif not _sentence_list(line):
            out_lines.append(line)
    return "\n".join(out_lines), removed


# ------------------------------------------------------------- entry checks
def verify_law_entries(entries: list[dict], evidence: list[Evidence],
                       classes: dict[str, dict]
                       ) -> tuple[list[dict], list[dict], list[str]]:
    """relevant_law: cited block must be LEGAL AUTHORITY and support the text."""
    by_id = {ev.id: ev for ev in evidence}
    legal_ids = [eid for eid, c in classes.items() if c.get("kind") in LEGAL_KINDS]
    kept, dropped, corrections = [], [], []

    for entry in _entry_list(entries):
        provision = str(entry.get("provision") or "").strip()
        covers = str(entry.get("covers") or "").strip()
        target = ". ".join(p for p in (provision, covers) if p).strip()
        cites = [c for c in extract_ids(entry.get("cites") or []) if c in by_id]
        if not target:
            dropped.append({"claim": provision or "(empty law entry)",
                            "reason": "law entry had no provision text"})
            continue
        legal_cites = [c for c in cites if classes.get(c, {}).get("kind") in LEGAL_KINDS]
        if not legal_cites:
            dropped.append({"claim": target,
                            "reason": "does not cite a retrieved legal-authority block"})
            continue
        grounded = [c for c in legal_cites if _grounded(target, [by_id[c]], classes, legal=True)]
        if not grounded:
            found = next((eid for eid in legal_ids
                          if _grounded(target, [by_id[eid]], classes, legal=True)), None)
            if found:
                grounded = [found]
                corrections.append(f"re-cited law entry {'/'.join(cites)} -> [{found}] — "
                                   f"only that block contains: {target[:90]}")
            else:
                dropped.append({"claim": target,
                                "reason": "no retrieved legal authority contains this "
                                          "provision as stated"})
                continue
        kept.append({"provision": provision, "covers": covers,
                     "evidence_ids": grounded})
    return kept, dropped, corrections


def verify_case_connections(entries: list[dict], evidence: list[Evidence],
                            classes: dict[str, dict]
                            ) -> tuple[list[dict], list[dict], list[str]]:
    """case_connections must hold on BOTH sides: law side + case-fact side."""
    by_id = {ev.id: ev for ev in evidence}
    legal_ids = [eid for eid, c in classes.items() if c.get("kind") in LEGAL_KINDS]
    case_ids = [eid for eid, c in classes.items() if c.get("kind") == "case_evidence"]
    kept, dropped, corrections = [], [], []

    for entry in _entry_list(entries):
        provision = str(entry.get("provision") or "").strip()
        covers = str(entry.get("what_it_covers") or "").strip()
        case_fact = str(entry.get("case_fact") or "").strip()
        sufficiency = str(entry.get("sufficiency") or "").strip()
        missing = str(entry.get("missing") or "").strip()
        cites = [c for c in extract_ids(entry.get("cites") or []) if c in by_id]
        label = provision or case_fact or "(empty connection)"

        if not provision or not case_fact:
            dropped.append({"claim": label,
                            "reason": "connection is missing a verified provision "
                                      "or a case fact"})
            continue

        # --- law side: provision (+ explanation) must be in legal authority
        law_target = ". ".join(p for p in (provision, covers) if p)
        law_grounded = [c for c in cites
                        if classes.get(c, {}).get("kind") in LEGAL_KINDS
                        and _grounded(law_target, [by_id[c]], classes, legal=True)]
        if not law_grounded:
            found = next((eid for eid in legal_ids
                          if _grounded(law_target, [by_id[eid]], classes, legal=True)), None)
            if found:
                law_grounded = [found]
                corrections.append(f"re-cited connection provision -> [{found}] — only "
                                   f"that legal block contains: {law_target[:90]}")
            else:
                dropped.append({"claim": law_target,
                                "reason": "no retrieved legal authority supports this "
                                          "provision as stated"})
                continue

        # --- fact side: case fact must be in case material
        fact_cites = [c for c in cites
                      if classes.get(c, {}).get("kind") == "case_evidence"
                      and _grounded(case_fact, [by_id[c]], classes)]
        if not fact_cites:
            found = next((eid for eid in case_ids
                          if _grounded(case_fact, [by_id[eid]], classes)), None)
            if found:
                fact_cites = [found]
                corrections.append(f"re-cited connection case fact -> [{found}] — only "
                                   f"that block contains: {case_fact[:90]}")
            else:
                dropped.append({"claim": case_fact,
                                "reason": "no retrieved case material contains this fact"})
                continue

        kept.append({"provision": provision, "what_it_covers": covers,
                     "case_fact": case_fact, "evidence_ids": fact_cites,
                     "law_evidence_ids": law_grounded,
                     "sufficiency": sufficiency, "missing": missing})
    return kept, dropped, corrections


def verify_judgments(entries: list[dict], evidence: list[Evidence],
                     classes: dict[str, dict]
                     ) -> tuple[list[dict], list[dict], list[str]]:
    """judgments: only a block classified JUDGMENT may support a precedent."""
    by_id = {ev.id: ev for ev in evidence}
    judgment_ids = [eid for eid, c in classes.items() if c.get("kind") == "judgment"]
    kept, dropped, corrections = [], [], []

    for entry in _entry_list(entries):
        case_name = str(entry.get("case_name") or "").strip()
        court = str(entry.get("court") or "").strip()
        date = str(entry.get("date") or "").strip()
        citation = str(entry.get("citation") or "").strip()
        principle = str(entry.get("principle") or "").strip()
        relationship = str(entry.get("relationship") or "").strip()
        cites = [c for c in extract_ids(entry.get("cites") or []) if c in by_id]
        target = ". ".join(p for p in (case_name, court, date, citation, principle) if p)
        label = case_name or citation or "(unnamed judgment)"
        if not target:
            dropped.append({"claim": label, "reason": "judgment entry was empty"})
            continue
        judgment_cites = [c for c in cites if classes.get(c, {}).get("kind") == "judgment"]
        if not judgment_cites:
            dropped.append({"claim": label,
                            "reason": "no retrieved judgment source supports this "
                                      "precedent (a case document is not a judgment)"})
            continue
        grounded = [c for c in judgment_cites if _grounded(target, [by_id[c]], classes)]
        if not grounded:
            found = next((eid for eid in judgment_ids
                          if _grounded(target, [by_id[eid]], classes)), None)
            if found:
                grounded = [found]
                corrections.append(f"re-cited judgment -> [{found}]")
            else:
                dropped.append({"claim": target,
                                "reason": "no retrieved judgment source contains this "
                                          "case name/citation/principle"})
                continue
        kept.append({"case_name": case_name, "court": court, "date": date,
                     "citation": citation, "principle": principle,
                     "relationship": relationship, "evidence_ids": grounded})
    return kept, dropped, corrections


# ------------------------------------------------------------ source records
def build_sources(cited_ids: list[str], grounded_ids: set[str],
                  evidence: list[Evidence], classes: dict[str, dict],
                  roles: dict[str, list[str]]) -> list[dict]:
    """Source records for the response — provenance from classes, never the model.

    `verified` is True only when at least one VERIFIED claim was actually
    supported by that block; a citation that failed verification is listed
    with verified=false (the claim itself was already removed).
    """
    by_id = {ev.id: ev for ev in evidence}
    sources = []
    for eid in cited_ids:
        ev, cls = by_id.get(eid), classes.get(eid, {})
        if ev is None:
            continue
        sources.append({
            "evidence_id": eid,
            "title": cls.get("title") or ev.filename,
            "source_type": cls.get("source_type") or "case_evidence",
            "authority": cls.get("authority"),
            "section": cls.get("section"),
            "url": cls.get("url"),
            "tier": cls.get("tier"),
            "verified": eid in grounded_ids,
            "relevance": "; ".join(roles.get(eid, [])) or
                         ("cited in the research answer" if eid in grounded_ids
                          else "cited but not verified — citation removed"),
            "filename": ev.filename,
            "page": ev.page,
            "excerpt": ev.text[:240],
        })
    return sources


def answer_grounded_ids(answer: str, evidence: list[Evidence],
                        classes: dict[str, dict]) -> set[str]:
    """E-IDs whose markers survive final verification (claims that checked out)."""
    by_id = {ev.id: ev for ev in evidence}
    grounded: set[str] = set()
    for sentence in re.split(r"\n|(?<=[.!?])\s+", answer):
        ids = _markers(sentence)
        if not ids:
            continue
        target = _strip_markers(sentence)
        if not target or len(target.split()) < 3:
            continue
        for eid in ids:
            ev = by_id.get(eid)
            if ev is None:
                continue
            legal = classes.get(eid, {}).get("kind") in LEGAL_KINDS
            if _grounded(target, [ev], classes, legal=legal):
                grounded.add(eid)
    return grounded
