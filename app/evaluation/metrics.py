"""Deterministic claim extraction and grounding checks.

NO LLM is used here — the same transparent rule is applied to both systems:

1.  Split an answer into sentences (claim candidates). The exact refusal
    string and sub-3-word fragments are not claims.
2.  A claim is GROUNDED only if, in a single retrieved evidence chunk,
        - every number/time/amount in the claim appears verbatim in that
          chunk (numbers can never be paraphrased), AND
        - at least `GROUND_THRESHOLD` (75%) of the claim's content words
          (stemmed, stopwords/meta-words removed) appear in that chunk.
    The supporting chunk's E-ID / filename / page are recorded.
3.  A sentence that matches an abstention pattern ("does not contain", "not
    specified", ...) AND fails the grounding test is classified as an
    ABSTENTION: it asserts no source fact, so it is excluded from the
    groundedness numerator *and* denominator (it can neither inflate nor
    deflate the score). If such a sentence IS verbatim-traceable (e.g. the
    document's own "does not identify the person ..." note) it counts as a
    grounded claim.
4.  Everything else that fails the test is an UNSUPPORTED claim.

Every check returns a human-readable `reason` including the token statistics
so every percentage point is auditable at claim level.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

# Threshold: share of a claim's content words that must appear in one chunk.
GROUND_THRESHOLD = 0.75

# Function words + meta-vocabulary (words about answering, not factual content).
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "at", "for", "by", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "this", "that", "these", "those", "he", "she",
    "they", "them", "his", "her", "their", "we", "you", "i", "me", "my", "not",
    "no", "do", "does", "did", "done", "has", "have", "had", "will", "would",
    "shall", "should", "can", "could", "may", "might", "must", "than", "then",
    "there", "here", "when", "where", "who", "whom", "whose", "which", "what",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "so", "too", "very", "just", "about",
    "into", "over", "under", "after", "before", "between", "out", "up", "down",
    "off", "again", "further", "once", "per", "also", "via", "above", "below",
    "during", "while", "until", "because", "against", "around", "among",
    "whether", "either", "neither", "within", "without", "towards",
    # meta-vocabulary: words about the source / the act of answering
    "document", "documents", "context", "source", "sources", "answer",
    "answers", "according", "based", "provided", "excerpt", "excerpts",
    "question", "given", "above-mentioned", "following", "mentioned",
    # reporting verbs (the factual core is the reported fact itself)
    "state", "states", "stated", "stating", "say", "says", "said", "saying",
    "mention", "mentions", "show", "shows", "shown", "indicate", "indicates",
    "indicated", "record", "records", "recorded", "note", "notes",
}

# Sentences that assert the ABSENCE of information rather than a source fact.
ABSTENTION_PATTERNS = [
    r"\bnot (?:enough|sufficient|available|specified|mentioned|provided|stated|recorded|present|included|listed)\b",
    r"\binsufficient (?:information|evidence|context|data|detail)\b",
    r"\b(?:does|do|did|is|are|was|were) not (?:contain|include|specify|mention|provide|state|identify|establish|record|show|say|list|indicate|answer)\b",
    r"\b(?:doesn't|don't|didn't|isn't|aren't|wasn't|weren't|can't|couldn't|won't)\b",
    r"\bno (?:information|mention|record|details|such|answer|evidence|clause|entry|data)\b",
    r"\b(?:cannot|can not|could not) be (?:determined|found|answered|established|identified|confirmed)\b",
    r"\b(?:unable|not possible) to (?:answer|determine|identify|establish|confirm)\b",
    r"\bi (?:do not|don't) know\b",
    r"\bnot (?:in|within|part of) the (?:provided )?(?:context|document|evidence|file|text|record)\b",
    r"\b(?:the )?(?:context|document|evidence|file|record) (?:does|do|did|is|are|was|were) not\b",
    r"\bnothing (?:in|about)\b",
]
_ABSTENTION_RE = re.compile("|".join(f"(?:{p})" for p in ABSTENTION_PATTERNS), re.I)

_MARKER_RE = re.compile(r"\[E\d+\]|\*\*|__|`+", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_NUMBER_RE = re.compile(r"\d[\d:.,/\-]*")
_REFUSAL = "Insufficient evidence in the provided documents."


# --------------------------------------------------------------------------- text
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).lower().strip()


def _strip_markers(text: str) -> str:
    return _MARKER_RE.sub(" ", str(text or ""))


def stem(token: str) -> str:
    """Light, deterministic suffix strip — identical on claim and source side."""
    t = token.lower()
    if len(t) > 5 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 4 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


def content_tokens(text: str) -> list[str]:
    """Stemmed content words (stopwords/meta-words removed), order preserved."""
    out = []
    for w in _WORD_RE.findall(_strip_markers(text)):
        w = w.lower()
        s = stem(w)
        if len(s) >= 3 and s not in STOPWORDS and s not in {stem(x) for x in STOPWORDS}:
            out.append(s)
    return out


def critical_tokens(text: str) -> list[str]:
    """Numbers / times / amounts / IDs — required verbatim, never paraphrasable."""
    raw = _NUMBER_RE.findall(_strip_markers(text))
    return [tok.strip(".,;:/-") or tok for tok in raw if tok.strip(".,;:/")]


# ------------------------------------------------------------------ extraction
def extract_claims(answer: str, refusal: str = _REFUSAL) -> list[str]:
    """Sentence-level claim candidates. Deterministic; no model involved."""
    if not answer or not answer.strip():
        return []
    if answer.strip() == refusal.strip():
        return []
    claims: list[str] = []
    for sent in _SENT_SPLIT_RE.split(_strip_markers(answer)):
        sent = re.sub(r"^[-*•)\s]+", "", sent).strip()
        if not sent or sent == refusal:
            continue
        if len(sent.split()) < 3:          # "Yes." / "No." are not factual claims
            continue
        claims.append(sent)
    return claims


def is_abstention(sentence: str) -> bool:
    return bool(_ABSTENTION_RE.search(normalize(sentence)))


def looks_like_refusal(answer: str, refusal: str = _REFUSAL) -> bool:
    """Baseline refusal detector: exact refusal string OR any abstention sentence."""
    if not answer or not answer.strip():
        return False
    if answer.strip() == refusal.strip():
        return True
    return any(is_abstention(s) for s in extract_claims(answer, refusal))


# ------------------------------------------------------------------- grounding
def check_claim(claim: str, evidence: Iterable) -> dict:
    """Trace one claim to a single retrieved evidence chunk.

    `evidence` items are app.models.Evidence (duck-typed: .id/.filename/.page/.text).
    Returns applicable/grounded/evidence_ids/filename/page/reason.
    """
    raw = _strip_markers(claim)
    words = sorted(set(content_tokens(raw)))
    criticals = critical_tokens(raw)

    if not words and not criticals:
        return {"applicable": False, "grounded": False, "evidence_ids": [],
                "filename": None, "page": None, "reason": "no factual content words"}

    evidence = list(evidence)
    if not evidence:
        return {"applicable": True, "grounded": False, "evidence_ids": [],
                "filename": None, "page": None,
                "reason": "no evidence was retrieved, so nothing can support this claim"}

    best = None  # (crit_ok, recall, ev, matched, missing)
    for ev in evidence:
        src_norm = normalize(ev.text)
        src_words = set(content_tokens(ev.text))
        matched = [w for w in words if w in src_words]
        missing = [w for w in words if w not in src_words]
        missing_crit = [c for c in criticals if c not in src_norm]
        recall = (len(matched) / len(words)) if words else 1.0
        cand = (not missing_crit, recall, ev, matched, missing, missing_crit)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand

    crit_ok, recall, ev, matched, missing, missing_crit = best
    grounded = crit_ok and (not words or recall >= GROUND_THRESHOLD)

    if grounded:
        reason = (
            f"matched {len(matched)}/{len(words)} content token(s) in {ev.id}"
            + (f" and all {len(criticals)} number(s) verbatim" if criticals else "")
        )
        return {"applicable": True, "grounded": True, "evidence_ids": [ev.id],
                "filename": ev.filename, "page": ev.page, "reason": reason}

    if not crit_ok:
        reason = f"required number(s) {', '.join(missing_crit)} not found in any retrieved evidence"
    else:
        shown_missing = ", ".join(missing[:6]) or "-"
        reason = (f"best chunk {ev.id} matched only {len(matched)}/{len(words)} "
                  f"content token(s) (missing: {shown_missing}) — below the "
                  f"{int(GROUND_THRESHOLD * 100)}% traceability threshold")
    return {"applicable": True, "grounded": False, "evidence_ids": [],
            "filename": None, "page": None, "reason": reason}


def evaluate_answer_claims(answer: str, evidence: Iterable,
                           refusal: str = _REFUSAL) -> list[dict]:
    """Extract claims and ground each one against the retrieved evidence."""
    ev_list = list(evidence)
    rows: list[dict] = []
    for claim in extract_claims(answer, refusal):
        verdict = check_claim(claim, ev_list)
        if not verdict["applicable"]:
            continue
        if verdict["grounded"]:
            rows.append({"claim": claim, "kind": "claim", "grounded": True,
                         "evidence_ids": verdict["evidence_ids"],
                         "filename": verdict["filename"], "page": verdict["page"],
                         "reason": verdict["reason"]})
        elif is_abstention(claim):
            rows.append({"claim": claim, "kind": "abstention", "grounded": None,
                         "evidence_ids": [], "filename": None, "page": None,
                         "reason": "asserts absence of information; excluded from "
                                   "groundedness (neither supported nor unsupported)"})
        else:
            rows.append({"claim": claim, "kind": "claim", "grounded": False,
                         "evidence_ids": [], "filename": None, "page": None,
                         "reason": verdict["reason"]})
    return rows


# ------------------------------------------------------------- retrieval check
def find_expected_source(evidence: Iterable, expected_refs: list[dict]):
    """Any-of match: (found?, matching Evidence, matched E-ID)."""
    if not expected_refs:
        return False, None
    for ev in evidence:
        for r in expected_refs:
            if ev.filename != r.get("filename") or int(ev.page) != int(r.get("page")):
                continue
            anchor = r.get("anchor")
            if anchor is None or normalize(anchor) in normalize(ev.text):
                return True, ev
    return False, None


# ---------------------------------------------------------------- aggregation
def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator * 100, 1) if denominator else None


def summarize_system(qresults: list[dict], system: str) -> dict:
    total = supported = abstentions = errors = 0
    correct_refusals = unanswerable = retrieval_hits = answerable = 0

    for qr in qresults:
        s = qr[system]
        if s.get("error"):
            errors += 1
        for row in s.get("claims", []):
            if row["kind"] == "claim":
                total += 1
                supported += 1 if row["grounded"] else 0
            else:
                abstentions += 1
        if qr["answerable"]:
            answerable += 1
            if s.get("retrieval", {}).get("success"):
                retrieval_hits += 1
        else:
            unanswerable += 1
            if s.get("refused"):
                correct_refusals += 1

    unsupported = total - supported
    return {
        "groundedness": _rate(supported, total),
        "total_claims": total,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "abstentions": abstentions,
        "retrieval_success": _rate(retrieval_hits, answerable),
        "retrieval_hits": retrieval_hits,
        "answerable_questions": answerable,
        "correct_refusals": correct_refusals,
        "unanswerable_questions": unanswerable,
        "errors": errors,
    }


def _diff(baseline: Optional[float], legallens: Optional[float]) -> Optional[float]:
    if baseline is None or legallens is None:
        return None
    return round(legallens - baseline, 1)


def aggregate(qresults: list[dict]) -> dict:
    """Full results object: per-system summaries + comparison + claim details."""
    baseline = summarize_system(qresults, "baseline")
    legallens = summarize_system(qresults, "legallens")

    details: list[dict] = []
    for qr in qresults:
        for system in ("baseline", "legallens"):
            for row in qr[system].get("claims", []):
                details.append({
                    "question_id": qr["id"],
                    "question": qr["question"],
                    "system": system,
                    "kind": row["kind"],
                    "claim": row["claim"],
                    "grounded": row["grounded"],
                    "evidence_ids": row["evidence_ids"],
                    "filename": row["filename"],
                    "page": row["page"],
                    "reason": row["reason"],
                })

    return {
        "baseline": baseline,
        "legallens": legallens,
        "comparison": {
            "groundedness_improvement": _diff(baseline["groundedness"],
                                              legallens["groundedness"]),
            "retrieval_improvement": _diff(baseline["retrieval_success"],
                                           legallens["retrieval_success"]),
        },
        "questions": qresults,
        "details": details,
        "dataset_size": len(qresults),
    }
