"""Deterministic premise check — pure, no LLM, no network.

Supports the chat pipeline's THREE-WAY evidence decision. The question's
presuppositions are extracted by Python, and each one is marked using ONLY
the retrieved chunks:

    SUPPORTED     the chunks contain the assumption (content-word
                  traceability relaxed to >= 50% because a presupposition is
                  a weaker statement than a full claim; any number in it must
                  still appear verbatim, and support only counts when it comes
                  from non-absence sentences)
    CONTRADICTED  the chunks assert the absence / non-establishment of what
                  the assumption needs (deterministic abstention detector +
                  domain overlap; a guilt presumption is defeated by any
                  identity/establishment denial in the offence domain)
    NOT_FOUND     the chunks are silent about it

The verdicts gate the pipeline (answer_generator.answer_question):
    any CONTRADICTED  -> the answer must open with a cited correction
                         ("The sources state X, not Y [E#]")
    every NOT_FOUND   -> exact refusal WITHOUT an LLM call

Whichever branch is taken, the produced answer still goes through the normal
citation verifier — the premise step gates, it never replaces verification.
"""
import re

from .evaluation.metrics import (STOPWORDS, content_tokens, critical_tokens,
                                 is_abstention, normalize, stem)
from .models import Evidence

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_FOUND = "NOT_FOUND"

# A presupposition only needs half of its content words present — it is a
# weaker statement than a claim (e.g. "the maximum aggregate liability" is
# supported by a chunk that says "aggregate liability ... shall not exceed").
PREMISE_SUPPORT = 0.5
MAX_ASSUMPTIONS = 4

# ---------------------------------------------------------------- extraction
_THAT_CLAUSE_RE = re.compile(
    r"\b(?:conclud\w*|confirm\w*|propos\w*|prov[ei]\w*|show\w*|assert\w*|"
    r"claim\w*|allege\w*|admit\w*|acknowledg\w*|believ\w*|think|know|"
    r"say(?:ing|s)?|state\w*|report\w*)\s+that\s+([^?.,;]{4,160})", re.I)

_WH_EVENT_RE = re.compile(
    r"\b(?:why|how|when|where)\s+(?:did|do|does|was|were|is|are|has|have|had)\s+"
    r"([^?]{4,160})", re.I)

_NP_RE = re.compile(
    r"\b(the\s+(?:[A-Za-z][A-Za-z'/-]*\s+){0,2}[A-Za-z][A-Za-z'/-]*)", re.I)

# Head nouns that are meta-reference, never a fact of the case.
_NP_DENY = {
    "case", "question", "matter", "document", "documents", "source", "sources",
    "following", "above", "below", "answer", "file", "text", "context", "user",
    "system", "thing", "things", "situation", "page", "pages", "list", "form",
}

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _clean(fragment: str) -> str:
    frag = re.sub(r"[\s.,;:!?\"'()\[\]]+", " ", str(fragment)).strip()
    # trim trailing function words ("the alleged theft in" -> "the alleged theft")
    words = frag.split()
    while len(words) > 1 and stem(words[-1].lower()) in STOPWORDS:
        words.pop()
    return " ".join(words)


def extract_assumptions(question: str) -> list[str]:
    """Presuppositions embedded in the question — deterministic, no model."""
    found: list[str] = []
    for regex in (_THAT_CLAUSE_RE, _WH_EVENT_RE):
        for m in regex.finditer(question or ""):
            frag = _clean(m.group(1))
            if len(frag.split()) >= 2:
                found.append(frag)
    for m in _NP_RE.finditer(question or ""):
        frag = _clean(m.group(1))
        words = frag.split()
        if not words:
            continue
        head = re.sub(r"[^a-z]", "", words[-1].lower())
        if not head or head in _NP_DENY:
            continue
        if not content_tokens(frag):     # pure meta/function words
            continue
        found.append(frag)

    out, seen = [], set()
    for a in found:
        key = a.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    # Drop assumptions contained in a longer one ("the accused" inside
    # "the accused is guilty") — one premise per underlying presupposition.
    out = [a for a in out
           if not any(a != b and a.lower() in b.lower() for b in out)]
    return out[:MAX_ASSUMPTIONS]


# ------------------------------------------------------------------- marking
# Unhedged assertions of guilt OR of arrest/charge/bail/conviction — the
# presuppositions legal questions most often get wrong.
_OFFENCE_RE = re.compile(
    r"\b(?:guilty|guilt|convict\w*|commit(?:ted|s|ting)?|stole|stolen|stealing|"
    r"theft|thief|robbery|murder|fraud\w*|cheat\w*|assault\w*|kill\w*|rape\w*|"
    r"offence|offense|crime|criminal|arrest\w*|charg\w*|detain\w*|custody|"
    r"bail|sentenc\w*)\b", re.I)

# Hedged assumptions ("the alleged theft") are not guilt presumptions: the
# question itself already treats the offence as unestablished.
_HEDGE_RE = re.compile(
    r"\b(?:alleg\w*|claim\w*|report\w*|if|whether|maybe|possible|potential|"
    r"presum\w*|assum\w*|hypothet\w*|might|may|could)\b", re.I)

_PERSON_RE = re.compile(
    r"\b(?:accused|appellant|respondent|petitioner|defendant|convict\w*|"
    r"perpetrator|assailant|suspect\w*)\b"
    r"|\b[A-Z][a-z]{2,}(?:\s+[A-Z][A-Za-z]{2,})?\b")

# Absence statements that deny responsibility / identity / charge.
_IDENTITY_DENIAL_RE = re.compile(
    r"\b(?:does|do|did)\s+not\s+(?:establish|identify|state|show|prove|record|"
    r"name|confirm|support|indicate)"
    r"|\bno\s+(?:evidence|record|material|information|witness|proof)"
    r"|\bnot\s+(?:established|identified|named|charged|arrested|convicted|"
    r"specified|recorded)\b"
    r"|\b(?:was|is|were|are)\s+not\s+(?:arrested|charged|convicted|identified|"
    r"named)\b", re.I)

_OFFENCE_DOMAIN_RE = re.compile(
    r"\b(?:theft|stole|stolen|steal|took|taking|taken|commit\w*|accus\w*|"
    r"arrest\w*|charg\w*|detain\w*|custody|guilt\w*|offence|offense|crime|"
    r"robbery|murder|fraud|cheat\w*|assault\w*|kill\w*|bail|liabilit\w*|"
    r"liable|negligen\w*)\b", re.I)


def _is_guilt_presumption(assumption: str) -> bool:
    """Unhedged assertion that a person committed an offence."""
    if not _OFFENCE_RE.search(assumption):
        return False
    if _HEDGE_RE.search(assumption):
        return False
    for m in _PERSON_RE.finditer(assumption):
        token = m.group(0)
        if len(token) >= 3 and stem(token.lower()) not in STOPWORDS \
                and token.lower() not in STOPWORDS:
            return True
    return False


def _supporting_chunk(assumption: str, evidence: list[Evidence]):
    """Chunk whose NON-absence sentences trace the assumption (>= 50%)."""
    words = sorted(set(content_tokens(assumption)))
    crits = critical_tokens(assumption)
    if not words and not crits:
        return None
    best = None
    for ev in evidence:
        positive = " ".join(
            s for s in _SENT_SPLIT_RE.split(ev.text or "") if not is_abstention(s))
        src_norm = normalize(positive)
        src_words = set(content_tokens(positive))
        missing_crit = [c for c in crits if c not in src_norm]
        matched = sum(1 for w in words if w in src_words)
        recall = (matched / len(words)) if words else 1.0
        cand = (not missing_crit, recall, ev)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    if best is None:
        return None
    crit_ok, recall, ev = best
    if crit_ok and (words or crits) and recall >= PREMISE_SUPPORT:
        return ev
    return None


def _contradiction_chunk(assumption: str, evidence: list[Evidence],
                         guilt: bool = False):
    """Chunk whose absence statements defeat the assumption."""
    a_tokens = set(content_tokens(assumption))
    for ev in evidence:
        for sent in _SENT_SPLIT_RE.split(ev.text or ""):
            if not is_abstention(sent):
                continue
            if guilt:
                if _IDENTITY_DENIAL_RE.search(sent) and _OFFENCE_DOMAIN_RE.search(sent):
                    return ev
                continue
            shared = a_tokens & set(content_tokens(sent))
            if len(shared) >= 2:
                return ev
            if shared and _OFFENCE_DOMAIN_RE.search(sent):
                return ev
    return None


def mark_assumption(assumption: str, evidence: list[Evidence]) -> dict:
    """Mark one assumption SUPPORTED / CONTRADICTED / NOT_FOUND from chunks."""
    if _is_guilt_presumption(assumption):
        ev = _contradiction_chunk(assumption, evidence, guilt=True)
        if ev is not None:
            return {"assumption": assumption, "verdict": CONTRADICTED,
                    "evidence_id": ev.id,
                    "note": f"the material denies that responsibility is "
                            f"established ({ev.id}, page {ev.page})"}
    ev = _supporting_chunk(assumption, evidence)
    if ev is not None:
        return {"assumption": assumption, "verdict": SUPPORTED,
                "evidence_id": ev.id,
                "note": f"found in {ev.id} ({ev.filename}, page {ev.page})"}
    ev = _contradiction_chunk(assumption, evidence, guilt=False)
    if ev is not None:
        return {"assumption": assumption, "verdict": CONTRADICTED,
                "evidence_id": ev.id,
                "note": f"the material states the opposite ({ev.id}, page {ev.page})"}
    return {"assumption": assumption, "verdict": NOT_FOUND,
            "evidence_id": None,
            "note": "no retrieved chunk mentions this assumption"}


def check_premises(question: str, evidence: list[Evidence]) -> list[dict]:
    """Extract the question's assumptions and mark each from the chunks."""
    return [mark_assumption(a, evidence) for a in extract_assumptions(question)]
