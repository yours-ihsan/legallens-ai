"""Deterministic premise check — pure, no LLM, no network.

Supports the chat pipeline's THREE-WAY evidence decision. The question's
presuppositions are extracted by Python, and each one is marked using ONLY
the retrieved chunks:

    SUPPORTED     a chunk sentence STATES the assumption: every content word
                  (hedging words like "alleged" carry no factual commitment
                  and are ignored) and every number in it appears in ONE
                  non-negated, non-absence sentence. A tiny paraphrase table
                  admits the usual legal restatements ("maximum" is stated by
                  "shall not exceed"), never bare word overlap.
    CONTRADICTED  a chunk sentence STATES THE OPPOSITE: the same content
                  words under a negated predicate ("The laptop was not found
                  in Manoj Verma's bag"), a competing number where the
                  assumption quotes its own, or (pre-existing detector) an
                  absence / non-establishment statement in the offence
                  domain that defeats a guilt presumption.
    NOT_FOUND     no chunk states it — silence is never support.

Possessives are part of the token: "laptop's serial number" requires a chunk
that actually states a serial number; a chunk mentioning a bare "laptop" does
not satisfy it.

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

# A question's presupposition must be stated in full by a chunk sentence —
# partial word overlap is NOT support (see the module docstring). Numbers in
# an assumption must always appear verbatim in the same sentence.
MAX_ASSUMPTIONS = 4

# Deterministic paraphrase table: assumption-side content word -> pattern that
# a chunk sentence must match to count as stating that word. Legal documents
# restate the same fact ("the maximum aggregate liability" is stated by a
# clause that says the aggregate liability "shall not exceed USD 50,000").
_PARAPHRASE_RE = {
    "maximum": r"\b(?:not\s+exceed|up\s+to|at\s+most|capped?\s+at|limit(?:ed)?\s+to)\b",
    "minimum": r"\b(?:at\s+least|not\s+less\s+than|no\s+less\s+than)\b",
}

# Scoped negation: a predicate actually being negated. A stray "not" is not
# enough — contract caps read "shall not exceed", which ASSERTS the cap, so
# the bare "shall ... not" is deliberately excluded from this pattern.
_SCOPED_NEG_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|am|do|does|did|has|have|had|"
    r"will|would|can|could|should|must)\s+not\s+\w+"
    r"|\bnever\s+\w+"
    r"|\bno\s+(?:[A-Za-z]+\s+){0,2}[A-Za-z]+\b"
    r"|\bnot\s+(?:in|from|at|by|with)\b", re.I)

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

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    """Sentence-split a chunk. PDF line breaks are layout, not semantics:
    chunks join extracted lines with newlines, so collapse all whitespace
    BEFORE splitting on [.!?] ("...shall\\n\\nnot exceed..." is one sentence)."""
    collapsed = re.sub(r"\s+", " ", str(text or ""))
    return [s for s in _SENT_SPLIT_RE.split(collapsed) if s.strip()]


# Placeholder so apostrophes BETWEEN letters survive the quote strip below.
_INNER_APOS_RE = re.compile(r"(?<=[A-Za-z])'(?=[A-Za-z])")
_APOS = "\u0007"


def _clean(fragment: str) -> str:
    # Possessives are part of the token: keep "laptop's" one word so it is
    # never confused with the bare noun "laptop" during marking.
    frag = _INNER_APOS_RE.sub(_APOS, str(fragment))
    frag = re.sub(r"[\s.,;:!?\"'()\[\]]+", " ", frag).strip().replace(_APOS, "'")
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


def _propositional_core(assumption: str) -> str:
    """Strip hedging so only the asserted content is matched for entailment.

    "the alleged theft" presupposes that a theft happened; the hedge carries
    no factual commitment, so it must not be required verbatim in the chunks.
    """
    kept = [w for w in assumption.split() if not _HEDGE_RE.fullmatch(w.lower())]
    return " ".join(kept) or assumption


def _negation_mismatch(core: str, sentence: str) -> bool:
    """Sentence negates a predicate the core asserts (or the core is un-negated)."""
    return bool(_SCOPED_NEG_RE.search(normalize(sentence))) and \
        not bool(_SCOPED_NEG_RE.search(normalize(core)))


def _word_stated(word: str, src_words: set, src_norm: str) -> bool:
    """Chunk sentence states this content word (literally or via paraphrase)."""
    if word in src_words:
        return True
    pattern = _PARAPHRASE_RE.get(word)
    return bool(pattern and re.search(pattern, src_norm))


def _supporting_chunk(core: str, evidence: list[Evidence]):
    """Chunk containing ONE sentence that states the core (full entailment)."""
    words = set(content_tokens(core))
    crits = critical_tokens(core)
    if not words and not crits:
        return None
    for ev in evidence:
        for sent in _sentences(ev.text):
            if is_abstention(sent) or _negation_mismatch(core, sent):
                continue
            src_norm = normalize(sent)
            if any(c not in src_norm for c in crits):
                continue
            src_words = set(content_tokens(sent))
            if all(_word_stated(w, src_words, src_norm) for w in words):
                return ev
    return None


def _opposite_chunk(core: str, evidence: list[Evidence]):
    """Chunk containing ONE sentence that states the OPPOSITE of the core:
    the same content words under a negated predicate, or a competing number
    where the assumption quotes its own (verbatim) figures."""
    words = set(content_tokens(core))
    crits = critical_tokens(core)
    if not words:
        return None
    for ev in evidence:
        for sent in _sentences(ev.text):
            if is_abstention(sent):   # absence statements -> _contradiction_chunk
                continue
            src_norm = normalize(sent)
            if not words <= set(content_tokens(sent)):
                continue
            if _negation_mismatch(core, sent):
                return ev
            if crits and not any(c in src_norm for c in crits) \
                    and critical_tokens(sent):
                return ev
    return None


def _contradiction_chunk(assumption: str, evidence: list[Evidence],
                         guilt: bool = False):
    """Chunk whose absence statements defeat the assumption."""
    a_tokens = set(content_tokens(assumption))
    for ev in evidence:
        for sent in _sentences(ev.text):
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
    """Mark one assumption by entailment against the chunks (three-way)."""
    if _is_guilt_presumption(assumption):
        ev = _contradiction_chunk(assumption, evidence, guilt=True)
        if ev is not None:
            return {"assumption": assumption, "verdict": CONTRADICTED,
                    "evidence_id": ev.id,
                    "note": f"the material denies that responsibility is "
                            f"established ({ev.id}, page {ev.page})"}
    core = _propositional_core(assumption)
    ev = _opposite_chunk(core, evidence)       # a chunk states the opposite
    if ev is not None:
        return {"assumption": assumption, "verdict": CONTRADICTED,
                "evidence_id": ev.id,
                "note": f"the material states the opposite ({ev.id}, page {ev.page})"}
    ev = _supporting_chunk(core, evidence)     # a chunk states it
    if ev is not None:
        return {"assumption": assumption, "verdict": SUPPORTED,
                "evidence_id": ev.id,
                "note": f"found in {ev.id} ({ev.filename}, page {ev.page})"}
    ev = _contradiction_chunk(assumption, evidence, guilt=False)  # absence
    if ev is not None:
        return {"assumption": assumption, "verdict": CONTRADICTED,
                "evidence_id": ev.id,
                "note": f"the material records the absence of what the "
                        f"assumption needs ({ev.id}, page {ev.page})"}
    return {"assumption": assumption, "verdict": NOT_FOUND,
            "evidence_id": None,
            "note": "no retrieved chunk states this assumption"}


def check_premises(question: str, evidence: list[Evidence]) -> list[dict]:
    """Extract the question's assumptions and mark each from the chunks."""
    return [mark_assumption(a, evidence) for a in extract_assumptions(question)]
