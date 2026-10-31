"""Grounded answer generation — the system's trust boundary.

HARD INVARIANT: the LLM receives ONLY the retrieved evidence packet
(build_packet). Page numbers in the response are derived from vector-store
metadata, never from the model: the model cites evidence IDs (E1, E2, ...)
and Python maps those IDs to (filename, page).

FALSE PREMISES: when the question contains a false or unsupported assumption
that the evidence contradicts, the model must produce a grounded correction
("The sources state X, not Y") with citations — refusal is reserved for
questions the evidence says nothing relevant about.

THREE-WAY EVIDENCE DECISION (replaces the old single threshold refusal):
  1. REFUSE  - best retrieval score < LOW_FLOOR (0.10): nothing relevant was
               retrieved, the LLM is not consulted (stage=retrieval), OR the
               deterministic premise check finds every assumption NOT_FOUND
               in the chunks (stage=premise).
  2. CORRECT - >= 1 assumption CONTRADICTED by the chunks: the answer must
               open with a cited correction; decision=premise_correction.
  3. ANSWER  - otherwise: the evidence is passed to the LLM (any score above
               the floor), decision=grounded_answer.
Whichever branch runs, validate() still checks EVERY claim of the produced
answer — the premise step gates, it never replaces verification.

Refusal stages (reported as refusal_stage, all return the exact REFUSAL string):
  retrieval - nothing indexed / top score < LOW_FLOOR (no LLM call)
  premise   - every assumption NOT_FOUND in the chunks (no LLM call)
  system    - LLM or JSON-parse failure
  model     - the model itself declined (insufficient=true / empty answer)
  verifier  - validate() rejected the model's claims (non-existent evidence IDs,
              missing citations, unverified page numbers in the answer)
"""
import re

from .config import BAND_HIGH, BAND_MEDIUM, LOW_FLOOR
from .evaluation.metrics import (
    check_claim,
    content_tokens,
    critical_tokens,
    extract_claims,
    is_abstention,
    normalize,
)
from .llm import LLMError, extract_json, run_llm
from .models import AnswerResponse, Citation, Confidence, Evidence
from .premise import (
    CONTRADICTED,
    NOT_FOUND,
    STOPWORDS,
    _IDENTITY_DENIAL_RE,
    _PERSON_RE,
    _sentences,
    check_premises,
)
from .retriever import retrieve

REFUSAL = "Insufficient evidence in the provided documents."

SYSTEM_PROMPT = f"""You are LegalLens AI, a grounded legal research assistant.

STRICT RULES:
1. Answer ONLY from the EVIDENCE provided below. The evidence is the entire universe of information available to you.
2. Never invent facts, laws, statutes, case names, legal citations, article numbers, party names, dates, or amounts.
3. Never write page numbers or filenames inside the answer text. You may reference evidence with [E1]-style markers, and you must list every supporting evidence ID in "citations".
4. Every factual sentence must be supported by at least one cited evidence ID.
5. CONTRADICTION & FALSE-PREMISE RULE: the question may contain a false, unsupported, or contradicted allegation or assumption.
   - If the evidence contradicts the allegation, or if the question asks whether the document proves or establishes an allegation that is contradicted or not established:
     You MUST explicitly state that the document does not establish it (e.g. "No. The document does not establish that [person] [allegation].").
     Explain what the evidence actually records (citing the absence of witness identification, no property recovered, or that access alone does not prove theft), citing the supporting evidence IDs [E#].
     NEVER state, agree, or imply that a contradicted or unproven allegation is established or supported.
   - For other contradicted assumptions, provide a grounded correction: "The sources state X, not Y." and cite the supporting evidence IDs.
   - DO NOT refuse to answer solely because the question contains a false premise. Correct the false premise using the available evidence, explain what the evidence actually establishes, and include citations. Never invent facts or citations.
6. REFUSAL RULE: set "insufficient" to true ONLY when the evidence says nothing relevant to the question. When refusing, set "citations" to [] and set "answer" to exactly: {REFUSAL}
7. "confidence" is your 0-1 confidence that the evidence fully supports the answer.
8. Respond with ONLY a JSON object and no other text, using exactly this schema:
{{"answer": string, "insufficient": boolean, "confidence": number, "citations": ["E1", ...]}}
9. A PREMISE CHECK block may appear above the evidence. Its verdicts are computed by the system from the evidence and are authoritative:
   - For every CONTRADICTED allegation, OPEN with an explicit correction that the document does not establish the claim, and explain using the source evidence.
   - Never assert a NOT_FOUND assumption as fact.
   - You may rely on SUPPORTED assumptions with citations."""


def premise_block(premises: list) -> str:
    """System-computed premise verdicts for the prompt (empty when none)."""
    if not premises:
        return ""
    lines = ["PREMISE CHECK (computed by the system from the evidence — authoritative):"]
    for p in premises:
        ref = f" -> see [{p['evidence_id']}]" if p.get("evidence_id") else ""
        lines.append(f'- ASSUMPTION "{p["assumption"]}" — {p["verdict"]}{ref}')
    lines += [
        "Follow these verdicts exactly:",
        "* CONTRADICTED: The evidence contradicts this assumption. OPEN the answer by explicitly stating "
        'that the document does not establish this claim (e.g. "No. The document does not establish that [X] [E#]" '
        'or "The sources state X, not Y [E#]"), and explain what the evidence actually records using the source facts. '
        "NEVER assert or imply that a contradicted assumption is true or established.",
        "* NOT_FOUND: the evidence does not establish this — never assert it "
        "as fact; state what the evidence does record instead, with citations.",
        "* SUPPORTED: you may rely on it and cite the supporting evidence ID.",
    ]
    return "\n".join(lines)


def _band(top_score: float) -> str:
    if top_score >= BAND_HIGH:
        return "High"
    if top_score >= BAND_MEDIUM:
        return "Medium"
    return "Low"


def build_packet(evidence: list[Evidence]) -> str:
    blocks = [
        f"[{ev.id}] ({ev.filename}, page {ev.page}, relevance {ev.score:.2f})\n{ev.text}"
        for ev in evidence
    ]
    return "EVIDENCE:\n\n" + "\n\n---\n\n".join(blocks)


def _confidence(top_score: float, model_conf: float) -> Confidence:
    retrieval = max(0.0, min(top_score, 1.0))
    model_conf = max(0.0, min(model_conf, 1.0))
    return Confidence(
        score=round(0.5 * retrieval + 0.5 * model_conf, 2),
        band=_band(top_score),
        retrieval=round(retrieval, 3),
        model=round(model_conf, 3),
    )


# Which case of the three-way decision produced this refusal (user-visible).
_STAGE_DECISION = {
    "retrieval": "refusal_low_evidence",
    "premise": "refusal_no_premise_support",
    "system": "refusal_system",
    "model": "refusal_model",
    "verifier": "refusal_verifier",
}


def _refusal(question: str, evidence: list[Evidence], top_score: float, reason: str,
             stage: str, model_conf: float = 0.0,
             premises: list[dict] | None = None) -> AnswerResponse:
    return AnswerResponse(
        question=question,
        answer=REFUSAL,
        insufficient=True,
        citations=[],
        evidence=evidence,
        confidence=_confidence(top_score, model_conf),
        reason=reason,
        refusal_stage=stage,
        decision=_STAGE_DECISION.get(stage, f"refusal_{stage}"),
        premises=list(premises or []),
    )


def _answer_decision(premises: list[dict] | None) -> str:
    """grounded_answer normally; premise_correction when >= 1 contradicted."""
    for p in premises or []:
        if p.get("verdict") == CONTRADICTED:
            return "premise_correction"
    return "grounded_answer"


def validate(question: str, evidence: list[Evidence], top_score: float, data: object,
             premises: list[dict] | None = None) -> AnswerResponse:
    """Pure validation of parsed model output — unit-tested without any LLM call."""
    if not isinstance(data, dict):
        return _refusal(question, evidence, top_score,
                        "model output was not a JSON object", stage="verifier",
                        premises=premises)

    if data.get("insufficient") is True:
        return _refusal(question, evidence, top_score,
                        "model reported insufficient evidence", stage="model",
                        premises=premises)

    answer = str(data.get("answer") or "").strip()
    if not answer or answer == REFUSAL:
        return _refusal(question, evidence, top_score,
                        "model returned no usable answer", stage="model",
                        premises=premises)

    by_id = {ev.id: ev for ev in evidence}
    raw_cites = data.get("citations")
    if not isinstance(raw_cites, list):
        raw_cites = []

    # Evidence IDs in the citations field must all exist.
    for cid in raw_cites:
        if str(cid).strip().upper() not in by_id:
            return _refusal(question, evidence, top_score,
                            f"model cited non-existent evidence {cid!r}", stage="verifier",
                            premises=premises)

    # [E1]-style markers inside the answer text must all exist too.
    text_ids = [m.upper() for m in re.findall(r"\[(E\d+)\]", answer, flags=re.I)]
    for tid in text_ids:
        if tid not in by_id:
            return _refusal(question, evidence, top_score,
                            f"answer references non-existent evidence {tid}", stage="verifier",
                            premises=premises)

    # Union of field citations and valid text markers -> real (filename, page) citations.
    citations: list[Citation] = []
    seen: set[str] = set()
    for cid in list(raw_cites) + text_ids:
        ev = by_id.get(str(cid).strip().upper())
        if ev is None or ev.id in seen:
            continue
        seen.add(ev.id)
        citations.append(
            Citation(evidence_id=ev.id, filename=ev.filename, page=ev.page, quote=ev.text[:240])
        )

    if not citations:
        return _refusal(question, evidence, top_score,
                        "model provided no citations", stage="verifier",
                        premises=premises)

    # The answer must not reference page numbers outside the retrieved evidence.
    evidence_pages = {ev.page for ev in evidence}
    for match in re.finditer(r"\b(?:page|pp?\.?)\s*(\d+)", answer, re.I):
        if int(match.group(1)) not in evidence_pages:
            return _refusal(question, evidence, top_score,
                            f"answer references unverified page {match.group(1)}",
                            stage="verifier", premises=premises)

    try:
        model_conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        model_conf = 0.0

    return AnswerResponse(
        question=question,
        answer=answer,
        insufficient=False,
        citations=citations,
        evidence=evidence,
        confidence=_confidence(top_score, model_conf),
        reason=None,
        refusal_stage=None,
        decision=_answer_decision(premises),
        premises=list(premises or []),
    )


def answer_question(question: str, doc_ids: list[str] | None = None) -> AnswerResponse:
    evidence = retrieve(question, doc_ids=doc_ids)
    top_score = evidence[0].score if evidence else 0.0

    if not evidence:
        return _refusal(question, [], 0.0,
                        "no indexed documents or no matching chunks", stage="retrieval")

    # --- three-way decision, step 1: LOW floor (weak retrieval -> refuse) ---
    if top_score < LOW_FLOOR:
        return _refusal(
            question, evidence, top_score,
            f"best retrieval score {top_score:.2f} is below the low floor {LOW_FLOOR:.2f}; "
            "the LLM was not consulted",
            stage="retrieval",
        )

    # --- step 2: deterministic premise check over the chunks only ---
    premises = check_premises(question, evidence)
    if premises and all(p["verdict"] == NOT_FOUND for p in premises):
        listed = "; ".join(f'"{p["assumption"]}"' for p in premises)
        return _refusal(
            question, evidence, top_score,
            f"every assumption in the question was NOT_FOUND in the retrieved evidence "
            f"({listed}); the LLM was not consulted",
            stage="premise", premises=premises,
        )

    # --- step 3: LLM on the evidence (any score above the floor) ---
    block = premise_block(premises)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{block + chr(10) + chr(10) if block else ''}"
        f"{build_packet(evidence)}\n\n"
        f"QUESTION: {question}\n\nRespond with the JSON object only."
    )
    try:
        raw = run_llm(prompt)
        data = extract_json(raw)
    except (LLMError, ValueError) as exc:
        return _refusal(question, evidence, top_score,
                        f"model error: {exc}", stage="system", premises=premises)

    # --- step 4: the verifier still checks every claim, whichever branch ---
    return validate(question, evidence, top_score, data, premises=premises)
