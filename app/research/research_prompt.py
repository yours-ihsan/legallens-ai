"""Legal Research prompt and DETERMINISTIC legal-issue identification.

EVIDENCE FIRST -> VERIFY -> REASON -> ANSWER.

The legal issue statement and the statute-search terms are computed by Python
from a fixed topic lexicon — never by the model — so they cannot hallucinate
and they are unit-testable offline. The model receives them alongside the
evidence packet and may only reason from that packet.
"""
import re

from ..answer_generator import REFUSAL

# Allowed research_type values (validated before any retrieval / LLM call).
RESEARCH_TYPES = ("general", "law", "judgment", "precedent")

# (keywords, short topic label, statute search terms) — first match wins.
# More specific topics come first so "identity theft" does not match "theft".
TOPICS: list[tuple[tuple[str, ...], str, str]] = [
    (("cyber", "hacking", "phishing", "identity theft", "data breach"),
     "cyber offences", "cyber offence information technology"),
    (("bail", "arrested", "arrest", "custody", "detention", "remand"),
     "bail and arrest", "bail arrest"),
    (("theft", "stolen", "steal", "stole", "misappropriat"),
     "theft", "theft dishonestly movable property"),
    (("robbery", "snatched", "snatch"),
     "robbery", "robbery"),
    (("cheating", "cheated", "fraud", "defraud", "forgery"),
     "cheating and fraud", "cheating fraud"),
    (("trespass", "break-in", "break in", "unauthorised entry",
      "unauthorized entry"),
     "criminal trespass", "criminal trespass"),
    (("assault", "criminal force", "hurt", "attacked"),
     "assault and criminal force", "assault criminal force"),
    (("murder", "homicide", "killed"),
     "homicide", "murder"),
    (("defamation", "libel", "slander"),
     "defamation", "defamation"),
    (("affidavit", "notary", "sworn"),
     "affidavit and sworn statements", "affidavit sworn statement"),
    (("notice", "demand notice"),
     "legal notice", "legal notice demand"),
    (("contract", "agreement", "liability", "clause", "indemnity"),
     "contract and civil liability", "contract breach liability"),
    (("employment", "terminated", "termination", "salary", "employer"),
     "employment and termination", "employment termination"),
    (("tenant", "tenancy", "landlord", "rent", "lease"),
     "tenancy and lease", "tenancy lease rent"),
    (("negligence", "negligent"),
     "negligence", "negligence"),
    (("witness", "statement", "evidence"),
     "evidence and witness statements", "evidence witness statement"),
    (("property", "possession"),
     "property and possession", "property possession"),
    (("guilty", "accused", "offence", "crime"),
     "criminal liability of an accused", "offence criminal liability"),
]

_WORD_RE = re.compile(r"[a-z]+")


def _normalize(text: str) -> str:
    return " " + re.sub(r"\s+", " ", str(text or "")).lower() + " "


def find_topic(question: str, case_context: str = "") -> tuple[str, str] | None:
    """(topic label, statute search terms) for the first matching topic, else None."""
    haystack = _normalize(f"{question} {case_context}")
    for keywords, label, terms in TOPICS:
        if any(kw in haystack for kw in keywords):
            return label, terms
    return None


def identify_legal_issue(question: str, case_context: str = "") -> str:
    """Cautious, deterministic legal-issue statement shown to the user.

    Never converts an allegation into an established fact: the wording is
    "potentially relevant" / "concerning" and is computed by Python only.
    """
    topic = find_topic(question, case_context)
    if topic:
        return f"Potentially relevant legal provisions concerning {topic[0]}."
    clean = re.sub(r"\s+", " ", question or "").strip()
    return f'The issue concerns the question as posed: "{clean}".'


def legal_search_terms(question: str, case_context: str = "") -> str:
    """Query sent to the legal-source search (India Code + local statute store)."""
    topic = find_topic(question, case_context)
    if topic:
        return topic[1]
    words = [w for w in _WORD_RE.findall(_normalize(question)) if len(w) > 3]
    return " ".join(words[:6]) or question.strip()[:80]


# --------------------------------------------------------------------- prompt
RESEARCH_SYSTEM_PROMPT = f"""You are a grounded legal research assistant.

You must reason ONLY from:
  1. VERIFIED LEGAL SOURCES retrieved by the system (blocks the classification
     list marks as LEGAL AUTHORITY), and
  2. VERIFIED FACTS from the uploaded case documents (blocks marked CASE
     MATERIAL).
The EVIDENCE packet below is the entire universe of information available to
you. The SOURCE CLASSIFICATION list is computed by the system and is
authoritative — a block is legal authority only if the list says so.

STRICT RULES:
1. Never invent a statute, legal section, judgment, case citation, court, URL,
   date, party name, case number or case fact. Ever.
2. Cite every material legal claim with the [E1]-style marker of the block
   that LITERALLY contains it, placed INSIDE the sentence before its full
   stop. Cite every case-specific factual claim the same way, using a CASE
   MATERIAL block. One block per sentence.
3. Use the source's own wording (quote or minimally adapt) for statutory
   language, names, dates and events so every statement stays verifiably
   traceable — paraphrase is not accepted by the verifier.
4. NEVER treat an allegation as an established fact. The source material
   ALLEGES or REPORTS; say "the source records/alleges/reports ...".
   NEVER state or imply that any person is guilty or committed an offence.
5. FALSE-PREMISE RULE: if the question presumes guilt or that an offence has
   been proved (e.g. "sections that prove the accused is guilty", "did X commit
   the theft"), DO NOT comply with the premise. Give a grounded correction:
   state what the source material records and what it does NOT establish, with
   citations, and explain what information would be needed.
6. Clearly distinguish legal authority from case facts, and verified
   information from missing information. Never present an unverified item as
   verified authority.
7. Only cite LEGAL AUTHORITY blocks for entries in "relevant_law" and
   "judgments". If no retrieved block supports a provision you were asked
   about, return "relevant_law": [] and include the limitation string:
   "Verified legal authority was not found in the available sources."
8. Only list a judgment in "judgments" if a retrieved block literally contains
   it (case name/court/citation in the source). If none does, return
   "judgments": [] — the system reports that no verified precedent was located.
   Never invent a case name, citation, court or date.
9. For "case_connections": state which verified legal provision may relate to
   the issue, which verified case fact is relevant, whether the available facts
   are SUFFICIENT to establish applicability (they usually are not), and what
   is missing. Use cautious wording: "potentially relevant", "the available
   material does not establish ...".
10. Never write page numbers or filenames in the text; use [E#] markers only.
11. Set "insufficient" to true ONLY when the packet contains nothing usable
    for the question. Then set "answer" to exactly: {REFUSAL} and every list to
    [].
12. Respond with ONLY a JSON object and no other text, using exactly this
schema:
{{"answer": string, "insufficient": boolean,
  "relevant_law": [{{"provision": string, "covers": string, "cites": ["E7"]}}],
  "case_connections": [{{"provision": string, "what_it_covers": string,
     "case_fact": string, "cites": ["E1"], "sufficiency": string,
     "missing": string}}],
  "judgments": [{{"case_name": string, "court": string, "date": string,
     "citation": string, "principle": string, "relationship": string,
     "cites": ["E9"]}}],
  "limitations": [string]}}"""


def classification_lines(classes: dict[str, dict], evidence) -> str:
    """Python-computed classification block fed to the model (never its own)."""
    lines = []
    for ev in evidence:
        c = classes.get(ev.id, {})
        if c.get("kind") == "case_evidence":
            lines.append(f"- {ev.id}: CASE MATERIAL — {ev.filename}, page {ev.page}")
        else:
            bits = [c.get("source_type") or "legal source", c.get("act") or ""]
            if c.get("section"):
                bits.append(f"Section {c['section']}")
            who = c.get("authority") or "authority not recorded"
            rep = " — REPEALED, do not present as in force" if c.get("repealed") else ""
            lines.append(f"- {ev.id}: LEGAL AUTHORITY — " + " · ".join(b for b in bits if b)
                         + f" — {who} (tier {c.get('tier')}){rep}")
    return "\n".join(lines)


def build_prompt(packet: str, classes: dict[str, dict], evidence,
                 legal_issue: str, research_type: str, question: str,
                 case_context: str = "") -> str:
    parts = [
        RESEARCH_SYSTEM_PROMPT,
        packet,
        "\nSOURCE CLASSIFICATION (computed by the system — authoritative):\n"
        + classification_lines(classes, evidence),
        f"\nLEGAL ISSUE (identified by the system): {legal_issue}",
        f"RESEARCH TYPE: {research_type}",
    ]
    if case_context:
        parts.append("USER-PROVIDED CASE CONTEXT (unverified notes, NOT evidence — "
                     "never cite it and never present it as sourced fact):\n"
                     + case_context)
    parts.append(f"QUESTION: {question}")
    parts.append("Respond with the JSON object only.")
    return "\n\n".join(parts)
