
"""Classify whether an uploaded PDF is a relevant legal case document.

This is a deterministic intake filter, not a legal opinion or proof that
a document is authentic. Ambiguous PDFs are rejected for confirmation.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentClassification:
    accepted: bool
    confidence: float
    category: str
    reason: str


# Official legal records and procedural documents.
OFFICIAL_PATTERNS = {
    "police report": r"\bfirst information report\b|\bfir\s*(?:no\.?|number)\b",
    "crime identifier": r"\bcrime\s*(?:no\.?|number)\s*[:#/-]?\s*[a-z0-9/-]+",
    "charge sheet": r"\bcharge[\s-]?sheet\b|\bchargesheet\b",
    "criminal proceedings": (
        r"\bcriminal appeal\b|\bcriminal revision\b|\bsessions case\b"
        r"|\bcriminal case\b|\bcase diary\b"
    ),
    "bail proceeding": r"\b(?:anticipatory\s+)?bail application\b|\bbail petition\b",
    "court decision": r"\bjudg(?:ment|ement)\b|\bcourt order\b|\border of the court\b",
    "police procedure": (
        r"\bremand report\b|\barrest memo\b|\bseizure memo\b"
        r"|\bpanchnama\b|\binquest report\b|\bpolice complaint\b"
    ),
    "legal case document": (
        r"\bin the court of\b|\bcase file\b|\bwitness statement\b"
        r"|\bstatement of the accused\b|\bfinal report\b"
    ),
}

CONTEXT_PATTERNS = {
    "case participants": (
        r"\baccused\b|\bcomplainant\b|\bprosecut(?:ion|or)\b"
        r"|\bappellant\b|\brespondent\b|\bpetitioner\b|\bwitness\b"
        r"|\binvestigating officer\b|\bvictim\b"
    ),
    "incident or investigation": (
        r"\bcrime\b|\boffence\b|\boffense\b|\bincident\b|\binvestigation\b"
        r"|\bseized\b|\brecovered\b|\bstatement recorded\b"
    ),
    "legal process": (
        r"\bpolice station\b|\bmagistrate\b|\bsessions court\b"
        r"|\bhigh court\b|\bsupreme court\b|\bcharges?\s+under\b"
    ),
    "legal provisions": (
        r"\bindian penal code\b|\bipc\b|\bcriminal procedure code\b"
        r"|\bcrpc\b|\bbharatiya nyaya sanhita\b|\bbns\b"
        r"|\bbharatiya nagarik suraksha sanhita\b|\bbnss\b"
        r"|\bindian evidence act\b"
    ),
}

REPORT_PATTERNS = {
    "software project report": r"\bsoftware project report\b",
    "academic project": r"\bcapstone project\b|\bacademic project report\b",
    "software design document": r"\bsoftware architecture\b|\bsoftware requirements specification\b",
    "project methodology": r"\bproject methodology\b|\bsprint planning\b",
}


def classify_legal_case_document(
    filename: str,
    text: str,
) -> DocumentClassification:
    """Accept relevant case records; reject unrelated PDFs before indexing."""

    body = (text or "").strip()
    if len(body) < 40:
        return DocumentClassification(
            False, 0.0, "unreadable",
            "Not enough readable text to identify this document. "
            "Please upload a text-based criminal/legal case PDF.",
        )

    # Use the extracted document text for classification. The filename is
    # supporting context, never sufficient evidence by itself.
    searchable = body[:30000].casefold()
    title = "\n".join(body.splitlines()[:12]).casefold()

    official = [
        label for label, pattern in OFFICIAL_PATTERNS.items()
        if re.search(pattern, searchable, re.IGNORECASE)
    ]
    context_groups = [
        label for label, pattern in CONTEXT_PATTERNS.items()
        if re.search(pattern, searchable, re.IGNORECASE)
    ]
    report_signals = [
        label for label, pattern in REPORT_PATTERNS.items()
        if re.search(pattern, searchable, re.IGNORECASE)
    ]

    has_case_file_label = bool(
        re.search(r"\b(?:criminal|crime|legal)\s+case\s+file\b", title)
        or re.search(r"\bcase file\b", title)
    )
    has_crime_context = bool(
        re.search(
            r"\bcrime\b|\baccused\b|\bcomplainant\b|\bwitness\b"
            r"|\bpolice\b|\bprosecution\b|\boffence\b|\boffense\b"
            r"|\binvestigation\b",
            searchable,
            re.IGNORECASE,
        )
    )

    # Avoid accepting an academic/software report merely because it mentions
    # legal terminology. Formal case-record markers can override this rule.
    strong_official = [
        item for item in official
        if item not in {"legal case document"}
    ]
    if len(report_signals) >= 2 and not strong_official:
        return DocumentClassification(
            False, 0.15, "unrelated_report",
            "This PDF appears to be a project or academic report, "
            "not a criminal/legal case document.",
        )

    # Formal records need contextual support. A clearly titled case file
    # must also contain crime-related content.
    if strong_official and context_groups:
        return DocumentClassification(
            True, 0.90, "legal_case",
            "Legal case indicators and supporting context were found.",
        )

    if has_case_file_label and has_crime_context:
        return DocumentClassification(
            True, 0.85, "legal_case",
            "The document appears to be a crime-related case file.",
        )

    # Allow other case documents that contain multiple independent legal
    # signals, even when they don't use a standard document title.
    if len(context_groups) >= 3 and official:
        return DocumentClassification(
            True, 0.75, "legal_case",
            "Multiple legal-case indicators were found.",
        )

    return DocumentClassification(
        False, 0.20, "unrecognized",
        "This PDF could not be identified as a relevant legal case "
        "document. Please upload a criminal/legal case file such as an "
        "FIR, police complaint, charge sheet, witness statement, bail "
        "application, or court order/judgment.",
    )
