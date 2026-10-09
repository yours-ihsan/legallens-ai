"""Drafting requirements configuration — what each draft type needs from the source.

DESIGN (safety-critical):
* Field AVAILABILITY is decided by Python with transparent regex patterns run
  over the retrieved evidence — never by the model. "Available" means the
  document literally contains text matching that field's patterns, and the
  supporting chunk (evidence ID, filename, page) is recorded alongside it.
* Anything not detected is reported as MISSING and must appear in the draft as
  a clearly labeled placeholder — it is never guessed.
* Patterns are deliberately STRICT (they look for specific, name-bearing
  contexts), erring toward "missing". A false "missing" is safe; a false
  "available" could smuggle an invented value into a legal document.
* The indexed documents were inspected before writing these patterns:
  they contain a Case ID, incident date, timeline, witness statements and a
  complainant, but NO court, FIR number, police station or accused name —
  so those must be reported missing for a bail application, which is exactly
  the honest outcome.
* Label words are NOT values: a source that merely LISTS field names (its own
  "Missing: Court name, jurisdiction, ... exact relief sought ..." note, or a
  section heading) must never mark those fields available. Patterns therefore
  require value syntax — "<field>: <Value>", "<field> is <Value>", or a
  number-bearing docket form — which is why the petition fields below are all
  strict. Prefer a false "missing" over a false "available".
"""

# Structure templates are jurisdiction-neutral: headings only, no legal
# language that the evidence does not support.
DRAFT_TYPES: dict[str, dict] = {
    "bail_application": {
        "label": "Bail Application",
        "query": (
            "case details incident date reported complainant accused applicant "
            "police station court fir number facts circumstances timeline witness"
        ),
        "structure": """IN THE [COURT NAME — NOT PROVIDED]
BAIL APPLICATION
Case/Reference No.: [CASE NUMBER — NOT PROVIDED]
Applicant: [APPLICANT NAME — NOT PROVIDED]
Versus
Respondent: [RESPONDENT NAME — NOT PROVIDED]

MOST RESPECTFULLY SUBMITTED:

1. Background
2. Facts supported by the source document
3. Relevant circumstances
4. Grounds — only where supported by available information
5. Prayer / Relief""",
        "fields": [
            {"key": "court", "label": "Court name",
             "patterns": [r"\b(?:in the|before the|of the|at the)\s+(?:hon['\u2019]ble\s+)?[\w-]+\s+court\b",
                          r"\b(?:sessions|district|high|civil|criminal|judicial|metropolitan|additional)\s+(?:\w+\s+){0,2}court\b",
                          r"\bmagistrate\b"]},
            {"key": "fir_number", "label": "FIR number",
             "patterns": [r"\bF\.?\s?I\.?\s?R\.?\s*(?:no\.?|number|#)?\s*[\w/-]*\d",
                          r"\bF\.?\s?I\.?\s?R\.?\s*[:\-\u2013]\s*[A-Z0-9]"]},
            {"key": "police_station", "label": "Police station",
             "patterns": [r"\bpolice\s+station\s*[:\-\u2013]\s*[A-Z0-9]",
                          r"\bstation house officer\b",
                          r"\bP\.?\s?S\.?\s*(?:no\.?)?\s*\d[\w-]*"]},
            {"key": "accused_name", "label": "Accused name",
             "patterns": [r"\baccused\s*(?:name)?\s*[:\-\u2013]\s*[A-Z][\w.'\u2013-]+",
                          r"\baccused\s+(?:is|was|named)\s+[A-Z][\w.'\u2013-]+"]},
            {"key": "applicant_name", "label": "Applicant name",
             "patterns": [r"\bapplicant\s*(?:name)?\s*[:\-\u2013]\s*[A-Z][\w.'\u2013-]+",
                          r"\bapplicant\s+(?:is|was)\s+[A-Z][\w.'\u2013-]+"]},
            {"key": "case_number", "label": "Case / reference number",
             "patterns": [r"\bcase\s*(?:id|no|number)\s*[:\-\u2013]?\s*[A-Z0-9][\w/-]*\d",
                          r"\bLL-CR-\d[\w-]*"]},
            {"key": "incident_date", "label": "Date of incident",
             "patterns": [r"\bincident date\b",
                          r"\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b"]},
            {"key": "case_details", "label": "Relevant case details",
             "patterns": [r"\bincident\b", r"\btimeline\b", r"\breported at\b", r"\bevidence log\b"]},
            {"key": "complainant", "label": "Complainant",
             "patterns": [r"\bcomplainant\b"]},
            {"key": "grounds", "label": "Grounds for bail",
             "patterns": [r"\bgrounds?\s+for\s+bail\b", r"\bantecedents?\b", r"\bmedical\s+grounds?\b",
                          r"\bno\s+criminal\s+history\b"]},
        ],
    },

    "petition": {
        "label": "Petition",
        "query": (
            "parties court case number relevant facts dispute relief requested "
            "petitioner respondent agreement obligations"
        ),
        "structure": """IN THE [COURT NAME — NOT PROVIDED]
PETITION
Case/Reference No.: [CASE NUMBER — NOT PROVIDED]

In the matter of:
Petitioner: [PETITIONER NAME — NOT PROVIDED]
Versus
Respondent: [RESPONDENT NAME — NOT PROVIDED]

MOST RESPECTFULLY SUBMITTED:

1. Parties
2. Relevant facts supported by the source document
3. Grounds — only where supported by available information
4. Relief requested""",
        # Petition fields are checked for VALUES, not label words. Sources that
        # merely list what is missing ("Missing: Court name, jurisdiction,
        # petitioner/respondent ..., exact relief sought ...") must NOT make a
        # field available: every pattern below requires value syntax
        # ("<field>: <Value>" / "<field> is <Value>" / a number-bearing docket
        # form), so a bare mention of the label word matches nothing. The
        # internal case-file ID ("Case ID LL-CR-...") is not a court case
        # number, so it does not satisfy "Case/reference number" either.
        "fields": [
            {"key": "court", "label": "Court name",
             "patterns": [r"\b(?:in the|before the|of the|at the)\s+(?:hon['\u2019]ble\s+)?[\w-]+\s+court\b",
                          r"\b(?:sessions|district|high|civil|criminal|judicial|metropolitan)\s+(?:\w+\s+){0,2}court\b",
                          r"\bmagistrate\b",
                          r"\bcourt\s+name\s*[:\-\u2013]\s*[A-Z]"]},
            {"key": "jurisdiction", "label": "Court jurisdiction",
             "patterns": [r"\bterritorial\s+jurisdiction\b",
                          r"\bcourt\s+of\s+jurisdiction\b",
                          r"\bjurisdiction\s+(?:of|lies\s+with|vested\s+in|shall\s+lie)\b",
                          r"\bjurisdiction\s*[:\-\u2013]\s*[A-Z]"]},
            {"key": "case_number", "label": "Case/reference number",
             "patterns": [r"\b(?:case|suit|petition|appeal|writ|original application|civil suit)\s*(?:no\.?|number)\s*[:.\-\u2013]?\s*[A-Z0-9][\w/-]*\d",
                          r"\bW\.?\s?P\.?\s*(?:No\.?|number)\s*[:.\-\u2013]?\s*\d",
                          r"\b(?:C\.?P\.?|R\.?\s?S\.?\s?A\.?|O\.?\s?S\.?)\s*(?:No\.?|number)\s*[:.\-\u2013]?\s*\d",
                          r"\bRM\.?\s*/\s?AP\b",
                          r"\(\d{4}\)\s*\d+\s*(?:SCC|SC|ALL)\b"]},
            {"key": "petitioner", "label": "Petitioner details",
             "patterns": [r"\bpetitioner\s*(?:name|address|details)?\s*[:\-\u2013]\s*[A-Z][\w.'\u2013-]+",
                          r"\bpetitioner\s+(?:is|was|has\s+filed|filed)\b"]},
            {"key": "respondent", "label": "Respondent details",
             "patterns": [r"\brespondent\s*(?:name|address|details)?\s*[:\-\u2013]\s*[A-Z][\w.'\u2013-]+",
                          r"\brespondent\s+(?:is|was|has)\b"]},
            {"key": "facts", "label": "Relevant facts",
             "patterns": [r"\bincident\b", r"\btimeline\b", r"\bagreement\b", r"\bobligations?\b"]},
            {"key": "grounds", "label": "Legal grounds",
             "patterns": [r"\bgrounds?\s+of\s+(?:attack|appeal|challenge)\b",
                          r"\blegal\s+grounds?\s*[:\-\u2013]",
                          r"\bon\s+the\s+grounds?\s+that\b",
                          r"\bgrounds?\s+are\s+[:\-\u2013]"]},
            {"key": "statutory", "label": "Statutory provisions",
             "patterns": [r"\bsection\s+\d+",
                          r"\bact,?\s+\d{4}\b",
                          r"\bArticle\s+\d+\s+of\b",
                          r"\b(?:IPC|CrPC|BNSS|BNS)\b",
                          r"\bstatute\s+(?:no\.?|number)\s*\d"]},
            {"key": "relief", "label": "Relief requested",
             "patterns": [r"\brelief\s+(?:sought|requested)\s*[:\-\u2013]\s*[A-Z]",
                          r"\bhereby\s+(?:prayed|requested)\b",
                          r"\bprayer\s*[:\-\u2013]\s*[A-Z]"]},
            {"key": "filing_date", "label": "Filing date",
             "patterns": [r"\bdate\s+of\s+filing\s*[:.\-\u2013]?\s*\d{1,2}\b",
                          r"\bfiling\s+date\s*[:\-\u2013]\s*\d{1,2}\b",
                          r"\bfiled\s+(?:on|this\s+day\s+of)\s+\d{1,2}\b"]},
            {"key": "advocate", "label": "Advocate details",
             "patterns": [r"\badvocate\s*(?:name)?\s*[:\-\u2013]\s*[A-Z]",
                          r"\blearned\s+counsel\b",
                          r"\badvocate\s+(?:is|was)\s+[A-Z]",
                          r"\bbar\s+council\b"]},
        ],
    },

    "legal_notice": {
        "label": "Legal Notice",
        "query": (
            "sender recipient date issue dispute breach notice period requested "
            "action obligations parties"
        ),
        "structure": """LEGAL NOTICE
Date: [DATE — NOT PROVIDED]

From: [SENDER — NOT PROVIDED]
To: [RECIPIENT — NOT PROVIDED]

Subject: [ISSUE / DISPUTE — NOT PROVIDED]

1. Facts supported by the source document
2. Relevant provisions — only if present in the source
3. Action requested: [REQUESTED ACTION — NOT PROVIDED]
4. Time limit, if stated in the source; otherwise [TIME LIMIT — NOT PROVIDED]

This notice is issued without prejudice to the contents of the source document.""",
        "fields": [
            {"key": "date", "label": "Date of notice",
             "patterns": [r"\bdated\s+(?:this|the)?\s*\d{1,2}",
                          r"\bdate of notice\b",
                          r"\bthis (?:notice|day)\b.{0,60}\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)"]},
            {"key": "sender", "label": "Sender",
             "patterns": [r"\bnotice\s+(?:is\s+)?(?:hereby\s+)?(?:given|issued|served)\s+by\b",
                          r"\bhereby serves?\s+(?:this\s+)?notice\b",
                          r"(?:^|\n)\s*from\s*:\s*[A-Z]"]},
            {"key": "recipient", "label": "Recipient",
             "patterns": [r"(?:^|\n)\s*to\s*:\s*[A-Z]",
                          r"\baddressed to\b",
                          r"\bnotice\s+(?:is\s+)?(?:hereby\s+)?(?:given|issued|served)\s+to\b"]},
            {"key": "issue", "label": "Issue / dispute",
             "patterns": [r"\bdispute\b", r"\bbreach\b", r"\bgrievance\b", r"\bnon-performance\b"]},
            {"key": "requested_action", "label": "Requested action",
             "patterns": [r"\bhereby called upon\b",
                          r"\bis hereby required to\b",
                          r"\bdemands?\s+(?:payment|performance)\b"]},
            {"key": "facts", "label": "Relevant facts",
             "patterns": [r"\bincident\b", r"\btimeline\b", r"\bagreement\b", r"\bobligations?\b"]},
        ],
    },

    "affidavit": {
        "label": "Affidavit",
        "query": (
            "deponent affiant statement sworn facts complainant incident "
            "date place circumstances"
        ),
        "structure": """AFFIDAVIT

I, [DEPONENT — NOT PROVIDED], do hereby solemnly affirm and state:

1. Facts supported by the source document
2. Statements available for affirmation, each traced to evidence
3. Verification

Deponent: [DEPONENT — NOT PROVIDED]
Place: [PLACE — NOT PROVIDED]
Date: [DATE — NOT PROVIDED]""",
        "fields": [
            {"key": "deponent", "label": "Deponent",
             "patterns": [r"\bdeponent\s*(?:name)?\s*[:\-\u2013]\s*[A-Z][\w.'\u2013-]+",
                          r"\bdeponent\s+(?:is|was)\s+[A-Z]",
                          r"\baffiant\s*[:\-\u2013]\s*[A-Z]"]},
            {"key": "statements", "label": "Statements to affirm",
             "patterns": [r"\bcomplainant statement\b", r"\bstatement\b", r"\bwitness statement\b"]},
            {"key": "facts", "label": "Relevant facts",
             "patterns": [r"\bincident\b", r"\btimeline\b", r"\bevidence log\b"]},
            {"key": "sworn_place_date", "label": "Date/place of sworn statement",
             "patterns": [r"\bsubscribe(?:d)?\b.{0,60}\bbefore\b", r"\bnotary public\b",
                          r"\bplace of (?:swearing|execution)\b", r"\bsolemnly affirm(?:ed)?\b.{0,60}\bdate\b"]},
        ],
    },
}


def type_label(draft_type: str) -> str:
    return DRAFT_TYPES[draft_type]["label"]
