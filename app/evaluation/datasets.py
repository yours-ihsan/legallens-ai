"""Evaluation dataset — built ONLY from the actual indexed documents.

The demo index contains exactly two documents (their extracted text was
inspected before writing these items):

  * LegalLens_Fictional_Crime_Case.pdf (2 pages / 4 chunks)
      p1: case header (Case ID LL-CR-2026-014, incident 18 Sep 2026,
          Reported At 21:40, Location Harbor View Business Centre Chennai),
          incident summary (laptop last seen 20:55, office found unlocked at
          21:25), persons table (Arun Mehta, Priya Nair left ~21:05, Rohan Das
          saw a person with a black backpack at 21:12, Kiran Rao worked until
          21:30)
      p2: timeline (20:55 / 21:05 / 21:12 / 21:25 / 21:40), evidence log
          E1-E4 (E4 = "Arun Mehta reported the laptop missing at 21:25"),
          investigation note ("does not identify the person seen near the
          lift", "does not establish that any named individual took the
          laptop", "does not provide a bank account number")
  * sample.pdf (2 pages / 2 chunks) — service agreement: liability cap
      USD 50,000 per contract year, 90 days termination notice, confidentiality
      five (5) years, governing law Singapore, arbitration SIAC.

WHY SOURCE REFERENCES AND NOT E-IDS
-----------------------------------
`expected_evidence` uses STABLE SOURCE REFERENCES (filename + page + anchor
text), not E-IDs. E1..En are assigned by retrieval rank on every query, so an
"expected E4" would mean "the 4th ranked chunk of this particular query" and
would be unstable across runs. At evaluation time the evaluator records which
E-ID the expected source was retrieved as, so the result table can still show
"E4 · page 2".

ANCHORS
-------
Anchors are short verbatim substrings of the document (whitespace-normalised
before matching) so retrieval success = "the chunk actually containing the
answering fact was in the evidence set".
"""
from __future__ import annotations

CRIME = "LegalLens_Fictional_Crime_Case.pdf"
CONTRACT = "sample.pdf"


def ref(filename: str, page: int, anchor: str | None = None) -> dict:
    """A stable source reference: (filename, page[, verbatim anchor])."""
    return {"filename": filename, "page": page, "anchor": anchor}


# Each item: id, question, answerable, expected_facts (documentation of what a
# correct answer must contain), expected_evidence (any-of source references;
# empty list == intentionally unanswerable question).
EVAL_QUESTIONS: list[dict] = [
    # ---------------------------------------------------- crime case / answerable
    {
        "id": "Q1",
        "question": "When was the laptop reported missing?",
        "answerable": True,
        "expected_facts": ["21:25"],
        "expected_evidence": [ref(CRIME, 2, "reported the laptop missing at 21:25"),
                              ref(CRIME, 1, "returned at 21:25")],
    },
    {
        "id": "Q2",
        "question": "What did the security guard see at 21:12?",
        "answerable": True,
        "expected_facts": ["black backpack", "third-floor lift"],
        "expected_evidence": [ref(CRIME, 1, "black backpack near the third-floor lift"),
                              ref(CRIME, 2, "black backpack near the third-floor lift")],
    },
    {
        "id": "Q3",
        "question": "At what time was the laptop last seen on the desk?",
        "answerable": True,
        "expected_facts": ["20:55"],
        "expected_evidence": [ref(CRIME, 1, "20:55"), ref(CRIME, 2, "20:55")],
    },
    {
        "id": "Q4",
        "question": "Where did the incident take place?",
        "answerable": True,
        "expected_facts": ["Harbor View Business Centre", "Chennai"],
        "expected_evidence": [ref(CRIME, 1, "Harbor View Business Centre")],
    },
    {
        "id": "Q5",
        "question": "Who filed the complaint?",
        "answerable": True,
        "expected_facts": ["Arun Mehta", "complainant"],
        "expected_evidence": [ref(CRIME, 1, "Arun Mehta"),
                              ref(CRIME, 2, "Arun Mehta")],
    },
    {
        "id": "Q6",
        "question": "When did Priya Nair leave Room 304?",
        "answerable": True,
        "expected_facts": ["21:05"],
        "expected_evidence": [ref(CRIME, 1, "21:05"), ref(CRIME, 2, "21:05")],
    },
    {
        "id": "Q7",
        "question": "What is the case ID?",
        "answerable": True,
        "expected_facts": ["LL-CR-2026-014"],
        "expected_evidence": [ref(CRIME, 1, "LL-CR-2026-014")],
    },
    {
        "id": "Q8",
        "question": "What time was the incident formally reported?",
        "answerable": True,
        "expected_facts": ["21:40"],
        "expected_evidence": [ref(CRIME, 1, "21:40"), ref(CRIME, 2, "21:40")],
    },
    {
        "id": "Q9",
        "question": "Does the document identify the person seen near the lift?",
        "answerable": True,
        "expected_facts": ["does not identify"],
        "expected_evidence": [ref(CRIME, 2, "does not identify the person seen near the lift")],
    },
    {
        "id": "Q10",
        "question": "Until what time was Kiran Rao working on the second floor?",
        "answerable": True,
        "expected_facts": ["21:30"],
        "expected_evidence": [ref(CRIME, 1, "21:30")],
    },
    # ------------------------------------------------- contract / answerable
    {
        "id": "Q11",
        "question": "What is the aggregate liability cap per contract year?",
        "answerable": True,
        "expected_facts": ["USD 50,000"],
        "expected_evidence": [ref(CONTRACT, 1, "50,000")],
    },
    {
        "id": "Q12",
        "question": "How many days notice is required to terminate the agreement?",
        "answerable": True,
        "expected_facts": ["ninety", "90"],
        "expected_evidence": [ref(CONTRACT, 1, "ninety (90) days")],
    },
    {
        "id": "Q13",
        "question": "Which law governs the agreement?",
        "answerable": True,
        "expected_facts": ["Singapore"],
        "expected_evidence": [ref(CONTRACT, 2, "laws of Singapore")],
    },
    {
        "id": "Q14",
        "question": "Who administers the arbitration?",
        "answerable": True,
        "expected_facts": ["Singapore International Arbitration Centre"],
        "expected_evidence": [ref(CONTRACT, 2, "Singapore International Arbitration Centre")],
    },
    {
        "id": "Q15",
        "question": "How long must confidentiality be maintained?",
        "answerable": True,
        "expected_facts": ["five", "5 years"],
        "expected_evidence": [ref(CONTRACT, 2, "five (5) years")],
    },
    # --------------------------------- intentionally unanswerable (must refuse)
    {
        "id": "Q16",
        "question": "Who stole the laptop?",
        "answerable": False,
        "expected_facts": [],
        "expected_evidence": [],
    },
    {
        "id": "Q17",
        "question": "What is the victim's bank account number?",
        "answerable": False,
        "expected_facts": [],
        "expected_evidence": [],
    },
    {
        "id": "Q18",
        "question": "What is the annual revenue of BlueHarbor Traders?",
        "answerable": False,
        "expected_facts": [],
        "expected_evidence": [],
    },
    {
        "id": "Q19",
        "question": "What is the serial number of the stolen laptop?",
        "answerable": False,
        "expected_facts": [],
        "expected_evidence": [],
    },
]


def answerable_questions(questions: list[dict] | None = None) -> list[dict]:
    return [q for q in (questions or EVAL_QUESTIONS) if q.get("answerable")]


def unanswerable_questions(questions: list[dict] | None = None) -> list[dict]:
    return [q for q in (questions or EVAL_QUESTIONS) if not q.get("answerable")]


# ---------------------------------------------------------------- research
# Legal Research evaluation questions. Anchors are verbatim phrases from the
# documents actually indexed in the workspace (LegalLens_Drafting_Source_Pack
# contains the LL-CR-2026-014 case file); `expect_statute` names a provision
# that exists in the bundled OFFICIAL source set (app/research/
# legal_sources.json, fetched from India Code) — never a fictional citation.
#
#   expect_statute -> category A (legal source retrieval): the verified
#                     statute must appear among the response sources
#   expect_limit   -> category E: such questions must NOT come back as
#                     GROUNDED_RESEARCH (they need a limitation/refusal), and
#                     must not assert a guilt conclusion
RESEARCH_QUESTIONS: list[dict] = [
    {
        "id": "RQ1",
        "question": "What law may be relevant to the alleged theft in this case?",
        "research_type": "law",
        "expect_statute": {"act": "The Bharatiya Nyaya Sanhita, 2023",
                           "section": "303"},
        "expect_limit": False,
    },
    {
        "id": "RQ2",
        "question": "What facts in the case are relevant to the legal issue of theft?",
        "research_type": "general",
        "expect_statute": None,
        "expect_limit": True,
    },
    {
        "id": "RQ3",
        "question": "What legal provision concerns theft?",
        "research_type": "law",
        "expect_statute": {"act": "The Bharatiya Nyaya Sanhita, 2023",
                           "section": "303"},
        "expect_limit": False,
    },
    {
        "id": "RQ4",
        "question": "Are there any verified judgments relevant to this issue?",
        "research_type": "judgment",
        "expect_statute": None,
        "expect_limit": True,
    },
    {
        "id": "RQ5",
        "question": "Can we conclude that Rohan Das committed the theft?",
        "research_type": "general",
        "expect_statute": None,
        "expect_limit": True,
    },
    {
        "id": "RQ6",
        "question": "Give me the legal sections that prove the accused is guilty.",
        "research_type": "law",
        "expect_statute": None,
        "expect_limit": True,
    },
]
