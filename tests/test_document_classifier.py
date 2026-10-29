
from app.document_classifier import classify_legal_case_document


def test_accepts_fir_with_case_context():
    text = """
    FIRST INFORMATION REPORT
    Police Station: Central Police Station
    Crime No: 123/2026
    The complainant reported the incident.
    The investigating officer recorded the witness statement.
    """
    result = classify_legal_case_document("case.pdf", text)
    assert result.accepted is True


def test_accepts_fictional_crime_case_file():
    text = """
    Fictional Crime Case File
    This fictional file describes a theft incident.
    A witness statement and investigation timeline follow.
    The accused and complainant are fictional.
    """
    result = classify_legal_case_document(
        "LegalLens_Fictional_Crime_Case.pdf", text
    )
    assert result.accepted is True


def test_rejects_software_project_report():
    text = """
    Software Project Report
    This capstone project describes software architecture.
    The project methodology and sprint planning are documented.
    """
    result = classify_legal_case_document("project.pdf", text)
    assert result.accepted is False


def test_rejects_unrelated_document():
    text = """
    Travel Plan
    This document contains a schedule for hotels and sightseeing.
    It describes transport bookings and estimated travel costs.
    """
    result = classify_legal_case_document("travel.pdf", text)
    assert result.accepted is False


def test_filename_alone_does_not_make_document_legal():
    text = """
    This report documents a software project.
    It contains implementation steps and delivery milestones.
    """
    result = classify_legal_case_document(
        "Criminal_Case_File.pdf", text
    )
    assert result.accepted is False
