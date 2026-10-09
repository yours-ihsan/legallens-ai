"""Page-preserving PDF text extraction.

Page numbers come from pypdf's physical page order (1-based) and are the ONLY
source of page numbers in the system.
"""
from pathlib import Path

from pypdf import PdfReader

from .models import PageText


def extract_pages(pdf_path: Path) -> list[PageText]:
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.replace("\x00", " ").strip()
        pages.append(PageText(page_number=number, text=text))

    if not pages:
        raise ValueError("PDF contains no pages.")
    if not any(p.text for p in pages):
        raise ValueError(
            "No extractable text found (scanned or image-only PDFs are not supported)."
        )
    return pages
