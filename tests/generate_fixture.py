"""Generate a tiny 2-page sample legal PDF without any extra dependencies.

The PDF is hand-assembled (raw objects + xref) so tests can always rebuild it.
"""
from pathlib import Path

PAGE_TEXTS = [
    (
        "SERVICE AGREEMENT\n"
        "This Service Agreement (the Agreement) is entered into between Meridian Logistics Pvt Ltd "
        "(Provider) and BlueHarbor Traders (Client), effective 14 February 2026.\n"
        "Clause 7 - Limitation of Liability. Neither party shall be liable for indirect or "
        "consequential damages. The aggregate liability of either party under this Agreement shall "
        "not exceed USD 50,000 per contract year.\n"
        "Clause 8 - Termination. Either party may terminate this Agreement upon ninety (90) days "
        "written notice to the other party. Termination does not relieve the Client of payment "
        "obligations accrued before the effective date of termination."
    ),
    (
        "CONFIDENTIALITY AND GOVERNING LAW\n"
        "Clause 12 - Confidentiality. Each party shall keep confidential all technical and "
        "commercial information disclosed by the other party for a period of five (5) years from "
        "the date of disclosure.\n"
        "Clause 13 - Governing Law. This Agreement shall be governed by and construed in "
        "accordance with the laws of Singapore.\n"
        "Clause 14 - Dispute Resolution. Any dispute arising out of or in connection with this "
        "Agreement shall be referred to and finally resolved by arbitration administered by the "
        "Singapore International Arbitration Centre (SIAC)."
    ),
]


def _esc(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _wrap(text: str, width: int = 92) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        words, current = raw.split(), ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > width:
                if current:
                    lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        lines.append("")  # paragraph gap
    while lines and not lines[-1]:
        lines.pop()
    return lines


def make_sample_pdf(path: Path) -> Path:
    """Write a 2-page PDF whose text is exactly PAGE_TEXTS, page by page."""
    # Layout: 1=Catalog, 2=Pages, 3=Font, then per page: content object, page object.
    # Page objects therefore live at 5, 7, 9, ... and content objects at 4, 6, 8, ...
    n_pages = len(PAGE_TEXTS)
    kids = " ".join(f"{5 + 2 * i} 0 R" for i in range(n_pages))

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]

    for text in PAGE_TEXTS:
        ops = [f"({_esc(line)}) Tj T*" for line in _wrap(text)]
        stream = ("BT /F1 11 Tf 72 724 Td 14 TL\n" + "\n".join(ops) + "\nET").encode("latin-1")
        content_obj = (
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
            + stream + b"\nendstream"
        )
        objects.append(content_obj)
        content_num = len(objects)
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_num} 0 R >>"
        ).encode("ascii")
        objects.append(page_obj)

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for num, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


if __name__ == "__main__":
    target = Path(__file__).parent / "fixtures" / "sample.pdf"
    make_sample_pdf(target)
    print(f"wrote {target}")
