"""Shared test setup: no API calls, nothing written outside a temp folder."""
import os
from pathlib import Path

import pytest

# Agents build their LLM clients at construction time; a placeholder key is
# enough because the dry-run patch stops any request from being made.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ.setdefault("COUNCIL_VERBOSE", "0")


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Keep journals, outputs and the workspace out of the real project folders."""
    monkeypatch.setenv("COUNCIL_JOURNAL", str(tmp_path / "journal.jsonl"))
    monkeypatch.setenv("COUNCIL_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("COUNCIL_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.delenv("COUNCIL_DOC_BUDGET", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_pdf(path: Path, pages: list[list[str]]) -> Path:
    """A minimal, valid PDF. Each page is a list of text lines; [] makes an image-only page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        ops = "".join(f"BT /F1 12 Tf 72 {720 - 16 * i} Td ({t}) Tj ET\n" for i, t in enumerate(lines))
        objs.append(f"<< /Length {len(ops)} >>\nstream\n{ops}endstream")
        objs.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objs)} 0 R >>")
        kids.append(len(objs))
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"

    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def sample_docs(sandbox) -> Path:
    """A folder with one file of every readable kind, plus files that should be skipped."""
    import docx
    import openpyxl

    root = sandbox / "rfp"
    (root / "appendix").mkdir(parents=True)
    (root / ".hidden").mkdir()

    write_pdf(root / "rfp.pdf", [["DOT training RFP", "Budget ceiling: 180000"], ["Proposals due November 15"]])
    write_pdf(root / "scan.pdf", [[]])

    d = docx.Document()
    d.add_heading("Capacity plan", 1)
    d.add_paragraph("Two staff, booked through March.")
    table = d.add_table(rows=2, cols=2)
    for (r, c), text in {(0, 0): "Month", (0, 1): "Hours", (1, 0): "Nov", (1, 1): "120"}.items():
        table.cell(r, c).text = text
    d.save(root / "appendix" / "capacity.docx")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Cash"
    ws.append(["Month", "Cash"])
    ws.append(["Oct", 42000])
    wb.save(root / "budget.xlsx")

    (root / "notes.md").write_text("# Notes\nClient prefers a fixed fee.", encoding="utf-8")
    (root / "logo.png").write_bytes(b"\x89PNG\r\n")
    (root / ".hidden" / "secret.txt").write_text("should not be read")
    (root / "~$capacity.docx").write_bytes(b"office lock file")
    return root
