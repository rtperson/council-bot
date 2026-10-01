"""Background and document loading (sources.py). Offline; no model calls."""
import pytest

from council.sources import (
    ReadDocumentTool,
    SourceError,
    extract_text,
    load_context,
    load_documents,
    prompt_inputs,
)

from conftest import write_pdf


def test_every_readable_format_is_extracted(sample_docs):
    docs, _ = load_documents([str(sample_docs)])
    text = {d.name: d.text for d in docs}

    assert "Budget ceiling: 180000" in text["rfp.pdf"] and "[page 2]" in text["rfp.pdf"]
    assert "# Capacity plan" in text["capacity.docx"] and "Nov | 120" in text["capacity.docx"]
    assert "## Sheet: Cash" in text["budget.xlsx"] and "Oct | 42000" in text["budget.xlsx"]
    assert "fixed fee" in text["notes.md"]


def test_folders_skip_hidden_and_unreadable_files(sample_docs):
    docs, skipped = load_documents([str(sample_docs)])
    assert sorted(d.name for d in docs) == ["budget.xlsx", "capacity.docx", "notes.md", "rfp.pdf"]

    reasons = {s.path.name: s.reason for s in skipped}
    assert "unsupported" in reasons["logo.png"]
    assert "scanned" in reasons["scan.pdf"]
    assert "secret.txt" not in reasons and "~$capacity.docx" not in reasons


def test_files_named_twice_are_loaded_once(sample_docs):
    docs, _ = load_documents([str(sample_docs), str(sample_docs / "notes.md")])
    assert [d.name for d in docs].count("notes.md") == 1


def test_missing_path_fails_before_anything_runs(sandbox):
    with pytest.raises(SourceError, match="Not found: 'nope.pdf'"):
        load_documents(["nope.pdf"])
    with pytest.raises(SourceError, match="put the question before --docs"):
        load_documents(["Should we bid on it?"])


def test_legacy_office_formats_explain_the_fix(sandbox):
    (sandbox / "old.doc").write_bytes(b"x")
    with pytest.raises(SourceError, match="save it as .docx"):
        extract_text(sandbox / "old.doc")


def test_context_is_text_or_a_file(sandbox):
    assert load_context("We have two staff.") == ("We have two staff.", None)
    assert load_context("") == ("", None)

    (sandbox / "background.md").write_text("Runway: 4 months.", encoding="utf-8")
    assert load_context("background.md") == ("Runway: 4 months.", "background.md")

    write_pdf(sandbox / "brief.pdf", [["Board wants a decision by Friday"]])
    text, source = load_context("brief.pdf")
    assert "decision by Friday" in text and source == "brief.pdf"

    with pytest.raises(SourceError, match="Context file not found"):
        load_context("backgroud.md")  # typo'd path is an error, not silently the background


def test_short_documents_go_in_whole(sample_docs):
    docs, _ = load_documents([str(sample_docs)])
    inputs = prompt_inputs(docs)
    assert all(d.text in inputs["documents"] for d in docs)
    assert "included in full" in inputs["document_guidance"]


def test_long_documents_are_excerpted_to_the_budget(sample_docs, monkeypatch):
    (sample_docs / "long.txt").write_text("x" * 5000)
    docs, _ = load_documents([str(sample_docs)])
    monkeypatch.setenv("COUNCIL_DOC_BUDGET", "1000")

    inputs = prompt_inputs(docs)
    assert "notes.md" in inputs["documents"]                      # small docs still whole
    assert "x" * 5000 not in inputs["documents"]                 # the long one is cut
    assert "excerpt" in inputs["documents"] and "read_document" in inputs["document_guidance"]
    assert len(inputs["documents"]) < 2000


def test_no_documents_points_the_moderator_at_the_workspace():
    inputs = prompt_inputs([])
    assert "workspace" in inputs["document_guidance"]
    assert inputs["documents_index"] == "(none)"


def test_read_document_tool_reads_only_attached_files(sample_docs):
    docs, _ = load_documents([str(sample_docs)])
    tool = ReadDocumentTool.for_documents(docs)
    assert "Budget ceiling" in tool._run("rfp.pdf")
    assert "Budget ceiling" in tool._run("RFP.PDF")               # forgiving about case
    assert tool._run("../../etc/passwd").startswith("Error")


def test_read_document_tool_pages_through_long_text(sandbox):
    (sandbox / "long.txt").write_text("a" * 25_000)
    docs, _ = load_documents([str(sandbox / "long.txt")])
    tool = ReadDocumentTool.for_documents(docs)
    first = tool._run("long.txt")
    assert "call again with start=20000" in first
    assert "more" not in tool._run("long.txt", start=20_000)


def test_workspace_tool_stays_inside_the_workspace(sandbox, sample_docs):
    tool = ReadDocumentTool.for_folder(sample_docs)
    assert "Nov | 120" in tool._run("appendix/capacity.docx")
    (sandbox / "outside.txt").write_text("private")
    assert "outside the workspace" in tool._run("../outside.txt")
