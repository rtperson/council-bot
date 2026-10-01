"""
Offline tests: no API calls, no cost. Run with:  uv run pytest
"""
import json

import pytest

from council.crew import ADVISORS, AdvisorCrew, IntakeCrew, ReviewCrew, TriageCrew  # noqa: E402
from council.dry_run import fake_llm_run  # noqa: E402
from council.flow import CouncilFlow  # noqa: E402
from council.journal import JournalSearchTool, append_entry, read_entries, record_outcome  # noqa: E402
from council.sources import ReadDocumentTool, load_documents  # noqa: E402

QUESTION = "Should we hire a second consultant or keep outsourcing overflow work?"
MODES = ["quick", "standard", "deep"]


# ---------------------------------------------------------------- wiring

@pytest.mark.parametrize("mode", MODES)
def test_crews_build_in_every_mode(mode):
    advisors = None if mode == "deep" else ["red_team", "outsider", "executor"]
    advisor_crew = AdvisorCrew(mode=mode, advisors=advisors).crew()
    review_crew = ReviewCrew(mode=mode, advisors=advisors).crew()

    names = [t.name for t in advisor_crew.tasks]
    assert ("decision_science" in names) == (mode == "deep")
    assert ("forced_debate" in [t.name for t in review_crew.tasks]) == (mode != "quick")


@pytest.mark.parametrize("crew_cls", [TriageCrew, IntakeCrew, AdvisorCrew, ReviewCrew])
def test_question_is_the_only_required_input(crew_cls):
    """Every crew must run with just a question; the crews fill in everything else."""
    with fake_llm_run():
        result = crew_cls().crew().kickoff(inputs={"question": QUESTION})
    assert result.tasks_output


def test_missing_question_is_a_clear_error():
    with fake_llm_run(), pytest.raises(ValueError, match="question"):
        TriageCrew().crew().kickoff(inputs={})


def test_yaml_model_is_used_unless_overridden(monkeypatch):
    monkeypatch.delenv("COUNCIL_MODEL", raising=False)
    crew = TriageCrew()
    yaml_model = crew.agents_config["moderator"]["llm"]
    assert crew.moderator().llm.model in yaml_model

    monkeypatch.setenv("COUNCIL_MODEL", "openai/gpt-4o-mini")
    assert TriageCrew().moderator().llm.model == "gpt-4o-mini"


# ------------------------------------------------------------- full flow

@pytest.mark.parametrize("mode", MODES)
def test_dry_run_completes_and_writes_outputs(mode, sandbox):
    flow = CouncilFlow()
    with fake_llm_run(mode=mode):
        flow.kickoff(inputs={"question": QUESTION})

    s = flow.state
    assert s.status == "complete" and s.mode == mode
    run_dir = sandbox / "output" / s.run_id
    assert (run_dir / "report.html").read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    transcript = (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert "Anonymization key" in transcript and "Response A" in transcript

    [entry] = read_entries()
    assert entry["run_id"] == s.run_id and entry["question"] == QUESTION
    assert entry["mode"] == mode and entry["outcome"] == "pending"
    assert {"red_team", "executor"} <= set(entry["advisors"])
    if mode == "deep":
        assert entry["advisors"] == ADVISORS


def test_forced_mode_overrides_the_moderator():
    flow = CouncilFlow()
    with fake_llm_run(mode="deep"):          # the moderator "chooses" deep...
        flow.kickoff(inputs={"question": QUESTION, "mode": "quick"})   # ...the user forces quick
    assert flow.state.mode == "quick"
    assert read_entries()[0]["advisors"] == ["red_team", "outsider", "executor"]


def test_rejected_question_stops_after_triage(sandbox):
    flow = CouncilFlow()
    with fake_llm_run(triage_proceeds=False):
        flow.kickoff(inputs={"question": "What is the capital of Florida?"})
    assert flow.state.status == "rejected"
    assert read_entries() == []
    assert not (sandbox / "output").exists()


def test_internal_state_is_not_an_input():
    public = set(CouncilFlow().state.model_fields)
    assert {"question", "context", "docs", "mode"} <= public
    assert not public & {"intake_brief", "anonymized_responses", "timestamp", "advisors", "label_map"}


# ------------------------------------------------- background + documents

BACKGROUND = "We have four months of runway and two staff."


def test_background_reaches_every_step_that_needs_it():
    """Regression: --context used to stop at triage."""
    with fake_llm_run(mode="quick") as seen:
        CouncilFlow().kickoff(inputs={"question": QUESTION, "context": BACKGROUND})
    for task in ("triage", "frame_and_bias_audit", "red_team_analysis", "outsider_analysis",
                 "executor_analysis", "peer_review_red_team", "consensus_synthesis", "dissent_synthesis"):
        assert BACKGROUND in seen[task], f"{task} never saw the background"


def test_background_can_come_from_a_file(sandbox):
    (sandbox / "notes.md").write_text(BACKGROUND, encoding="utf-8")
    flow = CouncilFlow()
    with fake_llm_run(mode="quick") as seen:
        flow.kickoff(inputs={"question": QUESTION, "context": "notes.md"})
    assert BACKGROUND in seen["red_team_analysis"]
    assert "(from notes.md)" in seen["red_team_analysis"]
    assert read_entries()[0]["context_source"] == "notes.md"


def test_attached_documents_reach_moderator_and_advisors_verbatim(sample_docs, sandbox):
    flow = CouncilFlow()
    with fake_llm_run(mode="deep") as seen:
        flow.kickoff(inputs={"question": QUESTION, "docs": [str(sample_docs)]})

    for task in ("frame_and_bias_audit", "red_team_analysis", "first_principles_analysis",
                 "expansionist_analysis", "outsider_analysis", "executor_analysis"):
        assert "Budget ceiling: 180000" in seen[task] and "Nov | 120" in seen[task], task
    assert "rfp.pdf" in seen["triage"] and "rfp.pdf" in seen["html_report"]   # listed, not pasted
    assert "Budget ceiling" not in seen["triage"]

    entry = read_entries()[0]
    assert sorted(entry["documents"]) == ["budget.xlsx", "capacity.docx", "notes.md", "rfp.pdf"]
    transcript = (sandbox / "output" / flow.state.run_id / "transcript.md").read_text(encoding="utf-8")
    assert "**Documents:** " in transcript and "**Skipped:** " in transcript and "scan.pdf" in transcript


def _task_tools(crew, name):
    return {t.name for t in next(t for t in crew.tasks if t.name == name).tools}


def test_moderator_reads_attached_documents_or_else_the_workspace(sample_docs):
    docs, _ = load_documents([str(sample_docs)])
    attached = IntakeCrew(documents=docs).crew()
    shared = IntakeCrew().crew()

    assert _task_tools(attached, "frame_and_bias_audit") == {"read_document"}
    workspace = _task_tools(shared, "frame_and_bias_audit")
    assert "read_document" in workspace and any("list" in n.lower() for n in workspace)

    # Each intake step gets only the tools it needs.
    assert _task_tools(attached, "journal_lookup") == {"council_journal_search"}
    assert _task_tools(attached, "mode_selection") == set()
    assert not TriageCrew(documents=docs).moderator().tools


def test_bad_document_path_stops_before_any_model_call():
    from council.sources import SourceError
    with fake_llm_run() as seen, pytest.raises(SourceError):
        CouncilFlow().kickoff(inputs={"question": QUESTION, "docs": ["missing.pdf"]})
    assert seen == {}


def test_command_line_accepts_context_and_docs(sample_docs, sandbox, capsys):
    from council.main import kickoff
    (sandbox / "notes.md").write_text(BACKGROUND, encoding="utf-8")
    state = kickoff([QUESTION, "--context", "notes.md", "--docs", str(sample_docs),
                     "--dry-run", "--quiet", "--mode", "quick"])
    out = capsys.readouterr().out
    assert state.status == "complete"
    assert "Background: from notes.md" in out and "Documents: 4 attached" in out
    assert "Skipped scan.pdf" in out


def test_command_line_reports_a_bad_path_plainly(sandbox, capsys):
    from council.main import kickoff
    with pytest.raises(SystemExit, match="Not found: 'nope.pdf'"):
        kickoff([QUESTION, "--docs", "nope.pdf", "--dry-run", "--quiet"])


# --------------------------------------------------------------- journal

def test_journal_search_and_outcome():
    append_entry({"run_id": "r1", "question": "Should we bid on state agency AI training contracts?",
                  "recommendation": "Bid on one pilot", "outcome": "pending"})
    append_entry({"run_id": "r2", "question": "Which CRM should we buy?", "outcome": "pending"})

    hits = json.loads(JournalSearchTool()._run("Should we pursue state agency training contracts?"))
    assert [h["run_id"] for h in hits] == ["r1"]

    assert record_outcome("r1", "success", "won the pilot")
    assert read_entries()[0]["outcome"] == "success"
    assert not record_outcome("nope", "failure")


# ------------------------------------------------- python -m council

def test_python_dash_m_runs_without_the_exe_launchers(sandbox):
    """`python -m council` is the Smart App Control-proof way to run the commands."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import council

    env = dict(os.environ)  # make sure the child finds this copy of the package
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(Path(next(iter(council.__path__))).parent), env.get("PYTHONPATH")) if p
    )
    done = subprocess.run(
        [sys.executable, "-m", "council", QUESTION, "--dry-run", "--quiet", "--mode", "quick"],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "Done. Mode: quick" in done.stdout


def test_python_dash_m_dispatches_subcommands(capsys):
    from council.__main__ import main
    main(["outcome", "no-such-run", "success"])
    assert "No run with id no-such-run" in capsys.readouterr().out
