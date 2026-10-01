#!/usr/bin/env python
"""
Command-line entry points (see [project.scripts] in pyproject.toml).

  uv run council "Should we ...?"                       full run
  uv run council "Should we ...?" --mode quick          cheapest real run (3 advisors, no debate)
  uv run council "Should we ...?" --dry-run             whole flow, no model calls, free
  uv run council "Should we ...?" --max-tokens 200000   stop the run past this many tokens
  uv run council "..." --context "We have 4 staff..."   add background as text
  uv run council "..." --context notes.md               ...or from a file
  uv run council "..." --docs rfp.pdf budget.xlsx specs/   attach files and folders
  uv run triage "Should we ...?"                        step 1 only: checks keys and model cheaply
  uv run outcome <run_id> success "won the pilot"       record how a decision turned out
  uv run plot                                           writes council_flow.html (flow diagram)

`crewai run` calls kickoff(); with no question given it asks for one.
Put the question first: everything after --docs is read as a file or folder.

If Windows blocks the .exe launchers (Smart App Control), run the same
commands through Python instead; see __main__.py:
  uv run python -m council "Should we ...?" --mode quick
  uv run python -m council triage "Should we ...?"
"""
import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

SAMPLE_QUESTION = (
    "Should a two-person AI consultancy focus next year on small businesses "
    "or on state-agency contracts?"
)


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="council",
        description="Put a question to the AI council.",
        epilog="Other commands: triage, outcome, plot. If Windows blocks council.exe, use "
               "`python -m council ...` (e.g. `uv run python -m council triage \"Should we ...?\"`).",
    )
    p.add_argument("question", nargs="*", help="The question (quotes optional).")
    p.add_argument("--context", default="",
                   help="Background for the council: the text itself, or a path to a file "
                        "(.md, .txt, .pdf, .docx, ...).")
    p.add_argument("--docs", nargs="+", default=[], metavar="PATH",
                   help="Files or folders for this question (PDF, Word, Excel, text). "
                        "Without this, the moderator checks the shared workspace folder.")
    p.add_argument("--mode", choices=["quick", "standard", "deep"],
                   help="Force a depth instead of letting the moderator choose.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run every step with canned outputs; no API calls, no cost.")
    p.add_argument("--max-tokens", type=int, metavar="N",
                   help="Stop the run once it has used N tokens (default: COUNCIL_MAX_TOKENS "
                        "or 500,000; 0 for no limit).")
    p.add_argument("--quiet", action="store_true", help="Hide agent-by-agent logging.")
    return p.parse_args(argv)


def _show_progress() -> None:
    """Print the council's own status lines (sources loaded, files skipped)."""
    logger = logging.getLogger("council")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _question(args: argparse.Namespace) -> str:
    q = " ".join(args.question).strip()
    if q:
        return q
    if sys.stdin.isatty():
        q = input(f"Question for the council (Enter for the sample):\n> ").strip()
    return q or SAMPLE_QUESTION


def _summarize(flow) -> None:
    s = flow.state
    print()
    if s.status == "rejected":
        t = s.triage
        print(f"Not sent to the council ({t.get('rejection_category')}): {t.get('rationale')}")
        if t.get("direct_answer"):
            print(f"\nAnswer: {t['direct_answer']}")
        for q in t.get("clarifying_questions") or []:
            print(f"  - {q}")
    else:
        print(f"Done. Mode: {s.mode}   Run: {s.run_id}")
        print(f"Tokens:     {s.tokens:,} (breakdown at the end of the transcript)")
        print(f"Report:     {s.report_path}")
        print(f"Transcript: {os.path.join(os.path.dirname(s.report_path), 'transcript.md')}")


def _preflight() -> None:
    """Fail in a second, with the fix, if a local model isn't ready."""
    from . import crew  # noqa: F401  (loads .env and applies OLLAMA_BASE_URL)
    from .local_models import LocalModelError, check_ollama

    try:
        checked = check_ollama()
    except LocalModelError as e:
        sys.exit(f"error: {e}")
    if checked:
        log = logging.getLogger("council")
        log.info("Ollama is up; using %s", ", ".join(checked))
        if not os.getenv("OLLAMA_CONTEXT_LENGTH"):
            log.info("  Reminder: Ollama silently drops the start of prompts longer than its context "
                     "window (4,096 tokens by default on most machines). Council prompts are longer; "
                     "set OLLAMA_CONTEXT_LENGTH for the Ollama app (see README).")


def kickoff(argv: list[str] | None = None):
    """Full pipeline. Also what `crewai run` calls."""
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.quiet:
        os.environ["COUNCIL_VERBOSE"] = "0"
    if args.max_tokens is not None:
        os.environ["COUNCIL_MAX_TOKENS"] = str(args.max_tokens)
    _show_progress()
    if not args.dry_run:
        _preflight()
    inputs = {"question": _question(args), "context": args.context, "docs": args.docs}
    if args.mode:
        inputs["mode"] = args.mode

    from .flow import CouncilFlow  # imported late so --quiet takes effect
    from .sources import SourceError
    from .usage import BudgetExceeded

    flow = CouncilFlow()
    try:
        if args.dry_run:
            from .dry_run import fake_llm_run
            os.environ.setdefault("COUNCIL_JOURNAL", "journal/dry-run.jsonl")
            os.environ.setdefault("COUNCIL_OUTPUT_DIR", "output/dry-run")
            with fake_llm_run(mode=args.mode or "standard"):
                flow.kickoff(inputs=inputs)
        else:
            flow.kickoff(inputs=inputs)
    except SourceError as e:
        sys.exit(f"error: {e}")
    except BudgetExceeded as e:
        sys.exit(f"\n{e}\nRaise the limit with --max-tokens, or try --mode quick.")
    _summarize(flow)
    return flow.state


def triage():
    """Step 1 only. A cheap way to confirm your API key and model work."""
    args = _parse(sys.argv[1:])
    _show_progress()
    _preflight()
    from .crew import TriageCrew
    from .sources import SourceError, load_context, load_documents

    try:
        context, _ = load_context(args.context)
        documents, _ = load_documents(args.docs)
    except SourceError as e:
        sys.exit(f"error: {e}")
    crew = TriageCrew(documents=documents).crew()
    result = crew.kickoff(inputs={"question": _question(args), "context": context})
    print(json.dumps(result.pydantic.model_dump(), indent=2))


def outcome():
    """Record how a past decision turned out:  outcome <run_id> <success|partial|failure|abandoned> [notes]"""
    if len(sys.argv) < 3:
        sys.exit("usage: outcome <run_id> <success|partial|failure|abandoned> [notes]")
    from .journal import record_outcome
    run_id, result, notes = sys.argv[1], sys.argv[2], " ".join(sys.argv[3:])
    print("Updated." if record_outcome(run_id, result, notes) else f"No run with id {run_id}.")


def plot():
    from .flow import CouncilFlow
    CouncilFlow().plot("council_flow")
    print("Wrote council_flow.html")


if __name__ == "__main__":
    kickoff()
