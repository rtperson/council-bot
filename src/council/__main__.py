# src/council/__main__.py
"""
Run the council's commands through Python itself:

  python -m council "Should we ...?" [options]       full run (same as `council`)
  python -m council triage "Should we ...?"          step 1 only
  python -m council outcome <run_id> <result> [notes]
  python -m council plot

With uv:  uv run python -m council triage "Should we ...?"

Why this exists: uv also creates small launcher programs for these commands
(.venv\\Scripts\\council.exe, triage.exe, ...). They're unsigned and unique to
your machine, so Windows Smart App Control can block them. Running through
python.exe avoids the launchers entirely.
"""
import sys

from .main import kickoff, outcome, plot, triage

COMMANDS = {"run": kickoff, "triage": triage, "outcome": outcome, "plot": plot}


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    name = "run"
    if argv and argv[0] in COMMANDS:
        name, argv = argv[0], argv[1:]
    if name == "run":
        return kickoff(argv)
    sys.argv = [f"council {name}", *argv]     # the other commands read sys.argv
    return COMMANDS[name]()


if __name__ == "__main__":
    main()
