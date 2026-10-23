# src/council/flow.py
"""
Runs the 10-step pipeline:

  run_triage ──router──► rejected ─► finish_rejected
                   └───► accepted ─► run_intake ─► run_advisors ─► run_review ─► write_outputs

Inputs a person supplies:  question (required)
                           context (optional: background text, or a path to a file)
                           docs    (optional: files/folders attached to this question)
                           mode    (optional: quick | standard | deep, overrides step 4)
Everything else (run id, timestamp, brief, anonymized responses...) is created
here and kept in private state, so it never shows up as an input.

Code (not agents) owns: stopping on rejection, choosing which tasks exist,
anonymizing advisor outputs, writing files, the atomic journal append, and
the token budget (see usage.py).
"""
import logging
import os
import random
import re
import string
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, PrivateAttr

from .crew import ADVISORS, AdvisorCrew, IntakeCrew, ReviewCrew, TriageCrew
from .journal import append_entry
from .models import render_for_transcript
from .usage import BudgetExceeded, UsageMeter, activate
from .sources import (
    WORKSPACE_HINT,
    Document,
    Skipped,
    doc_budget,
    load_context,
    load_documents,
    prompt_inputs,
    workspace_dir,
)

log = logging.getLogger("council")

# Role words scrubbed from advisor outputs before blind review (belt and braces;
# the task prompts already tell advisors not to name themselves).
ROLE_WORDS = re.compile(
    r"\b(red[\s-]?team(ing)?|first[\s-]principles?|expansionist|outsider|executor|"
    r"tree[\s-]of[\s-]thoughts?|pre[\s-]?mortem)\b",
    re.IGNORECASE,
)

# Advisors used when `mode` is forced and step 4's advisor choice is skipped.
DEFAULT_ADVISORS = {
    "quick": ["red_team", "outsider", "executor"],
    "standard": ["red_team", "first_principles", "expansionist", "executor"],
    "deep": list(ADVISORS),
}


def output_dir() -> Path:
    return Path(os.getenv("COUNCIL_OUTPUT_DIR", "output"))


class CouncilState(BaseModel):
    # ---- the only fields a person sets ----
    question: str = ""
    context: str = ""                      # text, or a path to a file
    docs: list[str] = []                   # files and/or folders
    mode: Optional[Literal["quick", "standard", "deep"]] = None

    # ---- results, readable after the run ----
    status: str = "started"
    run_id: str = ""
    triage: dict = {}
    report_path: str = ""
    tokens: int = 0

    # ---- internal working state (private: never an input) ----
    _timestamp: str = PrivateAttr("")
    _context_text: str = PrivateAttr("")
    _context_source: Optional[str] = PrivateAttr(None)
    _documents: list[Document] = PrivateAttr(default_factory=list)
    _skipped: list[Skipped] = PrivateAttr(default_factory=list)
    _advisors: list[str] = PrivateAttr(default_factory=list)
    _brief: str = PrivateAttr("")
    _review_brief: str = PrivateAttr("")
    _responses: str = PrivateAttr("")
    _label_map: dict[str, str] = PrivateAttr(default_factory=dict)
    _transcript: list[dict] = PrivateAttr(default_factory=list)
    _meter: Any = PrivateAttr(None)


def _outputs_by_name(result) -> dict:
    return {t.name: t for t in result.tasks_output}


class CouncilFlow(Flow[CouncilState]):

    def _inputs(self) -> dict:
        return {"question": self.state.question, "context": self.state._context_text}

    def _crew_args(self) -> dict:
        s = self.state
        return {"mode": s.mode or "standard", "advisors": s._advisors or None, "documents": s._documents}

    def _run(self, step: str, crew):
        """Kick off one crew under the token meter; stop the run cleanly if the budget runs out."""
        s, meter = self.state, self.state._meter
        if meter.over_budget:
            self._stop_for_budget(step, None)
        meter.step = step
        try:
            result = crew.kickoff(inputs=self._inputs())
        except Exception as e:
            meter.settle()
            if meter.blocked:
                self._stop_for_budget(step, e)
            raise
        finally:
            meter.settle()
            s.tokens = meter.total
        used = meter.by_step[step]
        log.info("  %s: %s tokens in %d model calls", step, f"{used['tokens']:,}", used["calls"])
        self._log(step, result)
        return result

    def _stop_for_budget(self, step: str, cause: Exception | None):
        s, meter = self.state, self.state._meter
        s.status = "stopped"
        run_dir = output_dir() / s.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        note = (f"Stopped during {step}: the run reached its token limit of {meter.limit:,} "
                f"(COUNCIL_MAX_TOKENS). Everything completed before that point is below.")
        self._write_transcript(run_dir, note)
        activate(None)
        raise BudgetExceeded(f"{note}\nTranscript so far: {run_dir / 'transcript.md'}") from cause

    def _write_transcript(self, run_dir: Path, note: str | None = None) -> None:
        """Verbatim record of the run, with the anonymization key and token usage at the end."""
        s = self.state
        lines = [f"# Council transcript: {s.run_id}", ""]
        if note:
            lines += [f"> **{note}**", ""]
        lines += [f"**Question:** {s.question}  ", f"**Mode:** {s.mode or 'not chosen yet'}  "]
        if s._documents:
            lines.append("**Documents:** " + ", ".join(f"{d.name} ({d.chars:,} chars)" for d in s._documents) + "  ")
        else:
            lines.append(f"**Documents:** none attached (moderator checked {workspace_dir()})  ")
        if s._skipped:
            lines.append("**Skipped:** " + "; ".join(f"{k.path.name} ({k.reason})" for k in s._skipped) + "  ")
        source = f" (from {s._context_source})" if s._context_source else ""
        lines += ["", f"## Background from the user{source}", "", s._context_text or "(none given)", ""]
        for entry in s._transcript:
            lines += [f"## {entry['step']} · {entry['task']} ({entry['agent']})", "", entry["output"], ""]
        if s._label_map:
            lines += ["## Anonymization key", ""] + [f"- {k}: {v}" for k, v in s._label_map.items()] + [""]
        lines += ["## Token usage", ""] + s._meter.table()
        (run_dir / "transcript.md").write_text("\n".join(lines), encoding="utf-8")

    def _log(self, step: str, result) -> None:
        for t in result.tasks_output:
            agent = (t.agent or "").strip()
            output = render_for_transcript(t.pydantic) if t.pydantic is not None else t.raw
            self.state._transcript.append({"step": step, "task": t.name, "agent": agent, "output": output})

    def _load_sources(self) -> None:
        """Read background and documents before any model is called; a bad path fails here."""
        s = self.state
        s._context_text, s._context_source = load_context(s.context)
        s._documents, s._skipped = load_documents(s.docs)

        if s._context_text:
            where = f"from {s._context_source}" if s._context_source else "inline"
            log.info("Background: %s (%s characters)", where, f"{len(s._context_text):,}")
        else:
            log.info("Background: none given")
        if s._documents:
            total = sum(d.chars for d in s._documents)
            log.info("Documents: %d attached (%s characters)", len(s._documents), f"{total:,}")
            for d in s._documents:
                log.info("  - %s (%s characters)", d.name, f"{d.chars:,}")
            if total > doc_budget():
                log.info("  Long documents will be excerpted in the advisors' brief (budget %s "
                         "characters, set with COUNCIL_DOC_BUDGET); the moderator reads them in full.",
                         f"{doc_budget():,}")
        else:
            log.info("Documents: none attached; %s (%s).", WORKSPACE_HINT, workspace_dir())
        for k in s._skipped:
            log.warning("  Skipped %s: %s", k.path.name, k.reason)

    def _compose_brief(self, framing: str, prior_runs: str, mode: str, why: str) -> str:
        s = self.state
        source = f" (from {s._context_source})" if s._context_source else ""
        parts = [
            f"## Background from the user{source}, verbatim\n{s._context_text or '(none given)'}",
            f"## Moderator's brief\n{framing}",
            f"## Prior council runs\n{prior_runs}",
            f"## Deliberation mode: {mode.upper()}\n{why}",
        ]
        if s._documents:
            docs = prompt_inputs(s._documents)
            parts.append(
                "## Documents attached to this question\n"
                "Cite a document by name when you rely on it.\n\n"
                f"{docs['documents_index']}\n\n{docs['documents']}"
            )
        return "\n\n".join(parts)

    # ---- Step 1
    @start()
    def run_triage(self):
        if not self.state.question.strip():
            raise ValueError("Please provide a question.")
        self._load_sources()
        self.state.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.state._timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.state._meter = UsageMeter()
        activate(self.state._meter)
        limit = self.state._meter.limit
        log.info("Token limit: %s", f"{limit:,} (COUNCIL_MAX_TOKENS)" if limit else "none")
        result = self._run("1 Triage", TriageCrew(**self._crew_args()).crew())
        self.state.triage = result.pydantic.model_dump()

    @router(run_triage)
    def gate(self):
        return "accepted" if self.state.triage.get("proceed") else "rejected"

    @listen("rejected")
    def finish_rejected(self):
        self.state.status = "rejected"
        activate(None)
        return self.state.triage          # direct answer or clarifying questions for the UI

    # ---- Steps 2-4
    @listen("accepted")
    def run_intake(self):
        result = self._run("2-4 Intake", IntakeCrew(**self._crew_args()).crew())
        out = _outputs_by_name(result)
        decision = out["mode_selection"].pydantic

        if self.state.mode:                              # a person forced the mode
            mode, advisors = self.state.mode, DEFAULT_ADVISORS[self.state.mode]
            why = f"Mode set to {mode.upper()} by the user (the moderator suggested {decision.mode.upper()})."
        else:
            mode, advisors, why = decision.mode, list(decision.advisors), decision.rationale
            for required in ("red_team", "executor"):    # enforce the rule in code, not just the prompt
                if required not in advisors:
                    advisors.append(required)
            if mode == "deep":
                advisors = list(ADVISORS)

        self.state.mode = mode
        self.state._advisors = [a for a in ADVISORS if a in advisors]
        self.state._brief = self._compose_brief(
            out["frame_and_bias_audit"].raw, out["journal_lookup"].raw, mode, why
        )

    # ---- Steps 5-6
    @listen(run_intake)
    def run_advisors(self):
        s = self.state
        result = self._run("5-6 Advisors", AdvisorCrew(**self._crew_args(), brief=s._brief).crew())
        out = _outputs_by_name(result)

        # Anonymize: shuffle, relabel A-E, scrub role words.
        order = s._advisors[:]
        random.shuffle(order)
        blocks = []
        for letter, name in zip(string.ascii_uppercase, order):
            label = f"Response {letter}"
            s._label_map[label] = name
            blocks.append(f"### {label}\n{ROLE_WORDS.sub('[advisor]', out[f'{name}_analysis'].raw)}")
        s._responses = "\n\n".join(blocks)

        s._review_brief = s._brief
        if "decision_science" in out:
            s._review_brief += f"\n\n## Decision matrix\n{out['decision_science'].raw}"

    # ---- Steps 7-10 (agent work)
    @listen(run_advisors)
    def run_review(self):
        s = self.state
        crew = ReviewCrew(**self._crew_args(), brief=s._review_brief, responses=s._responses).crew()
        result = self._run("7-10 Review", crew)
        return _outputs_by_name(result)

    # ---- Steps 9-10 (code work)
    @listen(run_review)
    def write_outputs(self, out):
        s = self.state
        run_dir = output_dir() / s.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Step 9a: HTML report (strip a markdown fence if the model added one)
        html = re.sub(r"^```(?:html)?\s*|\s*```$", "", out["html_report"].raw.strip())
        (run_dir / "report.html").write_text(html, encoding="utf-8")

        # Step 9b: transcript. Prose is copied as written. Structured outputs
        # are rendered as markdown from the same fields, with no extra model call.
        self._write_transcript(run_dir)

        # Step 10: journal. The LLM supplies the content; code supplies identity and outcome fields.
        entry = {
            "run_id": s.run_id,
            "timestamp": s._timestamp,
            "question": s.question,
            "mode": s.mode,
            "advisors": s._advisors,
            "context_source": s._context_source or ("inline" if s._context_text else None),
            "documents": [d.name for d in s._documents],
            "tokens": s._meter.total,
            **out["journal_record"].pydantic.model_dump(),
            "outcome": "pending",
            "outcome_notes": "",
        }
        append_entry(entry)

        s.status, s.report_path = "complete", str(run_dir / "report.html")
        activate(None)
        return {"run_id": s.run_id, "report": s.report_path, "journal_entry": entry}
