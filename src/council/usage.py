# src/council/usage.py
"""
Token accounting and a hard budget for a council run.

CrewAI calls two hooks around every model call, including retries, output
conversion and delegated questions. The meter uses them to:

  - count tokens and calls per pipeline step and per agent, so a run shows
    exactly where its tokens went;
  - refuse any further model call once the run has used COUNCIL_MAX_TOKENS.
    A refused call costs nothing; the step fails and the flow stops, keeping
    everything done so far.

Counts come from each model client's own running total, settled before every
call and after every crew, so calls whose after-hook CrewAI skips (tool calls)
are still counted, and the budget check always sees an up-to-date total.
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict

from crewai.hooks import register_after_llm_call_hook, register_before_llm_call_hook

DEFAULT_MAX_TOKENS = 500_000


def max_tokens() -> int:
    """Token ceiling for one run; 0 means no limit."""
    return int(os.getenv("COUNCIL_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))


class BudgetExceeded(RuntimeError):
    """The run hit its token ceiling and was stopped."""


class UsageMeter:
    def __init__(self, limit: int | None = None):
        self.limit = max_tokens() if limit is None else limit
        self.step = "setup"
        self.total = 0
        self.calls = 0
        self.blocked = False
        self.by_step: dict[str, dict[str, int]] = defaultdict(lambda: {"tokens": 0, "calls": 0})
        self.by_agent: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"tokens": 0, "calls": 0})
        # id(llm) -> (llm, its running total when last settled, (step, agent) that used it last)
        self._seen: dict[int, tuple[object, int, tuple[str, str]]] = {}
        self._lock = threading.Lock()                    # advisors run in parallel threads

    def __deepcopy__(self, memo):
        # CrewAI snapshots the flow state (which holds the meter) by deep copy.
        # There is one meter per run, and it holds live model clients that
        # can't be copied, so a copy is the same meter.
        return self

    @property
    def over_budget(self) -> bool:
        return bool(self.limit) and self.total >= self.limit

    # ---- hooks
    def _sync(self, llm) -> None:
        """Add whatever this model client has used since we last looked, to whoever used it."""
        if not hasattr(llm, "get_token_usage_summary"):
            return
        running = llm.get_token_usage_summary().total_tokens
        with self._lock:
            _, seen, owner = self._seen.get(id(llm), (llm, 0, (self.step, "(output conversion)")))
            delta = max(0, running - seen)
            self._seen[id(llm)] = (llm, running, owner)   # holding llm keeps its id from being reused
            if delta:
                self.total += delta
                self.by_step[owner[0]]["tokens"] += delta
                self.by_agent[owner]["tokens"] += delta

    def before(self, ctx) -> bool | None:
        # CrewAI skips the after-call hook when a model answers with a tool
        # call, so settle each client's usage here, before its next call.
        llm = ctx.llm
        self._sync(llm)
        if self.over_budget:
            self.blocked = True
            return False          # CrewAI refuses the call; nothing is spent
        owner = (self.step, (getattr(ctx.agent, "role", None) or "(output conversion)").strip())
        with self._lock:
            entry = self._seen.get(id(llm))       # set by _sync above
            if entry is not None:
                self._seen[id(llm)] = (llm, entry[1], owner)   # this call's tokens go to this agent
            self.calls += 1
            self.by_step[self.step]["calls"] += 1
            self.by_agent[owner]["calls"] += 1
        return None

    def after(self, ctx) -> None:
        self._sync(ctx.llm)
        return None               # returning a string would replace the model's answer

    def settle(self) -> None:
        """Count any calls still unreported (run after each crew finishes)."""
        for llm, _, _ in list(self._seen.values()):
            self._sync(llm)

    # ---- reporting
    def table(self) -> list[str]:
        """Markdown table of usage by step and agent."""
        rows = ["| Step | Agent | Calls | Tokens |", "|---|---|---:|---:|"]
        for (step, agent), u in self.by_agent.items():
            rows.append(f"| {step} | {agent} | {u['calls']} | {u['tokens']:,} |")
        limit = f" (limit {self.limit:,})" if self.limit else ""
        rows.append(f"| **Total** | | **{self.calls}** | **{self.total:,}**{limit} |")
        return rows

    def as_dict(self) -> dict:
        return {
            "total_tokens": self.total,
            "calls": self.calls,
            "limit": self.limit,
            "by_step": {k: dict(v) for k, v in self.by_step.items()},
        }


# CrewAI's hooks are global, so they're registered once and forward to
# whichever meter the current run has activated.
_active: UsageMeter | None = None
_registered = False


def _before(ctx):
    return _active.before(ctx) if _active else None


def _after(ctx):
    return _active.after(ctx) if _active else None


def activate(meter: UsageMeter | None) -> None:
    global _active, _registered
    if not _registered:
        register_before_llm_call_hook(_before)
        register_after_llm_call_hook(_after)
        _registered = True
    _active = meter
