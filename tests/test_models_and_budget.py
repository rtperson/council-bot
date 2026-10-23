"""
Real (non-dry) runs against a fake Ollama server: model routing, token
accounting, loop limits, the token budget, and the pre-run Ollama check.
Offline; the "model" is a local HTTP server in this process.
"""
import json
from collections import Counter

import pytest

from council.crew import (
    MAX_ROUNDS_DEBATE,
    MAX_ROUNDS_NO_TOOLS,
    MAX_ROUNDS_TOOLS,
    IntakeCrew,
    ReviewCrew,
)
from council.flow import CouncilFlow
from council.journal import read_entries
from council.preflight import PreflightError, check_ollama
from council.usage import BudgetExceeded

from fake_council_model import reply as good_model
from fake_ollama import FakeOllama, text_reply, tool_reply

TOKENS_PER_CALL = 1100   # FakeOllama's default usage per response
QUESTION = "Should we hire a second consultant?"


@pytest.fixture
def ollama(monkeypatch):
    """Start a fake Ollama with the given behaviour and point the council at it."""
    servers = []

    def start(reply=good_model, models=("fake-model",)):
        srv = FakeOllama(reply, models=models).__enter__()
        servers.append(srv)
        monkeypatch.setenv("OLLAMA_HOST", srv.host)
        monkeypatch.setenv("COUNCIL_MODEL", "ollama/fake-model")
        return srv

    yield start
    for srv in servers:
        srv.__exit__(None, None, None)


def _tool_names(request: dict) -> tuple:
    return tuple(sorted(t["function"]["name"] for t in request.get("tools", [])))


# ------------------------------------------------------------ routing + usage

def test_full_run_goes_to_ollama_and_counts_every_token(ollama, sandbox):
    srv = ollama()
    flow = CouncilFlow()
    flow.kickoff(inputs={"question": QUESTION})

    assert flow.state.status == "complete"
    assert {r["model"] for r in srv.requests} == {"fake-model"}
    assert flow.state.tokens == len(srv.requests) * TOKENS_PER_CALL      # nothing missed

    assert read_entries()[0]["tokens"] == flow.state.tokens
    transcript = (sandbox / "output" / flow.state.run_id / "transcript.md").read_text(encoding="utf-8")
    assert "## Token usage" in transcript and f"{flow.state.tokens:,}" in transcript


def test_well_behaved_model_makes_one_call_per_task(ollama):
    srv = ollama()
    flow = CouncilFlow()
    flow.kickoff(inputs={"question": QUESTION, "mode": "standard"})
    # triage 1 + intake 3 + 4 advisors + 4 reviews + debate + 2 chairs + report + journal
    assert len(srv.requests) == 17


def test_only_the_debate_can_delegate(ollama):
    srv = ollama()
    CouncilFlow().kickoff(inputs={"question": QUESTION, "mode": "standard"})

    tools = Counter(_tool_names(r) for r in srv.requests)
    delegating = [t for t in tools if any("coworker" in n for n in t)]
    assert len(delegating) == 1 and tools[delegating[0]] == 1          # one task: the debate
    assert tools[("council_journal_search",)] == 1                      # journal lookup only
    assert sum(n for t, n in tools.items() if not t) == len(srv.requests) - 3


def test_quick_mode_has_no_delegation_at_all(ollama):
    srv = ollama()
    CouncilFlow().kickoff(inputs={"question": QUESTION, "mode": "quick"})
    assert not any("coworker" in n for r in srv.requests for n in _tool_names(r))


def test_agents_have_round_limits():
    review = ReviewCrew(mode="standard")
    assert review.red_team().max_iter == MAX_ROUNDS_NO_TOOLS
    assert review.consensus_chair().max_iter == MAX_ROUNDS_NO_TOOLS
    assert review._debate_moderator().max_iter == MAX_ROUNDS_DEBATE
    assert IntakeCrew().moderator().max_iter == MAX_ROUNDS_TOOLS
    assert all(a.max_retry_limit == 1 for a in review.crew().agents)


# ------------------------------------------------------------ runaway models

def looping_model(body: dict) -> dict:
    """Never stops calling a tool while it has one; otherwise answers badly."""
    if body.get("tools"):
        return tool_reply("council_journal_search", {"query": "hire"})
    return text_reply("not json")


def test_a_looping_model_is_cut_off(ollama):
    srv = ollama(looping_model)
    with pytest.raises(Exception):
        IntakeCrew().crew().kickoff(inputs={"question": QUESTION})
    # Uncapped, CrewAI allowed 81 calls here; each resends the growing conversation.
    assert len(srv.requests) <= 2 * (MAX_ROUNDS_TOOLS + 2) + 4


def test_budget_stops_the_run_and_keeps_what_was_done(ollama, sandbox, monkeypatch):
    srv = ollama()
    monkeypatch.setenv("COUNCIL_MAX_TOKENS", str(3 * TOKENS_PER_CALL - 100))
    flow = CouncilFlow()
    with pytest.raises(BudgetExceeded, match="token limit"):
        flow.kickoff(inputs={"question": QUESTION})

    assert len(srv.requests) == 3                         # the 4th call was refused, costing nothing
    assert flow.state.status == "stopped"
    assert read_entries() == []                           # an unfinished run isn't journaled
    transcript = (sandbox / "output" / flow.state.run_id / "transcript.md").read_text(encoding="utf-8")
    assert "Stopped during 2-4 Intake" in transcript and "## 1 Triage" in transcript


def test_command_line_budget_flag(ollama, sandbox):
    from council.main import kickoff
    ollama()
    with pytest.raises(SystemExit, match="--max-tokens"):
        kickoff([QUESTION, "--max-tokens", "2000", "--quiet"])


# ------------------------------------------------------------ pre-run check

def test_ollama_check_passes_when_ready(ollama):
    ollama()
    assert check_ollama() == ["fake-model"]


def test_ollama_check_explains_a_missing_model(ollama):
    ollama(models=("llama3.1:8b",))
    with pytest.raises(PreflightError, match="ollama pull fake-model"):
        check_ollama()


def test_ollama_check_accepts_latest_tag(ollama):
    ollama(models=("fake-model:latest",))
    assert check_ollama() == ["fake-model"]


def test_ollama_check_explains_when_ollama_is_not_running(monkeypatch):
    monkeypatch.setenv("COUNCIL_MODEL", "ollama/qwen3:14b")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")      # nothing listens here
    with pytest.raises(PreflightError, match="Can't reach Ollama at http://127.0.0.1:9/v1"):
        check_ollama(timeout=1)


def test_ollama_check_skips_hosted_models(monkeypatch):
    monkeypatch.setenv("COUNCIL_MODEL", "openai/gpt-5.4-mini")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    assert check_ollama() == []
