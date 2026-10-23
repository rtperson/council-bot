# src/council/preflight.py
"""
A one-second check before a real run that uses Ollama models.

Without it, a stopped Ollama costs about a minute of CrewAI retries and ends
in a long traceback saying "Failed to connect to OpenAI API". This asks
CrewAI where it will look for Ollama (so the address comes from your
OLLAMA_HOST setting, read exactly the way CrewAI reads it) and fails fast,
with the fix, if Ollama isn't answering or the model isn't pulled.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from crewai import LLM

AGENTS_YAML = Path(__file__).parent / "config" / "agents.yaml"


class PreflightError(RuntimeError):
    pass


def models_in_use() -> set[str]:
    """The model strings a run will use: COUNCIL_MODEL/DECISION_MODEL, else agents.yaml."""
    override, decision = os.getenv("COUNCIL_MODEL"), os.getenv("DECISION_MODEL")
    if override:
        return {m for m in (override, decision) if m}
    config = yaml.safe_load(AGENTS_YAML.read_text(encoding="utf-8")) or {}
    models = {str(a["llm"]).strip() for a in config.values() if isinstance(a, dict) and a.get("llm")}
    return models | ({decision} if decision else set())


def check_ollama(timeout: float = 3.0) -> list[str]:
    """Raise PreflightError if an Ollama model in use isn't ready. Returns the models checked."""
    wanted = sorted(m for m in models_in_use() if m.split("/", 1)[0].lower() in ("ollama", "ollama_chat"))
    if not wanted:
        return []

    base_url = LLM(model=wanted[0]).base_url.rstrip("/")     # CrewAI's own reading of OLLAMA_HOST
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as r:
            available = {m["id"] for m in json.load(r).get("data", [])}
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise PreflightError(
            f"Can't reach Ollama at {base_url} ({getattr(e, 'reason', e)}). Start the Ollama app "
            "(or run `ollama serve`), or set OLLAMA_HOST if it runs somewhere else."
        ) from e

    names = [m.split("/", 1)[1] for m in wanted]
    missing = [n for n in names if n not in available and f"{n}:latest" not in available]
    if missing:
        raise PreflightError(
            f"Ollama doesn't have {', '.join(missing)}. Pull it first: "
            + "; ".join(f"ollama pull {n}" for n in missing)
            + f". Models it has: {', '.join(sorted(available)) or 'none'}."
        )
    return names
