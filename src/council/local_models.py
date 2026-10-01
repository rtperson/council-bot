# src/council/local_models.py
"""
Checks for local (Ollama) models, run before a real council run.

A missing server or an un-pulled model otherwise shows up as a connection
error deep inside the first crew. This fails in a second with the fix.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import yaml

AGENTS_YAML = Path(__file__).parent / "config" / "agents.yaml"


class LocalModelError(RuntimeError):
    pass


def models_in_use() -> set[str]:
    """Every model string this run will use: the overrides, else agents.yaml."""
    override = os.getenv("COUNCIL_MODEL")
    decision = os.getenv("DECISION_MODEL")
    if override:
        return {m for m in (override, decision) if m}
    config = yaml.safe_load(AGENTS_YAML.read_text(encoding="utf-8")) or {}
    models = {str(a["llm"]).strip() for a in config.values() if isinstance(a, dict) and a.get("llm")}
    return models | ({decision} if decision else set())


def ollama_host() -> str:
    """Ollama's own API root (not the /v1 OpenAI-compatible path)."""
    host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "localhost:11434"
    if "://" not in host:
        host = f"http://{host}"
    host = host.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    if host.startswith("http://") and host.count(":") == 1:   # no port given
        host = f"{host}:11434"
    return host


def _matches(wanted: str, available: set[str]) -> bool:
    return wanted in available or (":" not in wanted and f"{wanted}:latest" in available)


def check_ollama(timeout: float = 3.0) -> list[str]:
    """
    If any model in use is an Ollama model, make sure the server answers and
    has each model pulled. Returns the Ollama models checked (empty if none).
    """
    wanted = sorted(m.split("/", 1)[1] for m in models_in_use()
                    if m.split("/", 1)[0].lower() in ("ollama", "ollama_chat") and "/" in m)
    if not wanted:
        return []

    host = ollama_host()
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            tags = json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise LocalModelError(
            f"Can't reach Ollama at {host} ({getattr(e, 'reason', e)}). "
            "Start it (open the Ollama app, or run `ollama serve`), or set OLLAMA_HOST "
            "if it runs somewhere else."
        ) from e

    available = {m.get("name") or m.get("model") for m in tags.get("models", [])}
    missing = [m for m in wanted if not _matches(m, available)]
    if missing:
        pulls = "; ".join(f"ollama pull {m}" for m in missing)
        have = ", ".join(sorted(a for a in available if a)) or "none"
        raise LocalModelError(f"Ollama doesn't have {', '.join(missing)}. Pull it first: {pulls}. "
                              f"Models it has: {have}.")
    return wanted
