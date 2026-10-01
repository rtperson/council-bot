"""A well-behaved stand-in model: valid JSON for structured tasks, plain text otherwise."""
import json

from council import models
from council.dry_run import fake_model
from fake_ollama import text_reply

OVERRIDES = {
    "TriageResult": {"proceed": True, "rationale": "Worth deliberating."},
    "ModeDecision": {"mode": "standard", "advisors": ["red_team", "first_principles", "expansionist", "executor"],
                     "rationale": "Moderate stakes."},
}


def reply(body: dict) -> dict:
    fmt = body.get("response_format") or {}
    name = (fmt.get("json_schema") or {}).get("name")
    if name:
        data = OVERRIDES.get(name) or fake_model(getattr(models, name)).model_dump()
        return text_reply(json.dumps(data))
    return text_reply("A considered answer from the fake model.")
